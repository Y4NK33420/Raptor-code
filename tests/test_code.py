"""Offline tests for raptor.code (fixture repo + fake models from conftest)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tree_sitter")
pytest.importorskip("tree_sitter_python")

from conftest import EchoQA, EchoSummarizer, HashEmbedder

from raptor.chunking import default_tokenizer
from raptor.clustering import CLUSTERERS, ClusteringAlgorithm
from raptor.code import CodeAnswer, CodeConfig, CodeEdge, CodeHit, CodeIndex
from raptor.code.chunks import make_leaves
from raptor.code.config import ChunkConfig, IngestConfig
from raptor.code.ingest import discover_files, read_source
from raptor.code.languages import PythonExtractor
from raptor.code.lexical import BM25, split_identifier, tokenize
from raptor.models.cache import CachedSummarizationModel, DiskCache

FIXTURE = Path(__file__).parent / "fixtures" / "code_repo"


class _GroupByDirClusterer(ClusteringAlgorithm):
    """Deterministic, fast stand-in for UMAP+GMM: two clusters by embedding sign."""

    def __init__(self, **_: object) -> None:
        pass

    def cluster(self, embeddings, token_counts):
        emb = np.asarray(embeddings)
        axis = emb - emb.mean(axis=0)
        u, _, _ = np.linalg.svd(axis, full_matrices=False)
        labels = (u[:, 0] > 0).astype(int)
        out = [list(np.where(labels == i)[0]) for i in (0, 1)]
        return [c for c in out if c]


class DigestSummarizer(EchoSummarizer):
    """Like EchoSummarizer, but any change to the input changes the summary."""

    model_id = "test:digest"

    def summarize(self, context: str, max_tokens: int = 150) -> str:
        import hashlib

        digest = hashlib.sha256(context.encode()).hexdigest()[:8]
        return f"{super().summarize(context, max_tokens)} [{digest}]"


if "test_split" not in CLUSTERERS:
    CLUSTERERS.register("test_split")(_GroupByDirClusterer)


def make_config(depth: str = "file") -> CodeConfig:
    cfg = CodeConfig()
    cfg.summary.depth = depth  # type: ignore[assignment]
    cfg.summary.max_workers = 1
    cfg.concepts.clustering.strategy = "test_split"
    cfg.concepts.min_nodes = 4
    cfg.models.cache = False
    return cfg


def build(repo: Path = FIXTURE, depth: str = "file", **models):
    models.setdefault("embedding_model", HashEmbedder())
    models.setdefault("summarization_model", EchoSummarizer())
    models.setdefault("qa_model", EchoQA())
    return CodeIndex.build(repo, make_config(depth), **models)


@pytest.fixture(scope="module")
def index() -> CodeIndex:
    return build()


# --------------------------------------------------------------------------- parsing
def test_python_extractor_symbols_imports_calls():
    src = (FIXTURE / "shop" / "cart.py").read_bytes()
    pf = PythonExtractor().parse(src, "shop/cart.py")
    assert pf.module_docstring == "Shopping cart model."
    names = {s.qualname: s for s in pf.symbols}
    assert {"Cart", "Cart.__init__", "Cart.add_item", "Cart.total"} <= names.keys()
    total = names["Cart.total"]
    assert total.kind == "method" and total.parent == "Cart"
    assert total.signature == "def total(self, discount_code: str | None = None) -> float"
    assert total.docstring.startswith("Subtotal after applying")
    assert {"self.subtotal", "apply_discount", "round"} <= set(total.calls)
    assert names["Cart"].kind == "class" and "currency" in names["Cart"].text
    imp = next(i for i in pf.imports if i.name == "apply_discount")
    assert imp.module == "pricing" and imp.level == 1
    bases = {
        s.qualname: s.bases
        for s in PythonExtractor()
        .parse((FIXTURE / "shop" / "pricing.py").read_bytes(), "shop/pricing.py")
        .symbols
    }
    assert bases["PercentRule"] == ["PriceRule"]


def test_python_extractor_decorators_nested_and_module_code():
    src = b'''"""Doc."""
import os

X = 1


@decorate(1)
def f(a,
      b=2):
    def inner():
        return os.getcwd()
    return inner()


class A(Base, metaclass=M):
    @property
    def p(self):
        return 1

    class B:
        def g(self):
            pass
'''
    pf = PythonExtractor().parse(src, "pkg/m.py")
    names = {s.qualname: s for s in pf.symbols}
    f = names["f"]
    assert f.decorators == ["decorate(1)"] and f.start_line == 7 and f.signature == "def f(a, b=2)"
    assert "os.getcwd" in f.calls and "inner" in f.calls
    assert names["A"].bases == ["Base"]
    assert names["A.p"].decorators == ["property"]
    assert names["A.B.g"].parent == "A.B"
    assert pf.module_code and pf.module_code[0][0] == 1
    assert PythonExtractor().module_name("src/pkg/sub/__init__.py") == "pkg.sub"
    assert PythonExtractor().module_name("pkg/m.py") == "pkg.m"


def test_leaves_split_merge_and_header():
    tok = default_tokenizer()
    body = "\n".join(f"    x{i} = compute_value({i}) + other_value({i})" for i in range(80))
    src = f"import os\n\n\ndef big():\n{body}\n    return os.sep\n\n\ndef a():\n    return 1\n\n\ndef b():\n    return 2\n"
    pf = PythonExtractor().parse(src.encode(), "m.py")
    leaves = make_leaves(pf, src, tok, ChunkConfig(max_symbol_tokens=200, min_symbol_tokens=40))
    big = [lf for lf in leaves if lf.symbol == "big"]
    assert len(big) > 1 and all(lf.part is not None for lf in big)
    assert big[0].start_line == 4 and big[-1].end_line == pf.symbols[0].end_line
    merged = [lf for lf in leaves if lf.symbol_kind == "merged"]
    assert merged and merged[0].symbols == ["a", "b"]
    assert big[0].text.startswith("# m.py › def big() (part 1/")
    assert "# uses: import os" in big[-1].text or "# uses: import os" in big[0].text
    for lf in leaves:
        assert lf.metadata()["content_hash"] and lf.metadata()["language"] == "python"


# --------------------------------------------------------------------------- ingest
def test_discover_respects_gitignore_and_skips(tmp_path):
    (tmp_path / ".gitignore").write_text("ignored.py\nsecret/\n")
    (tmp_path / "keep.py").write_text("def f():\n    return 1\n")
    (tmp_path / "ignored.py").write_text("x = 1\n")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "s.py").write_text("x = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "n.py").write_text("x = 1\n")
    (tmp_path / "msg_pb2.py").write_text("x = 1\n")
    (tmp_path / "gen.py").write_text("# Generated by protoc. DO NOT EDIT.\nx = 1\n")
    (tmp_path / "bin.py").write_bytes(b"\x00\x01\x02")
    cfg = IngestConfig(use_git=False)
    files = discover_files(tmp_path, cfg, [".py"])
    assert "keep.py" in files and "ignored.py" not in files
    assert not any(f.startswith(("secret/", "node_modules/")) for f in files)
    assert "msg_pb2.py" not in files
    assert read_source(tmp_path, "gen.py", cfg) is None
    assert read_source(tmp_path, "bin.py", cfg) is None
    assert read_source(tmp_path, "keep.py", cfg) is not None


# -------------------------------------------------------------------------- lexical
def test_identifier_tokenization_and_bm25():
    assert split_identifier("getHTTPResponse_code") == ["get", "http", "response", "code"]
    toks = tokenize("def apply_discount(amount): return CartStore.save_cart(x)")
    assert {"apply_discount", "apply", "discount", "cartstore", "cart", "store", "save"} <= set(
        toks
    )
    bm = BM25([tokenize("apply_discount amount code"), tokenize("save_cart owner")])
    assert bm.top(tokenize("discount"), 1)[0][0] == 0


# ------------------------------------------------------------------------ hierarchy
def test_hierarchy_layers_and_metadata(index: CodeIndex):
    tree = index.tree
    kinds = {n.metadata["kind"] for n in tree.all_nodes.values()}
    assert kinds == {"symbol", "class", "file", "dir", "repo", "concept"}
    for n in tree.all_nodes.values():
        m = n.metadata
        if m["kind"] == "symbol":
            assert n.layer == 0 and not n.children
            for key in (
                "path",
                "start_line",
                "end_line",
                "symbol",
                "signature",
                "docstring",
                "language",
                "content_hash",
            ):
                assert key in m
        elif m["kind"] == "class":
            assert n.layer == 1
        elif m["kind"] == "file":
            assert n.layer == 2 and m["start_line"] == 1
    dirs = {n.metadata["path"]: n for n in tree.all_nodes.values() if n.metadata["kind"] == "dir"}
    assert set(dirs) == {"shop", "shop/storage"}
    assert dirs["shop"].layer > dirs["shop/storage"].layer
    repo = next(n for n in tree.root_nodes.values() if n.metadata["kind"] == "repo")
    child_paths = {tree.all_nodes[c].metadata["path"] for c in repo.children}
    assert child_paths == {"shop", "cli.py"}
    concepts = [n for n in tree.all_nodes.values() if n.metadata["kind"] == "concept"]
    assert concepts and all(n.index in tree.root_nodes for n in concepts)
    # The concept layer makes this a DAG: some file/class node has two parents.
    assert any(len(tree.parents_of(c)) > 1 for n in concepts for c in n.children)
    assert index.files() == [
        "cli.py",
        "shop/__init__.py",
        "shop/cart.py",
        "shop/pricing.py",
        "shop/storage/__init__.py",
        "shop/storage/db.py",
    ]


def test_graph_edges(index: CodeIndex):
    def named(e: CodeEdge) -> tuple[str, str, str]:
        return (e.kind, index.node_hit(e.src).symbol, index.node_hit(e.dst).symbol)

    edges = {named(e) for e in index.graph()}
    assert ("calls", "Cart.total", "apply_discount") in edges
    assert ("calls", "apply_discount", "PercentRule") in edges
    assert ("calls", "main", "CartStore.save_cart") in edges
    assert any(
        k == "inherits" and s == "PercentRule" and d.startswith("PriceRule") for k, s, d in edges
    )
    assert ("imports", "shop/cart.py", "shop/pricing.py") in edges
    assert ("imports", "shop/storage/db.py", "shop/cart.py") in edges
    assert all(e.kind == "calls" for e in index.graph(kinds=("calls",)))


# ------------------------------------------------------------------------ retrieval
def test_search_identifier_prefers_leaf(index: CodeIndex):
    hits = index.search("apply_discount", k=5)
    assert isinstance(hits[0], CodeHit)
    assert hits[0].kind == "symbol" and hits[0].symbol == "apply_discount"
    assert hits[0].location == "shop/pricing.py:29-32"
    assert index.searcher.query_type("apply_discount") == "identifier"
    assert index.searcher.query_type("how are carts persisted") == "broad"


def test_search_broad_question_reaches_summaries(index: CodeIndex):
    hits = index.search("how are carts saved and loaded from the sqlite database", k=8)
    assert any(h.path.startswith("shop/storage") for h in hits[:3])
    assert any(h.kind in ("file", "dir", "class", "concept") for h in hits)


def test_search_lexical_only_and_flags(index: CodeIndex):
    hits = index.search("save_cart", k=3, dense=False)
    assert hits[0].symbol == "CartStore.save_cart"
    plain = index.search("tax for region", k=5, bm25=False, graph_expansion=False, descend=False)
    assert plain
    only_leaves = index.search("cart total", k=5, kinds=("symbol",))
    assert all(h.kind == "symbol" for h in only_leaves)
    with pytest.raises(ValueError):
        index.search("x", dense=False, bm25=False)


def test_graph_expansion_adds_callees(index: CodeIndex):
    base = index.search("loyalty_total", k=6, graph_expansion=False, dense=False)
    expanded = index.search("loyalty_total", k=6, graph_expansion=True, dense=False)
    assert "PercentRule" not in {h.symbol for h in base[:1]}
    assert any(h.symbol.startswith("PercentRule") for h in expanded)


def test_ask_and_stream_ask_cite_locations(index: CodeIndex):
    ans = index.ask("How is the cart total computed with a discount code?", k=5)
    assert isinstance(ans, CodeAnswer) and ans.answer.startswith("Q=")
    assert ans.citations and all(
        ":" in c.location or c.location.endswith(")") or c.location.endswith("/")
        for c in ans.citations
    )
    assert any(c.location.startswith("shop/cart.py:") for c in ans.citations)
    pieces = list(index.stream_ask("where is tax computed?", k=4))
    assert isinstance(pieces[-1], CodeAnswer)
    assert all(isinstance(p, str) for p in pieces[:-1])
    assert pieces[-1].answer == "".join(pieces[:-1])


def test_symbol_info(index: CodeIndex):
    total = index.find_symbol("Cart.total")[0]
    info = index.symbol(total.node_index)
    assert set(info) == {"hit", "children", "parents", "callers", "callees"}
    assert any(h.symbol == "apply_discount" for h in info["callees"])
    assert any(h.kind == "class" and h.symbol == "Cart" for h in info["parents"])
    cart = next(h for h in index.find_symbol("Cart") if h.kind == "class")
    cart_info = index.symbol(cart.node_index)
    assert any(h.symbol == "main" for h in cart_info["callers"])
    assert index.module_node("shop/storage") is not None
    assert index.module_node("shop.pricing") == index.module_node("shop/pricing.py")


def test_save_load_roundtrip(index: CodeIndex, tmp_path):
    index.save(tmp_path / "idx")
    loaded = CodeIndex.load(tmp_path / "idx", embedding_model=HashEmbedder())
    assert loaded.files() == index.files()
    assert loaded.graph() == index.graph()
    q = "discount code percentage"
    assert [h.node_index for h in loaded.search(q)] == [h.node_index for h in index.search(q)]
    assert loaded.repo_path == index.repo_path


# ---------------------------------------------------------------------- incremental
def _ancestors(tree, idx: int) -> set[int]:
    out: set[int] = set()
    stack = [idx]
    while stack:
        for p in tree.parents_of(stack.pop()):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


@pytest.mark.parametrize("depth", ["symbol", "file"])
def test_update_resummarizes_only_ancestors(tmp_path, depth):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    cache = DiskCache(tmp_path / "cache.sqlite")
    inner = DigestSummarizer()
    summarizer = CachedSummarizationModel(inner, cache)
    idx = build(repo, depth, summarization_model=summarizer)
    assert idx.update() == {"changed_files": [], "reembedded": 0, "resummarized": 0}

    calls_before, misses_before = inner.calls, summarizer.stats.misses
    _edit(
        repo / "shop" / "pricing.py",
        "rates.get(region.lower(), 0.2)",
        "rates.get(region.strip().lower(), 0.2)",
    )
    result = idx.update()
    assert result["changed_files"] == ["shop/pricing.py"]

    leaf = next(h for h in idx.find_symbol("tax_for") if h.kind == "symbol")
    summarized_ancestors = {
        a
        for a in _ancestors(idx.tree, leaf.node_index)
        if "input_hash" in idx.tree.all_nodes[a].metadata
    }
    expected = len(summarized_ancestors) + (1 if depth == "symbol" else 0)
    kinds = {idx.tree.all_nodes[a].metadata["kind"] for a in summarized_ancestors}
    assert {"file", "dir", "repo"} <= kinds
    assert result["resummarized"] == expected
    assert inner.calls - calls_before == expected
    assert summarizer.stats.misses - misses_before == expected
    # Only the edited leaf and the re-summarized summary nodes need new embeddings
    # (symbol summaries live in leaf metadata and are not embedded separately).
    assert result["reembedded"] == 1 + len(summarized_ancestors)
    # Nothing outside the ancestor chain changed.
    others = [
        n
        for n in idx.tree.all_nodes.values()
        if n.metadata.get("path") == "shop/cart.py" and n.metadata["kind"] == "file"
    ]
    assert others and others[0].index not in summarized_ancestors


def test_update_method_edit_resummarizes_class_in_symbol_depth(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    inner = DigestSummarizer()
    idx = build(repo, "symbol", summarization_model=inner)
    before = inner.calls
    _edit(repo / "shop" / "cart.py", "self.items.pop(sku, None)", "del self.items[sku]")
    result = idx.update()
    leaf = next(h for h in idx.find_symbol("Cart.remove_item") if h.kind == "symbol")
    anc = _ancestors(idx.tree, leaf.node_index)
    assert any(idx.tree.all_nodes[a].metadata["kind"] == "class" for a in anc)
    assert result["resummarized"] == inner.calls - before
    assert result["resummarized"] == 1 + sum(
        1 for a in anc if "input_hash" in idx.tree.all_nodes[a].metadata
    )


def test_update_added_and_deleted_files(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    idx = build(repo)
    (repo / "shop" / "shipping.py").write_text(
        '"""Shipping costs."""\n\n\ndef shipping_cost(weight_kg: float, express: bool = False) -> float:\n'
        '    """Cost to ship a parcel of ``weight_kg`` kilograms."""\n'
        "    base = 4.5 + 1.2 * weight_kg\n    return base * 2 if express else base\n",
        encoding="utf-8",
    )
    (repo / "cli.py").unlink()
    result = idx.update()
    assert result["changed_files"] == ["cli.py", "shop/shipping.py"]
    assert "shop/shipping.py" in idx.files() and "cli.py" not in idx.files()
    assert idx.search("shipping_cost", k=1)[0].path == "shop/shipping.py"
    idx.save(tmp_path / "idx")
    again = CodeIndex.load(
        tmp_path / "idx", embedding_model=HashEmbedder(), summarization_model=EchoSummarizer()
    )
    assert again.update()["changed_files"] == []


# ------------------------------------------------------------------------- CLI / MCP
@pytest.fixture
def fake_default_models(monkeypatch):
    import raptor.code.models as cm

    monkeypatch.setattr(cm, "build_embedding_model", lambda cfg: HashEmbedder())
    monkeypatch.setattr(cm, "build_summarization_model", lambda cfg: EchoSummarizer())
    monkeypatch.setattr(cm, "build_qa_model", lambda cfg: EchoQA())
    monkeypatch.setenv("RAPTOR_CACHE_ENABLED", "false")
    from raptor.settings import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_cli_index_search_ask_tree_update(tmp_path, fake_default_models):
    from typer.testing import CliRunner

    from raptor.cli import app

    repo = tmp_path / "repo"
    shutil.copytree(FIXTURE, repo)
    cfg_path = tmp_path / "code.yaml"
    make_config().to_yaml(cfg_path)
    out = tmp_path / "idx"
    runner = CliRunner()
    r = runner.invoke(
        app,
        [
            "code",
            "index",
            str(repo),
            "--out",
            str(out),
            "-c",
            str(cfg_path),
            "--summary-depth",
            "symbol",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (out / "code.json").exists() and (out / "graph.json").exists()
    r = runner.invoke(app, ["code", "search", str(out), "apply_discount", "-k", "3"])
    assert r.exit_code == 0 and "shop/pricing.py:29-32" in r.output
    r = runner.invoke(app, ["code", "ask", str(out), "How is the total computed?"])
    assert r.exit_code == 0 and "Sources:" in r.output and "shop/cart.py:" in r.output
    r = runner.invoke(app, ["code", "tree", str(out), "--summaries"])
    assert r.exit_code == 0 and "shop/storage/" in r.output and "[concept]" in r.output
    r = runner.invoke(app, ["code", "update", str(out)])
    assert r.exit_code == 0 and "Changed files: 0" in r.output


def test_mcp_tools(index: CodeIndex):
    import anyio

    mcp = pytest.importorskip("mcp")
    from raptor.code.mcp_server import create_server

    server = create_server(index)

    async def run() -> dict:
        import json

        results: dict = {}
        if hasattr(mcp, "Client"):  # mcp >= 2
            client_cm = mcp.Client(server)
        else:  # pragma: no cover - mcp 1.x
            from mcp.shared.memory import create_connected_server_and_client_session

            client_cm = create_connected_server_and_client_session(server._mcp_server)
        async with client_cm as client:
            tools = await client.list_tools()
            results["tools"] = sorted(t.name for t in tools.tools)
            calls = {
                "search_code": {"query": "apply_discount", "k": 3},
                "ask_codebase": {"question": "How is tax computed?"},
                "get_symbol": {"symbol": "Cart.total"},
                "explain_module": {"path": "shop/storage"},
                "get_callers": {"symbol": "apply_discount"},
                "get_callees": {"symbol": "Cart.total"},
            }
            for name, args in calls.items():
                res = await client.call_tool(name, args)
                assert not res.is_error if hasattr(res, "is_error") else not res.isError
                results[name] = json.loads(res.content[0].text)
        return results

    out = anyio.run(run)
    assert out["tools"] == [
        "ask_codebase",
        "explain_module",
        "get_callees",
        "get_callers",
        "get_symbol",
        "search_code",
    ]
    assert out["search_code"]["results"][0]["location"] == "shop/pricing.py:29-32"
    assert out["ask_codebase"]["citations"]
    assert out["get_symbol"]["symbol"]["symbol"] == "Cart.total"
    assert any(c["symbol"] == "apply_discount" for c in out["get_symbol"]["callees"])
    assert out["explain_module"]["module"]["kind"] == "dir"
    assert any(c["symbol"] == "Cart.total" for c in out["get_callers"]["callers"])
    assert any(c["symbol"] == "apply_discount" for c in out["get_callees"]["callees"])
