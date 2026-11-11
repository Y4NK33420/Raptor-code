"""``raptor code ...``: index, update, search, ask, browse and serve a codebase."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import typer

code_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="RAPTOR for codebases: symbol trees, code graph, hybrid search and cited answers.",
)

IndexDir = Annotated[
    Path, typer.Argument(help="Code index directory.", exists=True, file_okay=False)
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")]


@code_app.callback()
def _main() -> None:
    # Node text contains non-ASCII (e.g. "›"); never crash on a narrow console encoding.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(errors="replace")


def _setup_logging(verbose: bool) -> None:
    from ..settings import get_settings

    level = "DEBUG" if verbose else get_settings().log_level
    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )
    for noisy in ("httpx", "google_genai", "numba", "urllib3", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _progress(ev: dict[str, Any]) -> None:
    stage = ev.get("stage")
    show = {
        "parsing": ev.get("done") == ev.get("total"),
        "embedding": bool(ev.get("total")),
        "summarizing": bool(ev.get("total")) and ev.get("done") in (0, ev.get("total")),
    }.get(str(stage), True)
    if show:
        typer.echo(f"  {ev}", err=True)


def _load(index: Path):
    from .index import CodeIndex

    return CodeIndex.load(index)


@code_app.command("index")
def index_cmd(
    path: Annotated[
        Path, typer.Argument(help="Repository to index.", exists=True, file_okay=False)
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Directory to write the index to.")],
    summary_depth: Annotated[
        str,
        typer.Option("--summary-depth", help="symbol: summarize every symbol; file: files only."),
    ] = "file",
    config: Annotated[Path | None, typer.Option("--config", "-c", help="YAML CodeConfig.")] = None,
    concepts: Annotated[bool, typer.Option(help="Build the semantic concept layer.")] = True,
    exclude: Annotated[
        list[str] | None,
        typer.Option("--exclude", "-x", help="Extra gitignore-style pattern to skip."),
    ] = None,
    summary_provider: Annotated[
        str | None,
        typer.Option(help="Summaries via 'gemini' or 'ollama' (default: models.provider)."),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing index at --out.")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Build a code index for a repository."""
    _setup_logging(verbose)
    from .config import CodeConfig
    from .index import CodeIndex

    if summary_depth not in ("symbol", "file"):
        raise typer.BadParameter("--summary-depth must be 'symbol' or 'file'")
    if (out / "code.json").exists() and not overwrite:
        raise typer.BadParameter(f"{out} already holds an index; pass --overwrite or use `update`")
    cfg = CodeConfig.from_yaml(config) if config else CodeConfig()
    cfg.summary.depth = summary_depth  # type: ignore[assignment]
    cfg.concepts.enabled = concepts
    if exclude:
        cfg.ingest.exclude = [*cfg.ingest.exclude, *exclude]
    if summary_provider:
        cfg.models.summary_provider = summary_provider  # type: ignore[assignment]
    idx = CodeIndex.build(path, cfg, progress=_progress)
    idx.save(out)
    st = idx.stats()
    typer.echo(
        f"Indexed {st['files']} files -> {sum(st['nodes'].values())} nodes {st['nodes']}, "
        f"edges {st['edges']} -> {out}"
    )
    typer.echo(f"Model calls: {idx.last_build_stats}")


@code_app.command()
def update(index: IndexDir, verbose: VerboseOpt = False) -> None:
    """Re-index files changed since the index was built (git diff or content hashes)."""
    _setup_logging(verbose)
    idx = _load(index)
    result = idx.update(progress=_progress)
    if result["changed_files"]:
        idx.save(index)
    typer.echo(
        f"Changed files: {len(result['changed_files'])}; re-embedded {result['reembedded']}; "
        f"re-summarized {result['resummarized']}"
    )
    for f in result["changed_files"]:
        typer.echo(f"  {f}")


@code_app.command()
def search(
    index: IndexDir,
    query: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    bm25: Annotated[bool, typer.Option(help="Use BM25 over identifiers.")] = True,
    dense: Annotated[bool, typer.Option(help="Use dense embeddings.")] = True,
    graph: Annotated[bool, typer.Option(help="Expand hits along the call graph.")] = True,
    as_json: JsonOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Search the index (hybrid BM25 + dense, with graph expansion)."""
    _setup_logging(verbose)
    idx = _load(index)
    hits = idx.search(query, k, bm25=bm25, dense=dense, graph_expansion=graph)
    if as_json:
        typer.echo(json.dumps([h.to_dict() for h in hits], indent=2))
        return
    for i, h in enumerate(hits, 1):
        first = next((ln for ln in h.text.splitlines() if ln.strip()), "")
        typer.echo(f"{i:2d}. [{h.kind:7s}] {h.location}  {h.symbol}  ({h.score:.4f})")
        typer.echo(f"      {first[:110]}")


@code_app.command()
def ask(
    index: IndexDir,
    question: Annotated[str, typer.Argument()],
    k: Annotated[int, typer.Option("--top-k", "-k")] = 10,
    show_context: Annotated[bool, typer.Option(help="List the retrieved nodes first.")] = False,
    as_json: JsonOpt = False,
    verbose: VerboseOpt = False,
) -> None:
    """Answer a question about the codebase with path:line citations."""
    _setup_logging(verbose)
    from .types import CodeAnswer

    idx = _load(index)
    if as_json:
        typer.echo(json.dumps(idx.ask(question, k).to_dict(), indent=2))
        return
    final: CodeAnswer | None = None
    for piece in idx.stream_ask(question, k):
        if isinstance(piece, CodeAnswer):
            final = piece
        else:
            typer.echo(piece, nl=False)
    typer.echo("")
    if final is not None and (show_context or final.citations):
        typer.echo("\nSources:")
        for c in final.citations:
            typer.echo(f"  - {c.location} ({c.kind} {c.symbol})")


@code_app.command()
def tree(
    index: IndexDir,
    depth: Annotated[int, typer.Option(help="Levels to print below the repo.")] = 3,
    summaries: Annotated[bool, typer.Option(help="Show the first line of each summary.")] = False,
) -> None:
    """Print the structural hierarchy (repo > dirs > files > classes > symbols) and concepts."""
    idx = _load(index)
    st = idx.stats()
    typer.echo(f"{st['repo_path']}: {st['files']} files, nodes {st['nodes']}, edges {st['edges']}")
    for d, h in idx.outline(max_depth=depth):
        label = {
            "repo": f"{h.symbol}/",
            "dir": f"{h.path}/",
            "file": h.path,
            "class": f"class {h.symbol}  :{h.start_line}-{h.end_line}",
            "symbol": f"{h.symbol}  :{h.start_line}-{h.end_line}",
            "concept": f"[concept] {h.symbol}",
        }.get(h.kind, h.symbol)
        line = "  " * d + label
        if summaries and h.kind != "symbol":
            body = h.text.split("\n", 1)[1] if "\n" in h.text else h.text
            first = next((ln for ln in body.splitlines() if ln.strip()), "")
            line += f"  — {first[:100]}"
        typer.echo(line)


@code_app.command("serve-mcp")
def serve_mcp(index: IndexDir, verbose: VerboseOpt = False) -> None:
    """Serve the index as an MCP server over stdio (for Claude Code, IDEs, ...)."""
    _setup_logging(verbose)
    from .mcp_server import serve

    serve(index)


if __name__ == "__main__":  # pragma: no cover
    code_app()
