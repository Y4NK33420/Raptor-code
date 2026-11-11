"""Alternative chunkers, clusterers and retrieval refinements (offline fakes only)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import TOPICS, HashEmbedder

from raptor.builder import TreeBuilder
from raptor.chunking import CHUNKERS, default_tokenizer, make_chunker
from raptor.chunking.strategies import paragraph_spans, prose_sentence_spans
from raptor.clustering import CLUSTERERS, make_clusterer
from raptor.config import ChunkingConfig, ClusteringConfig, RaptorConfig, RetrievalConfig
from raptor.retrieval import TreeRetriever
from raptor.retrieval.hybrid import BM25, mmr, rrf

TOK = default_tokenizer()
WRAPPED = (
    "THE TITLE\nBy AN AUTHOR\n\n\n The first paragraph starts here and it\n wraps onto a second "
    "line. It has two sentences.\nA new paragraph begins without a blank line. Short.\n\n\n "
    '"Dialogue line," she said.\n'
)


def ntok(text: str) -> int:
    return len(TOK.encode(text, disallowed_special=()))


# ----------------------------------------------------------------- chunking
def test_paragraph_and_sentence_spans_handle_wrapped_text():
    paras = [WRAPPED[a:b] for a, b in paragraph_spans(WRAPPED)]
    assert paras[0] == "THE TITLE"
    assert any(p.startswith("The first paragraph") and "second line" in p for p in paras)
    assert any(p.startswith("A new paragraph") for p in paras)
    sents = [WRAPPED[a:b] for a, b in prose_sentence_spans(WRAPPED)]
    assert "The first paragraph starts here and it\n wraps onto a second line." in sents
    assert "It has two sentences." in sents


@pytest.mark.parametrize(
    "strategy", ["sentence", "fixed_tokens", "paragraph", "recursive", "semantic"]
)
@pytest.mark.parametrize("max_tokens", [20, 60])
def test_chunkers_respect_limits_and_offsets(strategy, max_tokens, document):
    chunker = make_chunker(ChunkingConfig(strategy=strategy, max_tokens=max_tokens), HashEmbedder())
    chunks = chunker.split(document + "\n\n" + WRAPPED)
    assert chunks
    text = document + "\n\n" + WRAPPED
    for c in chunks:
        assert c.text.strip()
        assert ntok(c.text) <= max_tokens
        assert c.start is not None and text[c.start : c.end] == c.text
    # Every non-space character of the source ends up in some chunk.
    covered = np.zeros(len(text), dtype=bool)
    for c in chunks:
        covered[c.start : c.end] = True
    assert all(covered[i] for i, ch in enumerate(text) if not ch.isspace())


def test_all_chunkers_registered():
    assert {"sentence", "fixed_tokens", "paragraph", "recursive", "semantic"} <= set(
        CHUNKERS.names()
    )


def test_fixed_tokens_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = make_chunker(
        ChunkingConfig(strategy="fixed_tokens", max_tokens=30, overlap_tokens=10)
    ).split(text)
    assert len(chunks) > 3
    for a, b in itertools.pairwise(chunks):
        assert a.end is not None and b.start is not None and b.start < a.end  # windows overlap
    assert chunks[-1].end == len(text)


def test_paragraph_chunker_keeps_short_paragraphs_whole():
    chunks = make_chunker(ChunkingConfig(strategy="paragraph", max_tokens=200)).split(WRAPPED)
    joined = [c.text for c in chunks]
    assert any("The first paragraph" in t and "two sentences." in t for t in joined)


def test_semantic_chunker_cuts_at_topic_shifts(document):
    chunker = make_chunker(
        ChunkingConfig(strategy="semantic", max_tokens=400, params={"breakpoint_percentile": 95}),
        HashEmbedder(),
    )
    chunks = chunker.split(document)

    # A chunk should rarely mix vocabularies from different topics.
    def topics_of(t: str) -> set[str]:
        words = set(t.lower().split())
        return {name for name, vocab in TOPICS.items() if len(words & set(vocab.split())) >= 2}

    mixed = sum(1 for c in chunks if len(topics_of(c.text)) > 1)
    assert mixed <= 3


def test_semantic_chunker_requires_embedder():
    with pytest.raises(ValueError, match="embedding model"):
        make_chunker(ChunkingConfig(strategy="semantic"))


# --------------------------------------------------------------- clustering
NEW_CLUSTERERS = ["gmm", "pca_gmm", "agglomerative", "kmeans_silhouette", "contiguous", "hdbscan"]


def topic_embeddings(per_topic: int = 12, seed: int = 0) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, 32))
    emb = np.concatenate([c + 0.15 * rng.normal(size=(per_topic, 32)) for c in centers])
    labels = [i for i in range(4) for _ in range(per_topic)]
    return emb.astype(np.float32), labels


@pytest.mark.parametrize("strategy", NEW_CLUSTERERS)
def test_clusterers_cover_and_respect_token_limit(strategy):
    emb, _ = topic_embeddings()
    tokens = [100] * len(emb)
    algo = make_clusterer(ClusteringConfig(strategy=strategy, max_cluster_tokens=700))
    clusters = algo.cluster(emb, tokens)
    assert {i for c in clusters for i in c} == set(range(len(emb)))
    assert all(sum(tokens[i] for i in c) <= 700 or len(c) == 1 for c in clusters)
    assert clusters == algo.cluster(emb, tokens)  # deterministic


@pytest.mark.parametrize("strategy", ["gmm", "pca_gmm", "agglomerative", "kmeans_silhouette"])
def test_similarity_clusterers_recover_topics(strategy):
    emb, labels = topic_embeddings()
    params = {"target_size": 12} if strategy == "agglomerative" else {}
    clusters = make_clusterer(
        ClusteringConfig(strategy=strategy, max_cluster_tokens=10_000, params=params)
    ).cluster(emb, [10] * len(emb))
    purity = sum(max(np.bincount([labels[i] for i in c])) for c in clusters) / sum(
        map(len, clusters)
    )
    assert purity > 0.9


def test_contiguous_returns_ordered_runs():
    emb, _ = topic_embeddings(per_topic=10)
    clusters = make_clusterer(
        ClusteringConfig(
            strategy="contiguous", max_cluster_tokens=10_000, params={"target_size": 10}
        )
    ).cluster(emb, [10] * len(emb))
    flat = [i for c in clusters for i in c]
    assert flat == list(range(len(emb)))  # contiguous, in order, no overlap
    assert all(c == list(range(c[0], c[-1] + 1)) for c in clusters)
    # Cuts should land on the topic boundaries (every 10 rows).
    assert [c[0] for c in clusters] == [0, 10, 20, 30]


def test_all_clusterers_registered():
    assert set(NEW_CLUSTERERS) | {"raptor_gmm"} <= set(CLUSTERERS.names())


@pytest.mark.parametrize(
    "strategy", ["contiguous", "kmeans_silhouette", "agglomerative", "pca_gmm"]
)
def test_builder_with_new_clusterers(strategy, document, summarizer):
    cfg = RaptorConfig()
    cfg.chunking.max_tokens = 40
    cfg.clustering.strategy = strategy
    cfg.clustering.max_cluster_tokens = 400
    tree = TreeBuilder(cfg, HashEmbedder(), summarizer).build_from_text(document)
    tree.validate()
    assert tree.num_layers >= 1
    for layer in range(1, tree.num_layers + 1):
        for node in tree.layer_to_nodes[layer]:
            assert node.children and all(
                tree.all_nodes[c].layer == layer - 1 for c in node.children
            )


# ---------------------------------------------------------------- retrieval
def test_bm25_ranks_lexical_match_first():
    bm = BM25(["the cat sat on the mat", "dogs bark loudly", "a cat and a dog"])
    s = bm.scores("cat mat")
    assert int(np.argmax(s)) == 0 and s[1] == 0


def test_rrf_and_mmr():
    assert rrf([[1, 2], [1, 3]])[0] == 1
    vecs = np.asarray([[1, 0], [0.99, 0.141], [0, 1]], dtype=np.float64)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    rel = np.asarray([1.0, 0.95, 0.5])
    assert mmr([10, 11, 12], rel, vecs, lam=1.0) == [10, 11, 12]
    assert mmr([10, 11, 12], rel, vecs, lam=0.3)[:2] == [
        10,
        12,
    ]  # diversity skips the near-duplicate


@pytest.fixture(scope="module")
def small_tree(document):
    from conftest import EchoSummarizer

    cfg = RaptorConfig()
    cfg.chunking.max_tokens = 40
    cfg.clustering.strategy = "contiguous"
    return TreeBuilder(cfg, HashEmbedder(), EchoSummarizer()).build_from_text(document)


def test_default_retrieval_unchanged_by_new_options(small_tree):
    base = TreeRetriever(small_tree, HashEmbedder(), RetrievalConfig(top_k=5)).retrieve(
        "rocket orbit"
    )
    same = TreeRetriever(
        small_tree, HashEmbedder(), RetrievalConfig(top_k=5, candidates=10)
    ).retrieve("rocket orbit")
    assert [r.node.index for r in base.nodes] == [r.node.index for r in same.nodes]


def test_hybrid_mmr_and_rerank(small_tree):
    q = "astronaut comet launch"
    dense = TreeRetriever(small_tree, HashEmbedder(), RetrievalConfig(top_k=6)).retrieve(q)
    hybrid = TreeRetriever(
        small_tree, HashEmbedder(), RetrievalConfig(top_k=6, hybrid=True)
    ).retrieve(q)
    assert len(hybrid.nodes) == 6
    assert {r.node.index for r in hybrid.nodes} & {r.node.index for r in dense.nodes}

    class ReverseReranker:
        calls = 0

        def score(self, query, passages):
            ReverseReranker.calls += 1
            return -np.arange(len(passages), dtype=np.float64)[::-1]  # last candidate scores best

    cfg = RetrievalConfig(top_k=3, reranker="fake", candidates=5)
    reranked = TreeRetriever(small_tree, HashEmbedder(), cfg, reranker=ReverseReranker()).retrieve(
        q
    )
    dense5 = TreeRetriever(small_tree, HashEmbedder(), RetrievalConfig(top_k=5)).retrieve(q)
    assert ReverseReranker.calls == 1
    assert [r.node.index for r in reranked.nodes] == [r.node.index for r in dense5.nodes][::-1][:3]

    diverse = TreeRetriever(
        small_tree, HashEmbedder(), RetrievalConfig(top_k=6, mmr_lambda=0.3)
    ).retrieve(q)
    assert len(diverse.nodes) == 6
