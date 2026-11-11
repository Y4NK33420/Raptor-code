from pathlib import Path

import numpy as np
import pytest
from scipy import spatial

from raptor import FlatRAG, Node, RetrievalConfig, Tree, TreeRetriever, load_legacy_pickle
from raptor.chunking import default_tokenizer
from raptor.models.base import BaseEmbeddingModel

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_cinderella.pkl"


class FixedQuery(BaseEmbeddingModel):
    """Returns a preset query vector (lets tests drive retrieval without a real model)."""

    model_id = "test:fixed"

    def __init__(self, vec):
        self.vec = np.asarray(vec, dtype=np.float32)

    def embed_documents(self, texts):
        return np.stack([self.vec for _ in texts])


def toy_tree() -> Tree:
    def node(i, vec, layer=0, children=(), text=None):
        return Node(
            text or f"node {i}", i, set(children), {"EMB": np.asarray(vec, np.float32)}, layer
        )

    leaves = [node(0, [1, 0, 0]), node(1, [0.9, 0.1, 0]), node(2, [0, 1, 0]), node(3, [0, 0, 1])]
    tops = [node(4, [1, 0.05, 0], 1, (0, 1)), node(5, [0, 0.7, 0.7], 1, (2, 3))]
    return Tree.from_layers({0: leaves, 1: tops})


def test_collapsed_ranks_by_similarity_and_honours_top_k():
    r = TreeRetriever(toy_tree(), FixedQuery([1, 0, 0]))
    res = r.retrieve("q", top_k=3)
    assert [n.node.index for n in res.nodes] == [0, 4, 1]
    assert res.nodes[0].score == pytest.approx(1.0)
    assert res.layer_counts() == {0: 2, 1: 1}


def test_collapsed_token_budget_stops_early():
    r = TreeRetriever(toy_tree(), FixedQuery([1, 0, 0]))
    one = len(default_tokenizer().encode("node 0"))
    res = r.retrieve("q", top_k=10, max_tokens=one * 2)
    assert len(res.nodes) == 2


def test_threshold_keeps_similar_nodes_not_distant_ones():
    cfg = RetrievalConfig(selection_mode="threshold", threshold=0.8)
    res = TreeRetriever(toy_tree(), FixedQuery([1, 0, 0]), cfg).retrieve("q", top_k=10)
    assert {n.node.index for n in res.nodes} == {0, 1, 4}
    assert all(n.score >= 0.8 for n in res.nodes)


def test_traversal_descends_into_children_of_best_parent():
    res = TreeRetriever(toy_tree(), FixedQuery([0, 0, 1])).retrieve("q", mode="traversal", top_k=1)
    assert [n.node.index for n in res.nodes] == [5, 3]
    with pytest.raises(ValueError):
        TreeRetriever(toy_tree(), FixedQuery([0, 0, 1])).retrieve(
            "q", mode="traversal", start_layer=5
        )


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        TreeRetriever(toy_tree(), FixedQuery([1, 0, 0])).retrieve("  ")


def _old_collapsed(tree_dict, query, top_k, max_tokens):
    """The original RAPTOR collapsed-tree algorithm (scipy distances + argsort)."""
    tok = default_tokenizer()
    nodes = [tree_dict[i] for i in sorted(tree_dict)]
    d = [spatial.distance.cosine(query, n.embeddings["OpenAI"]) for n in nodes]
    out, total = [], 0
    for idx in np.argsort(d)[:top_k]:
        t = len(tok.encode(nodes[idx].text))
        if total + t > max_tokens:
            break
        out.append(nodes[idx].index)
        total += t
    return out


def test_matches_original_algorithm_on_legacy_tree():
    tree = load_legacy_pickle(FIXTURE)
    assert tree.embedding_key == "OpenAI"
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = rng.normal(size=len(tree.all_nodes[0].embeddings["OpenAI"]))
        new = TreeRetriever(tree, FixedQuery(q)).retrieve("q", top_k=10, max_tokens=3500)
        assert [n.node.index for n in new.nodes] == _old_collapsed(tree.all_nodes, q, 10, 3500)


def test_flat_rag_from_tree_uses_leaves_only():
    tree = toy_tree()
    flat = FlatRAG.from_tree(tree, FixedQuery([0, 0, 1]), top_k=2)
    ctx, idx = flat.retrieve("q")
    assert idx[0] == 3 and "node 3" in ctx
    assert len(flat.chunks) == 4
