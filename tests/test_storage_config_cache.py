import io
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from raptor import Node, RaptorConfig, Tree, load_legacy_pickle, load_tree, save_tree
from raptor.models import CachedEmbeddingModel, CachedSummarizationModel, DiskCache

FIXTURE = Path(__file__).parent / "fixtures" / "legacy_cinderella.pkl"


def tree() -> Tree:
    a = Node(
        "alpha. text", 0, set(), {"EMB": np.array([1, 0], np.float32)}, 0, {"start": 0, "end": 11}
    )
    b = Node("beta", 1, set(), {"EMB": np.array([0, 1], np.float32)}, 0)
    c = Node("summary", 2, {0, 1}, {"EMB": np.array([1, 1], np.float32)}, 1)
    return Tree.from_layers({0: [a, b], 1: [c]}, {"embedding_key": "EMB", "note": "ü"})


def test_round_trip(tmp_path):
    t = tree()
    save_tree(t, tmp_path / "idx")
    back = load_tree(tmp_path / "idx")
    assert back.num_layers == 1 and back.metadata["note"] == "ü"
    assert back.all_nodes[2].children == {0, 1}
    assert back.leaf_nodes[0].metadata == {"start": 0, "end": 11}
    assert np.allclose(back.all_nodes[2].embeddings["EMB"], [1, 1])
    save_tree(back, tmp_path / "idx")  # overwrite in place
    assert not list(tmp_path.glob(".tmp-*"))


def test_rejects_wrong_version(tmp_path):
    save_tree(tree(), tmp_path / "idx")
    doc = json.loads((tmp_path / "idx" / "tree.json").read_text(encoding="utf-8"))
    doc["format_version"] = 99
    (tmp_path / "idx" / "tree.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_tree(tmp_path / "idx")


def test_legacy_import_and_load_tree_accepts_pickle_file():
    t = load_legacy_pickle(FIXTURE)
    assert {k: len(v) for k, v in t.layer_to_nodes.items()} == {0: 33, 1: 7}
    assert t.all_nodes[35].layer == 1
    assert len(load_tree(FIXTURE).all_nodes) == 40


class _Evil:
    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


def test_legacy_loader_refuses_arbitrary_globals(tmp_path):
    p = tmp_path / "evil.pkl"
    p.write_bytes(pickle.dumps(_Evil()))
    with pytest.raises(pickle.UnpicklingError):
        load_legacy_pickle(p)


def test_config_yaml_round_trip_and_fingerprint(tmp_path):
    cfg = RaptorConfig()
    cfg.chunking.max_tokens = 250
    cfg.to_yaml(tmp_path / "c.yaml")
    back = RaptorConfig.from_yaml(tmp_path / "c.yaml")
    assert back == cfg and back.fingerprint() == cfg.fingerprint()
    cfg.retrieval.top_k = 3
    assert back.fingerprint(sections=("chunking",)) == cfg.fingerprint(sections=("chunking",))
    assert back.fingerprint() != cfg.fingerprint()
    with pytest.raises(ValueError):
        RaptorConfig.model_validate({"chunking": {"max_tokenz": 5}})


def test_disk_cache_hits_and_persistence(tmp_path, embedder, summarizer):
    cache = DiskCache(tmp_path / "c.sqlite")
    emb = CachedEmbeddingModel(embedder, cache)
    first = emb.embed_documents(["a b", "c d", "a b"])
    assert emb.stats.misses == 2 and emb.stats.hits == 1  # duplicate served from the batch result
    again = CachedEmbeddingModel(embedder, DiskCache(tmp_path / "c.sqlite")).embed_documents(
        ["c d"]
    )
    assert np.allclose(again[0], first[1])
    assert embedder.texts_embedded == 2

    summ = CachedSummarizationModel(summarizer, cache)
    s1 = summ.summarize("One. Two.", 10)
    s2 = summ.summarize("One. Two.", 10)
    assert s1 == s2 and summarizer.calls == 1 and summ.stats.hits == 1
    summ.summarize("One. Two.", 20)  # different params -> miss
    assert summarizer.calls == 2


def test_embedding_subclass_must_implement_something():
    from raptor.models.base import BaseEmbeddingModel

    with pytest.raises(TypeError):

        class Broken(BaseEmbeddingModel):
            pass

    class Legacy(BaseEmbeddingModel):
        def create_embedding(self, text):
            return [float(len(text)), 1.0]

    assert Legacy().embed_documents(["ab", "abc"]).tolist() == [[2.0, 1.0], [3.0, 1.0]]
    assert io  # keep import used
