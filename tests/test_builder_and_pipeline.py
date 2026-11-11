import numpy as np
import pytest

from raptor import RaptorConfig, RetrievalAugmentation, RetrievalAugmentationConfig, TreeBuilder

pytestmark = pytest.mark.slow


def small_config(**over) -> RaptorConfig:
    cfg = RaptorConfig()
    cfg.chunking.max_tokens = 40
    cfg.clustering.reduction_dimension = 5
    cfg.builder.max_layers = 3
    cfg.builder.summarization_length = 30
    for k, v in over.items():
        section, field = k.split("__")
        setattr(getattr(cfg, section), field, v)
    return cfg


def test_tree_structure_is_consistent(document, embedder, summarizer):
    events = []
    tree = TreeBuilder(small_config(), embedder, summarizer).build_from_text(
        document, progress=events.append
    )
    tree.validate()
    assert tree.num_layers >= 1
    assert set(tree.layer_to_nodes) == set(range(tree.num_layers + 1))
    for layer in range(1, tree.num_layers + 1):
        below = {n.index for n in tree.layer_to_nodes[layer - 1]}
        for node in tree.layer_to_nodes[layer]:
            assert node.children and node.children <= below
    # every lower-layer node has at least one parent
    for layer in range(tree.num_layers):
        parents = {c for n in tree.layer_to_nodes[layer + 1] for c in n.children}
        assert {n.index for n in tree.layer_to_nodes[layer]} <= parents
    assert tree.root_nodes == {n.index: n for n in tree.layer_to_nodes[tree.num_layers]}
    assert events[0]["stage"] == "chunking" and events[-1]["stage"] == "done"
    # leaves carry their source offsets
    leaf = tree.leaf_nodes[0]
    assert document[leaf.metadata["start"] : leaf.metadata["end"]] == leaf.text
    # embeddings are batched: one call per layer, not one per chunk
    assert embedder.doc_calls == tree.num_layers + 1


def test_builder_reuse_does_not_leak_layer_count(document, embedder, summarizer):
    builder = TreeBuilder(small_config(), embedder, summarizer)
    short = builder.build_from_text("One sentence only. And another.")
    assert short.num_layers == 0
    full = builder.build_from_text(document)
    assert full.num_layers >= 1  # the old builder kept the reduced count from the first build


def test_pipeline_build_query_save_load(tmp_path, document, embedder, summarizer, qa):
    ra = RetrievalAugmentation(
        small_config(), embedding_model=embedder, summarization_model=summarizer, qa_model=qa
    )
    ra.add_documents(document)
    with pytest.raises(ValueError, match="already exists"):
        ra.add_documents(document)
    ra.add_documents(document, overwrite=True)

    ctx, layers = ra.retrieve("rocket orbit astronaut", top_k=5, return_layer_information=True)
    assert "rocket" in ctx.lower() or "orbit" in ctx.lower()
    assert len(layers) <= 5 and all({"node_index", "layer_number"} <= set(d) for d in layers)
    assert ra.answer_question("rocket?").startswith("Q=rocket?")

    ra.save(tmp_path / "idx")
    ra2 = RetrievalAugmentation(
        small_config(), tree=tmp_path / "idx", embedding_model=embedder, qa_model=qa
    )
    assert ra2.retrieve("rocket orbit astronaut", top_k=5) == ctx


def test_add_to_existing_reuses_leaf_embeddings(document, embedder, summarizer, qa):
    ra = RetrievalAugmentation(
        small_config(), embedding_model=embedder, summarization_model=summarizer, qa_model=qa
    )
    ra.add_documents(document)
    n_leaves = len(ra.tree.leaf_nodes)
    before = embedder.texts_embedded
    extra = "Dragons guard the castle gate. The knight rides at dawn."
    ra.add_to_existing(extra)
    assert len(ra.tree.leaf_nodes) == n_leaves + 1
    new_leaf_and_summaries = embedder.texts_embedded - before
    summaries = len(ra.tree.all_nodes) - len(ra.tree.leaf_nodes)
    assert new_leaf_and_summaries == 1 + summaries  # old leaves were not re-embedded
    assert np.allclose(
        ra.tree.leaf_nodes[0].embeddings["EMB"], embedder.vector(ra.tree.leaf_nodes[0].text)
    )


def test_compat_config_maps_old_arguments(embedder, summarizer, qa):
    cfg = RetrievalAugmentationConfig(
        qa_model=qa,
        embedding_model=embedder,
        summarization_model=summarizer,
        tb_max_tokens=250,
        tb_num_layers=3,
        tr_top_k=7,
        tree_builder_type="cluster",
    )
    assert cfg.raptor.chunking.max_tokens == 250
    assert cfg.raptor.builder.max_layers == 3
    assert cfg.raptor.retrieval.top_k == 7
    ra = RetrievalAugmentation(config=cfg)
    assert ra.qa_model is qa and ra.embedding_model is embedder
    with pytest.raises(TypeError):
        RetrievalAugmentationConfig(bogus=1)
