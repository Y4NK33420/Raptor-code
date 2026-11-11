"""Interim Streamlit dashboard: Flat RAG vs RAPTOR on QuALITY.

Run with:  streamlit run app.py
(This is replaced by the React UI later; the API key comes from .env / the environment.)
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from raptor import FlatRAG, RaptorConfig, RetrievalAugmentation
from raptor.eval.datasets import load_quality
from raptor.models import ModelBundle, open_cache, with_cache
from raptor.models.gemini import GeminiEmbeddingModel, GeminiQAModel, GeminiSummarizationModel
from raptor.settings import get_settings

TREES = Path("demo/trees")
THEMATIC = Path("demo/results/thematic_multi_hop_results.json")
PAPER = "https://arxiv.org/abs/2401.18059"
N_STORIES = 5

st.set_page_config(page_title="RAPTOR vs Flat RAG", page_icon="🌳", layout="wide")
st.title("🌳 Hierarchical RAG with RAPTOR vs Flat RAG")
st.caption("Comparative evaluation and live demo on the NYU QuALITY benchmark")


@st.cache_data
def stories():
    try:
        return load_quality("dev", limit=N_STORIES)
    except FileNotFoundError:
        return []


@st.cache_resource
def models(legacy_embeddings: bool) -> ModelBundle:
    # Demo trees converted from the original code were embedded without task types.
    task = {"document_task_type": None, "query_task_type": None} if legacy_embeddings else {}
    bundle = ModelBundle(GeminiEmbeddingModel(**task), GeminiSummarizationModel(), GeminiQAModel())
    return with_cache(bundle, open_cache(get_settings().cache_dir))


def config(top_k: int) -> RaptorConfig:
    cfg = RaptorConfig()
    cfg.chunking.max_tokens = 250
    cfg.builder.max_layers = 3
    cfg.retrieval.top_k = top_k
    return cfg


def is_legacy(tree_dir: Path) -> bool:
    head = (tree_dir / "tree.json").read_text(encoding="utf-8")[:4000]
    return "doc=None" in head


all_stories = stories()
if not all_stories:
    st.error("QuALITY data not found. Run `python scripts/download_quality.py` first.")
    st.stop()

st.sidebar.header("📁 Document")
idx = st.sidebar.selectbox(
    "QuALITY story",
    range(len(all_stories)),
    format_func=lambda i: (
        f"Story {i + 1}: {all_stories[i].title} ({all_stories[i].word_count} words)"
    ),
)
story = all_stories[idx]
top_k = st.sidebar.slider("Retrieval budget (top-k nodes)", 3, 25, 15)
st.sidebar.info(
    "**Reader**: Gemini 3.5 Flash Lite\n\n**Embedder**: Gemini Embedding 001\n\n"
    "**Clustering**: UMAP + GMM (BIC-selected k)"
)

tree_dir = TREES / f"quality_story_{idx + 1}_gemini"
ra: RetrievalAugmentation | None = None
if (tree_dir / "tree.json").exists():
    m = models(is_legacy(tree_dir))
    ra = RetrievalAugmentation(
        config(top_k),
        tree=tree_dir,
        embedding_model=m.embedding,
        summarization_model=m.summarization,
        qa_model=m.qa,
    )
else:
    st.warning(f"The RAPTOR tree for '{story.title}' is not built yet.")
    if st.button(f"🔨 Build tree for story {idx + 1}", type="primary"):
        m = models(False)
        builder = RetrievalAugmentation(
            config(top_k),
            embedding_model=m.embedding,
            summarization_model=m.summarization,
            qa_model=m.qa,
        )
        status = st.status("Building…", expanded=True)
        builder.add_documents(story.article, progress=lambda ev: status.write(ev))
        builder.save(tree_dir)
        status.update(label="Tree built", state="complete")
        st.rerun()

tab_qa, tab_thematic, tab_tree, tab_story = st.tabs(
    ["⚔️ Multiple-choice arena", "🌟 Thematic examples (precomputed)", "🌲 Tree", "📖 Story"]
)

with tab_qa:
    st.subheader(f"QA arena: {story.title}")
    if story.questions:
        qi = st.selectbox(
            "Benchmark question",
            range(len(story.questions)),
            format_func=lambda i: f"Q{i + 1}: {story.questions[i].question}",
        )
        q = story.questions[qi]
        cols = st.columns(2)
        for i, opt in enumerate(q.options):
            gold = " ✅ **[gold]**" if i + 1 == q.gold_label else ""
            cols[i % 2].markdown(f"> **Option {i + 1}**: {opt}{gold}")

        if st.button("🚀 Run side-by-side comparison", type="primary", disabled=ra is None) and ra:
            prompt = q.mcq_prompt()
            left, right = st.columns(2)
            with left:
                st.markdown("### 📄 Flat RAG")
                st.caption("Dense retrieval over the leaf chunks only.")
                flat = FlatRAG.from_tree(ra.tree, ra.embedding_model, ra.qa_model, top_k=top_k)
                with st.spinner("Retrieving…"):
                    ctx, _ = flat.retrieve(q.question)
                    st.success(ra.qa_model.answer_question(ctx, prompt))
                with st.expander(f"Retrieved chunks (top {top_k})"):
                    st.text(ctx[:3000])
            with right:
                st.markdown("### 🌳 RAPTOR")
                st.caption("Collapsed-tree retrieval over summaries and leaves.")
                with st.spinner("Retrieving…"):
                    res = ra.retrieve_nodes(q.question, top_k=top_k)
                    st.success(ra.qa_model.answer_question(res.context, prompt))
                st.caption(f"Nodes per layer: {res.layer_counts()}")
                with st.expander("Retrieved multi-layer context"):
                    st.text(res.context[:3000])

with tab_thematic:
    st.subheader("Thematic and multi-hop questions")
    st.info(
        "These answers were generated earlier and saved to "
        f"`{THEMATIC}`; they are not re-run here. The status labels in the table are "
        "qualitative judgements, not computed metrics. See the "
        f"[RAPTOR paper]({PAPER}) for the motivation."
    )
    if THEMATIC.exists():
        examples = json.loads(THEMATIC.read_text(encoding="utf-8"))
        ti = st.selectbox(
            "Task",
            range(len(examples)),
            format_func=lambda i: (
                f"Q{examples[i]['id']}: [{examples[i]['category']}] {examples[i]['question']}"
            ),
        )
        ex = examples[ti]
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Corpus**: {ex['corpus']}")
        c2.warning(f"**Type**: {ex['category']}")
        c3.success(f"**RAPTOR summary nodes**: {ex['raptor']['summary_percentage']:.0f}%")
        st.markdown(f"**Question**: *{ex['question']}*")
        st.caption(f"Evaluation objective: {ex['eval_criteria']}")
        lc, rc = st.columns(2)
        lc.markdown("#### Flat RAG")
        lc.markdown(f"> {ex['flat_rag']['answer']}")
        rc.markdown("#### RAPTOR")
        rc.markdown(f"> {ex['raptor']['answer']}")

with tab_tree:
    st.subheader(f"Tree structure: {story.title}")
    if ra and ra.tree:
        tree = ra.tree
        cols = st.columns(len(tree.layer_to_nodes) + 1)
        cols[0].metric("Total nodes", len(tree.all_nodes))
        for i, layer in enumerate(sorted(tree.layer_to_nodes), start=1):
            label = "Leaves" if layer == 0 else f"Layer {layer} summaries"
            cols[i].metric(label, len(tree.layer_to_nodes[layer]))
        for layer in sorted(tree.layer_to_nodes, reverse=True):
            st.markdown(f"### Layer {layer}" + (" (leaf chunks, first 3)" if layer == 0 else ""))
            nodes = tree.layer_to_nodes[layer]
            for node in nodes[:3] if layer == 0 else nodes:
                title = f"Node {node.index}" + (
                    f" ← children {sorted(node.children)}" if node.children else ""
                )
                with st.expander(title):
                    st.write(node.text)
    else:
        st.info("Build the tree to inspect it.")

with tab_story:
    st.subheader(story.title)
    st.caption(f"Author: {story.author} | Topic: {story.topic}")
    st.text_area("Full text", story.article, height=400)
