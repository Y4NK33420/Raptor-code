"""Flat RAG vs RAPTOR on QuALITY questions, using the same leaf chunks for both.

    python examples/quality_compare.py --story 1            # all questions of story 1
    python examples/quality_compare.py --story 1 -q 2 -q 4  # selected questions
    python examples/quality_compare.py --story 3 --build    # build the tree first if missing

This is a quick qualitative tool; systematic evaluation lives in the eval harness.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from raptor import FlatRAG, RaptorConfig, RetrievalAugmentation
from raptor.eval.datasets import load_quality
from raptor.models import ModelBundle, open_cache, with_cache
from raptor.models.gemini import GeminiEmbeddingModel, GeminiQAModel, GeminiSummarizationModel
from raptor.settings import get_settings

TREES = Path("demo/trees")


def models_for(tree_dir: Path) -> ModelBundle:
    legacy = (
        tree_dir.exists()
        and "doc=None" in (tree_dir / "tree.json").read_text(encoding="utf-8")[:4000]
    )
    # Converted demo trees were embedded without task types; query them the same way.
    task = {} if not legacy else {"document_task_type": None, "query_task_type": None}
    bundle = ModelBundle(GeminiEmbeddingModel(**task), GeminiSummarizationModel(), GeminiQAModel())
    return with_cache(bundle, open_cache(get_settings().cache_dir))


def predicted_option(answer: str) -> int | None:
    m = re.search(r"option\s*\**\s*(\d)", answer, re.IGNORECASE)
    return int(m.group(1)) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", type=int, default=1, help="1-based index into the QuALITY dev split")
    ap.add_argument("-q", "--question", type=int, action="append", help="1-based question numbers")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--build", action="store_true", help="build the story's tree if it does not exist"
    )
    args = ap.parse_args()

    story = load_quality("dev", limit=args.story)[-1]
    tree_dir = TREES / f"quality_story_{args.story}_gemini"
    models = models_for(tree_dir)
    cfg = RaptorConfig()
    cfg.chunking.max_tokens = 250
    cfg.builder.max_layers = 3

    if tree_dir.exists():
        ra = RetrievalAugmentation(
            cfg,
            tree=tree_dir,
            embedding_model=models.embedding,
            summarization_model=models.summarization,
            qa_model=models.qa,
        )
    elif args.build:
        ra = RetrievalAugmentation(
            cfg,
            embedding_model=models.embedding,
            summarization_model=models.summarization,
            qa_model=models.qa,
        )
        ra.add_documents(story.article)
        ra.save(tree_dir)
    else:
        raise SystemExit(f"{tree_dir} not found; pass --build to create it")

    assert ra.tree is not None
    flat = FlatRAG.from_tree(ra.tree, models.embedding, models.qa, top_k=args.top_k)
    print(
        f"'{story.title}' ({story.word_count} words): {len(ra.tree.leaf_nodes)} leaves, "
        f"{len(ra.tree.all_nodes)} nodes, top layer {ra.tree.num_layers}\n"
    )

    numbers = args.question or list(range(1, len(story.questions) + 1))
    score = {"flat": 0, "raptor": 0}
    for n in numbers:
        q = story.questions[n - 1]
        prompt = q.mcq_prompt()
        flat_ctx, _ = flat.retrieve(q.question)
        flat_ans = models.qa.answer_question(flat_ctx, prompt)
        res = ra.retrieve_nodes(q.question, top_k=args.top_k)
        raptor_ans = models.qa.answer_question(res.context, prompt)
        fp, rp = predicted_option(flat_ans), predicted_option(raptor_ans)
        score["flat"] += fp == q.gold_label
        score["raptor"] += rp == q.gold_label
        print(
            f"Q{n}: {q.question}\n  gold={q.gold_label}  flat={fp}  raptor={rp}  "
            f"raptor layers={res.layer_counts()}"
        )
    print(
        f"\nAccuracy over {len(numbers)} questions: flat {score['flat']}/{len(numbers)}, "
        f"raptor {score['raptor']}/{len(numbers)}"
    )


if __name__ == "__main__":
    main()
