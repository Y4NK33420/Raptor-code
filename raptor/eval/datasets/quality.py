"""QuALITY (Pang et al., NAACL 2022) loader: long-document multiple-choice QA.

Get the data with ``python scripts/download_quality.py`` (writes to ``data/quality``).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_DIR = Path("data/quality")
FILE_PATTERN = "QuALITY.v1.0.1.htmlstripped.{split}"


@dataclass
class QualityQuestion:
    question_id: str
    question: str
    options: list[str]
    gold_label: int | None  # 1-based; None on the test split
    difficult: bool = False

    def mcq_prompt(self) -> str:
        opts = "\n".join(f"{i}. {o.strip()}" for i, o in enumerate(self.options, 1))
        return (
            f"Question: {self.question}\n\nOptions:\n{opts}\n\n"
            "Based strictly on the provided context, select the single best option "
            f"(1-{len(self.options)}). Start your response with 'Option X' and then justify it."
        )


@dataclass
class QualityArticle:
    set_id: str  # unique per (article, question set); the same article appears in several sets
    article_id: str
    title: str
    author: str
    article: str
    topic: str = ""
    questions: list[QualityQuestion] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return len(self.article.split())


def _parse(row: dict[str, Any]) -> QualityArticle:
    questions = [
        QualityQuestion(
            question_id=q.get("question_unique_id", f"{row['article_id']}-{i}"),
            question=q["question"],
            options=q["options"],
            gold_label=q.get("gold_label"),
            difficult=bool(q.get("difficult", 0)),
        )
        for i, q in enumerate(row.get("questions", []))
    ]
    return QualityArticle(
        set_id=row.get("set_unique_id", row["article_id"]),
        article_id=row["article_id"],
        title=row.get("title", ""),
        author=row.get("author", ""),
        article=row["article"],
        topic=row.get("topic", ""),
        questions=questions,
    )


def iter_quality(
    split: str = "dev", data_dir: str | Path = DEFAULT_DIR
) -> Iterator[QualityArticle]:
    path = Path(data_dir) / FILE_PATTERN.format(split=split)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_quality.py` first."
        )
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield _parse(json.loads(line))


def load_quality(
    split: str = "dev", data_dir: str | Path = DEFAULT_DIR, limit: int | None = None
) -> list[QualityArticle]:
    out = []
    for i, art in enumerate(iter_quality(split, data_dir)):
        if limit is not None and i >= limit:
            break
        out.append(art)
    return out
