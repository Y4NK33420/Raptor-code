"""QuALITY stories and questions for the Library picker and the Arena."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..eval.datasets.quality import QualityArticle, QualityQuestion, load_quality


class QualityStore:
    def __init__(self, data_dir: Path, split: str = "dev") -> None:
        self.data_dir = data_dir
        self.split = split
        self._articles: list[QualityArticle] | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return (self.data_dir / f"QuALITY.v1.0.1.htmlstripped.{self.split}").is_file()

    def articles(self) -> list[QualityArticle]:
        with self._lock:
            if self._articles is None:
                self._articles = load_quality(self.split, self.data_dir) if self.available else []
            return self._articles

    def get(self, set_id: str) -> QualityArticle:
        for a in self.articles():
            if a.set_id == set_id:
                return a
        raise KeyError(set_id)

    def by_position(self, position: int) -> dict[str, Any] | None:
        arts = self.articles()
        if 0 <= position < len(arts):
            a = arts[position]
            return {"set_id": a.set_id, "title": a.title}
        return None

    def question(
        self, question_id: str, set_id: str | None = None
    ) -> tuple[QualityArticle, QualityQuestion]:
        pool = [self.get(set_id)] if set_id else self.articles()
        for a in pool:
            for q in a.questions:
                if q.question_id == question_id:
                    return a, q
        raise KeyError(question_id)

    @staticmethod
    def story_summary(a: QualityArticle) -> dict[str, Any]:
        return {
            "set_id": a.set_id,
            "article_id": a.article_id,
            "title": a.title,
            "author": a.author,
            "topic": a.topic,
            "word_count": a.word_count,
            "n_questions": len(a.questions),
            "n_hard": sum(q.difficult for q in a.questions),
        }

    @staticmethod
    def question_dict(q: QualityQuestion) -> dict[str, Any]:
        return {
            "question_id": q.question_id,
            "question": q.question,
            "options": q.options,
            "gold": q.gold_label,
            "difficult": q.difficult,
        }
