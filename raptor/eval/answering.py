"""Multiple-choice answering for QuALITY.

The reader returns structured JSON ``{"option": int, "rationale": str}`` (Gemini's JSON
mode with a response schema). Models without JSON mode get the same prompt as plain
text, and the option is extracted with a regex fallback. Every reply goes through the
disk cache, so re-running an experiment costs nothing.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ..models.base import BaseQAModel
from ..models.cache import CacheStats, DiskCache

PROMPT_VERSION = "mcq-v1"

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "option": {"type": "INTEGER", "description": "The number of the chosen option."},
        "rationale": {"type": "STRING", "description": "One or two sentences of justification."},
    },
    "required": ["option", "rationale"],
    "propertyOrdering": ["rationale", "option"],
}

MCQ_PROMPT = (
    "You answer multiple-choice questions about a long story. You are given excerpts and "
    "summaries retrieved from the story, not the whole text.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Choose the single best option (1-{n}) based on the context. If the context does not "
    "settle it, choose the most plausible option anyway. Reply as JSON: "
    '{{"rationale": "<one or two sentences>", "option": <number>}}'
)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_JSON_OPTION = re.compile(r'"option"\s*:\s*"?(\d+)')
_TEXT_OPTION = re.compile(r"\boption\s*[#:]?\s*\**\s*\(?(\d+)", re.IGNORECASE)
_BARE_DIGIT = re.compile(r"\b([1-9])\b")


@dataclass
class MCQAnswer:
    option: int | None  # 1-based; None if nothing could be parsed
    rationale: str
    raw: str
    parsed_by: str  # "json" | "regex" | "none"
    cached: bool


def format_prompt(context: str, question: str, options: Sequence[str]) -> str:
    opts = "\n".join(f"{i}. {o.strip()}" for i, o in enumerate(options, 1))
    return MCQ_PROMPT.format(context=context, question=question, options=opts, n=len(options))


def parse_answer(raw: str, n_options: int) -> tuple[int | None, str, str]:
    """Return ``(option, rationale, parsed_by)`` from a model reply."""
    text = _FENCE.sub("", raw.strip())
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        data = None
    if isinstance(data, dict):
        try:
            opt = int(data.get("option"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            opt = None
        if opt is not None and 1 <= opt <= n_options:
            return opt, str(data.get("rationale", "")).strip(), "json"
    for pattern in (_JSON_OPTION, _TEXT_OPTION, _BARE_DIGIT):
        for m in pattern.finditer(text):
            opt = int(m.group(1))
            if 1 <= opt <= n_options:
                return opt, text, "regex"
    return None, text, "none"


class MCQAnswerer:
    """Answers QuALITY questions with a reader model, caching every reply on disk."""

    def __init__(
        self,
        generate: Callable[[str], str | None],
        model_id: str,
        cache: DiskCache | None = None,
    ) -> None:
        self.generate = generate
        self.model_id = model_id
        self.cache = cache
        self.stats = CacheStats()
        self.calls = 0  # reader calls actually made (cache misses)
        self._lock = threading.Lock()

    @classmethod
    def from_qa_model(cls, qa: BaseQAModel, cache: DiskCache | None = None) -> MCQAnswerer:
        """Use JSON mode when the model has it (Gemini), else plain text + regex."""
        inner = getattr(qa, "inner", qa)  # unwrap CachedQAModel: we cache here instead
        gen_json = getattr(inner, "generate_json", None)
        if callable(gen_json):
            return cls(lambda p: gen_json(p, ANSWER_SCHEMA), f"{inner.model_id}:json", cache)
        return cls(lambda p: inner.answer_question("", p), f"{inner.model_id}:text", cache)

    def answer(self, context: str, question: str, options: Sequence[str]) -> MCQAnswer:
        prompt = format_prompt(context, question, options)
        key = DiskCache.key("mcq", PROMPT_VERSION, self.model_id, prompt)
        cached = False
        raw: str | None = None
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                raw, cached = hit.decode("utf-8"), True
        if cached:
            with self._lock:
                self.stats.hits += 1
        else:
            with self._lock:
                self.stats.misses += 1
                self.calls += 1
            raw = self.generate(prompt) or ""
            if self.cache is not None:
                self.cache.set(key, raw.encode("utf-8"))
        option, rationale, how = parse_answer(raw or "", len(options))
        return MCQAnswer(option, rationale, raw or "", how, cached)
