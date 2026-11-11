"""Fixed, seeded evaluation splits over the QuALITY dev file.

QuALITY has 230 question sets over 115 articles in its dev file: each article appears in
about two sets with different questions. A split is a list of ``set_id`` values. The
``dev`` subset (used for sweeps) and the held-out ``confirm`` subset use disjoint
*articles*, so nothing tuned on dev leaks into the confirm numbers.

The split file is committed (``configs/splits.yaml``) so every run uses the same
questions; :func:`make_splits` only regenerates it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

from .datasets.quality import DEFAULT_DIR, QualityArticle, load_quality

DEFAULT_SPLITS_FILE = Path("configs/splits.yaml")
DEFAULT_SEED = 561
DEFAULT_SIZE = 8


@dataclass
class SplitSpec:
    name: str
    source: str  # QuALITY file split ("dev")
    set_ids: list[str]


def make_splits(
    articles: list[QualityArticle],
    *,
    seed: int = DEFAULT_SEED,
    size: int = DEFAULT_SIZE,
    source: str = "dev",
) -> dict[str, SplitSpec]:
    """Pick ``size`` sets for ``dev`` and ``size`` for ``confirm``, from disjoint articles.

    Only sets with gold labels are eligible. One set is taken per article.
    """
    by_article: dict[str, list[QualityArticle]] = {}
    for art in articles:
        if art.questions and all(q.gold_label is not None for q in art.questions):
            by_article.setdefault(art.article_id, []).append(art)
    article_ids = sorted(by_article)
    if len(article_ids) < 2 * size:
        raise ValueError(f"need {2 * size} articles, found {len(article_ids)}")
    rng = random.Random(seed)
    picked = rng.sample(article_ids, 2 * size)
    chosen = [rng.choice(sorted(by_article[a], key=lambda s: s.set_id)).set_id for a in picked]
    return {
        "dev": SplitSpec("dev", source, chosen[:size]),
        "confirm": SplitSpec("confirm", source, chosen[size:]),
    }


def save_splits(
    splits: dict[str, SplitSpec], path: str | Path = DEFAULT_SPLITS_FILE, *, seed: int
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "seed": seed,
        "dataset": "QuALITY v1.0.1 (htmlstripped)",
        "splits": {name: {"source": s.source, "set_ids": s.set_ids} for name, s in splits.items()},
    }
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def load_splits(path: str | Path = DEFAULT_SPLITS_FILE) -> dict[str, SplitSpec]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {
        name: SplitSpec(name, spec.get("source", "dev"), list(spec["set_ids"]))
        for name, spec in doc["splits"].items()
    }


def load_split_articles(
    name: str,
    *,
    splits_file: str | Path = DEFAULT_SPLITS_FILE,
    data_dir: str | Path = DEFAULT_DIR,
) -> list[QualityArticle]:
    """The question sets of split ``name``, in split order."""
    splits = load_splits(splits_file)
    if name not in splits:
        raise ValueError(f"unknown split {name!r}; available: {sorted(splits)}")
    spec = splits[name]
    by_id = {a.set_id: a for a in load_quality(spec.source, data_dir)}
    missing = [s for s in spec.set_ids if s not in by_id]
    if missing:
        raise ValueError(f"split {name!r} references unknown set ids: {missing}")
    return [by_id[s] for s in spec.set_ids]
