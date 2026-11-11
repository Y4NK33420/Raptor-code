"""2-D projection of all node embeddings for the Embedding Map (cached per index)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..types import Tree

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


def _pca2(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    if len(x) < 2:
        return np.zeros((len(x), 2), dtype=np.float32)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    out = centered @ vt[:2].T
    if out.shape[1] < 2:
        out = np.pad(out, ((0, 0), (0, 2 - out.shape[1])))
    return out


def _umap2(x: np.ndarray) -> np.ndarray:
    import umap

    n_neighbors = max(2, min(15, len(x) - 1))
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=0.15, metric="cosine", random_state=42
    )
    return np.asarray(reducer.fit_transform(x))


def project(tree: Tree, method: str = "umap") -> dict[str, Any]:
    indices, matrix = tree.normalized_matrix()
    if method == "umap" and len(indices) >= 6:
        try:
            coords = _umap2(matrix)
        except Exception:  # pragma: no cover - UMAP can fail on degenerate inputs
            logger.exception("UMAP failed; falling back to PCA")
            coords, method = _pca2(matrix), "pca"
    else:
        coords, method = _pca2(matrix), "pca"
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    coords = (coords - lo) / span  # normalise to [0, 1]

    parents: dict[int, list[int]] = {i: [] for i in indices}
    for n in tree.all_nodes.values():
        for c in n.children:
            parents.setdefault(c, []).append(n.index)
    points = []
    for row, idx in enumerate(indices):
        node = tree.all_nodes[idx]
        ps = sorted(parents.get(idx, []))
        points.append(
            {
                "id": idx,
                "x": round(float(coords[row, 0]), 5),
                "y": round(float(coords[row, 1]), 5),
                "layer": node.layer,
                "parents": ps,
                "cluster": ps[0] if ps else None,
            }
        )
    return {"method": method, "points": points}


def cached_projection(
    tree: Tree, cache_dir: Path, key: str, method: str = "umap"
) -> dict[str, Any]:
    path = cache_dir / "projections" / f"{key}-{method}.json"
    if path.is_file():
        return dict(json.loads(path.read_text(encoding="utf-8")))
    with _LOCK:
        if path.is_file():
            return dict(json.loads(path.read_text(encoding="utf-8")))
        data = project(tree, method)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return data
