"""The RAPTOR clustering algorithm: UMAP reduction + Gaussian mixture, global then local.

1. Reduce embeddings with UMAP using a wide neighbourhood (global structure) and fit
   a GMM whose component count minimises BIC. Soft-assign each point to every
   component with probability above ``threshold``.
2. Within each global cluster, repeat with a narrow neighbourhood (local structure).
3. Any resulting cluster whose text exceeds ``max_cluster_tokens`` is re-clustered.

Fixes over the original implementation: seeded UMAP (reproducible trees), direct index
tracking instead of an O(n²·d) equality search (which also mis-mapped duplicate
chunks), parameters preserved through recursion, guarded small-n cases, and a
termination guarantee when an oversized cluster cannot be split further.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence

import numpy as np
from sklearn.mixture import GaussianMixture

from .base import CLUSTERERS, ClusteringAlgorithm, check_clusters

logger = logging.getLogger(__name__)


def _umap(embeddings: np.ndarray, dim: int, n_neighbors: int, random_state: int) -> np.ndarray:
    import umap  # imported lazily: numba JIT makes the import slow

    n = len(embeddings)
    n_neighbors = max(2, min(n_neighbors, n - 1))
    dim = max(1, min(dim, n - 2))
    kwargs = {}
    if "compatibility_layout" in inspect.signature(umap.UMAP.__init__).parameters:
        kwargs["compatibility_layout"] = True  # keep the 0.5.x layout the paper used
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=dim,
        metric="cosine",
        random_state=random_state,
        n_jobs=1,
        **kwargs,
    )
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float64)


def _fit_gmm(x: np.ndarray, n: int, random_state: int) -> GaussianMixture | None:
    """Fit a GMM, returning ``None`` if it is degenerate (e.g. collapsed points)."""
    try:
        return GaussianMixture(n_components=n, random_state=random_state).fit(x)
    except ValueError:
        return None


def _optimal_components(x: np.ndarray, max_clusters: int, random_state: int) -> int:
    upper = min(max_clusters, len(x))
    best_n, best_bic = 1, np.inf
    for n in range(1, upper):
        gm = _fit_gmm(x, n, random_state)
        if gm is None:
            continue
        bic = gm.bic(x)
        if bic < best_bic:
            best_n, best_bic = n, bic
    return best_n


def _gmm_soft(
    x: np.ndarray, threshold: float, max_clusters: int, random_state: int
) -> list[list[int]]:
    """Soft-assign rows of ``x`` to GMM components. Returns clusters of row indices."""
    x = np.asarray(x, dtype=np.float64)
    n_comp = _optimal_components(x, max_clusters, random_state)
    gm = _fit_gmm(x, n_comp, random_state)
    if gm is None:  # even one component failed: treat everything as one cluster
        return [list(range(len(x)))]
    probs = gm.predict_proba(x)
    clusters: list[list[int]] = [[] for _ in range(n_comp)]
    for row, p in enumerate(probs):
        labels = np.where(p > threshold)[0]
        if len(labels) == 0:  # keep every point somewhere
            labels = np.array([int(np.argmax(p))])
        for label in labels:
            clusters[int(label)].append(row)
    return [c for c in clusters if c]


class RaptorGMMClustering(ClusteringAlgorithm):
    def __init__(
        self,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        max_cluster_tokens: int = 3500,
        max_clusters: int = 50,
        random_state: int = 224,
        global_neighbors: int | None = None,
        local_neighbors: int = 10,
    ) -> None:
        self.reduction_dimension = reduction_dimension
        self.threshold = threshold
        self.max_cluster_tokens = max_cluster_tokens
        self.max_clusters = max_clusters
        self.random_state = random_state
        self.global_neighbors = global_neighbors
        self.local_neighbors = local_neighbors

    # -------------------------------------------------------------- public API
    def cluster(self, embeddings: np.ndarray, token_counts: Sequence[int]) -> list[list[int]]:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        n = len(embeddings)
        if n != len(token_counts):
            raise ValueError("embeddings and token_counts differ in length")
        if n == 0:
            return []
        clusters = self._cluster_with_limit(np.arange(n), embeddings, np.asarray(token_counts))
        return check_clusters(clusters, n)

    # ---------------------------------------------------------------- internals
    def _cluster_with_limit(
        self, idx: np.ndarray, embeddings: np.ndarray, tokens: np.ndarray
    ) -> list[list[int]]:
        result: list[list[int]] = []
        for members in self._two_level(embeddings[idx]):
            group = idx[members]
            if len(group) == 1 or tokens[group].sum() <= self.max_cluster_tokens:
                result.append(group.tolist())
            elif len(group) == len(idx):
                # Re-clustering made no progress; split in order to guarantee termination.
                half = len(group) // 2
                result.extend(self._cluster_with_limit(group[:half], embeddings, tokens))
                result.extend(self._cluster_with_limit(group[half:], embeddings, tokens))
            else:
                result.extend(self._cluster_with_limit(group, embeddings, tokens))
        return result

    def _two_level(self, emb: np.ndarray) -> list[np.ndarray]:
        """Global-then-local clustering of ``emb``; returns clusters of local row indices."""
        n = len(emb)
        dim = self.reduction_dimension
        if n <= 3:  # too few points for UMAP; the caller splits further if needed
            return [np.arange(n)]
        neighbors = self.global_neighbors or int((n - 1) ** 0.5)
        reduced = _umap(emb, dim, neighbors, self.random_state)
        global_clusters = _gmm_soft(reduced, self.threshold, self.max_clusters, self.random_state)

        out: list[np.ndarray] = []
        for g in global_clusters:
            g_idx = np.asarray(g)
            if len(g_idx) <= dim + 1:
                out.append(g_idx)
                continue
            local_reduced = _umap(emb[g_idx], dim, self.local_neighbors, self.random_state)
            for local in _gmm_soft(
                local_reduced, self.threshold, self.max_clusters, self.random_state
            ):
                out.append(g_idx[np.asarray(local)])
        return out


@CLUSTERERS.register("raptor_gmm")
def _make_raptor_gmm(
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    max_cluster_tokens: int = 3500,
    random_state: int = 224,
    **params: object,
) -> ClusteringAlgorithm:
    return RaptorGMMClustering(
        reduction_dimension=reduction_dimension,
        threshold=threshold,
        max_cluster_tokens=max_cluster_tokens,
        random_state=random_state,
        **params,  # type: ignore[arg-type]
    )
