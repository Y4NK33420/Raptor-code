"""Alternative clustering strategies.

* ``gmm``: Gaussian mixture on the raw (unit-normalised) embeddings, BIC-chosen size, soft.
* ``pca_gmm``: PCA to ``reduction_dimension`` then a soft GMM (single level).
* ``agglomerative``: average-linkage cosine clustering (by distance threshold or size).
* ``hdbscan``: UMAP reduction + HDBSCAN; noise points join the nearest cluster.
* ``kmeans_silhouette``: k-means with ``k`` chosen by cosine silhouette.
* ``contiguous``: segments of *adjacent* nodes, cut where neighbours are least similar.
  This keeps narrative order, which suits stories.

Every strategy respects ``max_cluster_tokens``: an oversized cluster is re-clustered on
its own, and split in order if that makes no progress.
"""

from __future__ import annotations

import itertools
import math
from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np

from .base import CLUSTERERS, ClusteringAlgorithm, check_clusters
from .raptor_gmm import _gmm_soft, _umap


def _unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class TokenLimitedClustering(ClusteringAlgorithm):
    """Base class: implement :meth:`groups`; token limits and validation are handled here."""

    def __init__(self, max_cluster_tokens: int = 3500, random_state: int = 224) -> None:
        self.max_cluster_tokens = max_cluster_tokens
        self.random_state = random_state

    @abstractmethod
    def groups(self, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        """Cluster the rows of ``emb``; return lists of row indices covering every row."""

    def cluster(self, embeddings: np.ndarray, token_counts: Sequence[int]) -> list[list[int]]:
        emb = np.asarray(embeddings, dtype=np.float32)
        n = len(emb)
        if n != len(token_counts):
            raise ValueError("embeddings and token_counts differ in length")
        if n == 0:
            return []
        clusters = self._limited(np.arange(n), emb, np.asarray(token_counts))
        clusters = sorted(check_clusters(clusters, n), key=lambda c: (c[0], len(c)))
        return clusters

    def _limited(self, idx: np.ndarray, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        if len(idx) == 1:
            return [idx.tolist()]
        out: list[list[int]] = []
        for members in self.groups(emb[idx], tokens[idx]):
            group = idx[np.asarray(sorted(set(members)), dtype=np.int64)]
            if len(group) == 1 or tokens[group].sum() <= self.max_cluster_tokens:
                out.append(group.tolist())
            elif len(group) == len(idx):  # no progress: split in order
                half = len(group) // 2
                out.extend(self._limited(group[:half], emb, tokens))
                out.extend(self._limited(group[half:], emb, tokens))
            else:
                out.extend(self._limited(group, emb, tokens))
        return out


# ------------------------------------------------------------------------- GMM
def _gmm_soft_cov(
    x: np.ndarray, threshold: float, max_clusters: int, random_state: int, covariance_type: str
) -> list[list[int]]:
    from sklearn.mixture import GaussianMixture

    def fit(k: int) -> GaussianMixture | None:
        try:
            return GaussianMixture(
                n_components=k, covariance_type=covariance_type, random_state=random_state
            ).fit(x)
        except ValueError:
            return None

    best, best_bic = None, np.inf
    for k in range(1, max(2, min(max_clusters, len(x)))):
        gm = fit(k)
        if gm is not None and (bic := gm.bic(x)) < best_bic:
            best, best_bic = gm, bic
    if best is None:
        return [list(range(len(x)))]
    probs = best.predict_proba(x)
    clusters: list[list[int]] = [[] for _ in range(best.n_components)]
    for row, p in enumerate(probs):
        labels = np.where(p > threshold)[0]
        if len(labels) == 0:
            labels = np.array([int(np.argmax(p))])
        for lab in labels:
            clusters[int(lab)].append(row)
    return [c for c in clusters if c]


class GMMClustering(TokenLimitedClustering):
    """Soft GMM directly on the embeddings (no dimensionality reduction)."""

    def __init__(
        self,
        threshold: float = 0.1,
        max_clusters: int = 50,
        covariance_type: str = "spherical",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.threshold = threshold
        self.max_clusters = max_clusters
        self.covariance_type = covariance_type

    def groups(self, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        if len(emb) <= 2:
            return [list(range(len(emb)))]
        return _gmm_soft_cov(
            _unit_rows(emb),
            self.threshold,
            self.max_clusters,
            self.random_state,
            self.covariance_type,
        )


class PCAGMMClustering(TokenLimitedClustering):
    """PCA to ``reduction_dimension`` components, then one level of soft GMM."""

    def __init__(
        self,
        reduction_dimension: int = 10,
        threshold: float = 0.1,
        max_clusters: int = 50,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.reduction_dimension = reduction_dimension
        self.threshold = threshold
        self.max_clusters = max_clusters

    def groups(self, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        from sklearn.decomposition import PCA

        n = len(emb)
        if n <= 3:
            return [list(range(n))]
        dim = max(1, min(self.reduction_dimension, n - 2))
        x = PCA(n_components=dim, random_state=self.random_state).fit_transform(_unit_rows(emb))
        return _gmm_soft(x, self.threshold, self.max_clusters, self.random_state)


# ------------------------------------------------------------- hard clusterers
class AgglomerativeClustering(TokenLimitedClustering):
    """Average-linkage clustering on cosine distance.

    With ``distance_threshold`` set, merging stops at that cosine distance; otherwise the
    number of clusters is ``n / target_size``.
    """

    def __init__(
        self,
        distance_threshold: float | None = None,
        target_size: int = 6,
        linkage: str = "average",
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.distance_threshold = distance_threshold
        self.target_size = target_size
        self.linkage = linkage

    def groups(self, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        from sklearn.cluster import AgglomerativeClustering as Agg

        n = len(emb)
        if n <= 2:
            return [list(range(n))]
        if self.distance_threshold is not None:
            model = Agg(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="cosine",
                linkage=self.linkage,
            )
        else:
            k = max(1, min(n - 1, round(n / self.target_size)))
            model = Agg(n_clusters=k, metric="cosine", linkage=self.linkage)
        return _labels_to_groups(model.fit_predict(_unit_rows(emb)))


class HDBSCANClustering(TokenLimitedClustering):
    """UMAP to ``reduction_dimension`` then sklearn's HDBSCAN; noise joins the nearest cluster."""

    def __init__(
        self,
        reduction_dimension: int = 10,
        min_cluster_size: int = 3,
        min_samples: int | None = None,
        n_neighbors: int | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.reduction_dimension = reduction_dimension
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors

    def groups(self, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        from sklearn.cluster import HDBSCAN

        n = len(emb)
        if n <= max(3, self.min_cluster_size):
            return [list(range(n))]
        if n > self.reduction_dimension + 2:
            neighbors = self.n_neighbors or max(2, int((n - 1) ** 0.5))
            x = _umap(emb, self.reduction_dimension, neighbors, self.random_state)
        else:
            x = _unit_rows(emb)
        labels = HDBSCAN(
            min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, copy=True
        ).fit_predict(x)
        real = sorted(set(labels.tolist()) - {-1})
        if not real:
            return [list(range(n))]
        centroids = np.stack([x[labels == lab].mean(axis=0) for lab in real])
        for i in np.where(labels == -1)[0]:
            labels[i] = real[int(np.argmin(np.linalg.norm(centroids - x[i], axis=1)))]
        return _labels_to_groups(labels)


class KMeansSilhouetteClustering(TokenLimitedClustering):
    """k-means on unit vectors, with ``k`` picked by the cosine silhouette score.

    The smallest ``k`` tried is the one that makes the average cluster fit the token limit.
    """

    def __init__(self, max_clusters: int = 50, **kw: Any) -> None:
        super().__init__(**kw)
        self.max_clusters = max_clusters

    def groups(self, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n = len(emb)
        if n <= 3:
            return [list(range(n))]
        x = _unit_rows(emb)
        lo = max(2, math.ceil(float(tokens.sum()) / self.max_cluster_tokens))
        hi = min(self.max_clusters, n - 1)
        best_labels, best_score = None, -np.inf
        for k in range(lo, max(lo, hi) + 1):
            if k >= n:
                break
            labels = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(x)
            if len(set(labels.tolist())) < 2:
                continue
            score = silhouette_score(x, labels, metric="cosine")
            if score > best_score:
                best_labels, best_score = labels, score
        if best_labels is None:
            return [list(range(n))]
        return _labels_to_groups(best_labels)


class ContiguousClustering(TokenLimitedClustering):
    """Segment the (ordered) nodes into runs of adjacent nodes.

    Picks ``round(n / target_size)`` segments of ``min_size``..``max_size`` nodes that
    minimise the total similarity across the cut points (dynamic programming), so cuts
    fall where neighbouring nodes are least alike. Clusters come back in input order, so
    higher layers stay in narrative order too.
    """

    def __init__(
        self,
        target_size: int = 5,
        min_size: int | None = None,
        max_size: int | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(**kw)
        self.target_size = max(1, target_size)
        self.min_size = min_size if min_size is not None else max(1, self.target_size // 2)
        self.max_size = max_size if max_size is not None else 2 * self.target_size

    def groups(self, emb: np.ndarray, tokens: np.ndarray) -> list[list[int]]:
        n = len(emb)
        if n <= self.target_size:
            return [list(range(n))]
        x = _unit_rows(emb)
        gap_sim = np.sum(x[:-1] * x[1:], axis=1)  # gap i sits between rows i and i+1
        lo, hi = min(self.min_size, n), max(self.max_size, 1)
        k = max(2, round(n / self.target_size))
        k = min(max(k, math.ceil(n / hi)), max(1, n // lo))
        cuts = self._best_cuts(gap_sim, tokens, n, k, lo, hi)
        if cuts is None:  # infeasible with the size bounds: fall back to equal runs
            size = math.ceil(n / k)
            cuts = list(range(size, n, size))
        bounds = [0, *cuts, n]
        return [list(range(a, b)) for a, b in itertools.pairwise(bounds)]

    def _best_cuts(
        self, gap_sim: np.ndarray, tokens: np.ndarray, n: int, k: int, lo: int, hi: int
    ) -> list[int] | None:
        csum = np.concatenate([[0], np.cumsum(tokens)])
        inf = float("inf")
        # best[j][m]: min cost to split rows [0, j) into m segments; back-pointers in prev.
        best = np.full((n + 1, k + 1), inf)
        prev = np.full((n + 1, k + 1), -1, dtype=np.int64)
        best[0][0] = 0.0
        for m in range(1, k + 1):
            for j in range(1, n + 1):
                for size in range(lo, hi + 1):
                    i = j - size
                    if i < 0:
                        break
                    if best[i][m - 1] == inf:
                        continue
                    if size > 1 and csum[j] - csum[i] > self.max_cluster_tokens:
                        continue
                    cost = best[i][m - 1] + (gap_sim[i - 1] if i > 0 else 0.0)
                    if cost < best[j][m]:
                        best[j][m], prev[j][m] = cost, i
        if best[n][k] == inf:
            return None
        cuts, j = [], n
        for m in range(k, 0, -1):
            i = int(prev[j][m])
            if i > 0:
                cuts.append(i)
            j = i
        return sorted(cuts)


def _labels_to_groups(labels: np.ndarray) -> list[list[int]]:
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(np.asarray(labels).tolist()):
        groups.setdefault(int(lab), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


# ------------------------------------------------------------------ registry
_COMMON = ("max_cluster_tokens", "random_state")


def _pick(kw: dict[str, Any], *names: str) -> dict[str, Any]:
    return {k: kw[k] for k in (*names, *_COMMON) if k in kw}


@CLUSTERERS.register("gmm")
def _make_gmm(**kw: Any) -> ClusteringAlgorithm:
    return GMMClustering(**_pick(kw, "threshold", "max_clusters", "covariance_type"))


@CLUSTERERS.register("pca_gmm")
def _make_pca_gmm(**kw: Any) -> ClusteringAlgorithm:
    return PCAGMMClustering(**_pick(kw, "reduction_dimension", "threshold", "max_clusters"))


@CLUSTERERS.register("agglomerative")
def _make_agglomerative(**kw: Any) -> ClusteringAlgorithm:
    return AgglomerativeClustering(**_pick(kw, "distance_threshold", "target_size", "linkage"))


@CLUSTERERS.register("hdbscan")
def _make_hdbscan(**kw: Any) -> ClusteringAlgorithm:
    return HDBSCANClustering(
        **_pick(kw, "reduction_dimension", "min_cluster_size", "min_samples", "n_neighbors")
    )


@CLUSTERERS.register("kmeans_silhouette")
def _make_kmeans(**kw: Any) -> ClusteringAlgorithm:
    return KMeansSilhouetteClustering(**_pick(kw, "max_clusters"))


@CLUSTERERS.register("contiguous")
def _make_contiguous(**kw: Any) -> ClusteringAlgorithm:
    return ContiguousClustering(**_pick(kw, "target_size", "min_size", "max_size"))
