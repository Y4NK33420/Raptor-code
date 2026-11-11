"""Clustering strategies. Register new ones with ``@CLUSTERERS.register("name")``."""

from ..config import ClusteringConfig
from .base import CLUSTERERS, ClusteringAlgorithm, check_clusters
from .raptor_gmm import RaptorGMMClustering

__all__ = [
    "CLUSTERERS",
    "ClusteringAlgorithm",
    "RaptorGMMClustering",
    "check_clusters",
    "make_clusterer",
]


def make_clusterer(cfg: ClusteringConfig) -> ClusteringAlgorithm:
    return CLUSTERERS.create(
        cfg.strategy,
        reduction_dimension=cfg.reduction_dimension,
        threshold=cfg.threshold,
        max_cluster_tokens=cfg.max_cluster_tokens,
        random_state=cfg.random_state,
        **cfg.params,
    )
