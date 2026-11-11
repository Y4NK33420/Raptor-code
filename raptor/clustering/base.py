"""Clustering interface.

A clustering algorithm groups the nodes of one layer. It receives their embeddings and
token counts and returns clusters as lists of row indices. An index may appear in more
than one cluster (soft clustering); every index must appear in at least one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from ..registry import Registry


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def cluster(self, embeddings: np.ndarray, token_counts: Sequence[int]) -> list[list[int]]: ...


CLUSTERERS: Registry[ClusteringAlgorithm] = Registry("clustering algorithm")


def check_clusters(clusters: list[list[int]], n: int) -> list[list[int]]:
    """Validate and normalise clusters: sorted, de-duplicated, non-empty, full coverage."""
    out = [sorted(set(c)) for c in clusters if c]
    covered = {i for c in out for i in c}
    if covered != set(range(n)):
        missing = sorted(set(range(n)) - covered)
        raise ValueError(f"clustering left indices unassigned: {missing[:10]}")
    return out
