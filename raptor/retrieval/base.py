"""Retrieval result types and shared scoring helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..types import Node, Tree, nodes_text


@dataclass
class RetrievedNode:
    node: Node
    score: float  # cosine similarity to the query

    @property
    def layer(self) -> int:
        return self.node.layer


@dataclass
class RetrievalResult:
    nodes: list[RetrievedNode] = field(default_factory=list)

    @property
    def context(self) -> str:
        return nodes_text(r.node for r in self.nodes)

    def layer_information(self) -> list[dict[str, int]]:
        return [{"node_index": r.node.index, "layer_number": r.layer} for r in self.nodes]

    def layer_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for r in self.nodes:
            counts[r.layer] = counts.get(r.layer, 0) + 1
        return counts


def unit(vector: np.ndarray) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v


def score_all(tree: Tree, query_embedding: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Cosine similarity of the query to every node: ``(node_indices, similarities)``."""
    indices, matrix = tree.normalized_matrix()
    return indices, matrix @ unit(query_embedding)
