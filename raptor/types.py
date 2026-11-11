"""Core data structures: :class:`Node` and :class:`Tree`.

A RAPTOR index is a layered DAG. Layer 0 holds the leaf chunks; each higher layer
holds summaries of clusters of nodes from the layer below. With soft clustering a
node can belong to several clusters, so a node may have more than one parent.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_EMBEDDING_KEY = "EMB"


@dataclass(eq=False)
class Node:
    """One node of the index.

    The positional order ``(text, index, children, embeddings)`` matches the original
    RAPTOR ``Node`` so older call sites keep working.
    """

    text: str
    index: int
    children: set[int] = field(default_factory=set)
    embeddings: dict[str, Any] = field(default_factory=dict)
    layer: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(eq=False)
class Tree:
    """The full index.

    ``root_nodes`` and ``leaf_nodes`` are dicts keyed by node index (as in the original
    RAPTOR). ``layer_to_nodes`` maps a layer number to the list of nodes in that layer.
    """

    all_nodes: dict[int, Node]
    root_nodes: dict[int, Node]
    leaf_nodes: dict[int, Node]
    num_layers: int
    layer_to_nodes: dict[int, list[Node]]
    metadata: dict[str, Any] = field(default_factory=dict)
    _matrix_cache: dict[str, tuple[list[int], np.ndarray]] = field(
        default_factory=dict, init=False, repr=False
    )

    # ------------------------------------------------------------------ helpers
    @property
    def embedding_key(self) -> str:
        return str(self.metadata.get("embedding_key", DEFAULT_EMBEDDING_KEY))

    def node_layer(self, index: int) -> int:
        return self.all_nodes[index].layer

    def parents_of(self, index: int) -> list[int]:
        return [n.index for n in self.all_nodes.values() if index in n.children]

    def nodes_in_layer(self, layer: int) -> list[Node]:
        return list(self.layer_to_nodes.get(layer, []))

    def normalized_matrix(self, key: str | None = None) -> tuple[list[int], np.ndarray]:
        """Return ``(node_indices, unit-normalised embedding matrix)`` for ``key``.

        The result is cached on the tree; call :meth:`invalidate_cache` after mutating
        nodes in place.
        """
        key = key or self.embedding_key
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached
        indices = sorted(self.all_nodes)
        matrix = np.asarray([self.all_nodes[i].embeddings[key] for i in indices], dtype=np.float32)
        matrix = _normalize_rows(matrix)
        self._matrix_cache[key] = (indices, matrix)
        return indices, matrix

    def invalidate_cache(self) -> None:
        self._matrix_cache.clear()

    def validate(self) -> None:
        """Raise ``ValueError`` if the tree is internally inconsistent."""
        for idx, node in self.all_nodes.items():
            if idx != node.index:
                raise ValueError(f"node key {idx} != node.index {node.index}")
            missing = node.children - self.all_nodes.keys()
            if missing:
                raise ValueError(f"node {idx} has unknown children {sorted(missing)}")
        for layer, nodes in self.layer_to_nodes.items():
            for n in nodes:
                if n.layer != layer:
                    raise ValueError(f"node {n.index} in layer list {layer} has layer {n.layer}")

    @classmethod
    def from_layers(
        cls,
        layer_to_nodes: dict[int, list[Node]],
        metadata: dict[str, Any] | None = None,
    ) -> Tree:
        all_nodes = {n.index: n for nodes in layer_to_nodes.values() for n in nodes}
        top = max(layer_to_nodes) if layer_to_nodes else 0
        return cls(
            all_nodes=all_nodes,
            root_nodes={n.index: n for n in layer_to_nodes.get(top, [])},
            leaf_nodes={n.index: n for n in layer_to_nodes.get(0, [])},
            num_layers=top,
            layer_to_nodes=layer_to_nodes,
            metadata=dict(metadata or {}),
        )


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def nodes_text(nodes: Iterable[Node]) -> str:
    """Join node texts the way the original RAPTOR did (one paragraph per node)."""
    return "".join(" ".join(n.text.splitlines()) + "\n\n" for n in nodes)
