"""Saving and loading trees.

Format (version 1) is a directory::

    <path>/
      tree.json        # nodes (text, children, layer, metadata), layers, tree metadata
      embeddings.npz   # one float32 matrix per embedding key, rows ordered like tree.json nodes

Nothing here executes code on load. Legacy pickles written by the original RAPTOR can
be imported with :func:`load_legacy_pickle`, which uses an allow-listed unpickler that
only reconstructs the old ``Node``/``Tree`` classes and plain containers.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .types import Node, Tree

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
TREE_FILE = "tree.json"
EMBEDDINGS_FILE = "embeddings.npz"


# --------------------------------------------------------------------------- save
def save_tree(tree: Tree, path: str | Path) -> Path:
    """Write ``tree`` to the directory ``path`` atomically (replaces any existing index)."""
    path = Path(path)
    if path.exists() and path.is_file():
        raise FileExistsError(f"{path} is a file; pass a directory path for the new format")
    path.parent.mkdir(parents=True, exist_ok=True)

    order = sorted(tree.all_nodes)
    keys = sorted({k for n in tree.all_nodes.values() for k in n.embeddings})
    nodes_json = []
    for idx in order:
        n = tree.all_nodes[idx]
        nodes_json.append(
            {
                "index": n.index,
                "text": n.text,
                "children": sorted(n.children),
                "layer": n.layer,
                "metadata": n.metadata,
            }
        )
    doc: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "num_layers": tree.num_layers,
        "metadata": tree.metadata,
        "embedding_keys": keys,
        "layers": {
            str(layer): [n.index for n in nodes] for layer, nodes in tree.layer_to_nodes.items()
        },
        "root_nodes": sorted(tree.root_nodes),
        "leaf_nodes": sorted(tree.leaf_nodes),
        "nodes": nodes_json,
    }
    matrices = {}
    for key in keys:
        rows = [tree.all_nodes[i].embeddings.get(key) for i in order]
        if any(r is None for r in rows):
            raise ValueError(f"embedding '{key}' is missing on some nodes")
        matrices[key] = np.asarray(rows, dtype=np.float32)

    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=path.parent))
    try:
        (tmp / TREE_FILE).write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(tmp / EMBEDDINGS_FILE, **matrices)  # type: ignore[arg-type]
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    logger.info("Saved tree with %d nodes to %s", len(order), path)
    return path


# --------------------------------------------------------------------------- load
def load_tree(path: str | Path) -> Tree:
    """Load a tree saved with :func:`save_tree`.

    A legacy pickle file is also accepted (imported with the restricted unpickler) and
    a deprecation warning is logged; run ``raptor migrate`` to convert it.
    """
    path = Path(path)
    if path.is_file():
        logger.warning(
            "%s is a legacy pickle; importing with the restricted loader. "
            "Convert it with `raptor migrate %s <out_dir>`.",
            path,
            path,
        )
        return load_legacy_pickle(path)
    tree_file = path / TREE_FILE
    if not tree_file.exists():
        raise FileNotFoundError(f"no RAPTOR index at {path} (missing {TREE_FILE})")

    doc = json.loads(tree_file.read_text(encoding="utf-8"))
    version = doc.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported index format_version {version!r} (expected {FORMAT_VERSION})"
        )

    keys: list[str] = doc["embedding_keys"]
    with np.load(path / EMBEDDINGS_FILE, allow_pickle=False) as npz:
        matrices = {k: npz[k] for k in keys}

    all_nodes: dict[int, Node] = {}
    for row, nj in enumerate(doc["nodes"]):
        all_nodes[nj["index"]] = Node(
            text=nj["text"],
            index=nj["index"],
            children=set(nj["children"]),
            embeddings={k: matrices[k][row] for k in keys},
            layer=nj["layer"],
            metadata=nj.get("metadata", {}),
        )
    layer_to_nodes = {int(k): [all_nodes[i] for i in v] for k, v in doc["layers"].items()}
    tree = Tree(
        all_nodes=all_nodes,
        root_nodes={i: all_nodes[i] for i in doc["root_nodes"]},
        leaf_nodes={i: all_nodes[i] for i in doc["leaf_nodes"]},
        num_layers=doc["num_layers"],
        layer_to_nodes=layer_to_nodes,
        metadata=doc.get("metadata", {}),
    )
    tree.validate()
    return tree


# ------------------------------------------------------------------ legacy import
class _LegacyNode:
    pass


class _LegacyTree:
    pass


_ALLOWED_GLOBALS: dict[tuple[str, str], Any] = {
    ("raptor.tree_structures", "Node"): _LegacyNode,
    ("raptor.tree_structures", "Tree"): _LegacyTree,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "dict"): dict,
    ("builtins", "list"): list,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in _ALLOWED_GLOBALS:
            return _ALLOWED_GLOBALS[(module, name)]
        if module in ("numpy", "numpy.core.multiarray", "numpy._core.multiarray") and name in (
            "ndarray",
            "dtype",
            "_reconstruct",
            "scalar",
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load global {module}.{name} from legacy tree")


def load_legacy_pickle(path: str | Path) -> Tree:
    """Import a tree pickled by the original RAPTOR (``RetrievalAugmentation.save``)."""
    with open(path, "rb") as fh:
        obj = _RestrictedUnpickler(fh).load()
    if not isinstance(obj, _LegacyTree):
        raise ValueError(f"{path} does not contain a legacy RAPTOR Tree")
    legacy = obj.__dict__

    index_to_layer: dict[int, int] = {}
    for layer, nodes in legacy["layer_to_nodes"].items():
        for n in nodes:
            index_to_layer[n.__dict__["index"]] = int(layer)

    def convert(ln: _LegacyNode) -> Node:
        d = ln.__dict__
        return Node(
            text=d["text"],
            index=d["index"],
            children=set(d["children"]),
            embeddings={k: np.asarray(v, dtype=np.float32) for k, v in d["embeddings"].items()},
            layer=index_to_layer.get(d["index"], 0),
        )

    all_nodes = {i: convert(n) for i, n in legacy["all_nodes"].items()}
    layer_to_nodes = {
        int(layer): [all_nodes[n.__dict__["index"]] for n in nodes]
        for layer, nodes in legacy["layer_to_nodes"].items()
    }
    keys = {k for n in all_nodes.values() for k in n.embeddings}
    tree = Tree(
        all_nodes=all_nodes,
        root_nodes={i: all_nodes[i] for i in legacy["root_nodes"]},
        leaf_nodes={i: all_nodes[i] for i in legacy["leaf_nodes"]},
        num_layers=int(legacy["num_layers"]),
        layer_to_nodes=layer_to_nodes,
        metadata={
            "embedding_key": sorted(keys)[0] if len(keys) == 1 else "EMB",
            "source": "legacy-pickle",
        },
    )
    tree.validate()
    return tree
