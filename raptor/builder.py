"""Builds a RAPTOR tree: chunk → embed → (cluster → summarize → embed)*."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import tiktoken

from .chunking import Chunk, Chunker, default_tokenizer, make_chunker
from .clustering import ClusteringAlgorithm, make_clusterer
from .config import RaptorConfig
from .models.base import BaseEmbeddingModel, BaseSummarizationModel
from .types import Node, Tree, nodes_text

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]


def _noop(_: dict[str, Any]) -> None:
    pass


class TreeBuilder:
    def __init__(
        self,
        config: RaptorConfig,
        embedding_model: BaseEmbeddingModel,
        summarization_model: BaseSummarizationModel,
        *,
        chunker: Chunker | None = None,
        clusterer: ClusteringAlgorithm | None = None,
        tokenizer: tiktoken.Encoding | None = None,
    ) -> None:
        self.config = config
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.chunker = chunker or make_chunker(config.chunking)
        self.clusterer = clusterer or make_clusterer(config.clustering)
        self.tokenizer = tokenizer or default_tokenizer()

    # ------------------------------------------------------------------ public
    def build_from_text(self, text: str, *, progress: ProgressCallback | None = None) -> Tree:
        progress = progress or _noop
        progress({"stage": "chunking"})
        chunks = self.chunker.split(text)
        if not chunks:
            raise ValueError("document produced no chunks (is it empty?)")
        return self.build_from_chunks(chunks, progress=progress)

    def build_from_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        leaf_embeddings: np.ndarray | None = None,
        progress: ProgressCallback | None = None,
    ) -> Tree:
        progress = progress or _noop
        key = self.config.builder.embedding_key
        progress({"stage": "embedding", "layer": 0, "total": len(chunks)})
        if leaf_embeddings is None:
            leaf_embeddings = self.embedding_model.embed_documents([c.text for c in chunks])
        leaves = []
        for i, (chunk, emb) in enumerate(zip(chunks, leaf_embeddings, strict=True)):
            meta = dict(chunk.metadata)
            if chunk.start is not None:
                meta.update(start=chunk.start, end=chunk.end)
            leaves.append(
                Node(chunk.text, i, set(), {key: np.asarray(emb, dtype=np.float32)}, 0, meta)
            )
        logger.info("Created %d leaf nodes", len(leaves))

        layer_to_nodes: dict[int, list[Node]] = {0: leaves}
        current = leaves
        next_index = len(leaves)
        for layer in range(1, self.config.builder.max_layers + 1):
            if len(current) <= self.config.stop_threshold:
                logger.info(
                    "Stopping at layer %d: %d nodes <= %d",
                    layer - 1,
                    len(current),
                    self.config.stop_threshold,
                )
                break
            new_nodes = self._build_layer(current, layer, next_index, progress)
            if len(new_nodes) >= len(current):
                logger.info(
                    "Stopping at layer %d: clustering did not reduce the node count", layer - 1
                )
                break
            layer_to_nodes[layer] = new_nodes
            next_index += len(new_nodes)
            current = new_nodes

        tree = Tree.from_layers(
            layer_to_nodes,
            metadata={
                "embedding_key": key,
                "embedding_model": self.embedding_model.model_id,
                "summarization_model": self.summarization_model.model_id,
                "config": self.config.model_dump(mode="json"),
            },
        )
        progress({"stage": "done", "num_nodes": len(tree.all_nodes), "num_layers": tree.num_layers})
        return tree

    # ---------------------------------------------------------------- internals
    def _build_layer(
        self, nodes: list[Node], layer: int, first_index: int, progress: ProgressCallback
    ) -> list[Node]:
        key = self.config.builder.embedding_key
        progress({"stage": "clustering", "layer": layer, "total": len(nodes)})
        embeddings = np.stack([np.asarray(n.embeddings[key], dtype=np.float32) for n in nodes])
        tokens = [len(self.tokenizer.encode(n.text, disallowed_special=())) for n in nodes]
        clusters = self.clusterer.cluster(embeddings, tokens)
        logger.info("Layer %d: %d nodes -> %d clusters", layer, len(nodes), len(clusters))

        groups = [[nodes[i] for i in c] for c in clusters]
        length = self.config.builder.summarization_length
        done = 0

        def summarize(group: list[Node]) -> str:
            return self.summarization_model.summarize(nodes_text(group), max_tokens=length)

        progress({"stage": "summarizing", "layer": layer, "done": 0, "total": len(groups)})
        summaries: list[str] = [""] * len(groups)
        with ThreadPoolExecutor(max_workers=self.config.builder.max_workers) as pool:
            futures = {pool.submit(summarize, g): i for i, g in enumerate(groups)}
            for fut, i in futures.items():
                summaries[i] = fut.result()
                done += 1
                progress(
                    {"stage": "summarizing", "layer": layer, "done": done, "total": len(groups)}
                )

        progress({"stage": "embedding", "layer": layer, "total": len(summaries)})
        summary_embeddings = self.embedding_model.embed_documents(summaries)
        return [
            Node(
                text,
                first_index + i,
                {n.index for n in group},
                {key: np.asarray(emb, dtype=np.float32)},
                layer,
            )
            for i, (text, group, emb) in enumerate(
                zip(summaries, groups, summary_embeddings, strict=True)
            )
        ]
