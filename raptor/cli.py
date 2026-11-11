"""Command-line interface: ``raptor build | ask | info | migrate``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(
    add_completion=False, help="RAPTOR: tree-organized retrieval over long documents."
)


def _setup_logging(verbose: bool) -> None:
    from .settings import get_settings

    level = "DEBUG" if verbose else get_settings().log_level
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "google_genai", "numba", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _config(path: Path | None):
    from .config import RaptorConfig

    return RaptorConfig.from_yaml(path) if path else RaptorConfig()


ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML RaptorConfig file.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v")]


@app.command()
def build(
    source: Annotated[
        Path, typer.Argument(help="Text file to index.", exists=True, dir_okay=False)
    ],
    out: Annotated[Path, typer.Option("--out", "-o", help="Directory to write the index to.")],
    config: ConfigOpt = None,
    overwrite: Annotated[bool, typer.Option(help="Replace an existing index at --out.")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Build an index from a text file."""
    _setup_logging(verbose)
    from .pipeline import RetrievalAugmentation

    if out.exists() and not overwrite:
        raise typer.BadParameter(f"{out} exists; pass --overwrite to replace it")
    ra = RetrievalAugmentation(_config(config))

    def progress(ev: dict) -> None:
        if ev.get("stage") in {"clustering", "done"} or (
            ev.get("stage") == "summarizing" and ev.get("done") == ev.get("total")
        ):
            typer.echo(f"  {ev}")

    tree = ra.add_documents(source.read_text(encoding="utf-8"), progress=progress)
    ra.save(out)
    typer.echo(f"Built {len(tree.all_nodes)} nodes in {tree.num_layers + 1} layers -> {out}")


@app.command()
def ask(
    index: Annotated[Path, typer.Argument(help="Index directory.", exists=True)],
    question: Annotated[str, typer.Argument()],
    config: ConfigOpt = None,
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
    show_context: Annotated[bool, typer.Option(help="Print the retrieved nodes.")] = False,
    verbose: VerboseOpt = False,
) -> None:
    """Answer a question from an index."""
    _setup_logging(verbose)
    from .pipeline import RetrievalAugmentation

    ra = RetrievalAugmentation(_config(config), tree=index)
    result = ra.retrieve_nodes(question, top_k=top_k)
    if show_context:
        for r in result.nodes:
            typer.echo(
                f"[layer {r.layer} | node {r.node.index} | sim {r.score:.3f}] {r.node.text[:160]}..."
            )
        typer.echo("")
    for piece in ra.qa_model.stream_answer(result.context, question):
        typer.echo(piece, nl=False)
    typer.echo("")


@app.command()
def info(index: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Show the structure of an index."""
    from .storage import load_tree

    tree = load_tree(index)
    typer.echo(f"{index}: {len(tree.all_nodes)} nodes, top layer {tree.num_layers}")
    for layer in sorted(tree.layer_to_nodes):
        typer.echo(f"  layer {layer}: {len(tree.layer_to_nodes[layer])} nodes")
    for key in ("embedding_model", "summarization_model", "source"):
        if key in tree.metadata:
            typer.echo(f"  {key}: {tree.metadata[key]}")


@app.command()
def migrate(
    pickle_file: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False, help="Legacy pickled tree.")
    ],
    out: Annotated[Path, typer.Argument(help="Directory for the converted index.")],
) -> None:
    """Convert a tree pickled by the original RAPTOR into the current index format."""
    from .storage import load_legacy_pickle, save_tree

    tree = load_legacy_pickle(pickle_file)
    save_tree(tree, out)
    typer.echo(f"Converted {pickle_file} ({len(tree.all_nodes)} nodes) -> {out}")


from .eval.cli import app as eval_app  # noqa: E402

app.add_typer(
    eval_app, name="eval", help="QuALITY experiment harness (run, report, compare, list)."
)

if __name__ == "__main__":  # pragma: no cover
    app()
