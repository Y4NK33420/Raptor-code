"""``raptor eval run | report | compare | list | splits``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from .runner import DEFAULT_RUNS_DIR, DEFAULT_TREES_DIR

app = typer.Typer(add_completion=False, help="QuALITY experiment harness.", no_args_is_help=True)

RootOpt = Annotated[Path, typer.Option("--runs", help="Run artifacts directory.")]


def _logging(verbose: bool) -> None:
    from ..settings import get_settings

    logging.basicConfig(
        level="DEBUG" if verbose else get_settings().log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "google_genai", "numba", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command()
def run(
    configs: Annotated[list[Path], typer.Argument(help="Experiment YAML files.", exists=True)],
    runs: RootOpt = DEFAULT_RUNS_DIR,
    trees: Annotated[
        Path, typer.Option("--trees", help="Tree cache directory.")
    ] = DEFAULT_TREES_DIR,
    force: Annotated[
        bool, typer.Option(help="Redo finished runs (all calls come from cache).")
    ] = False,
    workers: Annotated[int, typer.Option(help="Questions answered in parallel.")] = 4,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run experiment configs (skips finished ones; resumes interrupted ones)."""
    _logging(verbose)
    from .runner import Runner

    runner = Runner(runs, trees, question_workers=workers)
    outcomes = runner.run_many(sorted(configs), force=force)
    for o in outcomes:
        acc = f"{o.summary['accuracy']:.3f}" if o.summary else "-"
        cost = o.session_cost
        typer.echo(
            f"{o.status:8s} {o.run_id:48s} acc={acc}  "
            f"llm={cost.get('llm_calls', 0)} emb={cost.get('embed_calls', 0)} "
            f"hits={cost.get('cache_hits', 0)} misses={cost.get('cache_misses', 0)}"
        )
    if any(o.status == "quota" for o in outcomes):
        raise typer.Exit(code=2)


@app.command("list")
def list_cmd(runs: RootOpt = DEFAULT_RUNS_DIR) -> None:
    """List finished runs."""
    from .runs import list_runs

    for s in list_runs(runs):
        lo, hi = s["ci95"]
        typer.echo(
            f"{s['run_id']:48s} {s.get('stage', ''):2s} {s['split']:8s} {s['method']:7s} "
            f"acc={s['accuracy']:.3f} [{lo:.2f},{hi:.2f}] n={s['n_questions']}"
        )


@app.command()
def compare(
    a: Annotated[str, typer.Argument(help="Baseline run id or name.")],
    b: Annotated[str, typer.Argument(help="Other run id or name.")],
    runs: RootOpt = DEFAULT_RUNS_DIR,
) -> None:
    """Paired comparison of run B against run A."""
    from .runs import compare_runs, find_run

    out = compare_runs(find_run(a, runs)["run_id"], find_run(b, runs)["run_id"], runs)
    typer.echo(json.dumps(out, indent=2))


@app.command()
def report(
    runs: RootOpt = DEFAULT_RUNS_DIR,
    out: Annotated[Path, typer.Option("--out", help="Report directory.")] = Path(
        "docs/experiments"
    ),
) -> None:
    """Write the Markdown report and figures."""
    from .report import generate_report

    typer.echo(f"wrote {generate_report(runs, out)}")


@app.command()
def splits(
    out: Annotated[Path, typer.Option("--out")] = Path("configs/splits.yaml"),
    seed: int = 561,
    size: int = 8,
) -> None:
    """Regenerate the fixed dev / confirm splits from the QuALITY dev file."""
    from .datasets.quality import load_quality
    from .splits import make_splits, save_splits

    result = make_splits(load_quality("dev"), seed=seed, size=size)
    save_splits(result, out, seed=seed)
    by_id = {a.set_id: a for a in load_quality("dev")}
    for name, spec in result.items():
        n = sum(len(by_id[s].questions) for s in spec.set_ids)
        typer.echo(f"{name}: {len(spec.set_ids)} sets, {n} questions")
