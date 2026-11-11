"""Generate the experiment report (Markdown + matplotlib figures) from run artifacts.

Every number in the report is computed from ``artifacts/runs``. Hand-written
conclusions live in ``docs/experiments/conclusions.md`` and are included verbatim, so
regenerating the report never loses them.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runner import DEFAULT_RUNS_DIR
from .runs import compare_runs, list_runs

DEFAULT_OUT_DIR = Path("docs/experiments")
FACTOR_ORDER = [
    "baseline",
    "method",
    "retrieval",
    "embedding",
    "chunking",
    "clustering",
    "combined",
]
ALPHA = 0.05

# Reference palette (validated categorical order); text stays in neutral ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def _baseline_for(run: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    name = run.get("baseline")
    if not name:
        return None
    same = [r for r in runs if r.get("name") == name and r.get("split") == run.get("split")]
    return same[-1] if same else None


def _verdict(cmp: dict[str, Any] | None) -> str:
    if cmp is None:
        return "baseline"
    if cmp["mcnemar_p"] < ALPHA:
        return "**better**" if cmp["delta"] > 0 else "**worse**"
    return "within noise"


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{100 * x:.{digits}f}"


def _setting(run: dict[str, Any]) -> str:
    f = run.get("factors", {})
    factor = run.get("factor", "")
    if factor in f:
        return f[factor]
    if factor == "method":
        return run.get("method", "")
    return " / ".join(f.get(k, "") for k in ("chunking", "clustering", "retrieval", "embedding"))


def _summary_share(run: dict[str, Any]) -> float:
    share = run.get("layer_share", {})
    return float(sum(v for k, v in share.items() if k != "0"))


def _order(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(r: dict[str, Any]) -> tuple[int, str]:
        f = r.get("factor") or ("baseline" if not r.get("baseline") else "combined")
        return (FACTOR_ORDER.index(f) if f in FACTOR_ORDER else len(FACTOR_ORDER), r["name"])

    return sorted(runs, key=key)


def stage_rows(runs: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Each run of a stage with its paired comparison to its baseline."""
    out = []
    for run in _order(runs):
        base = _baseline_for(run, runs) or _baseline_for(run, list_runs(root))
        cmp = compare_runs(base["run_id"], run["run_id"], root) if base else None
        out.append({"run": run, "base": base, "cmp": cmp, "verdict": _verdict(cmp)})
    return out


def _table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| Run | Varies | Setting | Acc % | 95% CI | Hard acc % | Δ vs base (pts) [95% CI] "
        "| McNemar p | fixed / broken | Summary-node share % | LLM calls | Build s/story "
        "| Nodes/tree | Verdict |\n"
        "|---|---|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|---|\n"
    )
    lines = []
    for row in rows:
        r, cmp = row["run"], row["cmp"]
        lo, hi = r["ci95"]
        if cmp:
            dlo, dhi = cmp["delta_ci95"]
            delta = f"{100 * cmp['delta']:+.1f} [{100 * dlo:+.1f}, {100 * dhi:+.1f}]"
            p = f"{cmp['mcnemar_p']:.3f}"
            fb = f"{len(cmp['fixed'])} / {len(cmp['broken'])}"
        else:
            delta, p, fb = "-", "-", "-"
        lines.append(
            f"| `{r['name']}` | {r.get('factor') or '-'} | {_setting(r)} | {_pct(r['accuracy'])} "
            f"| {_pct(lo)}–{_pct(hi)} | {_pct(r.get('accuracy_hard'))} | {delta} | {p} | {fb} "
            f"| {_pct(_summary_share(r), 0)} | {r['cost'].get('llm_calls', 0)} "
            f"| {r.get('build_seconds', 0):.0f} | {r['tree_stats'].get('mean_nodes', 0):.0f} "
            f"| {row['verdict']} |"
        )
    return head + "\n".join(lines) + "\n"


# ------------------------------------------------------------------- figures
def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _forest(rows: list[dict[str, Any]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 0.42 * n + 1.6), sharey=True, gridspec_kw={"width_ratios": [1, 1]}
    )
    fig.patch.set_facecolor(SURFACE)
    labels = [r["run"]["name"] for r in rows]
    ys = list(range(n))[::-1]
    for y, row in zip(ys, rows, strict=True):
        r = row["run"]
        lo, hi = r["ci95"]
        is_base = row["cmp"] is None
        color = INK_2 if is_base else SERIES[0]
        ax1.plot([100 * lo, 100 * hi], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax1.plot(100 * r["accuracy"], y, "o", color=color, markersize=8, markeredgecolor=SURFACE)
        if row["cmp"]:
            c = row["cmp"]
            dlo, dhi = c["delta_ci95"]
            sig = c["mcnemar_p"] < ALPHA
            dcol = (SERIES[2] if c["delta"] > 0 else SERIES[1]) if sig else SERIES[0]
            ax2.plot(
                [100 * dlo, 100 * dhi], [y, y], color=dcol, linewidth=2, solid_capstyle="round"
            )
            ax2.plot(100 * c["delta"], y, "o", color=dcol, markersize=8, markeredgecolor=SURFACE)
    bases = [row["run"]["accuracy"] for row in rows if row["cmp"] is None]
    for b in bases:
        ax1.axvline(100 * b, color=INK_2, linewidth=1, linestyle=(0, (3, 3)))
    ax2.axvline(0, color=INK_2, linewidth=1)
    ax1.set_yticks(ys, labels, color=INK, fontsize=9)
    ax1.set_xlabel("Accuracy % (95% bootstrap CI)", color=INK_2, fontsize=9)
    ax2.set_xlabel("Δ vs baseline, points (paired 95% CI)", color=INK_2, fontsize=9)
    for ax in (ax1, ax2):
        _style(ax)
    fig.suptitle(title, color=INK, fontsize=11, x=0.01, ha="left")
    fig.text(
        0.01,
        0.005,
        "Grey = baseline. Right panel: green/orange = McNemar p < 0.05 better/worse; blue = within noise.",
        color=INK_2,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def _layer_share(runs: list[dict[str, Any]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in runs if r.get("method") == "raptor"]
    if not runs:
        return
    layers = sorted({int(k) for r in runs for k in r.get("layer_share", {})})
    fig, ax = plt.subplots(figsize=(9, 0.36 * len(runs) + 1.4))
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(runs)))[::-1]
    left = [0.0] * len(runs)
    for li, layer in enumerate(layers):
        vals = [100 * r.get("layer_share", {}).get(str(layer), 0.0) for r in runs]
        label = "leaves (layer 0)" if layer == 0 else f"layer {layer} summaries"
        ax.barh(
            ys,
            vals,
            left=left,
            height=0.62,
            color=SERIES[min(li, len(SERIES) - 1)],
            edgecolor=SURFACE,
            linewidth=2,
            label=label,
        )
        left = [a + b for a, b in zip(left, vals, strict=True)]
    ax.set_yticks(ys, [f"{r['name']} ({r['split']})" for r in runs], color=INK, fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of retrieved nodes, %", color=INK_2, fontsize=9)
    _style(ax)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(layers), frameon=False, fontsize=8
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# -------------------------------------------------------------------- report
def generate_report(
    root: str | Path = DEFAULT_RUNS_DIR,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    *,
    title: str = "RAPTOR strategy sweep on QuALITY",
    figures: bool = True,
) -> Path:
    root, out_dir = Path(root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    runs = list_runs(root)
    stages: dict[str, list[dict[str, Any]]] = {}
    for r in runs:
        stages.setdefault(r.get("stage") or "other", []).append(r)

    parts = [f"# {title}\n"]
    parts.append(
        f"_Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `raptor eval report` from "
        f"{len(runs)} runs in `{root.as_posix()}`. Do not edit by hand: conclusions live in "
        "`conclusions.md`._\n"
    )
    splits = sorted({(r["split"], r["n_questions"], r.get("n_stories", 0)) for r in runs})
    if splits:
        desc = ", ".join(f"`{s}` = {n} questions over {k} stories" for s, n, k in splits)
        readers = sorted({r.get("reader_model", "?") for r in runs})
        parts.append(
            "## Setup\n\n"
            f"- Splits (`configs/splits.yaml`): {desc}. Dev and confirm use disjoint articles.\n"
            f"- Reader (fixed): {', '.join(f'`{m}`' for m in readers)} with JSON-mode MCQ answers.\n"
            "- Accuracy CIs: percentile bootstrap (10k resamples). Δ CIs: paired bootstrap over "
            "questions. p-values: exact two-sided McNemar test on the discordant pairs vs the "
            "run's baseline.\n"
            "- **Noise floor:** with ~70 questions a 95% CI on accuracy is about ±11 points, so "
            'only large, consistent differences mean anything. "Within noise" below means '
            f"McNemar p ≥ {ALPHA}.\n"
            '- "Summary-node share" is the fraction of retrieved nodes that are summaries '
            "(layer ≥ 1) rather than leaf chunks. LLM calls count summaries + answers that were "
            "not already cached when the run happened.\n"
        )
    conclusions = out_dir / "conclusions.md"
    if conclusions.exists():
        parts.append(conclusions.read_text(encoding="utf-8").strip() + "\n")

    for stage in sorted(stages):
        rows = stage_rows(stages[stage], root)
        label = {
            "A": "Stage A: one factor at a time (dev)",
            "B": "Stage B: combinations (dev)",
            "C": "Stage C: held-out confirm set",
        }.get(stage, f"Stage {stage}")
        parts.append(f"## {label}\n")
        if figures and rows:
            fig_dir.mkdir(parents=True, exist_ok=True)
            fname = f"stage_{stage.lower()}.png"
            _forest(rows, fig_dir / fname, label)
            parts.append(f"![{label}](figures/{fname})\n")
        parts.append(_table(rows))
        noisy = [r["run"]["name"] for r in rows if r["verdict"] == "within noise"]
        sig = [
            f"`{r['run']['name']}` ({r['verdict'].strip('*')})"
            for r in rows
            if r["verdict"].startswith("**")
        ]
        parts.append(
            f"\nSignificant vs baseline: {', '.join(sig) if sig else 'none'}. "
            f"Within noise: {len(noisy)} of {sum(1 for r in rows if r['cmp'])} comparisons.\n"
        )

    if figures and runs:
        fig_dir.mkdir(parents=True, exist_ok=True)
        _layer_share(runs, fig_dir / "layer_share.png")
        parts.append(
            "## Where retrieved context comes from\n\n![Layer share](figures/layer_share.png)\n"
        )

    if runs:
        total = {
            k: sum(r["cost"].get(k, 0) for r in runs)
            for k in ("llm_calls", "embed_calls", "cache_hits", "cache_misses")
        }
        parts.append(
            "## Cost\n\n"
            f"Across all runs: {total['llm_calls']} LLM calls, {total['embed_calls']} embedded "
            f"texts (local + API), {total['cache_hits']} cache hits, {total['cache_misses']} misses. "
            "Retrieval-only variants reuse cached trees and cost one reader call per question; "
            "re-running a finished config costs nothing.\n"
        )
        parts.append(
            "## Runs\n\n| Run id | Split | Method | Chunking | Clustering | Retrieval | Embedding |\n|---|---|---|---|---|---|---|\n"
        )
        for r in runs:
            f = r.get("factors", {})
            parts.append(
                f"| `{r['run_id']}` | {r['split']} | {r['method']} | {f.get('chunking', '')} "
                f"| {f.get('clustering', '')} | {f.get('retrieval', '')} | {f.get('embedding', '')} |\n"
            )

    path = out_dir / "README.md"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
