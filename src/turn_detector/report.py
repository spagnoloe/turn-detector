"""Evaluation results on disk, and the results table and figures built from every system there.

Each system writes its own folder under `results/`: `sweep.csv` (the sweep over its knob on all
development conversations) and `cross_validation.json` (the cross-validated operating point).
The table and figures are rebuilt from all system folders, so they compare every system run so far.
"""

import csv
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from turn_detector.data import REPO_ROOT  # noqa: E402
from turn_detector.evaluation import MAX_FALSE_CUT_IN_RATE, CrossValidation, Fold, Scores  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
TABLE_NAME = "results.md"
LATENCY_FIGURE_NAME = "detection_latency_vs_false_cut_in_rate.png"
RECALL_FIGURE_NAME = "recall_vs_false_cut_in_rate.png"

# Each system keeps its colour in every figure; systems are listed in this order.
SYSTEM_COLOURS = {"silence timeout": "#2a78d6"}
SPARE_COLOURS = ["#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED_INK, GRID = "#1a1a19", "#6b6a64", "#e4e3dc"


@dataclass(frozen=True)
class SystemResults:
    """One system family's development sweep and cross-validated operating point."""

    name: str
    knob_name: str
    sweep: list[Scores]
    cross_validation: CrossValidation


def system_dir(name: str, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / name.replace(" ", "-")


def scores_from(row: dict) -> Scores:
    """Scores from a CSV row or a JSON object."""
    return Scores(
        knob=float(row["knob"]),
        recall=float(row["recall"]),
        false_cut_in_rate=float(row["false_cut_in_rate"]),
        detection_latency_p10_ms=float(row["detection_latency_p10_ms"]),
        detection_latency_p50_ms=float(row["detection_latency_p50_ms"]),
        detection_latency_p90_ms=float(row["detection_latency_p90_ms"]),
        tp=int(row["tp"]),
        fn=int(row["fn"]),
        fp=int(row["fp"]),
        tn=int(row["tn"]),
    )


def save(results: SystemResults, results_dir: Path = RESULTS_DIR) -> None:
    directory = system_dir(results.name, results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "sweep.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(Scores)])
        writer.writeheader()
        writer.writerows(asdict(scores) for scores in results.sweep)
    stored = {"name": results.name, "knob_name": results.knob_name, **asdict(results.cross_validation)}
    (directory / "cross_validation.json").write_text(json.dumps(stored, indent=2) + "\n")


def load(directory: Path) -> SystemResults:
    stored = json.loads((directory / "cross_validation.json").read_text())
    with (directory / "sweep.csv").open() as file:
        sweep = [scores_from(row) for row in csv.DictReader(file)]
    cross_validation = CrossValidation(
        folds=[Fold(**fold) for fold in stored["folds"]],
        knob=stored["knob"],
        scores=scores_from(stored["scores"]),
    )
    return SystemResults(stored["name"], stored["knob_name"], sweep, cross_validation)


def load_all(results_dir: Path = RESULTS_DIR) -> list[SystemResults]:
    """Every system with results, in SYSTEM_COLOURS order, then by name."""
    results = [load(path.parent) for path in results_dir.glob("*/cross_validation.json")]
    order = list(SYSTEM_COLOURS)
    return sorted(
        results, key=lambda system: (order.index(system.name) if system.name in order else len(order), system.name)
    )


def colours(results: list[SystemResults]) -> dict[str, str]:
    spare = iter(SPARE_COLOURS)
    return {system.name: SYSTEM_COLOURS.get(system.name) or next(spare) for system in results}


def write_table(results: list[SystemResults], path: Path) -> None:
    """Recall and median detection latency at the operating point, cross-validated."""

    def number(value: float, digits: int) -> str:
        return "—" if math.isnan(value) else f"{value:.{digits}f}"

    lines = [
        f"# Results at a false-cut-in rate of {MAX_FALSE_CUT_IN_RATE:.2f} or less",
        "",
        "Development conversations, cross-validated: each speaker group is scored at the knob setting "
        "chosen without it (highest recall within the false-cut-in budget), and the folds are pooled, "
        "so the scores mix the per-fold settings. Scored with TurnBench's EOT scorer. The setting "
        "chosen on all development conversations is the one a final system would use.",
        "",
        "| System | Knob | Setting chosen on all development (per-fold range) | Recall | False-cut-in rate "
        "| Detection latency p10 / p50 / p90 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for system in results:
        scores = system.cross_validation.scores
        fold_settings = [fold.knob for fold in system.cross_validation.folds]
        detection_latency = " / ".join(
            number(ms, 0)
            for ms in (scores.detection_latency_p10_ms, scores.detection_latency_p50_ms, scores.detection_latency_p90_ms)
        )
        lines.append(
            f"| {system.name} | {system.knob_name} "
            f"| {system.cross_validation.knob:g} ({min(fold_settings):g}–{max(fold_settings):g}) "
            f"| {number(scores.recall, 3)} | {number(scores.false_cut_in_rate, 3)} | {detection_latency} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_against_false_cut_in_rate(results: list[SystemResults], metric: str, ylabel: str, path: Path) -> None:
    """One line per system: its development sweep, `metric` against the false-cut-in rate."""
    figure, axes = plt.subplots(figsize=(7, 4.5), dpi=150)
    colour = colours(results)
    axes.axvline(MAX_FALSE_CUT_IN_RATE, color=MUTED_INK, linewidth=1, linestyle="--", zorder=1)
    axes.annotate(
        f"budget {MAX_FALSE_CUT_IN_RATE:.2f}",
        (MAX_FALSE_CUT_IN_RATE, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        color=MUTED_INK,
        fontsize=8,
    )
    for system in results:
        points = sorted(
            (scores.false_cut_in_rate, getattr(scores, metric))
            for scores in system.sweep
            if not math.isnan(getattr(scores, metric))
        )
        x, y = zip(*points)
        axes.plot(x, y, color=colour[system.name], linewidth=2, marker="o", markersize=3, label=system.name, zorder=3)
        chosen = next(scores for scores in system.sweep if scores.knob == system.cross_validation.knob)
        axes.plot(
            chosen.false_cut_in_rate, getattr(chosen, metric), marker="o", markersize=8,
            color=colour[system.name], markeredgecolor="white", markeredgewidth=2, zorder=4,
        )  # fmt: skip
        axes.annotate(
            f"{system.knob_name} = {system.cross_validation.knob:g}",
            (chosen.false_cut_in_rate, getattr(chosen, metric)),
            xytext=(8, -12),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    axes.set_xlabel("False-cut-in rate (share of mid-turn pauses fired in)", color=INK)
    axes.set_ylabel(ylabel, color=INK)
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    axes.grid(color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED_INK)
    axes.tick_params(colors=MUTED_INK)
    axes.legend(frameon=False, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_report(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Rebuild the results table and both figures from every system's saved results."""
    results = load_all(results_dir)
    table, detection_latency, recall = (results_dir / n for n in (TABLE_NAME, LATENCY_FIGURE_NAME, RECALL_FIGURE_NAME))
    write_table(results, table)
    plot_against_false_cut_in_rate(results, "detection_latency_p50_ms", "Median detection latency (ms)", detection_latency)
    plot_against_false_cut_in_rate(results, "recall", "Recall (share of EOTs fired on in time)", recall)
    return [table, detection_latency, recall]
