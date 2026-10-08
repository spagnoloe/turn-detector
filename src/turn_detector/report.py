"""Evaluation results on disk: one folder per model, and a comparison across all of them.

    results/models/<model>/      that model only: sweep.csv (the sweep over its knob on all
                                 development conversations), cross_validation.json (the
                                 cross-validated operating point) and its two figures
    results/comparison/          every model evaluated so far: results.md (the results table)
                                 and the two figures with one line per model

The comparison is rebuilt from all model folders whenever a model is evaluated.
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
from turn_detector.models import MODELS, RegisteredModel  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
TABLE_NAME = "results.md"
LATENCY_FIGURE_NAME = "detection_latency_vs_false_cut_in_rate.png"
RECALL_FIGURE_NAME = "recall_vs_false_cut_in_rate.png"

INK, MUTED_INK, GRID = "#1a1a19", "#6b6a64", "#e4e3dc"


@dataclass(frozen=True)
class ModelResults:
    """One model's development sweep and cross-validated operating point."""

    model: RegisteredModel
    sweep: list[Scores]
    cross_validation: CrossValidation


def model_dir(model: RegisteredModel, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "models" / model.slug


def comparison_dir(results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "comparison"


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


def save(results: ModelResults, results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Write one model's results folder, its own figures included."""
    directory = model_dir(results.model, results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    sweep, cross_validation = directory / "sweep.csv", directory / "cross_validation.json"
    with sweep.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[f.name for f in fields(Scores)])
        writer.writeheader()
        writer.writerows(asdict(scores) for scores in results.sweep)
    stored = {"model": results.model.name, "knob_name": results.model.knob_name, **asdict(results.cross_validation)}
    cross_validation.write_text(json.dumps(stored, indent=2) + "\n")
    return [sweep, cross_validation, *write_figures([results], directory, title=results.model.name)]


def load(model: RegisteredModel, results_dir: Path = RESULTS_DIR) -> ModelResults:
    directory = model_dir(model, results_dir)
    stored = json.loads((directory / "cross_validation.json").read_text())
    with (directory / "sweep.csv").open() as file:
        sweep = [scores_from(row) for row in csv.DictReader(file)]
    cross_validation = CrossValidation(
        folds=[Fold(**fold) for fold in stored["folds"]],
        knob=stored["knob"],
        scores=scores_from(stored["scores"]),
    )
    return ModelResults(model, sweep, cross_validation)


def load_all(results_dir: Path = RESULTS_DIR) -> list[ModelResults]:
    """Every registered model that has results, in registry order."""
    return [load(model, results_dir) for model in MODELS.values() if model_dir(model, results_dir).exists()]


def write_table(results: list[ModelResults], path: Path) -> None:
    """Recall and median detection latency at the operating point, cross-validated."""

    def number(value: float, digits: int) -> str:
        return "—" if math.isnan(value) else f"{value:.{digits}f}"

    lines = [
        f"# Results at a false-cut-in rate of {MAX_FALSE_CUT_IN_RATE:.2f} or less",
        "",
        "Development conversations, cross-validated: each speaker group is scored at the knob setting "
        "chosen without it (highest recall within the false-cut-in budget), and the folds are pooled, "
        "so the scores mix the per-fold settings. Scored with TurnBench's EOT scorer. The setting "
        "chosen on all development conversations is the one a final model would use.",
        "",
        "| Model | Knob | Setting chosen on all development (per-fold range) | Recall | False-cut-in rate "
        "| Detection latency p10 / p50 / p90 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        scores = result.cross_validation.scores
        fold_settings = [fold.knob for fold in result.cross_validation.folds]
        detection_latency = " / ".join(
            number(ms, 0)
            for ms in (scores.detection_latency_p10_ms, scores.detection_latency_p50_ms, scores.detection_latency_p90_ms)
        )
        lines.append(
            f"| {result.model.name} | {result.model.knob_name} "
            f"| {result.cross_validation.knob:g} ({min(fold_settings):g}–{max(fold_settings):g}) "
            f"| {number(scores.recall, 3)} | {number(scores.false_cut_in_rate, 3)} | {detection_latency} |"
        )
    path.write_text("\n".join(lines) + "\n")


def plot_against_false_cut_in_rate(
    results: list[ModelResults], metric: str, ylabel: str, title: str, path: Path
) -> None:
    """One line per model: its development sweep, `metric` against the false-cut-in rate."""
    figure, axes = plt.subplots(figsize=(7, 4.5), dpi=150)
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
    for result in results:
        model = result.model
        points = sorted(
            (scores.false_cut_in_rate, getattr(scores, metric))
            for scores in result.sweep
            if not math.isnan(getattr(scores, metric))
        )
        x, y = zip(*points)
        axes.plot(x, y, color=model.colour, linewidth=2, marker="o", markersize=3, label=model.name, zorder=3)
        chosen = next(scores for scores in result.sweep if scores.knob == result.cross_validation.knob)
        axes.plot(
            chosen.false_cut_in_rate, getattr(chosen, metric), marker="o", markersize=8,
            color=model.colour, markeredgecolor="white", markeredgewidth=2, zorder=4,
        )  # fmt: skip
        axes.annotate(
            f"{model.knob_name} = {result.cross_validation.knob:g}",
            (chosen.false_cut_in_rate, getattr(chosen, metric)),
            xytext=(8, -12),
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    axes.set_title(title, color=INK, loc="left")
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
    if len(results) > 1:  # a single line is named by the title
        axes.legend(frameon=False, labelcolor=INK)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_figures(results: list[ModelResults], directory: Path, title: str) -> list[Path]:
    """The main figure (median detection latency) and the secondary one (recall), both against the
    false-cut-in rate, on the development sweep."""
    detection_latency, recall = directory / LATENCY_FIGURE_NAME, directory / RECALL_FIGURE_NAME
    plot_against_false_cut_in_rate(
        results, "detection_latency_p50_ms", "Median detection latency (ms)", title, detection_latency
    )
    plot_against_false_cut_in_rate(results, "recall", "Recall (share of EOTs fired on in time)", title, recall)
    return [detection_latency, recall]


def write_comparison(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Rebuild the comparison table and figures from every model's saved results."""
    results = load_all(results_dir)
    directory = comparison_dir(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / TABLE_NAME
    write_table(results, table)
    return [table, *write_figures(results, directory, title="All models")]
