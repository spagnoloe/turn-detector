"""Evaluation results on disk: one folder per model, and a comparison across all of them.

    results/models/<model>/      that model only: sweep.csv (the sweep over its settings on all
                                 development conversations), cross_validation.json (the
                                 cross-validated operating point), held_out.json (the final
                                 model's scores on the held-out conversations, once they exist)
                                 and its two figures
    results/comparison/          every model evaluated so far: results.md (the results table)
                                 and the two figures with one line per model

The comparison is rebuilt from all model folders whenever a model is evaluated or the held-out
conversations are scored. Once held-out scores exist, the table leads with them, next to the
published TurnBench references, and the figures mark them.
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
from turn_detector.model import Setting  # noqa: E402
from turn_detector.models import MODELS, RegisteredModel  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
TABLE_NAME = "results.md"
HELD_OUT_NAME = "held_out.json"
LATENCY_FIGURE_NAME = "detection_latency_vs_false_cut_in_rate.png"
RECALL_FIGURE_NAME = "recall_vs_false_cut_in_rate.png"

INK, MUTED_INK, GRID = "#1a1a19", "#6b6a64", "#e4e3dc"


@dataclass(frozen=True)
class Reference:
    """A published TurnBench result, for comparison only: scored on TurnBench's test set, by a
    model trained on far more data than ours, with its own pause detection."""

    name: str
    trained_on: str
    recall: float
    false_cut_in_rate: float
    detection_latency_ms: tuple[float, float, float]  # p10, p50, p90


# EOT scores on the TurnBench test set, from TurnBench's results/leaderboard-test.json at the commit
# pyproject.toml pins (38a6f87), each at its operating point chosen on the whole dev set.
PUBLISHED_REFERENCES = [
    Reference(
        "VAP (published)",
        "Switchboard and Fisher, then fine-tuned on TurnBench's 104 h training set",
        0.845, 0.055, (-57.0, 368.0, 1537.0),
    ),
    Reference(
        "Pipecat Smart Turn v3 (published)",
        "Pipecat's own turn-completion data (not TurnBench)",
        0.752, 0.047, (729.0, 1017.0, 1175.0),
    ),
]  # fmt: skip
REFERENCE_INK = "#9a988f"


@dataclass(frozen=True)
class ModelResults:
    """One model's development sweep and cross-validated operating point, and its final model's
    held-out scores once they exist."""

    model: RegisteredModel
    sweep: list[Scores]
    cross_validation: CrossValidation
    held_out: Scores | None = None


def model_dir(model: RegisteredModel, results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "models" / model.slug


def comparison_dir(results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "comparison"


def setting_from(stored: str | list[float]) -> Setting:
    """A setting from a CSV cell (values separated by spaces) or a JSON list."""
    return tuple(float(value) for value in (stored.split() if isinstance(stored, str) else stored))


def scores_from(row: dict) -> Scores:
    """Scores from a CSV row or a JSON object."""
    return Scores(
        setting=setting_from(row["setting"]),
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
        writer.writerows(
            asdict(scores) | {"setting": " ".join(f"{value:g}" for value in scores.setting)} for scores in results.sweep
        )
    stored = {
        "model": results.model.name,
        "knob_names": results.model.knob_names,
        **asdict(results.cross_validation),
    }
    cross_validation.write_text(json.dumps(stored, indent=2) + "\n")
    return [sweep, cross_validation, *write_figures([results], directory, title=results.model.name)]


def save_held_out(
    model: RegisteredModel, scores: Scores, conversation_ids: list[str], results_dir: Path = RESULTS_DIR
) -> Path:
    """Write the final model's held-out scores. They are written once: the held-out conversations
    are scored a single time, so this refuses to replace scores already there."""
    path = model_dir(model, results_dir) / HELD_OUT_NAME
    if path.exists():
        raise FileExistsError(f"{path} exists: the held-out conversations have already been scored")
    stored = {"model": model.name, "knob_names": model.knob_names, "conversations": conversation_ids, "scores": asdict(scores)}
    path.write_text(json.dumps(stored, indent=2) + "\n")
    return path


def load(model: RegisteredModel, results_dir: Path = RESULTS_DIR) -> ModelResults:
    directory = model_dir(model, results_dir)
    stored = json.loads((directory / "cross_validation.json").read_text())
    with (directory / "sweep.csv").open() as file:
        sweep = [scores_from(row) for row in csv.DictReader(file)]
    cross_validation = CrossValidation(
        folds=[Fold(fold["held_out"], setting_from(fold["setting"])) for fold in stored["folds"]],
        setting=setting_from(stored["setting"]),
        scores=scores_from(stored["scores"]),
    )
    held_out_path = directory / HELD_OUT_NAME
    held_out = scores_from(json.loads(held_out_path.read_text())["scores"]) if held_out_path.exists() else None
    return ModelResults(model, sweep, cross_validation, held_out)


def load_all(results_dir: Path = RESULTS_DIR) -> list[ModelResults]:
    """Every registered model that has results, in registry order."""
    return [load(model, results_dir) for model in MODELS.values() if model_dir(model, results_dir).exists()]


def number(value: float, digits: int) -> str:
    return "—" if math.isnan(value) else f"{value:.{digits}f}"


def latencies(p10: float, p50: float, p90: float) -> str:
    return " / ".join(number(ms, 0) for ms in (p10, p50, p90))


def uncertainty_lines(scores: Scores) -> list[str]:
    """How far the held-out rates can move: what one event is worth, and a 95% binomial interval."""
    eots, pauses = scores.tp + scores.fn, scores.fp + scores.tn

    def half_width(rate: float, n: int) -> float:
        return 1.96 * math.sqrt(rate * (1 - rate) / n)

    return [
        f"The held-out set is small: {eots} EOTs and {pauses} mid-turn pauses, so one EOT moves recall by "
        f"{1 / eots:.3f} and one mid-turn pause the false-cut-in rate by {1 / pauses:.3f}. A 95% binomial "
        f"interval is about ±{half_width(scores.recall, eots):.2f} on recall and "
        f"±{half_width(scores.false_cut_in_rate, pauses):.2f} on the false-cut-in rate, and wider in truth, "
        "since the pauses of one conversation are correlated."
    ]


def held_out_lines(results: list[ModelResults]) -> list[str]:
    """The held-out table: each final model once, then the published references."""
    lines = [
        "## Held-out conversations: the final scores",
        "",
        "The 12 held-out conversations, scored once, in a single run, by each final model (trained on "
        "all 26 development conversations) at the setting chosen in cross-validation, with no tuning "
        "afterwards. Scored with TurnBench's EOT scorer. These are the only clean scores: the audio "
        "model's encoder layer (8) and regularisation (C = 1e-4) were picked on a third of the "
        "development conversations, so the cross-validated development scores below are slightly "
        "optimistic for the audio-only and combined models.",
        "",
        *uncertainty_lines(next(result.held_out for result in results if result.held_out is not None)),
        "",
        "| Model | Knobs | Setting | Recall | False-cut-in rate | Detection latency p10 / p50 / p90 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        if (scores := result.held_out) is None:
            continue
        lines.append(
            f"| {result.model.name} | {', '.join(result.model.knob_names)} "
            f"| {', '.join(f'{value:g}' for value in scores.setting)} | {number(scores.recall, 3)} "
            f"| {number(scores.false_cut_in_rate, 3)} "
            f"| {latencies(scores.detection_latency_p10_ms, scores.detection_latency_p50_ms, scores.detection_latency_p90_ms)} |"
        )
    for reference in PUBLISHED_REFERENCES:
        lines.append(
            f"| {reference.name} | — | — | {reference.recall:.3f} | {reference.false_cut_in_rate:.3f} "
            f"| {latencies(*reference.detection_latency_ms)} |"
        )
    lines += [
        "",
        "The published rows are outside references, not comparable on equal terms:",
        "",
        "- they are scored on TurnBench's test set, a different and larger split than our held-out "
        "conversations, at an operating point chosen on the whole dev set;",
        *(f"- {reference.name.removesuffix(' (published)')} was trained on {reference.trained_on};" for reference in PUBLISHED_REFERENCES),
        "- they detect pauses from the audio themselves, whereas every model here is given the "
        "annotators' consensus segment ends (a perfect VAD) and, for text, human transcripts (a "
        "perfect ASR, read 200 ms after each segment ends).",
        "",
        "The full list of assumptions is in [`docs/solution.md`](../../docs/solution.md#assumptions).",
        "",
    ]
    return lines


def write_table(results: list[ModelResults], path: Path) -> None:
    """Recall and median detection latency at the operating point: on the held-out conversations
    once they are scored, then cross-validated on the development conversations."""
    lines = [f"# Results at a false-cut-in rate of {MAX_FALSE_CUT_IN_RATE:.2f} or less", ""]
    if any(result.held_out is not None for result in results):
        lines += [*held_out_lines(results), "## Development conversations, cross-validated", ""]
    lines += [
        "Development conversations, cross-validated: each speaker group is scored at the setting "
        "chosen without it (highest recall within the false-cut-in budget), and the folds are pooled, "
        "so the scores mix the per-fold settings. Scored with TurnBench's EOT scorer. The setting "
        "chosen on all development conversations is the one a final model would use.",
        "",
        "| Model | Knobs | Setting chosen on all development (per-fold range) | Recall | False-cut-in rate "
        "| Detection latency p10 / p50 / p90 (ms) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in results:
        scores = result.cross_validation.scores
        chosen = ", ".join(f"{value:g}" for value in result.cross_validation.setting)
        fold_ranges = ", ".join(
            f"{min(values):g}–{max(values):g}"
            for values in zip(*(fold.setting for fold in result.cross_validation.folds), strict=True)
        )
        lines.append(
            f"| {result.model.name} | {', '.join(result.model.knob_names)} | {chosen} ({fold_ranges}) "
            f"| {number(scores.recall, 3)} | {number(scores.false_cut_in_rate, 3)} "
            f"| {latencies(scores.detection_latency_p10_ms, scores.detection_latency_p50_ms, scores.detection_latency_p90_ms)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def frontier(points: list[tuple[float, float]], higher_is_better: bool) -> list[tuple[float, float]]:
    """The (false-cut-in rate, metric) points no other point beats: none has a false-cut-in rate
    as low and a metric as good. With one knob this is (nearly) the whole sweep; with two it is
    the best trade-off the model's settings reach."""
    sign = -1 if higher_is_better else 1
    best, kept = math.inf, []
    for rate, value in sorted(points, key=lambda point: (point[0], sign * point[1])):
        if sign * value < best:
            best = sign * value
            kept.append((rate, value))
    return kept


def mark_held_out(axes, results: list[ModelResults], metric: str) -> None:
    """Each final model's held-out score as a hollow square in its colour, nested (earlier models
    larger) so that coinciding scores all stay visible, and the published references (TurnBench
    test set) as grey triangles."""
    held_out = [result for result in results if result.held_out is not None]
    for index, result in enumerate(held_out):
        assert result.held_out is not None
        axes.plot(
            result.held_out.false_cut_in_rate, getattr(result.held_out, metric),
            marker="s", markersize=7 + 3 * (len(held_out) - 1 - index), markerfacecolor="none",
            markeredgecolor=result.model.colour, markeredgewidth=1.5, linestyle="none", zorder=5,
        )  # fmt: skip
    axes.plot([], [], marker="s", markersize=8, markerfacecolor="none", markeredgecolor=MUTED_INK, markeredgewidth=1.5, linestyle="none", label="held-out, final model")
    for index, reference in enumerate(PUBLISHED_REFERENCES):
        value = reference.recall if metric == "recall" else reference.detection_latency_ms[1]
        axes.plot(
            reference.false_cut_in_rate, value, marker="^", markersize=8, color=REFERENCE_INK, linestyle="none",
            label="published, TurnBench test set" if index == 0 else None, zorder=5,
        )  # fmt: skip
        axes.annotate(
            reference.name.removesuffix(" (published)"), (reference.false_cut_in_rate, value),
            xytext=(-6, -14), textcoords="offset points", ha="left", color=MUTED_INK, fontsize=7,
        )  # fmt: skip


def plot_against_false_cut_in_rate(
    results: list[ModelResults], metric: str, higher_is_better: bool, ylabel: str, title: str, path: Path
) -> None:
    """One line per model: the frontier of its development sweep, `metric` against the
    false-cut-in rate, with the chosen setting marked, and the held-out scores and published
    references once the held-out conversations are scored."""
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
    for index, result in enumerate(results):
        model = result.model
        points = [
            (scores.false_cut_in_rate, getattr(scores, metric))
            for scores in result.sweep
            if not math.isnan(getattr(scores, metric))
        ]
        x, y = zip(*frontier(points, higher_is_better))
        axes.plot(x, y, color=model.colour, linewidth=2, marker="o", markersize=3, label=model.name, zorder=3)
        chosen = next(scores for scores in result.sweep if scores.setting == result.cross_validation.setting)
        axes.plot(
            chosen.false_cut_in_rate, getattr(chosen, metric), marker="o", markersize=8,
            color=model.colour, markeredgecolor="white", markeredgewidth=2, zorder=4,
        )  # fmt: skip
        axes.annotate(
            model.describe(result.cross_validation.setting),
            (chosen.false_cut_in_rate, getattr(chosen, metric)),
            xytext=(8, -12 - 11 * index),  # stacked, since chosen settings can coincide
            textcoords="offset points",
            color=INK,
            fontsize=8,
        )
    if any(result.held_out is not None for result in results):
        mark_held_out(axes, results, metric)
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
    if len(results) > 1 or any(result.held_out is not None for result in results):  # else the title names the line
        axes.legend(frameon=False, labelcolor=INK, fontsize=8)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_figures(results: list[ModelResults], directory: Path, title: str) -> list[Path]:
    """The main figure (median detection latency) and the secondary one (recall), both against the
    false-cut-in rate, on the development sweep."""
    detection_latency, recall = directory / LATENCY_FIGURE_NAME, directory / RECALL_FIGURE_NAME
    plot_against_false_cut_in_rate(
        results, "detection_latency_p50_ms", False, "Median detection latency (ms)", title, detection_latency
    )
    plot_against_false_cut_in_rate(results, "recall", True, "Recall (share of EOTs fired on in time)", title, recall)
    return [detection_latency, recall]


def write_comparison(results_dir: Path = RESULTS_DIR) -> list[Path]:
    """Rebuild the comparison table and figures from every model's saved results."""
    results = load_all(results_dir)
    directory = comparison_dir(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    table = directory / TABLE_NAME
    write_table(results, table)
    return [table, *write_figures(results, directory, title="All models")]
