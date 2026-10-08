"""Results on disk: a model's sweep and cross-validation load back as they were saved."""

from turn_detector.evaluation import CrossValidation, Fold, Scores
from turn_detector.models import RegisteredModel, Rule
from turn_detector.models.baseline import Baseline
from turn_detector.report import ModelResults, frontier, load, save


def scores(setting: tuple[float, ...], recall: float, false_cut_in_rate: float) -> Scores:
    return Scores(setting, recall, false_cut_in_rate, 200.0, 1500.0, 1500.0, 9, 1, 2, 8)


def test_results_with_two_knobs_load_back(tmp_path):
    model = RegisteredModel(
        name="two knobs",
        knob_names=("threshold", "backstop (ms)"),
        settings=[(0.5, 1000.0), (0.9, 1500.0)],
        fit=lambda conversations: Rule(lambda setting: Baseline(timeout_ms=setting[1])),
        final=lambda setting: Baseline(timeout_ms=setting[1]),
        colour="#e07b39",
    )
    sweep = [scores((0.5, 1000.0), 0.9, 0.2), scores((0.9, 1500.0), 0.8, 0.05)]
    cross_validation = CrossValidation(
        folds=[Fold(["1", "2"], (0.9, 1500.0)), Fold(["3"], (0.5, 1000.0))],
        setting=(0.9, 1500.0),
        scores=scores((0.9, 1500.0), 0.8, 0.06),
    )
    save(ModelResults(model, sweep, cross_validation), tmp_path)
    assert load(model, tmp_path) == ModelResults(model, sweep, cross_validation)


def test_the_frontier_keeps_only_unbeaten_points():
    # (false-cut-in rate, median detection latency): (0.2, 900) is beaten by (0.1, 800).
    points = [(0.3, 400.0), (0.1, 800.0), (0.2, 900.0), (0.0, 1500.0), (0.1, 1000.0)]
    assert frontier(points, higher_is_better=False) == [(0.0, 1500.0), (0.1, 800.0), (0.3, 400.0)]
    assert frontier([(0.1, 0.8), (0.2, 0.7), (0.3, 0.9)], higher_is_better=True) == [(0.1, 0.8), (0.3, 0.9)]
