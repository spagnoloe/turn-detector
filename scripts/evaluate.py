"""Fit a model to the development conversations, sweep it over its settings, cross-validate the
setting, write the model's results to results/models/<model>/, save a trained model at the chosen
setting to
artifacts/<model>/, and rebuild the comparison in results/comparison/.

    uv run python scripts/evaluate.py baseline
    uv run python scripts/evaluate.py text-only

The held-out conversations are not touched.
"""

import argparse

from turn_detector.evaluation import cross_validate, load_conversations, score_settings
from turn_detector.models import MODELS
from turn_detector.report import ModelResults, save, write_comparison
from turn_detector.split import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=list(MODELS))
    model = MODELS[parser.parse_args().model]

    conversations = load_conversations(load_split().development)
    fitted = model.fit(conversations)
    setting_scores = score_settings(fitted, model.settings, conversations)
    cross_validation = cross_validate(setting_scores, [c.info for c in conversations])
    curve = setting_scores.curve([c.info.conversation_id for c in conversations])

    s = cross_validation.scores
    print(f"{model.name}: {len(conversations)} development conversations, {len(cross_validation.folds)} folds")
    print(f"  chosen {model.describe(cross_validation.setting)}")
    print(f"  per fold: {[fold.setting for fold in cross_validation.folds]}")
    print(
        f"  cross-validated: recall {s.recall:.3f}, false-cut-in rate {s.false_cut_in_rate:.3f}, "
        f"detection latency p50 {s.detection_latency_p50_ms:.0f} ms"
    )
    paths = save(ModelResults(model, curve, cross_validation)) + fitted.save(cross_validation.setting) + write_comparison()
    for path in paths:
        print(f"wrote {path}")
    print()
    print(next(path for path in paths if path.suffix == ".md").read_text())


if __name__ == "__main__":
    main()
