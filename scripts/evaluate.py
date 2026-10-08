"""Sweep a model over its knob on the development conversations, cross-validate the knob, write the
model's results to results/models/<model>/, and rebuild the comparison in results/comparison/.

    uv run python scripts/evaluate.py baseline

The held-out conversations are not touched.
"""

import argparse

from turn_detector.evaluation import cross_validate, load_conversations, score_knobs
from turn_detector.models import MODELS
from turn_detector.report import ModelResults, save, write_comparison
from turn_detector.split import load_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=list(MODELS))
    model = MODELS[parser.parse_args().model]

    conversations = load_conversations(load_split().development)
    knob_scores = score_knobs(model, conversations)
    cross_validation = cross_validate(knob_scores, [c.info for c in conversations])
    curve = knob_scores.curve([c.info.conversation_id for c in conversations])

    s = cross_validation.scores
    print(f"{model.name}: {len(conversations)} development conversations, {len(cross_validation.folds)} folds")
    print(f"  chosen {model.knob_name} = {cross_validation.knob:g} (per fold: {[f.knob for f in cross_validation.folds]})")
    print(
        f"  cross-validated: recall {s.recall:.3f}, false-cut-in rate {s.false_cut_in_rate:.3f}, "
        f"detection latency p50 {s.detection_latency_p50_ms:.0f} ms"
    )
    paths = save(ModelResults(model, curve, cross_validation)) + write_comparison()
    for path in paths:
        print(f"wrote {path}")
    print()
    print(next(path for path in paths if path.suffix == ".md").read_text())


if __name__ == "__main__":
    main()
