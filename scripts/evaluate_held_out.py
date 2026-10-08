"""Score the held-out conversations once, with every evaluated model, and rebuild the comparison.

    uv run python scripts/evaluate_held_out.py

Each final model (trained on all development conversations and saved by `scripts/evaluate.py`,
or the baseline's timeout) is scored on the held-out conversations at the setting chosen in
cross-validation, read from results/models/<model>/cross_validation.json; a saved model at any
other setting is refused. The scores go to results/models/<model>/held_out.json, and the
comparison in results/comparison/ then leads with them.

The held-out conversations are scored a single time, for every model in the same run, with no
tuning afterwards: the script refuses to run if any model already has held-out scores.
"""

import argparse

from turn_detector.evaluation import load_conversations, score_held_out
from turn_detector.report import HELD_OUT_NAME, load_all, model_dir, save_held_out, write_comparison
from turn_detector.split import load_split


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    results = load_all()
    if scored := [result.model.name for result in results if (model_dir(result.model) / HELD_OUT_NAME).exists()]:
        raise SystemExit(f"the held-out conversations have already been scored ({', '.join(scored)}); they are scored once")

    # Load every final model before scoring any, so a stale artifact stops the run before it starts.
    finals = [(result, result.model.final(result.cross_validation.setting)) for result in results]
    conversations = load_conversations(load_split().held_out)
    ids = [c.info.conversation_id for c in conversations]
    print(f"{len(conversations)} held-out conversations: {', '.join(ids)}")
    scores = {}
    for result, model in finals:
        print(f"{result.model.name}: {result.model.describe(result.cross_validation.setting)}", flush=True)
        s = scores[result.model.slug] = score_held_out(model, result.cross_validation.setting, conversations)
        print(
            f"  recall {s.recall:.3f}, false-cut-in rate {s.false_cut_in_rate:.3f}, "
            f"detection latency p50 {s.detection_latency_p50_ms:.0f} ms"
        )
    # Written only once every model is scored, so a failed run leaves nothing behind.
    paths = [save_held_out(result.model, scores[result.model.slug], ids) for result, _ in finals] + write_comparison()
    for path in paths:
        print(f"wrote {path}")
    print()
    print(next(path for path in paths if path.suffix == ".md").read_text())


if __name__ == "__main__":
    main()
