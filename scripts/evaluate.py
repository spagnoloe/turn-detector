"""Sweep a system over its knob on the development conversations, cross-validate the knob, and
rebuild the results table and figures in results/.

    uv run python scripts/evaluate.py baseline

The held-out conversations are not touched.
"""

import argparse

from turn_detector.baseline import SilenceTimeout
from turn_detector.evaluation import SystemSweep, cross_validate, load_conversations, score_knobs
from turn_detector.report import SystemResults, save, write_report
from turn_detector.split import load_split

SWEEPS = {
    "baseline": SystemSweep(
        name="silence timeout",
        knob_name="N (ms)",
        values=[float(n) for n in range(0, 3001, 50)],
        build=lambda n: SilenceTimeout(timeout_ms=n),
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("system", choices=sorted(SWEEPS))
    sweep = SWEEPS[parser.parse_args().system]

    conversations = load_conversations(load_split().development)
    knob_scores = score_knobs(sweep, conversations)
    cross_validation = cross_validate(knob_scores, [c.info for c in conversations])
    curve = knob_scores.curve([c.info.conversation_id for c in conversations])
    save(SystemResults(sweep.name, sweep.knob_name, curve, cross_validation))

    s = cross_validation.scores
    print(f"{sweep.name}: {len(conversations)} development conversations, {len(cross_validation.folds)} folds")
    print(f"  chosen {sweep.knob_name} = {cross_validation.knob:g} (per fold: {[f.knob for f in cross_validation.folds]})")
    print(
        f"  cross-validated: recall {s.recall:.3f}, false-cut-in rate {s.false_cut_in_rate:.3f}, "
        f"detection latency p50 {s.detection_latency_p50_ms:.0f} ms"
    )
    for path in write_report():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
