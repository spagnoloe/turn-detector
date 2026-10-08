# Turn Detector

Decides, while a user is speaking to a voice agent, whether the user has finished their turn and the agent should respond. Domain terms are defined in [`CONTEXT.md`](CONTEXT.md); decisions are recorded in [`docs/adr/`](docs/adr/).

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python 3.14 is installed by uv if missing.

```bash
uv sync
```

This also installs the [TurnBench](https://github.com/SesameAILabs/turnbench) repository (pinned to a commit) for its scorer, sweep tool and annotator-agreement logic.

## Data

The project uses the TurnBench dev set, [`mundo-ai/turn-benchmark-dev`](https://huggingface.co/datasets/mundo-ai/turn-benchmark-dev) (38 conversations, about 7 h, non-commercial licence). It is gated, with automatic approval:

1. Create a Hugging Face account and accept the dataset's terms on its page.
2. Create a **Read** token at <https://huggingface.co/settings/tokens>.
3. Log in from a regular terminal (the prompt needs hidden input): `uv run hf auth login`.
4. Download (about 4 GB) into the git-ignored `data/` folder, at the revision TurnBench pins:

```bash
uv run python scripts/download_data.py
```

Trained models go to the git-ignored `artifacts/` folder.

## Development / held-out split

The 38 conversations are split into 26 development and 12 held-out conversations ([ADR 0003](docs/adr/0003-train-on-turnbench-dev.md)). The split is stored in [`src/turn_detector/split.json`](src/turn_detector/split.json) and is the source of truth. The same actors appear in several conversations, so the split is speaker-disjoint: every conversation an actor took part in is on the same side. Among such splits it picks the one whose conversation-type mix best matches the whole set, so the result is deterministic. `uv run python scripts/make_split.py` regenerates it.

## Gold events

`turn_detector.events.build_events` turns a conversation's three annotator tracks into each speaker's gold **EOTs** and **mid-turn pauses**, using TurnBench's own gold construction. A segment needs 2-of-3 annotator agreement within ±200 ms, and its time is the agreeing annotators' median. Segments without agreement are dropped. This makes the training labels the same events the official scorer evaluates against.

```bash
uv run python scripts/count_events.py
```

This prints counts per split and per speaker. The dev-set totals (1904 EOTs, 1063 mid-turn pauses) match TurnBench's published numbers exactly.

## Detector systems

Every system implements one interface (`turn_detector.detector`): it takes one speaker's side of a conversation (that speaker's audio, their speech segments with transcripts, and the other speaker's turns) and returns its **firing** times. Each system owns its firing rule. The segments are TurnBench's 2-of-3 consensus segments (`turn_detector.timeline`), so their ends are the pauses every system sees: a perfect pause detector in place of a real VAD. Speech without annotator agreement is missing from the timeline, and each segment's transcript comes from the closest-matching annotator segment (the annotators' texts agree 99% of the time).

- **Silence timeout** (`turn_detector.baseline`): fires at segment end + N ms if the user hasn't resumed by then.

`tests/test_causality.py` checks that every system in its `SYSTEMS` list is causal: rewriting the audio, text or segments after a time t never changes the firings before t.

## Evaluation

```bash
uv run python scripts/evaluate.py baseline
```

This sweeps the system's knob (here N, 0 to 3000 ms) on the development conversations and writes to `results/`:

- `<system>/sweep.csv`: recall, **false-cut-in rate** and p10/p50/p90 **detection latency** at every knob setting, on all development conversations;
- `<system>/cross_validation.json`: the knob chosen per fold and the pooled cross-validated scores;
- `results.md`: recall and detection latency at a false-cut-in rate of 0.10 or less, one row per system run so far;
- `detection_latency_vs_false_cut_in_rate.png` (main figure) and `recall_vs_false_cut_in_rate.png`: one line per system.

The knob is chosen by cross-validation that leaves one speaker group out per fold (9 folds, [ADR 0003](docs/adr/0003-train-on-turnbench-dev.md)). Each fold picks the highest-recall setting within the 0.10 false-cut-in budget on the other folds (TurnBench's operating-point rule, ties to lower median latency), and the held-out folds' scores are pooled for the table. The figures show the sweep on all development conversations. The held-out conversations aren't scored.

Scores come from TurnBench's own code: firings are checked with its submission validators and scored with its `score_task` against its gold, exactly as `turnbench.score` does, on the EOT task only. Each conversation is scored once per knob setting, and folds are sums of those scores. Two departures: the sweep covers a subset of conversations, where `turnbench.score` insists on the whole dev set, and ties between knob settings with equal recall go to the lower median detection latency, where `turnbench.sweep.operating_point` leaves them unbroken.

## Tests

```bash
uv run pytest
uv run ty check src tests scripts   # typecheck
```
