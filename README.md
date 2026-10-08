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

## Models

A **model** is one complete way of deciding when the user's turn has ended: a plain rule (the baseline) or a trained classifier plus a firing rule. There are four: the baseline, text-only, audio-only and combined.

Code is split into what every model shares and what belongs to one model:

| Code | Scope |
|---|---|
| `turn_detector.model` | Shared: the interface every model implements. A model takes one speaker's side of a conversation (that speaker's audio, their speech segments with transcripts, and the other speaker's turns) and returns its **firing** times. |
| `turn_detector.timeline` | Shared: builds each speaker's side from the annotations. |
| `turn_detector.evaluation`, `turn_detector.report` | Shared: scoring, the sweep over settings, cross-validation, results table and figures. |
| `turn_detector.models` | One module per model, plus `MODELS`, the registry of each model's knobs, swept settings and figure colour. |
| `turn_detector.models.classified` | Shared by the trained models: P(EOT) at the times a model may fire in each pause, the threshold-plus-backstop firing rule, cross-fitting per speaker group, saving a head with its encoder's name, and each model's `Serving` description (which requests it answers, and how soon into a pause it may fire). |
| `turn_detector.models.baseline` | The baseline, a silence timeout: fires at segment end + N ms if the user hasn't resumed by then. This is what plain voice-activity detection achieves. |
| `turn_detector.models.text_only` | The text-only model: at each segment end, a frozen `all-MiniLM-L6-v2` encoder with a logistic-regression head reads the other speaker's previous turn and the user's turn so far (the last 64 tokens) and outputs P_text(EOT). It fires at segment end + 200 ms (the assumed ASR lag) if P_text ≥ the threshold, otherwise at segment end + a silence backstop, either only if the user is still silent. The threshold and the backstop are its two knobs, tuned together. With the threshold above every P_text it is the baseline, with the backstop as its timeout, so tuning both never does worse than the baseline on the data they are tuned on. |
| `turn_detector.models.audio_only` | The audio-only model: every 50 ms while the user is silent, a frozen `wav2vec2-base` encoder hears the user's last 1 s of audio (resampled from 48 kHz to 16 kHz), and a logistic-regression head reads three features: the mean of the window's frames, the mean of its last speech frames (the last 200 ms before the pause) and the silence duration so far. It outputs P_audio(EOT) and fires on the rising edge where it first reaches the threshold, or at segment end + a silence backstop if that comes first, at most once per pause and only if the user is still silent. It stops listening 3 s into a pause, since TurnBench never counts a later firing as a hit. Like the text-only model, the threshold and the backstop are its two knobs, tuned together, and with the threshold above every P_audio it is the baseline with the backstop as its timeout. |
| `turn_detector.models.combined` | The combined model: every 50 ms while the user is silent, a logistic regression over [P_audio, P_text, silence duration] outputs P(EOT), with the audio-only model's firing rule and knobs. P_text is the text-only classifier's, on the context at the user's latest segment end whose text has arrived (200 ms after it ended); until then the previous one is held. The audio-only model already has the backstop, so what this model adds is the text alone. The fusion head is trained on out-of-fold P_audio and P_text, at the audio-only head's pause samples. |

The segments are TurnBench's 2-of-3 consensus segments, so their ends are the pauses every model sees: a perfect pause detector in place of a real VAD. Speech without annotator agreement is missing from the timeline, and each segment's transcript comes from the closest-matching annotator segment (the annotators' texts agree 99% of the time).

Transcripts stand in for a streaming ASR. A segment's text is readable only once the segment has ended. Before the text model reads them, bracketed annotation tags such as `[laughs]` are stripped, since an ASR never outputs them, and fillers such as "um" and "uh" are kept, since an ASR usually does.

The audio-only model hears real audio, but its pauses and silence duration come from the same consensus segments. It is trained on samples every 50 ms through each pause with a gold label, up to 1 s into it (EOT or mid-turn pause), plus one sample per second of speech (not EOT). Its window is resampled on its own, so no audio after t can reach the window used at t.

`tests/test_causality.py` checks that every model in its `MODELS_UNDER_TEST` list is causal: rewriting the audio, text or segments after a time t never changes the firings before t.

## Evaluation

```bash
uv run python scripts/evaluate.py baseline
uv run python scripts/evaluate.py text-only
uv run python scripts/evaluate.py audio-only
uv run python scripts/evaluate.py combined
```

This fits the model to the development conversations, sweeps its settings, one value per knob (for the baseline, the silence timeout N from 0 to 3000 ms in steps of 50; for the text-only model, every pair of a P_text threshold from 0 to 1 in steps of 0.01 and a backstop from 200 to 3000 ms in steps of 50; for the audio-only model, every pair of a P_audio threshold from 0 to 1 in steps of 0.001, since its probabilities bunch up near the chosen threshold, and a backstop from 200 to 3000 ms in steps of 50; for the combined model, the audio-only model's settings), and writes:

- `results/models/<model>/`, that model only:
  - `sweep.csv`: recall, **false-cut-in rate** and p10/p50/p90 **detection latency** at every setting, on all development conversations;
  - `cross_validation.json`: the setting chosen per fold and the pooled cross-validated scores;
  - `detection_latency_vs_false_cut_in_rate.png` and `recall_vs_false_cut_in_rate.png`: its sweep's frontier: at each false-cut-in rate, the best value any setting reaches.
- `results/comparison/`, every model evaluated so far, rebuilt on each run:
  - `results.md`: recall and detection latency at a false-cut-in rate of 0.10 or less, one row per model;
  - `detection_latency_vs_false_cut_in_rate.png` (main figure) and `recall_vs_false_cut_in_rate.png`: one line per model.

- `artifacts/<model>/`, for a trained model: the final model at the chosen setting, trained on all development conversations, for serving. For the text-only model this is `text-only.json` (the head, its threshold and backstop, and the encoder's name), loaded back with `turn_detector.models.text_only.load_text_only`; for the audio-only model, `audio-only.json` (the head, its threshold and backstop, and the encoder's name and layer), loaded back with `turn_detector.models.audio_only.load_audio_only`; for the combined model, `combined.json` (all three heads, the threshold and backstop, and both encoders' names), loaded back with `turn_detector.models.combined.load_combined`.

The audio-only model's encoder features are cached in the git-ignored `data/cache/audio-only/`, at every 50 ms step the evaluation scores and every training sample. The first run extracts them (about 100,000 windows, roughly 15 minutes on the laptop's GPU); later runs train the heads and sweep the threshold in seconds. The combined model reads the same cache, so run the audio-only model first.

The script also prints the comparison table.

A trained model's classifier is cross-fitted: the head that scores a conversation is trained on the other speaker groups only, the same folds the setting's cross-validation uses, so no conversation is scored by a head that saw it. The combined model's fusion head is cross-fitted the same way, nested: the one that scores a speaker group is trained on the other groups' probabilities, each from base heads trained without both groups. The setting for a fold is still chosen on the other folds' cross-fitted scores, whose heads did see that fold.

The setting is chosen by cross-validation that leaves one speaker group out per fold (9 folds, [ADR 0003](docs/adr/0003-train-on-turnbench-dev.md)). Each fold picks the highest-recall setting within the 0.10 false-cut-in budget on the other folds (TurnBench's operating-point rule, ties to lower median latency), and the held-out folds' scores are pooled for the table. The figures show the sweep on all development conversations. The held-out conversations aren't scored.

Scores come from TurnBench's own code: firings are checked with its submission validators and scored with its `score_task` against its gold, exactly as `turnbench.score` does, on the EOT task only. Each conversation is scored once per setting, and folds are sums of those scores. Two departures: the sweep covers a subset of conversations, where `turnbench.score` insists on the whole dev set, and ties between settings with equal recall go to the lower median detection latency, where `turnbench.sweep.operating_point` leaves them unbroken.

## Serving

`turn_detector.serving` is a stateless FastAPI service around the trained text-only, audio-only and combined models ([ADR 0001](docs/adr/0001-asr-outside-the-detector.md): ASR runs upstream). The models are loaded once, at startup, from `artifacts/text-only/`, `artifacts/audio-only/` and `artifacts/combined/`, so run the three evaluations first. The combined model shares the other two's encoders.

- `POST /predict` takes `{audio, transcript, previous_turn, silence_ms}`, all optional, and returns `{p_eot, threshold, backstop_ms, model}`.
  - `audio` is base64 of the user's last 1 s as 16 kHz mono 16-bit little-endian PCM: exactly 32,000 bytes, or the request is rejected with 422.
  - `transcript` is the user's turn so far, as the ASR has finalised it, and `previous_turn` is the agent's last turn.
  - A request with `audio` and a `transcript` (even an empty one) is answered by the combined model, one with `audio` only by the audio-only model, and any other by the text-only model, which reads only the text. A model that hears audio also hears `silence_ms`, so it is required with audio, or the request is rejected with 422. `model` in the response says which answered.
- `GET /health` returns `{status, models}`.

The caller keeps the rolling audio buffer and applies the firing rule itself. It calls `/predict` every 50 ms while the user is silent, up to 3 s into the pause. With the audio-only and combined models it fires as soon as `p_eot` ≥ `threshold`; with the text-only model, only once the ASR's text has arrived (200 ms into the pause). With any, it also fires once the silence reaches `backstop_ms`, and at most once per pause. `turn_detector.streaming` is that caller. `tests/test_streaming.py` checks that streaming through the API fires where each evaluated model does, rounded up to the next 50 ms step.

```bash
uv run uvicorn turn_detector.serving:app      # http://localhost:8000/docs
```

### Docker

The CPU image bakes in the dependencies (torch from PyTorch's CPU index on Linux, so without CUDA), the trained heads and both encoders' weights. It runs with `HF_HUB_OFFLINE=1`, so it needs no network access. The server is uvicorn with `WEB_CONCURRENCY` worker processes (default 4), each using `OMP_NUM_THREADS` threads (default 1).

```bash
uv run python scripts/evaluate.py text-only    # writes artifacts/text-only/
uv run python scripts/evaluate.py audio-only   # writes artifacts/audio-only/
uv run python scripts/evaluate.py combined     # writes artifacts/combined/
docker build -t turn-detector .
docker run --rm -p 8000:8000 turn-detector
```

### Demo and stress test

```bash
uv run python scripts/stream_conversation.py   # streams a held-out conversation, prints the firings
uv run python scripts/stress_test.py --setup "<machine, CPUs, workers>"              # audio and text requests
uv run python scripts/stress_test.py --setup "<machine, CPUs, workers>" --no-text    # audio requests
uv run python scripts/stress_test.py --setup "<machine, CPUs, workers>" --no-audio   # text requests
```

`stream_conversation.py` streams both speakers of a held-out conversation through the API in 50 ms steps, sending real audio and transcripts (text only with `--no-audio`, audio only with `--no-text`). It prints each firing (when, how far into the pause, confident or backstop, `p_eot`) and the request latency the client saw. It shows no gold events or scores, since held-out conversations are scored only once, at the end.

`stress_test.py` sends real requests sampled from that stream with [Locust](https://locust.io), at several concurrency levels: with audio and text, so the combined model answers, with `--no-text`, so the audio-only model does, or with `--no-audio`, so the text-only model does. It writes request latency p50/p95/p99 and throughput per level to the answering model's `results/models/<model>/stress_test.{csv,md}`. The baseline has none: it is a silence timeout the caller applies itself, with no request to make. The findings are in [`docs/solution.md`](docs/solution.md#serving-and-request-latency).

## Tests

```bash
uv run pytest
uv run ty check src tests scripts   # typecheck
```
