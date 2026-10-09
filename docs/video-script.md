# Video script

About 7–8 minutes, about 1,200 spoken words at around 150 words a minute. Screen-share only: `docs/solution.md`, the README, the ADRs and the two comparison figures. `[ON SCREEN]` lines are cues, not narration. Each quoted phrase appears exactly once in its file, so Ctrl+F on it lands on the section to show.

---

## 1. Problem and result (0:00–0:45)

`[ON SCREEN: solution.md — find "Summary"]`

A voice agent has to decide, while the user is still talking, when they've finished. If it answers too early, it talks over them. That's a false cut-in. If it answers too late, it feels slow. A plain silence timeout can't tell "I'd like to fly to… Barcelona" from a finished turn.

I built four detectors, from a silence timeout up to a model that combines audio and text, and evaluated them all the same way. Here's the result up front: on my held-out conversations, none of the trained models beats a well-tuned silence timeout. The rest of this video covers why, the constraints that shaped it, and what I'd do in production.

## 2. Constraints and the decisions they forced (0:45–2:15)

`[ON SCREEN: docs/adr/ — 0002 laptop-only compute, then 0003 train on TurnBench dev]`

Two constraints drove almost every decision. Time: a few hours for data, models, serving and monitoring. And compute: everything ran on one MacBook Air, with no GPU and no cloud.

The first decision was the data. I used TurnBench, a turn-taking benchmark of real two-person conversations, each labelled by three annotators. Its 104-hour training set needs licence approval for commercial use, so I used only the dev set: 38 conversations, about seven hours. I split it by speaker, with 26 conversations for development and 12 held out, so the models can't just learn the actors' voices.

The second decision was the models. With a laptop and about five hours of training audio, fine-tuning wasn't realistic. So the encoders are small, pretrained and frozen: MiniLM for text and wav2vec2 for audio. I only train logistic-regression heads on top.

`[ON SCREEN: solution.md — find "Every result above rests on these"]`

Third, I simplified the inputs and wrote every simplification down. Pauses come from the annotators' segment ends, which is effectively a perfect VAD. Text comes from human transcripts, effectively a perfect ASR, with a fixed 200-millisecond lag. These assumptions make every number here look better than production would.

Because I evaluate on recordings, it's easy to cheat by accident, for example by peeking at whether the user speaks again. So every model only sees what it would have in a live call, and a test checks that. Every trained model also keeps a silence backstop, so no turn goes unanswered.

## 3. Four models (2:15–3:00)

`[ON SCREEN: README.md — find "Shared: the interface every model implements"]`

The **baseline** fires a fixed time after the user stops speaking.

**Text-only** reads the agent's last turn and what the user has said so far. If that sounds finished, it fires 200 milliseconds into the pause.

**Audio-only** listens to the last second of audio every 50 milliseconds, together with how long the user has been silent.

**Combined** is a tiny logistic regression over the audio probability, the text probability and the silence duration. It uses the same firing rule as audio-only, so any difference between them is exactly what text adds.

`[ON SCREEN: solution.md — find "Each model turns them into samples differently"]`

The training data comes from TurnBench's labels. Wherever the user stops speaking, at least two of three annotators agree on whether that was the end of the turn or just a pause. That gives about 2,100 labelled pauses. The text model learns from each pause once. The audio model learns from every 50 milliseconds of the first second of each pause. That's around 50,000 samples, but they're highly correlated, and turn ends supply most of the pause samples, because people stay silent longer after a real turn end.

Each trained model has two knobs: a confidence threshold and the backstop.

## 4. How I evaluate (3:00–4:00)

`[ON SCREEN: results/comparison/detection_latency_vs_false_cut_in_rate.png]`

I used TurnBench's official scorer. A firing counts as a hit if it lands between a quarter of a second before the end of the turn and three seconds after it. Any firing during a mid-turn pause counts as a false cut-in.

That gives three numbers: recall, false-cut-in rate and detection latency. Each model's knobs follow TurnBench's rule: the highest recall at a false-cut-in rate of 10% or less.

The knobs are chosen by cross-validation, holding out one group of speakers per fold. The 12 held-out conversations are scored once, at the very end, with no tuning afterwards.

The main figure is detection latency against false-cut-in rate: how long a caller waits against how often they get interrupted.

## 5. Results, and why the timeout wins (4:00–5:15)

`[ON SCREEN: solution.md — find "The 12 held-out conversations were scored once"]`

On the held-out set, all four models reach a recall between 0.85 and 0.88. Those differences are within noise: with about 500 turn ends, the error bars are around three points.

All four also go over the 10% budget, to between 12 and 14%. The same timeout cuts in on 10% of development pauses and 14% of held-out ones: those speakers simply pause longer mid-sentence. Tuning on 26 conversations doesn't transfer to new speakers, which is why production needs monitoring.

Text-only and combined end up behaving like the baseline: they almost never fire before the backstop. Audio-only is the only model that fires early, answering about one turn in ten in roughly half a second. But its backstop is later, so its median latency is worse.

`[ON SCREEN: solution.md — find "This is the central result"]`

So why does nothing beat the timeout? There are three main reasons. First, the metric rewards recall, not speed, and the timeout already has most of the recall. Second, whether a pause ends a turn depends partly on the listener, and my models only hear the user. Third, five hours of data is enough to learn that "longer silence means done", which the timeout already knows, but not the subtler cues in intonation or wording. Published models like VAP, trained on far more data and hearing both speakers, do clearly better.

If I had to ship one model, it would be audio. It skips the ASR lag, and it's the only one that fires early. Text needs a better encoder first: three handcrafted features rank turn ends better than MiniLM does here.

## 6. Serving (5:15–5:45)

`[ON SCREEN: solution.md — find "Text-only requests meet the <100 ms target"]`

The models are served by a stateless FastAPI service in a CPU Docker image and stress-tested with Locust. Text requests meet the 100-millisecond target easily, at 45 milliseconds p99 with eight requests in flight. Audio requests don't: they take about 130 milliseconds even one at a time. That's the laptop: Docker's Linux VM, a slower CPU build of PyTorch, and no GPU. In production I'd use a GPU or ONNX Runtime, batching, and a streaming WebSocket.

## 7. Monitoring and improving in production (5:45–6:45)

`[ON SCREEN: solution.md — find "How it is observed in production"]`

Production has no gold labels, so I'd track proxies on every call and keep them honest with a small weekly human audit.

For false cut-ins, the proxy is the user barging in within a second of the agent starting to speak. For latency, it's the firing time minus the end of speech, and how often we fall back to the backstop. For missed turns, it's long silences, or the user saying "hello?". Plus request latency measured by the caller, and drift in the model's inputs and outputs.

Everything is sliced by customer, language and channel. An average can stay flat while one customer on a noisy line gets twice the interruptions.

`[ON SCREEN: solution.md — find "Improving the model over time"]`

To improve the model: at a million calls a month, the bottleneck isn't audio, it's labels. Barge-ins and long waits give weak labels for free. Humans annotate the uncertain pauses near the threshold. Before a new model ships, I'd first replay recorded calls through it, which is offline replay. Then I'd run it silently next to the live model and log when it would have fired, which is shadow mode. Only then would I A/B test it on a small share of calls.

## 8. Close (6:45–7:00)

With one laptop and a few hours, a well-tuned timeout is a tough baseline to beat, and I'd rather show that honestly than oversell a model. The full write-up, including the discussion questions, is in `docs/solution.md`. Thanks for watching.
