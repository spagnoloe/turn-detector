# Results at a false-cut-in rate of 0.10 or less

## Held-out conversations: the final scores

The 12 held-out conversations, scored once, in a single run, by each final model (trained on all 26 development conversations) at the setting chosen in cross-validation, with no tuning afterwards. Scored with TurnBench's EOT scorer. These are the only clean scores: the audio model's encoder layer (8) and regularisation (C = 1e-4) were picked on a third of the development conversations, so the cross-validated development scores below are slightly optimistic for the audio-only and combined models.

The held-out set is small: 512 EOTs and 352 mid-turn pauses, so one EOT moves recall by 0.002 and one mid-turn pause the false-cut-in rate by 0.003. A 95% binomial interval is about ±0.03 on recall and ±0.04 on the false-cut-in rate, and wider in truth, since the pauses of one conversation are correlated.

| Model | Knobs | Setting | Recall | False-cut-in rate | Detection latency p10 / p50 / p90 (ms) |
|---|---|---:|---:|---:|---:|
| baseline | silence timeout N (ms) | 1150 | 0.877 | 0.142 | 1150 / 1150 / 1150 |
| text-only | P_text threshold, backstop (ms) | 0.9, 1150 | 0.881 | 0.142 | 1150 / 1150 / 1150 |
| audio-only | P_audio threshold, backstop (ms) | 0.946, 1500 | 0.854 | 0.125 | 551 / 1500 / 1500 |
| combined | P_combined threshold, backstop (ms) | 0.926, 1150 | 0.877 | 0.142 | 1150 / 1150 / 1150 |
| VAP (published) | — | — | 0.845 | 0.055 | -57 / 368 / 1537 |
| Pipecat Smart Turn v3 (published) | — | — | 0.752 | 0.047 | 729 / 1017 / 1175 |

The published rows are outside references, not comparable on equal terms:

- they are scored on TurnBench's test set, a different and larger split than our held-out conversations, at an operating point chosen on the whole dev set;
- VAP was trained on Switchboard and Fisher, then fine-tuned on TurnBench's 104 h training set;
- Pipecat Smart Turn v3 was trained on Pipecat's own turn-completion data (not TurnBench);
- they detect pauses from the audio themselves, whereas every model here is given the annotators' consensus segment ends (a perfect VAD) and, for text, human transcripts (a perfect ASR, read 200 ms after each segment ends).

The full list of assumptions is in [`docs/solution.md`](../../docs/solution.md#assumptions).

## Development conversations, cross-validated

Development conversations, cross-validated: each speaker group is scored at the setting chosen without it (highest recall within the false-cut-in budget), and the folds are pooled, so the scores mix the per-fold settings. Scored with TurnBench's EOT scorer. The setting chosen on all development conversations is the one a final model would use.

| Model | Knobs | Setting chosen on all development (per-fold range) | Recall | False-cut-in rate | Detection latency p10 / p50 / p90 (ms) |
|---|---|---:|---:|---:|---:|
| baseline | silence timeout N (ms) | 1150 (1100–1250) | 0.853 | 0.098 | 1100 / 1150 / 1200 |
| text-only | P_text threshold, backstop (ms) | 0.9, 1150 (0.86–0.9, 1100–1250) | 0.857 | 0.113 | 1100 / 1150 / 1200 |
| audio-only | P_audio threshold, backstop (ms) | 0.946, 1500 (0.936–0.954, 1300–1600) | 0.866 | 0.113 | 506 / 1300 / 1500 |
| combined | P_combined threshold, backstop (ms) | 0.926, 1150 (0.894–0.926, 1100–1500) | 0.856 | 0.122 | 569 / 1150 / 1500 |
