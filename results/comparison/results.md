# Results at a false-cut-in rate of 0.10 or less

Development conversations, cross-validated: each speaker group is scored at the setting chosen without it (highest recall within the false-cut-in budget), and the folds are pooled, so the scores mix the per-fold settings. Scored with TurnBench's EOT scorer. The setting chosen on all development conversations is the one a final model would use.

| Model | Knobs | Setting chosen on all development (per-fold range) | Recall | False-cut-in rate | Detection latency p10 / p50 / p90 (ms) |
|---|---|---:|---:|---:|---:|
| baseline | silence timeout N (ms) | 1150 (1100–1250) | 0.853 | 0.098 | 1100 / 1150 / 1200 |
| text-only | P_text threshold, backstop (ms) | 0.9, 1150 (0.86–0.9, 1100–1250) | 0.857 | 0.113 | 1100 / 1150 / 1200 |
| audio-only | P_audio threshold, backstop (ms) | 0.946, 1500 (0.936–0.954, 1300–1600) | 0.866 | 0.113 | 506 / 1300 / 1500 |
| combined | P_combined threshold, backstop (ms) | 0.926, 1150 (0.894–0.926, 1100–1500) | 0.856 | 0.122 | 569 / 1150 / 1500 |
