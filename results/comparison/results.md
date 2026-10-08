# Results at a false-cut-in rate of 0.10 or less

Development conversations, cross-validated: each speaker group is scored at the knob setting chosen without it (highest recall within the false-cut-in budget), and the folds are pooled, so the scores mix the per-fold settings. Scored with TurnBench's EOT scorer. The setting chosen on all development conversations is the one a final model would use.

| Model | Knob | Setting chosen on all development (per-fold range) | Recall | False-cut-in rate | Detection latency p10 / p50 / p90 (ms) |
|---|---|---:|---:|---:|---:|
| baseline | silence timeout N (ms) | 1150 (1100–1250) | 0.853 | 0.098 | 1100 / 1150 / 1200 |
