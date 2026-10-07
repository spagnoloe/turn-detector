# Train and evaluate on TurnBench dev, split by conversation

The TurnBench training set (`otoSpeech-full-duplex-turn-104h`) needs a gated access request, and its licence requires written approval for use by a for-profit company. Rather than wait or risk a licence breach in a hiring assignment, we use only `turn-benchmark-dev` (38 conversations, non-commercial licence). About 26 conversations are used for development, with cross-validation grouped by conversation for model choice and threshold tuning; about 12 are held out and scored once at the end. If training-set access is granted later, it can replace the development portion without changing the held-out evaluation.

## Consequences

- The model sees far less data than the TurnBench baselines, so its scores are not directly comparable on equal terms; the write-up must say so.
- Results on the held-out conversations are not official dev-set scores, since part of dev was used for training.
