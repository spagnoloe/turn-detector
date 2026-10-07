# ASR runs outside the turn detector

The detector's input is `(recent user audio, partial transcript so far)` and its output is P(EOT); it never runs speech recognition itself. This matches the assignment's reference architecture, where the transcriber is a separate upstream component, and it keeps the detector's own per-request latency well under the 100 ms target. ASR (or human transcripts) is only used offline to build training and evaluation data. A text-only or audio-only variant simply ignores the input it doesn't use.
