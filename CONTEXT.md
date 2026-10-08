# Turn Detector

Decides, while a user is speaking to a voice agent, whether the user has finished their turn and the agent should respond.

## Language

**End of Turn (EOT)**:
The moment the user has finished their turn and expects the agent to respond.
_Avoid_: endpoint, turn complete, turn boundary

**Mid-turn pause**:
A stretch of silence within the user's speech after which the same user continues.
_Avoid_: hold, floor hold

**Pause**:
Any stretch of silence in the user's speech; it ends either in an EOT or as a mid-turn pause.

**Detection latency**:
The time from the end of the user's speech to the detector firing.
_Avoid_: latency (unqualified)

**Request latency**:
The time the detector takes to answer one prediction request; the assignment's "<100 ms per request" target.
_Avoid_: latency (unqualified), inference time

**False cut-in**:
The detector firing during a mid-turn pause.
_Avoid_: false interruption

**False-cut-in rate**:
The share of mid-turn pauses in which the detector fires at least once.
_Avoid_: FPR, false positive rate

**Firing**:
The detector's single decision that the user's turn has ended and the agent should respond now; what gets scored, rather than the per-window probabilities.
_Avoid_: trigger, detection

**Model**:
One complete way of deciding when to fire: a plain rule (the baseline, a silence timeout) or a trained classifier plus a firing rule. The four models (baseline, text-only, audio-only, combined) are compared on the same evaluation.
_Avoid_: system, detector (for one of the four)

**Classifier**:
The trained part of a model, which outputs P(EOT); the model's firing rule turns it into firings.
_Avoid_: model (for the trained part alone)
