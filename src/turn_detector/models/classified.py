"""What the trained models share: P(EOT) at the times a model may fire in each pause, the
threshold-plus-backstop firing rule, cross-fitting per speaker group, saving a head with its
encoder's name, and how the API serves a model.

A trained model scores each pause of the user's at the times it may fire there: every 50 ms step
for a model that hears audio, once 200 ms in (when the ASR's text arrives) for the text-only
model. It fires at the first of those times where P(EOT) reaches its threshold (the rising edge:
while the user speaks there is no firing, so the first time counts as an edge), or at a silence
backstop if that comes first, at most once per pause and only while the user is still silent, up
to 3 s into the pause. The threshold and the backstop are its two knobs, tuned together; with the
threshold above every P(EOT), the model is the baseline with the backstop as its timeout.

The evaluation scores the pauses once, by classifiers cross-fitted per speaker group, and a sweep
over settings only re-applies the firing rule (`CrossFitted`, `Fitted`).
"""

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import EPSILON_S, HORIZON_S, Setting, SpeakerSide, next_speech_start, steps
from turn_detector.split import speaker_groups

type SideKey = tuple[str, int]  # (conversation id, speaker)


def side_key(side: SpeakerSide) -> SideKey:
    return (side.conversation_id, side.speaker)


@dataclass(frozen=True)
class Pause:
    """A segment end of the user's, when they stop being silent after it (they resume, the call
    ends or the model stops listening, whichever comes first), and the 50 ms steps until then."""

    end: float
    silent_until: float
    steps: list[float]


def step_times(pauses: Sequence[Pause]) -> tuple[list[float], np.ndarray]:
    """Every 50 ms step of the pauses, in step order, and the silence so far at each."""
    times = [t for pause in pauses for t in pause.steps]
    silence_s = np.array([t - pause.end for pause in pauses for t in pause.steps])
    return times, silence_s


def pauses(side: SpeakerSide) -> list[Pause]:
    """Every distinct segment end of the user's, in time order, as a pause."""
    result = []
    for end in sorted({segment.end for segment in side.segments}):
        silent_until = min(next_speech_start(side, end), side.duration_s, end + HORIZON_S)
        result.append(Pause(end, silent_until, list(steps(end, silent_until))))
    return result


@dataclass(frozen=True)
class ScoredPause:
    """A pause with P(EOT) at each time the model may fire in it, as (time, P(EOT)) in time order."""

    end: float
    silent_until: float
    scores: list[tuple[float, float]]


def scored_steps(pauses: Sequence[Pause], p_eot: Sequence[float]) -> list[ScoredPause]:
    """The pauses with P(EOT) at each 50 ms step, `p_eot` holding them in step order."""
    result, i = [], 0
    for pause in pauses:
        scores = list(zip(pause.steps, map(float, p_eot[i : i + len(pause.steps)])))
        result.append(ScoredPause(pause.end, pause.silent_until, scores))
        i += len(pause.steps)
    return result


def firings(pauses: Sequence[ScoredPause], threshold: float, backstop_s: float) -> list[float]:
    """The firing rule: in each pause, fire at the first time P(EOT) reaches the threshold, or at
    the backstop if that comes first, only while the user is still silent."""
    firings = []
    for pause in pauses:
        confident = next((t for t, p in pause.scores if p >= threshold), math.inf)
        firing = min(confident, pause.end + backstop_s)
        if firing <= pause.silent_until + EPSILON_S:
            firings.append(firing)
    return firings


@dataclass(frozen=True)
class LogisticHead:
    """P(EOT) = sigmoid(features · weights + bias)."""

    weights: np.ndarray
    bias: float

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(features @ self.weights + self.bias)))


def train_head(features: np.ndarray, labels: np.ndarray, regularisation_c: float = 1.0, standardise: bool = False) -> LogisticHead:
    """Fit a head: L2-regularised logistic regression (sklearn's C), labels True for EOT. With
    `standardise`, it is fitted on standardised features and the standardisation is folded into
    the weights, so the head reads raw features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not standardise:
        regression = LogisticRegression(C=regularisation_c, max_iter=2000).fit(features, labels)
        return LogisticHead(regression.coef_[0].copy(), float(regression.intercept_[0]))
    scaler = StandardScaler().fit(features)
    regression = LogisticRegression(C=regularisation_c, max_iter=2000).fit(scaler.transform(features), labels)
    weights = regression.coef_[0] / scaler.scale_
    return LogisticHead(weights, float(regression.intercept_[0] - weights @ scaler.mean_))


def head_json(head: LogisticHead) -> dict:
    return {"bias": head.bias, "weights": head.weights.tolist()}


def head_from_json(stored: dict) -> LogisticHead:
    return LogisticHead(np.array(stored["weights"]), stored["bias"])


class Named(Protocol):
    @property
    def name(self) -> str: ...


def check_encoder[E: Named](stored: dict, encoder: E) -> E:
    """`encoder`, if it is the one the stored head was trained on."""
    if encoder.name != stored["encoder"]:
        raise ValueError(f"the head was trained on encoder {stored['encoder']!r}, not {encoder.name!r}")
    return encoder


class Thresholded(Protocol):
    """A model that fires on a threshold or a backstop."""

    @property
    def name(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    @property
    def backstop_s(self) -> float: ...


def at_setting[M: Thresholded](model: M, setting: Setting) -> M:
    """`model`, a saved one, if it is at `setting` (threshold, backstop in ms); otherwise its
    artifact is stale, from an evaluation older than the results that chose `setting`."""
    threshold, backstop_ms = setting
    if not (math.isclose(model.threshold, threshold) and math.isclose(model.backstop_s * 1000, backstop_ms)):
        raise ValueError(
            f"the saved {model.name} model is at threshold {model.threshold:g}, backstop "
            f"{model.backstop_s * 1000:g} ms, not at the setting chosen in cross-validation, {setting}; "
            f"re-run scripts/evaluate.py {model.name}"
        )
    return model


def save_json(stored: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stored, indent=2) + "\n")
    return path


# Cross-fitting: the evaluation's view of a trained model.


def gold_pause_labels(conversation: EvaluationConversation, side: SpeakerSide) -> dict[float, bool]:
    """The side's segment ends that have a gold label: True at an EOT, False at the start of a
    mid-turn pause. Ends that are neither (disputed) are not trained on."""
    gold = conversation.gold
    eots = {event.time_s for event in gold.eot_positive_events if event.speaker == side.speaker}
    mid_turn = {span.start for span in gold.eot_negative_spans if span.speaker == side.speaker}
    ends = {segment.end for segment in side.segments}
    return {end: end in eots for end in ends & (eots | mid_turn)}


class Training[C](Protocol):
    """How to train a model's classifier on some development conversations, and score a side's
    pauses with it."""

    def classifier(self, conversation_ids: Sequence[str]) -> C: ...

    def scored_pauses(self, side: SpeakerSide, classifier: C) -> list[ScoredPause]: ...


def cross_fitted[C, S](
    conversations: Sequence[EvaluationConversation],
    train: Callable[[list[str]], C],
    score: Callable[[C, EvaluationConversation], S],
) -> dict[str, S]:
    """Each conversation's `score` by the classifier `train`ed on every speaker group but its own:
    the folds `cross_validate` uses, so a fold's held-out conversations are always scored by a
    classifier that never saw them."""
    all_ids = [c.info.conversation_id for c in conversations]
    result = {}
    for group in speaker_groups([c.info for c in conversations]):
        held_out = {c.conversation_id for c in group}
        classifier = train([i for i in all_ids if i not in held_out])
        for conversation in (c for c in conversations if c.info.conversation_id in held_out):
            result[conversation.info.conversation_id] = score(classifier, conversation)
    return result


@dataclass(frozen=True)
class CrossFitted:
    """A trained model as the evaluation scores it, at one setting. Each side's pauses were scored
    once, by the classifier that never saw that conversation's speakers, so a sweep over settings
    only re-applies the firing rule."""

    name: str
    pauses: dict[SideKey, list[ScoredPause]]
    threshold: float
    backstop_s: float

    def fire(self, side: SpeakerSide) -> list[float]:
        return firings(self.pauses[side_key(side)], self.threshold, self.backstop_s)


@dataclass(frozen=True)
class Fitted:
    """Every development side's pauses scored by cross-fitted classifiers, and how to save the
    final model, whose classifier is trained on all conversations. A setting is (threshold,
    backstop in ms); `check_setting` rejects one the model can't run at."""

    name: str
    pauses: dict[SideKey, list[ScoredPause]]
    save_final: Callable[[float, float], Path]  # (threshold, backstop in s) -> the saved model
    check_setting: Callable[[float, float], None] = lambda threshold, backstop_s: None

    def build(self, setting: Setting) -> CrossFitted:
        threshold, backstop_ms = setting
        self.check_setting(threshold, backstop_ms / 1000)
        return CrossFitted(self.name, self.pauses, threshold, backstop_ms / 1000)

    def save(self, setting: Setting) -> list[Path]:
        threshold, backstop_ms = setting
        return [self.save_final(threshold, backstop_ms / 1000)]


def fit[C](
    name: str,
    training: Training[C],
    conversations: Sequence[EvaluationConversation],
    save: Callable[[C, float, float], Path],
    check_setting: Callable[[float, float], None] = lambda threshold, backstop_s: None,
) -> Fitted:
    """Score every side with a classifier trained on the other speaker groups, and train the final
    classifier on every conversation; `save(classifier, threshold, backstop_s)` saves a model."""
    scored = cross_fitted(
        conversations,
        training.classifier,
        lambda classifier, conversation: [(side, training.scored_pauses(side, classifier)) for side in conversation.sides],
    )
    pauses = {side_key(side): side_pauses for sides in scored.values() for side, side_pauses in sides}
    final = training.classifier([c.info.conversation_id for c in conversations])
    return Fitted(name, pauses, lambda threshold, backstop_s: save(final, threshold, backstop_s), check_setting)


# Serving: what a request carries, and which model answers it.


@dataclass(frozen=True)
class PredictionInputs:
    """One prediction request, decoded: the last 1 s of the user's audio at 16 kHz as one row of
    samples in [-1, 1), the agent's previous turn, the user's turn so far as the ASR has finalised
    it, and the silence so far. Each is None when the caller didn't send it."""

    window: np.ndarray | None
    previous_turn: str | None
    transcript: str | None
    silence_s: float | None


@dataclass(frozen=True)
class Serving:
    """How a model is served, the same for the API and its caller: what a request must carry for
    this model to answer it, and how far into a pause its P(EOT) may first fire (the text-only
    model's waits for the ASR's text)."""

    name: str
    needs_audio: bool
    needs_transcript: bool
    earliest_firing_s: float = 0.0

    def answers(self, request: PredictionInputs) -> bool:
        return (request.window is not None or not self.needs_audio) and (
            request.transcript is not None or not self.needs_transcript
        )
