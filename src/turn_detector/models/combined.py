"""The combined model: what the user said and how they said it, fused late.

Every 50 ms while the user is silent, the audio-only model's classifier hears the user's last 1 s
of audio and the silence so far (P_audio), and the text-only model's classifier reads the text
the ASR has delivered (P_text). A logistic regression over [P_audio, P_text, silence duration]
outputs P(EOT). The text read is the context at the user's latest segment end whose text has
arrived, 200 ms (the ASR lag) after the segment ended: until then the previous segment end's
context is held, and before the user's first, the context is empty. So P_text changes only at
segment ends, 200 ms late.

The firing rule is the audio-only model's (`turn_detector.models.classified`): fire on the rising
edge where P(EOT) first reaches the threshold, or at a silence backstop if that comes first, the
two knobs tuned together. The audio-only model already has the backstop, so what this model adds
over it is the text alone.

The fusion head has three weights, so it is fitted on probabilities that are not overfitted: P_audio
and P_text from base classifiers that never saw the conversation's speaker group. Its samples are
the audio-only head's pause samples: every 50 ms step up to 1 s into each labelled pause, labelled
by whether it is a gold EOT or a mid-turn pause. For the evaluation this is nested: the fusion head
that scores a speaker group is trained on the other groups, each scored by base classifiers trained
without both groups, so no conversation's labels reach the classifier that scores it. The base
classifiers are trained exactly as the text-only and audio-only models' are.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from turn_detector.data import ARTIFACTS_DIR
from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import EPSILON_S, Setting, SpeakerSide
from turn_detector.models import audio_only, text_only
from turn_detector.models.audio_only import TRAINED_PAUSE_S, AudioClassifier, AudioTraining, Wav2Vec2Encoder, heard_pauses
from turn_detector.models.classified import (
    Fitted,
    LogisticHead,
    PredictionInputs,
    ScoredPause,
    Serving,
    SideKey,
    at_setting,
    cross_fitted,
    firings,
    gold_pause_labels,
    head_from_json,
    head_json,
    pauses,
    save_json,
    scored_steps,
    side_key,
    step_times,
    train_head,
)
from turn_detector.models.classified import fit as fit_classified
from turn_detector.models.text_only import SentenceEncoder, TextClassifier, TextContext, TextTraining, request_context, text_read_at

NAME = "combined"
ARTIFACT_PATH = ARTIFACTS_DIR / NAME / "combined.json"
# Answers requests that carry both audio and a transcript, from the first step of a pause.
SERVING = Serving(NAME, needs_audio=True, needs_transcript=True)


def p_text(classifier: text_only.Classifier, contexts: Sequence[TextContext]) -> np.ndarray:
    """P_text for each context, reading each distinct context once (a pause holds at most two)."""
    distinct = list(dict.fromkeys(contexts))
    by_context = dict(zip(distinct, classifier.p_eot(distinct)))
    return np.array([by_context[context] for context in contexts], dtype=float)


def fusion_features(p_audio: np.ndarray, p_text: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
    """What the fusion head reads, one row per step: [P_audio, P_text, silence duration]."""
    return np.column_stack([p_audio, p_text, silence_s])


class Classifier(Protocol):
    """1 s windows of 16 kHz audio, the text read and the silence so far in, P(EOT) out, one per window."""

    def p_eot(self, windows: np.ndarray, contexts: Sequence[TextContext], silence_s: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CombinedClassifier:
    """The two base classifiers and the fusion head over their probabilities and the silence."""

    text: text_only.Classifier
    audio: audio_only.Classifier
    fusion: LogisticHead

    def p_eot(self, windows: np.ndarray, contexts: Sequence[TextContext], silence_s: np.ndarray) -> np.ndarray:
        return self.fusion(fusion_features(self.audio.p_eot(windows, silence_s), p_text(self.text, contexts), silence_s))


def scored_pauses(side: SpeakerSide, classifier: Classifier) -> list[ScoredPause]:
    """Every pause of the user's, each 50 ms step scored by `classifier`."""
    return heard_pauses(side, lambda windows, times, silence_s: classifier.p_eot(windows, text_read_at(side, times), silence_s))


@dataclass(frozen=True)
class Combined:
    """Fire at the first 50 ms step of a pause where the fused P(EOT) >= `threshold`, or
    `backstop_s` into it if that comes first, up to 3 s into it and only while the user is still
    silent."""

    classifier: Classifier
    threshold: float
    backstop_s: float
    name: str = NAME

    def fire(self, side: SpeakerSide) -> list[float]:
        return firings(scored_pauses(side, self.classifier), self.threshold, self.backstop_s)

    def predict(self, request: PredictionInputs) -> float:
        assert request.window is not None and request.silence_s is not None
        [p_eot] = self.classifier.p_eot(request.window, [request_context(request)], np.array([request.silence_s]))
        return float(p_eot)


def save_combined(model: Combined, path: Path = ARTIFACT_PATH) -> Path:
    """Save the three heads, the threshold and backstop, and the encoders' names (and the audio
    encoder's layer); the encoders are downloaded."""
    classifier = model.classifier
    if not (
        isinstance(classifier, CombinedClassifier)
        and isinstance(classifier.text, TextClassifier)
        and isinstance(classifier.audio, AudioClassifier)
    ):
        raise TypeError("only a model with trained text, audio and fusion heads can be saved")
    stored = {
        "threshold": model.threshold,
        "backstop_s": model.backstop_s,
        "text": text_only.classifier_json(classifier.text),
        "audio": audio_only.classifier_json(classifier.audio),
        "fusion": head_json(classifier.fusion),
    }
    return save_json(stored, path)


def load_combined(
    path: Path = ARTIFACT_PATH,
    text_encoder: text_only.Encoder | None = None,
    audio_encoder: audio_only.Encoder | None = None,
) -> Combined:
    """The saved model, with its encoders loaded (or the ones given, which must be the ones it was
    trained on, as when serving shares them with the single-input models)."""
    stored = json.loads(path.read_text())
    classifier = CombinedClassifier(
        text_only.classifier_from_json(stored["text"], text_encoder),
        audio_only.classifier_from_json(stored["audio"], audio_encoder),
        head_from_json(stored["fusion"]),
    )
    return Combined(classifier, stored["threshold"], stored["backstop_s"])


def final(setting: Setting) -> Combined:
    """The saved final model, which must be at `setting`, the one cross-validation chose, with its
    audio encoder on the GPU if there is one."""
    import torch

    encoder = Wav2Vec2Encoder(device="mps" if torch.backends.mps.is_available() else "cpu")
    return at_setting(load_combined(audio_encoder=encoder), setting)


@dataclass
class CombinedTraining:
    """Trains the combined classifier on some development conversations: the base classifiers on
    them, and the fusion head on their out-of-fold probabilities."""

    conversations: Sequence[EvaluationConversation]
    text: TextTraining
    audio: AudioTraining
    contexts: dict[SideKey, list[TextContext]] = field(default_factory=dict, repr=False)

    def classifier(self, conversation_ids: Sequence[str]) -> CombinedClassifier:
        wanted = set(conversation_ids)
        samples = cross_fitted(
            [c for c in self.conversations if c.info.conversation_id in wanted],
            lambda ids: (self.text.classifier(ids), self.audio.classifier(ids)),
            lambda base, conversation: self.fusion_samples(conversation, *base),
        )
        features = np.concatenate([samples[i][0] for i in conversation_ids])
        labels = np.concatenate([samples[i][1] for i in conversation_ids])
        fusion = train_head(features, labels, standardise=True)
        return CombinedClassifier(self.text.classifier(conversation_ids), self.audio.classifier(conversation_ids), fusion)

    def step_features(self, side: SpeakerSide, text: TextClassifier, audio: AudioClassifier) -> np.ndarray:
        """The fusion head's features at every 50 ms step of the side's pauses, in step order."""
        times, silence_s = step_times(pauses(side))
        if side_key(side) not in self.contexts:
            self.contexts[side_key(side)] = text_read_at(side, times)
        return fusion_features(self.audio.p_audio(side, audio), p_text(text, self.contexts[side_key(side)]), silence_s)

    def fusion_samples(
        self, conversation: EvaluationConversation, text: TextClassifier, audio: AudioClassifier
    ) -> tuple[np.ndarray, np.ndarray]:
        """Features and labels at the steps up to 1 s into each gold EOT (True) or mid-turn pause (False)."""
        rows, labels = [], []
        for side in conversation.sides:
            is_eot = gold_pause_labels(conversation, side)
            features = self.step_features(side, text, audio)
            ends = [pause.end for pause in pauses(side) for _ in pause.steps]
            for row, end, silence_s in zip(features, ends, features[:, 2]):
                if end in is_eot and silence_s <= TRAINED_PAUSE_S + EPSILON_S:
                    rows.append(row)
                    labels.append(is_eot[end])
        return np.array(rows).reshape(-1, 3), np.array(labels, dtype=bool)

    def scored_pauses(self, side: SpeakerSide, classifier: CombinedClassifier) -> list[ScoredPause]:
        assert isinstance(classifier.text, TextClassifier) and isinstance(classifier.audio, AudioClassifier)
        return scored_steps(pauses(side), classifier.fusion(self.step_features(side, classifier.text, classifier.audio)).tolist())


def fit(conversations: Sequence[EvaluationConversation]) -> Fitted:
    """Train the combined classifier per speaker group (nested, see the module docstring) and on
    every conversation, reading the audio features the audio-only model cached."""
    import torch

    audio_encoder = Wav2Vec2Encoder(device="mps" if torch.backends.mps.is_available() else "cpu")
    training = CombinedTraining(conversations, TextTraining(conversations, SentenceEncoder()), AudioTraining(conversations, audio_encoder))

    def save(classifier: CombinedClassifier, threshold: float, backstop_s: float) -> Path:
        return save_combined(Combined(classifier, threshold, backstop_s))

    return fit_classified(NAME, training, conversations, save)
