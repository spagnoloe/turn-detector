"""The audio-only model: does the way the user stopped speaking sound like the end of their turn?

Every 50 ms while the user is silent, the classifier hears the user's last 1 s of audio, resampled
to 16 kHz, and the silence duration so far, and outputs P_audio(EOT). The firing rule is the one
every trained model shares (`turn_detector.models.classified`): in each pause, fire on the rising
edge where P_audio first reaches the threshold, or at a silence backstop if that comes first, at
most once per pause and only while the user is still silent. The
threshold and the backstop are the two knobs, tuned together, as for the text-only model; with
the threshold above every P_audio, the model is the baseline with the backstop as its timeout.
The model stops listening 3 s into a pause, since TurnBench never counts a later firing as a hit.

The classifier is a frozen `wav2vec2-base` encoder (ADR 0002) with a logistic-regression head over
three features: the mean of the window's frames, the mean of its last speech frames (how the
speech ended, kept even as the window fills with silence) and the silence duration so far (the
only signal left once the window is all silence). The pauses and the silence duration come from
the same segment timeline every model sees, a perfect pause detector standing in for a real VAD.

Training samples are taken every 50 ms through each labelled pause, up to 1 s into it, labelled by
whether the pause is a gold EOT or a mid-turn pause; plus one sample per second of speech, labelled
not EOT. Pauses that are neither (disputed) are not trained on. The encoder is the slow part, so
its features at every sample time and every 50 ms step the evaluation scores are cached to disk
once; training heads and sweeping thresholds then take seconds.
"""

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly

from turn_detector.data import ARTIFACTS_DIR, DATA_DIR, load_audio
from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import EPSILON_S, HORIZON_S, STEP_S, Audio, SpeakerSide
from turn_detector.models.classified import (
    Fitted,
    LogisticHead,
    Request,
    ScoredPause,
    Serving,
    SideKey,
    check_encoder,
    firings,
    gold_pause_labels,
    head_from_json,
    head_json,
    pauses,
    save_json,
    scored_steps,
    side_key,
    train_head,
)
from turn_detector.models.classified import fit as fit_classified

NAME = "audio-only"
ENCODER = "facebook/wav2vec2-base"
ENCODER_REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
# The encoder layer the features are read from, and the head's L2 regularisation (sklearn's C).
# Both were chosen once, on the training samples of a third of the development conversations, by
# out-of-fold AUC with one speaker group held out per fold: layer 8 beat layers 3, 6, 10 and 12,
# and weaker regularisation let the head overfit its 1537 features (C = 1 scored below the silence
# duration alone).
LAYER = 8
REGULARISATION_C = 1e-4
SAMPLE_RATE = 16_000
WINDOW_S = 1.0
WINDOW_SAMPLES = round(WINDOW_S * SAMPLE_RATE)
TRAINED_PAUSE_S = 1.0  # training samples go this far into a pause
SPEECH_SAMPLE_EVERY_S = 1.0
# wav2vec2's frames: one every 320 samples (20 ms), each heard through 400 samples (25 ms).
FRAME_STRIDE, FRAME_SIZE = 320, 400
LAST_SPEECH_FRAMES = 10  # 200 ms: how the speech ended
WINDOWS_PER_BATCH = 256  # windows cut and encoded at once, to bound memory
ARTIFACT_PATH = ARTIFACTS_DIR / NAME / "audio-only.json"
FEATURE_CACHE_DIR = DATA_DIR / "cache" / NAME
# Answers requests that carry audio (and so the silence duration), from the first step of a pause.
SERVING = Serving(NAME, needs_audio=True, needs_transcript=False)


def windows(audio: Audio, times: Sequence[float]) -> np.ndarray:
    """The last 1 s of `audio` before each time, resampled to 16 kHz, one row per time, left-padded
    with silence near the start of the call. Each window is resampled on its own, so no sample
    after its time can leak into it."""
    length = round(WINDOW_S * audio.sample_rate)
    native = np.zeros((len(times), length), np.float32)
    for row, t in zip(native, times):
        end = round(t * audio.sample_rate)
        heard = audio.samples[max(0, end - length) : end]
        row[length - len(heard) :] = heard
    if audio.sample_rate == SAMPLE_RATE:
        return native
    common = math.gcd(SAMPLE_RATE, audio.sample_rate)
    return resample_poly(native, SAMPLE_RATE // common, audio.sample_rate // common, axis=1).astype(np.float32)


def features(frames: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
    """One row per window: the mean of all its frames, the mean of its last speech frames (those
    centred before the pause began; zeros once the window is all silence) and the silence so far.

    `frames` is the encoder's output, (windows, frames, dimensions).
    """
    centres_s = (FRAME_STRIDE * np.arange(frames.shape[1]) + FRAME_SIZE / 2) / SAMPLE_RATE
    speech_ends_s = WINDOW_S - silence_s
    rows = []
    for window_frames, speech_end_s in zip(frames, speech_ends_s):
        speech = window_frames[centres_s < speech_end_s][-LAST_SPEECH_FRAMES:]
        last_speech = speech.mean(axis=0) if len(speech) else np.zeros(frames.shape[2], frames.dtype)
        rows.append(np.concatenate([window_frames.mean(axis=0), last_speech]))
    return np.column_stack([np.array(rows, np.float32), silence_s])


class Encoder(Protocol):
    """Turns 1 s windows of 16 kHz audio into frames, (windows, frames, dimensions)."""

    @property
    def name(self) -> str: ...

    def __call__(self, windows: np.ndarray) -> np.ndarray: ...


@dataclass
class Wav2Vec2Encoder:
    """A frozen wav2vec2 encoder: the hidden states of one of its layers, one frame per 20 ms.

    Each window is normalised to zero mean and unit variance first, as wav2vec2-base expects. The
    layers above `layer` are dropped, since nothing reads them.
    """

    name: str = ENCODER
    layer: int = LAYER
    device: str = "cpu"
    batch_size: int = 64

    @cached_property
    def model(self):
        from transformers import Wav2Vec2Model

        model = Wav2Vec2Model.from_pretrained(self.name, revision=ENCODER_REVISION).eval()
        model.encoder.layers = model.encoder.layers[: self.layer]
        return model.to(self.device)

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        import torch

        normalised = (windows - windows.mean(axis=1, keepdims=True)) / np.sqrt(windows.var(axis=1, keepdims=True) + 1e-7)
        frames = []
        with torch.inference_mode():
            for start in range(0, len(normalised), self.batch_size):
                batch = torch.from_numpy(normalised[start : start + self.batch_size].astype(np.float32)).to(self.device)
                frames.append(self.model(batch).last_hidden_state.float().cpu().numpy())
        return np.concatenate(frames)


def train_audio_head(features: np.ndarray, labels: np.ndarray) -> LogisticHead:
    """Fit the head: strongly L2-regularised logistic regression on standardised features, labels
    True for EOT."""
    return train_head(features, labels, REGULARISATION_C, standardise=True)


class Classifier(Protocol):
    """1 s windows of 16 kHz audio and the silence so far in, P_audio(EOT) out, one per window."""

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class AudioClassifier:
    """The encoder and head together."""

    encoder: Encoder
    head: LogisticHead

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
        return self.head(features(self.encoder(windows), silence_s))


def heard_pauses(
    side: SpeakerSide, p_eot: Callable[[np.ndarray, Sequence[float], np.ndarray], np.ndarray]
) -> list[ScoredPause]:
    """Every pause of the user's, each 50 ms step scored by `p_eot(windows, times, silence_s)`, a
    batch of steps at a time."""
    if side.audio is None:
        raise ValueError("a model that listens needs the user's audio")
    found = pauses(side)
    times = [t for pause in found for t in pause.steps]
    silence_s = np.array([t - pause.end for pause in found for t in pause.steps])
    scores = []
    for start in range(0, len(times), WINDOWS_PER_BATCH):
        batch = times[start : start + WINDOWS_PER_BATCH]
        scores.extend(p_eot(windows(side.audio, batch), batch, silence_s[start : start + WINDOWS_PER_BATCH]))
    return scored_steps(found, scores)


def scored_pauses(side: SpeakerSide, classifier: Classifier) -> list[ScoredPause]:
    """Every pause of the user's, each step scored by `classifier`."""
    return heard_pauses(side, lambda windows, times, silence_s: classifier.p_eot(windows, silence_s))


@dataclass(frozen=True)
class AudioOnly:
    """Fire at the first 50 ms step of a pause where P_audio >= `threshold`, or `backstop_s` into
    it if that comes first, up to 3 s into it and only while the user is still silent."""

    classifier: Classifier
    threshold: float
    backstop_s: float
    name: str = NAME

    def fire(self, side: SpeakerSide) -> list[float]:
        return firings(scored_pauses(side, self.classifier), self.threshold, self.backstop_s)

    def predict(self, request: Request) -> float:
        assert request.window is not None and request.silence_s is not None
        [p_eot] = self.classifier.p_eot(request.window, np.array([request.silence_s]))
        return float(p_eot)


def classifier_json(classifier: AudioClassifier) -> dict:
    """The head and the encoder's name and layer; the encoder itself is downloaded."""
    return {
        "encoder": classifier.encoder.name,
        "layer": getattr(classifier.encoder, "layer", LAYER),
        **head_json(classifier.head),
    }


def classifier_from_json(stored: dict, encoder: Encoder | None = None) -> AudioClassifier:
    """The saved classifier, with its encoder loaded on the CPU (or `encoder`, which must be the
    one it was trained on)."""
    encoder = check_encoder(stored, encoder or Wav2Vec2Encoder(stored["encoder"], stored["layer"]))
    return AudioClassifier(encoder, head_from_json(stored))


def save_audio_only(model: AudioOnly, path: Path = ARTIFACT_PATH) -> Path:
    """Save the head, its threshold and backstop, and the encoder's name and layer."""
    if not isinstance(model.classifier, AudioClassifier):
        raise TypeError("only a model with a trained AudioClassifier can be saved")
    return save_json({**classifier_json(model.classifier), "threshold": model.threshold, "backstop_s": model.backstop_s}, path)


def load_audio_only(path: Path = ARTIFACT_PATH, encoder: Encoder | None = None) -> AudioOnly:
    """The saved model, with its encoder loaded on the CPU (or `encoder`, which must be the one it
    was trained on)."""
    stored = json.loads(path.read_text())
    return AudioOnly(classifier_from_json(stored, encoder), stored["threshold"], stored["backstop_s"])


# Cached features. A side's features are kept for every 50 ms step the evaluation scores and every
# speech sample; the cache is keyed by everything that changes them.


@dataclass(frozen=True)
class SideFeatures:
    """Features at sample times of one side: `pause_ends` is the pause a time is a step of, or NaN
    for a speech sample."""

    times: np.ndarray
    pause_ends: np.ndarray
    features: np.ndarray


def speech_samples(side: SpeakerSide) -> list[float]:
    """One time per second into each of the user's segments, while they are speaking."""
    return [
        segment.start + k * SPEECH_SAMPLE_EVERY_S
        for segment in side.segments
        for k in range(1, math.ceil((segment.end - segment.start) / SPEECH_SAMPLE_EVERY_S))
    ]


def feature_cache_path(side: SpeakerSide, encoder: Encoder, cache_dir: Path = FEATURE_CACHE_DIR) -> Path:
    config = [
        encoder.name, ENCODER_REVISION, getattr(encoder, "layer", None),
        STEP_S, HORIZON_S, SPEECH_SAMPLE_EVERY_S, LAST_SPEECH_FRAMES,
    ]  # fmt: skip
    key = hashlib.sha256(json.dumps(config).encode()).hexdigest()[:12]
    return cache_dir / key / f"{side.conversation_id}-{side.speaker}.npz"


def side_features(side: SpeakerSide, encoder: Encoder, cache_dir: Path = FEATURE_CACHE_DIR) -> SideFeatures:
    """The features at every pause step and speech sample of `side`, from the cache if there; the
    user's audio is loaded from the dev set when they are not."""
    found = pauses(side)
    speech = speech_samples(side)
    times = np.array([t for pause in found for t in pause.steps] + speech)
    pause_ends = np.array([pause.end for pause in found for _ in pause.steps] + [math.nan] * len(speech))
    path = feature_cache_path(side, encoder, cache_dir)
    if path.exists():
        cached = np.load(path)
        if np.array_equal(cached["times"], times):
            return SideFeatures(times, pause_ends, cached["features"].astype(np.float32))
    audio = side.audio if side.audio is not None else load_audio(side.conversation_id, side.speaker)
    silence_s = np.where(np.isnan(pause_ends), 0.0, times - pause_ends)
    rows = []
    for start in range(0, len(times), WINDOWS_PER_BATCH):
        batch = windows(audio, list(times[start : start + WINDOWS_PER_BATCH]))
        rows.append(features(encoder(batch), silence_s[start : start + WINDOWS_PER_BATCH]))
    result = np.concatenate(rows) if rows else np.zeros((0, 0), np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, times=times, features=result.astype(np.float16))
    return SideFeatures(times, pause_ends, result.astype(np.float16).astype(np.float32))


def labelled_samples(conversation: EvaluationConversation, sides: dict[int, SideFeatures]) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels: steps up to 1 s into a gold EOT (True) or mid-turn pause (False), and
    speech samples (False)."""
    rows, labels = [], []
    for side in conversation.sides:
        is_eot = gold_pause_labels(conversation, side)
        found = sides[side.speaker]
        for row, t, end in zip(found.features, found.times, found.pause_ends):
            if math.isnan(end):
                rows.append(row)
                labels.append(False)
            elif end in is_eot and t - end <= TRAINED_PAUSE_S + EPSILON_S:
                rows.append(row)
                labels.append(is_eot[end])
    return np.array(rows), np.array(labels)


@dataclass
class AudioTraining:
    """Trains the audio classifier on some development conversations, from the features cached for
    every side; each set of conversations is trained on once."""

    conversations: Sequence[EvaluationConversation]
    encoder: Encoder
    trained: dict[frozenset[str], AudioClassifier] = field(default_factory=dict, repr=False)

    @cached_property
    def features(self) -> dict[SideKey, SideFeatures]:
        found = {}
        for conversation in self.conversations:
            for side in conversation.sides:
                print(f"  features: conversation {side.conversation_id}, speaker {side.speaker}", flush=True)
                found[side_key(side)] = side_features(side, self.encoder)
        return found

    @cached_property
    def samples(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {
            c.info.conversation_id: labelled_samples(c, {side.speaker: self.features[side_key(side)] for side in c.sides})
            for c in self.conversations
        }

    def classifier(self, conversation_ids: Sequence[str]) -> AudioClassifier:
        key = frozenset(conversation_ids)
        if key not in self.trained:
            head = train_audio_head(
                np.concatenate([self.samples[i][0] for i in conversation_ids]),
                np.concatenate([self.samples[i][1] for i in conversation_ids]),
            )
            self.trained[key] = AudioClassifier(self.encoder, head)
        return self.trained[key]

    def p_audio(self, side: SpeakerSide, classifier: AudioClassifier) -> np.ndarray:
        """P_audio at every 50 ms step of the side's pauses, in step order, from the cached features."""
        found = self.features[side_key(side)]
        return classifier.head(found.features[~np.isnan(found.pause_ends)])

    def scored_pauses(self, side: SpeakerSide, classifier: AudioClassifier) -> list[ScoredPause]:
        return scored_steps(pauses(side), self.p_audio(side, classifier).tolist())


def fit(conversations: Sequence[EvaluationConversation]) -> Fitted:
    """Extract (or read the cached) features, train one head per speaker group on every other
    group, and a final head on every conversation."""
    import torch

    encoder = Wav2Vec2Encoder(device="mps" if torch.backends.mps.is_available() else "cpu")

    def save(classifier: AudioClassifier, threshold: float, backstop_s: float) -> Path:
        return save_audio_only(AudioOnly(classifier, threshold, backstop_s))

    return fit_classified(NAME, AudioTraining(conversations, encoder), conversations, save)
