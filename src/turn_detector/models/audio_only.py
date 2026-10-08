"""The audio-only model: does the way the user stopped speaking sound like the end of their turn?

Every 50 ms while the user is silent, the classifier hears the user's last 1 s of audio, resampled
to 16 kHz, and the silence duration so far, and outputs P_audio(EOT). The firing rule: in each
pause, fire on the rising edge where P_audio first reaches the threshold, at most once per pause
and only while the user is still silent. The threshold is the one knob. The model stops listening
3 s into a pause, since TurnBench never counts a later firing as a hit.

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
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
from scipy.signal import resample_poly

from turn_detector.data import ARTIFACTS_DIR, DATA_DIR, load_audio
from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import EPSILON_S, HORIZON_S, STEP_S, Audio, Setting, SpeakerSide, next_speech_start, steps
from turn_detector.models.text_only import LogisticHead
from turn_detector.split import speaker_groups

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


def train_head(features: np.ndarray, labels: np.ndarray) -> LogisticHead:
    """Fit the head: strongly L2-regularised logistic regression on standardised features, labels
    True for EOT. The standardisation is folded into the weights, so the head reads raw features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(features)
    regression = LogisticRegression(C=REGULARISATION_C, max_iter=2000).fit(scaler.transform(features), labels)
    weights = regression.coef_[0] / scaler.scale_
    return LogisticHead(weights, float(regression.intercept_[0] - weights @ scaler.mean_))


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


@dataclass(frozen=True)
class ScoredPause:
    """A segment end with the 50 ms steps of the pause after it, each with P_audio there."""

    end: float
    steps: list[tuple[float, float]]


def pause_steps(side: SpeakerSide) -> list[tuple[float, list[float]]]:
    """Every distinct segment end of the user's, with the 50 ms steps at which the user is still
    silent after it, up to the horizon."""
    return [
        (end, list(steps(end, min(next_speech_start(side, end), side.duration_s, end + HORIZON_S))))
        for end in sorted({segment.end for segment in side.segments})
    ]


def scored(pauses: Sequence[tuple[float, list[float]]], p_eot: Sequence[float]) -> list[ScoredPause]:
    """`pause_steps` with P_audio at each step, `p_eot` holding them in step order."""
    result, i = [], 0
    for end, times in pauses:
        result.append(ScoredPause(end, list(zip(times, map(float, p_eot[i : i + len(times)])))))
        i += len(times)
    return result


def scored_pauses(side: SpeakerSide, classifier: Classifier) -> list[ScoredPause]:
    """Every pause of the user's, each step scored by `classifier`."""
    if side.audio is None:
        raise ValueError("the audio-only model needs the user's audio")
    pauses = pause_steps(side)
    times = [t for _, ts in pauses for t in ts]
    silence_s = np.array([t - end for end, ts in pauses for t in ts])
    p_eot = []
    for start in range(0, len(times), WINDOWS_PER_BATCH):
        batch = windows(side.audio, times[start : start + WINDOWS_PER_BATCH])
        p_eot.extend(classifier.p_eot(batch, silence_s[start : start + WINDOWS_PER_BATCH]))
    return scored(pauses, p_eot)


def firings(pauses: Sequence[ScoredPause], threshold: float) -> list[float]:
    """The firing rule: in each pause, fire on the rising edge where P_audio first reaches the
    threshold. While the user speaks there is no firing, so the first step counts as an edge."""
    firings = []
    for pause in pauses:
        firing = next((t for t, p in pause.steps if p >= threshold), None)
        if firing is not None:
            firings.append(firing)
    return firings


@dataclass(frozen=True)
class AudioOnly:
    """Fire at the first 50 ms step of a pause where P_audio >= `threshold`, up to 3 s into it."""

    classifier: Classifier
    threshold: float
    name: str = NAME

    def fire(self, side: SpeakerSide) -> list[float]:
        return firings(scored_pauses(side, self.classifier), self.threshold)


def save_audio_only(model: AudioOnly, path: Path = ARTIFACT_PATH) -> Path:
    """Save the head, its threshold and the encoder's name and layer; the encoder is downloaded."""
    if not isinstance(model.classifier, AudioClassifier):
        raise TypeError("only a model with a trained AudioClassifier can be saved")
    classifier = model.classifier
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {
        "encoder": classifier.encoder.name,
        "layer": getattr(classifier.encoder, "layer", LAYER),
        "threshold": model.threshold,
        "bias": classifier.head.bias,
        "weights": classifier.head.weights.tolist(),
    }
    path.write_text(json.dumps(stored, indent=2) + "\n")
    return path


def load_audio_only(path: Path = ARTIFACT_PATH, encoder: Encoder | None = None) -> AudioOnly:
    """The saved model, with its encoder loaded on the CPU (or `encoder`, which must be the one it
    was trained on)."""
    stored = json.loads(path.read_text())
    encoder = encoder or Wav2Vec2Encoder(stored["encoder"], stored["layer"])
    if encoder.name != stored["encoder"]:
        raise ValueError(f"the head was trained on encoder {stored['encoder']!r}, not {encoder.name!r}")
    head = LogisticHead(np.array(stored["weights"]), stored["bias"])
    return AudioOnly(AudioClassifier(encoder, head), stored["threshold"])


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
    pauses = pause_steps(side)
    speech = speech_samples(side)
    times = np.array([t for _, ts in pauses for t in ts] + speech)
    pause_ends = np.array([end for end, ts in pauses for _ in ts] + [math.nan] * len(speech))
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
    gold = conversation.gold
    rows, labels = [], []
    for side in conversation.sides:
        eots = {event.time_s for event in gold.eot_positive_events if event.speaker == side.speaker}
        mid_turn = {span.start for span in gold.eot_negative_spans if span.speaker == side.speaker}
        found = sides[side.speaker]
        for row, t, end in zip(found.features, found.times, found.pause_ends):
            if math.isnan(end):
                rows.append(row)
                labels.append(False)
            elif (end in eots or end in mid_turn) and t - end <= TRAINED_PAUSE_S + EPSILON_S:
                rows.append(row)
                labels.append(end in eots)
    return np.array(rows), np.array(labels)


@dataclass(frozen=True)
class CrossFitted:
    """The audio-only model as the evaluation scores it, at one threshold. Each side's steps were
    scored once, by the head that never saw that conversation's speakers."""

    pauses: dict[tuple[str, int], list[ScoredPause]]
    threshold: float
    name: str = NAME

    def fire(self, side: SpeakerSide) -> list[float]:
        return firings(self.pauses[(side.conversation_id, side.speaker)], self.threshold)


@dataclass(frozen=True)
class FittedAudioOnly:
    """Every development side's steps scored by cross-fitted heads, and a head trained on all
    conversations. A setting is (threshold,)."""

    pauses: dict[tuple[str, int], list[ScoredPause]]
    final_classifier: AudioClassifier

    def build(self, setting: Setting) -> CrossFitted:
        (threshold,) = setting
        return CrossFitted(self.pauses, threshold)

    def save(self, setting: Setting) -> list[Path]:
        (threshold,) = setting
        return [save_audio_only(AudioOnly(self.final_classifier, threshold))]


def fit(conversations: Sequence[EvaluationConversation]) -> FittedAudioOnly:
    """Extract (or read the cached) features, train one head per speaker group on every other
    group (the folds `cross_validate` uses), and a final head on every conversation."""
    import torch

    encoder = Wav2Vec2Encoder(device="mps" if torch.backends.mps.is_available() else "cpu")
    found = {}
    for conversation in conversations:
        for side in conversation.sides:
            print(f"  features: conversation {side.conversation_id}, speaker {side.speaker}", flush=True)
            found[(side.conversation_id, side.speaker)] = side_features(side, encoder)
    samples = {
        c.info.conversation_id: labelled_samples(c, {side.speaker: found[(c.info.conversation_id, side.speaker)] for side in c.sides})
        for c in conversations
    }

    def head_trained_on(conversation_ids: Sequence[str]) -> LogisticHead:
        return train_head(
            np.concatenate([samples[i][0] for i in conversation_ids]),
            np.concatenate([samples[i][1] for i in conversation_ids]),
        )

    all_ids = list(samples)
    pauses = {}
    for group in speaker_groups([c.info for c in conversations]):
        held_out = {c.conversation_id for c in group}
        head = head_trained_on([i for i in all_ids if i not in held_out])
        for conversation in (c for c in conversations if c.info.conversation_id in held_out):
            for side in conversation.sides:
                side_found = found[(side.conversation_id, side.speaker)]
                in_pause = ~np.isnan(side_found.pause_ends)
                pauses[(side.conversation_id, side.speaker)] = scored(
                    pause_steps(side), head(side_found.features[in_pause]).tolist()
                )
    return FittedAudioOnly(pauses, AudioClassifier(encoder, head_trained_on(all_ids)))
