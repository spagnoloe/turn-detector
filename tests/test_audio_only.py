"""The audio-only model through the model interface, with a stub classifier; its features, its
audio window and its saved head."""

from collections.abc import Sequence

import numpy as np
import pytest

from turn_detector.model import Audio, Segment, SpeakerSide
from turn_detector.models.classified import LogisticHead
from turn_detector.models.audio_only import (
    AudioClassifier,
    AudioOnly,
    features,
    load_audio_only,
    save_audio_only,
    windows,
)

SAMPLE_RATE = 48_000
NO_BACKSTOP_S = 4.0  # beyond the 3 s horizon, so it never fires


class RisesWithSilence:
    """P(EOT) = the silence so far in seconds, capped at 1; ignores the audio."""

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
        return np.minimum(silence_s, 1.0)


def side(*segments: tuple[float, float], duration_s=20.0, audio: np.ndarray | None = None) -> SpeakerSide:
    samples = np.zeros(round(duration_s * SAMPLE_RATE), np.float32) if audio is None else audio
    return SpeakerSide(
        conversation_id="synthetic",
        speaker=1,
        duration_s=duration_s,
        audio=Audio(samples, SAMPLE_RATE),
        segments=[Segment(start, end, "") for start, end in segments],
        other_turns=[],
    )


def test_fires_at_the_first_50_ms_step_where_p_reaches_the_threshold():
    # P = silence: 0.30 at 3.30 s is below 0.33, 0.35 at 3.35 s is above it.
    assert AudioOnly(RisesWithSilence(), threshold=0.33, backstop_s=NO_BACKSTOP_S).fire(side((1.0, 3.0))) == pytest.approx([3.35])


class Oscillates:
    """P(EOT) alternates between 0.9 and 0.1 every 50 ms step of a pause."""

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
        return np.where(np.round(silence_s / 0.05) % 2 == 0, 0.9, 0.1)


def test_fires_at_most_once_per_pause_on_the_rising_edge():
    # P rises above 0.5 again at every other step of both pauses; only the first edge fires.
    assert AudioOnly(Oscillates(), threshold=0.5, backstop_s=NO_BACKSTOP_S).fire(side((1.0, 3.0), (5.0, 6.0))) == pytest.approx([3.0, 6.0])


def test_does_not_fire_below_the_threshold():
    assert AudioOnly(Oscillates(), threshold=0.95, backstop_s=NO_BACKSTOP_S).fire(side((1.0, 3.0), (5.0, 6.0))) == []


def test_does_not_fire_once_the_user_resumes_or_the_conversation_ends():
    # P reaches 0.42 at 0.45 s of silence: too late for a 0.3 s pause and a call ending 0.4 s in.
    model = AudioOnly(RisesWithSilence(), threshold=0.42, backstop_s=NO_BACKSTOP_S)
    assert model.fire(side((1.0, 3.0), (3.3, 5.0), duration_s=5.4)) == []


class SureAfter:
    """P(EOT) = 1 once the silence so far is longer than `seconds`, else 0."""

    def __init__(self, seconds: float):
        self.seconds = seconds

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
        return (silence_s > self.seconds).astype(float)


def test_stops_listening_3_s_into_a_pause():
    # A firing more than 3 s after an EOT can never be a hit, so the model stops asking there.
    assert AudioOnly(SureAfter(2.92), threshold=0.5, backstop_s=NO_BACKSTOP_S).fire(side((1.0, 3.0))) == pytest.approx([5.95])
    assert AudioOnly(SureAfter(3.01), threshold=0.5, backstop_s=NO_BACKSTOP_S).fire(side((1.0, 3.0))) == []


def test_features_are_the_window_mean_the_last_speech_frames_mean_and_the_silence():
    # wav2vec2 frames are 20 ms apart: frame i of a 1 s window is centred at 12.5 + 20 i ms, so 49
    # frames. Each frame here holds its own index, in both dimensions.
    frames = np.repeat(np.arange(49, dtype=np.float32)[None, :, None], 2, axis=2).repeat(3, axis=0)
    silence_s = np.array([0.0, 0.2, 1.0])
    assert features(frames, silence_s).tolist() == [
        [24, 24, 43.5, 43.5, 0.0],  # speaking: the last 10 frames, 39-48
        [24, 24, 34.5, 34.5, 0.2],  # 200 ms silent: frames 0-39 are centred before the pause; 30-39
        [24, 24, 0, 0, 1.0],  # all silence: no speech frames
    ]


def test_the_window_is_the_last_second_at_16_khz_left_padded_with_silence():
    # 2 s of silence, then a constant 0.5, recorded at 48 kHz.
    audio = Audio(np.concatenate([np.zeros(2 * SAMPLE_RATE), np.full(SAMPLE_RATE, 0.5)]).astype(np.float32), SAMPLE_RATE)
    late, early = windows(audio, [2.5, 0.25])
    assert late.shape == (16_000,)
    np.testing.assert_allclose(late[:7_900], 0, atol=1e-3)  # 1.5-2 s: silence
    np.testing.assert_allclose(late[8_100:-100], 0.5, atol=1e-3)  # 2-2.5 s, away from the window's edge
    np.testing.assert_array_equal(early[:12_000], 0)  # before the call started


class ChunkEncoder:
    """Pseudo-frames: each of 49 frames is the mean and spread of its stretch of the window."""

    name = "chunks"

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        chunks = windows[:, : 49 * 320].reshape(len(windows), 49, 320)
        return np.stack([chunks.mean(axis=2), chunks.std(axis=2)], axis=2)


def test_saved_model_loads_back_and_predicts_the_same(tmp_path):
    head = LogisticHead(weights=np.random.default_rng(0).standard_normal(5), bias=0.3)
    model = AudioOnly(AudioClassifier(ChunkEncoder(), head), threshold=0.42, backstop_s=0.9)
    sound = np.random.default_rng(1).standard_normal((3, 16_000)).astype(np.float32)
    silence_s = np.array([0.0, 0.3, 1.2])
    path = tmp_path / "audio-only.json"
    save_audio_only(model, path)
    loaded = load_audio_only(path, encoder=ChunkEncoder())
    assert (loaded.threshold, loaded.backstop_s) == (0.42, 0.9)
    np.testing.assert_array_equal(loaded.classifier.p_eot(sound, silence_s), model.classifier.p_eot(sound, silence_s))


def test_loading_with_a_different_encoder_fails(tmp_path):
    path = tmp_path / "audio-only.json"
    save_audio_only(AudioOnly(AudioClassifier(ChunkEncoder(), LogisticHead(np.ones(5), 0.0)), threshold=0.5, backstop_s=1.5), path)
    other = ChunkEncoder()
    other.name = "another encoder"
    with pytest.raises(ValueError, match="encoder"):
        load_audio_only(path, encoder=other)


def test_fires_at_the_backstop_when_p_never_reaches_the_threshold():
    model = AudioOnly(Oscillates(), threshold=0.95, backstop_s=1.15)
    assert model.fire(side((1.0, 3.0), (5.0, 6.0))) == pytest.approx([4.15, 7.15])


def test_the_backstop_fires_before_a_later_confident_step():
    # P reaches 0.9 at 0.9 s of silence, but the 0.5 s backstop comes first.
    assert AudioOnly(RisesWithSilence(), threshold=0.88, backstop_s=0.5).fire(side((1.0, 3.0))) == pytest.approx([3.5])


def test_the_backstop_does_not_fire_once_the_user_resumes_or_the_conversation_ends():
    model = AudioOnly(Oscillates(), threshold=0.95, backstop_s=1.15)
    assert model.fire(side((1.0, 3.0), (4.0, 5.0), duration_s=6.0)) == []
