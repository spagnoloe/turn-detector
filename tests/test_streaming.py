"""Streaming a speaker's side through the prediction API, as the voice agent's orchestrator would."""

import base64
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from turn_detector.model import Audio, Segment, SpeakerSide
from turn_detector.models.audio_only import AudioOnly
from turn_detector.models.combined import Combined
from turn_detector.models.text_only import TextContext, TextOnly
from turn_detector.serving import AUDIO_BYTES, create_app
from turn_detector.streaming import Firing, audio_window, pcm_16k, stream


@dataclass(frozen=True)
class Stub:
    """P(EOT) = 1 when the user's turn so far ends with a full stop, else 0."""

    def p_eot(self, contexts: Sequence[TextContext]) -> list[float]:
        return [1.0 if context.turn_so_far.endswith(".") else 0.0 for context in contexts]


@dataclass(frozen=True)
class AudioStub:
    """P(EOT) = 1 once the silence so far is in (`after_s`, 3.2 s), else 0; ignores the audio."""

    after_s: float

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
        return ((silence_s > self.after_s) & (silence_s < 3.2)).astype(float)


@dataclass(frozen=True)
class CombinedStub:
    """P(EOT) = 1 once the user's turn so far ends with a full stop and the silence so far is over
    `after_s`, else 0; ignores the audio."""

    after_s: float

    def p_eot(self, windows: np.ndarray, contexts: Sequence[TextContext], silence_s: np.ndarray) -> np.ndarray:
        full_stop = np.array([context.turn_so_far.endswith(".") for context in contexts])
        return (full_stop & (silence_s > self.after_s)).astype(float)


MODEL = TextOnly(Stub(), threshold=0.5, backstop_s=1.5)
COMBINED = Combined(CombinedStub(0.12), threshold=0.5, backstop_s=1.5)


def side(*segments: tuple[float, float, str], other: Sequence[tuple[float, float, str]] = (), duration_s=20.0):
    return SpeakerSide(
        conversation_id="synthetic",
        speaker=1,
        duration_s=duration_s,
        audio=None,
        segments=[Segment(*segment) for segment in segments],
        other_turns=[Segment(*segment) for segment in other],
    )


def served(audio_model: AudioOnly, combined_model: Combined = COMBINED):
    with TestClient(create_app(load=lambda: [MODEL, audio_model, combined_model])) as client:
        yield lambda payload: client.post("/predict", json=payload).raise_for_status().json()


@pytest.fixture(scope="module")
def predict():
    yield from served(AudioOnly(AudioStub(0.12), threshold=0.5, backstop_s=1.5))


SIDES = [
    side((1.0, 3.0, "I'd like to fly to Barcelona.")),
    side((1.0, 3.0, "I'd like to fly to")),
    side((1.0, 3.0, "Yes."), (3.1, 5.0, "So"), (6.0, 7.0, "Friday.")),
    side((1.0, 3.0, "Well"), duration_s=4.0),
    side((1.0, 3.0, "To Barcelona."), (2.0, 3.0, "Barcelona.")),
    side((0.0, 1.0, "Hi."), (3.0, 4.0, "To Barcelona,"), (4.5, 5.0, "on Friday."), other=[(1.2, 2.0, "Where to?")]),
]


@pytest.mark.parametrize("user", SIDES)
def test_streaming_through_the_api_fires_where_the_model_does(user, predict):
    assert [firing.time_s for firing in stream(user, predict)] == pytest.approx(MODEL.fire(user))


def with_audio(user: SpeakerSide) -> SpeakerSide:
    return replace(user, audio=Audio(np.zeros(round(user.duration_s * 48_000), np.float32), 48_000))


@pytest.mark.parametrize("after_s", [0.12, 2.92, 3.01])
@pytest.mark.parametrize("user", [*SIDES, side((1.0, 3.0, "Hi."), (8.0, 9.0, "Bye."))])
def test_streaming_audio_through_the_api_fires_where_the_audio_model_does(user, after_s):
    # No wait for the ASR, the backstop when P never gets there, and nothing after 3 s into a pause.
    audio_model = AudioOnly(AudioStub(after_s), threshold=0.5, backstop_s=1.5)
    user = with_audio(user)
    assert user.audio is not None
    for predict in served(audio_model):
        assert [f.time_s for f in stream(user, predict, pcm_16k(user.audio), text=False)] == pytest.approx(audio_model.fire(user))


@pytest.mark.parametrize("after_s", [0.0, 0.12, 2.92])
@pytest.mark.parametrize(
    "user",
    [
        *SIDES,
        side((1.0, 3.0, "Hi."), (3.5, 6.0, "I'd like to")),  # the first pause's text is held into the second
        side((1.0, 2.0, "Paris."), (2.5, 3.0, "Rome."), other=[(2.1, 2.4, "Mm.")]),  # "Mm." is not read until 3.2 s
    ],
)
def test_streaming_audio_and_text_through_the_api_fires_where_the_combined_model_does(user, after_s):
    # Fires from the first step of a pause, on the text held until the new segment's text arrives.
    combined_model = Combined(CombinedStub(after_s), threshold=0.5, backstop_s=1.5)
    user = with_audio(user)
    assert user.audio is not None
    for predict in served(AudioOnly(AudioStub(0.12), threshold=0.5, backstop_s=1.5), combined_model):
        firings = stream(user, predict, pcm_16k(user.audio))
        assert [f.time_s for f in firings] == pytest.approx(combined_model.fire(user))


def test_firings_say_why_they_fired(predict):
    user = side((1.0, 3.0, "Barcelona."), (5.0, 6.0, "I'd like to"))
    assert stream(user, predict) == [Firing(3.2, 3.0, 1.0, "confident"), Firing(7.5, 6.0, 0.0, "backstop")]


def test_firings_land_on_the_next_50_ms_step(predict):
    # Confident at 3.23 + 0.2; the first step at or after that is 3.45.
    assert [firing.time_s for firing in stream(side((1.0, 3.23, "Barcelona.")), predict)] == pytest.approx([3.45])


def test_the_last_segments_text_arrives_after_the_asr_lag():
    payloads = []

    def predict(payload):
        payloads.append(payload)
        return {"p_eot": 0.0, "threshold": 0.5, "backstop_ms": 300, "model": "text-only"}

    stream(side((0.0, 1.0, "Hi."), (2.0, 3.0, "Barcelona.")), predict)
    in_second_pause = payloads[-7:]  # steps at 3.0, 3.05, ..., 3.3
    assert [p["silence_ms"] for p in in_second_pause] == pytest.approx([0, 50, 100, 150, 200, 250, 300])
    assert [p["transcript"] for p in in_second_pause] == ["Hi."] * 4 + ["Hi. Barcelona."] * 3


def test_requests_carry_the_last_second_of_audio_when_given(predict):
    samples = np.arange(16_000 * 4, dtype=np.int16)
    payloads = []

    def recording(payload):
        payloads.append(payload)
        return predict(payload)

    stream(side((1.0, 3.0, "Barcelona.")), recording, samples)
    assert all(len(base64.b64decode(p["audio"])) == AUDIO_BYTES for p in payloads)


def test_the_audio_window_is_the_last_second_left_padded_with_silence():
    samples = np.arange(1, 16_000 * 2 + 1, dtype=np.int16)
    np.testing.assert_array_equal(np.frombuffer(audio_window(samples, 2.0), "<i2"), samples[16_000:])
    early = np.frombuffer(audio_window(samples, 0.25), "<i2")
    np.testing.assert_array_equal(early, np.concatenate([np.zeros(12_000), samples[:4_000]]))


def test_audio_is_resampled_to_16_khz_pcm():
    t = np.arange(48_000 * 2) / 48_000
    pcm = pcm_16k(Audio((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 48_000))
    assert pcm.dtype == np.int16 and len(pcm) == 32_000
    assert 0.45 * 32767 < np.abs(pcm[1000:-1000]).max() <= 0.55 * 32767
