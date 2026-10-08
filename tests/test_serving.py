"""The prediction API over HTTP, serving the text-only and audio-only models with stub classifiers."""

import base64
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest
from fastapi.testclient import TestClient

from turn_detector.models.audio_only import AudioOnly
from turn_detector.models.text_only import TextContext, TextOnly
from turn_detector.serving import AUDIO_BYTES, create_app


@dataclass
class Stub:
    """P(EOT) = 0.9 when the user's turn so far ends with a full stop, else 0.1; remembers what it read."""

    read: list[TextContext]

    def p_eot(self, contexts: Sequence[TextContext]) -> list[float]:
        self.read.extend(contexts)
        return [0.9 if context.turn_so_far.endswith(".") else 0.1 for context in contexts]


@dataclass
class AudioStub:
    """P(EOT) = the window's loudest sample, capped at 1; remembers what it heard."""

    heard: list[tuple[np.ndarray, float]]

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
        self.heard.extend(zip(windows, silence_s.tolist()))
        return np.minimum(np.abs(windows).max(axis=1), 1.0)


@pytest.fixture
def stub():
    return Stub([])


@pytest.fixture
def audio_stub():
    return AudioStub([])


@pytest.fixture
def client(stub, audio_stub):
    app = create_app(
        load_text=lambda: TextOnly(stub, threshold=0.6, backstop_s=1.15),
        load_audio=lambda: AudioOnly(audio_stub, threshold=0.7, backstop_s=0.85),
    )
    with TestClient(app) as client:
        yield client


def audio(n_bytes=AUDIO_BYTES) -> str:
    return base64.b64encode(bytes(n_bytes)).decode()


def pcm(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype("<i2").tobytes()).decode()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "models": ["text-only", "audio-only"]}


def test_text_requests_return_p_eot_and_the_text_models_firing_rule_settings(client):
    response = client.post("/predict", json={"transcript": "To Barcelona.", "previous_turn": "Where to?"})
    assert response.status_code == 200
    assert response.json() == {"p_eot": 0.9, "threshold": 0.6, "backstop_ms": 1150, "model": "text-only"}


def test_requests_with_audio_are_answered_by_the_audio_model(client):
    samples = np.zeros(16_000)
    samples[100] = 16384  # half of full scale
    response = client.post(
        "/predict",
        json={"audio": pcm(samples), "transcript": "To Barcelona.", "previous_turn": "Where to?", "silence_ms": 250},
    )
    assert response.status_code == 200
    assert response.json() == {"p_eot": 0.5, "threshold": 0.7, "backstop_ms": 850, "model": "audio-only"}


def test_the_audio_model_hears_the_window_as_samples_in_minus_1_to_1_and_the_silence_in_seconds(client, audio_stub):
    samples = np.linspace(-32768, 32767, 16_000).astype(np.int16)
    client.post("/predict", json={"audio": pcm(samples), "silence_ms": 250})
    [(window, silence_s)] = audio_stub.heard[-1:]
    np.testing.assert_allclose(window, samples / 32768)
    assert silence_s == 0.25


def test_audio_only_requests_return_a_valid_p_eot(client):
    noise = np.random.default_rng(0).integers(-32768, 32767, 16_000)
    response = client.post("/predict", json={"audio": pcm(noise), "silence_ms": 0})
    assert response.status_code == 200
    assert 0 <= response.json()["p_eot"] <= 1


def test_audio_without_the_silence_duration_is_rejected(client):
    assert client.post("/predict", json={"audio": audio(), "transcript": "Hi."}).status_code == 422


def test_the_classifier_reads_the_previous_turn_and_the_transcript(client, stub):
    client.post("/predict", json={"transcript": "To Barcelona,", "previous_turn": "Where to?"})
    assert stub.read[-1] == TextContext("Where to?", "To Barcelona,")


def test_text_only_requests_work(client):
    response = client.post("/predict", json={"transcript": "I'd like to fly to"})
    assert response.status_code == 200
    assert 0 <= response.json()["p_eot"] <= 1


def test_missing_text_reads_as_empty(client, stub):
    assert client.post("/predict", json={}).status_code == 200
    assert stub.read[-1] == TextContext("", "")


@pytest.mark.parametrize("bad_audio", ["not base64!", audio(AUDIO_BYTES - 2), audio(AUDIO_BYTES + 2), audio(0)])
def test_malformed_or_wrong_length_audio_is_rejected(client, bad_audio):
    assert client.post("/predict", json={"audio": bad_audio, "transcript": "Hi.", "silence_ms": 0}).status_code == 422


def test_negative_silence_is_rejected(client):
    assert client.post("/predict", json={"transcript": "Hi.", "silence_ms": -50}).status_code == 422


def test_the_models_are_loaded_once_at_startup(stub, audio_stub):
    loads = []

    def load_text():
        loads.append("text")
        return TextOnly(stub, threshold=0.6, backstop_s=1.15)

    def load_audio():
        loads.append("audio")
        return AudioOnly(audio_stub, threshold=0.7, backstop_s=0.85)

    with TestClient(create_app(load_text=load_text, load_audio=load_audio)) as client:
        assert loads == ["text", "audio"]  # before any request
        client.post("/predict", json={"transcript": "Hi."})
        client.post("/predict", json={"audio": audio(), "silence_ms": 0})
    assert loads == ["text", "audio"]
