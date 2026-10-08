"""The prediction API over HTTP, serving the text-only model with a stub classifier."""

import base64
from collections.abc import Sequence
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from turn_detector.models.text_only import TextContext, TextOnly
from turn_detector.serving import AUDIO_BYTES, create_app


@dataclass
class Stub:
    """P(EOT) = 0.9 when the user's turn so far ends with a full stop, else 0.1; remembers what it read."""

    read: list[TextContext]

    def p_eot(self, contexts: Sequence[TextContext]) -> list[float]:
        self.read.extend(contexts)
        return [0.9 if context.turn_so_far.endswith(".") else 0.1 for context in contexts]


@pytest.fixture
def stub():
    return Stub([])


@pytest.fixture
def client(stub):
    with TestClient(create_app(lambda: TextOnly(stub, threshold=0.6, backstop_s=1.15))) as client:
        yield client


def audio(n_bytes=AUDIO_BYTES) -> str:
    return base64.b64encode(bytes(n_bytes)).decode()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "text-only"}


def test_predict_returns_p_eot_and_the_firing_rules_settings(client):
    response = client.post(
        "/predict",
        json={"audio": audio(), "transcript": "To Barcelona.", "previous_turn": "Where to?", "silence_ms": 250},
    )
    assert response.status_code == 200
    assert response.json() == {"p_eot": 0.9, "threshold": 0.6, "backstop_ms": 1150, "model": "text-only"}


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
    assert client.post("/predict", json={"audio": bad_audio, "transcript": "Hi."}).status_code == 422


def test_negative_silence_is_rejected(client):
    assert client.post("/predict", json={"transcript": "Hi.", "silence_ms": -50}).status_code == 422


def test_the_model_is_loaded_once_at_startup(stub):
    loads = []

    def load():
        loads.append(1)
        return TextOnly(stub, threshold=0.6, backstop_s=1.15)

    with TestClient(create_app(load)) as client:
        assert loads == [1]  # before any request
        client.post("/predict", json={"transcript": "Hi."})
        client.post("/predict", json={"transcript": "Hi."})
    assert loads == [1]
