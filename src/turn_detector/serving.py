"""The prediction API: P(EOT) for one moment of a call, from what the caller sends.

The service is stateless. The caller (the voice agent's orchestrator) keeps the rolling 1 s audio
buffer and the transcript, calls `POST /predict` every 50 ms while the user is silent, and applies
the firing rule itself with the threshold and backstop each response carries
(`turn_detector.streaming` is that caller). ASR runs upstream (ADR 0001): `transcript` is the
user's turn so far as the ASR has finalised it, and `previous_turn` is the agent's last turn.

Which model answers is the first in SERVED whose inputs the request carries (each model's
`Serving` description): the combined model a request with audio and a transcript, the audio-only
model one with audio only, and the text-only model any other, reading only the text. A model that
hears audio also hears the silence duration, so a request with audio must carry `silence_ms`. The
models are loaded once, at startup.

    uv run uvicorn turn_detector.serving:app
"""

import base64
import binascii
from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Protocol, Self

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from turn_detector.models import audio_only, combined, text_only
from turn_detector.models.audio_only import AudioClassifier, load_audio_only
from turn_detector.models.classified import Request as Inputs
from turn_detector.models.classified import Serving
from turn_detector.models.combined import load_combined
from turn_detector.models.text_only import TextClassifier, load_text_only

AUDIO_SAMPLE_RATE = 16_000
# The last 1 s of the user's channel: 16 kHz mono 16-bit little-endian PCM.
AUDIO_SAMPLES = AUDIO_SAMPLE_RATE
AUDIO_BYTES = AUDIO_SAMPLES * 2

# Most specific first: a request is answered by the first model whose inputs it carries.
SERVED = [combined.SERVING, audio_only.SERVING, text_only.SERVING]
SERVED_BY_NAME = {serving.name: serving for serving in SERVED}


class PredictRequest(BaseModel):
    audio: bytes | None = Field(None, description="Base64 of the last 1 s, 16 kHz mono 16-bit little-endian PCM")
    transcript: str | None = Field(None, description="The user's turn so far, as the ASR has finalised it")
    previous_turn: str | None = Field(None, description="The agent's previous turn")
    silence_ms: float | None = Field(None, ge=0, description="How long the user has been silent")

    @field_validator("audio", mode="before")
    @classmethod
    def decode_audio(cls, value: object) -> bytes | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("audio must be a base64 string")
        try:
            pcm = base64.b64decode(value, validate=True)
        except binascii.Error as error:
            raise ValueError(f"audio is not valid base64: {error}") from None
        if len(pcm) != AUDIO_BYTES:
            raise ValueError(f"audio must be {AUDIO_BYTES} bytes (1 s of 16 kHz 16-bit PCM), not {len(pcm)}")
        return pcm

    @model_validator(mode="after")
    def audio_needs_the_silence(self) -> Self:
        if self.audio is not None and self.silence_ms is None:
            raise ValueError("a request with audio needs silence_ms, which a model hearing audio hears too")
        return self

    def inputs(self) -> Inputs:
        window = None if self.audio is None else samples(self.audio)
        silence_s = None if self.silence_ms is None else self.silence_ms / 1000
        return Inputs(window, self.previous_turn, self.transcript, silence_s)


class PredictResponse(BaseModel):
    p_eot: float
    threshold: float
    backstop_ms: int | None = Field(description="The model's silence backstop; null for a model without one")
    model: str


class Health(BaseModel):
    status: str
    models: list[str]


def samples(pcm: bytes) -> np.ndarray:
    """16-bit little-endian PCM as float samples in [-1, 1), one row: a batch of one window."""
    return (np.frombuffer(pcm, "<i2") / 32768).astype(np.float32)[None, :]


class Served(Protocol):
    """A loaded model as the API serves it: P(EOT) for one request, and its firing rule's settings."""

    @property
    def name(self) -> str: ...

    @property
    def threshold(self) -> float: ...

    @property
    def backstop_s(self) -> float: ...

    def predict(self, request: Inputs) -> float: ...


def load_models() -> list[Served]:
    """The saved models; the combined model shares the single-input models' encoders."""
    text, audio = load_text_only(), load_audio_only()
    assert isinstance(text.classifier, TextClassifier) and isinstance(audio.classifier, AudioClassifier)
    return [text, audio, load_combined(text_encoder=text.classifier.encoder, audio_encoder=audio.classifier.encoder)]


def create_app(load: Callable[[], Sequence[Served]] = load_models) -> FastAPI:
    """The API, serving the models `load` returns, one per entry in SERVED; it is called once, at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        models = {model.name: model for model in load()}
        if set(models) != set(SERVED_BY_NAME):
            raise ValueError(f"expected the models {sorted(SERVED_BY_NAME)}, got {sorted(models)}")
        # Load the encoders now, not on the first request.
        for model in models.values():
            model.predict(Inputs(samples(bytes(AUDIO_BYTES)), "", "", 0.0))
        app.state.models = models
        yield

    app = FastAPI(title="Turn detector", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", models=list(request.app.state.models))

    # A plain `def`: the encoders block, so FastAPI runs it in its thread pool, off the event loop.
    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        inputs = body.inputs()
        serving: Serving = next(serving for serving in SERVED if serving.answers(inputs))
        model: Served = request.app.state.models[serving.name]
        return PredictResponse(
            p_eot=model.predict(inputs), threshold=model.threshold, backstop_ms=round(model.backstop_s * 1000), model=model.name
        )

    return app


app = create_app()
