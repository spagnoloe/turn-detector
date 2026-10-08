"""The prediction API: P(EOT) for one moment of a call, from what the caller sends.

The service is stateless. The caller (the voice agent's orchestrator) keeps the rolling 1 s audio
buffer and the transcript, calls `POST /predict` every 50 ms while the user is silent, and applies
the firing rule itself with the threshold and backstop each response carries
(`turn_detector.streaming` is that caller). ASR runs upstream (ADR 0001): `transcript` is the
user's turn so far as the ASR has finalised it, and `previous_turn` is the agent's last turn.

A request with audio is answered by the audio-only model, which hears the audio and the silence
duration (so a request with audio must carry `silence_ms`); any other request by the text-only
model, which reads only the text. Both models are loaded once, at startup.

    uv run uvicorn turn_detector.serving:app
"""

import base64
import binascii
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from typing import Self

import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from turn_detector.models.audio_only import AudioOnly, load_audio_only
from turn_detector.models.text_only import TextContext, TextOnly, load_text_only

AUDIO_SAMPLE_RATE = 16_000
# The last 1 s of the user's channel: 16 kHz mono 16-bit little-endian PCM.
AUDIO_SAMPLES = AUDIO_SAMPLE_RATE
AUDIO_BYTES = AUDIO_SAMPLES * 2


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
            raise ValueError("a request with audio needs silence_ms, which the audio model hears too")
        return self


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


def create_app(
    load_text: Callable[[], TextOnly] = load_text_only, load_audio: Callable[[], AudioOnly] = load_audio_only
) -> FastAPI:
    """The API, serving the models `load_text` and `load_audio` return; each is called once, at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        text, audio = load_text(), load_audio()
        # Load the encoders now, not on the first request.
        text.classifier.p_eot([TextContext("", "")])
        audio.classifier.p_eot(samples(bytes(AUDIO_BYTES)), np.zeros(1))
        app.state.text, app.state.audio = text, audio
        yield

    app = FastAPI(title="Turn detector", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", models=[request.app.state.text.name, request.app.state.audio.name])

    # A plain `def`: the encoders block, so FastAPI runs it in its thread pool, off the event loop.
    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        if body.audio is not None and body.silence_ms is not None:
            audio: AudioOnly = request.app.state.audio
            [p_eot] = audio.classifier.p_eot(samples(body.audio), np.array([body.silence_ms / 1000]))
            return PredictResponse(p_eot=float(p_eot), threshold=audio.threshold, backstop_ms=None, model=audio.name)
        text: TextOnly = request.app.state.text
        [p_eot] = text.classifier.p_eot([TextContext(body.previous_turn or "", body.transcript or "")])
        return PredictResponse(
            p_eot=p_eot, threshold=text.threshold, backstop_ms=round(text.backstop_s * 1000), model=text.name
        )

    return app


app = create_app()
