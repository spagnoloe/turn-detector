"""The prediction API: P(EOT) for one moment of a call, from what the caller sends.

The service is stateless. The caller (the voice agent's orchestrator) keeps the rolling 1 s audio
buffer and the transcript, calls `POST /predict` every 50 ms while the user is silent, and applies
the firing rule itself with the threshold and backstop each response carries
(`turn_detector.streaming` is that caller). ASR runs upstream (ADR 0001): `transcript` is the
user's turn so far as the ASR has finalised it, and `previous_turn` is the agent's last turn.

It serves the text-only model, which reads only the text: `audio` and `silence_ms` are validated
but not used yet. The model is loaded once, at startup.

    uv run uvicorn turn_detector.serving:app
"""

import base64
import binascii
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

from turn_detector.models.text_only import TextContext, TextOnly, load_text_only

AUDIO_SAMPLE_RATE = 16_000
AUDIO_WINDOW_S = 1.0
# The last 1 s of the user's channel: 16 kHz mono 16-bit little-endian PCM.
AUDIO_BYTES = int(AUDIO_SAMPLE_RATE * AUDIO_WINDOW_S) * 2


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


class PredictResponse(BaseModel):
    p_eot: float
    threshold: float
    backstop_ms: int | None = Field(description="The model's silence backstop; null for a model without one")
    model: str


class Health(BaseModel):
    status: str
    model: str


def create_app(load_model: Callable[[], TextOnly] = load_text_only) -> FastAPI:
    """The API, serving the model `load_model` returns; it is called once, at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        model = load_model()
        model.classifier.p_eot([TextContext("", "")])  # load the encoder now, not on the first request
        app.state.model = model
        yield

    app = FastAPI(title="Turn detector", lifespan=lifespan)

    def served(request: Request) -> TextOnly:
        return request.app.state.model

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(status="ok", model=served(request).name)

    # A plain `def`: the encoder blocks, so FastAPI runs it in its thread pool, off the event loop.
    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        model = served(request)
        context = TextContext(body.previous_turn or "", body.transcript or "")
        [p_eot] = model.classifier.p_eot([context])
        return PredictResponse(
            p_eot=p_eot, threshold=model.threshold, backstop_ms=round(model.backstop_s * 1000), model=model.name
        )

    return app


app = create_app()
