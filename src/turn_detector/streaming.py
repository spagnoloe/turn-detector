"""The caller's side of the prediction API: stream one speaker's side of a conversation through it.

This is what the voice agent's orchestrator does on a live call. Every 50 ms while the user is
silent, up to 3 s into the pause, it sends the last 1 s of their audio, the transcript so far, the
other speaker's previous turn and the silence duration, and applies the firing rule of the model
that answered, at most once per pause and only while the user is still silent: fire once P(EOT)
>= the returned threshold, but not before that model's earliest firing (`Serving`: for the
text-only model, once the ASR's text has arrived, 200 ms into the pause), or once the silence
reaches the returned backstop. Which model answers depends on what is sent: audio and text, the
combined model; audio only, the audio-only model; text only, the text-only model.

The pauses are the gaps after the user's consensus segments, the same perfect pause detector the
evaluation uses, so streaming fires where the model's own `fire` does, rounded up to the next 50
ms step (the confident firings of the models that hear audio are already on that grid; their
windows here are cut from audio resampled as a whole rather than window by window, a negligible
difference). The text sent at step t is what the ASR has delivered by t: the context at the
user's latest segment end at least 200 ms before t (`text_read_at`), so it matches what the
evaluated models read.
"""

import base64
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.signal import resample_poly

from turn_detector.model import EPSILON_S, HORIZON_S, Audio, SpeakerSide, next_speech_start, steps
from turn_detector.models.text_only import text_read_at
from turn_detector.serving import AUDIO_SAMPLE_RATE, AUDIO_SAMPLES, SERVED_BY_NAME

type Predict = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Firing:
    time_s: float
    pause_start_s: float
    p_eot: float
    reason: Literal["confident", "backstop"]


def pcm_16k(audio: Audio) -> np.ndarray:
    """A channel as the API takes it: resampled to 16 kHz, as 16-bit samples."""
    common = math.gcd(AUDIO_SAMPLE_RATE, audio.sample_rate)
    resampled = resample_poly(audio.samples, AUDIO_SAMPLE_RATE // common, audio.sample_rate // common)
    return (np.clip(resampled, -1, 1) * 32767).astype(np.int16)


def audio_window(samples: np.ndarray, t: float) -> bytes:
    """The last 1 s of 16 kHz 16-bit samples before `t`, as little-endian PCM, left-padded with
    silence near the start of the call."""
    end = round(t * AUDIO_SAMPLE_RATE)
    window = samples[max(0, end - AUDIO_SAMPLES) : end].astype("<i2")
    return np.concatenate([np.zeros(AUDIO_SAMPLES - len(window), "<i2"), window]).tobytes()


def stream(side: SpeakerSide, predict: Predict, audio: np.ndarray | None = None, text: bool = True) -> list[Firing]:
    """Every firing on `side`, calling `predict` (one `/predict` request) every 50 ms in each pause.

    `audio`, the user's channel as 16 kHz 16-bit samples, is sent with every request if given; the
    transcript and the previous turn are sent unless `text` is False.
    """
    firings = []
    for end in sorted({segment.end for segment in side.segments}):
        times = list(steps(end, min(next_speech_start(side, end), side.duration_s, end + HORIZON_S)))
        for t, context in zip(times, text_read_at(side, times)):
            payload: dict[str, Any] = {"silence_ms": (t - end) * 1000}
            if text:
                payload |= {"transcript": context.turn_so_far, "previous_turn": context.previous_turn}
            if audio is not None:
                payload["audio"] = base64.b64encode(audio_window(audio, t)).decode()
            response = predict(payload)
            silence_s = t - end + EPSILON_S
            earliest_s = SERVED_BY_NAME[response["model"]].earliest_firing_s
            if silence_s >= earliest_s and response["p_eot"] >= response["threshold"]:
                firings.append(Firing(t, end, response["p_eot"], "confident"))
                break
            if response["backstop_ms"] is not None and silence_s >= response["backstop_ms"] / 1000:
                firings.append(Firing(t, end, response["p_eot"], "backstop"))
                break
    return firings
