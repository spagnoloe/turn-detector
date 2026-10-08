"""The caller's side of the prediction API: stream one speaker's side of a conversation through it.

This is what the voice agent's orchestrator does on a live call. Every 50 ms while the user is
silent, up to 3 s into the pause, it sends the last 1 s of their audio, the transcript so far, the
other speaker's previous turn and the silence duration, and applies the response's model's firing
rule, at most once per pause and only while the user is still silent:

- the audio-only model (requests with audio): fire as soon as P(EOT) >= the returned threshold;
- the text-only model (requests without audio): fire once the ASR's text has arrived (200 ms into
  the pause) if P(EOT) >= the returned threshold, or once the silence reaches the returned backstop.

The pauses are the gaps after the user's consensus segments, the same perfect pause detector the
evaluation uses, so streaming fires where the model's own `fire` does, rounded up to the next 50
ms step for the text-only model (the audio-only model already runs on that grid; its windows here
are cut from audio resampled as a whole rather than window by window, a negligible difference).
The text sent at step t is what the ASR has finalised by t: segments that ended by t - 200 ms, and
at most up to the pause start (the other speaker's speech during the user's pause is not a new
agent turn), so it matches what the evaluated text model reads.
"""

import base64
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.signal import resample_poly

from turn_detector.model import EPSILON_S, HORIZON_S, Audio, SpeakerSide, next_speech_start, steps
from turn_detector.models import text_only
from turn_detector.models.text_only import ASR_LAG_S, text_context
from turn_detector.serving import AUDIO_SAMPLE_RATE, AUDIO_SAMPLES

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


def stream(side: SpeakerSide, predict: Predict, audio: np.ndarray | None = None) -> list[Firing]:
    """Every firing on `side`, calling `predict` (one `/predict` request) every 50 ms in each pause.

    `audio`, the user's channel as 16 kHz 16-bit samples, is sent with every request if given.
    """
    firings = []
    for end in sorted({segment.end for segment in side.segments}):
        for t in steps(end, min(next_speech_start(side, end), side.duration_s, end + HORIZON_S)):
            context = text_context(side, min(end, t - ASR_LAG_S + EPSILON_S))
            payload: dict[str, Any] = {
                "transcript": context.turn_so_far,
                "previous_turn": context.previous_turn,
                "silence_ms": (t - end) * 1000,
            }
            if audio is not None:
                payload["audio"] = base64.b64encode(audio_window(audio, t)).decode()
            response = predict(payload)
            silence_s = t - end + EPSILON_S
            # Only the text model waits for the ASR: before its lag, it would read stale text.
            earliest_s = ASR_LAG_S if response["model"] == text_only.NAME else 0.0
            if silence_s >= earliest_s and response["p_eot"] >= response["threshold"]:
                firings.append(Firing(t, end, response["p_eot"], "confident"))
                break
            if response["backstop_ms"] is not None and silence_s >= response["backstop_ms"] / 1000:
                firings.append(Firing(t, end, response["p_eot"], "backstop"))
                break
    return firings
