"""The caller's side of the prediction API: stream one speaker's side of a conversation through it.

This is what the voice agent's orchestrator does on a live call. Every 50 ms while the user is
silent it sends the last 1 s of their audio, the transcript so far, the other speaker's previous
turn and the silence duration, and applies the firing rule to the response: fire once the ASR's
text has arrived (200 ms into the pause) if P(EOT) >= the returned threshold, or once the silence
reaches the returned backstop, at most once per pause and only while the user is still silent.

The pauses are the gaps after the user's consensus segments, the same perfect pause detector the
evaluation uses, so streaming the text-only model fires where `TextOnly.fire` does, rounded up to
the next 50 ms step. The text sent at step t is what the ASR has finalised by t: segments that
ended by t - 200 ms, and at most up to the pause start (the other speaker's speech during the
user's pause is not a new agent turn), so it matches what the evaluated model reads.
"""

import base64
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.signal import resample_poly

from turn_detector.model import Audio, SpeakerSide, next_speech_start
from turn_detector.models.text_only import ASR_LAG_S, text_context
from turn_detector.serving import AUDIO_SAMPLE_RATE, AUDIO_SAMPLES

STEP_S = 0.05
EPSILON_S = 1e-6  # steps are on a float grid; compare times with this much slack

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


def steps(start: float, until: float) -> Iterator[float]:
    """The 50 ms grid points from `start` to `until`, both included."""
    k = math.ceil(start / STEP_S - EPSILON_S)
    while (t := k / round(1 / STEP_S)) <= until + EPSILON_S:
        yield t
        k += 1


def stream(side: SpeakerSide, predict: Predict, audio: np.ndarray | None = None) -> list[Firing]:
    """Every firing on `side`, calling `predict` (one `/predict` request) every 50 ms in each pause.

    `audio`, the user's channel as 16 kHz 16-bit samples, is sent with every request if given.
    """
    firings = []
    for end in sorted({segment.end for segment in side.segments}):
        for t in steps(end, min(next_speech_start(side, end), side.duration_s)):
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
            if silence_s >= ASR_LAG_S and response["p_eot"] >= response["threshold"]:
                firings.append(Firing(t, end, response["p_eot"], "confident"))
                break
            if response["backstop_ms"] is not None and silence_s >= response["backstop_ms"] / 1000:
                firings.append(Firing(t, end, response["p_eot"], "backstop"))
                break
    return firings
