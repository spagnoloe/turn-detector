"""The interface every model implements: one speaker's side in, firing times out.

A model treats one speaker of a conversation as "the user" and decides when that user's turn
has ended and the agent should respond. It sees what a live detector would have: the user's own
audio channel, the user's speech segments with their transcripts, and the other speaker's turns.
Each model owns its firing rule. Firings, not probabilities, are what gets scored; a model built
on a classifier's P(EOT) (the served detector's output, ADR 0001) turns it into firings with its
threshold. A trained model is fitted on the development conversations first (`Fitted`). Models
must be causal: a firing at time t may depend on nothing after t
(tests/test_causality.py checks every model).
"""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Segment:
    """A stretch of speech, in seconds, with its transcript."""

    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Audio:
    """Mono samples of one speaker's channel."""

    samples: np.ndarray
    sample_rate: int


@dataclass(frozen=True)
class SpeakerSide:
    """One speaker's side of a conversation, as seen by a model treating them as the user.

    `segments` is the user's speech timeline in start order; the gaps between segments are the
    pauses. `other_turns` is the other speaker's speech in start order. `audio` is the user's
    own channel, or None when it wasn't loaded (models that don't listen don't need it).
    """

    conversation_id: str
    speaker: int
    duration_s: float
    audio: Audio | None
    segments: list[Segment]
    other_turns: list[Segment]


class Model(Protocol):
    """A turn detector: decides when the user's turn has ended."""

    @property
    def name(self) -> str:
        """A short name for tables and figures."""
        ...

    def fire(self, side: SpeakerSide) -> list[float]:
        """Firing times in seconds, strictly increasing and within the conversation."""
        ...


# One value per knob of a model, in the order its registration names them: (N,) for the
# baseline's timeout, (threshold, backstop) for the text-only model.
type Setting = tuple[float, ...]


class Fitted(Protocol):
    """A model fitted to the development conversations, ready to build at any setting."""

    def build(self, setting: Setting) -> Model:
        """The model at this setting, as the evaluation scores it. A trained model's classifier
        never saw the speaker group of the conversation it is scoring."""
        ...

    def save(self, setting: Setting) -> list[Path]:
        """Save the final model at the chosen setting for serving; a plain rule saves nothing."""
        ...


def next_speech_start(side: SpeakerSide, end: float) -> float:
    """When the user is next heard after a segment end at `end`: the earliest start of their
    speech still going on after `end`, or infinity. A firing at t is in silence iff t <= this."""
    return min((segment.start for segment in side.segments if segment.end > end), default=float("inf"))


# A live detector is asked every 50 ms, on a grid counted from the start of the call.
STEP_S = 0.05
EPSILON_S = 1e-6  # steps are on a float grid; compare times with this much slack
# A model stops listening this far into a pause: TurnBench never counts a firing more than 3 s
# after an EOT as a hit.
HORIZON_S = 3.0


def steps(start: float, until: float) -> Iterator[float]:
    """The 50 ms grid points from `start` to `until`, both included."""
    k = math.ceil(start / STEP_S - EPSILON_S)
    while (t := k / round(1 / STEP_S)) <= until + EPSILON_S:
        yield t
        k += 1
