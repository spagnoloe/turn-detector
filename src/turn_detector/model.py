"""The interface every model implements: one speaker's side in, firing times out.

A model treats one speaker of a conversation as "the user" and decides when that user's turn
has ended and the agent should respond. It sees what a live detector would have: the user's own
audio channel, the user's speech segments with their transcripts, and the other speaker's turns.
Each model owns its firing rule. Firings, not probabilities, are what gets scored; a model built
on a classifier's P(EOT) (the served detector's output, ADR 0001) turns it into firings with its
threshold. Models must be causal: a firing at time t may depend on nothing after t
(tests/test_causality.py checks every model).
"""

from dataclasses import dataclass
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
