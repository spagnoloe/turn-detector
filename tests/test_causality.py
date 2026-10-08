"""Causality: changing audio, text or segments after time t never changes firings before t.

Every model goes in MODELS_UNDER_TEST. The check builds random synthetic sides, rewrites
everything after a cut time t (the user's audio, segments still in progress at t and their
text, segments starting after t, and the same for the other speaker's turns), and requires the
firings before t to be identical. Cuts fall at random times and just after each of the model's
firings, so a model reading even slightly ahead of its firing is caught.
"""

import random
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pytest

from turn_detector.model import Audio, Model, Segment, SpeakerSide
from turn_detector.models.audio_only import AudioOnly
from turn_detector.models.baseline import Baseline
from turn_detector.models.text_only import TextContext, TextOnly


@dataclass(frozen=True)
class HashClassifier:
    """A stand-in for the trained classifier whose P(EOT) changes with any change to its input."""

    def p_eot(self, contexts: Sequence[TextContext]) -> list[float]:
        return [zlib.crc32(repr(context).encode()) / 2**32 for context in contexts]


@dataclass(frozen=True)
class ProjectionClassifier:
    """A stand-in for the trained audio classifier whose P(EOT) changes with any change to its
    window or to the silence duration: a random projection, wrapped onto [0, 1]."""

    def p_eot(self, windows: np.ndarray, silence_s: np.ndarray) -> np.ndarray:
        projection = np.random.default_rng(0).standard_normal(windows.shape[1])
        return (np.sin(windows @ projection * 1e3 + silence_s * 7919) + 1) / 2


MODELS_UNDER_TEST: list[Model] = [
    Baseline(timeout_ms=200),
    Baseline(timeout_ms=1000),
    TextOnly(HashClassifier(), threshold=0.5, backstop_s=1.5),
    TextOnly(HashClassifier(), threshold=0.3, backstop_s=0.4),
    AudioOnly(ProjectionClassifier(), threshold=0.9, backstop_s=1.5),
    AudioOnly(ProjectionClassifier(), threshold=0.99, backstop_s=0.4),
]

DURATION_S = 30.0
SAMPLE_RATE = 16_000
WORDS = "yes no so I would like to fly to Barcelona on Friday um well maybe".split()


def random_text(rng: random.Random) -> str:
    return " ".join(rng.choices(WORDS, k=rng.randint(1, 8)))


def random_segments(rng: random.Random, start: float, end: float) -> list[Segment]:
    """Speech segments separated by pauses of 50 ms to 2 s, starting after `start`."""
    segments, time = [], start
    while True:
        time += rng.uniform(0.05, 2.0)
        length = rng.uniform(0.2, 3.0)
        if time + length > end:
            return segments
        segments.append(Segment(time, time + length, random_text(rng)))
        time += length


def random_side(rng: random.Random) -> SpeakerSide:
    samples = np.random.default_rng(rng.randrange(2**32)).standard_normal(int(DURATION_S * SAMPLE_RATE))
    return SpeakerSide(
        conversation_id="synthetic",
        speaker=1,
        duration_s=DURATION_S,
        audio=Audio(samples.astype(np.float32), SAMPLE_RATE),
        segments=random_segments(rng, 0.0, DURATION_S),
        other_turns=random_segments(rng, 0.0, DURATION_S),
    )


def rewrite_after(segments: list[Segment], t: float, rng: random.Random) -> list[Segment]:
    """Keep segments over by t; give those in progress a new end and text; replace the rest."""
    past = [segment for segment in segments if segment.end <= t]
    in_progress = [
        Segment(segment.start, rng.uniform(t + 0.01, t + 3.0), random_text(rng))
        for segment in segments
        if segment.start < t < segment.end
    ]
    future_start = max([t, *(segment.end for segment in in_progress)])
    return past + in_progress + random_segments(rng, future_start, DURATION_S)


def perturb_after(side: SpeakerSide, t: float, rng: random.Random) -> SpeakerSide:
    """The same side up to time t, and something different after it."""
    assert side.audio is not None
    samples = side.audio.samples.copy()
    cut = int(np.ceil(t * side.audio.sample_rate))
    samples[cut:] = np.random.default_rng(rng.randrange(2**32)).standard_normal(len(samples) - cut)
    return replace(
        side,
        audio=Audio(samples, side.audio.sample_rate),
        segments=rewrite_after(side.segments, t, rng),
        other_turns=rewrite_after(side.other_turns, t, rng),
    )


def firings_before(model: Model, side: SpeakerSide, t: float) -> list[float]:
    return [firing for firing in model.fire(side) if firing < t]


def assert_causal(model: Model, *, sides: int = 30, cuts_per_side: int = 10, seed: int = 0) -> None:
    """Cut at random times, and just after each firing, where reading slightly ahead shows up."""
    rng = random.Random(seed)
    for _ in range(sides):
        side = random_side(rng)
        random_cuts = [rng.uniform(0.0, DURATION_S) for _ in range(cuts_per_side)]
        cuts_after_firings = [min(firing + rng.uniform(0.001, 0.5), DURATION_S) for firing in model.fire(side)]
        for t in random_cuts + cuts_after_firings:
            perturbed = perturb_after(side, t, rng)
            assert firings_before(model, perturbed, t) == firings_before(model, side, t), (
                f"{model.name}: firings before t={t:.3f} changed when the future changed"
            )


@pytest.mark.parametrize("model", MODELS_UNDER_TEST, ids=lambda model: f"{model.name} {model}")
def test_firings_never_depend_on_the_future(model: Model):
    assert_causal(model)


@dataclass(frozen=True)
class PeeksAhead:
    """Not causal: fires 200 ms into a pause only when the pause will turn out to be long."""

    name: str = "peeks ahead"

    def fire(self, side: SpeakerSide) -> list[float]:
        starts = [segment.start for segment in side.segments[1:]] + [side.duration_s]
        return [s.end + 0.2 for s, next_start in zip(side.segments, starts) if next_start - s.end > 1.0]


def test_the_check_catches_a_model_that_peeks_ahead():
    with pytest.raises(AssertionError, match="changed when the future changed"):
        assert_causal(PeeksAhead())
