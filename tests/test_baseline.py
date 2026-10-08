"""The baseline (a silence timeout), through the model interface."""

from turn_detector.model import Segment, SpeakerSide
from turn_detector.models.baseline import Baseline


def side(*segments: tuple[float, float], duration_s: float = 20.0) -> SpeakerSide:
    """A user side with the given speech segments and no other speaker."""
    return SpeakerSide(
        conversation_id="synthetic",
        speaker=1,
        duration_s=duration_s,
        audio=None,
        segments=[Segment(start, end, "") for start, end in segments],
        other_turns=[],
    )


def test_fires_at_segment_end_plus_timeout():
    assert Baseline(timeout_ms=500).fire(side((1.0, 3.0))) == [3.5]


def test_does_not_fire_when_the_user_resumes_first():
    # The 0.4 s pause after 3.0 is shorter than the timeout; the 2 s pause after 5.0 is not.
    assert Baseline(timeout_ms=500).fire(side((1.0, 3.0), (3.4, 5.0), (7.0, 8.0))) == [5.5, 8.5]


def test_does_not_fire_after_the_conversation_ends():
    assert Baseline(timeout_ms=500).fire(side((1.0, 3.0), (18.0, 19.8), duration_s=20.0)) == [3.5]


def test_segments_ending_together_fire_once():
    # Overlapping consensus segments can share an end time; firings must be strictly increasing.
    assert Baseline(timeout_ms=500).fire(side((1.0, 3.0), (2.0, 3.0))) == [3.5]
