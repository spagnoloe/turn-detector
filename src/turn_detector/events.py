"""Gold events: each speaker's EOTs and mid-turn pauses, built from the three annotator tracks."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from turnbench.data import SPEAKERS, Annotation, Conversation
from turnbench.gold import events_for_conversation


def turnbench_conversation(
    annotations: Mapping[tuple[int, str], Sequence[Annotation]],
    conversation_id: str = "",
    duration_s: float = 0.0,
) -> Conversation:
    """A TurnBench conversation for its gold and consensus code, which read only the annotations.

    The id and duration are placeholders unless given; audio is never attached.
    """
    return Conversation(
        conversation_id=conversation_id,
        duration_s=duration_s,
        annotations={key: list(segments) for key, segments in annotations.items()},
        audio_bytes={},
    )


@dataclass(frozen=True)
class MidTurnPause:
    """A silence in a speaker's turn, in seconds, after which the same speaker continues."""

    start: float
    end: float


@dataclass(frozen=True)
class SpeakerEvents:
    """One speaker's gold events, each list in time order."""

    speaker: int
    eots: list[float]
    mid_turn_pauses: list[MidTurnPause]


def build_events(annotations: Mapping[tuple[int, str], Sequence[Annotation]]) -> dict[int, SpeakerEvents]:
    """Build each speaker's gold EOTs and mid-turn pauses from a conversation's annotator tracks.

    `annotations` maps (speaker, annotator) to that annotator's (start_s, end_s, label, text)
    segments, as stored in the dataset. TurnBench's own gold construction is used, so the
    events are exactly the ones its scorer evaluates against: a segment counts only when at
    least 2 of 3 annotators agree on both its start and end within ±200 ms (its times are their
    median), and non-agreed segments are dropped. A segment end is an EOT when the other speaker
    takes the next turn or it is the speaker's last turn; it is a mid-turn pause when the same
    speaker resumes first. Pauses are cut short at contrary evidence (disputed regions, the
    speaker's own backchannel, an interruption), but not at the other speaker's backchannel.
    """
    gold = events_for_conversation(turnbench_conversation(annotations))
    return {
        speaker: SpeakerEvents(
            speaker=speaker,
            eots=sorted(event.time_s for event in gold.eot_positive_events if event.speaker == speaker),
            mid_turn_pauses=sorted(
                (MidTurnPause(span.start, span.end) for span in gold.eot_negative_spans if span.speaker == speaker),
                key=lambda pause: pause.start,
            ),
        )
        for speaker in SPEAKERS
    }
