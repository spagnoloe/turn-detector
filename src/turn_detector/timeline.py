"""Speaker sides from the annotations: each speaker's speech timeline, with transcripts.

The timeline is TurnBench's turn-view consensus: the floor-claiming segments that at least 2 of
3 annotators agree on, which are the segments its gold EOTs and mid-turn pauses are anchored to.
Their ends are the pauses every system sees, a perfect pause detector standing in for a real
VAD. Two consequences: speech the annotators don't agree on is missing from the timeline, so it
can look like a pause (firings there mostly fall in TurnBench's disputed regions, which are
neither rewarded nor penalised); and consensus segments carry no text, so each takes the
transcript of the closest-matching annotator segment, which can be incomplete when annotators
split the speech differently.
"""

from collections.abc import Mapping, Sequence

from turnbench.data import ANNOTATORS, SPEAKERS, Annotation
from turnbench.gold import TURN_CANONICAL, collect_turns, consensus_for_conversation

from turn_detector.detector import Segment, SpeakerSide
from turn_detector.events import turnbench_conversation


def transcript(start: float, end: float, candidates: Sequence[Annotation]) -> str:
    """The text of the annotator segment whose start and end are closest to (start, end)."""
    closest = min(candidates, key=lambda c: abs(c[0] - start) + abs(c[1] - end))
    return closest[3]


def speech_timelines(annotations: Mapping[tuple[int, str], Sequence[Annotation]]) -> dict[int, list[Segment]]:
    """Each speaker's consensus speech segments with transcripts, in start order."""
    turn_events, _ = consensus_for_conversation(turnbench_conversation(annotations), canonical=TURN_CANONICAL)
    timelines = {}
    for speaker in SPEAKERS:
        candidates = [
            segment
            for annotator in ANNOTATORS
            for segment in annotations[(speaker, annotator)]
            if segment[2] in TURN_CANONICAL
        ]
        timelines[speaker] = [
            Segment(event.start, event.end, transcript(event.start, event.end, candidates))
            for event in collect_turns(turn_events, speaker).segments
        ]
    return timelines


def speaker_sides(
    conversation_id: str,
    duration_s: float,
    annotations: Mapping[tuple[int, str], Sequence[Annotation]],
) -> list[SpeakerSide]:
    """Both sides of a conversation, each speaker in turn as the user. Audio isn't loaded."""
    timelines = speech_timelines(annotations)
    return [
        SpeakerSide(
            conversation_id=conversation_id,
            speaker=speaker,
            duration_s=duration_s,
            audio=None,
            segments=timelines[speaker],
            other_turns=timelines[other],
        )
        for speaker, other in ((1, 2), (2, 1))
    ]
