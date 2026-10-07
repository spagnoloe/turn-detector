"""Event building: three annotator tracks per speaker -> gold EOTs and mid-turn pauses."""

from turn_detector.events import MidTurnPause, SpeakerEvents, build_events

TURN = "Normal Turn"
BACKCHANNEL = "Acknowledgement Backchannel"


def agreed(*segments: tuple[float, float, str]) -> dict[str, list[tuple[float, float, str, str]]]:
    """The same segments on all three annotator tracks."""
    return {annotator: [(start, end, label, "") for start, end, label in segments] for annotator in "abc"}


def tracks(speaker1: dict, speaker2: dict | None = None) -> dict:
    """Annotations keyed (speaker, annotator), as stored in the dataset."""
    by_speaker = {1: speaker1, 2: speaker2 or {}}
    return {
        (speaker, annotator): by_speaker[speaker].get(annotator, [])
        for speaker in (1, 2)
        for annotator in "abc"
    }


def test_agreement_within_200_ms_takes_the_median_time():
    events = build_events(
        tracks(
            {
                "a": [(0.0, 2.00, TURN, "")],
                "b": [(0.0, 2.15, TURN, "")],
                "c": [(0.0, 2.10, TURN, "")],
            }
        )
    )

    assert events[1] == SpeakerEvents(speaker=1, eots=[2.10], mid_turn_pauses=[])


def test_two_of_three_agreement_is_enough():
    events = build_events(
        tracks(
            {
                "a": [(0.0, 2.0, TURN, "")],
                "b": [(0.0, 2.1, TURN, "")],
                "c": [(0.0, 3.0, TURN, "")],
            }
        )
    )

    assert events[1].eots == [2.05]


def test_agreement_tolerance_is_200_ms():
    just_inside = build_events(
        tracks({"a": [(0.0, 2.0, TURN, "")], "b": [(0.0, 2.19, TURN, "")], "c": [(0.0, 3.0, TURN, "")]})
    )
    just_outside = build_events(
        tracks({"a": [(0.0, 2.0, TURN, "")], "b": [(0.0, 2.21, TURN, "")], "c": [(0.0, 2.42, TURN, "")]})
    )

    assert just_inside[1].eots == [2.095]
    assert just_outside[1].eots == []


def test_no_agreement_within_200_ms_gives_no_event():
    events = build_events(
        tracks(
            {
                "a": [(0.0, 2.0, TURN, "")],
                "b": [(0.0, 2.5, TURN, "")],
                "c": [(0.0, 3.0, TURN, "")],
            }
        )
    )

    assert events[1] == SpeakerEvents(speaker=1, eots=[], mid_turn_pauses=[])


def test_hand_over_to_the_other_speaker_is_an_eot():
    events = build_events(
        tracks(
            agreed((0.0, 2.0, TURN), (4.5, 6.0, TURN)),
            agreed((2.5, 4.0, TURN)),
        )
    )

    assert events[1].eots == [2.0, 6.0]
    assert events[1].mid_turn_pauses == []
    assert events[2].eots == [4.0]


def test_same_speaker_resuming_is_a_mid_turn_pause():
    events = build_events(tracks(agreed((0.0, 2.0, TURN), (2.6, 4.0, TURN))))

    assert events[1].mid_turn_pauses == [MidTurnPause(start=2.0, end=2.6)]
    assert events[1].eots == [4.0]


def test_the_speakers_last_turn_is_an_eot():
    events = build_events(tracks(agreed((0.0, 2.0, TURN)), agreed((2.5, 4.0, TURN), (5.0, 6.0, TURN))))

    assert events[2].eots == [6.0]


def test_other_speakers_backchannel_does_not_break_a_mid_turn_pause():
    events = build_events(
        tracks(
            agreed((0.0, 2.0, TURN), (2.6, 4.0, TURN)),
            agreed((2.2, 2.4, BACKCHANNEL)),
        )
    )

    assert events[1].mid_turn_pauses == [MidTurnPause(start=2.0, end=2.6)]
    assert events[1].eots == [4.0]
    assert events[2] == SpeakerEvents(speaker=2, eots=[], mid_turn_pauses=[])


def test_a_pause_only_one_annotator_marked_is_dropped():
    # Annotators a and b hear one turn; only c splits it with a pause at 2.0.
    events = build_events(
        tracks(
            {
                "a": [(0.0, 4.0, TURN, "")],
                "b": [(0.0, 4.0, TURN, "")],
                "c": [(0.0, 2.0, TURN, ""), (2.6, 4.0, TURN, "")],
            }
        )
    )

    assert events[1] == SpeakerEvents(speaker=1, eots=[4.0], mid_turn_pauses=[])


def test_a_turn_only_one_annotator_marked_is_dropped():
    events = build_events(
        tracks(
            agreed((0.0, 2.0, TURN)),
            {"a": [(2.5, 4.0, TURN, "")]},
        )
    )

    assert events[2] == SpeakerEvents(speaker=2, eots=[], mid_turn_pauses=[])
    # Without speaker 2's turn, speaker 1's segment is simply their last turn.
    assert events[1].eots == [2.0]


def test_events_are_returned_in_time_order():
    events = build_events(
        tracks(
            agreed((0.0, 1.0, TURN), (1.5, 2.0, TURN), (4.5, 5.0, TURN), (5.4, 6.0, TURN)),
            agreed((2.5, 4.0, TURN)),
        )
    )

    assert events[1].eots == [2.0, 6.0]
    assert events[1].mid_turn_pauses == [MidTurnPause(1.0, 1.5), MidTurnPause(5.0, 5.4)]
