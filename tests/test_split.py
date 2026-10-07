"""The development / held-out split of the dev-set conversations."""

from turn_detector.split import ConversationInfo, Split, load_split, make_split


def conversation(conversation_id: str, speakers: tuple[str, str], conversation_type: str = "Casual") -> ConversationInfo:
    return ConversationInfo(conversation_id, speakers, conversation_type)


CONVERSATIONS = [
    conversation("1", ("ann", "bob")),
    conversation("2", ("ann", "bob"), "Instructional"),
    conversation("3", ("bob", "cat")),
    conversation("4", ("dan", "eve"), "Instructional"),
    conversation("5", ("fay", "gus")),
    conversation("6", ("hal", "ivy"), "Instructional"),
    conversation("7", ("hal", "jon")),
    conversation("8", ("kim", "lee")),
]


def test_every_conversation_lands_in_exactly_one_side():
    split = make_split(CONVERSATIONS, held_out_size=3)

    assert sorted(split.development + split.held_out, key=int) == [c.conversation_id for c in CONVERSATIONS]
    assert not set(split.development) & set(split.held_out)


def test_no_speaker_appears_on_both_sides():
    split = make_split(CONVERSATIONS, held_out_size=3)

    speakers = {c.conversation_id: set(c.speakers) for c in CONVERSATIONS}
    development_speakers = set().union(*(speakers[i] for i in split.development))
    held_out_speakers = set().union(*(speakers[i] for i in split.held_out))
    assert not development_speakers & held_out_speakers


def test_held_out_size_is_met_when_speaker_groups_allow_it():
    assert len(make_split(CONVERSATIONS, held_out_size=3).held_out) == 3
    assert len(make_split(CONVERSATIONS, held_out_size=2).held_out) == 2


def test_held_out_keeps_conversation_types_balanced():
    conversations = [
        conversation("1", ("ann", "bob")),
        conversation("2", ("cat", "dan")),
        conversation("3", ("eve", "fay"), "Instructional"),
        conversation("4", ("gus", "hal"), "Instructional"),
    ]

    split = make_split(conversations, held_out_size=2)

    assert len(split.held_out) == 2
    assert {conversations[int(i) - 1].conversation_type for i in split.held_out} == {"Casual", "Instructional"}


def test_split_is_deterministic_and_independent_of_input_order():
    assert make_split(CONVERSATIONS, held_out_size=3) == make_split(list(reversed(CONVERSATIONS)), held_out_size=3)


def test_stored_split_covers_the_dev_set_once():
    split = load_split()

    assert isinstance(split, Split)
    assert len(split.development) + len(split.held_out) == 38
    assert not set(split.development) & set(split.held_out)
    assert 11 <= len(split.held_out) <= 13
