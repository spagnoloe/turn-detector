"""The fixed split of the dev-set conversations into development and held-out sets (ADR 0003).

The same actors appear in several conversations, so the split keeps speakers apart, not just
conversations: every conversation an actor took part in falls on the same side. The held-out
set is otherwise chosen to match a target size and the whole set's mix of conversation types.
The split is generated once by `scripts/make_split.py` and stored in `split.json`.
"""

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

SPLIT_PATH = Path(__file__).with_name("split.json")


@dataclass(frozen=True)
class ConversationInfo:
    """What the split needs to know about a conversation."""

    conversation_id: str
    speakers: tuple[str, str]
    conversation_type: str


@dataclass(frozen=True)
class Split:
    """Conversation ids on each side of the split, in numeric order."""

    development: list[str]
    held_out: list[str]


def speaker_groups(conversations: Sequence[ConversationInfo]) -> list[list[ConversationInfo]]:
    """Group conversations that share a speaker, directly or through other conversations."""
    groups: list[tuple[set[str], list[ConversationInfo]]] = []
    for conversation in sorted(conversations, key=lambda c: int(c.conversation_id)):
        speakers, members = set(conversation.speakers), [conversation]
        for group in [group for group in groups if group[0] & speakers]:
            groups.remove(group)
            speakers |= group[0]
            members += group[1]
        groups.append((speakers, members))
    return sorted(
        (sorted(members, key=lambda c: int(c.conversation_id)) for _, members in groups),
        key=lambda members: int(members[0].conversation_id),
    )


def make_split(conversations: Sequence[ConversationInfo], held_out_size: int) -> Split:
    """Pick whole speaker groups for the held-out set, trying every combination.

    Preference order: held-out size closest to `held_out_size`, then conversation types closest
    to their share of the whole set, then the lowest conversation ids (so the result never
    depends on input order or randomness).
    """
    groups = speaker_groups(conversations)
    all_types = Counter(c.conversation_type for c in conversations)
    share = held_out_size / len(conversations)

    def cost(chosen: tuple[list[ConversationInfo], ...]) -> tuple[int, float, list[int]]:
        members = [c for group in chosen for c in group]
        types = Counter(c.conversation_type for c in members)
        imbalance = sum(abs(types[t] - share * n) for t, n in all_types.items())
        return abs(len(members) - held_out_size), imbalance, sorted(int(c.conversation_id) for c in members)

    candidates = (chosen for size in range(len(groups) + 1) for chosen in combinations(groups, size))
    best = min(candidates, key=cost)
    held_out = {c.conversation_id for group in best for c in group}
    ids = sorted((c.conversation_id for c in conversations), key=int)
    return Split(
        development=[i for i in ids if i not in held_out],
        held_out=[i for i in ids if i in held_out],
    )


def save_split(split: Split, path: Path = SPLIT_PATH) -> None:
    path.write_text(json.dumps({"development": split.development, "held_out": split.held_out}, indent=2) + "\n")


def load_split(path: Path = SPLIT_PATH) -> Split:
    stored = json.loads(path.read_text())
    return Split(development=stored["development"], held_out=stored["held_out"])
