"""Print gold EOT and mid-turn-pause counts per split and per speaker.

    uv run python scripts/count_events.py
"""

from collections import Counter

from turn_detector.data import iter_annotations
from turn_detector.events import build_events
from turn_detector.split import load_split

# TurnBench's published majority-consensus counts for the 38-conversation dev set
# (turnbench/README.md, `python -m turnbench.gold stats`).
PUBLISHED_EOTS = 1904
PUBLISHED_MID_TURN_PAUSES = 1063


def main() -> None:
    split = load_split()
    side = {i: "development" for i in split.development} | {i: "held_out" for i in split.held_out}
    eots: Counter[tuple[str, str]] = Counter()
    pauses: Counter[tuple[str, str]] = Counter()
    conversations: Counter[str] = Counter()
    for conversation in iter_annotations():
        conversations[side[conversation.conversation_id]] += 1
        for speaker, events in build_events(conversation.annotations).items():
            key = (side[conversation.conversation_id], conversation.speaker_ids[speaker])
            eots[key] += len(events.eots)
            pauses[key] += len(events.mid_turn_pauses)

    print(f"{'split':<12} {'speaker':<30} {'EOTs':>6} {'pauses':>7}")
    for split_name in ("development", "held_out"):
        speakers = sorted(speaker for name, speaker in eots.keys() | pauses.keys() if name == split_name)
        for speaker in speakers:
            print(f"{split_name:<12} {speaker:<30} {eots[split_name, speaker]:>6} {pauses[split_name, speaker]:>7}")
        split_eots = sum(n for (name, _), n in eots.items() if name == split_name)
        split_pauses = sum(n for (name, _), n in pauses.items() if name == split_name)
        label = f"TOTAL ({conversations[split_name]} conversations)"
        print(f"{split_name:<12} {label:<30} {split_eots:>6} {split_pauses:>7}\n")

    total_eots, total_pauses = sum(eots.values()), sum(pauses.values())
    print(f"TurnBench dev set (all {sum(conversations.values())} conversations): {total_eots} EOTs, {total_pauses} mid-turn pauses")
    print(f"TurnBench published: {PUBLISHED_EOTS} EOTs, {PUBLISHED_MID_TURN_PAUSES} mid-turn pauses")


if __name__ == "__main__":
    main()
