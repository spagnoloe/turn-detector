"""Generate the development / held-out split and store it in src/turn_detector/split.json.

Run once after downloading the data; the stored split is the source of truth afterwards.

    uv run python scripts/make_split.py
"""

from turn_detector.data import iter_annotations
from turn_detector.split import SPLIT_PATH, ConversationInfo, make_split, save_split

HELD_OUT_SIZE = 12

if __name__ == "__main__":
    conversations = [
        ConversationInfo(c.conversation_id, (c.speaker_ids[1], c.speaker_ids[2]), c.conversation_type)
        for c in iter_annotations()
    ]
    split = make_split(conversations, held_out_size=HELD_OUT_SIZE)
    save_split(split)
    print(f"wrote {SPLIT_PATH}: {len(split.development)} development, {len(split.held_out)} held out")
