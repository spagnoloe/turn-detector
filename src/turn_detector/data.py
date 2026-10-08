"""Where the TurnBench dev set lives on disk, how it gets there, and how it is read back."""

import io
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import soundfile
from huggingface_hub import snapshot_download
from turnbench.data import ANNOTATORS, DEV_DATASET, DEV_REVISION, SPEAKERS, Annotation

from turn_detector.model import Audio

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
DEV_DATA_DIR = DATA_DIR / "turn-benchmark-dev"


def annotation_column(speaker: int, annotator: str) -> str:
    return f"speaker_{speaker}_annotation_{annotator}"


ANNOTATION_COLUMNS = [annotation_column(speaker, annotator) for speaker in SPEAKERS for annotator in ANNOTATORS]


@dataclass(frozen=True)
class ConversationAnnotations:
    """One conversation's annotator tracks and metadata, without its audio.

    `annotations` maps (speaker, annotator) to that annotator's (start_s, end_s, label, text)
    segments. `speaker_ids` maps each channel to the actor recorded on it.
    """

    conversation_id: str
    conversation_type: str
    speaker_ids: dict[int, str]
    annotations: dict[tuple[int, str], list[Annotation]]


def download_dev_set(dest: Path = DEV_DATA_DIR) -> Path:
    """Download the TurnBench dev set at the revision TurnBench pins, into `dest`.

    The dataset is gated: accept its terms on Hugging Face and log in (`hf auth login`) first.
    """
    snapshot_download(
        DEV_DATASET,
        repo_type="dataset",
        revision=DEV_REVISION,
        local_dir=dest,
    )
    return dest


def parquet_files(data_dir: Path = DEV_DATA_DIR) -> list[Path]:
    files = sorted(data_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {data_dir}; run scripts/download_data.py first")
    return files


def iter_annotations(data_dir: Path = DEV_DATA_DIR) -> Iterator[ConversationAnnotations]:
    """Every conversation's annotations and metadata, reading no audio, in numeric id order."""
    conversations = []
    for file in parquet_files(data_dir):
        table = pq.read_table(file, columns=["conversation_id", "metadata", *ANNOTATION_COLUMNS])
        for row in table.to_pylist():
            metadata = row["metadata"]
            conversations.append(
                ConversationAnnotations(
                    conversation_id=row["conversation_id"],
                    conversation_type=metadata["conversation_type"],
                    speaker_ids={speaker: metadata[f"speaker_{speaker}_actor_id"] for speaker in SPEAKERS},
                    annotations={
                        (speaker, annotator): [
                            (segment["start_s"], segment["end_s"], segment["label"], segment["text"])
                            for segment in row[annotation_column(speaker, annotator)]
                        ]
                        for speaker in SPEAKERS
                        for annotator in ANNOTATORS
                    },
                )
            )
    yield from sorted(conversations, key=lambda conversation: int(conversation.conversation_id))


def load_audio(conversation_id: str, speaker: int, data_dir: Path = DEV_DATA_DIR) -> Audio:
    """One speaker's channel of one conversation, decoded to float samples in [-1, 1]."""
    column = f"speaker_{speaker}_audio"
    for file in parquet_files(data_dir):
        table = pq.read_table(file, columns=[column], filters=[("conversation_id", "=", conversation_id)])
        if table.num_rows:
            samples, sample_rate = soundfile.read(io.BytesIO(table[column][0]["bytes"].as_py()), dtype="float32")
            return Audio(samples if samples.ndim == 1 else samples.mean(axis=1), sample_rate)
    raise KeyError(f"no conversation {conversation_id!r} under {data_dir}")
