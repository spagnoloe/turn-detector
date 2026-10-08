"""The text-only model: at each pause, do the user's words so far sound finished?

At every segment end the classifier reads the other speaker's previous turn and the user's turn
so far, and outputs P_text(EOT). The firing rule is the one every trained model shares
(`turn_detector.models.classified`), with P_text read once per pause: fire at segment end + 200 ms
(the assumed ASR finalisation lag) if P_text >= the threshold, otherwise at segment end + the
backstop, either only if the user is still silent. The threshold and the backstop are the two
knobs, tuned together; with the threshold above every P_text, the model is the baseline with the
backstop as its timeout.

The classifier is a frozen sentence encoder (`all-MiniLM-L6-v2`) with a logistic-regression head
(ADR 0002), trained on one example per labelled pause: segment ends that are a gold EOT or the
start of a gold mid-turn pause. Segment ends that are neither (disputed) are not trained on.

Transcripts stand in for a streaming ASR's output. A segment's text is readable only once the
segment has ended, so the context at a segment end holds only segments that ended by then; the
ASR lag is why the confident firing waits 200 ms. Bracketed annotation tags such as "[laughs]"
or "[unintelligible]" are stripped, since an ASR never outputs them; fillers such as "um", "uh"
and "hm" are kept, since an ASR usually does, including when an annotator bracketed them.
"""

import json
import re
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np

from turn_detector.data import ARTIFACTS_DIR
from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import EPSILON_S, SpeakerSide
from turn_detector.models.classified import (
    Fitted,
    LogisticHead,
    Request,
    ScoredPause,
    Serving,
    check_encoder,
    firings,
    gold_pause_labels,
    head_from_json,
    head_json,
    pauses,
    save_json,
    train_head,
)
from turn_detector.models.classified import fit as fit_classified

NAME = "text-only"
ASR_LAG_S = 0.2
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64
ARTIFACT_PATH = ARTIFACTS_DIR / NAME / "text-only.json"
# Answers any request, reading only the text; its P(EOT) may fire only once the ASR's text has arrived.
SERVING = Serving(NAME, needs_audio=False, needs_transcript=False, earliest_firing_s=ASR_LAG_S)

FILLERS = {"um", "uh", "hm", "hmm", "mm", "mhm", "mm-hmm", "uh-huh", "er", "ah"}
TAG = re.compile(r"\[([^\]]*)\]")


def asr_text(transcript: str) -> str:
    """A transcript as a streaming ASR would output it: annotation tags dropped, fillers kept."""

    def replace(tag: re.Match[str]) -> str:
        return tag[1] if tag[1].lower() in FILLERS else ""

    return " ".join(TAG.sub(replace, transcript).split())


@dataclass(frozen=True)
class TextContext:
    """What the classifier reads at a segment end, each part already as ASR text."""

    previous_turn: str
    turn_so_far: str


def text_context(side: SpeakerSide, end: float) -> TextContext:
    """The context at the user's segment end `end`, from the segments that have ended by then.

    The user's turn so far is their run of segments since the other speaker last finished a
    segment; the other speaker's previous turn is the run of theirs just before it.
    """
    ended = sorted(
        [(segment.end, True, segment.text) for segment in side.segments if segment.end <= end]
        + [(segment.end, False, segment.text) for segment in side.other_turns if segment.end <= end],
        key=lambda segment: (segment[0], segment[1]),  # on equal ends, the user's segment comes last
    )
    runs: list[tuple[bool, list[str]]] = []
    for _, is_user, text in ended:
        if not runs or runs[-1][0] != is_user:
            runs.append((is_user, []))
        runs[-1][1].append(text)

    def joined(texts: list[str]) -> str:
        return " ".join(text for text in map(asr_text, texts) if text)

    turn_so_far = joined(runs.pop()[1]) if runs and runs[-1][0] else ""
    previous_turn = joined(runs[-1][1]) if runs else ""
    return TextContext(previous_turn, turn_so_far)


def text_read_at(side: SpeakerSide, times: Sequence[float]) -> list[TextContext]:
    """At each time, what the ASR has delivered by then: the context at the user's latest segment
    end at least ASR_LAG_S before it, or the empty context before the first. It changes only at
    segment ends, 200 ms late."""
    ends = sorted({segment.end for segment in side.segments})
    at_end: dict[float, TextContext] = {}
    contexts = []
    for t in times:
        i = bisect_right(ends, t - ASR_LAG_S + EPSILON_S)
        if i == 0:
            contexts.append(TextContext("", ""))
            continue
        if ends[i - 1] not in at_end:
            at_end[ends[i - 1]] = text_context(side, ends[i - 1])
        contexts.append(at_end[ends[i - 1]])
    return contexts


def classifier_input(context: TextContext) -> str:
    """The one string the encoder reads; its tokenizer keeps the end if it is too long."""
    return f"{context.previous_turn} [SEP] {context.turn_so_far}"


class Encoder(Protocol):
    """Turns strings into fixed-size embeddings, one row per string."""

    @property
    def name(self) -> str: ...

    def __call__(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class SentenceEncoder:
    """A frozen sentence-transformers encoder reading at most `max_tokens` tokens, the last ones.

    Embeddings are remembered, so re-reading the same text (every threshold of a sweep) is free;
    a served encoder (`remember=False`) remembers nothing, since its texts are endless.
    """

    name: str = ENCODER
    max_tokens: int = MAX_TOKENS
    remember: bool = True
    cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @cached_property
    def transformer(self):
        from sentence_transformers import SentenceTransformer

        transformer = SentenceTransformer(self.name, device="cpu")
        transformer.max_seq_length = self.max_tokens
        transformer.tokenizer.truncation_side = "left"  # the end of the user's turn matters most
        return transformer

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if not self.remember:
            return self.transformer.encode(list(texts), batch_size=64, convert_to_numpy=True)
        new = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if new:
            self.cache.update(zip(new, self.transformer.encode(new, batch_size=64, convert_to_numpy=True)))
        return np.stack([self.cache[text] for text in texts])


def train_text_head(embeddings: np.ndarray, labels: np.ndarray) -> LogisticHead:
    """Fit the head: plain L2-regularised logistic regression, labels True for EOT."""
    return train_head(embeddings, labels)


class Classifier(Protocol):
    """Contexts in, P_text(EOT) out, one per context."""

    def p_eot(self, contexts: Sequence[TextContext]) -> list[float]: ...


@dataclass(frozen=True)
class TextClassifier:
    """The encoder and head together: contexts in, P_text(EOT) out."""

    encoder: Encoder
    head: LogisticHead

    def p_eot(self, contexts: Sequence[TextContext]) -> list[float]:
        if not contexts:
            return []
        return self.head(self.encoder([classifier_input(context) for context in contexts])).tolist()


def scored_pauses(side: SpeakerSide, classifier: Classifier) -> list[ScoredPause]:
    """Every pause of the user's, in time order, scored by `classifier` once, when the segment's
    text arrives 200 ms into it."""
    found = pauses(side)
    p_eot = classifier.p_eot([text_context(side, pause.end) for pause in found])
    return [ScoredPause(pause.end, pause.silent_until, [(pause.end + ASR_LAG_S, p)]) for pause, p in zip(found, p_eot)]


def check_backstop(threshold: float, backstop_s: float) -> None:
    # A shorter backstop would fire on unconfident ends before confident ones, and before the
    # words are read.
    if backstop_s < ASR_LAG_S:
        raise ValueError(f"the backstop ({backstop_s} s) can't be shorter than the ASR lag ({ASR_LAG_S} s)")


@dataclass(frozen=True)
class TextOnly:
    """Fire 200 ms after a segment end if P_text >= `threshold`, else `backstop_s` after it, if
    silent, up to 3 s into the pause."""

    classifier: Classifier
    threshold: float
    backstop_s: float
    name: str = NAME

    def __post_init__(self) -> None:
        check_backstop(self.threshold, self.backstop_s)

    def fire(self, side: SpeakerSide) -> list[float]:
        return firings(scored_pauses(side, self.classifier), self.threshold, self.backstop_s)

    def predict(self, request: Request) -> float:
        [p_eot] = self.classifier.p_eot([TextContext(request.previous_turn or "", request.transcript or "")])
        return float(p_eot)


def classifier_json(classifier: TextClassifier) -> dict:
    """The head and the encoder's name; the encoder itself is downloaded."""
    return {
        "encoder": classifier.encoder.name,
        "max_tokens": getattr(classifier.encoder, "max_tokens", MAX_TOKENS),
        **head_json(classifier.head),
    }


def classifier_from_json(stored: dict, encoder: Encoder | None = None) -> TextClassifier:
    """The saved classifier, with its encoder loaded (or `encoder`, which must be the one it was
    trained on). It is loaded for serving, so the encoder remembers nothing."""
    encoder = check_encoder(stored, encoder or SentenceEncoder(stored["encoder"], stored["max_tokens"], remember=False))
    return TextClassifier(encoder, head_from_json(stored))


def save_text_only(model: TextOnly, path: Path = ARTIFACT_PATH) -> Path:
    """Save the head, its threshold and backstop, and the encoder's name."""
    if not isinstance(model.classifier, TextClassifier):
        raise TypeError("only a model with a trained TextClassifier can be saved")
    return save_json({**classifier_json(model.classifier), "threshold": model.threshold, "backstop_s": model.backstop_s}, path)


def load_text_only(path: Path = ARTIFACT_PATH, encoder: Encoder | None = None) -> TextOnly:
    """The saved model, with its encoder loaded (or `encoder`, which must be the one it was trained on)."""
    stored = json.loads(path.read_text())
    return TextOnly(classifier_from_json(stored, encoder), stored["threshold"], stored["backstop_s"])


def labelled_pauses(conversation: EvaluationConversation) -> list[tuple[TextContext, bool]]:
    """One example per segment end that is a gold EOT (True) or starts a mid-turn pause (False)."""
    examples = []
    for side in conversation.sides:
        is_eot = gold_pause_labels(conversation, side)
        examples.extend((text_context(side, end), is_eot[end]) for end in sorted(is_eot))
    return examples


@dataclass
class TextTraining:
    """Trains the text classifier on some development conversations, one example per labelled
    pause; each set of conversations is trained on once."""

    conversations: Sequence[EvaluationConversation]
    encoder: SentenceEncoder
    trained: dict[frozenset[str], TextClassifier] = field(default_factory=dict, repr=False)

    @cached_property
    def examples(self) -> dict[str, list[tuple[TextContext, bool]]]:
        return {c.info.conversation_id: labelled_pauses(c) for c in self.conversations}

    def classifier(self, conversation_ids: Sequence[str]) -> TextClassifier:
        key = frozenset(conversation_ids)
        if key not in self.trained:
            pauses = [pause for conversation_id in conversation_ids for pause in self.examples[conversation_id]]
            embeddings = self.encoder([classifier_input(context) for context, _ in pauses])
            head = train_text_head(embeddings, np.array([is_eot for _, is_eot in pauses]))
            self.trained[key] = TextClassifier(self.encoder, head)
        return self.trained[key]

    def scored_pauses(self, side: SpeakerSide, classifier: TextClassifier) -> list[ScoredPause]:
        return scored_pauses(side, classifier)


def fit(conversations: Sequence[EvaluationConversation]) -> Fitted:
    """Train one head per speaker group, on every other group, and a final head on every conversation."""

    def save(classifier: TextClassifier, threshold: float, backstop_s: float) -> Path:
        return save_text_only(TextOnly(classifier, threshold, backstop_s))

    return fit_classified(NAME, TextTraining(conversations, SentenceEncoder()), conversations, save, check_backstop)
