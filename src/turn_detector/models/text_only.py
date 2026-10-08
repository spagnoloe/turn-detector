"""The text-only model: at each pause, do the user's words so far sound finished?

At every segment end the classifier reads the other speaker's previous turn and the user's turn
so far, and outputs P_text(EOT). The firing rule: fire at segment end + 200 ms (the assumed ASR
finalisation lag) if P_text >= the threshold, otherwise at segment end + the backstop, either only
if the user is still silent. The threshold and the backstop are the two knobs, tuned together; with
the threshold above every P_text, the model is the baseline with the backstop as its timeout.

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
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np

from turn_detector.data import ARTIFACTS_DIR
from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import Setting, SpeakerSide, next_speech_start
from turn_detector.split import speaker_groups

NAME = "text-only"
ASR_LAG_S = 0.2
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 64
ARTIFACT_PATH = ARTIFACTS_DIR / NAME / "text-only.json"

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


@dataclass(frozen=True)
class LogisticHead:
    """P(EOT) = sigmoid(embedding · weights + bias)."""

    weights: np.ndarray
    bias: float

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-(embeddings @ self.weights + self.bias)))


def train_head(embeddings: np.ndarray, labels: np.ndarray) -> LogisticHead:
    """Fit the head: plain L2-regularised logistic regression, labels True for EOT."""
    from sklearn.linear_model import LogisticRegression

    regression = LogisticRegression(max_iter=1000).fit(embeddings, labels)
    return LogisticHead(regression.coef_[0].copy(), float(regression.intercept_[0]))


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


@dataclass(frozen=True)
class ScoredPause:
    """A segment end with the classifier's P_text there, and when the user is next heard."""

    end: float
    p_eot: float
    next_speech_start: float


def scored_pauses(side: SpeakerSide, classifier: Classifier) -> list[ScoredPause]:
    """Every distinct segment end of the user's, in time order, scored by `classifier`."""
    ends = sorted({segment.end for segment in side.segments})
    p_eot = classifier.p_eot([text_context(side, end) for end in ends])
    return [ScoredPause(end, p, next_speech_start(side, end)) for end, p in zip(ends, p_eot)]


def firings(pauses: Sequence[ScoredPause], threshold: float, backstop_s: float, duration_s: float) -> list[float]:
    """The firing rule: 200 ms after a confident end, `backstop_s` after any other, if still silent."""
    firings = []
    for pause in pauses:
        firing = pause.end + (ASR_LAG_S if pause.p_eot >= threshold else backstop_s)
        if firing <= min(pause.next_speech_start, duration_s):
            firings.append(firing)
    return firings


def check_backstop(backstop_s: float) -> None:
    # A shorter backstop would fire on unconfident ends before confident ones, and before the
    # words are read.
    if backstop_s < ASR_LAG_S:
        raise ValueError(f"the backstop ({backstop_s} s) can't be shorter than the ASR lag ({ASR_LAG_S} s)")


@dataclass(frozen=True)
class TextOnly:
    """Fire 200 ms after a segment end if P_text >= `threshold`, else `backstop_s` after it, if silent."""

    classifier: Classifier
    threshold: float
    backstop_s: float
    name: str = NAME

    def __post_init__(self) -> None:
        check_backstop(self.backstop_s)

    def fire(self, side: SpeakerSide) -> list[float]:
        return firings(scored_pauses(side, self.classifier), self.threshold, self.backstop_s, side.duration_s)


def save_text_only(model: TextOnly, path: Path = ARTIFACT_PATH) -> Path:
    """Save the head, its threshold and backstop, and the encoder's name; the encoder itself is
    downloaded."""
    if not isinstance(model.classifier, TextClassifier):
        raise TypeError("only a model with a trained TextClassifier can be saved")
    classifier = model.classifier
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = {
        "encoder": classifier.encoder.name,
        "max_tokens": getattr(classifier.encoder, "max_tokens", MAX_TOKENS),
        "threshold": model.threshold,
        "backstop_s": model.backstop_s,
        "bias": classifier.head.bias,
        "weights": classifier.head.weights.tolist(),
    }
    path.write_text(json.dumps(stored, indent=2) + "\n")
    return path


def load_text_only(path: Path = ARTIFACT_PATH, encoder: Encoder | None = None) -> TextOnly:
    """The saved model, with its encoder loaded (or `encoder`, which must be the one it was trained on).
    It is loaded for serving, so the encoder remembers nothing."""
    stored = json.loads(path.read_text())
    encoder = encoder or SentenceEncoder(stored["encoder"], stored["max_tokens"], remember=False)
    if encoder.name != stored["encoder"]:
        raise ValueError(f"the head was trained on encoder {stored['encoder']!r}, not {encoder.name!r}")
    head = LogisticHead(np.array(stored["weights"]), stored["bias"])
    return TextOnly(TextClassifier(encoder, head), stored["threshold"], stored["backstop_s"])


def labelled_pauses(conversation: EvaluationConversation) -> list[tuple[TextContext, bool]]:
    """One example per segment end that is a gold EOT (True) or starts a mid-turn pause (False)."""
    gold = conversation.gold
    examples = []
    for side in conversation.sides:
        eots = {event.time_s for event in gold.eot_positive_events if event.speaker == side.speaker}
        mid_turn = {span.start for span in gold.eot_negative_spans if span.speaker == side.speaker}
        for end in sorted({segment.end for segment in side.segments} & (eots | mid_turn)):
            examples.append((text_context(side, end), end in eots))
    return examples


@dataclass(frozen=True)
class CrossFitted:
    """The text-only model as the evaluation scores it, at one setting.

    Each side's pauses were scored once, by the head that never saw that conversation's
    speakers, so a sweep over settings only re-applies the firing rule.
    """

    pauses: dict[tuple[str, int], list[ScoredPause]]
    threshold: float
    backstop_s: float
    name: str = NAME

    def __post_init__(self) -> None:
        check_backstop(self.backstop_s)

    def fire(self, side: SpeakerSide) -> list[float]:
        pauses = self.pauses[(side.conversation_id, side.speaker)]
        return firings(pauses, self.threshold, self.backstop_s, side.duration_s)


@dataclass(frozen=True)
class FittedTextOnly:
    """Every development side's pauses scored by cross-fitted heads, and a head trained on all
    conversations. A setting is (threshold, backstop in ms)."""

    pauses: dict[tuple[str, int], list[ScoredPause]]
    final_classifier: TextClassifier

    def build(self, setting: Setting) -> CrossFitted:
        threshold, backstop_ms = setting
        return CrossFitted(self.pauses, threshold, backstop_ms / 1000)

    def save(self, setting: Setting) -> list[Path]:
        threshold, backstop_ms = setting
        return [save_text_only(TextOnly(self.final_classifier, threshold, backstop_ms / 1000))]


def fit(conversations: Sequence[EvaluationConversation]) -> FittedTextOnly:
    """Train one head per speaker group, on every other group (the folds `cross_validate` uses,
    so a fold's held-out conversations are always scored by a head that never saw them), and a
    final head on every conversation."""
    encoder = SentenceEncoder()
    examples = {c.info.conversation_id: labelled_pauses(c) for c in conversations}

    def head_trained_on(conversation_ids: Sequence[str]) -> LogisticHead:
        pauses = [pause for conversation_id in conversation_ids for pause in examples[conversation_id]]
        embeddings = encoder([classifier_input(context) for context, _ in pauses])
        return train_head(embeddings, np.array([is_eot for _, is_eot in pauses]))

    all_ids = list(examples)
    pauses = {}
    for group in speaker_groups([c.info for c in conversations]):
        held_out = {c.conversation_id for c in group}
        classifier = TextClassifier(encoder, head_trained_on([i for i in all_ids if i not in held_out]))
        for conversation in (c for c in conversations if c.info.conversation_id in held_out):
            for side in conversation.sides:
                pauses[(conversation.info.conversation_id, side.speaker)] = scored_pauses(side, classifier)
    return FittedTextOnly(pauses, TextClassifier(encoder, head_trained_on(all_ids)))
