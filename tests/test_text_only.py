"""The text-only model through the model interface, with a stub classifier; its input preparation
and its saved head."""

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from turn_detector.model import Segment, SpeakerSide
from turn_detector.models.classified import LogisticHead
from turn_detector.models.text_only import (
    SentenceEncoder,
    TextContext,
    TextClassifier,
    TextOnly,
    asr_text,
    load_text_only,
    save_text_only,
    text_context,
)


@dataclass(frozen=True)
class Stub:
    """P(EOT) = 1 when the user's turn so far ends with a full stop, else 0."""

    def p_eot(self, contexts: Sequence[TextContext]) -> list[float]:
        return [1.0 if context.turn_so_far.endswith(".") else 0.0 for context in contexts]


def side(*segments: tuple[float, float, str], other: Sequence[tuple[float, float, str]] = (), duration_s=20.0):
    return SpeakerSide(
        conversation_id="synthetic",
        speaker=1,
        duration_s=duration_s,
        audio=None,
        segments=[Segment(*segment) for segment in segments],
        other_turns=[Segment(*segment) for segment in other],
    )


def test_fires_200_ms_after_the_segment_end_when_confident():
    assert TextOnly(Stub(), threshold=0.5, backstop_s=1.5).fire(side((1.0, 3.0, "I'd like to fly to Barcelona."))) == [3.2]


def test_fires_at_the_1_5_s_backstop_when_not_confident():
    assert TextOnly(Stub(), threshold=0.5, backstop_s=1.5).fire(side((1.0, 3.0, "I'd like to fly to"))) == [4.5]


def test_the_backstop_is_a_setting():
    assert TextOnly(Stub(), threshold=0.5, backstop_s=0.8).fire(side((1.0, 3.0, "I'd like to fly to"))) == [3.8]


def test_the_backstop_cannot_come_before_the_text_is_read():
    with pytest.raises(ValueError, match="backstop"):
        TextOnly(Stub(), threshold=0.5, backstop_s=0.1)


def test_does_not_fire_when_the_user_resumes_first():
    # Resumes 0.1 s after a confident end, and 1 s after an unconfident one.
    model = TextOnly(Stub(), threshold=0.5, backstop_s=1.5)
    assert model.fire(side((1.0, 3.0, "Yes."), (3.1, 5.0, "So"), (6.0, 7.0, "Friday."))) == [7.2]


def test_does_not_fire_after_the_conversation_ends():
    assert TextOnly(Stub(), threshold=0.5, backstop_s=1.5).fire(side((1.0, 3.0, "Well"), duration_s=4.0)) == []


def test_segments_ending_together_fire_once():
    model = TextOnly(Stub(), threshold=0.5, backstop_s=1.5)
    assert model.fire(side((1.0, 3.0, "To Barcelona."), (2.0, 3.0, "Barcelona."))) == [3.2]


def test_asr_text_strips_annotation_tags_and_keeps_fillers():
    assert asr_text("Um, I [laughs] tried, uh, [unintelligible] twice. [sighs]") == "Um, I tried, uh, twice."


def test_asr_text_unbrackets_annotated_fillers():
    assert asr_text("[um] so [mhm] [Uh] yes") == "um so mhm Uh yes"


def test_context_is_the_other_speakers_previous_turn_and_the_users_turn_so_far():
    user = side(
        (0.0, 1.0, "Hi."),
        (3.0, 4.0, "To Barcelona,"),
        (4.5, 5.0, "[sighs] on Friday."),
        (9.0, 10.0, "Thanks."),
        other=[(1.2, 2.0, "Where to?"), (2.1, 2.8, "[um] And when?"), (6.0, 8.0, "Done.")],
    )
    assert text_context(user, 5.0) == TextContext("Where to? um And when?", "To Barcelona, on Friday.")
    assert text_context(user, 4.0) == TextContext("Where to? um And when?", "To Barcelona,")
    assert text_context(user, 1.0) == TextContext("", "Hi.")


def test_context_reads_only_segments_that_have_ended():
    # The other speaker's turn ending after 5.0 isn't transcribed yet, so the previous turn is still
    # "Where to?", and the user's own later segment isn't there either.
    user = side((3.0, 5.0, "Barcelona."), (6.0, 7.0, "Friday."), other=[(1.0, 2.0, "Where to?"), (4.0, 5.5, "Mm.")])
    assert text_context(user, 5.0) == TextContext("Where to?", "Barcelona.")


class HashEncoder:
    """Deterministic pseudo-embeddings: each text seeds its own random vector."""

    name = "hash"

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([np.random.default_rng(zlib.crc32(text.encode())).standard_normal(8) for text in texts])


def test_saved_model_loads_back_and_predicts_the_same(tmp_path):
    rng = np.random.default_rng(0)
    head = LogisticHead(weights=rng.standard_normal(8), bias=0.3)
    model = TextOnly(TextClassifier(HashEncoder(), head), threshold=0.42, backstop_s=0.9)
    contexts = [TextContext("Where to?", "Barcelona."), TextContext("", "Hi"), TextContext("And when?", "Um,")]
    path = tmp_path / "text-only.json"
    save_text_only(model, path)
    loaded = load_text_only(path, encoder=HashEncoder())
    assert (loaded.threshold, loaded.backstop_s) == (0.42, 0.9)
    assert loaded.classifier.p_eot(contexts) == model.classifier.p_eot(contexts)


def test_loading_with_a_different_encoder_fails(tmp_path):
    path = tmp_path / "text-only.json"
    save_text_only(TextOnly(TextClassifier(HashEncoder(), LogisticHead(np.ones(8), 0.0)), threshold=0.5, backstop_s=1.5), path)
    other = HashEncoder()
    other.name = "another encoder"
    with pytest.raises(ValueError, match="encoder"):
        load_text_only(path, encoder=other)


class FakeTransformer:
    def encode(self, texts, **kwargs):
        return np.stack([np.full(4, len(text), dtype=float) for text in texts])


def encoder_with_fake_transformer(**kwargs) -> SentenceEncoder:
    encoder = SentenceEncoder(**kwargs)
    encoder.__dict__["transformer"] = FakeTransformer()  # fills the cached_property, so nothing is downloaded
    return encoder


def test_the_encoder_remembers_embeddings_by_default():
    encoder = encoder_with_fake_transformer()
    encoder(["Hi.", "Where to?"])
    assert set(encoder.cache) == {"Hi.", "Where to?"}


def test_a_served_encoder_remembers_nothing():
    # A server sees endless new transcripts; remembering them all would grow without bound.
    encoder = encoder_with_fake_transformer(remember=False)
    assert encoder(["Hi.", "Where to?"]).tolist() == [[3.0] * 4, [9.0] * 4]
    assert encoder.cache == {}
