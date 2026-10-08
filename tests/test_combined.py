"""The combined model through the model interface, with stub classifiers; the text it reads at each
step, its fusion samples and its saved heads."""

import zlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from turn_detector.model import Audio, Segment, SpeakerSide
from turn_detector.models.audio_only import AudioClassifier
from turn_detector.models.classified import LogisticHead, Request
from turn_detector.models.combined import Combined, CombinedClassifier, load_combined, save_combined
from turn_detector.models.text_only import TextClassifier, TextContext

SAMPLE_RATE = 16_000
NO_BACKSTOP_S = 4.0  # beyond the 3 s horizon, so it never fires


def side(*segments: tuple[float, float, str], other: Sequence[tuple[float, float, str]] = (), duration_s=20.0):
    return SpeakerSide(
        conversation_id="synthetic",
        speaker=1,
        duration_s=duration_s,
        audio=Audio(np.zeros(round(duration_s * SAMPLE_RATE), np.float32), SAMPLE_RATE),
        segments=[Segment(*segment) for segment in segments],
        other_turns=[Segment(*segment) for segment in other],
    )


class FullStop:
    """P(EOT) = 1 when the user's turn so far ends with a full stop, else 0; ignores the audio."""

    def p_eot(self, windows: np.ndarray, contexts: Sequence[TextContext], silence_s: np.ndarray) -> np.ndarray:
        return np.array([1.0 if context.turn_so_far.endswith(".") else 0.0 for context in contexts])


class RisesWithSilence:
    """P(EOT) = the silence so far in seconds, capped at 1; ignores the audio and the text."""

    def p_eot(self, windows: np.ndarray, contexts: Sequence[TextContext], silence_s: np.ndarray) -> np.ndarray:
        return np.minimum(silence_s, 1.0)


@dataclass
class Recording:
    """P(EOT) = 0; remembers the context and silence it read at each step."""

    read: list[tuple[TextContext, float]]

    def p_eot(self, windows: np.ndarray, contexts: Sequence[TextContext], silence_s: np.ndarray) -> np.ndarray:
        assert len(windows) == len(contexts) == len(silence_s)
        self.read.extend(zip(contexts, silence_s.tolist()))
        return np.zeros(len(contexts))


def test_fires_at_the_first_50_ms_step_where_p_reaches_the_threshold():
    assert Combined(RisesWithSilence(), threshold=0.33, backstop_s=NO_BACKSTOP_S).fire(side((1.0, 3.0, "Hi"))) == pytest.approx([3.35])


def test_fires_at_the_backstop_when_p_never_reaches_the_threshold():
    model = Combined(RisesWithSilence(), threshold=0.95, backstop_s=0.5)
    assert model.fire(side((1.0, 3.0, "Hi"), (5.0, 6.0, "Bye"))) == pytest.approx([3.5, 6.5])


def test_does_not_fire_once_the_user_resumes_or_the_conversation_ends():
    model = Combined(RisesWithSilence(), threshold=0.42, backstop_s=NO_BACKSTOP_S)
    assert model.fire(side((1.0, 3.0, "Hi"), (3.3, 5.0, "So"), duration_s=5.4)) == []


def test_the_new_text_is_read_200_ms_into_the_pause():
    # "Barcelona." is readable 200 ms after its segment ends; until then the turn so far is "I'd like".
    model = Combined(FullStop(), threshold=0.5, backstop_s=NO_BACKSTOP_S)
    assert model.fire(side((1.0, 3.0, "I'd like"), (4.0, 6.0, "to fly to Barcelona."))) == pytest.approx([6.2])


def test_until_then_the_previous_text_is_held():
    # The first pause's text is still what the model reads at the start of the second.
    model = Combined(FullStop(), threshold=0.5, backstop_s=NO_BACKSTOP_S)
    assert model.fire(side((1.0, 3.0, "Hi."), (3.5, 6.0, "I'd like to"))) == pytest.approx([3.2, 6.0])


def test_reads_the_text_finalised_at_the_latest_segment_end_the_asr_has_delivered():
    classifier = Recording([])
    user = side(
        (1.0, 2.0, "To Barcelona,"),
        (2.5, 3.0, "on Friday."),
        other=[(0.0, 0.8, "Where to?"), (2.1, 2.4, "Mm."), (3.05, 3.4, "Great.")],
        duration_s=3.5,
    )
    Combined(classifier, threshold=0.5, backstop_s=NO_BACKSTOP_S).fire(user)
    # Pauses at 2.0-2.5 s and 3.0-3.5 s; the other speaker's turns ending in a pause aren't read.
    nothing, first, second = TextContext("", ""), TextContext("Where to?", "To Barcelona,"), TextContext("Mm.", "on Friday.")
    assert [context for context, _ in classifier.read] == [nothing] * 4 + [first] * 7 + [first] * 4 + [second] * 7
    assert [silence_s for _, silence_s in classifier.read] == pytest.approx([k * 0.05 for k in range(11)] * 2)


def test_predicts_from_one_request():
    request = Request(np.zeros((1, 16_000), np.float32), "Where to?", "Barcelona.", 0.25)
    assert Combined(FullStop(), threshold=0.5, backstop_s=1.0).predict(request) == 1.0


class HashEncoder:
    """Deterministic pseudo-embeddings: each text seeds its own random vector."""

    name = "hash"

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([np.random.default_rng(zlib.crc32(text.encode())).standard_normal(8) for text in texts])


class ChunkEncoder:
    """Pseudo-frames: each of 49 frames is the mean and spread of its stretch of the window."""

    name = "chunks"

    def __call__(self, windows: np.ndarray) -> np.ndarray:
        chunks = windows[:, : 49 * 320].reshape(len(windows), 49, 320)
        return np.stack([chunks.mean(axis=2), chunks.std(axis=2)], axis=2)


def trained_classifier(text_encoder=None, audio_encoder=None) -> CombinedClassifier:
    rng = np.random.default_rng(0)
    return CombinedClassifier(
        TextClassifier(text_encoder or HashEncoder(), LogisticHead(rng.standard_normal(8), 0.1)),
        AudioClassifier(audio_encoder or ChunkEncoder(), LogisticHead(rng.standard_normal(5), -0.2)),
        LogisticHead(np.array([2.0, 1.5, 0.8]), -1.7),
    )


def test_the_classifier_fuses_p_audio_p_text_and_the_silence():
    classifier = trained_classifier()
    windows = np.random.default_rng(1).standard_normal((2, 16_000)).astype(np.float32)
    contexts = [TextContext("Where to?", "Barcelona."), TextContext("", "Um,")]
    silence_s = np.array([0.1, 0.6])
    p_audio = classifier.audio.p_eot(windows, silence_s)
    p_text = np.array(classifier.text.p_eot(contexts))
    expected = 1 / (1 + np.exp(-(2.0 * p_audio + 1.5 * p_text + 0.8 * silence_s - 1.7)))
    np.testing.assert_allclose(classifier.p_eot(windows, contexts, silence_s), expected)


def test_saved_model_loads_back_and_predicts_the_same(tmp_path):
    model = Combined(trained_classifier(), threshold=0.42, backstop_s=0.9)
    windows = np.random.default_rng(1).standard_normal((3, 16_000)).astype(np.float32)
    contexts = [TextContext("Where to?", "Barcelona."), TextContext("", "Hi"), TextContext("And when?", "Um,")]
    silence_s = np.array([0.0, 0.3, 1.2])
    path = tmp_path / "combined.json"
    save_combined(model, path)
    loaded = load_combined(path, text_encoder=HashEncoder(), audio_encoder=ChunkEncoder())
    assert (loaded.threshold, loaded.backstop_s) == (0.42, 0.9)
    np.testing.assert_array_equal(loaded.classifier.p_eot(windows, contexts, silence_s), model.classifier.p_eot(windows, contexts, silence_s))


@pytest.mark.parametrize("which", ["text", "audio"])
def test_loading_with_a_different_encoder_fails(tmp_path, which):
    path = tmp_path / "combined.json"
    save_combined(Combined(trained_classifier(), threshold=0.5, backstop_s=1.5), path)
    text, audio = HashEncoder(), ChunkEncoder()
    (text if which == "text" else audio).name = "another encoder"
    with pytest.raises(ValueError, match="encoder"):
        load_combined(path, text_encoder=text, audio_encoder=audio)
