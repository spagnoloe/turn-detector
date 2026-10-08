"""The held-out evaluation: each final model scored once on the held-out conversations, at the
setting chosen in cross-validation, and reported next to the published references."""

import numpy as np
import pytest

from turn_detector.data import ConversationAnnotations
from turn_detector.evaluation import CrossValidation, Fold, Scores, evaluation_conversation, score_held_out
from turn_detector.model import Audio, SpeakerSide
from turn_detector.models import MODELS, RegisteredModel, Rule
from turn_detector.models.baseline import Baseline
from turn_detector.models.classified import check_saved_setting
from turn_detector.report import PUBLISHED_REFERENCES, ModelResults, load, save, save_held_out, write_table

TURN = "Normal Turn"


def conversation():
    """Speaker 1 pauses for 0.5 s at 2.0 (a mid-turn pause) and ends their turn at 4.0; speaker 2
    answers at 5.0 and ends theirs at 6.0."""
    turns = {1: [(0.0, 2.0), (2.5, 4.0)], 2: [(5.0, 6.0)]}
    annotations = {
        (speaker, annotator): [(start, end, TURN, "words.") for start, end in turns[speaker]]
        for speaker in (1, 2)
        for annotator in "abc"
    }
    return evaluation_conversation(ConversationAnnotations("1", "Casual", {1: "a", 2: "b"}, annotations), 10.0)


def silent_audio(conversation_id: str, speaker: int) -> Audio:
    return Audio(np.zeros(10 * 16_000, np.float32), 16_000)


class NeedsAudio:
    """The baseline, but refusing a side whose audio wasn't loaded."""

    name = "needs audio"

    def fire(self, side: SpeakerSide) -> list[float]:
        assert side.audio is not None
        return Baseline(timeout_ms=600).fire(side)


def test_held_out_scores_count_every_conversation_once_at_the_setting():
    eager = score_held_out(Baseline(timeout_ms=300), (300.0,), [conversation()], silent_audio)
    patient = score_held_out(Baseline(timeout_ms=600), (600.0,), [conversation()], silent_audio)

    assert (eager.setting, eager.tp, eager.fn, eager.fp, eager.tn) == ((300.0,), 2, 0, 1, 0)
    assert (patient.setting, patient.tp, patient.fp, patient.tn) == ((600.0,), 2, 0, 1)
    assert patient.detection_latency_p50_ms == pytest.approx(600.0)


def test_held_out_sides_carry_their_audio():
    assert score_held_out(NeedsAudio(), (600.0,), [conversation()], silent_audio).tp == 2


def test_the_final_baseline_is_the_timeout_chosen():
    assert MODELS["baseline"].final((1150.0,)) == Baseline(timeout_ms=1150.0)


def test_a_saved_model_must_be_at_the_setting_chosen():
    check_saved_setting("text-only", 0.9, 1.15, (0.9, 1150.0))
    with pytest.raises(ValueError, match="text-only"):
        check_saved_setting("text-only", 0.9, 1.15, (0.86, 1150.0))


def scores(setting: tuple[float, ...], recall: float, false_cut_in_rate: float) -> Scores:
    return Scores(setting, recall, false_cut_in_rate, 200.0, 1500.0, 1500.0, 9, 1, 2, 8)


def results(tmp_path) -> ModelResults:
    model = RegisteredModel(
        name="timeout",
        knob_names=("N (ms)",),
        settings=[(1000.0,)],
        fit=lambda conversations: Rule(lambda setting: Baseline(timeout_ms=setting[0])),
        final=lambda setting: Baseline(timeout_ms=setting[0]),
        colour="#2a78d6",
    )
    cross_validation = CrossValidation([Fold(["1"], (1000.0,))], (1000.0,), scores((1000.0,), 0.8, 0.06))
    found = ModelResults(model, [scores((1000.0,), 0.8, 0.05)], cross_validation)
    save(found, tmp_path)
    return found


def test_held_out_scores_load_back_with_the_model(tmp_path):
    found = results(tmp_path)
    held_out = scores((1000.0,), 0.75, 0.08)
    save_held_out(found.model, held_out, ["31", "32"], tmp_path)

    assert load(found.model, tmp_path).held_out == held_out
    assert load(found.model, tmp_path).cross_validation == found.cross_validation


def test_held_out_scores_are_written_once(tmp_path):
    found = results(tmp_path)
    save_held_out(found.model, scores((1000.0,), 0.75, 0.08), ["31"], tmp_path)
    with pytest.raises(FileExistsError):
        save_held_out(found.model, scores((1000.0,), 0.9, 0.01), ["31"], tmp_path)


def test_the_table_leads_with_held_out_scores_and_the_published_references(tmp_path):
    found = results(tmp_path)
    save_held_out(found.model, scores((1000.0,), 0.75, 0.08), ["31"], tmp_path)
    path = tmp_path / "results.md"
    write_table([load(found.model, tmp_path)], path)
    table = path.read_text()

    held_out, development = table.index("| timeout | N (ms) | 1000 | 0.750"), table.index("| timeout | N (ms) | 1000 (1000–1000) | 0.800")
    assert held_out < development
    for reference in PUBLISHED_REFERENCES:
        assert f"| {reference.name} " in table
        assert table.index(f"| {reference.name} ") < development


def test_without_held_out_scores_the_table_is_the_cross_validated_one(tmp_path):
    found = results(tmp_path)
    path = tmp_path / "results.md"
    write_table([found], path)

    assert "held-out" not in path.read_text().lower()
