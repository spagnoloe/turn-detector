"""The evaluation harness every system goes through: knob sweep, scoring and cross-validation.

Scores come from TurnBench itself. Each system's firings are validated as a TurnBench
submission (strictly increasing, finite, within the audio) and scored with its `score_task`
against its gold EOTs and mid-turn pauses, exactly as `turnbench.score` does: a hit is the first
firing in [EOT - 0.25 s, EOT + 3 s], any firing in a mid-turn pause is one false cut-in, and
firings in disputed regions are ignored. Only the EOT task is scored.

A system has one knob (the baseline's timeout, a model's threshold). Each conversation is scored
once per knob value; a sweep, a cross-validation fold or a held-out evaluation is then just a
sum over a subset of conversations.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from turnbench.durations import load_durations
from turnbench.gold import ConversationEvents, events_for_conversation
from turnbench.score import TaskScore, merge, score_task
from turnbench.submission import ConversationPrediction, SpeakerEvents, validate_event_times

from turn_detector.data import ConversationAnnotations, iter_annotations
from turn_detector.detector import DetectorSystem, SpeakerSide
from turn_detector.events import turnbench_conversation
from turn_detector.split import ConversationInfo, speaker_groups
from turn_detector.timeline import speaker_sides

# The operating-point budget: the highest false-cut-in rate a system may run at.
MAX_FALSE_CUT_IN_RATE = 0.10


@dataclass(frozen=True)
class EvaluationConversation:
    """A conversation ready to evaluate: both speaker sides and TurnBench's gold events."""

    info: ConversationInfo
    duration_s: float
    sides: list[SpeakerSide]
    gold: ConversationEvents


def evaluation_conversation(conversation: ConversationAnnotations, duration_s: float) -> EvaluationConversation:
    gold = events_for_conversation(
        turnbench_conversation(conversation.annotations, conversation.conversation_id, duration_s)
    )
    return EvaluationConversation(
        info=ConversationInfo(
            conversation.conversation_id,
            (conversation.speaker_ids[1], conversation.speaker_ids[2]),
            conversation.conversation_type,
        ),
        duration_s=duration_s,
        sides=speaker_sides(conversation.conversation_id, duration_s, conversation.annotations),
        gold=gold,
    )


@dataclass(frozen=True)
class SystemSweep:
    """A system family over its knob: `build(value)` is the system at that knob setting."""

    name: str
    knob_name: str
    values: Sequence[float]
    build: Callable[[float], DetectorSystem]


@dataclass(frozen=True)
class Scores:
    """Aggregate EOT scores at one knob setting. Latencies are detection latencies in ms."""

    knob: float
    recall: float
    false_cut_in_rate: float
    detection_latency_p10_ms: float
    detection_latency_p50_ms: float
    detection_latency_p90_ms: float
    tp: int
    fn: int
    fp: int
    tn: int

    @staticmethod
    def of(knob: float, score: TaskScore) -> "Scores":
        detection_latency = score.latency()
        return Scores(
            knob, score.recall, score.fp_rate, detection_latency.p10, detection_latency.p50, detection_latency.p90,
            score.tp, score.fn, score.fp, score.tn,
        )  # fmt: skip


def score_conversation(system: DetectorSystem, conversation: EvaluationConversation) -> TaskScore:
    """Score one conversation's firings, both speakers, with TurnBench's validation and scorer."""
    firings = {side.speaker: system.fire(side) for side in conversation.sides}
    prediction = ConversationPrediction(
        conversation_id=conversation.info.conversation_id,
        speaker_1=SpeakerEvents(eot=firings[1], interruption=[]),
        speaker_2=SpeakerEvents(eot=firings[2], interruption=[]),
    )
    validate_event_times(prediction, conversation.duration_s)
    gold = conversation.gold
    return score_task(gold.eot_positive_events, gold.eot_negative_spans, firings, gold.eot_excluded)


@dataclass(frozen=True)
class KnobScores:
    """Per-conversation scores of one system family at every knob setting."""

    sweep: SystemSweep
    by_knob: dict[float, dict[str, TaskScore]]

    def total(self, knob: float, conversation_ids: Iterable[str]) -> TaskScore:
        total = TaskScore()
        for conversation_id in conversation_ids:
            merge(total, self.by_knob[knob][conversation_id])
        return total

    def curve(self, conversation_ids: Sequence[str]) -> list[Scores]:
        """The sweep over these conversations: scores at every knob setting, in knob order."""
        return [Scores.of(knob, self.total(knob, conversation_ids)) for knob in self.sweep.values]


def score_knobs(sweep: SystemSweep, conversations: Sequence[EvaluationConversation]) -> KnobScores:
    by_knob = {}
    for value in sweep.values:
        system = sweep.build(value)
        by_knob[value] = {c.info.conversation_id: score_conversation(system, c) for c in conversations}
    return KnobScores(sweep, by_knob)


def operating_point(curve: Sequence[Scores], max_false_cut_in_rate: float = MAX_FALSE_CUT_IN_RATE) -> Scores:
    """The highest-recall setting within the false-cut-in budget, lower median latency on ties.

    TurnBench's own operating-point rule (`turnbench.sweep.operating_point`), with a tie-break.
    """
    allowed = [s for s in curve if s.false_cut_in_rate <= max_false_cut_in_rate]
    if not allowed:
        raise ValueError(f"no knob setting reaches a false-cut-in rate of {max_false_cut_in_rate} or less")
    return max(allowed, key=lambda s: (s.recall, -s.detection_latency_p50_ms))


@dataclass(frozen=True)
class Fold:
    held_out: list[str]
    knob: float


@dataclass(frozen=True)
class CrossValidation:
    """Knob selection by cross-validation, one speaker group held out per fold.

    `scores` pools every fold's held-out conversations, each scored at the knob chosen without
    them; its `knob` field is `knob`, the setting chosen on all the conversations.
    """

    folds: list[Fold]
    knob: float
    scores: Scores


def cross_validate(knob_scores: KnobScores, conversations: Sequence[ConversationInfo]) -> CrossValidation:
    """Leave one speaker group out (ADR 0003): a fold holds out every conversation of a set of actors."""
    all_ids = [c.conversation_id for c in conversations]
    folds, pooled = [], TaskScore()
    for group in speaker_groups(conversations):
        held_out = [c.conversation_id for c in group]
        training = [i for i in all_ids if i not in held_out]
        knob = operating_point(knob_scores.curve(training)).knob
        folds.append(Fold(held_out, knob))
        merge(pooled, knob_scores.total(knob, held_out))
    knob = operating_point(knob_scores.curve(all_ids)).knob
    return CrossValidation(folds, knob, Scores.of(knob, pooled))


def load_conversations(conversation_ids: Iterable[str]) -> list[EvaluationConversation]:
    """These dev-set conversations, ready to evaluate, in numeric id order."""
    wanted = set(conversation_ids)
    durations = load_durations("dev")
    return [
        evaluation_conversation(c, durations[c.conversation_id])
        for c in iter_annotations()
        if c.conversation_id in wanted
    ]
