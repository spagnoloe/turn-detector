"""The models, and how the evaluation runs each one.

Each model lives in its own module here. MODELS registers it for `scripts/evaluate.py`: how to fit
it to the development conversations, its knobs, the settings to sweep, and its colour in the
comparison figures. What the trained models share (their firing rule, cross-fitting, saving and
how the API serves them) is in `classified`, and each trained model's `SERVING` describes which
requests it answers. Everything model-specific is in this package; the rest of `turn_detector` is
shared by every model.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import Fitted, Model, Setting
from turn_detector.models import audio_only, combined, text_only
from turn_detector.models.baseline import Baseline


@dataclass(frozen=True)
class Rule:
    """A model with nothing to train: fitting it changes nothing, and saving it writes nothing."""

    model_at: Callable[[Setting], Model]

    def build(self, setting: Setting) -> Model:
        return self.model_at(setting)

    def save(self, setting: Setting) -> list[Path]:
        return []


@dataclass(frozen=True)
class RegisteredModel:
    """A model as the evaluation runs it: `fit(conversations)` trains it on those conversations,
    and the result builds the model at any setting: one value per knob, in `knob_names` order."""

    name: str
    knob_names: tuple[str, ...]
    settings: Sequence[Setting]
    fit: Callable[[Sequence[EvaluationConversation]], Fitted]
    colour: str

    def describe(self, setting: Setting) -> str:
        """The setting with its knobs named, for figures and logs."""
        return ", ".join(f"{name} = {value:g}" for name, value in zip(self.knob_names, setting, strict=True))

    @property
    def slug(self) -> str:
        """The name used on the command line and for the model's results folder."""
        return self.name.replace(" ", "-")


# In the order the models appear in the comparison table and figures; colours are fixed per model.
MODELS = {
    model.slug: model
    for model in [
        RegisteredModel(
            name="baseline",
            knob_names=("silence timeout N (ms)",),
            settings=[(float(n),) for n in range(0, 3001, 50)],
            fit=lambda conversations: Rule(lambda setting: Baseline(timeout_ms=setting[0])),
            colour="#2a78d6",
        ),
        RegisteredModel(
            name=text_only.NAME,
            knob_names=("P_text threshold", "backstop (ms)"),
            # The backstop starts at the 200 ms ASR lag; a threshold of 1 never fires early.
            settings=[(i / 100, float(ms)) for i in range(101) for ms in range(200, 3001, 50)],
            fit=text_only.fit,
            colour="#e07b39",
        ),
        RegisteredModel(
            name=audio_only.NAME,
            knob_names=("P_audio threshold", "backstop (ms)"),
            # The threshold is finer than the text-only model's: the strongly regularised head's
            # P_audio is bunched near its threshold, where a step of 0.01 moves the false-cut-in
            # rate by several points. A threshold of 1 never fires early.
            settings=[(i / 1000, float(ms)) for i in range(1001) for ms in range(200, 3001, 50)],
            fit=audio_only.fit,
            colour="#3a9e6a",
        ),
        RegisteredModel(
            name=combined.NAME,
            knob_names=("P_combined threshold", "backstop (ms)"),
            # The audio-only model's settings, so the two are compared at the same firing rule.
            settings=[(i / 1000, float(ms)) for i in range(1001) for ms in range(200, 3001, 50)],
            fit=combined.fit,
            colour="#8a5cc2",
        ),
    ]
}
