"""The models, and how the evaluation runs each one.

Each model lives in its own module here. MODELS registers it for `scripts/evaluate.py`: how to fit
it to the development conversations, its knob, the values to sweep, and its colour in the
comparison figures. Everything model-specific is in
this package; the rest of `turn_detector` is shared by every model.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from turn_detector.evaluation import EvaluationConversation
from turn_detector.model import Fitted, Model
from turn_detector.models import text_only
from turn_detector.models.baseline import Baseline


@dataclass(frozen=True)
class Rule:
    """A model with nothing to train: fitting it changes nothing, and saving it writes nothing."""

    model_at: Callable[[float], Model]

    def build(self, knob: float) -> Model:
        return self.model_at(knob)

    def save(self, knob: float) -> list[Path]:
        return []


@dataclass(frozen=True)
class RegisteredModel:
    """A model as the evaluation runs it: `fit(conversations)` trains it on those conversations,
    and the result builds the model at any knob setting."""

    name: str
    knob_name: str
    knob_values: Sequence[float]
    fit: Callable[[Sequence[EvaluationConversation]], Fitted]
    colour: str

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
            knob_name="silence timeout N (ms)",
            knob_values=[float(n) for n in range(0, 3001, 50)],
            fit=lambda conversations: Rule(lambda n: Baseline(timeout_ms=n)),
            colour="#2a78d6",
        ),
        RegisteredModel(
            name=text_only.NAME,
            knob_name="P_text threshold",
            knob_values=[i / 100 for i in range(101)],
            fit=text_only.fit,
            colour="#e07b39",
        ),
    ]
}
