"""The models, and how the evaluation runs each one.

Each model lives in its own module here. MODELS registers it for `scripts/evaluate.py`: its knob,
the values to sweep, and its colour in the comparison figures. Everything model-specific is in
this package; the rest of `turn_detector` is shared by every model.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from turn_detector.model import Model
from turn_detector.models.silence_timeout import SilenceTimeout


@dataclass(frozen=True)
class RegisteredModel:
    """A model as the evaluation runs it: `build(value)` is the model at that knob setting."""

    name: str
    knob_name: str
    knob_values: Sequence[float]
    build: Callable[[float], Model]
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
            name="silence timeout",
            knob_name="N (ms)",
            knob_values=[float(n) for n in range(0, 3001, 50)],
            build=lambda n: SilenceTimeout(timeout_ms=n),
            colour="#2a78d6",
        ),
    ]
}
