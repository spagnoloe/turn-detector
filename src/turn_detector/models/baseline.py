"""The baseline: a silence timeout, which is what plain voice-activity detection achieves."""

from dataclasses import dataclass

from turn_detector.model import SpeakerSide, next_speech_start


@dataclass(frozen=True)
class Baseline:
    """Fire `timeout_ms` after each segment end if the user hasn't resumed speaking by then."""

    timeout_ms: float
    name: str = "baseline"

    def fire(self, side: SpeakerSide) -> list[float]:
        firings = set()
        for segment in side.segments:
            firing = segment.end + self.timeout_ms / 1000
            if firing <= min(next_speech_start(side, segment.end), side.duration_s):
                firings.add(firing)
        return sorted(firings)
