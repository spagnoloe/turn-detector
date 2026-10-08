"""The silence-timeout baseline: what plain voice-activity detection achieves."""

from dataclasses import dataclass

from turn_detector.detector import SpeakerSide


@dataclass(frozen=True)
class SilenceTimeout:
    """Fire `timeout_ms` after each segment end if the user hasn't resumed speaking by then."""

    timeout_ms: float
    name: str = "silence timeout"

    def fire(self, side: SpeakerSide) -> list[float]:
        firings = set()
        for segment in side.segments:
            firing = segment.end + self.timeout_ms / 1000
            # Silent throughout [end, firing): no other speech overlaps that stretch.
            resumed = any(
                other is not segment and other.start < firing and other.end > segment.end
                for other in side.segments
            )
            if not resumed and firing <= side.duration_s:
                firings.add(firing)
        return sorted(firings)
