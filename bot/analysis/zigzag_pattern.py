"""Zigzag (ABC) correction pattern detector for Elliott Wave 2nd wave."""

import logging
from dataclasses import dataclass
from bot.analysis.fractals import Pivot

logger = logging.getLogger(__name__)


@dataclass
class ZigzagCorrection:
    """ABC zigzag — sharp, steep correction."""
    wave_a: Pivot
    wave_b: Pivot
    wave_c: Pivot
    direction: str      # "down" for bearish correction in uptrend
    confidence: float = 0.0

    @property
    def depth(self) -> float:
        """Total depth of the correction."""
        if self.direction == "down":
            return self.wave_a.price - self.wave_c.price
        else:
            return self.wave_c.price - self.wave_a.price

    @property
    def retracement_ratio(self) -> float:
        """How much wave B retraces wave A (should be 38.2-78.6%)."""
        a_move = abs(self.wave_b.price - self.wave_a.price)
        b_retrace = abs(self.wave_b.price - self.wave_c.price) if self.wave_c else 0
        return a_move / max(abs(self.wave_a.price - self.wave_b.price), 1e-10)


def detect_zigzag(pivots: list[Pivot], direction: str = "down") -> list[ZigzagCorrection]:
    """Detect ABC zigzag correction patterns.

    For bearish zigzag (wave 2 in uptrend):
    - A starts from a high, moves sharply down
    - B bounces up (retraces 38-78% of A)
    - C moves down past A's low

    Characteristics vs triangle:
    - Sharp, impulsive moves (not sideways)
    - 3 waves only (not 5 like triangle)
    - Wave C typically equals or exceeds wave A length
    """
    zigzags = []

    for i in range(len(pivots) - 2):
        a, b, c = pivots[i], pivots[i+1], pivots[i+2]

        if direction == "down":
            # A = high, B = low, C = high is wrong — we need A=high->low, B=low->high, C=high->low
            # Actually: A = the first drop endpoint (low), B = the bounce (high), C = final drop (low)
            # So pivot sequence: start_high, a_low, b_high, c_low
            # With 3 pivots: a=high (start), b=low (wave A end), c=high (wave B end)
            # But we need a 4th for wave C... let's use the approach:
            # a = starting high pivot, b = the next low pivot (end of wave A),
            # c (wave B end) = next high

            if i + 3 > len(pivots):
                continue

            start = a  # high
            wave_a_end = b  # low
            wave_b_end = pivots[i+2]  # high

            if not (start.pivot_type == "high" and wave_a_end.pivot_type == "low"
                    and wave_b_end.pivot_type == "high"):
                continue

            # Wave A is a drop
            wave_a_size = start.price - wave_a_end.price
            if wave_a_size <= 0:
                continue

            # Wave B retraces upward
            wave_b_retrace = wave_b_end.price - wave_a_end.price
            if wave_b_retrace <= 0:
                continue

            # B should not exceed start (otherwise not a correction)
            if wave_b_end.price >= start.price:
                continue

            # B retracement ratio (of wave A)
            b_ratio = wave_b_retrace / wave_a_size

            # Check for wave C (if available)
            wave_c_end = None
            if i + 3 < len(pivots):
                candidate_c = pivots[i+3]
                if candidate_c.pivot_type == "low":
                    wave_c_end = candidate_c

            confidence = 0.4

            # Ideal B retracement: 38.2-78.6%
            if 0.382 <= b_ratio <= 0.786:
                confidence += 0.2
            elif 0.236 <= b_ratio <= 0.886:
                confidence += 0.1

            # Sharp move characteristic
            time_a = wave_a_end.index - start.index
            time_b = wave_b_end.index - wave_a_end.index
            if time_b > time_a:  # B takes longer (typical for corrective bounce)
                confidence += 0.05

            if wave_c_end:
                wave_c_size = wave_b_end.price - wave_c_end.price
                # Wave C often equals wave A (or 1.618x)
                c_to_a_ratio = wave_c_size / wave_a_size if wave_a_size > 0 else 0
                if 0.8 <= c_to_a_ratio <= 1.2:
                    confidence += 0.2  # C ≈ A
                elif 1.5 <= c_to_a_ratio <= 1.7:
                    confidence += 0.15  # C ≈ 1.618 * A

                # C must go below A (new low for bearish)
                if wave_c_end.price < wave_a_end.price:
                    confidence += 0.1

                zigzags.append(ZigzagCorrection(
                    wave_a=start,
                    wave_b=wave_b_end,
                    wave_c=wave_c_end,
                    direction=direction,
                    confidence=min(confidence, 1.0),
                ))
            else:
                # Partial — wave C not yet complete
                zigzags.append(ZigzagCorrection(
                    wave_a=start,
                    wave_b=wave_b_end,
                    wave_c=wave_a_end,  # placeholder
                    direction=direction,
                    confidence=min(confidence * 0.7, 1.0),
                ))

        else:  # direction == "up" (bearish impulse correction going up)
            if i + 3 > len(pivots):
                continue
            start = a
            wave_a_end = b
            wave_b_end = pivots[i+2]

            if not (start.pivot_type == "low" and wave_a_end.pivot_type == "high"
                    and wave_b_end.pivot_type == "low"):
                continue

            wave_a_size = wave_a_end.price - start.price
            if wave_a_size <= 0:
                continue

            wave_b_retrace = wave_a_end.price - wave_b_end.price
            if wave_b_retrace <= 0:
                continue

            if wave_b_end.price <= start.price:
                continue

            b_ratio = wave_b_retrace / wave_a_size
            confidence = 0.4

            if 0.382 <= b_ratio <= 0.786:
                confidence += 0.2

            wave_c_end = None
            if i + 3 < len(pivots):
                candidate_c = pivots[i+3]
                if candidate_c.pivot_type == "high":
                    wave_c_end = candidate_c

            if wave_c_end:
                wave_c_size = wave_c_end.price - wave_b_end.price
                c_to_a_ratio = wave_c_size / wave_a_size if wave_a_size > 0 else 0
                if 0.8 <= c_to_a_ratio <= 1.2:
                    confidence += 0.2
                if wave_c_end.price > wave_a_end.price:
                    confidence += 0.1

                zigzags.append(ZigzagCorrection(
                    wave_a=start, wave_b=wave_b_end, wave_c=wave_c_end,
                    direction=direction, confidence=min(confidence, 1.0),
                ))

    zigzags.sort(key=lambda z: z.confidence, reverse=True)
    return zigzags
