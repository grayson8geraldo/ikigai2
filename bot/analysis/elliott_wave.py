"""Elliott Wave labeling engine.

Applies strict Elliott Wave rules to a sequence of pivots to identify
impulse waves (1-2-3-4-5) and corrective structures.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from bot.analysis.fractals import Pivot

logger = logging.getLogger(__name__)


@dataclass
class WaveLabel:
    number: str         # "1", "2", "3", "4", "5" or "A", "B", "C", "D", "E"
    pivot: Pivot
    wave_type: str      # "impulse", "correction"


@dataclass
class ImpulseWave:
    """Represents a 5-wave impulse structure."""
    wave1_start: Pivot
    wave1_end: Pivot    # = wave 2 start
    wave2_end: Pivot    # = wave 3 start
    wave3_end: Pivot    # = wave 4 start
    wave4_end: Pivot    # = wave 5 start
    wave5_end: Optional[Pivot] = None
    direction: str = "up"  # "up" or "down"
    confidence: float = 0.0

    @property
    def wave1_length(self) -> float:
        return abs(self.wave1_end.price - self.wave1_start.price)

    @property
    def wave2_length(self) -> float:
        return abs(self.wave2_end.price - self.wave1_end.price)

    @property
    def wave3_length(self) -> float:
        return abs(self.wave3_end.price - self.wave2_end.price)

    @property
    def wave4_length(self) -> float:
        return abs(self.wave4_end.price - self.wave3_end.price)

    @property
    def wave5_length(self) -> float:
        if self.wave5_end is None:
            return 0.0
        return abs(self.wave5_end.price - self.wave4_end.price)


def validate_impulse_rules(pivots: list[Pivot], direction: str = "up") -> tuple[bool, float]:
    """Validate Elliott Wave impulse rules on 6 pivots (start + 5 wave endpoints).

    Returns (is_valid, confidence_score).

    Rules:
    1. Wave 2 never retraces more than 100% of wave 1
    2. Wave 3 is never the shortest of waves 1, 3, 5
    3. Wave 4 does not overlap wave 1 territory (no overlap rule)
    """
    if len(pivots) < 5:
        return False, 0.0

    p0, p1, p2, p3, p4 = pivots[0], pivots[1], pivots[2], pivots[3], pivots[4]

    if direction == "up":
        # Wave 1: p0 -> p1 (up), Wave 2: p1 -> p2 (down), Wave 3: p2 -> p3 (up)
        # Wave 4: p3 -> p4 (down)
        w1 = p1.price - p0.price
        w2 = p1.price - p2.price  # retracement (positive = valid)
        w3 = p3.price - p2.price
        w4_ret = p3.price - p4.price

        # Rule 1: Wave 2 cannot retrace more than 100% of wave 1
        if p2.price <= p0.price:
            return False, 0.0

        # Rule 3: Wave 4 cannot enter wave 1 territory
        if p4.price <= p1.price:
            return False, 0.0

        # For 5-wave check, we need wave 5
        if len(pivots) >= 6:
            p5 = pivots[5]
            w5 = p5.price - p4.price

            # Rule 2: Wave 3 is never the shortest
            lengths = [w1, w3, w5]
            if w3 == min(lengths):
                return False, 0.0

            # Wave 5 must make new high above wave 3
            if p5.price <= p3.price:
                # Truncated 5th — low confidence
                confidence = 0.15
            else:
                confidence = 0.4
        else:
            # Only 5 pivots — wave 5 not yet complete
            if w3 < w1 * 0.5:
                return False, 0.0
            confidence = 0.25

    else:  # direction == "down"
        w1 = p0.price - p1.price
        w2 = p2.price - p1.price
        w3 = p2.price - p3.price
        w4_ret = p4.price - p3.price

        if p2.price >= p0.price:
            return False, 0.0
        if p4.price >= p1.price:
            return False, 0.0

        if len(pivots) >= 6:
            p5 = pivots[5]
            w5 = p4.price - p5.price
            lengths = [w1, w3, w5]
            if w3 == min(lengths):
                return False, 0.0
            if p5.price >= p3.price:
                confidence = 0.15
            else:
                confidence = 0.4
        else:
            if w3 < w1 * 0.5:
                return False, 0.0
            confidence = 0.25

    # Bonus confidence: wave 3 is the longest (most common)
    if len(pivots) >= 6:
        if w3 == max(lengths):
            confidence += 0.1

    # Bonus: wave 2 retraces 50-78.6% of wave 1 (ideal)
    w2_retrace = w2 / w1 if w1 != 0 else 0
    if 0.38 <= w2_retrace <= 0.786:
        confidence += 0.05

    return True, min(confidence, 0.75)


def find_impulse_waves(pivots: list[Pivot], direction: str = "up") -> list[ImpulseWave]:
    """Scan pivot sequence for valid impulse wave patterns.

    Uses sliding window approach to find all possible impulse interpretations.
    """
    waves = []

    if direction == "up":
        # Filter to alternating low-high-low-high-low-high
        expected = ["low", "high", "low", "high", "low", "high"]
    else:
        expected = ["high", "low", "high", "low", "high", "low"]

    # Try every starting point
    for start_idx in range(len(pivots)):
        if pivots[start_idx].pivot_type != expected[0]:
            continue

        # Collect matching sequence
        sequence = [pivots[start_idx]]
        j = start_idx + 1
        for exp_type in expected[1:]:
            while j < len(pivots) and pivots[j].pivot_type != exp_type:
                j += 1
            if j >= len(pivots):
                break
            sequence.append(pivots[j])
            j += 1

        if len(sequence) < 5:
            continue

        is_valid, confidence = validate_impulse_rules(sequence, direction)
        if is_valid and confidence >= 0.4:
            wave = ImpulseWave(
                wave1_start=sequence[0],
                wave1_end=sequence[1],
                wave2_end=sequence[2],
                wave3_end=sequence[3],
                wave4_end=sequence[4],
                wave5_end=sequence[5] if len(sequence) >= 6 else None,
                direction=direction,
                confidence=confidence,
            )
            waves.append(wave)

    # Sort by confidence descending
    waves.sort(key=lambda w: w.confidence, reverse=True)
    return waves
