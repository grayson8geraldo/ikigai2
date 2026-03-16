"""Diagonal pattern detector (Leading and Ending diagonals).

Leading diagonal: forms in wave 1 or wave A — signals new trend start.
Ending diagonal: forms in wave 5 or wave C — signals imminent reversal.
"""

import logging
from dataclasses import dataclass
from bot.analysis.fractals import Pivot

logger = logging.getLogger(__name__)


@dataclass
class Diagonal:
    """5-wave diagonal structure with converging trendlines."""
    waves: list[Pivot]  # 6 points: start, 1, 2, 3, 4, 5
    diagonal_type: str  # "leading" or "ending"
    direction: str      # "up" or "down"
    confidence: float = 0.0

    @property
    def is_complete(self) -> bool:
        return len(self.waves) >= 6

    @property
    def reversal_target(self) -> float:
        """After ending diagonal completes, expect sharp reversal back to wave 2 territory."""
        if len(self.waves) >= 6:
            # Quick reversal target = wave 2 level
            return self.waves[2].price
        return 0.0


def detect_diagonals(pivots: list[Pivot], direction: str = "up") -> list[Diagonal]:
    """Detect diagonal patterns (leading and ending).

    Diagonal characteristics:
    - 5-wave structure like impulse, BUT:
    - Wave 4 DOES overlap wave 1 (unlike regular impulse)
    - Converging trendlines (wedge shape)
    - Each wave is corrective (3-wave structure internally)

    Leading diagonal (wave 1/A):
    - After significant decline, signals new trend
    - Wave 3 ≈ 62% of wave 1, wave 4 ≈ 62% of wave 2

    Ending diagonal (wave 5/C):
    - At end of trend, signals reversal
    - All waves overlap
    - Followed by sharp, fast reversal
    """
    diagonals = []

    if direction == "up":
        expected = ["low", "high", "low", "high", "low", "high"]
    else:
        expected = ["high", "low", "high", "low", "high", "low"]

    for i in range(len(pivots) - 5):
        seq = []
        j = i
        for exp_type in expected:
            while j < len(pivots) and pivots[j].pivot_type != exp_type:
                j += 1
            if j >= len(pivots):
                break
            seq.append(pivots[j])
            j += 1

        if len(seq) < 6:
            continue

        p0, p1, p2, p3, p4, p5 = seq[:6]

        if direction == "up":
            w1 = p1.price - p0.price
            w2 = p1.price - p2.price
            w3 = p3.price - p2.price
            w4 = p3.price - p4.price
            w5 = p5.price - p4.price

            # Must be overall upward
            if w1 <= 0 or w3 <= 0 or w5 <= 0:
                continue

            # KEY: Wave 4 overlaps wave 1 (this distinguishes diagonal from impulse)
            overlaps = p4.price < p1.price

            # Converging: wave 5 < wave 3 < wave 1 (typically)
            converging = (w5 < w3 and w3 <= w1 * 1.1)

            # Each successive high is higher
            highs_rising = p3.price > p1.price and p5.price > p3.price
            # Each successive low is higher
            lows_rising = p2.price > p0.price and p4.price > p2.price

        else:
            w1 = p0.price - p1.price
            w2 = p2.price - p1.price
            w3 = p2.price - p3.price
            w4 = p4.price - p3.price
            w5 = p4.price - p5.price

            if w1 <= 0 or w3 <= 0 or w5 <= 0:
                continue

            overlaps = p4.price > p1.price
            converging = (w5 < w3 and w3 <= w1 * 1.1)
            highs_rising = False
            lows_rising = False

        if not overlaps:
            continue

        confidence = 0.4

        # Determine type
        # Leading diagonal: wave 3 ≈ 62% of wave 1
        w3_to_w1 = w3 / w1 if w1 > 0 else 0
        w4_to_w2 = w4 / w2 if w2 > 0 else 0

        is_leading = False
        is_ending = False

        if 0.5 <= w3_to_w1 <= 0.786:
            confidence += 0.15
            is_leading = True
        if 0.5 <= w4_to_w2 <= 0.786:
            confidence += 0.1
            is_leading = True

        if converging:
            confidence += 0.2
            is_ending = True  # ending diagonals are more strongly converging

        if direction == "up" and highs_rising and lows_rising:
            confidence += 0.1

        # Ending diagonals have all waves overlapping
        if direction == "up":
            all_overlap = (p4.price < p1.price and p2.price < p3.price)
        else:
            all_overlap = (p4.price > p1.price and p2.price > p3.price)

        if all_overlap:
            confidence += 0.1
            is_ending = True

        diag_type = "ending" if is_ending and not is_leading else "leading"
        if is_leading and is_ending:
            diag_type = "ending" if converging else "leading"

        diagonals.append(Diagonal(
            waves=seq[:6],
            diagonal_type=diag_type,
            direction=direction,
            confidence=min(confidence, 1.0),
        ))

    diagonals.sort(key=lambda d: d.confidence, reverse=True)
    return diagonals
