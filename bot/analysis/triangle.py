"""Triangle pattern (ABCDE) detector for Elliott Wave 4th wave identification."""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional
from bot.analysis.fractals import Pivot

logger = logging.getLogger(__name__)


@dataclass
class Triangle:
    """ABCDE triangle pattern — contracting range."""
    wave_a: Pivot
    wave_b: Pivot
    wave_c: Pivot
    wave_d: Pivot
    wave_e: Pivot
    upper_slope: float   # slope of upper trendline (B-D)
    lower_slope: float   # slope of lower trendline (A-C-E)
    apex_index: float    # projected bar index where trendlines meet
    breakout_direction: str  # "up" or "down"
    confidence: float = 0.0

    @property
    def width(self) -> float:
        """Widest part of the triangle (wave A amplitude)."""
        return abs(self.wave_a.price - self.wave_b.price)

    @property
    def breakout_level(self) -> float:
        """Upper trendline value at wave E index (resistance to break)."""
        if self.breakout_direction == "up":
            # Use upper trendline (B-D) extrapolated to wave E
            slope = (self.wave_d.price - self.wave_b.price) / max(self.wave_d.index - self.wave_b.index, 1)
            return self.wave_d.price + slope * (self.wave_e.index - self.wave_d.index)
        else:
            slope = (self.wave_c.price - self.wave_a.price) / max(self.wave_c.index - self.wave_a.index, 1)
            return self.wave_c.price + slope * (self.wave_e.index - self.wave_c.index)

    @property
    def target_from_triangle(self) -> float:
        """Target = breakout level + triangle width."""
        if self.breakout_direction == "up":
            return self.breakout_level + self.width
        else:
            return self.breakout_level - self.width


def detect_triangles(pivots: list[Pivot], direction: str = "up",
                      tolerance: float = 0.02) -> list[Triangle]:
    """Detect contracting triangle patterns (ABCDE) in pivot sequence.

    For a bullish impulse (direction="up"):
    - Triangle forms as wave 4 correction (downward then sideways)
    - A = first down move, B = bounce, C = lower low than A, D = lower high than B, E = higher low than C
    - Converging trendlines: upper (B-D) slopes down, lower (A-C-E) slopes up

    Returns list of detected triangles sorted by confidence.
    """
    triangles = []

    # We need at least 5 pivots to form ABCDE
    for i in range(len(pivots) - 4):
        a, b, c, d, e = pivots[i], pivots[i+1], pivots[i+2], pivots[i+3], pivots[i+4]

        # For bullish triangle (wave 4 in uptrend):
        # A = low, B = high, C = low, D = high, E = low
        if direction == "up":
            if not (a.pivot_type == "low" and b.pivot_type == "high" and
                    c.pivot_type == "low" and d.pivot_type == "high" and
                    e.pivot_type == "low"):
                continue

            # Contracting: D < B (lower high) and C > A (higher low) and E > C (higher low)
            contracting = (d.price < b.price and c.price > a.price and e.price > c.price)
            if not contracting:
                # Allow small tolerance
                contracting = (
                    d.price < b.price * (1 + tolerance) and
                    c.price > a.price * (1 - tolerance) and
                    e.price > c.price * (1 - tolerance)
                )
            if not contracting:
                continue

            breakout_dir = "up"

        else:  # Bearish triangle
            if not (a.pivot_type == "high" and b.pivot_type == "low" and
                    c.pivot_type == "high" and d.pivot_type == "low" and
                    e.pivot_type == "high"):
                continue

            contracting = (d.price > b.price and c.price < a.price and e.price < c.price)
            if not contracting:
                contracting = (
                    d.price > b.price * (1 - tolerance) and
                    c.price < a.price * (1 + tolerance) and
                    e.price < c.price * (1 + tolerance)
                )
            if not contracting:
                continue

            breakout_dir = "down"

        # Calculate trendline slopes
        if direction == "up":
            upper_slope = (d.price - b.price) / max(d.index - b.index, 1)
            lower_slope = (e.price - a.price) / max(e.index - a.index, 1)
        else:
            upper_slope = (e.price - a.price) / max(e.index - a.index, 1)
            lower_slope = (d.price - b.price) / max(d.index - b.index, 1)

        # Apex: where trendlines intersect
        if direction == "up":
            denom = lower_slope - upper_slope
            if abs(denom) < 1e-10:
                apex_idx = e.index + 50  # parallel lines, far apex
            else:
                apex_idx = b.index + (b.price - a.price) / denom
        else:
            denom = upper_slope - lower_slope
            if abs(denom) < 1e-10:
                apex_idx = e.index + 50
            else:
                apex_idx = a.index + (a.price - b.price) / denom

        # Confidence scoring
        confidence = 0.5

        # Better if each successive swing is smaller (true contraction)
        ab = abs(b.price - a.price)
        bc = abs(c.price - b.price)
        cd = abs(d.price - c.price)
        de = abs(e.price - d.price)
        if ab > bc > cd > de:
            confidence += 0.25
        elif ab > bc and cd > de:
            confidence += 0.15

        # Better if roughly symmetric timing
        time_ab = b.index - a.index
        time_bc = c.index - b.index
        time_cd = d.index - c.index
        time_de = e.index - d.index
        avg_time = (time_ab + time_bc + time_cd + time_de) / 4
        time_variance = np.var([time_ab, time_bc, time_cd, time_de])
        if avg_time > 0 and time_variance / (avg_time ** 2) < 0.5:
            confidence += 0.1

        triangles.append(Triangle(
            wave_a=a, wave_b=b, wave_c=c, wave_d=d, wave_e=e,
            upper_slope=upper_slope,
            lower_slope=lower_slope,
            apex_index=apex_idx,
            breakout_direction=breakout_dir,
            confidence=min(confidence, 1.0),
        ))

    triangles.sort(key=lambda t: t.confidence, reverse=True)
    return triangles
