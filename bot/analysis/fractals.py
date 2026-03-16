"""Fractal/pivot point detection for identifying peaks and valleys."""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class Pivot:
    index: int          # bar index in the dataframe
    timestamp: pd.Timestamp
    price: float
    pivot_type: str     # "high" or "low"


def detect_fractals(df: pd.DataFrame, window: int = 5) -> list[Pivot]:
    """Detect fractal highs and lows using a rolling window.

    A fractal high: bar whose high is the highest within `window` bars on each side.
    A fractal low: bar whose low is the lowest within `window` bars on each side.
    """
    highs = df["high"].values
    lows = df["low"].values
    pivots = []

    for i in range(window, len(df) - window):
        # Check fractal high
        is_high = True
        for j in range(1, window + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_high = False
                break
        if is_high:
            pivots.append(Pivot(
                index=i,
                timestamp=df.index[i],
                price=highs[i],
                pivot_type="high",
            ))

        # Check fractal low
        is_low = True
        for j in range(1, window + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_low = False
                break
        if is_low:
            pivots.append(Pivot(
                index=i,
                timestamp=df.index[i],
                price=lows[i],
                pivot_type="low",
            ))

    pivots.sort(key=lambda p: p.index)
    return pivots


def get_zigzag(pivots: list[Pivot], min_change_pct: float = 0.03) -> list[Pivot]:
    """Filter pivots into alternating high/low zigzag with minimum swing size.

    Removes noise by requiring at least `min_change_pct` price change between pivots.
    Returns alternating high-low-high-low sequence.
    """
    if len(pivots) < 2:
        return pivots

    zigzag = [pivots[0]]

    for p in pivots[1:]:
        last = zigzag[-1]

        # Same type — keep the more extreme one
        if p.pivot_type == last.pivot_type:
            if p.pivot_type == "high" and p.price > last.price:
                zigzag[-1] = p
            elif p.pivot_type == "low" and p.price < last.price:
                zigzag[-1] = p
        else:
            # Check minimum swing
            change = abs(p.price - last.price) / last.price
            if change >= min_change_pct:
                zigzag.append(p)
            else:
                # Too small, skip or replace
                if p.pivot_type == "high" and p.price > last.price:
                    zigzag[-1] = p
                elif p.pivot_type == "low" and p.price < last.price:
                    zigzag[-1] = p

    return zigzag


def find_significant_pivots(df: pd.DataFrame, window: int = 5,
                             min_change_pct: float = 0.03) -> list[Pivot]:
    """Convenience function: detect fractals and filter to zigzag."""
    raw = detect_fractals(df, window)
    return get_zigzag(raw, min_change_pct)
