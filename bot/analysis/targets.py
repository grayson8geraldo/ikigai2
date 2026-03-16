"""Fibonacci extensions, parallel channels, and cluster analysis for target zones."""

import logging
import numpy as np
from dataclasses import dataclass
from bot.analysis.elliott_wave import ImpulseWave
from bot.analysis.triangle import Triangle
from bot.analysis.fractals import Pivot

logger = logging.getLogger(__name__)


@dataclass
class TargetLevel:
    price: float
    method: str     # "fibonacci", "triangle", "channel"
    description: str
    strength: float  # 0-1


@dataclass
class TargetCluster:
    """A zone where multiple target methods converge."""
    center_price: float
    low_price: float
    high_price: float
    levels: list[TargetLevel]
    strength: float  # combined strength

    @property
    def num_methods(self) -> int:
        return len(set(l.method for l in self.levels))


def fibonacci_extension_targets(impulse: ImpulseWave) -> list[TargetLevel]:
    """Calculate Fibonacci extension targets for wave 5.

    Projects wave 3 length from wave 4 end.
    Common targets: 61.8%, 100%, 161.8% of wave 3.
    """
    targets = []
    w3_length = impulse.wave3_length

    fib_levels = [
        (0.618, "Fib 61.8% of W3"),
        (0.786, "Fib 78.6% of W3"),
        (1.0,   "Fib 100% of W3"),
        (1.272, "Fib 127.2% of W3"),
        (1.618, "Fib 161.8% of W3"),
    ]

    for ratio, desc in fib_levels:
        if impulse.direction == "up":
            target = impulse.wave4_end.price + w3_length * ratio
        else:
            target = impulse.wave4_end.price - w3_length * ratio

        # Strength: 61.8% and 100% are most common for wave 5
        if ratio in (0.618, 1.0):
            strength = 0.8
        elif ratio == 0.786:
            strength = 0.6
        else:
            strength = 0.4

        targets.append(TargetLevel(
            price=target, method="fibonacci",
            description=desc, strength=strength,
        ))

    # Also: wave 5 = wave 1 (common equality)
    w1_length = impulse.wave1_length
    if impulse.direction == "up":
        w5_eq_w1 = impulse.wave4_end.price + w1_length
    else:
        w5_eq_w1 = impulse.wave4_end.price - w1_length

    targets.append(TargetLevel(
        price=w5_eq_w1, method="fibonacci",
        description="W5 = W1 equality", strength=0.7,
    ))

    return targets


def fibonacci_log_targets(impulse: ImpulseWave) -> list[TargetLevel]:
    """Fibonacci targets on logarithmic scale.

    Used when arithmetic targets produce unreasonably large values.
    """
    targets = []

    if impulse.direction == "up":
        log_w3 = np.log(impulse.wave3_end.price) - np.log(impulse.wave2_end.price)
        for ratio, desc in [(0.618, "Log Fib 61.8%"), (1.0, "Log Fib 100%")]:
            log_target = np.log(impulse.wave4_end.price) + log_w3 * ratio
            target = np.exp(log_target)
            targets.append(TargetLevel(
                price=target, method="fibonacci",
                description=desc, strength=0.7,
            ))
    else:
        log_w3 = np.log(impulse.wave2_end.price) - np.log(impulse.wave3_end.price)
        for ratio, desc in [(0.618, "Log Fib 61.8%"), (1.0, "Log Fib 100%")]:
            log_target = np.log(impulse.wave4_end.price) - log_w3 * ratio
            target = np.exp(log_target)
            targets.append(TargetLevel(
                price=target, method="fibonacci",
                description=desc, strength=0.7,
            ))

    return targets


def triangle_target(triangle: Triangle) -> list[TargetLevel]:
    """Target from triangle breakout: width of triangle projected from breakout."""
    targets = []

    target_price = triangle.target_from_triangle
    targets.append(TargetLevel(
        price=target_price, method="triangle",
        description="Triangle width projection",
        strength=0.75,
    ))

    return targets


def parallel_channel_target(impulse: ImpulseWave) -> list[TargetLevel]:
    """Parallel channel target for wave 5.

    Channel built from:
    - Base line: wave 1 end -> wave 3 end
    - Parallel through wave 4 end (or wave 2 end)
    - Target = upper channel at projected wave 5 completion
    """
    targets = []

    # Points for channel
    p1 = impulse.wave1_end
    p3 = impulse.wave3_end
    p2 = impulse.wave2_end
    p4 = impulse.wave4_end

    if p3.index == p1.index:
        return targets

    # Base line slope (connecting wave 1 and wave 3 ends)
    base_slope = (p3.price - p1.price) / (p3.index - p1.index)

    # Parallel line through wave 2 end
    # offset = p2.price - (p1.price + base_slope * (p2.index - p1.index))
    # For upper channel (targeting wave 5 in uptrend):
    # Upper line goes through waves 1 and 3 (peaks)
    # Lower line goes through wave 2 (trough)
    # Wave 5 target = upper line extrapolated

    if impulse.direction == "up":
        # Upper line through p1 and p3 (highs of waves 1 and 3)
        # Estimate wave 5 duration ≈ wave 1 duration
        w1_duration = p1.index - impulse.wave1_start.index
        estimated_w5_end_idx = p4.index + w1_duration

        # Upper channel value at estimated wave 5 end
        channel_target = p1.price + base_slope * (estimated_w5_end_idx - p1.index)

        # Alternative: channel through 2 and 4 (lower), then parallel through 1/3 for upper
        lower_slope = (p4.price - p2.price) / max(p4.index - p2.index, 1)
        upper_offset = max(p1.price - (p2.price + lower_slope * (p1.index - p2.index)),
                          p3.price - (p2.price + lower_slope * (p3.index - p2.index)))
        channel_target_alt = (p2.price + lower_slope * (estimated_w5_end_idx - p2.index)) + upper_offset

        targets.append(TargetLevel(
            price=channel_target, method="channel",
            description="Parallel channel (1-3 line)", strength=0.65,
        ))
        if abs(channel_target_alt - channel_target) / channel_target > 0.01:
            targets.append(TargetLevel(
                price=channel_target_alt, method="channel",
                description="Parallel channel (2-4 base)", strength=0.6,
            ))
    else:
        w1_duration = impulse.wave1_start.index - p1.index if p1.index < impulse.wave1_start.index else p1.index - impulse.wave1_start.index
        estimated_w5_end_idx = p4.index + max(w1_duration, 5)
        channel_target = p1.price + base_slope * (estimated_w5_end_idx - p1.index)
        targets.append(TargetLevel(
            price=channel_target, method="channel",
            description="Parallel channel", strength=0.65,
        ))

    return targets


def find_target_clusters(all_targets: list[TargetLevel],
                          tolerance_pct: float = 0.015) -> list[TargetCluster]:
    """Group nearby target levels into clusters.

    Clusters where multiple methods agree are the strongest trade targets.
    """
    if not all_targets:
        return []

    # Sort by price
    sorted_targets = sorted(all_targets, key=lambda t: t.price)
    clusters = []
    used = set()

    for i, target in enumerate(sorted_targets):
        if i in used:
            continue

        cluster_levels = [target]
        used.add(i)

        for j in range(i + 1, len(sorted_targets)):
            if j in used:
                continue
            if abs(sorted_targets[j].price - target.price) / target.price <= tolerance_pct:
                cluster_levels.append(sorted_targets[j])
                used.add(j)

        prices = [l.price for l in cluster_levels]
        center = np.mean(prices)
        combined_strength = min(sum(l.strength for l in cluster_levels) / len(cluster_levels), 1.0)

        # Bonus for multiple methods converging
        methods = set(l.method for l in cluster_levels)
        if len(methods) >= 3:
            combined_strength = min(combined_strength + 0.3, 1.0)
        elif len(methods) >= 2:
            combined_strength = min(combined_strength + 0.15, 1.0)

        clusters.append(TargetCluster(
            center_price=center,
            low_price=min(prices),
            high_price=max(prices),
            levels=cluster_levels,
            strength=combined_strength,
        ))

    clusters.sort(key=lambda c: c.strength, reverse=True)
    return clusters


def calculate_all_targets(impulse: ImpulseWave,
                           triangle_pattern: Triangle = None,
                           tolerance_pct: float = 0.015) -> list[TargetCluster]:
    """Master function: calculate all targets and find convergence clusters."""
    all_targets = []

    # 1. Fibonacci extension targets (arithmetic)
    fib_targets = fibonacci_extension_targets(impulse)
    all_targets.extend(fib_targets)

    # 2. Log-scale Fibonacci (if arithmetic targets are too far)
    current_price = impulse.wave4_end.price
    arith_max = max(t.price for t in fib_targets) if fib_targets else current_price
    if impulse.direction == "up" and arith_max > current_price * 3:
        # Arithmetic targets look extreme — add log scale
        log_targets = fibonacci_log_targets(impulse)
        all_targets.extend(log_targets)

    # 3. Triangle-based target
    if triangle_pattern:
        tri_targets = triangle_target(triangle_pattern)
        all_targets.extend(tri_targets)

    # 4. Parallel channel target
    channel_targets = parallel_channel_target(impulse)
    all_targets.extend(channel_targets)

    # Filter unreasonable targets
    if impulse.direction == "up":
        all_targets = [t for t in all_targets if t.price > current_price]
    else:
        all_targets = [t for t in all_targets if t.price < current_price]

    return find_target_clusters(all_targets, tolerance_pct)
