"""Master strategy: combines all Elliott Wave analysis into trade signals."""

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from bot.config import Config
from bot.analysis.fractals import find_significant_pivots, Pivot
from bot.analysis.elliott_wave import find_impulse_waves, ImpulseWave
from bot.analysis.triangle import detect_triangles, Triangle
from bot.analysis.zigzag_pattern import detect_zigzag, ZigzagCorrection
from bot.analysis.diagonal import detect_diagonals, Diagonal
from bot.analysis.targets import (
    calculate_all_targets, TargetCluster, fibonacci_extension_targets,
    triangle_target, parallel_channel_target, find_target_clusters,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    symbol: str
    timeframe: str
    direction: str          # "long" or "short"
    entry_price: float
    stop_loss: float
    take_profit_zones: list[TargetCluster]
    leverage: int
    confidence: float
    reason: str
    wave_context: str       # description of current wave position


@dataclass
class MarketAnalysis:
    symbol: str
    timeframe: str
    pivots: list[Pivot] = field(default_factory=list)
    impulse_waves: list[ImpulseWave] = field(default_factory=list)
    triangles: list[Triangle] = field(default_factory=list)
    zigzags: list[ZigzagCorrection] = field(default_factory=list)
    diagonals: list[Diagonal] = field(default_factory=list)
    signal: Optional[TradeSignal] = None


class StrategyAnalyzer:
    """Implements the full Elliott Wave trading strategy."""

    def __init__(self, config: Config):
        self.config = config

    def analyze(self, symbol: str, timeframe: str, df: pd.DataFrame) -> MarketAnalysis:
        """Run full analysis on a single symbol/timeframe."""
        analysis = MarketAnalysis(symbol=symbol, timeframe=timeframe)

        if df.empty or len(df) < 50:
            return analysis

        # Step 0: Find pivots
        window = self.config.FRACTAL_WINDOW
        min_swing = 0.02 if timeframe in ("1d", "1w") else 0.015
        pivots = find_significant_pivots(df, window=window, min_change_pct=min_swing)
        analysis.pivots = pivots

        if len(pivots) < 6:
            return analysis

        # Step 1: Search for triangles (primary trigger)
        triangles_up = detect_triangles(pivots, direction="up", tolerance=self.config.TRIANGLE_TOLERANCE)
        triangles_down = detect_triangles(pivots, direction="down", tolerance=self.config.TRIANGLE_TOLERANCE)
        analysis.triangles = triangles_up + triangles_down

        # Step 2: Search for impulse waves
        impulses_up = find_impulse_waves(pivots, direction="up")
        impulses_down = find_impulse_waves(pivots, direction="down")
        analysis.impulse_waves = impulses_up + impulses_down

        # Step 3: Search for zigzag corrections
        zigzags_down = detect_zigzag(pivots, direction="down")
        zigzags_up = detect_zigzag(pivots, direction="up")
        analysis.zigzags = zigzags_down + zigzags_up

        # Step 4: Search for diagonals
        diags_up = detect_diagonals(pivots, direction="up")
        diags_down = detect_diagonals(pivots, direction="down")
        analysis.diagonals = diags_up + diags_down

        # Step 5: Generate trade signal
        signal = self._generate_signal(symbol, timeframe, df, analysis)
        analysis.signal = signal

        return analysis

    def _generate_signal(self, symbol: str, timeframe: str,
                          df: pd.DataFrame, analysis: MarketAnalysis) -> Optional[TradeSignal]:
        """Generate trade signal based on analysis results."""
        current_price = df["close"].iloc[-1]

        # STRATEGY 1: Triangle breakout (highest priority)
        # If triangle found as wave 4, prepare for wave 5
        signal = self._check_triangle_breakout(symbol, timeframe, current_price, analysis)
        if signal:
            return signal

        # STRATEGY 2: Leading diagonal detected — new trend starting
        signal = self._check_leading_diagonal(symbol, timeframe, current_price, analysis)
        if signal:
            return signal

        # STRATEGY 3: Ending diagonal — reversal imminent
        signal = self._check_ending_diagonal(symbol, timeframe, current_price, analysis)
        if signal:
            return signal

        return None

    def _check_triangle_breakout(self, symbol: str, timeframe: str,
                                  current_price: float,
                                  analysis: MarketAnalysis) -> Optional[TradeSignal]:
        """Strategy 1: Triangle as wave 4 → trade wave 5 breakout."""
        if not analysis.triangles:
            return None

        best_triangle = analysis.triangles[0]

        # Validate alternation: wave 2 should be a zigzag (sharp correction)
        # If wave 4 is a triangle (sideways), wave 2 must be sharp
        has_alternation = False
        for zz in analysis.zigzags:
            # Check if zigzag occurred before the triangle in time
            if zz.wave_c.index < best_triangle.wave_a.index:
                has_alternation = True
                break

        if not has_alternation and analysis.zigzags:
            # Still accept but lower confidence
            pass

        # Find matching impulse wave where this triangle is wave 4
        matching_impulse = None
        for imp in analysis.impulse_waves:
            # Triangle should be near wave 4 position
            if (abs(imp.wave4_end.index - best_triangle.wave_e.index) <= 3 or
                abs(imp.wave3_end.index - best_triangle.wave_a.index) <= 5):
                matching_impulse = imp
                break

        if matching_impulse is None and analysis.impulse_waves:
            # Use best impulse as context even if not perfectly matched
            matching_impulse = analysis.impulse_waves[0]

        # Calculate targets
        if matching_impulse:
            target_clusters = calculate_all_targets(
                matching_impulse, best_triangle, self.config.CLUSTER_TOLERANCE,
                entry_price=current_price,
            )
        else:
            # Use triangle target alone
            from bot.analysis.targets import TargetLevel
            tri_targets = triangle_target(best_triangle)
            target_clusters = find_target_clusters(
                tri_targets + [TargetLevel(
                    price=current_price * 1.15, method="estimate",
                    description="15% estimate", strength=0.3,
                )],
                self.config.CLUSTER_TOLERANCE,
            )

        if not target_clusters:
            return None

        # Entry and stop loss
        if best_triangle.breakout_direction == "up":
            entry = best_triangle.breakout_level
            direction = "long"

            # Only enter if price has actually broken out above the level
            if current_price >= entry:
                entry = current_price
            else:
                # Breakout hasn't happened yet — skip signal
                logger.info(f"[{symbol}] Price {current_price:.4f} below breakout {entry:.4f} — waiting")
                return None

            # SL must be BELOW entry for a long — use lowest point of wave E or triangle
            wave_e_price = best_triangle.wave_e.price
            sl_candidate = min(wave_e_price, entry) * 0.98
            stop_loss = sl_candidate
        else:
            entry = best_triangle.breakout_level
            direction = "short"

            # Only enter if price has actually broken down below the level
            if current_price <= entry:
                entry = current_price
            else:
                logger.info(f"[{symbol}] Price {current_price:.4f} above breakout {entry:.4f} — waiting")
                return None

            # SL must be ABOVE entry for a short
            wave_e_price = best_triangle.wave_e.price
            sl_candidate = max(wave_e_price, entry) * 1.02
            stop_loss = sl_candidate

        # Leverage based on risk
        risk_pct = abs(entry - stop_loss) / entry
        leverage = min(int(0.05 / max(risk_pct, 0.005)), self.config.MAX_LEVERAGE)
        leverage = max(leverage, 1)

        confidence = best_triangle.confidence
        if has_alternation:
            confidence = min(confidence + 0.15, 1.0)
        if matching_impulse:
            confidence = min(confidence + 0.1, 1.0)

        wave_ctx = "Wave 4 Triangle detected"
        if matching_impulse:
            wave_ctx += f" within impulse (conf={matching_impulse.confidence:.0%})"
        if has_alternation:
            wave_ctx += ", alternation confirmed (W2=zigzag)"

        return TradeSignal(
            symbol=symbol, timeframe=timeframe,
            direction=direction,
            entry_price=entry, stop_loss=stop_loss,
            take_profit_zones=target_clusters,
            leverage=leverage,
            confidence=confidence,
            reason="Triangle breakout (Wave 5 entry)",
            wave_context=wave_ctx,
        )

    def _check_leading_diagonal(self, symbol: str, timeframe: str,
                                 current_price: float,
                                 analysis: MarketAnalysis) -> Optional[TradeSignal]:
        """Strategy 2: Leading diagonal → new trend, enter on completion."""
        leading = [d for d in analysis.diagonals if d.diagonal_type == "leading"]
        if not leading:
            return None

        diag = leading[0]
        if not diag.is_complete:
            return None

        if diag.direction == "up":
            # New uptrend starting — enter long after diagonal completes
            entry = current_price
            # Stop below diagonal start
            stop_loss = diag.waves[0].price * 0.97
            direction = "long"

            # Target: wave 3 of the larger impulse (typically 1.618x wave 1)
            diag_height = diag.waves[5].price - diag.waves[0].price
            tp_price = diag.waves[5].price + diag_height * 1.618

            from bot.analysis.targets import TargetLevel
            targets = [TargetLevel(price=tp_price, method="fibonacci",
                                    description="W3 target (1.618x diagonal)", strength=0.7)]
            target_clusters = find_target_clusters(targets, self.config.CLUSTER_TOLERANCE)
        else:
            entry = current_price
            stop_loss = diag.waves[0].price * 1.03
            direction = "short"
            diag_height = diag.waves[0].price - diag.waves[5].price
            tp_price = diag.waves[5].price - diag_height * 1.618
            from bot.analysis.targets import TargetLevel
            targets = [TargetLevel(price=tp_price, method="fibonacci",
                                    description="W3 target (1.618x diagonal)", strength=0.7)]
            target_clusters = find_target_clusters(targets, self.config.CLUSTER_TOLERANCE)

        risk_pct = abs(entry - stop_loss) / entry
        leverage = min(int(0.05 / max(risk_pct, 0.005)), self.config.MAX_LEVERAGE)

        return TradeSignal(
            symbol=symbol, timeframe=timeframe,
            direction=direction,
            entry_price=entry, stop_loss=stop_loss,
            take_profit_zones=target_clusters,
            leverage=max(leverage, 1),
            confidence=diag.confidence,
            reason="Leading diagonal completion (new trend)",
            wave_context=f"Leading diagonal ({diag.direction}) completed — wave 1 of new impulse",
        )

    def _check_ending_diagonal(self, symbol: str, timeframe: str,
                                current_price: float,
                                analysis: MarketAnalysis) -> Optional[TradeSignal]:
        """Strategy 3: Ending diagonal → imminent reversal, counter-trend trade."""
        ending = [d for d in analysis.diagonals if d.diagonal_type == "ending"]
        if not ending:
            return None

        diag = ending[0]
        if not diag.is_complete:
            return None

        # Ending diagonal in uptrend → go short (reversal)
        if diag.direction == "up":
            entry = current_price
            stop_loss = diag.waves[5].price * 1.02  # Above diagonal end
            direction = "short"
            # Reversal target: wave 2 of the diagonal
            tp_price = diag.reversal_target
        else:
            entry = current_price
            stop_loss = diag.waves[5].price * 0.98
            direction = "long"
            tp_price = diag.reversal_target

        from bot.analysis.targets import TargetLevel
        targets = [TargetLevel(price=tp_price, method="fibonacci",
                                description="Diagonal reversal target (W2)", strength=0.75)]
        target_clusters = find_target_clusters(targets, self.config.CLUSTER_TOLERANCE)

        risk_pct = abs(entry - stop_loss) / entry
        leverage = min(int(0.04 / max(risk_pct, 0.005)), self.config.MAX_LEVERAGE)

        return TradeSignal(
            symbol=symbol, timeframe=timeframe,
            direction=direction,
            entry_price=entry, stop_loss=stop_loss,
            take_profit_zones=target_clusters,
            leverage=max(leverage, 1),
            confidence=diag.confidence * 0.9,  # Slightly less confident on reversal trades
            reason="Ending diagonal reversal",
            wave_context=f"Ending diagonal ({diag.direction}) completed — sharp reversal expected",
        )

    def analyze_multi_timeframe(self, symbol: str,
                                 data: dict[str, pd.DataFrame]) -> Optional[TradeSignal]:
        """Analyze across multiple timeframes and find the strongest signal.

        Higher timeframes have priority. Signal must not conflict across timeframes.
        """
        tf_priority = {"1w": 4, "1d": 3, "4h": 2, "1h": 1}
        signals = []

        for tf, df in data.items():
            analysis = self.analyze(symbol, tf, df)
            if analysis.signal:
                priority = tf_priority.get(tf, 0)
                analysis.signal.confidence *= (0.7 + 0.1 * priority)
                signals.append((priority, analysis.signal))
                logger.info(f"[{symbol}] Signal on {tf}: {analysis.signal.reason} "
                           f"(conf={analysis.signal.confidence:.0%})")

        if not signals:
            return None

        # Sort by priority * confidence
        signals.sort(key=lambda s: s[0] * s[1].confidence, reverse=True)
        best = signals[0][1]

        # Check for conflicting signals across timeframes
        directions = [s[1].direction for s in signals]
        if "long" in directions and "short" in directions:
            logger.warning(f"[{symbol}] Conflicting signals across timeframes — skipping")
            return None

        return best
