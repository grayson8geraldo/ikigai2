"""Risk management module — controls position sizing, drawdown protection, and exposure."""

import logging
import time
from dataclasses import dataclass, field
from bot.config import Config
from bot.analysis.strategy import TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    direction: str
    entry_price: float
    size: float          # in base currency
    leverage: int
    stop_loss: float
    take_profit: float
    trailing_stop: float = 0.0
    pnl: float = 0.0
    status: str = "open"  # "open", "closed"


class RiskManager:
    def __init__(self, config: Config):
        self.config = config
        self.positions: list[Position] = []
        self.peak_balance = config.INITIAL_BALANCE
        self.current_balance = config.INITIAL_BALANCE
        self.trade_history: list[dict] = []
        self.total_trades = 0
        self.winning_trades = 0
        self._cooldowns: dict[str, float] = {}  # symbol -> timestamp when cooldown expires
        self.COOLDOWN_SECONDS = 3 * 3600  # 3 hours cooldown after stop-loss

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def win_rate(self) -> float:
        return self.winning_trades / max(self.total_trades, 1)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_balance == 0:
            return 0
        return (self.peak_balance - self.current_balance) / self.peak_balance

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        if len(self.open_positions) >= self.config.MAX_POSITIONS:
            logger.info("Max positions reached")
            return False
        if self.drawdown_pct >= self.config.MAX_DRAWDOWN_PCT:
            logger.warning(f"Max drawdown reached: {self.drawdown_pct:.1%}")
            return False
        return True

    def calculate_position_size(self, signal: TradeSignal) -> float:
        """Calculate position size based on risk per trade.

        Risk amount = balance * max_risk_per_trade
        Position size = risk_amount / (entry - stop_loss) * leverage
        """
        risk_amount = self.current_balance * self.config.MAX_RISK_PER_TRADE
        price_risk = abs(signal.entry_price - signal.stop_loss)

        if price_risk == 0:
            return 0

        # Position in USDT
        position_usdt = (risk_amount / price_risk) * signal.entry_price
        # Apply leverage
        margin_required = position_usdt / signal.leverage

        # Don't use more than 30% of balance per trade
        max_margin = self.current_balance * 0.30
        if margin_required > max_margin:
            margin_required = max_margin
            position_usdt = margin_required * signal.leverage

        # Size in base currency
        size = position_usdt / signal.entry_price
        return size

    def is_on_cooldown(self, symbol: str) -> bool:
        """Check if a symbol is on cooldown after a recent stop-loss."""
        if symbol not in self._cooldowns:
            return False
        if time.time() >= self._cooldowns[symbol]:
            del self._cooldowns[symbol]
            return False
        remaining = (self._cooldowns[symbol] - time.time()) / 60
        logger.info(f"{symbol} on cooldown for {remaining:.0f} more minutes")
        return True

    def create_position(self, signal: TradeSignal) -> Position | None:
        """Create a new position from a trade signal.

        Returns None if signal has invalid SL/TP values.
        """
        # Check cooldown
        if self.is_on_cooldown(signal.symbol):
            return None

        # Validate stop loss is on the correct side of entry
        if signal.direction == "long" and signal.stop_loss >= signal.entry_price:
            logger.warning(f"Rejected {signal.symbol}: SL={signal.stop_loss:.4f} >= entry={signal.entry_price:.4f} for LONG")
            return None
        if signal.direction == "short" and signal.stop_loss <= signal.entry_price:
            logger.warning(f"Rejected {signal.symbol}: SL={signal.stop_loss:.4f} <= entry={signal.entry_price:.4f} for SHORT")
            return None

        size = self.calculate_position_size(signal)
        if size <= 0:
            logger.warning(f"Rejected {signal.symbol}: calculated size={size:.6f}")
            return None

        # Best take-profit target (highest confidence cluster)
        if signal.direction == "long":
            default_tp = signal.entry_price * 1.10
        else:
            default_tp = signal.entry_price * 0.90

        # Max allowed TP distance: 50% per 1x leverage (scales with leverage)
        max_tp_distance_pct = 0.50 * signal.leverage

        tp = default_tp
        if signal.take_profit_zones:
            best_cluster = signal.take_profit_zones[0]
            candidate_tp = best_cluster.center_price
            tp_distance_pct = abs(candidate_tp - signal.entry_price) / signal.entry_price

            # Validate TP is on correct side of entry and within reasonable range
            if signal.direction == "long" and candidate_tp > signal.entry_price:
                if tp_distance_pct > max_tp_distance_pct:
                    logger.warning(f"{signal.symbol}: TP={candidate_tp:.4f} too far from entry "
                                 f"({tp_distance_pct:.0%}), capping to {max_tp_distance_pct:.0%}")
                    tp = signal.entry_price * (1 + max_tp_distance_pct)
                else:
                    tp = candidate_tp
            elif signal.direction == "short" and candidate_tp < signal.entry_price:
                if tp_distance_pct > max_tp_distance_pct:
                    logger.warning(f"{signal.symbol}: TP={candidate_tp:.4f} too far from entry "
                                 f"({tp_distance_pct:.0%}), capping to {max_tp_distance_pct:.0%}")
                    tp = signal.entry_price * (1 - max_tp_distance_pct)
                else:
                    tp = candidate_tp
            else:
                logger.warning(f"{signal.symbol}: TP={candidate_tp:.4f} on wrong side of entry={signal.entry_price:.4f} "
                             f"for {signal.direction}, using default TP={default_tp:.4f}")

        # Trailing stop
        trailing = signal.entry_price * (1 - self.config.TRAILING_STOP_PCT) if signal.direction == "long" \
            else signal.entry_price * (1 + self.config.TRAILING_STOP_PCT)

        position = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            size=size,
            leverage=signal.leverage,
            stop_loss=signal.stop_loss,
            take_profit=tp,
            trailing_stop=trailing,
        )
        self.positions.append(position)
        logger.info(f"Opened {signal.direction} {signal.symbol}: "
                    f"size={size:.4f}, entry={signal.entry_price:.4f}, "
                    f"SL={signal.stop_loss:.4f}, TP={tp:.4f}, lev={signal.leverage}x")
        return position

    def update_trailing_stop(self, position: Position, current_price: float):
        """Update trailing stop as price moves in our favor."""
        if position.direction == "long":
            new_trail = current_price * (1 - self.config.TRAILING_STOP_PCT)
            if new_trail > position.trailing_stop:
                position.trailing_stop = new_trail
        else:
            new_trail = current_price * (1 + self.config.TRAILING_STOP_PCT)
            if new_trail < position.trailing_stop:
                position.trailing_stop = new_trail

    def check_exit(self, position: Position, current_price: float) -> str:
        """Check if position should be closed. Returns reason or empty string."""
        if position.direction == "long":
            if current_price <= position.stop_loss:
                return "stop_loss"
            if current_price <= position.trailing_stop:
                return "trailing_stop"
            if current_price >= position.take_profit:
                return "take_profit"
        else:
            if current_price >= position.stop_loss:
                return "stop_loss"
            if current_price >= position.trailing_stop:
                return "trailing_stop"
            if current_price <= position.take_profit:
                return "take_profit"
        return ""

    def close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position and update balance."""
        if position.direction == "long":
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price

        pnl_usdt = pnl_pct * position.size * position.entry_price * position.leverage
        # Subtract fees (estimated 0.1% round trip)
        fee = position.size * position.entry_price * position.leverage * 0.001
        pnl_usdt -= fee

        position.pnl = pnl_usdt
        position.status = "closed"
        self.current_balance += pnl_usdt
        self.total_trades += 1

        # Set cooldown if closed by stop-loss or trailing stop
        if reason in ("stop_loss", "trailing_stop"):
            self._cooldowns[position.symbol] = time.time() + self.COOLDOWN_SECONDS
            logger.info(f"{position.symbol} on cooldown for {self.COOLDOWN_SECONDS // 3600}h after {reason}")

        if pnl_usdt > 0:
            self.winning_trades += 1

        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        self.trade_history.append({
            "symbol": position.symbol,
            "direction": position.direction,
            "entry": position.entry_price,
            "exit": exit_price,
            "pnl": pnl_usdt,
            "reason": reason,
            "leverage": position.leverage,
        })

        logger.info(f"Closed {position.symbol} ({reason}): PnL=${pnl_usdt:.2f}, "
                    f"Balance=${self.current_balance:.2f}")

    def get_status(self) -> dict:
        return {
            "balance": self.current_balance,
            "peak_balance": self.peak_balance,
            "drawdown": f"{self.drawdown_pct:.1%}",
            "open_positions": len(self.open_positions),
            "total_trades": self.total_trades,
            "win_rate": f"{self.win_rate:.0%}",
            "target": self.config.TARGET_BALANCE,
            "progress": f"{(self.current_balance / self.config.TARGET_BALANCE) * 100:.1f}%",
        }
