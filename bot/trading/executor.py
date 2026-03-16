"""Trade executor — places and manages orders on the exchange."""

import logging
import time
from typing import Optional
import ccxt

from bot.config import Config
from bot.data_fetcher import DataFetcher
from bot.trading.risk_manager import RiskManager, Position
from bot.analysis.strategy import TradeSignal

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Handles order placement and management via ccxt."""

    def __init__(self, config: Config, fetcher: DataFetcher, risk_manager: RiskManager):
        self.config = config
        self.exchange = fetcher.exchange
        self.risk = risk_manager
        self.paper_mode = config.MODE == "paper"

    def execute_signal(self, signal: TradeSignal) -> Optional[Position]:
        """Execute a trade signal — open position with SL/TP."""
        if not self.risk.can_open_position():
            return None

        # Check if already in position for this symbol
        for pos in self.risk.open_positions:
            if pos.symbol == signal.symbol:
                logger.info(f"Already in position for {signal.symbol}")
                return None

        # Minimum confidence threshold
        if signal.confidence < 0.5:
            logger.info(f"Signal confidence too low: {signal.confidence:.0%}")
            return None

        position = self.risk.create_position(signal)

        if self.paper_mode:
            logger.info(f"[PAPER] Opened {signal.direction} {signal.symbol} "
                       f"@ {signal.entry_price:.4f}")
            return position

        # Live execution
        try:
            side = "buy" if signal.direction == "long" else "sell"

            # Set leverage
            try:
                self.exchange.set_leverage(signal.leverage, signal.symbol)
            except Exception as e:
                logger.warning(f"Could not set leverage: {e}")

            # Market order
            order = self.exchange.create_order(
                symbol=signal.symbol,
                type="market",
                side=side,
                amount=position.size,
            )
            logger.info(f"Order placed: {order['id']}")

            # Stop loss order
            sl_side = "sell" if signal.direction == "long" else "buy"
            try:
                self.exchange.create_order(
                    symbol=signal.symbol,
                    type="stop",
                    side=sl_side,
                    amount=position.size,
                    price=position.stop_loss,
                    params={"stopPrice": position.stop_loss, "reduceOnly": True},
                )
            except Exception as e:
                logger.warning(f"Could not place stop loss: {e}")

            # Take profit order
            tp_side = sl_side
            try:
                self.exchange.create_order(
                    symbol=signal.symbol,
                    type="limit",
                    side=tp_side,
                    amount=position.size,
                    price=position.take_profit,
                    params={"reduceOnly": True},
                )
            except Exception as e:
                logger.warning(f"Could not place take profit: {e}")

            return position

        except ccxt.InsufficientFunds:
            logger.error(f"Insufficient funds for {signal.symbol}")
            position.status = "closed"
            return None
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            position.status = "closed"
            return None

    def check_and_manage_positions(self):
        """Check all open positions for exit conditions."""
        for position in self.risk.open_positions:
            try:
                ticker = self.exchange.fetch_ticker(position.symbol) if not self.paper_mode else None
                if self.paper_mode:
                    # In paper mode, we'd need current price from data
                    continue

                current_price = ticker["last"]
                self.risk.update_trailing_stop(position, current_price)

                exit_reason = self.risk.check_exit(position, current_price)
                if exit_reason:
                    self._close_live_position(position, current_price, exit_reason)

            except Exception as e:
                logger.error(f"Error managing position {position.symbol}: {e}")

    def _close_live_position(self, position: Position, exit_price: float, reason: str):
        """Close a live position on the exchange."""
        if self.paper_mode:
            self.risk.close_position(position, exit_price, reason)
            return

        try:
            side = "sell" if position.direction == "long" else "buy"
            self.exchange.create_order(
                symbol=position.symbol,
                type="market",
                side=side,
                amount=position.size,
                params={"reduceOnly": True},
            )
            self.risk.close_position(position, exit_price, reason)

            # Cancel remaining SL/TP orders
            try:
                open_orders = self.exchange.fetch_open_orders(position.symbol)
                for order in open_orders:
                    self.exchange.cancel_order(order["id"], position.symbol)
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Failed to close position {position.symbol}: {e}")

    def close_all_positions(self, reason: str = "manual"):
        """Emergency: close all open positions."""
        for position in self.risk.open_positions:
            try:
                ticker = self.exchange.fetch_ticker(position.symbol) if not self.paper_mode else None
                price = ticker["last"] if ticker else position.entry_price
                self._close_live_position(position, price, reason)
            except Exception as e:
                logger.error(f"Failed to close {position.symbol}: {e}")
