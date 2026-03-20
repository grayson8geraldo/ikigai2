"""Main bot runner — orchestrates scanning, analysis, and trading."""

import logging
import time
import sys
from datetime import datetime

from bot.config import Config
from bot.data_fetcher import DataFetcher
from bot.analysis.strategy import StrategyAnalyzer
from bot.trading.risk_manager import RiskManager
from bot.trading.executor import TradeExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("bot.log", mode="a"),
    ],
)
logger = logging.getLogger("bot")


class ElliottWaveBot:
    """Main trading bot using Elliott Wave analysis."""

    def __init__(self):
        self.config = Config()
        self.fetcher = DataFetcher(self.config)
        self.strategy = StrategyAnalyzer(self.config)
        self.risk = RiskManager(self.config)
        self.executor = TradeExecutor(self.config, self.fetcher, self.risk)

    def scan_markets(self):
        """Scan all configured symbols for trade signals."""
        signals = []

        for symbol in self.config.SYMBOLS:
            try:
                logger.info(f"Scanning {symbol}...")
                data = self.fetcher.fetch_multi_timeframe(symbol)

                if not data:
                    logger.warning(f"No data for {symbol}")
                    continue

                signal = self.strategy.analyze_multi_timeframe(symbol, data)

                if signal:
                    logger.info(
                        f">>> SIGNAL: {signal.direction.upper()} {symbol} "
                        f"@ {signal.entry_price:.4f} | "
                        f"SL: {signal.stop_loss:.4f} | "
                        f"Conf: {signal.confidence:.0%} | "
                        f"{signal.reason}"
                    )
                    if signal.take_profit_zones:
                        best_tp = signal.take_profit_zones[0]
                        logger.info(
                            f"    TP Zone: {best_tp.center_price:.4f} "
                            f"({best_tp.num_methods} methods, str={best_tp.strength:.0%})"
                        )
                    signals.append(signal)
                else:
                    logger.info(f"  No signal for {symbol}")

            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        return signals

    def execute_best_signals(self, signals):
        """Execute the best signal(s) respecting risk limits."""
        if not signals:
            return

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)

        for signal in signals:
            if not self.risk.can_open_position():
                break

            position = self.executor.execute_signal(signal)
            if position:
                logger.info(f"Position opened: {position.symbol} {position.direction}")

    def manage_positions(self, current_prices: dict[str, float]):
        """Check open positions and manage exits (paper mode)."""
        for position in self.risk.open_positions:
            price = current_prices.get(position.symbol)
            if price is None:
                ticker = self.fetcher.get_ticker(position.symbol)
                if ticker:
                    price = ticker["last"]
            if price is None:
                continue

            self.risk.update_trailing_stop(position, price)
            exit_reason = self.risk.check_exit(position, price)
            if exit_reason:
                self.risk.close_position(position, price, exit_reason)
                logger.info(f"Position closed: {position.symbol} ({exit_reason})")

    def print_status(self):
        """Print current bot status."""
        status = self.risk.get_status()
        logger.info("=" * 60)
        logger.info(f"Balance: ${status['balance']:.2f} | "
                    f"Peak: ${status['peak_balance']:.2f} | "
                    f"Drawdown: {status['drawdown']}")
        logger.info(f"Positions: {status['open_positions']} | "
                    f"Trades: {status['total_trades']} | "
                    f"Win Rate: {status['win_rate']}")
        logger.info(f"Target: ${status['target']:.2f} | "
                    f"Progress: {status['progress']}")
        logger.info("=" * 60)

        for pos in self.risk.open_positions:
            logger.info(f"  [{pos.direction.upper()}] {pos.symbol} "
                       f"entry={pos.entry_price:.4f} SL={pos.stop_loss:.4f} "
                       f"TP={pos.take_profit:.4f} lev={pos.leverage}x")

    def run(self, interval_minutes: int = 60):
        """Main loop: scan → analyze → trade → manage → repeat."""
        logger.info("=" * 60)
        logger.info(f"  Elliott Wave Trading Bot")
        logger.info(f"  Mode: {self.config.MODE}")
        logger.info(f"  Exchange: {self.config.EXCHANGE}")
        logger.info(f"  Symbols: {', '.join(self.config.SYMBOLS)}")
        logger.info(f"  Timeframes: {', '.join(self.config.TIMEFRAMES)}")
        logger.info(f"  Initial Balance: ${self.config.INITIAL_BALANCE}")
        logger.info(f"  Target: ${self.config.TARGET_BALANCE}")
        logger.info(f"  Max Risk/Trade: {self.config.MAX_RISK_PER_TRADE:.0%}")
        logger.info(f"  Max Leverage: {self.config.MAX_LEVERAGE}x")
        logger.info("=" * 60)

        cycle = 0
        while True:
            try:
                cycle += 1
                logger.info(f"\n--- Cycle {cycle} | {datetime.now().strftime('%Y-%m-%d %H:%M')} ---")

                # Check if target reached
                if self.risk.current_balance >= self.config.TARGET_BALANCE:
                    logger.info(f"TARGET REACHED! Balance: ${self.risk.current_balance:.2f}")
                    self.executor.close_all_positions("target_reached")
                    break

                # Check max drawdown — pause trading for 6 hours instead of stopping
                if self.risk.drawdown_pct >= self.config.MAX_DRAWDOWN_PCT:
                    logger.warning(f"MAX DRAWDOWN {self.risk.drawdown_pct:.1%} — closing positions and pausing 6h")
                    self.executor.close_all_positions("max_drawdown")
                    # Reset peak to current balance so drawdown resets
                    self.risk.peak_balance = self.risk.current_balance
                    logger.info(f"Peak balance reset to ${self.risk.current_balance:.2f}")
                    time.sleep(6 * 3600)  # Pause 6 hours
                    logger.info("Resuming after drawdown pause...")
                    continue

                # 1. Manage existing positions
                if self.risk.open_positions:
                    current_prices = {}
                    for pos in self.risk.open_positions:
                        ticker = self.fetcher.get_ticker(pos.symbol)
                        if ticker:
                            current_prices[pos.symbol] = ticker["last"]
                    self.manage_positions(current_prices)

                # 2. Scan for new signals
                signals = self.scan_markets()

                # 3. Execute best signals
                self.execute_best_signals(signals)

                # 4. Print status
                self.print_status()

                # 5. Wait for next cycle
                logger.info(f"Sleeping {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                logger.info(f"Waking up for cycle {cycle + 1}...")

            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                self.executor.close_all_positions("manual_stop")
                break
            except Exception as e:
                logger.error(f"Unexpected error in cycle {cycle}: {e}", exc_info=True)
                logger.info("Retrying in 60 seconds...")
                time.sleep(60)


def main():
    bot = ElliottWaveBot()
    # Default: scan every 60 minutes for daily/4h analysis
    interval = 60
    if "1h" in bot.config.TIMEFRAMES and "1d" not in bot.config.TIMEFRAMES:
        interval = 15
    bot.run(interval_minutes=interval)


if __name__ == "__main__":
    main()
