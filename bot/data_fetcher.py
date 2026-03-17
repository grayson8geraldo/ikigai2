"""OHLCV data fetcher using ccxt."""

import logging
import ccxt
import pandas as pd
import numpy as np
from typing import Optional
from bot.config import Config

logger = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self, config: Config):
        self.config = config
        exchange_class = getattr(ccxt, config.EXCHANGE)
        self.exchange = exchange_class({
            "apiKey": config.API_KEY,
            "secret": config.API_SECRET,
            "enableRateLimit": True,
            "timeout": 30000,  # 30 second timeout for all API calls
            "options": {"defaultType": "swap"},
        })
        if config.MODE == "paper":
            self.exchange.set_sandbox_mode(True)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """Fetch OHLCV candles and return as DataFrame."""
        try:
            data = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
            return pd.DataFrame()

    def fetch_multi_timeframe(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Fetch data for all configured timeframes."""
        result = {}
        for tf in self.config.TIMEFRAMES:
            df = self.fetch_ohlcv(symbol, tf)
            if not df.empty:
                result[tf] = df
        return result

    def get_ticker(self, symbol: str) -> Optional[dict]:
        """Get current ticker price."""
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch ticker {symbol}: {e}")
            return None

    def get_balance(self) -> float:
        """Get USDT balance."""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance.get("USDT", {}).get("free", 0))
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0
