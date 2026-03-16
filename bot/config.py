import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    EXCHANGE = os.getenv("EXCHANGE", "bybit")
    API_KEY = os.getenv("API_KEY", "")
    API_SECRET = os.getenv("API_SECRET", "")

    INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "200"))
    TARGET_BALANCE = float(os.getenv("TARGET_BALANCE", "1200"))
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.05"))
    MAX_LEVERAGE = int(os.getenv("MAX_LEVERAGE", "10"))

    TIMEFRAMES = os.getenv("TIMEFRAMES", "1d,4h,1h").split(",")
    SYMBOLS = os.getenv("SYMBOLS", "SOL/USDT,XRP/USDT,OP/USDT,ETH/USDT,BTC/USDT").split(",")

    MODE = os.getenv("MODE", "paper")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    # Elliott Wave parameters
    FRACTAL_WINDOW = 5
    MIN_WAVE_BARS = 10
    TRIANGLE_TOLERANCE = 0.02
    FIB_LEVELS = [0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618]
    CLUSTER_TOLERANCE = 0.015  # 1.5% for target clustering

    # Risk management
    MAX_POSITIONS = 3
    TRAILING_STOP_PCT = 0.02
    MAX_DRAWDOWN_PCT = 0.25  # stop bot if 25% drawdown from peak
