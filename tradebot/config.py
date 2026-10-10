"""Bot settings, loaded from environment variables (or a .env file)."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.getenv("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", ""))
    symbols: list[str] = field(
        default_factory=lambda: [s.strip() for s in os.getenv("SYMBOLS", "BTC/USD,ETH/USD").split(",") if s.strip()]
    )

    # Strategy
    fast_ema: int = field(default_factory=lambda: _i("FAST_EMA", 9))
    slow_ema: int = field(default_factory=lambda: _i("SLOW_EMA", 21))
    trend_ema: int = field(default_factory=lambda: _i("TREND_EMA", 0))  # 0 = no long-term trend filter
    exit_on_trend: bool = field(default_factory=lambda: os.getenv("EXIT_ON_TREND", "1") == "1")

    # Exits
    stop_loss_pct: float = field(default_factory=lambda: _f("STOP_LOSS_PCT", 0.0075))
    take_profit_pct: float = field(default_factory=lambda: _f("TAKE_PROFIT_PCT", 0.015))

    # Sizing and risk
    position_notional_usd: float = field(default_factory=lambda: _f("POSITION_NOTIONAL_USD", 1000))
    max_daily_loss_usd: float = field(default_factory=lambda: _f("MAX_DAILY_LOSS_USD", 150))
    max_trades_per_day: int = field(default_factory=lambda: _i("MAX_TRADES_PER_DAY", 12))

    # Alpaca crypto taker fee per side; used by the backtester to keep results honest.
    fee_pct: float = field(default_factory=lambda: _f("FEE_PCT", 0.0025))

    poll_seconds: int = field(default_factory=lambda: _i("POLL_SECONDS", 60))
