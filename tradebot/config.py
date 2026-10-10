"""Bot settings, loaded from environment variables (or a .env file). Empty values count as unset."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _s(name: str, default: str) -> str:
    return os.getenv(name) or default


def _f(name: str, default: float) -> float:
    return float(_s(name, str(default)))


def _i(name: str, default: int) -> int:
    return int(_s(name, str(default)))


ASSET_CLASS = _s("ASSET_CLASS", "stocks")  # "stocks" or "crypto"
STOCKS = ASSET_CLASS == "stocks"


@dataclass
class Config:
    api_key: str = field(default_factory=lambda: _s("ALPACA_API_KEY", ""))
    secret_key: str = field(default_factory=lambda: _s("ALPACA_SECRET_KEY", ""))
    asset_class: str = ASSET_CLASS
    symbols: list[str] = field(
        default_factory=lambda: [
            s.strip() for s in _s("SYMBOLS", "SPY,QQQ" if STOCKS else "BTC/USD,ETH/USD").split(",") if s.strip()
        ]
    )

    # Strategy: "vwap_ema", "orb" (opening range breakout) or "dip" (dip buying)
    strategy: str = field(default_factory=lambda: _s("STRATEGY", "vwap_ema"))
    bar_minutes: int = field(default_factory=lambda: _i("BAR_MINUTES", 1))
    fast_ema: int = field(default_factory=lambda: _i("FAST_EMA", 9))
    slow_ema: int = field(default_factory=lambda: _i("SLOW_EMA", 21))
    trend_ema: int = field(default_factory=lambda: _i("TREND_EMA", 0))  # 0 = no long-term trend filter
    exit_on_trend: bool = field(default_factory=lambda: _s("EXIT_ON_TREND", "1") == "1")
    or_minutes: int = field(default_factory=lambda: _i("OR_MINUTES", 15))  # orb: opening range length
    or_range_stop: bool = field(default_factory=lambda: _s("OR_RANGE_STOP", "1") == "1")  # orb: stop at range low
    band_len: int = field(default_factory=lambda: _i("BAND_LEN", 20))  # dip: bars in the moving average
    band_k: float = field(default_factory=lambda: _f("BAND_K", 2.0))  # dip: standard deviations below average

    # Exits
    stop_loss_pct: float = field(default_factory=lambda: _f("STOP_LOSS_PCT", 0.0075))
    take_profit_pct: float = field(default_factory=lambda: _f("TAKE_PROFIT_PCT", 0.015))

    # Sizing and risk
    position_notional_usd: float = field(default_factory=lambda: _f("POSITION_NOTIONAL_USD", 1000))
    max_daily_loss_usd: float = field(default_factory=lambda: _f("MAX_DAILY_LOSS_USD", 150))
    max_trades_per_day: int = field(default_factory=lambda: _i("MAX_TRADES_PER_DAY", 12))

    # Trading cost per side, used by the backtester. Crypto: Alpaca's 0.25% taker fee.
    # Stocks are commission-free; 0.02% allows for the bid/ask spread and slippage.
    fee_pct: float = field(default_factory=lambda: _f("FEE_PCT", 0.0002 if STOCKS else 0.0025))

    poll_seconds: int = field(default_factory=lambda: _i("POLL_SECONDS", 60))

    @property
    def stocks(self) -> bool:
        return self.asset_class == "stocks"

    @property
    def session_tz(self) -> str:
        """Time zone whose calendar day defines a trading day (VWAP reset, daily limits)."""
        return "America/New_York" if self.stocks else "UTC"
