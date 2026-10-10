"""Entry/exit signals. Long only. Pick one with STRATEGY:

vwap_ema  VWAP + EMA momentum. Enter on the bar where "close > VWAP and fast EMA > slow EMA" first
          becomes true; exit when that breaks (optional). VWAP restarts each trading day.
orb       Opening range breakout (stocks). Record the high/low of the first N minutes after the open,
          then buy the first close above that high, once per day. Stop at the range low or a fixed %.
dip       Dip buying (mean reversion). Buy when the close first drops below the lower Bollinger band
          (mean - k standard deviations); sell when price gets back to the mean.

Every strategy also exits on its stop loss, take profit, and (stocks) the end of the day.
Each signal function returns the bars plus:
  entry        bool, open a position at this bar's close
  exit_signal  bool, close an open position at this bar's close
  stop_price   optional, overrides the percentage stop loss
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

EXIT_REASONS = {"vwap_ema": "trend_broken", "orb": "signal", "dip": "back_to_average"}


def add_indicators(
    bars: pd.DataFrame, fast: int, slow: int, trend: int = 0, exit_on_trend: bool = True, tz: str = "UTC"
) -> pd.DataFrame:
    """VWAP + EMA momentum signals.

    trend: if > 0, only enter when close is also above an EMA of this many bars (a long-term trend filter).
    exit_on_trend: if False, positions only exit on their stop loss or take profit.
    tz: VWAP restarts at midnight in this time zone (UTC for crypto, New York for stocks).
    """
    df = bars.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.tz_convert(tz).date
    cum_pv = (typical * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    # Zero-volume bars at the start of a day leave VWAP undefined; fall back to price.
    df["vwap"] = (cum_pv / cum_v.replace(0, np.nan)).fillna(df["close"])
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["bullish"] = (df["close"] > df["vwap"]) & (df["ema_fast"] > df["ema_slow"])
    if trend:
        df["ema_trend"] = df["close"].ewm(span=trend, adjust=False).mean()
        df["bullish"] &= df["close"] > df["ema_trend"]
    df["entry"] = df["bullish"] & ~df["bullish"].shift(1, fill_value=True)
    if exit_on_trend:
        df["exit_signal"] = (df["close"] < df["vwap"]) | (df["ema_fast"] < df["ema_slow"])
    else:
        df["exit_signal"] = False
    # Don't trust EMAs until they've had time to warm up.
    df.loc[df.index[: max(slow * 2, trend)], ["entry"]] = False
    return df


def add_orb(bars: pd.DataFrame, or_minutes: int, tz: str = "America/New_York", range_stop: bool = True) -> pd.DataFrame:
    """Opening range breakout signals on 1-minute regular-hours stock bars."""
    df = bars.copy()
    local = df.index.tz_convert(tz)
    day = local.date
    minutes_since_open = (local.hour * 60 + local.minute) - (9 * 60 + 30)
    in_range = (minutes_since_open >= 0) & (minutes_since_open < or_minutes)
    df["or_high"] = df["high"].where(in_range).groupby(day).transform("max")
    df["or_low"] = df["low"].where(in_range).groupby(day).transform("min")
    breakout = (minutes_since_open >= or_minutes) & (df["close"] > df["or_high"])
    df["entry"] = breakout & (breakout.groupby(day).cumsum() == 1)  # first breakout of the day only
    df["exit_signal"] = False
    df["stop_price"] = df["or_low"] if range_stop else np.nan
    return df


def add_dip(bars: pd.DataFrame, length: int = 20, k: float = 2.0, trend: int = 0) -> pd.DataFrame:
    """Dip-buying signals: close drops below mean - k * std of the last `length` bars."""
    df = bars.copy()
    close = df["close"]
    df["mean"] = close.rolling(length).mean()
    df["lower_band"] = df["mean"] - k * close.rolling(length).std()
    dip = close < df["lower_band"]
    if trend:
        df["ema_trend"] = close.ewm(span=trend, adjust=False).mean()
        dip &= close > df["ema_trend"]
    df["entry"] = dip & ~dip.shift(1, fill_value=True)
    df["exit_signal"] = close >= df["mean"]
    df.loc[df.index[: max(length, trend)], ["entry"]] = False
    return df


def signals(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.strategy == "orb":
        return add_orb(bars, cfg.or_minutes, cfg.session_tz, cfg.or_range_stop)
    if cfg.strategy == "dip":
        return add_dip(bars, cfg.band_len, cfg.band_k, cfg.trend_ema)
    return add_indicators(bars, cfg.fast_ema, cfg.slow_ema, cfg.trend_ema, cfg.exit_on_trend, cfg.session_tz)


def stop_level(row: pd.Series, entry_price: float, cfg: Config) -> float:
    stop = row.get("stop_price", np.nan)
    return stop if pd.notna(stop) else entry_price * (1 - cfg.stop_loss_pct)


@dataclass
class Action:
    kind: str  # "enter", "exit" or "hold"
    reason: str = ""


def decide(row: pd.Series, entry_price: float | None, cfg: Config) -> Action:
    """Decide what to do given the latest *closed* bar (with signals) and the open position, if any."""
    price = row["close"]
    if entry_price is not None:
        if price <= stop_level(row, entry_price, cfg):
            return Action("exit", "stop_loss")
        if price >= entry_price * (1 + cfg.take_profit_pct):
            return Action("exit", "take_profit")
        if row["exit_signal"]:
            return Action("exit", EXIT_REASONS[cfg.strategy])
        return Action("hold")
    if row["entry"]:
        return Action("enter", cfg.strategy)
    return Action("hold")
