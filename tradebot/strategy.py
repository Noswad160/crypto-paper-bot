"""VWAP + EMA momentum strategy on 1-minute bars.

Long only.

Entry: the bar where "close > VWAP and fast EMA > slow EMA" first becomes true.
Exit:  stop loss, take profit, or the trend breaking (close < VWAP or fast EMA < slow EMA).

VWAP is anchored to each trading day: the stock market open, or 00:00 UTC for crypto.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


def add_indicators(
    bars: pd.DataFrame, fast: int, slow: int, trend: int = 0, exit_on_trend: bool = True, tz: str = "UTC"
) -> pd.DataFrame:
    """bars: DataFrame indexed by UTC timestamp with open/high/low/close/volume columns.

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
        df["trend_broken"] = (df["close"] < df["vwap"]) | (df["ema_fast"] < df["ema_slow"])
    else:
        df["trend_broken"] = False
    # Don't trust EMAs until they've had time to warm up.
    df.loc[df.index[: max(slow * 2, trend)], ["entry"]] = False
    return df


@dataclass
class Action:
    kind: str  # "enter", "exit" or "hold"
    reason: str = ""


def decide(row: pd.Series, entry_price: float | None, cfg: Config) -> Action:
    """Decide what to do given the latest *closed* bar (with indicators) and the open position, if any."""
    price = row["close"]
    if entry_price is not None:
        if price <= entry_price * (1 - cfg.stop_loss_pct):
            return Action("exit", "stop_loss")
        if price >= entry_price * (1 + cfg.take_profit_pct):
            return Action("exit", "take_profit")
        if row["trend_broken"]:
            return Action("exit", "trend_broken")
        return Action("hold")
    if row["entry"]:
        return Action("enter", "vwap_ema_long")
    return Action("hold")
