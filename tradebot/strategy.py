"""VWAP + EMA momentum strategy on 1-minute bars.

Long only (Alpaca doesn't allow shorting crypto).

Entry: the bar where "close > VWAP and fast EMA > slow EMA" first becomes true.
Exit:  stop loss, take profit, or the trend breaking (close < VWAP or fast EMA < slow EMA).

VWAP is anchored at 00:00 UTC each day, since crypto has no session open.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config


def add_indicators(bars: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    """bars: DataFrame indexed by UTC timestamp with open/high/low/close/volume columns."""
    df = bars.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("D")
    cum_pv = (typical * df["volume"]).groupby(day).cumsum()
    cum_v = df["volume"].groupby(day).cumsum()
    # Zero-volume bars at the start of a day leave VWAP undefined; fall back to price.
    df["vwap"] = (cum_pv / cum_v.replace(0, np.nan)).fillna(df["close"])
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["bullish"] = (df["close"] > df["vwap"]) & (df["ema_fast"] > df["ema_slow"])
    df["entry"] = df["bullish"] & ~df["bullish"].shift(1, fill_value=True)
    df["trend_broken"] = (df["close"] < df["vwap"]) | (df["ema_fast"] < df["ema_slow"])
    # Don't trust EMAs until they've had time to warm up.
    df.loc[df.index[: slow * 2], ["entry"]] = False
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
