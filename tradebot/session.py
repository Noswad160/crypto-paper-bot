"""Regular US stock market hours and the bot's trading window within them."""
from datetime import time

import pandas as pd

TZ = "America/New_York"
OPEN, CLOSE = time(9, 30), time(16, 0)
ENTRY_START = time(9, 45)  # skip the noisy first 15 minutes
ENTRY_END = time(15, 30)  # no new positions in the last half hour
FLATTEN = time(15, 55)  # sell everything before the close


def entry_start_for(strategy: str) -> time:
    # The breakout strategy times its own entries off the opening range.
    return OPEN if strategy == "orb" else ENTRY_START


def regular_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Keep only bars that start between 9:30 and 16:00 New York time."""
    t = bars.index.tz_convert(TZ).time
    return bars[(t >= OPEN) & (t < CLOSE)]


def add_session_flags(df: pd.DataFrame, stocks: bool, bar_minutes: int = 1, entry_start: time = ENTRY_START) -> pd.DataFrame:
    """can_enter: a new position may be opened on this bar (from entry_start until ENTRY_END).
    session_end: the position must be closed at this bar's close (last bar of the day)."""
    if not stocks:
        df["can_enter"] = True
        df["session_end"] = False
        return df
    local = df.index.tz_convert(TZ)
    bar_end = (local + pd.Timedelta(minutes=bar_minutes)).time
    df["can_enter"] = (local.time >= entry_start) & (bar_end <= ENTRY_END)
    dates = pd.Series(local.date, index=df.index)
    df["session_end"] = (dates != dates.shift(-1)) | (bar_end > FLATTEN)
    return df
