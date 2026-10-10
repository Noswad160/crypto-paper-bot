"""Fetch minute bars from Alpaca for stocks or crypto."""
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from .session import regular_hours


def resample(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Combine 1-minute bars into N-minute bars (bins start on the hour, so 9:30 lines up for N <= 30)."""
    if minutes == 1:
        return bars
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return bars.resample(f"{minutes}min", label="left", closed="left").agg(agg).dropna()


class MarketData:
    def __init__(self, stocks: bool, api_key: str = "", secret_key: str = ""):
        self.stocks = stocks
        if stocks:
            # Stock data needs (free) API keys; crypto data doesn't.
            self.client = StockHistoricalDataClient(api_key, secret_key)
        else:
            self.client = CryptoHistoricalDataClient(api_key or None, secret_key or None)

    def minute_bars(self, symbol: str, start: datetime, end: datetime | None = None, live: bool = False) -> pd.DataFrame:
        if self.stocks:
            # The free plan gets real-time data from the IEX exchange only. Full-market (SIP) data is
            # free for anything older than 15 minutes, so the backtester uses that.
            feed = DataFeed.IEX if live else DataFeed.SIP
            if not live:
                end = min(end or datetime.now(timezone.utc), datetime.now(timezone.utc) - timedelta(minutes=16))
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end, feed=feed)
            df = self.client.get_stock_bars(req).df
        else:
            req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end)
            df = self.client.get_crypto_bars(req).df
        if df.empty:
            return df
        df = df.xs(symbol, level="symbol")[["open", "high", "low", "close", "volume"]].sort_index()
        return regular_hours(df) if self.stocks else df

    def recent_closed_bars(self, symbol: str, bar_minutes: int, lookback: timedelta) -> pd.DataFrame:
        """Recent bars resampled to bar_minutes, dropping the bar that is still forming."""
        now = datetime.now(timezone.utc)
        df = resample(self.minute_bars(symbol, now - lookback, now, live=True), bar_minutes)
        if df.empty:
            return df
        # A bar stamped t covers [t, t + N min); it's only final once that time has passed.
        return df[df.index + timedelta(minutes=bar_minutes) <= now]
