"""Fetch 1-minute crypto bars from Alpaca (crypto market data doesn't need API keys)."""
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


class CryptoData:
    def __init__(self, api_key: str = "", secret_key: str = ""):
        self.client = CryptoHistoricalDataClient(api_key or None, secret_key or None)

    def minute_bars(self, symbol: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        req = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end)
        df = self.client.get_crypto_bars(req).df
        if df.empty:
            return df
        df = df.xs(symbol, level="symbol")
        return df[["open", "high", "low", "close", "volume"]].sort_index()

    def closed_bars_for_today(self, symbol: str, warmup_minutes: int = 120) -> pd.DataFrame:
        """Bars since UTC midnight (plus warm-up for the EMAs), dropping the still-forming bar."""
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        df = self.minute_bars(symbol, midnight - timedelta(minutes=warmup_minutes), now)
        if df.empty:
            return df
        # A bar stamped t covers [t, t+1min); it's only final once that minute has passed.
        return df[df.index + timedelta(minutes=1) <= now]
