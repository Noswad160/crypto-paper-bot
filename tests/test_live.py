"""Run the live bot's decision logic against a fake broker and fake data (no network)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

from tradebot import live
from tradebot.config import Config


class FakeBroker:
    def __init__(self, is_open=True, opens_in=timedelta(hours=10), entry=None):
        self.is_open, self.opens_in, self.entry = is_open, opens_in, entry
        self.bought, self.closed = [], []

    def clock(self):
        return SimpleNamespace(is_open=self.is_open, next_open=datetime.now(timezone.utc) + self.opens_in)

    def equity(self): return 100_000.0
    def day_start_equity(self): return 100_000.0
    def buys_since(self, since): return len(self.bought)
    def entry_price(self, symbol): return self.entry
    def has_open_order(self, symbol): return False

    def buy(self, symbol, notional):
        self.bought.append(symbol)
        self.entry = 100.0
        return SimpleNamespace(id="buy-1")

    def close(self, symbol):
        self.closed.append(symbol)
        self.entry = None
        return SimpleNamespace(id="sell-1")

    def close_all(self): pass


class FakeData:
    def __init__(self, bars): self.bars = bars
    def recent_closed_bars(self, symbol, bar_minutes, lookback): return self.bars


def make_bot(monkeypatch, tmp_path, broker, bars, **cfg_kw):
    monkeypatch.setattr(live, "LOG_DIR", tmp_path)
    monkeypatch.setattr(live, "notify", lambda *a, **k: None)
    bot = live.Bot.__new__(live.Bot)
    cfg = Config(api_key="k", secret_key="s", symbols=["SPY"], asset_class="stocks", **cfg_kw)
    bot.cfg, bot.dry_run, bot.broker, bot.data = cfg, False, broker, FakeData(bars)
    from zoneinfo import ZoneInfo
    bot.tz = ZoneInfo(cfg.session_tz)
    bot.day, bot.halted, bot.flattened, bot.was_open = None, False, False, False
    bot.last_entry_bar, bot.last_error_alert = {}, 0.0
    return bot


def bars_ending_now(closes):
    """1-minute bars whose last bar closed one minute ago, timestamped inside today's entry window."""
    end = pd.Timestamp.now(tz="UTC").floor("min") - pd.Timedelta(minutes=1)
    idx = pd.date_range(end=end, periods=len(closes), freq="1min")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)


def test_market_gate(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path, FakeBroker(is_open=False, opens_in=timedelta(hours=10)), None)
    assert bot.market_gate() == "exit"
    bot.broker.opens_in = timedelta(minutes=20)
    assert bot.market_gate() == "wait"
    bot.broker.is_open = True
    assert bot.market_gate() in ("trade", "flatten")
    bot.broker.is_open = False
    assert bot.market_gate() == "exit"  # it was open earlier this run, so the day is over


def test_flatten_sells_open_positions_once(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path, FakeBroker(entry=100.0), None)
    bot.flatten()
    bot.flatten()
    assert bot.broker.closed == ["SPY"]


def test_step_buys_on_fresh_signal_inside_window(monkeypatch, tmp_path):
    closes = [100] * 60 + [100.5]  # flat, then a pop above VWAP with EMAs turning up
    bot = make_bot(monkeypatch, tmp_path, FakeBroker(), bars_ending_now(closes))
    monkeypatch.setattr(live, "add_session_flags", lambda df, stocks, m, start: df.assign(can_enter=True, session_end=False))
    bot.step("SPY")
    assert bot.broker.bought == ["SPY"]
    bot.broker.entry = None
    bot.step("SPY")  # same bar again: must not buy twice
    assert bot.broker.bought == ["SPY"]


def test_step_skips_entry_outside_window(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path, FakeBroker(), bars_ending_now([100] * 60 + [100.5]))
    monkeypatch.setattr(live, "add_session_flags", lambda df, stocks, m, start: df.assign(can_enter=False, session_end=False))
    bot.step("SPY")
    assert bot.broker.bought == []


def test_step_sells_on_stop(monkeypatch, tmp_path):
    bot = make_bot(monkeypatch, tmp_path, FakeBroker(entry=100.0), bars_ending_now([100] * 60 + [98.0]))
    bot.step("SPY")
    assert bot.broker.closed == ["SPY"]
