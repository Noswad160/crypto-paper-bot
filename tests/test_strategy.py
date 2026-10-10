import numpy as np
import pandas as pd

from tradebot.backtest import backtest
from tradebot.config import Config
from tradebot.strategy import add_indicators, decide


def make_bars(closes, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": 1.0}, index=idx)


def cfg(**kw):
    c = Config(api_key="", secret_key="")
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_vwap_resets_each_utc_day():
    bars = make_bars([100] * 3 + [200] * 3, start="2026-01-01 23:57")
    df = add_indicators(bars, 2, 3)
    assert df["vwap"].iloc[2] == 100
    assert df["vwap"].iloc[3] == 200  # new day, fresh VWAP


def test_zero_volume_falls_back_to_close():
    bars = make_bars([100, 101])
    bars["volume"] = 0.0
    assert add_indicators(bars, 2, 3)["vwap"].tolist() == [100, 101]


def test_entry_after_downtrend_reverses():
    closes = list(np.linspace(100, 95, 120)) + list(np.linspace(95, 100, 60))
    df = add_indicators(make_bars(closes), 9, 21)
    assert df["entry"].sum() >= 1
    assert df.index[df["entry"]][0] > df.index[120]  # only fires once price turns up


def test_no_entries_during_warmup():
    df = add_indicators(make_bars(np.linspace(100, 110, 30)), 9, 21)
    assert not df["entry"].any()


def test_decide_exits():
    c = cfg(stop_loss_pct=0.01, take_profit_pct=0.02)
    row = pd.Series({"close": 98.9, "trend_broken": False, "entry": False})
    assert decide(row, 100, c).reason == "stop_loss"
    row["close"] = 102.1
    assert decide(row, 100, c).reason == "take_profit"
    row["close"], row["trend_broken"] = 100.5, True
    assert decide(row, 100, c).reason == "trend_broken"
    row["trend_broken"] = False
    assert decide(row, 100, c).kind == "hold"


def test_backtest_charges_fees():
    closes = [100] * 60 + list(np.linspace(100, 104, 60))
    no_fee = backtest(make_bars(closes), cfg(fee_pct=0.0, take_profit_pct=0.02))
    with_fee = backtest(make_bars(closes), cfg(fee_pct=0.0025, take_profit_pct=0.02))
    assert len(no_fee) == len(with_fee) >= 1
    assert with_fee.pnl.sum() < no_fee.pnl.sum()
