import numpy as np
import pandas as pd

from tradebot.backtest import backtest
from tradebot.config import Config
from tradebot.strategy import add_dip, add_indicators, add_orb, decide


def make_bars(closes, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="1min", tz="UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": 1.0}, index=idx)


def cfg(**kw):
    c = Config(api_key="", secret_key="", asset_class="crypto")
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
    row = pd.Series({"close": 98.9, "exit_signal": False, "entry": False})
    assert decide(row, 100, c).reason == "stop_loss"
    row["close"] = 102.1
    assert decide(row, 100, c).reason == "take_profit"
    row["close"], row["exit_signal"] = 100.5, True
    assert decide(row, 100, c).reason == "trend_broken"
    row["exit_signal"] = False
    assert decide(row, 100, c).kind == "hold"


def test_backtest_charges_fees():
    closes = [100] * 60 + list(np.linspace(100, 104, 60))
    no_fee = backtest(make_bars(closes), cfg(fee_pct=0.0, take_profit_pct=0.02))
    with_fee = backtest(make_bars(closes), cfg(fee_pct=0.0025, take_profit_pct=0.02))
    assert len(no_fee) == len(with_fee) >= 1
    assert with_fee.pnl.sum() < no_fee.pnl.sum()


def test_backtest_stop_fills_at_stop_price_intrabar():
    closes = [100] * 60 + list(np.linspace(100, 101, 10)) + [101] * 5
    bars = make_bars(closes)
    bars.iloc[-3, bars.columns.get_loc("low")] = 90  # one bar wicks far below the stop
    trades = backtest(bars, cfg(fee_pct=0.0, stop_loss_pct=0.01, take_profit_pct=0.5))
    stop = trades[trades.reason == "stop_loss"].iloc[0]
    assert stop.exit == stop.entry * 0.99


def test_resample_to_5_minutes():
    from tradebot.backtest import resample
    bars = make_bars(range(100, 110))
    r = resample(bars, 5)
    assert len(r) == 2 and r["open"].iloc[0] == 100 and r["close"].iloc[0] == 104 and r["volume"].iloc[0] == 5


def test_trend_filter_blocks_entries_below_long_term_trend():
    # Long decline, then a bounce on a new UTC day (fresh VWAP) that stays below the long-term EMA.
    closes = list(np.linspace(120, 100, 300)) + list(np.linspace(100, 103, 40))
    bars = make_bars(closes, start="2026-01-01 19:00")
    assert add_indicators(bars, 9, 21)["entry"].iloc[300:].any()
    assert not add_indicators(bars, 9, 21, trend=200)["entry"].iloc[300:].any()


def test_stop_target_only_ignores_trend_breaks():
    df = add_indicators(make_bars(np.linspace(110, 100, 60)), 9, 21, exit_on_trend=False)
    assert not df["exit_signal"].any()


def stock_bars(closes, day="2026-03-02"):
    """1-minute bars for one regular session starting 9:30 New York time."""
    idx = pd.date_range(f"{day} 09:30", periods=len(closes), freq="1min", tz="America/New_York").tz_convert("UTC")
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.0005, "low": c * 0.9995, "close": c, "volume": 1.0}, index=idx)


def test_regular_hours_filter():
    from tradebot.session import regular_hours
    idx = pd.date_range("2026-03-02 09:00", "2026-03-02 16:30", freq="30min", tz="America/New_York")
    bars = pd.DataFrame({"close": 1.0}, index=idx.tz_convert("UTC"))
    kept = regular_hours(bars).index.tz_convert("America/New_York")
    assert kept[0].strftime("%H:%M") == "09:30" and kept[-1].strftime("%H:%M") == "15:30"


def test_stock_session_flags():
    from tradebot.session import add_session_flags
    df = add_session_flags(stock_bars([100] * 390), stocks=True)
    t = df.index.tz_convert("America/New_York").strftime("%H:%M")
    assert not df["can_enter"][t == "09:40"].any()
    assert df["can_enter"][t == "10:00"].all()
    assert not df["can_enter"][t == "15:30"].any()
    assert df["session_end"][t == "15:55"].all() and not df["session_end"][t == "15:54"].any()


def test_stock_positions_close_by_end_of_day():
    # Flat first hour, then a steady climb: no stop, target or trend exit, so the close must be end_of_day.
    closes = [100] * 60 + list(np.linspace(100, 100.9, 330))
    trades = backtest(stock_bars(closes), cfg(asset_class="stocks", take_profit_pct=0.5, stop_loss_pct=0.5,
                                              exit_on_trend=False))
    assert len(trades) >= 1
    assert trades.iloc[-1].reason == "end_of_day"
    assert trades.iloc[-1].exit_time.tz_convert("America/New_York").strftime("%H:%M") == "15:55"


def test_stock_vwap_resets_at_new_york_midnight():
    two_days = pd.concat([stock_bars([100] * 390, "2026-03-02"), stock_bars([200] * 390, "2026-03-03")])
    df = add_indicators(two_days, 9, 21, tz="America/New_York")
    assert df["vwap"].iloc[389] == 100 and df["vwap"].iloc[390] == 200


def test_orb_buys_first_breakout_only_with_range_low_stop():
    # First 15 min range 100-101, quiet, breakout at 10:00, a second push later.
    closes = [100, 101] * 7 + [100.5] + [100.5] * 15 + [101.5] + [100.8] * 20 + [102] + [101] * 330
    bars = stock_bars(closes)
    bars["high"], bars["low"] = bars["close"], bars["close"]
    df = add_orb(bars, 15)
    assert df["entry"].sum() == 1
    first = df.index[df["entry"]][0].tz_convert("America/New_York")
    assert first.strftime("%H:%M") == "10:00"
    assert df["or_high"].iloc[0] == 101 and df["stop_price"].iloc[-1] == 100


def test_orb_backtest_stops_at_range_low():
    closes = [100, 101] * 7 + [100.5] * 16 + [101.5] + [99] * 60 + [100] * 300
    bars = stock_bars(closes)
    bars["high"], bars["low"] = bars["close"], bars["close"]
    c = cfg(asset_class="stocks", strategy="orb", or_minutes=15, or_range_stop=True, take_profit_pct=1.0,
            stop_loss_pct=1.0, max_trades_per_day=1)
    t = backtest(bars, c)
    assert len(t) == 1 and t.iloc[0].reason == "stop_loss" and t.iloc[0].exit == 99  # bar closed below the stop


def test_dip_buys_below_band_and_exits_at_average():
    closes = [100, 100.2] * 15 + [97] + [98, 99, 100.2, 100.2]
    df = add_dip(make_bars(closes), length=20, k=2.0)
    assert df["entry"].sum() == 1 and df["entry"].iloc[30]
    assert df["exit_signal"].iloc[33]


def test_dip_trend_filter_blocks_dips_in_downtrends():
    closes = list(np.linspace(120, 100, 250)) + [95]
    assert add_dip(make_bars(closes), 20, 2.0)["entry"].iloc[-1]
    assert not add_dip(make_bars(closes), 20, 2.0, trend=200)["entry"].iloc[-1]
