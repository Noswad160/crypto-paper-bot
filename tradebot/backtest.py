"""Replay the strategy over historical bars.

    python -m tradebot.backtest --symbols SPY --days 30
    python -m tradebot.backtest --sweep --days 90 --text results.txt   # compare strategy variants
    python -m tradebot.backtest --csv mybars.csv   # columns: timestamp,open,high,low,close,volume

Entries fill at the close of the signal bar. Stops and targets fill at their price as soon as a
later bar's low/high touches them (stop wins if both are touched in the same bar). Trend exits
and end-of-day exits fill at the bar close. Costs (FEE_PCT) are charged on both sides.
Stocks: only regular hours, entries 9:45-15:30 New York time, always flat by the close.
"""
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .config import Config
from .data import resample
from .session import add_session_flags, entry_start_for
from .strategy import EXIT_REASONS, signals

MIN_TRADES = 5  # a variant needs this many trades in the earlier period before its result counts


@dataclass(frozen=True)
class Variant:
    strategy: str
    bars: int  # bar size in minutes
    sl: float
    tp: float  # 1.0 (100%) = no target, hold until another exit
    max_per_day: int
    trend: int = 0  # long-term trend filter EMA length in bars, 0 = off
    exit_on_trend: bool = True  # vwap_ema: exit when the trend breaks
    or_minutes: int = 0  # orb: opening range length
    or_range_stop: bool = False  # orb: stop at the opening range low instead of sl
    band_k: float = 0.0  # dip: standard deviations below the average
    current: bool = False

    def label(self) -> str:
        target = "hold to close" if self.tp >= 1 else f"target {self.tp:.1%}"
        if self.strategy == "orb":
            stop = "stop at range low" if self.or_range_stop else f"stop {self.sl:.1%}"
            return f"ORB {self.or_minutes}min · {stop} · {target}"
        trend = " · trend filter" if self.trend else ""
        if self.strategy == "dip":
            return f"Dip {self.bars}m · {self.band_k:g} std{trend} · stop {self.sl:.0%}"
        exits = "trend exits" if self.exit_on_trend else "stop/target only"
        return f"VWAP/EMA {self.bars}m{trend} · {exits} · {self.sl:.1%}/{self.tp:.1%} · max {self.max_per_day}/day"

    def apply(self, cfg: Config) -> Config:
        return replace(
            cfg, strategy=self.strategy, bar_minutes=self.bars, trend_ema=self.trend, exit_on_trend=self.exit_on_trend,
            stop_loss_pct=self.sl, take_profit_pct=self.tp, max_trades_per_day=self.max_per_day,
            or_minutes=self.or_minutes or cfg.or_minutes, or_range_stop=self.or_range_stop,
            band_k=self.band_k or cfg.band_k,
        )


STRATEGY_NAMES = {"vwap_ema": "VWAP/EMA momentum", "orb": "Opening range breakout", "dip": "Dip buying"}
NO_TARGET = 1.0


def variants(cfg: Config) -> list[Variant]:
    """What the live bot does now, plus a grid of alternatives for each strategy."""
    current = Variant(cfg.strategy, cfg.bar_minutes, cfg.stop_loss_pct, cfg.take_profit_pct, cfg.max_trades_per_day,
                      cfg.trend_ema, cfg.exit_on_trend, cfg.or_minutes if cfg.strategy == "orb" else 0,
                      cfg.or_range_stop if cfg.strategy == "orb" else False,
                      cfg.band_k if cfg.strategy == "dip" else 0.0, current=True)
    if not cfg.stocks:
        others = [
            Variant("vwap_ema", bars, sl, tp, 2, trend, exit_on_trend)
            for bars in (15, 60) for trend in (0, 200) for exit_on_trend in (True, False)
            for sl, tp in ((0.015, 0.03), (0.025, 0.05), (0.04, 0.08))
        ]
    else:
        others = [
            Variant("vwap_ema", bars, sl, tp, 4, trend, exit_on_trend)
            for bars in (1, 5, 15) for trend in (0, 200) for exit_on_trend in (True, False)
            for sl, tp in ((0.005, 0.01), (0.01, 0.02), (0.02, 0.04))
        ] + [
            Variant("orb", 1, sl, tp, 1, exit_on_trend=False, or_minutes=m, or_range_stop=range_stop)
            for m in (5, 15, 30)
            for range_stop, sl in ((True, NO_TARGET), (False, 0.005), (False, 0.01))
            for tp in (NO_TARGET, 0.01, 0.02)
        ] + [
            Variant("dip", bars, sl, NO_TARGET, 2, trend, band_k=k)
            for bars in (5, 15) for k in (2.0, 2.5) for trend in (0, 200) for sl in (0.01, 0.02)
        ]
    return [current] + [v for v in others if replace(v, current=True) != current]


def backtest(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_session_flags(signals(bars, cfg), cfg.stocks, cfg.bar_minutes, entry_start_for(cfg.strategy))
    days = df.index.tz_convert(cfg.session_tz).date
    stops = df["stop_price"] if "stop_price" in df else pd.Series(np.nan, index=df.index)
    exit_reason = EXIT_REASONS[cfg.strategy]
    size = cfg.position_notional_usd
    trades, entry, entry_time, stop = [], None, None, None
    entries_per_day: dict = {}
    for ts, day, high, low, close, is_entry, exit_signal, can_enter, session_end, stop_price in zip(
        df.index, days, df["high"], df["low"], df["close"], df["entry"], df["exit_signal"],
        df["can_enter"], df["session_end"], stops,
    ):
        if entry is None:
            if is_entry and can_enter and entries_per_day.get(day, 0) < cfg.max_trades_per_day:
                entries_per_day[day] = entries_per_day.get(day, 0) + 1
                entry, entry_time = close, ts
                stop = stop_price if pd.notna(stop_price) else entry * (1 - cfg.stop_loss_pct)
            continue
        target = entry * (1 + cfg.take_profit_pct)
        if low <= stop:
            exit_price, reason = min(stop, close), "stop_loss"
        elif high >= target:
            exit_price, reason = target, "take_profit"
        elif exit_signal:
            exit_price, reason = close, exit_reason
        elif session_end:
            exit_price, reason = close, "end_of_day"
        else:
            continue
        gross = size * (exit_price / entry - 1)
        fees = size * cfg.fee_pct * (1 + exit_price / entry)
        trades.append({
            "entry_time": entry_time, "exit_time": ts, "entry": entry, "exit": exit_price,
            "reason": reason, "gross": gross, "fees": fees, "pnl": gross - fees,
        })
        entry = None
    return pd.DataFrame(trades)


def stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": 0.0, "net": 0.0, "avg": 0.0, "fees": 0.0, "max_dd": 0.0, "reasons": ""}
    equity = trades.pnl.cumsum()
    return {
        "trades": len(trades),
        "win_rate": (trades.pnl > 0).mean(),
        "net": trades.pnl.sum(),
        "avg": trades.pnl.mean(),
        "fees": trades.fees.sum(),
        "max_dd": (equity.cummax() - equity).max(),
        "reasons": ", ".join(f"{k}={v}" for k, v in trades.reason.value_counts().items()),
    }


def summarize(trades: pd.DataFrame) -> str:
    s = stats(trades)
    if not s["trades"]:
        return "No trades."
    return "\n".join([
        f"Trades:        {s['trades']}",
        f"Win rate:      {s['win_rate']:.1%}",
        f"Net P&L:       ${s['net']:,.2f}  (costs ${s['fees']:,.2f})",
        f"Avg trade:     ${s['avg']:,.2f}",
        f"Max drawdown:  ${s['max_dd']:,.2f}",
        f"Exit reasons:  {s['reasons']}",
    ])


def hold_pnl(bars: pd.DataFrame, cfg: Config) -> float:
    return cfg.position_notional_usd * (bars["close"].iloc[-1] / bars["close"].iloc[0] - 1) if len(bars) else 0.0


def rank_key(r: dict):
    return (r["early"]["trades"] >= MIN_TRADES, r["early"]["net"])


def sweep(bars_1m: pd.DataFrame, cfg: Config, cutoff: pd.Timestamp) -> list[dict]:
    """Run every variant. Trades are split by entry time into an earlier period, used to pick
    the best variant, and a recent period the pick never saw, used to check it."""
    rows = []
    for v in variants(cfg):
        trades = backtest(resample(bars_1m, v.bars), v.apply(cfg))
        early = trades[trades.entry_time < cutoff] if not trades.empty else trades
        late = trades[trades.entry_time >= cutoff] if not trades.empty else trades
        rows.append({"v": v, "all": stats(trades), "early": stats(early), "late": stats(late)})
    return sorted(rows, key=rank_key, reverse=True)


def combine(per_symbol: dict[str, list[dict]]) -> list[dict]:
    """Add each variant's results up across all symbols."""
    totals: dict[Variant, dict] = {}
    for rows in per_symbol.values():
        for r in rows:
            t = totals.setdefault(r["v"], {"v": r["v"], "early": {"net": 0.0, "trades": 0},
                                           "late": {"net": 0.0, "trades": 0}, "symbols_both": 0})
            for period in ("early", "late"):
                t[period]["net"] += r[period]["net"]
                t[period]["trades"] += r[period]["trades"]
            t["symbols_both"] += r["early"]["net"] > 0 and r["late"]["net"] > 0
    return sorted(totals.values(), key=rank_key, reverse=True)


def nums(r: dict) -> str:
    return (f"earlier ${r['early']['net']:+,.0f} ({r['early']['trades']}), "
            f"recent ${r['late']['net']:+,.0f} ({r['late']['trades']})")


def best_per_strategy(rows: list[dict]) -> dict[str, dict]:
    """Best variant of each strategy, picked on the earlier period (rows are already ranked)."""
    best: dict[str, dict] = {}
    for r in rows:
        if not r["v"].current:
            best.setdefault(r["v"].strategy, r)
    return best


def robust_count(rows: list[dict], strategy: str) -> str:
    group = [r for r in rows if r["v"].strategy == strategy and not r["v"].current]
    both = sum(r["early"]["net"] > 0 and r["late"]["net"] > 0 for r in group)
    return f"{both}/{len(group)}"


def symbol_text(symbol: str, early_hold: float, late_hold: float, rows: list[dict]) -> str:
    lines = [f"{symbol} (hold: earlier ${early_hold:+,.0f}, recent ${late_hold:+,.0f})"]
    for strategy, r in best_per_strategy(rows).items():
        lines.append(f"{r['v'].label()}: {nums(r)}")
    return "\n".join(lines) + "\n"


def overall_text(combined: list[dict], n_symbols: int, early_hold: float, late_hold: float) -> str:
    current = next(r for r in combined if r["v"].current)
    lines = [
        f"ALL {n_symbols} SYMBOLS COMBINED",
        f"Hold all: earlier ${early_hold:+,.0f}, recent ${late_hold:+,.0f}",
        f"Live bot now: {nums(current)}",
        "",
        "Best of each strategy (picked on earlier data):",
    ]
    for strategy, r in best_per_strategy(combined).items():
        lines.append(f"{STRATEGY_NAMES[strategy]}: {r['v'].label()}")
        lines.append(f"  {nums(r)}, works on {r['symbols_both']}/{n_symbols} symbols")
        lines.append(f"  versions profitable in both periods: {robust_count(combined, strategy)}")
    return "\n".join(lines) + "\n"


def markdown_table(title: str, rows: list[dict], extra_col: bool = False) -> str:
    head = "| Setup | Earlier trades | Earlier P&L | Recent trades | Recent P&L |" + (" Works on |" if extra_col else "")
    out = [f"### {title}", "", head, "|---" * (6 if extra_col else 5) + "|"]
    for r in rows:
        label = r["v"].label() + (" ← current" if r["v"].current else "")
        line = (f"| {label} | {r['early']['trades']} | ${r['early']['net']:+,.0f} | "
                f"{r['late']['trades']} | **${r['late']['net']:+,.0f}** |")
        out.append(line + (f" {r['symbols_both']} |" if extra_col else ""))
    return "\n".join(out) + "\n"


def load_bars(args, cfg: Config, symbol: str) -> pd.DataFrame:
    if args.csv:
        bars = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
        bars.index = pd.to_datetime(bars.index, utc=True)
        return bars
    from .data import MarketData
    end = datetime.now(timezone.utc)
    return MarketData(cfg.stocks, cfg.api_key, cfg.secret_key).minute_bars(symbol, end - timedelta(days=args.days), end)


def main():
    cfg = Config()
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default=",".join(cfg.symbols), help="comma-separated, e.g. SPY,QQQ")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--csv", help="load bars from a CSV instead of Alpaca")
    p.add_argument("--sweep", action="store_true", help="compare strategy variants")
    p.add_argument("--markdown", help="with --sweep, also write result tables to this file")
    p.add_argument("--text", help="with --sweep, also write a short plain-text summary to this file")
    p.add_argument("--out", help="write the trade list to this CSV")
    args = p.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    end = datetime.now(timezone.utc)
    cutoff = pd.Timestamp(end - timedelta(days=args.days / 3))  # last third of the window = "recent"
    per_symbol, text, md = {}, [], []
    early_hold_total = late_hold_total = 0.0
    for symbol in symbols:
        bars = load_bars(args, cfg, symbol)
        if bars.empty:
            print(f"\n{symbol}: no data")
            continue
        print(f"\n{symbol}: {len(bars)} bars, {bars.index[0]} -> {bars.index[-1]}")
        if not args.sweep:
            trades = backtest(bars, cfg)
            print(summarize(trades))
            if args.out and not trades.empty:
                trades.to_csv(args.out, index=False)
                print(f"Trades written to {args.out}")
            continue
        if args.csv:  # CSV data may not end today; split it two-thirds / one-third instead
            cutoff = bars.index[0] + (bars.index[-1] - bars.index[0]) * 2 / 3
        rows = sweep(bars, cfg, cutoff)
        per_symbol[symbol] = rows
        early_hold = hold_pnl(bars[bars.index < cutoff], cfg)
        late_hold = hold_pnl(bars[bars.index >= cutoff], cfg)
        early_hold_total += early_hold
        late_hold_total += late_hold
        text.append(symbol_text(symbol, early_hold, late_hold, rows))
        md.append(markdown_table(f"{symbol} (buy & hold: earlier ${early_hold:+,.0f}, recent ${late_hold:+,.0f})", rows))
        print(text[-1])

    if not args.sweep or not per_symbol:
        return
    combined = combine(per_symbol)
    overall = overall_text(combined, len(per_symbol), early_hold_total, late_hold_total)
    print(overall)
    header = (f"Last {args.days} days, after {cfg.fee_pct:.2%}/side costs. ${cfg.position_notional_usd:,.0f} per trade. "
              f"Dollar P&L, trade count in brackets. Recent = last {args.days // 3} days.\n\n")
    if args.text:
        with open(args.text, "w") as f:
            f.write(header + overall + "\n" + "\n".join(text))
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(f"## Backtest\n\n{header}" + markdown_table("All symbols combined", combined, extra_col=True)
                    + "\n" + "\n".join(md))


if __name__ == "__main__":
    main()
