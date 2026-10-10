"""Replay the strategy over historical bars.

    python -m tradebot.backtest --symbols BTC/USD --days 14
    python -m tradebot.backtest --sweep --days 90 --markdown results.md   # compare strategy variants
    python -m tradebot.backtest --csv mybars.csv   # columns: timestamp,open,high,low,close,volume

Entries fill at the close of the signal bar. Stops and targets fill at their price as soon as a
later bar's low/high touches them (stop wins if both are touched in the same bar). Trend exits
fill at the bar close. Fees are charged on both sides.
"""
import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import Config
from .strategy import add_indicators

@dataclass(frozen=True)
class Variant:
    bars: int  # bar size in minutes
    trend: int  # long-term trend filter EMA length in bars, 0 = off
    exit_on_trend: bool  # exit when VWAP/EMA trend breaks, or only on stop/target
    sl: float
    tp: float
    max_per_day: int
    current: bool = False

    def label(self) -> str:
        parts = [f"{self.bars}m", "trend filter" if self.trend else "no filter",
                 "trend exits" if self.exit_on_trend else "stop/target only",
                 f"{self.sl:.1%}/{self.tp:.1%}", f"max {self.max_per_day}/day"]
        return " · ".join(parts)

    def apply(self, cfg: Config) -> Config:
        return replace(cfg, trend_ema=self.trend, exit_on_trend=self.exit_on_trend,
                       stop_loss_pct=self.sl, take_profit_pct=self.tp, max_trades_per_day=self.max_per_day)


def variants(cfg: Config) -> list[Variant]:
    """What the live bot does now, plus slower, pickier ideas meant to trade less and aim for bigger moves."""
    current = Variant(1, cfg.trend_ema, cfg.exit_on_trend, cfg.stop_loss_pct, cfg.take_profit_pct,
                      cfg.max_trades_per_day, current=True)
    return [current] + [
        Variant(bars, trend, exit_on_trend, sl, tp, 2)
        for bars in (15, 60)
        for trend in (0, 200)
        for exit_on_trend in (True, False)
        for sl, tp in ((0.015, 0.03), (0.025, 0.05), (0.04, 0.08))
    ]


def resample(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return bars
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return bars.resample(f"{minutes}min", label="left", closed="left").agg(agg).dropna()


def backtest(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_indicators(bars, cfg.fast_ema, cfg.slow_ema, cfg.trend_ema, cfg.exit_on_trend)
    size = cfg.position_notional_usd
    trades, entry, entry_time = [], None, None
    entries_per_day: dict = {}
    for ts, high, low, close, is_entry, broken in zip(
        df.index, df["high"], df["low"], df["close"], df["entry"], df["trend_broken"]
    ):
        if entry is None:
            day = ts.date()
            if is_entry and entries_per_day.get(day, 0) < cfg.max_trades_per_day:
                entries_per_day[day] = entries_per_day.get(day, 0) + 1
                entry, entry_time = close, ts
            continue
        stop, target = entry * (1 - cfg.stop_loss_pct), entry * (1 + cfg.take_profit_pct)
        if low <= stop:
            exit_price, reason = stop, "stop_loss"
        elif high >= target:
            exit_price, reason = target, "take_profit"
        elif broken:
            exit_price, reason = close, "trend_broken"
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
        f"Net P&L:       ${s['net']:,.2f}  (fees ${s['fees']:,.2f})",
        f"Avg trade:     ${s['avg']:,.2f}",
        f"Max drawdown:  ${s['max_dd']:,.2f}",
        f"Exit reasons:  {s['reasons']}",
    ])


MIN_TRADES = 5


def hold_pnl(bars: pd.DataFrame, cfg: Config) -> float:
    return cfg.position_notional_usd * (bars["close"].iloc[-1] / bars["close"].iloc[0] - 1)


def sweep(bars_1m: pd.DataFrame, cfg: Config, split: float = 2 / 3) -> tuple[list[dict], pd.Timestamp]:
    """Run every variant. Trades are split by entry time into an earlier period, used to pick
    the best variant, and a recent period the pick never saw, used to check it."""
    cutoff = bars_1m.index[0] + (bars_1m.index[-1] - bars_1m.index[0]) * split
    rows = []
    for v in variants(cfg):
        trades = backtest(resample(bars_1m, v.bars), v.apply(cfg))
        early = trades[trades.entry_time < cutoff] if not trades.empty else trades
        late = trades[trades.entry_time >= cutoff] if not trades.empty else trades
        rows.append({"v": v, "all": stats(trades), "early": stats(early), "late": stats(late)})
    # A variant needs a handful of trades before its result means anything.
    rows.sort(key=lambda r: (r["early"]["trades"] >= MIN_TRADES, r["early"]["net"]), reverse=True)
    return rows, cutoff


def sweep_markdown(symbol: str, bars_1m: pd.DataFrame, rows: list[dict], cutoff, cfg: Config) -> str:
    early_hold = hold_pnl(bars_1m[bars_1m.index < cutoff], cfg)
    late_hold = hold_pnl(bars_1m[bars_1m.index >= cutoff], cfg)
    out = [
        f"### {symbol}",
        f"Earlier period: {bars_1m.index[0]:%b %d} → {cutoff:%b %d} (buy & hold **${early_hold:+,.0f}**). "
        f"Recent period: {cutoff:%b %d} → {bars_1m.index[-1]:%b %d} (buy & hold **${late_hold:+,.0f}**).",
        "",
        f"Sorted by the earlier period (setups with under {MIN_TRADES} earlier trades go last). "
        "The recent column is the honest check.",
        "",
        "| Setup | Earlier trades | Earlier P&L | Recent trades | Recent P&L | Win % | Fees |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        v, e, l, a = r["v"], r["early"], r["late"], r["all"]
        label = v.label() + (" ← current" if v.current else "")
        out.append(f"| {label} | {e['trades']} | ${e['net']:+,.0f} | {l['trades']} | **${l['net']:+,.0f}** | "
                   f"{a['win_rate']:.0%} | ${a['fees']:,.0f} |")
    return "\n".join(out) + "\n"


def sweep_text(symbol: str, bars_1m: pd.DataFrame, rows: list[dict], cutoff, cfg: Config) -> str:
    """Short plain-text version of the sweep, sized for a phone notification."""
    early_hold = hold_pnl(bars_1m[bars_1m.index < cutoff], cfg)
    late_hold = hold_pnl(bars_1m[bars_1m.index >= cutoff], cfg)
    current = next(r for r in rows if r["v"].current)
    both = sum(r["early"]["net"] > 0 and r["late"]["net"] > 0 for r in rows if not r["v"].current)

    def nums(r):
        return (f"earlier ${r['early']['net']:+,.0f} ({r['early']['trades']}), "
                f"recent ${r['late']['net']:+,.0f} ({r['late']['trades']})")

    lines = [
        f"{symbol}",
        f"Hold: earlier ${early_hold:+,.0f}, recent ${late_hold:+,.0f}",
        f"Current bot: {nums(current)}",
        "Top 3 (picked on earlier data):",
    ]
    for i, r in enumerate([r for r in rows if not r["v"].current][:3], 1):
        lines.append(f"{i}. {r['v'].label()}: {nums(r)}")
    lines.append(f"Profitable in both periods: {both} of {len(rows) - 1}")
    return "\n".join(lines) + "\n"


def load_bars(args, cfg: Config, symbol: str) -> pd.DataFrame:
    if args.csv:
        bars = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
        bars.index = pd.to_datetime(bars.index, utc=True)
        return bars
    from .data import CryptoData
    end = datetime.now(timezone.utc)
    return CryptoData(cfg.api_key, cfg.secret_key).minute_bars(symbol, end - timedelta(days=args.days), end)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", default="BTC/USD", help="comma-separated, e.g. BTC/USD,ETH/USD")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--csv", help="load bars from a CSV instead of Alpaca")
    p.add_argument("--sweep", action="store_true", help="compare bar sizes and stop/target settings")
    p.add_argument("--markdown", help="with --sweep, also write the results table to this file")
    p.add_argument("--text", help="with --sweep, also write a short plain-text summary to this file")
    p.add_argument("--out", help="write the trade list to this CSV")
    args = p.parse_args()
    cfg = Config()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    md = [
        f"## Backtest: last {args.days} days\n",
        f"${cfg.position_notional_usd:,.0f} per trade, {cfg.fee_pct:.2%} fee per side, "
        f"EMA {cfg.fast_ema}/{cfg.slow_ema}, trend filter EMA 200 bars. "
        "Past results don't predict future ones.\n",
    ]
    text = []
    for symbol in symbols:
        bars = load_bars(args, cfg, symbol)
        print(f"\n{symbol}: {len(bars)} bars, {bars.index[0]} -> {bars.index[-1]}")
        if args.sweep:
            rows, cutoff = sweep(bars, cfg)
            table = sweep_markdown(symbol, bars, rows, cutoff, cfg)
            print(table)
            md.append(table)
            text.append(sweep_text(symbol, bars, rows, cutoff, cfg))
        else:
            trades = backtest(bars, cfg)
            print(summarize(trades))
            if args.out and not trades.empty:
                trades.to_csv(args.out, index=False)
                print(f"Trades written to {args.out}")

    if args.sweep and args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(md))
    if args.sweep and args.text:
        with open(args.text, "w") as f:
            f.write(f"Last {args.days} days, after fees. Dollar P&L, trade count in brackets.\n\n" + "\n".join(text))


if __name__ == "__main__":
    main()
