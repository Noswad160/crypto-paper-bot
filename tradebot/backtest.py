"""Replay the strategy over historical bars.

    python -m tradebot.backtest --symbols BTC/USD --days 14
    python -m tradebot.backtest --sweep --days 30 --markdown results.md   # compare settings
    python -m tradebot.backtest --csv mybars.csv   # columns: timestamp,open,high,low,close,volume

Entries fill at the close of the signal bar. Stops and targets fill at their price as soon as a
later bar's low/high touches them (stop wins if both are touched in the same bar). Trend exits
fill at the bar close. Fees are charged on both sides.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import Config
from .strategy import add_indicators

# (bar minutes, stop loss, take profit) combinations tried by --sweep.
SWEEP = [
    (bars, sl, tp)
    for bars in (1, 5, 15)
    for sl, tp in ((0.0075, 0.015), (0.01, 0.02), (0.015, 0.03))
]


def resample(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return bars
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return bars.resample(f"{minutes}min", label="left", closed="left").agg(agg).dropna()


def backtest(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_indicators(bars, cfg.fast_ema, cfg.slow_ema)
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


def sweep(bars_1m: pd.DataFrame, cfg: Config) -> list[dict]:
    rows = []
    for minutes, sl, tp in SWEEP:
        c = replace(cfg, stop_loss_pct=sl, take_profit_pct=tp)
        s = stats(backtest(resample(bars_1m, minutes), c))
        current = minutes == 1 and sl == cfg.stop_loss_pct and tp == cfg.take_profit_pct
        rows.append({"bars": minutes, "sl": sl, "tp": tp, "current": current, **s})
    return rows


def sweep_markdown(symbol: str, bars_1m: pd.DataFrame, rows: list[dict], cfg: Config) -> str:
    hold = cfg.position_notional_usd * (bars_1m["close"].iloc[-1] / bars_1m["close"].iloc[0] - 1)
    best = max(rows, key=lambda r: r["net"])
    out = [
        f"### {symbol}",
        f"{bars_1m.index[0]:%b %d} → {bars_1m.index[-1]:%b %d %H:%M} UTC · "
        f"buy & hold ${cfg.position_notional_usd:,.0f} would have made **${hold:+,.2f}**",
        "",
        "| Bars | Stop | Target | Trades | Win % | Net P&L | Fees | Max drawdown |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        label = f"{r['bars']} min" + (" ← current" if r["current"] else "") + (" ⭐" if r is best else "")
        out.append(
            f"| {label} | {r['sl']:.2%} | {r['tp']:.1%} | {r['trades']} | {r['win_rate']:.0%} | "
            f"**${r['net']:+,.2f}** | ${r['fees']:,.2f} | ${r['max_dd']:,.2f} |"
        )
    return "\n".join(out) + "\n"


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
    p.add_argument("--out", help="write the trade list to this CSV")
    args = p.parse_args()
    cfg = Config()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    md = [
        f"## Backtest: last {args.days} days\n",
        f"${cfg.position_notional_usd:,.0f} per trade, {cfg.fee_pct:.2%} fee per side, "
        f"EMA {cfg.fast_ema}/{cfg.slow_ema}, max {cfg.max_trades_per_day} trades/day. "
        "Past results don't predict future ones.\n",
    ]
    for symbol in symbols:
        bars = load_bars(args, cfg, symbol)
        print(f"\n{symbol}: {len(bars)} bars, {bars.index[0]} -> {bars.index[-1]}")
        if args.sweep:
            rows = sweep(bars, cfg)
            table = sweep_markdown(symbol, bars, rows, cfg)
            print(table)
            md.append(table)
        else:
            trades = backtest(bars, cfg)
            print(summarize(trades))
            if args.out and not trades.empty:
                trades.to_csv(args.out, index=False)
                print(f"Trades written to {args.out}")

    if args.sweep and args.markdown:
        with open(args.markdown, "w") as f:
            f.write("\n".join(md))


if __name__ == "__main__":
    main()
