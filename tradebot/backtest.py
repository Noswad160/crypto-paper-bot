"""Replay the strategy over historical 1-minute bars.

    python -m tradebot.backtest --symbol BTC/USD --days 14
    python -m tradebot.backtest --csv mybars.csv   # columns: timestamp,open,high,low,close,volume

Fills happen at the bar close that triggered the signal, minus fees on both sides.
"""
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import Config
from .strategy import add_indicators, decide


def backtest(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = add_indicators(bars, cfg.fast_ema, cfg.slow_ema)
    trades, entry, entry_time = [], None, None
    for ts, row in df.iterrows():
        action = decide(row, entry, cfg)
        if action.kind == "enter":
            entry, entry_time = row["close"], ts
        elif action.kind == "exit":
            gross = cfg.position_notional_usd * (row["close"] / entry - 1)
            fees = cfg.position_notional_usd * cfg.fee_pct * (1 + row["close"] / entry)
            trades.append({
                "entry_time": entry_time, "exit_time": ts, "entry": entry, "exit": row["close"],
                "reason": action.reason, "pnl": gross - fees,
            })
            entry = None
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No trades."
    wins = trades[trades.pnl > 0]
    equity = trades.pnl.cumsum()
    lines = [
        f"Trades:        {len(trades)}",
        f"Win rate:      {len(wins) / len(trades):.1%}",
        f"Net P&L:       ${trades.pnl.sum():,.2f}",
        f"Avg trade:     ${trades.pnl.mean():,.2f}",
        f"Max drawdown:  ${(equity.cummax() - equity).max():,.2f}",
        "Exit reasons:  " + ", ".join(f"{k}={v}" for k, v in trades.reason.value_counts().items()),
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--csv", help="load bars from a CSV instead of Alpaca")
    p.add_argument("--out", help="write the trade list to this CSV")
    args = p.parse_args()
    cfg = Config()

    if args.csv:
        bars = pd.read_csv(args.csv, parse_dates=["timestamp"], index_col="timestamp")
        bars.index = pd.to_datetime(bars.index, utc=True)
    else:
        from .data import CryptoData
        end = datetime.now(timezone.utc)
        bars = CryptoData(cfg.api_key, cfg.secret_key).minute_bars(args.symbol, end - timedelta(days=args.days), end)

    print(f"{len(bars)} bars, {bars.index[0]} -> {bars.index[-1]}")
    trades = backtest(bars, cfg)
    print(summarize(trades))
    if args.out and not trades.empty:
        trades.to_csv(args.out, index=False)
        print(f"Trades written to {args.out}")


if __name__ == "__main__":
    main()
