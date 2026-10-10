"""Live paper-trading loop.

    python -m tradebot.live            # trade on your Alpaca paper account
    python -m tradebot.live --dry-run  # log signals only, place no orders
"""
import argparse
import csv
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .broker import PaperBroker
from .config import Config
from .data import CryptoData
from .notify import notify
from .strategy import add_indicators, decide

LOG_DIR = Path("logs")
log = logging.getLogger("tradebot")


def setup_logging():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(LOG_DIR / "bot.log")],
    )


def journal(row: dict):
    path = LOG_DIR / "trades.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


class Bot:
    def __init__(self, cfg: Config, dry_run: bool):
        self.cfg = cfg
        self.dry_run = dry_run
        self.data = CryptoData(cfg.api_key, cfg.secret_key)
        self.broker = PaperBroker(cfg.api_key, cfg.secret_key)
        self.day = None
        self.halted = False
        self.last_entry_bar = {}  # symbol -> timestamp of the bar we last bought on
        self.last_error_alert = 0.0

    def roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.day:
            self.day = today
            self.halted = False
            log.info("UTC day %s, equity at last daily close $%.2f", today, self.broker.day_start_equity())

    def trades_today(self) -> int:
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return self.broker.buys_since(midnight)

    def check_daily_loss(self) -> bool:
        # Both numbers come from Alpaca, so the limit still holds after a restart.
        pnl = self.broker.equity() - self.broker.day_start_equity()
        if pnl <= -self.cfg.max_daily_loss_usd and not self.halted:
            log.warning("Daily loss limit hit (%.2f). Flattening and halting until next UTC day.", pnl)
            if not self.dry_run:
                self.broker.close_all()
            self.halted = True
            notify("Daily loss limit hit", f"Down ${-pnl:,.2f} today. Sold everything; paused until 00:00 UTC.",
                   priority="high", tags="rotating_light")
        return self.halted

    def step(self, symbol: str):
        bars = self.data.closed_bars_for_today(symbol, warmup_minutes=max(self.cfg.slow_ema * 6, self.cfg.trend_ema * 3))
        if len(bars) < self.cfg.slow_ema * 2:
            log.info("%s: not enough bars yet (%d)", symbol, len(bars))
            return
        last = add_indicators(
            bars, self.cfg.fast_ema, self.cfg.slow_ema, self.cfg.trend_ema, self.cfg.exit_on_trend
        ).iloc[-1]
        entry = self.broker.entry_price(symbol)
        action = decide(last, entry, self.cfg)
        log.info(
            "%s close=%.2f vwap=%.2f ema%d=%.2f ema%d=%.2f pos=%s -> %s %s",
            symbol, last["close"], last["vwap"], self.cfg.fast_ema, last["ema_fast"],
            self.cfg.slow_ema, last["ema_slow"], entry, action.kind, action.reason,
        )
        if action.kind == "hold":
            return
        if action.kind == "enter":
            # Quiet markets can go minutes without a new bar. Only buy on a fresh bar, once.
            bar_time = last.name
            if datetime.now(timezone.utc) - bar_time > timedelta(minutes=2):
                log.info("%s: entry signal is on a stale bar (%s), skipping", symbol, bar_time)
                return
            if self.last_entry_bar.get(symbol) == bar_time:
                return
        if action.kind == "enter" and self.trades_today() >= self.cfg.max_trades_per_day:
            log.info("%s: max trades for today reached, skipping entry", symbol)
            return
        if self.dry_run or self.broker.has_open_order(symbol):
            return

        if action.kind == "enter":
            order = self.broker.buy(symbol, self.cfg.position_notional_usd)
            self.last_entry_bar[symbol] = last.name
            notify(f"Bought {symbol}", f"${self.cfg.position_notional_usd:,.0f} at ~${last['close']:,.2f}", tags="chart_with_upwards_trend")
        else:
            order = self.broker.close(symbol)
            pct = last["close"] / entry - 1
            notify(
                f"Sold {symbol} ({action.reason.replace('_', ' ')})",
                f"~${last['close']:,.2f}, entry ${entry:,.2f}: {pct:+.2%} (~${pct * self.cfg.position_notional_usd:+,.2f} before fees)",
                tags="white_check_mark" if pct > 0 else "x",
            )
        journal({
            "time": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "action": action.kind,
            "reason": action.reason, "bar_close": last["close"], "entry_price": entry, "order_id": order.id,
        })

    def run(self):
        log.info("Starting bot (%s) on %s", "DRY RUN" if self.dry_run else "PAPER", ", ".join(self.cfg.symbols))
        notify("Bot started", "Watching " + ", ".join(self.cfg.symbols), priority="min", tags="robot")
        while True:
            try:
                self.roll_day()
                if not self.check_daily_loss():
                    for symbol in self.cfg.symbols:
                        self.step(symbol)
            except Exception as e:
                log.exception("Loop error; retrying next cycle")
                if time.time() - self.last_error_alert > 1800:  # at most one error alert per 30 min
                    self.last_error_alert = time.time()
                    notify("Trading bot error", f"{type(e).__name__}: {e}"[:300], priority="high", tags="warning")
            # Wake a few seconds after each minute boundary so the last bar is final.
            time.sleep(self.cfg.poll_seconds - time.time() % self.cfg.poll_seconds + 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="log signals without placing orders")
    args = p.parse_args()
    setup_logging()
    try:
        Bot(Config(), args.dry_run).run()
    except KeyboardInterrupt:
        log.info("Stopped. Open positions are left as-is; close them in the Alpaca dashboard if you like.")


if __name__ == "__main__":
    main()
