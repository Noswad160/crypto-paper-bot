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
from zoneinfo import ZoneInfo

from .broker import PaperBroker
from .config import Config
from .data import MarketData
from .notify import notify
from .session import FLATTEN, add_session_flags
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
        self.data = MarketData(cfg.stocks, cfg.api_key, cfg.secret_key)
        self.broker = PaperBroker(cfg.api_key, cfg.secret_key)
        self.tz = ZoneInfo(cfg.session_tz)
        self.day = None
        self.halted = False
        self.flattened = False
        self.was_open = False
        self.last_entry_bar = {}  # symbol -> timestamp of the bar we last bought on
        self.last_error_alert = 0.0

    def local_now(self) -> datetime:
        return datetime.now(self.tz)

    def roll_day(self):
        today = self.local_now().date()
        if today != self.day:
            self.day = today
            self.halted = False
            self.flattened = False
            log.info("Trading day %s, equity at last daily close $%.2f", today, self.broker.day_start_equity())

    def trades_today(self) -> int:
        midnight = self.local_now().replace(hour=0, minute=0, second=0, microsecond=0)
        return self.broker.buys_since(midnight.astimezone(timezone.utc))

    def check_daily_loss(self) -> bool:
        # Both numbers come from Alpaca, so the limit still holds after a restart.
        pnl = self.broker.equity() - self.broker.day_start_equity()
        if pnl <= -self.cfg.max_daily_loss_usd and not self.halted:
            log.warning("Daily loss limit hit (%.2f). Flattening and halting until tomorrow.", pnl)
            if not self.dry_run:
                self.broker.close_all()
            self.halted = True
            notify("Daily loss limit hit", f"Down ${-pnl:,.2f} today. Sold everything; paused until tomorrow.",
                   priority="high", tags="rotating_light")
        return self.halted

    def flatten(self):
        """Sell everything before the stock market closes. Day trades don't hold overnight."""
        if self.flattened:
            return
        self.flattened = True
        for symbol in self.cfg.symbols:
            entry = self.broker.entry_price(symbol)
            if entry is None or self.dry_run:
                continue
            order = self.broker.close(symbol)
            log.info("%s: end of day, closing position", symbol)
            notify(f"Sold {symbol} (end of day)", f"Entry ${entry:,.2f}. Flat for the night.", tags="sunset")
            journal({
                "time": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "action": "exit",
                "reason": "end_of_day", "bar_close": "", "entry_price": entry, "order_id": order.id,
            })

    def lookback(self) -> timedelta:
        """How far back to fetch so the EMAs are warmed up and today's VWAP is complete."""
        warmup_minutes = max(self.cfg.slow_ema * 3, self.cfg.trend_ema * 2) * self.cfg.bar_minutes
        if self.cfg.stocks:
            # ~390 trading minutes per day, plus room for weekends and holidays.
            return timedelta(days=-(-warmup_minutes // 390) * 2 + 4)
        return timedelta(minutes=warmup_minutes) + timedelta(days=1)

    def step(self, symbol: str):
        cfg = self.cfg
        bars = self.data.recent_closed_bars(symbol, cfg.bar_minutes, self.lookback())
        if len(bars) < cfg.slow_ema * 2:
            log.info("%s: not enough bars yet (%d)", symbol, len(bars))
            return
        df = add_indicators(bars, cfg.fast_ema, cfg.slow_ema, cfg.trend_ema, cfg.exit_on_trend, cfg.session_tz)
        last = add_session_flags(df, cfg.stocks, cfg.bar_minutes).iloc[-1]
        entry = self.broker.entry_price(symbol)
        action = decide(last, entry, cfg)
        log.info(
            "%s close=%.2f vwap=%.2f ema%d=%.2f ema%d=%.2f pos=%s -> %s %s",
            symbol, last["close"], last["vwap"], cfg.fast_ema, last["ema_fast"],
            cfg.slow_ema, last["ema_slow"], entry, action.kind, action.reason,
        )
        if action.kind == "hold":
            return
        if action.kind == "enter":
            if not last["can_enter"]:
                log.info("%s: outside the entry window, skipping", symbol)
                return
            # Quiet markets can go minutes without a new bar. Only buy on a fresh bar, once.
            bar_end = last.name + timedelta(minutes=cfg.bar_minutes)
            if datetime.now(timezone.utc) - bar_end > timedelta(minutes=2):
                log.info("%s: entry signal is on a stale bar (%s), skipping", symbol, last.name)
                return
            if self.last_entry_bar.get(symbol) == last.name:
                return
            if self.trades_today() >= cfg.max_trades_per_day:
                log.info("%s: max trades for today reached, skipping entry", symbol)
                return
        if self.dry_run or self.broker.has_open_order(symbol):
            return

        if action.kind == "enter":
            order = self.broker.buy(symbol, cfg.position_notional_usd)
            self.last_entry_bar[symbol] = last.name
            notify(f"Bought {symbol}", f"${cfg.position_notional_usd:,.0f} at ~${last['close']:,.2f}", tags="chart_with_upwards_trend")
        else:
            order = self.broker.close(symbol)
            pct = last["close"] / entry - 1
            notify(
                f"Sold {symbol} ({action.reason.replace('_', ' ')})",
                f"~${last['close']:,.2f}, entry ${entry:,.2f}: {pct:+.2%} (~${pct * cfg.position_notional_usd:+,.2f})",
                tags="white_check_mark" if pct > 0 else "x",
            )
        journal({
            "time": datetime.now(timezone.utc).isoformat(), "symbol": symbol, "action": action.kind,
            "reason": action.reason, "bar_close": last["close"], "entry_price": entry, "order_id": order.id,
        })

    def market_gate(self) -> str:
        """For stocks: 'trade', 'flatten', 'wait' (opens soon) or 'exit' (closed for a while)."""
        if not self.cfg.stocks:
            return "trade"
        clock = self.broker.clock()
        if clock.is_open:
            self.was_open = True
            return "flatten" if self.local_now().time() >= FLATTEN else "trade"
        if self.was_open or clock.next_open - datetime.now(timezone.utc) > timedelta(hours=1):
            return "exit"
        return "wait"

    def run(self):
        mode = "DRY RUN" if self.dry_run else "PAPER"
        if self.market_gate() == "exit":
            log.info("Market closed and not opening within the hour. Exiting.")
            return
        log.info("Starting bot (%s) on %s", mode, ", ".join(self.cfg.symbols))
        notify("Bot started", "Watching " + ", ".join(self.cfg.symbols), priority="min", tags="robot")
        while True:
            try:
                gate = self.market_gate()
                if gate == "exit":
                    log.info("Market closed. Done for today.")
                    return
                if gate == "trade":
                    self.roll_day()
                    if not self.check_daily_loss():
                        for symbol in self.cfg.symbols:
                            self.step(symbol)
                elif gate == "flatten":
                    self.roll_day()
                    self.flatten()
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
