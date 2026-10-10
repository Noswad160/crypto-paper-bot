# Paper day-trading bot (Alpaca)

Day-trades US stocks (default) or crypto on an **Alpaca paper account**. Paper mode is hard-coded in `tradebot/broker.py`.

## Strategy: VWAP + EMA momentum (long only)
- **VWAP** restarts each trading day (the stock market day in New York, or 00:00 UTC for crypto).
- **Entry:** on the bar where *close > VWAP and EMA(9) > EMA(21)* first becomes true. An optional long-term trend filter (`TREND_EMA`) also requires the close to be above a slow EMA.
- **Exit:** stop loss, take profit, or (if `EXIT_ON_TREND=1`) the trend breaking.
- **Stocks:** regular hours only. No entries before 9:45 or after 15:30 New York time, and everything is sold at 15:55. Nothing is held overnight.
- **Risk:** fixed $ size per trade, max trades per day, and a daily loss limit that sells everything and pauses until the next day.

## Why stocks
The crypto version lost money in every backtest: Alpaca's ~0.25% crypto fee per side cost more than the strategy made. Stock trades on Alpaca are commission-free, so the backtester only charges 0.02% per side for the spread.

**Pattern day trader rule:** US accounts under $25k are limited to 3 day trades per 5 business days, and paper accounts enforce it too. The default $100k paper account is above that limit.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # paste your PAPER keys from app.alpaca.markets
```

## Use
```bash
python -m tradebot.backtest --symbols SPY,QQQ --days 30           # one backtest
python -m tradebot.backtest --sweep --days 90 --text results.txt  # compare strategy variants
python -m tradebot.live --dry-run                                 # live signals only, no orders
python -m tradebot.live                                           # paper trade
pytest                                                            # tests
```
Set `ASSET_CLASS=crypto` (and crypto `SYMBOLS`) to trade crypto instead. All settings are listed in `.env.example`.

## Running on GitHub (no computer needed)
- **Paper trading bot** (`.github/workflows/bot.yml`) runs on weekdays around market hours. Add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` under **Settings → Secrets and variables → Actions**, then enable it in the **Actions** tab.
- **Backtest** (`.github/workflows/backtest.yml`): **Actions → Backtest → Run workflow.** It compares variants over the earlier two thirds of the period, then reports how they did in the most recent third, which they never saw.
- **Phone alerts:** install the **ntfy** app, subscribe to a hard-to-guess topic name, and add it as the secret `NTFY_TOPIC`. You get a push for buys, sells, the daily loss limit, errors and backtest results.

To stop the bot: **Actions → Paper trading bot → ⋯ → Disable workflow**. GitHub pauses schedules after 60 days without commits; re-enable from the Actions tab.
