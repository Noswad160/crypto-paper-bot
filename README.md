# Crypto paper day-trading bot (Alpaca)

Trades BTC/USD and ETH/USD on 1-minute bars in an **Alpaca paper account**. Paper mode is hard-coded in `tradebot/broker.py`.

## Strategy: VWAP + EMA momentum (long only)
- **VWAP** is anchored at 00:00 UTC each day.
- **Entry:** on the bar where *close > VWAP and EMA(9) > EMA(21)* first becomes true.
- **Exit:** stop loss (-0.75%), take profit (+1.5%), or the trend breaking (close < VWAP or EMA9 < EMA21).
- **Risk:** fixed $ size per trade, max trades per UTC day, and a daily loss limit that flattens everything and pauses trading until the next UTC day.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # paste your PAPER keys from app.alpaca.markets
```

## Use
```bash
python -m tradebot.backtest --symbol BTC/USD --days 14   # try it on history first (no keys needed)
python -m tradebot.live --dry-run                         # live signals only, no orders
python -m tradebot.live                                   # paper trade
pytest                                                    # unit tests
```
Logs go to `logs/bot.log`, and every order is recorded in `logs/trades.csv`.

## Notes
- Alpaca charges crypto fees (~0.25% per side at the base tier). The backtest includes them via `FEE_PCT`, and on 1-minute bars fees make a big difference.
- Stops are checked on each closed 1-minute bar, so live fills can land past the stop level.
- All settings can be changed in `.env` (see `.env.example`).

## Running on GitHub (no computer needed)
`.github/workflows/bot.yml` runs the bot on GitHub Actions.
1. In the repo, open **Settings → Secrets and variables → Actions** and add `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` (your paper keys).
2. Open the **Actions** tab → **Paper trading bot** → **Run workflow** to start right away. After that it keeps itself going on an hourly schedule.
3. Watch trades in the Alpaca app. Each run's log is attached to its run in the Actions tab.

To stop it: **Actions → Paper trading bot → ⋯ → Disable workflow**, then cancel any running job.
GitHub pauses scheduled workflows after 60 days without commits to the repo; re-enable it from the Actions tab.
