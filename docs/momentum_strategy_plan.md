# Momentum Strategy Plan (Ross Cameron Style)

## Goals
- Systematically capture 1‑minute bull‑flag continuation moves (second/third waves).
- Cut losers quickly; let winners run; sell into the start of pullbacks; rebuy on fresh flags.
- Stay faithful to Ross’s rules while improving P&L with robust execution modeling and analysis.

## Current Architecture (Harmonized)
- `PatternMonitoringSession` (single engine): entries, exits, sizing, trims, add‑backs, logging.
- `BullFlagDetector` (entry‑only): strict validation (default) for the flag setup.
- Shared indicators per bar: EMA(9/20), MACD(12/26/9), VWAP.
- Orchestration: `warrior_backtest_main.py` (CLI, data loading, logging).

## Entry Rules (Strict by Default)
- 1‑minute timeframe.
- Flagpole followed by a 1–5 bar pullback near highs.
- Pullback retrace ≤ 50% of the pole; consolidation volume < pole volume.
- Pullback low holds above 9EMA and VWAP.
- Entry trigger: the first bar whose high breaks the last red bar’s high (pullback high).
- Confirmations at entry:
  - EMA trend: 9EMA > 20EMA (1‑minute).
  - MACD: MACD > Signal and histogram > 0 (12/26/9).

## Exit & Management Rules
- Stop loss at pullback low (executed intrabar at stop, with optional slippage).
- Bailout: breakeven within N=2 1‑minute bars if no immediate follow‑through (configurable).
- First target (2R): sell 1/2 and move stop to breakeven on the remainder.
- Trim on strength: extension‑bar trims (large green + high volume); optional level trims.
- Trim at start of pullback: sell ~33% on first meaningful red (1‑minute red that breaks prior low).
- Weakness exit (toggle): exit if price decisively breaks 1‑minute 9EMA or VWAP.
- Optional runner exit (toggle): first red 5‑minute candle.

## Add‑Backs (Second Wave)
- After partial exits, look for a fresh 1‑minute bull flag.
- On the next breakout, BUY additional shares sized via `PositionSizer` with the new pullback‑low stop.
- Controls: `max_add_backs` (default 2), cooldown bars (default 2), and optional higher‑high precondition.

## Execution Modeling (Fills)
- Stops: executed intrabar at the stop price; if gap below stop, fill at the bar open. Optional slippage in cents.
- Entries (proposed): execute at the breakout price (last red high) intrabar once the bar’s high ≥ trigger. Optional entry slippage in cents.
  - Indicators for confirmation can use the pre‑bar (previous bar) state to avoid waiting for bar close.
  - Without tick data, timestamp resolution remains 1‑minute; price accuracy improves via the fill model.
- Per‑lot stops (enhancement):
  - Treat each add‑back as its own lot with its own initial stop (the lot’s pullback low) and its own R.
  - Allow partial stop‑outs by lot while keeping runner lots alive.
  - Report per‑lot R and aggregate R/P&L at the trade level.

## P&L / Metrics
- End‑of‑run summary logged: alerts, entries, wins/losses/flats, total P&L, hit rate, avg/trade, exit reasons.
- Per‑trade journal (in session summary): executions, reasons, per‑trade P&L, and (next) MFE/MAE.

## Immediate Improvements (Data‑Driven)
- Time‑of‑day filter: emphasize strongest windows (e.g., 06:45–10:30) after measuring hit rate and MFE by hour.
- Bailout speed: evaluate `grace_bars=1` vs `2` on win rate and average loss.
- Dynamic trim sizing: 25% vs 33–50% at the start of pullbacks depending on prior extension/MFE.
- Add‑back conditions: require higher‑high + cooldown; keep strict ≤50% retrace and 9EMA/VWAP holds.

## Intrabar Entry Simulation (Design)
Problem: With 1‑minute OHLC we only know the high crossed the trigger during the bar; we don’t know the exact second.

Solution (practical and faithful to Ross):
- Detect trigger when current bar’s high ≥ last_red_high.
- Execute entry at fill_price = last_red_high + entry_slippage_cents.
- Confirm EMA/MACD using the prior bar’s values (pre‑bar state) to avoid waiting for the bar close.
- Manage the position immediately with the same intrabar stop model already implemented for exits.

Implementation sketch:
- In `BullFlagDetector.detect_entry_signal`: remove the `close > open` requirement; the trigger is the high crossing the last red high.
- In `PatternMonitoringSession._check_patterns`:
  - Use `signal.entry_price` (which equals the last_red_high) as the fill price (plus optional entry slippage).
  - For confirmations, use EMA/MACD values from the previous bar (`df.iloc[-2]`).
  - Add config: `entry_slippage_cents`.

## Analysis Tools (Next)
- Add a small analyzer (e.g., `tools/analyze_trades.py`) to compute per‑trade MFE/MAE, time‑to‑2R/−1R, hit rate by hour, and performance by pullback length and bailout setting.
- Feed this back into tuning: grace bars, trim sizes, add‑back limits, time‑of‑day filters.

## Current CLI Knobs
- `--bailout-grace-bars` (default 2)
- `--use-5min-first-red-exit` (off by default)
- `--disable-ema-vwap-weakness-exit` (on by default)
- `--stop-slippage-cents` (default 0.00)
- `--disable-add-back` (on by default = enabled)
- `--max-add-backs` (default 2)
- `--add-back-cooldown-bars` (default 2)
- Auto‑timestamped logs: `results/logs/backtest_<date>_<HHMMSS>.log`

## Next Actions
1) Implement intrabar entry simulation (fill at breakout + slippage; prior‑bar confirmations).
2) Add trade analyzer to quantify MFE/MAE and time‑to‑target metrics.
3) Tune grace bars, early‑pullback trim size, and add‑back conditions using 1–2 days of data.
4) Optionally require 5‑minute 9>20 EMA alignment for add‑backs only (trend continuation filter).
5) Implement per‑lot stops and partial stop‑outs to improve R accounting and risk control on add‑backs.
