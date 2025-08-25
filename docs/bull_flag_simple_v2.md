# Bull Flag Simple V2 — Current Strategy

This document describes the current implementation of the Bull Flag Simple V2 strategy and its runtime behavior, configuration, logging, and usage.

## Overview
- Focus: 1‑minute continuation entries following an ACTION alert.
- Trigger: Wait for a contiguous red pullback after the alert, then enter when the first green bar’s high breaks the last red bar’s high (intrabar).
- Timeout: Stop monitoring if no breakout occurs within 6 bars from the first red after the alert.
- Exit: “Flip” behavior — exit on the close of the first non‑alert bar after entry.
- Clustering: In per‑alert mode, redundant alerts within the same move are clustered so only one session is started per move.

## Rules (Current)
- Entry timing:
  - After an ACTION alert, ignore the alert bar itself; begin logic on bars strictly after the alert time.
  - Detect the first red bar (close < open) — this starts the pullback.
  - Track contiguous red bars; the “last red’s high” is updated as the pullback extends.
  - When a green bar (close > open) prints with `high >= last_red_high`, enter intrabar at `fill_price = last_red_high + entry_slippage_cents`.
- Timeout:
  - Start a 6‑bar window at the first red after the alert.
  - Increment the count each bar while the pullback is active and no entry has triggered.
  - If no green break occurs by bar 6, stop monitoring this alert (status → MONITORING_STOPPED).
- Exit (flip):
  - Exit on the first non‑alert bar close after entry.
  - If the next bar is an alert bar and is red, exit on that bar; otherwise defer to the first non‑alert bar.
- Stop loss (current, not Ross‑aligned):
  - Percent stop below entry using `risk_config.PREFERRED_STOP_DISTANCE_PCT` (default 2%).
  - Note: This will be changed to the pullback low in a subsequent iteration to match Ross more closely.

## Alert Clustering (Per‑Alert Mode)
To prevent redundant sessions for a single move, the backtest runner clusters alerts for `bull_flag_simple_v2`:
- Roll the seed alert forward across consecutive green bars until the first post‑alert red appears (still pre‑pullback).
- Once the first red appears, suppress subsequent alerts within that pullback until either:
  - Breakout (first green with high ≥ last red high), or
  - Timeout (6 bars from first red).
- After resolution, the next cluster can start.

This ensures we monitor only one alert per move — matching the intended trade selection logic.

## Logging
The strategy logs concise trace lines to help analysis:
- Pullback start: `BFSv2 pullback-start: <SYMBOL> | first_red=<TS> last_red_high=<PRICE>`
- Entry: `BFSv2 entry: <SYMBOL> | price=<PX> stop=<PX> trigger_bar=<TS> last_red_high=<PX>`
- Timeout: `BFSv2 timeout: <SYMBOL> | no breakout within 6 bars from first red after alert (first_red=<TS>)`

These appear in `results/logs/backtest_<DATE>_<HHMMSS>.log`.

## Usage
- CLI (example):
  - `python warrior_backtest_main.py --date 2025-08-13 --start-time 06:00 --end-time 11:30 --pattern bull_flag_simple_v2 --per-alert-sessions --symbols BSLK --log-level INFO`
- Notable options:
  - `--entry-slippage-cents`: entry slippage in dollars.
  - `--stop-slippage-cents`: stop fill slippage.
  - `--per-alert-sessions`: enables alert clustering logic for this pattern.
  - `--entry-cutoff-minutes`: freshness window for new entries.

## Example (BSLK 2025‑08‑13)
- Clusters selected (by alert time): 10:05, 10:24, 11:22.
- Trades (from entries CSV):
  - 10:05 → Entry 10:07 @ 4.3000 → Exit 10:09 @ 4.0302 → P&L −$1,176.33
  - 10:24 → Entry 10:28 @ 4.6700 → Exit 10:37 @ 5.0400 → P&L +$1,485.18
  - 11:22 → Entry 11:26 @ 6.8700 → Exit 11:30 @ 8.2400 → P&L +$3,738.73
- Note: Exit reason shows `session_end` in logs due to per‑alert slice clamping, but functionally aligns with the “flip” exit.

## Integration
- Strategy class: `tech_analysis/patterns/bull_flag_simple_v2.py`
- Runner (clustering): `warrior_backtest_main.py` (per‑alert sessions logic)
- Session engine: `tech_analysis/patterns/pattern_monitor.py` (entries/exits/sizing plumbing)
- Risk config: `risk_config.py` (percent stop parameter and other risk constants)

## Current Limitations
- Stop placement uses a fixed percent distance (2% default), not the pullback low (Ross’s rule). Scheduled for correction.
- Confirmations (EMA/MACD) are not enforced at entry in this v2 strategy — this is intentional for stepwise evolution.
- Exit reason label reads `session_end` in per‑alert runs; behavior is the flip exit.

## Next Steps
- Switch stop to the pullback low (from the red sequence).
- Add prior‑bar confirmations (EMA 9>20, MACD bullish) at entry to mirror Ross’s timing.
- Optionally extend clustering to other strategies if needed.

