# Changelog

All notable changes to this project will be documented in this file.

## v0.1-bfs2-clustering — 2025-08-24

- BullFlag Simple V2: start the 6-bar timeout from the first red bar after the ACTION alert (instead of from the alert). This aligns timing with pullback-based monitoring.
- BullFlag Simple V2: add trace logs for
  - pullback-start (first red after the alert + last red high),
  - entry (price, stop, trigger bar, last red high),
  - timeout (no breakout within 6 bars from first red).
- Backtest runner (`warrior_backtest_main.py`): implement per-alert clustering for `bull_flag_simple_v2` so redundant ACTION alerts within the same move are suppressed:
  - Roll the seed alert forward across consecutive green bars until the first red appears (still pre-pullback).
  - Ignore subsequent alerts within the pullback until either the breakout occurs or the 6-bar timeout is reached.
  - Start the next cluster only after the current one resolves.
- Result: Fewer redundant sessions, clearer logs, and behavior closer to the intended intrabar bull-flag timing.

Files touched:
- `tech_analysis/patterns/bull_flag_simple_v2.py`
- `warrior_backtest_main.py`

