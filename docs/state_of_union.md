# Warrior BT — State of the Union (as of now)

## What We’ve Built

- Pattern engine: `PatternMonitoringSession` orchestrates alerts → per‑symbol sessions → entries/exits → reporting.
- Strategies:
  - `bull_flag` (full detector with confirmations and management).
  - `bull_flag_simple_v2` (fast “break the last red high after alert” + pragmatic risk management).
  - `alert_flip` (buy alert close, sell next bar close/alert‑red).
- Risk + sizing:
  - `PositionSizer` (Ross dynamic sizing: ~1.25% risk per trade → $375 on a $30K account).
  - `PositionTracker` (max 4 positions, daily risk budget, duplicate guards).
- Backtesting: `warrior_backtest_main.py` (loads alerts, slices OHLC, runs sessions, writes logs/CSVs/visuals).
- Visuals: tools to plot 1‑min trades (headless, safe in CI), now grouped by run folder.

## Bull Flag Simple v2 — Current Rules (ELI5)

- When we hear “ding!” (ACTION alert): watch 1‑min candles after the alert.
- Wait for a “breather” (first red starts a red pullback). Track:
  - The “doorway” (last red candle’s high) and the lowest low of that red pullback (true pivot low).
  - Pullback red volumes for an average.
- Entry trigger: first green that pokes above the doorway.
- Volume quality: require green breakout volume ≥ (pullback red average × `breakout_vol_mult`, e.g., 1.5). Thin pullbacks are OK if `min_pullback_avg_volume` = 0.
- Pullback depth: must be ≤ 50% of the “pole” (approximate pole = `alert_high − alert_price`).
- Fill realism: entry never above current bar’s high.
- Stop: lowest low of the red pullback (technical). Intrabar stop fills simulate gaps/slippage.
- Target(s):
  - Primary: `alert_high` if above entry.
  - Fallback: if `alert_high` ≤ entry or missing, set 2R target = `entry + 2×(entry − stop)`.
- Scaling/BE:
  - On fallback 2R target: sell 50% and move remaining stop to breakeven (entry).
  - On `alert_high` target: optionally do the same (50% + BE) — controlled by `v2_partial_on_alert_high` (default ON).
- Fast fail: if the entry bar closes red, exit immediately (reason: NO_IMMEDIATE_BREAKOUT).
- Timeout: if no entry within 6 one‑minute bars from the first red, stop watching the alert.
- Managing past end: if enabled, sessions run until the last bar in the day and force‑flatten then (reason: SESSION_END).

## Recent Fixes & Improvements

- Correctness & timing
  - Fixed double‑calling v2 per bar (timeouts are truly 6 minutes now).
  - Entries honor cutoffs; only exits run after cutoffs.
  - Robust 6‑bar timeout: index‑based counting.
- Risk realism
  - Stop = true pullback low (not previous bar low).
  - Entry capped to bar high (no fantasy fills).
  - Target guard: ignore `alert_high` if ≤ entry.
- Quality gates
  - Added `breakout_vol_mult` and `min_pullback_avg_volume` (thin pullbacks OK with `0`).
  - Added retrace cap ≤ 50% of pole (alert_high − alert_price).
- Better exits
  - 2R fallback target + 50% scale + move stop to breakeven.
  - Optional 50%+BE on `alert_high` targets (default ON).
  - Exit reasons are explicit: STOP_LOSS, FIRST_TARGET, NO_IMMEDIATE_BREAKOUT, SESSION_END.
- Backtest ergonomics
  - Per‑alert slicing honors `--manage-past-end` (lets trades play to day end).
  - Plotting uses headless backend and proper datetime geometry.

## Config Knobs You Can Turn

- `--breakout-vol-mult <x>`: green breakout volume ≥ red pullback avg × x (e.g., 1.5, 2.0).
- `--min-pullback-avg-volume <n>`: require pullback red avg volume ≥ n shares/bar (0 to disable; thin pullbacks allowed).
- `--manage-past-end/--no-manage-past-end`: let trades run past end‑time or clamp at `--end-time`.
- `--pattern bull_flag_simple_v2 --per-alert-sessions`: enable v2 per alert.
- Session defaults added:
  - `v2_partial_on_alert_high` (default True) — partial on `alert_high` + stop to BE.

## Why Many Stop‑outs Show ~−$375

- Position sizing risks ~1.25% per trade of a $30K account = ~$375.
- Risk per share = entry − stop; shares = 375 / risk_per_share.
- A clean stop ≈ −$375. Slippage can wiggle it slightly.

## Results Snapshot

- 2025‑08‑13 (per‑alert, manage‑past‑end, v2 with 1.5× green>red, thin pullbacks allowed):
  - Before partial‑on‑alert_high: Alerts 50 | Entries 16 | Wins 11 | Losses 5 | P&L $4,143 | Hit 68.8% | session_end 2.
  - After partial‑on‑alert_high enabled (2R+BE already on): Entries 10 | Wins 9 | Loss 1 | P&L $76,087 (date had big runners; BSLK heavy contributor).
- 2025‑07‑07 (per‑alert, earlier v2 without all gates): Alerts 92 | Entries 81 | Wins 17 | Losses 63 | P&L −$4,146 (looser rules; many low‑quality entries).

Note: The large improvement with partials is date‑sensitive. The setting is a sensible default; A/B across more dates is advised for a stable policy.

## Known Gaps / Open Questions

- No post‑entry “breakout or bailout” timer (e.g., exit flat if no progress in 10 minutes).
- No “first red after entry” exit (code comment implied it; behavior wasn’t implemented in v2).
- No explicit “no‑new‑high during pullback” or trendline breakout gate (kept for later; full detector has richer structure checks).
- `SESSION_END` still occurs if a position remains open at day end; optional `--no-force-flatten` could be added for purist backtests.

## Recommended Next Steps

- Add a simple post‑entry sanity timer: if high ≤ entry for N minutes, exit flat (configurable N).
- Add “first red after entry” exit (or small trim) to keep momentum trades tight.
- A/B test partial‑on‑alert_high across several dates (P&L, hit rate, drawdown) to confirm default.
- Optional: EMA/VWAP overlays in plots (already supported in manager logic) and richer chart annotations.
- Later (if desired): add “no new high in pullback” and trendline breakout gate inspired by research repo.

### Planned Enhancement (v2 Runner Logic)

- 10‑minute MACD gate (post‑entry):
  - At 10 minutes after entry, check 1‑minute MACD.
  - If bearish (macd < signal or histogram ≤ 0): exit remaining shares.
  - If bullish: continue to hold the runner.
- Runner management while holding:
  - Trail stop at EMA9 (1‑minute), but never below breakeven.
  - Hard cap target at 3R (entry + 3 × risk per share); exit remaining at that level if hit.
  - Keep existing 2R partial (50%) and move‑to‑breakeven behavior.
- Visual overlays:
  - Add EMA9/EMA20/VWAP to BFSv2 plots to verify trend strength and show where the 10‑minute MACD check occurred.

## File/Commit Pointers

- Core: `tech_analysis/patterns/pattern_monitor.py`, `warrior_backtest_main.py`.
- V2: `tech_analysis/patterns/bull_flag_simple_v2.py` (volume/retrace gates, 2R+BE, partial on alert_high).
- Risk: `position_management/position_sizer.py`, `position_management/position_tracker.py`, `risk_config.py`.
- Visuals: `tools/plot_bfs2_trades.py`.
- Recent commits: see `git log` — highlights include `0a982ad`, `afb777e`, `c1f2c64`, `9d33254`, `d1adb61`, `013d615`.

---
This document updates as we evolve the rules. Ping me when you want to lock a “baseline” and start formal A/B runs.
