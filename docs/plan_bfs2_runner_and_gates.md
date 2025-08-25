# BFSv2 Improvements Plan

## Goals
- Reduce -1R stop-outs on weak days without zeroing entries.
- Make 10-minute MACD exit gate effective on all trades (optionally), not just runners.
- Keep winners running with EMA9 trail + 3R hard cap.
- Expose key strategy limits as knobs for A/B testing.

## Changes (Scope)
- MACD Gate Enhancements (exit-only):
  - Add `v2_macd_gate_require_runner` (default: true). If false, apply the MACD gate at T+X minutes even without a prior partial (runner).
  - Add `v2_macd_gate_require_no_progress` (default: false) + `v2_no_progress_thresh_r` (default: 0.0): only gate if max-high since entry ≤ entry + R × thresh.
  - Keep existing `v2_runner_macd_gate_minutes` for the gate timing.
- BFSv2 Knobs:
  - `v2_max_wait_bars` (default 6): bars to wait from first red after alert before timing out.
  - `v2_retrace_cap_pct` (default 0.50): pullback depth cap as a fraction of pole height.
- Optional Entry Guards:
  - `v2_require_vwap_above` (default false): require price ≥ VWAP at entry.
  - `v2_entry_confirm_ema5m` (default false): require EMA9>EMA20 on 5-min resampled close at entry bar.

## CLI Flags
- `--v2-macd-gate-require-runner / --no-v2-macd-gate-require-runner`
- `--v2-macd-gate-require-no-progress`
- `--v2-no-progress-thresh-r <float>`
- `--v2-max-wait-bars <int>`
- `--v2-retrace-cap-pct <float>`
- `--v2-require-vwap-above`
- `--v2-entry-confirm-ema5m`

## Acceptance Criteria
- A/B on 2025-07-01 (weak day, “66-entry” baseline):
  - Baseline loose vs baseline + MACD gate (require_runner=false, no-progress OFF): fewer stop_loss, more MACD gate exits, improved P&L (or reduced drawdown).
  - Same with no-progress ON (0.1R) to verify it prevents cutting trades that did move.
- A/B on 2025-07-29 (strong day):
  - Runner ON vs OFF: runner exits (MACD gate, hard cap) present and P&L uplift with runner ON.

## Run Recipes
- Loose baseline (repro):
  - `python warrior_backtest_main.py --date 2025-07-01 --pattern bull_flag_simple_v2 --per-alert-sessions --manage-past-end --breakout-vol-mult 0 --min-pullback-avg-volume 0 --v2-entry-confirmations none --entry-confirm-mode prior --spread-cap-bps 0 --v2-min-stop-dollars 0.10`
- Baseline + MACD gate (full position):
  - `... --no-v2-macd-gate-require-runner --v2-runner-macd-gate-minutes 10`
- Baseline + MACD gate + no-progress 0.1R:
  - `... --no-v2-macd-gate-require-runner --v2-macd-gate-require-no-progress --v2-no-progress-thresh-r 0.1`

## Notes
- Spread cap currently uses 1-min bar range proxy: (high-low)/close in bps; tighten cautiously (start 200 bps).
- VWAP/EMA5m guards are optional; enable for quality without eliminating all entries.

