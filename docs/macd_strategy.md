# MACD Strategy Guide

This document defines a simple, testable MACD‑led momentum strategy inspired by the transcript. It focuses on MACD for entries/exits, with clear guardrails and backtest knobs.

## Goals
- Trade obvious momentum names using MACD for timing and discipline.
- Keep entries simple (MACD cross or first pullback while MACD is positive).
- Exit into strength, then honor weakness fast (bailout, cross back down, below VWAP/EMA).

## Stock Selection (Pre‑Filter)
- Top Gainers: Up ≥ 30% intraday and among top 5 leaders.
- Price Range: $3–$20 preferred.
- Float: ≤ 5M shares (lower supply rotates harder).
- Relative Volume (RVOL): ≥ 5× average (higher is better).
- Catalyst: Fresh, tradeable news (PR, earnings, FDA, etc.).
- Attention: Obvious on scanners; many traders watching.

## Context Filters
- MACD Zone: Only trade long when MACD is positive (green zone).
- VWAP Behavior: Above VWAP or reclaiming VWAP on rising volume.
- Volume Profile: Buying dominates (more/larger green bars). Avoid high‑volume red at trigger.
- Level 2: No large sell wall pinned just above entry.

## Entries (MACD‑Led)
- Primary Trigger: First bullish MACD crossover (MACD > signal) after a consolidation/hold near VWAP.
- Micro Pullback (if missed cross): Buy the first 1‑min pullback while MACD remains positive:
  - Pullback holds above VWAP/9EMA or shows bottoming tails.
  - Enter on the first candle to make a new high.
- Optional Guards:
  - Skip if a large sell wall sits at/near breakout price.
  - Skip if pullback retraces > 50% of the prior leg.

## Stops & Risk
- Initial Stop: Below pullback low, or a fixed distance (≈ 2% under entry) if structure unclear.
- Bailout: If no immediate follow‑through (1–2 bars), exit near breakeven (“breakout or bailout”).
- Position Size: Risk‑based sizing using risk_per_share (entry − stop) vs per‑trade risk budget.

## Exits (Priority)
- Sell Into Strength (optional but recommended):
  - Extension Bar: Large green range (~2× recent) + high volume (~1.5× recent). Scale 25–50%.
- Targets & Stops:
  - First Target at 2R: Sell 50%, move stop to breakeven on remainder.
  - Continue scaling on strength; trail if keeping a runner.
- Weakness Signals (exit remainder on any):
  - MACD Bearish Cross (MACD < signal).
  - 1‑min Close below 9EMA or below VWAP.
  - High‑volume red or large topping tail.
  - Lost obviousness (another symbol becomes the clear leader).
- Optional: First red 5‑minute candle can be the runner exit.

## Disqualifiers (Do Not Enter)
- MACD negative (red zone).
- High‑volume selling dominates the profile.
- Large Level‑2 sell wall blocking breakout.
- Not obvious (not a top gainer, no news, thin volume).
- Deep retrace (> 50%) before trigger.

## When It Works Best
- Strong, obvious momentum names with extreme demand vs supply (low float, high RVOL, catalyst).
- The first bullish MACD crossover after a “hold‑up” consolidation near VWAP (accumulation).
- Morning momentum windows with fast resolution (minutes).

## Backtest Knobs (CLI)
The backtester supports MACD‑only triggers via an opt‑in mode. Examples:

- Pattern entries (default) but MACD‑only confirmations:
  
  ```bash
  python warrior_backtest_main.py \
    --date 2025-08-13 --start-time 06:00 --end-time 11:30 --account 30000 \
    --symbols BSLK \
    --entry-confirmations macd_only \
    --log-level INFO
  ```

- MACD cross as trigger + MACD cross exit (minimalist MACD system):
  
  ```bash
  python warrior_backtest_main.py \
    --date 2025-08-13 --start-time 06:00 --end-time 11:30 --account 30000 \
    --symbols BSLK \
    --trigger-mode macd_cross \
    --exit-on-macd-cross \
    --entry-slippage-cents 0.01 --stop-slippage-cents 0.01 \
    --log-level INFO
  ```

- Keep default exits enabled (EMA/VWAP weakness, extension bar, early trim) while using MACD cross trigger:
  
  ```bash
  python warrior_backtest_main.py \
    --date 2025-08-13 --start-time 06:00 --end-time 11:30 --account 30000 \
    --symbols BSLK \
    --trigger-mode macd_cross \
    --log-level INFO
  ```

Notes:
- Defaults remain unchanged; MACD‑only behavior is opt‑in via flags.
- Slippage flags model conservative fills; adjust per testing.

## Trade Verification (Optional)
Use Alpaca historical trades to verify entry/stop/target ordering at sub‑minute resolution:

```bash
python tools/analyze_trades.py \
  --file results/logs/backtest_2025-08-13_<TS>_sessions.json \
  --date 2025-08-13 \
  --verify-trades --alpaca-feed sip --session-tz US/Eastern
```

This checks:
- entry_seen (trade ≥ entry after entry time)
- stop_seen_after_entry (trade ≤ stop after entry)
- target_seen_after_entry (trade ≥ 2R after entry)

Ensure `.env` contains `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.

## Testing Suggestions
- A/B compare: (1) baseline pattern entries, (2) MACD‑only confirmations, (3) MACD cross trigger + MACD cross exit.
- Evaluate entries, P&L, hit rate, exit reasons; drill into early vs mid‑morning windows.
- Use verify‑trades for ambiguous 1‑minute bar ordering.

---
This strategy doc is intentionally concise and operational. Tune thresholds (e.g., 2% stop, RVOL≥5×) with data. 
