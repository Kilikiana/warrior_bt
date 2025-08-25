# Engineering Plan — Momentum Bot Backtesting Roadmap

This document tracks near‑term priorities and staged improvements beyond the hard daily stop.

## Phase 1 — Backtester Realism (Immediate Next)
- Costs & spread: apply `fees_per_share`, `per_trade_fee`, and a simple spread model to entries/exits.
- R metrics & analyzer: compute R multiple distribution, MFE/MAE, time‑to‑2R, and net‑of‑fees stats.
- Partial fills (approx): simulate 2–3 slice fills at targets/stops using volume‑aware heuristics.
- Config freeze + repro: persist a config hash per run; avoid rule tweaks mid‑batch.

## Phase 2 — Market Replay / Live‑like Sim
- Replay runner: stream historical 1‑minute at live speed with latency budgets (100–400ms) and throttles.
- Order/state safety: surface rejects and race conditions; halt/SSR constraints modeled at a basic level.

## Phase 3 — Paper Trading
- Broker adapter: unified interface (e.g., IB/Alpaca paper) sharing the same engine.
- Shadow fills: record broker fills vs internal simulator; track slippage deltas and incidents.
- Guardrails: daily stop, per‑trade R cap, max concurrent names, cooldown after two losers.

## Phase 4 — Tiny Live Capital
- Kill‑switch automation and position size limiter (fraction of paper size).
- Promotion rules: scale only after meeting pre‑defined profit/stability milestones.

## Data Realism Checklist
- Fees (SEC/FINRA), spreads, partial fills, SSR, LULD halts, pre/post‑market liquidity, basic latency.
- Validate against a few hand‑reviewed days: simulated entries/exits look plausible vs prints.

## Anti‑Overfit Discipline
- Walk‑forward batches (frozen ruleset) across multiple months.
- Report net expectancy, drawdown, W/L stability; iterate only between batches.

## Nice‑to‑Haves
- Per‑lot stops for add‑backs; partial stop‑outs by lot with per‑lot R accounting.
- 5‑minute trend alignment filter for add‑backs (9>20 EMA).

## Current Status Snapshot
- Event‑driven backtester with intrabar breakout fills and confirmations.
- ACTION alert scanners + replay present; position sizing and basic risk in place.
- Hard daily stop: added in orchestrator (halts new entries once limit breached).

