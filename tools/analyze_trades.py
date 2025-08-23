#!/usr/bin/env python3
"""
Analyze backtest session summaries for P&L and behavior insights.

Usage:
  python tools/analyze_trades.py --file results/logs/backtest_2025-08-13_233046_sessions.json

Outputs summary stats: entries, wins/losses, total P&L, hit rate, exit reasons,
and simple breakdowns by alert hour and pullback length.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime

def load_sessions(path: str):
    with open(path) as f:
        return json.load(f)

def try_parsetime(x):
    try:
        return datetime.fromisoformat(x)
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', required=True, help='Path to *_sessions.json produced by backtest')
    args = ap.parse_args()

    sessions = load_sessions(args.file)
    entries = [s for s in sessions if s.get('position_entered')]

    total_entries = len(entries)
    total_pnl = sum(float(s.get('total_pnl', 0.0)) for s in entries)
    wins = sum(1 for s in entries if float(s.get('total_pnl', 0.0)) > 0)
    losses = sum(1 for s in entries if float(s.get('total_pnl', 0.0)) < 0)
    flats = total_entries - wins - losses

    exit_reasons = Counter()
    for s in entries:
        for r, c in (s.get('sell_reasons') or {}).items():
            exit_reasons[r] += int(c)

    by_hour = Counter()
    for s in entries:
        t = s.get('alert_time')
        dt = try_parsetime(t) if isinstance(t, str) else t
        if dt:
            by_hour[dt.hour] += 1

    by_pullback_len = Counter()
    for s in entries:
        k = s.get('last_pullback_candles')
        if k is not None:
            by_pullback_len[int(k)] += 1

    print("=== Trade Analysis ===")
    print(f"Sessions: {len(sessions)} | Entries: {total_entries}")
    print(f"Wins: {wins} | Losses: {losses} | Flats: {flats}")
    arpt = (total_pnl / total_entries) if total_entries else 0.0
    hit = (100.0 * wins / total_entries) if total_entries else 0.0
    print(f"Total P&L: ${total_pnl:.0f} | Hit Rate: {hit:.1f}% | Avg P&L/trade: ${arpt:.0f}")
    if exit_reasons:
        print("Exit reasons:")
        for r, c in exit_reasons.most_common():
            print(f"  - {r}: {c}")
    if by_hour:
        print("Entries by alert hour:")
        for h, c in sorted(by_hour.items()):
            print(f"  - {h:02d}: {c}")
    if by_pullback_len:
        print("Entries by pullback bars (last signal):")
        for k, c in sorted(by_pullback_len.items()):
            print(f"  - {k} bars: {c}")

if __name__ == '__main__':
    main()

