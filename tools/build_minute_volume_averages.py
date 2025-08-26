from __future__ import annotations

import json
import gzip
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

from core.config import SHARED_CACHE_DIR, OHLCV_1MIN_DIR


def trading_days_before(date_str: str, n: int) -> List[str]:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    days: List[str] = []
    cur = dt - timedelta(days=1)
    while len(days) < n:
        if cur.weekday() < 5:
            days.append(cur.strftime("%Y-%m-%d"))
        cur -= timedelta(days=1)
    days.reverse()
    return days


def compute_minute_volume_averages(date: str, lookback_days: int = 30) -> Path:
    days = trading_days_before(date, lookback_days)
    # Map: symbol -> time -> [volumes]
    accum: Dict[str, Dict[str, List[int]]] = {}
    for d in days:
        f = OHLCV_1MIN_DIR / f"ohlcv_1min_{d}.json.gz"
        if not f.exists():
            continue
        try:
            with gzip.open(f, 'rt') as g:
                all_data = json.load(g)
        except Exception:
            continue
        for rec in all_data:
            sym = rec.get('symbol')
            minutes = rec.get('minutes') or rec.get('bars') or []
            if not sym or not minutes:
                continue
            slot_map = accum.setdefault(sym, {})
            for m in minutes:
                t = m.get('time')
                if not t:
                    continue
                v = int(m.get('volume') or 0)
                slot_map.setdefault(t, []).append(v)
    # Produce averages list in the format expected by RVOL tools
    out_list: List[Dict] = []
    for sym, slot_map in accum.items():
        minute_avgs = {}
        for t, vols in slot_map.items():
            if vols:
                minute_avgs[t] = sum(vols) / len(vols)
        out_list.append({
            'symbol': sym,
            'minute_avgs': minute_avgs,
        })
    out_path = SHARED_CACHE_DIR / 'minute_volume_averages_30d.json'
    with open(out_path, 'w') as f:
        json.dump(out_list, f)
    return out_path


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Build minute volume averages from cached 1-min bars')
    p.add_argument('--date', required=True, help='Target date (YYYY-MM-DD); build averages from prior N trading days')
    p.add_argument('--days', type=int, default=30, help='Lookback trading days (default 30)')
    args = p.parse_args()
    path = compute_minute_volume_averages(args.date, args.days)
    print(f"Wrote averages: {path}")

