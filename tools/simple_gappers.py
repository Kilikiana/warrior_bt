from __future__ import annotations

import json
import gzip
from pathlib import Path
from typing import List, Dict
import pandas as pd

from core.config import OHLCV_1MIN_DIR, OHLCV_DAILY_DIR, RESULTS_DIR


def compute_open_gappers(date: str, min_gap_pct: float = 10.0) -> Path:
    """Compute simple 9:30 open gappers from cached 1-min and daily bars.

    Writes a CSV at results/criteria_scans/open_gap_results_<date>_fallback.csv
    with columns [generated_at,symbol,date,prev_date,open_time,prev_close,open_price,gap_pct].
    """
    out_dir = RESULTS_DIR / 'criteria_scans'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'open_gap_results_{date}_fallback.csv'

    src = OHLCV_1MIN_DIR / f"ohlcv_1min_{date}.json.gz"
    if not src.exists():
        # Write empty file and return
        pd.DataFrame(columns=['generated_at','symbol','date','prev_date','open_time','prev_close','open_price','gap_pct']).to_csv(out_path, index=False)
        return out_path

    with gzip.open(src, 'rt') as f:
        all_data = json.load(f)

    rows: List[Dict] = []
    for rec in all_data:
        sym = rec.get('symbol')
        if not sym:
            continue
        minutes = rec.get('minutes') or rec.get('bars') or []
        # find 09:30
        open_row = next((m for m in minutes if m.get('time') == '09:30'), None)
        if not open_row:
            continue
        open_px = float(open_row.get('open', 0.0) or 0.0)
        if open_px <= 0:
            continue
        # premarket volume 04:00-09:29
        pm_vol = 0
        try:
            for m in minutes:
                t = m.get('time')
                if not t:
                    continue
                # include 04:00 up to (but not including) 09:30
                if '04:00' <= t < '09:30':
                    pm_vol += int(m.get('volume') or 0)
        except Exception:
            pm_vol = 0
        # prev close from daily
        daily = OHLCV_DAILY_DIR / f"daily_{sym}.json"
        prev_close = None
        if daily.exists():
            try:
                with open(daily, 'r') as df:
                    d = json.load(df)
                bars = d.get('bars', [])
                # choose last bar earlier than date
                prev = None
                for b in bars:
                    if b.get('date') and b['date'] < date:
                        prev = b
                if prev and prev.get('close') is not None:
                    prev_close = float(prev['close'])
            except Exception:
                prev_close = None
        if prev_close is None or prev_close <= 0:
            continue
        gap_pct = (open_px - prev_close) / prev_close * 100.0
        if gap_pct < min_gap_pct:
            continue
        rows.append({
            'generated_at': pd.Timestamp.utcnow().isoformat(),
            'symbol': sym,
            'date': date,
            'prev_date': None,
            'open_time': '09:30',
            'prev_close': prev_close,
            'open_price': open_px,
            'gap_pct': gap_pct,
            'premarket_volume': pm_vol,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('gap_pct', ascending=False)
    df.to_csv(out_path, index=False)
    return out_path
