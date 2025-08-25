"""
Plot Bull Flag Simple V2 trades as 1-min visualizations with key markers.

Usage:
  python tools/plot_bfs2_trades.py --date 2025-08-13 --symbol BSLK

Outputs PNGs to results/visuals/bfs2/<SYMBOL>_<ENTRYTIME>.png
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt

from core.config import RESULTS_DIR
import matplotlib.gridspec as gridspec
from data.ohlc_loader import load_symbol_ohlc_data


def _find_latest_sessions_file(date: str) -> Optional[Path]:
    logs_dir = RESULTS_DIR / "logs"
    candidates = sorted(logs_dir.glob(f"backtest_{date}_*_sessions.json"), reverse=True)
    # Prefer the most recent file that contains entries with entry_time
    import json
    for p in candidates:
        try:
            data = json.load(open(p))
            if isinstance(data, list):
                for s in data:
                    if s.get('position_entered') and s.get('entry_time'):
                        return p
        except Exception:
            continue
    return candidates[0] if candidates else None


def _parse_dt(s: str) -> datetime:
    return datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S")


def _first_red_after(df: pd.DataFrame, ts: datetime) -> Optional[pd.Timestamp]:
    try:
        window = df.loc[df.index > ts]
    except Exception:
        return None
    for t, row in window.iterrows():
        try:
            if float(row['close']) < float(row['open']):
                return t
        except Exception:
            continue
    return None


def _last_red_high_between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> Optional[float]:
    try:
        seg = df.loc[(df.index >= start) & (df.index <= end)]
    except Exception:
        return None
    last_high = None
    for _, row in seg.iterrows():
        try:
            if float(row['close']) < float(row['open']):
                last_high = float(row['high'])
            else:
                # stop at first non-red (contiguity)
                break
        except Exception:
            pass
    return last_high


def _plot_candles(ax, df: pd.DataFrame):
    # Simple manual candlesticks
    width = 0.5
    up_color = '#26a69a'
    dn_color = '#ef5350'
    for t, row in df.iterrows():
        o, h, l, c = float(row['open']), float(row['high']), float(row['low']), float(row['close'])
        color = up_color if c >= o else dn_color
        ax.vlines(t, l, h, color=color, linewidth=1)
        top, bottom = (c, o) if c >= o else (o, c)
        ax.add_patch(plt.Rectangle((t - pd.Timedelta(minutes=0.5), bottom), pd.Timedelta(minutes=1), top - bottom, 
                                   edgecolor=color, facecolor=color, linewidth=0.8))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def plot_trades(date: str, symbol: str, outdir: Optional[Path] = None, sessions_file: Optional[str] = None) -> List[Path]:
    base_out = outdir or (RESULTS_DIR / "visuals" / "bfs2")

    sessions_file = Path(sessions_file) if sessions_file else _find_latest_sessions_file(date)
    if not sessions_file or not sessions_file.exists():
        raise SystemExit(f"No sessions file found for {date}")
    import json
    raw = json.load(open(sessions_file))
    df_sess = pd.DataFrame(raw)
    entries = df_sess[df_sess["position_entered"] == True]
    entries = entries[entries["symbol"].str.upper() == symbol.upper()]
    if entries.empty:
        raise SystemExit("No entries found to plot.")

    df = load_symbol_ohlc_data(symbol.upper(), date, timeframe="1min")
    if df is None or df.empty:
        raise SystemExit("No OHLC data found.")

    # Use a per-run subfolder if sessions_file provided to avoid mixing outputs
    run_tag = None
    try:
        if sessions_file:
            st = Path(sessions_file).stem  # backtest_YYYY-MM-DD_HHMMSS_sessions
            run_tag = st.replace('_sessions', '')
    except Exception:
        run_tag = None
    outdir = base_out / run_tag if run_tag else base_out
    _ensure_dir(outdir)

    outputs: List[Path] = []
    for _, row in entries.iterrows():
        alert_time = _parse_dt(row['alert_time']) if isinstance(row['alert_time'], str) else row['alert_time']
        entry_time = _parse_dt(row['entry_time']) if isinstance(row['entry_time'], str) else row['entry_time']
        exit_time = _parse_dt(row.get('exit_time')) if isinstance(row.get('exit_time'), str) else row.get('exit_time')
        entry_price = float(row['entry_price'])
        exit_price = float(row.get('exit_price', float('nan')))

        # Compute pullback first red and last red high near entry
        fr = _first_red_after(df, alert_time)
        lrh = None
        if fr is not None:
            # scan from fr forward until the bar just before entry_time
            try:
                end = df.index[df.index.get_loc(entry_time)] if entry_time in df.index else df.index[df.index.get_indexer([entry_time], method='pad')[0]]
                lrh = _last_red_high_between(df, fr, end)
            except Exception:
                lrh = None

        # Window to plot
        start = alert_time - timedelta(minutes=10)
        end = (exit_time or entry_time) + timedelta(minutes=10)
        window = df.loc[(df.index >= start) & (df.index <= end)].copy()
        if window.empty:
            continue

        # Layout with volume sub-plot
        fig = plt.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(5, 1, figure=fig)
        ax = fig.add_subplot(gs[0:3, 0])
        axv = fig.add_subplot(gs[3:5, 0], sharex=ax)

        _plot_candles(ax, window)
        title = (
            f"{symbol.upper()} {date} | "
            f"Alert {alert_time.strftime('%H:%M')} | "
            f"Entry {entry_time.strftime('%H:%M')} @ {entry_price:.2f}"
        )
        if exit_time is not None and pd.notna(exit_price):
            title += f" | Exit {exit_time.strftime('%H:%M')} @ {float(exit_price):.2f}"
        ax.set_title(title)
        ax.set_ylabel("Price")

        # Markers on price
        ax.axvline(alert_time, color='#1976d2', linestyle='--', linewidth=1.0, label='Alert')
        if fr is not None:
            ax.axvline(fr, color='#8e24aa', linestyle='--', linewidth=1.0, label='First Red')
        ax.scatter([entry_time], [entry_price], color='#2e7d32', marker='^', s=80, zorder=3, label='Entry')
        if exit_time is not None and pd.notna(exit_price):
            ax.scatter([exit_time], [exit_price], color='#d32f2f', marker='v', s=80, zorder=3, label='Exit')
        # Last red high line (trigger)
        if lrh is not None:
            ax.axhline(lrh, color='#f9a825', linestyle=':', linewidth=1.2, label='Last Red High')

        # Volume bars (colored by candle direction)
        try:
            vol_colors = ['#26a69a' if float(window.loc[t, 'close']) >= float(window.loc[t, 'open']) else '#ef5350' for t in window.index]
        except Exception:
            vol_colors = '#90a4ae'
        axv.bar(window.index, window['volume'], width=0.0006 * len(window) if len(window) > 0 else 0.5, color=vol_colors, alpha=0.6)
        axv.set_ylabel('Volume')
        axv.grid(True, axis='y', linestyle=':', alpha=0.3)

        ax.legend(loc='best')
        fig.autofmt_xdate()

        exit_tag = exit_time.strftime('%H%M') if exit_time is not None else 'NA'
        outfile = outdir / f"{symbol.upper()}_{entry_time.strftime('%H%M')}_{exit_tag}.png"
        fig.savefig(outfile, dpi=140, bbox_inches='tight')
        plt.close(fig)
        outputs.append(outfile)

    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True)
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--sessions-file', help='Optional explicit sessions JSON to use')
    args = ap.parse_args()
    outs = plot_trades(args.date, args.symbol, sessions_file=args.sessions_file)
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
