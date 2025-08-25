#!/usr/bin/env python3
"""
Visualize a 1-minute OHLC window around a trade with optional annotations.

Usage:
  python tools/visualize_trade.py \
    --symbol MNTS --date 2025-08-13 --time 08:36 \
    --pre 10 --post 10 \
    --entry 1.72 --stop 1.66 --target 1.84 \
    --out results/visuals/MNTS_2025-08-13_0836.png

If --entry/--stop/--target are omitted, only candles are plotted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Prefer a non-interactive backend to avoid GUI/config issues
try:
    import matplotlib
    matplotlib.use('Agg')  # Safe backend for headless rendering
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
except Exception:
    plt = None

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from data.ohlc_loader import load_symbol_ohlc_data
try:
    # For default alerts file path
    from core.config import get_scan_file
except Exception:
    get_scan_file = None
import json


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', required=True)
    ap.add_argument('--date', required=True, help='YYYY-MM-DD')
    ap.add_argument('--time', help='HH:MM (local to data timestamps); ignored if --sessions-file used without --alert-time')
    ap.add_argument('--pre', type=int, default=10, help='Minutes before time to include')
    ap.add_argument('--post', type=int, default=10, help='Minutes after time to include')
    ap.add_argument('--entry', type=float)
    ap.add_argument('--stop', type=float)
    ap.add_argument('--target', type=float)
    ap.add_argument('--out', help='Output PNG path')
    # Sessions-driven auto mode
    ap.add_argument('--sessions-file', help='Path to *_sessions.json to auto-center on entry_time for symbol')
    ap.add_argument('--trade-index', type=int, default=0, help='Which trade index for the symbol (0-based) when using sessions')
    ap.add_argument('--alert-time', help='HH:MM to match specific alert_time from sessions (optional)')
    ap.add_argument('--alerts-file', help='Path to hod_momentum_scan_<date>.json (optional; auto-detected if omitted)')
    return ap.parse_args()


def plot_candles(ax, df: pd.DataFrame):
    # Basic candlestick plot without external deps
    # df indexed by timestamp, columns: open, high, low, close, volume
    width = 0.6  # minutes width
    for ts, row in df.iterrows():
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = '#2ca02c' if c >= o else '#d62728'
        # Wick
        ax.vlines(ts, l, h, color=color, linewidth=1)
        # Body
        top = max(o, c)
        bottom = min(o, c)
        ax.add_patch(plt.Rectangle((ts - pd.Timedelta(minutes=width/2), bottom),
                                   pd.Timedelta(minutes=width), top - bottom,
                                   facecolor=color, edgecolor=color, alpha=0.8))


def _pick_trade_from_sessions(sessions_path: str, symbol: str, date: str, trade_index: int = 0, alert_time: str | None = None):
    with open(sessions_path) as f:
        sessions = json.load(f)
    sym_entries = []
    for s in sessions:
        if not s.get('position_entered'):
            continue
        sym = s.get('symbol') or s.get('alert_symbol')
        if str(sym).upper() != symbol.upper():
            continue
        et = s.get('entry_time')
        if not et:
            continue
        # Normalize times as strings
        sym_entries.append(s)
    if not sym_entries:
        raise RuntimeError(f"No entries for {symbol} in sessions file")
    if alert_time:
        wanted = f"{date} {alert_time}"
        cand = [s for s in sym_entries if (s.get('alert_time') == wanted or str(s.get('alert_time')).endswith(alert_time))]
        if cand:
            return cand[0]
    # fallback by index
    idx = max(0, min(trade_index, len(sym_entries)-1))
    return sym_entries[idx]


def main():
    args = parse_args()
    if plt is None:
        print('matplotlib not available. Please install matplotlib to generate plots.', file=sys.stderr)
        sys.exit(2)

    symbol = args.symbol.upper()
    date = args.date

    # Derive center time and defaults from sessions if provided
    entry_price = args.entry
    stop_price = args.stop
    target_price = args.target
    center_time_str = args.time
    entry_dt = None
    sell_times = []
    alert_dt = None
    alert_meta = {}
    if args.sessions_file:
        trade = _pick_trade_from_sessions(args.sessions_file, symbol, date, args.trade_index, args.alert_time)
        # Center on entry_time from sessions
        center_time_str = None  # prefer entry_dt
        et = trade.get('entry_time')
        if isinstance(et, str):
            try:
                entry_dt = datetime.fromisoformat(et)
            except Exception:
                # fallback parse
                entry_dt = datetime.strptime(et, "%Y-%m-%d %H:%M:%S") if len(et) > 16 else datetime.strptime(et, "%Y-%m-%d %H:%M")
        # Defaults for price lines
        if entry_price is None:
            try:
                entry_price = float(trade.get('entry_price'))
            except Exception:
                pass
        rps = trade.get('risk_per_share')
        try:
            rps = float(rps) if rps is not None else None
        except Exception:
            rps = None
        if rps and entry_price is not None:
            stop_price = stop_price if stop_price is not None else (entry_price - rps)
            target_price = target_price if target_price is not None else (entry_price + 2.0 * rps)
        # Collect sell execution timestamps (if present)
        ex_list = trade.get('executions_list') or trade.get('executions') or []
        for e in ex_list:
            if str(e.get('action')).upper() == 'SELL' and e.get('timestamp'):
                try:
                    sell_times.append(datetime.fromisoformat(e['timestamp']))
                except Exception:
                    pass
        # Pull alert time from session if not provided
        at = trade.get('alert_time')
        if isinstance(at, str):
            try:
                alert_dt = datetime.fromisoformat(at)
            except Exception:
                try:
                    alert_dt = datetime.strptime(at, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    try:
                        alert_dt = datetime.strptime(f"{date} {at}", "%Y-%m-%d %H:%M")
                    except Exception:
                        alert_dt = None
        # Load alert metadata (strategy/description/price) to annotate
        alerts_path = args.alerts_file
        if alerts_path is None and get_scan_file is not None:
            try:
                alerts_path = str(get_scan_file(date))
            except Exception:
                alerts_path = None
        if alerts_path and alert_dt is not None:
            try:
                with open(alerts_path) as f:
                    alerts_json = json.load(f)
                all_alerts = alerts_json.get('all_alerts', [])
                # match by symbol (case-insensitive) and time HH:MM
                at_str = alert_dt.strftime('%H:%M')
                cand = [a for a in all_alerts if str(a.get('symbol','')).upper() == symbol and a.get('time') == at_str]
                if cand:
                    a = cand[0]
                    alert_meta = {
                        'time': a.get('time'),
                        'price': a.get('price'),
                        'strategy': a.get('strategy'),
                        'description': a.get('description')
                    }
            except Exception:
                pass

    # Determine center time for window
    if entry_dt is not None:
        base_dt = entry_dt
    else:
        if not center_time_str:
            print("--time is required when --sessions-file is not provided", file=sys.stderr)
            sys.exit(2)
        base_dt = datetime.strptime(f"{date} {center_time_str}", "%Y-%m-%d %H:%M")
    start_dt = base_dt - timedelta(minutes=int(args.pre))
    end_dt = base_dt + timedelta(minutes=int(args.post))

    df = load_symbol_ohlc_data(symbol, date, timeframe="1min")
    if df is None or df.empty:
        print(f"No OHLC data for {symbol} on {date}")
        sys.exit(1)

    window = df.loc[(df.index >= pd.Timestamp(start_dt)) & (df.index <= pd.Timestamp(end_dt))]
    if window.empty:
        print("Selected window is empty; check date/time.")
        sys.exit(1)

    # Default output path if none provided
    out_path = Path(args.out) if args.out else Path(f"results/visuals/{symbol}_{date}_{base_dt.strftime('%H%M')}.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure with 2 rows: price (top) + volume (bottom)
    fig, (ax_price, ax_vol) = plt.subplots(nrows=2, ncols=1, figsize=(12, 7), sharex=True,
                                           gridspec_kw={'height_ratios': [3, 1]})
    # Price plot
    ax = ax_price
    plot_candles(ax, window)
    # Overlay EMA9/EMA20 and VWAP for clarity
    try:
        close = window['close']
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema20 = close.ewm(span=20, adjust=False).mean()
        tp = (window['high'] + window['low'] + window['close']) / 3.0
        vwap = (tp * window['volume']).cumsum() / window['volume'].cumsum()
        ax.plot(window.index, ema9, color='#1f77b4', linewidth=1.2, alpha=0.9, label='EMA9')
        ax.plot(window.index, ema20, color='#ff7f0e', linewidth=1.0, alpha=0.9, label='EMA20')
        ax.plot(window.index, vwap, color='#9467bd', linewidth=1.0, alpha=0.8, linestyle='--', label='VWAP')
    except Exception:
        pass
    title_center = base_dt.strftime('%H:%M')
    title = f"{symbol} {date} {title_center} +/- {args.pre}/{args.post} min"
    # If we have alert metadata, enrich title
    if alert_meta:
        strat = alert_meta.get('strategy') or ''
        price = alert_meta.get('price')
        if price is not None:
            title += f" | Alert {alert_meta.get('time')} @ ${price:.2f} ({strat})"
        else:
            title += f" | Alert {alert_meta.get('time')} ({strat})"
    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.2)

    # Volume subplot
    try:
        colors = ['#2ca02c' if c >= o else '#d62728' for o, c in zip(window['open'], window['close'])]
        ax_vol.bar(window.index, window['volume'], color=colors, width=0.8, alpha=0.6)
        ax_vol.set_ylabel('Volume')
        ax_vol.grid(True, alpha=0.15)
    except Exception:
        pass

    # Annotation lines
    if entry_price is not None:
        ax.axhline(entry_price, color='#1f77b4', linestyle='--', linewidth=1.5, label=f"entry {entry_price:.2f}")
    if stop_price is not None:
        ax.axhline(stop_price, color='#ff7f0e', linestyle='--', linewidth=1.2, label=f"stop {stop_price:.2f}")
    if target_price is not None:
        ax.axhline(target_price, color='#2ca02c', linestyle='--', linewidth=1.2, label=f"target {target_price:.2f}")

    # Mark entry and sells as vertical lines
    ax.axvline(base_dt, color='#1f77b4', linestyle=':', linewidth=1.2, alpha=0.8)
    for st in sell_times:
        if start_dt <= st <= end_dt:
            ax.axvline(st, color='#ff7f0e', linestyle=':', linewidth=1.0, alpha=0.7)
    # Also mark alert time if available and within window
    if alert_dt is not None and start_dt <= alert_dt <= end_dt:
        ax.axvline(alert_dt, color='#9467bd', linestyle='--', linewidth=1.0, alpha=0.8)
        # Brief annotation box with strategy/desc snippet
        if alert_meta:
            desc = alert_meta.get('description') or ''
            try:
                snippet = desc[:60] + ('…' if len(desc) > 60 else '')
            except Exception:
                snippet = ''
            ax.text(alert_dt, ax.get_ylim()[1], f"Alert {alert_meta.get('time')}\n{alert_meta.get('strategy') or ''}\n{snippet}",
                    va='top', ha='left', fontsize=8, color='#4d4d4d',
                    bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.8))
    # Build legend including vertical markers
    handles, labels = ax.get_legend_handles_labels()
    extras = [
        mlines.Line2D([], [], color='#1f77b4', linestyle=':', label='entry time'),
        mlines.Line2D([], [], color='#ff7f0e', linestyle=':', label='sell time'),
        mlines.Line2D([], [], color='#9467bd', linestyle='--', label='alert time'),
    ]
    handles = handles + extras
    labels = labels + [h.get_label() for h in extras]
    ax.legend(handles, labels, loc='upper left')

    # Improve x formatting
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
