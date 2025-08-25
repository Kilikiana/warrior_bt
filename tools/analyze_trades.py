#!/usr/bin/env python3
"""
Analyze backtest session summaries for P&L and behavior insights.

Usage:
  python tools/analyze_trades.py --file results/logs/backtest_2025-08-13_233046_sessions.json [--date 2025-08-13]

Outputs summary stats: entries, wins/losses, total P&L, hit rate, exit reasons;
breakdowns by alert hour and pullback length; top symbols by P&L; top 10 winners/losers.
If --date is provided, computes MFE/MAE and time-to-2R using 1-minute OHLC in shared_cache.
With --validate, also checks per-trade conditions (entry trigger, 2R hit when claimed,
stop hit when claimed, bailout within grace bars, extension bar characteristics) and
prints failure counts; can export a CSV of validations with --validate-csv.
"""

import argparse
import json
from collections import Counter, defaultdict
import gzip
import json as jsonlib
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None
from datetime import datetime
import pytz

# Optional Alpaca trades verification
try:
    from data.alpaca_trades import fetch_trades_for_symbols, verify_ordering
except Exception:
    fetch_trades_for_symbols = None
    verify_ordering = None

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
    ap.add_argument('--date', help='Date (YYYY-MM-DD) to load OHLC for MFE/MAE/validation (optional)')
    ap.add_argument('--ohlc-dir', default='shared_cache/ohlcv_1min_bars', help='Directory with ohlcv_1min_<date>.json.gz')
    ap.add_argument('--validate', action='store_true', help='Validate per-trade logic using OHLC data')
    ap.add_argument('--grace-bars', type=int, default=2, help='Grace bars for bailout validation')
    ap.add_argument('--validate-csv', help='Path to write per-trade validation CSV')
    # Trades verification (Alpaca)
    ap.add_argument('--verify-trades', action='store_true', help='Use Alpaca trades to verify entry/stop/target ordering (requires API keys)')
    ap.add_argument('--alpaca-feed', default='sip', choices=['sip','iex','boats','otc'], help='Alpaca data feed')
    ap.add_argument('--alpaca-limit', type=int, default=10000, help='Max datapoints per page for trades fetch')
    ap.add_argument('--alpaca-asof', help='asof date (YYYY-MM-DD) for symbol mapping')
    ap.add_argument('--session-tz', default='US/Eastern', help='Timezone of session timestamps (e.g., US/Eastern)')
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

    # Top symbols by P&L
    sym_pnl = Counter()
    sym_reasons = defaultdict(Counter)
    for s in entries:
        sym = s.get('symbol')
        sym_pnl[sym] += float(s.get('total_pnl', 0.0))
        for r, c in (s.get('sell_reasons') or {}).items():
            sym_reasons[sym][r] += int(c)
    if sym_pnl:
        print("\nTop symbols by P&L:")
        for sym, pnl in sym_pnl.most_common(10):
            reasons = ', '.join(f"{r}:{c}" for r,c in sym_reasons[sym].most_common())
            print(f"  - {sym}: ${pnl:.0f} | exits: {reasons}")

    # Top 10 winners/losers (entries)
    entries_sorted = sorted(entries, key=lambda s: float(s.get('total_pnl', 0.0)), reverse=True)
    print("\nTop 10 Winners:")
    for s in entries_sorted[:10]:
        print(f"  - {s.get('symbol')} ${float(s.get('total_pnl',0.0)):.0f} | entry {s.get('entry_time')} | reasons: {','.join((s.get('sell_reasons') or {}).keys())}")
    print("Top 10 Losers:")
    for s in entries_sorted[-10:][::-1]:
        print(f"  - {s.get('symbol')} ${float(s.get('total_pnl',0.0)):.0f} | entry {s.get('entry_time')} | reasons: {','.join((s.get('sell_reasons') or {}).keys())}")

    # Optional MFE/MAE using OHLC data (and validations)
    if args.date and pd is not None:
        ohlc_path = Path(args.ohlc_dir) / f"ohlcv_1min_{args.date}.json.gz"
        if not ohlc_path.exists():
            print(f"MFE/MAE: OHLC file not found: {ohlc_path}")
            return
        # Load OHLC once
        with gzip.open(ohlc_path, 'rt') as f:
            all_data = jsonlib.load(f)
        sym_to_df = {}
        for rec in all_data:
            sym = rec.get('symbol')
            mins = rec.get('minutes') or []
            if not mins:
                continue
            df = pd.DataFrame(mins)
            if 'time' not in df.columns:
                continue
            df['timestamp'] = pd.to_datetime(f"{args.date} " + df['time'])
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            sym_to_df[sym] = df[['open','high','low','close','volume']]
        # Compute MFE/MAE per entry
        print("\nMFE/MAE (per trade):")
        for s in entries:
            sym = s.get('symbol')
            df = sym_to_df.get(sym)
            if df is None:
                continue
            entry_price = s.get('entry_price')
            rps = s.get('risk_per_share') or 0.0
            if entry_price is None or rps is None or rps == 0:
                continue
            et = try_parsetime(s.get('entry_time'))
            # infer exit time: last SELL execution timestamp
            ex_list = s.get('executions_list') or []
            sell_times = [try_parsetime(e.get('timestamp')) for e in ex_list if e.get('action') == 'SELL']
            xt = max([t for t in sell_times if t], default=None)
            if et is None:
                continue
            if xt is None:
                # fallback window of 30 minutes after entry
                xt = et
            w = df[(df.index >= et) & (df.index <= xt)]
            if w.empty:
                continue
            mfe_abs = (w['high'].max() - float(entry_price))
            mae_abs = (float(entry_price) - w['low'].min())
            mfe_r = mfe_abs / float(rps)
            mae_r = mae_abs / float(rps)
            # time to 2R
            target_2r = float(entry_price) + 2*float(rps)
            tt2 = None
            hit = w[w['high'] >= target_2r]
            if not hit.empty:
                tt2 = (hit.index[0] - et).seconds // 60
            print(f"  - {sym}: MFE {mfe_r:.1f}R / MAE {mae_r:.1f}R | time_to_2R: {tt2 if tt2 is not None else '-'} min")

        if args.validate:
            # Per-trade validation with failure counts
            print("\nValidation (per trade):")
            fails = Counter()
            rows = []
            for s in entries:
                sym = s.get('symbol')
                df = sym_to_df.get(sym)
                if df is None:
                    continue
                entry_price = s.get('entry_price')
                rps = s.get('risk_per_share') or 0.0
                if entry_price is None or rps is None or rps == 0:
                    continue
                et = try_parsetime(s.get('entry_time'))
                ex_list = s.get('executions_list') or []
                sell_times = [try_parsetime(e.get('timestamp')) for e in ex_list if e.get('action') == 'SELL']
                xt = max([t for t in sell_times if t], default=None)
                if et is None:
                    continue
                w = df[(df.index >= et) & ((df.index <= xt) if xt else (df.index >= et))]
                entry_ok = False
                first2R_ok = None
                stop_ok = None
                bailout_ok = None
                extension_ok = None
                # Entry high ≥ entry price at entry minute
                if et in df.index:
                    entry_ok = df.loc[et]['high'] + 1e-9 >= float(entry_price)
                else:
                    entry_ok = True  # cannot verify if missing
                # First target claim check
                if 'first_target' in (s.get('sell_reasons') or {}):
                    target_2r = float(entry_price) + 2*float(rps)
                    first2R_ok = not w.empty and (w['high'] >= target_2r).any()
                # Stop claim check
                if 'stop_loss' in (s.get('sell_reasons') or {}):
                    stop = float(entry_price) - float(rps)
                    stop_ok = not w.empty and (w['low'] <= stop + 1e-6).any()
                # Bailout timing check
                if 'breakout_or_bailout' in (s.get('sell_reasons') or {}) or 'no_immediate_breakout' in (s.get('sell_reasons') or {}):
                    if xt is not None and et in df.index and xt in df.index:
                        bars = len(df.loc[et:xt]) - 1
                        bailout_ok = bars <= max(1, int(args.grace_bars))
                # Extension bar check (approx)
                if 'extension_bar' in (s.get('sell_reasons') or {}):
                    ext_time = next((try_parsetime(e.get('timestamp')) for e in ex_list if e.get('action')=='SELL' and e.get('reason')=='extension_bar'), None)
                    if ext_time is not None and ext_time in df.index:
                        recent = df.loc[:ext_time].tail(11).iloc[:-1]
                        if not recent.empty:
                            avg_range = (recent['high']-recent['low']).mean()
                            avg_vol = recent['volume'].mean()
                            cur = df.loc[ext_time]
                            extension_ok = ((cur['high']-cur['low']) > 2.0*avg_range) and (cur['volume'] > 1.5*avg_vol)
                # Count fails
                if not entry_ok:
                    fails['entry'] += 1
                if first2R_ok is False:
                    fails['first2R'] += 1
                if stop_ok is False:
                    fails['stop'] += 1
                if bailout_ok is False:
                    fails['bailout'] += 1
                if extension_ok is False:
                    fails['extension'] += 1
                rows.append({
                    'symbol': sym,
                    'entry_time': s.get('entry_time'),
                    'entry_ok': entry_ok,
                    'first2R_ok': first2R_ok,
                    'stop_ok': stop_ok,
                    'bailout_ok': bailout_ok,
                    'extension_ok': extension_ok,
                    'total_pnl': s.get('total_pnl'),
                    'sell_reasons': ','.join((s.get('sell_reasons') or {}).keys()),
                })
            if fails:
                print("\nValidation failures:")
                for k, v in fails.items():
                    print(f"  - {k}: {v}")
            if args.validate_csv:
                import csv
                with open(args.validate_csv, 'w', newline='') as cf:
                    writer = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Wrote validation CSV: {args.validate_csv}")
        # Optional: verify entry/stop ordering using Alpaca trades
        if args.verify_trades and fetch_trades_for_symbols is not None:
            print("\nTrades verification (Alpaca):")
            # Window: per entry, from entry_time to exit time (last SELL) or +2 minutes fallback
            # Convert session times to UTC using provided timezone for accurate Alpaca queries
            try:
                session_tz = pytz.timezone(args.session_tz)
            except Exception:
                print(f"Invalid session timezone: {args.session_tz}; falling back to US/Eastern")
                session_tz = pytz.timezone('US/Eastern')

            def to_utc(dt: datetime) -> datetime:
                if dt.tzinfo is None:
                    # Localize then convert
                    dt_local = session_tz.localize(dt)
                else:
                    dt_local = dt
                return dt_local.astimezone(pytz.UTC)

            for s in entries:
                sym = s.get('symbol') or s.get('alert_symbol')
                et = try_parsetime(s.get('entry_time'))
                ex_list = s.get('executions_list') or s.get('executions') or []
                sell_times = [try_parsetime(e.get('timestamp')) for e in ex_list if e.get('action') == 'SELL']
                xt = max([t for t in sell_times if t], default=None)
                if not sym or not et:
                    continue
                # small default window if no exit timestamp
                if xt is None:
                    xt = et
                et_utc = to_utc(et)
                xt_utc = to_utc(xt)
                entry_px = s.get('entry_price')
                rps = s.get('risk_per_share') or 0.0
                stop_px = None
                target_px = None
                # derive stop if present in reasons
                if 'stop_loss' in (s.get('sell_reasons') or {}):
                    if entry_px is not None and rps:
                        stop_px = float(entry_px) - float(rps)
                if 'first_target' in (s.get('sell_reasons') or {}):
                    if entry_px is not None and rps:
                        target_px = float(entry_px) + 2*float(rps)
                try:
                    tf_map = fetch_trades_for_symbols([sym], start=et_utc.isoformat(), end=xt_utc.isoformat(),
                                                      limit=args.alpaca_limit, feed=args.alpaca_feed, asof=args.alpaca_asof)
                    tdf = tf_map.get(sym)
                    res = verify_ordering(tdf, et_utc, xt_utc, float(entry_px) if entry_px is not None else None,
                                          stop_price=stop_px, target_price=target_px)
                    print(f"  - {sym} {et_utc} → {xt_utc}: entry_seen={res['entry_seen']} stop_after_entry={res['stop_seen_after_entry']} target_after_entry={res['target_seen_after_entry']}")
                except Exception as e:
                    print(f"  - {sym} {et_utc} → {xt_utc}: trades verification failed: {e}")
    elif args.date and pd is None:
        print("pandas not available; skipping MFE/MAE computation")

if __name__ == '__main__':
    main()
