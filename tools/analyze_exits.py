#!/usr/bin/env python3
"""
Exit Quality Analyzer

For each trade in a backtest sessions JSON, evaluates:
- Pre-exit MFE (max favorable excursion) and captured R
- Post-exit window highs/lows (e.g., 10 minutes) to flag early exits
- Late exits (gave back > 1R vs MFE before exit)
- Data coverage (missing minutes) and average volume

Usage:
  python tools/analyze_exits.py \
    --file results/logs/backtest_2025-08-13_152248_sessions.json \
    --date 2025-08-13 \
    --window-minutes 10 \
    --r-threshold 1.0 \
    --csv results/logs/exit_audit_2025-08-13.csv
"""

import argparse
import csv
import gzip
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def load_sessions(path: str):
    with open(path) as f:
        return json.load(f)


def load_ohlc_1min(date: str, ohlc_dir: Path) -> dict[str, pd.DataFrame]:
    path = ohlc_dir / f"ohlcv_1min_{date}.json.gz"
    if not path.exists():
        return {}
    with gzip.open(path, "rt") as f:
        all_data = json.load(f)
    out: dict[str, pd.DataFrame] = {}
    for rec in all_data:
        sym = rec.get("symbol")
        mins = rec.get("minutes") or []
        if not mins:
            continue
        df = pd.DataFrame(mins)
        if "time" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(f"{date} " + df["time"])  # local timestamps per cache format
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        out[sym] = df[["open", "high", "low", "close", "volume"]]
    return out


def try_time(x):
    if isinstance(x, datetime):
        return x
    try:
        return datetime.fromisoformat(x)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to *_sessions.json")
    ap.add_argument("--date", required=True, help="Date YYYY-MM-DD for OHLC lookup")
    ap.add_argument("--ohlc-dir", default="shared_cache/ohlcv_1min_bars", help="Dir with ohlcv_1min_<date>.json.gz")
    ap.add_argument("--window-minutes", type=int, default=10, help="Post-exit window minutes")
    ap.add_argument("--r-threshold", type=float, default=1.0, help="R threshold for early-exit flag")
    ap.add_argument("--min-avg-volume", type=float, default=0.0, help="Flag low volume when avg vol/min below this")
    ap.add_argument("--csv", help="Optional path to write CSV results")
    args = ap.parse_args()

    sessions = load_sessions(args.file)
    entries = [s for s in sessions if s.get("position_entered")]
    if not entries:
        print("No entries found in sessions file.")
        return

    sym_to_df = load_ohlc_1min(args.date, Path(args.ohlc_dir))
    if not sym_to_df:
        print(f"1-min OHLC file not found or empty for date {args.date} in {args.ohlc_dir}")
        return

    rows = []
    flags_counter = Counter()
    labels_counter = Counter()
    skipped = Counter()
    for s in entries:
        sym = s.get("symbol")
        df = sym_to_df.get(sym)
        if df is None or df.empty:
            skipped['no_ohlc'] += 1
            continue

        entry_time = try_time(s.get("entry_time"))
        entry_price = s.get("entry_price")
        rps = s.get("risk_per_share") or 0.0
        if entry_time is None or entry_price is None:
            skipped['missing_entry'] += 1
            continue
        if not rps:
            skipped['missing_risk_per_share'] += 1
            continue

        # Determine exit time and price from last SELL execution
        ex_list = s.get("executions_list") or []
        sell_execs = [e for e in ex_list if (e.get("action") == "SELL")]
        exit_time = try_time(sell_execs[-1].get("timestamp")) if sell_execs else None
        exit_price = sell_execs[-1].get("price") if sell_execs else None
        exit_reason = sell_execs[-1].get("reason") if sell_execs else None

        if exit_time is None or exit_price is None:
            skipped['no_exit'] += 1
            continue

        # Window A: entry -> exit
        wa = df[(df.index >= entry_time) & (df.index <= exit_time)]
        # Window B: post-exit window
        wb_end = exit_time + timedelta(minutes=int(args.window_minutes))
        wb = df[(df.index > exit_time) & (df.index <= wb_end)]

        # Coverage metrics (1-min bars)
        # Expected minutes inclusive between entry and exit
        exp_a = max(1, int((exit_time - entry_time).total_seconds() // 60) + 1)
        got_a = len(wa)
        cov_a = got_a / exp_a if exp_a > 0 else 1.0
        exp_b = int((wb_end - exit_time).total_seconds() // 60)
        got_b = len(wb)
        cov_b = got_b / exp_b if exp_b > 0 else 1.0

        # Pre-exit MFE
        mfe_abs = (wa["high"].max() - float(entry_price)) if not wa.empty else 0.0
        mfe_r = mfe_abs / float(rps) if rps else 0.0

        # Captured R (approx): total_pnl / (rps * initial_shares)
        total_pnl = float(s.get("total_pnl", 0.0) or 0.0)
        initial_shares = int(s.get("initial_shares") or 0)
        denom = float(rps) * max(1, initial_shares)
        captured_r = total_pnl / denom if denom else 0.0
        left_on_table_r = max(0.0, mfe_r - captured_r)

        # Post-exit window extremes
        post_high = float(wb["high"].max()) if not wb.empty else float(exit_price)
        post_low = float(wb["low"].min()) if not wb.empty else float(exit_price)
        post_high_r = (post_high - float(entry_price)) / float(rps) if rps else 0.0
        post_low_r = (float(entry_price) - post_low) / float(rps) if rps else 0.0

        # Time metrics
        time_to_exit_min = max(0, int((exit_time - entry_time).total_seconds() // 60))
        # First time of MFE
        mfe_time_min = None
        if not wa.empty:
            maxh = wa["high"].max()
            first_idx = wa.index[wa["high"] >= maxh]
            if len(first_idx) > 0:
                mfe_time_min = max(0, int((first_idx[0] - entry_time).total_seconds() // 60))

        # Flags
        early_exit = (exit_reason not in ("stop_loss",)) and (post_high_r >= float(args.r_threshold))
        too_tight_bailout = (exit_reason in ("breakout_or_bailout", "no_immediate_breakout")) and (post_high_r >= 2.0)
        # Late exit: only if there was meaningful MFE before exit and we captured at least 1R less than available
        late_exit = (mfe_r >= 1.0) and ((mfe_r - captured_r) >= 1.0)
        # Favorable post-exit context: no further upside (<0.5R) or flushed >=0.5R after exit
        favorable_post_exit = (post_high_r < 0.5) or ((float(exit_price) - post_low) / float(rps) >= 0.5 if rps else False)

        # Volume
        avg_vol_a = float(wa["volume"].mean()) if not wa.empty else 0.0
        low_volume = (args.min_avg_volume > 0 and avg_vol_a > 0 and avg_vol_a < args.min_avg_volume)

        # Aggregate flags
        for name, val in (
            ("early_exit", early_exit),
            ("late_exit", late_exit),
            ("too_tight_bailout", too_tight_bailout),
            ("favorable_post_exit", favorable_post_exit),
            ("low_data_coverage", (cov_a < 0.8 or cov_b < 0.8)),
            ("low_volume", low_volume),
        ):
            if val:
                flags_counter[name] += 1

        # Exclusive label (best-effort) for quick scanning
        label = None
        if exit_reason == "stop_loss":
            label = "stopped"
        elif too_tight_bailout:
            label = "too_tight_bailout"
        elif early_exit:
            label = "early_exit"
        elif late_exit:
            label = "late_exit"
        else:
            label = "on_time"
        labels_counter[label] += 1

        rows.append({
            "symbol": sym,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "risk_per_share": rps,
            "captured_R": round(captured_r, 2),
            "MFE_R_before_exit": round(mfe_r, 2),
            "left_on_table_R": round(left_on_table_r, 2),
            "post_exit_high": round(post_high, 4),
            "post_exit_low": round(post_low, 4),
            "post_exit_high_R": round(post_high_r, 2),
            "post_exit_low_R": round(post_low_r, 2),
            "time_to_exit_min": time_to_exit_min,
            "time_to_MFE_min": mfe_time_min,
            "exit_reason": exit_reason,
            "pullback_bars": s.get("last_pullback_candles"),
            "retrace_pct": s.get("last_retrace_percentage"),
            "coverage_entry_exit": round(cov_a, 2),
            "coverage_post_exit": round(cov_b, 2),
            "avg_vol_per_min": round(avg_vol_a, 1),
            "early_exit": early_exit,
            "late_exit": late_exit,
            "too_tight_bailout": too_tight_bailout,
            "favorable_post_exit": favorable_post_exit,
            "low_data_coverage": (cov_a < 0.8 or cov_b < 0.8),
            "low_volume": low_volume,
            "label": label,
        })

    # Print summary
    total = len(rows)
    print("=== Exit Quality ===")
    print(f"Trades analyzed: {total}")
    if skipped:
        print("Skipped trades (reasons):")
        for k, v in skipped.items():
            print(f"  - {k}: {v}")
    print("Flags (non-exclusive; a trade can set multiple):")
    for k in ("early_exit", "late_exit", "too_tight_bailout", "favorable_post_exit", "low_data_coverage", "low_volume"):
        print(f"  - {k}: {flags_counter.get(k, 0)}")
    print("Exclusive labels:")
    for k, v in labels_counter.items():
        print(f"  - {k}: {v}")

    # Write CSV if requested
    if args.csv and rows:
        fieldnames = list(rows[0].keys())
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {len(rows)} rows to {outp}")


if __name__ == "__main__":
    main()
