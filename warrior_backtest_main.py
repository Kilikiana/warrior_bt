"""
Ross Cameron's Complete Trading Strategy Implementation

STRATEGY OVERVIEW:
This implements Ross Cameron's momentum trading strategy focusing on ACTION alerts,
bull flag patterns, and disciplined risk management with MACD confirmation.

KEY COMPONENTS:
1. MACDCalculator - Standard MACD (12,26,9) for trend confirmation
2. PullbackDetector - Detects Ross's bull flag entry pattern after ACTION alerts
3. TrendAnalyzer - 9/20 EMA trends and extension bar detection
4. RiskManager - 2% risk rule, 2:1 profit targets, position scaling
5. RossCameronStrategy - Main orchestrator combining all components

ENTRY LOGIC (CRITICAL):
1. ACTION Alert triggers (20-30%+ move) → Add to watchlist (DON'T ENTER!)
2. Wait for pullback (1-10% from alert high) - this forms the "flag"
3. First GREEN candle that breaks ABOVE pullback high = ENTRY SIGNAL
4. Must have MACD bullish confirmation (MACD > signal line)
5. Must have EMA uptrend confirmation (9 EMA > 20 EMA)

POSITION MANAGEMENT:
- Scale out 50% at 2:1 profit target (4% gain if risking 2%)
- Move stop to breakeven after first scale
- Scale out more on extension bars (big green candles with volume)
- 10% trailing stop after 15% gains
- "Breakout or Bailout" - exit if down 2% after 10 minutes
- Exit on MACD bearish crossover

DISCIPLINE RULES:
- Only one entry attempt per ACTION alert
- Stop trading when market MACD turns bearish
- No chasing - wait for proper setup
- Focus on process, not profits

Example: CLRO July 1st 7:00 AM
- 6:58: $13.16 → 7:00: ACTION alert $15.80 → 7:01: Pullback to $14.50-$15.33
- 7:02: Green candle breaks $15.33 (pullback high) → ENTRY at $15.34
- Result: Ran to $19.28 for massive profit
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
import logging
from pathlib import Path

# Import our TA-Lib based calculators
from tech_analysis.ema_calculator import EMACalculator, RossCameronEMAConfig, EMACrossType
from tech_analysis.macd_calculator import (MACDCalculator, RossCameronMACDConfig, 
                             MACDSignalType, MACDState, MACDAnalysis)
from core.config import get_scan_file, get_log_file, LOGS_DIR
from data.ohlc_loader import load_symbol_ohlc_data


def setup_logging(log_file: str | None = None, log_level: str = "INFO") -> None:
    """Configure console and optional file logging.

    Args:
        log_file: Path to log file (creates parent dirs). If None, logs only to console.
        log_level: Logging level string (e.g., DEBUG, INFO, WARNING, ERROR).
    """
    # Normalize level
    level = getattr(logging, str(log_level).upper(), logging.INFO)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates on re-init
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode="a")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            # Fall back to console-only but surface the issue
            logger.warning(f"Failed to initialize file logging at {log_file}: {e}")

    logger.info("Logging initialized | level=%s | file=%s", logging.getLevelName(level), log_file or "<console-only>")
    # Include git SHA (if available) for reproducibility
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        logger.info("Git SHA: %s", sha)
    except Exception:
        pass



    # moved to data.ohlc_loader.load_symbol_ohlc_data

def run_backtest(
    date: str,
    start_time: str = "06:00",
    end_time: str = "11:30",
    account_balance: float = 30000,
    bailout_grace_bars: int = 2,
    use_5min_first_red_exit: bool = False,
    exit_on_ema_vwap_break: bool = True,
    stop_slippage_cents: float = 0.0,
    entry_slippage_cents: float = 0.0,
    add_back_enabled: bool = True,
    max_add_backs: int = 2,
    add_back_cooldown_bars: int = 2,
    max_daily_loss: float | None = None,
    symbols: List[str] | None = None,
    exclude_symbols: List[str] | None = None,
    per_alert_sessions: bool = False,
    lockout_after_losers: int | None = None,
    lockout_minutes: int = 30,
):
    """
    Run backtest using existing modular components
    
    Args:
        date: Date in YYYY-MM-DD format
        start_time: Start time in HH:MM format 
        end_time: End time in HH:MM format
        account_balance: Starting account balance
    """
    import argparse
    import json
    import csv
    
    # Import our existing modular components
    try:
        from position_management.position_sizer import PositionSizer
        from position_management.position_tracker import PositionTracker
        from risk_config import RISK_PER_TRADE_PCT
        from tech_analysis.patterns.pattern_monitor import PatternMonitoringSession, ActionAlert
    except ImportError:
        logging.error("Required modules not available. Using simplified simulation.")
        return
    
    logging.info("🚀 Running Ross Cameron Backtest: %s (%s - %s)", date, start_time, end_time)
    logging.info("💰 Account Balance: $%s", f"{account_balance:,}")
    
    # Load ACTION alerts for the date (skip if seeding from gappers)
    seed_mode = getattr(args, 'seed_mode', 'alerts') if 'args' in globals() else 'alerts'
    scan_data = {}
    alerts_file = get_scan_file(date)
    if seed_mode == 'alerts':
        if not alerts_file.exists():
            logging.warning("No alerts found for %s", date)
            return
        with open(alerts_file) as f:
            scan_data = json.load(f)
    
    # Build alert stream
    # Optional symbol filtering
    symbols_set = set(s.upper() for s in symbols) if symbols else None
    exclude_set = set(s.upper() for s in exclude_symbols) if exclude_symbols else None
    alerts_by_symbol: Dict[str, List[dict]] = {}
    seed_mode = getattr(args, 'seed_mode', 'alerts') if 'args' in globals() else 'alerts'
    if seed_mode == 'alerts':
        # Use ACTION alerts (STRONG_SQUEEZE_HIGH_RVOL)
        all_alerts = scan_data.get('all_alerts', [])
        for alert in all_alerts:
            if alert.get('strategy') != 'STRONG_SQUEEZE_HIGH_RVOL':
                continue
            t = str(alert.get('time', ''))
            if not (start_time <= t <= end_time):
                continue
            symu = str(alert.get('symbol','')).upper()
            if symbols_set is not None and symu not in symbols_set:
                continue
            if exclude_set is not None and symu in exclude_set:
                continue
            alerts_by_symbol.setdefault(symu, []).append(dict(alert))
    else:
        # Seed from Top Gappers at 9:30 using results/criteria_scans open_gap_results
        import csv as _csv
        from pathlib import Path as _Path
        gap_file = _Path('results/criteria_scans/open_gap_results.csv')
        # Try dated file fallback
        if not gap_file.exists():
            import glob as _glob
            cands = sorted(_glob.glob(f"results/criteria_scans/open_gap_results_{date}_*.csv"))
            gap_file = _Path(cands[0]) if cands else gap_file
        # If still missing, compute a simple fallback from cached bars
        if not gap_file.exists():
            try:
                from tools.simple_gappers import compute_open_gappers
                gap_file = compute_open_gappers(date, min_gap_pct=float(getattr(args, 'gng_min_gap_pct', 4.0) or 4.0))
                logging.info("Generated fallback top gappers: %s", gap_file)
            except Exception as e:
                logging.info("Failed to generate fallback gappers: %s", e)
        gap_rows: List[dict] = []
        if gap_file.exists():
            try:
                with open(gap_file, 'r') as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        # accept both standard and fallback csv
                        if row.get('date') == date or gap_file.name.endswith('_fallback.csv'):
                            gap_rows.append(row)
            except Exception:
                gap_rows = []
        # Best-effort float map from HOD scanner (if present)
        hod_float_map = {}
        try:
            from core.config import get_scan_file as _get_scan
            _af = _get_scan_file(date) if 'get_scan_file' in globals() else None
        except Exception:
            _af = None
        if _af and _af.exists():
            try:
                import json as _json
                _d = _json.load(open(_af))
                for a in _d.get('all_alerts', []):
                    symu = str(a.get('symbol','')).upper()
                    if 'float' in a and symu not in hod_float_map:
                        try:
                            hod_float_map[symu] = float(a['float'])
                        except Exception:
                            pass
            except Exception:
                pass
        # Compute/attach premarket volume if not present
        def _compute_pm_vol(sym: str) -> int:
            try:
                import json, gzip
                from core.config import OHLCV_1MIN_DIR as _MIN_DIR
                p = _MIN_DIR / f"ohlcv_1min_{date}.json.gz"
                if not p.exists():
                    return 0
                with gzip.open(p, 'rt') as f:
                    allm = json.load(f)
                rec = next((r for r in allm if r.get('symbol') == sym), None)
                if not rec:
                    return 0
                minutes = rec.get('minutes') or rec.get('bars') or []
                tot = 0
                for m in minutes:
                    t = m.get('time')
                    if t and '04:00' <= t < '09:30':
                        tot += int(m.get('volume') or 0)
                return tot
            except Exception:
                return 0
        # Apply min premarket volume and rank by gap%
        min_pm = int(getattr(args, 'gappers_min_premarket_volume', 0) or 0)
        ranked: List[dict] = []
        for r in gap_rows:
            try:
                symu = str(r.get('symbol','')).upper()
                pmv = int(float(r.get('premarket_volume') or 0))
                if pmv == 0:
                    pmv = _compute_pm_vol(symu)
                if min_pm > 0 and pmv < min_pm:
                    continue
                r['premarket_volume'] = pmv
                ranked.append(r)
            except Exception:
                continue
        ranked.sort(key=lambda x: float(x.get('gap_pct') or 0.0), reverse=True)
        top_n = max(1, int(getattr(args, 'gappers_top_n', 5) or 5))
        ranked = ranked[:top_n]
        # Load simple news table if available to provide a catalyst string
        news_map = {}
        try:
            news_file = _Path('results/criteria_scans/backtest_news_results.csv')
            if news_file.exists():
                with open(news_file, 'r') as nf:
                    nreader = _csv.DictReader(nf)
                    for r in nreader:
                        sym = str(r.get('symbol','')).upper()
                        headline = r.get('first_headline') or ''
                        try:
                            cnt = int(float(r.get('news_count') or 0))
                        except Exception:
                            cnt = 0
                        news_map[sym] = (cnt, headline)
        except Exception:
            news_map = {}
        # Build pseudo-alerts at 09:30 (selected)
        for r in ranked:
            try:
                symu = str(r.get('symbol','')).upper()
                if not symu:
                    continue
                if symbols_set is not None and symu not in symbols_set:
                    continue
                if exclude_set is not None and symu in exclude_set:
                    continue
                open_price = float(r.get('open_price') or r.get('open') or 0.0)
                prev_close = float(r.get('prev_close') or 0.0)
                gap_pct = float(r.get('gap_pct') or 0.0)
                # Respect Gap & Go min gap
                min_gap = float(getattr(args, 'gng_min_gap_pct', 4.0) or 4.0)
                if gap_pct < min_gap:
                    continue
                # Time at 09:30
                t = '09:30'
                alert = {
                    'symbol': symu,
                    'time': t,
                    'price': open_price,
                    'strategy': 'STRONG_SQUEEZE_HIGH_RVOL',
                    'description': (news_map.get(symu, (0, ''))[1] if news_map else ''),
                    'prev_close': prev_close,
                    'float': hod_float_map.get(symu),
                    'premarket_volume': r.get('premarket_volume'),
                }
                alerts_by_symbol.setdefault(symu, []).append(alert)
            except Exception:
                continue
    # For each symbol, pick a pole-anchored alert (first alert at/after the first detected pole high)
    def _find_first_pole_anchor_time(df: pd.DataFrame, start_ts: datetime, end_ts: datetime) -> Optional[datetime]:
        if df is None or df.empty:
            return None
        # clamp to window
        try:
            window = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            if window.empty:
                return None
        except Exception:
            window = df
        # Prefer a local high followed by 2 red bars (flagpole high then pullback), scanning forward
        lookback = 5
        for i in range(lookback, len(window) - 2):
            try:
                # Local max check over previous 'lookback' bars
                prev_highs = window['high'].iloc[i - lookback:i]
                cur_high = float(window['high'].iloc[i])
                if cur_high >= prev_highs.max():
                    # Next two bars red?
                    o1, c1 = window.iloc[i+1]['open'], window.iloc[i+1]['close']
                    o2, c2 = window.iloc[i+2]['open'], window.iloc[i+2]['close']
                    if c1 < o1 and c2 < o2:
                        return window.index[i].to_pydatetime()
            except Exception:
                pass
        # Fallback: first occurrence of two consecutive red bars; anchor at first red
        red_len = 0
        for i in range(1, len(window)):
            o, c = window.iloc[i]['open'], window.iloc[i]['close']
            if c < o:
                red_len += 1
            else:
                red_len = 0
            if red_len == 2:
                return window.index[i-1].to_pydatetime()
        return None

    if per_alert_sessions:
        selected_alerts: List[dict] = []
        for symu, sym_alerts in alerts_by_symbol.items():
            sym_alerts = sorted(sym_alerts, key=lambda a: a.get('time',''))
            # For BullFlag Simple V2, cluster redundant alerts within the same move
            # to avoid starting multiple sessions unnecessarily.
            if getattr(args, 'pattern', 'bull_flag') == 'bull_flag_simple_v2':
                try:
                    ohlc_df = load_symbol_ohlc_data(symu, date, timeframe="1min")
                except Exception:
                    ohlc_df = None
                if ohlc_df is not None and not ohlc_df.empty:
                    clustered_any = False
                    def _ts(hm: str) -> datetime:
                        return datetime.strptime(f"{date} {hm}", "%Y-%m-%d %H:%M")
                    # Removed red-alert prefilter to preserve original alert stream
                    n = len(sym_alerts)
                    idx = 0
                    while idx < n:
                        clustered_any = True
                        seed = sym_alerts[idx]
                        t_seed = _ts(seed.get('time',''))
                        # Roll forward start alert while no red has occurred and subsequent alerts are green bars
                        # Find the first red bar after current seed time
                        def _first_red_after(ts: datetime) -> Optional[datetime]:
                            try:
                                df = ohlc_df.loc[ohlc_df.index > ts]
                                for t, row in df.iterrows():
                                    if float(row['close']) < float(row['open']):
                                        return t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
                            except Exception:
                                pass
                            return None
                        # Helper to check if a given alert time bar is green
                        def _is_green_at(ts: datetime) -> bool:
                            try:
                                if ts in ohlc_df.index:
                                    r = ohlc_df.loc[ts]
                                else:
                                    # pad to next index at/after ts
                                    pos = ohlc_df.index.get_indexer([ts], method='pad')
                                    r = ohlc_df.iloc[int(pos[0])] if pos[0] != -1 else None
                                if r is None:
                                    return False
                                return float(r['close']) > float(r['open'])
                            except Exception:
                                return False
                        # Advance seed to latest alert prior to first red, if each interim alert bar is green
                        first_red = _first_red_after(t_seed)
                        last_idx = idx
                        if first_red is not None:
                            j = idx + 1
                            while j < n:
                                t_next = _ts(sym_alerts[j].get('time',''))
                                if t_next >= first_red:
                                    break
                                # Only roll forward if the interim alert bar is green (still in pre-pullback)
                                if _is_green_at(t_next):
                                    last_idx = j
                                    j += 1
                                    continue
                                else:
                                    break
                            seed = sym_alerts[last_idx]
                            t_seed = _ts(seed.get('time',''))
                        # Determine cluster end: first breakout after first red, or timeout (6 bars from first red)
                        def _cluster_end_after(first_red_ts: Optional[datetime]) -> datetime:
                            if first_red_ts is None:
                                # No red observed; cluster ends at next alert or end_time
                                try:
                                    return datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
                                except Exception:
                                    return ohlc_df.index.max().to_pydatetime()
                            try:
                                df = ohlc_df
                                # Locate first red index
                                if first_red_ts in df.index:
                                    fr_pos = int(df.index.get_loc(first_red_ts))
                                else:
                                    pos = df.index.get_indexer([first_red_ts], method='backfill')
                                    fr_pos = int(pos[0]) if pos[0] != -1 else None
                                if fr_pos is None:
                                    return first_red_ts
                                # Build contiguous red segment and track last_red_high
                                last_red_high = float(df['high'].iloc[fr_pos])
                                k = fr_pos
                                while k + 1 < len(df):
                                    k += 1
                                    row = df.iloc[k]
                                    if float(row['close']) < float(row['open']):
                                        last_red_high = float(row['high'])
                                    else:
                                        break
                                # Now search for breakout within remaining bars up to timeout
                                timeout_idx = min(len(df) - 1, fr_pos + 5)  # first red counts as 1
                                br_time: Optional[datetime] = None
                                m = k
                                while m <= timeout_idx:
                                    row = df.iloc[m]
                                    if float(row['close']) > float(row['open']) and float(row['high']) >= last_red_high:
                                        t = df.index[m]
                                        br_time = t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
                                        break
                                    m += 1
                                if br_time is not None:
                                    return br_time
                                t = df.index[timeout_idx]
                                return t.to_pydatetime() if hasattr(t, 'to_pydatetime') else t
                            except Exception:
                                return first_red_ts
                        cluster_end = _cluster_end_after(first_red)
                        # Add the chosen (rolled) seed alert for this cluster
                        selected_alerts.append(seed)
                        # Advance idx beyond cluster_end
                        while idx < n:
                            t_cur = _ts(sym_alerts[idx].get('time',''))
                            if t_cur <= cluster_end:
                                idx += 1
                            else:
                                break
                        continue
                    # If we clustered at least one item for this symbol, skip fallback extension
                    if clustered_any:
                        continue
                # Fallback: if no OHLC or error, include raw alerts
                selected_alerts.extend(sym_alerts)
            else:
                selected_alerts.extend(sym_alerts)
        action_alerts = sorted(selected_alerts, key=lambda a: (a.get('time',''), a.get('symbol','')))
    else:
        selected_alerts: List[dict] = []
        for symu, sym_alerts in alerts_by_symbol.items():
            sym_alerts = sorted(sym_alerts, key=lambda a: a.get('time',''))
            # Load symbol OHLC to compute anchor
            ohlc_df = load_symbol_ohlc_data(symu, date, timeframe="1min")
            start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
            anchor_time = _find_first_pole_anchor_time(ohlc_df, start_dt, end_dt)
            chosen = None
            if anchor_time is not None:
                anchor_hm = anchor_time.strftime('%H:%M')
                for a in sym_alerts:
                    if str(a.get('time','')) >= anchor_hm:
                        chosen = a
                        break
            # Fallback to earliest alert
            if chosen is None and sym_alerts:
                chosen = sym_alerts[0]
            if chosen is not None:
                selected_alerts.append(chosen)
            # Log counts to ensure we aren't missing alerts
            try:
                total_in_win = len(sym_alerts)
                if total_in_win > 0:
                    first_t = sym_alerts[0].get('time','?')
                    last_t = sym_alerts[-1].get('time','?')
                    logging.info("Alerts in window for %s: %d (first=%s, last=%s) | chosen=%s",
                                 symu, total_in_win, first_t, last_t, chosen.get('time','?') if chosen else '-')
            except Exception:
                pass
        # Build sorted list by time to start sessions in order
        action_alerts = sorted(selected_alerts, key=lambda a: a.get('time',''))
    # If an alerts-file is provided, load alerts directly (replay mode)
    action_alerts_loaded = False
    try:
        if 'args' in globals():
            alerts_file = getattr(args, 'alerts_file', None)
        else:
            alerts_file = None
    except Exception:
        alerts_file = None
    if alerts_file:
        try:
            import json as _json
            with open(alerts_file) as af:
                loaded = _json.load(af)
            # Expect list of dicts with symbol,time,price,strategy,description
            if isinstance(loaded, list):
                action_alerts = loaded
                action_alerts_loaded = True
                logging.info("Replayed alerts from file: %s (count=%d)", alerts_file, len(action_alerts))
            else:
                logging.warning("Alerts file does not contain a list: %s", alerts_file)
        except Exception as e:
            logging.warning("Failed to load alerts file %s: %s", alerts_file, e)

    if not action_alerts:
        logging.warning("No ACTION alerts found in time window")
        return
    
    if symbols_set is not None or exclude_set is not None:
        logging.info(
            "📊 Found %d ACTION alerts in window (include=%s | exclude=%s)",
            len(action_alerts),
            ','.join(sorted(symbols_set)) if symbols_set else '-',
            ','.join(sorted(exclude_set)) if exclude_set else '-',
        )
    else:
        logging.info("📊 Found %d ACTION alerts in trading window", len(action_alerts))
    
    # Initialize position management (tracker owns hard daily stop)
    position_tracker = PositionTracker(
        account_balance,
        max_positions=4,
        daily_risk_percentage=0.05,
        max_daily_loss=max_daily_loss,
    )
    position_sizer = PositionSizer()
    
    # Process each alert using the real pattern monitoring system
    run_ts = datetime.now().strftime("%H%M%S")
    # Persist selected alerts for reproducibility (save even if replayed, to keep a run-scoped copy)
    try:
        import json as _json
        alerts_out = LOGS_DIR / f"backtest_{date}_{run_ts}_alerts.json"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        to_dump = [{
            'symbol': a.get('symbol'),
            'time': a.get('time'),
            'price': a.get('price'),
            'strategy': a.get('strategy'),
            'description': a.get('description'),
        } for a in action_alerts]
        with open(alerts_out, 'w') as af:
            _json.dump(to_dump, af, indent=2)
        logging.info("Saved selected alerts: %s", alerts_out)
        # If gappers seed, also persist the watchlist used
        try:
            if seed_mode == 'gappers':
                wl = LOGS_DIR / f"backtest_{date}_{run_ts}_watchlist.txt"
                with open(wl, 'w') as wf:
                    wf.write("# Watchlist used (seed-mode gappers)\n")
                    for a in action_alerts:
                        wf.write(f"{a.get('symbol')}\n")
                logging.info("Saved watchlist: %s", wl)
        except Exception:
            pass
    except Exception as e:
        logging.warning("Failed to save selected alerts: %s", e)
    summaries_path = LOGS_DIR / f"backtest_{date}_{run_ts}_sessions.json"
    entries_csv_path = LOGS_DIR / f"backtest_{date}_{run_ts}_entries.csv"
    session_summaries: List[Dict] = []
    # Track symbol busy intervals to prevent overlapping trades across sequential per-alert sessions
    symbol_busy_intervals: Dict[str, List[Tuple[datetime, datetime]]] = {}
    lockout_until: Optional[datetime] = None
    recent_negatives: List[datetime] = []
    ohlc_data_cache: Dict[str, pd.DataFrame] = {}
    for i, alert_data in enumerate(action_alerts, 1):
        # If tracker indicates daily stop hit, halt new alerts
        try:
            if hasattr(position_tracker, 'should_halt_trading') and position_tracker.should_halt_trading():
                logging.warning("⛔ Daily stop active. Halting new entries.")
                break
        except Exception:
            pass
        symbol = alert_data['symbol']
        alert_time = datetime.strptime(f"{date} {alert_data['time']}", "%Y-%m-%d %H:%M")
        # Lockout window check
        if lockout_until is not None and alert_time < lockout_until:
            logging.info("⏸ In lockout until %s; skipping %s at %s", lockout_until.strftime('%H:%M'), symbol, alert_data['time'])
            continue
        alert_price = alert_data['price']
        
        logging.info("\n--- Processing Alert %d/%d ---", i, len(action_alerts))
        logging.info("Symbol: %s, Time: %s, Price: $%s", symbol, alert_data['time'], alert_price)
        logging.info("Strategy: %s, Description: %s", alert_data.get('strategy'), alert_data.get('description'))
        
        # If prior trade window for this symbol spans this alert, skip to avoid overlap
        try:
            busy = False
            for (s_et, s_xt) in symbol_busy_intervals.get(symbol, []):
                if s_et is not None and s_xt is not None and s_et <= alert_time < s_xt:
                    busy = True
                    logging.info("⏭️  Skipping %s at %s: overlaps active trade window %s → %s", symbol, alert_data['time'], s_et.strftime('%H:%M'), s_xt.strftime('%H:%M'))
                    break
            if busy:
                continue
        except Exception:
            pass

        # Calculate risk amount using centralized config (Ross: 1.25% default)
        try:
            risk_amount = float(account_balance) * float(RISK_PER_TRADE_PCT)
        except Exception:
            risk_amount = account_balance * 0.0125
        
        # Skip if symbol already has an active position (single active trade per symbol)
        try:
            if hasattr(position_tracker, 'active_positions') and symbol in getattr(position_tracker, 'active_positions', {}):
                logging.info("⏭️  Skipping %s: position already active", symbol)
                continue
        except Exception:
            pass
        # Check position limits
        try:
            ok_reason = position_tracker.can_open_position(risk_amount)
            try:
                ok, reason = ok_reason
            except Exception:
                ok, reason = bool(ok_reason), ""
            if not ok:
                logging.info("⚠️  Position limits/risk gating: skipping %s (%s)", symbol, reason)
                continue
        except Exception as e:
            logging.warning("⚠️  Position check failed for %s: %s", symbol, e)
            continue
        
        # Create ActionAlert object
        # Default alert object; will adjust to pole anchor after loading OHLC
        action_alert = ActionAlert(
            symbol=symbol,
            alert_time=alert_time,
            alert_price=alert_price,
            alert_high=alert_price,  # temporary; replaced with pole high when available
            volume_spike=alert_data.get('rvol_5min', 0) / 100,  # Convert % to ratio
            news_catalyst=alert_data.get('description')
        )
        # If seeded from gappers, carry prev_close if available
        try:
            if getattr(args, 'seed_mode', 'alerts') == 'gappers':
                pc = alert_data.get('prev_close')
                if pc is not None:
                    monitor_prev_close = float(pc)
                else:
                    monitor_prev_close = None
            else:
                monitor_prev_close = None
        except Exception:
            monitor_prev_close = None
        
        # Start pattern monitoring using our improved system
        from tech_analysis.patterns.pattern_monitor import SessionConfig
        # Build list of all alert datetimes for this symbol within the window
        try:
            sym_alerts_all = alerts_by_symbol.get(symbol.upper(), [])
            all_alert_times = []
            for a in sym_alerts_all:
                try:
                    all_alert_times.append(datetime.strptime(f"{date} {a.get('time','')}", "%Y-%m-%d %H:%M"))
                except Exception:
                    pass
        except Exception:
            all_alert_times = []

        session_cfg = SessionConfig(
            account_balance=account_balance,
            sizing_method="ross_dynamic",
            use_5min_first_red_exit=use_5min_first_red_exit,
            bailout_grace_bars=bailout_grace_bars,
            exit_on_ema_vwap_break=exit_on_ema_vwap_break,
            add_back_enabled=add_back_enabled,
            max_add_backs=max_add_backs,
            add_back_cooldown_bars=add_back_cooldown_bars,
            stop_slippage_cents=stop_slippage_cents,
            entry_slippage_cents=entry_slippage_cents,
            entry_confirm_mode=args.entry_confirm_mode,
            trigger_mode=args.trigger_mode,
            entry_confirmations=args.entry_confirmations,
            exit_on_macd_cross=bool(getattr(args, 'exit_on_macd_cross', False)),
            enable_extension_bar_exit=not bool(getattr(args, 'disable_extension_bar_exit', False)),
            enable_early_pullback_trim=not bool(getattr(args, 'disable_early_pullback_trim', False)),
            entry_cutoff_minutes=int(getattr(args, 'entry_cutoff_minutes', 15) or 15),
            entry_cutoff_time=datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M"),
            allow_manage_past_end=bool(args.manage_past_end),
            breakout_vol_mult=float(getattr(args, 'breakout_vol_mult', 0.0) or 0.0),
            min_pullback_avg_volume=float(getattr(args, 'min_pullback_avg_volume', 0.0) or 0.0),
            spread_cap_bps=float(getattr(args, 'spread_cap_bps', 0.0) or 0.0),
            require_macd_positive=bool(getattr(args, 'require_macd_positive', False)),
            all_alert_times=all_alert_times,
            # BFSv2 toggles
            v2_partial_on_alert_high=bool(getattr(args, 'v2_partial_on_alert_high', True)),
            v2_runner_enabled=bool(getattr(args, 'v2_runner_enabled', True)),
            v2_runner_macd_gate_minutes=int(getattr(args, 'v2_runner_macd_gate_minutes', 10) or 10),
            v2_runner_hard_cap_R=float(getattr(args, 'v2_runner_hard_cap_r', 3.0) or 3.0),
            v2_entry_confirm_ema=bool(getattr(args, 'v2_entry_confirm_ema', False)),
            v2_entry_confirm_macd=bool(getattr(args, 'v2_entry_confirm_macd', False)),
            v2_entry_confirmations=str(getattr(args, 'v2_entry_confirmations', 'none')),
            v2_enter_on_close_with_gate=bool(getattr(args, 'v2_enter_on_close_with_gate', False)),
            v2_min_stop_dollars=float(getattr(args, 'v2_min_stop_dollars', 0.10) or 0.10),
            # MACD gate + structure knobs
            v2_macd_gate_require_runner=bool(getattr(args, 'v2_macd_gate_require_runner', True)),
            v2_macd_gate_require_no_progress=bool(getattr(args, 'v2_macd_gate_require_no_progress', False)),
            v2_no_progress_thresh_r=float(getattr(args, 'v2_no_progress_thresh_r', 0.0) or 0.0),
            v2_max_wait_bars=int(getattr(args, 'v2_max_wait_bars', 6) or 6),
            v2_retrace_cap_pct=float(getattr(args, 'v2_retrace_cap_pct', 0.50) or 0.50),
            v2_require_vwap_above=bool(getattr(args, 'v2_require_vwap_above', False)),
            v2_entry_confirm_ema5m=bool(getattr(args, 'v2_entry_confirm_ema5m', False)),
            v2_first_partial_pct=float(getattr(args, 'v2_first_partial_pct', 0.5) or 0.5),
            v2_add_back_enabled=bool(getattr(args, 'v2_add_back_enabled', False)),
            v2_add_back_pct=float(getattr(args, 'v2_add_back_pct', 0.2) or 0.2),
            v2_require_2r_potential=bool(getattr(args, 'v2_require_2r_potential', False)),
            v2_no_progress_exit_minutes=int(getattr(args, 'v2_no_progress_exit_minutes', 0) or 0),
            v2_no_progress_exit_macd_only=bool(getattr(args, 'v2_no_progress_exit_macd_only', True)),
            v2_weakness_exit_bars=int(getattr(args, 'v2_weakness_exit_bars', 0) or 0),
            v2_giveback_exit_frac=float(getattr(args, 'v2_giveback_exit_frac', 0.0) or 0.0),
            # BFSv3
            v3_max_pullback_candles=int(getattr(args, 'v3_max_pullback_candles', 3) or 3),
            v3_min_pullback_candles=int(getattr(args, 'v3_min_pullback_candles', 2) or 2),
            v3_max_retrace_pct=float(getattr(args, 'v3_max_retrace_pct', 0.50) or 0.50),
            v3_no_progress_minutes=int(getattr(args, 'v3_no_progress_minutes', 0) or 0),
            v3_use_ticks=bool(getattr(args, 'v3_use_ticks', False)),
            v3_tick_feed=str(getattr(args, 'v3_tick_feed', 'sip')),
            v3_require_vwap_above=bool(getattr(args, 'v3_require_vwap_above', False)),
            v3_weakness_exit_bars=int(getattr(args, 'v3_weakness_exit_bars', 0) or 0),
            v3_require_macd_positive=bool(getattr(args, 'v3_require_macd_positive', False)),
            v3_require_2r_potential=bool(getattr(args, 'v3_require_2r_potential', False)),
            v3_require_ema_trend=bool(getattr(args, 'v3_require_ema_trend', False)),
            # Gap & Go
            gng_min_gap_pct=float(getattr(args, 'gng_min_gap_pct', 4.0) or 4.0),
            gng_entry_mode=str(getattr(args, 'gng_entry_mode', 'both')),
            gng_entry_window_minutes=int(getattr(args, 'gng_entry_window_minutes', 30) or 30),
            gng_require_news=bool(getattr(args, 'gng_require_news', False)),
            gng_max_float_millions=float(getattr(args, 'gng_max_float_millions', 0.0) or 0.0),
        )
        monitor = PatternMonitoringSession(
            alert=action_alert,
            patterns_to_monitor=[getattr(args, 'pattern', 'bull_flag')] if getattr(args, 'pattern', 'bull_flag') in ('bull_flag','alert_flip','bull_flag_simple','bull_flag_simple_v2','bull_flag_simple_v3','gap_and_go') else ['bull_flag'],
            position_sizer=position_sizer,
            config=session_cfg,
            position_tracker=position_tracker,
        )
        # Provide float info from scanner to strategies that use it (e.g., Gap & Go)
        try:
            fm = alert_data.get('float')
            if fm is not None:
                monitor.float_millions = float(fm)
        except Exception:
            pass
        # Provide prev_close to the monitor if we have it (from gappers seed)
        try:
            if monitor_prev_close is not None:
                monitor.prev_close = float(monitor_prev_close)
        except Exception:
            pass
        
        try:
            _pat = getattr(args, 'pattern', 'bull_flag')
            if _pat == 'alert_flip':
                _label = 'Alert Flip'
            elif _pat == 'bull_flag_simple':
                _label = 'Bull Flag Simple'
            elif _pat == 'bull_flag_simple_v2':
                _label = 'Bull Flag Simple V2'
            elif _pat == 'bull_flag_simple_v3':
                _label = 'Bull Flag Simple V3'
            elif _pat == 'gap_and_go':
                _label = 'Gap & Go'
            else:
                _label = 'Bull Flag'
            logging.info("🔍 Started monitoring %s for patterns (strategy: %s)", symbol, _label)
        except Exception:
            logging.info("🔍 Started monitoring %s for patterns", symbol)
        # processed_symbols not needed since we de-dupe to one alert per symbol above
        
        # Load OHLC data for this symbol (from cache if available)
        if symbol in ohlc_data_cache:
            ohlc_data = ohlc_data_cache[symbol]
        else:
            ohlc_data = load_symbol_ohlc_data(symbol, date, timeframe="1min")
            if ohlc_data is not None and not ohlc_data.empty:
                ohlc_data_cache[symbol] = ohlc_data
        if ohlc_data is not None and not ohlc_data.empty:
            # If using Gap & Go, optionally fetch true PMH from Alpaca ticks
            try:
                if getattr(args, 'pattern', 'bull_flag') == 'gap_and_go':
                    from tools.alpaca_ticks import fetch_premarket_high
                    pmh = fetch_premarket_high(symbol, date, feed='sip')
                    if pmh is not None:
                        monitor.external_premarket_high = float(pmh)
                        logging.info("PMH (ticks) for %s: %.4f", symbol, float(pmh))
            except Exception as e:
                logging.info("PMH (ticks) unavailable for %s: %s", symbol, e)
            # Compute pole anchor time (first bar before a 2-red streak) and set known flagpole high/time
            try:
                start_dt = datetime.strptime(f"{date} {start_time}", "%Y-%m-%d %H:%M")
                end_dt = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
                # Reuse helper by local definition (mirrors above)
                def _find_first_pole_anchor_time(df: pd.DataFrame, start_ts: datetime, end_ts: datetime):
                    try:
                        w = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
                    except Exception:
                        w = df
                    red_len = 0
                    for i in range(1, len(w)):
                        o, c = w.iloc[i]['open'], w.iloc[i]['close']
                        if c < o:
                            red_len += 1
                        else:
                            red_len = 0
                        if red_len == 2:
                            # Anchor to the FIRST red bar (the bar where the pole high often occurs), not pre-red
                            anchor_idx = i - 1
                            if anchor_idx >= 0:
                                return w.index[anchor_idx]
                    return None
                pole_time = _find_first_pole_anchor_time(ohlc_data, start_dt, end_dt)
                # Override alert_time only for the full bull_flag strategy
                if pole_time is not None and getattr(args, 'pattern', 'bull_flag') == 'bull_flag':
                    pole_high = float(ohlc_data.loc[pole_time]['high'])
                    action_alert = ActionAlert(
                        symbol=symbol,
                        alert_time=pole_time.to_pydatetime() if hasattr(pole_time, 'to_pydatetime') else pole_time,
                        alert_price=action_alert.alert_price,
                        alert_high=pole_high,
                        volume_spike=action_alert.volume_spike,
                        news_catalyst=action_alert.news_catalyst,
                    )
                    monitor.alert = action_alert
            except Exception:
                pass
            # In per-alert mode, slice data from alert_time forward to reduce work
            try:
                if bool(getattr(args, 'per_alert_sessions', False)):
                    # Include a warmup lookback so indicators (e.g., MACD/EMA) are populated
                    from datetime import timedelta as _td
                    warmup_minutes = 40
                    slice_start = alert_time - _td(minutes=warmup_minutes)
                    # Decide end of slice: if managing past end, run to last available bar; else clamp at end_time
                    if bool(getattr(args, 'manage_past_end', True)):
                        slice_end = ohlc_data.index.max()
                    else:
                        try:
                            slice_end = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
                        except Exception:
                            slice_end = ohlc_data.index.max()
                    # For alert_flip only, slice through the first non-alert minute after alert (preserve behavior)
                    if getattr(args, 'pattern', 'bull_flag') in ('alert_flip',):
                        # Include all bars from alert_time up to the first non-alert minute (cap +30m or slice_end)
                        try:
                            sym_alerts_all = alerts_by_symbol.get(symbol.upper(), [])
                            alert_set = set(datetime.strptime(f"{date} {a.get('time','')}", "%Y-%m-%d %H:%M") for a in sym_alerts_all)
                        except Exception:
                            alert_set = set()
                        t = alert_time + _td(minutes=1)
                        # Search up to the overall end slice
                        limit = slice_end
                        while t in alert_set and t <= limit:
                            t = t + _td(minutes=1)
                        slice_end = min(limit, t)
                    # Clamp slice_start to available data start
                    try:
                        data_start = ohlc_data.index.min()
                        if slice_start < data_start:
                            slice_start = data_start
                    except Exception:
                        pass
                    ohlc_data = ohlc_data.loc[(ohlc_data.index >= slice_start) & (ohlc_data.index <= slice_end)]
            except Exception:
                pass
            # Otherwise optionally clamp to end_time if not managing past end
            if not args.manage_past_end:
                try:
                    end_dt2 = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
                    ohlc_data = ohlc_data.loc[:end_dt2]
                except Exception:
                    pass
            logging.info("📊 Loaded %d bars for %s", len(ohlc_data), symbol)
            
            # Feed each bar to the pattern monitor
            for idx, row in ohlc_data.iterrows():
                monitor.add_price_data(
                    timestamp=idx,
                    open_price=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
            
            # Force-flatten at slice end to close any dangling positions for reporting
            try:
                if getattr(monitor, 'position', None) is not None and len(ohlc_data) > 0:
                    last_idx = ohlc_data.index[-1]
                    last_close = float(ohlc_data['close'].iloc[-1])
                    monitor.force_flatten(last_idx, last_close)
            except Exception:
                pass
            # Collect trade summary for this session
            trade_summary = monitor.get_trade_summary()
            trade_summary["alert_symbol"] = symbol
            trade_summary["alert_time"] = alert_time
            session_summaries.append(trade_summary)
            # Record symbol busy interval to avoid overlapping subsequent alerts
            try:
                if trade_summary.get("position_entered"):
                    et = trade_summary.get("entry_time")
                    xt = trade_summary.get("exit_time")
                    if isinstance(et, str):
                        from datetime import datetime as _dt
                        et = _dt.strptime(et, "%Y-%m-%d %H:%M:%S")
                    if isinstance(xt, str):
                        from datetime import datetime as _dt
                        xt = _dt.strptime(xt, "%Y-%m-%d %H:%M:%S")
                    if et and xt and et <= xt:
                        symbol_busy_intervals.setdefault(symbol, []).append((et, xt))
            except Exception:
                pass
            # Record realized P&L into tracker and honor daily stop
            try:
                session_pnl = float(trade_summary.get("total_pnl", 0.0) or 0.0)
                if hasattr(position_tracker, 'record_realized_pnl'):
                    position_tracker.record_realized_pnl(session_pnl)
                # Lockout update
                if lockout_after_losers is not None and session_pnl < 0:
                    # Keep only negatives within window
                    recent_negatives = [t for t in recent_negatives if (alert_time - t).total_seconds() <= max(1,int(lockout_minutes))*60]
                    recent_negatives.append(alert_time)
                    if len(recent_negatives) >= int(lockout_after_losers):
                        lockout_until = alert_time + timedelta(minutes=int(lockout_minutes))
                        recent_negatives.clear()
                if hasattr(position_tracker, 'should_halt_trading') and position_tracker.should_halt_trading():
                    logging.warning("⛔ Daily stop reached after %s. Halting new entries.", symbol)
                    break
            except Exception:
                pass
            # Log per-alert decision if no entry was taken
            if not trade_summary.get("position_entered"):
                last_stage = trade_summary.get("last_stage", "unknown")
                last_val = trade_summary.get("last_validation", "unknown")
                conf_rej = int(trade_summary.get("confirmations_rejects", 0) or 0)
                brk_att = int(trade_summary.get("breakout_attempts", 0) or 0)
                logging.info(
                    "No trade for %s: stage=%s | validation=%s | conf_rejects=%d | breakout_attempts=%d",
                    symbol, last_stage, last_val, conf_rej, brk_att
                )
        else:
            logging.warning("⚠️  No OHLC data found for %s", symbol)
            
        # Check if we've hit position limits
        active_positions = len([p for p in getattr(position_tracker, 'positions', []) if getattr(p, 'status', 'active') == 'active'])
        if active_positions >= 4:
            logging.info("⚠️  Maximum 4 positions reached - stopping new alerts")
            break
        
    logging.info("\n✅ Backtest completed for %s", date)

    # Aggregate P&L/metrics summary
    entries = [s for s in session_summaries if s.get("position_entered")]
    total_entries = len(entries)
    total_pnl = sum(float(s.get("total_pnl", 0.0)) for s in entries)
    wins = sum(1 for s in entries if float(s.get("total_pnl", 0.0)) > 0)
    losses = sum(1 for s in entries if float(s.get("total_pnl", 0.0)) < 0)
    flat = total_entries - wins - losses
    # Exit reason distribution
    reason_counts: Dict[str, int] = {}
    for s in entries:
        for reason, cnt in (s.get("sell_reasons") or {}).items():
            reason_counts[reason] = reason_counts.get(reason, 0) + cnt

    logging.info("--- Summary (%s) ---", date)
    logging.info("Alerts: %d | Entries: %d | Wins: %d | Losses: %d | Flats: %d",
                 len(action_alerts), total_entries, wins, losses, flat)
    logging.info("Total P&L: $%s", f"{int(total_pnl):,}")
    if total_entries > 0:
        logging.info("Hit Rate: %.1f%% | Avg P&L/trade: $%.0f",
                     100.0 * wins / total_entries, total_pnl / total_entries)
    if reason_counts:
        reasons_str = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
        logging.info("Exit reasons: %s", reasons_str)

    # Persist per-session summaries (JSON)
    try:
        with open(summaries_path, 'w') as jf:
            json.dump(session_summaries, jf, default=str, indent=2)
        logging.info("Saved session summaries: %s", summaries_path)
    except Exception as e:
        logging.warning("Failed to save session summaries: %s", e)

    # Persist entries CSV
    try:
        fieldnames = [
            'symbol','alert_time','entry_time','entry_price','risk_per_share','total_pnl',
            'executions','status','last_stage','last_validation','confirmations_rejects','breakout_attempts',
            'last_pullback_candles','last_retrace_percentage','last_volume_confirmation','last_broke_vwap','last_broke_9ema','last_strength_score'
        ]
        with open(entries_csv_path, 'w', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=fieldnames)
            writer.writeheader()
            for s in entries:
                row = {
                    'symbol': s.get('symbol'),
                    'alert_time': s.get('alert_time'),
                    'entry_time': s.get('entry_time'),
                    'entry_price': s.get('entry_price'),
                    'risk_per_share': s.get('risk_per_share'),
                    'total_pnl': s.get('total_pnl'),
                    'executions': s.get('executions'),
                    'status': s.get('status'),
                    'last_stage': s.get('last_stage'),
                    'last_validation': s.get('last_validation'),
                    'confirmations_rejects': s.get('confirmations_rejects'),
                    'breakout_attempts': s.get('breakout_attempts'),
                    'last_pullback_candles': s.get('last_pullback_candles'),
                    'last_retrace_percentage': s.get('last_retrace_percentage'),
                    'last_volume_confirmation': s.get('last_volume_confirmation'),
                    'last_broke_vwap': s.get('last_broke_vwap'),
                    'last_broke_9ema': s.get('last_broke_9ema'),
                    'last_strength_score': s.get('last_strength_score'),
                }
                writer.writerow(row)
        logging.info("Saved entries CSV: %s", entries_csv_path)
    except Exception as e:
        logging.warning("Failed to save entries CSV: %s", e)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ross Cameron Strategy Backtest')
    parser.add_argument('--date', required=True, help='Date to backtest (YYYY-MM-DD)')
    parser.add_argument('--start-time', default='06:00', help='Start time (HH:MM)')
    parser.add_argument('--end-time', default='11:30', help='End time (HH:MM)')
    parser.add_argument('--account', type=float, default=30000, help='Account balance')
    parser.add_argument('--log-file', help='Path to log file (defaults to results/logs/backtest_<date>.log)')
    parser.add_argument('--log-level', default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    parser.add_argument('--bailout-grace-bars', type=int, default=2, help='1-min bars to allow before breakeven bailout')
    parser.add_argument('--use-5min-first-red-exit', action='store_true', help='Enable first red 5-min exit for runners')
    parser.add_argument('--disable-ema-vwap-weakness-exit', action='store_true', help='Disable 1-min EMA/VWAP weakness exit')
    parser.add_argument('--stop-slippage-cents', type=float, default=0.0, help='Stop fill slippage in dollars (per share)')
    parser.add_argument('--entry-slippage-cents', type=float, default=0.0, help='Entry fill slippage in dollars (per share)')
    parser.add_argument('--disable-add-back', action='store_true', help='Disable add-back entries on subsequent flags')
    parser.add_argument('--max-add-backs', type=int, default=2, help='Maximum number of add-backs per symbol')
    parser.add_argument('--add-back-cooldown-bars', type=int, default=2, help='Cooldown (bars) between add-backs')
    parser.add_argument('--entry-confirm-mode', choices=['prior','current'], default='current', help='Confirm entry on prior bar (fast) or current bar (at close)')
    parser.add_argument('--trigger-mode', choices=['pattern','macd_cross'], default='pattern', help='Entry trigger: bull-flag pattern (default) or MACD bullish cross')
    parser.add_argument('--entry-confirmations', choices=['both','macd_only','ema_only','none'], default='both', help='Which confirmations to require at entry')
    parser.add_argument('--exit-on-macd-cross', action='store_true', help='Enable exit on MACD bearish cross (macd < signal)')
    parser.add_argument('--disable-extension-bar-exit', action='store_true', help='Disable extension-bar sell-into-strength exits')
    parser.add_argument('--disable-early-pullback-trim', action='store_true', help='Disable early pullback trim on first meaningful 1-min red')
    parser.add_argument('--entry-cutoff-minutes', type=int, default=15, help='Minutes after alert to allow new entries (per-alert freshness window)')
    parser.add_argument('--require-green-breakout', action='store_true', help='Require breakout candle to close green before entry (conservative)')
    parser.add_argument('--breakout-vol-mult', type=float, default=1.0, help='Require breakout volume ≥ pullback avg * mult (0 disables)')
    parser.add_argument('--min-pullback-avg-volume', type=float, default=0.0, help='Require pullback average 1-min volume ≥ threshold (0 disables)')
    parser.add_argument('--max-daily-loss', type=float, help='Dollar max loss to halt new entries (kill switch)')
    parser.add_argument('--pattern', choices=['bull_flag','alert_flip','bull_flag_simple','bull_flag_simple_v2','bull_flag_simple_v3','gap_and_go'], default='bull_flag', help='Choose pattern/strategy to run')
    parser.add_argument('--seed-mode', choices=['alerts','gappers'], default='alerts', help='Seed sessions from ACTION alerts (default) or Top Gappers at 9:30')
    parser.add_argument('--gappers-top-n', type=int, default=5, help='Seed-mode gappers: take top-N ranked by gap% (or available score)')
    parser.add_argument('--gappers-min-premarket-volume', type=int, default=0, help='Seed-mode gappers: require premarket volume >= this (0 disables)')
    # Gap & Go (entry-only) options
    parser.add_argument('--gng-min-gap-pct', type=float, default=4.0, help='Gap & Go: minimum gap percent vs prior close')
    parser.add_argument('--gng-entry-mode', choices=['premarket_high','opening_range','premarket_flag','both'], default='both', help='Gap & Go: entry mode')
    parser.add_argument('--gng-entry-window-minutes', type=int, default=30, help='Gap & Go: entry window minutes from 9:30')
    parser.add_argument('--gng-require-news', action='store_true', help='Gap & Go: require news catalyst')
    parser.add_argument('--gng-max-float-millions', type=float, default=20.0, help='Gap & Go: require float <= this value (0 disables)')
    parser.add_argument('--per-alert-sessions', action='store_true', help='Start a new monitoring session for every ACTION alert (no de-dup)')
    parser.add_argument('--spread-cap-bps', type=float, default=0.0, help='Reject entries if (high-low)/close exceeds this bps cap (0 disables)')
    parser.add_argument('--require-macd-positive', action='store_true', help='Require MACD > 0 and histogram > 0 at entry bar')
    parser.add_argument('--lockout-after-losers', type=int, help='Lockout new entries after this many losses within lockout-minutes')
    parser.add_argument('--lockout-minutes', type=int, default=30, help='Lockout window in minutes for consecutive losers')
    # Manage past end-time toggles (default: manage past end)
    parser.add_argument('--manage-past-end', dest='manage_past_end', action='store_true', help='Continue managing open positions after end-time')
    parser.add_argument('--no-manage-past-end', dest='manage_past_end', action='store_false', help='Clamp bars at end-time; do not manage past end')
    parser.set_defaults(manage_past_end=True)
    # BFSv2 toggles
    parser.add_argument('--v2-partial-on-alert-high', dest='v2_partial_on_alert_high', action='store_true', help='On alert_high target, sell 50% and move stop to BE (default ON)')
    parser.add_argument('--no-v2-partial-on-alert-high', dest='v2_partial_on_alert_high', action='store_false', help='Disable partial at alert_high (sell all)')
    parser.set_defaults(v2_partial_on_alert_high=True)
    parser.add_argument('--v2-runner-enabled', dest='v2_runner_enabled', action='store_true', help='Enable BFSv2 runner management after first scale (default ON)')
    parser.add_argument('--no-v2-runner-enabled', dest='v2_runner_enabled', action='store_false', help='Disable BFSv2 runner management')
    parser.set_defaults(v2_runner_enabled=True)
    parser.add_argument('--v2-runner-macd-gate-minutes', type=int, default=10, help='Minutes after entry to apply MACD gate on runner')
    parser.add_argument('--v2-runner-hard-cap-r', type=float, default=3.0, help='Hard cap target for runner in R-multiples (e.g., 3.0R)')
    parser.add_argument('--v2-entry-confirm-ema', action='store_true', help='Require EMA9>EMA20 on BFSv2 entry')
    parser.add_argument('--v2-entry-confirm-macd', action='store_true', help='Require MACD bullish (macd>signal and hist>0) on BFSv2 entry')
    parser.add_argument('--v2-entry-confirmations', choices=['both','macd_only','ema_only','none'], default='none', help='BFSv2: select which confirmations to require (overrides individual flags)')
    parser.add_argument('--v2-enter-on-close-with-gate', action='store_true', help='When volume gate active, enter on bar close instead of intrabar breakout')
    parser.add_argument('--v2-min-stop-dollars', type=float, default=0.10, help='Minimum stop distance (dollars) for BFSv2 sizing/targets')
    # MACD gate + structure knobs
    parser.add_argument('--v2-macd-gate-require-runner', dest='v2_macd_gate_require_runner', action='store_true', help='Require runner (post-partial) for MACD gate at T+X (default)')
    parser.add_argument('--no-v2-macd-gate-require-runner', dest='v2_macd_gate_require_runner', action='store_false', help='Apply MACD gate at T+X even without runner (full position)')
    parser.set_defaults(v2_macd_gate_require_runner=True)
    parser.add_argument('--v2-macd-gate-require-no-progress', action='store_true', help='Require no progress (max-high ≤ entry + thresh_R×R) to trigger MACD gate')
    parser.add_argument('--v2-no-progress-thresh-r', type=float, default=0.0, help='No-progress threshold in R for MACD gate (e.g., 0.1)')
    parser.add_argument('--v2-max-wait-bars', type=int, default=6, help='BFSv2: bars to wait from first red after alert before timing out')
    parser.add_argument('--v2-retrace-cap-pct', type=float, default=0.50, help='BFSv2: pullback retrace cap as fraction of pole height (0-1)')
    parser.add_argument('--v2-require-vwap-above', action='store_true', help='BFSv2: require price >= VWAP at entry')
    parser.add_argument('--v2-entry-confirm-ema5m', action='store_true', help='BFSv2: require EMA9>EMA20 on 5-min timeframe at entry')
    # Discipline/exit enhancements
    parser.add_argument('--v2-require-2r-potential', action='store_true', help='BFSv2: require target potential >= 2R at entry (skip otherwise)')
    parser.add_argument('--v2-no-progress-exit-minutes', type=int, default=0, help='BFSv2: exit at T+N minutes if price never > entry and (optionally) MACD bearish (0 disables)')
    parser.add_argument('--v2-no-progress-exit-macd-only', action='store_true', help='BFSv2: require MACD bearish to trigger no-progress exit')
    parser.add_argument('--v2-weakness-exit-bars', type=int, default=0, help='BFSv2: in first M bars after entry, exit if close < VWAP or EMA9 (0 disables)')
    parser.add_argument('--v2-giveback-exit-frac', type=float, default=0.0, help='BFSv2: exit remainder if giveback exceeds this fraction of max-open R after partial (0 disables)')
    # BFSv3 (tick-level entry options)
    parser.add_argument('--v3-max-pullback-candles', type=int, default=5, help='BFSv3: max consecutive red candles in pullback')
    parser.add_argument('--v3-min-pullback-candles', type=int, default=1, help='BFSv3: min consecutive red candles before entry allowed')
    parser.add_argument('--v3-max-retrace-pct', type=float, default=0.50, help='BFSv3: max retrace as fraction of pole (0-1)')
    parser.add_argument('--v3-no-progress-minutes', type=int, default=0, help='BFSv3: bail if no progress after N minutes (0 disables)')
    parser.add_argument('--v3-use-ticks', action='store_true', help='BFSv3: refine entry with Alpaca trades (env keys required)')
    parser.add_argument('--v3-tick-feed', choices=['sip','iex'], default='sip', help='BFSv3: Alpaca trades feed to use')
    parser.add_argument('--v3-require-vwap-above', action='store_true', help='BFSv3: require close ≥ VWAP at trigger bar')
    parser.add_argument('--v3-weakness-exit-bars', type=int, default=0, help='BFSv3: in first M bars, exit if close < VWAP or EMA9 (0 disables)')
    parser.add_argument('--v3-require-macd-positive', action='store_true', help='BFSv3: require MACD>0 and histogram>0 at trigger bar')
    parser.add_argument('--v3-require-2r-potential', action='store_true', help='BFSv3: require target distance ≥ 2R at entry')
    parser.add_argument('--v3-require-ema-trend', action='store_true', help='BFSv3: require EMA9 > EMA20 at trigger bar')
    parser.add_argument('--symbols', help='Comma-separated symbols to include (e.g., BSLK,DOGZ)')
    parser.add_argument('--exclude-symbols', help='Comma-separated symbols to exclude')
    parser.add_argument('--symbols-file', help='Path to file with one or comma-separated symbols per line')
    parser.add_argument('--exclude-symbols-file', help='Path to file with one or comma-separated symbols per line to exclude')
    # Repro/replay: allow running with a saved alerts file (bypass selection)
    parser.add_argument('--alerts-file', help='Path to saved alerts JSON (from *_alerts.json) to replay alerts exactly')
    
    args = parser.parse_args()

    # Default log path: include timestamp so each run creates a new file
    now_ts = datetime.now().strftime("%H%M%S")
    default_log = str(LOGS_DIR / f"backtest_{args.date}_{now_ts}.log")
    log_path = args.log_file if args.log_file else default_log

    setup_logging(log_path, args.log_level)
    # Persist CLI flags for reproducibility
    try:
        import json as _json
        cfg_out = LOGS_DIR / f"backtest_{args.date}_{now_ts}_config.json"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with open(cfg_out, 'w') as cf:
            _json.dump(vars(args), cf, indent=2)
        logging.info("Saved run config: %s", cfg_out)
    except Exception as e:
        logging.warning("Failed to save run config: %s", e)
    logging.info(
        "Config: trigger=%s | grace_bars=%d | use_5min_first_red_exit=%s | ema_vwap_exit=%s | macd_exit=%s | entry_conf=%s | stop_slip=%.3f | entry_slip=%.3f | add_back=%s/%d | ext_exit=%s | early_trim=%s | max_daily_loss=%s | brk_vol_mult=%.2f | min_pb_avg_vol=%.0f",
        args.trigger_mode,
        args.bailout_grace_bars,
        args.use_5min_first_red_exit,
        not args.disable_ema_vwap_weakness_exit,
        args.exit_on_macd_cross,
        args.entry_confirmations,
        args.stop_slippage_cents,
        args.entry_slippage_cents,
        "on" if not args.disable_add_back else "off",
        args.max_add_backs,
        "on" if not args.disable_extension_bar_exit else "off",
        "on" if not args.disable_early_pullback_trim else "off",
        (f"${args.max_daily_loss:.0f}" if args.max_daily_loss is not None else "<none>"),
        float(args.breakout_vol_mult or 0.0),
        float(args.min_pullback_avg_volume or 0.0),
    )
    if getattr(args, 'pattern', 'bull_flag') == 'bull_flag_simple_v3':
        logging.info(
            "BFSv3: min_reds=%d | max_reds=%d | retrace_cap=%.2f | no_progress=%dm | vwap=%s | macd_pos=%s | ema_trend=%s | weak_exit_bars=%d | vol_mult=%.2f | spread_cap_bps=%.0f | min_stop=$%.2f",
            int(getattr(args, 'v3_min_pullback_candles', 1) or 1),
            int(getattr(args, 'v3_max_pullback_candles', 5) or 5),
            float(getattr(args, 'v3_max_retrace_pct', 0.5) or 0.5),
            int(getattr(args, 'v3_no_progress_minutes', 0) or 0),
            bool(getattr(args, 'v3_require_vwap_above', False)),
            bool(getattr(args, 'v3_require_macd_positive', False)),
            bool(getattr(args, 'v3_require_ema_trend', False)),
            int(getattr(args, 'v3_weakness_exit_bars', 0) or 0),
            float(getattr(args, 'breakout_vol_mult', 0.0) or 0.0),
            float(getattr(args, 'spread_cap_bps', 0.0) or 0.0),
            float(getattr(args, 'v2_min_stop_dollars', 0.0) or 0.0),
        )
    else:
        logging.info(
            "BFSv2: runner=%s | macd_gate=%d | hard_cap_R=%.1f | partial_on_alert_high=%s | enter_on_close_gate=%s | min_stop=$%.2f",
            "on" if args.v2_runner_enabled else "off",
            int(args.v2_runner_macd_gate_minutes or 10),
            float(args.v2_runner_hard_cap_r or 3.0),
            "on" if args.v2_partial_on_alert_high else "off",
            "on" if args.v2_enter_on_close_with_gate else "off",
            float(args.v2_min_stop_dollars or 0.10),
        )
        logging.info(
            "BFSv2 extras: macd_gate_require_runner=%s | require_no_progress=%s | no_progress_R=%.2f | max_wait_bars=%d | retrace_cap=%.2f | vwap_guard=%s | ema5m=%s",
            bool(getattr(args, 'v2_macd_gate_require_runner', True)),
            bool(getattr(args, 'v2_macd_gate_require_no_progress', False)),
            float(getattr(args, 'v2_no_progress_thresh_r', 0.0) or 0.0),
            int(getattr(args, 'v2_max_wait_bars', 6) or 6),
            float(getattr(args, 'v2_retrace_cap_pct', 0.50) or 0.50),
            bool(getattr(args, 'v2_require_vwap_above', False)),
            bool(getattr(args, 'v2_entry_confirm_ema5m', False)),
        )
    def _load_symfile(p):
        if not p:
            return []
        try:
            syms=[]
            with open(p) as f:
                for line in f:
                    line=line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts=[s.strip() for s in line.replace(',', ' ').split() if s.strip()]
                    syms.extend(parts)
            return syms
        except Exception:
            return []

    include_syms = []
    if args.symbols:
        include_syms.extend([s.strip().upper() for s in args.symbols.split(',') if s.strip()])
    include_syms.extend([s.upper() for s in _load_symfile(args.symbols_file)])

    exclude_syms = []
    if args.exclude_symbols:
        exclude_syms.extend([s.strip().upper() for s in args.exclude_symbols.split(',') if s.strip()])
    exclude_syms.extend([s.upper() for s in _load_symfile(args.exclude_symbols_file)])

    run_backtest(
        args.date,
        args.start_time,
        args.end_time,
        args.account,
        bailout_grace_bars=args.bailout_grace_bars,
        use_5min_first_red_exit=args.use_5min_first_red_exit,
        exit_on_ema_vwap_break=(not args.disable_ema_vwap_weakness_exit),
        stop_slippage_cents=args.stop_slippage_cents,
        entry_slippage_cents=args.entry_slippage_cents,
        add_back_enabled=(not args.disable_add_back),
        max_add_backs=args.max_add_backs,
        add_back_cooldown_bars=args.add_back_cooldown_bars,
        max_daily_loss=args.max_daily_loss,
        symbols=include_syms or None,
        exclude_symbols=exclude_syms or None,
        per_alert_sessions=bool(args.per_alert_sessions),
        lockout_after_losers=args.lockout_after_losers,
        lockout_minutes=args.lockout_minutes,
    )
