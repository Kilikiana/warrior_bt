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
    
    # Load ACTION alerts for the date
    alerts_file = get_scan_file(date)
    if not alerts_file.exists():
        logging.warning("No alerts found for %s", date)
        return
    
    with open(alerts_file) as f:
        scan_data = json.load(f)
    
    # Get all alerts and filter for ACTION alerts (STRONG_SQUEEZE_HIGH_RVOL) in time window
    all_alerts = scan_data.get('all_alerts', [])
    # Optional symbol filtering
    symbols_set = set(s.upper() for s in symbols) if symbols else None
    exclude_set = set(s.upper() for s in exclude_symbols) if exclude_symbols else None
    # Group alerts by symbol (preserve those in the time window)
    alerts_by_symbol: Dict[str, List[dict]] = {}
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
                    # Pre-filter: ignore ACTION alerts that occur on a red candle
                    def _is_red_at(ts: datetime) -> bool:
                        try:
                            if ts in ohlc_df.index:
                                r = ohlc_df.loc[ts]
                            else:
                                pos = ohlc_df.index.get_indexer([ts], method='pad')
                                r = ohlc_df.iloc[int(pos[0])] if pos[0] != -1 else None
                            if r is None:
                                return False
                            return float(r['close']) < float(r['open'])
                        except Exception:
                            return False
                    sym_alerts = [a for a in sym_alerts if not _is_red_at(_ts(a.get('time','')))]
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
    summaries_path = LOGS_DIR / f"backtest_{date}_{run_ts}_sessions.json"
    entries_csv_path = LOGS_DIR / f"backtest_{date}_{run_ts}_entries.csv"
    session_summaries: List[Dict] = []
    # Track symbol busy intervals to prevent overlapping trades across sequential per-alert sessions
    symbol_busy_intervals: Dict[str, List[Tuple[datetime, datetime]]] = {}
    lockout_until: Optional[datetime] = None
    recent_negatives: List[datetime] = []
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
        )
        monitor = PatternMonitoringSession(
            alert=action_alert,
            patterns_to_monitor=[getattr(args, 'pattern', 'bull_flag')] if getattr(args, 'pattern', 'bull_flag') in ('bull_flag','alert_flip','bull_flag_simple','bull_flag_simple_v2') else ['bull_flag'],
            position_sizer=position_sizer,
            config=session_cfg,
            position_tracker=position_tracker,
        )
        
        try:
            _pat = getattr(args, 'pattern', 'bull_flag')
            if _pat == 'alert_flip':
                _label = 'Alert Flip'
            elif _pat == 'bull_flag_simple':
                _label = 'Bull Flag Simple'
            elif _pat == 'bull_flag_simple_v2':
                _label = 'Bull Flag Simple V2'
            else:
                _label = 'Bull Flag'
            logging.info("🔍 Started monitoring %s for patterns (strategy: %s)", symbol, _label)
        except Exception:
            logging.info("🔍 Started monitoring %s for patterns", symbol)
        # processed_symbols not needed since we de-dupe to one alert per symbol above
        
        # Load OHLC data for this symbol and feed it to the pattern monitor (optionally clamp at end_time)
        ohlc_data = load_symbol_ohlc_data(symbol, date, timeframe="1min")
        if ohlc_data is not None and not ohlc_data.empty:
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
                    try:
                        slice_end = datetime.strptime(f"{date} {end_time}", "%Y-%m-%d %H:%M")
                    except Exception:
                        slice_end = ohlc_data.index.max()
                    # For alert_flip only, slice through the first non-alert minute after alert
                    if getattr(args, 'pattern', 'bull_flag') in ('alert_flip',):
                        from datetime import timedelta as _td
                        # Include all bars from alert_time up to the first non-alert minute (cap +30m).
                        # This allows exits on red alert bars if they occur, or on the first non-alert bar.
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
                    # For bull_flag_simple, honor MACD exits beyond next alert: keep slice to end_time
                    elif getattr(args, 'pattern', 'bull_flag') == 'bull_flag_simple':
                        # slice_end already set to end_time above; no change here to allow MACD-driven exits
                        pass
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
    parser.add_argument('--breakout-vol-mult', type=float, default=0.0, help='Require breakout volume ≥ pullback avg * mult (0 disables)')
    parser.add_argument('--min-pullback-avg-volume', type=float, default=0.0, help='Require pullback average 1-min volume ≥ threshold (0 disables)')
    parser.add_argument('--max-daily-loss', type=float, help='Dollar max loss to halt new entries (kill switch)')
    parser.add_argument('--pattern', choices=['bull_flag','alert_flip','bull_flag_simple','bull_flag_simple_v2'], default='bull_flag', help='Choose pattern/strategy to run')
    parser.add_argument('--per-alert-sessions', action='store_true', help='Start a new monitoring session for every ACTION alert (no de-dup)')
    parser.add_argument('--spread-cap-bps', type=float, default=0.0, help='Reject entries if (high-low)/close exceeds this bps cap (0 disables)')
    parser.add_argument('--require-macd-positive', action='store_true', help='Require MACD > 0 and histogram > 0 at entry bar')
    parser.add_argument('--lockout-after-losers', type=int, help='Lockout new entries after this many losses within lockout-minutes')
    parser.add_argument('--lockout-minutes', type=int, default=30, help='Lockout window in minutes for consecutive losers')
    # Manage past end-time toggles (default: manage past end)
    parser.add_argument('--manage-past-end', dest='manage_past_end', action='store_true', help='Continue managing open positions after end-time')
    parser.add_argument('--no-manage-past-end', dest='manage_past_end', action='store_false', help='Clamp bars at end-time; do not manage past end')
    parser.set_defaults(manage_past_end=True)
    parser.add_argument('--symbols', help='Comma-separated symbols to include (e.g., BSLK,DOGZ)')
    parser.add_argument('--exclude-symbols', help='Comma-separated symbols to exclude')
    parser.add_argument('--symbols-file', help='Path to file with one or comma-separated symbols per line')
    parser.add_argument('--exclude-symbols-file', help='Path to file with one or comma-separated symbols per line to exclude')
    
    args = parser.parse_args()

    # Default log path: include timestamp so each run creates a new file
    now_ts = datetime.now().strftime("%H%M%S")
    default_log = str(LOGS_DIR / f"backtest_{args.date}_{now_ts}.log")
    log_path = args.log_file if args.log_file else default_log

    setup_logging(log_path, args.log_level)
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
