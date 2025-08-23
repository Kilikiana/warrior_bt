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
    action_alerts = [
        alert for alert in all_alerts 
        if alert.get('strategy') == 'STRONG_SQUEEZE_HIGH_RVOL'
        and start_time <= alert.get('time', '') <= end_time
    ]
    
    if not action_alerts:
        logging.warning("No ACTION alerts found in time window")
        return
    
    logging.info("📊 Found %d ACTION alerts in trading window", len(action_alerts))
    
    # Initialize position management
    position_tracker = PositionTracker(account_balance, max_positions=4, daily_risk_percentage=0.05)
    position_sizer = PositionSizer()
    
    # Process each alert using the real pattern monitoring system
    run_ts = datetime.now().strftime("%H%M%S")
    summaries_path = LOGS_DIR / f"backtest_{date}_{run_ts}_sessions.json"
    entries_csv_path = LOGS_DIR / f"backtest_{date}_{run_ts}_entries.csv"
    session_summaries: List[Dict] = []
    for i, alert_data in enumerate(action_alerts, 1):
        symbol = alert_data['symbol']
        alert_time = datetime.strptime(f"{date} {alert_data['time']}", "%Y-%m-%d %H:%M")
        alert_price = alert_data['price']
        
        logging.info("\n--- Processing Alert %d/%d ---", i, len(action_alerts))
        logging.info("Symbol: %s, Time: %s, Price: $%s", symbol, alert_data['time'], alert_price)
        logging.info("Strategy: %s, Description: %s", alert_data.get('strategy'), alert_data.get('description'))
        
        # Calculate risk amount (2% of account)
        risk_amount = account_balance * 0.02
        
        # Check position limits
        try:
            if not position_tracker.can_open_position(risk_amount):
                logging.info("⚠️  Position limits exceeded, skipping %s", symbol)
                continue
        except Exception as e:
            logging.warning("⚠️  Position check failed for %s: %s", symbol, e)
            continue
        
        # Create ActionAlert object
        action_alert = ActionAlert(
            symbol=symbol,
            alert_time=alert_time,
            alert_price=alert_price,
            alert_high=alert_price,  # Will need actual high from OHLC data
            volume_spike=alert_data.get('rvol_5min', 0) / 100,  # Convert % to ratio
            news_catalyst=alert_data.get('description')
        )
        
        # Start pattern monitoring using our improved system
        monitor = PatternMonitoringSession(
            alert=action_alert,
            patterns_to_monitor=['bull_flag'],
            account_balance=account_balance,
            position_sizer=position_sizer,
            sizing_method="ross_dynamic",
            bailout_grace_bars=bailout_grace_bars,
            use_5min_first_red_exit=use_5min_first_red_exit,
            exit_on_ema_vwap_break=exit_on_ema_vwap_break,
            stop_slippage_cents=stop_slippage_cents,
            entry_slippage_cents=entry_slippage_cents,
            add_back_enabled=add_back_enabled,
            max_add_backs=max_add_backs,
            add_back_cooldown_bars=add_back_cooldown_bars,
        )
        
        logging.info("🔍 Started monitoring %s for bull flag patterns", symbol)
        
        # Load OHLC data for this symbol and feed it to the pattern monitor
        ohlc_data = load_symbol_ohlc_data(symbol, date, timeframe="1min")
        if ohlc_data is not None and not ohlc_data.empty:
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
            
            # Collect trade summary for this session
            trade_summary = monitor.get_trade_summary()
            trade_summary["alert_symbol"] = symbol
            trade_summary["alert_time"] = alert_time
            session_summaries.append(trade_summary)
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
    
    args = parser.parse_args()

    # Default log path: include timestamp so each run creates a new file
    now_ts = datetime.now().strftime("%H%M%S")
    default_log = str(LOGS_DIR / f"backtest_{args.date}_{now_ts}.log")
    log_path = args.log_file if args.log_file else default_log

    setup_logging(log_path, args.log_level)
    logging.info(
        "Config: grace_bars=%d | use_5min_first_red_exit=%s | ema_vwap_exit=%s | stop_slip=%.3f | entry_slip=%.3f | add_back=%s/%d",
        args.bailout_grace_bars,
        args.use_5min_first_red_exit,
        not args.disable_ema_vwap_weakness_exit,
        args.stop_slippage_cents,
        args.entry_slippage_cents,
        "on" if not args.disable_add_back else "off",
        args.max_add_backs,
    )
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
    )
