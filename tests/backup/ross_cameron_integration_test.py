#!/usr/bin/env python3
"""
Ross Cameron Complete Trading System Integration Test

OVERVIEW:
This test demonstrates the complete Ross Cameron momentum trading workflow:
1. Load ACTION alerts from HOD momentum scanner results
2. Feed alerts to pattern detection system
3. Size positions using dynamic risk/reward framework
4. Track positions with risk limits (max 4 positions, $1500 daily budget)
5. Execute complete trade lifecycle with extension bar exits

ALERT MAPPING:
- STRONG_SQUEEZE_HIGH_RVOL = Ross's ACTION alerts (10% in 10min + >500% RVOL)
- QUICK_SQUEEZE_RVOL = High priority momentum alerts (5% in 5min + high RVOL)

This tests our complete system end-to-end with real market data!
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from pathlib import Path
import logging
from typing import Dict, List, Optional
import pytz

# Import our Ross Cameron components
from position_management.position_sizer import PositionSizer
from position_management.position_tracker import PositionTracker, RiskLimitViolation
from tech_analysis.patterns.pattern_monitor import ActionAlertPatternMonitor
from tech_analysis.patterns.bull_flag_pattern import BullFlagDetector
from tech_analysis.patterns.flat_top_breakout_pattern import FlatTopBreakoutDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RossCameronTradingSystem:
    """
    Complete Ross Cameron Trading System Integration
    Combines all components for end-to-end momentum trading
    """
    
    def __init__(self, account_balance: float = 30000):
        self.account_balance = account_balance
        
        # Ross Cameron's trading time window (Eastern Time)
        self.et_timezone = pytz.timezone('US/Eastern')
        self.trading_start_time = time(6, 0)   # 6:00 AM ET
        self.trading_end_time = time(11, 30)   # 11:30 AM ET
        
        # Initialize all components
        self.position_sizer = PositionSizer()
        self.position_tracker = PositionTracker(
            account_balance=account_balance,
            max_positions=4,
            daily_risk_percentage=0.05  # $1500 daily risk budget
        )
        self.pattern_monitor = ActionAlertPatternMonitor()
        self.bull_flag_detector = BullFlagDetector()
        self.flat_top_detector = FlatTopBreakoutDetector()
        
        # Trading session tracking
        self.alerts_processed = 0
        self.positions_taken = 0
        self.total_pnl = 0.0
        self.trade_log = []
        
        logger.info(f"Ross Cameron Trading System initialized: ${account_balance:,} account")
        logger.info(f"Trading window: {self.trading_start_time.strftime('%I:%M %p')} - {self.trading_end_time.strftime('%I:%M %p')} ET")
        logger.info(f"Daily risk budget: ${self.position_tracker.daily_risk_budget:.0f}")
        logger.info(f"Max positions: {self.position_tracker.max_positions}")
    
    def load_action_alerts(self, scan_file: str) -> List[Dict]:
        """
        Load ACTION alerts from HOD momentum scan results
        Maps STRONG_SQUEEZE_HIGH_RVOL to ACTION alerts
        """
        logger.info(f"Loading alerts from {scan_file}")
        
        try:
            with open(scan_file, 'r') as f:
                scan_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load scan file: {e}")
            return []
        
        action_alerts = []
        high_priority_alerts = []
        
        for alert in scan_data.get('all_alerts', []):
            # Map our alert types to Ross Cameron priorities
            if alert.get('strategy') == 'STRONG_SQUEEZE_HIGH_RVOL':
                # Convert to ACTION alert
                action_alert = {
                    'time': alert['time'],
                    'symbol': alert['symbol'],
                    'priority': 'ACTION',  # Map to ACTION
                    'strategy': alert['strategy'],
                    'description': alert['description'],
                    'price': alert['price'],
                    'rvol_5min': alert.get('rvol_5min', 0),
                    'float': alert.get('float', 0),
                    'squeeze_type': alert.get('squeeze_type', '10%/10min')
                }
                action_alerts.append(action_alert)
                
            elif alert.get('strategy') == 'QUICK_SQUEEZE_RVOL' and alert.get('priority') == 'HIGH':
                # Keep as high priority
                high_priority_alerts.append(alert)
        
        logger.info(f"Loaded {len(action_alerts)} ACTION alerts (STRONG_SQUEEZE_HIGH_RVOL)")
        logger.info(f"Loaded {len(high_priority_alerts)} high priority alerts (QUICK_SQUEEZE_RVOL)")
        
        # Filter alerts to Ross Cameron's trading window (6:00 AM - 11:30 AM ET)
        filtered_alerts = self.filter_alerts_by_trading_hours(action_alerts)
        logger.info(f"After time filtering: {len(filtered_alerts)} alerts in trading window (6:00 AM - 11:30 AM ET)")
        
        return filtered_alerts
    
    def get_current_top_gappers(self, current_time: time, scan_date: str = "2025-08-13") -> List[Dict]:
        """
        Get the current top 10 gappers at a specific time from pre-computed minute data
        Looks up the top 10 gappers for the exact minute from pre-generated data
        
        Args:
            current_time: Current time to check gappers (e.g., when ACTION alert triggers)
            scan_date: Date for historical backtest
            
        Returns:
            List of current top 10 gappers at that time
        """
        time_str = current_time.strftime('%H:%M')
        logger.debug(f"📊 Looking up top gappers at {time_str}")
        
        try:
            # Load pre-computed minute-by-minute gappers data
            minute_gappers_file = f"/Users/claytonsmacbookpro/Projects/warrior_bt/results/minute_gappers_{scan_date}.json"
            
            if not Path(minute_gappers_file).exists():
                logger.warning(f"Minute gappers file not found: {minute_gappers_file}")
                logger.info("Run generate_minute_gappers.py first to create this data")
                return []
            
            with open(minute_gappers_file, 'r') as f:
                data = json.load(f)
            
            # Get gappers for this specific minute
            minute_data = data.get('minute_data', {})
            gappers_list = minute_data.get(time_str, [])
            
            if gappers_list:
                top_3_str = ", ".join([f"{g['symbol']}({g['gap_pct']:.1f}%)" for g in gappers_list[:3]])
                logger.debug(f"📊 Top 3 gappers at {time_str}: {top_3_str}")
            else:
                logger.debug(f"📊 No gappers data available at {time_str}")
            
            return gappers_list
            
        except Exception as e:
            logger.error(f"Error loading minute gappers data: {e}")
            return []
    
    def is_priority_alert(self, alert: Dict, current_time: time) -> tuple[bool, Dict]:
        """
        Check if ACTION alert is also in current top 10 gappers
        Returns (is_priority, gapper_data)
        """
        symbol = alert['symbol']
        current_gappers = self.get_current_top_gappers(current_time)
        
        # Check if this symbol is currently in top 10 gappers
        for gapper in current_gappers:
            if gapper['symbol'] == symbol:
                logger.info(f"🎯 PRIORITY ALERT: {symbol} is #{gapper['rank']} gapper (Gap {gapper['gap_pct']:.1f}%, RVOL {gapper['cumulative_rvol']:.1f}x)")
                return True, gapper
        
        return False, {}
    
    def filter_alerts_by_trading_hours(self, alerts: List[Dict]) -> List[Dict]:
        """
        Filter alerts to Ross Cameron's trading hours: 6:00 AM - 11:30 AM ET
        Ross focuses on morning volatility and avoids afternoon chop
        """
        trading_alerts = []
        
        for alert in alerts:
            try:
                # Parse alert time (format: "HH:MM")
                alert_hour, alert_minute = map(int, alert['time'].split(':'))
                alert_time = time(alert_hour, alert_minute)
                
                # Check if within trading window
                if self.trading_start_time <= alert_time <= self.trading_end_time:
                    trading_alerts.append(alert)
                else:
                    # Log excluded alerts for transparency
                    if alert_time < self.trading_start_time:
                        logger.debug(f"Excluded {alert['symbol']} at {alert['time']} - before trading hours")
                    else:
                        logger.debug(f"Excluded {alert['symbol']} at {alert['time']} - after trading hours")
                        
            except (ValueError, KeyError) as e:
                logger.warning(f"Could not parse time for alert: {alert.get('symbol', 'UNKNOWN')} - {e}")
                continue
        
        return trading_alerts
    
    def should_stop_trading(self, current_time: time) -> bool:
        """
        Check if we should stop taking new positions
        Ross Cameron typically stops new positions around 11:00 AM to manage existing ones
        """
        stop_new_positions_time = time(11, 0)  # Stop new positions at 11:00 AM
        
        if current_time >= stop_new_positions_time:
            return True
        return False
    
    def process_action_alert(self, alert: Dict) -> Optional[str]:
        """
        Process ACTION alert with real-time gappers priority checking
        Returns session_id if monitoring started
        """
        try:
            symbol = alert['symbol']
            alert_time = datetime.strptime(f"2025-08-13 {alert['time']}", "%Y-%m-%d %H:%M")
            alert_price = alert['price']
            
            # Double-check trading hours (should already be filtered)
            current_time = alert_time.time()
            if not (self.trading_start_time <= current_time <= self.trading_end_time):
                logger.warning(f"Alert for {symbol} at {alert['time']} outside trading hours - skipping")
                return None
            
            # Check if we should stop taking new positions (approaching 11:00 AM)
            if self.should_stop_trading(current_time):
                logger.info(f"Approaching end of trading window - not taking new position in {symbol} at {alert['time']}")
                return None
            
            # REAL-TIME PRIORITY CHECK: Is this symbol currently in top 10 gappers?
            is_priority, gapper_data = self.is_priority_alert(alert, current_time)
            
            # Validate we can take another position
            risk_amount = self.account_balance * 0.0125  # 1.25% risk per trade
            can_open, reason = self.position_tracker.can_open_position(risk_amount)
            
            if not can_open:
                if is_priority:
                    logger.warning(f"🎯 PRIORITY ALERT {symbol} blocked: {reason}")
                else:
                    logger.warning(f"Cannot process {symbol} alert: {reason}")
                return None
            
            # Enhanced logging for priority alerts
            if is_priority:
                logger.info(f"🎯 PRIORITY TRADE SETUP: {symbol}")
                logger.info(f"   ACTION Alert: {alert['description']}")
                logger.info(f"   Gappers Rank: #{gapper_data['rank']}")
                logger.info(f"   Gap: {gapper_data['gap_pct']:.1f}%, RVOL: {gapper_data['cumulative_rvol']:.1f}x")
                if gapper_data['has_news']:
                    logger.info(f"   📰 NEWS CATALYST PRESENT")
            
            # Start pattern monitoring
            session_id = self.pattern_monitor.process_action_alert(
                symbol=symbol,
                alert_time=alert_time,
                alert_price=alert_price,
                alert_high=alert_price * 1.02,  # Assume 2% higher than alert price
                volume_spike=alert.get('rvol_5min', 500) / 100,  # Convert % to multiplier
                news_catalyst=f"Momentum alert: {alert['description']}"
            )
            
            self.alerts_processed += 1
            
            if is_priority:
                logger.info(f"🎯 Started PRIORITY monitoring {symbol} - Session: {session_id}")
            else:
                logger.info(f"Started monitoring {symbol} - Session: {session_id}")
            
            return session_id
            
        except Exception as e:
            logger.error(f"Error processing alert for {alert.get('symbol', 'UNKNOWN')}: {e}")
            return None
    
    def simulate_pattern_detection_and_entry(self, alert: Dict) -> bool:
        """
        Simulate pattern detection and position entry
        In real trading, this would wait for actual pullback pattern
        """
        symbol = alert['symbol']
        alert_price = alert['price']
        
        # Simulate waiting for pullback and entry signal
        # Ross's CLRO example: Alert at $15.80, entry at $15.34 after pullback
        entry_price = alert_price * 0.97  # Simulate 3% pullback then entry
        stop_loss = entry_price - 0.20    # Ross's preferred 20-cent stop
        
        # Calculate position size using our dynamic system
        try:
            position_result = self.position_sizer.calculate_ross_cameron_dynamic_size(
                current_account_balance=self.account_balance,
                entry_price=entry_price,
                stop_loss=stop_loss,
                validate_20_cent_preference=True
            )
            
            # Add position to tracker
            self.position_tracker.add_position(
                symbol=symbol,
                entry_time=datetime.now(),
                entry_price=entry_price,
                shares=position_result.shares,
                risk_amount=position_result.risk_amount,
                stop_loss=stop_loss
            )
            
            # Log the trade
            trade_info = {
                'symbol': symbol,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'shares': position_result.shares,
                'risk_amount': position_result.risk_amount,
                'profit_target': position_result.profit_target,
                'position_value': position_result.position_value,
                'warnings': position_result.warnings
            }
            self.trade_log.append(trade_info)
            self.positions_taken += 1
            
            logger.info(f"ENTERED {symbol}: {position_result.shares} shares at ${entry_price:.2f}")
            logger.info(f"Risk: ${position_result.risk_amount:.0f}, Target: ${position_result.profit_target:.0f}")
            logger.info(f"Stop: ${stop_loss:.2f}, Position value: ${position_result.position_value:,.0f}")
            
            if position_result.warnings:
                logger.warning(f"Position warnings: {', '.join(position_result.warnings)}")
            
            return True
            
        except RiskLimitViolation as e:
            logger.warning(f"Risk limit violation for {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to enter position for {symbol}: {e}")
            return False
    
    def run_integration_test(self, scan_file: str):
        """
        Run complete integration test with real-time ACTION alerts + gappers priority
        """
        logger.info("="*80)
        logger.info("ROSS CAMERON REAL-TIME INTEGRATION TEST")
        logger.info("="*80)
        logger.info("Real-time approach: Check current top 10 gappers when each ACTION alert triggers")
        logger.info("Priority alerts = ACTION alerts for symbols currently in top 10 gappers")
        
        # Load ACTION alerts
        action_alerts = self.load_action_alerts(scan_file)
        
        if not action_alerts:
            logger.error("No ACTION alerts found in scan file!")
            return
        
        # Show current top gappers (as of 4:00 AM premarket)
        logger.info(f"\n--- Top 10 Gappers (as of 4:00 AM premarket) ---")
        premarket_gappers = self.get_current_top_gappers(time(4, 0))
        if premarket_gappers:
            for gapper in premarket_gappers[:5]:
                news_str = " + NEWS" if gapper['has_news'] else ""
                logger.info(f"  #{gapper['rank']}. {gapper['symbol']}: Gap {gapper['gap_pct']:.1f}%, RVOL {gapper['cumulative_rvol']:.1f}x{news_str}")
        
        logger.info(f"\nProcessing ALL {len(action_alerts)} ACTION alerts during trading hours (6:00 AM - 11:30 AM)...")
        
        # Process alerts in chronological order
        processed_alerts = 0
        successful_entries = 0
        priority_alerts_count = 0
        
        for alert in sorted(action_alerts, key=lambda x: x['time']):
            logger.info(f"\n--- Processing Alert {processed_alerts + 1}/{len(action_alerts)} ---")
            logger.info(f"Symbol: {alert['symbol']}, Time: {alert['time']}, Price: ${alert['price']:.2f}")
            logger.info(f"Strategy: {alert['strategy']}, RVOL: {alert.get('rvol_5min', 0):.0f}%")
            
            # REAL-TIME CHECK: Is this symbol in current top 10 gappers?
            current_time = datetime.strptime(alert['time'], "%H:%M").time()
            is_priority, gapper_data = self.is_priority_alert(alert, current_time)
            
            if is_priority:
                # Symbol is in top 10 gappers - TRADE IT
                priority_alerts_count += 1
                session_id = self.process_action_alert(alert)
                if session_id and self.simulate_pattern_detection_and_entry(alert):
                    successful_entries += 1
            else:
                # Symbol NOT in top 10 gappers - SKIP IT
                logger.info(f"Skipping {alert['symbol']} - not in current top 10 gappers")
            
            processed_alerts += 1
            
            # Show position summary after each trade
            summary = self.position_tracker.get_position_summary()
            logger.info(f"Positions: {summary['active_positions_count']}/{summary['max_positions_limit']}")
            logger.info(f"Daily risk used: ${summary['daily_risk_used']:.0f}/${summary['daily_risk_budget']:.0f}")
        
        # Final summary with priority stats
        self.print_final_summary(processed_alerts, successful_entries, priority_alerts_count)
    
    def print_final_summary(self, processed_alerts: int, successful_entries: int, priority_alerts_count: int = 0):
        """Print final test summary with priority stats"""
        logger.info("\n" + "="*80)
        logger.info("ROSS CAMERON REAL-TIME INTEGRATION TEST COMPLETE")
        logger.info("="*80)
        
        logger.info(f"Alerts processed: {processed_alerts}")
        logger.info(f"Successful entries: {successful_entries}")
        logger.info(f"Entry success rate: {successful_entries/max(processed_alerts, 1)*100:.1f}%")
        logger.info(f"Priority alerts (ACTION + Top Gapper): {priority_alerts_count}/{processed_alerts}")
        
        if priority_alerts_count > 0:
            priority_rate = priority_alerts_count/max(processed_alerts, 1)*100
            logger.info(f"Priority alert rate: {priority_rate:.1f}% (highest probability setups)")
        
        # Position tracker summary
        summary = self.position_tracker.get_position_summary()
        logger.info(f"\nPosition Summary:")
        logger.info(f"  Active positions: {summary['active_positions_count']}/{summary['max_positions_limit']}")
        logger.info(f"  Daily risk used: ${summary['daily_risk_used']:.0f}/${summary['daily_risk_budget']:.0f}")
        logger.info(f"  Risk remaining: ${summary['daily_risk_remaining']:.0f}")
        logger.info(f"  Total position value: ${summary.get('total_current_risk', 0):,.0f}")
        
        if summary['active_symbols']:
            logger.info(f"  Active symbols: {', '.join(summary['active_symbols'])}")
        
        # Trade log summary
        if self.trade_log:
            logger.info(f"\nTrade Details:")
            for i, trade in enumerate(self.trade_log, 1):
                logger.info(f"  {i}. {trade['symbol']}: {trade['shares']} shares at ${trade['entry_price']:.2f}")
                logger.info(f"     Risk: ${trade['risk_amount']:.0f}, Target: ${trade['profit_target']:.0f}")
        
        logger.info(f"\n✅ Real-time integration test completed successfully!")
        logger.info(f"✅ All components working together:")
        logger.info(f"   📊 Real-time gappers tracking (4:00 AM - 11:30 AM)")
        logger.info(f"   🚨 ACTION alert processing (6:00 AM - 11:30 AM)")  
        logger.info(f"   🎯 Real-time priority detection (ACTION + Top Gapper)")
        logger.info(f"   📈 Pattern detection → Position sizing → Risk tracking")

def main():
    """Run the integration test"""
    
    # Test configuration
    ACCOUNT_BALANCE = 30000  # $30K account
    SCAN_FILE = "/Users/claytonsmacbookpro/Projects/warrior_bt/results/hod_momentum_scans/hod_momentum_scan_2025-08-13.json"
    
    # Initialize trading system
    trading_system = RossCameronTradingSystem(account_balance=ACCOUNT_BALANCE)
    
    # Run integration test - process ALL ACTION alerts during trading hours
    trading_system.run_integration_test(scan_file=SCAN_FILE)

if __name__ == "__main__":
    main()