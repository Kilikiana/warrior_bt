#!/usr/bin/env python3
"""
REAL Pattern Detection Test for 8/13

This test uses ACTUAL OHLC data and the REAL BullFlagDetector to:
1. Load real 1-minute bars for 8/13
2. Feed them to ActionAlertPatternMonitor 
3. Use actual BullFlagDetector to find patterns
4. Enter positions ONLY when real patterns are detected

NO HARDCODED SIMULATION - This is the real deal!
"""

import json
import gzip
import pandas as pd
from datetime import datetime, time
from pathlib import Path
import logging

from tech_analysis.patterns.pattern_monitor import ActionAlertPatternMonitor, ActionAlert
from tech_analysis.patterns.bull_flag_pattern import BullFlagDetector, BullFlagStage, BullFlagValidation
from position_management.position_sizer import PositionSizer
from position_management.position_tracker import PositionTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealPatternTester:
    """Test actual pattern detection with real OHLC data"""
    
    def __init__(self):
        self.pattern_monitor = ActionAlertPatternMonitor()
        self.bull_flag_detector = BullFlagDetector()
        self.position_sizer = PositionSizer()
        self.position_tracker = PositionTracker(
            account_balance=30000,
            max_positions=4,
            daily_risk_percentage=0.05
        )
        
        # Load real OHLC data
        self.ohlc_data = self.load_8_13_ohlc_data()
        
    def load_8_13_ohlc_data(self) -> pd.DataFrame:
        """Load real 8/13 OHLC data"""
        ohlc_file = "/Users/claytonsmacbookpro/Projects/warrior_bt/shared_cache/ohlcv_1min_bars/ohlcv_1min_2025-08-13.json.gz"
        
        logger.info(f"Loading real OHLC data from {ohlc_file}")
        
        with gzip.open(ohlc_file, 'rt') as f:
            data = json.load(f)
        
        # Convert to DataFrame - data is a list of symbols
        rows = []
        for symbol_data in data:
            symbol = symbol_data['symbol']
            for bar in symbol_data.get('minutes', []):
                # Parse time - format like "09:30"
                try:
                    hour, minute = map(int, bar['time'].split(':'))
                    timestamp = datetime(2025, 8, 13, hour, minute)
                    
                    rows.append({
                        'timestamp': timestamp,
                        'symbol': symbol,
                        'open': bar['open'],
                        'high': bar['high'], 
                        'low': bar['low'],
                        'close': bar['close'],
                        'volume': bar['volume']
                    })
                except (ValueError, KeyError) as e:
                    continue
        
        df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(df)} OHLC bars for {df['symbol'].nunique()} symbols")
        return df
    
    def load_action_alerts(self) -> list:
        """Load ACTION alerts from scanner results"""
        scan_file = "/Users/claytonsmacbookpro/Projects/warrior_bt/results/hod_momentum_scans/hod_momentum_scan_2025-08-13.json"
        
        with open(scan_file, 'r') as f:
            scan_data = json.load(f)
        
        # Get STRONG_SQUEEZE_HIGH_RVOL alerts (ACTION alerts)
        action_alerts = []
        for alert in scan_data.get('all_alerts', []):
            if alert.get('strategy') == 'STRONG_SQUEEZE_HIGH_RVOL':
                # Filter to trading hours only
                try:
                    hour, minute = map(int, alert['time'].split(':'))
                    alert_time = time(hour, minute)
                    if time(6, 0) <= alert_time <= time(11, 30):
                        action_alerts.append(alert)
                except (ValueError, KeyError):
                    continue
        
        logger.info(f"Loaded {len(action_alerts)} ACTION alerts in trading window")
        return action_alerts
    
    def test_real_pattern_detection(self, symbol: str, alert_time: datetime, alert_price: float):
        """Test real pattern detection for a specific symbol"""
        logger.info(f"\n{'='*60}")
        logger.info(f"REAL PATTERN TEST: {symbol} at {alert_time.strftime('%H:%M')}")
        logger.info(f"{'='*60}")
        
        # Get OHLC data for this symbol
        symbol_data = self.ohlc_data[self.ohlc_data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            logger.warning(f"No OHLC data found for {symbol}")
            return False
            
        # Sort by timestamp
        symbol_data = symbol_data.sort_values('timestamp')
        logger.info(f"Found {len(symbol_data)} bars for {symbol}")
        
        # Start monitoring session
        alert = ActionAlert(
            symbol=symbol,
            alert_time=alert_time,
            alert_price=alert_price,
            alert_high=alert_price * 1.02,
            volume_spike=10.0,
            news_catalyst="ACTION alert"
        )
        
        session_id = self.pattern_monitor.process_action_alert(
            symbol=symbol,
            alert_time=alert_time,
            alert_price=alert_price,
            alert_high=alert_price * 1.02,
            volume_spike=10.0
        )
        
        logger.info(f"Started monitoring session: {session_id}")
        
        # Feed real OHLC data bar by bar (only bars AFTER alert time)
        pattern_found = False
        entry_made = False
        
        for _, bar in symbol_data.iterrows():
            if bar['timestamp'] <= alert_time:
                continue  # Skip bars before alert
                
            # Feed this bar to the pattern monitor
            self.pattern_monitor.update_price_data(
                symbol=symbol,
                timestamp=bar['timestamp'],
                open_price=bar['open'],
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                volume=bar['volume']
            )
            
            # Check session status
            session = self.pattern_monitor.active_sessions.get(session_id)
            if not session:
                session = next((s for s in self.pattern_monitor.completed_sessions 
                              if f"{s.symbol}_{s.start_time.strftime('%Y%m%d_%H%M%S')}" == session_id), None)
            
            if session:
                # Check if pattern was detected
                if session.pattern_signals:
                    latest_signal = session.pattern_signals[-1]
                    logger.info(f"  {bar['timestamp'].strftime('%H:%M')}: {latest_signal.stage.value} - {latest_signal.validation.value}")
                    
                    if (latest_signal.stage == BullFlagStage.BREAKOUT_CONFIRMED and 
                        latest_signal.validation == BullFlagValidation.VALID):
                        pattern_found = True
                        
                        # Check if position was entered
                        if session.position:
                            logger.info(f"🎯 REAL PATTERN DETECTED & POSITION ENTERED!")
                            logger.info(f"   Entry: ${session.position.entry_price:.2f}")
                            logger.info(f"   Stop: ${session.position.stop_loss:.2f}")
                            logger.info(f"   Shares: {session.position.initial_shares}")
                            entry_made = True
                            break
                
                # Check if monitoring stopped
                if session.status.value in ['pattern_failed', 'monitoring_stopped']:
                    logger.info(f"Monitoring stopped: {session.status.value}")
                    break
        
        if not pattern_found:
            logger.info("❌ No valid bull flag pattern detected")
        elif not entry_made:
            logger.info("⚠️ Pattern detected but no position entered")
            
        return entry_made
    
    def run_real_test(self, max_symbols: int = None):
        """Run real pattern detection test"""
        logger.info("="*80)
        logger.info("REAL PATTERN DETECTION TEST - 8/13 DATA")
        logger.info("="*80)
        logger.info("Using ACTUAL OHLC data and REAL BullFlagDetector")
        logger.info("NO hardcoded simulation - waiting for real patterns!")
        
        alerts = self.load_action_alerts()
        
        if not alerts:
            logger.error("No ACTION alerts found!")
            return
        
        # Test all symbols or first few
        test_alerts = alerts if max_symbols is None else alerts[:max_symbols]
        successful_patterns = 0
        
        for i, alert in enumerate(test_alerts, 1):
            symbol = alert['symbol']
            hour, minute = map(int, alert['time'].split(':'))
            alert_time = datetime(2025, 8, 13, hour, minute)
            alert_price = alert['price']
            
            logger.info(f"\n--- Test {i}/{len(test_alerts)} ---")
            logger.info(f"Symbol: {symbol}, Alert: ${alert_price:.2f} at {alert['time']}")
            
            if self.test_real_pattern_detection(symbol, alert_time, alert_price):
                successful_patterns += 1
        
        logger.info(f"\n{'='*80}")
        logger.info("REAL PATTERN TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Symbols tested: {len(test_alerts)}")
        logger.info(f"Real patterns found: {successful_patterns}")
        logger.info(f"Pattern success rate: {successful_patterns/len(test_alerts)*100:.1f}%")
        
        if successful_patterns == 0:
            logger.info("🔍 This is GOOD - it means we're not taking fake trades!")
            logger.info("🔍 Ross Cameron waits for REAL pullback patterns, not hardcoded simulation")

def main():
    tester = RealPatternTester()
    # Test ALL alerts for the entire day
    tester.run_real_test(max_symbols=None)

if __name__ == "__main__":
    main()