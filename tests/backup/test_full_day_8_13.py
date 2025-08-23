#!/usr/bin/env python3
"""
FULL DAY 8/13 PATTERN DETECTION TEST

Key insight: The ACTION alert surge IS the flagpole!
- LGHL: $2.24 → $2.47 (10.27%) = FLAGPOLE
- Now look for pullback and breakout in subsequent bars

This tests the ENTIRE day with all 158 ACTION alerts.
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

class FullDayPatternTester:
    """Test pattern detection for ENTIRE 8/13 day"""
    
    def __init__(self):
        self.pattern_monitor = ActionAlertPatternMonitor()
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
    
    def find_pullback_and_breakout(self, symbol: str, alert_time: datetime, alert_price: float) -> dict:
        """
        Find pullback and breakout after ACTION alert
        The ACTION alert surge IS the flagpole - now look for pullback and entry
        """
        
        # Get OHLC data for this symbol
        symbol_data = self.ohlc_data[self.ohlc_data['symbol'] == symbol].copy()
        
        if symbol_data.empty:
            return {'pattern': 'no_data', 'entry': None}
            
        # Sort by timestamp and get bars AFTER alert
        symbol_data = symbol_data.sort_values('timestamp')
        post_alert_data = symbol_data[symbol_data['timestamp'] > alert_time].copy()
        
        if len(post_alert_data) < 5:
            return {'pattern': 'insufficient_data', 'entry': None}
        
        # Look for pullback from alert price
        pullback_found = False
        pullback_low = alert_price
        pullback_low_time = alert_time
        red_candle_count = 0
        
        # Phase 1: Look for pullback (1-4 red candles max)
        for i, (_, bar) in enumerate(post_alert_data.iterrows()):
            if i > 10:  # Don't look too far
                break
                
            # Track pullback low
            if bar['low'] < pullback_low:
                pullback_low = bar['low']
                pullback_low_time = bar['timestamp']
            
            # Check if this is a red candle
            if bar['close'] < bar['open']:
                red_candle_count += 1
                if red_candle_count <= 4:  # Ross's max pullback candles
                    pullback_found = True
            else:
                # Green candle - potential breakout
                if pullback_found and red_candle_count <= 4:
                    # Check if this green candle breaks above pullback high
                    pullback_high = post_alert_data.iloc[:i]['high'].max() if i > 0 else alert_price
                    
                    if bar['close'] > pullback_high:
                        # Calculate pullback depth
                        pullback_depth = (alert_price - pullback_low) / alert_price * 100
                        
                        if pullback_depth <= 50:  # Ross's 50% rule
                            return {
                                'pattern': 'bull_flag_detected',
                                'entry': bar['close'],
                                'entry_time': bar['timestamp'],
                                'stop_loss': pullback_low,
                                'pullback_depth': pullback_depth,
                                'red_candles': red_candle_count,
                                'flagpole_start': 'ACTION_ALERT',
                                'flagpole_high': alert_price
                            }
                        else:
                            return {'pattern': 'pullback_too_deep', 'entry': None, 'pullback_depth': pullback_depth}
                
                # Reset if too many green candles without setup
                if red_candle_count == 0:
                    continue
                    
        # Check if pullback was too deep or too many red candles
        if red_candle_count > 4:
            return {'pattern': 'too_many_red_candles', 'entry': None}
        
        pullback_depth = (alert_price - pullback_low) / alert_price * 100
        if pullback_depth > 50:
            return {'pattern': 'pullback_too_deep', 'entry': None, 'pullback_depth': pullback_depth}
        
        return {'pattern': 'no_breakout_yet', 'entry': None}
    
    def test_symbol(self, alert: dict) -> dict:
        """Test pattern detection for one symbol"""
        symbol = alert['symbol']
        hour, minute = map(int, alert['time'].split(':'))
        alert_time = datetime(2025, 8, 13, hour, minute)
        alert_price = alert['price']
        
        result = self.find_pullback_and_breakout(symbol, alert_time, alert_price)
        
        return {
            'symbol': symbol,
            'alert_time': alert['time'],
            'alert_price': alert_price,
            **result
        }
    
    def run_full_day_test(self):
        """Run pattern detection for ALL ACTION alerts on 8/13"""
        logger.info("="*80)
        logger.info("FULL DAY PATTERN DETECTION TEST - 8/13")
        logger.info("="*80)
        logger.info("Testing ALL 158 ACTION alerts with REAL pattern detection")
        logger.info("ACTION alert surge = FLAGPOLE, then look for pullback + breakout")
        
        alerts = self.load_action_alerts()
        
        if not alerts:
            logger.error("No ACTION alerts found!")
            return
        
        # Track results
        results = {
            'bull_flag_detected': 0,
            'no_data': 0,
            'insufficient_data': 0,
            'pullback_too_deep': 0,
            'too_many_red_candles': 0,
            'no_breakout_yet': 0,
            'successful_entries': []
        }
        
        logger.info(f"\nTesting {len(alerts)} ACTION alerts...")
        
        for i, alert in enumerate(alerts, 1):
            if i % 20 == 0:  # Progress update every 20
                logger.info(f"Progress: {i}/{len(alerts)} alerts tested")
            
            result = self.test_symbol(alert)
            pattern = result['pattern']
            
            # Count pattern types
            if pattern in results:
                results[pattern] += 1
            
            # Track successful entries
            if pattern == 'bull_flag_detected':
                results['successful_entries'].append(result)
                logger.info(f"🎯 BULL FLAG FOUND: {result['symbol']} at {result['alert_time']}")
                logger.info(f"   Entry: ${result['entry']:.2f}, Stop: ${result['stop_loss']:.2f}")
                logger.info(f"   Pullback: {result['pullback_depth']:.1f}%, Red candles: {result['red_candles']}")
        
        # Print final results
        logger.info(f"\n{'='*80}")
        logger.info("FULL DAY RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Total ACTION alerts tested: {len(alerts)}")
        logger.info(f"Bull flag patterns detected: {results['bull_flag_detected']}")
        logger.info(f"Success rate: {results['bull_flag_detected']/len(alerts)*100:.1f}%")
        
        logger.info(f"\nPattern breakdown:")
        for pattern, count in results.items():
            if pattern != 'successful_entries':
                logger.info(f"  {pattern}: {count}")
        
        if results['successful_entries']:
            logger.info(f"\n🎯 SUCCESSFUL BULL FLAG ENTRIES:")
            for entry in results['successful_entries']:
                logger.info(f"  {entry['symbol']} at {entry['alert_time']}: Entry ${entry['entry']:.2f}, Stop ${entry['stop_loss']:.2f}")
        else:
            logger.info(f"\n❌ No valid bull flag patterns found")
            logger.info(f"💡 This means Ross Cameron's strict criteria are working!")

def main():
    tester = FullDayPatternTester()
    tester.run_full_day_test()

if __name__ == "__main__":
    main()