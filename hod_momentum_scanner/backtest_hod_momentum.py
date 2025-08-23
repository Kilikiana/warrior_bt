#!/usr/bin/env python3
"""
Backtest HOD Momentum Scanner - Replay Aug 13 6AM-11:30AM minute by minute
Shows exactly when alerts would have fired throughout Ross Cameron's prime trading window
"""

import os
import sys
import json
import gzip
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque
import pytz

# Import backtest-specific scanners
from .momentum_scanner_components.backtest_squeeze_scanner import BacktestSqueezeScanner
from .momentum_scanner_components.backtest_rvol_scanner import BacktestRvolScanner
from .momentum_scanner_components.backtest_former_momo_scanner import BacktestFormerMomoScanner
from foundational_stock_screeners.float_scanner import FloatScanner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HODMomentumBacktester:
    """Backtest HOD Momentum Scanner on historical minute data"""
    
    def __init__(self, test_date: str):
        """Initialize backtester"""
        self.test_date = test_date
        self.et = pytz.timezone('US/Eastern')
        
        # Initialize backtest-specific scanners with Ross Cameron's parameters
        self.squeeze_scanner = BacktestSqueezeScanner()  # Detects both 5%/5min and 10%/10min
        self.rvol_scanner = BacktestRvolScanner(min_rvol_5min=500.0)  # 500% RVOL threshold
        self.former_momo_scanner = BacktestFormerMomoScanner(lookback_minutes=60)
        self.float_scanner = FloatScanner()
        
        # Data paths
        self.cache_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/shared_cache')
        self.results_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/results/hod_momentum_scans')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.minute_data = self.load_minute_data()
        self.five_min_data = self.load_5min_data()
        self.float_data = self.load_float_data()
        
        # Tracking
        self.alerts_fired = []
        self.price_history = defaultdict(deque)  # symbol -> deque of (time, price)
        self.volume_history = defaultdict(deque)  # symbol -> deque of (time, volume)
        
    def load_minute_data(self) -> Dict:
        """Load 1-minute bar data for test date"""
        file_path = self.cache_dir / 'ohlcv_1min_bars' / f'ohlcv_1min_{self.test_date}.json.gz'
        
        if not file_path.exists():
            logger.error(f"No minute data found for {self.test_date}")
            return {}
        
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)
        
        # Organize by symbol and time
        organized = {}
        for item in data:
            symbol = item['symbol']
            if symbol not in organized:
                organized[symbol] = {}
            
            for minute in item['minutes']:
                time_str = minute['time']
                organized[symbol][time_str] = minute
        
        logger.info(f"Loaded minute data for {len(organized)} symbols")
        return organized
    
    def load_5min_data(self) -> Dict:
        """Load 5-minute bar data for test date"""
        file_path = self.cache_dir / 'ohlcv_5min_bars' / f'ohlcv_5min_{self.test_date}.json.gz'
        
        if not file_path.exists():
            logger.error(f"No 5-min data found for {self.test_date}")
            return {}
        
        with gzip.open(file_path, 'rt') as f:
            data = json.load(f)
        
        # Organize by symbol
        organized = {}
        for item in data:
            symbol = item['symbol']
            organized[symbol] = item['bars']
        
        logger.info(f"Loaded 5-min data for {len(organized)} symbols")
        return organized
    
    def load_float_data(self) -> Dict:
        """Load float data from all float categories"""
        float_data = {}
        
        # Load LOW float stocks (<20M)
        low_float_file = self.cache_dir / 'low_float_stocks.json'
        if low_float_file.exists():
            with open(low_float_file, 'r') as f:
                stocks = json.load(f)
                for stock in stocks:
                    float_data[stock['symbol']] = stock.get('floatShares', 0) / 1_000_000
                logger.info(f"Loaded {len(stocks)} low float stocks")
        
        # Load MID float stocks (20-50M)
        mid_float_file = self.cache_dir / 'mid_float_stocks.json'
        if mid_float_file.exists():
            with open(mid_float_file, 'r') as f:
                stocks = json.load(f)
                for stock in stocks:
                    float_data[stock['symbol']] = stock.get('floatShares', 0) / 1_000_000
                logger.info(f"Loaded {len(stocks)} mid float stocks")
        
        # Load HIGH float stocks (>50M) - for tracking/warnings
        high_float_file = self.cache_dir / 'high_float_stocks.json'
        if high_float_file.exists():
            with open(high_float_file, 'r') as f:
                stocks = json.load(f)
                for stock in stocks:
                    float_data[stock['symbol']] = stock.get('floatShares', 0) / 1_000_000
                logger.info(f"Loaded {len(stocks)} high float stocks")
        
        logger.info(f"Total float data loaded for {len(float_data)} stocks")
        return float_data
    
    def scan_at_time(self, current_time: datetime, symbols: List[str]) -> List[Dict]:
        """Run scanners at a specific time"""
        alerts = []
        time_str = current_time.strftime('%H:%M')
        
        # Get last 5 minutes for RVOL calculation
        last_5_minutes = []
        for i in range(5):
            check_time = current_time - timedelta(minutes=i)
            last_5_minutes.append(check_time.strftime('%H:%M'))
        last_5_minutes.reverse()
        
        for symbol in symbols:
            if symbol not in self.minute_data:
                continue
            
            # Get current price and volume
            if time_str not in self.minute_data[symbol]:
                continue
            
            bar = self.minute_data[symbol][time_str]
            price = bar['close']
            volume = bar['volume']
            
            # PRICE FILTER: Skip penny stocks (<$1) and high-priced stocks (>$50)
            # Ross Cameron criteria: $1-$50 range
            if price < 1.0 or price > 50.0:
                continue  # Skip this symbol at this time
            
            # Determine price category for tracking
            price_category = 'LOW' if price <= 20.0 else 'MID'  # LOW: $1-20, MID: $20-50
            
            # Update squeeze scanner with price
            self.squeeze_scanner.update_price(symbol, price, time_str)
            
            # Check for squeezes (both 5%/5min and 10%/10min)
            squeeze_results = self.squeeze_scanner.check_squeezes(symbol)
            has_any_squeeze = squeeze_results.get('any_squeeze', False)
            has_quick_squeeze = squeeze_results.get('QUICK_SQUEEZE', {}).get('detected', False)
            has_strong_squeeze = squeeze_results.get('STRONG_SQUEEZE', {}).get('detected', False)
            
            # Calculate 5-min RVOL % using the RVOL scanner
            rvol_5min = self.rvol_scanner.calculate_5min_rvol(
                symbol, 
                self.minute_data[symbol], 
                last_5_minutes
            )
            
            # Check if former momo
            is_former_momo = self.former_momo_scanner.is_former_momo(symbol)
            
            # Get float and categorize
            float_m = self.float_data.get(symbol, 0)
            is_low_float = 0 < float_m < 20
            
            # Determine float category
            if float_m <= 0:
                float_category = 'UNKNOWN'
            elif float_m < 20:
                float_category = 'LOW_FLOAT'  # <20M shares - Ross Cameron's preference
            elif float_m <= 50:
                float_category = 'MID_FLOAT'  # 20-50M shares - acceptable
            else:
                float_category = 'HIGH_FLOAT'  # >50M shares - avoid for momentum
            
            # Evaluate alert conditions
            
            # SPECIAL CASE: Zero baseline with high volume (999999 flag)
            if has_strong_squeeze and rvol_5min >= 999999:
                # ACTION alert for zero baseline explosive moves
                if float_category != 'HIGH_FLOAT':
                    alert = {
                        'time': time_str,
                        'symbol': symbol,
                        'priority': 'ACTION',
                        'strategy': 'SQUEEZE_ZERO_BASELINE',
                        'description': f'10% in 10min + >100K vol (ZERO BASELINE) = ACTION!',
                        'squeeze_type': '10%/10min',
                        'price': price,
                        'price_category': price_category,
                        'rvol_5min': rvol_5min,
                        'float': float_m,
                        'float_category': float_category,
                        'is_low_float': is_low_float,
                        'former_momo': is_former_momo,
                        'action': 'IMMEDIATE_TECH_ANALYSIS'
                    }
                    alerts.append(alert)
                    # Track in former momo scanner
                    self.former_momo_scanner.add_alert(symbol, time_str, 'ACTION')
            
            # HIGHEST PRIORITY: 10% in 10min + RVOL >500% = ACTION!
            # BUT exclude high float stocks (>50M shares)
            elif has_strong_squeeze and rvol_5min > 500:
                if float_category != 'HIGH_FLOAT':
                    # ACTION alert for low/mid float stocks
                    alert = {
                        'time': time_str,
                        'symbol': symbol,
                        'priority': 'ACTION',
                        'strategy': 'STRONG_SQUEEZE_HIGH_RVOL',
                        'description': f'10% in 10min + {rvol_5min:.0f}% RVOL = TRIGGER ACTION!',
                        'squeeze_type': '10%/10min',
                        'price': price,
                        'price_category': price_category,  # LOW or MID price
                        'rvol_5min': rvol_5min,
                        'float': float_m,
                        'float_category': float_category,  # LOW_FLOAT or MID_FLOAT
                        'is_low_float': is_low_float,
                        'former_momo': is_former_momo,
                        'action': 'IMMEDIATE_TECH_ANALYSIS'
                    }
                    alerts.append(alert)
                    # Track in former momo scanner
                    self.former_momo_scanner.add_alert(symbol, time_str, 'ACTION')
                else:
                    # HIGH_FLOAT_WARNING for high float stocks that otherwise qualify
                    alert = {
                        'time': time_str,
                        'symbol': symbol,
                        'priority': 'HIGH_FLOAT_WARNING',
                        'strategy': 'STRONG_SQUEEZE_HIGH_FLOAT',
                        'description': f'⚠️ HIGH FLOAT ({float_m:.0f}M) - 10%/10min + {rvol_5min:.0f}% RVOL',
                        'squeeze_type': '10%/10min',
                        'price': price,
                        'price_category': price_category,
                        'rvol_5min': rvol_5min,
                        'float': float_m,
                        'float_category': float_category,
                        'is_low_float': False,
                        'former_momo': is_former_momo,
                        'action': 'MONITOR_ONLY_HIGH_FLOAT'
                    }
                    alerts.append(alert)
            
            # HIGH PRIORITY: 5% in 5min + High RVOL
            elif has_quick_squeeze and rvol_5min > 300:
                alert = {
                    'time': time_str,
                    'symbol': symbol,
                    'priority': 'HIGH',
                    'strategy': 'QUICK_SQUEEZE_RVOL',
                    'description': f'5% in 5min + {rvol_5min:.0f}% RVOL',
                    'squeeze_type': '5%/5min',
                    'price': price,
                    'price_category': price_category,
                    'rvol_5min': rvol_5min,
                    'float': float_m,
                    'is_low_float': is_low_float,
                    'former_momo': is_former_momo,
                    'action': 'WATCH_FOR_ENTRY'
                }
                alerts.append(alert)
                # Track in former momo scanner
                self.former_momo_scanner.add_alert(symbol, time_str, 'HIGH')
            
            # MEDIUM: Low Float + High RVOL (lowered to 200% for testing)
            elif is_low_float and rvol_5min > 200:
                alerts.append({
                    'time': time_str,
                    'symbol': symbol,
                    'priority': 'MEDIUM',
                    'strategy': 'LOW_FLOAT_HIGH_RVOL',
                    'description': f'Low Float + High RVOL (${price:.2f})',
                    'price': price,
                    'rvol_5min': rvol_5min,
                    'float': float_m,
                    'former_momo': is_former_momo
                })
            
            # MEDIUM: Just Squeeze (either type)
            elif has_any_squeeze:
                squeeze_type = '10%/10min' if has_strong_squeeze else '5%/5min'
                alerts.append({
                    'time': time_str,
                    'symbol': symbol,
                    'priority': 'MEDIUM',
                    'strategy': 'SQUEEZE_ONLY',
                    'description': f'{squeeze_type} squeeze',
                    'squeeze_type': squeeze_type,
                    'price': price,
                    'rvol_5min': rvol_5min,
                    'float': float_m,
                    'former_momo': is_former_momo
                })
            
            # MEDIUM: Just High RVOL
            elif rvol_5min > 300:
                alerts.append({
                    'time': time_str,
                    'symbol': symbol,
                    'priority': 'MEDIUM',
                    'strategy': 'HIGH_RVOL_ONLY',
                    'description': f'High RVOL {rvol_5min:.0f}%',
                    'price': price,
                    'rvol_5min': rvol_5min,
                    'float': float_m,
                    'is_low_float': is_low_float,
                    'former_momo': is_former_momo
                })
        
        return alerts
    
    def run_backtest(self, start_hour: int = 4, start_minute: int = 0,
                     end_hour: int = 20, end_minute: int = 0) -> Dict:
        """
        Run minute-by-minute backtest
        
        Args:
            start_hour: Start hour (default 4 AM - pre-market)
            start_minute: Start minute
            end_hour: End hour (default 8 PM - extended hours)
            end_minute: End minute (default 00 for 8:00 PM)
        """
        logger.info(f"Starting backtest for {self.test_date} {start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}")
        
        # Get all symbols we have data for
        symbols = list(self.minute_data.keys())
        logger.info(f"Scanning {len(symbols)} symbols")
        
        # Results tracking
        all_alerts = []
        high_priority_alerts = []
        alerts_by_time = defaultdict(list)
        alerts_by_symbol = defaultdict(list)
        
        # Create time range
        base_date = datetime.strptime(self.test_date, '%Y-%m-%d')
        base_date = self.et.localize(base_date)
        
        current_time = base_date.replace(hour=start_hour, minute=start_minute)
        end_time = base_date.replace(hour=end_hour, minute=end_minute)
        
        # Progress tracking
        total_minutes = int((end_time - current_time).total_seconds() / 60)
        minutes_processed = 0
        
        # Run minute by minute
        while current_time <= end_time:
            time_str = current_time.strftime('%H:%M')
            
            # Scan at this time
            alerts = self.scan_at_time(current_time, symbols)
            
            if alerts:
                all_alerts.extend(alerts)
                alerts_by_time[time_str] = alerts
                
                for alert in alerts:
                    alerts_by_symbol[alert['symbol']].append(alert)
                    
                    if alert['priority'] == 'ACTION':
                        high_priority_alerts.append(alert)
                        
                        # Log ACTION alerts immediately
                        print(f"\n{'='*60}")
                        print(f"🚨🚨🚨 {time_str} - ACTION ALERT (10%/10min + >500% RVOL)")
                        print(f"Symbol: {alert['symbol']}")
                        print(f"Strategy: {alert['strategy']}")
                        print(f"Price: ${alert['price']:.2f} ({alert.get('price_category', 'UNKNOWN')})")
                        print(f"5-min RVOL: {alert['rvol_5min']:.1f}%")
                        print(f"Float: {alert['float']:.1f}M ({alert.get('float_category', 'UNKNOWN')})")
                        print(f"Former Momo: {alert['former_momo']}")
                        print(f"Action: {alert['action']}")
                        print(f"{'='*60}")
                    
                    elif alert['priority'] == 'HIGH_FLOAT_WARNING':
                        # Log HIGH_FLOAT_WARNING alerts
                        print(f"\n{'⚠'*60}")
                        print(f"⚠️  {time_str} - HIGH FLOAT WARNING")
                        print(f"Symbol: {alert['symbol']}")
                        print(f"Price: ${alert['price']:.2f} ({alert.get('price_category', 'UNKNOWN')})")
                        print(f"Float: {alert['float']:.1f}M (HIGH_FLOAT - AVOID!)")
                        print(f"5-min RVOL: {alert['rvol_5min']:.1f}%")
                        print(f"Would qualify for ACTION but float too high for momentum trading")
                        print(f"{'⚠'*60}")
                        
                    elif alert['priority'] == 'HIGH':
                        high_priority_alerts.append(alert)
                        
                        # Log HIGH priority alerts
                        print(f"\n{'-'*60}")
                        print(f"⚠️  {time_str} - HIGH ALERT (5%/5min)")
                        print(f"Symbol: {alert['symbol']}")
                        print(f"Price: ${alert['price']:.2f}")
                        print(f"5-min RVOL: {alert['rvol_5min']:.1f}%")
                        print(f"{'-'*60}")
            
            # Progress update every 30 minutes
            minutes_processed += 1
            if minutes_processed % 30 == 0:
                logger.info(f"Processed {time_str} - {minutes_processed}/{total_minutes} minutes "
                          f"({len(all_alerts)} alerts so far)")
            
            # Move to next minute
            current_time += timedelta(minutes=1)
        
        # Compile results
        results = {
            'test_date': self.test_date,
            'time_range': f"{start_hour:02d}:{start_minute:02d} - {end_hour:02d}:{end_minute:02d}",
            'symbols_scanned': len(symbols),
            'total_minutes': total_minutes,
            'total_alerts': len(all_alerts),
            'high_priority_alerts': len(high_priority_alerts),
            'all_alerts': all_alerts,
            'alerts_by_time': dict(alerts_by_time),
            'alerts_by_symbol': dict(alerts_by_symbol),
            'high_priority_details': high_priority_alerts
        }
        
        return results
    
    def save_results(self, results: Dict):
        """Save backtest results"""
        # Save with predictable name for simulator (overwrites previous)
        filename = f"hod_momentum_scan_{self.test_date}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath
    
    def print_summary(self, results: Dict):
        """Print backtest summary"""
        print("\n" + "="*80)
        print(f"HOD MOMENTUM BACKTEST RESULTS - {results['test_date']}")
        print(f"Time Range: {results['time_range']} ET")
        print("="*80)
        
        print(f"\nSymbols Scanned: {results['symbols_scanned']}")
        print(f"Total Minutes: {results['total_minutes']}")
        print(f"Total Alerts: {results['total_alerts']}")
        print(f"High Priority Alerts: {results['high_priority_alerts']}")
        
        # Count specific alert combinations
        strong_squeeze_high_rvol_count = 0
        strong_squeeze_high_rvol_former_momo_count = 0
        strong_squeeze_high_rvol_symbols = set()
        strong_squeeze_high_rvol_former_momo_symbols = set()
        
        for alert in results.get('all_alerts', []):
            # Check for 10% squeeze in 10 mins AND >500% RVOL
            if (alert.get('squeeze_type') == '10%/10min' and 
                alert.get('rvol_5min', 0) > 500):
                strong_squeeze_high_rvol_count += 1
                strong_squeeze_high_rvol_symbols.add(alert['symbol'])
                
                # Check if also has Former Momo
                if alert.get('former_momo', False):
                    strong_squeeze_high_rvol_former_momo_count += 1
                    strong_squeeze_high_rvol_former_momo_symbols.add(alert['symbol'])
        
        # Show special alert combinations
        print("\n" + "="*40)
        print("🎯 SPECIAL ALERT COMBINATIONS:")
        print("="*40)
        print(f"\n📊 10% Squeeze + >500% RVOL Combined:")
        print(f"   Total Alerts: {strong_squeeze_high_rvol_count}")
        print(f"   Unique Symbols: {len(strong_squeeze_high_rvol_symbols)}")
        if strong_squeeze_high_rvol_symbols:
            print(f"   Symbols: {', '.join(sorted(strong_squeeze_high_rvol_symbols)[:10])}")
            if len(strong_squeeze_high_rvol_symbols) > 10:
                print(f"            ... and {len(strong_squeeze_high_rvol_symbols) - 10} more")
        
        print(f"\n🔥 10% Squeeze + >500% RVOL + Former Momo:")
        print(f"   Total Alerts: {strong_squeeze_high_rvol_former_momo_count}")
        print(f"   Unique Symbols: {len(strong_squeeze_high_rvol_former_momo_symbols)}")
        if strong_squeeze_high_rvol_former_momo_symbols:
            print(f"   Symbols: {', '.join(sorted(strong_squeeze_high_rvol_former_momo_symbols))}")
        
        # Show high priority alerts
        if results['high_priority_details']:
            print("\n" + "-"*40)
            print("HIGH PRIORITY ALERTS (Squeeze + High RVOL):")
            print("-"*40)
            for alert in results['high_priority_details']:
                print(f"\n{alert['time']} - {alert['symbol']}")
                print(f"  Price: ${alert['price']:.2f}")
                print(f"  5-min RVOL: {alert['rvol_5min']:.1f}%")
                print(f"  Float: {alert['float']:.1f}M")
                print(f"  Former Momo: {alert['former_momo']}")
        
        # Show alert distribution by time
        print("\n" + "-"*40)
        print("ALERT DISTRIBUTION BY TIME:")
        print("-"*40)
        
        # Group by 30-minute windows
        time_buckets = defaultdict(int)
        for time_str in results['alerts_by_time']:
            hour, minute = map(int, time_str.split(':'))
            bucket = f"{hour:02d}:{(minute // 30) * 30:02d}"
            time_buckets[bucket] += len(results['alerts_by_time'][time_str])
        
        for bucket in sorted(time_buckets.keys()):
            print(f"  {bucket}: {time_buckets[bucket]} alerts")
        
        # Show top symbols by alert count
        if results['alerts_by_symbol']:
            print("\n" + "-"*40)
            print("TOP SYMBOLS BY ALERT COUNT:")
            print("-"*40)
            
            symbol_counts = [(sym, len(alerts)) for sym, alerts in results['alerts_by_symbol'].items()]
            symbol_counts.sort(key=lambda x: x[1], reverse=True)
            
            for symbol, count in symbol_counts[:10]:
                print(f"  {symbol}: {count} alerts")


def main():
    """Run HOD Momentum backtest for specified date, 6AM-11:30AM"""
    from dotenv import load_dotenv
    import argparse
    
    load_dotenv('/Users/claytonsmacbookpro/Projects/warrior_bt/.env')
    
    parser = argparse.ArgumentParser(description='Run HOD momentum backtest')
    parser.add_argument('--date', type=str, required=True, help='Date to test (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Initialize backtester with specified date
    backtester = HODMomentumBacktester(test_date=args.date)
    
    # Run backtest for full day including extended hours (4 AM - 8 PM)
    results = backtester.run_backtest(
        start_hour=4, start_minute=0,
        end_hour=20, end_minute=0
    )
    
    # Print summary
    backtester.print_summary(results)
    
    # Save results
    filepath = backtester.save_results(results)
    
    print(f"\n✅ Backtest complete! Full results saved to:\n{filepath}")


if __name__ == "__main__":
    main()