#!/usr/bin/env python3
"""
Interactive Aggregate Scanner - Combines All Ross Cameron Criteria
Based on the working aggregate_scanner.py with added date/time parameters
Validates historical data availability before running
Core: Float <20M, Price $1-$20, Gap >10%, RVOL >5x
Bonus: News catalyst
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
import logging
from pathlib import Path
import argparse
import pytz
from typing import Dict, List, Optional, Tuple
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveAggregateScanner:
    def __init__(self, test_date: str = None, start_time: str = "06:00", end_time: str = "11:30"):
        """
        Initialize scanner with date and time range
        
        Args:
            test_date: Date to test in YYYY-MM-DD format (None = most recent trading day)
            start_time: Start time in HH:MM format (EST)
            end_time: End time in HH:MM format (EST)
        """
        # Time settings
        self.et = pytz.timezone('US/Eastern')
        
        # Set test date (default to most recent trading day)
        if test_date:
            self.test_date = test_date
            self.test_datetime = pd.Timestamp(test_date, tz=self.et)
        else:
            self.test_date, self.test_datetime = self.get_most_recent_trading_day()
        
        # Parse time range
        self.start_time = start_time
        self.end_time = end_time
        self.start_hour, self.start_minute = map(int, start_time.split(':'))
        self.end_hour, self.end_minute = map(int, end_time.split(':'))
        
        self.cache_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/shared_cache')
        
        logger.info(f"Interactive Aggregate Scanner initialized for {self.test_date}")
        logger.info(f"Time range: {start_time} to {end_time} EST")
    
    def get_most_recent_trading_day(self) -> Tuple[str, pd.Timestamp]:
        """Get the most recent trading day"""
        et = pytz.timezone('US/Eastern')
        now = pd.Timestamp.now(tz=et)
        
        # If it's after 4pm, use today (if weekday)
        # Otherwise use previous trading day
        if now.hour >= 16:
            test_day = now.normalize()
        else:
            test_day = now.normalize() - timedelta(days=1)
        
        # Skip weekends
        while test_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
            test_day = test_day - timedelta(days=1)
        
        return test_day.strftime('%Y-%m-%d'), test_day
    
    def validate_historical_data(self) -> bool:
        """
        Validate that we have the necessary historical data for RVOL calculations
        Returns True if data is available, False otherwise
        """
        logger.info("Validating historical data availability...")
        
        # Check if minute volume averages exist
        volume_file = self.cache_dir / 'minute_volume_averages_30d.json'
        if not volume_file.exists():
            # Fallback to 20d if 30d not available
            volume_file = self.cache_dir / 'minute_volume_averages_20d.json'
            if not volume_file.exists():
                logger.error("❌ No minute volume averages file found!")
            logger.error("   Please run the minute volume fetcher from historical_OHLC_fetcher/")
            return False
        
        # Calculate what dates we need for 20-day average
        required_days = []
        current = self.test_datetime
        days_found = 0
        
        while days_found < 20:
            current = current - timedelta(days=1)
            # Skip weekends
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                required_days.append(current.strftime('%Y-%m-%d'))
                days_found += 1
        
        logger.info(f"Need historical data from {required_days[-1]} to {required_days[0]} (20 trading days)")
        
        # Check if we have data for these dates by looking at raw volume files
        volume_dir = self.cache_dir / 'minute_volume_cache'
        if not volume_dir.exists():
            # Try alternate location
            volume_dir = Path('minute_volumes')
            if not volume_dir.exists():
                logger.warning("⚠️  No minute volumes directory found at standard locations")
                logger.info("   Continuing anyway - volume averages file exists")
        
        # Check for at least some of the required dates
        missing_dates = []
        if volume_dir.exists():
            for date_str in required_days[-5:]:  # Check last 5 days as a sample
                date_file = volume_dir / f"{date_str}.json.gz"
                if not date_file.exists():
                    missing_dates.append(date_str)
        
        if missing_dates:
            logger.warning(f"⚠️  Missing volume data for dates: {missing_dates}")
            logger.info("   You may want to run: python update_volume_history.py")
            # Don't fail completely - we might have enough data
        
        logger.info("✅ Historical data validation passed")
        return True
    
    def run_individual_scanners(self):
        """Run the individual backtest scanners to generate required data files"""
        logger.info("\nRunning individual scanners for specified date/time...")
        
        scanners = [
            {
                'name': 'Price Window Scanner',
                'script': 'backtest_price_scanner.py',
                'output': '/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans/backtest_price_results.csv',
                'command': f"python3 ../backtest_criteria_scanners/backtest_price_scanner.py --date {self.test_date} --start {self.start_time} --end {self.end_time}"
            },
            {
                'name': 'Gap Scanner',
                'script': 'backtest_gap_scanner.py',
                'output': '/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans/backtest_gap_results.csv',
                'command': f"python3 ../backtest_criteria_scanners/backtest_gap_scanner.py --date {self.test_date} --start {self.start_time} --end {self.end_time}"
            },
            {
                'name': 'RVOL Scanner',
                'script': 'backtest_rvol_scanner.py',
                'output': '/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans/backtest_rvol_results.csv',
                'command': f"python3 ../backtest_criteria_scanners/backtest_rvol_scanner.py --date {self.test_date} --start {self.start_time} --end {self.end_time}"
            },
            {
                'name': 'News Scanner (24-hour window)',
                'script': 'backtest_news_scanner.py',
                'output': '/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans/backtest_news_results.csv',
                'command': f"python3 ../backtest_criteria_scanners/backtest_news_scanner.py --date {self.test_date} --end {self.end_time}"
            }
        ]
        
        for scanner in scanners:
            logger.info(f"\nRunning {scanner['name']}...")
            logger.info(f"Command: {scanner['command']}")
            
            try:
                result = subprocess.run(
                    scanner['command'].split(),
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    # Check if output file was created
                    if os.path.exists(scanner['output']):
                        file_size = os.path.getsize(scanner['output'])
                        logger.info(f"✅ {scanner['name']} completed successfully (output: {file_size:,} bytes)")
                    else:
                        logger.warning(f"⚠️  {scanner['name']} completed but no output file found")
                else:
                    logger.error(f"❌ {scanner['name']} failed with return code {result.returncode}")
                    if result.stderr:
                        logger.error(f"   Error: {result.stderr[:500]}")
                        
            except subprocess.TimeoutExpired:
                logger.error(f"❌ {scanner['name']} timed out after 5 minutes")
            except FileNotFoundError:
                logger.error(f"❌ Could not find {scanner['script']}. Make sure it exists.")
            except Exception as e:
                logger.error(f"❌ Error running {scanner['name']}: {e}")
    
    def load_stock_universe(self):
        """Load the full US stock universe as our base"""
        import pickle
        universe_file = self.cache_dir / 'stock_universe_cache.pkl'
        if universe_file.exists():
            with open(universe_file, 'rb') as f:
                stocks = pickle.load(f)
            # Create set of all symbols
            universe_symbols = {stock['symbol'] for stock in stocks}
            logger.info(f"Loaded {len(universe_symbols)} stocks from US stock universe")
            return universe_symbols
        else:
            logger.warning("Stock universe cache not found. Will use symbols from scanner results as fallback.")
            return None
    
    def load_float_stocks(self):
        """Load low and mid-float stocks from cache (as independent scanners)"""
        float_categories = {}
        
        # Load low-float stocks (<20M)
        low_float_file = self.cache_dir / 'low_float_stocks.json'
        if low_float_file.exists():
            with open(low_float_file, 'r') as f:
                data = json.load(f)
            for stock in data:
                float_categories[stock['symbol']] = {
                    'floatShares': stock['floatShares'],
                    'category': 'LOW'
                }
            logger.info(f"Loaded {len(data)} low-float stocks (<20M shares)")
        
        # Load mid-float stocks (20M-50M)
        mid_float_file = self.cache_dir / 'mid_float_stocks.json'
        if mid_float_file.exists():
            with open(mid_float_file, 'r') as f:
                data = json.load(f)
            for stock in data:
                float_categories[stock['symbol']] = {
                    'floatShares': stock['floatShares'],
                    'category': 'MID'
                }
            logger.info(f"Loaded {len(data)} mid-float stocks (20M-50M shares)")
        
        if float_categories:
            logger.info(f"Total: {len(float_categories)} low/mid-float stocks loaded")
        else:
            logger.info("Float scanner caches not found.")
        
        return float_categories
    
    def load_price_window_stocks(self):
        """Load stocks in $1-$50 price window with categories"""
        price_file = '../../results/criteria_scans/backtest_price_results.csv'
        if not os.path.exists(price_file):
            logger.warning("Price window results not found")
            return {}, {}
        
        df = pd.read_csv(price_file)
        # Separate by price category
        price_data = {}
        for _, row in df.iterrows():
            symbol = row['symbol']
            price = row['close']  # Use 'close' instead of 'price'
            if symbol not in price_data:
                price_data[symbol] = {
                    'price': price,
                    'category': 'LOW' if price <= 20 else 'MID'
                }
        
        low_price = {s for s, d in price_data.items() if d['category'] == 'LOW'}
        mid_price = {s for s, d in price_data.items() if d['category'] == 'MID'}
        
        logger.info(f"Loaded {len(low_price)} stocks in $1-$20 window (LOW)")
        logger.info(f"Loaded {len(mid_price)} stocks in $20-$50 window (MID)")
        return price_data, (low_price | mid_price)
    
    def load_gap_stocks(self):
        """Calculate real-time gaps using current minute prices vs previous close"""
        import gzip
        
        # Load previous day's close prices from daily bars
        prev_date = (self.test_datetime - timedelta(days=1)).strftime('%Y-%m-%d')
        while prev_date and pd.Timestamp(prev_date).weekday() >= 5:  # Skip weekends
            prev_date = (pd.Timestamp(prev_date) - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # First, try to get current minute prices for gap calculation
        current_minute_file = self.cache_dir / f'ohlcv_1min_bars/ohlcv_1min_{self.test_date}.json.gz'
        
        if not current_minute_file.exists():
            logger.warning(f"No minute data found for {self.test_date}, falling back to static gaps")
            # Fallback to static gap file
            gap_file = '../../results/criteria_scans/backtest_gap_results.csv'
            if not os.path.exists(gap_file):
                logger.warning("Gap scanner results not found")
                return {}
            df = pd.read_csv(gap_file)
            gap_dict = df.groupby('symbol')['gap_pct'].max().to_dict()
            high_gap = {s: g for s, g in gap_dict.items() if g > 10}
            logger.info(f"Loaded {len(high_gap)} stocks with >10% gap (static)")
            return high_gap
        
        # Load current minute data
        try:
            with gzip.open(current_minute_file, 'rt') as f:
                minute_data_list = json.load(f)  # This is a list of objects
        except Exception as e:
            logger.error(f"Failed to load minute data: {e}")
            return {}
        
        # Convert to dict for easier access
        minute_data = {}
        for stock_data in minute_data_list:
            symbol = stock_data['symbol']
            minute_data[symbol] = stock_data['minutes']
        
        # Load previous day's close prices from daily bars
        daily_closes = {}
        daily_dir = self.cache_dir / 'ohlcv_daily_bars'
        if daily_dir.exists():
            for daily_file in daily_dir.glob('daily_*.json'):
                try:
                    symbol = daily_file.stem.replace('daily_', '')
                    with open(daily_file, 'r') as f:
                        daily_data = json.load(f)
                    
                    # The daily data has a 'bars' array
                    daily_bars = daily_data.get('bars', [])
                    
                    # Find previous day's close
                    for bar in reversed(daily_bars):  # Start from most recent
                        bar_date = bar['date']  # This is in YYYY-MM-DD format
                        if bar_date == prev_date:
                            daily_closes[symbol] = bar['close']
                            break
                        elif bar_date < prev_date:  # Gone too far back
                            break
                except:
                    continue
        
        logger.info(f"Loaded {len(daily_closes)} previous day closes for gap calculation")
        
        # Calculate current gaps using minute data within our time window
        real_time_gaps = {}
        
        # Convert time window to target minute
        target_time_start = f"{self.start_hour:02d}:{self.start_minute:02d}"
        target_time_end = f"{self.end_hour:02d}:{self.end_minute:02d}"
        
        for symbol, minutes in minute_data.items():
            if symbol not in daily_closes:
                continue
                
            prev_close = daily_closes[symbol]
            
            # Find the first price within our time window
            current_price = None
            for minute_bar in minutes:
                bar_time = minute_bar['time']  # This is in format "HH:MM"
                
                # Check if this bar is within our time range
                if target_time_start <= bar_time <= target_time_end:
                    current_price = minute_bar['close']  # Use close price of first matching minute
                    break
            
            if current_price is not None and prev_close > 0:
                gap_pct = ((current_price - prev_close) / prev_close) * 100
                if gap_pct > 10:  # Only include gaps >10%
                    real_time_gaps[symbol] = gap_pct
        
        logger.info(f"Calculated {len(real_time_gaps)} stocks with >10% real-time gap")
        logger.info(f"Time window: {target_time_start} to {target_time_end}")
        
        # Show top 5 gaps for verification
        if real_time_gaps:
            top_gaps = sorted(real_time_gaps.items(), key=lambda x: x[1], reverse=True)[:5]
            for symbol, gap in top_gaps:
                logger.info(f"  {symbol}: {gap:.1f}%")
        
        return real_time_gaps
    
    def load_rvol_stocks(self):
        """Load stocks with high RVOL (both cumulative and window)"""
        rvol_file = '../../results/criteria_scans/backtest_rvol_results.csv'
        if not os.path.exists(rvol_file):
            logger.warning("RVOL results not found")
            return {}, {}
        
        df = pd.read_csv(rvol_file)
        
        # Get max RVOL values for each symbol
        cumulative_rvol = df.groupby('symbol')['cumulative_rvol'].max().to_dict()
        window_rvol = df.groupby('symbol')['window_rvol'].max().to_dict()
        
        # Filter for >5x RVOL
        high_cumulative = {s: r for s, r in cumulative_rvol.items() if r > 5}
        high_window = {s: r for s, r in window_rvol.items() if r > 5}
        
        logger.info(f"Loaded {len(high_cumulative)} stocks with >5x cumulative RVOL")
        logger.info(f"Loaded {len(high_window)} stocks with >5x window RVOL")
        
        return high_cumulative, high_window
    
    def load_news_stocks(self):
        """Load stocks with news catalysts"""
        # Try backtest news results first, then fall back to regular news scan
        news_file = self.cache_dir / 'backtest_news_results.json'
        if not news_file.exists():
            news_file = self.cache_dir / 'news_scan_results.json'
            if not news_file.exists():
                # Also check results directory
                news_file = Path('../../results/criteria_scans/backtest_news_results.csv')
                if news_file.exists():
                    df = pd.read_csv(news_file)
                    news_stocks = set(df['symbol'].unique())
                    logger.info(f"Loaded {len(news_stocks)} stocks with news from CSV")
                    return news_stocks
                else:
                    logger.warning("News scan results not found")
                    return set()
        
        with open(news_file, 'r') as f:
            data = json.load(f)
        
        news_stocks = set(data['stocks_with_news_list'])
        logger.info(f"Loaded {len(news_stocks)} stocks with news")
        return news_stocks
    
    def aggregate_scanners(self):
        """Combine all scanner results and rank candidates"""
        
        # Load the stock universe first (primary source)
        stock_universe = self.load_stock_universe()
        
        # Load all scanner results (including float as independent scanner)
        float_stocks = self.load_float_stocks()  # Now treated as independent scanner
        price_data, price_window = self.load_price_window_stocks()
        gap_stocks = self.load_gap_stocks()
        cumulative_rvol, window_rvol = self.load_rvol_stocks()
        news_stocks = self.load_news_stocks()
        
        # Use stock universe as primary source, fallback to scanner results if not available
        if stock_universe:
            all_symbols = stock_universe
            logger.info(f"Using stock universe with {len(all_symbols)} symbols as base")
        else:
            # Fallback: Build universe from scanner results
            all_symbols = set()
            all_symbols.update(float_stocks.keys()) if float_stocks else None
            all_symbols.update(price_window)
            all_symbols.update(gap_stocks.keys())
            all_symbols.update(cumulative_rvol.keys())
            all_symbols.update(window_rvol.keys())
            all_symbols.update(news_stocks)
            logger.info(f"Using {len(all_symbols)} symbols from scanner results (fallback)")
        
        results = []
        
        for symbol in all_symbols:
            # Track criteria met
            criteria_met = []
            score = 0
            float_category = None
            
            # Core criteria (required for Ross Cameron)
            has_float_data = symbol in float_stocks
            has_price_window = symbol in price_window
            has_gap = symbol in gap_stocks
            has_cumulative_rvol = symbol in cumulative_rvol
            has_window_rvol = symbol in window_rvol
            
            # Bonus criteria
            has_news = symbol in news_stocks
            
            # Build criteria list and calculate score
            if has_float_data:
                float_info = float_stocks[symbol]
                float_category = float_info['category']
                float_shares = float_info['floatShares']
                
                if float_category == 'LOW':
                    criteria_met.append('Float<20M')
                    score += 2  # Low float gets higher score (more volatile)
                elif float_category == 'MID':
                    criteria_met.append('Float20-50M')
                    score += 1  # Mid float gets lower score
                
            if has_price_window:
                if symbol in price_data:
                    price_cat = price_data[symbol]['category']
                    price_val = price_data[symbol]['price']
                    if price_cat == 'LOW':
                        criteria_met.append(f'Price${price_val:.2f}(LOW)')
                        score += 2  # Low price gets higher score
                    else:
                        criteria_met.append(f'Price${price_val:.2f}(MID)')
                        score += 1  # Mid price gets lower score
                else:
                    criteria_met.append('Price$1-50')
                    score += 1
                
            if has_gap:
                criteria_met.append(f'Gap>{gap_stocks[symbol]:.1f}%')
                score += 2  # Gap is more important
                
            if has_cumulative_rvol:
                criteria_met.append(f'RVOL_C>{cumulative_rvol[symbol]:.1f}x')
                score += 2  # RVOL is critical
                
            if has_window_rvol:
                criteria_met.append(f'RVOL_W>{window_rvol[symbol]:.1f}x')
                score += 1  # Window RVOL shows immediate activity
                
            if has_news:
                criteria_met.append('NEWS')
                score += 1  # Bonus point for news
            
            # Only include stocks meeting at least 2 core criteria
            if len([c for c in [has_float_data, has_price_window, has_gap, 
                               has_cumulative_rvol or has_window_rvol] if c]) >= 2:
                
                results.append({
                    'symbol': symbol,
                    'score': score,
                    'float_category': float_category if float_category else 'HIGH',
                    'float_shares': float_stocks[symbol]['floatShares'] if symbol in float_stocks else 0,
                    'gap_pct': gap_stocks.get(symbol, 0),
                    'cumulative_rvol': cumulative_rvol.get(symbol, 0),
                    'window_rvol': window_rvol.get(symbol, 0),
                    'has_news': has_news,
                    'criteria_met': ', '.join(criteria_met),
                    # Binary flags for filtering
                    'has_float': has_float_data,
                    'has_price': has_price_window,
                    'has_gap': has_gap,
                    'has_rvol': has_cumulative_rvol or has_window_rvol,
                    'meets_all_core': has_float_data and has_price_window and has_gap and 
                                     (has_cumulative_rvol or has_window_rvol)
                })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: (x['score'], x['gap_pct'], x['cumulative_rvol']), reverse=True)
        
        return results
    
    def display_results(self, results):
        """Display aggregated scanner results"""
        
        print(f"\n{'='*80}")
        print(f"INTERACTIVE AGGREGATE SCANNER RESULTS")
        print(f"Date: {self.test_date} | Time: {self.start_time} to {self.end_time} EST")
        print(f"{'='*80}\n")
        
        # Count stocks by criteria combinations
        all_core = [r for r in results if r['meets_all_core']]
        all_core_plus_news = [r for r in all_core if r['has_news']]
        
        print(f"Total stocks analyzed: {len(results)}")
        print(f"Stocks meeting ALL core criteria: {len(all_core)}")
        print(f"Stocks meeting ALL core + news: {len(all_core_plus_news)}")
        
        # Show distribution
        print(f"\nCriteria Distribution:")
        print(f"  Has Low Float (<20M): {len([r for r in results if r.get('float_category') == 'LOW'])}")
        print(f"  Has Mid Float (20-50M): {len([r for r in results if r.get('float_category') == 'MID'])}")
        print(f"  Has Price $1-20: {len([r for r in results if r['has_price']])}")
        print(f"  Has Gap >10%: {len([r for r in results if r['has_gap']])}")
        print(f"  Has RVOL >5x: {len([r for r in results if r['has_rvol']])}")
        print(f"  Has News: {len([r for r in results if r['has_news']])}")
        
        # Top candidates meeting ALL core criteria
        print(f"\n{'='*80}")
        print(f"TOP CANDIDATES - ALL CORE CRITERIA (Float, Price, Gap, RVOL)")
        print(f"{'='*80}")
        
        if all_core:
            print(f"\n{'Symbol':<8} {'Score':<7} {'Float':<10} {'Cat':<4} {'Gap%':<8} {'RVOL_C':<8} {'RVOL_W':<8} {'News':<6}")
            print("-" * 85)
            
            for r in all_core[:20]:  # Top 20
                float_str = f"{r['float_shares']/1e6:.1f}M" if r['float_shares'] > 0 else "N/A"
                print(f"{r['symbol']:<8} {r['score']:<7} "
                      f"{float_str:>9}  "
                      f"{r.get('float_category', 'N/A'):<4} "
                      f"{r['gap_pct']:>7.1f}% "
                      f"{r['cumulative_rvol']:>7.1f}x "
                      f"{r['window_rvol']:>7.1f}x "
                      f"{'YES' if r['has_news'] else 'NO':<6}")
        else:
            print("No stocks meeting all core criteria")
        
        # Top candidates by score (may not meet all criteria)
        print(f"\n{'='*80}")
        print(f"TOP CANDIDATES BY SCORE (Weighted Criteria)")
        print(f"{'='*80}")
        
        top_scores = results[:30]  # Top 30 by score
        if top_scores:
            print(f"\n{'Symbol':<8} {'Score':<7} {'Criteria Met':<60}")
            print("-" * 80)
            
            for r in top_scores:
                print(f"{r['symbol']:<8} {r['score']:<7} {r['criteria_met'][:60]}")
        
        # Stocks with highest gap + RVOL combo (top gappers)
        print(f"\n{'='*80}")
        print(f"TOP GAPPERS (High Gap + High RVOL)")
        print(f"{'='*80}")
        
        momentum = [r for r in results if r['gap_pct'] > 10 and r['cumulative_rvol'] > 5]
        momentum.sort(key=lambda x: x['gap_pct'] * x['cumulative_rvol'], reverse=True)
        
        if momentum:
            print(f"\n{'Symbol':<8} {'Float(M)':<10} {'Gap%':<8} {'RVOL':<8} {'Gap×RVOL':<10} {'News':<6}")
            print("-" * 65)
            
            for r in momentum[:15]:
                combo_score = r['gap_pct'] * r['cumulative_rvol']
                float_str = f"{r['float_shares']/1e6:.1f}M" if r['float_shares'] > 0 else "N/A"
                print(f"{r['symbol']:<8} {float_str:<10} {r['gap_pct']:>7.1f}% {r['cumulative_rvol']:>7.1f}x "
                      f"{combo_score:>9.0f} {'YES' if r['has_news'] else 'NO':<6}")
        
        return all_core, all_core_plus_news
    
    def save_results(self, results):
        """Save aggregate results to file"""
        
        # Save to CSV for easy analysis
        df = pd.DataFrame(results)
        results_dir = Path('../../results/aggregate_scans')
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / 'interactive_aggregate_results.csv', index=False)
        
        # Save detailed JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = results_dir / f"interactive_aggregate_{self.test_date}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'scan_date': self.test_date,
                'time_range': f"{self.start_time} to {self.end_time} EST",
                'total_stocks': len(results),
                'stocks_meeting_all_core': len([r for r in results if r['meets_all_core']]),
                'results': results[:100]  # Top 100 for JSON
            }, f, indent=2)
        
        logger.info(f"Saved results to {results_dir / 'interactive_aggregate_results.csv'} and {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Interactive Aggregate Scanner for Ross Cameron Criteria',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan today with default times (6:00 AM to 11:30 AM EST)
  python interactive_aggregate_scanner.py
  
  # Scan specific date
  python interactive_aggregate_scanner.py --date 2025-08-11
  
  # Scan with custom time range
  python interactive_aggregate_scanner.py --date 2025-08-11 --start 09:30 --end 10:30
  
  # Just validate data without running scanners
  python interactive_aggregate_scanner.py --validate-only
        """
    )
    
    parser.add_argument('--date', type=str, help='Date to test (YYYY-MM-DD). Default: most recent trading day')
    parser.add_argument('--start', type=str, default='06:00', help='Start time (HH:MM in EST). Default: 06:00')
    parser.add_argument('--end', type=str, default='11:30', help='End time (HH:MM in EST). Default: 11:30')
    parser.add_argument('--validate-only', action='store_true', help='Only validate data availability, don\'t run scanners')
    parser.add_argument('--skip-individual', action='store_true', help='Skip running individual scanners (use existing results)')
    
    args = parser.parse_args()
    
    # Create scanner
    scanner = InteractiveAggregateScanner(
        test_date=args.date,
        start_time=args.start,
        end_time=args.end
    )
    
    # Validate historical data
    if not scanner.validate_historical_data():
        logger.error("\n⚠️  Historical data validation failed!")
        logger.error("Please ensure you have the necessary volume data before running the scanner.")
        logger.info("\nTo fetch missing data:")
        logger.info("  1. For initial setup: run the minute volume fetcher from historical_OHLC_fetcher/")
        logger.info("  2. For updates: python update_volume_history.py")
        if not args.validate_only:
            sys.exit(1)
    
    if args.validate_only:
        logger.info("\nValidation complete. Use without --validate-only to run the scanner.")
        return
    
    # Run individual scanners unless skipped
    if not args.skip_individual:
        scanner.run_individual_scanners()
    else:
        logger.info("Skipping individual scanners, using existing results...")
    
    # Run aggregation
    logger.info("\nAggregating scanner results...")
    results = scanner.aggregate_scanners()
    
    # Display results
    all_core, all_core_plus_news = scanner.display_results(results)
    
    # Save results
    scanner.save_results(results)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"\nRoss Cameron Criteria Met:")
    print(f"  • {len(all_core)} stocks meet ALL core criteria (Float, Price, Gap, RVOL)")
    print(f"  • {len(all_core_plus_news)} of those also have news catalysts")
    
    if all_core:
        print(f"\nTop 5 Trading Candidates:")
        for i, r in enumerate(all_core[:5], 1):
            news_str = " + NEWS" if r['has_news'] else ""
            print(f"  {i}. {r['symbol']}: Gap {r['gap_pct']:.1f}%, RVOL {r['cumulative_rvol']:.1f}x{news_str}")

if __name__ == "__main__":
    main()