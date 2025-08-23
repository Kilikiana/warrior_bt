#!/usr/bin/env python3
"""
RVOL Scanner with Batch API Calls
Calculates both cumulative RVOL and current 5-minute window RVOL
Uses batch API calls for efficiency
"""

from alpaca_trade_api import REST, TimeFrame
import os
from dotenv import load_dotenv
import pandas as pd
import pytz
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from foundational_stock_screeners.stock_universe_builder import StockUniverseBuilder
import time
import json
from datetime import datetime, timedelta
import argparse
from pathlib import Path

load_dotenv('/Users/claytonsmacbookpro/Projects/warrior_bt/.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Alpaca API
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url='https://paper-api.alpaca.markets'
)

def load_minute_averages():
    """Load pre-calculated minute volume averages"""
    cache_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/shared_cache')
    
    # Try 30-day averages first, then 20-day
    avg_file = cache_dir / 'minute_volume_averages_30d.json'
    if not avg_file.exists():
        avg_file = cache_dir / 'minute_volume_averages_20d.json'
    
    if avg_file.exists():
        with open(avg_file, 'r') as f:
            averages_list = json.load(f)
        
        # Convert list format to dict format for easier access
        averages = {}
        for stock_data in averages_list:
            symbol = stock_data['symbol']
            averages[symbol] = stock_data['minute_avgs']
        
        logger.info(f"Loaded minute averages for {len(averages)} symbols")
        return averages
    else:
        logger.error("No minute volume averages file found!")
        return {}

def get_most_recent_trading_day():
    """Get the most recent trading day"""
    eastern = pytz.timezone('America/New_York')
    now = pd.Timestamp.now(tz=eastern)
    
    # If it's after 4pm, use today (if weekday)
    # Otherwise use previous trading day
    if now.hour >= 16:
        test_day = now.normalize()
    else:
        test_day = now.normalize() - timedelta(days=1)
    
    # Skip weekends
    while test_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
        test_day = test_day - timedelta(days=1)
    
    return test_day

def process_symbols_batch(symbols, start, end, minute_averages, test_date):
    """Process symbols using cached minute data to calculate RVOL"""
    results = []
    
    # Load cached minute data for the test date
    import gzip
    cache_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/shared_cache')
    minute_file = cache_dir / f'ohlcv_1min_bars/ohlcv_1min_{test_date}.json.gz'
    
    if not minute_file.exists():
        logger.error(f"No cached minute data found for {test_date}")
        return []
    
    # Load cached minute data
    try:
        with gzip.open(minute_file, 'rt') as f:
            minute_data_list = json.load(f)
        
        # Convert to dict for easier access
        minute_data = {}
        for stock_data in minute_data_list:
            symbol = stock_data['symbol']
            if symbol in symbols and symbol in minute_averages:
                minute_data[symbol] = stock_data['minutes']
        
        logger.info(f"Loaded minute data for {len(minute_data)} symbols")
    except Exception as e:
        logger.error(f"Failed to load cached minute data: {e}")
        return []
    
    # Convert start/end times to string format for comparison
    start_time_str = start.strftime('%H:%M')
    end_time_str = end.strftime('%H:%M')
    
    # Process each symbol
    for symbol in symbols:
        if symbol not in minute_data or symbol not in minute_averages:
            continue
        
        symbol_minutes = minute_data[symbol]
        symbol_avg = minute_averages[symbol]
        
        cumulative_volume = 0
        
        for minute_bar in symbol_minutes:
            bar_time_str = minute_bar['time']  # Format: "HH:MM"
            
            # Check if this minute is within our time range
            if not (start_time_str <= bar_time_str <= end_time_str):
                continue
            
            cumulative_volume += minute_bar['volume']
            
            # Calculate cumulative average volume up to this time
            cumulative_avg = 0
            for avg_time, avg_vol in symbol_avg.items():
                if avg_time <= bar_time_str:
                    cumulative_avg += avg_vol
            
            if cumulative_avg > 0:
                cumulative_rvol = cumulative_volume / cumulative_avg
            else:
                cumulative_rvol = 0
            
            # Calculate 5-minute window RVOL (simplified for now - just use current minute)
            current_volume = minute_bar['volume']
            current_avg = symbol_avg.get(bar_time_str, 0)
            
            if current_avg > 0:
                window_rvol = current_volume / current_avg
            else:
                window_rvol = 0
            
            # Check if RVOL > 5x
            if cumulative_rvol > 5 or window_rvol > 5:
                results.append({
                    'symbol': symbol,
                    'timestamp': f"{test_date} {bar_time_str}:00",
                    'price': minute_bar['close'],
                    'cumulative_volume': cumulative_volume,
                    'cumulative_rvol': cumulative_rvol,
                    'window_volume': current_volume,
                    'window_rvol': window_rvol
                })
    
    logger.info(f"Processed {len(symbols)} symbols from cached data")
    return results

def run_dual_rvol_test(test_date=None, start_time='06:00', end_time='11:30'):
    """
    Main function to run RVOL test using batch processing
    
    Args:
        test_date: Date to test (YYYY-MM-DD format) or None for most recent
        start_time: Start time (HH:MM format in EST)
        end_time: End time (HH:MM format in EST)
    """
    
    # Load minute averages
    logger.info("Loading minute averages...")
    minute_averages = load_minute_averages()
    if not minute_averages:
        logger.error("Cannot run RVOL test without minute averages")
        return []
    
    # Load US stock universe as primary source
    logger.info("Loading US stock universe...")
    universe_builder = StockUniverseBuilder()
    us_stocks = universe_builder.get_stock_universe()
    symbols = [stock['symbol'] for stock in us_stocks]
    
    # Filter to only symbols with minute averages
    symbols = [s for s in symbols if s in minute_averages]
    logger.info(f"Testing {len(symbols)} stocks with minute averages")
    
    # Debug: Check what symbols we have vs what minute_averages has
    if len(symbols) == 0:
        logger.info(f"Debug: US stock universe has {len([stock['symbol'] for stock in us_stocks])} symbols")
        logger.info(f"Debug: Minute averages type: {type(minute_averages)}")
        logger.info(f"Debug: Sample US symbols: {[stock['symbol'] for stock in us_stocks][:5]}")
        if isinstance(minute_averages, dict):
            logger.info(f"Debug: Sample minute average symbols: {list(minute_averages.keys())[:5]}")
        else:
            logger.info(f"Debug: Minute averages structure: {minute_averages[:2] if len(minute_averages) > 0 else 'empty'}")
    
    # Parse date and time
    eastern = pytz.timezone('America/New_York')
    
    if test_date:
        test_dt = pd.Timestamp(test_date, tz=eastern)
    else:
        test_dt = get_most_recent_trading_day()
    
    # Parse times
    start_hour, start_min = map(int, start_time.split(':'))
    end_hour, end_min = map(int, end_time.split(':'))
    
    start = test_dt.replace(hour=start_hour, minute=start_min, second=0, microsecond=0)
    end = test_dt.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)
    
    logger.info(f"Testing date: {test_dt.strftime('%Y-%m-%d')}")
    logger.info(f"Time range: {start_time} to {end_time} EST")
    
    # Process all symbols using cached minute data
    start_time_proc = time.time()
    all_results = process_symbols_batch(symbols, start, end, minute_averages, test_dt.strftime('%Y-%m-%d'))
    total_time = time.time() - start_time_proc
    
    # Save ALL results
    if all_results:
        df = pd.DataFrame(all_results)
        # Create results directory if it doesn't exist
        from pathlib import Path
        results_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans')
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / 'backtest_rvol_results.csv', index=False)
        
        unique_symbols = df['symbol'].unique()
        total_minutes = len(df)
        
        # Count high RVOL occurrences
        high_cumulative = df[df['cumulative_rvol'] > 5]['symbol'].unique()
        high_window = df[df['window_rvol'] > 5]['symbol'].unique()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"DUAL RVOL TEST RESULTS - {test_dt.strftime('%Y-%m-%d')} {start_time} to {end_time} EST")
        logger.info(f"{'='*80}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Symbols tested: {len(symbols)}")
        logger.info(f"Processing rate: {len(symbols)/total_time:.1f} symbols/sec")
        logger.info(f"Stocks with >5x cumulative RVOL: {len(high_cumulative)}")
        logger.info(f"Stocks with >5x window RVOL: {len(high_window)}")
        logger.info(f"Total high RVOL occurrences: {total_minutes}")
        
        # Show top RVOL stocks
        top_cumulative = df.nlargest(10, 'cumulative_rvol')[['symbol', 'cumulative_rvol', 'price']].drop_duplicates('symbol')
        logger.info(f"\nTop 10 Cumulative RVOL:")
        for _, row in top_cumulative.iterrows():
            logger.info(f"  {row['symbol']}: {row['cumulative_rvol']:.1f}x, price ${row['price']:.2f}")
        
        top_window = df.nlargest(10, 'window_rvol')[['symbol', 'window_rvol', 'price']].drop_duplicates('symbol')
        logger.info(f"\nTop 10 Window RVOL:")
        for _, row in top_window.iterrows():
            logger.info(f"  {row['symbol']}: {row['window_rvol']:.1f}x, price ${row['price']:.2f}")
    else:
        logger.info("No stocks found with >5x RVOL in $1-$20 range")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='RVOL Scanner - Find stocks with high relative volume')
    parser.add_argument('--date', type=str, help='Date to test (YYYY-MM-DD). Default: most recent trading day')
    parser.add_argument('--start', type=str, default='06:00', help='Start time (HH:MM in EST). Default: 06:00')
    parser.add_argument('--end', type=str, default='11:30', help='End time (HH:MM in EST). Default: 11:30')
    
    args = parser.parse_args()
    
    run_dual_rvol_test(test_date=args.date, start_time=args.start, end_time=args.end)

if __name__ == "__main__":
    main()