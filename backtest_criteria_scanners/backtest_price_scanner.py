#!/usr/bin/env python3
"""
Price Window Scanner with Batch API Calls
Processes ALL stocks quickly using batch Alpaca API calls
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
import argparse
from datetime import datetime, timedelta

load_dotenv('/Users/claytonsmacbookpro/Projects/warrior_bt/.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Alpaca API
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url='https://paper-api.alpaca.markets'
)

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

def process_symbols_batch(symbols, start, end):
    """Process symbols in batches to get price data"""
    results = []
    batch_size = 100  # Alpaca allows up to 100 symbols per request
    
    # Process in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batch_str = ','.join(batch)
        
        try:
            # Get bars for the batch
            bars = api.get_bars(
                batch_str,
                TimeFrame.Minute,
                start=start.isoformat(),
                end=end.isoformat(),
                feed='sip',
                limit=10000  # Large limit to get all bars in range
            )
            
            # Process bars
            for bar in bars:
                # Check if price is in $1-$20 window (low price) or $20-$50 (mid price)
                if 1.0 <= bar.c <= 50.0:
                    price_category = 'LOW' if bar.c <= 20.0 else 'MID'
                    results.append({
                        'symbol': bar.S,
                        'timestamp': bar.t,
                        'open': bar.o,
                        'high': bar.h,
                        'low': bar.l,
                        'close': bar.c,
                        'volume': bar.v,
                        'trade_count': bar.n if hasattr(bar, 'n') else 0,
                        'vwap': bar.vw if hasattr(bar, 'vw') else bar.c,
                        'price_category': price_category
                    })
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
            
        except Exception as e:
            logger.warning(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    
    return results

def run_fast_price_window_test(test_date=None, start_time='06:00', end_time='11:30'):
    """
    Main function to run price window test using batch processing
    
    Args:
        test_date: Date to test (YYYY-MM-DD format) or None for most recent
        start_time: Start time (HH:MM format in EST)
        end_time: End time (HH:MM format in EST)
    """
    
    # Load US stock universe as primary source
    logger.info("Loading US stock universe...")
    universe_builder = StockUniverseBuilder()
    us_stocks = universe_builder.get_stock_universe()
    symbols = [stock['symbol'] for stock in us_stocks]
    
    logger.info(f"Using full US stock universe: {len(symbols)} stocks")
    logger.info(f"Testing {len(symbols)} stocks with batch processing...")
    
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
    
    # Process all symbols using batch API calls
    start_time = time.time()
    all_results = process_symbols_batch(symbols, start, end)
    total_time = time.time() - start_time
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        # Create results directory if it doesn't exist
        from pathlib import Path
        results_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans')
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / 'backtest_price_results.csv', index=False)
        
        unique_symbols = df['symbol'].unique()
        total_minutes = len(df)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"PRICE WINDOW TEST RESULTS - {test_dt.strftime('%Y-%m-%d')} {start_time} to {end_time} EST")
        logger.info(f"{'='*80}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Symbols tested: {len(symbols)}")
        logger.info(f"Processing rate: {len(symbols)/total_time:.1f} symbols/sec")
        low_price_symbols = df[df['close'] <= 20]['symbol'].unique()
        mid_price_symbols = df[(df['close'] > 20) & (df['close'] <= 50)]['symbol'].unique()
        logger.info(f"Stocks in $1-$20 window (LOW): {len(low_price_symbols)}")
        logger.info(f"Stocks in $20-$50 window (MID): {len(mid_price_symbols)}")
        logger.info(f"Total stocks in price window: {len(unique_symbols)}")
        logger.info(f"Total minute bars: {total_minutes}")
        
        # Show sample of stocks
        sample_stocks = unique_symbols[:20]
        logger.info(f"\nSample stocks in window: {', '.join(sample_stocks)}")
    else:
        logger.info("No stocks found in $1-$20 price window")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Price Window Scanner - Find stocks in $1-$20 range')
    parser.add_argument('--date', type=str, help='Date to test (YYYY-MM-DD). Default: most recent trading day')
    parser.add_argument('--start', type=str, default='06:00', help='Start time (HH:MM in EST). Default: 06:00')
    parser.add_argument('--end', type=str, default='11:30', help='End time (HH:MM in EST). Default: 11:30')
    
    args = parser.parse_args()
    
    run_fast_price_window_test(test_date=args.date, start_time=args.start, end_time=args.end)

if __name__ == "__main__":
    main()