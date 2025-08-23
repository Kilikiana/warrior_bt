#!/usr/bin/env python3
"""
FIXED Gap Scanner with Batch API Calls and Fallback
Tests which stocks have >10% gap from previous day's close
Uses batch API calls for efficiency, with individual fallback for failed batches
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
from foundational_stock_screeners.float_scanner import FloatScanner
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

def process_single_symbol(symbol, start, end, prev_date):
    """Process a single symbol when batch processing fails"""
    results = []
    
    try:
        # Get previous close for this symbol
        prev_bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=prev_date.strftime('%Y-%m-%d'),
            end=prev_date.strftime('%Y-%m-%d'),
            feed='sip',
            limit=1,
            asof=prev_date.strftime('%Y-%m-%d')
        )
        
        prev_close = None
        for bar in prev_bars:
            prev_close = bar.c
            break
            
        if prev_close is None or prev_close <= 0:
            return results
            
        # Get current day bars - format dates properly for API
        start_str = start.strftime('%Y-%m-%dT%H:%M:%S-04:00')  # EST timezone
        end_str = end.strftime('%Y-%m-%dT%H:%M:%S-04:00')
        
        curr_bars = api.get_bars(
            symbol,
            TimeFrame.Minute,
            start=start_str,
            end=end_str,
            feed='sip',
            limit=10000
        )
        
        # Process bars to find gaps
        for bar in curr_bars:
            gap_pct = ((bar.c - prev_close) / prev_close) * 100
            
            # Check if gap > 10%
            if gap_pct > 10.0:
                results.append({
                    'symbol': symbol,
                    'timestamp': bar.t,
                    'prev_close': prev_close,
                    'price': bar.c,
                    'gap_pct': gap_pct,
                    'volume': bar.v
                })
                
    except Exception as e:
        # Individual symbol failed - this is ok, just skip it
        logger.debug(f"Skipping {symbol}: {e}")
        
    return results

def process_symbols_batch(symbols, start, end, test_date):
    """Process symbols in batches to get gap data - WITH FALLBACK"""
    results = []
    batch_size = 100  # Alpaca allows up to 100 symbols per request
    
    # Get previous trading day
    prev_date = test_date - timedelta(days=1)
    while prev_date.weekday() >= 5:  # Skip weekends
        prev_date = prev_date - timedelta(days=1)
    
    # Process in batches
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batch_str = ','.join(batch)
        batch_success = False
        
        try:
            # Get previous day's closing prices (daily bars)
            prev_bars = api.get_bars(
                batch_str,
                TimeFrame.Day,
                start=prev_date.strftime('%Y-%m-%d'),
                end=prev_date.strftime('%Y-%m-%d'),
                feed='sip',
                limit=1000,
                asof=prev_date.strftime('%Y-%m-%d')
            )
            
            # Store previous closes
            prev_closes = {}
            for bar in prev_bars:
                prev_closes[bar.S] = bar.c
            
            if not prev_closes:
                logger.warning(f"No previous closes found for batch {i//batch_size + 1}")
            else:
                # Get current day bars for the time range - format dates properly
                start_str = start.strftime('%Y-%m-%dT%H:%M:%S-04:00')  # EST timezone
                end_str = end.strftime('%Y-%m-%dT%H:%M:%S-04:00')
                
                curr_bars = api.get_bars(
                    batch_str,
                    TimeFrame.Minute,
                    start=start_str,
                    end=end_str,
                    feed='sip',
                    limit=10000
                )
                
                # Process bars to find gaps
                for bar in curr_bars:
                    symbol = bar.S
                    if symbol in prev_closes:
                        prev_close = prev_closes[symbol]
                        if prev_close > 0:
                            gap_pct = ((bar.c - prev_close) / prev_close) * 100
                            
                            # Check if gap > 10%
                            if gap_pct > 10.0:
                                results.append({
                                    'symbol': symbol,
                                    'timestamp': bar.t,
                                    'prev_close': prev_close,
                                    'price': bar.c,
                                    'gap_pct': gap_pct,
                                    'volume': bar.v
                                })
                
                batch_success = True
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size} successfully")
                
        except Exception as e:
            logger.warning(f"Batch {i//batch_size + 1} failed: {e}")
            logger.info(f"Processing batch {i//batch_size + 1} symbols individually...")
            
        # FALLBACK: Process failed batch symbols individually
        if not batch_success:
            individual_count = 0
            for symbol in batch:
                symbol_results = process_single_symbol(symbol, start, end, prev_date)
                if symbol_results:
                    results.extend(symbol_results)
                    individual_count += 1
            logger.info(f"  Recovered {individual_count} symbols from failed batch")
    
    return results

def run_gap_test(test_date=None, start_time='06:00', end_time='11:30'):
    """
    Main function to run gap test
    
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
    
    # Parse date and time
    eastern = pytz.timezone('America/New_York')
    
    if test_date:
        test_dt = pd.Timestamp(test_date, tz=eastern)
    else:
        test_dt = get_most_recent_trading_day()
    
    # Create start and end timestamps
    start_hour, start_minute = map(int, start_time.split(':'))
    end_hour, end_minute = map(int, end_time.split(':'))
    
    start = test_dt.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    end = test_dt.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
    
    logger.info(f"Testing from {start} to {end}")
    
    # Process all symbols
    all_results = process_symbols_batch(symbols, start, end, test_dt)
    
    # Consolidate results (keep highest gap for each symbol)
    consolidated = {}
    for result in all_results:
        symbol = result['symbol']
        if symbol not in consolidated or result['gap_pct'] > consolidated[symbol]['gap_pct']:
            consolidated[symbol] = result
    
    # Convert to list and sort by gap percentage
    final_results = list(consolidated.values())
    final_results.sort(key=lambda x: x['gap_pct'], reverse=True)
    
    logger.info(f"Found {len(final_results)} stocks with >10% gap")
    
    # Save results
    if final_results:
        df = pd.DataFrame(final_results)
        output_file = '/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans/backtest_gap_results_fixed.csv'
        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
        
        # Show top 10
        logger.info("\nTop 10 gaps:")
        for i, row in enumerate(final_results[:10], 1):
            logger.info(f"{i}. {row['symbol']}: {row['gap_pct']:.1f}% gap (${row['prev_close']:.2f} -> ${row['price']:.2f})")
    
    return final_results

def main():
    parser = argparse.ArgumentParser(description='Gap Scanner with fallback for failed batches')
    parser.add_argument('--date', type=str, help='Date to test (YYYY-MM-DD format)')
    parser.add_argument('--start', type=str, default='06:00', help='Start time (HH:MM format in EST)')
    parser.add_argument('--end', type=str, default='11:30', help='End time (HH:MM format in EST)')
    
    args = parser.parse_args()
    
    results = run_gap_test(
        test_date=args.date,
        start_time=args.start,
        end_time=args.end
    )
    
    return results

if __name__ == "__main__":
    main()