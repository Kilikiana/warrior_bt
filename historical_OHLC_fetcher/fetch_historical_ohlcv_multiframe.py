#!/usr/bin/env python3
"""
Fetch Historical OHLCV Data - Multiple Timeframes
Fetches 1-minute, 5-minute, and daily bars for the full trading day
Stores in separate folders maintaining the same structure as existing 1-min fetcher
"""

from alpaca_trade_api import REST, TimeFrame
import os
from dotenv import load_dotenv
import pandas as pd
import pytz
import logging
import sys
import os
from foundational_stock_screeners.float_scanner import FloatScanner
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import gzip

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Alpaca API
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url="https://paper-api.alpaca.markets"
)

def fetch_minute_volume_for_symbol_date(symbol, date):
    """Fetch 1-minute volume data for a symbol on a specific date with 3 market sessions"""
    try:
        eastern = pytz.timezone('America/New_York')
        date_str = date.strftime('%Y-%m-%d')
        
        # Full extended trading day: 4:00 AM - 8:00 PM
        start = pd.Timestamp(f'{date_str} 04:00:00', tz=eastern)
        end = pd.Timestamp(f'{date_str} 20:00:00', tz=eastern)
        
        bars = api.get_bars(
            symbol,
            TimeFrame.Minute,
            start=start.isoformat(),
            end=end.isoformat(),
            feed='sip',
            limit=None
        )
        
        # Separate into 3 market sessions
        premarket = []   # 04:00 - 09:30
        regular = []     # 09:30 - 16:00 
        afterhours = []  # 16:00 - 20:00
        
        for bar in bars:
            bar_time = bar.t.strftime('%H:%M')
            bar_data = {
                'time': bar_time,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v,
                'trades': bar.n,
                'vwap': bar.vw
            }
            
            # Classify by trading session
            hour_min = bar.t.strftime('%H%M')
            if '0400' <= hour_min < '0930':
                premarket.append(bar_data)
            elif '0930' <= hour_min < '1600':
                regular.append(bar_data)
            elif '1600' <= hour_min <= '2000':
                afterhours.append(bar_data)
        
        # Calculate session statistics
        premarket_volume = sum(b['volume'] for b in premarket)
        regular_volume = sum(b['volume'] for b in regular)
        afterhours_volume = sum(b['volume'] for b in afterhours)
        
        if premarket or regular or afterhours:
            return {
                'symbol': symbol,
                'date': date_str,
                'sessions': {
                    'premarket': premarket,
                    'regular': regular,
                    'afterhours': afterhours
                },
                'stats': {
                    'premarket_bars': len(premarket),
                    'regular_bars': len(regular),
                    'afterhours_bars': len(afterhours),
                    'total_bars': len(premarket) + len(regular) + len(afterhours),
                    'premarket_volume': premarket_volume,
                    'regular_volume': regular_volume,
                    'afterhours_volume': afterhours_volume
                }
            }
        return None
        
    except Exception as e:
        logger.debug(f"Error fetching 1-min {symbol} for {date}: {e}")
        return None

def fetch_5min_volume_for_symbol_date(symbol, date):
    """Fetch 5-minute volume data for a symbol on a specific date with 3 market sessions"""
    try:
        eastern = pytz.timezone('America/New_York')
        date_str = date.strftime('%Y-%m-%d')
        
        # Full extended trading day: 4:00 AM - 8:00 PM
        start = pd.Timestamp(f'{date_str} 04:00:00', tz=eastern)
        end = pd.Timestamp(f'{date_str} 20:00:00', tz=eastern)
        
        # Use 5-minute timeframe
        bars = api.get_bars(
            symbol,
            '5Min',  # 5-minute bars - using string format
            start=start.isoformat(),
            end=end.isoformat(),
            feed='sip',
            limit=None
        )
        
        # Separate into 3 market sessions
        premarket = []   # 04:00 - 09:30
        regular = []     # 09:30 - 16:00 
        afterhours = []  # 16:00 - 20:00
        
        for bar in bars:
            bar_time = bar.t.strftime('%H:%M')
            bar_data = {
                'time': bar_time,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v,
                'trades': bar.n,
                'vwap': bar.vw
            }
            
            # Classify by trading session
            hour_min = bar.t.strftime('%H%M')
            if '0400' <= hour_min < '0930':
                premarket.append(bar_data)
            elif '0930' <= hour_min < '1600':
                regular.append(bar_data)
            elif '1600' <= hour_min <= '2000':
                afterhours.append(bar_data)
        
        # Calculate session statistics
        premarket_volume = sum(b['volume'] for b in premarket)
        regular_volume = sum(b['volume'] for b in regular)
        afterhours_volume = sum(b['volume'] for b in afterhours)
        
        if premarket or regular or afterhours:
            return {
                'symbol': symbol,
                'date': date_str,
                'sessions': {
                    'premarket': premarket,
                    'regular': regular,
                    'afterhours': afterhours
                },
                'stats': {
                    'premarket_bars': len(premarket),
                    'regular_bars': len(regular),
                    'afterhours_bars': len(afterhours),
                    'total_bars': len(premarket) + len(regular) + len(afterhours),
                    'premarket_volume': premarket_volume,
                    'regular_volume': regular_volume,
                    'afterhours_volume': afterhours_volume
                }
            }
        return None
        
    except Exception as e:
        logger.debug(f"Error fetching 5-min {symbol} for {date}: {e}")
        return None

def fetch_daily_bars_for_symbol(symbol, days=50):
    """Fetch daily bars for a symbol"""
    try:
        eastern = pytz.timezone('America/New_York')
        end_date = pd.Timestamp.now(tz=eastern).normalize()
        start_date = end_date - timedelta(days=days*2)  # Extra buffer for weekends/holidays
        
        bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            feed='sip',
            limit=days
        )
        
        # Store each daily bar
        daily_data = []
        for bar in bars:
            daily_data.append({
                'date': bar.t.strftime('%Y-%m-%d'),
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v,
                'trades': bar.n,
                'vwap': bar.vw
            })
        
        if daily_data:
            return {
                'symbol': symbol,
                'bars': daily_data[-days:],  # Keep only requested number of days
                'total_bars': len(daily_data)
            }
        return None
        
    except Exception as e:
        logger.debug(f"Error fetching daily {symbol}: {e}")
        return None

def fetch_historical_multiframe(specific_date=None, fetch_1min=True, fetch_5min=True, fetch_daily=True):
    """Fetch historical OHLCV bars for all US stocks in multiple timeframes
    
    Args:
        specific_date: Optional specific date string (YYYY-MM-DD) to fetch
        fetch_1min: Whether to fetch 1-minute data
        fetch_5min: Whether to fetch 5-minute data  
        fetch_daily: Whether to fetch daily data
    """
    
    # Create cache directory structure for each timeframe
    cache_dirs = {
        '1min': Path('shared_cache/ohlcv_1min_bars'),
        '5min': Path('shared_cache/ohlcv_5min_bars'),
        'daily': Path('shared_cache/ohlcv_daily_bars')
    }
    
    for dir_path in cache_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load stocks from US stock universe ONLY
    logger.info("Loading US stock universe...")
    
    # Get US stock universe - ALL stocks, not filtered by float
    from foundational_stock_screeners.stock_universe_builder import StockUniverseBuilder
    universe_builder = StockUniverseBuilder()
    us_stocks = universe_builder.get_stock_universe()
    symbols = [stock['symbol'] for stock in us_stocks]
    
    logger.info(f"Found {len(symbols)} US stocks in universe")
    logger.info(f"Will fetch historical data for ALL stocks (not just low-float)")
    
    logger.info(f"Fetching multi-timeframe data for {len(symbols)} stocks...")
    
    # Generate list of trading days
    eastern = pytz.timezone('America/New_York')
    
    if specific_date:
        # Use specific date provided
        dates_to_fetch = [pd.Timestamp(specific_date, tz=eastern)]
        logger.info(f"Fetching data for specific date: {specific_date}")
    else:
        # Default: last 90 trading days for 1-min and 5-min
        end_date = pd.Timestamp.now(tz=eastern).normalize()
        
        # Use Alpaca calendar API to get actual trading days (excludes holidays)
        start_date = end_date - timedelta(days=150)  # Go back 150 days to ensure 90 trading days
        
        calendar = api.get_calendar(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d')
        )
        
        # Convert to pandas timestamps and get last 90 trading days
        dates_to_fetch = []
        for day in calendar:
            trading_day = pd.Timestamp(day.date, tz=eastern)
            if trading_day <= end_date:
                dates_to_fetch.append(trading_day)
        
        # Sort descending and take most recent 90, keep in reverse chronological order
        dates_to_fetch.sort(reverse=True)
        dates_to_fetch = dates_to_fetch[:90]
        # Keep in reverse chronological order (newest first) for processing
    
    logger.info(f"Processing {len(dates_to_fetch)} trading days for 1-min and 5-min data")
    
    # Process 1-minute and 5-minute data for each date
    for date in dates_to_fetch:
        date_str = date.strftime('%Y-%m-%d')
        
        # Skip today if market is still open or after-hours trading is active
        now_eastern = pd.Timestamp.now(tz=eastern)
        today = now_eastern.strftime('%Y-%m-%d')
        
        if date_str == today:
            # After-hours trading ends at 8:00 PM EST
            if now_eastern.hour < 20:  # Before 8 PM
                logger.info(f"Skipping {date_str} - today's data not complete (market still active until 8 PM EST)")
                continue
            else:
                logger.info(f"Including {date_str} - after-hours trading closed (after 8 PM EST)")
        
        # Check if files already exist
        min1_file = cache_dirs['1min'] / f'ohlcv_1min_{date_str}.json.gz'
        min5_file = cache_dirs['5min'] / f'ohlcv_5min_{date_str}.json.gz'
        
        # Skip 1-min if already cached (keeping existing data intact)
        if min1_file.exists():
            logger.info(f"Skipping 1-min {date_str} - already cached")
        else:
            logger.info(f"Fetching 1-minute data for {date_str}...")
            start_time = time.time()
            
            # Process all symbols for this date in parallel
            all_1min_results = []
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_minute_volume_for_symbol_date, symbol, date): symbol 
                          for symbol in symbols}
                
                completed = 0
                for future in as_completed(futures):
                    symbol = futures[future]
                    completed += 1
                    
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(symbols) - completed) / rate
                        logger.info(f"  1-min Progress: {completed}/{len(symbols)} ({rate:.1f} symbols/sec, ETA: {eta:.0f}s)")
                    
                    try:
                        result = future.result()
                        if result and result['stats']['premarket_volume'] + result['stats']['regular_volume'] + result['stats']['afterhours_volume'] > 0:
                            all_1min_results.append(result)
                    except Exception as e:
                        logger.debug(f"Error processing {symbol}: {e}")
            
            # Save compressed to save space (these files will be large)
            with gzip.open(min1_file, 'wt') as f:
                json.dump(all_1min_results, f)
            
            elapsed = time.time() - start_time
            logger.info(f"Completed 1-min {date_str} in {elapsed:.1f} seconds - {len(all_1min_results)} stocks with volume")
        
        # Process 5-minute data
        if min5_file.exists():
            logger.info(f"Skipping 5-min {date_str} - already cached")
        else:
            logger.info(f"Fetching 5-minute data for {date_str}...")
            start_time = time.time()
            
            all_5min_results = []
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(fetch_5min_volume_for_symbol_date, symbol, date): symbol 
                          for symbol in symbols}
                
                completed = 0
                for future in as_completed(futures):
                    symbol = futures[future]
                    completed += 1
                    
                    if completed % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed
                        eta = (len(symbols) - completed) / rate
                        logger.info(f"  5-min Progress: {completed}/{len(symbols)} ({rate:.1f} symbols/sec, ETA: {eta:.0f}s)")
                    
                    try:
                        result = future.result()
                        if result and result['stats']['premarket_volume'] + result['stats']['regular_volume'] + result['stats']['afterhours_volume'] > 0:
                            all_5min_results.append(result)
                    except Exception as e:
                        logger.debug(f"Error processing {symbol}: {e}")
            
            with gzip.open(min5_file, 'wt') as f:
                json.dump(all_5min_results, f)
            
            elapsed = time.time() - start_time
            logger.info(f"Completed 5-min {date_str} in {elapsed:.1f} seconds - {len(all_5min_results)} stocks with volume")
    
    # Process daily data (different structure - one file per symbol)
    logger.info("Fetching daily data for all symbols (50 days each)...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_daily_bars_for_symbol, symbol, 50): symbol 
                  for symbol in symbols}
        
        completed = 0
        for future in as_completed(futures):
            symbol = futures[future]
            completed += 1
            
            if completed % 100 == 0:
                elapsed = time.time() - start_time
                rate = completed / elapsed
                eta = (len(symbols) - completed) / rate
                logger.info(f"  Daily Progress: {completed}/{len(symbols)} ({rate:.1f} symbols/sec, ETA: {eta:.0f}s)")
            
            try:
                result = future.result()
                if result and result['total_bars'] > 0:
                    # Save each symbol's daily data to its own file
                    daily_file = cache_dirs['daily'] / f'daily_{symbol}.json'
                    with open(daily_file, 'w') as f:
                        json.dump(result, f)
            except Exception as e:
                logger.debug(f"Error processing daily {symbol}: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Completed daily data fetch in {elapsed:.1f} seconds")
    
    logger.info("Multi-timeframe historical data fetch complete!")
    
    # Display summary
    for timeframe, cache_dir in cache_dirs.items():
        if timeframe == 'daily':
            files = list(cache_dir.glob('daily_*.json'))
            logger.info(f"{timeframe}: {len(files)} symbol files")
        else:
            files = list(cache_dir.glob(f'ohlcv_{timeframe}_*.json.gz'))
            logger.info(f"{timeframe}: {len(files)} date files")

if __name__ == "__main__":
    # Fetch data for last 90 trading days with all 3 market sessions (1min, 5min, daily)
    fetch_historical_multiframe()