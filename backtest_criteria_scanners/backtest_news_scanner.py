#!/usr/bin/env python3
"""
Backtest News Scanner
Scans all US stocks for news catalysts within specified time window
Works as an independent scanner (not a filter)
"""

import requests
import os
from dotenv import load_dotenv
import pandas as pd
import pytz
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from foundational_stock_screeners.stock_universe_builder import StockUniverseBuilder
from datetime import datetime, timedelta
import json
import argparse
import time
from time import sleep

load_dotenv('/Users/claytonsmacbookpro/Projects/warrior_bt/.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestNewsScanner:
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = "https://data.alpaca.markets/v1beta1"
        
        self.headers = {
            "accept": "application/json",
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key
        }
    
    def get_news_for_symbols(self, symbols, start_time, end_time):
        """Get news for a batch of symbols with pagination support"""
        try:
            # Alpaca limits symbols per request, so batch them
            batch_size = 50
            all_news = []
            total_pages = 0
            
            total_batches = (len(symbols) + batch_size - 1) // batch_size
            
            for i in range(0, len(symbols), batch_size):
                batch_num = (i // batch_size) + 1
                batch = symbols[i:i+batch_size]
                symbols_str = ','.join(batch)
                
                # Log progress every 10 batches
                if batch_num % 10 == 0 or batch_num == 1:
                    logger.info(f"Processing batch {batch_num}/{total_batches} ({batch_num/total_batches*100:.1f}%)")
                
                # Get all pages for this batch
                page_token = None
                batch_articles = 0
                
                while True:
                    params = {
                        'symbols': symbols_str,
                        'start': start_time,
                        'end': end_time,
                        'limit': 50,
                        'sort': 'desc',
                        'exclude_contentless': True
                    }
                    
                    # Add page token if we have one (for pagination)
                    if page_token:
                        params['page_token'] = page_token
                    
                    # Retry logic for API calls
                    max_retries = 3
                    retry_count = 0
                    response = None
                    
                    while retry_count < max_retries:
                        try:
                            response = requests.get(
                                f"{self.base_url}/news",
                                headers=self.headers,
                                params=params,
                                timeout=10
                            )
                            
                            if response.status_code == 429:  # Rate limit
                                retry_after = int(response.headers.get('Retry-After', 1))
                                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                                sleep(retry_after)
                                retry_count += 1
                                continue
                            elif response.status_code >= 500:  # Server error
                                logger.warning(f"Server error {response.status_code}. Retrying...")
                                sleep(2 ** retry_count)  # Exponential backoff
                                retry_count += 1
                                continue
                            else:
                                break  # Success or client error
                                
                        except requests.exceptions.Timeout:
                            logger.warning(f"Request timeout. Retry {retry_count + 1}/{max_retries}")
                            retry_count += 1
                            sleep(2)
                        except requests.exceptions.ConnectionError:
                            logger.warning(f"Connection error. Retry {retry_count + 1}/{max_retries}")
                            retry_count += 1
                            sleep(2)
                    
                    if response and response.status_code == 200:
                        data = response.json()
                        
                        if 'news' in data and data['news']:
                            all_news.extend(data['news'])
                            batch_articles += len(data['news'])
                        
                        # Check if there's a next page
                        if 'next_page_token' in data and data['next_page_token']:
                            page_token = data['next_page_token']
                            total_pages += 1
                            # Limit pages per batch to avoid infinite loops
                            if total_pages > 10:
                                logger.warning(f"Reached page limit for batch {i//batch_size + 1}")
                                break
                        else:
                            # No more pages for this batch
                            break
                    else:
                        logger.debug(f"Error fetching news (batch {i//batch_size + 1}): {response.status_code}")
                        break
                
                if batch_articles > 0:
                    logger.debug(f"Batch {i//batch_size + 1}: Found {batch_articles} articles")
            
            logger.info(f"Total news articles retrieved: {len(all_news)}")
            return all_news
            
        except Exception as e:
            logger.error(f"Error getting news: {e}")
            return []

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

def run_news_scan(test_date=None, end_time="11:30"):
    """
    Scan all US stocks for news catalysts in the 24 hours before end_time
    
    Args:
        test_date: Date to test (YYYY-MM-DD format) or None for most recent
        end_time: End time (HH:MM format in EST) - default 11:30 AM
                  News window will be 24 hours before this time
    """
    
    scanner = BacktestNewsScanner()
    
    # Load US stock universe as primary source
    logger.info("Loading US stock universe...")
    universe_builder = StockUniverseBuilder()
    us_stocks = universe_builder.get_stock_universe()
    symbols = [stock['symbol'] for stock in us_stocks]
    
    logger.info(f"Using full US stock universe: {len(symbols)} stocks")
    logger.info(f"Scanning {len(symbols)} US stocks for news...")
    
    # Parse date and time
    eastern = pytz.timezone('America/New_York')
    
    if test_date:
        test_dt = pd.Timestamp(test_date, tz=eastern)
    else:
        test_dt = get_most_recent_trading_day()
    
    # Parse times and create window
    end_hour, end_min = map(int, end_time.split(':'))
    
    # End: Test day at end_time (typically 11:30 AM)
    end_datetime = test_dt.replace(hour=end_hour, minute=end_min, second=0, microsecond=0)
    
    # Start: Exactly 24 hours before the end time
    # This ensures we capture all news catalysts from the last 24 hours
    start_datetime = end_datetime - timedelta(hours=24)
    
    # Convert to ISO format for API
    start_datetime_str = start_datetime.isoformat()
    end_datetime_str = end_datetime.isoformat()
    
    logger.info(f"Testing date: {test_dt.strftime('%Y-%m-%d')}")
    logger.info(f"News window (24 hours): {start_datetime_str} to {end_datetime_str}")
    
    # Get news for all symbols
    start_time_proc = time.time()
    all_news = scanner.get_news_for_symbols(symbols, start_datetime_str, end_datetime_str)
    
    # Process results
    news_by_symbol = {}
    news_counts = {}
    
    for article in all_news:
        if 'symbols' in article:
            for symbol in article['symbols']:
                if symbol in symbols:  # Only count if in our universe
                    if symbol not in news_by_symbol:
                        news_by_symbol[symbol] = []
                        news_counts[symbol] = 0
                    
                    news_by_symbol[symbol].append({
                        'headline': article.get('headline', ''),
                        'summary': article.get('summary', ''),
                        'author': article.get('author', ''),
                        'created_at': article.get('created_at', ''),
                        'updated_at': article.get('updated_at', ''),
                        'url': article.get('url', '')
                    })
                    news_counts[symbol] += 1
    
    # Calculate processing time
    total_time = time.time() - start_time_proc
    
    # Prepare results
    results = {
        'test_date': test_dt.strftime('%Y-%m-%d'),
        'news_window_start': start_datetime_str,
        'news_window_end': end_datetime_str,
        'news_window_hours': 24,  # Always 24 hours
        'total_stocks_scanned': len(symbols),
        'stocks_with_news': len(news_by_symbol),
        'total_articles': len(all_news),
        'stocks_with_news_list': list(news_by_symbol.keys()),
        'news_counts': news_counts,
        'news_by_symbol': news_by_symbol,
        'processing_time': total_time
    }
    
    # Save to cache file for aggregate scanner
    cache_file = '/Users/claytonsmacbookpro/Projects/warrior_bt/results/news_scans/backtest_news_results.json'
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved news scan results to {cache_file}")
    
    # Save CSV for analysis
    if news_by_symbol:
        news_data = []
        for symbol, count in news_counts.items():
            news_data.append({
                'symbol': symbol,
                'news_count': count,
                'first_headline': news_by_symbol[symbol][0]['headline'] if news_by_symbol[symbol] else ''
            })
        
        df = pd.DataFrame(news_data)
        # Create results directory if it doesn't exist
        from pathlib import Path
        results_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/results/criteria_scans')
        results_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(results_dir / 'backtest_news_results.csv', index=False)
        logger.info(f"Saved {len(df)} stocks with news to CSV")
    
    # Display summary
    print(f"\n{'='*80}")
    print(f"BACKTEST NEWS SCANNER RESULTS")
    print(f"Date: {test_dt.strftime('%Y-%m-%d')} | End: {end_time} EST | Window: 24 hours")
    print(f"{'='*80}")
    print(f"Total stocks scanned: {len(symbols)}")
    print(f"Stocks with news: {len(news_by_symbol)} ({len(news_by_symbol)/len(symbols)*100:.1f}%)")
    print(f"Total articles found: {len(all_news)}")
    print(f"Processing time: {total_time:.1f} seconds")
    
    if news_by_symbol:
        # Sort by news count
        sorted_symbols = sorted(news_counts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 20 stocks by news count:")
        print(f"{'Symbol':<8} {'Articles':<10} {'First Headline':<60}")
        print("-" * 80)
        
        for symbol, count in sorted_symbols[:20]:
            headline = news_by_symbol[symbol][0]['headline'][:57] + '...' if len(news_by_symbol[symbol][0]['headline']) > 60 else news_by_symbol[symbol][0]['headline']
            print(f"{symbol:<8} {count:<10} {headline:<60}")
    
    return results

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Backtest news scanner for all US stocks (24-hour window)')
    parser.add_argument('--date', type=str, help='Test date (YYYY-MM-DD format)')
    parser.add_argument('--end', type=str, default='11:30', help='End time (HH:MM in EST, default 11:30). News window will be 24 hours before this time.')
    args = parser.parse_args()
    
    # Run the scan
    run_news_scan(test_date=args.date, end_time=args.end)

if __name__ == "__main__":
    main()