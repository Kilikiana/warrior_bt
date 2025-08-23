#!/usr/bin/env python3
"""
Float Scanner - Gets real float data for all stocks using Financial Modeling Prep
This scanner identifies low-float stocks that are ideal for Ross Cameron's gap trading strategy
"""

import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict, Optional
import pickle
from datetime import datetime, timedelta
import logging
from urllib.parse import urlencode
from .stock_universe_builder import StockUniverseBuilder

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FloatScanner:
    def __init__(self):
        """Initialize float scanner with FMP API"""
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        if not self.fmp_api_key:
            raise ValueError("FMP_API_KEY not found in environment variables!")
        
        self.base_url = "https://financialmodelingprep.com"
        
        # Use shared cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "shared_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.float_cache_file = os.path.join(self.cache_dir, "float_data_cache.pkl")
        self.cache_duration_hours = 24  # Refresh float data daily
        
        # Initialize stock universe builder for US stock filtering
        self.universe_builder = StockUniverseBuilder()
    
    def get_all_float_data(self, 
                          use_cache: bool = True,
                          limit: int = 5000) -> List[Dict]:
        """
        Get float data for all available stocks from FMP
        
        Args:
            use_cache: Whether to use cached data if fresh
            limit: Number of results per page (default 5000)
        
        Returns:
            List of dictionaries containing float data
        """
        
        # Check cache first
        if use_cache and self._is_cache_valid():
            logger.info("Using cached float data")
            return self._load_cache()
        
        logger.info("Fetching fresh float data from FMP (this may take a minute)...")
        
        all_float_data = []
        current_page = 0
        max_pages = 20  # Safety limit to prevent infinite loops
        
        while current_page < max_pages:
            # Build the URL with parameters
            params = {
                'apikey': self.fmp_api_key,
                'limit': limit,
                'page': current_page
            }
            
            url = f"{self.base_url}/stable/shares-float-all"
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data:
                    # No more data, we've fetched everything
                    logger.info(f"Completed fetching all pages (total pages: {current_page})")
                    break
                
                all_float_data.extend(data)
                logger.info(f"Fetched page {current_page} with {len(data)} records (Total: {len(all_float_data)})")
                
                # If we got less than limit, we've reached the end
                if len(data) < limit:
                    logger.info(f"Last page reached (got {len(data)} records, less than limit of {limit})")
                    break
                
                current_page += 1
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching float data from FMP: {e}")
                break
        
        if current_page >= max_pages:
            logger.warning(f"Hit max pages limit ({max_pages}). There may be more data available.")
        
        logger.info(f"Total float data fetched: {len(all_float_data)} stocks")
        
        # Save to cache
        if all_float_data:
            self._save_cache(all_float_data)
        
        return all_float_data
    
    def get_low_float_stocks(self, 
                           max_float: float = 20_000_000) -> List[Dict]:
        """
        Get all stocks with float under max_float
        Cross-referenced with US stock universe
        
        Args:
            max_float: Maximum float in shares (default: 20M)
        
        Returns:
            List of low-float stocks sorted by float (lowest first)
        """
        
        # Get all float data
        all_float_data = self.get_all_float_data()
        
        if not all_float_data:
            logger.warning("No float data available")
            return []
        
        # Get US stock universe for cross-reference
        logger.info("Loading US stock universe for cross-reference...")
        us_stocks = self.universe_builder.get_stock_universe()
        us_symbols = {stock['symbol'] for stock in us_stocks}
        logger.info(f"US universe contains {len(us_symbols)} stocks")
        
        # Filter for low-float stocks IN the US universe
        low_float_stocks = []
        skipped_non_us = 0
        
        for stock in all_float_data:
            symbol = stock.get('symbol')
            
            # Skip if not in US stock universe
            if symbol not in us_symbols:
                skipped_non_us += 1
                continue
            
            float_shares = stock.get('floatShares')
            free_float = stock.get('freeFloat')
            outstanding_shares = stock.get('outstandingShares')
            
            # Skip if no float data
            if not float_shares or float_shares <= 0:
                continue
            
            # Apply float filter
            if float_shares > max_float:
                continue
            
            # Calculate float percentage if we have outstanding shares
            float_percentage = None
            if outstanding_shares and outstanding_shares > 0:
                float_percentage = (float_shares / outstanding_shares) * 100
            
            low_float_stocks.append({
                'symbol': symbol,
                'floatShares': float_shares,
                'freeFloat': free_float,
                'outstandingShares': outstanding_shares,
                'floatPercentage': float_percentage,
                'floatMillions': float_shares / 1_000_000
            })
        
        # Sort by float (lowest first)
        low_float_stocks.sort(key=lambda x: x['floatShares'])
        
        logger.info(f"Filtered out {skipped_non_us} non-US stocks")
        logger.info(f"Found {len(low_float_stocks)} US stocks with float < {max_float/1_000_000:.0f}M shares")
        
        # Save low-float stocks to JSON for other scanners
        import json
        from pathlib import Path
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        with open(cache_dir / 'low_float_stocks.json', 'w') as f:
            json.dump(low_float_stocks, f, indent=2)
        logger.info(f"Saved {len(low_float_stocks)} low-float US stocks to shared_cache/low_float_stocks.json")
        
        return low_float_stocks
    
    def generate_categorized_float_caches(self):
        """
        Generate separate cache files for different float categories
        All cross-referenced with US stock universe
        
        Categories:
        - Low float: < 20M shares
        - Mid float: 20-50M shares  
        - High float: > 50M shares
        """
        
        # Get all float data
        all_float_data = self.get_all_float_data()
        
        if not all_float_data:
            logger.warning("No float data available")
            return
        
        # Get US stock universe for cross-reference
        logger.info("Loading US stock universe for cross-reference...")
        us_stocks = self.universe_builder.get_stock_universe()
        us_symbols = {stock['symbol'] for stock in us_stocks}
        logger.info(f"US universe contains {len(us_symbols)} stocks")
        
        # Initialize categories
        low_float_stocks = []   # < 20M
        mid_float_stocks = []   # 20-50M
        high_float_stocks = []  # > 50M
        
        skipped_non_us = 0
        skipped_no_data = 0
        
        for stock in all_float_data:
            symbol = stock.get('symbol')
            
            # Skip if not in US stock universe
            if symbol not in us_symbols:
                skipped_non_us += 1
                continue
            
            float_shares = stock.get('floatShares')
            free_float = stock.get('freeFloat')
            outstanding_shares = stock.get('outstandingShares')
            
            # Skip if no float data
            if not float_shares or float_shares <= 0:
                skipped_no_data += 1
                continue
            
            # Calculate float percentage if we have outstanding shares
            float_percentage = None
            if outstanding_shares and outstanding_shares > 0:
                float_percentage = (float_shares / outstanding_shares) * 100
            
            stock_data = {
                'symbol': symbol,
                'floatShares': float_shares,
                'freeFloat': free_float,
                'outstandingShares': outstanding_shares,
                'floatPercentage': float_percentage,
                'floatMillions': float_shares / 1_000_000
            }
            
            # Categorize by float
            if float_shares < 20_000_000:
                low_float_stocks.append(stock_data)
            elif float_shares <= 50_000_000:
                mid_float_stocks.append(stock_data)
            else:
                high_float_stocks.append(stock_data)
        
        # Sort each category by float (lowest first)
        low_float_stocks.sort(key=lambda x: x['floatShares'])
        mid_float_stocks.sort(key=lambda x: x['floatShares'])
        high_float_stocks.sort(key=lambda x: x['floatShares'])
        
        # Save each category to separate JSON files
        import json
        from pathlib import Path
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(exist_ok=True)
        
        # Save low float
        with open(cache_dir / 'low_float_stocks.json', 'w') as f:
            json.dump(low_float_stocks, f, indent=2)
        logger.info(f"Saved {len(low_float_stocks)} low-float US stocks (<20M) to shared_cache/low_float_stocks.json")
        
        # Save mid float
        with open(cache_dir / 'mid_float_stocks.json', 'w') as f:
            json.dump(mid_float_stocks, f, indent=2)
        logger.info(f"Saved {len(mid_float_stocks)} mid-float US stocks (20-50M) to shared_cache/mid_float_stocks.json")
        
        # Save high float
        with open(cache_dir / 'high_float_stocks.json', 'w') as f:
            json.dump(high_float_stocks, f, indent=2)
        logger.info(f"Saved {len(high_float_stocks)} high-float US stocks (>50M) to shared_cache/high_float_stocks.json")
        
        # Print summary
        print("\n" + "="*70)
        print("📊 FLOAT CATEGORIZATION COMPLETE")
        print("="*70)
        print(f"Filtered out {skipped_non_us} non-US stocks")
        print(f"Skipped {skipped_no_data} stocks with no float data")
        print(f"\nUS Stocks by Float Category:")
        print(f"  Low Float (<20M):   {len(low_float_stocks):,} stocks")
        print(f"  Mid Float (20-50M): {len(mid_float_stocks):,} stocks")
        print(f"  High Float (>50M):  {len(high_float_stocks):,} stocks")
        print(f"  TOTAL:              {len(low_float_stocks) + len(mid_float_stocks) + len(high_float_stocks):,} stocks")
        
        # Show examples from each category
        print("\n📌 Example Stocks from Each Category:")
        
        if low_float_stocks:
            print("\nLow Float (<20M):")
            for stock in low_float_stocks[:5]:
                print(f"  {stock['symbol']:6} - {stock['floatMillions']:7.2f}M shares")
        
        if mid_float_stocks:
            print("\nMid Float (20-50M):")
            for stock in mid_float_stocks[:5]:
                print(f"  {stock['symbol']:6} - {stock['floatMillions']:7.2f}M shares")
        
        if high_float_stocks:
            print("\nHigh Float (>50M):")
            for stock in high_float_stocks[:5]:
                print(f"  {stock['symbol']:6} - {stock['floatMillions']:7.2f}M shares")
        
        return {
            'low': low_float_stocks,
            'mid': mid_float_stocks,
            'high': high_float_stocks
        }
    
    def get_float_for_symbols(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get float data for specific symbols
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary mapping symbol to float data
        """
        
        # Get all float data
        all_float_data = self.get_all_float_data()
        
        # Create a lookup dictionary
        float_lookup = {}
        for stock in all_float_data:
            symbol = stock.get('symbol')
            if symbol in symbols:
                float_lookup[symbol] = {
                    'floatShares': stock.get('floatShares'),
                    'freeFloat': stock.get('freeFloat'),
                    'outstandingShares': stock.get('outstandingShares'),
                    'floatMillions': stock.get('floatShares', 0) / 1_000_000 if stock.get('floatShares') else None
                }
        
        return float_lookup
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is fresh"""
        if not os.path.exists(self.float_cache_file):
            return False
        
        # Check age of cache
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.float_cache_file))
        return cache_age < timedelta(hours=self.cache_duration_hours)
    
    def _save_cache(self, data: List[Dict]):
        """Save data to cache file"""
        # Save as pickle for backwards compatibility
        with open(self.float_cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save as JSON for readability
        json_file = self.float_cache_file.replace('.pkl', '.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} float records to cache (both .pkl and .json)")
    
    def _load_cache(self) -> List[Dict]:
        """Load data from cache file"""
        with open(self.float_cache_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} float records from cache")
        return data
    
    def display_low_float_stats(self):
        """Display statistics about low-float stocks"""
        
        # Get low-float stocks
        low_float_stocks = self.get_low_float_stocks()
        
        if not low_float_stocks:
            print("No low-float US stocks found!")
            return
        
        # Calculate statistics
        float_ranges = {
            '0-5M': [],
            '5-10M': [],
            '10-15M': [],
            '15-20M': []
        }
        
        for stock in low_float_stocks:
            float_millions = stock['floatMillions']
            if float_millions <= 5:
                float_ranges['0-5M'].append(stock)
            elif float_millions <= 10:
                float_ranges['5-10M'].append(stock)
            elif float_millions <= 15:
                float_ranges['10-15M'].append(stock)
            else:
                float_ranges['15-20M'].append(stock)
        
        # Display results
        print("=" * 60)
        print("🎯 US LOW-FLOAT STOCKS FOR ROSS CAMERON STRATEGY")
        print("=" * 60)
        print(f"Total US Low-Float Stocks (<20M shares): {len(low_float_stocks)}")
        
        print("\nFloat Distribution:")
        for range_name, stocks in float_ranges.items():
            print(f"  {range_name}: {len(stocks)} stocks ({len(stocks)/len(low_float_stocks)*100:.1f}%)")
        
        print("\n🔥 ULTRA LOW-FLOAT STOCKS (Top 20 by Float):")
        print("-" * 60)
        for i, stock in enumerate(low_float_stocks[:20], 1):
            symbol = stock['symbol']
            float_millions = stock['floatMillions']
            float_pct = stock.get('floatPercentage')
            
            print(f"{i:2}. {symbol:6} - Float: {float_millions:6.2f}M shares", end="")
            if float_pct:
                print(f" ({float_pct:.1f}% of outstanding)")
            else:
                print()
        
        print("\n💎 MICRO-FLOAT STOCKS (<5M shares):")
        print("-" * 60)
        micro_floats = float_ranges['0-5M'][:10]
        for stock in micro_floats:
            symbol = stock['symbol']
            float_millions = stock['floatMillions']
            print(f"  {symbol:6} - {float_millions:4.2f}M shares")
        
        if len(float_ranges['0-5M']) > 10:
            print(f"  ... and {len(float_ranges['0-5M']) - 10} more")
        
        print("\n📊 Float Percentage Stats (where available):")
        stocks_with_pct = [s for s in low_float_stocks if s.get('floatPercentage')]
        if stocks_with_pct:
            avg_float_pct = sum(s['floatPercentage'] for s in stocks_with_pct) / len(stocks_with_pct)
            min_float_pct = min(s['floatPercentage'] for s in stocks_with_pct)
            max_float_pct = max(s['floatPercentage'] for s in stocks_with_pct)
            
            print(f"  Average Float %: {avg_float_pct:.1f}%")
            print(f"  Min Float %: {min_float_pct:.1f}%")
            print(f"  Max Float %: {max_float_pct:.1f}%")

def main():
    """Test the float scanner"""
    
    # Check for API key
    if not os.getenv('FMP_API_KEY'):
        print("❌ ERROR: Please add FMP_API_KEY to your .env file")
        return
    
    scanner = FloatScanner()
    
    # Display low-float statistics
    scanner.display_low_float_stats()
    
    # Example: Get float data for specific symbols
    print("\n" + "=" * 60)
    print("📈 EXAMPLE: Float Data for Specific Symbols")
    print("=" * 60)
    
    test_symbols = ['AAPL', 'TSLA', 'GME', 'AMC', 'BBBY']
    float_data = scanner.get_float_for_symbols(test_symbols)
    
    for symbol in test_symbols:
        if symbol in float_data:
            data = float_data[symbol]
            if data['floatShares']:
                print(f"{symbol:5} - Float: {data['floatMillions']:.2f}M shares")
            else:
                print(f"{symbol:5} - No float data available")
        else:
            print(f"{symbol:5} - Not found in float database")
    
    print("\n✅ Float scanner ready for integration with gap scanner!")
    print("💡 Usage in gap scanner:")
    print("  from float_scanner import FloatScanner")
    print("  float_scanner = FloatScanner()")
    print("  low_float_stocks = float_scanner.get_low_float_stocks()")

if __name__ == "__main__":
    main()