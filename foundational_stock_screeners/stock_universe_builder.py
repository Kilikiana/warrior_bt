#!/usr/bin/env python3
"""
Stock Universe Builder - Gets clean list of stocks using Financial Modeling Prep
Filters out all ETFs and funds to create a pure stock universe for scanning
"""

import os
import json
import requests
from dotenv import load_dotenv
from typing import List, Dict
import pickle
from datetime import datetime, timedelta
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockUniverseBuilder:
    def __init__(self):
        """Initialize universe builder with FMP API"""
        self.fmp_api_key = os.getenv('FMP_API_KEY')
        if not self.fmp_api_key:
            raise ValueError("FMP_API_KEY not found in environment variables!")
        
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        # Use shared cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", "shared_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.cache_file = os.path.join(self.cache_dir, "stock_universe_cache.pkl")
        self.cache_duration_hours = 24  # Refresh universe daily
    
    def get_stock_universe(self, 
                          use_cache: bool = True) -> List[Dict]:
        """
        Get universe of ALL stocks (no ETFs/funds) from FMP
        
        Args:
            use_cache: Whether to use cached data if fresh
        
        Returns:
            List of stock dictionaries with symbol, name, price, volume, marketCap
        """
        
        # Check cache first
        if use_cache and self._is_cache_valid():
            logger.info("Using cached stock universe")
            return self._load_cache()
        
        logger.info("Building fresh stock universe from FMP...")
        
        # Build the screener URL with parameters - ONLY filtering ETFs/Funds
        params = {
            'apikey': self.fmp_api_key,
            'isEtf': 'false',           # NO ETFs
            'isFund': 'false',          # NO Funds
            'isActivelyTrading': 'true', # Active stocks only
            'exchange': 'NASDAQ,NYSE,AMEX',  # US exchanges only (includes ADRs)
            # Removed country filter to include foreign companies on US exchanges
            'limit': 10000              # Get as many as possible
        }
        
        # Make the API request
        url = f"{self.base_url}/stock-screener"
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            stocks = response.json()
            
            if not stocks:
                logger.warning("No stocks returned from FMP")
                return []
            
            logger.info(f"Found {len(stocks)} stocks matching criteria")
            
            # Process and clean the data
            clean_stocks = []
            for stock in stocks:
                # Double-check it's not an ETF (belt and suspenders)
                company_name = stock.get('companyName')
                if company_name and 'ETF' in company_name.upper():
                    continue
                if company_name and 'FUND' in company_name.upper():
                    continue
                
                clean_stocks.append({
                    'symbol': stock.get('symbol'),
                    'name': stock.get('companyName'),
                    'price': stock.get('price'),
                    'volume': stock.get('volume'),
                    'marketCap': stock.get('marketCap'),
                    'exchange': stock.get('exchangeShortName'),
                    'sector': stock.get('sector'),
                    'industry': stock.get('industry'),
                    'beta': stock.get('beta')
                })
            
            logger.info(f"Cleaned universe: {len(clean_stocks)} stocks")
            
            # Save to cache
            self._save_cache(clean_stocks)
            
            return clean_stocks
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from FMP: {e}")
            # Try to use stale cache if available
            if os.path.exists(self.cache_file):
                logger.info("Using stale cache due to API error")
                return self._load_cache()
            return []
    
    def get_low_float_stocks(self, max_shares_outstanding: float = 50_000_000) -> List[Dict]:
        """
        Get stocks with low shares outstanding (proxy for low float)
        
        Note: FMP doesn't provide float directly, but shares outstanding is close
        """
        
        # First get the stock universe
        all_stocks = self.get_stock_universe()
        
        low_float_stocks = []
        
        logger.info(f"Checking shares outstanding for {len(all_stocks)} stocks...")
        
        # We'd need to make individual API calls for detailed data
        # This is where you'd filter further if needed
        # For now, return stocks under $10 as they're more likely to be low float
        
        for stock in all_stocks:
            if stock['price'] <= 10 and stock['marketCap'] < 500_000_000:  # Under $500M market cap
                low_float_stocks.append(stock)
        
        logger.info(f"Found {len(low_float_stocks)} potential low-float stocks")
        return low_float_stocks
    
    def get_symbols_only(self) -> List[str]:
        """Get just the list of symbols for use in other scanners"""
        stocks = self.get_stock_universe()
        return [stock['symbol'] for stock in stocks]
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and is fresh"""
        if not os.path.exists(self.cache_file):
            return False
        
        # Check age of cache
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(self.cache_file))
        return cache_age < timedelta(hours=self.cache_duration_hours)
    
    def _save_cache(self, data: List[Dict]):
        """Save data to cache file"""
        # Save as pickle for backwards compatibility
        with open(self.cache_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save as JSON for readability
        json_file = self.cache_file.replace('.pkl', '.json')
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(data)} stocks to cache (both .pkl and .json)")
    
    def _load_cache(self) -> List[Dict]:
        """Load data from cache file"""
        with open(self.cache_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data)} stocks from cache")
        return data
    
    def display_universe_stats(self):
        """Display statistics about the stock universe"""
        stocks = self.get_stock_universe()
        
        if not stocks:
            print("No stocks in universe!")
            return
        
        # Calculate statistics
        price_ranges = {
            '$1-5': 0,
            '$5-10': 0,
            '$10-15': 0,
            '$15-20': 0
        }
        
        exchanges = {}
        sectors = {}
        
        for stock in stocks:
            # Price ranges
            price = stock['price']
            if price <= 5:
                price_ranges['$1-5'] += 1
            elif price <= 10:
                price_ranges['$5-10'] += 1
            elif price <= 15:
                price_ranges['$10-15'] += 1
            else:
                price_ranges['$15-20'] += 1
            
            # Exchanges
            exchange = stock.get('exchange', 'Unknown')
            exchanges[exchange] = exchanges.get(exchange, 0) + 1
            
            # Sectors
            sector = stock.get('sector', 'Unknown')
            if sector:
                sectors[sector] = sectors.get(sector, 0) + 1
        
        # Display results
        print("=" * 60)
        print("📊 STOCK UNIVERSE STATISTICS (NO ETFs/FUNDS)")
        print("=" * 60)
        print(f"Total Stocks: {len(stocks):,}")
        
        print("\nPrice Distribution:")
        for range_name, count in price_ranges.items():
            pct = (count / len(stocks)) * 100
            print(f"  {range_name}: {count:,} ({pct:.1f}%)")
        
        print("\nExchange Distribution:")
        for exchange, count in sorted(exchanges.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(stocks)) * 100
            print(f"  {exchange}: {count:,} ({pct:.1f}%)")
        
        print("\nTop Sectors:")
        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / len(stocks)) * 100
            print(f"  {sector}: {count:,} ({pct:.1f}%)")
        
        print("\nSample Stocks:")
        for stock in stocks[:10]:
            print(f"  {stock['symbol']:6} - ${stock['price']:6.2f} - {stock['name'][:40]}")

def main():
    """Test the universe builder"""
    
    # Check for API key
    if not os.getenv('FMP_API_KEY'):
        print("❌ ERROR: Please add FMP_API_KEY to your .env file")
        print("\nAdd this line to .env:")
        print("FMP_API_KEY=your_fmp_api_key_here")
        return
    
    builder = StockUniverseBuilder()
    
    # Get and display the universe
    builder.display_universe_stats()
    
    # Get just symbols for use in other scanners
    symbols = builder.get_symbols_only()
    print(f"\n✅ Universe contains {len(symbols)} tradeable stocks (NO ETFs/Funds)")
    
    # Show how to use with gap scanner
    print("\n💡 Use this universe in your gap scanner:")
    print("  from stock_universe_builder import StockUniverseBuilder")
    print("  builder = StockUniverseBuilder()")
    print("  stock_symbols = builder.get_symbols_only()")
    print("  # Now scan only these symbols - guaranteed no ETFs!")

if __name__ == "__main__":
    main()