#!/usr/bin/env python3
"""Quick test of gap scanner with just a few symbols"""

from alpaca_trade_api import REST, TimeFrame
import os
from dotenv import load_dotenv
import pandas as pd
import pytz

load_dotenv('/Users/claytonsmacbookpro/Projects/warrior_bt/.env')

# Initialize Alpaca API
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url='https://paper-api.alpaca.markets'
)

def test_gap():
    eastern = pytz.timezone('America/New_York')
    test_date = '2024-08-13'
    test_dt = pd.Timestamp(test_date, tz=eastern)
    prev_date = '2024-08-12'
    
    # Test with just AAPL
    symbol = 'AAPL'
    
    print(f"Testing {symbol} for date {test_date}")
    
    # Get previous close
    try:
        prev_bars = api.get_bars(
            symbol,
            TimeFrame.Day,
            start=prev_date,
            end=prev_date,
            feed='sip',
            limit=1
        )
        prev_close = None
        for bar in prev_bars:
            prev_close = bar.c
            print(f"Previous close for {symbol}: ${prev_close}")
            break
            
        if prev_close:
            # Get opening price
            start = test_dt.replace(hour=9, minute=30, second=0)
            end = test_dt.replace(hour=9, minute=31, second=0)
            
            curr_bars = api.get_bars(
                symbol,
                TimeFrame.Minute,
                start=start.isoformat(),
                end=end.isoformat(),
                feed='sip',
                limit=1
            )
            
            for bar in curr_bars:
                open_price = bar.o
                gap_pct = ((open_price - prev_close) / prev_close) * 100
                print(f"Opening price: ${open_price}")
                print(f"Gap: {gap_pct:.2f}%")
                break
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_gap()