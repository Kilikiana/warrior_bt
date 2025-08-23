#!/usr/bin/env python3
"""
Backtest Squeeze Scanner - Detects rapid price increases in historical data
Replays historical minute data to detect 10% in 10min patterns
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestSqueezeScanner:
    """Backtest version of squeeze scanner for historical data replay"""
    
    def __init__(self):
        """
        Initialize Backtest Squeeze Scanner with multiple thresholds
        
        Detects both:
        - 5% in 5 minutes (quick squeeze)
        - 10% in 10 minutes (strong squeeze)
        """
        self.et = pytz.timezone('US/Eastern')
        
        # Multiple squeeze thresholds
        self.squeeze_configs = [
            {'threshold': 5.0, 'window': 5, 'name': 'QUICK_SQUEEZE'},
            {'threshold': 10.0, 'window': 10, 'name': 'STRONG_SQUEEZE'}
        ]
        
        # Price history: symbol -> deque of (time_str, price) tuples
        self.price_history = {}
        
        # Track current squeeze status for each type
        self.squeeze_status = {}  # symbol -> {type: bool}
        
    def update_price(self, symbol: str, price: float, time_str: str) -> None:
        """
        Update price history for a symbol
        
        Args:
            symbol: Stock symbol
            price: Current price
            time_str: Time string (HH:MM format)
        """
        # Initialize history if needed (keep enough for longest window)
        if symbol not in self.price_history:
            max_window = max(cfg['window'] for cfg in self.squeeze_configs)
            self.price_history[symbol] = deque(maxlen=max_window + 5)
        
        # Add new price point
        self.price_history[symbol].append((time_str, price))
    
    def check_squeezes(self, symbol: str) -> Dict:
        """
        Check if symbol is experiencing any type of squeeze
        
        Returns:
            Dict with squeeze info for each type
        """
        results = {}
        
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            return {'any_squeeze': False}
        
        # Get the most recent price
        latest_time, latest_price = self.price_history[symbol][-1]
        
        # Check each squeeze configuration
        for config in self.squeeze_configs:
            threshold = config['threshold']
            window = config['window']
            name = config['name']
            
            # FIXED: Use the price at the START of the window (not cherry-picked low)
            # This gives us the actual directional move over the time period
            window_start_idx = max(0, len(self.price_history[symbol]) - window - 1)
            
            if window_start_idx >= len(self.price_history[symbol]) - 1:
                results[name] = {'detected': False}
                continue
                
            # Get the price at the start of our window
            start_time, start_price = self.price_history[symbol][window_start_idx]
            
            # Calculate the ACTUAL move from window start to now
            actual_pct_change = ((latest_price - start_price) / start_price) * 100
            
            # Also find the highest price in the window (to ensure we're near highs)
            window_high = latest_price
            for i in range(window_start_idx, len(self.price_history[symbol])):
                _, price = self.price_history[symbol][i]
                window_high = max(window_high, price)
            
            # Calculate how far we are from the window high
            pct_from_high = ((latest_price - window_high) / window_high) * 100
            
            # FIXED: Only detect squeeze if:
            # 1. We have positive movement over the period (actual_pct_change > threshold)
            # 2. We're within 5% of the window high (not in a deep pullback)
            squeeze_detected = (
                actual_pct_change >= threshold and  # Positive directional move
                pct_from_high >= -5.0  # Near the highs (not in pullback)
            )
            
            results[name] = {
                'detected': squeeze_detected,
                'pct_change': actual_pct_change if squeeze_detected else None,
                'from_time': start_time if squeeze_detected else None,
                'threshold': threshold,
                'window': window,
                'pct_from_high': pct_from_high if squeeze_detected else None
            }
        
        # Track if any squeeze detected
        results['any_squeeze'] = any(r['detected'] for r in results.values() if isinstance(r, dict))
        
        # Initialize squeeze status if needed
        if symbol not in self.squeeze_status:
            self.squeeze_status[symbol] = {}
        
        # Update status for each type
        for name, data in results.items():
            if isinstance(data, dict) and 'detected' in data:
                self.squeeze_status[symbol][name] = data['detected']
        
        return results
    
    def get_squeeze_status(self, symbol: str) -> bool:
        """Get current squeeze status for a symbol"""
        return self.squeeze_status.get(symbol, False)
    
    def clear_history(self, symbol: Optional[str] = None) -> None:
        """Clear price history"""
        if symbol:
            if symbol in self.price_history:
                del self.price_history[symbol]
            if symbol in self.squeeze_status:
                del self.squeeze_status[symbol]
        else:
            self.price_history.clear()
            self.squeeze_status.clear()