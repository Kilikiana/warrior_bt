#!/usr/bin/env python3
"""
Backtest RVOL Scanner - Calculates relative volume from historical data
Uses pre-loaded volume baselines for 5-minute rolling RVOL % calculation
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestRvolScanner:
    """Backtest version of RVOL scanner for historical data"""
    
    def __init__(self, min_rvol_5min: float = 500.0):
        """
        Initialize Backtest RVOL Scanner
        
        Args:
            min_rvol_5min: Minimum 5-minute RVOL % to flag (default 500%)
        """
        self.min_rvol_5min = min_rvol_5min
        
        # Load volume baselines
        self.volume_baselines = self.load_volume_baselines()
        
    def load_volume_baselines(self) -> Dict:
        """Load pre-calculated volume baselines"""
        cache_dir = Path('/Users/claytonsmacbookpro/Projects/warrior_bt/shared_cache')
        baseline_file = cache_dir / 'volume_baselines_20d.json'
        
        if not baseline_file.exists():
            baseline_file = cache_dir / 'volume_baselines_5min.json'
        
        if not baseline_file.exists():
            logger.warning("No volume baselines found")
            return {}
        
        logger.info(f"Loading baselines from {baseline_file.name}")
        with open(baseline_file, 'r') as f:
            data = json.load(f)
            
        # Parse the nested baseline format
        baselines = {}
        for symbol, symbol_data in data.items():
            if isinstance(symbol_data, dict):
                # Combine all time-based averages into one dict
                combined = {}
                
                # Add regular market hours
                if 'regular_avg' in symbol_data:
                    combined.update(symbol_data['regular_avg'])
                
                # Add pre-market hours
                if 'premarket_avg' in symbol_data:
                    combined.update(symbol_data['premarket_avg'])
                
                # Add after-hours
                if 'afterhours_avg' in symbol_data:
                    combined.update(symbol_data['afterhours_avg'])
                
                # Add daily average
                if 'daily_avg' in symbol_data:
                    combined['daily_avg'] = symbol_data['daily_avg']
                
                baselines[symbol] = combined
        
        logger.info(f"Loaded baselines for {len(baselines)} symbols")
        return baselines
    
    def calculate_5min_rvol(self, symbol: str, minute_data: Dict, last_5_minutes: List[str]) -> float:
        """
        Calculate 5-minute rolling RVOL %
        
        Args:
            symbol: Stock symbol
            minute_data: Dict of time -> bar data for this symbol
            last_5_minutes: List of time strings for last 5 minutes
            
        Returns:
            RVOL percentage
        """
        # Sum volume for last 5 minutes
        current_volume = 0
        for time_str in last_5_minutes:
            if time_str in minute_data:
                current_volume += minute_data[time_str]['volume']
        
        if current_volume == 0:
            return 0.0
        
        # Get baseline for these minutes
        if symbol not in self.volume_baselines:
            # No baseline at all - if high volume, flag it
            if current_volume >= 100000:
                return 999999.0  # Special flag for zero baseline with high volume
            return 0.0
        
        baseline_data = self.volume_baselines[symbol]
        baseline_volume = 0
        
        for time_str in last_5_minutes:
            if time_str in baseline_data:
                baseline_volume += baseline_data[time_str]
        
        # CRITICAL FIX: If no baseline for these specific minutes but we have daily average
        if baseline_volume == 0:
            # Try to use daily average as fallback
            if 'daily_avg' in baseline_data and baseline_data['daily_avg'] > 0:
                # Use daily average divided by 78 (390 minutes / 5-min periods)
                baseline_volume = baseline_data['daily_avg'] / 78
            # If still no baseline but high volume, flag it
            elif current_volume >= 100000:
                return 999999.0  # Special flag for zero baseline with high volume
        
        if baseline_volume > 0:
            return (current_volume / baseline_volume) * 100
        
        return 0.0
    
    def check_high_rvol(self, rvol_pct: float) -> bool:
        """Check if RVOL meets minimum threshold"""
        return rvol_pct >= self.min_rvol_5min
    
    def has_baseline(self, symbol: str) -> bool:
        """Check if symbol has volume baseline data"""
        return symbol in self.volume_baselines