#!/usr/bin/env python3
"""
Backtest Former Momo Scanner - Tracks symbols with recent momentum
For backtesting, tracks symbols that had alerts in recent history
"""

import logging
from typing import Set, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestFormerMomoScanner:
    """Backtest version of former momo scanner"""
    
    def __init__(self, lookback_minutes: int = 60):
        """
        Initialize Backtest Former Momo Scanner
        
        Args:
            lookback_minutes: How many minutes to look back for recent activity
        """
        self.lookback_minutes = lookback_minutes
        
        # Track alert history: symbol -> list of alert times
        self.alert_history = defaultdict(list)
        
        # Current former momo symbols
        self.former_momo_symbols = set()
        
    def add_alert(self, symbol: str, time_str: str, alert_type: str = 'HIGH') -> None:
        """
        Record an alert for a symbol
        
        Args:
            symbol: Stock symbol
            time_str: Time of alert (HH:MM format)
            alert_type: Type of alert (HIGH, MEDIUM, etc)
        """
        self.alert_history[symbol].append({
            'time': time_str,
            'type': alert_type
        })
        
        # Add to former momo set
        self.former_momo_symbols.add(symbol)
    
    def is_former_momo(self, symbol: str, current_time: str = None) -> bool:
        """
        Check if symbol is a former momo (had recent alerts)
        
        Args:
            symbol: Stock symbol
            current_time: Current time string (HH:MM format)
            
        Returns:
            True if symbol had recent momentum
        """
        # For simplified backtest, just check if symbol had any alerts
        return symbol in self.former_momo_symbols
    
    def get_alert_count(self, symbol: str) -> int:
        """Get number of alerts for a symbol"""
        return len(self.alert_history.get(symbol, []))
    
    def get_recent_alerts(self, symbol: str, limit: int = 5) -> List[Dict]:
        """Get recent alerts for a symbol"""
        alerts = self.alert_history.get(symbol, [])
        return alerts[-limit:] if alerts else []
    
    def clear_history(self, symbol: Optional[str] = None) -> None:
        """Clear alert history"""
        if symbol:
            if symbol in self.alert_history:
                del self.alert_history[symbol]
            self.former_momo_symbols.discard(symbol)
        else:
            self.alert_history.clear()
            self.former_momo_symbols.clear()