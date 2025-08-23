"""
Exponential Moving Average (EMA) Calculator - TA-Lib Implementation

OVERVIEW:
Professional-grade EMA implementation using TA-Lib for backtesting and technical analysis.
EMA gives more weight to recent prices, making it more responsive than SMA.

TA-LIB ADVANTAGES:
- Battle-tested C implementation for speed and accuracy
- Industry-standard calculations used by professional platforms
- Optimized for large dataset backtesting
- Consistent with major trading platforms (TradingView, MetaTrader, etc.)

INSTALLATION:
pip install TA-Lib

Note: TA-Lib requires compilation. On some systems you may need:
- Windows: Download precompiled wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/
- macOS: brew install ta-lib
- Linux: Install dependencies first, then pip install

KEY FEATURES:
- TA-Lib powered EMA calculations for maximum accuracy
- Crossover signal detection (golden cross, death cross)
- Support/resistance level identification
- Trend direction analysis
- Multiple timeframe support (8, 20, 50, 200 periods)
- Dynamic signal generation for entry/exit points

TRADING APPLICATIONS:
- Trend identification and direction
- Entry/exit signal generation via crossovers
- Dynamic support and resistance levels
- Price action confirmation
- Multi-timeframe analysis

COMMON PERIODS:
- Day Trading: 8 EMA, 20 EMA
- Swing Trading: 20 EMA, 50 EMA
- Long-term: 50 EMA, 200 EMA
- Ross Cameron Style: 9 EMA, 20 EMA
"""

import pandas as pd
import numpy as np
try:
    import talib
except Exception:
    talib = None
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
from datetime import datetime

class EMACrossType(Enum):
    """Types of EMA crossovers"""
    GOLDEN_CROSS = "golden_cross"  # Fast EMA crosses above slow EMA (bullish)
    DEATH_CROSS = "death_cross"    # Fast EMA crosses below slow EMA (bearish)
    NO_CROSS = "no_cross"

class TrendDirection(Enum):
    """Trend direction based on EMA analysis"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"

class EMACrossSignal(NamedTuple):
    """EMA crossover signal details"""
    timestamp: datetime
    cross_type: EMACrossType
    fast_ema_value: float
    slow_ema_value: float
    price: float
    strength: float  # 0-1 scale based on separation

class EMASupportResistance(NamedTuple):
    """EMA support/resistance level"""
    level: float
    ema_period: int
    touches: int
    strength: float  # Based on number of touches and respect

class EMACalculator:
    """
    Professional-grade EMA calculator for backtesting and analysis
    """
    
    def __init__(self):
        self.ema_cache = {}  # Cache calculated EMAs for performance
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average using TA-Lib
        
        Args:
            prices: Price series (typically close prices)
            period: EMA period (e.g., 20 for 20-day EMA)
            
        Returns:
            Series with EMA values
        """
        if len(prices) < period:
            return pd.Series(index=prices.index, dtype=float)
        
        # Create cache key
        cache_key = f"{id(prices)}_{period}"
        if cache_key in self.ema_cache:
            return self.ema_cache[cache_key]
        
        # Use TA-Lib for EMA calculation with fallback
        if talib is not None:
            ema_values = talib.EMA(prices.values, timeperiod=period)
        else:
            ema_values = pd.Series(prices).ewm(span=period, adjust=False).mean().values
        ema_series = pd.Series(ema_values, index=prices.index)
        
        # Cache result
        self.ema_cache[cache_key] = ema_series
        return ema_series
    
    def calculate_multiple_emas(self, prices: pd.Series, 
                               periods: List[int]) -> Dict[int, pd.Series]:
        """
        Calculate multiple EMAs efficiently
        
        Args:
            prices: Price series
            periods: List of EMA periods to calculate
            
        Returns:
            Dictionary with period as key, EMA series as value
        """
        emas = {}
        for period in periods:
            emas[period] = self.calculate_ema(prices, period)
        
        return emas
    
    def detect_crossovers(self, fast_ema: pd.Series, slow_ema: pd.Series,
                         prices: pd.Series) -> List[EMACrossSignal]:
        """
        Detect EMA crossovers (golden cross and death cross)
        
        Args:
            fast_ema: Faster EMA series (lower period)
            slow_ema: Slower EMA series (higher period)
            prices: Price series for context
            
        Returns:
            List of crossover signals
        """
        signals = []
        
        if len(fast_ema) < 2 or len(slow_ema) < 2:
            return signals
        
        for i in range(1, len(fast_ema)):
            if pd.isna(fast_ema.iloc[i]) or pd.isna(slow_ema.iloc[i]):
                continue
            if pd.isna(fast_ema.iloc[i-1]) or pd.isna(slow_ema.iloc[i-1]):
                continue
                
            current_fast = fast_ema.iloc[i]
            current_slow = slow_ema.iloc[i]
            prev_fast = fast_ema.iloc[i-1]
            prev_slow = slow_ema.iloc[i-1]
            
            # Golden Cross: Fast EMA crosses above slow EMA
            if prev_fast <= prev_slow and current_fast > current_slow:
                separation = abs(current_fast - current_slow) / current_slow
                strength = min(separation * 100, 1.0)  # Cap at 1.0
                
                signals.append(EMACrossSignal(
                    timestamp=fast_ema.index[i],
                    cross_type=EMACrossType.GOLDEN_CROSS,
                    fast_ema_value=current_fast,
                    slow_ema_value=current_slow,
                    price=prices.iloc[i],
                    strength=strength
                ))
            
            # Death Cross: Fast EMA crosses below slow EMA
            elif prev_fast >= prev_slow and current_fast < current_slow:
                separation = abs(current_fast - current_slow) / current_slow
                strength = min(separation * 100, 1.0)  # Cap at 1.0
                
                signals.append(EMACrossSignal(
                    timestamp=fast_ema.index[i],
                    cross_type=EMACrossType.DEATH_CROSS,
                    fast_ema_value=current_fast,
                    slow_ema_value=current_slow,
                    price=prices.iloc[i],
                    strength=strength
                ))
        
        return signals
    
    def identify_trend_direction(self, emas: Dict[int, pd.Series], 
                               current_price: float) -> TrendDirection:
        """
        Identify overall trend direction using multiple EMAs
        
        Args:
            emas: Dictionary of EMA periods and their series
            current_price: Current price to compare against EMAs
            
        Returns:
            Trend direction enum
        """
        if not emas:
            return TrendDirection.UNKNOWN
        
        # Get most recent EMA values
        recent_emas = {}
        for period, ema_series in emas.items():
            if len(ema_series) > 0 and not pd.isna(ema_series.iloc[-1]):
                recent_emas[period] = ema_series.iloc[-1]
        
        if not recent_emas:
            return TrendDirection.UNKNOWN
        
        # Sort EMAs by period
        sorted_periods = sorted(recent_emas.keys())
        
        # Check if EMAs are in ascending order (bullish alignment)
        bullish_alignment = True
        bearish_alignment = True
        
        for i in range(len(sorted_periods) - 1):
            short_period = sorted_periods[i]
            long_period = sorted_periods[i + 1]
            
            if recent_emas[short_period] <= recent_emas[long_period]:
                bullish_alignment = False
            if recent_emas[short_period] >= recent_emas[long_period]:
                bearish_alignment = False
        
        # Price position relative to EMAs
        price_above_all = all(current_price > ema for ema in recent_emas.values())
        price_below_all = all(current_price < ema for ema in recent_emas.values())
        
        # Determine trend
        if bullish_alignment and price_above_all:
            return TrendDirection.BULLISH
        elif bearish_alignment and price_below_all:
            return TrendDirection.BEARISH
        else:
            return TrendDirection.SIDEWAYS
    
    def find_support_resistance_levels(self, prices: pd.Series, 
                                     ema: pd.Series, 
                                     tolerance_pct: float = 0.5) -> List[EMASupportResistance]:
        """
        Find support and resistance levels using EMA
        
        Args:
            prices: Price series
            ema: EMA series to analyze
            tolerance_pct: Percentage tolerance for level touches
            
        Returns:
            List of support/resistance levels
        """
        if len(prices) < 10 or len(ema) < 10:
            return []
        
        levels = []
        touches = []
        
        # Find where price touches EMA (within tolerance)
        for i in range(len(prices)):
            if pd.isna(ema.iloc[i]):
                continue
                
            price = prices.iloc[i]
            ema_value = ema.iloc[i]
            
            # Calculate percentage difference
            diff_pct = abs(price - ema_value) / ema_value * 100
            
            if diff_pct <= tolerance_pct:
                touches.append({
                    'index': i,
                    'price': price,
                    'ema_value': ema_value,
                    'timestamp': prices.index[i]
                })
        
        # Group touches into levels
        if len(touches) >= 2:
            # For simplicity, create one level based on average EMA value
            avg_ema = np.mean([touch['ema_value'] for touch in touches])
            
            levels.append(EMASupportResistance(
                level=avg_ema,
                ema_period=None,  # Would need to track this separately
                touches=len(touches),
                strength=min(len(touches) / 10.0, 1.0)  # Normalize to 0-1
            ))
        
        return levels
    
    def get_ema_signals(self, prices: pd.Series, fast_period: int = 20, 
                       slow_period: int = 50) -> Dict:
        """
        Get comprehensive EMA analysis for trading signals
        
        Args:
            prices: Price series
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            
        Returns:
            Dictionary with all EMA analysis results
        """
        # Calculate EMAs
        fast_ema = self.calculate_ema(prices, fast_period)
        slow_ema = self.calculate_ema(prices, slow_period)
        
        # Detect crossovers
        crossovers = self.detect_crossovers(fast_ema, slow_ema, prices)
        
        # Identify trend
        emas = {fast_period: fast_ema, slow_period: slow_ema}
        current_price = prices.iloc[-1] if len(prices) > 0 else 0
        trend = self.identify_trend_direction(emas, current_price)
        
        # Find support/resistance
        fast_sr = self.find_support_resistance_levels(prices, fast_ema)
        slow_sr = self.find_support_resistance_levels(prices, slow_ema)
        
        # Current values
        current_fast_ema = fast_ema.iloc[-1] if len(fast_ema) > 0 else None
        current_slow_ema = slow_ema.iloc[-1] if len(slow_ema) > 0 else None
        
        return {
            'fast_ema': fast_ema,
            'slow_ema': slow_ema,
            'current_fast_ema': current_fast_ema,
            'current_slow_ema': current_slow_ema,
            'crossovers': crossovers,
            'trend_direction': trend,
            'support_resistance': {
                'fast_ema_levels': fast_sr,
                'slow_ema_levels': slow_sr
            },
            'price_above_fast_ema': current_price > current_fast_ema if current_fast_ema else False,
            'price_above_slow_ema': current_price > current_slow_ema if current_slow_ema else False,
            'ema_separation_pct': ((current_fast_ema - current_slow_ema) / current_slow_ema * 100) 
                                 if current_fast_ema and current_slow_ema else 0
        }
    
    def clear_cache(self):
        """Clear EMA calculation cache"""
        self.ema_cache.clear()

# Ross Cameron specific EMA configurations
class RossCameronEMAConfig:
    """EMA configurations specific to Ross Cameron's style"""
    
    FAST_EMA = 9    # Ross's preferred fast EMA
    SLOW_EMA = 20   # Ross's preferred slow EMA
    
    @staticmethod
    def get_ross_ema_signals(prices: pd.Series) -> Dict:
        """Get EMA signals using Ross Cameron's preferred settings"""
        calculator = EMACalculator()
        return calculator.get_ema_signals(
            prices, 
            fast_period=RossCameronEMAConfig.FAST_EMA,
            slow_period=RossCameronEMAConfig.SLOW_EMA
        )

# Standard day trading EMA configurations
class DayTradingEMAConfig:
    """Common EMA configurations for day trading"""
    
    SCALPING_FAST = 8
    SCALPING_SLOW = 20
    
    SWING_FAST = 20
    SWING_SLOW = 50
    
    TREND_FAST = 50
    TREND_SLOW = 200

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='1min')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates
    )
    
    # Initialize calculator
    calculator = EMACalculator()
    
    # Get Ross Cameron style signals
    signals = RossCameronEMAConfig.get_ross_ema_signals(prices)
    
    print("EMA Calculator Implementation Complete")
    print("Features:")
    print("- Mathematically accurate EMA calculation")
    print("- Crossover signal detection (golden/death cross)")
    print("- Trend direction analysis")
    print("- Support/resistance level identification")
    print("- Multiple timeframe support")
    print("- Ross Cameron specific configurations")
    print("- Performance optimized with caching")
    
    print(f"\nCurrent trend: {signals['trend_direction']}")
    print(f"Fast EMA (9): {signals['current_fast_ema']:.2f}")
    print(f"Slow EMA (20): {signals['current_slow_ema']:.2f}")
    print(f"EMA Separation: {signals['ema_separation_pct']:.2f}%")