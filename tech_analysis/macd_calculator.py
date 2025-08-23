"""
MACD Calculator - TA-Lib Implementation

OVERVIEW:
Professional-grade MACD implementation using TA-Lib for backtesting and technical analysis.
MACD (Moving Average Convergence Divergence) measures momentum and trend changes.

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

MACD COMPONENTS:
1. MACD Line = EMA(12) - EMA(26)
2. Signal Line = EMA(9) of MACD Line  
3. Histogram = MACD Line - Signal Line

ROSS CAMERON'S MACD USAGE:
- Standard settings: 12, 26, 9 (DON'T change these)
- Entry confirmation: MACD > Signal Line (bullish)
- Exit signal: MACD crosses below Signal Line (bearish)
- Front-side trading: Only trade when MACD > 0
- Walk away: When market MACD turns bearish

TRADING SIGNALS:
- Bullish Crossover: MACD crosses above Signal Line
- Bearish Crossover: MACD crosses below Signal Line
- Zero Line Cross: MACD crosses above/below zero
- Divergence: Price vs MACD direction mismatch
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

class MACDSignalType(Enum):
    """Types of MACD signals"""
    BULLISH_CROSSOVER = "bullish_crossover"    # MACD crosses above signal
    BEARISH_CROSSOVER = "bearish_crossover"    # MACD crosses below signal
    ZERO_LINE_BULLISH = "zero_line_bullish"    # MACD crosses above zero
    ZERO_LINE_BEARISH = "zero_line_bearish"    # MACD crosses below zero
    BULLISH_DIVERGENCE = "bullish_divergence"  # Price down, MACD up
    BEARISH_DIVERGENCE = "bearish_divergence"  # Price up, MACD down
    NO_SIGNAL = "no_signal"

class MACDState(Enum):
    """Current MACD state for Ross Cameron strategy"""
    BULLISH = "bullish"           # MACD > Signal, good for trading
    BEARISH = "bearish"           # MACD < Signal, stop trading
    FRONT_SIDE = "front_side"     # MACD > 0, Ross's preferred zone
    BACK_SIDE = "back_side"       # MACD < 0, avoid trading

class MACDSignal(NamedTuple):
    """MACD signal with all relevant information"""
    timestamp: datetime
    signal_type: MACDSignalType
    macd_value: float
    signal_value: float
    histogram: float
    price: float
    strength: float  # 0-1 scale based on histogram size

class MACDAnalysis(NamedTuple):
    """Comprehensive MACD analysis results"""
    macd_line: pd.Series
    signal_line: pd.Series
    histogram: pd.Series
    current_macd: float
    current_signal: float
    current_histogram: float
    state: MACDState
    is_bullish: bool
    is_front_side: bool
    signals: List[MACDSignal]
    divergences: List[MACDSignal]

class MACDCalculator:
    """
    Professional MACD calculator using TA-Lib
    Implements Ross Cameron's MACD methodology
    """
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD calculator with Ross Cameron's standard settings
        
        Args:
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26) 
            signal_period: Signal line EMA period (default 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.macd_cache = {}
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD using TA-Lib
        
        Args:
            prices: Price series (typically close prices)
            
        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        if len(prices) < max(self.fast_period, self.slow_period, self.signal_period):
            empty_series = pd.Series(index=prices.index, dtype=float)
            return empty_series, empty_series, empty_series
        
        # Create cache key
        cache_key = f"{id(prices)}_{self.fast_period}_{self.slow_period}_{self.signal_period}"
        if cache_key in self.macd_cache:
            return self.macd_cache[cache_key]
        
        # Use TA-Lib for MACD calculation with fallback
        if talib is not None:
            macd_values, signal_values, histogram_values = talib.MACD(
                prices.values,
                fastperiod=self.fast_period,
                slowperiod=self.slow_period,
                signalperiod=self.signal_period
            )
        else:
            fast_ema = pd.Series(prices).ewm(span=self.fast_period, adjust=False).mean()
            slow_ema = pd.Series(prices).ewm(span=self.slow_period, adjust=False).mean()
            macd = fast_ema - slow_ema
            signal = macd.ewm(span=self.signal_period, adjust=False).mean()
            histogram = macd - signal
            macd_values, signal_values, histogram_values = macd.values, signal.values, histogram.values
        
        # Convert to pandas Series
        macd_series = pd.Series(macd_values, index=prices.index)
        signal_series = pd.Series(signal_values, index=prices.index)
        histogram_series = pd.Series(histogram_values, index=prices.index)
        
        # Cache results
        result = (macd_series, signal_series, histogram_series)
        self.macd_cache[cache_key] = result
        
        return result
    
    def detect_crossovers(self, macd_line: pd.Series, signal_line: pd.Series,
                         prices: pd.Series) -> List[MACDSignal]:
        """
        Detect MACD crossover signals
        
        Args:
            macd_line: MACD line series
            signal_line: Signal line series
            prices: Price series for context
            
        Returns:
            List of MACD crossover signals
        """
        signals = []
        
        if len(macd_line) < 2 or len(signal_line) < 2:
            return signals
        
        for i in range(1, len(macd_line)):
            if pd.isna(macd_line.iloc[i]) or pd.isna(signal_line.iloc[i]):
                continue
            if pd.isna(macd_line.iloc[i-1]) or pd.isna(signal_line.iloc[i-1]):
                continue
                
            current_macd = macd_line.iloc[i]
            current_signal = signal_line.iloc[i]
            prev_macd = macd_line.iloc[i-1]
            prev_signal = signal_line.iloc[i-1]
            
            # Calculate signal strength based on histogram size
            histogram = current_macd - current_signal
            strength = min(abs(histogram) / max(abs(current_macd), 0.001), 1.0)
            
            # Bullish Crossover: MACD crosses above Signal
            if prev_macd <= prev_signal and current_macd > current_signal:
                signals.append(MACDSignal(
                    timestamp=macd_line.index[i],
                    signal_type=MACDSignalType.BULLISH_CROSSOVER,
                    macd_value=current_macd,
                    signal_value=current_signal,
                    histogram=histogram,
                    price=prices.iloc[i],
                    strength=strength
                ))
            
            # Bearish Crossover: MACD crosses below Signal
            elif prev_macd >= prev_signal and current_macd < current_signal:
                signals.append(MACDSignal(
                    timestamp=macd_line.index[i],
                    signal_type=MACDSignalType.BEARISH_CROSSOVER,
                    macd_value=current_macd,
                    signal_value=current_signal,
                    histogram=histogram,
                    price=prices.iloc[i],
                    strength=strength
                ))
        
        return signals
    
    def detect_zero_line_crosses(self, macd_line: pd.Series, 
                                prices: pd.Series) -> List[MACDSignal]:
        """
        Detect MACD zero line crossovers
        
        Args:
            macd_line: MACD line series
            prices: Price series for context
            
        Returns:
            List of zero line cross signals
        """
        signals = []
        
        if len(macd_line) < 2:
            return signals
        
        for i in range(1, len(macd_line)):
            if pd.isna(macd_line.iloc[i]) or pd.isna(macd_line.iloc[i-1]):
                continue
                
            current_macd = macd_line.iloc[i]
            prev_macd = macd_line.iloc[i-1]
            
            # Bullish Zero Line Cross: MACD crosses above zero
            if prev_macd <= 0 and current_macd > 0:
                signals.append(MACDSignal(
                    timestamp=macd_line.index[i],
                    signal_type=MACDSignalType.ZERO_LINE_BULLISH,
                    macd_value=current_macd,
                    signal_value=0.0,
                    histogram=current_macd,  # No signal line at zero
                    price=prices.iloc[i],
                    strength=min(abs(current_macd) / max(abs(prices.iloc[i]), 0.001), 1.0)
                ))
            
            # Bearish Zero Line Cross: MACD crosses below zero
            elif prev_macd >= 0 and current_macd < 0:
                signals.append(MACDSignal(
                    timestamp=macd_line.index[i],
                    signal_type=MACDSignalType.ZERO_LINE_BEARISH,
                    macd_value=current_macd,
                    signal_value=0.0,
                    histogram=current_macd,
                    price=prices.iloc[i],
                    strength=min(abs(current_macd) / max(abs(prices.iloc[i]), 0.001), 1.0)
                ))
        
        return signals
    
    def determine_macd_state(self, macd_value: float, signal_value: float) -> MACDState:
        """
        Determine current MACD state for Ross Cameron strategy
        
        Args:
            macd_value: Current MACD value
            signal_value: Current signal line value
            
        Returns:
            MACD state enum
        """
        if pd.isna(macd_value) or pd.isna(signal_value):
            return MACDState.BEARISH
        
        # Ross Cameron's key rules
        if macd_value > signal_value:
            if macd_value > 0:
                return MACDState.FRONT_SIDE  # Best condition for Ross
            else:
                return MACDState.BULLISH     # Good but not ideal
        else:
            if macd_value < 0:
                return MACDState.BACK_SIDE   # Avoid trading
            else:
                return MACDState.BEARISH     # Stop trading
    
    def is_bullish(self, macd_value: float, signal_value: float, 
                   histogram: float) -> bool:
        """
        Check if MACD is in bullish state (Ross Cameron style)
        
        Args:
            macd_value: Current MACD value
            signal_value: Current signal line value
            histogram: Current histogram value
            
        Returns:
            True if bullish conditions met
        """
        return (macd_value > signal_value and histogram > 0)
    
    def is_front_side(self, macd_value: float) -> bool:
        """
        Check if MACD is on "front side" (Ross Cameron terminology)
        
        Args:
            macd_value: Current MACD value
            
        Returns:
            True if MACD > 0 (front side)
        """
        return macd_value > 0 if not pd.isna(macd_value) else False
    
    def should_stop_trading(self, macd_value: float, signal_value: float) -> bool:
        """
        Check if should stop trading based on Ross Cameron's rules
        
        Args:
            macd_value: Current MACD value
            signal_value: Current signal line value
            
        Returns:
            True if should stop trading
        """
        if pd.isna(macd_value) or pd.isna(signal_value):
            return True
        
        # Ross stops trading when MACD crosses below signal line
        return macd_value < signal_value
    
    def get_comprehensive_analysis(self, prices: pd.Series) -> MACDAnalysis:
        """
        Get comprehensive MACD analysis for trading decisions
        
        Args:
            prices: Price series
            
        Returns:
            Complete MACD analysis object
        """
        # Calculate MACD components
        macd_line, signal_line, histogram = self.calculate_macd(prices)
        
        # Get current values
        current_macd = macd_line.iloc[-1] if len(macd_line) > 0 and not pd.isna(macd_line.iloc[-1]) else 0.0
        current_signal = signal_line.iloc[-1] if len(signal_line) > 0 and not pd.isna(signal_line.iloc[-1]) else 0.0
        current_histogram = histogram.iloc[-1] if len(histogram) > 0 and not pd.isna(histogram.iloc[-1]) else 0.0
        
        # Determine state
        state = self.determine_macd_state(current_macd, current_signal)
        is_bullish = self.is_bullish(current_macd, current_signal, current_histogram)
        is_front_side = self.is_front_side(current_macd)
        
        # Detect signals
        crossover_signals = self.detect_crossovers(macd_line, signal_line, prices)
        zero_line_signals = self.detect_zero_line_crosses(macd_line, prices)
        
        # Combine all signals
        all_signals = crossover_signals + zero_line_signals
        
        return MACDAnalysis(
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            current_macd=current_macd,
            current_signal=current_signal,
            current_histogram=current_histogram,
            state=state,
            is_bullish=is_bullish,
            is_front_side=is_front_side,
            signals=all_signals,
            divergences=[]  # TODO: Implement divergence detection
        )
    
    def clear_cache(self):
        """Clear MACD calculation cache"""
        self.macd_cache.clear()

# Ross Cameron specific MACD configurations
class RossCameronMACDConfig:
    """MACD configurations specific to Ross Cameron's methodology"""
    
    FAST_PERIOD = 12
    SLOW_PERIOD = 26
    SIGNAL_PERIOD = 9
    
    @staticmethod
    def get_ross_macd_analysis(prices: pd.Series) -> MACDAnalysis:
        """Get MACD analysis using Ross Cameron's exact settings"""
        calculator = MACDCalculator(
            fast_period=RossCameronMACDConfig.FAST_PERIOD,
            slow_period=RossCameronMACDConfig.SLOW_PERIOD,
            signal_period=RossCameronMACDConfig.SIGNAL_PERIOD
        )
        return calculator.get_comprehensive_analysis(prices)
    
    @staticmethod
    def should_enter_trade(analysis: MACDAnalysis) -> bool:
        """
        Check if should enter trade based on Ross Cameron's MACD rules
        
        Args:
            analysis: MACD analysis results
            
        Returns:
            True if MACD confirms entry
        """
        return (analysis.is_bullish and 
                analysis.state in [MACDState.FRONT_SIDE, MACDState.BULLISH])
    
    @staticmethod
    def should_exit_trade(analysis: MACDAnalysis) -> bool:
        """
        Check if should exit trade based on Ross Cameron's MACD rules
        
        Args:
            analysis: MACD analysis results
            
        Returns:
            True if MACD suggests exit
        """
        return analysis.state in [MACDState.BEARISH, MACDState.BACK_SIDE]

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='1min')
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(100) * 0.5),
        index=dates
    )
    
    # Get Ross Cameron style MACD analysis
    analysis = RossCameronMACDConfig.get_ross_macd_analysis(prices)
    
    print("MACD Calculator Implementation Complete")
    print("Features:")
    print("- TA-Lib powered MACD calculation")
    print("- Ross Cameron methodology implementation")
    print("- Crossover and zero-line signal detection")
    print("- Front-side/back-side analysis")
    print("- Entry/exit confirmation logic")
    print("- Performance optimized with caching")
    
    print(f"\nCurrent MACD State: {analysis.state}")
    print(f"MACD: {analysis.current_macd:.4f}")
    print(f"Signal: {analysis.current_signal:.4f}")
    print(f"Histogram: {analysis.current_histogram:.4f}")
    print(f"Is Bullish: {analysis.is_bullish}")
    print(f"Is Front Side: {analysis.is_front_side}")
    print(f"Should Enter: {RossCameronMACDConfig.should_enter_trade(analysis)}")
    print(f"Should Exit: {RossCameronMACDConfig.should_exit_trade(analysis)}")