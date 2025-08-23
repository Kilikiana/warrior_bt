"""
Multi-Timeframe Pattern Monitor for Ross Cameron Strategy

ROSS'S MULTI-TIMEFRAME PHILOSOPHY (HIS EXACT WORDS):
"Whether you trade from a tick chart or a daily chart, it's always useful to see price action 
from multiple timeframes and ensure they're all telling a similar story."

KEY PRINCIPLES:
1. CONFIRMATION: "Confirm short-term setups and price action with longer time frames"
2. TREND CONTINUITY: "You're not playing the odds" if timeframes conflict
3. VITAL LEVELS: "A resistance level on daily chart is much more significant than one on 5-minute"
4. HOME RUN TRADES: "Sometimes the setup on 5-minute turns into daily breakout"

TIMEFRAME HIERARCHY:
- Daily: Long-term trend, vital S/R levels (weeks/months/years of data)
- 5-Minute: Primary pattern detection and entry timeframe
- 1-Minute: Precise entry timing and micro-movements

ROSS'S WARNING:
"Don't get bogged down on one time frame, it will cloud your judgement"
"You might be looking for long entries until daily chart prompts you to shift to shorts"
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
from datetime import datetime, timedelta
import logging

from .bull_flag_pattern import BullFlagDetector, BullFlagSignal, BullFlagStage, BullFlagValidation

class Timeframe(Enum):
    """Available timeframes"""
    DAILY = "daily"
    FIVE_MIN = "5min"
    ONE_MIN = "1min"

class MarketContext(Enum):
    """Daily chart market context - Ross's trend analysis"""
    STRONG_UPTREND = "strong_uptrend"      # Clear higher highs/lows, strong volume
    UPTREND = "uptrend"                    # Generally rising, some pullbacks
    SIDEWAYS = "sideways"                  # Consolidation, no clear direction
    DOWNTREND = "downtrend"               # Lower highs/lows, declining trend
    STRONG_DOWNTREND = "strong_downtrend" # Clear bearish trend, heavy selling

class TimeframeContinuity(Enum):
    """Timeframe alignment assessment"""
    BULLISH_ALIGNMENT = "bullish_alignment"      # All timeframes bullish
    MIXED_BULLISH = "mixed_bullish"              # Mostly bullish, some conflict
    CONFLICTED = "conflicted"                    # Major timeframe conflicts
    MIXED_BEARISH = "mixed_bearish"              # Mostly bearish, some conflict
    BEARISH_ALIGNMENT = "bearish_alignment"      # All timeframes bearish

class VitalLevel(NamedTuple):
    """Vital support/resistance level identified across timeframes"""
    price: float
    level_type: str  # "support" or "resistance"
    timeframe_significance: Timeframe  # Highest timeframe where level is significant
    touches: int  # Number of times level was tested
    last_test: datetime
    strength: float  # 0-1, based on timeframe and touches

class TimeframeData:
    """Stores OHLCV data for a specific timeframe"""
    
    def __init__(self, timeframe: Timeframe):
        self.timeframe = timeframe
        self.data: List[Dict] = []
        self.df: Optional[pd.DataFrame] = None
        
    def add_bar(self, timestamp: datetime, open_price: float, high: float, 
                low: float, close: float, volume: int) -> None:
        """Add new price bar"""
        bar = {
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
        self.data.append(bar)
        self._update_dataframe()
    
    def _update_dataframe(self) -> None:
        """Update pandas DataFrame"""
        if len(self.data) > 0:
            self.df = pd.DataFrame(self.data)
            self.df.set_index('timestamp', inplace=True)
    
    def get_latest_bar(self) -> Optional[Dict]:
        """Get most recent price bar"""
        return self.data[-1] if self.data else None
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Get DataFrame for analysis"""
        return self.df

class MultiTimeframeAnalysis(NamedTuple):
    """Results from multi-timeframe analysis - Ross's methodology"""
    daily_context: MarketContext
    timeframe_continuity: TimeframeContinuity  # Ross's "telling similar story"
    primary_timeframe: Timeframe  # Where pattern was detected
    bull_flag_signal: Optional[BullFlagSignal]
    vital_levels: List[VitalLevel]  # Significant S/R levels across timeframes
    entry_timeframe: Timeframe    # Recommended entry timeframe
    exit_timeframe: Timeframe     # Recommended exit monitoring timeframe
    confluence_score: float       # 0-1, higher = better confluence
    home_run_potential: bool      # Could this become a daily breakout?
    bias_shift_needed: bool       # Ross: "prompts you to shift to shorts"
    
class MultiTimeframePatternMonitor:
    """
    Ross Cameron's Multi-Timeframe Pattern Detection and Trade Management
    """
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        
        # Timeframe data storage
        self.timeframes = {
            Timeframe.DAILY: TimeframeData(Timeframe.DAILY),
            Timeframe.FIVE_MIN: TimeframeData(Timeframe.FIVE_MIN),
            Timeframe.ONE_MIN: TimeframeData(Timeframe.ONE_MIN)
        }
        
        # Pattern detectors
        self.bull_flag_detector = BullFlagDetector()
        
        # Ross's preferences
        self.primary_pattern_timeframe = Timeframe.FIVE_MIN  # Ross's main timeframe
        self.entry_timeframe = Timeframe.ONE_MIN             # Precise entries
        self.exit_timeframe = Timeframe.FIVE_MIN             # "First red 5min candle"
        
        # Current position state
        self.position_entry_timeframe: Optional[Timeframe] = None
        self.last_5min_candle_color: Optional[str] = None  # For exit rule
        
    def update_timeframe_data(self, timeframe: Timeframe, timestamp: datetime,
                             open_price: float, high: float, low: float, 
                             close: float, volume: int) -> None:
        """Update data for specific timeframe"""
        if timeframe in self.timeframes:
            self.timeframes[timeframe].add_bar(timestamp, open_price, high, low, close, volume)
    
    def analyze_daily_context(self) -> MarketContext:
        """
        Analyze daily chart for market context using Ross's approach
        Ross: "A level on daily chart has been defended for weeks, months, or years"
        """
        daily_df = self.timeframes[Timeframe.DAILY].get_dataframe()
        
        if daily_df is None or len(daily_df) < 50:
            return MarketContext.SIDEWAYS
        
        # Ross's approach: Look for higher highs/lows vs EMA analysis
        closes = daily_df['close']
        highs = daily_df['high']
        lows = daily_df['low']
        
        # Get recent price action (last 10-20 days)
        recent_closes = closes.tail(20)
        recent_highs = highs.tail(20)
        recent_lows = lows.tail(20)
        
        # Identify trend using Ross's approach: higher highs and higher lows
        current_price = closes.iloc[-1]
        
        # Look for pattern of higher highs and higher lows (uptrend)
        recent_peaks = []
        recent_troughs = []
        
        for i in range(2, len(recent_highs) - 2):
            # Peak detection
            if (recent_highs.iloc[i] > recent_highs.iloc[i-1] and 
                recent_highs.iloc[i] > recent_highs.iloc[i+1]):
                recent_peaks.append(recent_highs.iloc[i])
            
            # Trough detection  
            if (recent_lows.iloc[i] < recent_lows.iloc[i-1] and 
                recent_lows.iloc[i] < recent_lows.iloc[i+1]):
                recent_troughs.append(recent_lows.iloc[i])
        
        # Analyze trend direction
        higher_highs = len(recent_peaks) >= 2 and recent_peaks[-1] > recent_peaks[0]
        higher_lows = len(recent_troughs) >= 2 and recent_troughs[-1] > recent_troughs[0]
        lower_highs = len(recent_peaks) >= 2 and recent_peaks[-1] < recent_peaks[0]
        lower_lows = len(recent_troughs) >= 2 and recent_troughs[-1] < recent_troughs[0]
        
        # Calculate momentum (Ross looks at volume and price action)
        recent_volume = daily_df['volume'].tail(10)
        avg_volume = daily_df['volume'].tail(50).mean()
        volume_strength = recent_volume.mean() / avg_volume
        
        price_momentum = (current_price - closes.iloc[-10]) / closes.iloc[-10] * 100
        
        # Ross's context determination
        if higher_highs and higher_lows and price_momentum > 10 and volume_strength > 1.5:
            return MarketContext.STRONG_UPTREND
        elif higher_highs and higher_lows and price_momentum > 0:
            return MarketContext.UPTREND
        elif lower_highs and lower_lows and price_momentum < -10 and volume_strength > 1.5:
            return MarketContext.STRONG_DOWNTREND
        elif lower_highs and lower_lows and price_momentum < 0:
            return MarketContext.DOWNTREND
        else:
            return MarketContext.SIDEWAYS
    
    def identify_vital_levels(self) -> List[VitalLevel]:
        """
        Identify vital support/resistance levels across timeframes
        Ross: "A resistance level on daily chart is much more significant than one on 5-minute"
        """
        vital_levels = []
        
        # Check each timeframe for significant levels
        for timeframe, data in self.timeframes.items():
            df = data.get_dataframe()
            if df is None or len(df) < 20:
                continue
                
            levels = self._find_significant_levels(df, timeframe)
            vital_levels.extend(levels)
        
        # Sort by significance (daily > 5min > 1min)
        timeframe_priority = {Timeframe.DAILY: 3, Timeframe.FIVE_MIN: 2, Timeframe.ONE_MIN: 1}
        vital_levels.sort(key=lambda x: (timeframe_priority[x.timeframe_significance], x.strength), reverse=True)
        
        return vital_levels[:10]  # Top 10 most significant levels
    
    def _find_significant_levels(self, df: pd.DataFrame, timeframe: Timeframe) -> List[VitalLevel]:
        """Find significant S/R levels in a timeframe"""
        levels = []
        
        highs = df['high']
        lows = df['low']
        closes = df['close']
        
        # Find levels that have been tested multiple times
        price_range = highs.max() - lows.min()
        tolerance = price_range * 0.005  # 0.5% tolerance
        
        # Look for resistance levels (multiple touches of highs)
        for i in range(len(highs)):
            current_high = highs.iloc[i]
            touches = 0
            
            # Count how many times this level was tested
            for j in range(len(highs)):
                if abs(highs.iloc[j] - current_high) <= tolerance:
                    touches += 1
            
            # Significant if tested 3+ times
            if touches >= 3:
                strength = min(touches / 10.0, 1.0)  # Cap at 1.0
                
                # Boost strength for higher timeframes
                if timeframe == Timeframe.DAILY:
                    strength *= 1.5
                elif timeframe == Timeframe.FIVE_MIN:
                    strength *= 1.2
                
                levels.append(VitalLevel(
                    price=current_high,
                    level_type="resistance", 
                    timeframe_significance=timeframe,
                    touches=touches,
                    last_test=df.index[i],
                    strength=min(strength, 1.0)
                ))
        
        # Look for support levels (multiple touches of lows)
        for i in range(len(lows)):
            current_low = lows.iloc[i]
            touches = 0
            
            for j in range(len(lows)):
                if abs(lows.iloc[j] - current_low) <= tolerance:
                    touches += 1
            
            if touches >= 3:
                strength = min(touches / 10.0, 1.0)
                
                if timeframe == Timeframe.DAILY:
                    strength *= 1.5
                elif timeframe == Timeframe.FIVE_MIN:
                    strength *= 1.2
                
                levels.append(VitalLevel(
                    price=current_low,
                    level_type="support",
                    timeframe_significance=timeframe, 
                    touches=touches,
                    last_test=df.index[i],
                    strength=min(strength, 1.0)
                ))
        
        # Remove duplicates (similar price levels)
        unique_levels = []
        for level in levels:
            is_duplicate = False
            for existing in unique_levels:
                if abs(level.price - existing.price) <= tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_levels.append(level)
        
        return unique_levels
    
    def assess_timeframe_continuity(self) -> TimeframeContinuity:
        """
        Assess if timeframes are "telling a similar story" (Ross's words)
        """
        daily_context = self.analyze_daily_context()
        
        # Analyze 5min trend
        five_min_df = self.timeframes[Timeframe.FIVE_MIN].get_dataframe()
        five_min_bullish = self._is_timeframe_bullish(five_min_df) if five_min_df is not None else None
        
        # Analyze 1min trend  
        one_min_df = self.timeframes[Timeframe.ONE_MIN].get_dataframe()
        one_min_bullish = self._is_timeframe_bullish(one_min_df) if one_min_df is not None else None
        
        # Daily trend assessment
        daily_bullish = daily_context in [MarketContext.UPTREND, MarketContext.STRONG_UPTREND]
        daily_bearish = daily_context in [MarketContext.DOWNTREND, MarketContext.STRONG_DOWNTREND]
        
        # Count bullish vs bearish timeframes
        bullish_count = sum([daily_bullish, five_min_bullish, one_min_bullish])
        bearish_count = sum([daily_bearish, 
                           five_min_bullish == False if five_min_bullish is not None else False,
                           one_min_bullish == False if one_min_bullish is not None else False])
        
        # Ross's continuity assessment
        if bullish_count >= 2 and bearish_count == 0:
            return TimeframeContinuity.BULLISH_ALIGNMENT
        elif bullish_count >= 2:
            return TimeframeContinuity.MIXED_BULLISH
        elif bearish_count >= 2 and bullish_count == 0:
            return TimeframeContinuity.BEARISH_ALIGNMENT
        elif bearish_count >= 2:
            return TimeframeContinuity.MIXED_BEARISH
        else:
            return TimeframeContinuity.CONFLICTED
    
    def _is_timeframe_bullish(self, df: pd.DataFrame) -> bool:
        """Determine if a timeframe is showing bullish price action"""
        if df is None or len(df) < 10:
            return False
        
        closes = df['close']
        recent_closes = closes.tail(10)
        
        # Simple trend: recent price > older price
        current_price = recent_closes.iloc[-1]
        past_price = recent_closes.iloc[0]
        
        return current_price > past_price
    
    def detect_home_run_potential(self, bull_flag_signal: Optional[BullFlagSignal]) -> bool:
        """
        Detect if setup could become a "home run trade" (Ross's words)
        "Sometimes the setup on 5-minute turns into daily breakout"
        """
        if not bull_flag_signal or bull_flag_signal.validation != BullFlagValidation.VALID:
            return False
        
        daily_df = self.timeframes[Timeframe.DAILY].get_dataframe()
        if daily_df is None or len(daily_df) < 50:
            return False
        
        # Check if current price is near daily resistance
        daily_highs = daily_df['high'].tail(20)
        current_price = bull_flag_signal.entry_price or bull_flag_signal.flagpole_high
        
        # If within 5% of recent daily highs, could break to new highs
        recent_daily_high = daily_highs.max()
        proximity_to_daily_high = (recent_daily_high - current_price) / recent_daily_high
        
        # Home run potential if close to daily breakout and strong volume
        return proximity_to_daily_high <= 0.05 and bull_flag_signal.strength_score >= 0.7
    
    def detect_bull_flag_5min(self) -> Optional[BullFlagSignal]:
        """
        Detect bull flag on 5-minute chart (Ross's primary timeframe)
        """
        five_min_df = self.timeframes[Timeframe.FIVE_MIN].get_dataframe()
        
        if five_min_df is None or len(five_min_df) < 10:
            return None
        
        return self.bull_flag_detector.detect_bull_flag(five_min_df, self.symbol)
    
    def detect_bull_flag_1min(self) -> Optional[BullFlagSignal]:
        """
        Detect bull flag on 1-minute chart for scalping/precise entries
        """
        one_min_df = self.timeframes[Timeframe.ONE_MIN].get_dataframe()
        
        if one_min_df is None or len(one_min_df) < 20:
            return None
        
        return self.bull_flag_detector.detect_bull_flag(one_min_df, self.symbol)
    
    def get_comprehensive_analysis(self) -> MultiTimeframeAnalysis:
        """
        Get comprehensive multi-timeframe analysis using Ross's methodology
        Ross: "Ensure they're all telling a similar story"
        """
        # Step 1: Daily context (Ross's foundation)
        daily_context = self.analyze_daily_context()
        
        # Step 2: Assess timeframe continuity (Ross's key principle)
        timeframe_continuity = self.assess_timeframe_continuity()
        
        # Step 3: Identify vital levels across timeframes
        vital_levels = self.identify_vital_levels()
        
        # Step 4: Pattern detection on primary timeframe (5min)
        bull_flag_5min = self.detect_bull_flag_5min()
        bull_flag_1min = self.detect_bull_flag_1min()
        
        # Step 5: Determine best pattern and timeframes
        primary_signal = None
        primary_timeframe = Timeframe.FIVE_MIN
        entry_timeframe = Timeframe.ONE_MIN
        exit_timeframe = Timeframe.FIVE_MIN
        
        # Ross's logic: Use 5min for pattern, 1min for entry
        if bull_flag_5min and bull_flag_5min.validation == BullFlagValidation.VALID:
            primary_signal = bull_flag_5min
            primary_timeframe = Timeframe.FIVE_MIN
        elif bull_flag_1min and bull_flag_1min.validation == BullFlagValidation.VALID:
            primary_signal = bull_flag_1min
            primary_timeframe = Timeframe.ONE_MIN
            # If pattern is on 1min, use 1min for exits too (scalping)
            exit_timeframe = Timeframe.ONE_MIN
        
        # Step 6: Detect home run potential
        home_run_potential = self.detect_home_run_potential(primary_signal)
        
        # Step 7: Determine if bias shift needed (Ross's warning)
        bias_shift_needed = self._should_shift_bias(daily_context, timeframe_continuity, primary_signal)
        
        # Step 8: Calculate confluence score
        confluence_score = self._calculate_confluence_score(
            daily_context, timeframe_continuity, primary_signal, vital_levels
        )
        
        return MultiTimeframeAnalysis(
            daily_context=daily_context,
            timeframe_continuity=timeframe_continuity,
            primary_timeframe=primary_timeframe,
            bull_flag_signal=primary_signal,
            vital_levels=vital_levels,
            entry_timeframe=entry_timeframe,
            exit_timeframe=exit_timeframe,
            confluence_score=confluence_score,
            home_run_potential=home_run_potential,
            bias_shift_needed=bias_shift_needed
        )
    
    def _should_shift_bias(self, daily_context: MarketContext, 
                          timeframe_continuity: TimeframeContinuity,
                          signal: Optional[BullFlagSignal]) -> bool:
        """
        Determine if trader should shift bias (Ross's words)
        "You might be looking for long entries until daily chart prompts you to shift to shorts"
        """
        # If daily is bearish but looking for bull flags, should shift bias
        daily_bearish = daily_context in [MarketContext.DOWNTREND, MarketContext.STRONG_DOWNTREND]
        conflicted_timeframes = timeframe_continuity == TimeframeContinuity.CONFLICTED
        
        # Ross would shift from bullish to bearish bias
        return daily_bearish or conflicted_timeframes
    
    def _calculate_confluence_score(self, daily_context: MarketContext,
                                   timeframe_continuity: TimeframeContinuity,
                                   primary_signal: Optional[BullFlagSignal],
                                   vital_levels: List[VitalLevel]) -> float:
        """
        Calculate confluence score using Ross's multi-timeframe approach
        """
        score = 0.0
        
        # Daily context weight (35% - Ross's foundation)
        if daily_context == MarketContext.STRONG_UPTREND:
            score += 0.35
        elif daily_context == MarketContext.UPTREND:
            score += 0.25
        elif daily_context == MarketContext.SIDEWAYS:
            score += 0.1
        # Downtrend contexts get 0 points
        
        # Timeframe continuity weight (25% - Ross's "similar story")
        if timeframe_continuity == TimeframeContinuity.BULLISH_ALIGNMENT:
            score += 0.25
        elif timeframe_continuity == TimeframeContinuity.MIXED_BULLISH:
            score += 0.15
        elif timeframe_continuity == TimeframeContinuity.CONFLICTED:
            score += 0.05
        # Bearish continuity gets 0 for bull flags
        
        # Pattern strength weight (25%)
        if primary_signal and primary_signal.validation == BullFlagValidation.VALID:
            score += 0.25 * primary_signal.strength_score
        
        # Vital levels confluence weight (15% - Ross's significant levels)
        if vital_levels:
            # Check if near significant support (positive) or resistance (negative)
            current_price = primary_signal.entry_price if primary_signal else 0
            level_confluence = 0.0
            
            for level in vital_levels[:3]:  # Top 3 most significant
                distance_pct = abs(level.price - current_price) / current_price
                if distance_pct <= 0.02:  # Within 2%
                    if level.level_type == "support" and current_price >= level.price:
                        level_confluence += level.strength * 0.05
                    elif level.level_type == "resistance" and current_price <= level.price:
                        level_confluence -= level.strength * 0.05
            
            score += max(0, level_confluence)  # Only positive confluence
        
        return min(1.0, score)
    
    def _calculate_confluence_score(self, daily_context: MarketContext,
                                   primary_signal: Optional[BullFlagSignal],
                                   bull_flag_5min: Optional[BullFlagSignal],
                                   bull_flag_1min: Optional[BullFlagSignal]) -> float:
        """
        Calculate confluence score across timeframes
        """
        score = 0.0
        
        # Daily context weight (40%)
        if daily_context == MarketContext.STRONG_UPTREND:
            score += 0.4
        elif daily_context == MarketContext.UPTREND:
            score += 0.25
        elif daily_context == MarketContext.SIDEWAYS:
            score += 0.1
        # Downtrend contexts get 0 points
        
        # 5min pattern weight (35%)
        if bull_flag_5min and bull_flag_5min.validation == BullFlagValidation.VALID:
            score += 0.35 * bull_flag_5min.strength_score
        
        # 1min pattern weight (25%)
        if bull_flag_1min and bull_flag_1min.validation == BullFlagValidation.VALID:
            score += 0.25 * bull_flag_1min.strength_score
        
        return min(1.0, score)
    
    def should_enter_trade(self, analysis: MultiTimeframeAnalysis) -> bool:
        """
        Determine if should enter trade based on Ross's multi-timeframe analysis
        Ross: "Confirm short-term setups with longer time frames"
        """
        # Ross's core requirements
        basic_conditions = [
            analysis.bull_flag_signal is not None,
            analysis.bull_flag_signal.stage == BullFlagStage.BREAKOUT_CONFIRMED,
            analysis.bull_flag_signal.validation == BullFlagValidation.VALID,
        ]
        
        # Ross's timeframe continuity requirement
        timeframe_conditions = [
            analysis.timeframe_continuity in [TimeframeContinuity.BULLISH_ALIGNMENT, 
                                            TimeframeContinuity.MIXED_BULLISH],
            not analysis.bias_shift_needed,  # Don't trade if should shift bias
            analysis.confluence_score >= 0.6  # High confluence threshold
        ]
        
        # Ross's daily context requirement (but allow strong patterns in sideways)
        daily_condition = (
            analysis.daily_context in [MarketContext.UPTREND, MarketContext.STRONG_UPTREND] or
            (analysis.daily_context == MarketContext.SIDEWAYS and analysis.confluence_score >= 0.8)
        )
        
        return all(basic_conditions) and all(timeframe_conditions) and daily_condition
    
    def get_entry_signal(self, analysis: MultiTimeframeAnalysis) -> Optional[Dict]:
        """
        Get precise entry signal based on recommended timeframe
        """
        if not self.should_enter_trade(analysis):
            return None
        
        # Use 1-minute for precise entry (Ross's approach)
        one_min_signal = self.detect_bull_flag_1min()
        
        if (one_min_signal and 
            one_min_signal.stage == BullFlagStage.BREAKOUT_CONFIRMED and
            one_min_signal.entry_price is not None):
            
            self.position_entry_timeframe = analysis.entry_timeframe
            
            return {
                'entry_price': one_min_signal.entry_price,
                'stop_loss': one_min_signal.stop_loss,
                'timeframe': analysis.entry_timeframe,
                'pattern_strength': one_min_signal.strength_score,
                'confluence_score': analysis.confluence_score
            }
        
        return None
    
    def check_exit_signals(self, current_price: float) -> Optional[str]:
        """
        Check for exit signals based on position's timeframe
        Ross's key rule: "first red candle on the 5min chart"
        """
        if not self.position_entry_timeframe:
            return None
        
        # Get current timeframe data based on entry
        if self.position_entry_timeframe == Timeframe.FIVE_MIN:
            timeframe_data = self.timeframes[Timeframe.FIVE_MIN]
        else:
            # Use 1min for scalp exits, 5min for swing exits
            timeframe_data = self.timeframes[Timeframe.ONE_MIN]
        
        latest_bar = timeframe_data.get_latest_bar()
        if not latest_bar:
            return None
        
        # Check for red candle (Ross's exit rule)
        is_red_candle = latest_bar['close'] < latest_bar['open']
        
        # For 5min charts - Ross's exact rule
        if (self.position_entry_timeframe == Timeframe.FIVE_MIN and 
            is_red_candle and self.last_5min_candle_color != 'red'):
            self.last_5min_candle_color = 'red'
            return "first_red_5min_candle"
        
        # For 1min scalps - immediate red candle exit
        elif (self.position_entry_timeframe == Timeframe.ONE_MIN and 
              is_red_candle):
            return "first_red_1min_candle"
        
        # Update candle color tracking
        self.last_5min_candle_color = 'green' if not is_red_candle else 'red'
        
        return None
    
    def reset_position_tracking(self) -> None:
        """Reset position tracking when trade is closed"""
        self.position_entry_timeframe = None
        self.last_5min_candle_color = None

class ActionAlertMultiTimeframeMonitor:
    """
    Enhanced action alert monitor with multi-timeframe capabilities
    """
    
    def __init__(self):
        self.monitors: Dict[str, MultiTimeframePatternMonitor] = {}
        self.active_positions: Dict[str, Dict] = {}
        
    def process_action_alert(self, symbol: str, alert_time: datetime) -> str:
        """Start multi-timeframe monitoring for symbol"""
        if symbol not in self.monitors:
            self.monitors[symbol] = MultiTimeframePatternMonitor(symbol)
        
        logging.info(f"Started multi-timeframe monitoring for {symbol}")
        return f"{symbol}_{alert_time.strftime('%Y%m%d_%H%M%S')}"
    
    def update_price_data(self, symbol: str, timeframe: Timeframe, 
                         timestamp: datetime, open_price: float, high: float,
                         low: float, close: float, volume: int) -> None:
        """Update price data for specific symbol and timeframe"""
        if symbol in self.monitors:
            monitor = self.monitors[symbol]
            monitor.update_timeframe_data(timeframe, timestamp, open_price, 
                                        high, low, close, volume)
            
            # Check for entry/exit signals if this is the primary update
            if timeframe == Timeframe.ONE_MIN:  # Use 1min as trigger for analysis
                self._check_trading_signals(symbol, close)
    
    def _check_trading_signals(self, symbol: str, current_price: float) -> None:
        """Check for entry and exit signals"""
        monitor = self.monitors[symbol]
        
        # If no position, check for entry
        if symbol not in self.active_positions:
            analysis = monitor.get_comprehensive_analysis()
            entry_signal = monitor.get_entry_signal(analysis)
            
            if entry_signal:
                self.active_positions[symbol] = {
                    'entry_price': entry_signal['entry_price'],
                    'stop_loss': entry_signal['stop_loss'],
                    'entry_time': datetime.now(),
                    'timeframe': entry_signal['timeframe'],
                    'confluence_score': entry_signal['confluence_score']
                }
                logging.info(f"ENTERED {symbol} at ${entry_signal['entry_price']:.2f} "
                           f"(Confluence: {entry_signal['confluence_score']:.2f})")
        
        # If position exists, check for exit
        else:
            exit_reason = monitor.check_exit_signals(current_price)
            if exit_reason:
                position = self.active_positions.pop(symbol)
                pnl = current_price - position['entry_price']
                
                monitor.reset_position_tracking()
                logging.info(f"EXITED {symbol} at ${current_price:.2f} "
                           f"(P&L: ${pnl:.2f}, Reason: {exit_reason})")
    
    def get_analysis_summary(self, symbol: str) -> Optional[Dict]:
        """Get current multi-timeframe analysis for symbol"""
        if symbol not in self.monitors:
            return None
        
        analysis = self.monitors[symbol].get_comprehensive_analysis()
        
        return {
            'symbol': symbol,
            'daily_context': analysis.daily_context.value,
            'primary_timeframe': analysis.primary_timeframe.value,
            'has_bull_flag': analysis.bull_flag_signal is not None,
            'pattern_stage': analysis.bull_flag_signal.stage.value if analysis.bull_flag_signal else None,
            'confluence_score': analysis.confluence_score,
            'should_enter': self.monitors[symbol].should_enter_trade(analysis)
        }

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize multi-timeframe monitor
    monitor = ActionAlertMultiTimeframeMonitor()
    
    # Process ACTION alert
    session_id = monitor.process_action_alert("CLRO", datetime.now())
    
    print("Multi-Timeframe Pattern Monitor Initialized")
    print("\nFeatures:")
    print("- Daily chart context analysis")
    print("- 5-minute primary pattern detection")
    print("- 1-minute precise entry timing")
    print("- Ross's 'first red 5min candle' exit rule")
    print("- Multi-timeframe confluence scoring")
    print("- Automatic timeframe selection for entry/exit")
    
    # Example of feeding data
    timestamp = datetime.now()
    # monitor.update_price_data("CLRO", Timeframe.DAILY, timestamp, 15.0, 15.5, 14.8, 15.3, 100000)
    # monitor.update_price_data("CLRO", Timeframe.FIVE_MIN, timestamp, 15.0, 15.2, 14.9, 15.1, 5000)
    # monitor.update_price_data("CLRO", Timeframe.ONE_MIN, timestamp, 15.0, 15.1, 14.95, 15.05, 1000)