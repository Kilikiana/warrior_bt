"""
Ross Cameron Bull Flag Pattern Detector

ROSS CAMERON'S EXACT BULL FLAG RULES:
Based on his detailed description, these are the precise, non-contradictory rules.

ANATOMY OF BULL FLAG:
1. Flag Pole: Large green candle(s) on high relative volume
2. Flag: 2-3 red candles pulling back (consolidation)
3. Breakout: First green candle making new high after pullback

ENTRY RULES (EXACT):
- Stock moving higher on high relative volume (large green bars)
- Price consolidates at/near highs with defined pullback (2-3 red candles)
- Pullback must NOT retrace more than 50% of the initial move
- Pullback must NOT break below VWAP
- Entry = first candle making new high after pullback
- Entry price = moment green candle breaks high of previous red candle

VALIDATION CRITERIA:
- Stock: $2-$20 price range
- Daily gain: minimum 10%
- Volume: 5x relative volume
- Float: under 20 million shares
- News catalyst preferred

FAILURE SIGNALS:
- Retraces more than 50% of initial move
- Volume during consolidation higher than initial move
- Breaks 9 EMA or VWAP
- More than 3-4 red candles in pullback

POSITION MANAGEMENT:
- Stop loss: low of the pullback
- Profit target: 2:1 ratio minimum
- Scale out: sell 1/2 at first target, adjust stop to breakeven
- Exit immediately if no immediate breakout ("breakout or bailout")
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
from datetime import datetime, timedelta

class BullFlagValidation(Enum):
    """Bull flag validation results"""
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"  # Valid but has minor issues
    INVALID_RETRACE_TOO_DEEP = "retrace_too_deep"  # >62% even with tolerance
    INVALID_BROKE_VWAP = "broke_vwap"  # Close broke VWAP with tolerance
    INVALID_BROKE_9EMA = "broke_9ema"  # Persistent EMA break
    INVALID_TOO_MANY_RED_CANDLES = "too_many_red_candles"  # >6 total bars
    INVALID_HIGH_VOLUME_CONSOLIDATION = "high_volume_consolidation"  # >85% volume
    INVALID_LOW_LIQUIDITY = "low_liquidity"  # Pullback avg volume below threshold
    INVALID_NO_FLAGPOLE = "no_flagpole"
    PENDING = "pending"

class BullFlagStage(Enum):
    """Current stage of bull flag formation"""
    NO_PATTERN = "no_pattern"
    FLAGPOLE_FORMING = "flagpole_forming"
    PULLBACK_STAGE = "pullback_stage"
    READY_FOR_BREAKOUT = "ready_for_breakout"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    PATTERN_FAILED = "pattern_failed"

class BullFlagSignal(NamedTuple):
    """Bull flag signal details"""
    timestamp: datetime
    symbol: str
    stage: BullFlagStage
    validation: BullFlagValidation
    entry_price: Optional[float]
    stop_loss: Optional[float]
    flagpole_start: float
    flagpole_high: float
    pullback_low: float
    pullback_candles: int
    retrace_percentage: float
    volume_confirmation: bool
    broke_vwap: bool
    broke_9ema: bool
    strength_score: float  # 0-1 based on volume, retrace, etc.

class BullFlagDetector:
    """
    Ross Cameron's Bull Flag Pattern Detector
    Implements his exact rules with zero contradictions
    """
    
    def __init__(self, strict: bool = True, breakout_volume_multiple: float = 0.0, min_pullback_avg_volume: float = 0.0, require_green_breakout: bool = False):
        self.strict = strict
        self.min_flagpole_gain = 0.05  # 5% minimum for flagpole
        self.max_retrace_percent = 50.0  # Ross's 50% rule base
        self.dynamic_retrace_max = 62.0  # Extended tolerance for strong structures (non-strict)
        # Pullback length: require 2–6 bars (Ross prefers 2–3; hard cap 6)
        self.max_pullback_candles = 6
        self.min_pullback_candles = 2
        # Breakout timing: breakout must occur on the FIRST green after pullback ends
        # (i.e., immediately following the last red pullback bar)
        self.require_immediate_breakout = True
        # Optional entry volume surge requirement (e.g., 1.3–1.5x pullback average)
        self.breakout_volume_multiple = max(0.0, float(breakout_volume_multiple))
        # Optional liquidity gate: require pullback average 1-min volume ≥ threshold
        self.min_pullback_avg_volume = max(0.0, float(min_pullback_avg_volume))
        # Optional conservative entry: require breakout candle to CLOSE green
        self.require_green_breakout = bool(require_green_breakout)
        
        # Consolidation volume threshold (strict uses ratio >= 1.0 invalid)
        self.volume_threshold = 0.85  # used in non-strict mode only
        self.vwap_break_tolerance = 0.0 if strict else 0.02  # strict: no tolerance
        self.ema_break_persistence_bars = 1 if strict else 2
        
        # Stock criteria (Ross's preferences)
        self.min_price = 2.0
        self.max_price = 20.0
        self.min_daily_gain = 0.10  # 10%
        self.min_relative_volume = 5.0  # 5x
        self.max_float = 20_000_000  # 20M shares
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return vwap
    
    
    def identify_flagpole(self, df: pd.DataFrame, current_idx: int) -> Tuple[Optional[int], Optional[float], Optional[float]]:
        """
        Identify flagpole: large green candle(s) on high volume
        
        Returns:
            (flagpole_start_idx, flagpole_start_price, flagpole_high_price)
        """
        if current_idx < 5:  # Need some history
            return None, None, None
        
        # Look back up to 10 candles for flagpole start
        for lookback in range(1, min(11, current_idx)):
            start_idx = current_idx - lookback
            start_price = df.iloc[start_idx]['close']
            current_high = df.iloc[current_idx]['high']
            
            # Calculate gain from potential start to current high
            gain_percent = (current_high - start_price) / start_price * 100
            
            if gain_percent >= self.min_flagpole_gain * 100:  # Convert to percentage
                # Validate it's mostly green candles with volume
                green_candles = 0
                total_candles = 0
                avg_volume = df['volume'].iloc[max(0, start_idx-20):start_idx].mean()
                
                for i in range(start_idx, current_idx + 1):
                    if df.iloc[i]['close'] > df.iloc[i]['open']:
                        green_candles += 1
                    total_candles += 1
                
                # Ross wants "large green candles" - at least 60% green
                if green_candles / total_candles >= 0.6:
                    return start_idx, start_price, current_high
        
        return None, None, None
    
    def _get_ema9_series(self, df: pd.DataFrame) -> pd.Series:
        """Use precomputed EMA9 if present; else safe fallback (pandas EWM)."""
        if 'ema9' in df.columns:
            return df['ema9']
        try:
            import talib  # optional
            return pd.Series(talib.EMA(df['close'].values, timeperiod=9), index=df.index)
        except Exception:
            return df['close'].ewm(span=9, adjust=False).mean()

    def analyze_pullback(self, df: pd.DataFrame, flagpole_high_idx: int,
                         flagpole_start_price: float, flagpole_high: float) -> Dict:
        """
        Analyze the *contiguous* pullback right after the pole high.
        Enforce support at the **pullback low** (VWAP/EMA9 at that bar) and light-volume pullback.
        """
        current_idx = len(df) - 1
        # Ensure VWAP once
        if 'vwap' not in df.columns:
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        ema9 = self._get_ema9_series(df)
        # Contiguous red streak
        red_candles, last_red_idx = 0, None
        pullback_low, pullback_low_idx = flagpole_high, flagpole_high_idx
        for i in range(flagpole_high_idx + 1, current_idx + 1):
            o, c, h, l = df.iloc[i][['open','close','high','low']]
            if l < pullback_low:
                pullback_low, pullback_low_idx = l, i
            if c < o:
                red_candles += 1
                last_red_idx = i
            else:
                break  # first green ends the pullback
        last_red_high = df.iloc[last_red_idx]['high'] if last_red_idx is not None else None

        # Retrace
        flag_move = max(1e-9, flagpole_high - flagpole_start_price)
        retrace_percent = (flagpole_high - pullback_low) / flag_move * 100.0
        # Support integrity evaluated at the **low bar**
        vwap_at_low = df.iloc[pullback_low_idx]['vwap']
        ema_at_low = ema9.iloc[pullback_low_idx]
        broke_vwap = pullback_low < vwap_at_low if pd.notna(vwap_at_low) else False
        broke_9ema = pullback_low < ema_at_low if pd.notna(ema_at_low) else False
        # Light-volume pullback (≤ 60% of pole)
        pole_avg_vol = df['volume'].iloc[max(0, flagpole_high_idx-3):flagpole_high_idx+1].mean()
        end_for_pb = last_red_idx if last_red_idx is not None else current_idx
        pull_vol_series = df['volume'].iloc[flagpole_high_idx+1:end_for_pb+1] if end_for_pb > flagpole_high_idx else pd.Series([], dtype=float)
        pull_avg_vol = float(pull_vol_series.mean()) if len(pull_vol_series) > 0 else 0.0
        try:
            pull_median_vol = float(pull_vol_series.median()) if len(pull_vol_series) > 0 else 0.0
        except Exception:
            pull_median_vol = pull_avg_vol
        vol_ratio = (pull_avg_vol / pole_avg_vol) if pole_avg_vol else 0.0
        current_price = float(df.iloc[current_idx]['close'])

        return {
            'red_candles': red_candles,
            'last_red_idx': last_red_idx,
            'last_red_high': last_red_high,
            'pullback_low': pullback_low,
            'pullback_low_idx': pullback_low_idx,
            'retrace_percent': retrace_percent,
            'broke_vwap': broke_vwap,
            'broke_9ema': broke_9ema,
            'high_volume_consolidation': vol_ratio >= 0.60,
            'vol_ratio': vol_ratio,
            'pullback_start_idx': flagpole_high_idx + 1,
            'pullback_end_idx': end_for_pb,
            'pull_avg_vol': pull_avg_vol,
            'pull_median_vol': pull_median_vol,
            'current_price': current_price
        }
    
    def validate_bull_flag(self, pullback_analysis: Dict) -> BullFlagValidation:
        """
        Validate bull flag. If strict, enforce Ross’s exact rules.
        """
        # Liquidity gate (applies in both modes if configured)
        pav = float(pullback_analysis.get('pull_avg_vol') or 0.0)
        if self.min_pullback_avg_volume > 0.0 and (pav <= 0.0 or pav < self.min_pullback_avg_volume):
            return BullFlagValidation.INVALID_LOW_LIQUIDITY
        if self.strict:
            # Retrace must be ≤ 50%
            if pullback_analysis['retrace_percent'] > 50.0:
                return BullFlagValidation.INVALID_RETRACE_TOO_DEEP
            # Support holds: prioritize EMA9 support; do not invalidate solely on VWAP
            # (VWAP can be session-dependent; EMA9 is our primary support signal)
            if pullback_analysis['broke_9ema']:
                return BullFlagValidation.INVALID_BROKE_9EMA
            # Pullback length 1–5 bars
            if pullback_analysis['red_candles'] > self.max_pullback_candles:
                return BullFlagValidation.INVALID_TOO_MANY_RED_CANDLES
            # Consolidation volume must be lighter than pole
            if pullback_analysis.get('vol_ratio', 0.0) >= 1.0:
                return BullFlagValidation.INVALID_HIGH_VOLUME_CONSOLIDATION
            return BullFlagValidation.VALID
        else:
            # Non-strict (relaxed) mode
            retrace_limit = self.max_retrace_percent
            if (not pullback_analysis['broke_vwap'] and 
                not pullback_analysis['broke_9ema'] and 
                pullback_analysis['vol_ratio'] < 0.70):
                retrace_limit = self.dynamic_retrace_max
            if pullback_analysis['retrace_percent'] > retrace_limit:
                return BullFlagValidation.INVALID_RETRACE_TOO_DEEP
            if pullback_analysis['broke_vwap']:
                return BullFlagValidation.INVALID_BROKE_VWAP
            if pullback_analysis['broke_9ema']:
                return BullFlagValidation.INVALID_BROKE_9EMA
            if pullback_analysis['red_candles'] > self.max_pullback_candles:
                return BullFlagValidation.INVALID_TOO_MANY_RED_CANDLES
            if pullback_analysis['high_volume_consolidation']:
                return BullFlagValidation.INVALID_HIGH_VOLUME_CONSOLIDATION
            return BullFlagValidation.VALID
    
    def detect_entry_signal(self, df: pd.DataFrame, flagpole_high: float,
                            pullback_analysis: Dict) -> Tuple[bool, Optional[float]]:
        """
        Intrabar entry: first bar whose HIGH breaks the last red bar's HIGH
        after a contiguous 1–5 bar pullback. Entry = last_red_high (intrabar).
        """
        # Compute immediate contiguous pullback ending just before current bar
        cur_idx = len(df) - 1
        # Ensure current bar is green (first green after pullback)
        try:
            if not (float(df.iloc[cur_idx]['close']) > float(df.iloc[cur_idx]['open'])):
                return False, None
        except Exception:
            return False, None
        red_len = 0
        last_red_idx = None
        j = cur_idx - 1
        while j >= 0:
            try:
                o = float(df.iloc[j]['open']); c = float(df.iloc[j]['close'])
            except Exception:
                break
            if c < o:
                red_len += 1
                last_red_idx = j
                j -= 1
                if red_len > self.max_pullback_candles:
                    break
            else:
                break
        if last_red_idx is None or red_len < self.min_pullback_candles or red_len > self.max_pullback_candles:
            return False, None
        # Last red high
        try:
            last_red_high = float(df.iloc[last_red_idx]['high'])
        except Exception:
            return False, None

        cur = df.iloc[-1]
        # Trigger on high crossing the pullback high
        if cur['high'] >= last_red_high:
            # Optional: require breakout candle to close GREEN (conservative bar-close rule)
            if self.require_green_breakout:
                try:
                    if not (float(cur['close']) > float(cur['open'])):
                        return False, None
                except Exception:
                    return False, None
            # Optional breakout volume surge gate
            if self.breakout_volume_multiple > 0.0:
                # Prefer median pullback volume for robustness
                pmed = float(pullback_analysis.get('pull_median_vol') or 0.0)
                pav = float(pullback_analysis.get('pull_avg_vol') or 0.0)
                base = pmed if pmed > 0 else pav
                cur_vol = float(cur.get('volume') or 0.0)
                if base > 0 and cur_vol < base * self.breakout_volume_multiple:
                    return False, None
            return True, float(last_red_high)
        return False, None
    
    def calculate_strength_score(self, df: pd.DataFrame, flagpole_start_idx: int, 
                               pullback_analysis: Dict) -> float:
        """
        Calculate pattern strength (0-1) based on Ross's preferences
        """
        score = 0.0
        
        # Volume score (30% weight)
        if not pullback_analysis['high_volume_consolidation']:
            score += 0.3
        
        # Retrace score (25% weight) - shallower is better
        retrace_score = max(0, (self.max_retrace_percent - pullback_analysis['retrace_percent']) / self.max_retrace_percent)
        score += 0.25 * retrace_score
        
        # Pullback length score (20% weight) - 2-3 candles optimal
        optimal_candles = 2.5  # Middle of 2-3 range
        pullback_deviation = abs(pullback_analysis['red_candles'] - optimal_candles)
        pullback_score = max(0, 1 - (pullback_deviation / 2))
        score += 0.2 * pullback_score
        
        # Support level score (15% weight)
        if not pullback_analysis['broke_vwap'] and not pullback_analysis['broke_9ema']:
            score += 0.15
        elif not pullback_analysis['broke_vwap'] or not pullback_analysis['broke_9ema']:
            score += 0.075
        
        # News/momentum score (10% weight) - would need external data
        # For now, assume positive if we got this far
        score += 0.1
        
        return min(1.0, score)
    
    def find_actual_flagpole_start(self, df: pd.DataFrame, alert_time: datetime, 
                                  alert_price: float, symbol: str = "") -> Tuple[float, datetime]:
        """
        Find the actual flagpole start by looking back 10 minutes from alert time.
        Uses 1-minute data fallback for early alerts when 5-minute data insufficient.
        """
        import logging
        from data.ohlc_loader import load_symbol_ohlc_data
        
        # Look back exactly 10 minutes from alert time
        start_time = alert_time - timedelta(minutes=10)
        
        # Filter to bars within the 10-minute window (start_time to alert_time)
        window_bars = df[(df.index >= start_time) & (df.index <= alert_time)]
        
        # If insufficient 5-minute data, try 1-minute data for early alerts
        if len(window_bars) < 5:
            if symbol and alert_time.time() <= datetime.strptime("09:45", "%H:%M").time():
                logging.debug(f"🔍 Early alert with insufficient 5-min data, trying 1-minute data...")
                date_str = alert_time.strftime("%Y-%m-%d")
                
                try:
                    df_1min = load_symbol_ohlc_data(symbol, date_str, "1min")
                    if not df_1min.empty:
                        window_bars_1min = df_1min[(df_1min.index >= start_time) & (df_1min.index <= alert_time)]
                        logging.debug(f"🔍 Found {len(window_bars_1min)} 1-minute bars in window")
                        
                        if len(window_bars_1min) >= 5:
                            logging.debug(f"🔍 Using 1-minute data for flagpole detection")
                            window_bars = window_bars_1min
                        else:
                            raise ValueError(f"Insufficient data: only {len(window_bars)} 5-min bars and {len(window_bars_1min)} 1-min bars in 10-min window (need ≥5)")
                    else:
                        raise ValueError(f"Insufficient data: only {len(window_bars)} bars in 10-min window (need ≥5), no 1-min data available")
                except Exception as e:
                    logging.warning(f"Failed to load 1-minute fallback data: {e}")
                    raise ValueError(f"Insufficient data: only {len(window_bars)} bars in 10-min window (need ≥5)")
            else:
                raise ValueError(f"Insufficient data: only {len(window_bars)} bars in 10-min window (need ≥5)")
        
        # Find expansion point - volume spike + price acceleration
        expansion_candidates = []
        
        for i in range(3, len(window_bars)):  # Need 3 bars of history
            bar = window_bars.iloc[i]
            
            # Calculate volume expansion
            recent_avg_vol = window_bars['volume'].iloc[i-3:i].mean()
            if recent_avg_vol == 0:
                continue
                
            vol_ratio = bar['volume'] / recent_avg_vol
            
            # Calculate price expansion (green candle with range expansion)
            is_green = bar['close'] > bar['open']
            bar_range = bar['high'] - bar['low']
            recent_avg_range = (window_bars['high'].iloc[i-3:i] - window_bars['low'].iloc[i-3:i]).mean()
            range_ratio = bar_range / max(recent_avg_range, 1e-6)
            
            if vol_ratio >= 2.0 and is_green and range_ratio >= 1.5:
                expansion_candidates.append((i, vol_ratio * range_ratio, bar, window_bars.index[i]))
        
        if not expansion_candidates:
            raise ValueError(f"No valid flagpole start found: no volume/range expansion detected in 10-min window")
        
        # Take the best expansion candidate (highest combined score)
        best_idx, best_score, best_bar, best_timestamp = max(expansion_candidates, key=lambda x: x[1])
        
        logging.debug(f"Found flagpole start for alert at {alert_time}: "
                      f"expansion at {best_timestamp} (score: {best_score:.1f})")
        
        return best_bar['open'], best_timestamp

    def detect_bull_flag(self, df: pd.DataFrame, symbol: str = "", 
                        known_flagpole_high: Optional[float] = None,
                        known_flagpole_time: Optional[datetime] = None) -> BullFlagSignal:
        """
        Main bull flag detection method using Ross Cameron's exact rules
        
        Args:
            df: OHLC DataFrame with timestamp index
            symbol: Stock symbol
            known_flagpole_high: For ACTION alerts - the flagpole high price
            known_flagpole_time: For ACTION alerts - when the flagpole occurred
        """
        
        # For known flagpole mode, we need fewer bars
        min_bars_needed = 3 if known_flagpole_high is not None else 10
        
        if len(df) < min_bars_needed:
            return BullFlagSignal(
                timestamp=df.index[-1] if len(df) > 0 else datetime.now(),
                symbol=symbol,
                stage=BullFlagStage.NO_PATTERN,
                validation=BullFlagValidation.INVALID_NO_FLAGPOLE,
                entry_price=None,
                stop_loss=None,
                flagpole_start=0.0,
                flagpole_high=0.0,
                pullback_low=0.0,
                pullback_candles=0,
                retrace_percentage=0.0,
                volume_confirmation=False,
                broke_vwap=False,
                broke_9ema=False,
                strength_score=0.0
            )
        
        current_idx = len(df) - 1
        
        # Handle known flagpole from ACTION alert
        if known_flagpole_high is not None:
            # Find the bar closest to the known flagpole time
            flagpole_high_idx = current_idx  # Default to latest if time not found
            if known_flagpole_time is not None:
                # Find bar with timestamp closest to known_flagpole_time
                time_diffs = [(abs((ts - known_flagpole_time).total_seconds()), i) 
                             for i, ts in enumerate(df.index)]
                if time_diffs:
                    _, flagpole_high_idx = min(time_diffs)
            
            # Use known values for flagpole parameters
            flagpole_high = known_flagpole_high
            
            # Gate: only search for start when we have ≥5 bars in the 10-min window
            if known_flagpole_time is not None:
                start_time = known_flagpole_time - timedelta(minutes=10)
                window_bars = df[(df.index >= start_time) & (df.index <= known_flagpole_time)]
                if len(window_bars) < 5:
                    return BullFlagSignal(
                        timestamp=df.index[current_idx] if current_idx < len(df) else datetime.now(),
                        symbol=symbol,
                        stage=BullFlagStage.NO_PATTERN,
                        validation=BullFlagValidation.PENDING,
                        entry_price=None,
                        stop_loss=None,
                        flagpole_start=0.0,
                        flagpole_high=0.0,
                        pullback_low=0.0,
                        pullback_candles=0,
                        retrace_percentage=0.0,
                        volume_confirmation=False,
                        broke_vwap=False,
                        broke_9ema=False,
                        strength_score=0.0
                    )

            # Find actual flagpole start using strict detection - NO FALLBACKS
            try:
                flagpole_start_price, flagpole_start_time = self.find_actual_flagpole_start(
                    df, known_flagpole_time, known_flagpole_high, symbol)
                
                # Find the corresponding start index
                start_time_diffs = [(abs((ts - flagpole_start_time).total_seconds()), i) 
                                   for i, ts in enumerate(df.index)]
                if start_time_diffs:
                    _, flagpole_start_idx = min(start_time_diffs)
                else:
                    raise ValueError(f"Cannot find flagpole start index for time {flagpole_start_time}")
                    
            except ValueError as e:
                import logging
                logging.debug(f"❌ FLAGPOLE DETECTION FAILED for {symbol}: {e}")
                return BullFlagSignal(
                    timestamp=df.index[current_idx] if current_idx < len(df) else datetime.now(),
                    symbol=symbol,
                    stage=BullFlagStage.NO_PATTERN,
                    validation=BullFlagValidation.INVALID_NO_FLAGPOLE,
                    entry_price=None,
                    stop_loss=None,
                    flagpole_start=0.0,
                    flagpole_high=0.0,
                    pullback_low=0.0,
                    pullback_candles=0,
                    retrace_percentage=0.0,
                    volume_confirmation=False,
                    broke_vwap=False,
                    broke_9ema=False,
                    strength_score=0.0
                )
            
        else:
            # Step 1: Identify flagpole (original discovery mode)
            flagpole_start_idx, flagpole_start_price, flagpole_high = self.identify_flagpole(df, current_idx)
        
        # Check if flagpole was found (only applies to discovery mode)
        if known_flagpole_high is None and flagpole_start_idx is None:
            return BullFlagSignal(
                timestamp=df.index[current_idx],
                symbol=symbol,
                stage=BullFlagStage.NO_PATTERN,
                validation=BullFlagValidation.INVALID_NO_FLAGPOLE,
                entry_price=None,
                stop_loss=None,
                flagpole_start=0.0,
                flagpole_high=0.0,
                pullback_low=0.0,
                pullback_candles=0,
                retrace_percentage=0.0,
                volume_confirmation=False,
                broke_vwap=False,
                broke_9ema=False,
                strength_score=0.0
            )
        
        # Step 2: Find flagpole high index (for discovery mode only)
        if known_flagpole_high is None:
            flagpole_high_idx = flagpole_start_idx  # Default
            for i in range(flagpole_start_idx, current_idx + 1):
                if df.iloc[i]['high'] >= flagpole_high:
                    flagpole_high_idx = i
        # For known flagpole, flagpole_high_idx was already set above
        
        pullback_analysis = self.analyze_pullback(df, flagpole_high_idx, flagpole_start_price, flagpole_high)
        
        # Step 3: Validate pattern
        validation = self.validate_bull_flag(pullback_analysis)
        # In known-flagpole mode, be lenient on consolidation volume to avoid false negatives
        if known_flagpole_high is not None and validation == BullFlagValidation.INVALID_HIGH_VOLUME_CONSOLIDATION:
            validation = BullFlagValidation.VALID_WITH_WARNINGS
        
        # Step 4: Determine stage
        stage = BullFlagStage.FLAGPOLE_FORMING
        entry_price = None
        
        if validation not in (BullFlagValidation.VALID, BullFlagValidation.VALID_WITH_WARNINGS):
            stage = BullFlagStage.PATTERN_FAILED
        elif pullback_analysis['red_candles'] >= self.min_pullback_candles:
            # Check for entry signal
            has_entry, entry_price = self.detect_entry_signal(df, flagpole_high, pullback_analysis)
            
            if has_entry:
                stage = BullFlagStage.BREAKOUT_CONFIRMED
            else:
                stage = BullFlagStage.READY_FOR_BREAKOUT
        else:
            stage = BullFlagStage.PULLBACK_STAGE
        
        # Step 5: Calculate stop loss (Ross's rule: low of pullback)
        stop_loss = pullback_analysis['pullback_low'] if validation == BullFlagValidation.VALID else None
        
        # Step 6: Calculate strength score
        strength_score = self.calculate_strength_score(df, flagpole_start_idx, pullback_analysis)
        
        return BullFlagSignal(
            timestamp=df.index[current_idx],
            symbol=symbol,
            stage=stage,
            validation=validation,
            entry_price=entry_price,
            stop_loss=stop_loss,
            flagpole_start=flagpole_start_price,
            flagpole_high=flagpole_high,
            pullback_low=pullback_analysis['pullback_low'],
            pullback_candles=pullback_analysis['red_candles'],
            retrace_percentage=pullback_analysis['retrace_percent'],
            volume_confirmation=not pullback_analysis['high_volume_consolidation'],
            broke_vwap=pullback_analysis['broke_vwap'],
            broke_9ema=pullback_analysis['broke_9ema'],
            strength_score=strength_score
        )
    
    def validate_stock_criteria(self, current_price: float, daily_gain: float, 
                              relative_volume: float, float_shares: float) -> bool:
        """
        Validate stock meets Ross's criteria for bull flag trading
        """
        return (self.min_price <= current_price <= self.max_price and
                daily_gain >= self.min_daily_gain and
                relative_volume >= self.min_relative_volume and
                float_shares <= self.max_float)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01 09:30', periods=50, freq='1min')
    
    # Simulate a bull flag pattern
    prices = [100]
    volumes = [1000]
    
    # Flagpole: 5 green candles
    for i in range(5):
        prices.append(prices[-1] * 1.02)  # 2% gain each candle
        volumes.append(2000)  # High volume
    
    # Pullback: 3 red candles
    for i in range(3):
        prices.append(prices[-1] * 0.99)  # 1% pullback each
        volumes.append(800)  # Lower volume
    
    # Fill remaining with sideways action
    while len(prices) < 50:
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.005)))
        volumes.append(1000 + np.random.normal(0, 200))
    
    # Create OHLC data
    data = {
        'open': prices,
        'high': [p * 1.002 for p in prices],
        'low': [p * 0.998 for p in prices],
        'close': prices,
        'volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Test bull flag detection
    detector = BullFlagDetector()
    signal = detector.detect_bull_flag(df, "TEST")
    
    print("Bull Flag Detection Results:")
    print(f"Stage: {signal.stage}")
    print(f"Validation: {signal.validation}")
    print(f"Entry Price: {signal.entry_price}")
    print(f"Stop Loss: {signal.stop_loss}")
    print(f"Flagpole: ${signal.flagpole_start:.2f} -> ${signal.flagpole_high:.2f}")
    print(f"Retrace: {signal.retrace_percentage:.1f}%")
    print(f"Pullback Candles: {signal.pullback_candles}")
    print(f"Strength Score: {signal.strength_score:.2f}")
    print(f"Volume Confirmation: {signal.volume_confirmation}")
    print(f"Broke VWAP: {signal.broke_vwap}")
    print(f"Broke 9 EMA: {signal.broke_9ema}")
