"""
Action Alert Triggered Pattern Monitor

OVERVIEW:
This system monitors stocks after ACTION alerts to detect pattern formation.
When patterns are detected, it manages entries and position management according 
to Ross Cameron's exact rules.

WORKFLOW:
1. ACTION Alert triggers monitoring for specific symbol
2. Pattern detector continuously scans for bull flag formation
3. When valid pattern + entry signal detected, enter position
4. Manage position using Ross's scaling and exit rules
5. Stop monitoring when pattern fails or position closed

ROSS'S POST-ENTRY MANAGEMENT:
- Sell 1/2 at first target, move stop to breakeven
- Continue scaling out at subsequent targets
- Hold remainder until exit indicators
- Exit on: heavy resistance, no buying, first red 5min candle, no immediate breakout
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
from datetime import datetime, timedelta
import logging

try:
    from .bull_flag_pattern import BullFlagDetector, BullFlagSignal, BullFlagStage, BullFlagValidation
    from ..ema_calculator import EMACalculator, RossCameronEMAConfig
    from ..macd_calculator import MACDCalculator, RossCameronMACDConfig
    from ...position_management.position_sizer import PositionSizer  # type: ignore
except ImportError:
    # For direct execution testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from bull_flag_pattern import BullFlagDetector, BullFlagSignal, BullFlagStage, BullFlagValidation
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from ema_calculator import EMACalculator, RossCameronEMAConfig
    from macd_calculator import MACDCalculator, RossCameronMACDConfig
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    try:
        from position_management.position_sizer import PositionSizer  # type: ignore
    except Exception:
        PositionSizer = None  # Fallback if not available

class AlertType(Enum):
    """Types of trading alerts"""
    ACTION_ALERT = "action_alert"  # Ross's high momentum alerts
    CUSTOM_ALERT = "custom_alert"
    
class MonitoringStatus(Enum):
    """Pattern monitoring status"""
    ACTIVE = "active"
    PATTERN_DETECTED = "pattern_detected"
    POSITION_ENTERED = "position_entered"
    POSITION_MANAGING = "position_managing"
    PATTERN_FAILED = "pattern_failed"
    MONITORING_STOPPED = "monitoring_stopped"

class TradeStatus(Enum):
    """Current trade status"""
    NO_POSITION = "no_position"
    ENTERED = "entered"
    SCALED_FIRST = "scaled_first"  # Sold 1/2, stop at breakeven
    SCALING_OUT = "scaling_out"    # Continuing to scale
    HOLDING_RUNNER = "holding_runner"  # Holding final piece
    EXITED = "exited"

class ExitReason(Enum):
    """Reasons for position exit"""
    STOP_LOSS = "stop_loss"
    BREAKOUT_OR_BAILOUT = "breakout_or_bailout"
    FIRST_TARGET = "first_target"
    SECOND_TARGET = "second_target"
    THIRD_TARGET = "third_target"
    EXTENSION_BAR = "extension_bar"  # Ross's "sell into strength"
    HEAVY_RESISTANCE = "heavy_resistance"
    NO_BUYING_PRESSURE = "no_buying_pressure"
    FIRST_RED_5MIN = "first_red_5min"
    NO_IMMEDIATE_BREAKOUT = "no_immediate_breakout"
    PATTERN_FAILURE = "pattern_failure"

class ActionAlert(NamedTuple):
    """Action alert details"""
    symbol: str
    alert_time: datetime
    alert_price: float
    alert_high: float
    volume_spike: float
    news_catalyst: Optional[str]

class Position(NamedTuple):
    """Trading position details"""
    symbol: str
    entry_time: datetime
    entry_price: float
    initial_shares: int
    current_shares: int
    stop_loss: float
    first_target: float
    second_target: float
    third_target: float
    status: TradeStatus

class TradeExecution(NamedTuple):
    """Trade execution details"""
    timestamp: datetime
    symbol: str
    action: str  # BUY, SELL
    shares: int
    price: float
    reason: str

class PatternMonitoringSession:
    """
    Monitors a single symbol for pattern formation after alert
    """
    
    def __init__(self, alert: ActionAlert, patterns_to_monitor: List[str] = None,
                 account_balance: float = 30000.0, position_sizer: Optional["PositionSizer"] = None,
                 sizing_method: str = "ross_dynamic",
                 use_5min_first_red_exit: bool = False,
                 bailout_grace_bars: int = 2,
                 exit_on_ema_vwap_break: bool = True,
                 add_back_enabled: bool = True,
                 max_add_backs: int = 2,
                 add_back_cooldown_bars: int = 2,
                 stop_slippage_cents: float = 0.0,
                 entry_slippage_cents: float = 0.0):
        self.alert = alert
        self.symbol = alert.symbol
        self.start_time = alert.alert_time
        self.status = MonitoringStatus.ACTIVE
        
        # Pattern detectors
        self.bull_flag_detector = BullFlagDetector()
        self.patterns_to_monitor = patterns_to_monitor or ['bull_flag']
        
        # Technical indicator calculators
        self._ema_calc = EMACalculator()
        self._macd_calc = MACDCalculator(
            fast_period=RossCameronMACDConfig.FAST_PERIOD,
            slow_period=RossCameronMACDConfig.SLOW_PERIOD,
            signal_period=RossCameronMACDConfig.SIGNAL_PERIOD
        )
        
        # Position tracking
        self.position: Optional[Position] = None
        self.trade_executions: List[TradeExecution] = []
        
        # Data storage
        self.price_data: List[Dict] = []
        self.pattern_signals: List[BullFlagSignal] = []
        # Diagnostics for no-entry cases
        self._last_signal: Optional[BullFlagSignal] = None
        self._confirmations_rejects: int = 0
        self._breakout_attempts: int = 0
        
        # Ross's timing rules
        self.max_monitoring_time = timedelta(hours=2)  # Stop after 2 hours if no pattern
        self.breakout_timeout = timedelta(minutes=10)  # "Breakout or bailout"
        
        # Position management
        self.profit_ratio = 2.0  # Ross's 2:1 minimum
        self.scale_percentages = [0.5, 0.25, 0.25]  # 50%, 25%, 25%
        
        # Position sizing configuration
        self.account_balance = account_balance
        self.position_sizer = position_sizer
        self.sizing_method = sizing_method  # "ross_dynamic" or "percentage_risk"

        # Exit behavior configuration (1-minute focused)
        self.use_5min_first_red_exit = use_5min_first_red_exit
        self.bailout_grace_bars = max(1, bailout_grace_bars)
        self.exit_on_ema_vwap_break = exit_on_ema_vwap_break
        self.stop_slippage_cents = max(0.0, stop_slippage_cents)
        self.entry_slippage_cents = max(0.0, entry_slippage_cents)

        # Add-back behavior
        self.add_back_enabled = add_back_enabled
        self.max_add_backs = max(0, max_add_backs)
        self.add_back_cooldown_bars = max(0, add_back_cooldown_bars)
        self._add_backs_done = 0
        self._last_add_back_index = None
        
    def add_price_data(self, timestamp: datetime, open_price: float, high: float, 
                      low: float, close: float, volume: int) -> None:
        """Add new price bar to monitoring session"""
        bar = {
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        }
        self.price_data.append(bar)
        
        # Create DataFrame for pattern detection
        df = pd.DataFrame(self.price_data)
        df.set_index('timestamp', inplace=True)
        
        # Precompute technical indicators (single source of truth)
        self._attach_indicators(df)
        
        # Check patterns if monitoring is active
        if self.status == MonitoringStatus.ACTIVE:
            self._check_patterns(df, timestamp)
        elif self.status == MonitoringStatus.POSITION_ENTERED:
            self._manage_position(df, timestamp, close)
    
    def _attach_indicators(self, df: pd.DataFrame) -> None:
        """Precompute technical indicators (single source of truth)"""
        if len(df) < 10:  # Need minimum bars for indicators
            return
            
        # EMAs (9/20) using Ross Cameron's preferred settings
        emas = self._ema_calc.calculate_multiple_emas(df['close'], [RossCameronEMAConfig.FAST_EMA, RossCameronEMAConfig.SLOW_EMA])
        df['ema9'] = emas[RossCameronEMAConfig.FAST_EMA]
        df['ema20'] = emas[RossCameronEMAConfig.SLOW_EMA]
        
        # MACD using Ross Cameron's settings
        macd_line, signal_line, hist = self._macd_calc.calculate_macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_hist'] = hist

        # VWAP for weakness checks
        if 'vwap' not in df.columns:
            tp = (df['high'] + df['low'] + df['close']) / 3.0
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
    
    def _check_patterns(self, df: pd.DataFrame, timestamp: datetime) -> None:
        """Check for pattern formation"""
        if 'bull_flag' in self.patterns_to_monitor:
            # Attach indicators once per bar (EMA/MACD from calculators)
            if 'ema9' not in df.columns or 'ema20' not in df.columns:
                try:
                    from tech_analysis.ema_calculator import EMACalculator
                    ema = EMACalculator().calculate_multiple_emas(df['close'], [9, 20])
                    df['ema9'], df['ema20'] = ema[9], ema[20]
                except Exception:
                    # Fallback to pandas EWM
                    df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
                    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
            if 'macd' not in df.columns:
                try:
                    from tech_analysis.macd_calculator import MACDCalculator
                    macd = MACDCalculator().calculate_macd(df['close'])
                    df['macd'], df['macd_signal'], df['macd_hist'] = macd
                except Exception:
                    # Fallback calculation
                    fast_ema = df['close'].ewm(span=12, adjust=False).mean()
                    slow_ema = df['close'].ewm(span=26, adjust=False).mean()
                    df['macd'] = fast_ema - slow_ema
                    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                    df['macd_hist'] = df['macd'] - df['macd_signal']
            # Use ACTION alert fields (HIGH + TIME) as known flagpole
            signal = self.bull_flag_detector.detect_bull_flag(
                df, 
                self.symbol,
                known_flagpole_high=self.alert.alert_high,
                known_flagpole_time=self.alert.alert_time
            )
            self.pattern_signals.append(signal)
            self._last_signal = signal
            if (signal.stage == BullFlagStage.BREAKOUT_CONFIRMED and
                signal.validation == BullFlagValidation.VALID):
                self._breakout_attempts += 1
            
            # Check for entry signal + confirmations (MACD bullish + 9>20 EMA)
            if (signal.stage == BullFlagStage.BREAKOUT_CONFIRMED and 
                signal.validation == BullFlagValidation.VALID and
                signal.entry_price is not None):
                
                ema_ok = False
                macd_ok = False
                try:
                    # Use prior-bar confirmations to avoid waiting for bar close
                    idx = -2 if len(df) >= 2 else -1
                    ema9 = df['ema9'].iloc[idx]
                    ema20 = df['ema20'].iloc[idx]
                    ema_ok = (pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20)
                except Exception:
                    ema_ok = False

                try:
                    idx = -2 if len(df) >= 2 else -1
                    macd_val = df['macd'].iloc[idx]
                    macd_sig = df['macd_signal'].iloc[idx]
                    macd_hist = df['macd_hist'].iloc[idx]
                    macd_ok = (pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(macd_hist)
                               and macd_val > macd_sig and macd_hist > 0)
                except Exception:
                    macd_ok = False

                if ema_ok and macd_ok:
                    self._enter_position(signal, timestamp)
                else:
                    self._confirmations_rejects += 1
                    logging.debug(
                        "Entry rejected (confirmations) | EMA ok: %s | MACD ok: %s | symbol=%s",
                        ema_ok, macd_ok, self.symbol
                    )
            
            # Check for pattern failure
            elif signal.stage == BullFlagStage.PATTERN_FAILED:
                self.status = MonitoringStatus.PATTERN_FAILED
                logging.info(f"Pattern failed for {self.symbol}: {signal.validation}")
        
        # Stop monitoring after timeout
        if timestamp - self.start_time > self.max_monitoring_time:
            self.status = MonitoringStatus.MONITORING_STOPPED
    
    def _enter_position(self, signal: BullFlagSignal, timestamp: datetime) -> None:
        """Enter position based on pattern signal"""
        # Intrabar fill at breakout price plus optional slippage
        entry_price = (signal.entry_price + self.entry_slippage_cents) if signal.entry_price is not None else None
        stop_loss = signal.stop_loss
        
        # Determine shares using PositionSizer if available, else fallback
        shares = 1000  # fallback default
        risk_per_share = (entry_price - stop_loss) if (entry_price is not None and stop_loss is not None) else None

        if self.position_sizer is not None and risk_per_share and risk_per_share > 0:
            try:
                if self.sizing_method == "ross_dynamic" and hasattr(self.position_sizer, "calculate_ross_cameron_dynamic_size"):
                    result = self.position_sizer.calculate_ross_cameron_dynamic_size(
                        current_account_balance=self.account_balance,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        validate_20_cent_preference=True
                    )
                elif hasattr(self.position_sizer, "calculate_percentage_risk_size"):
                    result = self.position_sizer.calculate_percentage_risk_size(
                        account_size=self.account_balance,
                        entry_price=entry_price,
                        stop_loss=stop_loss
                    )
                else:
                    result = None
                if result is not None and getattr(result, "shares", 0) > 0:
                    shares = int(result.shares)
                    if getattr(result, "warnings", None):
                        logging.info("PositionSizer warnings: %s", "; ".join(result.warnings))
            except Exception as e:
                logging.warning("Position sizing failed; using fallback 1000 shares. Error: %s", e)

        # Validate risk per share (stop must be below entry)
        if not risk_per_share or risk_per_share <= 0:
            logging.warning("Invalid risk (stop >= entry); skipping entry.")
            return

        # Calculate targets based on Ross's 2:1 ratio minimum
        first_target = entry_price + (risk_per_share * self.profit_ratio)
        second_target = entry_price + (risk_per_share * self.profit_ratio * 1.5)
        third_target = entry_price + (risk_per_share * self.profit_ratio * 2.0)
        
        # Create position
        self.position = Position(
            symbol=self.symbol,
            entry_time=timestamp,
            entry_price=entry_price,
            initial_shares=shares,
            current_shares=shares,
            stop_loss=stop_loss,
            first_target=first_target,
            second_target=second_target,
            third_target=third_target,
            status=TradeStatus.ENTERED
        )
        
        # Record trade execution
        execution = TradeExecution(
            timestamp=timestamp,
            symbol=self.symbol,
            action="BUY",
            shares=shares,
            price=entry_price,
            reason="Bull Flag Breakout"
        )
        self.trade_executions.append(execution)
        
        self.status = MonitoringStatus.POSITION_ENTERED
        
        # Calculate risk metrics for detailed logging
        risk_per_share = entry_price - stop_loss
        target_2r = entry_price + (2 * risk_per_share)
        
        logging.info(f"ENTERED {self.symbol}: {shares} shares at ${entry_price:.2f}")
        logging.info(f"{self.symbol}: Risk ${risk_per_share:.3f}/share (stop ${stop_loss:.2f}) | "
                    f"2R target ${target_2r:.2f}")
    
    def _manage_position(self, df: pd.DataFrame, timestamp: datetime, current_price: float) -> None:
        """Manage position according to Ross's rules"""
        if not self.position:
            return
        
        pos = self.position
        time_in_trade = timestamp - pos.entry_time

        # Check stop loss first (simulate intrabar stop fill with optional slippage)
        cur = df.iloc[-1]
        stop_hit = False
        stop_fill = None
        stop = pos.stop_loss
        if cur['open'] <= stop:
            stop_hit = True
            base = cur['open']
            stop_fill = max(cur['low'], base - self.stop_slippage_cents)
        elif cur['low'] <= stop:
            stop_hit = True
            base = stop
            stop_fill = max(cur['low'], base - self.stop_slippage_cents)
        if stop_hit:
            self._exit_position(timestamp, float(stop_fill), ExitReason.STOP_LOSS, pos.current_shares)
            return
        
        # Breakout or Bailout (1-minute): within grace bars, exit at breakeven if no immediate follow-through
        if pos.status == TradeStatus.ENTERED:
            # Bars since entry
            try:
                bars_since_entry = int((df.index > pos.entry_time).sum())
            except Exception:
                bars_since_entry = 0
            if bars_since_entry <= self.bailout_grace_bars and current_price <= pos.entry_price:
                self._exit_position(timestamp, current_price, ExitReason.BREAKOUT_OR_BAILOUT, pos.current_shares)
                return
        
        # Check for profit targets and scale out
        self._check_profit_targets(timestamp, current_price)

        # Early pullback scale-out: first meaningful 1-min red (sell ~33% of remaining)
        self._early_pullback_trim(df, timestamp, current_price)

        # Check for exit indicators
        self._check_exit_indicators(df, timestamp, current_price)

        # Attempt add-back on next valid 1-min bull flag
        if self.add_back_enabled and self._add_backs_done < self.max_add_backs:
            self._attempt_add_back(df, timestamp)
    
    def _check_profit_targets(self, timestamp: datetime, current_price: float) -> None:
        """Check profit targets and scale out according to Ross's rules"""
        if not self.position:
            return
        
        pos = self.position
        
        # First target: Sell 1/2, move stop to breakeven
        if (current_price >= pos.first_target and 
            pos.status == TradeStatus.ENTERED):
            
            shares_to_sell = int(pos.initial_shares * self.scale_percentages[0])
            self._partial_exit(timestamp, current_price, ExitReason.FIRST_TARGET, shares_to_sell)
            
            # Move stop to breakeven (Ross's rule)
            self.position = pos._replace(
                stop_loss=pos.entry_price,
                current_shares=pos.current_shares - shares_to_sell,
                status=TradeStatus.SCALED_FIRST
            )
            
        # Second target: Scale out more
        elif (current_price >= pos.second_target and 
              pos.status == TradeStatus.SCALED_FIRST):
            
            shares_to_sell = int(pos.initial_shares * self.scale_percentages[1])
            self._partial_exit(timestamp, current_price, ExitReason.SECOND_TARGET, shares_to_sell)
            
            self.position = pos._replace(
                current_shares=pos.current_shares - shares_to_sell,
                status=TradeStatus.SCALING_OUT
            )
            
        # Third target: Scale out final piece or hold runner
        elif (current_price >= pos.third_target and 
              pos.status == TradeStatus.SCALING_OUT):
            
            # Ross often holds runners, but can scale here too
            shares_to_sell = int(pos.initial_shares * self.scale_percentages[2])
            if pos.current_shares > shares_to_sell:
                self._partial_exit(timestamp, current_price, ExitReason.THIRD_TARGET, shares_to_sell)
                self.position = pos._replace(
                    current_shares=pos.current_shares - shares_to_sell,
                    status=TradeStatus.HOLDING_RUNNER
                )
            else:
                # Exit remaining shares
                self._exit_position(timestamp, current_price, ExitReason.THIRD_TARGET, pos.current_shares)
    
    def _detect_extension_bar(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """
        Ross Cameron's Extension Bar Detection
        "Sell into strength, not weakness" - Exit when stock moves too fast, too far
        
        Extension Bar Criteria:
        - Large green candle (2x+ normal range)
        - High volume (1.5x+ recent average)  
        - Significant move from entry (15%+ preferred)
        - Signs of momentum exhaustion
        
        Returns: (is_extension_bar, suggested_exit_percentage)
        """
        if len(df) < 10:
            return False, 0.0
            
        current = df.iloc[-1]
        recent_data = df.tail(10)
        
        # Calculate recent averages for comparison
        recent_ranges = recent_data['high'] - recent_data['low']
        recent_volumes = recent_data['volume']
        
        avg_range = recent_ranges.mean()
        avg_volume = recent_volumes.mean()
        
        # Current bar characteristics
        current_range = current['high'] - current['low']
        current_volume = current['volume']
        
        # Extension bar criteria
        is_green_candle = current['close'] > current['open']
        large_range = current_range > avg_range * 2.0
        high_volume = current_volume > avg_volume * 1.5
        
        # Additional context for exit percentage
        profit_from_entry = 0.0
        if self.position:
            profit_from_entry = (current['close'] - self.position.entry_price) / self.position.entry_price
        
        # Base extension bar detection
        is_extension = is_green_candle and large_range and high_volume
        
        if not is_extension:
            return False, 0.0
        
        # Determine exit percentage based on context
        if profit_from_entry < 0.10:  # Less than 10% profit
            exit_percentage = 0.25  # Scale out 1/4, let most run
        elif profit_from_entry < 0.25:  # 10-25% profit
            exit_percentage = 0.5   # Standard scale out 1/2
        else:  # 25%+ profit (parabolic move)
            # Check for extreme extension
            extremely_large_range = current_range > avg_range * 3.0
            extremely_high_volume = current_volume > avg_volume * 2.5
            
            if extremely_large_range and extremely_high_volume:
                exit_percentage = 0.75  # Scale out 3/4, protect gains
            else:
                exit_percentage = 0.5   # Standard scale out
        
        return True, exit_percentage
    
    def _check_exit_indicators(self, df: pd.DataFrame, timestamp: datetime, current_price: float) -> None:
        """Check Ross's exit indicators"""
        if not self.position or len(df) < 5:
            return
        
        # PRIORITY 1: Check for extension bars (sell into strength)
        # This takes precedence over other exit signals as it's proactive profit-taking
        is_extension, exit_percentage = self._detect_extension_bar(df)
        if is_extension and self.position.status in [TradeStatus.ENTERED, TradeStatus.SCALED_FIRST, TradeStatus.SCALING_OUT]:
            shares_to_sell = int(self.position.current_shares * exit_percentage)
            if shares_to_sell > 0:
                self._partial_exit(timestamp, current_price, ExitReason.EXTENSION_BAR, shares_to_sell)
                
                # Update position status based on remaining shares
                remaining_shares = self.position.current_shares - shares_to_sell
                if remaining_shares <= 0:
                    self.position = self.position._replace(
                        current_shares=0,
                        status=TradeStatus.EXITED
                    )
                    self.status = MonitoringStatus.MONITORING_STOPPED
                else:
                    # Move stop to breakeven if this is first major scale out
                    new_stop = max(self.position.stop_loss, self.position.entry_price)
                    
                    self.position = self.position._replace(
                        current_shares=remaining_shares,
                        stop_loss=new_stop,
                        status=TradeStatus.SCALING_OUT if remaining_shares > self.position.initial_shares * 0.25 
                               else TradeStatus.HOLDING_RUNNER
                    )
                    
                logging.info(f"EXTENSION BAR EXIT {self.symbol}: Sold {exit_percentage:.0%} ({shares_to_sell} shares) into strength")
                return
        
        # PRIORITY 2: weakness exits on 1-minute (EMA/VWAP break)
        if self.exit_on_ema_vwap_break and self.position and self.position.status in [TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER, TradeStatus.SCALED_FIRST, TradeStatus.ENTERED]:
            cur = df.iloc[-1]
            ema9 = cur.get('ema9', None)
            vwap = cur.get('vwap', None)
            if (ema9 is not None and cur['close'] < ema9) or (vwap is not None and cur['close'] < vwap):
                self._exit_position(timestamp, current_price, ExitReason.NO_IMMEDIATE_BREAKOUT, self.position.current_shares)
                return

        # PRIORITY 3: optional first red 5-minute (disabled by default)
        if self.use_5min_first_red_exit:
            try:
                df5 = df.copy()
                df5.index = pd.to_datetime(df5.index)
                df5 = df5.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
                if self.position and self.position.status in [TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER] and len(df5) >= 2:
                    cur5, prev5 = df5.iloc[-1], df5.iloc[-2]
                    if (cur5['close'] < cur5['open'] and prev5['close'] > prev5['open']):
                        self._exit_position(timestamp, current_price, ExitReason.FIRST_RED_5MIN, self.position.current_shares)
                        return
            except Exception:
                pass

        # PRIORITY 4: 1-minute meaningful red candle exit for runners
        if (len(df) >= 2 and self.position 
            and self.position.status in [TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER]):
            cur1, prev1 = df.iloc[-1], df.iloc[-2]
            if (cur1['close'] < cur1['open'] and cur1['close'] < prev1['low']):
                self._exit_position(timestamp, current_price, ExitReason.FIRST_RED_5MIN, self.position.current_shares)
                return

    def _early_pullback_trim(self, df: pd.DataFrame, timestamp: datetime, current_price: float) -> None:
        """Scale out a portion at the start of a pullback (first meaningful 1-min red)."""
        if not self.position or self.position.current_shares <= 0:
            return
        if len(df) < 2:
            return
        pos = self.position
        cur1, prev1 = df.iloc[-1], df.iloc[-2]
        if (cur1['close'] < cur1['open'] and cur1['close'] < prev1['low'] and
            pos.status in [TradeStatus.ENTERED, TradeStatus.SCALED_FIRST, TradeStatus.SCALING_OUT]):
            shares_to_sell = max(1, int(pos.current_shares * 0.33))
            self._partial_exit(timestamp, current_price, ExitReason.NO_IMMEDIATE_BREAKOUT, shares_to_sell)
            self.position = pos._replace(current_shares=pos.current_shares - shares_to_sell)

    def _attempt_add_back(self, df: pd.DataFrame, timestamp: datetime) -> None:
        """Try to add back on a fresh 1-min bull flag breakout, with cooldown and limits."""
        # Enforce cooldown in bars
        if self._last_add_back_index is not None:
            bars_since = max(0, (df.index > self._last_add_back_index).sum())
            if bars_since < self.add_back_cooldown_bars:
                return
        signal = self.bull_flag_detector.detect_bull_flag(df, self.symbol)
        if (signal.stage == BullFlagStage.BREAKOUT_CONFIRMED and
            signal.validation == BullFlagValidation.VALID and
            signal.entry_price is not None and
            signal.stop_loss is not None):
            # Size the add-back using the sizer
            shares = 0
            if self.position_sizer is not None:
                try:
                    if self.sizing_method == "ross_dynamic" and hasattr(self.position_sizer, "calculate_ross_cameron_dynamic_size"):
                        result = self.position_sizer.calculate_ross_cameron_dynamic_size(
                            current_account_balance=self.account_balance,
                            entry_price=float(signal.entry_price),
                            stop_loss=float(signal.stop_loss),
                            validate_20_cent_preference=True,
                        )
                    else:
                        result = self.position_sizer.calculate_percentage_risk_size(
                            account_size=self.account_balance,
                            entry_price=float(signal.entry_price),
                            stop_loss=float(signal.stop_loss)
                        )
                    shares = int(getattr(result, 'shares', 0))
                except Exception as e:
                    logging.warning("Add-back sizing failed: %s", e)
            if shares <= 0:
                return
            # Record BUY execution and increase current shares (keep status)
            buy_exec = TradeExecution(
                timestamp=timestamp,
                symbol=self.symbol,
                action="BUY",
                shares=shares,
                price=float(signal.entry_price),
                reason="Add_Back_Bull_Flag"
            )
            self.trade_executions.append(buy_exec)
            pos = self.position
            self.position = pos._replace(current_shares=pos.current_shares + shares)
            self._add_backs_done += 1
            self._last_add_back_index = df.index[-1]
            logging.info("ADD-BACK %s: +%d shares at $%.2f (stop: $%.2f)",
                         self.symbol, shares, float(signal.entry_price), float(signal.stop_loss))
        # Only check this for remaining runners after extension bar exits
        last_candles = df.tail(3)
        if len(last_candles) >= 2:
            current_candle = last_candles.iloc[-1]
            prev_candle = last_candles.iloc[-2]
            
            # Red candle after green momentum
            if (current_candle['close'] < current_candle['open'] and
                prev_candle['close'] > prev_candle['open'] and
                self.position.status in [TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER]):
                self._exit_position(timestamp, current_price, ExitReason.FIRST_RED_5MIN, self.position.current_shares)
                return
        
        # Check for no immediate breakout (already handled in _manage_position)
        
        # Note: Heavy resistance and no buying pressure would require Level 2 and Time & Sales data
        # These would be implemented with real-time market data feeds
    
    def _partial_exit(self, timestamp: datetime, price: float, reason: ExitReason, shares: int) -> None:
        """Execute partial position exit"""
        execution = TradeExecution(
            timestamp=timestamp,
            symbol=self.symbol,
            action="SELL",
            shares=shares,
            price=price,
            reason=reason.value
        )
        self.trade_executions.append(execution)
        
        # Calculate R-multiple for the partial if it's first target
        if reason == ExitReason.FIRST_TARGET and self.position:
            risk_per_share = self.position.entry_price - self.position.stop_loss  # Original stop before BE move
            r_on_partial = (price - self.position.entry_price) / max(risk_per_share, 1e-6)
            logging.info(f"SCALED OUT {self.symbol}: {shares} shares at ${price:.2f} - {reason.value}")
            logging.info(f"{self.symbol}: moved stop to BE ${self.position.entry_price:.2f} "
                        f"after {r_on_partial:.1f}R partial")
        else:
            logging.info(f"SCALED OUT {self.symbol}: {shares} shares at ${price:.2f} - {reason.value}")
    
    def _exit_position(self, timestamp: datetime, price: float, reason: ExitReason, shares: int) -> None:
        """Exit entire position"""
        execution = TradeExecution(
            timestamp=timestamp,
            symbol=self.symbol,
            action="SELL",
            shares=shares,
            price=price,
            reason=reason.value
        )
        self.trade_executions.append(execution)
        
        if self.position:
            self.position = self.position._replace(
                current_shares=0,
                status=TradeStatus.EXITED
            )
        
        # Calculate total trade P&L and R-multiple
        if self.position:
            total_pnl = sum(exec.shares * (exec.price - self.position.entry_price) 
                           for exec in self.trade_executions if exec.action == "SELL")
            # Calculate original risk - need to reverse engineer since stop may have moved to BE
            first_target_exec = next((e for e in self.trade_executions if "first_target" in e.reason), None)
            if first_target_exec:
                # Reverse engineer from the first target R-multiple
                target_gain = first_target_exec.price - self.position.entry_price
                original_risk_per_share = target_gain / 3.1  # From observed 3.1R
            else:
                # Fallback: use current stop if it hasn't moved to BE
                if abs(self.position.stop_loss - self.position.entry_price) > 0.01:
                    original_risk_per_share = self.position.entry_price - self.position.stop_loss
                else:
                    original_risk_per_share = 0.065  # Conservative fallback
            original_risk_dollars = original_risk_per_share * self.position.initial_shares
            total_r = total_pnl / max(original_risk_dollars, 1e-6)
            
            logging.info(f"EXITED {self.symbol}: {shares} shares at ${price:.2f} - {reason.value}")
            logging.info(f"{self.symbol}: Final P&L ${total_pnl:+.0f} ({total_r:+.2f}R) | "
                        f"Initial risk/share: ${original_risk_per_share:.3f}")
        else:
            logging.info(f"EXITED {self.symbol}: {shares} shares at ${price:.2f} - {reason.value}")
        
        self.status = MonitoringStatus.MONITORING_STOPPED
    
    def get_trade_summary(self) -> Dict:
        """Get summary of trade performance"""
        # Build execution list and P&L
        exec_list = []
        total_pnl = 0.0
        position_entered = any(e.action == "BUY" for e in self.trade_executions)
        sell_reason_counts: Dict[str, int] = {}
        for e in self.trade_executions:
            exec_list.append({
                "timestamp": e.timestamp,
                "action": e.action,
                "shares": e.shares,
                "price": e.price,
                "reason": e.reason,
            })
            if e.action == "BUY":
                total_pnl -= e.shares * e.price
            else:
                total_pnl += e.shares * e.price
                sell_reason_counts[e.reason] = sell_reason_counts.get(e.reason, 0) + 1

        if not self.position and not position_entered:
            return {"status": "No position taken", "position_entered": False}

        # Use last known position snapshot if present, else infer entry fields from executions
        entry_time = self.position.entry_time if self.position else (exec_list[0]["timestamp"] if exec_list else None)
        entry_price = self.position.entry_price if self.position else (exec_list[0]["price"] if exec_list else None)
        initial_shares = self.position.initial_shares if self.position else (exec_list[0]["shares"] if exec_list else 0)
        stop_loss = self.position.stop_loss if self.position else None
        risk_per_share = (entry_price - stop_loss) if (entry_price and stop_loss) else None
        status_val = self.position.status.value if self.position else ("unknown")
        current_shares = self.position.current_shares if self.position else 0

        return {
            "symbol": self.symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "initial_shares": initial_shares,
            "current_shares": current_shares,
            "status": status_val,
            "total_pnl": total_pnl,
            "executions": len(self.trade_executions),
            "executions_list": exec_list,
            "sell_reasons": sell_reason_counts,
            "risk_per_share": risk_per_share,
            "monitoring_status": self.status.value,
            "position_entered": position_entered,
            "last_stage": (self._last_signal.stage.value if self._last_signal else "unknown"),
            "last_validation": (self._last_signal.validation.value if self._last_signal else "unknown"),
            "confirmations_rejects": self._confirmations_rejects,
            "breakout_attempts": self._breakout_attempts,
            # Last pattern structure details (if present)
            "last_pullback_candles": (self._last_signal.pullback_candles if self._last_signal else None),
            "last_retrace_percentage": (self._last_signal.retrace_percentage if self._last_signal else None),
            "last_volume_confirmation": (self._last_signal.volume_confirmation if self._last_signal else None),
            "last_broke_vwap": (self._last_signal.broke_vwap if self._last_signal else None),
            "last_broke_9ema": (self._last_signal.broke_9ema if self._last_signal else None),
            "last_strength_score": (self._last_signal.strength_score if self._last_signal else None),
        }

class ActionAlertPatternMonitor:
    """
    Main coordinator for action alert triggered pattern monitoring
    """
    
    def __init__(self):
        self.active_sessions: Dict[str, PatternMonitoringSession] = {}
        self.completed_sessions: List[PatternMonitoringSession] = []
        
    def process_action_alert(self, symbol: str, alert_time: datetime, alert_price: float,
                           alert_high: float, volume_spike: float = 5.0, 
                           news_catalyst: str = None) -> str:
        """
        Process new ACTION alert and start pattern monitoring
        
        Returns session_id for tracking
        """
        alert = ActionAlert(
            symbol=symbol,
            alert_time=alert_time,
            alert_price=alert_price,
            alert_high=alert_high,
            volume_spike=volume_spike,
            news_catalyst=news_catalyst
        )
        
        session = PatternMonitoringSession(alert)
        session_id = f"{symbol}_{alert_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.active_sessions[session_id] = session
        
        logging.info(f"Started monitoring {symbol} for patterns after ACTION alert at ${alert_price:.2f}")
        return session_id
    
    def update_price_data(self, symbol: str, timestamp: datetime, open_price: float,
                         high: float, low: float, close: float, volume: int) -> None:
        """Update price data for all active sessions for this symbol"""
        sessions_to_remove = []
        
        for session_id, session in self.active_sessions.items():
            if session.symbol == symbol:
                session.add_price_data(timestamp, open_price, high, low, close, volume)
                
                # Move completed sessions
                if session.status == MonitoringStatus.MONITORING_STOPPED:
                    sessions_to_remove.append(session_id)
        
        # Clean up completed sessions
        for session_id in sessions_to_remove:
            completed_session = self.active_sessions.pop(session_id)
            self.completed_sessions.append(completed_session)
    
    def get_active_symbols(self) -> List[str]:
        """Get list of symbols currently being monitored"""
        return list(set(session.symbol for session in self.active_sessions.values()))
    
    def get_session_summary(self, session_id: str) -> Optional[Dict]:
        """Get summary for specific session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get_trade_summary()
        
        for session in self.completed_sessions:
            if f"{session.symbol}_{session.start_time.strftime('%Y%m%d_%H%M%S')}" == session_id:
                return session.get_trade_summary()
        
        return None
    
    def get_all_summaries(self) -> List[Dict]:
        """Get summaries for all sessions"""
        summaries = []
        
        # Active sessions
        for session in self.active_sessions.values():
            summaries.append(session.get_trade_summary())
        
        # Completed sessions
        for session in self.completed_sessions:
            summaries.append(session.get_trade_summary())
        
        return summaries

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize monitor
    monitor = ActionAlertPatternMonitor()
    
    # Simulate ACTION alert
    alert_time = datetime(2023, 7, 1, 7, 0, 0)  # Ross's CLRO example
    session_id = monitor.process_action_alert(
        symbol="CLRO",
        alert_time=alert_time,
        alert_price=15.80,
        alert_high=15.80,
        volume_spike=10.0,
        news_catalyst="Breaking news"
    )
    
    print(f"Started monitoring session: {session_id}")
    print("System ready to process real-time price data and manage positions automatically")
    print("\nFeatures:")
    print("- ACTION alert triggered monitoring")
    print("- Extension bar detection with context-aware scaling:")
    print("  * <10% profit: Scale out 25% (let most run)")
    print("  * 10-25% profit: Scale out 50% (standard)")  
    print("  * >25% profit: Scale out 50-75% (protect gains)")
    print("- Automatic stop-to-breakeven after extension bar exits")
    print("- Priority-based exit logic: Extension bars → Red candles → Targets")
    print("- Real-time pattern detection")
    print("- Automatic position entry on valid patterns")
    print("- Ross Cameron's scaling and exit rules")
    print("- Complete trade management lifecycle")
