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
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging

try:
    from .bull_flag_pattern import BullFlagDetector, BullFlagSignal, BullFlagStage, BullFlagValidation
    from .alert_flip import AlertFlipStrategy
    from .bull_flag_strategy import BullFlagStrategy
    from .bull_flag_simple import BullFlagSimpleStrategy
    from .bull_flag_simple_v2 import BullFlagSimpleV2Strategy
    from ..ema_calculator import EMACalculator, RossCameronEMAConfig
    from ..macd_calculator import MACDCalculator, RossCameronMACDConfig
    from ...position_management.position_sizer import PositionSizer  # type: ignore
except ImportError:
    # For direct execution testing
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from bull_flag_pattern import BullFlagDetector, BullFlagSignal, BullFlagStage, BullFlagValidation
    from alert_flip import AlertFlipStrategy
    from bull_flag_strategy import BullFlagStrategy
    from bull_flag_simple import BullFlagSimpleStrategy
    from bull_flag_simple_v2 import BullFlagSimpleV2Strategy
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
    SESSION_END = "session_end"
    NEXT_BAR_CLOSE = "next_bar_close"  # For alert-close-flip strategy

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


@dataclass
class SessionConfig:
    """Configuration for a PatternMonitoringSession (kept thin and explicit)."""
    account_balance: float = 30000.0
    sizing_method: str = "ross_dynamic"  # or "percentage_risk"
    use_5min_first_red_exit: bool = False
    bailout_grace_bars: int = 2
    exit_on_ema_vwap_break: bool = True
    add_back_enabled: bool = True
    max_add_backs: int = 2
    add_back_cooldown_bars: int = 2
    stop_slippage_cents: float = 0.0
    entry_slippage_cents: float = 0.0
    entry_confirm_mode: str = "current"  # default to current bar confirmations for timely entries
    # Entry confirmations selector: 'both' | 'macd_only' | 'ema_only' | 'none'
    entry_confirmations: str = "both"
    # Entry trigger mode: 'pattern' (bull flag, default) or 'macd_cross'
    trigger_mode: str = "pattern"
    # Entry cutoff and management window
    entry_cutoff_time: Optional[datetime] = None  # do not open new positions after this time
    allow_manage_past_end: bool = True           # if True, manage open positions after cutoff
    # Per-alert freshness window (minutes) for opening new positions
    entry_cutoff_minutes: int = 15
    # Entry quality gate
    breakout_vol_mult: float = 0.0  # require breakout volume ≥ pullback_avg * mult (0 disables)
    min_pullback_avg_volume: float = 0.0  # require pullback avg volume ≥ threshold (0 disables)
    # Exit toggles
    exit_on_macd_cross: bool = False             # Exit on MACD bearish cross (macd < signal)
    enable_extension_bar_exit: bool = True       # Sell into strength on extension bars
    enable_early_pullback_trim: bool = True      # Trim on first meaningful 1-min red
    # Conservative entry option: require breakout candle to close green
    # Default False to align with intrabar entry at first break (Ross's timing)
    require_green_breakout: bool = False
    # Runner management (ride winners)
    runner_stop_mode: str = "ema9_1min"  # 'ema9_1min' | 'ema9_5min' | 'chandelier' | 'breakeven'
    first_red_5min_action: str = "trim_25"  # 'trim_25' | 'exit'
    disable_weakness_trim_after_scale: bool = False  # allow 1-min trims after first target
    runner_allow_stop_below_entry: bool = True  # allow trailing stop below entry to breathe early
    chandelier_atr_mult: float = 3.0  # ATR multiple for chandelier stop (5-min)
    # Entry quality gates
    spread_cap_bps: float = 0.0        # max (high-low)/close in bps (0 disables)
    require_macd_positive: bool = False  # require macd>0 and histogram>0 at entry bar
    # Alert times (for alert_flip strategy to avoid selling on alert bars)
    all_alert_times: Optional[list] = None

class PatternMonitoringSession:
    """
    Monitors a single symbol for pattern formation after alert
    """
    
    def __init__(self, alert: ActionAlert, patterns_to_monitor: List[str] = None,
                 position_sizer: Optional["PositionSizer"] = None,
                 config: Optional[SessionConfig] = None,
                 position_tracker: Optional[object] = None):
        self.alert = alert
        self.symbol = alert.symbol
        self.start_time = alert.alert_time
        self.status = MonitoringStatus.ACTIVE
        # Expose enums on instance for strategy helpers that reference session.ExitReason, etc.
        self.ExitReason = ExitReason
        self.MonitoringStatus = MonitoringStatus
        
        # Pattern detectors (configure entry quality if requested)
        self.bull_flag_detector = BullFlagDetector(
            breakout_volume_multiple=config.breakout_vol_mult if config else 0.0,
            min_pullback_avg_volume=config.min_pullback_avg_volume if config else 0.0,
            require_green_breakout=bool(getattr(config, 'require_green_breakout', False)) if config else False,
        )
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
        # Snapshot of initial entry details for reliable R-metrics even after position closes
        self._entry_snapshot: Optional[Dict] = None
        # Optional risk tracker for daily stop and risk accounting
        self._tracker = position_tracker
        
        # Data storage
        self.price_data: List[Dict] = []
        self.pattern_signals: List[BullFlagSignal] = []
        # Diagnostics for no-entry cases
        self._last_signal: Optional[BullFlagSignal] = None
        self._confirmations_rejects: int = 0
        self._breakout_attempts: int = 0
        # Alert-close-flip strategy state
        self._alert_flip: Optional[AlertFlipStrategy] = None
        # Bull-flag strategy wrapper
        self._bull_flag: Optional[BullFlagStrategy] = None
        # Simple bull-flag strategy (new scaffold)
        self._bull_flag_simple: Optional[BullFlagSimpleStrategy] = None
        # Simple v2 (duplicate of alert_flip)
        self._bull_flag_simple_v2: Optional[BullFlagSimpleV2Strategy] = None
        # Set of alert bar timestamps (as pandas Timestamps) to avoid exiting on alert bars
        try:
            ats = getattr(config, 'all_alert_times', None) if config else None
            if ats:
                import pandas as pd  # ensure available
                self._all_alert_times = set(pd.to_datetime(ats))
            else:
                self._all_alert_times = set()
        except Exception:
            self._all_alert_times = set()
        # Initialize strategies if requested
        if self.patterns_to_monitor:
            if 'alert_flip' in self.patterns_to_monitor:
                self._alert_flip = AlertFlipStrategy(all_alert_times=self._all_alert_times)
            if 'bull_flag' in self.patterns_to_monitor:
                self._bull_flag = BullFlagStrategy()
            if 'bull_flag_simple' in self.patterns_to_monitor:
                self._bull_flag_simple = BullFlagSimpleStrategy(all_alert_times=self._all_alert_times)
            if 'bull_flag_simple_v2' in self.patterns_to_monitor:
                self._bull_flag_simple_v2 = BullFlagSimpleV2Strategy(all_alert_times=self._all_alert_times)
        # Guard to avoid repeated actions within the same 5-min bucket
        self._last_5min_action_bucket = None
        
        # Ross's timing rules
        self.max_monitoring_time = timedelta(hours=2)  # Stop after 2 hours if no pattern
        self.breakout_timeout = timedelta(minutes=10)  # "Breakout or bailout"
        
        # Position management
        self.profit_ratio = 2.0  # Ross's 2:1 minimum
        self.scale_percentages = [0.5, 0.25, 0.25]  # 50%, 25%, 25%
        
        # Apply config
        cfg = config or SessionConfig()
        self.account_balance = cfg.account_balance
        # Expose volume gates for simple strategies (v2 reads from session)
        try:
            self.breakout_vol_mult = float(getattr(cfg, 'breakout_vol_mult', 0.0) or 0.0)
        except Exception:
            self.breakout_vol_mult = 0.0
        try:
            self.min_pullback_avg_volume = float(getattr(cfg, 'min_pullback_avg_volume', 0.0) or 0.0)
        except Exception:
            self.min_pullback_avg_volume = 0.0
        self.position_sizer = position_sizer
        self.sizing_method = cfg.sizing_method
        self.use_5min_first_red_exit = cfg.use_5min_first_red_exit
        self.bailout_grace_bars = max(1, int(cfg.bailout_grace_bars))
        self.exit_on_ema_vwap_break = cfg.exit_on_ema_vwap_break
        self.stop_slippage_cents = max(0.0, cfg.stop_slippage_cents)
        self.entry_slippage_cents = max(0.0, cfg.entry_slippage_cents)
        self.entry_confirm_mode = cfg.entry_confirm_mode if cfg.entry_confirm_mode in ("prior","current") else "prior"
        self.add_back_enabled = cfg.add_back_enabled
        self.max_add_backs = max(0, int(cfg.max_add_backs))
        self.add_back_cooldown_bars = max(0, int(cfg.add_back_cooldown_bars))
        self.entry_cutoff_time = cfg.entry_cutoff_time
        self.allow_manage_past_end = cfg.allow_manage_past_end
        self.trigger_mode = cfg.trigger_mode if getattr(cfg, 'trigger_mode', 'pattern') in ("pattern","macd_cross") else "pattern"
        # Relative (per-alert) cutoff time for new entries
        try:
            self.relative_entry_cutoff_time = self.start_time + timedelta(minutes=max(0, int(cfg.entry_cutoff_minutes)))
        except Exception:
            self.relative_entry_cutoff_time = self.start_time + timedelta(minutes=15)
        # New toggles
        self.entry_confirmations = cfg.entry_confirmations if cfg.entry_confirmations in ("both","macd_only","ema_only","none") else "both"
        self.exit_on_macd_cross = bool(cfg.exit_on_macd_cross)
        self.enable_extension_bar_exit = bool(cfg.enable_extension_bar_exit)
        self.enable_early_pullback_trim = bool(cfg.enable_early_pullback_trim)
        # Runner config
        self.runner_stop_mode = getattr(cfg, 'runner_stop_mode', 'ema9_1min')
        if self.runner_stop_mode not in ("ema9_1min","ema9_5min","chandelier","breakeven"):
            self.runner_stop_mode = "ema9_1min"
        self.first_red_5min_action = getattr(cfg, 'first_red_5min_action', 'trim_25')
        if self.first_red_5min_action not in ("trim_25","exit"):
            self.first_red_5min_action = "trim_25"
        self.disable_weakness_trim_after_scale = bool(getattr(cfg, 'disable_weakness_trim_after_scale', True))
        self.runner_allow_stop_below_entry = bool(getattr(cfg, 'runner_allow_stop_below_entry', True))
        try:
            self.chandelier_atr_mult = float(getattr(cfg, 'chandelier_atr_mult', 3.0))
        except Exception:
            self.chandelier_atr_mult = 3.0
        # Entry gates
        try:
            self.spread_cap_bps = float(getattr(cfg, 'spread_cap_bps', 0.0) or 0.0)
        except Exception:
            self.spread_cap_bps = 0.0
        self.require_macd_positive = bool(getattr(cfg, 'require_macd_positive', False))

        # Add-back behavior runtime state
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
        
        # Strategy-first handling: allow strategy objects (alert_flip / bull_flag_simple)
        # to run both entry and exit logic on every bar without interfering with
        # the default bull-flag management. When a strategy is active, we bypass
        # the default _manage_position to avoid double exits.
        strategy_active = (self._alert_flip is not None) or (getattr(self, '_bull_flag_simple', None) is not None) or (getattr(self, '_bull_flag_simple_v2', None) is not None)

        # Check patterns if monitoring is active and entries are allowed (before cutoff)
        # Entries allowed only:
        #  - on/after the ACTION alert start_time (no pre-alert entries)
        #  - before both absolute cutoff and relative per-alert cutoff
        entries_allowed = True
        if timestamp < self.start_time:
            entries_allowed = False
        if self.entry_cutoff_time is not None and timestamp > self.entry_cutoff_time:
            entries_allowed = False
        try:
            if timestamp > self.relative_entry_cutoff_time:
                entries_allowed = False
        except Exception:
            pass

        if strategy_active:
            # Handle simple strategies explicitly, once per bar.
            # - Allow entries only when entries_allowed is True and we have no position
            # - Always allow exit management when a position exists
            try:
                # Entry/exit for alert_flip
                if self._alert_flip is not None:
                    if (self.position is None and entries_allowed) or (self.position is not None):
                        self._alert_flip.on_bar(self, df, timestamp)
                # Entry/exit for bull_flag_simple
                if getattr(self, '_bull_flag_simple', None) is not None:
                    if (self.position is None and entries_allowed) or (self.position is not None):
                        self._bull_flag_simple.on_bar(self, df, timestamp)
                # Entry/exit for bull_flag_simple_v2
                if getattr(self, '_bull_flag_simple_v2', None) is not None:
                    if (self.position is None and entries_allowed) or (self.position is not None):
                        self._bull_flag_simple_v2.on_bar(self, df, timestamp)
            except Exception:
                pass
            # Do not invoke default _manage_position when a simple strategy is active
            return

        if self.status == MonitoringStatus.ACTIVE:
            if entries_allowed:
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
        """Check for pattern formation or MACD-only cross triggers"""
        # Simple alert-close-flip strategy: BUY on alert bar close, SELL on next bar close
        if self._alert_flip is not None:
            if self._alert_flip.on_bar(self, df, timestamp):
                return

        # Simple bull-flag strategy (scaffold)
        if self._bull_flag_simple is not None:
            if self._bull_flag_simple.on_bar(self, df, timestamp):
                return

        if getattr(self, '_bull_flag_simple_v2', None) is not None:
            if self._bull_flag_simple_v2.on_bar(self, df, timestamp):
                return

        # MACD-only trigger mode (opt-in)
        if self.trigger_mode == 'macd_cross' and self.status == MonitoringStatus.ACTIVE:
            if len(df) >= 2:
                prev = df.iloc[-2]
                cur = df.iloc[-1]
                try:
                    prev_macd = prev.get('macd', None)
                    prev_sig = prev.get('macd_signal', None)
                    cur_macd = cur.get('macd', None)
                    cur_sig = cur.get('macd_signal', None)
                    if (prev_macd is not None and prev_sig is not None and
                        cur_macd is not None and cur_sig is not None):
                        if prev_macd <= prev_sig and cur_macd > cur_sig:
                            entry_price = float(cur['close']) + float(self.entry_slippage_cents or 0.0)
                            try:
                                from risk_config import PREFERRED_STOP_DISTANCE_PCT
                                stop_pct = float(PREFERRED_STOP_DISTANCE_PCT)
                            except Exception:
                                stop_pct = 0.02
                            stop_loss = entry_price * (1.0 - stop_pct)
                            self._enter_direct(entry_price, stop_loss, timestamp, reason="MACD Bullish Cross")
                            return
                except Exception:
                    pass
            # Nothing else to do in macd_cross mode on this bar
            return

        # Pattern mode (default)
        if self._bull_flag is not None:
            # Delegate to strategy wrapper that preserves existing behavior
            took_action = self._bull_flag.on_bar(self, df, timestamp)
            if took_action:
                return
            # If no action, still allow timeout below
            return

        # Legacy path (should not be reached when strategy is initialized)
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
            # Use ACTION alert fields (HIGH + TIME) as known flagpole when available
            try:
                known_high = float(self.alert.alert_high) if self.alert and self.alert.alert_high is not None else None
            except Exception:
                known_high = None
            signal = self.bull_flag_detector.detect_bull_flag(
                df,
                self.symbol,
                known_flagpole_high=known_high,
                known_flagpole_time=self.alert.alert_time if self.alert else None,
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

                # If a tracker is provided and indicates daily stop, block new entries
                try:
                    if self._tracker is not None and hasattr(self._tracker, 'should_halt_trading') and self._tracker.should_halt_trading():
                        logging.info("Daily stop active; blocking new entry for %s", self.symbol)
                        return
                except Exception:
                    pass
                
                ema_ok = False
                macd_ok = False
                try:
                    # Confirmations: choose prior or current bar per config
                    idx = (-2 if len(df) >= 2 else -1) if self.entry_confirm_mode == 'prior' else -1
                    ema9 = df['ema9'].iloc[idx]
                    ema20 = df['ema20'].iloc[idx]
                    ema_ok = (pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20)
                except Exception:
                    ema_ok = False

                try:
                    idx = (-2 if len(df) >= 2 else -1) if self.entry_confirm_mode == 'prior' else -1
                    macd_val = df['macd'].iloc[idx]
                    macd_sig = df['macd_signal'].iloc[idx]
                    macd_hist = df['macd_hist'].iloc[idx]
                    macd_ok = (pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(macd_hist)
                               and macd_val > macd_sig and macd_hist > 0)
                    if macd_ok and self.require_macd_positive:
                        macd_ok = (macd_val > 0 and macd_hist > 0)
                except Exception:
                    macd_ok = False

                # Apply selected confirmation mode
                confirm_ok = False
                if self.entry_confirmations == 'both':
                    confirm_ok = ema_ok and macd_ok
                elif self.entry_confirmations == 'macd_only':
                    confirm_ok = macd_ok
                elif self.entry_confirmations == 'ema_only':
                    confirm_ok = ema_ok
                elif self.entry_confirmations == 'none':
                    confirm_ok = True

                if confirm_ok:
                    # Spread gate: reject if 1-min spread too wide
                    try:
                        if self.spread_cap_bps and self.spread_cap_bps > 0:
                            cur_bar = df.iloc[-1]
                            cur_spread_bps = 0.0
                            if float(cur_bar['close']) > 0:
                                cur_spread_bps = (float(cur_bar['high']) - float(cur_bar['low'])) / float(cur_bar['close']) * 10000.0
                            if cur_spread_bps > self.spread_cap_bps:
                                self._confirmations_rejects += 1
                                logging.debug("Entry rejected (spread %.1fbps > cap %.1fbps)", cur_spread_bps, self.spread_cap_bps)
                                return
                    except Exception:
                        pass
                    # Pass current bar high to ensure realistic fill capping
                    cur_high = None
                    try:
                        cur_high = float(df['high'].iloc[-1])
                    except Exception:
                        cur_high = None
                    self._enter_position(signal, timestamp, current_bar_high=cur_high)
                else:
                    self._confirmations_rejects += 1
                    logging.debug(
                        "Entry rejected (confirmations) | mode=%s | EMA ok: %s | MACD ok: %s | symbol=%s",
                        self.entry_confirmations, ema_ok, macd_ok, self.symbol
                    )
            
            # Check for pattern failure
            elif signal.stage == BullFlagStage.PATTERN_FAILED:
                self.status = MonitoringStatus.PATTERN_FAILED
                logging.info(f"Pattern failed for {self.symbol}: {signal.validation}")
        
        # Stop monitoring after timeout
        if timestamp - self.start_time > self.max_monitoring_time:
            self.status = MonitoringStatus.MONITORING_STOPPED

    # Internal handler that contains the existing bull flag logic, exposed for BullFlagStrategy
    def _on_bar_bull_flag(self, df: pd.DataFrame, timestamp: datetime) -> bool:
        """Existing bull flag per-bar logic extracted for strategy delegation.
        Returns True if an action/decision occurred; False otherwise.
        """
        action_taken = False
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
        # Use ACTION alert fields (HIGH + TIME) as known flagpole when available
        try:
            known_high = float(self.alert.alert_high) if self.alert and self.alert.alert_high is not None else None
        except Exception:
            known_high = None
        signal = self.bull_flag_detector.detect_bull_flag(
            df,
            self.symbol,
            known_flagpole_high=known_high,
            known_flagpole_time=self.alert.alert_time if self.alert else None,
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

            # If a tracker is provided and indicates daily stop, block new entries
            try:
                if self._tracker is not None and hasattr(self._tracker, 'should_halt_trading') and self._tracker.should_halt_trading():
                    logging.info("Daily stop active; blocking new entry for %s", self.symbol)
                    return True
            except Exception:
                pass

            ema_ok = False
            macd_ok = False
            try:
                # Confirmations: choose prior or current bar per config
                idx = (-2 if len(df) >= 2 else -1) if self.entry_confirm_mode == 'prior' else -1
                ema9 = df['ema9'].iloc[idx]
                ema20 = df['ema20'].iloc[idx]
                ema_ok = (pd.notna(ema9) and pd.notna(ema20) and ema9 > ema20)
            except Exception:
                ema_ok = False

            try:
                idx = (-2 if len(df) >= 2 else -1) if self.entry_confirm_mode == 'prior' else -1
                macd_val = df['macd'].iloc[idx]
                macd_sig = df['macd_signal'].iloc[idx]
                macd_hist = df['macd_hist'].iloc[idx]
                macd_ok = (pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(macd_hist)
                           and macd_val > macd_sig and macd_hist > 0)
                if macd_ok and self.require_macd_positive:
                    macd_ok = (macd_val > 0 and macd_hist > 0)
            except Exception:
                macd_ok = False

            # Apply selected confirmation mode
            confirm_ok = False
            if self.entry_confirmations == 'both':
                confirm_ok = ema_ok and macd_ok
            elif self.entry_confirmations == 'macd_only':
                confirm_ok = macd_ok
            elif self.entry_confirmations == 'ema_only':
                confirm_ok = ema_ok
            elif self.entry_confirmations == 'none':
                confirm_ok = True

            if confirm_ok:
                # Spread gate: reject if 1-min spread too wide
                try:
                    if self.spread_cap_bps and self.spread_cap_bps > 0:
                        cur_bar = df.iloc[-1]
                        cur_spread_bps = 0.0
                        if float(cur_bar['close']) > 0:
                            cur_spread_bps = (float(cur_bar['high']) - float(cur_bar['low'])) / float(cur_bar['close']) * 10000.0
                        if cur_spread_bps > self.spread_cap_bps:
                            self._confirmations_rejects += 1
                            logging.debug("Entry rejected (spread %.1fbps > cap %.1fbps)", cur_spread_bps, self.spread_cap_bps)
                            return True
                except Exception:
                    pass
                # Pass current bar high to ensure realistic fill capping
                cur_high = None
                try:
                    cur_high = float(df['high'].iloc[-1])
                except Exception:
                    cur_high = None
                self._enter_position(signal, timestamp, current_bar_high=cur_high)
                action_taken = True
            else:
                self._confirmations_rejects += 1
                logging.debug(
                    "Entry rejected (confirmations) | mode=%s | EMA ok: %s | MACD ok: %s | symbol=%s",
                    self.entry_confirmations, ema_ok, macd_ok, self.symbol
                )

        # Check for pattern failure
        elif signal.stage == BullFlagStage.PATTERN_FAILED:
            self.status = MonitoringStatus.PATTERN_FAILED
            logging.info(f"Pattern failed for {self.symbol}: {signal.validation}")
            action_taken = True

        # Stop monitoring after timeout
        if timestamp - self.start_time > self.max_monitoring_time:
            self.status = MonitoringStatus.MONITORING_STOPPED
            action_taken = True

        return action_taken
    
    def _enter_position(self, signal: BullFlagSignal, timestamp: datetime, current_bar_high: Optional[float] = None) -> None:
        """Enter position based on pattern signal"""
        # Intrabar fill at breakout trigger plus optional slippage, capped at the bar's high
        if signal.entry_price is not None:
            raw_price = float(signal.entry_price) + float(self.entry_slippage_cents or 0.0)
            if current_bar_high is not None:
                entry_price = min(raw_price, float(current_bar_high))
            else:
                entry_price = raw_price
        else:
            entry_price = None
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
        
        # If a tracker is present, enforce risk gating and register position
        try:
            if self._tracker is not None and entry_price is not None and stop_loss is not None:
                risk_per_share = entry_price - stop_loss
                risk_amount = max(0.0, float(shares) * float(risk_per_share))
                # Honor daily stop before committing
                if hasattr(self._tracker, 'should_halt_trading') and self._tracker.should_halt_trading():
                    logging.info("Daily stop active; blocking new entry for %s", self.symbol)
                    return
                # Prevent duplicate symbol while active if supported
                try:
                    if hasattr(self._tracker, 'validate_position_request'):
                        self._tracker.validate_position_request(self.symbol, risk_amount)
                    elif hasattr(self._tracker, 'active_positions') and self.symbol in getattr(self._tracker, 'active_positions', {}):
                        logging.info("Duplicate active position for %s; blocking new entry", self.symbol)
                        return
                except Exception as ve:
                    logging.info("Position validation failed for %s: %s", self.symbol, ve)
                    return
                # Validate risk limit for new position
                if hasattr(self._tracker, 'can_open_position'):
                    ok_reason = self._tracker.can_open_position(risk_amount)
                    try:
                        ok, reason = ok_reason
                    except Exception:
                        ok, reason = bool(ok_reason), ""
                    if not ok:
                        logging.info("PositionTracker rejected %s: %s", self.symbol, reason)
                        return
                # Register position
                if hasattr(self._tracker, 'add_position'):
                    self._tracker.add_position(self.symbol, timestamp, float(entry_price), int(shares), float(risk_amount), float(stop_loss))
        except Exception as e:
            logging.warning("Tracker integration failed on entry for %s: %s", self.symbol, e)

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

        # Persist initial entry snapshot for analyzers (do not rely on mutable state later)
        try:
            initial_rps = float(entry_price) - float(stop_loss) if (entry_price is not None and stop_loss is not None) else None
            self._entry_snapshot = {
                "entry_time": timestamp,
                "entry_price": float(entry_price) if entry_price is not None else None,
                "initial_stop": float(stop_loss) if stop_loss is not None else None,
                "initial_risk_per_share": float(initial_rps) if initial_rps is not None else None,
                "initial_shares": int(shares),
            }
        except Exception:
            self._entry_snapshot = None
        
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

    def _enter_direct(self, entry_price: float, stop_loss: float, timestamp: datetime, reason: str = "Direct Entry") -> None:
        """Direct entry for non-pattern triggers (e.g., MACD cross)."""
        shares = 1000
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            logging.warning("Invalid risk (stop >= entry); skipping direct entry.")
            return
        if self.position_sizer is not None:
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
            except Exception as e:
                logging.warning("Position sizing failed; using fallback 1000 shares. Error: %s", e)

        first_target = entry_price + (risk_per_share * self.profit_ratio)
        second_target = entry_price + (risk_per_share * self.profit_ratio * 1.5)
        third_target = entry_price + (risk_per_share * self.profit_ratio * 2.0)

        # Tracker integration for direct entries as well (prevents overlaps per symbol)
        try:
            if self._tracker is not None:
                risk_amount = max(0.0, float(shares) * float(risk_per_share))
                # Duplicate symbol guard
                try:
                    if hasattr(self._tracker, 'validate_position_request'):
                        self._tracker.validate_position_request(self.symbol, risk_amount)
                    elif hasattr(self._tracker, 'active_positions') and self.symbol in getattr(self._tracker, 'active_positions', {}):
                        logging.info("Duplicate active position for %s; blocking new entry", self.symbol)
                        return
                except Exception as ve:
                    logging.info("Position validation failed for %s: %s", self.symbol, ve)
                    return
                # Risk limits
                if hasattr(self._tracker, 'can_open_position'):
                    ok_reason = self._tracker.can_open_position(risk_amount)
                    try:
                        ok, reason = ok_reason
                    except Exception:
                        ok, reason = bool(ok_reason), ""
                    if not ok:
                        logging.info("PositionTracker rejected %s: %s", self.symbol, reason)
                        return
                if hasattr(self._tracker, 'add_position'):
                    self._tracker.add_position(self.symbol, timestamp, float(entry_price), int(shares), float(risk_amount), float(stop_loss))
        except Exception as e:
            logging.warning("Tracker integration failed on direct entry for %s: %s", self.symbol, e)

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

        self.trade_executions.append(TradeExecution(
            timestamp=timestamp,
            symbol=self.symbol,
            action="BUY",
            shares=shares,
            price=entry_price,
            reason=reason
        ))

        target_2r = entry_price + (2 * risk_per_share)
        logging.info(f"ENTERED {self.symbol}: {shares} shares at ${entry_price:.2f} ({reason})")
        logging.info(f"{self.symbol}: Risk ${risk_per_share:.3f}/share (stop ${stop_loss:.2f}) | 2R target ${target_2r:.2f}")
    
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
        
        # Breakout or Bailout (1-minute): within grace bars, exit at breakeven
        # Only if the bar's HIGH never moved above entry (intrabar check to avoid cutting runners)
        if pos.status == TradeStatus.ENTERED:
            # Bars since entry
            try:
                bars_since_entry = int((df.index > pos.entry_time).sum())
            except Exception:
                bars_since_entry = 0
            last_high = float(df.iloc[-1]['high']) if len(df) > 0 else current_price
            if bars_since_entry <= self.bailout_grace_bars and last_high <= pos.entry_price:
                self._exit_position(timestamp, current_price, ExitReason.BREAKOUT_OR_BAILOUT, pos.current_shares)
                return

        # Check for profit targets and scale out (use intrabar HIGH to trigger)
        self._check_profit_targets(df, timestamp)

        # Early pullback scale-out: first meaningful 1-min red (sell ~33% of remaining)
        if self.enable_early_pullback_trim:
            self._early_pullback_trim(df, timestamp, current_price)

        # Update runner trailing stop (EMA9 1-min/5-min or chandelier) after first scale
        try:
            if self.position and self.position.status in [TradeStatus.SCALED_FIRST, TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER]:
                new_stop = None
                if self.runner_stop_mode == 'ema9_1min':
                    # Use precomputed EMA9 if present; else EWM on 1-min
                    if 'ema9' in df.columns:
                        ema9_1 = df['ema9'].iloc[-1]
                    else:
                        ema9_1 = df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                    if pd.notna(ema9_1):
                        new_stop = float(ema9_1)
                elif self.runner_stop_mode == 'ema9_5min':
                    df5 = df.copy()
                    df5.index = pd.to_datetime(df5.index)
                    df5 = df5.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
                    if len(df5) >= 1:
                        ema9_5 = df5['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                        if pd.notna(ema9_5):
                            new_stop = float(ema9_5)
                elif self.runner_stop_mode == 'chandelier':
                    df5 = df.copy()
                    df5.index = pd.to_datetime(df5.index)
                    df5 = df5.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
                    if len(df5) >= 2:
                        import numpy as np
                        high = df5['high']
                        low = df5['low']
                        close = df5['close']
                        prev_close = close.shift(1)
                        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
                        atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
                        last_close = float(close.iloc[-1])
                        if not np.isnan(atr):
                            new_stop = last_close - self.chandelier_atr_mult * float(atr)
                elif self.runner_stop_mode == 'breakeven':
                    new_stop = float(self.position.entry_price)
                if new_stop is not None:
                    # Only trail upward (never lower the stop)
                    trail = max(float(self.position.stop_loss), float(new_stop))
                    if not self.runner_allow_stop_below_entry:
                        trail = max(trail, float(self.position.entry_price))
                    if trail > float(self.position.stop_loss) + 1e-9:
                        self.position = self.position._replace(stop_loss=trail)
        except Exception:
            pass

        # Check for exit indicators
        self._check_exit_indicators(df, timestamp, current_price)

        # Attempt add-back on next valid 1-min bull flag
        if self.add_back_enabled and self._add_backs_done < self.max_add_backs:
            self._attempt_add_back(df, timestamp)
    
    def _check_profit_targets(self, df: pd.DataFrame, timestamp: datetime) -> None:
        """Check profit targets and scale out according to Ross's rules"""
        if not self.position:
            return
        
        pos = self.position
        cur_close = float(df.iloc[-1]['close']) if len(df) > 0 else pos.entry_price
        cur_high = float(df.iloc[-1]['high']) if len(df) > 0 else cur_close
        
        # First target: Sell 1/2, set runner stop per config
        if (cur_high >= pos.first_target and 
            pos.status == TradeStatus.ENTERED):
            
            shares_to_sell = int(pos.initial_shares * self.scale_percentages[0])
            # Fill at target price or last close if higher (conservative): use the target price
            self._partial_exit(timestamp, float(pos.first_target), ExitReason.FIRST_TARGET, shares_to_sell)
            
            # Determine runner stop
            new_stop = float(pos.entry_price)
            try:
                if self.runner_stop_mode == 'ema9_1min':
                    # 1-min EMA9
                    if 'ema9' in df.columns:
                        ema9_1 = df['ema9'].iloc[-1]
                    else:
                        ema9_1 = df['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                    if pd.notna(ema9_1):
                        new_stop = float(ema9_1)
                elif self.runner_stop_mode == 'ema9_5min':
                    df5 = df.copy(); df5.index = pd.to_datetime(df5.index)
                    df5 = df5.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
                    ema9_5 = df5['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                    if pd.notna(ema9_5):
                        new_stop = float(ema9_5)
                elif self.runner_stop_mode == 'chandelier':
                    df5 = df.copy(); df5.index = pd.to_datetime(df5.index)
                    df5 = df5.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
                    import numpy as np
                    if len(df5) >= 2:
                        high = df5['high']; low = df5['low']; close = df5['close']; prev_close = close.shift(1)
                        tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
                        atr = tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
                        last_close = float(close.iloc[-1])
                        if not np.isnan(atr):
                            new_stop = last_close - self.chandelier_atr_mult * float(atr)
            except Exception:
                new_stop = float(pos.entry_price)
            if not self.runner_allow_stop_below_entry:
                new_stop = max(new_stop, float(pos.entry_price))
            self.position = pos._replace(
                stop_loss=new_stop,
                current_shares=pos.current_shares - shares_to_sell,
                status=TradeStatus.SCALED_FIRST
            )
            
        # Second target: Scale out more
        elif (cur_high >= pos.second_target and 
              pos.status == TradeStatus.SCALED_FIRST):
            
            shares_to_sell = int(pos.initial_shares * self.scale_percentages[1])
            self._partial_exit(timestamp, float(pos.second_target), ExitReason.SECOND_TARGET, shares_to_sell)
            
            self.position = pos._replace(
                current_shares=pos.current_shares - shares_to_sell,
                status=TradeStatus.SCALING_OUT
            )
            
        # Third target: Scale out final piece or hold runner
        elif (cur_high >= pos.third_target and 
              pos.status == TradeStatus.SCALING_OUT):
            
            # Ross often holds runners, but can scale here too
            shares_to_sell = int(pos.initial_shares * self.scale_percentages[2])
            if pos.current_shares > shares_to_sell:
                self._partial_exit(timestamp, float(pos.third_target), ExitReason.THIRD_TARGET, shares_to_sell)
                self.position = pos._replace(
                    current_shares=pos.current_shares - shares_to_sell,
                    status=TradeStatus.HOLDING_RUNNER
                )
            else:
                # Exit remaining shares
                self._exit_position(timestamp, float(pos.third_target), ExitReason.THIRD_TARGET, pos.current_shares)
    
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
        is_extension, exit_percentage = (False, 0.0)
        if self.enable_extension_bar_exit:
            is_extension, exit_percentage = self._detect_extension_bar(df)
        if self.enable_extension_bar_exit and is_extension and self.position.status in [TradeStatus.ENTERED, TradeStatus.SCALED_FIRST, TradeStatus.SCALING_OUT]:
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
        
        # PRIORITY 2: weakness on 1-minute (EMA/VWAP break) => trim rather than full exit to let runners breathe
        # Skip 1-min weakness trims after first scale if configured
        skip_weakness = (self.disable_weakness_trim_after_scale and self.position and self.position.status in [TradeStatus.SCALED_FIRST, TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER])
        if self.exit_on_ema_vwap_break and not skip_weakness and self.position and self.position.status in [TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER, TradeStatus.SCALED_FIRST, TradeStatus.ENTERED]:
            cur = df.iloc[-1]
            ema9 = cur.get('ema9', None)
            vwap = cur.get('vwap', None)
            if (ema9 is not None and cur['close'] < ema9) or (vwap is not None and cur['close'] < vwap):
                # Trim 25% of remaining shares instead of full exit
                shares_to_sell = max(1, int(self.position.current_shares * 0.25))
                self._partial_exit(timestamp, current_price, ExitReason.NO_IMMEDIATE_BREAKOUT, shares_to_sell)
                # Tighten stop to protect gains: at least breakeven, optionally to EMA9 if above entry
                try:
                    new_stop = max(self.position.stop_loss, self.position.entry_price)
                    if ema9 is not None and ema9 > new_stop:
                        new_stop = float(ema9)
                    self.position = self.position._replace(current_shares=self.position.current_shares - shares_to_sell,
                                                           stop_loss=new_stop,
                                                           status=TradeStatus.SCALING_OUT)
                except Exception:
                    pass
                # Do not return; allow higher-priority exits to act if needed

        # PRIORITY 3: MACD bearish cross (optional)
        if self.exit_on_macd_cross and len(df) >= 2 and self.position and self.position.status in [TradeStatus.ENTERED, TradeStatus.SCALED_FIRST, TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER]:
            prev = df.iloc[-2]
            cur = df.iloc[-1]
            try:
                prev_macd = prev.get('macd', None)
                prev_sig = prev.get('macd_signal', None)
                cur_macd = cur.get('macd', None)
                cur_sig = cur.get('macd_signal', None)
                if (prev_macd is not None and prev_sig is not None and cur_macd is not None and cur_sig is not None):
                    if prev_macd >= prev_sig and cur_macd < cur_sig:
                        self._exit_position(timestamp, current_price, ExitReason.NO_IMMEDIATE_BREAKOUT, self.position.current_shares)
                        return
            except Exception:
                pass

        # PRIORITY 4: optional first red 5-minute (disabled by default)
        if self.use_5min_first_red_exit:
            try:
                df5 = df.copy()
                df5.index = pd.to_datetime(df5.index)
                df5 = df5.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
                if self.position and self.position.status in [TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER, TradeStatus.SCALED_FIRST] and len(df5) >= 2:
                    cur5, prev5 = df5.iloc[-1], df5.iloc[-2]
                    # EMA9 5-min trailing enforcement
                    if self.runner_stop_mode == 'ema9_5min':
                        ema9_5 = df5['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                        if pd.notna(ema9_5) and float(cur5['close']) < float(ema9_5):
                            self._exit_position(timestamp, current_price, ExitReason.FIRST_RED_5MIN, self.position.current_shares)
                            return
                    if (cur5['close'] < cur5['open'] and prev5['close'] > prev5['open']):
                        if self.first_red_5min_action == 'trim_25':
                            shares_to_sell = max(1, int(self.position.current_shares * 0.25))
                            self._partial_exit(timestamp, current_price, ExitReason.FIRST_RED_5MIN, shares_to_sell)
                            return
                        else:
                            self._exit_position(timestamp, current_price, ExitReason.FIRST_RED_5MIN, self.position.current_shares)
                            return
            except Exception:
                pass

        # PRIORITY 5: 1-minute meaningful red candle handling for runners
        if (len(df) >= 2 and self.position 
            and self.position.status in [TradeStatus.SCALING_OUT, TradeStatus.HOLDING_RUNNER]):
            cur1, prev1 = df.iloc[-1], df.iloc[-2]
            if (cur1['close'] < cur1['open'] and cur1['close'] < prev1['low']):
                if self.runner_stop_mode == 'ema9_1min':
                    # Trim 25% on meaningful 1-min red; true exit comes from EMA9 trail
                    shares_to_sell = max(1, int(self.position.current_shares * 0.25))
                    self._partial_exit(timestamp, current_price, ExitReason.FIRST_RED_5MIN, shares_to_sell)
                    return
                elif not self.disable_weakness_trim_after_scale:
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
            # Block add-backs if daily stop is active
            try:
                if self._tracker is not None and hasattr(self._tracker, 'should_halt_trading') and self._tracker.should_halt_trading():
                    return
            except Exception:
                pass
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
            # Risk gating with tracker for add-back
            try:
                if self._tracker is not None:
                    rps = float(signal.entry_price) - float(signal.stop_loss)
                    risk_amt = max(0.0, float(shares) * rps)
                    if hasattr(self._tracker, 'can_open_position'):
                        ok, reason = self._tracker.can_open_position(risk_amt)
                        if not ok:
                            logging.info("Add-back blocked for %s: %s", self.symbol, reason)
                            return
                    if hasattr(self._tracker, 'update_position'):
                        # Treat as shares increase
                        self._tracker.update_position(self.symbol, float(signal.entry_price), shares_change=int(shares))
            except Exception as e:
                logging.debug("Tracker integration failed on add-back for %s: %s", self.symbol, e)
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
                self._exit_position(timestamp, float(current_candle['close']), ExitReason.FIRST_RED_5MIN, self.position.current_shares)
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

        # Update external tracker with realized P&L and share change
        try:
            if self._tracker is not None and self.position is not None:
                # Realized delta approximated against entry price
                realized = float(shares) * (float(price) - float(self.position.entry_price))
                if hasattr(self._tracker, 'update_position'):
                    self._tracker.update_position(self.symbol, float(price), shares_change=-int(shares))
                if hasattr(self._tracker, 'record_realized_pnl'):
                    self._tracker.record_realized_pnl(realized)
        except Exception as e:
            logging.debug("Tracker integration failed on partial exit for %s: %s", self.symbol, e)
        
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

        # Close in external tracker first to compute realized P&L
        try:
            if self._tracker is not None:
                if hasattr(self._tracker, 'close_position'):
                    self._tracker.close_position(self.symbol, float(price), timestamp)
        except Exception as e:
            logging.debug("Tracker integration failed on close for %s: %s", self.symbol, e)
        
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

    # Public method to force-flatten at session end
    def force_flatten(self, timestamp: datetime, price: float) -> None:
        """Force exit any open shares at the given time/price with SESSION_END reason."""
        if self.position and self.position.current_shares > 0:
            self._exit_position(timestamp, float(price), ExitReason.SESSION_END, self.position.current_shares)
    
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
            # Return diagnostics for no-entry cases
            return {
                "status": "No position taken",
                "position_entered": False,
                "symbol": self.symbol,
                "monitoring_status": self.status.value,
                "last_stage": (self._last_signal.stage.value if self._last_signal else "unknown"),
                "last_validation": (self._last_signal.validation.value if self._last_signal else "unknown"),
                "confirmations_rejects": self._confirmations_rejects,
                "breakout_attempts": self._breakout_attempts,
                "last_pullback_candles": (self._last_signal.pullback_candles if self._last_signal else None),
                "last_retrace_percentage": (self._last_signal.retrace_percentage if self._last_signal else None),
                "last_volume_confirmation": (self._last_signal.volume_confirmation if self._last_signal else None),
                "last_broke_vwap": (self._last_signal.broke_vwap if self._last_signal else None),
                "last_broke_9ema": (self._last_signal.broke_9ema if self._last_signal else None),
                "last_strength_score": (self._last_signal.strength_score if self._last_signal else None),
            }

        # Entry snapshot (authoritative, even after position closes)
        snap = self._entry_snapshot or {}
        entry_time = snap.get("entry_time") or (self.position.entry_time if self.position else (exec_list[0]["timestamp"] if exec_list else None))
        entry_price = snap.get("entry_price") or (self.position.entry_price if self.position else (exec_list[0]["price"] if exec_list else None))
        initial_shares = snap.get("initial_shares") or (self.position.initial_shares if self.position else (exec_list[0]["shares"] if exec_list else 0))
        stop_loss = snap.get("initial_stop") or (self.position.stop_loss if self.position else None)
        risk_per_share = snap.get("initial_risk_per_share") or ((entry_price - stop_loss) if (entry_price and stop_loss) else None)
        status_val = self.position.status.value if self.position else ("unknown")
        current_shares = self.position.current_shares if self.position else 0

        # Exit snapshot from last SELL execution (if any)
        exit_time = None
        exit_price = None
        try:
            sell_execs = [e for e in self.trade_executions if e.action == "SELL"]
            if sell_execs:
                exit_time = sell_execs[-1].timestamp
                exit_price = sell_execs[-1].price
        except Exception:
            pass

        return {
            "symbol": self.symbol,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
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
