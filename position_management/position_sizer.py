"""
Position Sizing Module

OVERVIEW:
Comprehensive position sizing system implementing multiple methodologies for optimal
capital allocation. Based on the principle that "position sizing accounted for 91% 
of portfolio performance" (Brinson, Singer & Beebower study).

SIZING METHODS IMPLEMENTED:
1. Percentage of Portfolio (Ross Cameron's 2% rule)
2. Fixed Dollar Amount 
3. Volatility-Based (ATR multiplier)
4. Standard Deviation-Based (Bollinger Bands)
5. Vital Market Levels (Support/Resistance)
6. Percentage Loss Method (William O'Neil style)

KEY PRINCIPLES:
- Larger accounts risk smaller percentages
- Position size = Risk Amount / Risk Per Share
- Risk Per Share = Entry Price - Stop Loss Price
- Account for trading psychology and stress levels
- Scalable across different account sizes

ROSS CAMERON'S APPROACH:
- 1.25% risk per trade (dynamic with account balance)
- 2.5% profit target per trade (1:2 risk/reward ratio)
- Preferred stop distance: 20 cents when technically valid
- Position size = Current Risk Amount / Stop Distance
- Daily risk budget: 10% of account (split across 4 trades)
- Maximum 3-4 positions to avoid over-concentration
- "I like to risk about 20 cents on my trades" - Ross Cameron
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple, Union
from enum import Enum
from datetime import datetime
import math
import logging

# Import centralized risk configuration
from risk_config import (
    DAILY_RISK_PCT, RISK_PER_TRADE_PCT, MAX_CONCURRENT_POSITIONS,
    PROFIT_TARGET_RATIO, MAX_POSITION_PCT, PREFERRED_STOP_DISTANCE_PCT,
    ATR_PERIOD, ATR_MULTIPLIER, DEFAULT_FIXED_DOLLAR_AMOUNT
)

# TA-Lib with graceful fallback
try:
    import talib
    _HAS_TALIB = True
except Exception:
    talib = None
    _HAS_TALIB = False

class SizingMethod(Enum):
    """Position sizing methodologies"""
    ROSS_CAMERON_DYNAMIC = "ross_cameron_dynamic"  # Ross's 1:2 risk/reward system
    PERCENTAGE_RISK = "percentage_risk"          # Traditional percentage risk
    FIXED_DOLLAR = "fixed_dollar"               # Fixed $ amount risk
    VOLATILITY_ATR = "volatility_atr"           # ATR-based sizing
    STANDARD_DEVIATION = "standard_deviation"    # Bollinger Band based
    VITAL_LEVELS = "vital_levels"               # Support/Resistance based
    PERCENTAGE_LOSS = "percentage_loss"         # William O'Neil method
    KELLY_CRITERION = "kelly_criterion"         # Optimal growth formula

class AccountTier(Enum):
    """Account size tiers for risk scaling"""
    SMALL = "small"           # < $25K
    MEDIUM = "medium"         # $25K - $100K  
    LARGE = "large"           # $100K - $500K
    VERY_LARGE = "very_large" # $500K - $1M
    INSTITUTIONAL = "institutional" # > $1M

class PositionSizeResult(NamedTuple):
    """Position sizing calculation result"""
    shares: int
    dollar_amount: float
    risk_amount: float
    risk_per_share: float
    risk_percentage: float
    profit_target: float  # Added for 1:2 risk/reward
    method_used: SizingMethod
    max_loss_if_stopped: float
    position_value: float
    buying_power_used: float
    warnings: List[str]
    metadata: Dict

class PositionSizer:
    """
    Comprehensive position sizing calculator implementing multiple methodologies
    """
    
    def __init__(self):
        # Ross Cameron's dynamic risk/reward system (using centralized config)
        self.ross_risk_per_trade = RISK_PER_TRADE_PCT      # 1.25% risk per trade
        self.ross_target_per_trade = RISK_PER_TRADE_PCT * PROFIT_TARGET_RATIO     # 2.5% target per trade (1:2 ratio)
        self.ross_preferred_stop = PREFERRED_STOP_DISTANCE_PCT        # 2% preferred stop
        self.ross_daily_risk_budget = DAILY_RISK_PCT     # 5% daily risk budget
        self.ross_max_positions = MAX_CONCURRENT_POSITIONS            # Max 4 positions at once
        
        # Ross Cameron's legacy risk percentages by account tier
        self.default_risk_percentages = {
            AccountTier.SMALL: 0.02,        # 2% for small accounts
            AccountTier.MEDIUM: 0.015,      # 1.5% for medium accounts  
            AccountTier.LARGE: 0.01,        # 1% for large accounts
            AccountTier.VERY_LARGE: 0.008,  # 0.8% for very large accounts
            AccountTier.INSTITUTIONAL: 0.005 # 0.5% for institutional
        }
        
        # Minimum and maximum position sizes
        self.min_position_value = 100    # $100 minimum position
        self.max_position_percentage = MAX_POSITION_PCT  # 33% max of account in one position
        self.ross_max_position_percentage = MAX_POSITION_PCT * 2  # 67% max for Ross's system (with 2x margin)
        
        # ATR multipliers for different strategies
        self.atr_multipliers = {
            'conservative': 2.0,
            'moderate': 2.5, 
            'aggressive': 3.0,
            'very_aggressive': 3.5
        }
    
    def _atr_fallback(self, price_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR using pure pandas if TA-Lib not available"""
        high, low, close = price_data['high'], price_data['low'], price_data['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
        
    def determine_account_tier(self, account_size: float) -> AccountTier:
        """Determine account tier for risk scaling"""
        if account_size < 25000:
            return AccountTier.SMALL
        elif account_size < 100000:
            return AccountTier.MEDIUM
        elif account_size < 500000:
            return AccountTier.LARGE
        elif account_size < 1000000:
            return AccountTier.VERY_LARGE
        else:
            return AccountTier.INSTITUTIONAL
    
    def calculate_ross_cameron_dynamic_size(self, current_account_balance: float, entry_price: float,
                                          stop_loss: float, validate_20_cent_preference: bool = True) -> PositionSizeResult:
        """
        Ross Cameron's Dynamic Risk/Reward Position Sizing System
        
        FRAMEWORK:
        - Risk per trade: 1.25% of current account balance (dynamic)
        - Profit target: 2.5% of current account balance (1:2 risk/reward)
        - Preferred stop distance: 20 cents when technically valid
        - Position size = Current Risk Amount ÷ Stop Distance
        
        Args:
            current_account_balance: Real-time account balance (dynamic)
            entry_price: Entry price per share
            stop_loss: Stop loss price per share  
            validate_20_cent_preference: Warn if stop distance isn't close to 20 cents
        """
        warnings = []
        
        # Calculate dynamic risk and target amounts
        risk_amount = current_account_balance * self.ross_risk_per_trade
        profit_target = current_account_balance * self.ross_target_per_trade
        
        # Calculate risk per share (stop distance)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            warnings.append("Risk per share is zero - invalid stop loss")
            return self._create_invalid_result(SizingMethod.ROSS_CAMERON_DYNAMIC, warnings)
        
        # Validate 20-cent preference
        if validate_20_cent_preference:
            stop_deviation = abs(risk_per_share - self.ross_preferred_stop)
            if stop_deviation > 0.05:  # More than 5 cents off
                warnings.append(f"Stop distance ${risk_per_share:.2f} deviates from Ross's 20-cent preference")
            if risk_per_share > 0.40:  # Very wide stop
                warnings.append(f"Stop distance ${risk_per_share:.2f} is quite wide for momentum trading")
            elif risk_per_share < 0.10:  # Very tight stop
                warnings.append(f"Stop distance ${risk_per_share:.2f} may be too tight for momentum stocks")
        
        # Calculate position size using Ross's formula
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Validate position against account constraints (Ross uses higher limits with margin)
        position_percentage = position_value / current_account_balance
        
        if position_percentage > self.ross_max_position_percentage:
            warnings.append(f"Position {position_percentage:.1%} exceeds Ross's maximum {self.ross_max_position_percentage:.1%}")
            # Adjust to maximum allowed
            max_position_value = current_account_balance * self.ross_max_position_percentage
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        if position_value < self.min_position_value:
            warnings.append(f"Position value ${position_value:.2f} below minimum ${self.min_position_value}")
        
        # Calculate actual risk with final share count
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / current_account_balance
        
        # Calculate profit targets for Ross's 1:2 system
        target_profit_1_to_1 = actual_risk  # Break even point
        target_profit_2_to_1 = actual_risk * 2  # Primary target
        
        # Buying power calculation (assume 2x margin)
        buying_power_used = position_value / 2  # With 2x margin
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=actual_risk,
            risk_per_share=risk_per_share,
            risk_percentage=actual_risk_pct,
            profit_target=target_profit_2_to_1,
            method_used=SizingMethod.ROSS_CAMERON_DYNAMIC,
            max_loss_if_stopped=actual_risk,
            position_value=position_value,
            buying_power_used=buying_power_used,
            warnings=warnings,
            metadata={
                'account_balance': current_account_balance,
                'target_risk_pct': self.ross_risk_per_trade,
                'target_profit_pct': self.ross_target_per_trade,
                'profit_target_1_to_1': target_profit_1_to_1,
                'profit_target_2_to_1': target_profit_2_to_1,
                'preferred_stop_distance': self.ross_preferred_stop,
                'actual_stop_distance': risk_per_share,
                'risk_reward_ratio': '1:2',
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'daily_risk_budget': current_account_balance * self.ross_daily_risk_budget,
                'remaining_daily_trades': 3  # Assuming this is first trade of day
            }
        )
    
    def calculate_percentage_risk_size(self, account_size: float, entry_price: float,
                                     stop_loss: float, risk_percentage: Optional[float] = None,
                                     custom_risk_amount: Optional[float] = None) -> PositionSizeResult:
        """
        Calculate position size using percentage risk method (Ross Cameron's approach)
        
        Args:
            account_size: Total account value
            entry_price: Entry price per share
            stop_loss: Stop loss price per share
            risk_percentage: Custom risk % (overrides default tier-based %)
            custom_risk_amount: Custom $ risk amount (overrides percentage)
        """
        warnings = []
        
        # Determine risk amount
        if custom_risk_amount:
            risk_amount = custom_risk_amount
            actual_risk_pct = risk_amount / account_size
        else:
            # Use custom percentage or default based on account tier
            if risk_percentage:
                actual_risk_pct = risk_percentage
            else:
                account_tier = self.determine_account_tier(account_size)
                actual_risk_pct = self.default_risk_percentages[account_tier]
            
            risk_amount = account_size * actual_risk_pct
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            warnings.append("Risk per share is zero - invalid stop loss")
            return self._create_invalid_result(SizingMethod.PERCENTAGE_RISK, warnings)
        
        # Calculate shares
        shares = int(risk_amount / risk_per_share)
        
        # Validate position size
        position_value = shares * entry_price
        
        # Check minimum position size
        if position_value < self.min_position_value:
            warnings.append(f"Position value ${position_value:.2f} below minimum ${self.min_position_value}")
        
        # Check maximum position percentage
        position_percentage = position_value / account_size
        if position_percentage > self.max_position_percentage:
            warnings.append(f"Position {position_percentage:.1%} exceeds maximum {self.max_position_percentage:.1%}")
            # Reduce to maximum allowed
            max_position_value = account_size * self.max_position_percentage
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        # Recalculate actual risk with final share count
        actual_risk = shares * risk_per_share
        actual_risk_pct = actual_risk / account_size
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=actual_risk,
            risk_per_share=risk_per_share,
            risk_percentage=actual_risk_pct,
            profit_target=actual_risk * 2,  # Traditional 2:1 target
            method_used=SizingMethod.PERCENTAGE_RISK,
            max_loss_if_stopped=actual_risk,
            position_value=position_value,
            buying_power_used=position_value,
            warnings=warnings,
            metadata={
                'account_tier': self.determine_account_tier(account_size).value,
                'target_risk_pct': actual_risk_pct,
                'entry_price': entry_price,
                'stop_loss': stop_loss
            }
        )
    
    def calculate_fixed_dollar_size(self, entry_price: float, stop_loss: float,
                                  fixed_risk_amount: float, account_size: float) -> PositionSizeResult:
        """
        Calculate position size using fixed dollar risk method
        """
        warnings = []
        
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            warnings.append("Risk per share is zero - invalid stop loss")
            return self._create_invalid_result(SizingMethod.FIXED_DOLLAR, warnings)
        
        # Calculate shares
        shares = int(fixed_risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Validate against account size and enforce cap
        position_percentage = position_value / account_size
        if position_percentage > self.max_position_percentage:
            warnings.append(f"Fixed-dollar position {position_percentage:.1%} exceeds maximum")
            max_position_value = account_size * self.max_position_percentage
            shares = max(1, int(max_position_value / entry_price))
            position_value = shares * entry_price
        
        # Check if fixed risk is reasonable for account size
        risk_percentage = fixed_risk_amount / account_size
        if risk_percentage > 0.05:  # 5% threshold
            warnings.append(f"Fixed risk ${fixed_risk_amount} is {risk_percentage:.1%} of account")
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=fixed_risk_amount,
            risk_per_share=risk_per_share,
            risk_percentage=risk_percentage,
            profit_target=fixed_risk_amount * 2,  # Traditional 2:1 target
            method_used=SizingMethod.FIXED_DOLLAR,
            max_loss_if_stopped=fixed_risk_amount,
            position_value=position_value,
            buying_power_used=position_value,
            warnings=warnings,
            metadata={
                'fixed_risk_amount': fixed_risk_amount,
                'entry_price': entry_price,
                'stop_loss': stop_loss
            }
        )
    
    def calculate_volatility_atr_size(self, price_data: pd.DataFrame, entry_price: float,
                                    account_size: float, atr_period: int = 14,
                                    atr_multiplier: float = 2.5, risk_percentage: float = 0.02) -> PositionSizeResult:
        """
        Calculate position size using ATR-based volatility method
        """
        warnings = []
        
        if len(price_data) < atr_period:
            warnings.append(f"Insufficient data for ATR calculation (need {atr_period} periods)")
            return self._create_invalid_result(SizingMethod.VOLATILITY_ATR, warnings)
        
        # Calculate ATR using TA-Lib or fallback
        if _HAS_TALIB:
            high = price_data['high'].values
            low = price_data['low'].values  
            close = price_data['close'].values
            atr_values = talib.ATR(high, low, close, timeperiod=atr_period)
            current_atr = float(atr_values[-1])
        else:
            current_atr = float(self._atr_fallback(price_data, atr_period))
        
        if pd.isna(current_atr) or current_atr == 0:
            warnings.append("Invalid ATR calculation")
            return self._create_invalid_result(SizingMethod.VOLATILITY_ATR, warnings)
        
        # Calculate stop loss based on ATR
        stop_loss = entry_price - (current_atr * atr_multiplier)
        risk_per_share = abs(entry_price - stop_loss)
        
        # Calculate position size using risk percentage
        risk_amount = account_size * risk_percentage
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Validate position
        position_percentage = position_value / account_size
        if position_percentage > self.max_position_percentage:
            warnings.append(f"ATR-based position {position_percentage:.1%} exceeds maximum")
            # Adjust to maximum
            max_position_value = account_size * self.max_position_percentage
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        actual_risk = shares * risk_per_share
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=actual_risk,
            risk_per_share=risk_per_share,
            risk_percentage=actual_risk / account_size,
            profit_target=actual_risk * 2,  # Traditional 2:1 target
            method_used=SizingMethod.VOLATILITY_ATR,
            max_loss_if_stopped=actual_risk,
            position_value=position_value,
            buying_power_used=position_value,
            warnings=warnings,
            metadata={
                'atr_value': current_atr,
                'atr_multiplier': atr_multiplier,
                'calculated_stop': stop_loss,
                'entry_price': entry_price
            }
        )
    
    def calculate_standard_deviation_size(self, price_data: pd.DataFrame, entry_price: float,
                                        account_size: float, lookback_period: int = 20,
                                        std_multiplier: float = 2.0, risk_percentage: float = 0.02) -> PositionSizeResult:
        """
        Calculate position size using standard deviation method (Bollinger Bands approach)
        """
        warnings = []
        
        if len(price_data) < lookback_period:
            warnings.append(f"Insufficient data for standard deviation calculation")
            return self._create_invalid_result(SizingMethod.STANDARD_DEVIATION, warnings)
        
        # Calculate moving average and standard deviation
        closes = price_data['close']
        moving_avg = closes.rolling(window=lookback_period).mean().iloc[-1]
        std_dev = closes.rolling(window=lookback_period).std().iloc[-1]
        
        if pd.isna(moving_avg) or pd.isna(std_dev) or std_dev == 0:
            warnings.append("Invalid standard deviation calculation")
            return self._create_invalid_result(SizingMethod.STANDARD_DEVIATION, warnings)
        
        # Calculate stop loss based on standard deviation below moving average
        stop_loss = moving_avg - (std_dev * std_multiplier)
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            warnings.append("Calculated stop loss equals entry price")
            return self._create_invalid_result(SizingMethod.STANDARD_DEVIATION, warnings)
        
        # Calculate position size
        risk_amount = account_size * risk_percentage
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Validate position
        position_percentage = position_value / account_size
        if position_percentage > self.max_position_percentage:
            warnings.append(f"Std dev position {position_percentage:.1%} exceeds maximum")
            max_position_value = account_size * self.max_position_percentage
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        actual_risk = shares * risk_per_share
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=actual_risk,
            risk_per_share=risk_per_share,
            risk_percentage=actual_risk / account_size,
            profit_target=actual_risk * 2,  # Traditional 2:1 target
            method_used=SizingMethod.STANDARD_DEVIATION,
            max_loss_if_stopped=actual_risk,
            position_value=position_value,
            buying_power_used=position_value,
            warnings=warnings,
            metadata={
                'moving_average': moving_avg,
                'std_deviation': std_dev,
                'std_multiplier': std_multiplier,
                'calculated_stop': stop_loss,
                'entry_price': entry_price
            }
        )
    
    def calculate_vital_levels_size(self, entry_price: float, support_level: float,
                                  account_size: float, buffer_percentage: float = 0.5,
                                  risk_percentage: float = 0.02) -> PositionSizeResult:
        """
        Calculate position size using vital market levels (support/resistance)
        """
        warnings = []
        
        # Calculate stop loss below support level with buffer
        buffer_amount = support_level * (buffer_percentage / 100)
        stop_loss = support_level - buffer_amount
        
        # Validate that entry is above stop
        if entry_price <= stop_loss:
            warnings.append("Entry price is below or at calculated stop loss")
            return self._create_invalid_result(SizingMethod.VITAL_LEVELS, warnings)
        
        risk_per_share = entry_price - stop_loss
        
        # Calculate position size
        risk_amount = account_size * risk_percentage
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Check for asymmetric risk/reward (key benefit of this method)
        distance_to_support = entry_price - support_level
        if distance_to_support / entry_price < 0.01:  # Very close to support
            warnings.append("Very tight stop - high potential for asymmetric R:R")
        
        # Validate position size
        position_percentage = position_value / account_size
        if position_percentage > self.max_position_percentage:
            warnings.append(f"Vital levels position {position_percentage:.1%} exceeds maximum")
            max_position_value = account_size * self.max_position_percentage
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        actual_risk = shares * risk_per_share
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=actual_risk,
            risk_per_share=risk_per_share,
            risk_percentage=actual_risk / account_size,
            profit_target=actual_risk * 2,  # Traditional 2:1 target
            method_used=SizingMethod.VITAL_LEVELS,
            max_loss_if_stopped=actual_risk,
            position_value=position_value,
            buying_power_used=position_value,
            warnings=warnings,
            metadata={
                'support_level': support_level,
                'buffer_percentage': buffer_percentage,
                'buffer_amount': buffer_amount,
                'calculated_stop': stop_loss,
                'entry_price': entry_price,
                'distance_to_support_pct': distance_to_support / entry_price * 100
            }
        )
    
    def calculate_percentage_loss_size(self, entry_price: float, account_size: float,
                                     max_loss_percentage: float = 7.0, risk_percentage: float = 0.02) -> PositionSizeResult:
        """
        Calculate position size using percentage loss method (William O'Neil's approach)
        """
        warnings = []
        
        # Calculate stop loss based on percentage below entry
        stop_loss = entry_price * (1 - max_loss_percentage / 100)
        risk_per_share = entry_price - stop_loss
        
        # Calculate position size
        risk_amount = account_size * risk_percentage
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Validate position
        position_percentage = position_value / account_size
        if position_percentage > self.max_position_percentage:
            warnings.append(f"Percentage loss position {position_percentage:.1%} exceeds maximum")
            max_position_value = account_size * self.max_position_percentage
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        # Check if stop loss percentage is reasonable
        if max_loss_percentage > 10:
            warnings.append(f"Stop loss of {max_loss_percentage}% is quite wide")
        elif max_loss_percentage < 3:
            warnings.append(f"Stop loss of {max_loss_percentage}% is very tight")
        
        actual_risk = shares * risk_per_share
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=actual_risk,
            risk_per_share=risk_per_share,
            risk_percentage=actual_risk / account_size,
            profit_target=actual_risk * 2,  # Traditional 2:1 target
            method_used=SizingMethod.PERCENTAGE_LOSS,
            max_loss_if_stopped=actual_risk,
            position_value=position_value,
            buying_power_used=position_value,
            warnings=warnings,
            metadata={
                'max_loss_percentage': max_loss_percentage,
                'calculated_stop': stop_loss,
                'entry_price': entry_price
            }
        )
    
    def calculate_kelly_criterion_size(self, win_rate: float, avg_win: float, avg_loss: float,
                                     account_size: float, entry_price: float, stop_loss: float) -> PositionSizeResult:
        """
        Calculate position size using Kelly Criterion for optimal growth
        """
        warnings = []
        
        # Validate inputs
        if not (0 < win_rate < 1):
            warnings.append("Win rate must be between 0 and 1")
            return self._create_invalid_result(SizingMethod.KELLY_CRITERION, warnings)
        
        if avg_win <= 0 or avg_loss <= 0:
            warnings.append("Average win and loss must be positive")
            return self._create_invalid_result(SizingMethod.KELLY_CRITERION, warnings)
        
        # Calculate Kelly percentage: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Cap Kelly percentage (never risk more than 25% on Kelly)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        if kelly_fraction <= 0:
            warnings.append("Kelly criterion suggests no position (negative edge)")
            return self._create_invalid_result(SizingMethod.KELLY_CRITERION, warnings)
        
        # Often use fractional Kelly (e.g., 25% of full Kelly) to reduce volatility
        fractional_kelly = kelly_fraction * 0.25  # Conservative fractional Kelly
        
        # Calculate position size
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            warnings.append("Risk per share is zero")
            return self._create_invalid_result(SizingMethod.KELLY_CRITERION, warnings)
        
        risk_amount = account_size * fractional_kelly
        shares = int(risk_amount / risk_per_share)
        position_value = shares * entry_price
        
        # Validate position
        position_percentage = position_value / account_size
        if position_percentage > self.max_position_percentage:
            warnings.append(f"Kelly position {position_percentage:.1%} exceeds maximum")
            max_position_value = account_size * self.max_position_percentage
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        actual_risk = shares * risk_per_share
        
        return PositionSizeResult(
            shares=shares,
            dollar_amount=position_value,
            risk_amount=actual_risk,
            risk_per_share=risk_per_share,
            risk_percentage=actual_risk / account_size,
            profit_target=actual_risk * 2,  # Traditional 2:1 target
            method_used=SizingMethod.KELLY_CRITERION,
            max_loss_if_stopped=actual_risk,
            position_value=position_value,
            buying_power_used=position_value,
            warnings=warnings,
            metadata={
                'kelly_fraction': kelly_fraction,
                'fractional_kelly': fractional_kelly,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': b,
                'entry_price': entry_price,
                'stop_loss': stop_loss
            }
        )
    
    def _create_invalid_result(self, method: SizingMethod, warnings: List[str]) -> PositionSizeResult:
        """Create invalid result for error cases"""
        return PositionSizeResult(
            shares=0,
            dollar_amount=0.0,
            risk_amount=0.0,
            risk_per_share=0.0,
            risk_percentage=0.0,
            profit_target=0.0,
            method_used=method,
            max_loss_if_stopped=0.0,
            position_value=0.0,
            buying_power_used=0.0,
            warnings=warnings,
            metadata={}
        )
    
    def compare_sizing_methods(self, account_size: float, entry_price: float,
                             stop_loss: float, price_data: Optional[pd.DataFrame] = None,
                             support_level: Optional[float] = None) -> Dict[str, PositionSizeResult]:
        """
        Compare multiple sizing methods for the same trade setup
        """
        results = {}
        
        # Method 1: Percentage Risk (Ross Cameron default)
        results['percentage_risk'] = self.calculate_percentage_risk_size(
            account_size, entry_price, stop_loss
        )
        
        # Method 2: Fixed Dollar (example with $1000 risk)
        results['fixed_dollar'] = self.calculate_fixed_dollar_size(
            entry_price, stop_loss, 1000.0, account_size
        )
        
        # Method 3: Percentage Loss (7% stop like O'Neil)
        loss_percentage = abs(entry_price - stop_loss) / entry_price * 100
        results['percentage_loss'] = self.calculate_percentage_loss_size(
            entry_price, account_size, loss_percentage
        )
        
        # Method 4: ATR-based (if price data available)
        if price_data is not None and len(price_data) >= 14:
            results['volatility_atr'] = self.calculate_volatility_atr_size(
                price_data, entry_price, account_size
            )
            
            results['standard_deviation'] = self.calculate_standard_deviation_size(
                price_data, entry_price, account_size
            )
        
        # Method 5: Vital Levels (if support level provided)
        if support_level is not None:
            results['vital_levels'] = self.calculate_vital_levels_size(
                entry_price, support_level, account_size
            )
        
        return results
    
    def get_recommended_size(self, account_size: float, entry_price: float, stop_loss: float,
                           strategy_type: str = "momentum", risk_tolerance: str = "moderate") -> PositionSizeResult:
        """
        Get recommended position size based on strategy type and risk tolerance
        """
        # Ross Cameron momentum trading (default)
        if strategy_type == "momentum":
            return self.calculate_percentage_risk_size(account_size, entry_price, stop_loss)
        
        # Swing trading with wider stops
        elif strategy_type == "swing":
            # Use lower risk percentage for swing trades
            tier = self.determine_account_tier(account_size)
            base_risk = self.default_risk_percentages[tier]
            swing_risk = base_risk * 0.75  # Reduce risk for swing trades
            
            return self.calculate_percentage_risk_size(
                account_size, entry_price, stop_loss, swing_risk
            )
        
        # Scalping with tight stops
        elif strategy_type == "scalp":
            # Can use higher risk percentage due to tight stops
            tier = self.determine_account_tier(account_size)
            base_risk = self.default_risk_percentages[tier]
            scalp_risk = min(base_risk * 1.5, 0.03)  # Max 3% for scalping
            
            return self.calculate_percentage_risk_size(
                account_size, entry_price, stop_loss, scalp_risk
            )
        
        else:
            # Default to percentage risk
            return self.calculate_percentage_risk_size(account_size, entry_price, stop_loss)

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize position sizer
    sizer = PositionSizer()
    
    # Example trade setup (using your $30K account framework)
    account_size = 30000  # $30K account
    entry_price = 10.00
    stop_loss = 9.80  # 20-cent stop (Ross's preference)
    
    print("=== ROSS CAMERON'S DYNAMIC POSITION SIZING ===")
    print(f"Current Account Balance: ${account_size:,}")
    print(f"Entry Price: ${entry_price}")
    print(f"Stop Loss: ${stop_loss}")
    print(f"Risk Per Share: ${entry_price - stop_loss:.2f}")
    print()
    
    # Ross Cameron's new dynamic method
    result = sizer.calculate_ross_cameron_dynamic_size(account_size, entry_price, stop_loss)
    
    print("=== ROSS CAMERON'S 1:2 RISK/REWARD SYSTEM ===")
    print(f"Position Size: {result.shares:,} shares")
    print(f"Position Value: ${result.position_value:,.2f}")
    print(f"Risk Amount: ${result.risk_amount:.2f} ({result.risk_percentage:.2%})")
    print(f"Profit Target: ${result.profit_target:.2f} ({result.metadata['target_profit_pct']:.2%})")
    print(f"Risk/Reward Ratio: {result.metadata['risk_reward_ratio']}")
    print(f"Daily Risk Budget: ${result.metadata['daily_risk_budget']:.2f}")
    print(f"Buying Power Used: ${result.buying_power_used:,.2f} (with 2x margin)")
    
    if result.warnings:
        print("Warnings:", ", ".join(result.warnings))
    
    print("\n=== ACCOUNT SCALING EXAMPLES ===")
    for test_account in [25000, 35000, 50000]:
        test_result = sizer.calculate_ross_cameron_dynamic_size(test_account, entry_price, stop_loss)
        print(f"${test_account:,} Account: {test_result.shares:,} shares, "
              f"Risk ${test_result.risk_amount:.0f}, Target ${test_result.profit_target:.0f}")
    
    print("\n=== COMPARISON OF ALL METHODS ===")
    
    # Create sample price data for ATR calculation
    dates = pd.date_range('2023-01-01', periods=30, freq='D')
    sample_data = pd.DataFrame({
        'high': np.random.normal(entry_price * 1.02, 0.5, 30),
        'low': np.random.normal(entry_price * 0.98, 0.5, 30),
        'close': np.random.normal(entry_price, 0.3, 30)
    }, index=dates)
    
    # Compare all methods
    comparison = sizer.compare_sizing_methods(
        account_size, entry_price, stop_loss, sample_data, support_level=14.50
    )
    
    for method_name, result in comparison.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Shares: {result.shares}")
        print(f"  Position Value: ${result.position_value:,.2f}")
        print(f"  Risk: ${result.risk_amount:.2f} ({result.risk_percentage:.2%})")
        if result.warnings:
            print(f"  Warnings: {len(result.warnings)}")
    
    print("\nPosition Sizer Module Complete!")
    print("Features:")
    print("- Ross Cameron's 2% risk rule with account tier scaling")
    print("- Multiple sizing methodologies (6 different methods)")
    print("- Comprehensive validation and warnings")
    print("- Metadata tracking for analysis")
    print("- Strategy-specific recommendations")