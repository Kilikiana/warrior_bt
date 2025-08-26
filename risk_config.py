# risk_config.py
"""
Centralized Risk Management Configuration

Single source of truth for all risk parameters to prevent drift between modules.
Based on Ross Cameron's actual trading rules.
"""

# Daily risk limits
DAILY_RISK_PCT = 0.05           # 5% daily risk (conservative for Ross's 4-position limit)
RISK_PER_TRADE_PCT = 0.0125     # 1.25% risk per trade (Ross's actual rule)
MAX_CONCURRENT_POSITIONS = 4    # Ross's position limit

# Profit targets
PROFIT_TARGET_RATIO = 2.0       # 2:1 profit target (Ross's minimum)
SCALE_OUT_FIRST_TARGET_PCT = 0.5 # Scale out 50% at first target

# Position sizing limits
MAX_POSITION_PCT = 0.33         # Max 33% of account per position (with 2x margin = 67% buying power)
PREFERRED_STOP_DISTANCE_PCT = 0.02  # Preferred 2% stop distance

# ATR volatility sizing
ATR_PERIOD = 14                 # Standard ATR period
ATR_MULTIPLIER = 2.0           # Stop distance = ATR * multiplier

# Fixed dollar amounts (fallback sizing)
DEFAULT_FIXED_DOLLAR_AMOUNT = 1000  # Default position size if other methods fail
