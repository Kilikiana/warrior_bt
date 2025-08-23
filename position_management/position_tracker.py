"""
Position Tracker and Risk Manager

OVERVIEW:
Comprehensive position tracking system that enforces Ross Cameron's risk limits:
- Maximum 4 positions simultaneously 
- Daily risk budget: $1500 for $30K account (5% of account)
- Real-time exposure monitoring
- Position validation before entry

ROSS'S RISK MANAGEMENT RULES:
- Never exceed 4 positions at once (concentration risk)
- Daily risk budget prevents overexposure
- Each position properly sized using dynamic risk/reward
- Automatic risk limit validation before new positions

INTEGRATION:
- Works with position_sizer.py for sizing calculations
- Works with pattern_monitor.py for trade execution
- Provides real-time risk validation and limits
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, NamedTuple, Set
from enum import Enum
from datetime import datetime, date
import logging

# Import centralized risk configuration
from risk_config import DAILY_RISK_PCT, MAX_CONCURRENT_POSITIONS

class PositionStatus(Enum):
    """Current position status"""
    ACTIVE = "active"
    SCALING_OUT = "scaling_out"
    HOLDING_RUNNER = "holding_runner"
    CLOSED = "closed"

class TrackedPosition(NamedTuple):
    """Position being tracked"""
    symbol: str
    entry_time: datetime
    entry_price: float
    initial_shares: int
    current_shares: int
    initial_risk: float  # $ amount risked on entry
    current_risk: float  # $ amount currently at risk
    stop_loss: float
    status: PositionStatus
    unrealized_pnl: float

class DailyRiskSummary(NamedTuple):
    """Daily risk tracking summary"""
    date: date
    risk_budget: float
    risk_used: float
    risk_remaining: float
    positions_taken: int
    max_positions_limit: int
    current_active_positions: int

class RiskLimitViolation(Exception):
    """Raised when a trade would violate risk limits"""
    pass

class PositionTracker:
    """
    Ross Cameron Position Tracker and Risk Manager
    Enforces maximum positions and daily risk budgets
    """
    
    def __init__(self, account_balance: float, max_positions: int = MAX_CONCURRENT_POSITIONS, 
                 daily_risk_percentage: float = DAILY_RISK_PCT):
        self.account_balance = account_balance
        self.max_positions = max_positions
        self.daily_risk_percentage = daily_risk_percentage
        self.daily_risk_budget = account_balance * daily_risk_percentage
        
        # Position tracking
        self.active_positions: Dict[str, TrackedPosition] = {}
        self.closed_positions: List[TrackedPosition] = []
        self.daily_risk_used = 0.0
        self.current_date = datetime.now().date()
        
        logging.info(f"Position Tracker initialized: ${account_balance:,.0f} account, "
                    f"max {max_positions} positions, ${self.daily_risk_budget:.0f} daily risk budget")
    
    def update_account_balance(self, new_balance: float) -> None:
        """Update account balance and recalculate risk budget"""
        old_budget = self.daily_risk_budget
        self.account_balance = new_balance
        self.daily_risk_budget = new_balance * self.daily_risk_percentage
        
        logging.info(f"Account balance updated: ${new_balance:,.0f}, "
                    f"daily risk budget: ${old_budget:.0f} → ${self.daily_risk_budget:.0f}")
    
    def reset_daily_limits(self) -> None:
        """Reset daily risk tracking (called at market open)"""
        self.daily_risk_used = 0.0
        self.current_date = datetime.now().date()
        
        # Move closed positions to history
        for symbol in list(self.active_positions.keys()):
            if self.active_positions[symbol].status == PositionStatus.CLOSED:
                closed_position = self.active_positions.pop(symbol)
                self.closed_positions.append(closed_position)
        
        logging.info(f"Daily limits reset for {self.current_date}: "
                    f"${self.daily_risk_budget:.0f} budget available")
    
    def can_open_position(self, risk_amount: float) -> Tuple[bool, str]:
        """
        Check if new position can be opened within limits
        
        Returns: (can_open, reason_if_not)
        """
        # Check if it's a new day (reset limits automatically)
        if datetime.now().date() != self.current_date:
            self.reset_daily_limits()
        
        # Check maximum positions limit
        active_count = len([p for p in self.active_positions.values() 
                           if p.status != PositionStatus.CLOSED])
        
        if active_count >= self.max_positions:
            return False, f"Maximum {self.max_positions} positions already active ({active_count})"
        
        # Check daily risk budget
        if self.daily_risk_used + risk_amount > self.daily_risk_budget:
            remaining = self.daily_risk_budget - self.daily_risk_used
            return False, f"Daily risk budget exceeded: ${risk_amount:.0f} requested, " \
                         f"${remaining:.0f} remaining of ${self.daily_risk_budget:.0f} budget"
        
        return True, "Position approved within risk limits"
    
    def add_position(self, symbol: str, entry_time: datetime, entry_price: float,
                    shares: int, risk_amount: float, stop_loss: float) -> None:
        """Add new position to tracking"""
        
        # Validate position can be opened
        can_open, reason = self.can_open_position(risk_amount)
        if not can_open:
            raise RiskLimitViolation(f"Cannot open {symbol} position: {reason}")
        
        # Create tracked position
        position = TrackedPosition(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=entry_price,
            initial_shares=shares,
            current_shares=shares,
            initial_risk=risk_amount,
            current_risk=risk_amount,
            stop_loss=stop_loss,
            status=PositionStatus.ACTIVE,
            unrealized_pnl=0.0
        )
        
        self.active_positions[symbol] = position
        self.daily_risk_used += risk_amount
        
        logging.info(f"Added position {symbol}: {shares} shares at ${entry_price:.2f}, "
                    f"risk ${risk_amount:.0f}, daily risk used: ${self.daily_risk_used:.0f}")
    
    def update_position(self, symbol: str, current_price: float, 
                       shares_change: int = 0, status_change: Optional[PositionStatus] = None,
                       stop_loss_update: Optional[float] = None) -> None:
        """Update existing position"""
        if symbol not in self.active_positions:
            logging.warning(f"Attempted to update non-existent position: {symbol}")
            return
        
        position = self.active_positions[symbol]
        
        # Calculate new values
        new_shares = max(0, position.current_shares + shares_change)
        new_risk = (new_shares / position.initial_shares) * position.initial_risk if new_shares > 0 else 0.0
        unrealized_pnl = new_shares * (current_price - position.entry_price)
        new_status = status_change if status_change else position.status
        new_stop = stop_loss_update if stop_loss_update else position.stop_loss
        
        # Update position
        self.active_positions[symbol] = position._replace(
            current_shares=new_shares,
            current_risk=new_risk,
            unrealized_pnl=unrealized_pnl,
            status=new_status,
            stop_loss=new_stop
        )
        
        # Update daily risk tracking
        risk_reduction = position.current_risk - new_risk
        self.daily_risk_used = max(0, self.daily_risk_used - risk_reduction)
        
        logging.info(f"Updated {symbol}: {new_shares} shares, ${new_risk:.0f} risk, "
                    f"${unrealized_pnl:+.0f} P&L, daily risk: ${self.daily_risk_used:.0f}")
    
    def close_position(self, symbol: str, exit_price: float, exit_time: datetime) -> float:
        """Close position completely and return realized P&L"""
        if symbol not in self.active_positions:
            logging.warning(f"Attempted to close non-existent position: {symbol}")
            return 0.0
        
        position = self.active_positions[symbol]
        realized_pnl = position.current_shares * (exit_price - position.entry_price)
        
        # Update to closed status and move to closed_positions immediately
        closed_position = position._replace(
            current_shares=0,
            current_risk=0.0,
            unrealized_pnl=realized_pnl,
            status=PositionStatus.CLOSED
        )
        
        # Move to closed_positions immediately to preserve invariants
        self.closed_positions.append(closed_position)
        del self.active_positions[symbol]
        
        # Reduce daily risk used
        self.daily_risk_used = max(0, self.daily_risk_used - position.current_risk)
        
        logging.info(f"Closed {symbol}: {position.current_shares} shares at ${exit_price:.2f}, "
                    f"realized P&L: ${realized_pnl:+.0f}")
        
        return realized_pnl
    
    def get_position_summary(self) -> Dict:
        """Get current position summary"""
        active_positions = [p for p in self.active_positions.values() 
                           if p.status != PositionStatus.CLOSED]
        
        total_risk = sum(p.current_risk for p in active_positions)
        total_unrealized_pnl = sum(p.unrealized_pnl for p in active_positions)
        
        return {
            'account_balance': self.account_balance,
            'active_positions_count': len(active_positions),
            'max_positions_limit': self.max_positions,
            'positions_remaining': self.max_positions - len(active_positions),
            'daily_risk_budget': self.daily_risk_budget,
            'daily_risk_used': self.daily_risk_used,
            'daily_risk_remaining': self.daily_risk_budget - self.daily_risk_used,
            'total_current_risk': total_risk,
            'total_unrealized_pnl': total_unrealized_pnl,
            'active_symbols': [p.symbol for p in active_positions]
        }
    
    def get_daily_summary(self) -> DailyRiskSummary:
        """Get daily risk summary"""
        active_count = len([p for p in self.active_positions.values() 
                           if p.status != PositionStatus.CLOSED])
        
        return DailyRiskSummary(
            date=self.current_date,
            risk_budget=self.daily_risk_budget,
            risk_used=self.daily_risk_used,
            risk_remaining=self.daily_risk_budget - self.daily_risk_used,
            positions_taken=len(self.active_positions) + len(self.closed_positions),
            max_positions_limit=self.max_positions,
            current_active_positions=active_count
        )
    
    def validate_position_request(self, symbol: str, risk_amount: float) -> None:
        """Validate position request and raise exception if invalid"""
        if symbol in self.active_positions and self.active_positions[symbol].status != PositionStatus.CLOSED:
            raise RiskLimitViolation(f"Position {symbol} already exists")
        
        can_open, reason = self.can_open_position(risk_amount)
        if not can_open:
            raise RiskLimitViolation(f"Position validation failed: {reason}")

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize tracker for $30K account with $1500 daily risk budget
    tracker = PositionTracker(account_balance=30000, max_positions=4, daily_risk_percentage=0.05)
    
    print("=== POSITION TRACKER TEST ===")
    print(f"Account: ${tracker.account_balance:,}")
    print(f"Daily Risk Budget: ${tracker.daily_risk_budget:.0f}")
    print(f"Max Positions: {tracker.max_positions}")
    print()
    
    try:
        # Test adding positions within limits
        print("Adding position 1: CLRO")
        tracker.add_position("CLRO", datetime.now(), 15.34, 1875, 375.0, 15.14)
        
        print("Adding position 2: ABCD") 
        tracker.add_position("ABCD", datetime.now(), 8.50, 2200, 375.0, 8.30)
        
        print("Adding position 3: EFGH")
        tracker.add_position("EFGH", datetime.now(), 12.00, 1560, 375.0, 11.76)
        
        print("Adding position 4: IJKL")
        tracker.add_position("IJKL", datetime.now(), 6.75, 2700, 375.0, 6.61)
        
        # Test position limit
        print("\nTrying to add 5th position (should fail):")
        tracker.add_position("FAIL", datetime.now(), 10.00, 1500, 375.0, 9.75)
        
    except RiskLimitViolation as e:
        print(f"Risk Limit Violation: {e}")
    
    # Show summary
    print("\n=== POSITION SUMMARY ===")
    summary = tracker.get_position_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Test position scaling
    print("\n=== TESTING POSITION SCALING ===")
    tracker.update_position("CLRO", 17.50, shares_change=-900, status_change=PositionStatus.SCALING_OUT)
    
    summary = tracker.get_position_summary()
    print(f"After scaling CLRO: Risk remaining ${summary['daily_risk_remaining']:.0f}")