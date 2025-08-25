from __future__ import annotations

from datetime import datetime
import pandas as pd


class BullFlagStrategy:
    """
    Thin wrapper to route per-bar handling for the existing bull flag
    logic through a strategy interface compatible with AlertFlipStrategy.

    All logic remains in PatternMonitoringSession; this simply calls the
    session's internal handler to preserve behavior and minimize risk.
    """

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        # Delegate to the session's existing bull-flag on-bar implementation.
        # Returns True if an action (entry/exit/state change) occurred.
        return bool(session._on_bar_bull_flag(df, timestamp))

