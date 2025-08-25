from __future__ import annotations

from datetime import datetime
from typing import Optional, Set

import pandas as pd


class AlertFlipStrategy:
    """
    Minimal "alert close flip" strategy:
    - Buy on the close of the ACTION alert bar (or the first bar after it)
    - Sell on:
        * the close of the first NON-alert bar after entry, or
        * the close of an ALERT bar if that alert bar is RED (close < open)

    This module is intentionally small and stateless beyond a few flags so it
    can be imported and used by PatternMonitoringSession without tangling logic.
    """

    def __init__(self, all_alert_times: Optional[Set[pd.Timestamp]] = None) -> None:
        self._entry_done = False
        self._exit_pending = False
        self._entry_index: Optional[pd.Timestamp] = None
        self._alert_times: Set[pd.Timestamp] = set(all_alert_times or set())

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        """Handle one bar for the alert-flip strategy.

        Returns True if this strategy took an action (entry/exit) on this bar.
        """
        # Entry at/after the alert bar close
        try:
            if (not self._entry_done) and (session.position is None) and (session.alert is not None):
                at = session.alert.alert_time
                if df.index[-1] >= at:
                    entry_price = float(df['close'].iloc[-1]) + float(session.entry_slippage_cents or 0.0)
                    try:
                        from risk_config import PREFERRED_STOP_DISTANCE_PCT
                        stop_pct = float(PREFERRED_STOP_DISTANCE_PCT)
                    except Exception:
                        stop_pct = 0.02
                    stop_loss = max(0.01, entry_price * (1.0 - stop_pct))
                    session._enter_direct(entry_price, stop_loss, timestamp, reason="Alert_Close_Flip_Entry")
                    self._entry_done = True
                    self._exit_pending = True
                    self._entry_index = df.index[-1]
                    return True
        except Exception:
            pass

        # Exit at first NON-alert bar after entry
        try:
            if self._exit_pending and session.position is not None:
                if self._entry_index is not None and len(df) >= 2 and df.index[-1] > self._entry_index:
                    # If this bar is an alert itself, exit early if RED; otherwise defer to next
                    try:
                        if self._alert_times and (df.index[-1] in self._alert_times):
                            try:
                                cur_open = float(df['open'].iloc[-1])
                                cur_close = float(df['close'].iloc[-1])
                                is_red = cur_close < cur_open
                            except Exception:
                                is_red = False
                            if not is_red:
                                # Alert bar but green/doji: defer exit
                                return False
                            # Alert bar and red: fall through to exit below
                    except Exception:
                        pass
                    exit_price = float(df['close'].iloc[-1])
                    session._exit_position(timestamp, exit_price, session.ExitReason.NEXT_BAR_CLOSE, session.position.current_shares)
                    self._exit_pending = False
                    return True
        except Exception:
            pass

        return False
