from __future__ import annotations

from datetime import datetime
from typing import Optional, Set

import pandas as pd
import logging


class BullFlagSimpleStrategy:
    """
    Initial simple bull-flag strategy scaffold (duplicated from alert_flip):
    - Buy on the close of the ACTION alert bar (or the first bar after it)
    - Sell on:
        * the close of the first NON-alert bar after entry, or
        * the close of an ALERT bar if that alert bar is RED (close < open)

    NOTE: This is intentionally identical to AlertFlipStrategy at creation time,
    and will be evolved into a proper bull flag detector step-by-step.
    """

    def __init__(self, all_alert_times: Optional[Set[pd.Timestamp]] = None) -> None:
        self._entry_done = False
        self._exit_pending = False
        self._entry_index: Optional[pd.Timestamp] = None
        self._alert_times: Set[pd.Timestamp] = set(all_alert_times or set())

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        """Handle one bar for the simple bull-flag strategy.

        Returns True if this strategy took an action (entry/exit) on this bar.
        """
        # Entry at/after the alert bar close
        try:
            if (not self._entry_done) and (session.position is None) and (session.alert is not None):
                at = session.alert.alert_time
                # Strict: enter only on the ACTION alert bar close
                if df.index[-1] == at:
                    entry_price = float(df['close'].iloc[-1]) + float(session.entry_slippage_cents or 0.0)
                    try:
                        from risk_config import PREFERRED_STOP_DISTANCE_PCT
                        stop_pct = float(PREFERRED_STOP_DISTANCE_PCT)
                    except Exception:
                        stop_pct = 0.02
                    stop_loss = max(0.01, entry_price * (1.0 - stop_pct))
                    logging.info("BFS ENTRY %s | price=%.4f stop=%.4f", timestamp, entry_price, stop_loss)
                    session._enter_direct(entry_price, stop_loss, timestamp, reason="BullFlag_Simple_Entry")
                    self._entry_done = True
                    self._exit_pending = True
                    self._entry_index = df.index[-1]
                    return True
        except Exception:
            pass

        # Exit logic: first candle (alert or non-alert) with MACD bearish
        try:
            if self._exit_pending and session.position is not None:
                if self._entry_index is not None and len(df) >= 2 and df.index[-1] > self._entry_index:
                    try:
                        macd_val = float(df['macd'].iloc[-1])
                        macd_sig = float(df['macd_signal'].iloc[-1])
                        macd_bearish = (macd_val < macd_sig) or (macd_val < 0.0)
                    except Exception:
                        macd_bearish = False
                    logging.info("BFS CHECK %s | macd=%.6f sig=%.6f bearish=%s", timestamp, macd_val if 'macd_val' in locals() else float('nan'), macd_sig if 'macd_sig' in locals() else float('nan'), macd_bearish)

                    if macd_bearish:
                        exit_price = float(df['close'].iloc[-1])
                        logging.info("BFS EXIT %s | price=%.4f (MACD bearish)", timestamp, exit_price)
                        session._exit_position(timestamp, exit_price, session.ExitReason.NEXT_BAR_CLOSE, session.position.current_shares)
                        self._exit_pending = False
                        return True
                    else:
                        return False
        except Exception:
            pass

        return False
