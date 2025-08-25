from __future__ import annotations

from datetime import datetime
from typing import Optional, Set

import pandas as pd
import logging


class BullFlagSimpleV2Strategy:
    """
    Bull flag simple v2: Wait for a contiguous red pullback after ACTION alert,
    then enter intrabar when the first green breaks the last red's high.
    - Entry price: last red candle's high (intrabar stop trigger)
    - Stop loss: configurable percent below entry (via risk_config)
    - Exit behavior: keep the simple flip exits for now (same as alert_flip)
    """

    def __init__(self, all_alert_times: Optional[Set[pd.Timestamp]] = None) -> None:
        self._entry_done = False
        self._exit_pending = False
        self._entry_index: Optional[pd.Timestamp] = None
        self._alert_times: Set[pd.Timestamp] = set(all_alert_times or set())
        # Pullback state
        self._pullback_started: bool = False
        self._last_red_high: Optional[float] = None
        # Count bars starting from the FIRST RED bar AFTER the ACTION alert
        self._bars_since_pullback_start: int = 0
        self._max_wait_candles: int = 6  # stop waiting after N bars from first red after alert
        self._pullback_start_index: Optional[pd.Timestamp] = None

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        # Entry: after ACTION alert, wait for contiguous red pullback, then
        # first green that breaks last_red_high. Intrabar trigger at last_red_high.
        try:
            if (not self._entry_done) and (session.position is None) and (session.alert is not None):
                at = session.alert.alert_time
                # Only consider bars strictly AFTER the alert bar for pullback logic
                if df.index[-1] > at:
                    cur_open = float(df['open'].iloc[-1])
                    cur_close = float(df['close'].iloc[-1])
                    cur_high = float(df['high'].iloc[-1])
                    is_red = cur_close < cur_open
                    is_green = cur_close > cur_open

                    # Build/extend the contiguous red pullback
                    if not self._pullback_started:
                        if is_red:
                            self._pullback_started = True
                            self._last_red_high = cur_high
                            # Initialize bar count window starting at first red after alert
                            self._bars_since_pullback_start = 1
                            try:
                                self._pullback_start_index = df.index[-1]
                            except Exception:
                                self._pullback_start_index = None
                            # Log pullback start for traceability
                            try:
                                logging.info(
                                    "BFSv2 pullback-start: %s | first_red=%s last_red_high=%.4f",
                                    getattr(session, 'symbol', '?'),
                                    str(self._pullback_start_index) if self._pullback_start_index is not None else 'n/a',
                                    float(self._last_red_high) if self._last_red_high is not None else float('nan')
                                )
                            except Exception:
                                pass
                        # else: still waiting for first red
                    else:
                        if is_red:
                            # Extend pullback; update last red candle's high
                            self._last_red_high = cur_high
                        elif is_green and self._last_red_high is not None:
                            # Trigger if green breaks last red high
                            if cur_high >= self._last_red_high:
                                entry_price = float(self._last_red_high) + float(session.entry_slippage_cents or 0.0)
                                try:
                                    from risk_config import PREFERRED_STOP_DISTANCE_PCT
                                    stop_pct = float(PREFERRED_STOP_DISTANCE_PCT)
                                except Exception:
                                    stop_pct = 0.02
                                stop_loss = max(0.01, entry_price * (1.0 - stop_pct))
                                # Log entry trigger details
                                try:
                                    logging.info(
                                        "BFSv2 entry: %s | price=%.4f stop=%.4f trigger_bar=%s last_red_high=%.4f",
                                        getattr(session, 'symbol', '?'),
                                        float(entry_price),
                                        float(stop_loss),
                                        str(df.index[-1]),
                                        float(self._last_red_high) if self._last_red_high is not None else float('nan')
                                    )
                                except Exception:
                                    pass
                                session._enter_direct(entry_price, stop_loss, timestamp, reason="BullFlag_Simple_V2_Entry")
                                self._entry_done = True
                                self._exit_pending = True
                                self._entry_index = df.index[-1]
                                return True
                        # doji or green without break: neither extend nor trigger; just wait

                        # Increment bar counter within the pullback window (no entry on this bar)
                        try:
                            self._bars_since_pullback_start += 1
                        except Exception:
                            self._bars_since_pullback_start = max(1, int(self._bars_since_pullback_start or 0) + 1)

                    # Stop waiting after N bars from first red after alert without entry
                    if self._pullback_started and (not self._entry_done):
                        if self._bars_since_pullback_start >= self._max_wait_candles:
                            # Log a concise timeout message for backtest traceability
                            try:
                                logging.info(
                                    "BFSv2 timeout: %s | no breakout within %d bars from first red after alert (first_red=%s)",
                                    getattr(session, 'symbol', '?'),
                                    int(self._max_wait_candles),
                                    str(self._pullback_start_index) if self._pullback_start_index is not None else 'n/a'
                                )
                            except Exception:
                                pass
                            session.status = session.MonitoringStatus.MONITORING_STOPPED  # type: ignore
                            return False
        except Exception:
            pass

        # Exit on first RED candle after entry (regardless of alert status)
        try:
            if self._exit_pending and session.position is not None:
                if self._entry_index is not None and len(df) >= 2 and df.index[-1] > self._entry_index:
                    try:
                        cur_open = float(df['open'].iloc[-1])
                        cur_close = float(df['close'].iloc[-1])
                        is_red = cur_close < cur_open
                    except Exception:
                        is_red = False
                    # Trace exit check for debugging
                    try:
                        logging.info(
                            "BFSv2 exit-check: %s | bar=%s open=%.4f close=%.4f is_red=%s entry_idx=%s",
                            getattr(session, 'symbol', '?'),
                            str(df.index[-1]),
                            float(df['open'].iloc[-1]) if 'open' in df.columns else float('nan'),
                            float(df['close'].iloc[-1]) if 'close' in df.columns else float('nan'),
                            str(is_red),
                            str(self._entry_index)
                        )
                    except Exception:
                        pass
                    if is_red:
                        exit_price = float(df['close'].iloc[-1])
                        try:
                            logging.info(
                                "BFSv2 exit: %s | bar=%s price=%.4f (first red after entry)",
                                getattr(session, 'symbol', '?'),
                                str(df.index[-1]),
                                float(exit_price)
                            )
                        except Exception:
                            pass
                        try:
                            session._exit_position(timestamp, exit_price, session.ExitReason.NEXT_BAR_CLOSE, session.position.current_shares)
                            self._exit_pending = False
                            return True
                        except Exception as e:
                            try:
                                logging.warning("BFSv2 exit error for %s at %s: %s", getattr(session, 'symbol', '?'), str(df.index[-1]), e)
                            except Exception:
                                pass
                            # Do not swallow; keep exit pending for next bar
                            return False
                    else:
                        return False
        except Exception:
            pass

        return False
