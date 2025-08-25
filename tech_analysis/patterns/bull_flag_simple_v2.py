from __future__ import annotations

from datetime import datetime
from typing import Optional, Set

import pandas as pd
import logging


class BullFlagSimpleV2Strategy:
    """
    Bull flag simple v2 (updated entry): After an ACTION alert, buy the break
    of the first candle that makes a new high above the immediately prior
    candle (prior candle may be green or red). Intrabar fill at the prior
    candle's high (+ optional slippage). If the entry candle closes red,
    immediately exit on that same bar close. Retains the 6-bar timeout counted
    from the first red after the alert to avoid monitoring stale setups.
    - Entry price: prior candle's high (intrabar stop trigger)
    - Stop loss: configurable percent below entry (via risk_config)
    - Exit behavior: first red after entry; plus immediate-same-bar exit if entry bar closes red
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
        # Target/stop tracking
        self._target_price: Optional[float] = None
        self._stop_price: Optional[float] = None

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        # Entry: after ACTION alert, buy first bar whose HIGH >= prior bar's HIGH
        # (prior bar may be green or red). Intrabar trigger at prior bar's high.
        try:
            if (not self._entry_done) and (session.position is None) and (session.alert is not None):
                at = session.alert.alert_time
                if df.index[-1] > at and len(df) >= 2:
                    prev_bar_ts = df.index[-2]
                    # Ensure prior bar is not before alert (require a post-alert prior)
                    if prev_bar_ts > at:
                        prev_high = float(df['high'].iloc[-2])
                        cur_open = float(df['open'].iloc[-1])
                        cur_close = float(df['close'].iloc[-1])
                        cur_high = float(df['high'].iloc[-1])
                        # Trigger: current high breaks prior bar high
                        if cur_high >= prev_high:
                            entry_price = float(prev_high) + float(session.entry_slippage_cents or 0.0)
                            # New stop/target rules:
                            # - Stop loss: Low of the pivot candle (prior bar)
                            # - Profit target: Action alert high; sell ALL at target
                            pivot_low = float(df['low'].iloc[-2])
                            self._stop_price = max(0.01, float(pivot_low))
                            try:
                                target = float(getattr(session.alert, 'alert_high', None) or float('nan'))
                            except Exception:
                                target = float('nan')
                            self._target_price = target if target == target else None  # nan-safe
                            try:
                                from risk_config import PREFERRED_STOP_DISTANCE_PCT
                                _ = float(PREFERRED_STOP_DISTANCE_PCT)  # no-op, kept for compatibility
                            except Exception:
                                pass
                            stop_loss = float(self._stop_price)
                            try:
                                logging.info(
                                    "BFSv2 entry: %s | price=%.4f stop=%.4f trigger_bar=%s prev_high=%.4f",
                                    getattr(session, 'symbol', '?'),
                                    float(entry_price),
                                    float(stop_loss),
                                    str(df.index[-1]),
                                    float(prev_high)
                                )
                            except Exception:
                                pass
                            session._enter_direct(entry_price, stop_loss, timestamp, reason="BullFlag_Simple_V2_Entry")
                            self._entry_done = True
                            self._exit_pending = True
                            self._entry_index = df.index[-1]
                            # Immediate exit if entry bar closes red (kept unless instructed otherwise)
                            try:
                                if cur_close < cur_open and session.position is not None:
                                    exit_price = float(cur_close)
                                    logging.info(
                                        "BFSv2 exit: %s | bar=%s price=%.4f (entry bar closed red)",
                                        getattr(session, 'symbol', '?'),
                                        str(df.index[-1]),
                                        float(exit_price)
                                    )
                                    session._exit_position(timestamp, exit_price, session.ExitReason.NEXT_BAR_CLOSE, session.position.current_shares)
                                    self._exit_pending = False
                            except Exception:
                                pass
                            return True

                # Maintain pullback anchor and timeout accounting (from first red after alert)
                if df.index[-1] > at:
                    try:
                        cur_open = float(df['open'].iloc[-1])
                        cur_close = float(df['close'].iloc[-1])
                        is_red = cur_close < cur_open
                    except Exception:
                        is_red = False
                    if not self._pullback_started and is_red:
                        self._pullback_started = True
                        self._bars_since_pullback_start = 1
                        try:
                            self._pullback_start_index = df.index[-1]
                        except Exception:
                            self._pullback_start_index = None
                        try:
                            logging.info(
                                "BFSv2 pullback-start: %s | first_red=%s last_red_high=%.4f",
                                getattr(session, 'symbol', '?'),
                                str(self._pullback_start_index) if self._pullback_start_index is not None else 'n/a',
                                float('nan')
                            )
                        except Exception:
                            pass
                    elif self._pullback_started and not self._entry_done:
                        self._bars_since_pullback_start += 1
                        if self._bars_since_pullback_start >= self._max_wait_candles:
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

        # Exit/management after entry: stop at pivot low; sell ALL at action alert high
        try:
            if self._exit_pending and session.position is not None:
                if self._entry_index is not None and len(df) >= 2 and df.index[-1] > self._entry_index:
                    cur = df.iloc[-1]
                    cur_open = float(cur['open'])
                    cur_high = float(cur['high'])
                    cur_low = float(cur['low'])
                    # 1) Profit target hit: sell ALL at target
                    if self._target_price is not None and cur_high >= float(self._target_price):
                        price = float(self._target_price)
                        try:
                            logging.info(
                                "BFSv2 exit: %s | bar=%s price=%.4f (target: alert_high)",
                                getattr(session, 'symbol', '?'), str(df.index[-1]), price)
                        except Exception:
                            pass
                        session._exit_position(timestamp, price, session.ExitReason.NEXT_BAR_CLOSE, session.position.current_shares)
                        self._exit_pending = False
                        return True
                    # 2) Stop loss hit: simulate intrabar stop fill (open gap or low breach)
                    stop = float(self._stop_price) if self._stop_price is not None else float(getattr(session.position, 'stop_loss', 0.0) or 0.0)
                    if stop > 0.0:
                        stop_hit = False
                        stop_fill = None
                        if cur_open <= stop:
                            stop_hit = True
                            base = cur_open
                            stop_fill = max(cur_low, base - float(getattr(session, 'stop_slippage_cents', 0.0) or 0.0))
                        elif cur_low <= stop:
                            stop_hit = True
                            base = stop
                            stop_fill = max(cur_low, base - float(getattr(session, 'stop_slippage_cents', 0.0) or 0.0))
                        if stop_hit:
                            price = float(stop_fill)
                            try:
                                logging.info(
                                    "BFSv2 exit: %s | bar=%s price=%.4f (stop: pivot low)",
                                    getattr(session, 'symbol', '?'), str(df.index[-1]), price)
                            except Exception:
                                pass
                            session._exit_position(timestamp, price, session.ExitReason.NEXT_BAR_CLOSE, session.position.current_shares)
                            self._exit_pending = False
                            return True
        except Exception:
            pass

        return False
