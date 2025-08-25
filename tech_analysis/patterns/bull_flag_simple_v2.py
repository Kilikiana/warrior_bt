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
        self._pullback_low: Optional[float] = None
        self._pullback_volumes: list[float] = []
        # Count bars starting from the FIRST RED bar AFTER the ACTION alert
        self._bars_since_pullback_start: int = 0
        self._max_wait_candles: int = 6  # stop waiting after N bars from first red after alert
        self._pullback_start_index: Optional[pd.Timestamp] = None
        # Target/stop tracking
        self._target_price: Optional[float] = None
        self._stop_price: Optional[float] = None
        # Fallback/scale state
        self._fallback_target: bool = False
        self._first_target_done: bool = False

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        # Entry: after ACTION alert, wait for a contiguous red pullback, then
        # enter intrabar when the first green breaks the last pullback candle's high.
        try:
            if (not self._entry_done) and (session.position is None) and (session.alert is not None):
                at = session.alert.alert_time
                if df.index[-1] > at:
                    cur_open = float(df['open'].iloc[-1])
                    cur_close = float(df['close'].iloc[-1])
                    cur_high = float(df['high'].iloc[-1])
                    cur_low = float(df['low'].iloc[-1])
                    is_red = cur_close < cur_open
                    is_green = cur_close > cur_open

                    # Build/extend the contiguous red pullback
                    if not self._pullback_started:
                        if is_red:
                            self._pullback_started = True
                            self._last_red_high = cur_high
                            self._pullback_low = float(df['low'].iloc[-1])
                            # seed volumes list
                            try:
                                v = float(df['volume'].iloc[-1])
                            except Exception:
                                v = float('nan')
                            self._pullback_volumes = [v]
                            # Initialize bar count window starting at first red after alert
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
                                    float(self._last_red_high) if self._last_red_high is not None else float('nan')
                                )
                            except Exception:
                                pass
                        # else still waiting for first red
                    else:
                        if is_red:
                            # Extend pullback; update last red high
                            self._last_red_high = cur_high
                            try:
                                self._pullback_low = min(float(self._pullback_low) if self._pullback_low is not None else cur_low,
                                                         float(df['low'].iloc[-1]))
                            except Exception:
                                self._pullback_low = float(df['low'].iloc[-1])
                            try:
                                v = float(df['volume'].iloc[-1])
                            except Exception:
                                v = float('nan')
                            self._pullback_volumes.append(v)
                        elif is_green and self._last_red_high is not None:
                            # Trigger if green breaks last pullback candle's high
                            if cur_high >= float(self._last_red_high):
                                # Entry quality gates (volume + retrace cap)
                                try:
                                    cur_vol = float(df['volume'].iloc[-1])
                                except Exception:
                                    cur_vol = float('nan')
                                # Pullback average volume
                                try:
                                    vals = [float(x) for x in (self._pullback_volumes or []) if x == x]
                                    pb_avg = (sum(vals) / len(vals)) if vals else float('nan')
                                except Exception:
                                    pb_avg = float('nan')
                                # Volume multiple gate
                                try:
                                    mult = float(getattr(session, 'breakout_vol_mult', 0.0) or 0.0)
                                except Exception:
                                    mult = 0.0
                                if mult > 0 and (pb_avg == pb_avg) and (cur_vol == cur_vol):
                                    if cur_vol < pb_avg * mult:
                                        # Reject entry due to insufficient breakout volume
                                        return False
                                # Minimum pullback average volume gate
                                try:
                                    min_pb = float(getattr(session, 'min_pullback_avg_volume', 0.0) or 0.0)
                                except Exception:
                                    min_pb = 0.0
                                if min_pb > 0 and (pb_avg == pb_avg):
                                    if pb_avg < min_pb:
                                        return False
                                # Retrace cap (<= 50% of pole height), using alert high/price as proxy for pole
                                try:
                                    ah = float(getattr(session.alert, 'alert_high', None) or float('nan'))
                                    ap = float(getattr(session.alert, 'alert_price', None) or float('nan'))
                                except Exception:
                                    ah = float('nan'); ap = float('nan')
                                pole_h = (ah - ap) if (ah == ah and ap == ap) else float('nan')
                                if (self._pullback_low is not None) and (self._last_red_high is not None) and (pole_h == pole_h) and pole_h > 0:
                                    pullback_depth = float(self._last_red_high) - float(self._pullback_low)
                                    retrace_pct = pullback_depth / pole_h
                                    if retrace_pct > 0.5:
                                        return False
                                # Cap entry to current bar's high for realistic fill
                                raw_entry = float(self._last_red_high) + float(session.entry_slippage_cents or 0.0)
                                entry_price = min(cur_high, raw_entry)
                                # New stop/target rules
                                # Use true pullback pivot low if available; fallback to prior bar low
                                pivot_low = None
                                try:
                                    if self._pullback_low is not None:
                                        pivot_low = float(self._pullback_low)
                                    elif len(df) >= 2:
                                        pivot_low = float(df['low'].iloc[-2])
                                except Exception:
                                    pivot_low = None
                                self._stop_price = max(0.01, float(pivot_low)) if pivot_low is not None else None
                                try:
                                    target = float(getattr(session.alert, 'alert_high', None) or float('nan'))
                                except Exception:
                                    target = float('nan')
                                # Target selection: prefer alert_high if above entry; else fallback to 2R
                                if (target == target) and (target > entry_price + 1e-6):
                                    self._target_price = target
                                    self._fallback_target = False
                                else:
                                    try:
                                        rps = float(entry_price) - float(self._stop_price) if self._stop_price is not None else None
                                        if rps is not None and rps > 0:
                                            self._target_price = float(entry_price) + 2.0 * float(rps)
                                            self._fallback_target = True
                                        else:
                                            self._target_price = None
                                            self._fallback_target = False
                                    except Exception:
                                        self._target_price = None
                                        self._fallback_target = False
                                stop_loss = float(self._stop_price) if self._stop_price is not None else max(0.01, entry_price * 0.98)
                                try:
                                    logging.info(
                                        "BFSv2 entry: %s | price=%.4f stop=%.4f trigger_bar=%s last_pullback_high=%.4f",
                                        getattr(session, 'symbol', '?'),
                                        float(entry_price),
                                        float(stop_loss),
                                        str(df.index[-1]),
                                        float(self._last_red_high)
                                    )
                                except Exception:
                                    pass
                                session._enter_direct(entry_price, stop_loss, timestamp, reason="BullFlag_Simple_V2_Entry")
                                self._entry_done = True
                                self._exit_pending = True
                                self._entry_index = df.index[-1]
                                # Immediate exit if entry bar closes red
                                try:
                                    if cur_close < cur_open and session.position is not None:
                                        exit_price = float(cur_close)
                                        logging.info(
                                            "BFSv2 exit: %s | bar=%s price=%.4f (entry bar closed red)",
                                            getattr(session, 'symbol', '?'),
                                            str(df.index[-1]),
                                            float(exit_price)
                                        )
                                        # Use a more descriptive reason for analytics
                                        session._exit_position(timestamp, exit_price, session.ExitReason.NO_IMMEDIATE_BREAKOUT, session.position.current_shares)
                                        self._exit_pending = False
                                except Exception:
                                    pass
                                return True
                        # doji/green without break: neither extend nor trigger

                        # Increment bar counter within the pullback window (no entry on this bar)
                        # Robustly update bars-since using index math to avoid drift
                        try:
                            if self._pullback_start_index is not None:
                                # Count bars strictly after first red up to current
                                idx = df.index
                                self._bars_since_pullback_start = int(((idx > self._pullback_start_index) & (idx <= idx[-1])).sum())
                            else:
                                self._bars_since_pullback_start = max(1, int(self._bars_since_pullback_start or 0) + 1)
                        except Exception:
                            self._bars_since_pullback_start = max(1, int(self._bars_since_pullback_start or 0) + 1)

                    # Stop waiting after N bars from first red after alert without entry
                    if self._pullback_started and (not self._entry_done):
                        # Recompute defensively in case counter missed a bar
                        try:
                            if self._pullback_start_index is not None:
                                idx = df.index
                                self._bars_since_pullback_start = int(((idx > self._pullback_start_index) & (idx <= idx[-1])).sum())
                        except Exception:
                            pass
                        if int(self._bars_since_pullback_start) >= int(self._max_wait_candles):
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
                    # 1) Profit target hit
                    if self._target_price is not None and cur_high >= float(self._target_price):
                        price = float(self._target_price)
                        # If target comes from fallback 2R and we haven't scaled yet, sell 1/2 and move stop to BE
                        if self._fallback_target and (not self._first_target_done) and session.position is not None:
                            try:
                                shares_to_sell = max(1, int(session.position.current_shares * 0.5))
                                session._partial_exit(timestamp, price, session.ExitReason.FIRST_TARGET, shares_to_sell)
                                # Move stop to breakeven on remaining
                                try:
                                    if session.position and session.position.current_shares > 0:
                                        new_stop = max(float(session.position.stop_loss), float(session.position.entry_price))
                                        remaining = int(session.position.current_shares) - int(shares_to_sell)
                                        if remaining < 0:
                                            remaining = 0
                                        session.position = session.position._replace(
                                            current_shares=remaining,
                                            stop_loss=new_stop,
                                            status=session.TradeStatus.SCALED_FIRST if remaining > 0 else session.TradeStatus.EXITED
                                        )
                                        if remaining == 0:
                                            session.status = session.MonitoringStatus.MONITORING_STOPPED
                                except Exception:
                                    pass
                                self._first_target_done = True
                                # Clear target to avoid repeated scales in v2 simple mode
                                self._target_price = None
                                return True
                            except Exception:
                                # Fallback to full exit if partial fails
                                pass
                        # Non-fallback target (alert_high) or already scaled: exit all
                        try:
                            logging.info(
                                "BFSv2 exit: %s | bar=%s price=%.4f (target hit)",
                                getattr(session, 'symbol', '?'), str(df.index[-1]), price)
                        except Exception:
                            pass
                        session._exit_position(timestamp, price, session.ExitReason.FIRST_TARGET, session.position.current_shares)
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
                            session._exit_position(timestamp, price, session.ExitReason.STOP_LOSS, session.position.current_shares)
                            self._exit_pending = False
                            return True
        except Exception:
            pass

        return False
