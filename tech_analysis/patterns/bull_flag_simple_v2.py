from __future__ import annotations

from datetime import datetime
from typing import Optional, Set

import pandas as pd
import logging


class BullFlagSimpleV2Strategy:
    """
    Bull flag simple v2: After an ACTION alert, wait for the first red
    pullback (“breather”). Track the pullback’s last red high (doorway) and the
    true pullback low. Enter when the first green candle breaks above the
    doorway, subject to quality gates (volume multiple, minimum pullback avg
    volume, retrace cap, optional VWAP/spread guards, optional EMA/MACD
    confirmations). Enforce a 6‑bar timeout from the first red after the alert.

    - Entry: doorway high capped to bar high (or bar close when
      `v2_enter_on_close_with_gate` to avoid same‑bar volume look‑ahead).
    - Stop: true pullback low with configurable minimum dollar floor.
    - Target: alert_high if above entry else 2R fallback.
    - Partials: 50% + move stop to BE on 2R fallback; optional same on alert_high.
    - Exits: immediate if entry bar closes red; bailout on no progress within
      grace bars; T+X MACD gate; EMA9 trail and 3R hard cap for runner.
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
        # Runner management state
        self._macd_gate_checked: bool = False
        self._initial_entry_price: Optional[float] = None
        self._initial_stop_price: Optional[float] = None
        self._initial_rps: Optional[float] = None
        self._runner_hard_cap_price: Optional[float] = None
        # Tracking for giveback and weakness
        self._max_price_since_partial: Optional[float] = None

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
                                        try:
                                            logging.debug(
                                                "BFSv2 reject: volume gate | sym=%s bar=%s cur_vol=%.0f pb_avg=%.0f mult=%.2f thresh=%.0f",
                                                getattr(session, 'symbol', '?'), str(df.index[-1]), float(cur_vol), float(pb_avg), float(mult), float(pb_avg * mult))
                                        except Exception:
                                            pass
                                        return False
                                # Minimum pullback average volume gate
                                try:
                                    min_pb = float(getattr(session, 'min_pullback_avg_volume', 0.0) or 0.0)
                                except Exception:
                                    min_pb = 0.0
                                if min_pb > 0 and (pb_avg == pb_avg):
                                    if pb_avg < min_pb:
                                        try:
                                            logging.debug(
                                                "BFSv2 reject: min pullback avg vol | sym=%s bar=%s pb_avg=%.0f < min=%.0f",
                                                getattr(session, 'symbol', '?'), str(df.index[-1]), float(pb_avg), float(min_pb))
                                        except Exception:
                                            pass
                                        return False
                                # Retrace cap (<= configured pct of pole height), using alert high/price as proxy for pole
                                try:
                                    ah = float(getattr(session.alert, 'alert_high', None) or float('nan'))
                                    ap = float(getattr(session.alert, 'alert_price', None) or float('nan'))
                                except Exception:
                                    ah = float('nan'); ap = float('nan')
                                pole_h = (ah - ap) if (ah == ah and ap == ap) else float('nan')
                                if (self._pullback_low is not None) and (self._last_red_high is not None) and (pole_h == pole_h) and pole_h > 0:
                                    pullback_depth = float(self._last_red_high) - float(self._pullback_low)
                                    retrace_pct = pullback_depth / pole_h
                                    try:
                                        cap_pct = float(getattr(session, 'v2_retrace_cap_pct', 0.50) or 0.50)
                                    except Exception:
                                        cap_pct = 0.50
                                    if retrace_pct > cap_pct:
                                        try:
                                            logging.debug(
                                                "BFSv2 reject: retrace cap | sym=%s bar=%s retrace=%.2f cap=%.2f pole=%.3f depth=%.3f",
                                                getattr(session, 'symbol', '?'), str(df.index[-1]), float(retrace_pct), float(cap_pct), float(pole_h), float(pullback_depth))
                                        except Exception:
                                            pass
                                        return False
                                # Optional VWAP guard: require price >= VWAP at entry
                                try:
                                    if getattr(session, 'v2_require_vwap_above', False):
                                        vwap = df.get('vwap', None)
                                        if vwap is not None:
                                            v = float(vwap.iloc[-1])
                                            if not (v == v) or float(cur_close) < v:
                                                try:
                                                    logging.debug(
                                                        "BFSv2 reject: vwap guard | sym=%s bar=%s close=%.4f vwap=%.4f",
                                                        getattr(session, 'symbol', '?'), str(df.index[-1]), float(cur_close), float(v))
                                                except Exception:
                                                    pass
                                                return False
                                except Exception:
                                    pass
                                # Spread gate (bps)
                                try:
                                    if getattr(session, 'spread_cap_bps', 0.0) and float(df['close'].iloc[-1]) > 0:
                                        spread_bps = (float(df['high'].iloc[-1]) - float(df['low'].iloc[-1])) / float(df['close'].iloc[-1]) * 10000.0
                                        if spread_bps > float(getattr(session, 'spread_cap_bps', 0.0)):
                                            try:
                                                logging.debug(
                                                    "BFSv2 reject: spread cap | sym=%s bar=%s spread_bps=%.1f cap=%.1f",
                                                    getattr(session, 'symbol', '?'), str(df.index[-1]), float(spread_bps), float(getattr(session, 'spread_cap_bps', 0.0)))
                                            except Exception:
                                                pass
                                            return False
                                except Exception:
                                    pass
                                # Cap entry to realistic price
                                if getattr(session, 'v2_enter_on_close_with_gate', False) and mult > 0:
                                    # To avoid same-bar volume look-ahead when gating by volume,
                                    # enter on bar close (no intrabar fill advantage)
                                    raw_entry = float(df['close'].iloc[-1]) + float(session.entry_slippage_cents or 0.0)
                                    entry_price = min(cur_high, raw_entry)
                                else:
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
                                # Enforce minimum stop-distance floor for sizing realism
                                if pivot_low is not None:
                                    min_floor = float(getattr(session, 'v2_min_stop_dollars', 0.0) or 0.0)
                                    est_stop = float(pivot_low)
                                    try:
                                        if min_floor > 0.0:
                                            est_stop = min(est_stop, float(entry_price) - float(min_floor))
                                    except Exception:
                                        pass
                                    self._stop_price = max(0.01, est_stop)
                                else:
                                    self._stop_price = None
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
                                # Require 2R potential if enabled
                                try:
                                    if getattr(session, 'v2_require_2r_potential', False) and self._stop_price is not None:
                                        rps = float(entry_price) - float(self._stop_price)
                                        if rps <= 0:
                                            return False
                                        tgt = float(self._target_price) if self._target_price is not None else float('nan')
                                        if tgt == tgt:  # valid
                                            if (tgt - float(entry_price)) < 2.0 * rps - 1e-9:
                                                return False
                                except Exception:
                                    pass
                                # Entry confirmations (optional): EMA9>EMA20 and/or MACD bullish; optional 5-min EMA trend
                                try:
                                    # Determine which bar's indicators to use
                                    idx = -1
                                    try:
                                        if getattr(session, 'entry_confirm_mode', 'current') == 'prior' and len(df) >= 2:
                                            idx = -2
                                    except Exception:
                                        idx = -1
                                    ema_ok = True
                                    macd_ok = True
                                    # EMA rule
                                    if getattr(session, 'v2_entry_confirm_ema', False) or getattr(session, 'v2_entry_confirmations', 'none') in ('both','ema_only'):
                                        ema9 = df.get('ema9', pd.Series(dtype=float)).iloc[idx]
                                        ema20 = df.get('ema20', pd.Series(dtype=float)).iloc[idx]
                                        ema_ok = (pd.notna(ema9) and pd.notna(ema20) and float(ema9) > float(ema20))
                                    # MACD rule
                                    if getattr(session, 'v2_entry_confirm_macd', False) or getattr(session, 'v2_entry_confirmations', 'none') in ('both','macd_only'):
                                        macd_val = df.get('macd', pd.Series(dtype=float)).iloc[idx]
                                        macd_sig = df.get('macd_signal', pd.Series(dtype=float)).iloc[idx]
                                        macd_hist = df.get('macd_hist', pd.Series(dtype=float)).iloc[idx]
                                        macd_ok = (pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(macd_hist)
                                                   and float(macd_val) > float(macd_sig) and float(macd_hist) > 0.0)
                                    # Optional 5-min EMA trend confirm
                                    if getattr(session, 'v2_entry_confirm_ema5m', False):
                                        try:
                                            df5 = df.copy(); df5.index = pd.to_datetime(df5.index)
                                            df5 = df5.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
                                            ema9_5 = df5['close'].ewm(span=9, adjust=False).mean().iloc[-1]
                                            ema20_5 = df5['close'].ewm(span=20, adjust=False).mean().iloc[-1]
                                            if pd.notna(ema9_5) and pd.notna(ema20_5):
                                                ema_ok = ema_ok and (float(ema9_5) > float(ema20_5))
                                        except Exception:
                                            pass
                                    # Combine per selector
                                    selector = getattr(session, 'v2_entry_confirmations', 'none')
                                    if selector == 'both':
                                        confirm_ok = ema_ok and macd_ok
                                    elif selector == 'ema_only':
                                        confirm_ok = ema_ok
                                    elif selector == 'macd_only':
                                        confirm_ok = macd_ok
                                    else:
                                        # If selector is 'none', fall back to boolean toggles if any requested
                                        if getattr(session, 'v2_entry_confirm_ema', False) or getattr(session, 'v2_entry_confirm_macd', False):
                                            confirm_ok = (not getattr(session, 'v2_entry_confirm_ema', False) or ema_ok) and \
                                                         (not getattr(session, 'v2_entry_confirm_macd', False) or macd_ok)
                                        else:
                                            confirm_ok = True
                                    if not confirm_ok:
                                        try:
                                            logging.debug(
                                                "BFSv2 reject: confirmations | sym=%s bar=%s selector=%s ema_ok=%s macd_ok=%s",
                                                getattr(session, 'symbol', '?'), str(df.index[-1]), str(selector), str(ema_ok), str(macd_ok))
                                        except Exception:
                                            pass
                                        return False
                                except Exception:
                                    pass
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
                                # Cache initial risk/targets for runner management
                                try:
                                    self._initial_entry_price = float(entry_price)
                                    self._initial_stop_price = float(stop_loss)
                                    self._initial_rps = max(0.0, float(entry_price) - float(stop_loss))
                                    self._max_price_since_partial = float(entry_price)
                                except Exception:
                                    self._initial_entry_price = float(entry_price)
                                    self._initial_stop_price = float(stop_loss)
                                    self._initial_rps = None
                                    self._max_price_since_partial = float(entry_price)
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

                    # Allow override of wait bars from session config
                    try:
                        self._max_wait_candles = int(getattr(session, 'v2_max_wait_bars', self._max_wait_candles) or self._max_wait_candles)
                    except Exception:
                        pass
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
        except Exception as e:
            try:
                logging.warning("BFSv2 on_bar error for %s @ %s: %s", getattr(session, 'symbol', '?'), str(timestamp), str(e))
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
                    cur_close = float(cur['close'])
                    # Breakout-or-Bailout: within N bars after entry, if price has not traded above entry, exit at breakeven
                    try:
                        grace = int(getattr(session, 'bailout_grace_bars', 2) or 0)
                    except Exception:
                        grace = 0
                    if grace > 0 and self._initial_entry_price is not None:
                        try:
                            bars_since_entry = int((df.index > self._entry_index).sum())
                        except Exception:
                            bars_since_entry = 0
                        if bars_since_entry <= grace and float(cur_high) <= float(self._initial_entry_price) + 1e-9:
                            try:
                                logging.info("BFSv2 bailout: %s | bar=%s close=%.4f (no progress within %d bars)",
                                             getattr(session,'symbol','?'), str(df.index[-1]), float(cur_close), int(grace))
                            except Exception:
                                pass
                            session._exit_position(timestamp, float(cur_close), session.ExitReason.BREAKOUT_OR_BAILOUT, session.position.current_shares)
                            self._exit_pending = False
                            return True
                    # Runner hard-cap target after first scale
                    if self._first_target_done and getattr(session, 'v2_runner_enabled', True):
                        try:
                            if self._runner_hard_cap_price is None and self._initial_entry_price is not None and self._initial_rps is not None:
                                cap_r = float(getattr(session, 'v2_runner_hard_cap_R', 3.0) or 3.0)
                                if cap_r > 0.0 and self._initial_rps > 0.0:
                                    self._runner_hard_cap_price = float(self._initial_entry_price) + cap_r * float(self._initial_rps)
                        except Exception:
                            pass
                        if self._runner_hard_cap_price is not None and cur_high >= float(self._runner_hard_cap_price):
                            price = float(self._runner_hard_cap_price)
                            try:
                                logging.info("BFSv2 runner hard cap exit: %s | bar=%s price=%.4f (%.1fR)",
                                             getattr(session, 'symbol', '?'), str(df.index[-1]), price,
                                             float(getattr(session, 'v2_runner_hard_cap_R', 3.0) or 3.0))
                            except Exception:
                                pass
                            session._exit_position(timestamp, price, session.ExitReason.RUNNER_HARDCAP, session.position.current_shares)
                            self._exit_pending = False
                            return True
                    # 1) Profit target hit
                    if self._target_price is not None and cur_high >= float(self._target_price):
                        price = float(self._target_price)
                        # If target comes from fallback 2R and we haven't scaled yet, sell 1/2 and move stop to BE
                        if self._fallback_target and (not self._first_target_done) and session.position is not None:
                            try:
                                shares_to_sell = max(1, int(session.position.current_shares * 0.5))
                                session._partial_exit(timestamp, price, session.ExitReason.FIRST_TARGET, shares_to_sell)
                                self._first_target_done = True
                                # Clear target to avoid repeated scales in v2 simple mode
                                self._target_price = None
                                # Initialize runner hard cap at first scale
                                try:
                                    if getattr(session, 'v2_runner_enabled', True) and self._initial_entry_price is not None and self._initial_rps is not None:
                                        cap_r = float(getattr(session, 'v2_runner_hard_cap_R', 3.0) or 3.0)
                                        if cap_r > 0.0 and self._initial_rps > 0.0:
                                            self._runner_hard_cap_price = float(self._initial_entry_price) + cap_r * float(self._initial_rps)
                                    self._max_price_since_partial = float(price)
                                except Exception:
                                    pass
                                return True
                            except Exception:
                                # Fallback to full exit if partial fails
                                pass
                        # Non-fallback target (alert_high) or already scaled
                        # Optionally do partial on alert_high targets too (config), then move stop to BE
                        if (not self._fallback_target) and (not self._first_target_done) and getattr(session, 'v2_partial_on_alert_high', False) and session.position is not None:
                            try:
                                shares_to_sell = max(1, int(session.position.current_shares * 0.5))
                                session._partial_exit(timestamp, price, session.ExitReason.FIRST_TARGET, shares_to_sell)
                                self._first_target_done = True
                                # Clear target to avoid repeated scales in v2 simple mode
                                self._target_price = None
                                # Initialize runner hard cap at first scale
                                try:
                                    if getattr(session, 'v2_runner_enabled', True) and self._initial_entry_price is not None and self._initial_rps is not None:
                                        cap_r = float(getattr(session, 'v2_runner_hard_cap_R', 3.0) or 3.0)
                                        if cap_r > 0.0 and self._initial_rps > 0.0:
                                            self._runner_hard_cap_price = float(self._initial_entry_price) + cap_r * float(self._initial_rps)
                                    self._max_price_since_partial = float(price)
                                except Exception:
                                    pass
                                return True
                            except Exception:
                                pass
                        # Otherwise exit all
                        try:
                            logging.info(
                                "BFSv2 exit: %s | bar=%s price=%.4f (target hit)",
                                getattr(session, 'symbol', '?'), str(df.index[-1]), price)
                        except Exception:
                            pass
                        session._exit_position(timestamp, price, session.ExitReason.FIRST_TARGET, session.position.current_shares)
                        self._exit_pending = False
                        return True
                    # MACD gate at T+X minutes (runner by default; optional full-position gate)
                    if getattr(session, 'v2_runner_enabled', True) and (not self._macd_gate_checked):
                        try:
                            from datetime import timedelta
                            gate_mins = int(getattr(session, 'v2_runner_macd_gate_minutes', 10) or 10)
                            if gate_mins > 0 and self._entry_index is not None:
                                if df.index[-1] >= (self._entry_index + pd.Timedelta(minutes=gate_mins)):
                                    require_runner = bool(getattr(session, 'v2_macd_gate_require_runner', True))
                                    if (not require_runner) or (require_runner and self._first_target_done):
                                        macd_val = df.get('macd', pd.Series(dtype=float)).iloc[-1]
                                        macd_sig = df.get('macd_signal', pd.Series(dtype=float)).iloc[-1]
                                        macd_hist = df.get('macd_hist', pd.Series(dtype=float)).iloc[-1]
                                        bearish = (pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(macd_hist) and
                                                   (float(macd_val) < float(macd_sig) or float(macd_hist) <= 0.0))
                                        # Optional no-progress guard in R
                                        if bearish and bool(getattr(session, 'v2_macd_gate_require_no_progress', False)):
                                            try:
                                                if (self._initial_entry_price is not None) and (self._initial_rps is not None) and self._initial_rps > 0:
                                                    thresh_r = float(getattr(session, 'v2_no_progress_thresh_r', 0.0) or 0.0)
                                                    hi_since = float(df.loc[df.index >= self._entry_index, 'high'].max())
                                                    if hi_since > (float(self._initial_entry_price) + thresh_r * float(self._initial_rps) + 1e-9):
                                                        bearish = False
                                            except Exception:
                                                pass
                                        self._macd_gate_checked = True
                                        if bearish:
                                            try:
                                                logging.info("BFSv2 runner MACD gate exit: %s | bar=%s close=%.4f", getattr(session,'symbol','?'), str(df.index[-1]), float(cur_close))
                                            except Exception:
                                                pass
                                            session._exit_position(timestamp, float(cur_close), session.ExitReason.RUNNER_MACD_GATE, session.position.current_shares)
                                            self._exit_pending = False
                                            return True
                        except Exception:
                            pass
                    # 2) Stop loss hit: prefer current position stop (e.g., breakeven after partial),
                    #    else fallback to initial pivot-low stop
                    pos_stop = 0.0
                    try:
                        if session.position is not None:
                            pos_stop = float(getattr(session.position, 'stop_loss', 0.0) or 0.0)
                    except Exception:
                        pos_stop = 0.0
                    stop = pos_stop if pos_stop > 0.0 else (float(self._stop_price) if self._stop_price is not None else 0.0)
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
                    # 3) Trailing stop update (EMA9 1-min) with BE floor for runner
                    if self._first_target_done and getattr(session, 'v2_runner_enabled', True):
                        try:
                            ema9 = df.get('ema9', pd.Series(dtype=float)).iloc[-1]
                            if pd.notna(ema9):
                                new_trail = float(ema9)
                                # Enforce BE floor
                                be_floor = float(session.position.entry_price)
                                new_trail = max(new_trail, be_floor)
                                if new_trail > float(session.position.stop_loss) + 1e-6:
                                    session.position = session.position._replace(stop_loss=new_trail)
                            # Track max since partial for giveback logic
                            try:
                                self._max_price_since_partial = max(float(self._max_price_since_partial or cur_high), cur_high)
                            except Exception:
                                self._max_price_since_partial = cur_high
                        except Exception:
                            pass
                    # 4) No-progress time gate: exit at T+N if no push above entry and (optionally) MACD bearish
                    try:
                        np_minutes = int(getattr(session, 'v2_no_progress_exit_minutes', 0) or 0)
                    except Exception:
                        np_minutes = 0
                    if np_minutes > 0 and self._entry_index is not None and session.position is not None:
                        try:
                            if df.index[-1] >= (self._entry_index + pd.Timedelta(minutes=np_minutes)):
                                # Check if price ever traded above entry
                                hi_since = float(df.loc[df.index > self._entry_index, 'high'].max())
                                crossed = (hi_since > float(self._initial_entry_price) + 1e-9) if self._initial_entry_price is not None else True
                                macd_ok = True
                                if bool(getattr(session, 'v2_no_progress_exit_macd_only', True)):
                                    macd_val = df.get('macd', pd.Series(dtype=float)).iloc[-1]
                                    macd_sig = df.get('macd_signal', pd.Series(dtype=float)).iloc[-1]
                                    macd_hist = df.get('macd_hist', pd.Series(dtype=float)).iloc[-1]
                                    macd_ok = (pd.notna(macd_val) and pd.notna(macd_sig) and pd.notna(macd_hist) and (float(macd_val) < float(macd_sig) or float(macd_hist) <= 0.0))
                                if (not crossed) and macd_ok:
                                    session._exit_position(timestamp, float(cur_close), session.ExitReason.BREAKOUT_OR_BAILOUT, session.position.current_shares)
                                    self._exit_pending = False
                                    return True
                        except Exception:
                            pass
                    # 5) Weakness exit in first M bars: close below VWAP/EMA9
                    try:
                        wb = int(getattr(session, 'v2_weakness_exit_bars', 0) or 0)
                    except Exception:
                        wb = 0
                    if wb > 0 and self._entry_index is not None and session.position is not None:
                        try:
                            bars = int((df.index > self._entry_index).sum())
                        except Exception:
                            bars = 0
                        if bars <= wb:
                            vwap_ok = True
                            ema_ok = True
                            try:
                                vwap_val = df.get('vwap', None)
                                if vwap_val is not None:
                                    vwap_ok = float(cur_close) >= float(vwap_val.iloc[-1])
                            except Exception:
                                vwap_ok = True
                            try:
                                ema9_val = df.get('ema9', None)
                                if ema9_val is not None:
                                    ema_ok = float(cur_close) >= float(ema9_val.iloc[-1])
                            except Exception:
                                ema_ok = True
                            if not (vwap_ok and ema_ok):
                                session._exit_position(timestamp, float(cur_close), session.ExitReason.NO_IMMEDIATE_BREAKOUT, session.position.current_shares)
                                self._exit_pending = False
                                return True
                    # 6) Giveback exit after partial: if price gives back > frac of max-open R, exit remainder
                    try:
                        gb = float(getattr(session, 'v2_giveback_exit_frac', 0.0) or 0.0)
                    except Exception:
                        gb = 0.0
                    if gb > 0.0 and self._first_target_done and session.position is not None and self._initial_rps is not None and self._initial_rps > 0 and self._max_price_since_partial is not None:
                        drop_r = (float(self._max_price_since_partial) - float(cur_close)) / float(self._initial_rps)
                        if drop_r >= gb - 1e-9:
                            session._exit_position(timestamp, float(cur_close), session.ExitReason.NO_IMMEDIATE_BREAKOUT, session.position.current_shares)
                            self._exit_pending = False
                            return True
        except Exception:
            pass

        return False
