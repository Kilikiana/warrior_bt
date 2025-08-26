from __future__ import annotations

"""
Bull Flag Simple V3 (minimal, Ross-aligned) with optional tick-level entry fills.

Core rules (1-min structure, optional tick entry):
- After ACTION alert, wait for N consecutive red candles (pullback), default
  min=2 and max=3. If more than max, abandon.
- Entry = first candle to make a new high over the last red candle's high.
- Stop = pullback low. Target1 = alert_high (or 2R if alert_high <= entry).
- Scale 1/2 at T1, move stop to BE for remainder; then trail runner at 9EMA 1-min
  with BE floor.
- Optional bail: no-progress exit after N minutes.
- Optional caps: max retrace ≤ 50% of pole (alert_high - alert_price).
- Optional tick entry: refine entry fill to first trade >= trigger in the entry minute.
"""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import logging


class BullFlagSimpleV3Strategy:
    def __init__(self) -> None:
        self._state = "waiting_pullback"
        self._reds = 0
        self._last_red_high: Optional[float] = None
        self._pullback_low: Optional[float] = None
        self._pullback_volumes: list[float] = []
        self._entry_index: Optional[pd.Timestamp] = None
        self._first_target_done = False
        self._max_price_since_partial: Optional[float] = None

    # --- helpers (readability only; no behavior changes) ---
    def _should_skip(self, session) -> bool:
        try:
            if getattr(session, 'status', None) == session.MonitoringStatus.MONITORING_STOPPED:
                return True
        except Exception:
            pass
        try:
            if getattr(session, 'position', None) is not None and int(getattr(session.position, 'current_shares', 0)) <= 0:
                return True
        except Exception:
            pass
        return False

    def _update_pullback_state(self, df: pd.DataFrame, symbol: str = "?") -> None:
        cur_high = float(df['high'].iloc[-1])
        cur_low = float(df['low'].iloc[-1])
        if self._state == "waiting_pullback":
            self._state = "in_pullback"
            self._reds = 1
            self._last_red_high = cur_high
            self._pullback_low = cur_low
            try:
                self._pullback_volumes = [float(df['volume'].iloc[-1])]
            except Exception:
                self._pullback_volumes = []
            try:
                logging.info(
                    "BFSv3 pullback-start: %s | first_red=%s last_red_high=%.4f",
                    symbol, str(df.index[-1]), float(self._last_red_high or cur_high)
                )
            except Exception:
                pass
            return
        # extend
        self._reds += 1
        self._last_red_high = max(float(self._last_red_high or cur_high), cur_high)
        self._pullback_low = min(float(self._pullback_low or cur_low), cur_low)
        try:
            self._pullback_volumes.append(float(df['volume'].iloc[-1]))
        except Exception:
            pass

    def _passes_entry_gates(self, session, df: pd.DataFrame) -> bool:
        # minimum reds
        try:
            min_reds = int(getattr(session, 'v3_min_pullback_candles', 1) or 1)
        except Exception:
            min_reds = 1
        if self._reds < max(1, min_reds):
            return False
        # retrace cap
        try:
            ah = float(getattr(session.alert, 'alert_high', None) or float('nan'))
            ap = float(getattr(session.alert, 'alert_price', None) or float('nan'))
            pole = ah - ap if (ah == ah and ap == ap) else float('nan')
            max_retrace = float(getattr(session, 'v3_max_retrace_pct', 0.5) or 0.5)
        except Exception:
            pole = float('nan'); max_retrace = 0.5
        if self._pullback_low is not None and pole == pole and pole > 0:
            retrace = float(self._last_red_high) - float(self._pullback_low)
            if retrace > max_retrace * pole + 1e-9:
                self._state = "abandoned"
                try:
                    logging.debug(
                        "BFSv3 reject: retrace cap | retrace=%.3f cap=%.3f pole=%.3f", retrace, max_retrace, pole
                    )
                except Exception:
                    pass
                return False
        # VWAP guard
        cur_close = float(df['close'].iloc[-1])
        try:
            if bool(getattr(session, 'v3_require_vwap_above', False)):
                vwap = df.get('vwap', None)
                if vwap is not None and pd.notna(vwap.iloc[-1]):
                    if float(cur_close) < float(vwap.iloc[-1]) - 1e-9:
                        try:
                            logging.debug("BFSv3 reject: vwap guard | close=%.4f vwap=%.4f", cur_close, float(vwap.iloc[-1]))
                        except Exception:
                            pass
                        return False
        except Exception:
            pass
        # MACD positive
        try:
            if bool(getattr(session, 'v3_require_macd_positive', False)):
                macd_val = df.get('macd', pd.Series(dtype=float)).iloc[-1]
                macd_hist = df.get('macd_hist', pd.Series(dtype=float)).iloc[-1]
                if not (pd.notna(macd_val) and pd.notna(macd_hist) and float(macd_val) > 0.0 and float(macd_hist) > 0.0):
                    try:
                        logging.debug("BFSv3 reject: macd guard | macd=%.4f hist=%.4f", float(macd_val), float(macd_hist))
                    except Exception:
                        pass
                    return False
        except Exception:
            pass
        # spread cap (proxy)
        cur_high = float(df['high'].iloc[-1])
        cur_low = float(df['low'].iloc[-1])
        try:
            cap_bps = float(getattr(session, 'spread_cap_bps', 0.0) or 0.0)
        except Exception:
            cap_bps = 0.0
        if cap_bps > 0.0:
            try:
                if float(cur_close) > 0.0:
                    cur_spread_bps = (float(cur_high) - float(cur_low)) / float(cur_close) * 10000.0
                    if cur_spread_bps > cap_bps:
                        try:
                            logging.debug("BFSv3 reject: spread cap | spread=%.1fbps cap=%.1fbps", cur_spread_bps, cap_bps)
                        except Exception:
                            pass
                        return False
            except Exception:
                pass
        # volume multiple
        try:
            mult = float(getattr(session, 'breakout_vol_mult', 1.0) or 0.0)
        except Exception:
            mult = 0.0
        if mult > 0.0:
            try:
                vals = [float(x) for x in (self._pullback_volumes or []) if x == x]
                pb_avg = (sum(vals) / len(vals)) if vals else float('nan')
                cur_vol = float(df['volume'].iloc[-1])
                if pb_avg == pb_avg and cur_vol == cur_vol:
                    if cur_vol < pb_avg * mult:
                        try:
                            logging.debug(
                                "BFSv3 reject: volume mult | cur_vol=%.0f pb_avg=%.0f mult=%.2f thresh=%.0f",
                                cur_vol, pb_avg, mult, pb_avg * mult
                            )
                        except Exception:
                            pass
                        return False
            except Exception:
                pass
        # EMA trend
        try:
            if bool(getattr(session, 'v3_require_ema_trend', False)):
                ema9 = df.get('ema9', pd.Series(dtype=float)).iloc[-1]
                ema20 = df.get('ema20', pd.Series(dtype=float)).iloc[-1]
                if not (pd.notna(ema9) and pd.notna(ema20) and float(ema9) > float(ema20)):
                    return False
        except Exception:
            pass
        return True

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        try:
            # If session is done or already flat, skip
            if self._should_skip(session):
                return False
            # Basic guards
            if session.alert is None:
                return False
            if len(df) < 3:
                return False
            cur_open = float(df['open'].iloc[-1])
            cur_close = float(df['close'].iloc[-1])
            cur_high = float(df['high'].iloc[-1])
            cur_low = float(df['low'].iloc[-1])
            is_red = cur_close < cur_open
            is_green = cur_close > cur_open

            # Before entry: build the pullback per Ross (2-3 reds)
            if session.position is None and self._entry_index is None:
                # Start tracking after alert time
                if df.index[-1] <= session.alert.alert_time:
                    return False
                if self._state == "waiting_pullback":
                    if is_red:
                        self._update_pullback_state(df, getattr(session, 'symbol', '?'))
                        return False
                    else:
                        return False
                if self._state == "in_pullback":
                    if is_red:
                        self._update_pullback_state(df, getattr(session, 'symbol', '?'))
                        # Too many red bars? give up
                        max_reds = int(getattr(session, 'v3_max_pullback_candles', 5) or 5)
                        if self._reds > max_reds:
                            self._state = "abandoned"
                        return False
                    if is_green and self._last_red_high is not None:
                        if not self._passes_entry_gates(session, df):
                            return False
                        # Trigger: CLOSE over last red high (entry confirmation)
                        if cur_close > float(self._last_red_high):
                            # The primary gates (VWAP, MACD, EMA, etc.) are now checked in _passes_entry_gates.
                            # We can proceed directly to defining the trade.
                            entry_price = float(self._last_red_high)
                            # Stop at pullback low (with optional min-dollar floor)
                            stop = float(self._pullback_low or cur_low) * 0.995
                            try:
                                min_floor = float(getattr(session, 'v2_min_stop_dollars', 0.0) or 0.0)
                                if min_floor > 0.0:
                                    stop = min(stop, entry_price - min_floor)
                            except Exception:
                                pass
                            stop = max(0.01, stop)
                            # Optional tick-level entry refinement (first trade >= trigger within this minute)
                            refined = None
                            try:
                                if bool(getattr(session, 'v3_use_ticks', False)):
                                    from data.alpaca_trades import first_trade_at_or_above
                                    start = df.index[-1].to_pydatetime().replace(second=0, microsecond=0)
                                    end = start + timedelta(minutes=1)
                                    feed = getattr(session, 'v3_tick_feed', 'sip')
                                    tr = first_trade_at_or_above(session.symbol, start, end, entry_price, feed=feed)  # type: ignore
                                    if tr is not None:
                                        refined = float(tr.p)
                                    else:
                                        try:
                                            logging.info(
                                                "BFSv3 tick refine unavailable: %s | trigger=%.4f",
                                                getattr(session,'symbol','?'), entry_price
                                            )
                                        except Exception:
                                            pass
                            except Exception:
                                refined = None
                            if refined is not None:
                                try:
                                    logging.info(
                                        "BFSv3 tick refine: %s | trigger=%.4f refined=%.4f",
                                        getattr(session, 'symbol', '?'), float(self._last_red_high), float(refined)
                                    )
                                except Exception:
                                    pass
                                entry_price = float(refined)
                            # Target 1: alert_high if above entry, else 2R fallback
                            try:
                                tgt = float(getattr(session.alert, 'alert_high', None) or float('nan'))
                            except Exception:
                                tgt = float('nan')
                            if not (tgt == tgt and tgt > entry_price + 1e-9):
                                rps = entry_price - stop
                                if rps > 0:
                                    tgt = entry_price + 2.0 * rps
                                else:
                                    return False
                            # Optional: require ≥ 2R potential even if using alert_high
                            try:
                                if bool(getattr(session, 'v3_require_2r_potential', False)):
                                    rps2 = entry_price - stop
                                    if rps2 <= 0:
                                        return False
                                    if (tgt - entry_price) + 1e-9 < 2.0 * rps2:
                                        return False
                            except Exception:
                                pass
                            # Enter
                            logging.info(
                                "BFSv3 entry: %s | price=%.4f stop=%.4f trigger_bar=%s last_red_high=%.4f",
                                getattr(session, 'symbol', '?'), entry_price, stop, str(df.index[-1]), float(self._last_red_high))
                            session._enter_direct(entry_price, stop, timestamp, reason="BullFlag_Simple_V3_Entry")
                            self._entry_index = df.index[-1]
                            # Attach first target for management
                            try:
                                session.position = session.position._replace(first_target=float(tgt))
                            except Exception:
                                pass
                            return True

            # Manage position
            if session.position is not None and self._entry_index is not None:
                # 1) Immediate stop
                pos_stop = float(getattr(session.position, 'stop_loss', 0.0) or 0.0)
                if pos_stop > 0.0 and cur_low <= pos_stop + 1e-9:
                    price = max(cur_low, pos_stop)
                    session._exit_position(timestamp, float(price), session.ExitReason.STOP_LOSS, session.position.current_shares)
                    return True
                # 2) First target hit → sell 1/2, stop to BE
                try:
                    t1 = float(getattr(session.position, 'first_target', 0.0) or 0.0)
                except Exception:
                    t1 = 0.0
                if (not self._first_target_done) and t1 > 0.0 and cur_high >= t1 - 1e-9:
                    # sell 100% at target
                    session._exit_position(timestamp, float(t1), session.ExitReason.FIRST_TARGET, session.position.current_shares)
                    return True
                # 3) Trail runner with EMA9 1-min, BE floor
                if self._first_target_done and session.position is not None:
                    try:
                        ema9 = df.get('ema9', pd.Series(dtype=float)).iloc[-1]
                        if pd.notna(ema9):
                            be = float(session.position.entry_price)
                            new_stop = max(float(ema9), be)
                            if new_stop > float(session.position.stop_loss) + 1e-9:
                                session.position = session.position._replace(stop_loss=new_stop)
                    except Exception:
                        pass
                # 4) No-progress timeout N minutes after entry
                try:
                    np_min = int(getattr(session, 'v3_no_progress_minutes', 0) or 0)
                except Exception:
                    np_min = 0
                if np_min > 0 and self._entry_index is not None:
                    if df.index[-1] >= (self._entry_index + pd.Timedelta(minutes=np_min)):
                        hi_since = float(df.loc[df.index > self._entry_index, 'high'].max())
                        crossed = hi_since > float(session.position.entry_price) + 1e-9
                        if not crossed:
                            session._exit_position(timestamp, float(cur_close), session.ExitReason.BREAKOUT_OR_BAILOUT, session.position.current_shares)
                            return True
                # 5) Early weakness exit: first M bars, close < VWAP or EMA9
                try:
                    wb = int(getattr(session, 'v3_weakness_exit_bars', 0) or 0)
                except Exception:
                    wb = 0
                if wb > 0 and self._entry_index is not None:
                    try:
                        bars = int((df.index > self._entry_index).sum())
                    except Exception:
                        bars = 0
                    if bars <= wb:
                        vwap_ok = True
                        ema_ok = True
                        try:
                            vwap_val = df.get('vwap', None)
                            if vwap_val is not None and pd.notna(vwap_val.iloc[-1]):
                                vwap_ok = float(cur_close) >= float(vwap_val.iloc[-1])
                        except Exception:
                            vwap_ok = True
                        try:
                            ema9_val = df.get('ema9', None)
                            if ema9_val is not None and pd.notna(ema9_val.iloc[-1]):
                                ema_ok = float(cur_close) >= float(ema9_val.iloc[-1])
                        except Exception:
                            ema_ok = True
                        if not (vwap_ok and ema_ok):
                            session._exit_position(timestamp, float(cur_close), session.ExitReason.NO_IMMEDIATE_BREAKOUT, session.position.current_shares)
                            return True
        except Exception as e:
            logging.warning("BFSv3 on_bar error for %s @ %s: %s", getattr(session, 'symbol', '?'), str(timestamp), str(e))
        return False