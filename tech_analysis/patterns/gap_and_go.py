from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Optional

import pandas as pd
import logging


class GapAndGoStrategy:
    """
    Ross Cameron's Gap and Go entry logic (entry-only strategy).

    Core idea:
    - Focus on big gappers at the open (default gap ≥ 4%).
    - Two common entries during 9:30–10:00 ET window:
        1) Break of pre‑market high
        2) 1‑min Opening Range Breakout (ORB)
    - Stop: low of the breakout candle (or ORB low).

    This strategy only handles entries. Post‑entry management is delegated to
    PatternMonitoringSession's default management (targets, scales, runner).

    Assumptions/inputs:
    - Previous day close is best provided via `session.prev_close`.
      If unavailable, we attempt to infer from the last 16:00 bar of the prior
      trading day within the provided DataFrame. If still unavailable, the
      gap filter is skipped (conservative fallback).
    - Optionally require a news catalyst via `session.alert.news_catalyst` or
      `session.news_catalyst` if available (configurable).
    - If an external premarket high is computed from tick data, it can be
      exposed as `session.external_premarket_high` and will take precedence.
    """

    def __init__(self) -> None:
        self._entry_done = False
        self._first_open_bar_index: Optional[pd.Timestamp] = None
        self._orb_high: Optional[float] = None
        self._orb_low: Optional[float] = None
        self._first_open_bar_open: Optional[float] = None

    @staticmethod
    def _day_open(dt: datetime) -> datetime:
        return datetime(dt.year, dt.month, dt.day, 9, 30)

    @staticmethod
    def _day_window_end(dt: datetime, minutes: int) -> datetime:
        return datetime(dt.year, dt.month, dt.day, 9, 30) + timedelta(minutes=minutes)

    def _infer_prev_close(self, session, df: pd.DataFrame, open_day: datetime) -> Optional[float]:
        # Preferred: explicit prev_close provided by caller/session
        try:
            pc = getattr(session, 'prev_close', None)
            if pc is not None and float(pc) > 0:
                return float(pc)
        except Exception:
            pass
        # Next: daily file lookup from shared cache
        try:
            symbol = str(getattr(session, 'symbol', '')).upper()
            if symbol:
                import json, os
                from core.config import OHLCV_DAILY_DIR
                daily_path = OHLCV_DAILY_DIR / f"daily_{symbol}.json"
                if daily_path.exists():
                    with open(daily_path, 'r') as f:
                        rec = json.load(f)
                    bars = rec.get('bars', [])
                    # find last date < open_day.date()
                    prev = None
                    for b in bars:
                        try:
                            d = b.get('date')
                            # date strings like YYYY-MM-DD
                            if d and d < open_day.date().isoformat():
                                prev = b
                        except Exception:
                            continue
                    if prev and prev.get('close') is not None:
                        return float(prev['close'])
        except Exception:
            pass
        # Fallback: last bar from previous calendar day around 16:00
        try:
            idx: pd.DatetimeIndex = df.index
            prev_day_mask = idx < pd.Timestamp(open_day)
            if prev_day_mask.any():
                prev_df = df.loc[prev_day_mask]
                # Try to pick the last bar between 15:59 and 16:00 of the prior day
                prev_day = prev_df.index.max().date()
                cands = prev_df[(prev_df.index.date == prev_day) &
                                (prev_df.index.time >= time(15, 59)) &
                                (prev_df.index.time <= time(16, 0))]
                if len(cands) > 0:
                    return float(cands['close'].iloc[-1])
                # Else use the very last available bar of prior day
                return float(prev_df['close'].iloc[-1])
        except Exception:
            pass
        return None

    def _compute_premarket_high(self, session, df: pd.DataFrame, open_day: datetime) -> Optional[float]:
        # If caller provides a premarket high (e.g., from tick aggregation), use it
        try:
            ext = getattr(session, 'external_premarket_high', None)
            if ext is not None and float(ext) > 0:
                return float(ext)
        except Exception:
            pass
        try:
            idx: pd.DatetimeIndex = df.index
            same_day_mask = (idx.date == open_day.date())
            pre_mask = same_day_mask & (idx < pd.Timestamp(open_day))
            if pre_mask.any():
                return float(df.loc[pre_mask]['high'].max())
        except Exception:
            pass
        return None

    def on_bar(self, session, df: pd.DataFrame, timestamp: datetime) -> bool:
        if self._entry_done:
            return False

        # Enforce trading window: entries between 9:30 and 9:30+window
        try:
            window_min = int(getattr(session, 'gng_entry_window_minutes', 30) or 30)
        except Exception:
            window_min = 30
        open_dt = self._day_open(timestamp)
        window_end = self._day_window_end(timestamp, window_min)
        if timestamp < open_dt or timestamp > window_end:
            return False

        # Compute ORB reference (first 1-min bar at/after 9:30)
        try:
            if self._first_open_bar_index is None:
                # Identify the first bar that is at or after 9:30 for this date
                day_mask = (df.index.date == open_dt.date())
                post_open = df.loc[day_mask & (df.index >= pd.Timestamp(open_dt))]
                if len(post_open) > 0:
                    self._first_open_bar_index = post_open.index[0]
                    self._orb_high = float(post_open['high'].iloc[0])
                    self._orb_low = float(post_open['low'].iloc[0])
                    self._first_open_bar_open = float(post_open['open'].iloc[0])
        except Exception:
            pass

        # Gap filter (>= min %). Use session.prev_close if available; otherwise infer.
        try:
            min_gap = float(getattr(session, 'gng_min_gap_pct', 4.0) or 4.0)
        except Exception:
            min_gap = 4.0
        try:
            # Opening reference price = open of first 9:30 bar
            if self._first_open_bar_index is None:
                return False  # wait until ORB known
            open_price = float(df.loc[self._first_open_bar_index, 'open'])
            prev_close = self._infer_prev_close(session, df, open_dt)
            gap_pct_ok = True
            if prev_close is not None and prev_close > 0:
                gap_pct = (open_price - float(prev_close)) / float(prev_close) * 100.0
                gap_pct_ok = gap_pct >= min_gap
                if not gap_pct_ok:
                    return False
        except Exception:
            # If we cannot compute a gap cleanly, do not block on it (conservative)
            pass

        # Optional catalyst/news requirement
        try:
            if bool(getattr(session, 'gng_require_news', False)):
                news = getattr(session.alert, 'news_catalyst', None) if getattr(session, 'alert', None) else getattr(session, 'news_catalyst', None)
                if not news:
                    return False
        except Exception:
            pass

        # Low-float guard (optional)
        try:
            max_float = float(getattr(session, 'gng_max_float_millions', 0.0) or 0.0)
        except Exception:
            max_float = 0.0
        if max_float > 0:
            try:
                fm = getattr(session, 'float_millions', None)
                if fm is not None and float(fm) > max_float:
                    return False
            except Exception:
                pass

        # Determine entry mode(s)
        try:
            mode = getattr(session, 'gng_entry_mode', 'both')
        except Exception:
            mode = 'both'

        cur_high = float(df['high'].iloc[-1])
        cur_low = float(df['low'].iloc[-1])

        # Helper: compute pre-market flag high (last local max below PMH in last N minutes)
        def _last_premarket_flag_high(pmh: float) -> Optional[float]:
            try:
                idx: pd.DatetimeIndex = df.index
                same_day_mask = (idx.date == open_dt.date())
                pre_mask_all = same_day_mask & (idx < pd.Timestamp(open_dt))
                if not pre_mask_all.any():
                    return None
                # Last window minutes before open
                try:
                    win_min = int(getattr(session, 'gng_flag_window_minutes', 30) or 30)
                except Exception:
                    win_min = 30
                cutoff = pd.Timestamp(open_dt - timedelta(minutes=win_min))
                pre_window = df.loc[pre_mask_all & (idx >= cutoff)]
                if len(pre_window) < 7:
                    return None
                highs = pre_window['high'].values
                # Local max: greater than prev 3 and next 3 bars, and strictly below PMH
                last_local = None
                for i in range(3, len(highs) - 3):
                    h = float(highs[i])
                    if h < pmh and h == max(highs[i-3:i+4]):
                        last_local = h
                # Ensure not too far below PMH (avoid random spikes): within 2% of PMH
                if last_local is not None:
                    if (pmh - last_local) / pmh <= 0.02:
                        return last_local
                return None
            except Exception:
                return None

        # 1) Premarket high break
        if mode in ('both', 'premarket_high', 'premarket_flag'):
            pm_high = self._compute_premarket_high(session, df, open_dt)
            # premarket_flag: break of last premarket flag high beneath PMH
            if mode == 'premarket_flag' and pm_high is not None:
                pfh = _last_premarket_flag_high(float(pm_high))
                if pfh is not None and cur_high >= float(pfh):
                    try:
                        entry_price = min(cur_high, float(pfh) + float(getattr(session, 'entry_slippage_cents', 0.0) or 0.0))
                    except Exception:
                        entry_price = float(pfh)
                    stop_loss = max(0.01, cur_low)
                    logging.info("GapAndGo entry (premarket_flag): price=%.4f stop=%.4f pfh=%.4f pmh=%.4f", entry_price, stop_loss, float(pfh), float(pm_high))
                    session._enter_direct(entry_price, stop_loss, timestamp, reason="GapAndGo PremarketFlag Break")
                    self._entry_done = True
                    return True
            # Standard PMH break
            if mode in ('both', 'premarket_high') and pm_high is not None and cur_high >= float(pm_high):
                try:
                    entry_price = min(cur_high, float(pm_high) + float(getattr(session, 'entry_slippage_cents', 0.0) or 0.0))
                except Exception:
                    entry_price = float(pm_high)
                # Stop: low of the breakout candle
                stop_loss = max(0.01, cur_low)
                logging.info("GapAndGo entry (premarket_high): price=%.4f stop=%.4f pmh=%.4f", entry_price, stop_loss, float(pm_high))
                session._enter_direct(entry_price, stop_loss, timestamp, reason="GapAndGo PremarketHigh Break")
                self._entry_done = True
                return True

        # 2) 1-min ORB break
        if mode in ('both', 'opening_range') and self._orb_high is not None and self._orb_low is not None:
            if cur_high >= float(self._orb_high) and df.index[-1] > self._first_open_bar_index:
                try:
                    entry_price = min(cur_high, float(self._orb_high) + float(getattr(session, 'entry_slippage_cents', 0.0) or 0.0))
                except Exception:
                    entry_price = float(self._orb_high)
                stop_loss = max(0.01, float(self._orb_low))
                logging.info("GapAndGo entry (ORB): price=%.4f stop=%.4f orbH=%.4f orbL=%.4f", entry_price, stop_loss, float(self._orb_high), float(self._orb_low))
                session._enter_direct(entry_price, stop_loss, timestamp, reason="GapAndGo ORB Break")
                self._entry_done = True
                return True

        # 3) Red-to-Green: if first 9:30 bar is red, enter when price reclaims its open
        try:
            if self._first_open_bar_index is not None and self._first_open_bar_open is not None:
                # First open bar being red
                first_bar = df.loc[self._first_open_bar_index]
                if float(first_bar['close']) < float(first_bar['open']):
                    rtg_trigger = float(self._first_open_bar_open)
                    # Only consider bars after the first open bar
                    if df.index[-1] > self._first_open_bar_index and cur_high >= rtg_trigger:
                        try:
                            entry_price = min(cur_high, rtg_trigger + float(getattr(session, 'entry_slippage_cents', 0.0) or 0.0))
                        except Exception:
                            entry_price = rtg_trigger
                        stop_loss = max(0.01, cur_low)
                        logging.info("GapAndGo entry (Red-to-Green): price=%.4f stop=%.4f rtg_level=%.4f", entry_price, stop_loss, rtg_trigger)
                        session._enter_direct(entry_price, stop_loss, timestamp, reason="GapAndGo RedToGreen")
                        self._entry_done = True
                        return True
        except Exception:
            pass

        return False
