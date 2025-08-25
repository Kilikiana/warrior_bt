"""
Simplified Ross-style Bull Flag pattern detector.

Keeps the same API (classes, return types, and detect_bull_flag signature)
so the rest of the codebase continues to work unchanged.

Core logic (deterministic, 1‑min friendly):
- Flagpole: recent impulse into a local high
- Pullback: contiguous 2–6 red candles after the pole high
- Breakout: first green bar whose HIGH breaks the last red HIGH

Validation:
- Retrace ≤ 50% of the pole move
- Prefer EMA9 support during pullback (soft gate)
- Prefer lighter pullback volume than the pole (soft gate)

If known_flagpole_high/time are provided (ACTION alerts), we anchor around that bar.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, NamedTuple
from enum import Enum
from datetime import datetime


class BullFlagValidation(Enum):
    """Bull flag validation results"""
    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID_RETRACE_TOO_DEEP = "retrace_too_deep"
    INVALID_BROKE_VWAP = "broke_vwap"
    INVALID_BROKE_9EMA = "broke_9ema"
    INVALID_TOO_MANY_RED_CANDLES = "too_many_red_candles"
    INVALID_HIGH_VOLUME_CONSOLIDATION = "high_volume_consolidation"
    INVALID_LOW_LIQUIDITY = "low_liquidity"
    INVALID_NO_FLAGPOLE = "no_flagpole"
    PENDING = "pending"


class BullFlagStage(Enum):
    """Current stage of bull flag formation"""
    NO_PATTERN = "no_pattern"
    FLAGPOLE_FORMING = "flagpole_forming"
    PULLBACK_STAGE = "pullback_stage"
    READY_FOR_BREAKOUT = "ready_for_breakout"
    BREAKOUT_CONFIRMED = "breakout_confirmed"
    PATTERN_FAILED = "pattern_failed"


class BullFlagSignal(NamedTuple):
    """Bull flag signal details"""
    timestamp: datetime
    symbol: str
    stage: BullFlagStage
    validation: BullFlagValidation
    entry_price: Optional[float]
    stop_loss: Optional[float]
    flagpole_start: float
    flagpole_high: float
    pullback_low: float
    pullback_candles: int
    retrace_percentage: float
    volume_confirmation: bool
    broke_vwap: bool
    broke_9ema: bool
    strength_score: float


class BullFlagDetector:
    """Simplified bull flag detector (Ross-style entry timing)"""

    def __init__(self, strict: bool = True, breakout_volume_multiple: float = 0.0,
                 min_pullback_avg_volume: float = 0.0, require_green_breakout: bool = False):
        self.strict = bool(strict)
        self.min_flagpole_gain = 0.06  # 6% minimum impulse move
        self.max_retrace_percent = 50.0
        self.max_pullback_candles = 6
        self.min_pullback_candles = 2
        self.breakout_volume_multiple = max(0.0, float(breakout_volume_multiple))
        self.min_pullback_avg_volume = max(0.0, float(min_pullback_avg_volume))
        self.require_green_breakout = bool(require_green_breakout)

    def calculate_vwap(self, df: pd.DataFrame) -> pd.Series:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

    def _get_ema9_series(self, df: pd.DataFrame) -> pd.Series:
        if 'ema9' in df.columns:
            return df['ema9']
        try:
            import talib  # optional
            return pd.Series(talib.EMA(df['close'].values, timeperiod=9), index=df.index)
        except Exception:
            return df['close'].ewm(span=9, adjust=False).mean()

    def _analyze_pullback(self, df: pd.DataFrame, flagpole_high_idx: int,
                          flagpole_start_price: float, flagpole_high: float) -> Dict:
        current_idx = len(df) - 1
        if 'vwap' not in df.columns:
            df['vwap'] = self.calculate_vwap(df)
        ema9 = self._get_ema9_series(df)

        red_candles, last_red_idx = 0, None
        pullback_low, pullback_low_idx = flagpole_high, flagpole_high_idx
        for i in range(flagpole_high_idx + 1, current_idx + 1):
            o, c, h, l = df.iloc[i][['open', 'close', 'high', 'low']]
            if l < pullback_low:
                pullback_low, pullback_low_idx = l, i
            if c < o:
                red_candles += 1
                last_red_idx = i
            else:
                break
        last_red_high = df.iloc[last_red_idx]['high'] if last_red_idx is not None else None

        flag_move = max(1e-9, flagpole_high - flagpole_start_price)
        retrace_percent = (flagpole_high - pullback_low) / flag_move * 100.0

        vwap_at_low = df.iloc[pullback_low_idx]['vwap']
        ema_at_low = ema9.iloc[pullback_low_idx]
        broke_vwap = pullback_low < vwap_at_low if pd.notna(vwap_at_low) else False
        broke_9ema = pullback_low < ema_at_low if pd.notna(ema_at_low) else False

        pole_avg_vol = df['volume'].iloc[max(0, flagpole_high_idx-3):flagpole_high_idx+1].mean()
        end_for_pb = last_red_idx if last_red_idx is not None else current_idx
        pull_vol_series = df['volume'].iloc[flagpole_high_idx+1:end_for_pb+1] if end_for_pb > flagpole_high_idx else pd.Series([], dtype=float)
        pull_avg_vol = float(pull_vol_series.mean()) if len(pull_vol_series) > 0 else 0.0
        try:
            pull_median_vol = float(pull_vol_series.median()) if len(pull_vol_series) > 0 else 0.0
        except Exception:
            pull_median_vol = pull_avg_vol
        vol_ratio = (pull_avg_vol / pole_avg_vol) if pole_avg_vol else 0.0

        return {
            'red_candles': red_candles,
            'last_red_idx': last_red_idx,
            'last_red_high': last_red_high,
            'pullback_low': pullback_low,
            'pullback_low_idx': pullback_low_idx,
            'retrace_percent': retrace_percent,
            'broke_vwap': broke_vwap,
            'broke_9ema': broke_9ema,
            'high_volume_consolidation': vol_ratio >= 1.0,
            'vol_ratio': vol_ratio,
            'pullback_start_idx': flagpole_high_idx + 1,
            'pullback_end_idx': end_for_pb,
            'pull_avg_vol': pull_avg_vol,
            'pull_median_vol': pull_median_vol,
        }

    def _validate(self, pull: Dict) -> BullFlagValidation:
        pav = float(pull.get('pull_avg_vol') or 0.0)
        if self.min_pullback_avg_volume > 0.0 and (pav <= 0.0 or pav < self.min_pullback_avg_volume):
            return BullFlagValidation.INVALID_LOW_LIQUIDITY
        if pull['retrace_percent'] > self.max_retrace_percent:
            return BullFlagValidation.INVALID_RETRACE_TOO_DEEP
        if pull['red_candles'] > self.max_pullback_candles:
            return BullFlagValidation.INVALID_TOO_MANY_RED_CANDLES
        if pull['broke_9ema']:
            return BullFlagValidation.INVALID_BROKE_9EMA if self.strict else BullFlagValidation.VALID_WITH_WARNINGS
        if pull['high_volume_consolidation']:
            return BullFlagValidation.VALID_WITH_WARNINGS
        return BullFlagValidation.VALID

    def _entry_signal(self, df: pd.DataFrame, pull: Dict) -> Tuple[bool, Optional[float]]:
        cur = df.iloc[-1]
        try:
            if not (float(cur['close']) > float(cur['open'])):
                return False, None
        except Exception:
            return False, None
        last_red_idx = pull.get('last_red_idx')
        if last_red_idx is None:
            return False, None
        try:
            last_red_high = float(df.iloc[int(last_red_idx)]['high'])
        except Exception:
            return False, None
        if float(cur['high']) >= last_red_high:
            if self.breakout_volume_multiple > 0.0:
                pmed = float(pull.get('pull_median_vol') or 0.0)
                pav = float(pull.get('pull_avg_vol') or 0.0)
                base = pmed if pmed > 0 else pav
                cur_vol = float(cur.get('volume') or 0.0)
                if base > 0 and cur_vol < base * self.breakout_volume_multiple:
                    return False, None
            return True, last_red_high
        return False, None

    def detect_bull_flag(self, df: pd.DataFrame, symbol: str = "",
                         known_flagpole_high: Optional[float] = None,
                         known_flagpole_time: Optional[datetime] = None) -> BullFlagSignal:
        if df is None or len(df) < 10:
            return BullFlagSignal(
                timestamp=df.index[-1] if len(df) else datetime.now(),
                symbol=symbol,
                stage=BullFlagStage.NO_PATTERN,
                validation=BullFlagValidation.INVALID_NO_FLAGPOLE,
                entry_price=None,
                stop_loss=None,
                flagpole_start=0.0,
                flagpole_high=0.0,
                pullback_low=0.0,
                pullback_candles=0,
                retrace_percentage=0.0,
                volume_confirmation=False,
                broke_vwap=False,
                broke_9ema=False,
                strength_score=0.0,
            )

        # Ensure indicators
        if 'vwap' not in df.columns:
            df['vwap'] = self.calculate_vwap(df)
        if 'ema9' not in df.columns or df['ema9'].isna().all():
            df['ema9'] = self._get_ema9_series(df)

        cur_idx = len(df) - 1

        # Locate flagpole high idx
        flagpole_high_idx: Optional[int] = None
        flagpole_start_idx: int
        start_price: float
        if known_flagpole_time is not None:
            # Post-alert mode: treat the ACTION alert as the surge start.
            # Find the highest high within a short forward window after the alert (to avoid using later breakout as pole).
            try:
                ts = pd.to_datetime(known_flagpole_time)
                if ts in df.index:
                    alert_idx = int(df.index.get_loc(ts))
                else:
                    idxs = df.index.get_indexer([ts], method='pad')
                    alert_idx = int(idxs[0]) if idxs[0] != -1 else 0
            except Exception:
                alert_idx = 0
            # Forward search window (bars)
            fwd = 5
            search_end = min(cur_idx, alert_idx + fwd)
            if search_end < alert_idx:
                search_end = alert_idx
            window = df.iloc[alert_idx:search_end+1]
            rel = int(np.argmax(window['high'].values)) if len(window) > 0 else 0
            flagpole_high_idx = alert_idx + rel
            flagpole_start_idx = alert_idx
            start_price = float(df.iloc[flagpole_start_idx]['close'])
        else:
            # Discovery mode: scan recent bars for local high and adequate impulse
            look = max(10, min(15, len(df)))
            window = df.iloc[-look:]
            rel = int(np.argmax(window['high'].values))
            flagpole_high_idx = len(df) - look + rel
            # Determine start idx: scan back up to 10 bars for impulse ≥ min_flagpole_gain
            start_scan_from = max(0, flagpole_high_idx - 10)
            flagpole_start_idx = start_scan_from
            start_price = float(df.iloc[flagpole_start_idx]['close'])
            for i in range(start_scan_from, flagpole_high_idx):
                p0 = float(df.iloc[i]['close'])
                gain = (float(df.iloc[flagpole_high_idx]['high']) - p0) / max(1e-9, p0)
                if gain >= self.min_flagpole_gain:
                    flagpole_start_idx = i
                    start_price = p0
                    break

        flagpole_high = float(df.iloc[flagpole_high_idx]['high']) if known_flagpole_high is None else float(known_flagpole_high)

        # Analyze pullback
        pull = self._analyze_pullback(df, flagpole_high_idx, start_price, flagpole_high)

        # If not enough red candles, report stage
        if pull['last_red_idx'] is None or pull['red_candles'] < self.min_pullback_candles:
            return BullFlagSignal(
                timestamp=df.index[cur_idx], symbol=symbol,
                stage=BullFlagStage.PULLBACK_STAGE,
                validation=BullFlagValidation.PENDING,
                entry_price=None, stop_loss=None,
                flagpole_start=start_price, flagpole_high=flagpole_high,
                pullback_low=pull['pullback_low'], pullback_candles=pull['red_candles'],
                retrace_percentage=pull['retrace_percent'],
                volume_confirmation=(pull['vol_ratio'] < 1.0),
                broke_vwap=False, broke_9ema=pull['broke_9ema'],
                strength_score=0.0,
            )

        # Validate
        validation = self._validate(pull)
        if validation not in (BullFlagValidation.VALID, BullFlagValidation.VALID_WITH_WARNINGS):
            return BullFlagSignal(
                timestamp=df.index[cur_idx], symbol=symbol,
                stage=BullFlagStage.PATTERN_FAILED, validation=validation,
                entry_price=None, stop_loss=None,
                flagpole_start=start_price, flagpole_high=flagpole_high,
                pullback_low=pull['pullback_low'], pullback_candles=pull['red_candles'],
                retrace_percentage=pull['retrace_percent'],
                volume_confirmation=(pull['vol_ratio'] < 1.0),
                broke_vwap=False, broke_9ema=pull['broke_9ema'],
                strength_score=0.0,
            )

        # Entry
        has_entry, entry_price = self._entry_signal(df, pull)
        stage = BullFlagStage.BREAKOUT_CONFIRMED if has_entry else BullFlagStage.READY_FOR_BREAKOUT
        stop_loss = pull['pullback_low'] if validation in (BullFlagValidation.VALID, BullFlagValidation.VALID_WITH_WARNINGS) else None

        # Simple strength score
        strength = 0.5
        if not pull['high_volume_consolidation']:
            strength += 0.2
        if not pull['broke_9ema']:
            strength += 0.2
        strength = min(1.0, max(0.0, strength))

        return BullFlagSignal(
            timestamp=df.index[cur_idx], symbol=symbol,
            stage=stage, validation=validation,
            entry_price=entry_price, stop_loss=stop_loss,
            flagpole_start=start_price, flagpole_high=flagpole_high,
            pullback_low=pull['pullback_low'], pullback_candles=pull['red_candles'],
            retrace_percentage=pull['retrace_percent'],
            volume_confirmation=(pull['vol_ratio'] < 1.0),
            broke_vwap=False, broke_9ema=pull['broke_9ema'],
            strength_score=strength,
        )
