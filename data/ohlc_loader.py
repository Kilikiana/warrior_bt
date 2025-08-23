"""
OHLCV data loader utilities.

Provides functions to load 1-minute and 5-minute bar data for a symbol and date
from the shared cache directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal
import json
import gzip
import pandas as pd

from core.config import OHLCV_5MIN_DIR, OHLCV_1MIN_DIR


Timeframe = Literal["1min", "5min"]


def load_symbol_ohlc_data(symbol: str, date: str, timeframe: Timeframe = "5min") -> pd.DataFrame:
    """Load OHLC data for a specific symbol and date from shared cache.

    Args:
        symbol: Ticker symbol
        date: YYYY-MM-DD
        timeframe: "1min" or "5min"

    Returns:
        DataFrame indexed by timestamp with columns open, high, low, close, volume.
        Empty DataFrame if not found or parse error.
    """
    if timeframe == "5min":
        data_path = OHLCV_5MIN_DIR / f"ohlcv_5min_{date}.json.gz"
    else:
        data_path = OHLCV_1MIN_DIR / f"ohlcv_1min_{date}.json.gz"

    if not data_path.exists():
        return pd.DataFrame()

    try:
        with gzip.open(data_path, "rt") as f:
            all_data = json.load(f)

        # Find record for symbol
        symbol_record = next((rec for rec in all_data if rec.get("symbol") == symbol), None)
        if not symbol_record:
            return pd.DataFrame()

        if timeframe == "5min":
            bar_data = symbol_record.get("bars", [])
        else:
            bar_data = symbol_record.get("minutes", [])

        if not bar_data:
            return pd.DataFrame()

        df = pd.DataFrame(bar_data)
        # Construct timestamps from date + time string
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(f"{date} " + df["time"].astype(str))
        else:
            # Some datasets may already contain a timestamp
            if "timestamp" not in df.columns:
                return pd.DataFrame()

        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()

