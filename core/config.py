"""
Core configuration and paths for the Warrior BT project.

Loads environment variables if available and centralizes common directories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None


def _load_env() -> None:
    """Load environment variables from .env if python-dotenv is installed."""
    if load_dotenv is not None:
        # Look for .env at repo root
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.exists():
            load_dotenv(env_path)


_load_env()


# Project root is two levels up from this file (core/config.py)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Data directories
SHARED_CACHE_DIR: Path = PROJECT_ROOT / "shared_cache"
OHLCV_5MIN_DIR: Path = SHARED_CACHE_DIR / "ohlcv_5min_bars"
OHLCV_1MIN_DIR: Path = SHARED_CACHE_DIR / "ohlcv_1min_bars"
OHLCV_DAILY_DIR: Path = SHARED_CACHE_DIR / "ohlcv_daily_bars"

# Results directories
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGS_DIR: Path = RESULTS_DIR / "logs"
SCANS_DIR: Path = RESULTS_DIR / "hod_momentum_scans"


def get_scan_file(date: str) -> Path:
    """Return path to scan file for a given date (YYYY-MM-DD)."""
    return SCANS_DIR / f"hod_momentum_scan_{date}.json"


def get_log_file(date: str) -> Path:
    """Default log file path for a given backtest date."""
    return LOGS_DIR / f"backtest_{date}.log"


def ensure_dirs(*paths: Path) -> None:
    """Ensure directories exist for the provided paths (interpreted as directories)."""
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

