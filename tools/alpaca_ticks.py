from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import requests
import pytz


ALPACA_DATA_BASE = "https://data.alpaca.markets/v2/stocks"


def _get_keys() -> tuple[str, str]:
    """Resolve Alpaca API keys from multiple common env var names.

    Supported keys (checked in order):
      - APCA_API_KEY_ID, APCA_API_SECRET_KEY (official)
      - APCA-API-KEY-ID, APCA-API-SECRET-KEY (hyphen style)
      - ALPACA_API_KEY, ALPACA_SECRET_KEY (project .env)
    """
    key = (
        os.getenv("APCA_API_KEY_ID")
        or os.getenv("APCA-API-KEY-ID")
        or os.getenv("ALPACA_API_KEY")
        or ""
    )
    secret = (
        os.getenv("APCA_API_SECRET_KEY")
        or os.getenv("APCA-API-SECRET-KEY")
        or os.getenv("ALPACA_SECRET_KEY")
        or ""
    )
    if not key or not secret:
        raise RuntimeError("Alpaca API keys not found in environment")
    return key, secret


def _et_range_utc(date_str: str, start_hm: str, end_hm: str) -> tuple[str, str]:
    tz = pytz.timezone("US/Eastern")
    st = tz.localize(datetime.strptime(f"{date_str} {start_hm}", "%Y-%m-%d %H:%M"))
    en = tz.localize(datetime.strptime(f"{date_str} {end_hm}", "%Y-%m-%d %H:%M"))
    return st.astimezone(pytz.UTC).isoformat().replace("+00:00", "Z"), en.astimezone(pytz.UTC).isoformat().replace("+00:00", "Z")


def fetch_premarket_high(symbol: str, date: str, feed: str = "sip", timeout: int = 20) -> Optional[float]:
    """Return premarket high from 04:00 to 09:30 ET for given symbol/date using Alpaca trades.

    Args:
        symbol: ticker
        date: YYYY-MM-DD (US/Eastern trading date)
        feed: 'sip' or 'iex'
        timeout: per-request timeout seconds
    """
    key, secret = _get_keys()
    start, end = _et_range_utc(date, "04:00", "09:30")
    url = f"{ALPACA_DATA_BASE}/{symbol}/trades"
    headers = {"accept": "application/json", "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    params = {"start": start, "end": end, "limit": 10000, "feed": feed, "sort": "asc"}
    pmh = None
    token = None
    pages = 0
    while True:
        if token:
            params["page_token"] = token
        resp = requests.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json() or {}
        trades = data.get("trades", [])
        for t in trades:
            try:
                p = float(t.get("p")) if "p" in t else float(t.get("price"))
            except Exception:
                continue
            pmh = p if pmh is None or p > pmh else pmh
        token = data.get("next_page_token")
        pages += 1
        if not token or pages > 50:
            break
    return pmh
