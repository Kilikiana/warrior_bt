from __future__ import annotations

"""
Lightweight Alpaca trades client for optional tick-level fills.

Environment variables:
- APCA_API_KEY_ID
- APCA_API_SECRET_KEY

Notes:
- If keys are missing or network fails, callers should gracefully fall back
  to bar-level prices.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any
import os
import time

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore


ALPACA_BASE = os.environ.get("APCA_DATA_URL", "https://data.alpaca.markets")


@dataclass
class Trade:
    t: datetime  # timestamp
    p: float     # price
    s: int       # size


def _auth_headers() -> Dict[str, str]:
    key = (
        os.environ.get("APCA-API-KEY-ID")
        or os.environ.get("APCA_API_KEY_ID")
        or os.environ.get("ALPACA_API_KEY")
    )
    sec = (
        os.environ.get("APCA-API-SECRET-KEY")
        or os.environ.get("APCA_API_SECRET_KEY")
        or os.environ.get("ALPACA_SECRET_KEY")
    )
    headers = {"accept": "application/json"}
    if key and sec:
        headers["APCA-API-KEY-ID"] = key
        headers["APCA-API-SECRET-KEY"] = sec
    return headers


def fetch_trades(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    limit: int = 1000,
    feed: str = "sip",
    max_pages: int = 20,
    timeout: float = 10.0,
) -> List[Trade]:
    """Fetch trades between start and end (UTC) with simple pagination.

    Returns empty list if requests is unavailable, creds missing, or on errors.
    """
    if requests is None:
        return []
    headers = _auth_headers()
    if "APCA-API-KEY-ID" not in headers:
        return []
    url = f"{ALPACA_BASE}/v2/stocks/trades"
    params: Dict[str, Any] = {
        "symbols": symbol.upper(),
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": limit,
        "feed": feed,
        "sort": "asc",
    }
    out: List[Trade] = []
    page = 0
    next_token: Optional[str] = None
    while page < max_pages:
        if next_token:
            params["page_token"] = next_token
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout)
            if resp.status_code != 200:
                break
            data = resp.json() or {}
            # Alpaca returns {trades: {SYMBOL: [ ... ]}, next_page_token: ...}
            trades_by_symbol = data.get("trades") or {}
            raw = trades_by_symbol.get(symbol.upper()) or []
            for t in raw:
                ts = t.get("t") or t.get("timestamp")
                try:
                    # Example: 2025-07-01T15:32:10.123456Z
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ")
                except Exception:
                    try:
                        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                    except Exception:
                        continue
                price = float(t.get("p")) if t.get("p") is not None else None
                size = int(t.get("s")) if t.get("s") is not None else 0
                if price is None:
                    continue
                out.append(Trade(t=dt, p=price, s=size))
            next_token = data.get("next_page_token")
            if not next_token:
                break
            page += 1
            # tiny pause to be nice to API
            time.sleep(0.05)
        except Exception:
            break
    return out


def first_trade_at_or_above(
    symbol: str,
    start: datetime,
    end: datetime,
    threshold_price: float,
    *,
    feed: str = "sip",
) -> Optional[Trade]:
    """Return the first trade with price >= threshold in [start,end].

    Falls back to None if API not available or none found.
    """
    trades = fetch_trades(symbol, start, end, feed=feed, max_pages=5)
    best: Optional[Trade] = None
    for tr in trades:
        if tr.p + 1e-9 >= float(threshold_price):
            if best is None or tr.t < best.t:
                best = tr
    return best
