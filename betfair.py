"""
betfair.py — Australian racing data via the official Betfair Exchange API.

Replaces the blocked punters.com.au scrape with a legitimate, reliable source.
Betfair returns real AU WIN markets with runners, live odds, traded volume,
settled winners AND form metadata (jockey, trainer, weight, barrier, days since
last run) — which powers the model's form features.

Returns a pandas DataFrame matching pipeline.py's ENTRY_COLUMNS + "result".

Auth (set as env / GitHub secrets):
    BF_APP_KEY    Betfair Application Key (a free "delayed" key is fine)
    BF_USERNAME   Betfair account username
    BF_PASSWORD   Betfair account password
    BF_IDENTITY_URL (optional) override login host; AU accounts may need
                    https://identitysso.betfair.com.au/api/login

Docs: https://developer.betfair.com/  (Sports API, Exchange Betting)
"""

import json
import os
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

IDENTITY_URL = os.getenv("BF_IDENTITY_URL", "https://identitysso.betfair.com/api/login")
BETTING_URL = "https://api.betfair.com/exchange/betting/json-rpc/v1"
EVENT_TYPE_HORSE = "7"
DEBUG = os.getenv("BF_DEBUG") == "1"

ENTRY_COLUMNS = [
    "meeting", "race_number", "race_time", "distance", "track_condition",
    "horse", "barrier", "jockey", "trainer", "weight", "last_start_date",
    "odds", "bookmaker", "selection_id", "last_traded_price", "total_matched",
    "status",
]


def _log(*args):
    if DEBUG:
        print("[betfair]", *args)


# ---------------------------------------------------------------------------
# Auth + JSON-RPC
# ---------------------------------------------------------------------------

def login() -> str:
    app_key = os.environ["BF_APP_KEY"]
    resp = requests.post(
        IDENTITY_URL,
        data={"username": os.environ["BF_USERNAME"], "password": os.environ["BF_PASSWORD"]},
        headers={
            "X-Application": app_key,
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "SUCCESS":
        raise RuntimeError(f"Betfair login failed: {body.get('error') or body}")
    _log("login OK")
    return body["token"]


def _rpc(method: str, params: dict, token: str):
    payload = {"jsonrpc": "2.0", "method": f"SportsAPING/v1.0/{method}", "params": params, "id": 1}
    resp = requests.post(
        BETTING_URL,
        data=json.dumps(payload),
        headers={
            "X-Application": os.environ["BF_APP_KEY"],
            "X-Authentication": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if isinstance(body, dict) and body.get("error"):
        raise RuntimeError(f"Betfair {method} error: {body['error']}")
    return body["result"]


# ---------------------------------------------------------------------------
# Market discovery + book
# ---------------------------------------------------------------------------

def list_markets(token: str) -> list[dict]:
    """All AU horse-racing WIN markets for today, with runners + form metadata."""
    now = datetime.now(timezone.utc)
    end = now.replace(hour=23, minute=59, second=59)
    params = {
        "filter": {
            "eventTypeIds": [EVENT_TYPE_HORSE],
            "marketCountries": ["AU"],
            "marketTypeCodes": ["WIN"],
            # Include earlier today so already-run races come back for results.
            "marketStartTime": {
                "from": now.replace(hour=0, minute=0, second=0).isoformat(),
                "to": end.isoformat(),
            },
        },
        "marketProjection": [
            "MARKET_START_TIME", "EVENT", "RUNNER_DESCRIPTION", "RUNNER_METADATA",
        ],
        "sort": "FIRST_TO_START",
        "maxResults": 200,
    }
    markets = _rpc("listMarketCatalogue", params, token)
    _log(f"{len(markets)} AU WIN markets today")
    return markets


def list_books(token: str, market_ids: list[str]) -> dict:
    """Map market_id -> marketBook (prices + settlement status), batched."""
    books: dict[str, dict] = {}
    for i in range(0, len(market_ids), 25):
        batch = market_ids[i:i + 25]
        params = {"marketIds": batch, "priceProjection": {"priceData": ["EX_BEST_OFFERS"]}}
        for book in _rpc("listMarketBook", params, token):
            books[book["marketId"]] = book
    return books


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _race_number(market_name: str) -> int:
    import re
    m = re.search(r"R(?:ace)?\s*(\d+)", market_name or "", re.IGNORECASE)
    return _int(m.group(1)) if m else 0


def _distance(market_name: str, meta: dict):
    import re
    if meta.get("RACE_TYPE") and (d := _int(meta.get("DISTANCE"))):
        return d
    m = re.search(r"(\d{3,4})\s*m", market_name or "", re.IGNORECASE)
    return _int(m.group(1)) if m else None


def _last_start_date(race_time_iso: str | None, meta: dict):
    days = _int(meta.get("DAYS_SINCE_LAST_RUN"))
    if days is None or race_time_iso is None:
        return None
    try:
        rt = datetime.fromisoformat(race_time_iso.replace("Z", "+00:00"))
        return (rt - timedelta(days=days)).date().isoformat()
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_from_betfair(_day: str | None = None) -> pd.DataFrame:
    """Pull today's AU WIN markets into the pipeline's race_entries schema.

    `result`: 1 for the settled winner, 2 for any settled (beaten) runner, None
    while the market is still open. Betfair exposes winner/loser, not full
    placings, so beaten runners carry won=False with an approximate position —
    enough for the top-pick strike rate, which keys off the winner.
    """
    token = login()
    markets = list_markets(token)
    if not markets:
        print("• Betfair returned no AU WIN markets for today.")
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])

    market_ids = [m["marketId"] for m in markets]
    books = list_books(token, market_ids)

    rows: list[dict] = []
    for market in markets:
        mid = market["marketId"]
        event = market.get("event", {})
        meeting = event.get("venue") or event.get("name")
        market_name = market.get("marketName", "")
        race_time = market.get("marketStartTime")
        book = books.get(mid, {})
        market_closed = book.get("status") == "CLOSED"

        # selectionId -> runner book entry (prices + status)
        book_runners = {r["selectionId"]: r for r in book.get("runners", [])}

        for runner in market.get("runners", []):
            sid = runner["selectionId"]
            meta = runner.get("metadata", {}) or {}
            br = book_runners.get(sid, {})

            best_back = None
            ex = br.get("ex", {})
            if ex.get("availableToBack"):
                best_back = ex["availableToBack"][0].get("price")

            r_status = br.get("status")  # ACTIVE / WINNER / LOSER / REMOVED
            if r_status == "WINNER":
                result = 1
            elif market_closed and r_status == "LOSER":
                result = 2
            else:
                result = None

            rows.append({
                "meeting": meeting,
                "race_number": _race_number(market_name),
                "race_time": race_time,
                "distance": _distance(market_name, meta),
                "track_condition": meta.get("GOING"),
                "horse": runner.get("runnerName"),
                "barrier": _int(meta.get("STALL_DRAW")),
                "jockey": meta.get("JOCKEY_NAME"),
                "trainer": meta.get("TRAINER_NAME"),
                "weight": _float(meta.get("WEIGHT_VALUE")),
                "last_start_date": _last_start_date(race_time, meta),
                "odds": best_back,
                "bookmaker": "betfair",
                "selection_id": sid,
                "last_traded_price": br.get("lastPriceTraded"),
                "total_matched": br.get("totalMatched"),
                "status": r_status,
                "result": result,
            })

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(
        subset=["meeting", "race_number", "race_time", "horse"]
    ).reset_index(drop=True)
    n_races = df[["meeting", "race_number", "race_time"]].drop_duplicates().shape[0]
    print(f"• Betfair: {len(df)} runners across {n_races} AU races.")
    return df[ENTRY_COLUMNS + ["result"]]


if __name__ == "__main__":
    print(fetch_from_betfair().head(20).to_string())
