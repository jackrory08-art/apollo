"""
betfair.py - Australian racing data via the official Betfair Exchange API.

The module returns a DataFrame matching pipeline.py's ENTRY_COLUMNS plus
"result", so the pipeline can ingest Betfair markets without depending on the
legacy PuntersEdge SQL Server scrape.

Auth:
    BF_APP_KEY
    BF_USERNAME
    BF_PASSWORD

Recommended unattended auth:
    BF_LOGIN_MODE=cert
    BF_CERT_FILE=/path/to/client-2048.crt
    BF_KEY_FILE=/path/to/client-2048.key

For local testing only, BF_LOGIN_MODE=interactive can use the interactive SSO
endpoint without a client certificate. Betfair's documented unattended/bot
flow uses certificate login.

Docs:
    https://developer.betfair.com/
    https://betfair-developer-docs.atlassian.net/wiki/spaces/1smk3cen4v3lu3yomq5qye0ni/pages/2687915/Non-Interactive+bot+login
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import requests

BETTING_URL = os.getenv("BF_BETTING_URL", "https://api.betfair.com/exchange/betting/json-rpc/v1")
CERT_IDENTITY_URL = os.getenv(
    "BF_CERT_IDENTITY_URL",
    os.getenv("BF_IDENTITY_URL", "https://identitysso-cert.betfair.com/api/certlogin"),
)
INTERACTIVE_IDENTITY_URL = os.getenv(
    "BF_INTERACTIVE_IDENTITY_URL",
    "https://identitysso.betfair.com/api/login",
)
EVENT_TYPE_HORSE = os.getenv("BF_EVENT_TYPE_ID", "7")
DEFAULT_TIMEZONE = os.getenv("BF_TIMEZONE", "Australia/Sydney")
DEBUG = os.getenv("BF_DEBUG") == "1"

ENTRY_COLUMNS = [
    "meeting", "race_number", "race_time", "distance", "track_condition",
    "horse", "barrier", "jockey", "trainer", "weight", "last_start_date",
    "odds", "bookmaker", "selection_id", "last_traded_price", "total_matched",
    "status",
]


def _log(*args) -> None:
    if DEBUG:
        print("[betfair]", *args)


def _csv_env(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}.") from exc


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Set {name} in your environment or .env file.")
    return value


def _local_zone() -> ZoneInfo:
    try:
        return ZoneInfo(DEFAULT_TIMEZONE)
    except ZoneInfoNotFoundError:
        _log(f"unknown timezone {DEFAULT_TIMEZONE!r}; falling back to UTC")
        return ZoneInfo("UTC")


def _day_window(day: str | None) -> tuple[str, str]:
    """Return Betfair UTC ISO bounds for one local racing date."""
    zone = _local_zone()
    racing_day = date.fromisoformat(day) if day else datetime.now(zone).date()
    start_local = datetime.combine(racing_day, time.min, zone)
    end_local = datetime.combine(racing_day, time.max, zone)
    start_utc = start_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    end_utc = end_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return start_utc, end_utc


# ---------------------------------------------------------------------------
# Auth + JSON-RPC
# ---------------------------------------------------------------------------

def _cert_argument() -> str | tuple[str, str]:
    cert_file = os.getenv("BF_CERT_FILE") or os.getenv("BF_CERT_PATH") or os.getenv("BF_CERT_PEM")
    key_file = os.getenv("BF_KEY_FILE") or os.getenv("BF_KEY_PATH")

    if cert_file and key_file:
        return cert_file, key_file
    if cert_file:
        return cert_file

    raise RuntimeError(
        "BF_LOGIN_MODE=cert requires BF_CERT_FILE plus BF_KEY_FILE, or one PEM "
        "file containing both the certificate and private key."
    )


def cert_login() -> str:
    """Login using Betfair's certificate-based non-interactive flow."""
    resp = requests.post(
        CERT_IDENTITY_URL,
        data={"username": _required_env("BF_USERNAME"), "password": _required_env("BF_PASSWORD")},
        headers={
            "X-Application": _required_env("BF_APP_KEY"),
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        cert=_cert_argument(),
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("loginStatus") != "SUCCESS":
        raise RuntimeError(f"Betfair certificate login failed: {body.get('loginStatus') or body}")
    _log("certificate login OK")
    return body["sessionToken"]


def interactive_login() -> str:
    """Login via the interactive SSO endpoint. Useful for local development."""
    resp = requests.post(
        INTERACTIVE_IDENTITY_URL,
        data={"username": _required_env("BF_USERNAME"), "password": _required_env("BF_PASSWORD")},
        headers={
            "X-Application": _required_env("BF_APP_KEY"),
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "SUCCESS":
        raise RuntimeError(f"Betfair interactive login failed: {body.get('error') or body}")
    _log("interactive login OK")
    return body["token"]


def login() -> str:
    mode = os.getenv("BF_LOGIN_MODE", "auto").strip().lower()
    if mode == "auto":
        mode = "cert" if (os.getenv("BF_CERT_FILE") or os.getenv("BF_CERT_PEM")) else "interactive"
    if mode == "cert":
        return cert_login()
    if mode == "interactive":
        return interactive_login()
    raise RuntimeError("BF_LOGIN_MODE must be one of: auto, cert, interactive.")


def _rpc(method: str, params: dict, token: str):
    payload = {
        "jsonrpc": "2.0",
        "method": f"SportsAPING/v1.0/{method}",
        "params": params,
        "id": 1,
    }
    resp = requests.post(
        BETTING_URL,
        data=json.dumps(payload),
        headers={
            "X-Application": _required_env("BF_APP_KEY"),
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

def list_markets(token: str, day: str | None = None) -> list[dict]:
    """All configured horse-racing WIN markets for a local racing date."""
    start_utc, end_utc = _day_window(day)
    params = {
        "filter": {
            "eventTypeIds": [EVENT_TYPE_HORSE],
            "marketCountries": _csv_env("BF_MARKET_COUNTRIES", "AU"),
            "marketTypeCodes": _csv_env("BF_MARKET_TYPES", "WIN"),
            "marketStartTime": {"from": start_utc, "to": end_utc},
        },
        "marketProjection": [
            "MARKET_START_TIME", "EVENT", "RUNNER_DESCRIPTION", "RUNNER_METADATA",
        ],
        "sort": "FIRST_TO_START",
        "maxResults": _int_env("BF_MAX_RESULTS", 200),
    }
    markets = _rpc("listMarketCatalogue", params, token)
    _log(f"{len(markets)} markets from {start_utc} to {end_utc}")
    return markets


def list_books(token: str, market_ids: list[str]) -> dict[str, dict]:
    """Map market_id -> marketBook (prices + settlement status), batched."""
    books: dict[str, dict] = {}
    for i in range(0, len(market_ids), 25):
        batch = market_ids[i:i + 25]
        params = {
            "marketIds": batch,
            "priceProjection": {"priceData": ["EX_BEST_OFFERS", "EX_TRADED"]},
        }
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
    match = re.search(r"\bR(?:ace)?\s*(\d+)\b", market_name or "", re.IGNORECASE)
    return _int(match.group(1)) if match else 0


def _distance(market_name: str, meta: dict):
    meta_distance = _int(meta.get("DISTANCE"))
    if meta_distance:
        return meta_distance
    match = re.search(r"(\d{3,4})\s*m\b", market_name or "", re.IGNORECASE)
    return _int(match.group(1)) if match else None


def _last_start_date(race_time_iso: str | None, meta: dict):
    days = _int(meta.get("DAYS_SINCE_LAST_RUN"))
    if days is None or race_time_iso is None:
        return None
    try:
        rt = datetime.fromisoformat(race_time_iso.replace("Z", "+00:00"))
        return (rt - timedelta(days=days)).date().isoformat()
    except ValueError:
        return None


def _best_back_price(runner_book: dict):
    ex = runner_book.get("ex", {})
    available = ex.get("availableToBack") or []
    if available:
        return available[0].get("price")
    return runner_book.get("lastPriceTraded")


def _result_from_status(market_closed: bool, runner_status: str | None):
    if runner_status == "WINNER":
        return 1
    if market_closed and runner_status == "LOSER":
        return 2
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_from_betfair(day: str | None = None) -> pd.DataFrame:
    """Pull configured Betfair horse-racing markets into race_entries shape."""
    token = login()
    markets = list_markets(token, day)
    if not markets:
        print("* Betfair returned no matching markets.")
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])

    market_ids = [market["marketId"] for market in markets]
    books = list_books(token, market_ids)

    rows: list[dict] = []
    for market in markets:
        market_id = market["marketId"]
        event = market.get("event", {})
        meeting = event.get("venue") or event.get("name")
        market_name = market.get("marketName", "")
        race_time = market.get("marketStartTime")
        book = books.get(market_id, {})
        market_closed = book.get("status") == "CLOSED"
        book_runners = {runner["selectionId"]: runner for runner in book.get("runners", [])}

        for runner in market.get("runners", []):
            selection_id = runner["selectionId"]
            meta = runner.get("metadata", {}) or {}
            runner_book = book_runners.get(selection_id, {})
            runner_status = runner_book.get("status")

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
                "odds": _best_back_price(runner_book),
                "bookmaker": "betfair",
                "selection_id": selection_id,
                "last_traded_price": runner_book.get("lastPriceTraded"),
                "total_matched": runner_book.get("totalMatched"),
                "status": runner_status,
                "result": _result_from_status(market_closed, runner_status),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])

    df = df.drop_duplicates(
        subset=["meeting", "race_number", "race_time", "horse"]
    ).reset_index(drop=True)
    n_races = df[["meeting", "race_number", "race_time"]].drop_duplicates().shape[0]
    print(f"* Betfair: {len(df)} runners across {n_races} races.")
    return df[ENTRY_COLUMNS + ["result"]]


if __name__ == "__main__":
    print(fetch_from_betfair(os.getenv("SCRAPE_DATE") or None).head(20).to_string())
