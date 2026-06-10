"""
history.py - historical horse-form ingestion helpers.

Apollo's live Betfair integration finds today's runners. This module lets the
pipeline enrich those runners with past-start history from either:

* HISTORY_SOURCE=csv - a local CSV export from a form/history provider.
* HISTORY_SOURCE=api - a JSON API endpoint queried once per horse.

The generic API mode is intentionally simple because each form provider exposes
different fields. It expects either a JSON list of runs or an object containing
a "runs", "history", "results", or "data" list.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import requests

HISTORY_COLUMNS = [
    "source", "horse", "meeting", "race_number", "race_time", "distance",
    "track_condition", "barrier", "jockey", "trainer", "weight", "odds",
    "starting_price", "finishing_position", "won", "margin",
]

ALIASES = {
    "horse": ["horse", "horse_name", "runner", "runner_name", "name"],
    "meeting": ["meeting", "venue", "track", "course"],
    "race_number": ["race_number", "race_no", "race", "race_num"],
    "race_time": ["race_time", "start_time", "date", "race_date", "run_date"],
    "distance": ["distance", "distance_m", "distance_metres", "distance_meters"],
    "track_condition": ["track_condition", "going", "track", "condition"],
    "barrier": ["barrier", "stall", "stall_draw", "draw"],
    "jockey": ["jockey", "jockey_name"],
    "trainer": ["trainer", "trainer_name"],
    "weight": ["weight", "weight_value", "carried_weight"],
    "odds": ["odds", "bf_odds", "betfair_odds"],
    "starting_price": ["starting_price", "sp", "bsp", "betfair_sp"],
    "finishing_position": ["finishing_position", "position", "result", "finish"],
    "won": ["won", "winner", "is_winner"],
    "margin": ["margin", "beaten_margin"],
}


def _clean_horse(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).strip().upper()


def _coerce_bool(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "winner", "win"}:
        return True
    if text in {"0", "false", "f", "no", "n", "loser", "loss"}:
        return False
    return None


def _first_present(df: pd.DataFrame, names: list[str]):
    lowered = {str(col).lower(): col for col in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def normalize_history(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Map provider-specific columns to Apollo's horse_history schema."""
    if df.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    out = pd.DataFrame(index=df.index)
    for target, aliases in ALIASES.items():
        column = _first_present(df, aliases)
        out[target] = df[column] if column else None

    out["source"] = source
    out["horse"] = out["horse"].map(_clean_horse)
    out["race_time"] = pd.to_datetime(out["race_time"], errors="coerce", utc=True)

    numeric_cols = [
        "race_number", "distance", "barrier", "weight", "odds",
        "starting_price", "finishing_position", "margin",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "won" not in out or out["won"].isna().all():
        out["won"] = out["finishing_position"].eq(1)
    else:
        out["won"] = out["won"].map(_coerce_bool)

    out = out.dropna(subset=["horse", "race_time"])
    return out[HISTORY_COLUMNS].drop_duplicates(
        subset=["source", "horse", "race_time", "meeting"]
    )


def fetch_history_csv() -> pd.DataFrame:
    path = os.getenv("HISTORY_CSV_PATH")
    if not path:
        print("* HISTORY_SOURCE=csv but HISTORY_CSV_PATH is not set.")
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    df = pd.read_csv(path)
    history = normalize_history(df, os.getenv("HISTORY_SOURCE_NAME", "csv"))
    print(f"* Loaded {len(history)} historical starts from CSV.")
    return history


def _extract_runs(body):
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        for key in ("runs", "history", "results", "data"):
            value = body.get(key)
            if isinstance(value, list):
                return value
    return []


def fetch_history_api(current_entries: pd.DataFrame) -> pd.DataFrame:
    url = os.getenv("HISTORY_API_URL")
    if not url:
        print("* HISTORY_SOURCE=api but HISTORY_API_URL is not set.")
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    token = os.getenv("HISTORY_API_KEY")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    rows = []
    horses = sorted({_clean_horse(h) for h in current_entries.get("horse", []) if _clean_horse(h)})
    for horse in horses:
        resp = requests.get(
            url,
            params={"horse": horse},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        rows.extend(_extract_runs(resp.json()))

    history = normalize_history(
        pd.DataFrame(rows),
        os.getenv("HISTORY_SOURCE_NAME", "api"),
    )
    print(f"* Loaded {len(history)} historical starts from API.")
    return history


def fetch_external_history(current_entries: pd.DataFrame) -> pd.DataFrame:
    source = os.getenv("HISTORY_SOURCE", "none").strip().lower()
    if source in {"", "none", "off"}:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    if source == "csv":
        return fetch_history_csv()
    if source == "api":
        return fetch_history_api(current_entries)
    print(f"* Unknown HISTORY_SOURCE={source!r}; skipping external history.")
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def observed_history_from_results(df_finished: pd.DataFrame) -> pd.DataFrame:
    """Turn Apollo's own settled race rows into reusable horse history."""
    if df_finished.empty:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    out = pd.DataFrame(index=df_finished.index)
    for col in HISTORY_COLUMNS:
        out[col] = df_finished[col] if col in df_finished else None
    out["source"] = "observed"
    out["horse"] = out["horse"].map(_clean_horse)
    out["race_time"] = pd.to_datetime(out["race_time"], errors="coerce", utc=True)
    out["finishing_position"] = pd.to_numeric(df_finished.get("result"), errors="coerce")
    out["won"] = out["finishing_position"].eq(1)
    if "starting_price" not in out or out["starting_price"].isna().all():
        out["starting_price"] = out.get("odds")
    out = out.dropna(subset=["horse", "race_time", "finishing_position"])
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    return out[HISTORY_COLUMNS]
