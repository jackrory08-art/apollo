"""
pipeline.py — Apollo unified on-demand racing pipeline.

Flow
----
1. Pull race/runner/odds/result rows from the configured data source. Betfair
   Exchange API is the default and recommended source for Australian WIN
   markets.
2. Upsert those rows into Supabase `race_entries` via the HTTP client (no TCP/IPv6).
3. For each race:
     - upcoming  → engineer form features, run the XGBoost model, write `predictions`
     - finished  → read official outcomes, write `results`
4. Print the top-pick precision strike rate.

Run:  python pipeline.py

Config via env / .env (see .env.example):
    SUPABASE_URL, SUPABASE_ANON_KEY,
    BF_APP_KEY, BF_USERNAME, BF_PASSWORD, BF_LOGIN_MODE

NOTE: Betfair runner metadata can populate form columns where available. The
feature code still degrades gracefully to neutral defaults when any field is
missing.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from history import (
    HISTORY_COLUMNS,
    fetch_external_history,
    observed_history_from_results,
)

load_dotenv()

MODEL_VERSION = "v2"


# ---------------------------------------------------------------------------
# Supabase HTTP client (avoids all TCP/IPv6 issues)
# ---------------------------------------------------------------------------

def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise SystemExit("ERROR: set SUPABASE_URL and SUPABASE_ANON_KEY (see .env.example).")
    return create_client(url, key)


def _clean(val):
    """Convert pandas/numpy scalars to JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if np.isnan(val) else float(val)
    if isinstance(val, pd.Timestamp):
        return val.isoformat()
    if isinstance(val, bool):
        return val
    return val


# ---------------------------------------------------------------------------
# 1. Fetch from the configured market-data source
# ---------------------------------------------------------------------------

# Column mapping: scraper field  ->  our race_entries column.
#
# Real schema (from PuntersEdgeScraper App.config / Module1.vb):
#   Database : PuntersEdge  (AWS RDS SQL Server)
#   TodaysRaces  — Meeting, RaceTime, Horse, Odds, BookMaker
#   BetFairData  — Meeting, RaceTime, Horse, SelectionID, LastTradedprice,
#                  Market_TotalMatched, selection_TotalMatched
#   Results      — Meeting, RaceTime, Horse, Result
#
# Form columns (barrier, jockey, trainer, weight, last_start_date,
# track_condition) are NOT captured by the scraper — they remain NULL and
# the model's form features activate once a form-data source populates them.
SCRAPER_QUERY = """
    SELECT
        t.Meeting                    AS meeting,
        0                            AS race_number,
        t.RaceTime                   AS race_time,
        NULL                         AS distance,
        NULL                         AS track_condition,
        t.Horse                      AS horse,
        NULL                         AS barrier,
        NULL                         AS jockey,
        NULL                         AS trainer,
        NULL                         AS weight,
        NULL                         AS last_start_date,
        t.Odds                       AS odds,
        t.BookMaker                  AS bookmaker,
        b.SelectionID                AS selection_id,
        b.LastTradedprice            AS last_traded_price,
        b.Market_TotalMatched        AS total_matched,
        NULL                         AS status,
        r.Result                     AS result
    FROM       TodaysRaces  t
    LEFT JOIN  BetFairData  b ON  b.Meeting  = t.Meeting
                              AND b.RaceTime  = t.RaceTime
                              AND b.Horse     = t.Horse
    LEFT JOIN  Results      r ON  r.Meeting  = t.Meeting
                              AND r.RaceTime  = t.RaceTime
                              AND r.Horse     = t.Horse
"""

ENTRY_COLUMNS = [
    "meeting", "race_number", "race_time", "distance", "track_condition",
    "horse", "barrier", "jockey", "trainer", "weight", "last_start_date",
    "odds", "bookmaker", "selection_id", "last_traded_price", "total_matched",
    "status",
]


def _horse_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def fetch_data() -> pd.DataFrame:
    """
    Fetch today's race/runner rows from the configured data source.

    DATA_SOURCE=betfair (default) → official Betfair Exchange API (AU racing).
    DATA_SOURCE=punters           → scrape punters.com.au (blocked by bot
                                    protection; kept for reference).
    DATA_SOURCE=sqlserver         → legacy PuntersEdge SQL Server DB.
    """
    source = os.getenv("DATA_SOURCE", "betfair").lower()
    try:
        if source == "sqlserver":
            return fetch_from_sqlserver()
        if source == "punters":
            from scraper import fetch_from_punters
            return fetch_from_punters(os.getenv("SCRAPE_DATE") or None)
        from betfair import fetch_from_betfair
        return fetch_from_betfair(os.getenv("SCRAPE_DATE") or None)
    except Exception as exc:
        print(f"• Data fetch failed ({exc}); continuing without ingest.")
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])


def fetch_from_sqlserver() -> pd.DataFrame:
    """
    Read the latest race/runner rows from the PuntersEdge SQL Server DB.
    Returns an empty frame if SQL Server is unreachable or unconfigured.
    """
    host = os.getenv("SQLSERVER_HOST")
    if not host:
        print("• SQLSERVER_HOST not set — skipping scraper ingest.")
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])

    try:
        import pymssql
    except ImportError:
        print("• pymssql not installed — skipping scraper ingest.")
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])

    try:
        conn = pymssql.connect(
            server=host,
            user=os.getenv("SQLSERVER_USER"),
            password=os.getenv("SQLSERVER_PASSWORD"),
            database=os.getenv("SQLSERVER_DB", "PuntersEdge"),
        )
        try:
            df = pd.read_sql(SCRAPER_QUERY, conn)
        finally:
            conn.close()
        print(f"• Fetched {len(df)} rows from SQL Server.")
        return df
    except Exception as exc:
        print(f"• Could not read from SQL Server ({exc}); continuing without ingest.")
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])


# ---------------------------------------------------------------------------
# 2. Upsert race entries into Supabase
# ---------------------------------------------------------------------------

def upsert_race_entries(client, df: pd.DataFrame) -> None:
    if df.empty:
        return
    records = [
        {col: _clean(row[col]) for col in ENTRY_COLUMNS}
        for _, row in df[ENTRY_COLUMNS].iterrows()
    ]
    client.table("race_entries").upsert(
        records, on_conflict="meeting,race_number,race_time,horse"
    ).execute()
    print(f"• Upserted {len(records)} race entries.")


def upsert_horse_history(client, df: pd.DataFrame) -> None:
    if df.empty:
        return
    records = [
        {col: _clean(row[col]) for col in HISTORY_COLUMNS}
        for _, row in df[HISTORY_COLUMNS].iterrows()
    ]
    try:
        client.table("horse_history").upsert(
            records, on_conflict="source,horse,race_time,meeting"
        ).execute()
        print(f"• Upserted {len(records)} horse-history starts.")
    except Exception as exc:
        print(f"• Could not write horse_history ({exc}). Run setup_db.py after adding SUPABASE_DB_URL.")


# ---------------------------------------------------------------------------
# 3a. Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "odds_implied_prob", "market_support", "jockey_trainer_combo",
    "weight_shift", "rest_days", "track_condition_suitability",
    "horse_history_starts", "horse_history_win_rate",
    "horse_history_place_rate", "horse_history_avg_finish",
    "horse_history_best_finish", "horse_history_days_since_run",
    "distance_history_win_rate", "track_history_win_rate",
    "going_history_win_rate",
]


def _history_feature_rows(df: pd.DataFrame, history: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "horse_history_starts", "horse_history_win_rate",
        "horse_history_place_rate", "horse_history_avg_finish",
        "horse_history_best_finish", "horse_history_days_since_run",
        "distance_history_win_rate", "track_history_win_rate",
        "going_history_win_rate",
    ]
    defaults = pd.DataFrame(0.0, index=df.index, columns=columns)
    if history is None or history.empty:
        return defaults

    hist = history.copy()
    hist["horse_key"] = hist["horse"].map(_horse_key)
    hist["race_time"] = pd.to_datetime(hist["race_time"], errors="coerce", utc=True)
    hist["finishing_position"] = pd.to_numeric(hist.get("finishing_position"), errors="coerce")
    hist["won"] = hist.get("won", pd.Series(False, index=hist.index)).fillna(False).astype(bool)
    hist["placed"] = hist["finishing_position"].le(3)

    rows = []
    for _, row in df.iterrows():
        horse = _horse_key(row.get("horse"))
        race_time = pd.to_datetime(row.get("race_time"), errors="coerce", utc=True)
        past = hist[(hist["horse_key"] == horse) & (hist["race_time"] < race_time)]
        if past.empty:
            rows.append({col: 0.0 for col in columns})
            continue

        past = past.sort_values("race_time")
        valid_finish = past["finishing_position"].dropna()
        row_distance = pd.to_numeric(row.get("distance"), errors="coerce")
        same_distance = past[pd.to_numeric(past.get("distance"), errors="coerce").eq(row_distance)]
        same_track = past[past.get("meeting").fillna("").str.upper().eq(
            str(row.get("meeting") or "").upper()
        )]
        same_going = past[past.get("track_condition").fillna("").str.upper().eq(
            str(row.get("track_condition") or "").upper()
        )]
        days_since = (race_time - past.iloc[-1]["race_time"]).days if pd.notna(race_time) else 0

        rows.append({
            "horse_history_starts": float(len(past)),
            "horse_history_win_rate": float(past["won"].mean()),
            "horse_history_place_rate": float(past["placed"].fillna(False).mean()),
            "horse_history_avg_finish": float(valid_finish.mean()) if not valid_finish.empty else 0.0,
            "horse_history_best_finish": float(valid_finish.min()) if not valid_finish.empty else 0.0,
            "horse_history_days_since_run": float(max(days_since, 0)),
            "distance_history_win_rate": float(same_distance["won"].mean()) if not same_distance.empty else 0.0,
            "track_history_win_rate": float(same_track["won"].mean()) if not same_track.empty else 0.0,
            "going_history_win_rate": float(same_going["won"].mean()) if not same_going.empty else 0.0,
        })

    return pd.DataFrame(rows, index=df.index, columns=columns).fillna(0.0)


def engineer_features(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build model features from a frame of race_entries rows.
    NaN-safe: falls back to neutral values where form columns are missing.
    """
    f = pd.DataFrame(index=df.index)

    odds = pd.to_numeric(df.get("odds"), errors="coerce")
    f["odds_implied_prob"] = (1.0 / odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matched = pd.to_numeric(df.get("total_matched"), errors="coerce").fillna(0.0)
    f["market_support"] = np.log1p(matched)

    won = df["won"] if "won" in df else pd.Series(0, index=df.index)
    combo = (
        df.get("jockey", pd.Series("?", index=df.index)).fillna("?") + "|" +
        df.get("trainer", pd.Series("?", index=df.index)).fillna("?")
    )
    combo_strike = won.groupby(combo).transform("mean") if "won" in df else 0.0
    f["jockey_trainer_combo"] = pd.Series(combo_strike, index=df.index).fillna(0.0)

    weight = pd.to_numeric(df.get("weight"), errors="coerce")
    f["weight_shift"] = (weight - weight.mean()).fillna(0.0)

    last = pd.to_datetime(df.get("last_start_date"), errors="coerce", utc=True)
    race_t = pd.to_datetime(df.get("race_time"), errors="coerce", utc=True)
    f["rest_days"] = (race_t - last).dt.days.fillna(0).clip(lower=0)

    if "track_condition" in df and "won" in df:
        cond_strike = won.groupby(df["track_condition"].fillna("?")).transform("mean")
        f["track_condition_suitability"] = pd.Series(cond_strike, index=df.index).fillna(0.0)
    else:
        f["track_condition_suitability"] = 0.0

    hist_features = _history_feature_rows(df, history)
    for col in hist_features.columns:
        f[col] = hist_features[col]

    return f[FEATURE_COLUMNS].astype(float)


# ---------------------------------------------------------------------------
# 3b. Model
# ---------------------------------------------------------------------------

class RacingModel:
    """XGBoost win classifier with odds-heuristic fallback."""

    def __init__(self, model_version: str = MODEL_VERSION):
        self.model_version = model_version
        self.model = None

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        labels = labels.astype(int)
        if features.empty or labels.nunique() < 2 or len(features) < 30:
            print("• Not enough labelled history — using odds heuristic.")
            return
        from xgboost import XGBClassifier
        self.model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=2,
        )
        self.model.fit(features, labels)
        print(f"• Trained XGBoost on {len(features)} labelled runners.")

    def win_probability(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is not None:
            return self.model.predict_proba(features)[:, 1]
        p = features["odds_implied_prob"].to_numpy(dtype=float)
        total = p.sum()
        return p / total if total > 0 else np.full(len(p), 1.0 / max(len(p), 1))


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------

def load_entries(client) -> pd.DataFrame:
    """Load all race_entries, left-joined with any known results."""
    entries_resp = client.table("race_entries").select("*").execute()
    entries = entries_resp.data or []
    if not entries:
        return pd.DataFrame()

    df = pd.DataFrame(entries)
    results_resp = client.table("results").select(
        "race_entry_id, won, finishing_position"
    ).execute()
    results = results_resp.data or []

    if results:
        rdf = pd.DataFrame(results)
        df = df.merge(rdf, left_on="id", right_on="race_entry_id", how="left")
    else:
        df["won"] = None
        df["finishing_position"] = None

    return df


def load_horse_history(client) -> pd.DataFrame:
    try:
        resp = client.table("horse_history").select("*").limit(10000).execute()
    except Exception as exc:
        print(f"• Could not read horse_history ({exc}); using market-only features.")
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    rows = resp.data or []
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=HISTORY_COLUMNS)


def write_predictions(client, race_entry_ids, ranks, probs, top_pick_id) -> None:
    records = [
        {
            "race_entry_id": int(eid),
            "model_version": MODEL_VERSION,
            "predicted_rank": int(rank),
            "win_probability": float(prob),
            "is_top_pick": bool(int(eid) == int(top_pick_id)),
        }
        for eid, rank, prob in zip(race_entry_ids, ranks, probs)
    ]
    client.table("predictions").upsert(
        records, on_conflict="race_entry_id,model_version"
    ).execute()


def write_results(client, df_finished: pd.DataFrame) -> None:
    records = []
    for _, row in df_finished.iterrows():
        pos = pd.to_numeric(row.get("result"), errors="coerce")
        if pd.isna(pos):
            continue
        pos = int(pos)
        records.append({
            "race_entry_id": int(row["id"]),
            "finishing_position": pos,
            "won": pos == 1,
        })
    if not records:
        return
    client.table("results").upsert(
        records, on_conflict="race_entry_id"
    ).execute()
    print(f"• Recorded {len(records)} official results.")


def top_pick_strike_rate(client) -> tuple[float, int, int]:
    preds_resp = client.table("predictions").select(
        "race_entry_id"
    ).eq("is_top_pick", True).execute()
    entry_ids = [p["race_entry_id"] for p in (preds_resp.data or [])]

    if not entry_ids:
        return 0.0, 0, 0

    results_resp = client.table("results").select(
        "race_entry_id, won"
    ).in_("race_entry_id", entry_ids).execute()
    results = results_resp.data or []

    settled = len(results)
    hits = sum(1 for r in results if r["won"])
    return (hits / settled if settled else 0.0), hits, settled


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> None:
    client = get_supabase_client()

    scraped = fetch_data()
    upsert_race_entries(client, scraped)
    if not scraped.empty:
        upsert_horse_history(client, fetch_external_history(scraped))

    if not scraped.empty and "result" in scraped:
        entries = load_entries(client)
        finished_scraped = scraped[scraped["result"].notna()]
        if not finished_scraped.empty and not entries.empty:
            merged = finished_scraped.merge(
                entries[["id", "meeting", "race_number", "race_time", "horse"]],
                on=["meeting", "race_number", "race_time", "horse"],
                how="inner",
            )
            write_results(client, merged)
            upsert_horse_history(client, observed_history_from_results(finished_scraped))

    entries = load_entries(client)
    history = load_horse_history(client)
    if entries.empty:
        print("• No race entries in the database yet.")
    else:
        entries["race_time"] = pd.to_datetime(entries["race_time"], utc=True)
        now = datetime.now(timezone.utc)

        settled = entries[entries["won"].notna()]
        model = RacingModel()
        if not settled.empty:
            model.train(engineer_features(settled, history), settled["won"].fillna(False))
        else:
            model.train(pd.DataFrame(), pd.Series(dtype=int))

        upcoming = entries[entries["race_time"] > now]
        n_races = 0
        for _, grp in upcoming.groupby(["meeting", "race_number", "race_time"]):
            if grp.empty:
                continue
            feats = engineer_features(grp, history)
            probs = model.win_probability(feats)
            order = np.argsort(-probs)
            ranks = np.empty(len(probs), dtype=int)
            ranks[order] = np.arange(1, len(probs) + 1)
            top_pick_id = int(grp.iloc[int(order[0])]["id"])
            write_predictions(client, grp["id"].tolist(), ranks, probs, top_pick_id)
            n_races += 1
        print(f"• Generated predictions for {n_races} upcoming races.")

    rate, hits, settled_n = top_pick_strike_rate(client)
    print("\n" + "=" * 48)
    print(f"  TOP-PICK PRECISION STRIKE RATE: {rate:.1%}")
    print(f"  ({hits} winners from {settled_n} settled top picks)")
    print("=" * 48)


if __name__ == "__main__":
    run()
