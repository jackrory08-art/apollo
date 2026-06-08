"""
pipeline.py — Apollo unified on-demand racing pipeline.

Flow
----
1. Pull race/runner/odds/result rows from the PuntersEdge scraper's SQL Server DB
   (the VB.NET app is not importable into Python, so its SQL Server output is the
   integration seam — triggering a *fresh* scrape means running the VB .exe).
2. Upsert those rows into Supabase `race_entries`.
3. For each race:
     - upcoming  → engineer form features, run the XGBoost model, write `predictions`
     - finished  → read official outcomes, write `results`
4. Print the top-pick precision strike rate.

Run:  python pipeline.py

Config via env / .env (see .env.example):
    SUPABASE_DB_URL, SQLSERVER_HOST, SQLSERVER_DB, SQLSERVER_USER, SQLSERVER_PASSWORD

NOTE: The SQL Server source only carries odds/market fields. The form features
(jockey/trainer combo, weight shift, rest days, track-condition suitability) are
written to operate on the schema's form columns and degrade gracefully to neutral
defaults while those columns are empty — they sharpen automatically once a
form-data source populates them.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

MODEL_VERSION = "v1"


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def supabase_conn():
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("ERROR: SUPABASE_DB_URL not set (see .env.example).")
    return psycopg2.connect(db_url)


# ---------------------------------------------------------------------------
# 1. Fetch from the scraper's SQL Server output
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
        -- RaceNumber is not stored by the scraper; derive a stable int from time.
        -- Use 0 as a placeholder; adjust below if you add a RaceNumber column.
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

# Columns we expect to hand around the pipeline.
ENTRY_COLUMNS = [
    "meeting", "race_number", "race_time", "distance", "track_condition",
    "horse", "barrier", "jockey", "trainer", "weight", "last_start_date",
    "odds", "bookmaker", "selection_id", "last_traded_price", "total_matched",
    "status",
]


def fetch_from_sqlserver() -> pd.DataFrame:
    """
    Read the latest race/runner rows from the PuntersEdge SQL Server DB.

    Returns a DataFrame with ENTRY_COLUMNS plus a `result` column (the scraper's
    official finishing position; NULL/None while the race is still upcoming).

    If SQL Server is unreachable or unconfigured we return an empty frame so the
    rest of the pipeline can still run (e.g. against already-ingested data).
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
    except Exception as exc:  # noqa: BLE001 - surface but don't crash the run
        print(f"• Could not read from SQL Server ({exc}); continuing without ingest.")
        return pd.DataFrame(columns=ENTRY_COLUMNS + ["result"])


# ---------------------------------------------------------------------------
# 2. Upsert race entries into Supabase
# ---------------------------------------------------------------------------

def upsert_race_entries(conn, df: pd.DataFrame) -> None:
    """Insert/refresh runners in race_entries (ON CONFLICT updates live odds)."""
    if df.empty:
        return

    rows = [tuple(r) for r in df[ENTRY_COLUMNS].itertuples(index=False, name=None)]
    cols = ", ".join(ENTRY_COLUMNS)
    update_cols = [c for c in ENTRY_COLUMNS
                   if c not in ("meeting", "race_number", "race_time", "horse")]
    update_set = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

    sql = (
        f"INSERT INTO race_entries ({cols}) VALUES %s "
        f"ON CONFLICT (meeting, race_number, race_time, horse) "
        f"DO UPDATE SET {update_set}"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    print(f"• Upserted {len(rows)} race entries.")


# ---------------------------------------------------------------------------
# 3a. Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "odds_implied_prob",
    "market_support",
    "jockey_trainer_combo",
    "weight_shift",
    "rest_days",
    "track_condition_suitability",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model features from a frame of race_entries rows.

    Every feature is NaN-safe: where the underlying form column is missing the
    feature falls back to a neutral value, so the model still runs on odds/market
    signals alone and improves as form data arrives.
    """
    f = pd.DataFrame(index=df.index)

    # --- market signals (always available) ---
    odds = pd.to_numeric(df.get("odds"), errors="coerce")
    f["odds_implied_prob"] = (1.0 / odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    matched = pd.to_numeric(df.get("total_matched"), errors="coerce").fillna(0.0)
    f["market_support"] = np.log1p(matched)

    # --- jockey/trainer combo strike rate (historical, within this batch) ---
    won = df["won"] if "won" in df else pd.Series(0, index=df.index)
    combo = (df.get("jockey").fillna("?") + "|" + df.get("trainer").fillna("?")
             if "jockey" in df and "trainer" in df
             else pd.Series("?|?", index=df.index))
    combo_strike = won.groupby(combo).transform("mean") if "won" in df else 0.0
    f["jockey_trainer_combo"] = pd.Series(combo_strike, index=df.index).fillna(0.0)

    # --- weight shift vs. field average (proxy for class/weight pressure) ---
    weight = pd.to_numeric(df.get("weight"), errors="coerce")
    f["weight_shift"] = (weight - weight.mean()).fillna(0.0)

    # --- rest days since last start ---
    last = pd.to_datetime(df.get("last_start_date"), errors="coerce", utc=True)
    race_t = pd.to_datetime(df.get("race_time"), errors="coerce", utc=True)
    f["rest_days"] = (race_t - last).dt.days.fillna(0).clip(lower=0)

    # --- track-condition suitability (runner's avg result on this condition) ---
    if "track_condition" in df and "won" in df:
        cond_strike = won.groupby(df["track_condition"].fillna("?")).transform("mean")
        f["track_condition_suitability"] = pd.Series(cond_strike, index=df.index).fillna(0.0)
    else:
        f["track_condition_suitability"] = 0.0

    return f[FEATURE_COLUMNS].astype(float)


# ---------------------------------------------------------------------------
# 3b. Model
# ---------------------------------------------------------------------------

class RacingModel:
    """
    Thin XGBoost wrapper. Trains a win classifier on labelled history when we have
    enough of it; otherwise falls back to a transparent odds-based heuristic so the
    pipeline always produces ranked predictions.
    """

    def __init__(self, model_version: str = MODEL_VERSION):
        self.model_version = model_version
        self.model = None

    def train(self, features: pd.DataFrame, labels: pd.Series) -> None:
        labels = labels.astype(int)
        # Need both classes and a minimum sample size for a meaningful fit.
        if features.empty or labels.nunique() < 2 or len(features) < 30:
            print("• Not enough labelled history — using odds heuristic.")
            return
        from xgboost import XGBClassifier

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=2,
        )
        self.model.fit(features, labels)
        print(f"• Trained XGBoost on {len(features)} labelled runners.")

    def win_probability(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is not None:
            return self.model.predict_proba(features)[:, 1]
        # Heuristic fallback: normalised implied probability from the odds.
        p = features["odds_implied_prob"].to_numpy(dtype=float)
        total = p.sum()
        return p / total if total > 0 else np.full(len(p), 1.0 / max(len(p), 1))


# ---------------------------------------------------------------------------
# Data access helpers
# ---------------------------------------------------------------------------

def load_entries(conn) -> pd.DataFrame:
    """Load race_entries joined to any known result (for labels + outcomes)."""
    sql = """
        SELECT e.*, r.won, r.finishing_position
        FROM race_entries e
        LEFT JOIN results r ON r.race_entry_id = e.id
    """
    return pd.read_sql(sql, conn)


def write_predictions(conn, race_entry_ids, ranks, probs, top_pick_id) -> None:
    rows = [
        (int(eid), MODEL_VERSION, int(rank), float(prob), bool(eid == top_pick_id))
        for eid, rank, prob in zip(race_entry_ids, ranks, probs)
    ]
    sql = (
        "INSERT INTO predictions "
        "(race_entry_id, model_version, predicted_rank, win_probability, is_top_pick) "
        "VALUES %s "
        "ON CONFLICT (race_entry_id, model_version) DO UPDATE SET "
        "predicted_rank = EXCLUDED.predicted_rank, "
        "win_probability = EXCLUDED.win_probability, "
        "is_top_pick = EXCLUDED.is_top_pick, created_at = now()"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()


def write_results(conn, df_finished: pd.DataFrame) -> None:
    """Upsert official outcomes from the scraper's `result` field."""
    rows = []
    for _, row in df_finished.iterrows():
        pos = pd.to_numeric(row.get("result"), errors="coerce")
        if pd.isna(pos):
            continue
        pos = int(pos)
        rows.append((int(row["id"]), pos, pos == 1))
    if not rows:
        return
    sql = (
        "INSERT INTO results (race_entry_id, finishing_position, won) VALUES %s "
        "ON CONFLICT (race_entry_id) DO UPDATE SET "
        "finishing_position = EXCLUDED.finishing_position, "
        "won = EXCLUDED.won, updated_at = now()"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    print(f"• Recorded {len(rows)} official results.")


# ---------------------------------------------------------------------------
# Strike rate
# ---------------------------------------------------------------------------

def top_pick_strike_rate(conn) -> tuple[float, int, int]:
    """Precision of our top picks: top-picks that won / settled top-picks."""
    sql = """
        SELECT COUNT(*) FILTER (WHERE r.won) AS hits,
               COUNT(*)                      AS settled
        FROM predictions p
        JOIN results r ON r.race_entry_id = p.race_entry_id
        WHERE p.is_top_pick
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        hits, settled = cur.fetchone()
    rate = (hits / settled) if settled else 0.0
    return rate, hits or 0, settled or 0


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run() -> None:
    conn = supabase_conn()
    try:
        # 1 + 2: ingest from the scraper and persist.
        scraped = fetch_from_sqlserver()
        upsert_race_entries(conn, scraped)

        # If the scraper handed us finished races, record their results now.
        if not scraped.empty and "result" in scraped:
            # Re-read entries so we have their DB ids to link results against.
            entries = load_entries(conn)
            finished_scraped = scraped[scraped["result"].notna()]
            if not finished_scraped.empty:
                # Match scraped finished rows to entry ids by full unique key.
                merged = finished_scraped.merge(
                    entries[["id", "meeting", "race_number", "race_time", "horse"]],
                    on=["meeting", "race_number", "race_time", "horse"],
                    how="inner",
                )
                write_results(conn, merged)

        # 3: per-race predictions for upcoming races.
        entries = load_entries(conn)
        if entries.empty:
            print("• No race entries in the database yet.")
        else:
            entries["race_time"] = pd.to_datetime(entries["race_time"], utc=True)
            now = datetime.now(timezone.utc)

            # Train once on all settled history.
            settled = entries[entries["won"].notna()]
            model = RacingModel()
            if not settled.empty:
                model.train(engineer_features(settled), settled["won"].fillna(False))
            else:
                model.train(pd.DataFrame(), pd.Series(dtype=int))

            upcoming = entries[entries["race_time"] > now]
            n_races = 0
            for _, grp in upcoming.groupby(["meeting", "race_number", "race_time"]):
                if grp.empty:
                    continue
                feats = engineer_features(grp)
                probs = model.win_probability(feats)
                order = np.argsort(-probs)  # highest prob first
                ranks = np.empty(len(probs), dtype=int)
                ranks[order] = np.arange(1, len(probs) + 1)
                top_pick_id = int(grp.iloc[int(order[0])]["id"])
                write_predictions(conn, grp["id"].tolist(), ranks, probs, top_pick_id)
                n_races += 1
            print(f"• Generated predictions for {n_races} upcoming races.")

        # 4: report top-pick precision.
        rate, hits, settled_n = top_pick_strike_rate(conn)
        print("\n" + "=" * 48)
        print(f"  TOP-PICK PRECISION STRIKE RATE: {rate:.1%}")
        print(f"  ({hits} winners from {settled_n} settled top picks)")
        print("=" * 48)
    finally:
        conn.close()


if __name__ == "__main__":
    run()
