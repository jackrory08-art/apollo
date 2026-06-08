"""
setup_db.py — Provision the Apollo racing pipeline schema on Supabase Postgres.

Creates three linked tables:

    race_entries  ──<  predictions
                  └──<  results

`race_entries` holds one row per runner per race. `predictions` and `results`
each reference a race entry via a foreign key, giving us the three linked tables.

Connection: set SUPABASE_DB_URL in your environment (or a local .env file), e.g.
    SUPABASE_DB_URL=postgresql://postgres:<password>@db.<ref>.supabase.co:5432/postgres

The script is idempotent — it can be run repeatedly without error.

NOTE on form columns: the upstream PuntersEdge scraper (VB.NET → SQL Server) only
captures odds/market fields. The form columns below (barrier, jockey, trainer,
weight, last_start_date, track_condition) are therefore nullable; they start empty
and populate automatically once a form-data source feeds them.
"""

import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()


# --- DDL -------------------------------------------------------------------

SCHEMA_SQL = """
-- 1. race_entries: one row per runner per race ----------------------------
CREATE TABLE IF NOT EXISTS race_entries (
    id                  BIGSERIAL PRIMARY KEY,
    -- race identity (sourced from the scraper)
    meeting             TEXT        NOT NULL,
    race_number         INTEGER     NOT NULL,
    race_time           TIMESTAMPTZ NOT NULL,
    distance            INTEGER,
    -- form / context fields (nullable; populated when a form source exists)
    track_condition     TEXT,
    horse               TEXT        NOT NULL,
    barrier             INTEGER,
    jockey              TEXT,
    trainer             TEXT,
    weight              NUMERIC(5, 2),
    last_start_date     DATE,
    -- market fields (sourced from the scraper)
    odds                NUMERIC(8, 2),
    bookmaker           TEXT,
    selection_id        BIGINT,
    last_traded_price   NUMERIC(8, 2),
    total_matched       NUMERIC(14, 2),
    status              TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- a runner is uniquely identified by its race + horse
    CONSTRAINT uq_race_entry UNIQUE (meeting, race_number, race_time, horse)
);

CREATE INDEX IF NOT EXISTS idx_race_entries_race_time ON race_entries (race_time);

-- 2. predictions: model output, linked to a race entry --------------------
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    race_entry_id   BIGINT      NOT NULL
                      REFERENCES race_entries (id) ON DELETE CASCADE,
    model_version   TEXT        NOT NULL DEFAULT 'v1',
    predicted_rank  INTEGER     NOT NULL,
    win_probability NUMERIC(6, 5),
    is_top_pick     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- one prediction per runner per model version
    CONSTRAINT uq_prediction UNIQUE (race_entry_id, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_top_pick
    ON predictions (is_top_pick) WHERE is_top_pick;

-- 3. results: official outcome, linked to a race entry --------------------
CREATE TABLE IF NOT EXISTS results (
    id                 BIGSERIAL PRIMARY KEY,
    race_entry_id      BIGINT      NOT NULL
                         REFERENCES race_entries (id) ON DELETE CASCADE,
    finishing_position INTEGER,
    won                BOOLEAN     NOT NULL DEFAULT FALSE,
    margin             NUMERIC(6, 2),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- one official result per runner
    CONSTRAINT uq_result UNIQUE (race_entry_id)
);
"""


def get_connection():
    """Open a connection to Supabase Postgres using SUPABASE_DB_URL."""
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        sys.exit(
            "ERROR: set SUPABASE_DB_URL (your Supabase Postgres connection string) "
            "in the environment or a .env file. See .env.example."
        )
    return psycopg2.connect(db_url)


def setup():
    """Create all tables, constraints and indexes (idempotent)."""
    conn = get_connection()
    try:
        with conn:  # commits on success, rolls back on error
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
        print("✓ Schema ready: race_entries, predictions, results")
    finally:
        conn.close()


if __name__ == "__main__":
    setup()
