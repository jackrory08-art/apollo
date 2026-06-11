CREATE TABLE IF NOT EXISTS horse_history (
    id                  BIGSERIAL PRIMARY KEY,
    source              TEXT        NOT NULL DEFAULT 'observed',
    horse               TEXT        NOT NULL,
    meeting             TEXT,
    race_number         INTEGER,
    race_time           TIMESTAMPTZ NOT NULL,
    distance            INTEGER,
    track_condition     TEXT,
    barrier             INTEGER,
    jockey              TEXT,
    trainer             TEXT,
    weight              NUMERIC(5, 2),
    odds                NUMERIC(8, 2),
    starting_price      NUMERIC(8, 2),
    finishing_position  INTEGER,
    won                 BOOLEAN,
    margin              NUMERIC(6, 2),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_horse_history_start UNIQUE (source, horse, race_time, meeting)
);

CREATE INDEX IF NOT EXISTS idx_horse_history_horse_time
    ON horse_history (horse, race_time DESC);

CREATE INDEX IF NOT EXISTS idx_horse_history_race_time
    ON horse_history (race_time DESC);
