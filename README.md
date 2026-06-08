# Apollo — Automated Racing Pipeline

An on-demand pipeline that ingests racing data from the **PuntersEdge** scraper,
stores it in **Supabase Postgres**, ranks runners with an **XGBoost** model, and
serves a **Streamlit** dashboard.

## How the pieces fit

```
PuntersEdge scraper (VB.NET → SQL Server)
        │  (pipeline.py reads its SQL Server output)
        ▼
   Supabase Postgres ──  race_entries ──< predictions
                                       └─< results
        │
        ▼
   Streamlit dashboard (app.py)
```

> **About the scraper:** `PuntersEdgeScraper` is a Visual Basic .NET app that
> scrapes Oddschecker + the Betfair API and writes **odds/market fields** to a
> SQL Server database. It cannot be imported as a Python module, so `pipeline.py`
> integrates by **reading its SQL Server output**. Form fields (jockey, trainer,
> weight, barrier, rest days, track condition) are **not** captured by the
> scraper; the schema and model are built for them and they activate automatically
> once a form-data source populates those columns.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your values
python setup_db.py          # create the Supabase tables
```

## Run

```bash
python pipeline.py          # ingest, predict upcoming, settle finished, print strike rate
streamlit run app.py        # dashboard (PIN: 2828)
```

## Files

| File             | Purpose                                                      |
|------------------|-------------------------------------------------------------|
| `setup_db.py`    | Create the 3 linked Supabase tables with unique constraints |
| `pipeline.py`    | Ingest → feature-engineer → XGBoost → predictions/results   |
| `app.py`         | PIN-gated Streamlit dashboard                               |
| `requirements.txt` | Python dependencies                                       |
| `.env.example`   | Required environment variables                              |
