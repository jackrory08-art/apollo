# Apollo - Automated Racing Pipeline

An on-demand pipeline that ingests Australian racing markets from the
**official Betfair Exchange API**, stores them in **Supabase Postgres**, ranks
runners with an **XGBoost** model, and serves a **Streamlit** dashboard.

## How the pieces fit

```text
Betfair Exchange API
        |  (pipeline.py reads AU WIN markets)
        v
   Supabase Postgres -- race_entries --< predictions
                                      \--< results
        |
        v
   Streamlit dashboard (app.py)
```

> **Betfair auth:** unattended pipeline runs should use Betfair certificate
> login (`BF_LOGIN_MODE=cert`, `BF_CERT_FILE`, `BF_KEY_FILE`). The interactive
> login endpoint can be used for local experiments, but the certificate flow is
> the documented bot/login method for automation.

The legacy PuntersEdge SQL Server and punters.com.au paths are still available
through `DATA_SOURCE=sqlserver` and `DATA_SOURCE=punters`, but Betfair is the
default source.

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

For a specific racing date:

```bash
SCRAPE_DATE=2026-06-10 python pipeline.py
```

## Files

| File               | Purpose                                                     |
|--------------------|-------------------------------------------------------------|
| `setup_db.py`      | Create the 3 linked Supabase tables with unique constraints |
| `betfair.py`       | Betfair Exchange API ingestion and market parsing           |
| `pipeline.py`      | Ingest -> feature-engineer -> XGBoost -> predictions/results |
| `app.py`           | PIN-gated Streamlit dashboard                              |
| `requirements.txt` | Python dependencies                                         |
| `.env.example`     | Required environment variables                              |
