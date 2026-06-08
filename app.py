"""
app.py — Apollo racing dashboard (Streamlit).

A PIN-gated dashboard over the Supabase data: overall top-pick win %, today's
top picks, and win% over time.

Run:  streamlit run app.py

Config via env / .env:
    SUPABASE_DB_URL   Supabase Postgres connection string (read-only use here)
    DASHBOARD_PIN     Optional; defaults to 2828
"""

import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PIN = "2828"

st.set_page_config(page_title="Apollo Racing", page_icon="🏇", layout="wide")


# ---------------------------------------------------------------------------
# PIN login gate
# ---------------------------------------------------------------------------

def require_pin() -> None:
    """Block the app until the correct PIN is entered."""
    pin = os.getenv("DASHBOARD_PIN", DEFAULT_PIN)
    if st.session_state.get("authed"):
        return

    st.title("🏇 Apollo")
    st.caption("Enter PIN to continue")
    entered = st.text_input("PIN", type="password", max_chars=8)
    if st.button("Enter") or entered:
        if entered == pin:
            st.session_state["authed"] = True
            st.rerun()
        elif entered:
            st.error("Incorrect PIN")
    st.stop()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_resource
def get_conn():
    db_url = os.getenv("SUPABASE_DB_URL") or os.getenv("DATABASE_URL")
    if not db_url:
        st.error("SUPABASE_DB_URL is not set. See .env.example.")
        st.stop()
    return psycopg2.connect(db_url)


@st.cache_data(ttl=60)
def load_overview() -> dict:
    conn = get_conn()
    settled = pd.read_sql(
        """
        SELECT r.won, e.race_time
        FROM predictions p
        JOIN results r       ON r.race_entry_id = p.race_entry_id
        JOIN race_entries e  ON e.id = p.race_entry_id
        WHERE p.is_top_pick
        """,
        conn,
    )
    total_picks = pd.read_sql(
        "SELECT COUNT(*) AS n FROM predictions WHERE is_top_pick", conn
    )["n"].iloc[0]
    races = pd.read_sql("SELECT COUNT(*) AS n FROM race_entries", conn)["n"].iloc[0]
    return {"settled": settled, "total_picks": int(total_picks), "races": int(races)}


@st.cache_data(ttl=60)
def load_today_top_picks() -> pd.DataFrame:
    conn = get_conn()
    return pd.read_sql(
        """
        SELECT e.meeting, e.race_number, e.race_time, e.horse,
               e.jockey, e.odds, p.win_probability, p.predicted_rank
        FROM predictions p
        JOIN race_entries e ON e.id = p.race_entry_id
        WHERE p.is_top_pick
          AND e.race_time::date = CURRENT_DATE
        ORDER BY e.race_time
        """,
        conn,
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

require_pin()

st.title("🏇 Apollo — Racing Dashboard")

ov = load_overview()
settled = ov["settled"]
win_rate = settled["won"].mean() if not settled.empty else 0.0

c1, c2, c3 = st.columns(3)
c1.metric("Top-pick win %", f"{win_rate:.1%}")
c2.metric("Top picks made", ov["total_picks"])
c3.metric("Runners tracked", ov["races"])

st.subheader("Today's top picks")
today = load_today_top_picks()
if today.empty:
    st.info("No top picks for today yet — run the pipeline to generate predictions.")
else:
    st.dataframe(today, use_container_width=True, hide_index=True)

st.subheader("Win % over time")
if settled.empty:
    st.info("No settled results yet.")
else:
    s = settled.copy()
    s["race_time"] = pd.to_datetime(s["race_time"], utc=True)
    s = s.sort_values("race_time")
    s["cumulative_win_pct"] = s["won"].expanding().mean()
    st.line_chart(s.set_index("race_time")["cumulative_win_pct"])
