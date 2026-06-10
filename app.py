"""
app.py — Apollo racing dashboard (Streamlit).

Uses the Supabase HTTP client (supabase-py) instead of psycopg2 to avoid
TCP/IPv6 connection issues on Streamlit Cloud.

Streamlit secrets required:
    SUPABASE_URL      https://<ref>.supabase.co
    SUPABASE_ANON_KEY anon/public JWT key
    DASHBOARD_PIN     optional; defaults to 2828
"""

import os
from datetime import date

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

DEFAULT_PIN = "2828"

st.set_page_config(page_title="Apollo Racing", page_icon="🏇", layout="wide")


# ---------------------------------------------------------------------------
# PIN login gate
# ---------------------------------------------------------------------------

def require_pin() -> None:
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
# Supabase HTTP client — no TCP, no IPv6 issues
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        st.error("Set SUPABASE_URL and SUPABASE_ANON_KEY in Streamlit secrets.")
        st.stop()
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_overview() -> dict:
    client = get_client()

    preds_resp = client.table("predictions").select(
        "race_entry_id, race_entries(race_time)"
    ).eq("is_top_pick", True).execute()
    preds = preds_resp.data or []
    entry_ids = [p["race_entry_id"] for p in preds]

    settled_rows = []
    if entry_ids:
        res_resp = client.table("results").select(
            "race_entry_id, won"
        ).in_("race_entry_id", entry_ids).execute()
        results_map = {r["race_entry_id"]: r["won"] for r in (res_resp.data or [])}
        for p in preds:
            eid = p["race_entry_id"]
            if eid in results_map:
                entry = p.get("race_entries") or {}
                settled_rows.append({"won": results_map[eid], "race_time": entry.get("race_time")})

    settled = pd.DataFrame(settled_rows)
    races_resp = client.table("race_entries").select("id", count="exact").execute()

    return {"settled": settled, "total_picks": len(preds), "races": races_resp.count or 0}


@st.cache_data(ttl=60)
def load_today_top_picks() -> pd.DataFrame:
    client = get_client()

    today = date.today().isoformat()
    entries_resp = client.table("race_entries").select(
        "id, meeting, race_number, race_time, horse, jockey, odds"
    ).gte("race_time", f"{today}T00:00:00+00:00").lte(
        "race_time", f"{today}T23:59:59+00:00"
    ).execute()

    entries = entries_resp.data or []
    if not entries:
        return pd.DataFrame()

    entry_ids = [e["id"] for e in entries]
    entries_map = {e["id"]: e for e in entries}

    preds_resp = client.table("predictions").select(
        "race_entry_id, predicted_rank, win_probability"
    ).eq("is_top_pick", True).in_("race_entry_id", entry_ids).execute()

    rows = []
    for p in (preds_resp.data or []):
        e = entries_map.get(p["race_entry_id"], {})
        rows.append({
            "meeting":         e.get("meeting"),
            "race_number":     e.get("race_number"),
            "race_time":       e.get("race_time"),
            "horse":           e.get("horse"),
            "jockey":          e.get("jockey"),
            "odds":            e.get("odds"),
            "win_probability": p.get("win_probability"),
            "predicted_rank":  p.get("predicted_rank"),
        })

    return pd.DataFrame(rows).sort_values("race_time") if rows else pd.DataFrame()


@st.cache_data(ttl=60)
def load_today_market_rows() -> pd.DataFrame:
    client = get_client()

    today = date.today().isoformat()
    entries_resp = client.table("race_entries").select(
        "meeting, race_number, race_time, horse, jockey, trainer, odds, "
        "last_traded_price, total_matched, status, bookmaker"
    ).gte("race_time", f"{today}T00:00:00+00:00").lte(
        "race_time", f"{today}T23:59:59+00:00"
    ).execute()

    rows = entries_resp.data or []
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["race_time", "meeting", "race_number", "horse"])


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
today_picks = load_today_top_picks()
if today_picks.empty:
    st.info("No top picks for today yet — run the pipeline to generate predictions.")
else:
    st.dataframe(today_picks, use_container_width=True, hide_index=True)

st.subheader("Betfair market watch")
market_rows = load_today_market_rows()
if market_rows.empty:
    st.info("No Betfair market rows for today yet.")
else:
    total_matched = pd.to_numeric(
        market_rows.get("total_matched"), errors="coerce"
    ).fillna(0).sum()
    st.caption(f"{len(market_rows)} runners tracked | total matched ${total_matched:,.0f}")
    st.dataframe(market_rows, use_container_width=True, hide_index=True)

st.subheader("Win % over time")
if settled.empty:
    st.info("No settled results yet.")
else:
    s = settled.copy()
    s["race_time"] = pd.to_datetime(s["race_time"], utc=True)
    s = s.sort_values("race_time")
    s["cumulative_win_pct"] = s["won"].expanding().mean()
    st.line_chart(s.set_index("race_time")["cumulative_win_pct"])
