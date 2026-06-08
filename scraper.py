"""
scraper.py — Live Australian racing scraper for punters.com.au.

Replaces the dead PuntersEdge SQL Server source with a real, on-demand scrape
of punters.com.au race results / form pages.

Returns a pandas DataFrame matching the columns pipeline.py expects
(ENTRY_COLUMNS + "result").

Selectors are grounded in the `predictive-punter/punters_client` reference
library. punters.com.au changes its markup periodically and applies bot
protection, so every field is parsed defensively — a missing/renamed selector
yields None for that field rather than crashing the whole run. Run with
SCRAPER_DEBUG=1 to print what each step found, which makes selector drift easy
to spot and fix.

Usage:
    from scraper import fetch_from_punters
    df = fetch_from_punters("2026-06-08")   # date defaults to today
"""

import os
import re
from datetime import date, datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.punters.com.au"
RESULTS_URL = BASE + "/racing-results/{date}/"

# Australian state/territory slugs that appear in race URLs.
STATES = ("nsw", "vic", "qld", "sa", "wa", "tas", "act", "nt")
STATE_RE = re.compile(r"/(" + "|".join(STATES) + r")/", re.IGNORECASE)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-AU,en;q=0.9",
}

DEBUG = os.getenv("SCRAPER_DEBUG") == "1"

# Mirrors pipeline.ENTRY_COLUMNS so the two stay in lockstep.
ENTRY_COLUMNS = [
    "meeting", "race_number", "race_time", "distance", "track_condition",
    "horse", "barrier", "jockey", "trainer", "weight", "last_start_date",
    "odds", "bookmaker", "selection_id", "last_traded_price", "total_matched",
    "status",
]


def _log(*args):
    if DEBUG:
        print("[scraper]", *args)


def _get(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        _log(f"GET {url} -> HTTP {resp.status_code} ({len(resp.content)} bytes)")
        if resp.status_code != 200:
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as exc:  # network / parse errors shouldn't kill the run
        _log(f"GET {url} failed: {exc}")
        return None


def _text(node) -> str | None:
    if node is None:
        return None
    t = node.get_text(strip=True)
    return t or None


def _to_int(value) -> int | None:
    try:
        return int(re.sub(r"[^\d-]", "", str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> float | None:
    try:
        return float(re.sub(r"[^\d.]", "", str(value)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Discovery: find race-page URLs for a given date
# ---------------------------------------------------------------------------

def get_race_urls(day: str) -> list[str]:
    """Return de-duplicated race-page URLs listed on the results page for `day`."""
    soup = _get(RESULTS_URL.format(date=day))
    if soup is None:
        return []

    urls: list[str] = []
    seen = set()
    for a in soup.select("a[href]"):
        href = a["href"]
        if STATE_RE.search(href) and "/form-guide/" not in href:
            full = href if href.startswith("http") else BASE + href
            if full not in seen:
                seen.add(full)
                urls.append(full)
    _log(f"{day}: found {len(urls)} candidate race URLs")
    return urls


# ---------------------------------------------------------------------------
# Parsing: race header + runners
# ---------------------------------------------------------------------------

def parse_race(url: str) -> list[dict]:
    """Parse one race page into a list of runner dicts (ENTRY_COLUMNS + result)."""
    soup = _get(url)
    if soup is None:
        return []

    # --- race-level metadata ---
    meeting = _text(soup.select_one("a.label-link"))

    ts = soup.select_one("div.details-line abbr.timestamp")
    race_time = None
    if ts and ts.get("data-utime"):
        race_time = datetime.fromtimestamp(
            int(ts["data-utime"]), tz=timezone.utc
        ).isoformat()

    dist_node = soup.select_one("span.distance abbr.conversion")
    distance = _to_int(dist_node["data-value"]) if dist_node and dist_node.get("data-value") else None

    # Race number is usually in the URL (…-race-3 / /race/3) or a header.
    rn = re.search(r"race[-/](\d+)", url, re.IGNORECASE)
    race_number = _to_int(rn.group(1)) if rn else 0

    # --- runners ---
    runners: list[dict] = []
    horse_links = soup.select("a.form-guide-overview__horse-link")
    for h in horse_links:
        row = h.find_parent("tr") or h.find_parent("div")

        def pick(selector):
            return row.select_one(selector) if row else None

        horse = _text(h)
        jockey = _text(pick("a.form-guide-overview__jockey-link"))
        trainer = _text(pick("a.form-guide-overview__trainer-link"))
        barrier = _to_int(_text(pick("td.form-guide-overview__competitor-barrier")))
        weight = _to_float(_text(pick("td.form-guide-overview__competitor-weight")))

        pos_node = pick("span.formSummaryPosition")
        pos_text = _text(pos_node)
        result = None if (pos_text in (None, "Abn", "NR")) else _to_int(pos_text)

        if not horse:
            continue

        runners.append({
            "meeting": meeting,
            "race_number": race_number,
            "race_time": race_time,
            "distance": distance,
            "track_condition": None,
            "horse": horse,
            "barrier": barrier,
            "jockey": jockey,
            "trainer": trainer,
            "weight": weight,
            "last_start_date": None,
            "odds": None,
            "bookmaker": None,
            "selection_id": None,
            "last_traded_price": None,
            "total_matched": None,
            "status": None,
            "result": result,
        })

    _log(f"{url}: parsed {len(runners)} runners (meeting={meeting!r})")
    return runners


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_from_punters(day: str | None = None) -> pd.DataFrame:
    """Scrape all races for `day` (YYYY-MM-DD, default today) into a DataFrame."""
    day = day or date.today().isoformat()
    print(f"• Scraping punters.com.au for {day} …")

    all_rows: list[dict] = []
    for url in get_race_urls(day):
        all_rows.extend(parse_race(url))

    cols = ENTRY_COLUMNS + ["result"]
    if not all_rows:
        print("• Scraper found no races (selectors may need updating — set "
              "SCRAPER_DEBUG=1 for detail).")
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(all_rows)
    # Drop exact dupes (same runner can appear under multiple bookmaker tabs).
    df = df.drop_duplicates(
        subset=["meeting", "race_number", "race_time", "horse"]
    ).reset_index(drop=True)
    print(f"• Scraped {len(df)} runners across "
          f"{df[['meeting', 'race_number', 'race_time']].drop_duplicates().shape[0]} races.")
    return df[cols]


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else None
    out = fetch_from_punters(target)
    print(out.head(20).to_string())
