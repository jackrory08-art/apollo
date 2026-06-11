r"""
Import Betfair BASIC historical horse-racing data into Apollo.

Expected input is a Betfair Historical Data .tar containing BASIC/**/*.bz2
stream files. The importer filters to AU WIN markets by default and writes:

* race_entries
* results
* horse_history

Run from PowerShell:
    .venv\Scripts\python.exe import_betfair_history.py --tar C:\path\to\data.tar

Use --dry-run first to inspect counts without writing to Supabase.
"""

from __future__ import annotations

import argparse
import bz2
import json
import os
import re
import tarfile
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

from history import HISTORY_COLUMNS
from pipeline import ENTRY_COLUMNS, _clean

load_dotenv()

SOURCE = "betfair_historical"


@dataclass
class ParsedMarket:
    market_id: str
    meeting: str
    race_number: int
    race_time: str
    distance: int | None
    market_name: str
    rows: list[dict]


def get_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env.")
    return create_client(url, key)


def clean_horse(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = re.sub(r"^\s*\d+\s*[.\-)]\s*", "", str(name)).strip()
    return re.sub(r"\s+", " ", cleaned)


def race_number(market_name: str | None) -> int:
    match = re.search(r"\bR(?:ace)?\s*(\d+)\b", market_name or "", re.IGNORECASE)
    return int(match.group(1)) if match else 0


def distance_metres(market_name: str | None) -> int | None:
    match = re.search(r"(\d{3,4})\s*m\b", market_name or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def meeting_name(event_name: str | None) -> str:
    name = event_name or "Unknown"
    name = re.sub(r"\s*\(AUS\)\s*", " ", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+\d{1,2}(st|nd|rd|th)?\s+\w+\s*$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip()


def is_market_file(name: str) -> bool:
    return "/1." in name.replace("\\", "/") and name.endswith(".bz2")


def parse_market_bytes(name: str, raw: bytes, country: str, market_type: str) -> ParsedMarket | None:
    last_definition = None
    runner_prices: dict[int, dict] = {}
    market_id = None

    for line in raw.splitlines():
        if not line:
            continue
        obj = json.loads(line)
        for market_change in obj.get("mc", []):
            market_id = market_change.get("id") or market_id
            if market_change.get("marketDefinition"):
                last_definition = market_change["marketDefinition"]
            for runner_change in market_change.get("rc", []) or []:
                runner_id = runner_change.get("id")
                if runner_id is None:
                    continue
                state = runner_prices.setdefault(int(runner_id), {})
                if runner_change.get("ltp") is not None:
                    state["last_traded_price"] = runner_change.get("ltp")
                if runner_change.get("tv") is not None:
                    state["total_matched"] = runner_change.get("tv")

    if not last_definition:
        return None
    if last_definition.get("countryCode") != country:
        return None
    if last_definition.get("marketType") != market_type:
        return None
    if last_definition.get("numberOfWinners") != 1:
        return None

    market_name = last_definition.get("name") or ""
    race_time = last_definition.get("marketTime")
    meeting = meeting_name(last_definition.get("eventName"))
    rows = []

    for runner in last_definition.get("runners", []):
        horse = clean_horse(runner.get("name"))
        if not horse:
            continue
        runner_id = int(runner["id"])
        prices = runner_prices.get(runner_id, {})
        status = runner.get("status")
        if status == "WINNER":
            result = 1
        elif last_definition.get("status") == "CLOSED" and status == "LOSER":
            result = 2
        else:
            result = None

        rows.append({
            "meeting": meeting,
            "race_number": race_number(market_name),
            "race_time": race_time,
            "distance": distance_metres(market_name),
            "track_condition": None,
            "horse": horse,
            "barrier": None,
            "jockey": None,
            "trainer": None,
            "weight": None,
            "last_start_date": None,
            "odds": prices.get("last_traded_price"),
            "bookmaker": SOURCE,
            "selection_id": runner_id,
            "last_traded_price": prices.get("last_traded_price"),
            "total_matched": prices.get("total_matched"),
            "status": status,
            "result": result,
        })

    if not rows or not race_time:
        return None
    return ParsedMarket(
        market_id=market_id or name,
        meeting=meeting,
        race_number=race_number(market_name),
        race_time=race_time,
        distance=distance_metres(market_name),
        market_name=market_name,
        rows=rows,
    )


def iter_markets(tar_path: str, country: str, market_type: str, limit: int | None = None):
    seen_market_ids = set()
    parsed = 0
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if not member.isfile() or not is_market_file(member.name):
                continue
            f = archive.extractfile(member)
            if f is None:
                continue
            try:
                raw = bz2.decompress(f.read())
                market = parse_market_bytes(member.name, raw, country, market_type)
            except Exception as exc:
                print(f"* Skipped {member.name}: {exc}")
                continue
            if market is None or market.market_id in seen_market_ids:
                continue
            seen_market_ids.add(market.market_id)
            parsed += 1
            yield market
            if limit and parsed >= limit:
                break


def chunks(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def dedupe_rows(rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    deduped: dict[tuple, dict] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        deduped[key] = row
    return list(deduped.values())


def history_records(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        if row.get("result") is None:
            continue
        records.append({
            "source": SOURCE,
            "horse": row["horse"].upper(),
            "meeting": row["meeting"],
            "race_number": row["race_number"],
            "race_time": row["race_time"],
            "distance": row["distance"],
            "track_condition": row["track_condition"],
            "barrier": row["barrier"],
            "jockey": row["jockey"],
            "trainer": row["trainer"],
            "weight": row["weight"],
            "odds": row["odds"],
            "starting_price": row["last_traded_price"],
            "finishing_position": row["result"],
            "won": row["result"] == 1,
            "margin": None,
        })
    return records


def result_records(entry_rows: list[dict], upserted_entries: list[dict]) -> list[dict]:
    ids = {
        (r.get("meeting"), r.get("race_number"), r.get("race_time"), r.get("horse")): r.get("id")
        for r in upserted_entries
    }
    records = []
    for row in entry_rows:
        if row.get("result") is None:
            continue
        entry_id = ids.get((row["meeting"], row["race_number"], row["race_time"], row["horse"]))
        if not entry_id:
            continue
        records.append({
            "race_entry_id": int(entry_id),
            "finishing_position": int(row["result"]),
            "won": row["result"] == 1,
        })
    return records


def write_batch(client, rows: list[dict], batch_size: int) -> tuple[int, int, int]:
    rows = dedupe_rows(rows, ("meeting", "race_number", "race_time", "horse"))
    entry_rows = [{col: _clean(row[col]) for col in ENTRY_COLUMNS} for row in rows]
    hist_source_rows = dedupe_rows(
        history_records(rows),
        ("source", "horse", "race_time", "meeting"),
    )
    hist_rows = [{col: _clean(row[col]) for col in HISTORY_COLUMNS} for row in hist_source_rows]

    entries_written = 0
    results_written = 0
    history_written = 0

    for row_batch in chunks(rows, batch_size):
        batch = [{col: _clean(row[col]) for col in ENTRY_COLUMNS} for row in row_batch]
        resp = client.table("race_entries").upsert(
            batch, on_conflict="meeting,race_number,race_time,horse"
        ).execute()
        upserted = resp.data or []
        entries_written += len(batch)
        results = result_records(row_batch, upserted)
        if results:
            client.table("results").upsert(results, on_conflict="race_entry_id").execute()
            results_written += len(results)

    for batch in chunks(hist_rows, batch_size):
        client.table("horse_history").upsert(
            batch, on_conflict="source,horse,race_time,meeting"
        ).execute()
        history_written += len(batch)

    return entries_written, results_written, history_written


def normalize_time(value) -> str:
    return pd.to_datetime(value, errors="coerce", utc=True).isoformat()


def merge_key(row: dict) -> tuple:
    return (
        str(row.get("meeting") or "").strip().upper(),
        normalize_time(row.get("race_time")),
        str(row.get("horse") or "").strip().upper(),
    )


def fetch_all(client, table: str, columns: str, page_size: int = 1000, **filters) -> list[dict]:
    rows: list[dict] = []
    start = 0
    while True:
        query = client.table(table).select(columns)
        for key, value in filters.items():
            query = query.eq(key, value)
        resp = query.range(start, start + page_size - 1).execute()
        page = resp.data or []
        rows.extend(page)
        if len(page) < page_size:
            return rows
        start += page_size


def sync_results_from_history(client, batch_size: int) -> int:
    print("* Loading imported race entries for result sync...")
    entries = fetch_all(
        client,
        "race_entries",
        "id,meeting,race_time,horse",
        bookmaker=SOURCE,
    )
    print(f"* Loaded {len(entries)} race entries.")

    print("* Loading historical settled starts...")
    history = fetch_all(
        client,
        "horse_history",
        "horse,meeting,race_time,finishing_position,won",
        source=SOURCE,
    )
    print(f"* Loaded {len(history)} horse-history rows.")

    entry_ids = {merge_key(row): row["id"] for row in entries}
    records = []
    for row in history:
        entry_id = entry_ids.get(merge_key(row))
        if not entry_id or row.get("finishing_position") is None:
            continue
        records.append({
            "race_entry_id": int(entry_id),
            "finishing_position": int(row["finishing_position"]),
            "won": bool(row.get("won")),
        })

    records = dedupe_rows(records, ("race_entry_id",))
    written = 0
    for batch in chunks(records, batch_size):
        client.table("results").upsert(batch, on_conflict="race_entry_id").execute()
        written += len(batch)
        if written % 10000 == 0:
            print(f"* Synced {written} results...")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tar", default=os.getenv("BETFAIR_HISTORY_TAR"))
    parser.add_argument("--country", default=os.getenv("BETFAIR_HISTORY_COUNTRY", "AU"))
    parser.add_argument("--market-type", default=os.getenv("BETFAIR_HISTORY_MARKET_TYPE", "WIN"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-results-only", action="store_true")
    args = parser.parse_args()

    if not args.tar:
        raise SystemExit("Pass --tar or set BETFAIR_HISTORY_TAR.")

    client = None if args.dry_run else get_client()
    if args.sync_results_only:
        if args.dry_run:
            raise SystemExit("--sync-results-only cannot be combined with --dry-run.")
        written = sync_results_from_history(client, args.batch_size)
        print(f"Result sync complete. Results: {written}")
        return

    total_markets = total_entries = total_results = total_history = 0
    pending: list[dict] = []
    started = datetime.now()

    for market in iter_markets(args.tar, args.country, args.market_type, args.limit):
        total_markets += 1
        pending.extend(market.rows)
        if total_markets % 100 == 0:
            print(f"* Parsed {total_markets} markets...")
        if len(pending) >= args.batch_size:
            if args.dry_run:
                total_entries += len(pending)
                total_results += sum(1 for row in pending if row.get("result") is not None)
                total_history += sum(1 for row in pending if row.get("result") is not None)
            else:
                e, r, h = write_batch(client, pending, args.batch_size)
                total_entries += e
                total_results += r
                total_history += h
            pending = []

    if pending:
        if args.dry_run:
            total_entries += len(pending)
            total_results += sum(1 for row in pending if row.get("result") is not None)
            total_history += sum(1 for row in pending if row.get("result") is not None)
        else:
            e, r, h = write_batch(client, pending, args.batch_size)
            total_entries += e
            total_results += r
            total_history += h

    mode = "Dry run" if args.dry_run else "Import"
    elapsed = datetime.now() - started
    print(f"{mode} complete in {elapsed}.")
    print(f"Markets: {total_markets}")
    print(f"Race entries: {total_entries}")
    print(f"Results: {total_results}")
    print(f"Horse-history starts: {total_history}")


if __name__ == "__main__":
    main()
