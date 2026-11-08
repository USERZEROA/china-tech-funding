"""Fetch, clean, merge, store, and analyze the two public data sources."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
RUNS = ROOT / "artifacts" / "runs"
ZENODO_API = "https://zenodo.org/api/records/19194192"
INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "gdp_per_capita": "NY.GDP.PCAP.CD",
}
REQUIRED = {
    "Investee Company", "Investing Company", "Round Investors", "Round",
    "Amount (USD)", "Year", "Type", "Sector", "Industry", "Country",
}


def run_dir(run_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    path = RUNS / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def latest_run_summary() -> Path:
    """Choose the latest completed pipeline run by actual finish time."""
    paths = list(RUNS.glob("*/run_summary.json"))
    if not paths:
        raise ValueError("No completed Part 1 run was found")
    return max(paths, key=lambda path: datetime.fromisoformat(
        json.loads(path.read_text(encoding="utf-8"))["task_times"]["analyze_from_sql"]["ended_utc"]))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded(func):
    @wraps(func)
    def wrapped(run_id, *args):
        started = datetime.now(timezone.utc).isoformat()
        try:
            result = func(run_id, *args)
            save_json(run_dir(run_id) / f"{func.__name__}.json", {
                "run_id": run_id, "started_utc": started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "status": "success", "result": result,
            })
            if func.__name__ == "analyze_from_sql":
                summarize_run(run_id)
            return result
        except Exception as exc:
            save_json(run_dir(run_id) / f"{func.__name__}.json", {
                "run_id": run_id, "started_utc": started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "status": "failed", "error": str(exc),
            })
            raise
    return wrapped


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "IDS706-china-tech-funding/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        path.write_bytes(response.read())


@recorded
def fetch_zenodo(run_id: str) -> dict:
    folder = run_dir(run_id) / "source"
    folder.mkdir(exist_ok=True)
    metadata = folder / "zenodo_record.json"
    download(ZENODO_API, metadata)
    record = json.loads(metadata.read_text(encoding="utf-8"))
    files = {}
    for item in record["files"]:
        short = item["key"].split(" - ")[-1]
        if short not in {"Dataset.csv", "Codebook.csv", "Documentation.csv"}:
            continue
        path = folder / short
        download(item["links"]["self"], path)
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        expected = item["checksum"].split(":")[-1]
        if md5 != expected:
            raise ValueError(f"Zenodo checksum mismatch: {short}")
        files[short] = {"path": str(path), "url": item["links"]["self"],
                        "sha256": sha256(path), "md5": md5, "bytes": path.stat().st_size}
    if set(files) != {"Dataset.csv", "Codebook.csv", "Documentation.csv"}:
        raise ValueError("Zenodo record does not contain all required CSV files")
    return {"record": ZENODO_API, "record_sha256": sha256(metadata), "files": files}


@recorded
def fetch_macro(run_id: str) -> dict:
    folder = run_dir(run_id) / "source"
    folder.mkdir(exist_ok=True)
    files = {}
    for name, code in INDICATORS.items():
        url = f"https://api.worldbank.org/v2/country/CHN/indicator/{code}?format=json&date=1998:2023&per_page=100"
        path = folder / f"{name}.json"
        download(url, path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if len(data) != 2 or data[0]["pages"] != 1:
            raise ValueError(f"Unexpected World Bank response for {name}")
        files[name] = {"path": str(path), "url": url, "sha256": sha256(path),
                       "rows": len(data[1]), "lastupdated": data[0].get("lastupdated")}
    return {"files": files}


def text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def amount_value(value: str) -> float | None:
    clean = text(value).removeprefix("$").replace(",", "")
    if clean == "":
        return None
    result = float(clean)
    if not 0 <= result < float("inf"):
        raise ValueError("invalid amount")
    return result or None


@recorded
def clean_investments(run_id: str, source: dict) -> dict:
    path = Path(source["files"]["Dataset.csv"]["path"])
    data = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if not REQUIRED.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(REQUIRED - set(data.columns))}")
    original_columns = list(data.columns)
    data["source_row"] = data.index + 2

    def in_scope(row):
        return text(row["Country"]) == "China" and text(row["Industry"]) == "Technology & Software"

    def group_fields(row):
        return (text(row["Investee Company"]).casefold(), text(row["Year"]),
                text(row["Round"]).casefold(), text(row["Type"]))

    # Check candidate peers in the full source before the country/industry filter.
    all_groups = {}
    for _, row in data.iterrows():
        all_groups.setdefault(group_fields(row), []).append(row)
    scope_conflicts = {}
    for fields, group in all_groups.items():
        inside = [row for row in group if in_scope(row)]
        outside = [row for row in group if not in_scope(row)]
        if not inside or not outside:
            continue
        different_country = any(text(row["Country"]) != "China" for row in outside)
        different_industry = any(text(row["Industry"]) != "Technology & Software" for row in outside)
        reason = "cross_scope_country_and_industry" if different_country and different_industry else (
            "cross_scope_country" if different_country else "cross_scope_industry")
        scope_conflicts[fields] = {
            "reason": reason,
            "selected_source_rows": [int(row["source_row"]) for row in inside],
            "peer_source_rows": [int(row["source_row"]) for row in outside],
            "peer_classifications": [{"source_row": int(row["source_row"]),
                                      "country": text(row["Country"]),
                                      "industry": text(row["Industry"]),
                                      "sector": text(row["Sector"]),
                                      "amount_usd_raw": text(row["Amount (USD)"])}
                                     for row in outside],
            "peers": outside,
        }

    selected = data.loc[data.apply(in_scope, axis=1)].copy()
    candidates = {}
    links = []
    for _, row in selected.iterrows():
        raw = {key: str(row[key]) for key in original_columns}
        name, round_name, kind = (text(row[key]) for key in ("Investee Company", "Round", "Type"))
        reason = None
        try:
            year = int(text(row["Year"]))
            if not 1999 <= year <= 2024:
                raise ValueError("year outside source coverage")
        except ValueError:
            year, reason = None, "invalid_year"
        if not name:
            reason = "missing_company"
        elif not round_name:
            reason = "missing_round"
        elif kind not in {"investment", "acquisition"}:
            reason = "invalid_type"
        try:
            amount = amount_value(row["Amount (USD)"])
        except ValueError:
            amount, reason = None, "invalid_amount"
        fields = group_fields(row)
        if reason is None and fields in scope_conflicts:
            reason = scope_conflicts[fields]["reason"]
        key = None
        if name and round_name and year is not None and kind in {"investment", "acquisition"}:
            key_fields = [name.casefold(), year, round_name.casefold(), kind]
            key = hashlib.sha256(json.dumps(key_fields, ensure_ascii=False).encode()).hexdigest()[:24]
        if reason is None:
            candidates.setdefault(key, []).append((row, amount, name, round_name, year, kind))
        links.append({"source_row": int(row["source_row"]), "candidate_key": key,
                      "status": "quarantined" if reason else "pending", "reason": reason or "",
                      "raw_record": json.dumps(raw, ensure_ascii=False)})

    for conflict in scope_conflicts.values():
        for peer in conflict["peers"]:
            name, year_text, round_name, kind = group_fields(peer)
            try:
                year = int(year_text)
                key = hashlib.sha256(json.dumps([name, year, round_name, kind],
                                                ensure_ascii=False).encode()).hexdigest()[:24]
            except ValueError:
                key = None
            links.append({"source_row": int(peer["source_row"]), "candidate_key": key,
                          "status": "scope_peer", "reason": conflict["reason"],
                          "raw_record": json.dumps({col: str(peer[col]) for col in original_columns},
                                                   ensure_ascii=False)})

    by_row = {item["source_row"]: item for item in links}
    events = []
    for key, group in candidates.items():
        amounts = {item[1] for item in group if item[1] is not None}
        sectors = {text(item[0]["Sector"]) for item in group}
        reason = "conflicting_amounts" if len(amounts) > 1 else (
            "conflicting_sectors" if len(sectors) > 1 else None)
        if reason:
            for row, *_ in group:
                by_row[int(row["source_row"])].update(status="quarantined", reason=reason)
            continue
        row, _, name, round_name, year, kind = group[0]
        events.append({"event_key": key, "company": name, "year": year,
                       "round": round_name, "deal_type": kind, "sector": next(iter(sectors)),
                       "amount_usd": next(iter(amounts)) if amounts else None,
                       "is_ai": next(iter(sectors)) == "Artificial Intelligence",
                       "source_rows": len(group)})
        for linked, *_ in group:
            by_row[int(linked["source_row"])].update(status="accepted", reason="")

    folder = run_dir(run_id)
    event_path, link_path = folder / "events.csv", folder / "source_rows.csv"
    pd.DataFrame(events).to_csv(event_path, index=False)
    pd.DataFrame(links).to_csv(link_path, index=False)
    scope_report = [{key: value for key, value in info.items() if key != "peers"}
                    for info in scope_conflicts.values()]
    save_json(folder / "scope_conflicts.json", scope_report)
    reasons = pd.Series([x["reason"] for x in links if x["status"] == "quarantined"]).value_counts().to_dict()
    report = {"source_rows": len(data), "selected_rows": len(selected),
              "selected_investment_rows": int((selected["Type"] == "investment").sum()),
              "selected_ai_rows": int((selected["Sector"] == "Artificial Intelligence").sum()),
              "accepted_events": len(events), "accepted_investment_events": sum(x["deal_type"] == "investment" for x in events),
              "accepted_ai_events": sum(x["is_ai"] for x in events),
              "disclosed_events": sum(x["amount_usd"] is not None for x in events),
              "accepted_source_rows": sum(x["status"] == "accepted" for x in links),
              "quarantined_rows": sum(x["status"] == "quarantined" for x in links),
              "scope_conflict_groups": len(scope_conflicts),
              "scope_peer_rows": sum(x["status"] == "scope_peer" for x in links),
              "source_link_rows": len(links),
              "quarantine_reasons": reasons}
    save_json(folder / "cleaning_report.json", report)
    return {"events": str(event_path), "source_rows": str(link_path), "report": report}


@recorded
def clean_macro(run_id: str, source: dict) -> dict:
    parts = []
    for name, file in source["files"].items():
        body = json.loads(Path(file["path"]).read_text(encoding="utf-8"))[1]
        part = pd.DataFrame([{"country": item["countryiso3code"], "year": int(item["date"]),
                              name: item["value"]} for item in body])
        if part.duplicated(["country", "year"]).any():
            raise ValueError(f"Duplicate country-year in {name}")
        if len(part) != 26 or set(part["year"]) != set(range(1998, 2024)) or set(part["country"]) != {"CHN"}:
            raise ValueError(f"Incomplete World Bank coverage for {name}")
        parts.append(part)
    merged = parts[0].merge(parts[1], on=["country", "year"], validate="one_to_one", how="outer", indicator=True)
    if (merged["_merge"] != "both").any() or merged.duplicated(["country", "year"]).any():
        raise ValueError("Macro country-year keys do not match uniquely")
    merged = merged.drop(columns="_merge")
    path = run_dir(run_id) / "macro.csv"
    merged.to_csv(path, index=False)
    return {"path": str(path), "rows": len(merged), "unique_country_year": True,
            "missing_values": {name: int(merged[name].isna().sum()) for name in INDICATORS}}


@recorded
def merge_data(run_id: str, investments: dict, macro: dict) -> dict:
    events = pd.read_csv(investments["events"])
    years = pd.read_csv(macro["path"])
    if years.duplicated(["country", "year"]).any():
        raise ValueError("Duplicate country-year before merge")
    events["macro_year"] = events["year"] - 1
    merged = events.merge(years, left_on="macro_year", right_on="year", how="left",
                          validate="many_to_one", suffixes=("", "_macro"), indicator=True)
    if len(merged) != len(events):
        raise ValueError("Merge changed event row count")
    covered = (merged["_merge"] == "both") & merged["gdp_growth"].notna() & merged["gdp_per_capita"].notna()
    report = {"before_rows": len(events), "after_rows": len(merged),
              "macro_rows": len(years), "matched_country_year_rows": int((merged["_merge"] == "both").sum()),
              "complete_macro_rows": int(covered.sum()),
              "complete_macro_coverage": float(covered.mean()),
              "missing_macro_event_keys": merged.loc[~covered, "event_key"].tolist()}
    merged = merged.drop(columns=["year_macro", "_merge"])
    path = run_dir(run_id) / "merged_events.csv"
    merged.to_csv(path, index=False)
    save_json(run_dir(run_id) / "merge_report.json", report)
    return {"path": str(path), "source_rows": investments["source_rows"],
            "macro": macro["path"], "report": report}


def connection():
    url = os.environ.get("FUNDING_DATABASE_URL")
    if not url:
        raise RuntimeError("FUNDING_DATABASE_URL is required")
    return psycopg2.connect(url)


@recorded
def load_postgres(run_id: str, merged: dict) -> dict:
    events = pd.read_csv(merged["path"])
    links = pd.read_csv(merged["source_rows"], keep_default_na=False)
    macro = pd.read_csv(merged["macro"])
    with connection() as db:
        with db.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS china_macro (
                country TEXT NOT NULL, year INTEGER NOT NULL, gdp_growth DOUBLE PRECISION,
                gdp_per_capita DOUBLE PRECISION, PRIMARY KEY (country, year))""")
            cur.execute("""CREATE TABLE IF NOT EXISTS funding_events (
                event_key TEXT PRIMARY KEY, company TEXT NOT NULL, year INTEGER NOT NULL,
                round TEXT NOT NULL, deal_type TEXT NOT NULL, sector TEXT NOT NULL,
                amount_usd NUMERIC, is_ai BOOLEAN NOT NULL, source_rows INTEGER NOT NULL,
                macro_year INTEGER NOT NULL, gdp_growth DOUBLE PRECISION,
                gdp_per_capita DOUBLE PRECISION)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS funding_source_rows (
                source_row INTEGER PRIMARY KEY, candidate_key TEXT, status TEXT NOT NULL,
                reason TEXT NOT NULL, raw_record JSONB NOT NULL)""")
            cur.execute("SELECT count(*) FROM funding_events")
            prior_event_rows = cur.fetchone()[0]
            cur.execute("TRUNCATE funding_source_rows, funding_events, china_macro")
            execute_values(cur, "INSERT INTO china_macro VALUES %s", [
                (r.country, int(r.year), None if pd.isna(r.gdp_growth) else float(r.gdp_growth),
                 None if pd.isna(r.gdp_per_capita) else float(r.gdp_per_capita))
                for r in macro.itertuples(index=False)])
            execute_values(cur, "INSERT INTO funding_events VALUES %s", [
                (r.event_key, r.company, int(r.year), r.round, r.deal_type, r.sector,
                 None if pd.isna(r.amount_usd) else float(r.amount_usd), bool(r.is_ai),
                 int(r.source_rows), int(r.macro_year),
                 None if pd.isna(r.gdp_growth) else float(r.gdp_growth),
                 None if pd.isna(r.gdp_per_capita) else float(r.gdp_per_capita))
                for r in events.itertuples(index=False)])
            execute_values(cur, "INSERT INTO funding_source_rows VALUES %s", [
                (int(r.source_row), r.candidate_key or None, r.status, r.reason,
                 Json(json.loads(r.raw_record))) for r in links.itertuples(index=False)])
            cur.execute("SELECT (SELECT count(*) FROM china_macro), (SELECT count(*) FROM funding_events), (SELECT count(*) FROM funding_source_rows)")
            readback = cur.fetchone()
            if readback != (len(macro), len(events), len(links)):
                raise ValueError(f"Database readback mismatch: {readback}")
            cur.execute("SELECT event_key FROM funding_events")
            if {row[0] for row in cur.fetchall()} != set(events["event_key"]):
                raise ValueError("Database event keys differ from merged file")
            cur.execute("SELECT source_row FROM funding_source_rows")
            if {row[0] for row in cur.fetchall()} != set(links["source_row"]):
                raise ValueError("Database source rows differ from cleaned file")
    report = {"macro_rows": readback[0], "event_rows": readback[1],
              "source_rows": readback[2], "prior_event_rows": prior_event_rows,
              "key_readback_match": True,
              "idempotence": "snapshot replace in one transaction"}
    save_json(run_dir(run_id) / "database_report.json", report)
    return report


@recorded
def analyze_from_sql(run_id: str, loaded: dict) -> dict:
    with connection() as db:
        with db.cursor() as cur:
            cur.execute("""SELECT deal_type, count(*), count(amount_usd),
                count(*) FILTER (WHERE is_ai),
                count(*) FILTER (WHERE amount_usd >= 100000000)
                FROM funding_events GROUP BY deal_type ORDER BY deal_type""")
            by_type = [{"type": kind, "events": total, "disclosed": disclosed,
                        "ai_events": ai, "at_least_100m": large}
                       for kind, total, disclosed, ai, large in cur.fetchall()]
            cur.execute("""SELECT year, count(*), count(amount_usd)
                FROM funding_events WHERE deal_type = 'investment'
                GROUP BY year ORDER BY year""")
            by_year = [{"year": year, "events": total, "disclosed": disclosed}
                       for year, total, disclosed in cur.fetchall()]
            cur.execute("SELECT count(*) FROM funding_events")
            total = cur.fetchone()[0]
    if total != loaded["event_rows"]:
        raise ValueError("SQL analysis readback differs from load count")
    result = {"source": "PostgreSQL SQL queries", "total_events": total,
              "by_type": by_type, "investment_by_year": by_year}
    save_json(run_dir(run_id) / "analysis.json", result)
    return result


def summarize_run(run_id: str) -> None:
    folder = run_dir(run_id)
    names = ["fetch_zenodo", "fetch_macro", "clean_investments", "clean_macro",
             "merge_data", "load_postgres", "analyze_from_sql"]
    steps = {name: json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
             for name in names}
    left, right = steps["fetch_zenodo"], steps["fetch_macro"]
    overlap = max(left["started_utc"], right["started_utc"]) < min(
        left["ended_utc"], right["ended_utc"])
    save_json(folder / "run_summary.json", {
        "run_id": run_id,
        "source_hashes": {
            name: item["sha256"]
            for name, item in left["result"]["files"].items()
        } | {
            name: item["sha256"]
            for name, item in right["result"]["files"].items()
        },
        "cleaning": steps["clean_investments"]["result"]["report"],
        "macro": {key: value for key, value in steps["clean_macro"]["result"].items() if key != "path"},
        "merge": steps["merge_data"]["result"]["report"],
        "database": steps["load_postgres"]["result"],
        "analysis": steps["analyze_from_sql"]["result"],
        "fetch_tasks_overlapped": overlap,
        "task_times": {name: {"started_utc": step["started_utc"],
                              "ended_utc": step["ended_utc"], "status": step["status"]}
                       for name, step in steps.items()},
    })
