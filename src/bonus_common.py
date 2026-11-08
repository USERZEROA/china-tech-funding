"""Shared data and scoring code for the three workflow comparison."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import pipeline


ROOT = pipeline.ROOT / "artifacts" / "bonus_runs"
SPLIT_FILE = pipeline.ROOT / "artifacts" / "splits" / "funding_v1.json"
NUMERIC = ["year", "gdp_growth", "gdp_per_capita"]
CATEGORICAL = ["round", "sector"]
FEATURES = NUMERIC + CATEGORICAL


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def folder_for(run_id: str) -> Path:
    folder = ROOT / re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def record(run_id: str, node: str, phase: str, details: dict | None = None) -> None:
    row = {"at_utc": now(), "node": node, "phase": phase, "details": details or {}}
    path = folder_for(run_id) / "events.jsonl"
    descriptor = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    finally:
        os.close(descriptor)


def prepare(run_id: str) -> dict:
    """Read the Part 1 SQL table and apply the existing frozen split."""
    folder = folder_for(run_id)
    record(run_id, "data_steward", "started")
    sql = """SELECT event_key, company, year, gdp_growth, gdp_per_capita,
             round, sector, amount_usd::double precision FROM funding_events
             WHERE deal_type = 'investment' AND amount_usd IS NOT NULL AND amount_usd > 0
             ORDER BY event_key"""
    with pipeline.connection() as db:
        with db.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
    frame = pd.DataFrame(rows, columns=["event_key", "company", *FEATURES, "amount_usd"])
    if frame.empty or frame.event_key.duplicated().any() or (frame.amount_usd <= 0).any():
        raise ValueError("Invalid disclosed investment events from SQL")
    frame["target"] = (frame.amount_usd >= 100_000_000).astype(int)
    fingerprint_rows = sorted((r.event_key, r.company.strip().casefold(), float(r.amount_usd))
                              for r in frame.itertuples(index=False))
    fingerprint = hashlib.sha256(json.dumps(fingerprint_rows).encode()).hexdigest()
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split["data_fingerprint"] != fingerprint:
        raise ValueError("Frozen company split differs from the SQL snapshot")
    keys = {name: set(split["event_keys"][name]) for name in ("train", "validation", "test")}
    if set.union(*keys.values()) != set(frame.event_key) or any(keys[a] & keys[b] for a, b in
            (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Frozen split keys do not partition SQL events")
    groups = {}
    counts = {}
    for name, part_keys in keys.items():
        part = frame.loc[frame.event_key.isin(part_keys)]
        groups[name] = set(part.company.str.strip().str.casefold())
        counts[name] = {"rows": len(part), "positive": int(part.target.sum())}
        part[["event_key", "target", *FEATURES]].to_csv(folder / f"{name}.csv", index=False)
    if any(groups[a] & groups[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Company group crosses split partitions")
    source_path = pipeline.latest_run_summary()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_events = pd.read_csv(source_path.parent / "merged_events.csv")
    source_keys = set(source_events.loc[
        (source_events.deal_type == "investment") & source_events.amount_usd.notna(), "event_key"])
    if source_keys != set(frame.event_key):
        raise ValueError("SQL events do not match the Part 1 source run")
    manifest = {"run_id": run_id, "source_run_id": source["run_id"],
                "source_hashes": source["source_hashes"], "data_fingerprint": fingerprint,
                "split_version": split["version"], "split": counts,
                "eligible": {"rows": len(frame), "positive": int(frame.target.sum())},
                "company_group_overlap": 0, "features": FEATURES,
                "excluded_from_features": ["amount_usd", "company", "event_key"], "sql": sql}
    pipeline.save_json(folder / "data_manifest.json", manifest)
    record(run_id, "data_steward", "completed", {"eligible": manifest["eligible"], "split": counts})
    return manifest


def fixed_specs() -> list[dict]:
    return [
        {"family": "logistic_regression", "C": 1.0, "class_weight": "balanced"},
        {"family": "random_forest", "n_estimators": 200, "max_depth": 6,
         "min_samples_leaf": 5, "class_weight": "balanced_subsample"},
    ]


def model_for(spec: dict) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    preprocess = ColumnTransformer([("numeric", numeric, NUMERIC),
                                    ("categorical", categorical, CATEGORICAL)])
    if spec["family"] == "logistic_regression":
        estimator = LogisticRegression(C=spec["C"], class_weight=spec["class_weight"],
                                       max_iter=2000, random_state=42)
    elif spec["family"] == "random_forest":
        estimator = RandomForestClassifier(n_estimators=spec["n_estimators"],
                                           max_depth=spec["max_depth"],
                                           min_samples_leaf=spec["min_samples_leaf"],
                                           class_weight=spec["class_weight"],
                                           random_state=42, n_jobs=1)
    else:
        raise ValueError("Model family is not allowed")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def score(model: Pipeline, frame: pd.DataFrame) -> dict:
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    predicted = (probability >= 0.5).astype(int)
    truth = frame.target.to_numpy()
    return {"f1": float(f1_score(truth, predicted, zero_division=0)),
            "precision": float(precision_score(truth, predicted, zero_division=0)),
            "recall": float(recall_score(truth, predicted, zero_division=0)),
            "average_precision": float(average_precision_score(truth, probability)),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
            "threshold": 0.5, "rows": len(frame), "positive": int(frame.target.sum())}


def train_validate(run_id: str, cycle: int, spec: dict, agent: str) -> dict:
    folder = folder_for(run_id)
    record(run_id, agent, "started", {"cycle": cycle, "spec": spec})
    train = pd.read_csv(folder / "train.csv")
    validation = pd.read_csv(folder / "validation.csv")
    model = model_for(spec)
    model.fit(train[FEATURES], train.target)
    model_path = folder / f"model_{cycle:02d}.joblib"
    joblib.dump(model, model_path)
    metrics = score(model, validation)
    trial = {"cycle": cycle, "agent": agent, "spec": spec,
             "model_path": str(model_path), "validation": metrics}
    pipeline.save_json(folder / f"validation_{cycle:02d}.json", trial)
    record(run_id, agent, "completed", {"cycle": cycle, "validation": metrics})
    return trial


def finish(run_id: str, mode: str, trials: list[dict], stop_reason: str,
           extra: dict | None = None) -> dict:
    folder = folder_for(run_id)
    summary_path = folder / "run_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    best = max(trials, key=lambda x: (x["validation"]["f1"],
                                      x["validation"]["average_precision"]))
    record(run_id, "judge_agent", "selected", {"best_cycle": best["cycle"],
                                                 "validation_f1": best["validation"]["f1"]})
    metrics_path = folder / "test_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        record(run_id, "final_test", "resumed", {"best_cycle": best["cycle"]})
    else:
        record(run_id, "final_test", "started", {"best_cycle": best["cycle"]})
        model = joblib.load(best["model_path"])
        metrics = score(model, pd.read_csv(folder / "test.csv"))
        pipeline.save_json(metrics_path, metrics)
        record(run_id, "final_test", "completed", {"test_metrics": metrics})
    manifest = json.loads((folder / "data_manifest.json").read_text(encoding="utf-8"))
    summary = {"run_id": run_id, "mode": mode, "status": "success",
               "source_run_id": manifest["source_run_id"],
               "source_hashes": manifest["source_hashes"],
               "data_fingerprint": manifest["data_fingerprint"],
               "split": manifest["split"], "eligible": manifest["eligible"],
               "company_group_overlap": 0, "experiment_budget": 2,
               "experiments": trials, "best_cycle": best["cycle"],
               "test_metrics": metrics, "stop_reason": stop_reason,
               "ended_utc": now(), **(extra or {})}
    pipeline.save_json(summary_path, summary)
    return summary
