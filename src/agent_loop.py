"""A small LangGraph research loop over the cleaned PostgreSQL events."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import joblib
import pandas as pd
from langgraph.graph import END, START, StateGraph
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import pipeline


NUMERIC = ["year", "gdp_growth", "gdp_per_capita"]
CATEGORICAL = ["round", "sector"]
FEATURES = NUMERIC + CATEGORICAL
SPLIT_FILE = pipeline.ROOT / "artifacts" / "splits" / "funding_v1.json"
AGENT_RUNS = pipeline.ROOT / "artifacts" / "agent_runs"
TARGET_USD = 100_000_000

NODES = [
    {"id": "data_steward", "title": "Data Steward", "role": "SQL eligibility and frozen company split", "x": 30, "y": 100},
    {"id": "hypothesis_agent", "title": "Hypothesis Agent", "role": "Choose experiment from feedback", "x": 255, "y": 100},
    {"id": "model_engineer", "title": "Model Engineer", "role": "Fit on training data only", "x": 480, "y": 100},
    {"id": "evaluation_agent", "title": "Evaluation Agent", "role": "Score validation data", "x": 705, "y": 100},
    {"id": "judge_agent", "title": "Judge Agent", "role": "Compare validation and route", "x": 930, "y": 100},
    {"id": "final_test", "title": "Final Test", "role": "Open holdout once, then stop", "x": 1155, "y": 100},
]
EDGES = [
    {"source": "data_steward", "target": "hypothesis_agent"},
    {"source": "hypothesis_agent", "target": "model_engineer"},
    {"source": "model_engineer", "target": "evaluation_agent"},
    {"source": "evaluation_agent", "target": "judge_agent"},
    {"source": "judge_agent", "target": "hypothesis_agent", "route": "revise"},
    {"source": "judge_agent", "target": "final_test", "route": "finish"},
]


class LoopState(TypedDict, total=False):
    run_id: str
    source_run_id: str
    data_fingerprint: str
    split: dict
    train_path: str
    validation_path: str
    test_path: str
    cycle: int
    spec: dict
    model_path: str
    validation: dict
    experiments: list[dict]
    best_index: int
    feedback: dict
    feedback_history: list[dict]
    route: str
    stop_reason: str
    test_metrics: dict


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def agent_run_dir(run_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    folder = AGENT_RUNS / safe
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def event(run_id: str, node: str, phase: str, details: dict | None = None) -> None:
    item = {"at_utc": now(), "node": node, "phase": phase}
    if details:
        item["details"] = details
    with (agent_run_dir(run_id) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def graph_spec() -> dict:
    return {"nodes": NODES, "edges": EDGES, "entry": "data_steward", "exit": "final_test"}


def target_counts(frame: pd.DataFrame) -> dict:
    return {"rows": int(len(frame)), "positive": int(frame["target"].sum()),
            "negative": int((frame["target"] == 0).sum())}


def grouped_split(frame: pd.DataFrame, fingerprint: str) -> dict:
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    groups = frame["company"].str.strip().str.casefold()
    if SPLIT_FILE.exists():
        split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
        if split["data_fingerprint"] != fingerprint:
            raise ValueError("Frozen split does not match the current SQL snapshot; create a reviewed split version")
    else:
        split = None
        for seed in range(42, 142):
            outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
            train_val_idx, test_idx = next(outer.split(frame, frame["target"], groups))
            inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
            train_local, validation_local = next(inner.split(
                frame.iloc[train_val_idx], frame.iloc[train_val_idx]["target"], groups.iloc[train_val_idx]))
            partitions = {
                "train": frame.iloc[train_val_idx[train_local]],
                "validation": frame.iloc[train_val_idx[validation_local]],
                "test": frame.iloc[test_idx],
            }
            if all(part["target"].sum() >= 5 and (part["target"] == 0).sum() >= 5
                   for part in partitions.values()):
                split = {"version": 1, "seed": seed, "method": "GroupShuffleSplit by company",
                         "data_fingerprint": fingerprint,
                         "event_keys": {name: part["event_key"].tolist()
                                        for name, part in partitions.items()},
                         "counts": {name: target_counts(part)
                                    for name, part in partitions.items()}}
                pipeline.save_json(SPLIT_FILE, split)
                break
        if split is None:
            raise ValueError("No company-grouped split with both classes in each partition")

    key_sets = {name: set(keys) for name, keys in split["event_keys"].items()}
    if set.union(*key_sets.values()) != set(frame["event_key"]):
        raise ValueError("Frozen split does not cover the SQL events")
    if any(key_sets[a] & key_sets[b] for a, b in
           [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Frozen split has overlapping event keys")
    group_sets = {name: set(groups[frame["event_key"].isin(keys)])
                  for name, keys in key_sets.items()}
    if any(group_sets[a] & group_sets[b] for a, b in
           [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Company appears in more than one partition")
    split["company_group_counts"] = {name: len(values) for name, values in group_sets.items()}
    split["company_group_overlap"] = 0
    return split


def data_steward(state: LoopState) -> dict:
    sql = """SELECT event_key, company, year, gdp_growth, gdp_per_capita,
             round, sector, amount_usd::double precision FROM funding_events
             WHERE deal_type = 'investment' AND amount_usd IS NOT NULL AND amount_usd > 0
             ORDER BY event_key"""
    with pipeline.connection() as db:
        with db.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()
    columns = ["event_key", "company", *FEATURES, "amount_usd"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty or frame["event_key"].duplicated().any():
        raise ValueError("SQL modeling events are empty or not unique")
    if (frame["amount_usd"] <= 0).any():
        raise ValueError("Undisclosed amount entered the modeling subset")
    frame["target"] = (frame["amount_usd"] >= TARGET_USD).astype(int)

    source_path = pipeline.latest_run_summary()
    source_folder = source_path.parent
    source_summary = json.loads(source_path.read_text(encoding="utf-8"))
    source_events = pd.read_csv(source_folder / "merged_events.csv")
    source_keys = set(source_events.loc[
        (source_events["deal_type"] == "investment") & source_events["amount_usd"].notna(),
        "event_key"])
    if source_keys != set(frame["event_key"]):
        raise ValueError("PostgreSQL modeling rows differ from the latest Part 1 run")
    fingerprint_rows = sorted((r.event_key, r.company.strip().casefold(), float(r.amount_usd))
                              for r in frame.itertuples(index=False))
    fingerprint = hashlib.sha256(json.dumps(fingerprint_rows).encode()).hexdigest()
    split = grouped_split(frame, fingerprint)

    folder = agent_run_dir(state["run_id"])
    keep = ["event_key", "target", *FEATURES]
    paths = {}
    for name, keys in split["event_keys"].items():
        path = folder / f"{name}.csv"
        frame.loc[frame["event_key"].isin(keys), keep].to_csv(path, index=False)
        paths[name] = str(path)
    pipeline.save_json(folder / "split_manifest.json", split)
    pipeline.save_json(folder / "data_profile.json", {
        "source_run_id": source_summary["run_id"],
        "data_fingerprint": fingerprint, "eligible": target_counts(frame),
        "features": FEATURES, "excluded_from_features": ["amount_usd", "company", "event_key"],
        "target": "amount_usd >= 100000000", "split": split["counts"],
        "company_group_overlap": split["company_group_overlap"], "sql": sql,
    })
    return {"source_run_id": source_summary["run_id"],
            "data_fingerprint": fingerprint, "split": split,
            "train_path": paths["train"], "validation_path": paths["validation"],
            "test_path": paths["test"], "cycle": 0, "experiments": [],
            "feedback_history": []}


def hypothesis_agent(state: LoopState) -> dict:
    cycle = state["cycle"] + 1
    if cycle == 1:
        spec = {"family": "logistic_regression", "C": 1.0, "class_weight": "balanced",
                "reason": "Start with a regularized linear baseline"}
    elif cycle == 2:
        focus = state["feedback"]["focus"]
        spec = {"family": "random_forest", "n_estimators": 200,
                "max_depth": 10 if focus == "improve_recall" else 6,
                "min_samples_leaf": 2 if focus == "improve_recall" else 5,
                "class_weight": "balanced_subsample",
                "reason": f"Validation feedback requested {focus.replace('_', ' ')}"}
    else:
        raise ValueError("The simple loop allows two experiments")
    pipeline.save_json(agent_run_dir(state["run_id"]) / f"experiment_{cycle:02d}_plan.json", spec)
    return {"cycle": cycle, "spec": spec}


def make_model(spec: dict) -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
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
        raise ValueError(f"Unknown model family: {spec['family']}")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def model_engineer(state: LoopState) -> dict:
    train = pd.read_csv(state["train_path"])
    model = make_model(state["spec"])
    model.fit(train[FEATURES], train["target"])
    path = agent_run_dir(state["run_id"]) / f"model_{state['cycle']:02d}.joblib"
    joblib.dump(model, path)
    return {"model_path": str(path)}


def score(model: Pipeline, frame: pd.DataFrame) -> dict:
    probability = model.predict_proba(frame[FEATURES])[:, 1]
    prediction = (probability >= 0.5).astype(int)
    truth = frame["target"].to_numpy()
    return {"f1": float(f1_score(truth, prediction, zero_division=0)),
            "precision": float(precision_score(truth, prediction, zero_division=0)),
            "recall": float(recall_score(truth, prediction, zero_division=0)),
            "average_precision": float(average_precision_score(truth, probability)),
            "confusion_matrix": confusion_matrix(truth, prediction, labels=[0, 1]).tolist(),
            "threshold": 0.5, "rows": int(len(frame)),
            "positive": int(frame["target"].sum())}


def evaluation_agent(state: LoopState) -> dict:
    model = joblib.load(state["model_path"])
    validation = pd.read_csv(state["validation_path"])
    metrics = score(model, validation)
    pipeline.save_json(agent_run_dir(state["run_id"]) / f"validation_{state['cycle']:02d}.json", metrics)
    return {"validation": metrics}


def judge_agent(state: LoopState) -> dict:
    experiments = list(state.get("experiments", []))
    experiments.append({"cycle": state["cycle"], "spec": state["spec"],
                        "model_path": state["model_path"], "validation": state["validation"]})
    best_index = max(range(len(experiments)), key=lambda i: (
        experiments[i]["validation"]["f1"], experiments[i]["validation"]["average_precision"]))
    if state["cycle"] == 1:
        focus = "improve_recall" if state["validation"]["recall"] < state["validation"]["precision"] else "improve_precision"
        route, stop_reason = "revise", ""
    else:
        focus, route, stop_reason = "select_best_validation", "finish", "two_experiment_budget"
    feedback = {"cycle": state["cycle"], "focus": focus, "route": route,
                "best_cycle": experiments[best_index]["cycle"],
                "validation_f1": state["validation"]["f1"]}
    history = list(state.get("feedback_history", [])) + [feedback]
    pipeline.save_json(agent_run_dir(state["run_id"]) / "feedback.json", history)
    return {"experiments": experiments, "best_index": best_index,
            "feedback": feedback, "feedback_history": history,
            "route": route, "stop_reason": stop_reason}


def final_test(state: LoopState) -> dict:
    best = state["experiments"][state["best_index"]]
    model = joblib.load(best["model_path"])
    test = pd.read_csv(state["test_path"])
    metrics = score(model, test)
    folder = agent_run_dir(state["run_id"])
    pipeline.save_json(folder / "test_metrics.json", metrics)
    shutil.copyfile(best["model_path"], folder / "chosen_model.joblib")
    return {"test_metrics": metrics}


def tracked(name, function):
    def wrapped(state: LoopState):
        for attempt in (1, 2):
            event(state["run_id"], name, "started", {"attempt": attempt})
            try:
                result = function(state)
                details = {key: result[key] for key in ("cycle", "spec", "validation", "feedback",
                                                       "route", "stop_reason", "test_metrics") if key in result}
                details["attempt"] = attempt
                event(state["run_id"], name, "completed", details)
                return result
            except (OSError, ConnectionError, TimeoutError) as exc:
                event(state["run_id"], name, "retry" if attempt == 1 else "failed",
                      {"attempt": attempt, "error": str(exc)})
                if attempt == 2:
                    raise
            except Exception as exc:
                event(state["run_id"], name, "failed", {"attempt": attempt, "error": str(exc)})
                raise
    return wrapped


def build_graph():
    graph = StateGraph(LoopState)
    functions = {"data_steward": data_steward, "hypothesis_agent": hypothesis_agent,
                 "model_engineer": model_engineer, "evaluation_agent": evaluation_agent,
                 "judge_agent": judge_agent, "final_test": final_test}
    for name, function in functions.items():
        graph.add_node(name, tracked(name, function))
    graph.add_edge(START, "data_steward")
    for edge in EDGES:
        if "route" not in edge:
            graph.add_edge(edge["source"], edge["target"])
    branch_map = {edge["route"]: edge["target"] for edge in EDGES if "route" in edge}
    graph.add_conditional_edges("judge_agent", lambda state: state["route"], branch_map)
    graph.add_edge("final_test", END)
    return graph.compile()


def run_agent_loop(run_id: str | None = None) -> dict:
    run_id = run_id or "agent__" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    folder = agent_run_dir(run_id)
    pipeline.save_json(folder / "graph_spec.json", graph_spec())
    pipeline.save_json(folder / "status.json", {"run_id": run_id, "status": "running", "started_utc": now()})
    try:
        final = build_graph().invoke({"run_id": run_id}, config={"recursion_limit": 20})
        source_folder = next(p.parent for p in pipeline.RUNS.glob("*/run_summary.json")
                             if json.loads(p.read_text(encoding="utf-8"))["run_id"] == final["source_run_id"])
        source_summary = json.loads((source_folder / "run_summary.json").read_text(encoding="utf-8"))
        summary = {"run_id": run_id, "status": "success", "source_run_id": final["source_run_id"],
                   "source_hashes": source_summary["source_hashes"],
                   "data_fingerprint": final["data_fingerprint"],
                   "eligible": target_counts(pd.concat([pd.read_csv(final[f"{name}_path"])
                                                        for name in ("train", "validation", "test")])),
                   "features": FEATURES, "excluded_from_features": ["amount_usd", "company", "event_key"],
                   "split": final["split"]["counts"], "company_group_overlap": 0,
                   "experiments": final["experiments"],
                   "best_cycle": final["experiments"][final["best_index"]]["cycle"],
                   "stop_reason": final["stop_reason"], "test_metrics": final["test_metrics"],
                   "ended_utc": now()}
        pipeline.save_json(folder / "run_summary.json", summary)
        pipeline.save_json(folder / "status.json", {"run_id": run_id, "status": "success",
                                                       "started_utc": json.loads((folder / "status.json").read_text())["started_utc"],
                                                       "ended_utc": now()})
        return summary
    except Exception as exc:
        status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
        status.update(status="failed", ended_utc=now(), error=str(exc))
        pipeline.save_json(folder / "status.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the funding agent loop")
    parser.add_argument("--run-id", default=None)
    options = parser.parse_args()
    result = run_agent_loop(options.run_id)
    print(json.dumps({"run_id": result["run_id"], "best_cycle": result["best_cycle"],
                      "test_metrics": result["test_metrics"]}, indent=2))
