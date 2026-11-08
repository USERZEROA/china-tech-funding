"""Planner-led graph with validation routing and bounded recovery."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src import bonus_common, pipeline


AGENTS = {"linear_agent", "forest_agent"}


class AdaptiveState(TypedDict, total=False):
    run_id: str
    plan: dict
    cycle: int
    trials: list[dict]
    focus: str
    route: str
    routes: list[dict]
    summary: dict


def fallback_plan() -> dict:
    return {"first_agent": "linear_agent", "second_on_low_precision": "forest_agent",
            "second_on_low_recall": "forest_agent", "precision_gate": 0.35,
            "recall_gate": 0.50, "retry_limit": 1}


def validate_plan(plan: dict) -> dict:
    required = {"first_agent", "second_on_low_precision", "second_on_low_recall",
                "precision_gate", "recall_gate", "retry_limit"}
    if not isinstance(plan, dict) or not required.issubset(plan):
        raise ValueError("Planner omitted required graph choices")
    if plan["first_agent"] not in AGENTS:
        raise ValueError("First model agent is not allowed")
    for key in ("second_on_low_precision", "second_on_low_recall"):
        if plan[key] not in AGENTS or plan[key] == plan["first_agent"]:
            raise ValueError("Second agent must be a different allowed agent")
    for key in ("precision_gate", "recall_gate"):
        if type(plan[key]) not in (int, float) or not 0.15 <= plan[key] <= 0.80:
            raise ValueError("Quality gate is outside the allowed range")
    if type(plan["retry_limit"]) is not int or plan["retry_limit"] != 1:
        raise ValueError("The shared comparison budget permits one retry")
    return {key: plan[key] for key in required}


def api_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    secret_file = pipeline.ROOT / "private" / "openrouter.env"
    if secret_file.exists():
        for line in secret_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""


def ask_planner(profile: dict) -> tuple[dict, dict]:
    key = api_key()
    if not key:
        return fallback_plan(), {"origin": "rules_fallback", "reason": "API key unavailable"}
    prompt = (
        "You are the Planner Agent for a classroom funding model workflow. "
        "Return ONLY one JSON object with keys first_agent, second_on_low_precision, "
        "second_on_low_recall, precision_gate, recall_gate, retry_limit. "
        "Agent values are linear_agent or forest_agent. The second agents MUST differ "
        "from first_agent. Gates must be numbers from 0.15 to 0.80. retry_limit is 1. "
        "Choose a first model and a different second model for validation feedback. "
        "Use only this aggregate profile; no test outcomes are available: "
        + json.dumps(profile, sort_keys=True)
    )
    body = {"model": "openrouter/free", "messages": [
        {"role": "system", "content": "Return valid JSON only. Do not add prose."},
        {"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1000,
        "reasoning": {"enabled": False}}
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/USERZEROA/china-tech-funding",
                 "X-Title": "IDS706 funding workflow"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("Planner returned no text content")
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            raise ValueError("Planner returned no JSON object")
        plan = validate_plan(json.loads(match.group(0)))
        return plan, {"origin": "openrouter_free", "model": result.get("model"),
                      "response_id": result.get("id")}
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
            ValueError, json.JSONDecodeError) as exc:
        return fallback_plan(), {"origin": "rules_fallback", "reason": type(exc).__name__,
                                 "http_status": getattr(exc, "code", None)}


def graph_plan(plan: dict) -> dict:
    first = plan["first_agent"]
    return {"entry": first, "exit": "final_test", "budget": 2,
            "nodes": [
                {"id": "data_steward", "role": "Check SQL data and frozen split"},
                {"id": "planner_agent", "role": "Create this structured plan"},
                {"id": "linear_agent", "role": "Train regularized logistic regression"},
                {"id": "forest_agent", "role": "Train random forest"},
                {"id": "quality_agent", "role": "Score validation and name weakness"},
                {"id": "judge_agent", "role": "Route by plan and validation"},
                {"id": "final_test", "role": "Open holdout once and stop"}],
            "edges": [
                {"source": "data_steward", "target": "planner_agent"},
                {"source": "planner_agent", "target": first, "route": "plan.first_agent"},
                {"source": "linear_agent", "target": "quality_agent"},
                {"source": "forest_agent", "target": "quality_agent"},
                {"source": "quality_agent", "target": "judge_agent"},
                {"source": "judge_agent", "target": plan["second_on_low_precision"],
                 "route": "low_precision"},
                {"source": "judge_agent", "target": plan["second_on_low_recall"],
                 "route": "low_recall"},
                {"source": "judge_agent", "target": "final_test", "route": "budget_reached"}]}


def retry_step(run_id: str, node: str, work, limit: int = 1):
    for attempt in range(1, limit + 2):
        try:
            return work()
        except (OSError, ConnectionError, TimeoutError) as exc:
            phase = "retry" if attempt <= limit else "failed"
            bonus_common.record(run_id, node, phase,
                                {"attempt": attempt, "error_type": type(exc).__name__})
            if attempt > limit:
                raise


def make_spec(agent: str, cycle: int, focus: str) -> dict:
    if agent == "linear_agent":
        return {"family": "logistic_regression", "C": 1.0 if cycle == 1 else 0.3,
                "class_weight": "balanced"}
    return {"family": "random_forest", "n_estimators": 200,
            "max_depth": 10 if focus == "low_recall" else 6,
            "min_samples_leaf": 2 if focus == "low_recall" else 5,
            "class_weight": "balanced_subsample"}


def build_graph(plan: dict):
    graph = StateGraph(AdaptiveState)

    def model_node(agent):
        def work(state: AdaptiveState):
            cycle = state.get("cycle", 0) + 1
            spec = make_spec(agent, cycle, state.get("focus", "start"))
            trial = retry_step(state["run_id"], agent,
                               lambda: bonus_common.train_validate(state["run_id"], cycle, spec, agent),
                               state["plan"]["retry_limit"])
            return {"cycle": cycle, "trials": list(state.get("trials", [])) + [trial]}
        return work

    def quality(state: AdaptiveState):
        metrics = state["trials"][-1]["validation"]
        if metrics["precision"] < state["plan"]["precision_gate"]:
            focus = "low_precision"
        elif metrics["recall"] < state["plan"]["recall_gate"]:
            focus = "low_recall"
        else:
            focus = "quality_pass"
        bonus_common.record(state["run_id"], "quality_agent", "completed",
                            {"cycle": state["cycle"], "focus": focus, "validation": metrics})
        return {"focus": focus}

    def judge(state: AdaptiveState):
        if state["cycle"] >= 2:
            route = "final_test"
            reason = "two_experiment_budget"
        else:
            key = "second_on_low_recall" if state["focus"] == "low_recall" else "second_on_low_precision"
            route = state["plan"][key]
            reason = state["focus"]
        choice = {"cycle": state["cycle"], "focus": reason, "target": route}
        bonus_common.record(state["run_id"], "judge_agent", "routed", choice)
        return {"route": route, "routes": list(state.get("routes", [])) + [choice]}

    def final(state: AdaptiveState):
        summary = bonus_common.finish(state["run_id"], "adaptive_agent", state["trials"],
                                      "two_experiment_budget",
                                      {"plan": state["plan"], "routes": state["routes"]})
        return {"summary": summary}

    graph.add_node("linear_agent", model_node("linear_agent"))
    graph.add_node("forest_agent", model_node("forest_agent"))
    graph.add_node("quality_agent", quality)
    graph.add_node("judge_agent", judge)
    graph.add_node("final_test", final)
    graph.add_edge(START, plan["first_agent"])
    graph.add_edge("linear_agent", "quality_agent")
    graph.add_edge("forest_agent", "quality_agent")
    graph.add_edge("quality_agent", "judge_agent")
    graph.add_conditional_edges("judge_agent", lambda state: state["route"],
                                {"linear_agent": "linear_agent", "forest_agent": "forest_agent",
                                 "final_test": "final_test"})
    graph.add_edge("final_test", END)
    return graph.compile()


def run_adaptive(run_id: str | None = None) -> dict:
    run_id = run_id or "adaptive__" + datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    folder = bonus_common.folder_for(run_id)
    pipeline.save_json(folder / "status.json", {"run_id": run_id, "status": "running",
                                                 "started_utc": bonus_common.now()})
    try:
        profile = bonus_common.prepare(run_id)
        planner_input = {"eligible": profile["eligible"], "split": profile["split"],
                         "features": profile["features"], "company_group_overlap": 0,
                         "experiment_budget": 2}
        bonus_common.record(run_id, "planner_agent", "started", planner_input)
        plan, provenance = ask_planner(planner_input)
        plan = validate_plan(plan)
        graph = graph_plan(plan)
        pipeline.save_json(folder / "planner_input.json", planner_input)
        pipeline.save_json(folder / "graph_plan.json", graph)
        pipeline.save_json(folder / "planner_result.json", {"plan": plan, **provenance})
        bonus_common.record(run_id, "planner_agent", "completed",
                            {"plan": plan, "origin": provenance["origin"]})
        state = build_graph(plan).invoke({"run_id": run_id, "plan": plan, "cycle": 0,
                                          "trials": [], "routes": []},
                                         config={"recursion_limit": 20})
        summary = state["summary"]
        summary["planner"] = provenance
        pipeline.save_json(folder / "run_summary.json", summary)
        pipeline.save_json(folder / "status.json", {"run_id": run_id, "status": "success",
                                                     "ended_utc": bonus_common.now()})
        return summary
    except Exception as exc:
        pipeline.save_json(folder / "status.json", {"run_id": run_id, "status": "failed",
                                                     "ended_utc": bonus_common.now(),
                                                     "error": str(exc)})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id")
    options = parser.parse_args()
    result = run_adaptive(options.run_id)
    print(json.dumps({"run_id": result["run_id"], "planner": result["planner"],
                      "best_cycle": result["best_cycle"],
                      "test_metrics": result["test_metrics"]}, indent=2))
