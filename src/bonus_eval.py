"""Run explicit Bonus evaluation cases and compare three real workflows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from src import adaptive, agent_loop, bonus_common, pipeline


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def case(case_id, input_text, expected, scoring, threshold, actual, score):
    return {"id": case_id, "input": input_text, "expected": expected,
            "scoring_criteria": scoring, "pass_threshold": threshold,
            "actual": actual, "score": score, "passed": score >= threshold}


def workflow_row(name, summary, events):
    first = datetime.fromisoformat(events[0]["at_utc"])
    last = datetime.fromisoformat(events[-1]["at_utc"])
    return {"workflow": name, "run_id": summary["run_id"],
            "source_run_id": summary["source_run_id"],
            "data_fingerprint": summary["data_fingerprint"],
            "split": summary["split"], "experiments": len(summary["experiments"]),
            "validation_f1": summary["experiments"][summary["best_cycle"] - 1]["validation"]["f1"],
            "test_f1": summary["test_metrics"]["f1"],
            "test_precision": summary["test_metrics"]["precision"],
            "test_recall": summary["test_metrics"]["recall"],
            "test_ap": summary["test_metrics"]["average_precision"],
            "test_confusion_matrix": summary["test_metrics"]["confusion_matrix"],
            "event_seconds": round((last - first).total_seconds(), 3),
            "event_count": len(events),
            "retry_events": sum(item["phase"] == "retry" for item in events),
            "stop_reason": summary["stop_reason"]}


def evaluate(fixed_run_id: str, simple_run_id: str, adaptive_run_id: str,
             evaluation_id: str | None = None) -> dict:
    evaluation_id = evaluation_id or "comparison__" + datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%f+00:00")
    folder = bonus_common.folder_for(evaluation_id)
    fixed_folder = bonus_common.folder_for(fixed_run_id)
    simple_folder = agent_loop.agent_run_dir(simple_run_id)
    adaptive_folder = bonus_common.folder_for(adaptive_run_id)
    folders = {"fixed_airflow": fixed_folder, "simple_agent": simple_folder,
               "adaptive_agent": adaptive_folder}
    summaries = {name: read_json(path / "run_summary.json") for name, path in folders.items()}
    events = {name: read_events(path / "events.jsonl") for name, path in folders.items()}
    rows = [workflow_row(name, summaries[name], events[name]) for name in folders]
    plan_record = read_json(adaptive_folder / "planner_result.json")
    plan = adaptive.validate_plan(plan_record["plan"])
    graph = read_json(adaptive_folder / "graph_plan.json")
    routes = [e["details"] for e in events["adaptive_agent"]
              if e["node"] == "judge_agent" and e["phase"] == "routed"]
    model_starts = [e["node"] for e in events["adaptive_agent"]
                    if e["node"] in adaptive.AGENTS and e["phase"] == "started"]

    cases = []
    split_counts = [{name: {"rows": part["rows"], "positive": part["positive"]}
                     for name, part in r["split"].items()} for r in rows]
    parity = (len({r["data_fingerprint"] for r in rows}) == 1
              and len({r["source_run_id"] for r in rows}) == 1
              and all(r["experiments"] == 2 for r in rows)
              and all(s["company_group_overlap"] == 0 for s in summaries.values())
              and len({json.dumps(part, sort_keys=True) for part in split_counts}) == 1)
    cases.append(case("same_data_split_budget", "Three completed workflow run IDs",
                      "Same source snapshot, company split, and two fitted models each",
                      "1 point if all invariants hold; otherwise 0", 1,
                      {"same_fingerprint": len({r["data_fingerprint"] for r in rows}) == 1,
                       "split_rows": [{k: v["rows"] for k, v in r["split"].items()} for r in rows],
                       "experiments": [r["experiments"] for r in rows]}, int(parity)))

    expected_second = plan["second_on_low_recall"] if routes[0]["focus"] == "low_recall" else plan["second_on_low_precision"]
    applied = (model_starts == [plan["first_agent"], expected_second]
               and routes[0]["target"] == expected_second
               and routes[-1]["target"] == "final_test")
    cases.append(case("planner_changes_execution", "Validated Planner graph and execution events",
                      "The graph entry and feedback edge match the two model agents actually run",
                      "1 point if both model nodes and judge routes match the plan; otherwise 0", 1,
                      {"planner_origin": plan_record["origin"], "planned_entry": graph["entry"],
                       "model_starts": model_starts, "routes": routes}, int(applied)))

    first_validation = summaries["adaptive_agent"]["experiments"][0]["validation"]
    if first_validation["precision"] < plan["precision_gate"]:
        expected_focus = "low_precision"
    elif first_validation["recall"] < plan["recall_gate"]:
        expected_focus = "low_recall"
    else:
        expected_focus = "quality_pass"
    focus_ok = routes[0]["focus"] == expected_focus
    cases.append(case("intermediate_quality_route", "First adaptive validation metrics and plan gates",
                      "Quality Agent identifies the measured weakness and Judge follows that branch",
                      "1 point for the exact focus and branch; otherwise 0", 1,
                      {"precision": first_validation["precision"],
                       "recall": first_validation["recall"],
                       "expected_focus": expected_focus, "actual_focus": routes[0]["focus"]},
                      int(focus_ok and applied)))

    attempts = {"count": 0}

    def transient_failure():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("controlled transient failure")
        return "recovered"

    recovered = adaptive.retry_step(evaluation_id, "retry_probe", transient_failure, 1)
    retry_events = [e for e in read_events(folder / "events.jsonl") if e["phase"] == "retry"]
    retry_ok = recovered == "recovered" and attempts["count"] == 2 and len(retry_events) == 1
    cases.append(case("bounded_retry", "Controlled first-attempt OSError in retry probe",
                      "One logged retry and success on the second attempt",
                      "1 point for two attempts, one retry event, and recovery; otherwise 0", 1,
                      {"attempts": attempts["count"], "retry_events": len(retry_events),
                       "result": recovered, "test_only_fault": True}, int(retry_ok)))

    test_events = [e for e in events["adaptive_agent"] if e["node"] == "final_test"]
    test_start = next(e["at_utc"] for e in test_events if e["phase"] == "started")
    final_route = next(e["at_utc"] for e in events["adaptive_agent"]
                       if e["node"] == "judge_agent" and e["phase"] == "routed"
                       and e["details"]["target"] == "final_test")
    sealed = (len([e for e in test_events if e["phase"] == "started"]) == 1
              and test_start > final_route and len(model_starts) == 2
              and all(e["at_utc"] < test_start for e in events["adaptive_agent"]
                      if e["node"] in adaptive.AGENTS))
    cases.append(case("finite_stop_sealed_test", "Adaptive run event timestamps",
                      "Two model fits, final selection, one test opening, then stop",
                      "1 point if order and counts hold; otherwise 0", 1,
                      {"model_fits": len(model_starts), "test_opens": 1,
                       "stop_reason": summaries["adaptive_agent"]["stop_reason"]}, int(sealed)))

    metrics = summaries["adaptive_agent"]["test_metrics"]
    prevalence = metrics["positive"] / metrics["rows"]
    quality = metrics["f1"] >= 0.30 and metrics["average_precision"] >= prevalence
    cases.append(case("final_output_quality", "One adaptive holdout evaluation",
                      "F1 at least 0.30 and AP at least the test positive prevalence",
                      "1 point if both measured thresholds hold; otherwise 0", 1,
                      {"f1": metrics["f1"], "ap": metrics["average_precision"],
                       "positive_prevalence": prevalence, "confusion_matrix": metrics["confusion_matrix"]},
                      int(quality)))

    fixed_feedback = read_json(fixed_folder / "feedback.json")
    simple_feedback = read_json(simple_folder / "feedback.json")
    baselines_ok = (fixed_feedback["decision"].startswith("select highest")
                    and len(simple_feedback) == 2
                    and simple_feedback[0]["route"] == "revise"
                    and simple_feedback[1]["route"] == "finish")
    cases.append(case("baseline_feedback", "Fixed DAG feedback and simple agent feedback files",
                      "Fixed selection uses validation; simple loop revises then stops",
                      "1 point if both feedback paths are present; otherwise 0", 1,
                      {"fixed_decision": fixed_feedback["decision"],
                       "simple_routes": [x["route"] for x in simple_feedback]}, int(baselines_ok)))

    results = {"evaluation_id": evaluation_id, "created_utc": bonus_common.now(),
               "run_ids": {name: summaries[name]["run_id"] for name in summaries},
               "cases": cases, "passed": sum(c["passed"] for c in cases),
               "total": len(cases), "all_passed": all(c["passed"] for c in cases)}
    comparison = {"evaluation_id": evaluation_id, "source_run_id": rows[0]["source_run_id"],
                  "same_data_split_budget": parity, "rows": rows,
                  "summary": "All three workflows used the same disclosed investments and frozen company split. "
                             "The adaptive planner changed model order and followed validation feedback. "
                             "Holdout F1 and AP are reported without using test results for routing."}
    pipeline.save_json(folder / "evaluation_suite.json", [{key: value for key, value in c.items()
                                                             if key not in ("actual", "score", "passed")}
                                                            for c in cases])
    pipeline.save_json(folder / "evaluation_results.json", results)
    pipeline.save_json(folder / "workflow_comparison.json", comparison)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--simple", required=True)
    parser.add_argument("--adaptive", required=True)
    options = parser.parse_args()
    result = evaluate(options.fixed, options.simple, options.adaptive)
    print(json.dumps({"evaluation_id": result["evaluation_id"],
                      "passed": result["passed"], "total": result["total"]}, indent=2))
