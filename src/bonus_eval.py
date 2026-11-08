"""Run explicit Bonus evaluation cases and compare three real workflows."""

from __future__ import annotations

import argparse
import hashlib
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
              and rows[0]["experiments"] == rows[1]["experiments"] == 2
              and 1 <= rows[2]["experiments"] <= 2
              and all(s["company_group_overlap"] == 0 for s in summaries.values())
              and len({json.dumps(part, sort_keys=True) for part in split_counts}) == 1)
    cases.append(case("same_data_split_budget", "Three completed workflow run IDs",
                      "Same source snapshot and company split; each workflow permits at most two fits",
                      "1 point if all invariants hold; otherwise 0", 1,
                      {"same_fingerprint": len({r["data_fingerprint"] for r in rows}) == 1,
                       "split_rows": [{k: v["rows"] for k, v in r["split"].items()} for r in rows],
                       "experiments": [r["experiments"] for r in rows]}, int(parity)))

    partition_hashes = {part: {name: hashlib.sha256((path / f"{part}.csv").read_bytes()).hexdigest()
                               for name, path in folders.items()}
                        for part in ("train", "validation", "test")}
    same_files = all(len(set(hashes.values())) == 1 for hashes in partition_hashes.values())
    cases.append(case("same_partition_files", "Three workflows' train, validation, and test CSV files",
                      "Each matching partition has identical SHA-256 bytes across workflows",
                      "1 point if all three partitions match byte for byte; otherwise 0", 1,
                      {"sha256": partition_hashes}, int(same_files)))

    first_validation = summaries["adaptive_agent"]["experiments"][0]["validation"]
    expected_focus = adaptive.quality_focus(first_validation, plan)
    expected_route, expected_stop = adaptive.next_step(1, expected_focus, plan)
    expected_models = [plan["first_agent"]]
    if expected_route in adaptive.AGENTS:
        expected_models.append(expected_route)
    applied = (model_starts == expected_models
               and routes[0]["target"] == expected_route
               and routes[-1]["target"] == "final_test"
               and graph["entry"] == plan["first_agent"])
    cases.append(case("planner_changes_execution", "Validated Planner graph and execution events",
                      "The graph entry, feedback route, and executed model nodes match the plan",
                      "1 point if all executed nodes and routes match; otherwise 0", 1,
                      {"planner_origin": plan_record["origin"], "planned_entry": graph["entry"],
                       "model_starts": model_starts, "routes": routes}, int(applied)))

    actions = graph.get("branch_actions", {})
    graph_actions_ok = (set(actions) == {"low_precision", "low_recall", "quality_pass"}
                        and actions["low_precision"]["model_spec"] != actions["low_recall"]["model_spec"]
                        and actions["quality_pass"]["target"] == "final_test"
                        and (expected_route == "final_test" or
                             summaries["adaptive_agent"]["experiments"][1]["spec"]
                             == actions[expected_focus]["model_spec"]))
    cases.append(case("saved_graph_branch_actions", "Saved graph plan and executed second experiment",
                      "Different feedback branches carry different model settings; executed settings match the graph",
                      "1 point if graph actions differ and the executed branch matches; otherwise 0", 1,
                      {"branch_actions": actions,
                       "executed_second_spec": summaries["adaptive_agent"]["experiments"][1]["spec"]
                       if len(summaries["adaptive_agent"]["experiments"]) > 1 else None},
                      int(graph_actions_ok)))

    focus_ok = (routes[0]["focus"] == expected_focus
                and routes[0]["target"] == expected_route)
    cases.append(case("intermediate_quality_route", "First adaptive validation metrics and plan gates",
                      "Quality Agent identifies the measured weakness and Judge follows that branch",
                      "1 point for the exact focus and branch; otherwise 0", 1,
                      {"precision": first_validation["precision"],
                       "recall": first_validation["recall"],
                       "expected_focus": expected_focus, "actual_focus": routes[0]["focus"],
                       "expected_target": expected_route, "actual_target": routes[0]["target"]},
                      int(focus_ok and applied)))

    planner_input = read_json(adaptive_folder / "planner_input.json")
    sealed_input = (set(planner_input) == {"training", "validation_rows", "features", "experiment_budget"}
                    and set(planner_input["training"]) == {"rows", "positive"})
    cases.append(case("planner_input_sealed", "Saved Planner API input",
                      "Only training labels, validation size, features, and fit budget are sent",
                      "1 point if the input has only the allowed fields; otherwise 0", 1,
                      {"fields": sorted(planner_input), "training": planner_input["training"],
                       "validation_rows": planner_input["validation_rows"]}, int(sealed_input)))

    api_used = plan_record.get("origin") == "openrouter_free" and bool(plan_record.get("response_id"))
    cases.append(case("api_planner_used", "Planner response metadata",
                      "A free API model returned the structured plan used for this run",
                      "1 point for an API response ID and API origin; otherwise 0", 1,
                      {"origin": plan_record.get("origin"), "model": plan_record.get("model"),
                       "response_id": plan_record.get("response_id")}, int(api_used)))

    controlled = [
        ("low_precision", {"precision": 0.10, "recall": 0.90}, plan["second_on_low_precision"]),
        ("low_recall", {"precision": 0.90, "recall": 0.10}, plan["second_on_low_recall"]),
        ("quality_pass", {"precision": 0.90, "recall": 0.90}, "final_test"),
    ]
    controlled_actions = []
    for expected_name, example_metrics, expected_agent in controlled:
        actual_focus = adaptive.quality_focus(example_metrics, plan)
        target, reason = adaptive.next_step(1, actual_focus, plan)
        spec = adaptive.make_spec(target, 2, actual_focus) if target in adaptive.AGENTS else None
        if expected_name == "quality_pass":
            expected_settings = None
            settings_ok = spec is None and reason == "quality_gate_met"
        elif expected_agent == "linear_agent":
            expected_settings = ({"C": 0.3, "class_weight": None} if expected_name == "low_precision"
                                 else {"C": 3.0, "class_weight": "balanced"})
            settings_ok = all(spec[key] == value for key, value in expected_settings.items())
        else:
            expected_settings = ({"max_depth": 6, "min_samples_leaf": 5, "class_weight": None}
                                 if expected_name == "low_precision" else
                                 {"max_depth": 10, "min_samples_leaf": 2,
                                  "class_weight": "balanced_subsample"})
            settings_ok = all(spec[key] == value for key, value in expected_settings.items())
        actual = {"focus": actual_focus, "next_agent": target, "model_spec": spec,
                  "stop_reason": reason if target == "final_test" else None,
                  "controlled_input": True}
        controlled_actions.append(actual)
        cases.append(case("route_" + expected_name,
                          {"plan": plan, "validation_metrics": example_metrics},
                          {"focus": expected_name, "next_agent": expected_agent,
                           "settings": expected_settings},
                          "1 point if the real routing functions choose the expected action; otherwise 0",
                          1, actual,
                          int(actual_focus == expected_name and target == expected_agent and settings_ok)))
    distinct_actions = (controlled_actions[0]["model_spec"] != controlled_actions[1]["model_spec"]
                        and controlled_actions[2]["next_agent"] == "final_test")
    cases.append(case("feedback_changes_action", "Three controlled validation outcomes under one saved plan",
                      "Low precision and low recall change model settings; quality pass stops early",
                      "1 point if the three actual actions differ as specified; otherwise 0", 1,
                      {"actions": controlled_actions}, int(distinct_actions)))

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
    actual_fits = len(summaries["adaptive_agent"]["experiments"])
    expected_final_reason = expected_stop if expected_route == "final_test" else "two_experiment_budget"
    sealed = (len([e for e in test_events if e["phase"] == "started"]) == 1
              and test_start > final_route and 1 <= actual_fits <= 2
              and len(model_starts) == actual_fits
              and summaries["adaptive_agent"]["stop_reason"] == expected_final_reason
              and all(e["at_utc"] < test_start for e in events["adaptive_agent"]
                      if e["node"] in adaptive.AGENTS))
    cases.append(case("finite_stop_sealed_test", "Adaptive run event timestamps",
                      "At most two model fits, final selection, one test opening, then stop",
                      "1 point if order and counts hold; otherwise 0", 1,
                      {"model_fits": actual_fits, "test_opens": 1,
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
                             f"Fixed and simple workflows used two fits; adaptive used {actual_fits} of two allowed fits. "
                             "The adaptive planner and validation feedback determined the execution route. "
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
