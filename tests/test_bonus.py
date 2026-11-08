import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from src import adaptive, bonus_common


class BonusRulesTest(unittest.TestCase):
    def test_planner_choices_build_the_execution_graph(self):
        plan = adaptive.fallback_plan()
        plan["first_agent"] = "forest_agent"
        plan["second_on_low_precision"] = "linear_agent"
        plan["second_on_low_recall"] = "linear_agent"
        checked = adaptive.validate_plan(plan)
        graph = adaptive.graph_plan(checked)
        self.assertEqual(graph["entry"], "forest_agent")
        self.assertIn({"source": "judge_agent", "target": "linear_agent",
                       "route": "low_precision"}, graph["edges"])
        self.assertNotEqual(checked["first_agent"], checked["second_on_low_precision"])
        self.assertNotEqual(graph["branch_actions"]["low_precision"]["model_spec"],
                            graph["branch_actions"]["low_recall"]["model_spec"])
        self.assertEqual(graph["branch_actions"]["quality_pass"]["target"], "final_test")

    def test_planner_rejects_unsafe_route_or_budget(self):
        plan = adaptive.fallback_plan()
        plan["second_on_low_precision"] = plan["first_agent"]
        with self.assertRaises(ValueError):
            adaptive.validate_plan(plan)
        plan = adaptive.fallback_plan()
        plan["retry_limit"] = 5
        with self.assertRaises(ValueError):
            adaptive.validate_plan(plan)

    def test_validation_feedback_changes_action_for_one_plan(self):
        plan = adaptive.fallback_plan()
        plan["first_agent"] = "forest_agent"
        plan["second_on_low_precision"] = "linear_agent"
        plan["second_on_low_recall"] = "linear_agent"
        plan = adaptive.validate_plan(plan)
        actions = []
        for metrics in ({"precision": 0.10, "recall": 0.90},
                        {"precision": 0.90, "recall": 0.10},
                        {"precision": 0.90, "recall": 0.90}):
            focus = adaptive.quality_focus(metrics, plan)
            target, reason = adaptive.next_step(1, focus, plan)
            spec = adaptive.make_spec(target, 2, focus) if target in adaptive.AGENTS else None
            actions.append((focus, target, reason, spec))
        self.assertEqual([item[0] for item in actions],
                         ["low_precision", "low_recall", "quality_pass"])
        self.assertNotEqual(actions[0][3], actions[1][3])
        self.assertEqual(actions[2][1:3], ("final_test", "quality_gate_met"))

    def test_planner_input_excludes_holdout_labels_and_total_positives(self):
        manifest = {"split": {"train": {"rows": 315, "positive": 56},
                              "validation": {"rows": 100, "positive": 15},
                              "test": {"rows": 104, "positive": 14}},
                    "eligible": {"rows": 519, "positive": 85},
                    "features": bonus_common.FEATURES}
        profile = adaptive.planner_profile(manifest)
        self.assertEqual(profile["training"], manifest["split"]["train"])
        self.assertEqual(profile["validation_rows"], 100)
        self.assertNotIn("test", str(profile))
        self.assertNotIn("85", str(profile))
        self.assertNotIn("14", str(profile))

    def test_free_planner_retries_one_invalid_response(self):
        class Response:
            def __init__(self, content):
                self.content = content

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"id": "example", "model": "free-model",
                                   "choices": [{"message": {"content": self.content}}]}).encode()

        valid = json.dumps(adaptive.fallback_plan())
        with patch.object(adaptive, "api_key", return_value="test-key"):
            with patch.object(adaptive.urllib.request, "urlopen",
                              side_effect=[Response("not JSON"), Response(valid)]):
                plan, provenance = adaptive.ask_planner({"training": {"rows": 10, "positive": 2}})
        self.assertEqual(plan, adaptive.fallback_plan())
        self.assertEqual(provenance["origin"], "openrouter_free")
        self.assertEqual(provenance["attempt"], 2)
        self.assertEqual(len(provenance["prior_errors"]), 1)

    def test_bonus_models_use_only_allowed_features(self):
        self.assertEqual(bonus_common.FEATURES,
                         ["year", "gdp_growth", "gdp_per_capita", "round", "sector"])
        self.assertFalse({"amount_usd", "company", "event_key"} & set(bonus_common.FEATURES))
        model = bonus_common.model_for(bonus_common.fixed_specs()[0])
        self.assertEqual(model.steps[0][0], "preprocess")

    def test_completed_final_step_is_idempotent_on_airflow_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(bonus_common, "ROOT", Path(temporary)):
                folder = bonus_common.folder_for("fixed_example")
                (folder / "run_summary.json").write_text('{"run_id":"fixed_example"}', encoding="utf-8")
                result = bonus_common.finish("fixed_example", "fixed_airflow", [], "done")
                self.assertEqual(result["run_id"], "fixed_example")
                self.assertFalse((folder / "test_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
