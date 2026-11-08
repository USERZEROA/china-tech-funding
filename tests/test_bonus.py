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

    def test_planner_rejects_unsafe_route_or_budget(self):
        plan = adaptive.fallback_plan()
        plan["second_on_low_precision"] = plan["first_agent"]
        with self.assertRaises(ValueError):
            adaptive.validate_plan(plan)
        plan = adaptive.fallback_plan()
        plan["retry_limit"] = 5
        with self.assertRaises(ValueError):
            adaptive.validate_plan(plan)

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
