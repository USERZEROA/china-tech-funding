import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import agent_loop


class SplitRulesTest(unittest.TestCase):
    def test_split_is_frozen_and_company_groups_do_not_cross(self):
        rows = []
        for company in range(90):
            for event in range(2):
                rows.append({"event_key": f"{company}-{event}", "company": f"Company {company}",
                             "target": int(company % 4 == 0)})
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(agent_loop, "SPLIT_FILE", Path(folder) / "split.json"):
                first = agent_loop.grouped_split(frame, "fixed-source")
                second = agent_loop.grouped_split(frame, "fixed-source")
                self.assertEqual(first["event_keys"], second["event_keys"])
                self.assertEqual(first["company_group_overlap"], 0)
                with self.assertRaisesRegex(ValueError, "Frozen split"):
                    agent_loop.grouped_split(frame, "different-source")

    def test_model_features_exclude_label_inputs_and_ids(self):
        self.assertFalse({"amount_usd", "company", "event_key", "source_rows"} & set(agent_loop.FEATURES))


if __name__ == "__main__":
    unittest.main()
