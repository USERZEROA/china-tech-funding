import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import pipeline


class CleaningRulesTest(unittest.TestCase):
    def test_zero_is_undisclosed(self):
        self.assertIsNone(pipeline.amount_value("0"))
        self.assertEqual(pipeline.amount_value("100,000,000"), 100000000)
        self.assertEqual(pipeline.amount_value("$68,645,500"), 68645500)

    def test_conflicting_candidate_is_quarantined_and_links_remain(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "Dataset.csv"
            rows = []
            for amount, investor in [("0", "Alibaba"), ("10,000,000", "Tencent"),
                                     ("20,000,000", "Baidu")]:
                rows.append({"Investee Company": "Example AI", "Investing Company": investor,
                             "Round Investors": investor, "Round": "A", "Amount (USD)": amount,
                             "Year": "2020", "Type": "investment", "Sector": "Artificial Intelligence",
                             "Industry": "Technology & Software", "Country": "China"})
            pd.DataFrame(rows).to_csv(data, index=False)
            with patch.object(pipeline, "RUNS", Path(folder) / "runs"):
                result = pipeline.clean_investments("test", {"files": {"Dataset.csv": {"path": str(data)}}})
            self.assertEqual(result["report"]["accepted_events"], 0)
            self.assertEqual(result["report"]["quarantined_rows"], 3)
            self.assertEqual(result["report"]["quarantine_reasons"], {"conflicting_amounts": 3})
            links = pd.read_csv(result["source_rows"])
            self.assertEqual(len(links), 3)
            self.assertEqual(json.loads(links.iloc[1]["raw_record"])["Investing Company"], "Tencent")

    def test_outside_industry_peer_is_seen_before_filter(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "Dataset.csv"
            common = {"Investee Company": "Example AI", "Round Investors": "Alibaba",
                      "Round": "A", "Amount (USD)": "100,000,000", "Year": "2020",
                      "Type": "investment", "Country": "China"}
            rows = [
                {**common, "Investing Company": "Alibaba", "Sector": "Artificial Intelligence",
                 "Industry": "Technology & Software"},
                {**common, "Investing Company": "Tencent", "Sector": "Manufacturing",
                 "Industry": "Resource & Production Industries"},
            ]
            pd.DataFrame(rows).to_csv(data, index=False)
            with patch.object(pipeline, "RUNS", Path(folder) / "runs"):
                result = pipeline.clean_investments("test", {"files": {"Dataset.csv": {"path": str(data)}}})
            self.assertEqual(result["report"]["selected_rows"], 1)
            self.assertEqual(result["report"]["accepted_events"], 0)
            self.assertEqual(result["report"]["quarantine_reasons"], {"cross_scope_industry": 1})
            self.assertEqual(result["report"]["scope_peer_rows"], 1)
            self.assertEqual(len(pd.read_csv(result["source_rows"])), 2)


if __name__ == "__main__":
    unittest.main()
