import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import agent_web


class AgentPageTest(unittest.TestCase):
    def test_latest_uses_start_time_and_selected_run_can_be_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "agent__old"
            new = root / "agent__new"
            bonus = root / "simple_bonus__later"
            for folder, started, status in (
                (old, "2026-11-08T05:00:00+00:00", "success"),
                (new, "2026-11-08T06:00:00+00:00", "starting"),
                (bonus, "2026-11-08T07:00:00+00:00", "success"),
            ):
                folder.mkdir()
                (folder / "status.json").write_text(json.dumps(
                    {"run_id": folder.name, "started_utc": started, "status": status}),
                    encoding="utf-8")
            with patch.object(agent_web, "AGENT_RUNS", root):
                client = agent_web.app.test_client()
                latest = client.get("/api/latest").json
                selected = client.get("/api/runs/agent__new").json
                self.assertEqual(latest["status"]["run_id"], "agent__new")
                self.assertEqual(selected["status"]["status"], "starting")
                self.assertIsNone(selected["summary"])
                self.assertEqual(client.get("/api/runs/simple_bonus__later").status_code, 404)


if __name__ == "__main__":
    unittest.main()
