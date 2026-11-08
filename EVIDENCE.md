# Evidence ledger

This file tracks assignment requirements and actual evidence. Run artifacts use `artifacts/runs/<run_id>/`.

Final Part 1 run: `manual__2026-11-08T04:19:45+00:00` (2026-11-07 23:19 EST). Its folder is `artifacts/runs/manual__2026-11-08T04_19_45_00_00/`. Earlier runs are preserved as development evidence. The 03:53:48 UTC run exposed a dollar-prefixed amount; the 04:19:16 UTC run first applied the cross-scope conflict check.

| Requirement | Implementation | Run ID / time | Result | Screenshot | Status |
| --- | --- | --- | --- | --- | --- |
| Scheduled Airflow DAG, ingestion through analysis | `dags/china_tech_funding.py` | Final Part 1 run, 23:19 EST | Seven tasks succeeded; DAG timezone verified as UTC; Monday 09:00 UTC schedule | `screenshots/01_airflow_success.png` | Complete; automatic scheduled dispatch not yet observed |
| Independent source paths and DAG graph | `dags/china_tech_funding.py` | Final Part 1 run, 23:19 EST | `run_summary.json`: fetch task intervals overlap | `screenshots/02_airflow_graph.png` | Complete |
| Zenodo Dataset, Codebook, Documentation and World Bank GDP indicators | `src/pipeline.py` | Final Part 1 run, 23:19 EST | `source/` and `run_summary.json`: five files with SHA-256; Zenodo MD5 verified | — | Complete |
| Cleaning, candidate events, source links, conflict quarantine | `src/pipeline.py` | Final Part 1 run, 23:19 EST | `cleaning_report.json`, `scope_conflicts.json`, `events.csv`, `source_rows.csv`: 6,741 raw; 1,532 selected; 1,471 accepted source rows become 1,428 candidate events; 61 selected rows quarantined; 36 outside-scope peers retained | — | Complete |
| Lagged macro merge, unique country-year, coverage and row counts | `src/pipeline.py` | Final Part 1 run, 23:19 EST | `merge_report.json`: 26 unique macro years; 1,428 before and after; 100% complete macro coverage | — | Complete |
| PostgreSQL load, idempotent snapshot, SQL readback and analysis | `src/pipeline.py`, `compose.yaml` | Final Part 1 run, 23:19 EST | `database_report.json`, `analysis.json`: 1,428 events before and after repeat load, 1,568 linked source/peer rows, matching SQL keys; 1,310 investment events, 519 disclosed | — | Complete |
| Agent training and feedback loop | Pending Part 2 | Pending | Pending | `screenshots/03_agent_graph.png`, `screenshots/04_model_metrics.png` | Pending |
| Adaptive graph plan, routing, retries, stop reason | Pending Bonus | Pending | Pending | `screenshots/05_adaptive_execution.png` | Pending |
| Evaluation cases and three-workflow comparison | Pending Bonus | Pending | Pending | `screenshots/06_workflow_comparison.png` | Pending |

Screenshots are complete only when the image file exists and shows a real run. The first four images satisfy the assignment's named screenshot requests; the last two support the Bonus.

The 61 quarantined selected rows comprise 38 cross-scope classification conflicts (37 industry, one country), 13 conflicting sectors, and 10 conflicting disclosed amounts. The 38 cross-scope rows belong to 33 candidate groups with 36 outside-scope peers. Of the 1,532 selected rows, 1,471 are accepted source links; 43 extra links collapse into the 1,428 candidate events. The candidate grouping is not a verified transaction ID. The selected rows include 1,411 investment rows and 427 AI-sector rows; these are row counts, not event counts. SQL analysis reports 365 AI investment events out of 1,310 (27.9%), and 85 investments at or above $100 million out of 519 with disclosed amounts (16.4%). No model metric is claimed in Part 1.

Verification: `docker compose config --quiet`, `python -m pip check`, and three `unittest` cleaning checks passed inside the Airflow container. The DAG timezone was read as UTC from the parsed Airflow DAG. The final two manual DAG runs succeeded with the corrected cross-scope rule; the two Part 1 screenshots were refreshed from the live local Airflow UI. Part 2 and Bonus screenshots remain pending.
