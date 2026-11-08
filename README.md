# China tech funding: IDS706 Week 11

[![CI](https://github.com/USERZEROA/china-tech-funding/actions/workflows/ci.yml/badge.svg)](https://github.com/USERZEROA/china-tech-funding/actions/workflows/ci.yml)

This project studies financing events involving eight Chinese technology investors. It joins [Zenodo investment records](https://zenodo.org/records/19194192) for Chinese Technology & Software investees with [World Bank China GDP growth](https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.MKTP.KD.ZG?format=json&date=1998:2023&per_page=100) and [GDP per capita](https://api.worldbank.org/v2/country/CHN/indicator/NY.GDP.PCAP.CD?format=json&date=1998:2023&per_page=100). A financing event in year `t` uses macro data from `t-1`.

## Run Part 1

Start Docker Desktop with Linux containers. From this directory:

```powershell
docker compose up --build -d
docker compose exec airflow-webserver airflow dags trigger china_tech_funding
```

Open <http://localhost:8088> and sign in with `admin` / `admin`. The DAG is scheduled for Monday at 09:00 UTC. A manual run succeeded; an automatic scheduled run has not yet been observed. The two source branches ran in parallel. Each run saves inputs, SHA-256 hashes, cleaning and merge checks, database readback, SQL analysis, and task times under `artifacts/runs/<run_id>/`.

The pipeline retains original investor rows in PostgreSQL, groups plausible funding events, and quarantines conflicting candidates. Zero amounts are treated as undisclosed. The load replaces one validated snapshot in a transaction, so another run does not add duplicate rows.

The [verified run](artifacts/runs/manual__2026-11-08T04_19_45_00_00/run_summary.json) selected 1,532 source rows and accepted 1,428 candidate events. It checked related records before filtering, quarantined 61 selected rows with unresolved conflicts, and kept 36 outside-scope peers for review. All accepted events matched the prior-year macro data. The repeat load kept 1,428 events in PostgreSQL.

SQL analysis found 1,310 investment events; 365 were in the AI sector (27.9%). Of 519 investments with disclosed amounts, 85 raised at least $100 million (16.4%). The [Airflow run](screenshots/01_airflow_success.png) and [DAG graph](screenshots/02_airflow_graph.png) were captured from the live UI. See [EVIDENCE.md](EVIDENCE.md) for checks and remaining screenshots.

## Run Part 2

```powershell
docker compose up --build -d agent-ui
docker compose exec -T agent-ui python -m src.agent_loop
```

Open <http://localhost:8090> for the live graph and results. The [verified agent run](artifacts/agent_runs/agent__2026-11-08T04_31_51.037232_00_00/run_summary.json) used 519 disclosed investment events, including 85 at or above $100 million. Its frozen company-grouped split has 315 training, 100 validation, and 104 test rows, with no company overlap. A validation feedback loop compared logistic regression (F1 0.359) with random forest (F1 0.360) and selected the latter. The final test returned F1 0.351, precision 0.233, recall 0.714, AP 0.397, and confusion matrix `[[57, 33], [4, 10]]`. Precision is limited, so these are exploratory results. The [agent graph](screenshots/03_agent_graph.png) and [model metrics](screenshots/04_model_metrics.png) show the real run.

The optional adaptive workflow and three-way comparison remain pending.
