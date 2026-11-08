"""A fixed Airflow model workflow for the Bonus comparison."""

from datetime import datetime, timedelta, timezone

from airflow.decorators import dag, task

from src import bonus_common, pipeline


@dag(
    dag_id="funding_fixed_baseline",
    start_date=datetime(2026, 11, 1, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=15)},
    tags=["IDS706", "Bonus", "baseline"],
)
def funding_fixed_baseline():
    @task
    def prepare_data(**context):
        run_id = context["run_id"]
        bonus_common.record(run_id, "airflow", "started", {"workflow": "fixed"})
        return bonus_common.prepare(run_id)["run_id"]

    @task
    def linear_trial(prepared_run):
        return bonus_common.train_validate(prepared_run, 1, bonus_common.fixed_specs()[0], "linear_agent")

    @task
    def forest_trial(prepared_run):
        return bonus_common.train_validate(prepared_run, 2, bonus_common.fixed_specs()[1], "forest_agent")

    @task
    def select_and_test(linear, forest, **context):
        run_id = context["run_id"]
        trials = [linear, forest]
        feedback = {"cycle_1_f1": linear["validation"]["f1"],
                    "cycle_2_f1": forest["validation"]["f1"],
                    "decision": "select highest validation F1; AP breaks ties"}
        pipeline.save_json(bonus_common.folder_for(run_id) / "feedback.json", feedback)
        result = bonus_common.finish(run_id, "fixed_airflow", trials,
                                     "predefined_steps_completed", {"feedback": feedback})
        bonus_common.record(run_id, "airflow", "completed", {"status": result["status"]})
        return result["run_id"]

    prepared = prepare_data()
    linear = linear_trial(prepared)
    forest = forest_trial(prepared)
    linear >> forest
    select_and_test(linear, forest)


funding_fixed_baseline()
