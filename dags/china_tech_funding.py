"""Scheduled two-source funding pipeline for IDS706 Week 11."""

from datetime import datetime, timedelta

from airflow.decorators import dag, task

from src import pipeline


@dag(
    dag_id="china_tech_funding",
    start_date=datetime(2026, 11, 1),
    schedule="0 9 * * 1",
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=2)},
    tags=["IDS706", "China", "funding"],
)
def china_tech_funding():
    @task
    def get_investments(**context):
        return pipeline.fetch_zenodo(context["run_id"])

    @task
    def get_macro(**context):
        return pipeline.fetch_macro(context["run_id"])

    @task
    def clean_investments(source, **context):
        return pipeline.clean_investments(context["run_id"], source)

    @task
    def clean_macro(source, **context):
        return pipeline.clean_macro(context["run_id"], source)

    @task
    def merge_data(investments, macro, **context):
        return pipeline.merge_data(context["run_id"], investments, macro)

    @task
    def load_postgres(merged, **context):
        return pipeline.load_postgres(context["run_id"], merged)

    @task
    def analyze_from_sql(loaded, **context):
        return pipeline.analyze_from_sql(context["run_id"], loaded)

    investments = clean_investments(get_investments())
    macro = clean_macro(get_macro())
    analyze_from_sql(load_postgres(merge_data(investments, macro)))


china_tech_funding()
