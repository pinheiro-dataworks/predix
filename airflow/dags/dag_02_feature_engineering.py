"""
Predix — DAG 02: Feature Engineering
────────────────────────────────────────────────────────────────────────────────
Schedule: @daily (depends on dag_01 success)
Tasks:
  1. trigger_dbt_silver    — dbt run: Bronze → Silver transforms
  2. trigger_dbt_gold      — dbt run: Silver → Gold / feature tables
  3. build_python_features — Python: RFM + session features
  4. validate_features     — Row count + schema checks
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

DEFAULT_ARGS = {
    "owner":            "predix-data-science",
    "depends_on_past":  False,
    "email_on_failure": True,
    "retries":          1,
    "retry_delay":      timedelta(minutes=10),
    "execution_timeout": timedelta(hours=1, minutes=30),
}


def _events_path(context, root):
    """Event sample location; override with the `predix_events_path` Variable."""
    default = root / "dataset" / "sample_events.parquet"
    return context["var"]["value"].get("predix_events_path", str(default))


with DAG(
    dag_id            = "dag_02_feature_engineering",
    description       = "Predix — Build Gold feature tables from Silver events",
    schedule_interval = "@daily",
    start_date        = datetime(2019, 10, 1),
    end_date          = datetime(2019, 11, 30),
    catchup           = True,
    max_active_runs   = 1,
    default_args      = DEFAULT_ARGS,
    tags              = ["predix", "features", "dbt", "gold"],
) as dag:

    # ── Sensor: wait for ingestion ──────────────────────────────────────────
    wait_ingestion = ExternalTaskSensor(
        task_id             = "wait_ingestion_dag",
        external_dag_id     = "dag_01_ingestion",
        external_task_id    = "notify_success",
        allowed_states      = ["success"],
        mode                = "reschedule",
        poke_interval       = 60,
        timeout             = 3600,
    )

    # ── Task 1: dbt Silver ─────────────────────────────────────────────────
    dbt_silver = BashOperator(
        task_id      = "trigger_dbt_silver",
        bash_command = """
            cd {{ var.value.predix_home }}/dbt
            dbt run \
              --profiles-dir . \
              --select models/silver \
              --vars '{"run_date": "{{ ds }}"}'
        """,
        doc_md = "Runs dbt Silver models: events_cleaned, sessions, user_daily_agg.",
    )

    # ── Task 2: dbt Gold ───────────────────────────────────────────────────
    dbt_gold = BashOperator(
        task_id      = "trigger_dbt_gold",
        bash_command = """
            cd {{ var.value.predix_home }}/dbt
            dbt run \
              --profiles-dir . \
              --select models/gold \
              --vars '{"run_date": "{{ ds }}"}'
        """,
        doc_md = "Runs dbt Gold models: customer_rfm, churn_features, upsell_features.",
    )

    # ── Task 3: dbt tests ──────────────────────────────────────────────────
    dbt_test = BashOperator(
        task_id      = "dbt_test",
        bash_command = """
            cd {{ var.value.predix_home }}/dbt
            dbt test --profiles-dir . --select models/gold
        """,
    )

    # ── Task 4: Build Python features ──────────────────────────────────────
    def build_python_features(**context):
        import sys, logging
        from pathlib import Path

        log = logging.getLogger(__name__)
        root = Path(context["var"]["value"].get("predix_home", "."))
        sys.path.insert(0, str(root))

        from pipeline.feature_store.feature_engineering import (
            build_rfm,
            build_session_features,
            load_events,
        )

        out_dir  = root / "pipeline" / "scoring"
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info("Loading data for feature engineering…")
        df = load_events(_events_path(context, root))

        rfm = build_rfm(df)
        rfm.to_parquet(out_dir / "rfm.parquet", index=False)
        log.info("RFM saved: %d users.", len(rfm))

        sessions = build_session_features(df)
        sessions.to_parquet(out_dir / "sessions.parquet", index=False)
        log.info("Sessions saved: %d sessions.", len(sessions))

    build_features = PythonOperator(
        task_id         = "build_python_features",
        python_callable = build_python_features,
    )

    # ── Task 5: Validate ───────────────────────────────────────────────────
    def validate_features(**context):
        from pathlib import Path
        import pandas as pd, logging
        log = logging.getLogger(__name__)

        root    = Path(context["var"]["value"].get("predix_home", "."))
        rfm     = pd.read_parquet(root / "pipeline" / "scoring" / "rfm.parquet")
        assert len(rfm) > 100,  "RFM feature table too small — check ingestion."
        assert "segment" in rfm.columns, "segment column missing from RFM."
        log.info("Feature validation passed: %d RFM users.", len(rfm))

    validate_features_task = PythonOperator(
        task_id         = "validate_features",
        python_callable = validate_features,
    )

    # ── Dependencies ───────────────────────────────────────────────────────
    wait_ingestion >> dbt_silver >> dbt_gold >> dbt_test >> build_features >> validate_features_task
