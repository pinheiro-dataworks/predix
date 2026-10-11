"""
Predix — DAG 04: Daily Batch Scoring
────────────────────────────────────────────────────────────────────────────────
Schedule: @daily
Tasks:
  1. load_features   — Read Gold feature tables
  2. score_churn     — Apply champion churn model to full user base
  3. score_upsell    — Apply upsell scoring to full user base
  4. write_scores    — Write output to Snowflake scores table
  5. api_cache_warm  — Trigger FastAPI cache refresh
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.trigger_rule import TriggerRule

DEFAULT_ARGS = {
    "owner":            "predix-data-science",
    "depends_on_past":  False,
    "email_on_failure": True,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "execution_timeout": timedelta(hours=1),
}

with DAG(
    dag_id            = "dag_04_batch_scoring",
    description       = "Predix — Daily churn + upsell scoring for all users",
    schedule_interval = "@daily",
    start_date        = datetime(2019, 11, 1),
    end_date          = datetime(2019, 11, 30),
    catchup           = True,
    max_active_runs   = 1,
    default_args      = DEFAULT_ARGS,
    tags              = ["predix", "scoring", "batch", "api"],
) as dag:

    # ── Sensor ────────────────────────────────────────────────────────────
    wait_features = ExternalTaskSensor(
        task_id          = "wait_feature_dag",
        external_dag_id  = "dag_02_feature_engineering",
        external_task_id = "validate_features",
        allowed_states   = ["success"],
        mode             = "reschedule",
        poke_interval    = 60,
        timeout          = 3600,
    )

    # ── Task 1: Load & validate features ─────────────────────────────────
    def load_and_validate(**context):
        import logging
        from pathlib import Path

        import pandas as pd
        log  = logging.getLogger(__name__)
        root = Path(context["var"]["value"].get("predix_home", "."))

        rfm_path = root / "pipeline" / "scoring" / "rfm.parquet"
        if not rfm_path.exists():
            raise FileNotFoundError(f"Feature file not found: {rfm_path}")

        rfm = pd.read_parquet(rfm_path)
        assert len(rfm) > 100, "Feature table too small."
        log.info("Features loaded: %d users.", len(rfm))
        context["ti"].xcom_push(key="n_users", value=len(rfm))

    load_features = PythonOperator(
        task_id         = "load_features",
        python_callable = load_and_validate,
    )

    # ── Task 2: Score churn ───────────────────────────────────────────────
    score_churn = BashOperator(
        task_id      = "score_churn",
        bash_command = """
            python {{ var.value.predix_home }}/pipeline/scoring/batch_scoring.py \
              --task          churn \
              --churn-features {{ var.value.predix_home }}/pipeline/scoring/rfm.parquet \
              --output-dir    {{ var.value.predix_home }}/pipeline/scoring/output \
              --mlflow-uri    {{ var.value.mlflow_uri }}
        """,
        doc_md = "Loads champion churn model from MLflow Registry and scores all users.",
    )

    # ── Task 3: Score upsell ──────────────────────────────────────────────
    score_upsell = BashOperator(
        task_id      = "score_upsell",
        bash_command = """
            python {{ var.value.predix_home }}/pipeline/scoring/batch_scoring.py \
              --task          upsell \
              --upsell-scores {{ var.value.predix_home }}/pipeline/scoring/user_scores.parquet \
              --output-dir    {{ var.value.predix_home }}/pipeline/scoring/output \
              --mlflow-uri    {{ var.value.mlflow_uri }}
        """,
    )

    # ── Task 4: Write to Snowflake ────────────────────────────────────────
    def write_snowflake(**context):
        import logging
        log = logging.getLogger(__name__)
        # In production: use snowflake-connector-python to COPY INTO scores table
        log.info(
            "Writing scores to Snowflake table predix.gold.customer_scores "
            "for run_date=%s.", context["ds"]
        )
        # Example (requires snowflake-connector-python):
        # import snowflake.connector
        # conn = snowflake.connector.connect(...)
        # conn.cursor().execute("COPY INTO predix.gold.customer_scores FROM @stage/output/...")

    write_scores = PythonOperator(
        task_id         = "write_scores",
        python_callable = write_snowflake,
    )

    # ── Task 5: Warm API cache ────────────────────────────────────────────
    api_cache_warm = BashOperator(
        task_id      = "api_cache_warm",
        bash_command = """
            curl -s -X POST \
              {{ var.value.predix_api_url }}/admin/reload-scores \
              -H "Authorization: Bearer {{ var.value.predix_api_token }}" \
              || echo "API cache warm skipped (non-critical)."
        """,
        trigger_rule = TriggerRule.ALL_SUCCESS,
    )

    # ── Dependencies ──────────────────────────────────────────────────────
    wait_features >> load_features >> [score_churn, score_upsell] >> write_scores >> api_cache_warm
