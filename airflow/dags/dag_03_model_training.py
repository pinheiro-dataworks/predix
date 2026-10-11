"""
Predix — DAG 03: Model Training & Registration
────────────────────────────────────────────────────────────────────────────────
Schedule: @weekly (every Sunday 02:00 UTC)
Tasks:
  1. train_churn_model   — Fit 3 classifiers, log to MLflow
  2. train_upsell_model  — Compute opportunity + affinity scores
  3. evaluate_models     — Compare AUC; gate on threshold
  4. register_champion   — Promote best model to Production in MLflow Registry
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

DEFAULT_ARGS = {
    "owner":            "predix-data-science",
    "depends_on_past":  False,
    "email_on_failure": True,
    "retries":          1,
    "retry_delay":      timedelta(minutes=15),
    "execution_timeout": timedelta(hours=3),
}

MIN_AUC_THRESHOLD = 0.60  # gate: do not promote model below this AUC


def _events_path(context, root):
    """Event sample location; override with the `predix_events_path` Variable."""
    default = root / "dataset" / "sample_events.parquet"
    return context["var"]["value"].get("predix_events_path", str(default))


def _churn_window(context):
    """Churn cutoff and horizon from Airflow Variables (see churn_dataset.py)."""
    variables = context["var"]["value"]
    return {
        "as_of":        variables.get("churn_as_of", "2019-11-01"),
        "horizon_days": int(variables.get("churn_horizon_days", 30)),
    }


with DAG(
    dag_id            = "dag_03_model_training",
    description       = "Predix — Weekly model training and MLflow registration",
    schedule_interval = "0 2 * * 0",  # every Sunday at 02:00 UTC
    start_date        = datetime(2019, 11, 3),
    catchup           = False,
    max_active_runs   = 1,
    default_args      = DEFAULT_ARGS,
    tags              = ["predix", "ml", "training", "mlflow"],
) as dag:

    # ── Sensor: wait for latest features ───────────────────────────────────
    wait_features = ExternalTaskSensor(
        task_id          = "wait_feature_dag",
        external_dag_id  = "dag_02_feature_engineering",
        external_task_id = "validate_features",
        allowed_states   = ["success"],
        mode             = "reschedule",
        poke_interval    = 120,
        timeout          = 7200,
        execution_delta  = timedelta(days=0),
    )

    # ── Task 1: Train churn model ───────────────────────────────────────────
    def train_churn(**context):
        import logging
        import sys
        from pathlib import Path

        import mlflow

        log  = logging.getLogger(__name__)
        root = Path(context["var"]["value"].get("predix_home", "."))
        sys.path.insert(0, str(root))

        mlflow.set_tracking_uri(context["var"]["value"].get("mlflow_uri", str(root / "mlruns")))

        from pipeline.feature_store.feature_engineering import load_events
        from pipeline.models.churn_model import ChurnModel

        events = load_events(_events_path(context, root))
        feat   = ChurnModel.build_features(events, **_churn_window(context))
        model = ChurnModel()
        model.fit(feat)
        summary = model.summary()
        log.info("\n%s", summary.to_string(index=False))

        best_auc = summary["auc_roc"].max()
        context["ti"].xcom_push(key="best_auc", value=float(best_auc))
        log.info("Best AUC: %.4f", best_auc)

    train_churn_task = PythonOperator(
        task_id         = "train_churn_model",
        python_callable = train_churn,
    )

    # ── Task 2: Train upsell model ─────────────────────────────────────────
    def train_upsell(**context):
        import logging
        import sys
        from pathlib import Path

        import mlflow

        log  = logging.getLogger(__name__)
        root = Path(context["var"]["value"].get("predix_home", "."))
        sys.path.insert(0, str(root))

        mlflow.set_tracking_uri(context["var"]["value"].get("mlflow_uri", str(root / "mlruns")))

        from pipeline.feature_store.feature_engineering import build_rfm, load_events
        from pipeline.models.upsell_model import UpsellModel

        df = load_events(_events_path(context, root))

        rfm   = build_rfm(df)
        model = UpsellModel()
        model.fit(df, rfm)
        model.export_scores(str(root / "pipeline" / "scoring" / "user_scores.parquet"))
        log.info("Upsell training complete.")

    train_upsell_task = PythonOperator(
        task_id         = "train_upsell_model",
        python_callable = train_upsell,
    )

    # ── Task 3: AUC gate ───────────────────────────────────────────────────
    def evaluate_and_gate(**context):
        best_auc = context["ti"].xcom_pull(task_ids="train_churn_model", key="best_auc")
        if best_auc and float(best_auc) >= MIN_AUC_THRESHOLD:
            return "register_champion"
        return "skip_registration"

    auc_gate = BranchPythonOperator(
        task_id         = "evaluate_models",
        python_callable = evaluate_and_gate,
    )

    # ── Task 4a: Register champion ─────────────────────────────────────────
    def register(**context):
        import logging
        import sys
        from pathlib import Path

        import mlflow

        log  = logging.getLogger(__name__)
        root = Path(context["var"]["value"].get("predix_home", "."))
        sys.path.insert(0, str(root))

        mlflow.set_tracking_uri(context["var"]["value"].get("mlflow_uri", str(root / "mlruns")))

        from pipeline.feature_store.feature_engineering import load_events
        from pipeline.models.churn_model import ChurnModel

        events = load_events(_events_path(context, root))
        feat   = ChurnModel.build_features(events, **_churn_window(context))
        model = ChurnModel()
        model.fit(feat)
        run_id = model.register_champion()
        log.info("Registered champion. run_id=%s", run_id)

    register_task = PythonOperator(
        task_id         = "register_champion",
        python_callable = register,
    )

    # ── Task 4b: Skip ─────────────────────────────────────────────────────
    skip_task = BashOperator(
        task_id      = "skip_registration",
        bash_command = 'echo "AUC below threshold. Champion not promoted."',
    )

    # ── Dependencies ───────────────────────────────────────────────────────
    wait_features >> [train_churn_task, train_upsell_task]
    train_churn_task >> auc_gate >> [register_task, skip_task]
