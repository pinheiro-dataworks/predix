"""
Predix — Training Entrypoint
────────────────────────────────────────────────────────────────────────────────
Orchestrates the full local training loop:
  1. Load the user-level event sample (scripts/build_sample.py)
  2. Build the point-in-time churn dataset at `--as-of`
  3. Train and evaluate all classifiers, log them to MLflow
  4. Register the champion in the MLflow Model Registry
  5. Score a current snapshot (features as of the end of the data) and write
     the Parquet files served by the FastAPI app
  6. Compute RFM segments and upsell scores

Usage:
    python pipeline/models/train_and_register.py \
        --events ./dataset/sample_events.parquet \
        --mlflow-uri sqlite:///mlflow.db \
        --as-of 2019-11-01 --horizon-days 30
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.feature_store.churn_dataset import build_features  # noqa: E402
from pipeline.feature_store.feature_engineering import build_rfm, prepare_events  # noqa: E402
from pipeline.models.churn_model import ChurnModel, ChurnModelConfig  # noqa: E402
from pipeline.models.upsell_model import UpsellModel, UpsellModelConfig  # noqa: E402
from pipeline.scoring.batch_scoring import assign_churn_risk  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

SCORING_DIR = ROOT / "pipeline" / "scoring"
OUTPUT_DIR = SCORING_DIR / "output"


def score_snapshot(model: ChurnModel, events: pd.DataFrame) -> pd.DataFrame:
    """Score every buyer using all history up to the end of the data.

    Training needs a cutoff with a known future (`--as-of`); serving does not.
    The snapshot cutoff is the day after the last event, so features use the
    full history while staying on the same point-in-time code path.
    """
    snapshot_as_of = events["event_time"].max().normalize() + pd.Timedelta(days=1)
    observation = events[events["event_time"] < snapshot_as_of]
    feat = build_features(observation, snapshot_as_of).reset_index()
    feat["churn_score"] = model.predict_proba(feat)
    feat["churn_risk"] = assign_churn_risk(feat["churn_score"])
    feat["as_of"] = snapshot_as_of
    log.info("Snapshot scored as of %s: %d users.", snapshot_as_of.date(), len(feat))
    return feat


def main(args: argparse.Namespace) -> None:
    mlflow.set_tracking_uri(args.mlflow_uri)

    # ── Data ──────────────────────────────────────────────────────────────────
    log.info("Loading events from %s", args.events)
    events = prepare_events(pd.read_parquet(args.events))
    log.info("%d events | %d users", len(events), events["user_id"].nunique())

    # ── Churn model ───────────────────────────────────────────────────────────
    log.info("Starting churn model training…")
    churn_cfg = ChurnModelConfig(
        experiment_name  = "predix_churn_v1",
        gb_n_estimators  = 150,
        gb_max_depth     = 4,
        gb_learning_rate = 0.05,
    )
    churn_model = ChurnModel(cfg=churn_cfg)
    feat = ChurnModel.build_features(events, args.as_of, horizon_days=args.horizon_days)
    churn_model.fit(feat)

    log.info("\n%s", churn_model.summary().to_string(index=False))

    run_id = churn_model.register_champion()
    log.info("Champion registered. MLflow run_id=%s", run_id)

    # ── Churn scores (current snapshot) ──────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = score_snapshot(churn_model, events)
    churn_path = OUTPUT_DIR / "churn_scores.parquet"
    scores[["user_id", "churn_score", "churn_risk", "as_of"]].to_parquet(churn_path, index=False)
    log.info("Churn scores written to %s", churn_path)

    # ── RFM (served by the API and needed for upsell) ─────────────────────────
    rfm = build_rfm(events)
    rfm_path = SCORING_DIR / "rfm.parquet"
    rfm.to_parquet(rfm_path, index=False)
    log.info("RFM segments written to %s", rfm_path)

    # ── Upsell model ──────────────────────────────────────────────────────────
    log.info("Starting upsell model training…")
    upsell_cfg = UpsellModelConfig(experiment_name="predix_upsell_v1")
    upsell_model = UpsellModel(cfg=upsell_cfg)
    upsell_model.fit(events, rfm)

    upsell_path = OUTPUT_DIR / "upsell_scores.parquet"
    upsell_model.export_scores(str(upsell_path))
    log.info("Upsell scores written to %s", upsell_path)

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predix — Train & Register models")
    parser.add_argument("--events",       default="./dataset/sample_events.parquet", type=str)
    # A database backend is required: MLflow 3 rejects the ./mlruns file store.
    parser.add_argument("--mlflow-uri",   default="sqlite:///mlflow.db", type=str)
    parser.add_argument("--as-of",        default="2019-11-01",  type=str)
    parser.add_argument("--horizon-days", default=30,            type=int)
    main(parser.parse_args())
