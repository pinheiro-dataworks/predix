#!/usr/bin/env python3
"""
Predix - evaluate_churn.py
Builds the point-in-time churn dataset from the local sample, compares the
models against the recency and logistic-regression baselines, and writes
reports/churn_metrics.json.

Usage (from project root, after scripts/build_sample.py):
    python scripts/evaluate_churn.py
    python scripts/evaluate_churn.py --as-of 2019-11-01 --horizon-days 14
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.feature_store.churn_dataset import build_churn_dataset  # noqa: E402
from pipeline.feature_store.feature_engineering import prepare_events  # noqa: E402
from pipeline.models.churn_evaluation import evaluate_churn_models, results_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate churn models on the local sample.")
    parser.add_argument("--events", type=Path, default=ROOT / "dataset" / "sample_events.parquet")
    parser.add_argument("--as-of", default="2019-11-01")
    parser.add_argument("--horizon-days", type=int, default=30)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "churn_metrics.json")
    args = parser.parse_args()

    started = time.time()
    events = prepare_events(pd.read_parquet(args.events))
    dataset = build_churn_dataset(
        events, args.as_of, horizon_days=args.horizon_days, lookback_days=args.lookback_days
    )
    results, _, split = evaluate_churn_models(dataset)

    print(results_table(results).to_string(index=False))
    print(f"\nEligible users: {split['n_users']:,} | churn rate: {split['churn_rate']:.1%}")

    report = {
        "target": {
            "as_of": args.as_of,
            "horizon_days": args.horizon_days,
            "lookback_days": args.lookback_days,
            "definition": "eligible = >=1 purchase before as_of; "
                          "churned = no purchase in [as_of, as_of + horizon_days)",
        },
        "data": {
            "source": args.events.name,
            "events": int(len(events)),
            "users": int(events["user_id"].nunique()),
            "first_event": str(events["event_time"].min()),
            "last_event": str(events["event_time"].max()),
        },
        "split": {"test_size": 0.2, "random_state": 42, "stratified_by": "churned", **split},
        "models": [
            {
                "model": r.name,
                "roc_auc": round(r.roc_auc, 4),
                "pr_auc": round(r.pr_auc, 4),
                "cv_roc_auc_mean": None if r.cv_roc_auc_mean is None else round(r.cv_roc_auc_mean, 4),
                "cv_roc_auc_std": None if r.cv_roc_auc_std is None else round(r.cv_roc_auc_std, 4),
            }
            for r in results
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"OK  Report written to {args.out.relative_to(ROOT)} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
