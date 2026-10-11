"""
Predix — Churn Prediction Model
────────────────────────────────────────────────────────────────────────────────
Binary classifier that predicts whether a buyer will make no purchase in the
N days following a cutoff date (churned=1, retained=0). The target and the
point-in-time features are defined in `pipeline.feature_store.churn_dataset`.

Three estimators are compared:
  1. Logistic Regression (baseline, interpretable)
  2. Random Forest       (ensemble, handles non-linearity)
  3. Gradient Boosting   (boosted trees)

The champion (GB by default, see ChurnModelConfig) is registered in the MLflow
Model Registry. On the local sample the three estimators are within ~0.01
ROC-AUC of each other; see reports/churn_metrics.json.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.feature_store.churn_dataset import CHURN_FEATURE_COLS, build_churn_dataset

log = logging.getLogger(__name__)

FEATURE_COLS = CHURN_FEATURE_COLS


@dataclass
class ChurnModelConfig:
    experiment_name:   str  = "predix_churn_v1"
    registered_name:   str  = "predix_churn_model"
    test_size:         float = 0.20
    random_state:      int   = 42
    cv_folds:          int   = 5
    champion_model:    str   = "Gradient Boosting"
    gb_n_estimators:   int   = 200
    gb_max_depth:      int   = 4
    gb_learning_rate:  float = 0.05
    rf_n_estimators:   int   = 150
    rf_max_depth:      int   = 6


@dataclass
class ModelEvalResult:
    model_name: str
    auc_roc:    float
    auc_pr:     float
    f1:         float
    precision:  float
    recall:     float
    accuracy:   float
    cv_auc_mean: float
    cv_auc_std:  float
    fpr: list   = field(default_factory=list)
    tpr: list   = field(default_factory=list)


class ChurnModel:
    """Train, evaluate, and register churn prediction models."""

    def __init__(self, cfg: ChurnModelConfig | None = None):
        self.cfg     = cfg or ChurnModelConfig()
        self.models_: dict[str, Pipeline] = {}
        self.results_: list[ModelEvalResult] = []
        self.champion_: Pipeline | None = None

    # ── Build pipelines ────────────────────────────────────────────────────────
    def _build_pipelines(self) -> dict[str, Pipeline]:
        cfg = self.cfg
        return {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=cfg.random_state,
                    solver="lbfgs",
                )),
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    RandomForestClassifier(
                    n_estimators=cfg.rf_n_estimators,
                    max_depth=cfg.rf_max_depth,
                    class_weight="balanced",
                    random_state=cfg.random_state,
                    n_jobs=-1,
                )),
            ]),
            "Gradient Boosting": Pipeline([
                ("scaler", StandardScaler()),
                ("clf",    GradientBoostingClassifier(
                    n_estimators=cfg.gb_n_estimators,
                    max_depth=cfg.gb_max_depth,
                    learning_rate=cfg.gb_learning_rate,
                    subsample=0.85,
                    random_state=cfg.random_state,
                )),
            ]),
        }

    # ── Feature engineering ────────────────────────────────────────────────────
    @staticmethod
    def build_features(
        events: pd.DataFrame,
        as_of: pd.Timestamp | str,
        horizon_days: int = 30,
        lookback_days: int | None = None,
    ) -> pd.DataFrame:
        """Return point-in-time features and the `churned` label for eligible users."""
        return build_churn_dataset(
            events, as_of, horizon_days=horizon_days, lookback_days=lookback_days
        )

    # ── Train & evaluate ───────────────────────────────────────────────────────
    def fit(self, feat: pd.DataFrame) -> ChurnModel:
        X = feat[FEATURE_COLS].values
        y = feat["churned"].values

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y,
            test_size=self.cfg.test_size,
            random_state=self.cfg.random_state,
            stratify=y,
        )

        pipelines = self._build_pipelines()
        cv = StratifiedKFold(
            n_splits=self.cfg.cv_folds, shuffle=True, random_state=self.cfg.random_state
        )

        mlflow.set_experiment(self.cfg.experiment_name)

        for name, pipe in pipelines.items():
            with mlflow.start_run(run_name=name):
                pipe.fit(X_tr, y_tr)

                prob = pipe.predict_proba(X_te)[:, 1]
                pred = pipe.predict(X_te)
                fpr, tpr, _ = roc_curve(y_te, prob)

                cv_scores = cross_val_score(
                    pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1
                )

                result = ModelEvalResult(
                    model_name   = name,
                    auc_roc      = float(roc_auc_score(y_te, prob)),
                    auc_pr       = float(average_precision_score(y_te, prob)),
                    f1           = float(f1_score(y_te, pred)),
                    precision    = float(precision_score(y_te, pred, zero_division=0)),
                    recall       = float(recall_score(y_te, pred)),
                    accuracy     = float(accuracy_score(y_te, pred)),
                    cv_auc_mean  = float(cv_scores.mean()),
                    cv_auc_std   = float(cv_scores.std()),
                    fpr          = fpr.tolist(),
                    tpr          = tpr.tolist(),
                )
                self.results_.append(result)
                self.models_[name] = pipe

                # MLflow logging
                params = pipe.named_steps["clf"].get_params()
                mlflow.log_params({
                    k: v for k, v in params.items()
                    if isinstance(v, (int, float, str, bool))
                })
                mlflow.log_metrics({
                    "auc_roc":     result.auc_roc,
                    "auc_pr":      result.auc_pr,
                    "f1":          result.f1,
                    "precision":   result.precision,
                    "recall":      result.recall,
                    "accuracy":    result.accuracy,
                    "cv_auc_mean": result.cv_auc_mean,
                    "cv_auc_std":  result.cv_auc_std,
                })
                mlflow.set_tags({
                    "model_type": name,
                    "task":       "churn_prediction",
                    "dataset":    "ecommerce_oct_nov_2019",
                })
                mlflow.sklearn.log_model(pipe, artifact_path="model")

                log.info(
                    "[%s] AUC=%.4f | F1=%.4f | CV-AUC=%.4f±%.4f",
                    name, result.auc_roc, result.f1, result.cv_auc_mean, result.cv_auc_std,
                )

        self.champion_ = self.models_[self.cfg.champion_model]
        return self

    # ── Register champion ──────────────────────────────────────────────────────
    def register_champion(self) -> str:
        """Register the champion model in MLflow Model Registry."""
        if self.champion_ is None:
            raise RuntimeError("Call fit() before register_champion().")

        with mlflow.start_run(run_name=f"register_{self.cfg.champion_model}"):
            mlflow.sklearn.log_model(
                self.champion_,
                artifact_path="champion",
                registered_model_name=self.cfg.registered_name,
            )
            run_id = mlflow.active_run().info.run_id

        log.info(
            "Champion '%s' registered as '%s'",
            self.cfg.champion_model, self.cfg.registered_name,
        )
        return run_id

    # ── Predict ────────────────────────────────────────────────────────────────
    def predict_proba(self, feat: pd.DataFrame) -> np.ndarray:
        if self.champion_ is None:
            raise RuntimeError("Call fit() before predict_proba().")
        return self.champion_.predict_proba(feat[FEATURE_COLS].values)[:, 1]

    # ── Summary ────────────────────────────────────────────────────────────────
    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "model":     r.model_name,
                "auc_roc":   round(r.auc_roc,  4),
                "auc_pr":    round(r.auc_pr,   4),
                "f1":        round(r.f1,       4),
                "cv_auc":    f"{r.cv_auc_mean:.4f}±{r.cv_auc_std:.4f}",
            }
            for r in self.results_
        ])
