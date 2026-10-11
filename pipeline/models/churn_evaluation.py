"""
Predix — Churn Model Evaluation
────────────────────────────────────────────────────────────────────────────────
Compares learned classifiers against two simple baselines on the same
user-level holdout:

  * Recency score        days since last purchase, used directly as the score
                         (no training — the bar any model has to clear)
  * Logistic Regression  scaled features, class-balanced

Reported metrics are ROC-AUC and PR-AUC (average precision). With a churn
rate around 70%, PR-AUC must be read against the prevalence: a random score
has PR-AUC equal to the churn rate, not 0.5.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.feature_store.churn_dataset import CHURN_FEATURE_COLS

RECENCY_BASELINE = "Recency baseline"


@dataclass
class ModelResult:
    name: str
    roc_auc: float
    pr_auc: float
    cv_roc_auc_mean: float | None
    cv_roc_auc_std: float | None
    fpr: np.ndarray
    tpr: np.ndarray
    test_scores: np.ndarray


def build_classifiers(random_state: int = 42) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state),
        ),
        "Random Forest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", random_state=random_state, n_jobs=-1,
            ),
        ),
        "Gradient Boosting": make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.85, random_state=random_state,
            ),
        ),
    }


def evaluate_churn_models(
    dataset: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> tuple[list[ModelResult], dict[str, Pipeline], dict]:
    """Fit all models on a stratified user split and score them on the holdout.

    Returns the per-model results, the fitted pipelines and a split summary.
    """
    X = dataset[CHURN_FEATURE_COLS].to_numpy()
    y = dataset["churned"].to_numpy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results: list[ModelResult] = []

    recency_idx = CHURN_FEATURE_COLS.index("recency")
    rec_scores = X_te[:, recency_idx]
    fpr, tpr, _ = roc_curve(y_te, rec_scores)
    results.append(ModelResult(
        name=RECENCY_BASELINE,
        roc_auc=float(roc_auc_score(y_te, rec_scores)),
        pr_auc=float(average_precision_score(y_te, rec_scores)),
        cv_roc_auc_mean=None,
        cv_roc_auc_std=None,
        fpr=fpr, tpr=tpr, test_scores=rec_scores,
    ))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fitted: dict[str, Pipeline] = {}
    for name, pipe in build_classifiers(random_state).items():
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, prob)
        results.append(ModelResult(
            name=name,
            roc_auc=float(roc_auc_score(y_te, prob)),
            pr_auc=float(average_precision_score(y_te, prob)),
            cv_roc_auc_mean=float(cv_auc.mean()),
            cv_roc_auc_std=float(cv_auc.std()),
            fpr=fpr, tpr=tpr, test_scores=prob,
        ))
        fitted[name] = pipe

    split = {
        "n_users": int(len(y)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "churn_rate": float(y.mean()),
        "churn_rate_test": float(y_te.mean()),
    }
    return results, fitted, split


def results_table(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "model": r.name,
            "roc_auc": round(r.roc_auc, 4),
            "pr_auc": round(r.pr_auc, 4),
            "cv_roc_auc": (
                f"{r.cv_roc_auc_mean:.4f} +/- {r.cv_roc_auc_std:.4f}"
                if r.cv_roc_auc_mean is not None else "-"
            ),
        }
        for r in results
    ])
