"""
Predix — Upsell / Cross-sell Scoring Model
────────────────────────────────────────────────────────────────────────────────
Generates three complementary scores for each user×category pair:
  1. Opportunity Score   — gap between views and conversions (unmet demand)
  2. Affinity Score      — co-purchase probability from the affinity matrix
  3. RFM Upsell Index    — user value weight × category opportunity

The combined score drives personalised CRM campaigns, push notifications,
and product recommendation widgets in real time via the FastAPI scoring API.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

SCORE_VERSION = "1.0"


@dataclass
class UpsellModelConfig:
    experiment_name:  str   = "predix_upsell_v1"
    registered_name:  str   = "predix_upsell_model"
    opportunity_alpha: float = 8.0    # views - purchases × alpha
    top_k_categories: int   = 5      # max categories per user recommendation
    min_views:        int   = 10     # minimum views to compute gap score


class UpsellModel:
    """Compute opportunity + affinity scores and register artefacts."""

    def __init__(self, cfg: UpsellModelConfig | None = None):
        self.cfg              = cfg or UpsellModelConfig()
        self.affinity_matrix_ : pd.DataFrame | None = None
        self.category_scores_ : pd.DataFrame | None = None
        self.user_scores_     : pd.DataFrame | None = None

    # ── Category-level opportunity ─────────────────────────────────────────────
    def compute_category_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute aggregate opportunity score per category."""
        views     = df[df["event_type"] == "view"].groupby("category").size().rename("views")
        purchases = (
            df[df["event_type"] == "purchase"].groupby("category").size().rename("purchases")
        )

        scores = pd.concat([views, purchases], axis=1).fillna(0)
        scores = scores[scores["views"] >= self.cfg.min_views]

        scores["gap_score"] = (
            scores["views"] - scores["purchases"] * self.cfg.opportunity_alpha
        ).clip(lower=0)
        scores["conversion_rate"] = scores["purchases"] / scores["views"].replace(0, np.nan)

        scaler = MinMaxScaler()
        scores["opportunity_score"] = scaler.fit_transform(
            scores[["gap_score"]]
        ).flatten()

        self.category_scores_ = scores.sort_values("opportunity_score", ascending=False)
        log.info("Category scores computed for %d categories.", len(self.category_scores_))
        return self.category_scores_

    # ── Affinity matrix ────────────────────────────────────────────────────────
    def compute_affinity_matrix(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Co-purchase affinity: P(bought Y | bought X)."""
        pur = df[df["event_type"] == "purchase"]
        top_cats = pur["category"].value_counts().head(top_n).index.tolist()

        user_cat = (
            pur[pur["category"].isin(top_cats)]
            .groupby(["user_id", "category"])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1)
        )

        affinity_raw  = user_cat.T.dot(user_cat).astype(float)
        row_sums      = affinity_raw.sum(axis=1).replace(0, 1)
        self.affinity_matrix_ = (affinity_raw.div(row_sums, axis=0) * 100).round(2)

        log.info("Affinity matrix computed (%d × %d).", top_n, top_n)
        return self.affinity_matrix_

    # ── User-level upsell recommendations ─────────────────────────────────────
    def compute_user_scores(
        self, df: pd.DataFrame, rfm: pd.DataFrame
    ) -> pd.DataFrame:
        """Score each user for the top-K upsell categories."""
        if self.category_scores_ is None or self.affinity_matrix_ is None:
            raise RuntimeError(
                "Call compute_category_scores() and compute_affinity_matrix() first."
            )

        pur = df[df["event_type"] == "purchase"]

        # Categories each user already bought
        user_bought = (
            pur.groupby("user_id")["category"]
            .apply(set)
            .reset_index()
            .rename(columns={"category": "bought_cats"})
        )

        rfm_scored = rfm.merge(user_bought, on="user_id", how="left")
        rfm_scored["bought_cats"] = rfm_scored["bought_cats"].apply(
            lambda x: x if isinstance(x, set) else set()
        )

        top_opps = self.category_scores_.head(20).index.tolist()

        records = []
        for _, row in rfm_scored.iterrows():
            bought  = row["bought_cats"]
            rfm_w   = row.get("rfm_score", 5) / 15.0  # normalise to 0–1

            for cat in top_opps:
                if cat in bought:
                    continue  # skip categories user already converts on

                opp = float(
                    self.category_scores_.loc[cat, "opportunity_score"]
                    if cat in self.category_scores_.index else 0
                )

                # Affinity boost from categories user bought
                affinity_boost = 0.0
                if self.affinity_matrix_ is not None and cat in self.affinity_matrix_.columns:
                    for bc in bought:
                        if bc in self.affinity_matrix_.index:
                            affinity_boost += float(self.affinity_matrix_.loc[bc, cat])
                    affinity_boost = min(affinity_boost / 100.0, 1.0)

                combined = 0.5 * opp + 0.3 * affinity_boost + 0.2 * rfm_w
                records.append({
                    "user_id":         row["user_id"],
                    "category":        cat,
                    "opportunity":     round(opp, 4),
                    "affinity":        round(affinity_boost, 4),
                    "rfm_weight":      round(rfm_w, 4),
                    "upsell_score":    round(combined, 4),
                })

        user_scores = pd.DataFrame(records)
        if len(user_scores):
            user_scores = (
                user_scores
                .sort_values(["user_id", "upsell_score"], ascending=[True, False])
                .groupby("user_id")
                .head(self.cfg.top_k_categories)
                .reset_index(drop=True)
            )

        self.user_scores_ = user_scores
        log.info("User-level scores computed: %d user×category rows.", len(user_scores))
        return user_scores

    # ── Fit (all in one) ───────────────────────────────────────────────────────
    def fit(self, df: pd.DataFrame, rfm: pd.DataFrame) -> UpsellModel:
        self.compute_category_scores(df)
        self.compute_affinity_matrix(df)
        self.compute_user_scores(df, rfm)

        mlflow.set_experiment(self.cfg.experiment_name)
        with mlflow.start_run(run_name=f"upsell_v{SCORE_VERSION}"):
            mlflow.log_params({
                "opportunity_alpha": self.cfg.opportunity_alpha,
                "top_k":            self.cfg.top_k_categories,
                "min_views":        self.cfg.min_views,
                "version":          SCORE_VERSION,
            })
            mlflow.log_metrics({
                "n_categories_scored": len(self.category_scores_),
                "n_user_recs":         (
                    len(self.user_scores_) if self.user_scores_ is not None else 0
                ),
            })
            mlflow.set_tags({
                "model_type": "rule_based_upsell",
                "task":       "upsell_cross_sell",
                "dataset":    "ecommerce_oct_nov_2019",
            })
        return self

    # ── Score a single user ────────────────────────────────────────────────────
    def score_user(self, user_id: int) -> pd.DataFrame:
        """Return top-K upsell recommendations for a single user."""
        if self.user_scores_ is None:
            raise RuntimeError("Call fit() first.")
        user_recs = self.user_scores_[self.user_scores_["user_id"] == user_id]
        return user_recs.sort_values("upsell_score", ascending=False)

    # ── Export for batch API ───────────────────────────────────────────────────
    def export_scores(self, path: str) -> None:
        if self.user_scores_ is None:
            raise RuntimeError("Call fit() first.")
        self.user_scores_.to_parquet(path, index=False)
        log.info("User scores exported to %s", path)
