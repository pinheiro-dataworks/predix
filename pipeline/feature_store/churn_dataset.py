"""
Predix — Churn Dataset (target + point-in-time features)
────────────────────────────────────────────────────────────────────────────────
Churn is defined relative to a cutoff date `as_of`:

  observation window   [as_of - lookback_days, as_of)   → features
  prediction window    [as_of, as_of + horizon_days)    → label

  eligible population  users with at least one purchase in the observation window
  churned = 1          eligible user with no purchase in the prediction window

Every feature is computed from the observation slice only, so no event at or
after `as_of` can influence the model inputs. The label is the only output
that reads the prediction window.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CHURN_FEATURE_COLS = [
    "frequency",
    "monetary",
    "avg_price",
    "recency",
    "unique_cats",
    "unique_brands",
    "unique_prods",
    "n_views",
    "n_carts",
    "cart_to_pur_ratio",
    "n_sessions",
    "active_days",
    "tenure_days",
]

# Events are logged with second precision but the raw files are split by
# calendar day, so a window may end up to one day after the last event.
_COVERAGE_TOLERANCE = pd.Timedelta(days=1)


@dataclass(frozen=True)
class ChurnWindow:
    """Cutoff and window lengths that define the churn target."""

    as_of: pd.Timestamp
    horizon_days: int = 30
    lookback_days: int | None = None

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")
        if self.lookback_days is not None and self.lookback_days <= 0:
            raise ValueError(f"lookback_days must be positive, got {self.lookback_days}")

    @property
    def observation_start(self) -> pd.Timestamp | None:
        if self.lookback_days is None:
            return None
        return self.as_of - pd.Timedelta(days=self.lookback_days)

    @property
    def prediction_end(self) -> pd.Timestamp:
        return self.as_of + pd.Timedelta(days=self.horizon_days)


def _align_tz(ts: pd.Timestamp | str, reference: pd.Series) -> pd.Timestamp:
    """Return `ts` in the same timezone as the event_time column."""
    ts = pd.Timestamp(ts)
    tz = getattr(reference.dt, "tz", None)
    if tz is not None and ts.tzinfo is None:
        return ts.tz_localize(tz)
    if tz is None and ts.tzinfo is not None:
        return ts.tz_convert(None)
    return ts


def split_by_cutoff(
    events: pd.DataFrame, window: ChurnWindow
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split events into the observation slice and the prediction slice."""
    t = events["event_time"]
    obs_mask = t < window.as_of
    if window.observation_start is not None:
        obs_mask &= t >= window.observation_start
    pred_mask = (t >= window.as_of) & (t < window.prediction_end)
    return events.loc[obs_mask], events.loc[pred_mask]


def build_features(observation: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-user features for users with at least one purchase in `observation`.

    The caller is responsible for passing only pre-cutoff events; this function
    also refuses any event at or after `as_of` as a second line of defence.
    """
    if len(observation) and observation["event_time"].max() >= as_of:
        raise ValueError("observation contains events at or after as_of")

    pur = observation[observation["event_type"] == "purchase"]
    feat = pur.groupby("user_id").agg(
        frequency=("event_time", "size"),
        monetary=("price", "sum"),
        avg_price=("price", "mean"),
        last_purchase=("event_time", "max"),
        unique_cats=("category", "nunique"),
        unique_brands=("brand", "nunique"),
        unique_prods=("product_id", "nunique"),
    )

    # Activity features cover every event type, restricted to eligible users.
    activity = observation[observation["user_id"].isin(feat.index)]
    by_type = activity.groupby(["user_id", "event_type"]).size().unstack(fill_value=0)
    feat["n_views"] = by_type.get("view", pd.Series(dtype=int)).reindex(feat.index, fill_value=0)
    feat["n_carts"] = by_type.get("cart", pd.Series(dtype=int)).reindex(feat.index, fill_value=0)

    per_user = (
        activity.assign(day=activity["event_time"].dt.normalize())
        .groupby("user_id")
        .agg(
            n_sessions=("user_session", "nunique"),
            active_days=("day", "nunique"),
            first_event=("event_time", "min"),
        )
    )
    feat = feat.join(per_user)

    feat["recency"] = (as_of - feat["last_purchase"]).dt.total_seconds() / 86_400
    feat["tenure_days"] = (as_of - feat["first_event"]).dt.total_seconds() / 86_400
    feat["cart_to_pur_ratio"] = feat["n_carts"] / feat["frequency"]

    feat = feat.drop(columns=["last_purchase", "first_event"])
    return feat[CHURN_FEATURE_COLS].astype(float)


def build_labels(prediction: pd.DataFrame, users: pd.Index) -> pd.Series:
    """1 if the user has no purchase in the prediction slice, else 0."""
    buyers = prediction.loc[prediction["event_type"] == "purchase", "user_id"].unique()
    return pd.Series(np.where(users.isin(buyers), 0, 1), index=users, name="churned")


def build_churn_dataset(
    events: pd.DataFrame,
    as_of: pd.Timestamp | str,
    horizon_days: int = 30,
    lookback_days: int | None = None,
    require_full_horizon: bool = True,
) -> pd.DataFrame:
    """Return one row per eligible user with point-in-time features and `churned`.

    Parameters
    ----------
    events : prepared event log (see `feature_engineering.prepare_events`).
    as_of : cutoff; features use events strictly before it.
    horizon_days : length N of the prediction window.
    lookback_days : optional length of the observation window (None = all history).
    require_full_horizon : raise if the data ends before the prediction window
        does. A truncated window would label active users as churned.
    """
    window = ChurnWindow(
        as_of=_align_tz(as_of, events["event_time"]),
        horizon_days=horizon_days,
        lookback_days=lookback_days,
    )

    if require_full_horizon:
        data_end = events["event_time"].max()
        if data_end + _COVERAGE_TOLERANCE < window.prediction_end:
            raise ValueError(
                f"prediction window ends {window.prediction_end} but data ends {data_end}"
            )

    observation, prediction = split_by_cutoff(events, window)
    feat = build_features(observation, window.as_of)
    labels = build_labels(prediction, feat.index)

    dataset = feat.join(labels).reset_index()
    log.info(
        "Churn dataset as_of=%s horizon=%dd: %d eligible users | churn rate=%.1f%%",
        window.as_of.date(), horizon_days, len(dataset),
        dataset["churned"].mean() * 100 if len(dataset) else float("nan"),
    )
    return dataset
