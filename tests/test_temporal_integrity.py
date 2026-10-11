"""Point-in-time guarantees of the churn dataset.

The central property: features at `as_of` are a function of pre-cutoff events
only. Each test perturbs the data on or after the cutoff and requires every
feature column to stay identical. If any feature starts reading future events,
these tests fail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.feature_store.churn_dataset import CHURN_FEATURE_COLS, build_churn_dataset
from pipeline.feature_store.feature_engineering import prepare_events

DATA_START = pd.Timestamp("2019-10-01", tz="UTC")
DATA_END = pd.Timestamp("2019-12-01", tz="UTC")
CATEGORIES = ["electronics.smartphone", "appliances.kitchen", "apparel.shoes", ""]


def _random_events(seed: int, n_users: int = 300, n_events: int = 6_000) -> pd.DataFrame:
    """Synthetic two-month log with views, carts and purchases on both sides of any cutoff."""
    rng = np.random.default_rng(seed)
    span = (DATA_END - DATA_START).total_seconds()
    times = DATA_START + pd.to_timedelta(np.sort(rng.uniform(0, span - 1, n_events)), unit="s")
    users = rng.integers(1, n_users + 1, n_events)
    raw = pd.DataFrame({
        "event_time": times,
        "event_type": rng.choice(["view", "cart", "purchase"], n_events, p=[0.75, 0.15, 0.10]),
        "product_id": rng.integers(1, 400, n_events),
        "category_id": 1,
        "category_code": rng.choice(CATEGORIES, n_events),
        "brand": rng.choice(["acme", "globex", None], n_events),
        "price": rng.gamma(2.0, 40.0, n_events).round(2),
        "user_id": users,
        "user_session": [f"s{u}-{d}" for u, d in zip(users, times.day, strict=True)],
    })
    return prepare_events(raw)


def _features(events: pd.DataFrame, as_of, horizon_days: int = 30) -> pd.DataFrame:
    dataset = build_churn_dataset(
        events, as_of, horizon_days=horizon_days, require_full_horizon=False
    )
    return dataset.set_index("user_id")[CHURN_FEATURE_COLS].sort_index()


def _future_purchases(events: pd.DataFrame, as_of: pd.Timestamp, seed: int) -> pd.DataFrame:
    """Expensive purchases after the cutoff, for existing and brand-new users."""
    rng = np.random.default_rng(seed)
    users = np.concatenate([events["user_id"].unique()[:50], np.arange(10_000, 10_020)])
    offsets = pd.to_timedelta(rng.uniform(0, 20 * 86_400, len(users)), unit="s")
    extra = pd.DataFrame({
        "event_time": as_of + offsets,
        "event_type": "purchase",
        "product_id": 999_999,
        "category_id": 1,
        "category_code": "luxury.watch",
        "brand": "future",
        "price": 50_000.0,
        "user_id": users,
        "user_session": [f"future-{u}" for u in users],
    })
    return pd.concat([events, prepare_events(extra)], ignore_index=True)


CUTOFFS = ["2019-10-20", "2019-11-01", "2019-11-10"]


@pytest.mark.parametrize("as_of", CUTOFFS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_features_ignore_removal_of_future_events(seed, as_of):
    events = _random_events(seed)
    cutoff = pd.Timestamp(as_of, tz="UTC")

    full = _features(events, cutoff)
    past_only = _features(events[events["event_time"] < cutoff], cutoff)

    assert len(full) > 0
    pd.testing.assert_frame_equal(full, past_only)


@pytest.mark.parametrize("as_of", CUTOFFS)
def test_features_ignore_injected_future_purchases(as_of):
    events = _random_events(seed=7)
    cutoff = pd.Timestamp(as_of, tz="UTC")

    before = _features(events, cutoff)
    after = _features(_future_purchases(events, cutoff, seed=7), cutoff)

    pd.testing.assert_frame_equal(before, after)


def test_features_ignore_edits_to_future_events():
    events = _random_events(seed=3)
    cutoff = pd.Timestamp("2019-11-01", tz="UTC")
    future = events["event_time"] >= cutoff

    edited = events.copy()
    edited.loc[future, "price"] *= 100
    edited.loc[future, "event_type"] = "purchase"
    edited.loc[future, "brand"] = "edited"

    pd.testing.assert_frame_equal(_features(events, cutoff), _features(edited, cutoff))


def test_event_exactly_at_cutoff_is_not_a_feature():
    events = _random_events(seed=4)
    cutoff = pd.Timestamp("2019-11-01", tz="UTC")
    user = int(events.loc[events["event_time"] < cutoff, "user_id"].iloc[0])
    at_cutoff = events.iloc[[0]].assign(
        event_time=cutoff, event_type="purchase", user_id=user, price=1e6
    )

    before = _features(events, cutoff)
    after = _features(pd.concat([events, at_cutoff], ignore_index=True), cutoff)

    pd.testing.assert_frame_equal(before, after)


def test_future_buyers_do_not_enter_the_eligible_population():
    events = _random_events(seed=5)
    cutoff = pd.Timestamp("2019-11-01", tz="UTC")

    eligible = set(_features(events, cutoff).index)
    with_future = set(_features(_future_purchases(events, cutoff, seed=5), cutoff).index)

    assert with_future == eligible
    assert not any(u >= 10_000 for u in with_future)


def test_labels_do_react_to_future_purchases():
    """Control: the perturbation used above must be visible to the label."""
    events = _random_events(seed=6)
    cutoff = pd.Timestamp("2019-11-01", tz="UTC")

    before = build_churn_dataset(events, cutoff, require_full_horizon=False)
    after = build_churn_dataset(
        _future_purchases(events, cutoff, seed=6), cutoff, require_full_horizon=False
    )

    assert after["churned"].sum() < before["churned"].sum()


@pytest.mark.parametrize("horizon_days", [7, 14, 30])
def test_horizon_changes_labels_but_not_features(horizon_days):
    events = _random_events(seed=8)
    cutoff = pd.Timestamp("2019-10-25", tz="UTC")

    reference = build_churn_dataset(events, cutoff, horizon_days=30).set_index("user_id")
    other = build_churn_dataset(events, cutoff, horizon_days=horizon_days).set_index("user_id")

    pd.testing.assert_frame_equal(reference[CHURN_FEATURE_COLS], other[CHURN_FEATURE_COLS])
    # a shorter window can only turn retained users into churned ones
    assert (other["churned"] >= reference["churned"]).all()
