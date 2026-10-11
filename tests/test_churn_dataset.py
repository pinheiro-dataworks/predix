from __future__ import annotations

import pandas as pd
import pytest

from pipeline.feature_store.churn_dataset import (
    CHURN_FEATURE_COLS,
    ChurnWindow,
    build_churn_dataset,
    build_features,
)
from tests.conftest import AS_OF


def _labels(dataset: pd.DataFrame) -> dict[int, int]:
    return dict(zip(dataset["user_id"], dataset["churned"], strict=True))


def test_eligible_population_is_users_with_a_prior_purchase(events):
    dataset = build_churn_dataset(events, AS_OF, horizon_days=30)

    assert set(dataset["user_id"]) == {1, 2, 4, 5, 6, 7}


def test_label_is_absence_of_purchase_in_prediction_window(events):
    labels = _labels(build_churn_dataset(events, AS_OF, horizon_days=30))

    assert labels == {1: 0, 2: 1, 4: 0, 5: 1, 6: 0, 7: 0}


def test_horizon_is_configurable(events):
    labels = _labels(build_churn_dataset(events, AS_OF, horizon_days=14))

    # user 4 returns on day 24, user 7 exactly on day 14 (end is exclusive)
    assert labels[4] == 1
    assert labels[7] == 1
    assert labels[1] == 0


def test_event_at_cutoff_counts_for_label_not_features(events):
    dataset = build_churn_dataset(events, AS_OF, horizon_days=30).set_index("user_id")

    assert dataset.loc[6, "churned"] == 0
    assert dataset.loc[6, "frequency"] == 1
    assert dataset.loc[6, "monetary"] == 40.0


def test_features_are_measured_relative_to_cutoff(events):
    dataset = build_churn_dataset(events, AS_OF, horizon_days=30).set_index("user_id")

    assert dataset.loc[2, "recency"] == pytest.approx(11.5)  # Oct 20 12:00 → Nov 1
    assert dataset.loc[1, "n_views"] == 1
    assert dataset.loc[1, "n_carts"] == 1
    assert dataset.loc[1, "cart_to_pur_ratio"] == 1.0
    assert list(dataset.columns) == [*CHURN_FEATURE_COLS, "churned"]


def test_lookback_restricts_observation_window(events):
    dataset = build_churn_dataset(events, AS_OF, horizon_days=30, lookback_days=14)

    # only purchases from Oct 18 onwards make a user eligible
    assert set(dataset["user_id"]) == {2, 5}


def test_truncated_prediction_window_is_rejected(events):
    with pytest.raises(ValueError, match="prediction window"):
        build_churn_dataset(events, "2019-11-15", horizon_days=30)


def test_truncated_window_allowed_when_explicitly_requested(events):
    dataset = build_churn_dataset(
        events, "2019-11-15", horizon_days=30, require_full_horizon=False
    )

    assert len(dataset) > 0


def test_naive_cutoff_is_aligned_to_event_timezone(events):
    naive = build_churn_dataset(events, "2019-11-01", horizon_days=30)
    aware = build_churn_dataset(events, AS_OF, horizon_days=30)

    pd.testing.assert_frame_equal(naive, aware)


def test_build_features_rejects_post_cutoff_events(events):
    with pytest.raises(ValueError, match="at or after as_of"):
        build_features(events, AS_OF)


@pytest.mark.parametrize("kwargs", [{"horizon_days": 0}, {"lookback_days": -1}])
def test_window_rejects_non_positive_lengths(kwargs):
    with pytest.raises(ValueError):
        ChurnWindow(as_of=AS_OF, **kwargs)
