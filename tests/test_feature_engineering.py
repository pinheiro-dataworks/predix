from __future__ import annotations

import pandas as pd
import pytest

from pipeline.feature_store.feature_engineering import (
    build_product_features,
    build_rfm,
    build_session_features,
    build_user_features,
    prepare_events,
    score_quantiles,
)
from pipeline.scoring.batch_scoring import assign_churn_risk
from tests.conftest import make_events


def _raw(**overrides) -> pd.DataFrame:
    row = {
        "event_time": "2019-10-05 10:15:30 UTC",
        "event_type": "purchase",
        "product_id": 1,
        "category_id": 1,
        "category_code": "electronics.smartphone",
        "brand": "acme",
        "price": "99.90",
        "user_id": 1,
        "user_session": "s1",
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ── prepare_events ─────────────────────────────────────────────────────────────
def test_prepare_events_parses_raw_utc_suffix():
    out = prepare_events(_raw())

    assert str(out["event_time"].dt.tz) == "UTC"
    assert out["event_time"].iloc[0] == pd.Timestamp("2019-10-05 10:15:30", tz="UTC")


@pytest.mark.parametrize("value", [
    "2019-10-05T10:15:30+00:00",
    pd.Timestamp("2019-10-05 10:15:30"),
    pd.Timestamp("2019-10-05 07:15:30", tz="America/Sao_Paulo"),
])
def test_prepare_events_normalises_other_timestamp_forms(value):
    out = prepare_events(_raw(event_time=value))

    assert out["event_time"].iloc[0] == pd.Timestamp("2019-10-05 10:15:30", tz="UTC")


def test_prepare_events_fills_missing_values():
    out = prepare_events(_raw(brand=None, category_code=None, price="not-a-number"))

    assert out["brand"].iloc[0] == "unknown"
    assert out["category_code"].iloc[0] == ""
    assert out["category"].iloc[0] == "unknown"
    assert out["price"].iloc[0] == 0.0


def test_prepare_events_extracts_top_level_category():
    out = prepare_events(_raw(category_code="appliances.kitchen.refrigerators"))

    assert out["category"].iloc[0] == "appliances"


def test_prepare_events_does_not_mutate_input():
    raw = _raw()
    prepare_events(raw)

    assert raw["event_time"].iloc[0] == "2019-10-05 10:15:30 UTC"
    assert "category" not in raw


# ── score_quantiles ────────────────────────────────────────────────────────────
def test_score_quantiles_spans_one_to_q():
    scores = score_quantiles(pd.Series(range(100)))

    assert scores.min() == 1
    assert scores.max() == 5
    assert scores.value_counts().tolist() == [20] * 5


def test_score_quantiles_handles_heavy_ties():
    # most buyers have a single purchase: pd.qcut(duplicates="drop") fails here
    frequency = pd.Series([1] * 80 + [2] * 10 + [5] * 10)
    scores = score_quantiles(frequency)

    assert scores[frequency == 1].nunique() == 1
    assert scores[frequency == 5].iloc[0] == 5
    assert scores.is_monotonic_increasing


def test_score_quantiles_descending_rewards_low_values():
    recency = pd.Series([1, 10, 20, 30, 40])
    scores = score_quantiles(recency, ascending=False)

    assert scores.iloc[0] == 5
    assert scores.iloc[-1] == 1


# ── RFM ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def buyers() -> pd.DataFrame:
    rows = []
    for user in range(1, 41):
        n_purchases = 1 if user <= 25 else user // 5
        for k in range(n_purchases):
            rows.append((f"2019-11-{(user + k) % 28 + 1:02d} 12:00:00", "purchase", user,
                         10.0 * user, "electronics.audio"))
    rows.append(("2019-11-10 12:00:00", "view", 99, 5.0, ""))  # non-buyer
    return make_events(rows)


def test_build_rfm_scores_and_segments_every_buyer(buyers):
    rfm = build_rfm(buyers)

    assert set(rfm["user_id"]) == set(range(1, 41))
    for col in ["r_score", "f_score", "m_score"]:
        assert rfm[col].between(1, 5).all()
    assert (rfm["rfm_score"] == rfm["r_score"] + rfm["f_score"] + rfm["m_score"]).all()
    assert rfm["segment"].notna().all()


def test_build_rfm_recency_is_measured_to_reference_date(buyers):
    rfm = build_rfm(buyers).set_index("user_id")
    last = buyers[buyers["event_type"] == "purchase"].groupby("user_id")["event_time"].max()

    expected = (pd.Timestamp("2019-12-01", tz="UTC") - last).dt.days
    pd.testing.assert_series_equal(
        rfm["recency"].sort_index(), expected.sort_index(), check_names=False
    )


# ── Other feature tables ───────────────────────────────────────────────────────
def test_user_features_ratios(events):
    users = build_user_features(events).set_index("user_id")

    assert users.loc[1, "total_purchases"] == 2
    assert users.loc[1, "total_views"] == 1
    assert users.loc[1, "cart_to_pur_ratio"] == 0.5


def test_product_features_conversion_rate():
    events = make_events([
        ("2019-10-01 10:00:00", "view", 1, 10.0, "a.b"),
        ("2019-10-01 10:01:00", "view", 2, 10.0, "a.b"),
        ("2019-10-01 10:02:00", "purchase", 1, 10.0, "a.b"),
    ]).assign(product_id=7)

    prod = build_product_features(events).set_index("product_id")

    assert prod.loc[7, "conversion_rate"] == 0.5


def test_session_features_duration_and_conversion(events):
    sessions = build_session_features(events).set_index("user_session")
    s = sessions.loc["s-1-2019-10-05"]

    assert s["session_events"] == 3
    assert s["duration_minutes"] == 10
    assert s["converted"] == 1


# ── Risk tiers ─────────────────────────────────────────────────────────────────
def test_assign_churn_risk_bins():
    tiers = assign_churn_risk(pd.Series([0.0, 0.35, 0.36, 0.60, 0.61, 1.0]))

    assert tiers.tolist() == ["Low", "Low", "Medium", "Medium", "High", "High"]
