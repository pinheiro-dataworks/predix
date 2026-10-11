from __future__ import annotations

import pandas as pd
import pytest

from pipeline.feature_store.feature_engineering import prepare_events

AS_OF = pd.Timestamp("2019-11-01", tz="UTC")


def make_events(rows: list[tuple]) -> pd.DataFrame:
    """Build a prepared event log from (time, type, user_id, price, category_code) tuples."""
    records = [
        {
            "event_time": time,
            "event_type": event_type,
            "product_id": 1000 + i,
            "category_id": 1,
            "category_code": category_code,
            "brand": "acme",
            "price": price,
            "user_id": user_id,
            "user_session": f"s-{user_id}-{str(time)[:10]}",
        }
        for i, (time, event_type, user_id, price, category_code) in enumerate(rows)
    ]
    return prepare_events(pd.DataFrame(records))


@pytest.fixture
def events() -> pd.DataFrame:
    """Small two-month log with one user per edge case around AS_OF = 2019-11-01."""
    return make_events([
        # 1: buys before and inside the window → retained
        ("2019-10-05 10:00:00", "view", 1, 50.0, "electronics.smartphone"),
        ("2019-10-05 10:05:00", "cart", 1, 50.0, "electronics.smartphone"),
        ("2019-10-05 10:10:00", "purchase", 1, 50.0, "electronics.smartphone"),
        ("2019-11-10 09:00:00", "purchase", 1, 80.0, "appliances.kitchen"),
        # 2: buys before, nothing after → churned
        ("2019-10-20 12:00:00", "purchase", 2, 20.0, "apparel.shoes"),
        # 3: only browses before the cutoff → not eligible
        ("2019-10-15 08:00:00", "view", 3, 10.0, ""),
        ("2019-11-05 08:00:00", "purchase", 3, 10.0, ""),
        # 4: comes back late in the window → retained at N=30, churned at N=14
        ("2019-10-01 00:00:00", "purchase", 4, 15.0, "computers.notebook"),
        ("2019-11-25 18:00:00", "purchase", 4, 15.0, "computers.notebook"),
        # 5: last purchase just before the cutoff, only views afterwards → churned
        ("2019-10-31 23:59:59", "purchase", 5, 30.0, "electronics.audio"),
        ("2019-11-02 14:00:00", "view", 5, 30.0, "electronics.audio"),
        # 6: purchase exactly at as_of belongs to the prediction window → retained
        ("2019-10-10 11:00:00", "purchase", 6, 40.0, "electronics.smartphone"),
        ("2019-11-01 00:00:00", "purchase", 6, 40.0, "electronics.smartphone"),
        # 7: purchase exactly at as_of + 14d is outside a 14-day window
        ("2019-10-12 11:00:00", "purchase", 7, 25.0, "apparel.shoes"),
        ("2019-11-15 00:00:00", "purchase", 7, 25.0, "apparel.shoes"),
        # filler: data covers the whole of November
        ("2019-11-30 23:00:00", "view", 99, 5.0, ""),
    ])
