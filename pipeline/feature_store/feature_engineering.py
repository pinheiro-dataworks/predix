"""
Predix — Feature Engineering
────────────────────────────────────────────────────────────────────────────────
Builds the analytical feature store from Silver-layer events.
Entities: user, product, session.
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REF_DATE = pd.Timestamp("2019-12-01", tz="UTC")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Event cleaning ─────────────────────────────────────────────────────────────
def prepare_events(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw events: UTC timestamps, numeric price, filled brand/category.

    `category` is the top level of `category_code` ("electronics.smartphone"
    → "electronics"); events without a code fall into "unknown".
    """
    out = df.copy()
    out["event_time"] = _parse_event_time(out["event_time"])
    out["price"] = pd.to_numeric(out["price"], errors="coerce").fillna(0.0)
    out["brand"] = out["brand"].fillna("unknown")
    out["category_code"] = out["category_code"].fillna("")
    out["category"] = out["category_code"].str.split(".").str[0].replace("", "unknown")
    return out


def load_events(path: str | Path) -> pd.DataFrame:
    """Read the event sample written by scripts/build_sample.py and prepare it."""
    return prepare_events(pd.read_parquet(path))


def _parse_event_time(col: pd.Series) -> pd.Series:
    """Parse event timestamps as UTC.

    Raw files store "2019-10-01 00:00:00 UTC". Parsing the fixed-width prefix
    with an explicit format is ~15x faster than letting pandas infer the
    timezone suffix, which matters on millions of rows.
    """
    if pd.api.types.is_datetime64_any_dtype(col):
        return col.dt.tz_localize("UTC") if col.dt.tz is None else col.dt.tz_convert("UTC")
    as_str = col.astype(str)
    if as_str.str.endswith(" UTC").all():
        parsed = pd.to_datetime(as_str.str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")
        return parsed.dt.tz_localize("UTC")
    return pd.to_datetime(as_str, utc=True)


# ── RFM ───────────────────────────────────────────────────────────────────────
def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Recency, Frequency, Monetary + segment for all buyers."""
    pur = df[df["event_type"] == "purchase"].copy()

    rfm = (
        pur.groupby("user_id")
        .agg(
            recency   = ("event_time", lambda x: int((REF_DATE - x.max()).days)),
            frequency = ("event_time", "count"),
            monetary  = ("price", "sum"),
            avg_price = ("price", "mean"),
        )
        .reset_index()
    )
    rfm = rfm[rfm["monetary"] > 0]

    for col, labels, _ in [
        ("recency",   [5, 4, 3, 2, 1], True),
        ("frequency", [1, 2, 3, 4, 5], False),
        ("monetary",  [1, 2, 3, 4, 5], False),
    ]:
        rfm[f"{col[0]}_score"] = pd.qcut(
            rfm[col], q=5, labels=labels, duplicates="drop"
        ).astype(int)

    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    rfm["segment"]   = rfm.apply(_classify_segment, axis=1)
    log.info("RFM built: %d users | segments: %s", len(rfm), rfm["segment"].value_counts().to_dict())
    return rfm


def _classify_segment(row: pd.Series) -> str:
    r, f = int(row["r_score"]), int(row["f_score"])
    if r >= 4 and f >= 4:  return "Champions"
    if r >= 3 and f >= 3:  return "Loyal"
    if r >= 3 and f <= 2:  return "Potential Loyalist"
    if r == 2 and f >= 3:  return "At Risk"
    if r == 2 and f == 2:  return "Needs Attention"
    if r == 1 and f >= 3:  return "Can't Lose"
    if r == 1 and f == 2:  return "Hibernating"
    return "Lost"


# ── User features ──────────────────────────────────────────────────────────────
def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate all event types into a per-user feature table."""
    pur = df[df["event_type"] == "purchase"]
    viw = df[df["event_type"] == "view"]
    crt = df[df["event_type"] == "cart"]

    base = (
        pur.groupby("user_id")
        .agg(
            total_purchases   = ("event_time", "count"),
            total_spend       = ("price", "sum"),
            avg_purchase_price= ("price", "mean"),
            unique_products   = ("product_id", "nunique"),
            unique_categories = ("category_code", "nunique"),
            unique_brands     = ("brand", "nunique"),
            first_purchase_dt = ("event_time", "min"),
            last_purchase_dt  = ("event_time", "max"),
        )
        .reset_index()
    )
    base = base.merge(viw.groupby("user_id").size().rename("total_views"),  on="user_id", how="left")
    base = base.merge(crt.groupby("user_id").size().rename("total_carts"),  on="user_id", how="left")
    base = base.fillna(0)

    base["recency_days"]       = (REF_DATE - base["last_purchase_dt"]).dt.days.astype(int)
    base["cart_to_pur_ratio"]  = base["total_carts"] / base["total_purchases"].replace(0, np.nan)
    base["view_to_pur_ratio"]  = base["total_views"] / base["total_purchases"].replace(0, np.nan)
    base = base.fillna(0)

    log.info("User features built: %d users.", len(base))
    return base


# ── Product features ───────────────────────────────────────────────────────────
def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    pur = df[df["event_type"] == "purchase"]
    viw = df[df["event_type"] == "view"]
    crt = df[df["event_type"] == "cart"]

    prod = (
        pur.groupby(["product_id", "category_code", "brand"])
        .agg(
            total_purchases = ("price", "count"),
            total_revenue   = ("price", "sum"),
            avg_price       = ("price", "mean"),
        )
        .reset_index()
    )
    prod = prod.merge(viw.groupby("product_id").size().rename("total_views"),  on="product_id", how="left")
    prod = prod.merge(crt.groupby("product_id").size().rename("total_carts"),  on="product_id", how="left")
    prod = prod.fillna(0)
    prod["conversion_rate"] = prod["total_purchases"] / prod["total_views"].replace(0, np.nan)
    prod = prod.fillna(0)

    log.info("Product features built: %d products.", len(prod))
    return prod


# ── Session features ───────────────────────────────────────────────────────────
def build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    sess = (
        df.groupby("user_session")
        .agg(
            user_id           = ("user_id", "first"),
            session_events    = ("event_type", "count"),
            session_purchases = ("event_type", lambda x: (x == "purchase").sum()),
            session_revenue   = ("price",      "sum"),
            session_start     = ("event_time", "min"),
            session_end       = ("event_time", "max"),
        )
        .reset_index()
    )
    sess["duration_minutes"] = (
        (sess["session_end"] - sess["session_start"]).dt.total_seconds() / 60
    ).clip(lower=0)
    sess["converted"] = (sess["session_purchases"] > 0).astype(int)

    log.info("Session features built: %d sessions.", len(sess))
    return sess
