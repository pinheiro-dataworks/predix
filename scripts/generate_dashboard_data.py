#!/usr/bin/env python3
"""
Predix - generate_dashboard_data.py
Pre-computes all chart data from the local event sample, evaluates the churn
models, and exports dashboard/data/dashboard_data.json for the static frontend.

Usage (from project root):
    pip install -r requirements.txt
    python scripts/build_sample.py
    python scripts/generate_dashboard_data.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.feature_store.churn_dataset import (  # noqa: E402
    CHURN_FEATURE_COLS,
    build_churn_dataset,
)
from pipeline.feature_store.feature_engineering import (  # noqa: E402
    prepare_events,
    score_quantiles,
)
from pipeline.models.churn_evaluation import evaluate_churn_models  # noqa: E402

warnings.filterwarnings("ignore")

SAMPLE_PATH = ROOT / "dataset" / "sample_events.parquet"
OUT_DIR = ROOT / "dashboard" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AS_OF = pd.Timestamp("2019-11-01", tz="UTC")
HORIZON_DAYS = 30
REF_DATE = pd.Timestamp("2019-12-01", tz="UTC")
RFM_SCATTER_CAP = 5_000
RFM_3D_CAP = 3_000
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def sf(x) -> float:
    try:
        v = float(x)
        return 0.0 if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    except Exception:
        return 0.0


def si(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def safe_list(it) -> list:
    out = []
    for v in it:
        if isinstance(v, (np.integer,)):
            out.append(int(v))
        elif isinstance(v, (float, np.floating)):
            out.append(sf(v))
        else:
            out.append(v)
    return out


print(">> Loading data...")

df = prepare_events(pd.read_parquet(SAMPLE_PATH))
df["month"]    = np.where(df["event_time"].dt.month == 10, "october", "november")
df["date"]     = df["event_time"].dt.strftime("%Y-%m-%d")
df["hour"]     = df["event_time"].dt.hour.astype(int)
df["weekday"]  = df["event_time"].dt.day_name()

total_rows  = len(df)
total_users = df["user_id"].nunique()
print(f"   {total_rows:,} rows | {total_users:,} unique users")

frames = {
    "all":      df,
    "october":  df[df["month"] == "october"].copy(),
    "november": df[df["month"] == "november"].copy(),
}


# OVERVIEW
print(">> Computing overview...")


def build_overview(data: pd.DataFrame) -> dict:
    pur = data[data["event_type"] == "purchase"]
    crt = data[data["event_type"] == "cart"]
    viw = data[data["event_type"] == "view"]
    n_viw, n_crt, n_pur = len(viw), len(crt), len(pur)

    kpis = {
        "total_revenue":   sf(pur["price"].sum()),
        "total_orders":    n_pur,
        "total_customers": si(data["user_id"].nunique()),
        "avg_order_value": sf(pur["price"].mean()),
        "conversion_rate": sf(n_pur / max(n_viw, 1) * 100),
        "cart_rate":       sf(n_crt / max(n_viw, 1) * 100),
    }

    dly = (
        data.groupby(["date", "event_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("date")
    )
    for col in ["view", "cart", "purchase"]:
        if col not in dly.columns:
            dly[col] = 0

    daily_events = {
        "dates":     dly["date"].tolist(),
        "views":     safe_list(dly["view"]),
        "carts":     safe_list(dly["cart"]),
        "purchases": safe_list(dly["purchase"]),
    }

    funnel = {
        "stages":      ["Views", "Cart Adds", "Purchases"],
        "values":      [n_viw, n_crt, n_pur],
        "percentages": [100.0, sf(n_crt / max(n_viw, 1) * 100), sf(n_pur / max(n_viw, 1) * 100)],
    }

    ec = data["event_type"].value_counts()
    event_split = {
        "labels": ["View", "Cart", "Purchase"],
        "values": [si(ec.get("view", 0)), si(ec.get("cart", 0)), si(ec.get("purchase", 0))],
    }

    cr = pur.groupby("category")["price"].sum().sort_values(ascending=False).head(10)
    category_revenue = {
        "categories": cr.index.tolist(),
        "revenues":   safe_list(cr.values),
    }

    dr = pur.groupby("date")["price"].sum().reset_index().sort_values("date")
    daily_revenue = {
        "dates":    dr["date"].tolist(),
        "revenues": safe_list(dr["price"]),
    }

    return {
        "kpis":             kpis,
        "daily_events":     daily_events,
        "funnel":           funnel,
        "event_split":      event_split,
        "category_revenue": category_revenue,
        "daily_revenue":    daily_revenue,
    }


overview_out = {m: build_overview(f) for m, f in frames.items()}


# CHURN INTELLIGENCE
print(">> Computing churn analysis...")

# Target: eligible = bought before AS_OF; churned = no purchase in the next
# HORIZON_DAYS. Features only see events before AS_OF.
churn_ds = build_churn_dataset(df, AS_OF, horizon_days=HORIZON_DAYS)
eligible = set(churn_ds["user_id"])
churned  = set(churn_ds.loc[churn_ds["churned"] == 1, "user_id"])
retained = eligible - churned

horizon_end = AS_OF + pd.Timedelta(days=HORIZON_DAYS)
window_buyers = set(df.loc[
    (df["event_type"] == "purchase")
    & (df["event_time"] >= AS_OF)
    & (df["event_time"] < horizon_end),
    "user_id",
])
new_buyers = window_buyers - eligible

churn_rate = sf(len(churned) / max(len(eligible), 1) * 100)

cohort = {
    "categories": ["Buyers before cutoff", f"Buyers in next {HORIZON_DAYS} days"],
    "retained":   [len(retained), len(retained)],
    "churned":    [len(churned), 0],
    "new":        [0, len(new_buyers)],
    "totals":     [len(eligible), len(window_buyers)],
}

pur_all = df[df["event_type"] == "purchase"].copy()

rfm = (
    pur_all.groupby("user_id")
    .agg(
        recency   = ("event_time", lambda x: int((REF_DATE - x.max()).days)),
        frequency = ("event_time", "count"),
        monetary  = ("price", "sum"),
    )
    .reset_index()
)
rfm = rfm[rfm["monetary"] > 0].copy()


def add_rfm_scores(df_rfm: pd.DataFrame) -> pd.DataFrame:
    df_rfm["r_score"] = score_quantiles(df_rfm["recency"],   5, ascending=False)
    df_rfm["f_score"] = score_quantiles(df_rfm["frequency"], 5, ascending=True)
    df_rfm["m_score"] = score_quantiles(df_rfm["monetary"],  5, ascending=True)
    df_rfm["rfm_score"] = df_rfm["r_score"] + df_rfm["f_score"] + df_rfm["m_score"]
    return df_rfm


rfm = add_rfm_scores(rfm)


def classify_segment(row) -> str:
    r, f = row["r_score"], row["f_score"]
    if r >= 4 and f >= 4:  return "Champions"
    if r >= 3 and f >= 3:  return "Loyal"
    if r >= 3 and f <= 2:  return "Potential Loyalist"
    if r == 2 and f >= 3:  return "At Risk"
    if r == 2 and f == 2:  return "Needs Attention"
    if r == 1 and f >= 3:  return "Can't Lose"
    if r == 1 and f == 2:  return "Hibernating"
    return "Lost"


rfm["segment"] = rfm.apply(classify_segment, axis=1)

seg_colors = {
    "Champions":          "#F59E0B",
    "Loyal":              "#34D399",
    "Potential Loyalist": "#60A5FA",
    "At Risk":            "#EF4444",
    "Needs Attention":    "#F97316",
    "Can't Lose":         "#A78BFA",
    "Hibernating":        "#6B7280",
    "Lost":               "#374151",
}

rfm_samp = rfm.sample(min(RFM_SCATTER_CAP, len(rfm)), random_state=42)
rfm_map = {
    "recency":   safe_list(rfm_samp["recency"]),
    "monetary":  safe_list(rfm_samp["monetary"]),
    "frequency": safe_list(rfm_samp["frequency"]),
    "segment":   rfm_samp["segment"].tolist(),
    "rfm_score": safe_list(rfm_samp["rfm_score"]),
}

seg_counts = rfm["segment"].value_counts()
segments = {
    "names":  seg_counts.index.tolist(),
    "counts": safe_list(seg_counts.values),
    "colors": [seg_colors.get(s, "#7B5CFA") for s in seg_counts.index],
}

rfm_3d = rfm.sample(min(RFM_3D_CAP, len(rfm)), random_state=7)
rfm_3d_out = {
    "r":       safe_list(rfm_3d["recency"]),
    "f":       safe_list(rfm_3d["frequency"]),
    "m":       safe_list(rfm_3d["monetary"]),
    "score":   safe_list(rfm_3d["rfm_score"]),
    "segment": rfm_3d["segment"].tolist(),
}

churn_out = {
    "churn_rate":    churn_rate,
    "as_of":         AS_OF.strftime("%Y-%m-%d"),
    "horizon_days":  HORIZON_DAYS,
    "eligible":      len(eligible),
    "window_buyers": len(window_buyers),
    "retained":      len(retained),
    "churned":       len(churned),
    "new_buyers":    len(new_buyers),
    "cohort":      cohort,
    "rfm_map":     rfm_map,
    "segments":    segments,
    "rfm_3d":      rfm_3d_out,
}


# PRODUCTS & CATEGORIES
print(">> Computing product/category metrics...")


def build_products(data: pd.DataFrame) -> dict:
    pur = data[data["event_type"] == "purchase"]
    crt = data[data["event_type"] == "cart"]
    viw = data[data["event_type"] == "view"]

    tr = pur.groupby("category")["price"].sum().reset_index()
    tr.columns = ["label", "value"]
    treemap = {
        "labels": tr["label"].tolist(),
        "values": safe_list(tr["value"]),
    }

    top_brands_all = (
        pur.groupby("brand")["price"].sum().sort_values(ascending=False).head(20)
    )
    by_cat = {}
    for cat in pur["category"].unique():
        top = (
            pur[pur["category"] == cat]
            .groupby("brand")["price"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        if len(top):
            by_cat[cat] = {
                "brands":   top.index.tolist(),
                "revenues": safe_list(top.values),
            }

    brands = {
        "all_brands":   top_brands_all.index.tolist(),
        "all_revenues": safe_list(top_brands_all.values),
        "by_category":  by_cat,
    }

    top_cats = pur.groupby("category")["price"].sum().sort_values(ascending=False).head(8).index.tolist()
    cv = viw[viw["category"].isin(top_cats)].groupby("category").size()
    cc = crt[crt["category"].isin(top_cats)].groupby("category").size()
    cp = pur[pur["category"].isin(top_cats)].groupby("category").size()
    category_funnel = {
        "categories": top_cats,
        "views":      [si(cv.get(c, 0)) for c in top_cats],
        "carts":      [si(cc.get(c, 0)) for c in top_cats],
        "purchases":  [si(cp.get(c, 0)) for c in top_cats],
    }

    ap = pur.groupby("category")["price"].mean().sort_values(ascending=False).head(8)
    avg_price = {
        "categories": ap.index.tolist(),
        "prices":     safe_list(ap.values),
    }

    all_cats = viw["category"].unique()
    conv_rows = []
    for cat in all_cats:
        nv = len(viw[viw["category"] == cat])
        np_ = len(pur[pur["category"] == cat])
        if nv > 50:
            conv_rows.append({"category": cat, "rate": np_ / nv * 100})
    conv_df = pd.DataFrame(conv_rows).sort_values("rate", ascending=False).head(10)
    conversion_rate = {
        "categories": conv_df["category"].tolist(),
        "rates":      safe_list(conv_df["rate"]),
    }

    tp = (
        pur.groupby(["product_id", "category", "brand"])
        .agg(revenue=("price", "sum"), orders=("price", "count"), avg_price=("price", "mean"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(20)
    )
    top_products = []
    for _, row in tp.iterrows():
        top_products.append({
            "product_id": str(row["product_id"]),
            "category":   str(row["category"]),
            "brand":      str(row["brand"]),
            "revenue":    sf(row["revenue"]),
            "orders":     si(row["orders"]),
            "avg_price":  sf(row["avg_price"]),
        })

    return {
        "treemap":         treemap,
        "brands":          brands,
        "category_funnel": category_funnel,
        "avg_price":       avg_price,
        "conversion_rate": conversion_rate,
        "top_products":    top_products,
    }


products_out = {m: build_products(f) for m, f in frames.items()}


# USER BEHAVIOR
print(">> Computing user behavior metrics...")


def build_behavior(data: pd.DataFrame) -> dict:
    pur = data[data["event_type"] == "purchase"]
    viw = data[data["event_type"] == "view"]

    hm = pur.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
    for h in range(24):
        if h not in hm.columns:
            hm[h] = 0
    for w in WEEKDAYS:
        if w not in hm.index:
            hm.loc[w] = 0
    hm = hm.reindex(WEEKDAYS)[sorted(hm.columns)]
    heatmap = {
        "weekdays": WEEKDAYS,
        "hours":    list(range(24)),
        "z":        hm.values.tolist(),
    }

    hp = pur.groupby("hour").size().reindex(range(24), fill_value=0)
    hourly = {
        "hours":     list(range(24)),
        "purchases": safe_list(hp.values),
    }

    wp = pur.groupby("weekday").size()
    weekday = {
        "days":      WEEKDAYS,
        "purchases": [si(wp.get(d, 0)) for d in WEEKDAYS],
    }

    sd = (
        data.groupby("user_session")
        .size()
        .clip(upper=20)
        .value_counts()
        .sort_index()
    )
    session_depth = {
        "depth": safe_list(sd.index),
        "count": safe_list(sd.values),
    }

    hr = (
        pur.groupby("hour")["price"]
        .sum()
        .reindex(range(24), fill_value=0)
        .cumsum()
    )
    cumulative = {
        "hours":              list(range(24)),
        "cumulative_revenue": safe_list(hr.values),
    }

    daily_v = viw.groupby("date").size().rename("views")
    daily_p = pur.groupby("date").size().rename("purchases")
    ratio_df = pd.concat([daily_v, daily_p], axis=1).fillna(0).reset_index().sort_values("date")
    ratio_df["ratio"] = (
        ratio_df["purchases"]
        / ratio_df["views"].replace(0, np.nan)
        * 100
    )
    views_purchases_ratio = {
        "dates":  ratio_df["date"].tolist(),
        "ratios": safe_list(ratio_df["ratio"]),
    }

    return {
        "heatmap":               heatmap,
        "hourly":                hourly,
        "weekday":               weekday,
        "session_depth":         session_depth,
        "cumulative_revenue":    cumulative,
        "views_purchases_ratio": views_purchases_ratio,
    }


behavior_out = {m: build_behavior(f) for m, f in frames.items()}


# ML INTELLIGENCE
print(">> Training ML models...")

results, fitted, split = evaluate_churn_models(churn_ds)

roc_curves_out = {}
model_auc      = {}
model_pr_auc   = {}
for r in results:
    model_auc[r.name]    = sf(r.roc_auc)
    model_pr_auc[r.name] = sf(r.pr_auc)
    roc_curves_out[r.name] = {
        "fpr":    safe_list(r.fpr),
        "tpr":    safe_list(r.tpr),
        "auc":    sf(r.roc_auc),
        "pr_auc": sf(r.pr_auc),
    }
    print(f"   {r.name:<20} ROC-AUC = {r.roc_auc:.4f} | PR-AUC = {r.pr_auc:.4f}")

rf_clf = fitted["Random Forest"][-1]
fi_pairs = sorted(
    zip(CHURN_FEATURE_COLS, rf_clf.feature_importances_, strict=True),
    key=lambda x: x[1], reverse=True
)
feature_importance = {
    "features":    [f for f, _ in fi_pairs],
    "importances": safe_list([v for _, v in fi_pairs]),
}

# Score distribution on the holdout only, so it reflects unseen users.
gb_scores = next(r.test_scores for r in results if r.name == "Gradient Boosting")
hist_vals, hist_bins = np.histogram(gb_scores, bins=20, range=(0, 1))
churn_scores_out = {
    "bins":   safe_list(hist_bins[:-1]),
    "counts": safe_list(hist_vals),
}

top_cats = pur_all["category"].value_counts().head(8).index.tolist()
user_cat = (
    pur_all[pur_all["category"].isin(top_cats)]
    .groupby(["user_id", "category"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)
)
affinity_raw  = user_cat.T.dot(user_cat)
row_sums      = affinity_raw.sum(axis=1).replace(0, 1)
affinity_norm = (affinity_raw.div(row_sums, axis=0) * 100).round(1)
affinity_aligned = (
    affinity_norm
    .reindex(index=top_cats, columns=top_cats)
    .fillna(0)
)
affinity_matrix = {
    "categories": top_cats,
    "z":          affinity_aligned.values.tolist(),
}

viw_cnt     = df[df["event_type"] == "view"].groupby("category").size()
pur_cnt_all = pur_all.groupby("category").size()
upsell_df   = pd.DataFrame({"views": viw_cnt, "purchases": pur_cnt_all}).fillna(0)
upsell_df["score"] = (upsell_df["views"] - upsell_df["purchases"] * 8).clip(lower=0)
upsell_df = upsell_df.sort_values("score", ascending=False).head(10)
upsell_out = {
    "categories": upsell_df.index.tolist(),
    "scores":     safe_list(upsell_df["score"]),
}

seg_churn_risk = {
    "Champions":          "Low",
    "Loyal":              "Low",
    "Potential Loyalist": "Low",
    "At Risk":            "High",
    "Needs Attention":    "Medium",
    "Can't Lose":         "High",
    "Hibernating":        "High",
    "Lost":               "Very High",
}
seg_upsell_priority = {
    "Champions":          "High",
    "Loyal":              "High",
    "Potential Loyalist": "Medium",
    "At Risk":            "Medium",
    "Needs Attention":    "Low",
    "Can't Lose":         "Low",
    "Hibernating":        "Very Low",
    "Lost":               "Very Low",
}

segment_scores_out = [
    {
        "segment":         seg,
        "count":           si(seg_counts.get(seg, 0)),
        "churn_risk":      seg_churn_risk.get(seg, "Medium"),
        "upsell_priority": seg_upsell_priority.get(seg, "Low"),
    }
    for seg in seg_counts.index.tolist()
    if seg in seg_churn_risk
]

ml_out = {
    "roc_curves":         roc_curves_out,
    "feature_importance": feature_importance,
    "churn_scores":       churn_scores_out,
    "affinity_matrix":    affinity_matrix,
    "upsell":             upsell_out,
    "segment_scores":     segment_scores_out,
    "model_auc":          model_auc,
    "model_pr_auc":       model_pr_auc,
    "split":              split,
    "gb_auc":             model_auc.get("Gradient Boosting", 0),
    "rf_auc":             model_auc.get("Random Forest", 0),
    "lr_auc":             model_auc.get("Logistic Regression", 0),
    "recency_auc":        model_auc.get("Recency baseline", 0),
}

categories_list = sorted(
    [c for c in df[df["event_type"] == "purchase"]["category"].unique() if c != "unknown"]
)

# ASSEMBLE & SAVE
print(">> Saving dashboard_data.json...")

dashboard_data = {
    "metadata": {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sample":       "user-level sample (scripts/build_sample.py)",
        "total_events": total_rows,
        "total_users":  si(total_users),
        "date_range":   [df["date"].min(), df["date"].max()],
        "model_auc":    model_auc,
        "model_pr_auc": model_pr_auc,
    },
    "filters": {
        "months":     ["all", "october", "november"],
        "categories": categories_list,
    },
    "overview":  overview_out,
    "churn":     churn_out,
    "products":  products_out,
    "behavior":  behavior_out,
    "ml":        ml_out,
}

out_file = OUT_DIR / "dashboard_data.json"
with open(out_file, "w", encoding="utf-8") as fh:
    json.dump(dashboard_data, fh, separators=(",", ":"), allow_nan=False)

size_mb = out_file.stat().st_size / 1_048_576
print(f"OK  Saved -> {out_file}")
print(f"    File size : {size_mb:.2f} MB")
print(f"    Churn rate: {churn_out['churn_rate']:.1f}%")
print(f"    GB AUC    : {model_auc.get('Gradient Boosting', 0):.4f}")
