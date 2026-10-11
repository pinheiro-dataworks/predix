<div align="center">
  <img src="asset/Logo_01.png" alt="Predix — Ecommerce Intelligence Platform" width="420" />
</div>

<div align="center">

**Predix** is a churn prediction, upsell scoring and RFM segmentation project built on
two months of e-commerce behavioural events.

[![CI](https://github.com/pinheiro-dataworks/predix/actions/workflows/ci.yml/badge.svg)](https://github.com/pinheiro-dataworks/predix/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?style=flat-square)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## Scope: what runs here and what is design

The repository contains two kinds of code. Keeping them apart is important when you read the numbers below.

### Runs locally

Everything in this section has been executed end to end on a **user-level sample of 4,045,612 events** (197,295 users, 3.7% of all users, every event of each sampled user for October and November 2019).

| Component | Entry point | What it produces |
|---|---|---|
| Sampling | `scripts/build_sample.py` | `dataset/sample_events.parquet` (deterministic hash of `user_id`) |
| Churn target and features | `pipeline/feature_store/churn_dataset.py` | One row per eligible user, point-in-time features plus label |
| Model evaluation | `scripts/evaluate_churn.py` | `reports/churn_metrics.json` |
| Training and registry | `pipeline/models/train_and_register.py` | MLflow runs and registered champion (SQLite backend), churn, RFM and upsell score files |
| Scoring API | `api/main.py` (FastAPI) | Serves the score files written by the training step |
| Dashboard data | `scripts/generate_dashboard_data.py` | `dashboard/data/dashboard_data.json` |
| Static dashboard | `dashboard/` | Plotly.js site, deployed on Vercel from `main` |
| Tests and lint | `pytest`, `ruff` | Run in GitHub Actions on Python 3.11 and 3.12 |

### Designed, not executed in this repo

These parts describe how the pipeline would run in production. They are kept consistent with the local code (same churn definition, same feature names), but **none of them has been run in this repository**, and no performance or scale figure in this README comes from them.

| Component | Location | Status |
|---|---|---|
| Airflow orchestration (4 DAGs) | `airflow/dags/` | Syntax-checked only; no Airflow deployment |
| Snowflake + dbt medallion models | `dbt/models/` | Not compiled or run; no Snowflake account is involved |
| PySpark ingestion to Delta Lake | `pipeline/ingestion/` | Not run; local processing uses pandas and pyarrow |
| Full-dataset scale | — | The raw files hold 109,950,743 events (~14.7 GB). Models have only been trained on the ~4M-event sample |
| Streaming, Kubernetes, feature store | `requirements-pipeline.txt` | Listed as target technologies only |

---

## Churn definition

Churn is defined relative to a cutoff date `as_of` (default `2019-11-01`) and a prediction horizon of `N` days (default 30, configurable):

```
              observation window                    prediction window
   ───────────────────────────────────────┼──────────────────────────────┤
 2019-10-01                             as_of                       as_of + N
   features: only events before as_of     label: any purchase here?
```

- **Eligible population:** users with at least one purchase before `as_of`.
- **Label:** `churned = 1` when an eligible user makes no purchase in `[as_of, as_of + N)`.
- **Features (13):** recency, frequency, monetary, average price, distinct categories, brands and products, views, carts, cart-to-purchase ratio, sessions, active days, tenure. All are computed from events strictly before `as_of`.
- **Guards:** `build_features` refuses any event at or after `as_of`. `build_churn_dataset` raises if the data ends before the prediction window does, because a truncated window would label active users as churned.

`tests/test_temporal_integrity.py` removes, injects and edits post-cutoff events and requires every feature to be unchanged. Two deliberately injected leaks fail 17 and 10 of its 19 tests respectively.

An earlier version read the first 2M rows of each monthly file. That covers only October 1–2 and November 1–2, so "churn" meant "bought on Oct 1–2 but not on Nov 1–2" and the reported churn rate was 95.3%.

---

## Results (local sample)

`as_of = 2019-11-01`, `N = 30` days, stratified 80/20 split by user (`random_state = 42`). Source: [`reports/churn_metrics.json`](reports/churn_metrics.json).

| | Value |
|---|---|
| Eligible users | 12,691 |
| Churn rate | **73.4%** |
| Train / test users | 10,152 / 2,539 |

| Model | ROC-AUC (test) | PR-AUC (test) | 5-fold CV ROC-AUC (train) |
|---|---|---|---|
| Recency baseline (days since last purchase, no training) | 0.595 | 0.797 | — |
| Logistic Regression | **0.657** | 0.818 | 0.668 ± 0.016 |
| Random Forest | 0.646 | 0.817 | 0.666 ± 0.017 |
| Gradient Boosting | 0.653 | **0.819** | 0.668 ± 0.015 |

How to read this:

- A random score has PR-AUC equal to the churn rate (0.734), not 0.5. The models add about 0.085 PR-AUC over random and about 0.02 over the recency baseline.
- The three learned models are within cross-validation noise of each other. Gradient Boosting is registered as champion by configuration, not because it clearly wins.
- The signal is modest. With only one month of history before the cutoff, 62% of eligible users have a single purchase, which limits what any model can learn.

---

## Architecture

The diagram shows the target design. Solid boxes run locally; boxes marked *(design)* are described in the previous section.

```
Raw CSVs (Oct + Nov 2019, ~110M events)
    │
    ├──► scripts/build_sample.py ──► sample_events.parquet (~4M events)        [runs]
    │         │
    │         ├──► churn_dataset.py ──► evaluate_churn.py ──► churn_metrics.json  [runs]
    │         ├──► train_and_register.py ──► MLflow (SQLite) + score files     [runs]
    │         │         └──► FastAPI /api/v1/score/churn/{uid} …               [runs]
    │         └──► generate_dashboard_data.py ──► static dashboard (Vercel)    [runs]
    │
    └──► PySpark ingestion ──► dbt Bronze/Silver/Gold on Snowflake
              └──► Airflow DAGs: features ─► training ─► daily batch scoring   (design)
```

### Why these technologies

- **pandas + pyarrow (local):** the 4M-event sample fits in memory. Streaming the raw CSVs with pyarrow builds the sample in about 80 seconds.
- **scikit-learn:** logistic regression, random forest and gradient boosting, all wrapped in pipelines, with a recency baseline as the minimum bar.
- **MLflow:** experiment tracking and model registry. A SQLite backend is used because MLflow 3 no longer accepts the `./mlruns` file store.
- **FastAPI:** serves pre-computed batch scores from Parquet, with Pydantic response models.
- **Plotly.js on a static site:** the dashboard is pre-computed JSON plus HTML, so there is no server to run.
- **Airflow, dbt, Snowflake, PySpark (design):** the intended production stack for scheduling, SQL transformations with tests and lineage, and distributed processing of the full dataset.

---

## Quickstart

Requires Python 3.11+ and the two raw CSVs from Kaggle in `dataset/` (see [Dataset](#dataset)).

```bash
pip install -r requirements-dev.txt

# 1. Build the ~4M-event user-level sample (~80 s)
python scripts/build_sample.py

# 2. Evaluate the churn models against the baselines
python scripts/evaluate_churn.py                    # as_of=2019-11-01, N=30
python scripts/evaluate_churn.py --horizon-days 14  # any other horizon

# 3. Train, register the champion in MLflow and write score files
python pipeline/models/train_and_register.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional

# 4. Serve the scores
uvicorn api.main:app --port 8000
curl http://localhost:8000/health

# 5. Regenerate and validate the dashboard data
python scripts/generate_dashboard_data.py
python scripts/check_json.py
cd dashboard && python -m http.server 3000          # http://localhost:3000

# Tests and lint
pytest
ruff check .
```

---

## Dashboard

| Tab | Content |
|---|---|
| **Overview** | Daily events, funnel, event split, category and daily revenue |
| **Churn Intelligence** | Churn rate with its definition, eligible and retained buyers, cohort before and after the cutoff, RFM maps |
| **Products & Categories** | Revenue treemap, top brands, category funnel, conversion rates, top products |
| **User Behavior** | Hour × weekday heatmap, hourly and weekday curves, session depth |
| **ML Intelligence** | ROC curves on the holdout set (recency baseline and three models), ROC-AUC and PR-AUC, feature importance, score distribution, category affinity, upsell scores |
| **Architecture** | Target pipeline design, model cards, feature store schema, DAG and dbt layer descriptions |

---

## Project structure

```
Predix/
├── .github/workflows/ci.yml      ← ruff + pytest (Python 3.11 / 3.12)
├── api/main.py                   ← FastAPI scoring service
├── dashboard/                    ← static site (Vercel)
├── dataset/                      ← raw CSVs and sample (not versioned)
├── pipeline/
│   ├── feature_store/
│   │   ├── churn_dataset.py      ← churn target + point-in-time features
│   │   └── feature_engineering.py← event cleaning, RFM, user/product/session tables
│   ├── models/
│   │   ├── churn_evaluation.py   ← baselines vs models, ROC-AUC / PR-AUC
│   │   ├── churn_model.py        ← MLflow-tracked training and registration
│   │   ├── upsell_model.py       ← rule-based opportunity + affinity scores
│   │   └── train_and_register.py ← local training entrypoint
│   ├── scoring/batch_scoring.py
│   └── ingestion/                ← PySpark ingestion (design)
├── airflow/dags/                 ← orchestration (design)
├── dbt/models/                   ← Snowflake medallion models (design)
├── reports/churn_metrics.json    ← versioned evaluation results
├── scripts/                      ← sampling, evaluation, dashboard export
└── tests/                        ← temporal integrity, API contract, transformations
```

---

## Limitations

- **One cutoff, two months of data.** Results come from a single `as_of`. A rolling, out-of-time evaluation would need a longer history.
- **Sample, not full data.** All metrics are from 3.7% of users. The full 110M-event dataset has not been processed.
- **Churn risk tiers** in the API use fixed probability thresholds (0.35 / 0.60). With a 73% base rate, most users fall into the High tier, so the thresholds should be recalibrated before any campaign use.
- **Upsell scoring is rule-based** (view/purchase gap, co-purchase affinity, RFM weight). It is not a trained model and has no offline evaluation.

---

## Dataset

**Source:** [Kaggle — eCommerce behavior data from multi category store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)
**Files used:** `2019-Oct.csv` (42,448,764 events) and `2019-Nov.csv` (67,501,979 events), ~14.7 GB in total.
**Schema:** `event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session`

---

## License

[MIT](LICENSE) © Renan Pinheiro
