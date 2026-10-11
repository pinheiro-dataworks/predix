-- Gold: churn_features
-- ──────────────────────────────────────────────────────────────────────────────
-- ML-ready feature table for the churn prediction model. SQL counterpart of
-- pipeline/feature_store/churn_dataset.py; both must stay in sync.
--
--   observation window   event_time <  churn_as_of            → features
--   prediction window    [churn_as_of, churn_as_of + N days)  → label
--   eligible population  users with >= 1 purchase in the observation window
--   churned = 1          no purchase in the prediction window
--
-- Features are computed from `observation` only. RFM scores from
-- customer_rfm are deliberately not joined: they are computed over the full
-- history, which would leak post-cutoff purchases into the features.
-- ──────────────────────────────────────────────────────────────────────────────

{{ config(
    materialized = 'table',
    schema       = 'gold',
    tags         = ['gold', 'features', 'ml', 'churn'],
) }}

{% set as_of   = "TO_TIMESTAMP('" ~ var('churn_as_of') ~ "')" %}
{% set horizon = var('churn_window_days') %}

WITH observation AS (
    SELECT *
    FROM {{ ref('events_cleaned') }}
    WHERE event_time < {{ as_of }}
),

prediction_buyers AS (
    SELECT DISTINCT user_id
    FROM {{ ref('events_cleaned') }}
    WHERE event_type = 'purchase'
      AND event_time >= {{ as_of }}
      AND event_time <  DATEADD('day', {{ horizon }}, {{ as_of }})
),

purchases AS (
    SELECT
        user_id,
        COUNT(*)                                                    AS frequency,
        SUM(price)                                                  AS monetary,
        AVG(price)                                                  AS avg_price,
        MAX(event_time)                                             AS last_purchase,
        COUNT(DISTINCT COALESCE(NULLIF(category_root, ''), 'unknown')) AS unique_cats,
        COUNT(DISTINCT brand)                                       AS unique_brands,
        COUNT(DISTINCT product_id)                                  AS unique_prods
    FROM observation
    WHERE event_type = 'purchase'
    GROUP BY user_id
),

activity AS (
    SELECT
        user_id,
        COUNT_IF(event_type = 'view')       AS n_views,
        COUNT_IF(event_type = 'cart')       AS n_carts,
        COUNT(DISTINCT user_session)        AS n_sessions,
        COUNT(DISTINCT event_time::DATE)    AS active_days,
        MIN(event_time)                     AS first_event
    FROM observation
    GROUP BY user_id
)

SELECT
    p.user_id,
    p.frequency,
    p.monetary,
    p.avg_price,
    DATEDIFF('second', p.last_purchase, {{ as_of }}) / 86400.0  AS recency,
    p.unique_cats,
    p.unique_brands,
    p.unique_prods,
    a.n_views,
    a.n_carts,
    a.n_carts / p.frequency                                     AS cart_to_pur_ratio,
    a.n_sessions,
    a.active_days,
    DATEDIFF('second', a.first_event, {{ as_of }}) / 86400.0    AS tenure_days,
    CASE WHEN pb.user_id IS NULL THEN 1 ELSE 0 END              AS churned,
    {{ as_of }}                                                 AS as_of,
    CURRENT_TIMESTAMP()                                         AS computed_at

FROM purchases p
JOIN activity               a  ON p.user_id = a.user_id
LEFT JOIN prediction_buyers pb ON p.user_id = pb.user_id
