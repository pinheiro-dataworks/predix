"""
Predix — PySpark Ingestion Pipeline
────────────────────────────────────────────────────────────────────────────────
Reads raw e-commerce event CSVs from the landing zone, applies schema
enforcement and data quality checks, and writes Bronze-layer Parquet to
Snowflake-staged storage (or local Delta Lake for development).

Production run (Spark submit):
    spark-submit \
      --master spark://cluster:7077 \
      --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
      pipeline/ingestion/pyspark_ingestion.py \
      --env prod --month 2019-11
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────
RAW_SCHEMA = T.StructType([
    T.StructField("event_time",    T.StringType(),  True),
    T.StructField("event_type",    T.StringType(),  True),
    T.StructField("product_id",    T.LongType(),    True),
    T.StructField("category_id",   T.LongType(),    True),
    T.StructField("category_code", T.StringType(),  True),
    T.StructField("brand",         T.StringType(),  True),
    T.StructField("price",         T.DoubleType(),  True),
    T.StructField("user_id",       T.LongType(),    True),
    T.StructField("user_session",  T.StringType(),  True),
])

VALID_EVENT_TYPES = {"view", "cart", "purchase"}


# ── SparkSession ──────────────────────────────────────────────────────────────
def build_spark(app_name: str = "predix_ingestion", env: str = "dev") -> SparkSession:
    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )
    if env == "prod":
        builder = (
            builder
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config(
                "spark.sql.catalog.spark_catalog",
                "org.apache.spark.sql.delta.catalog.DeltaCatalog",
            )
        )
    return builder.getOrCreate()


# ── Read raw ──────────────────────────────────────────────────────────────────
def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    log.info("Reading raw CSV: %s", path)
    return (
        spark.read
        .option("header", "true")
        .option("mode",   "PERMISSIVE")
        .schema(RAW_SCHEMA)
        .csv(path)
    )


# ── Data quality checks ───────────────────────────────────────────────────────
def run_dq_checks(df: DataFrame) -> tuple[DataFrame, dict]:
    total      = df.count()
    null_uid   = df.filter(F.col("user_id").isNull()).count()
    null_price = df.filter(F.col("price").isNull() | (F.col("price") < 0)).count()
    bad_events = df.filter(~F.col("event_type").isin(list(VALID_EVENT_TYPES))).count()
    dup_keys   = total - df.dropDuplicates(["user_id", "event_time", "user_session"]).count()

    metrics = {
        "total_rows":          total,
        "null_user_id":        null_uid,
        "negative_price":      null_price,
        "invalid_event_type":  bad_events,
        "duplicate_events":    dup_keys,
        "valid_rows":          total - null_uid - bad_events,
        "null_user_pct":       round(null_uid / max(total, 1) * 100, 2),
    }
    log.info("DQ results: %s", metrics)
    return df, metrics


# ── Transform ─────────────────────────────────────────────────────────────────
def transform_bronze(df: DataFrame, month: str | None = None) -> DataFrame:
    return (
        df
        # Parse timestamp (UTC)
        .withColumn(
            "event_time",
            F.to_timestamp(F.regexp_replace(F.col("event_time"), " UTC$", ""))
        )
        # Normalise event type
        .withColumn("event_type", F.lower(F.trim(F.col("event_type"))))
        # Filter valid event types only
        .filter(F.col("event_type").isin(list(VALID_EVENT_TYPES)))
        # Null guard for user_id
        .filter(F.col("user_id").isNotNull())
        # Clean price (cap at 99th percentile or mark as 0 if null)
        .withColumn(
            "price",
            F.when(F.col("price").isNull() | (F.col("price") < 0), F.lit(0.0))
            .otherwise(F.col("price")),
        )
        # Derive category_root and sub_category
        .withColumn("category_root", F.split(F.col("category_code"), r"\.").getItem(0))
        .withColumn("category_sub",  F.split(F.col("category_code"), r"\.").getItem(1))
        # Add partition columns
        .withColumn("event_date",  F.to_date(F.col("event_time")))
        .withColumn("event_month", F.date_format(F.col("event_time"), "yyyy-MM"))
        # Ingestion metadata
        .withColumn("_ingested_at", F.lit(datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")))
        .withColumn("_source", F.input_file_name())
        # Remove exact duplicates
        .dropDuplicates(["user_id", "user_session", "event_time", "event_type", "product_id"])
    )


# ── Write Bronze ──────────────────────────────────────────────────────────────
def write_bronze(df: DataFrame, output_path: str, env: str = "dev") -> None:
    log.info("Writing Bronze to: %s (env=%s)", output_path, env)
    writer = (
        df.repartition(32, "event_date")
        .write
        .partitionBy("event_month", "event_date")
        .mode("append")
    )
    if env == "prod":
        writer.format("delta").save(output_path)
    else:
        writer.format("parquet").save(output_path)
    log.info("Bronze write complete.")


# ── Main ──────────────────────────────────────────────────────────────────────
def run(input_path: str, output_path: str, env: str = "dev", month: str | None = None) -> dict:
    spark = build_spark(env=env)
    spark.sparkContext.setLogLevel("WARN")

    raw   = read_raw_csv(spark, input_path)
    raw, dq = run_dq_checks(raw)
    clean = transform_bronze(raw, month=month)

    write_bronze(clean, output_path, env=env)

    out_count = clean.count()
    log.info("Ingestion complete: %d valid rows written to Bronze.", out_count)
    spark.stop()
    return {**dq, "written_rows": out_count}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predix — PySpark Bronze Ingestion")
    parser.add_argument("--input",  required=True, help="Path to raw CSV (glob supported)")
    parser.add_argument("--output", required=True, help="Bronze output path")
    parser.add_argument("--env",    default="dev", choices=["dev", "prod"])
    parser.add_argument("--month",  default=None,  help="YYYY-MM filter (optional)")
    args = parser.parse_args()

    result = run(args.input, args.output, args.env, args.month)
    log.info("Final ingestion metrics: %s", result)
