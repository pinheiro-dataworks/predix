#!/usr/bin/env python3
"""
Predix - build_sample.py
Builds a reproducible, user-level sample of the raw October/November CSVs.

Sampling is done on users, not on rows: a user is either fully in the sample
(every event of both months) or fully out of it. Row-level sampling, or
reading the first N rows of each file, truncates purchase histories and
makes any time-window target (churn, recency) meaningless.

Users are selected with a deterministic hash of `user_id`, so the same
fraction always yields the same users regardless of file order.

Usage (from project root):
    python scripts/build_sample.py                      # ~3.7% of users (~4M events)
    python scripts/build_sample.py --fraction 0.10 --out dataset/sample_10pct.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILES = ["2019-Oct.csv", "2019-Nov.csv"]
HASH_BUCKETS = 10_000

COLUMN_TYPES = {
    "event_time": pa.string(),
    "event_type": pa.string(),
    "product_id": pa.int64(),
    "category_id": pa.int64(),
    "category_code": pa.string(),
    "brand": pa.string(),
    "price": pa.float64(),
    "user_id": pa.int64(),
    "user_session": pa.string(),
}


def user_bucket(user_ids: np.ndarray) -> np.ndarray:
    """Map user ids to [0, HASH_BUCKETS) with a multiplicative (Fibonacci) hash.

    Plain `user_id % k` would only be safe if ids were assigned at random; ids
    that grow with signup time would couple the sample to user tenure. Hashing
    removes that dependency.
    """
    mixed = user_ids.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    return ((mixed >> np.uint64(32)) % np.uint64(HASH_BUCKETS)).astype(np.int64)


def build_sample(data_dir: Path, files: list[str], fraction: float, out_path: Path) -> int:
    threshold = int(round(fraction * HASH_BUCKETS))
    if not 0 < threshold <= HASH_BUCKETS:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(COLUMN_TYPES.items())
    convert = pv.ConvertOptions(column_types=COLUMN_TYPES, strings_can_be_null=True)
    read = pv.ReadOptions(block_size=64 << 20)

    kept = 0
    with pq.ParquetWriter(out_path, schema, compression="zstd") as writer:
        for name in files:
            started, seen = time.time(), 0
            reader = pv.open_csv(data_dir / name, read_options=read, convert_options=convert)
            for batch in reader:
                seen += batch.num_rows
                uids = batch.column("user_id").to_numpy(zero_copy_only=False)
                mask = user_bucket(uids) < threshold
                if mask.any():
                    table = pa.Table.from_batches([batch]).filter(pa.array(mask))
                    writer.write_table(table.cast(schema))
                    kept += table.num_rows
            print(f"   {name}: {seen:,} rows scanned in {time.time() - started:,.0f}s")
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a user-level sample of the raw events.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "dataset")
    parser.add_argument("--fraction", type=float, default=0.037, help="share of users to keep")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "sample_events.parquet")
    args = parser.parse_args()

    print(f">> Sampling {args.fraction:.1%} of users from {args.data_dir}")
    kept = build_sample(args.data_dir, DEFAULT_FILES, args.fraction, args.out)
    print(f"OK  {kept:,} events written to {args.out}")


if __name__ == "__main__":
    main()
