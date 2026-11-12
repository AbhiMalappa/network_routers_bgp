"""Scan every raw DQL parquet pull for device-coverage anomalies.

Catches the pattern found manually in the Sep 2026 pull: a subset of queries
(2a_i/2a_ii/2c) silently collapsing to a handful of devices for several days
while every other query in the same pull kept full fleet coverage. For each
(pull, query), each day's device count is compared against that SAME query's
own median device count across the whole pull — not against other queries'
counts, since some queries (e.g. fan/temperature/fan-tray status) only ever
apply to a subset of device models and are legitimately low every day. A
within-query day-over-own-baseline drop is what actually signals a gap. Any
query/day whose device count falls below --threshold of that query's own
pull-median is flagged.

Usage:
    python data_pipeline/audit_dql_source_coverage.py
    python data_pipeline/audit_dql_source_coverage.py --threshold 0.5 --out data_pipeline/processed/dql_coverage_gaps.csv
"""

import argparse
import glob
import re
import os

import pandas as pd

# 2026-11-10: moved into data_pipeline/ (QA tool for this pipeline's own
# raw Dynatrace pulls) and switched from CWD-relative strings to
# __file__-based paths pointing at data_pipeline/raw, data_pipeline/
# processed -- the old relative strings only worked when run from the
# project root as cwd.
PIPELINE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(PIPELINE_DIR, "processed"), exist_ok=True)
RAW_PARQUET_DIR = os.path.join(PIPELINE_DIR, "raw", "dql", "parquet")
DEFAULT_OUT = os.path.join(PIPELINE_DIR, "processed", "dql_coverage_gaps.csv")

ENTITY_COL_CANDIDATES = ("dt.entity.network:device",)
TIME_COL_CANDIDATES = ("time",)


def _find_col(columns, candidates, contains_fallback):
    for c in candidates:
        if c in columns:
            return c
    for c in columns:
        if contains_fallback in c.lower():
            return c
    raise ValueError(f"No column matching {candidates} or containing '{contains_fallback}' in {list(columns)}")


def per_day_device_counts(path: str) -> pd.Series:
    df = pd.read_parquet(path)
    entity_col = _find_col(df.columns, ENTITY_COL_CANDIDATES, "entity")
    time_col = _find_col(df.columns, TIME_COL_CANDIDATES, "time")
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df["date"] = df[time_col].dt.date
    return df.groupby("date")[entity_col].nunique()


def audit(raw_dir: str, threshold: float) -> pd.DataFrame:
    pull_dirs = sorted(d for d in glob.glob(os.path.join(raw_dir, "*")) if os.path.isdir(d))
    rows = []
    for pull_dir in pull_dirs:
        pull_name = os.path.basename(pull_dir)
        files = sorted(glob.glob(os.path.join(pull_dir, "*.parquet")))
        if not files:
            continue

        for f in files:
            query_name = re.sub(r"_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}\.parquet$", "", os.path.basename(f))
            try:
                col = per_day_device_counts(f)
            except Exception as exc:
                print(f"  [skip] {pull_name}/{os.path.basename(f)}: {exc}")
                continue

            own_median = col.median()
            if own_median == 0:
                continue
            ratio = col / own_median
            flagged = ratio[ratio < threshold]
            for date, r in flagged.items():
                rows.append(
                    {
                        "pull": pull_name,
                        "query": query_name,
                        "date": date,
                        "device_count": int(col.loc[date]),
                        "query_own_pull_median_device_count": own_median,
                        "ratio_vs_own_median": round(r, 4),
                    }
                )

    return pd.DataFrame(rows).sort_values(["pull", "date", "query"]) if rows else pd.DataFrame(
        columns=["pull", "query", "date", "device_count", "query_own_pull_median_device_count", "ratio_vs_own_median"]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-dir", default=RAW_PARQUET_DIR)
    ap.add_argument("--threshold", type=float, default=0.5, help="Flag query/day if device_count < threshold * that query's own median device count across the whole pull")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    result = audit(args.raw_dir, args.threshold)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.to_csv(args.out, index=False)

    print(f"\n{len(result)} query/day coverage anomalies found (threshold={args.threshold}).")
    if len(result):
        print(f"Affected pulls: {sorted(result['pull'].unique())}")
        print(f"Affected queries: {sorted(result['query'].unique())}")
        print(f"Date range: {result['date'].min()} to {result['date'].max()}")
    print(f"Full detail written to {args.out}")


if __name__ == "__main__":
    main()
