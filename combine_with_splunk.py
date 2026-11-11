"""
BGP flap prediction -- data pipeline, stage 5: combine (Dynatrace+SNOW) + Splunk.

Left-joins the Splunk-derived log feature table (log_features_wide.parquet,
from build_log_features.py) and device platform/tier (log_device_platform.csv)
onto combine_dynatrace_snow.py's output, resolving each Dynatrace device to
its Splunk-side key first. Also one-hot encodes device_platform/device_tier
(top-10 + "Others" binning) and router_type/network_type (already present
on the Dynatrace side from build_and_clean_dynatrace_device_data.py's
metadata join) -- kept one-hot here, unlike the SNOW columns, per an
explicit decision: platform/tier/router_type/network_type stay one-hot
encoded same as the pre-existing pipeline; only the SNOW columns
(haiku_root_cause/predictability/subcategory) are deferred to plain strings.

Logic reused verbatim from the old build_features.py's build_splunk_key_pool/
resolve_splunk_keys/merge_log_features/merge_platform/merge_router_network_type
-- same two-tier Splunk-key resolution (resolved device name first, a
dns_names alias fallback only if exactly one alias matches), same NaN-vs-0
discipline (devices with log coverage but no activity in a bucket get 0;
devices with no log coverage or no resolvable Splunk key at all get NaN --
"confirmed zero" vs "unknown" are not the same thing).

No lag/trend/z-score/volatility computed here -- that's stage 6
(build_secondary_features.py), applied uniformly over the fully combined
table so Dynatrace, SNOW, and Splunk-derived columns all go through the
same rolling-window machinery in one place.

Input: data/processed/dynatrace_snow_combined.parquet,
       data/processed/log_features_wide.parquet,
       data/processed/log_device_coverage.csv,
       data/processed/log_device_platform.csv
Output: data/processed/dynatrace_snow_splunk_combined.parquet
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_device_coverage import normalize_device  # noqa: E402

# 2026-11-10: raw/processed moved to data_pipeline/raw, data_pipeline/
# processed (self-contained pipeline).
PIPELINE_DIR = os.path.dirname(__file__)
IN_PATH = os.path.join(PIPELINE_DIR, "processed", "dynatrace_snow_combined.parquet")
LOG_FEATURES_PATH = os.path.join(PIPELINE_DIR, "processed", "log_features_wide.parquet")
LOG_COVERAGE_PATH = os.path.join(PIPELINE_DIR, "processed", "log_device_coverage.csv")
LOG_PLATFORM_PATH = os.path.join(PIPELINE_DIR, "processed", "log_device_platform.csv")
OUT_PATH = os.path.join(PIPELINE_DIR, "processed", "dynatrace_snow_splunk_combined.parquet")

ENTITY_COL = "dt.entity.network:device"
IDENTITY_COL = "device_name_resolved"
PLATFORM_TOP_N = 10  # matches the original Splunk-side `top limit=10` cutoff


def build_splunk_key_pool() -> set:
    """Union of normalized device names across the three Splunk-derived
    tables -- the target pool for resolving each Dynatrace device's
    identity onto a Splunk-side key."""
    log_wide = pd.read_parquet(LOG_FEATURES_PATH, columns=["sys.name"])
    coverage = pd.read_csv(LOG_COVERAGE_PATH)
    platform = pd.read_csv(LOG_PLATFORM_PATH)
    pool = set()
    pool |= set(log_wide["sys.name"].map(normalize_device))
    pool |= set(coverage["sys_name_lower"].map(normalize_device))
    pool |= set(platform["sys_name_lower"].map(normalize_device))
    return pool


def resolve_splunk_keys(devices: pd.DataFrame, splunk_pool: set) -> pd.Series:
    """devices: one row per unique ENTITY_COL, with IDENTITY_COL and
    dt.dns_names. Returns a Series indexed by ENTITY_COL -> normalized
    Splunk key (or NaN if no match). Same two-tier priority rule as
    build_device_coverage.py's load_dynatrace_directory(): try the
    resolved device name first, fall back to a dns_names alias only if
    exactly one alias matches -- never guess among multiple candidates."""
    resolved = {}
    n_primary = n_fallback = n_unmatched = n_ambiguous = 0
    for _, row in devices.iterrows():
        entity_id = row[ENTITY_COL]
        primary = normalize_device(row[IDENTITY_COL]) if pd.notna(row[IDENTITY_COL]) else None
        if primary is not None and primary in splunk_pool:
            resolved[entity_id] = primary
            n_primary += 1
            continue

        dns_names = row.get("dt.dns_names")
        alias_matches = set()
        if pd.notna(dns_names):
            alias_keys = {normalize_device(a.strip()) for a in dns_names.split(",")}
            alias_matches = alias_keys & splunk_pool

        if len(alias_matches) == 1:
            resolved[entity_id] = next(iter(alias_matches))
            n_fallback += 1
        else:
            resolved[entity_id] = None
            n_unmatched += 1
            if len(alias_matches) > 1:
                n_ambiguous += 1

    print(f"Splunk key resolution: {n_primary} matched directly on {IDENTITY_COL}, "
          f"{n_fallback} matched via a unique dns_names alias fallback, "
          f"{n_unmatched} unmatched ({n_ambiguous} of those had 2+ ambiguous alias "
          f"candidates, left unresolved rather than guessed)")
    return pd.Series(resolved, name="_splunk_key")


def merge_log_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Left-joins the log-derived wide feature table onto df (resolved
    Splunk key, time_utc), returning the merged frame and the list of log
    feature column names added. Devices with log coverage but no matching
    bucket get 0 (confirmed no activity); devices with no log coverage at
    all, OR no resolvable Splunk key at all, get NaN (unknown)."""
    log_wide = pd.read_parquet(LOG_FEATURES_PATH)
    log_wide["_splunk_key"] = log_wide["sys.name"].map(normalize_device)
    log_wide = log_wide.drop(columns=["sys.name"])
    log_cols = [c for c in log_wide.columns if c not in ("_splunk_key", "time_utc")]

    coverage = pd.read_csv(LOG_COVERAGE_PATH)
    coverage["_splunk_key"] = coverage["sys_name_lower"].map(normalize_device)
    covered_keys = set(coverage.loc[coverage["has_log_coverage"], "_splunk_key"])

    df = df.merge(log_wide, on=["_splunk_key", "time_utc"], how="left")

    is_covered = df["_splunk_key"].isin(covered_keys)
    for c in log_cols:
        df[c] = df[c].where(~(is_covered & df[c].isna()), 0)
        # rows for uncovered (or unresolved) devices stay NaN (no fillna applied there)
    return df, log_cols


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def _one_hot_known(df: pd.DataFrame, raw_col: str, prefix: str, known_col: str) -> tuple[pd.DataFrame, list[str]]:
    """One-hots df[raw_col] (already binned/finalized) with a `{prefix}_`
    naming scheme, plus a `{known_col}` flag distinguishing "confirmed not
    any of the one-hot categories" (all-zero one-hot, known=1) from "no
    info at all" (all-zero one-hot AND known=0)."""
    sanitized = df[raw_col].apply(lambda v: f"{prefix}_{_sanitize(v)}" if pd.notna(v) else np.nan)
    dummies = pd.get_dummies(sanitized).astype(int)
    cols = list(dummies.columns)
    df = pd.concat([df, dummies], axis=1)
    df[known_col] = df[raw_col].notna().astype(int)
    return df, cols + [known_col]


def merge_platform(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Static (device, not time) join of hardware platform + tier, one-hot
    encoded. Top-N-by-DEVICE-count-plus-"Others" binning for platform
    happens HERE, over devices actually present in this table (post
    Dynatrace<->Splunk key resolution) -- not server-side or baked into an
    earlier stage, so "top N" reflects the real modeled population."""
    platform = pd.read_csv(LOG_PLATFORM_PATH)
    platform["_splunk_key"] = platform["sys_name_lower"].map(normalize_device)
    platform = platform.drop_duplicates(subset=["_splunk_key"], keep="first")

    df = df.merge(platform[["_splunk_key", "device_platform", "device_tier"]], on="_splunk_key", how="left")

    device_platforms = df.drop_duplicates(subset=[ENTITY_COL])["device_platform"]
    top_platforms = set(device_platforms.value_counts().head(PLATFORM_TOP_N).index)
    print(f"Platform binning: {device_platforms.nunique()} distinct platforms among devices in this table, "
          f"top {PLATFORM_TOP_N} kept as their own category, rest binned to 'Others': {sorted(top_platforms)}")

    df["_platform_binned"] = df["device_platform"].where(
        df["device_platform"].isin(top_platforms) | df["device_platform"].isna(),
        "Others",
    )

    df, platform_cols = _one_hot_known(df, "_platform_binned", "platform", "platform_known")
    df, tier_cols = _one_hot_known(df, "device_tier", "tier", "tier_known")

    df = df.drop(columns=["device_platform", "device_tier", "_platform_binned"])
    return df, platform_cols + tier_cols


def merge_router_network_type(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One-hot encodes router_type/network_type -- raw string columns
    already present in df from the Dynatrace device-metadata join, joined
    on the entity ID (no Splunk-key resolution risk)."""
    df, rtype_cols = _one_hot_known(df, "router_type", "router_type", "router_type_known")
    df, ntype_cols = _one_hot_known(df, "network_type", "network_type", "network_type_known")
    return df, rtype_cols + ntype_cols


def main():
    df = pd.read_parquet(IN_PATH)

    devices = df[[ENTITY_COL, IDENTITY_COL, "dt.dns_names"]].drop_duplicates(subset=[ENTITY_COL])
    splunk_pool = build_splunk_key_pool()
    splunk_key_by_device = resolve_splunk_keys(devices, splunk_pool)
    df["_splunk_key"] = df[ENTITY_COL].map(splunk_key_by_device)

    df, log_cols = merge_log_features(df)
    df, platform_cols = merge_platform(df)
    df, router_network_type_cols = merge_router_network_type(df)
    df = df.drop(columns=["_splunk_key"])

    print(f"\nRows: {len(df)}  Devices: {df[ENTITY_COL].nunique()}")
    print(f"Log-derived columns added: {len(log_cols)}")
    print(f"Platform/tier one-hot columns added: {len(platform_cols)}")
    print(f"router_type/network_type one-hot columns added: {len(router_network_type_cols)}")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
