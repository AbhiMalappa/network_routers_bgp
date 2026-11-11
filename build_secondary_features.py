"""
BGP flap prediction -- data pipeline, stage 6 (final): secondary features.

Takes dynatrace_snow_splunk_combined.parquet (the fully merged Dynatrace +
SNOW + Splunk table) and adds every derived/lag feature this project has
established, applied over the WHOLE combined table in one place -- so
Dynatrace, SNOW, and Splunk-derived columns all go through the same
rolling-window machinery, rather than each source computing its own lag
logic in its own stage. This is the generic, model-agnostic output table:
no target columns, no row filtering, no train/test split. That's all
deferred to a separate, model-specific step -- which model (XGBoost,
survival, LSTM, ...) decides what the target even looks like, and this
pipeline's job ends at "every raw and derived feature a model could want."

Three different secondary treatments, assigned deliberately per metric
group (see docs/feature_reference.md for the original per-metric reasoning
this carries forward):

  1. LEVEL_METRICS / SPREAD_STAT_METRICS (Dynatrace raw metrics, plus
     snow_ticket_count -- 2026-11-09, treated as a count-style level metric
     just like peer_count/peers_with_data) -- full treatment: trend
     (_delta_24h), self-baseline z-score (_zscore_vs_own_history), rolling
     volatility (_std_prior_24h/_7d).
  2. The 59 Splunk log-derived base columns, plus device_flap_count,
     num_peers_flapped, flap_concentration, and the 10 hardware abnormal
     flags (+ any_hardware_abnormal rollup) -- prior_24h/_7d SUM lag only
     (flap_concentration gets a prior_7d AVG instead, being a ratio, not a
     count). No trend/z-score/volatility -- these are either already-rare
     binary flags (trend/z-score of a flag is a weak construction) or
     persistence-adjacent counts where the "how much has this fired
     recently" lag is the only secondary signal that was ever established
     as useful.

     NOTE on device_flap_count/num_peers_flapped/flap_concentration
     specifically: these remain ordinary feature columns in THIS generic
     table (no target-based exclusion here, since there's no target here).
     A future model-specific step choosing a forward-looking flap target
     will likely need to exclude these (and their lags) from ITS OWN
     feature set, for the same label-leakage reason the old pipeline's
     PERSISTENCE_FEATURE_PREFIXES policy existed -- that's a decision for
     that step to make, not this one.
  3. Everything else established in this project's "additional derived
     features" set: post-reboot risk window (recently_rebooted_24h/_48h),
     interface churn imbalance (down - up, at each lag horizon), and
     error/discard rate normalized by traffic volume (+ prior_7d avg).

Deliberately NOT ported from the old pipeline: the three permanent
`keep`-mask row exclusions (flat_cpu_device, chronic_flap_window,
restart_co_occurring_flap) and the known-raw-pull-gap date exclusion.
Those existed to protect TARGET construction from contaminated rows --
with no target built here, there's nothing to protect. The three flag
columns themselves are left as ordinary features (computed upstream in
build_and_clean_dynatrace_device_data.py, untouched); a model-specific step
can still filter on them later if it chooses to. The known raw-pull gap
(Aug 29-Sept 17 2026) isn't masked out either -- every rolling/expanding
window here already correctly propagates NaN through min_periods when its
window spans missing data, so the gap shows up as NaN in the right derived
columns on its own, without needing an explicit date-range drop.

Input: data/processed/dynatrace_snow_splunk_combined.parquet,
       docs/log_message_text_to_feature_reference.csv (for the list of 59
       Splunk-derived base column names)
Output: data/processed/bgp_combined_data_table.parquet
"""
import os

import pandas as pd

# 2026-11-11: raw/processed/docs all live inside data_pipeline/ -- fully
# self-contained, no path here reaches outside this folder.
PIPELINE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(PIPELINE_DIR, "processed"), exist_ok=True)
IN_PATH = os.path.join(PIPELINE_DIR, "processed", "dynatrace_snow_splunk_combined.parquet")
LOG_REF_PATH = os.path.join(PIPELINE_DIR, "docs", "log_message_text_to_feature_reference.csv")
OUT_PATH = os.path.join(PIPELINE_DIR, "processed", "bgp_combined_data_table.parquet")

ENTITY_COL = "dt.entity.network:device"

BUCKET_HOURS = 4
PRIOR_24H_PERIODS = 24 // BUCKET_HOURS   # 6 buckets
PRIOR_7D_PERIODS = (7 * 24) // BUCKET_HOURS  # 42 buckets

# Same gap-tolerance reasoning as the old pipeline: a single short reporting
# blip (1-2 buckets) shouldn't poison a full 24h/7d rolling window.
GAP_TOLERANCE_BUCKETS = 2
PRIOR_24H_MIN_PERIODS = PRIOR_24H_PERIODS - GAP_TOLERANCE_BUCKETS  # 4
PRIOR_7D_MIN_PERIODS = PRIOR_7D_PERIODS - GAP_TOLERANCE_BUCKETS    # 40
ZSCORE_EPSILON = 1e-9

# Group 1: full trend + z-score + volatility treatment.
LEVEL_METRICS = [
    "max_cpu_usage", "avg_cpu_usage", "memory_utilization_avg",
    "ifcsTrafficIn_bps", "ifcsTrafficOut_bps", "total_interface_errors", "total_interface_discards",
    "worst_interface_errors", "num_interfaces_with_errors",
    "worst_interface_discards", "num_interfaces_with_discards",
    "interfaces_with_errors_data", "interfaces_with_discards_data",
    "interfaces_with_data", "abnormal_interface_count",
    "traffic_in_min_in_bin", "traffic_in_max_in_bin", "traffic_in_samples",
    "traffic_out_min_in_bin", "traffic_out_max_in_bin", "traffic_out_samples",
    "discards_min_in_bin", "discards_max_in_bin", "discards_samples",
    "peer_count", "peers_with_data", "worst_established_time",
    "power_worst_dBm", "power_best_dBm", "power_avg_dBm", "tx_power_worst_dBm", "rx_power_worst_dBm",
    "bias_current_worst_mA", "bias_current_min_mA", "bias_current_avg_mA",
    "module_temp_worst_C", "module_temp_min_C", "module_temp_avg_C",
    "chassis_temp_worst_C", "chassis_temp_min_C", "chassis_temp_avg_C",
    "temp_value",
    "snow_ticket_count",  # 2026-11-09 -- a count-style level metric, same treatment as peer_count
]
SPREAD_STAT_METRICS = [
    "traffic_in_std", "traffic_out_std", "discards_std",
    "traffic_in_range", "traffic_out_range", "discards_range",
    "power_range_dBm", "power_volatility_std", "power_volatility_range",
    "bias_current_range_mA", "module_temp_range_C", "chassis_temp_range_C",
]
TREND_METRICS = LEVEL_METRICS + SPREAD_STAT_METRICS
ZSCORE_METRICS = LEVEL_METRICS + SPREAD_STAT_METRICS
VOLATILITY_METRICS = LEVEL_METRICS + SPREAD_STAT_METRICS

# Group 2: prior_24h/_7d SUM lag only (binary flags + persistence-adjacent counts).
ABNORMAL_FLAG_COLS = [
    "any_interface_abnormal", "bias_abnormal_any", "temp_abnormal_any",
    "chassis_temp_abnormal_any", "fan_notPresent_any", "fan_abnormal_any",
    "fan_tray_abnormal", "fru_power_abnormal_any", "ps_notPresent_any", "ps_abnormal_any",
]
HARDWARE_FAULT_FLAG_COLS = [c for c in ABNORMAL_FLAG_COLS if c != "any_interface_abnormal"]


def load_log_cols() -> list:
    ref = pd.read_csv(LOG_REF_PATH).fillna({"feature_col_name": ""})
    return sorted(set(ref.loc[ref["category"] != "dropped", "feature_col_name"].unique()))


def add_coverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per has_X_data source (14 of them): missing-bins count in the
    trailing 24h/7d window, running lifetime-observed-bins count, and
    hours-since-last-real-reading staleness. All strictly backward-looking
    (shift(1) before any accumulation)."""
    has_cols = [c for c in df.columns if c.startswith("has_") and c.endswith("_data")]
    g_cov = df.groupby(ENTITY_COL)
    coverage_cols = {}
    for has_col in has_cols:
        stem = has_col[len("has_"):-len("_data")]

        coverage_cols[f"{stem}_missing_bins_prior_24h"] = g_cov[has_col].transform(
            lambda s: (1 - s.astype(int)).shift(1).rolling(PRIOR_24H_PERIODS, min_periods=1).sum()
        )
        coverage_cols[f"{stem}_missing_bins_prior_7d"] = g_cov[has_col].transform(
            lambda s: (1 - s.astype(int)).shift(1).rolling(PRIOR_7D_PERIODS, min_periods=1).sum()
        )
        coverage_cols[f"{stem}_bins_observed_lifetime"] = g_cov[has_col].transform(
            lambda s: s.astype(int).shift(1).fillna(0).cumsum()
        )

        last_real_time = df["time_utc"].where(df[has_col])
        last_real_time_ffilled = last_real_time.groupby(df[ENTITY_COL]).ffill()
        prior_last_real_time = last_real_time_ffilled.groupby(df[ENTITY_COL]).shift(1)
        coverage_cols[f"{stem}_hours_since_last_real_reading"] = (
            (df["time_utc"] - prior_last_real_time).dt.total_seconds() / 3600
        )
    return pd.concat([df, pd.DataFrame(coverage_cols, index=df.index)], axis=1)


def add_persistence_lags(df: pd.DataFrame) -> pd.DataFrame:
    """device_flap_count, num_peers_flapped (counts -> SUM lag) and
    flap_concentration (a ratio -> AVG lag). See module docstring for why
    these stay ordinary features here with no target-based exclusion."""
    g = df.groupby(ENTITY_COL)
    df["flap_count_prior_24h"] = g["device_flap_count"].transform(
        lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
    )
    df["flap_count_prior_7d"] = g["device_flap_count"].transform(
        lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).sum()
    )
    df["num_peers_flapped_prior_24h"] = g["num_peers_flapped"].transform(
        lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
    )
    df["num_peers_flapped_prior_7d"] = g["num_peers_flapped"].transform(
        lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).sum()
    )
    df["flap_concentration_prior_7d_avg"] = g["flap_concentration"].transform(
        lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).mean()
    )
    return df


def add_log_lags(df: pd.DataFrame, log_cols: list) -> pd.DataFrame:
    g = df.groupby(ENTITY_COL)
    lag_cols = {}
    for c in log_cols:
        lag_cols[f"{c}_prior_24h"] = g[c].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
        )
        lag_cols[f"{c}_prior_7d"] = g[c].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).sum()
        )
    return pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)


def add_hardware_abnormal_lags(df: pd.DataFrame) -> pd.DataFrame:
    """any_hardware_abnormal rollup (max(skipna=True) across the 9 real
    fault categories, excluding any_interface_abnormal -- see
    HARDWARE_FAULT_FLAG_COLS), plus prior_24h/_7d SUM lag for it and every
    individual abnormal flag."""
    df["any_hardware_abnormal"] = df[HARDWARE_FAULT_FLAG_COLS].max(axis=1, skipna=True)
    g = df.groupby(ENTITY_COL)
    hw_lag_cols = {}
    for c in ABNORMAL_FLAG_COLS + ["any_hardware_abnormal"]:
        hw_lag_cols[f"{c}_prior_24h"] = g[c].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
        )
        hw_lag_cols[f"{c}_prior_7d"] = g[c].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).sum()
        )
    return pd.concat([df, pd.DataFrame(hw_lag_cols, index=df.index)], axis=1)


def add_trend_zscore_volatility(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(ENTITY_COL)

    trend_cols = {}
    for m in TREND_METRICS:
        trend_cols[f"{m}_delta_24h"] = df[m] - g[m].shift(PRIOR_24H_PERIODS)
    df = pd.concat([df, pd.DataFrame(trend_cols, index=df.index)], axis=1)

    g = df.groupby(ENTITY_COL)  # fresh -- df was just reassigned by the concat above
    zscore_cols = {}
    for m in ZSCORE_METRICS:
        shifted = g[m].transform(lambda s: s.shift(1))
        baseline_mean = shifted.groupby(df[ENTITY_COL]).transform(
            lambda s: s.expanding(min_periods=PRIOR_24H_PERIODS).mean()
        )
        baseline_std = shifted.groupby(df[ENTITY_COL]).transform(
            lambda s: s.expanding(min_periods=PRIOR_24H_PERIODS).std()
        )
        zscore_cols[f"{m}_zscore_vs_own_history"] = (df[m] - baseline_mean) / (baseline_std + ZSCORE_EPSILON)
    df = pd.concat([df, pd.DataFrame(zscore_cols, index=df.index)], axis=1)

    g = df.groupby(ENTITY_COL)  # fresh again
    vol_cols = {}
    for m in VOLATILITY_METRICS:
        vol_cols[f"{m}_std_prior_24h"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).std()
        )
        vol_cols[f"{m}_std_prior_7d"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).std()
        )
    df = pd.concat([df, pd.DataFrame(vol_cols, index=df.index)], axis=1)
    return df


def add_misc_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Post-reboot risk window, interface churn imbalance (needs the log
    lag columns already added by add_log_lags -- must run after it), and
    traffic-normalized error/discard rate + their prior_7d avg lag."""
    df["recently_rebooted_24h"] = (df["uptime_hrs"] < 24).astype("Int64")
    df["recently_rebooted_48h"] = (df["uptime_hrs"] < 48).astype("Int64")
    df.loc[df["uptime_hrs"].isna(), ["recently_rebooted_24h", "recently_rebooted_48h"]] = pd.NA

    churn_cols = {}
    for suffix in ["", "_prior_24h", "_prior_7d"]:
        down_col, up_col = f"interface_link_down_count{suffix}", f"interface_link_up_count{suffix}"
        if down_col in df.columns and up_col in df.columns:
            churn_cols[f"interface_churn_imbalance{suffix}"] = df[down_col] - df[up_col]
    df = pd.concat([df, pd.DataFrame(churn_cols, index=df.index)], axis=1)

    total_traffic = df["ifcsTrafficIn_bps"] + df["ifcsTrafficOut_bps"] + 1
    df["error_rate"] = df["total_interface_errors"] / total_traffic
    df["discard_rate"] = df["total_interface_discards"] / total_traffic
    g = df.groupby(ENTITY_COL)
    rate_cols = {
        "error_rate_prior_7d_avg": g["error_rate"].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).mean()
        ),
        "discard_rate_prior_7d_avg": g["discard_rate"].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).mean()
        ),
    }
    df = pd.concat([df, pd.DataFrame(rate_cols, index=df.index)], axis=1)
    return df


def main():
    df = pd.read_parquet(IN_PATH)
    df = df.sort_values([ENTITY_COL, "time_utc"]).reset_index(drop=True)
    n_cols_start = len(df.columns)

    log_cols = load_log_cols()

    df = add_coverage_features(df)
    df = add_persistence_lags(df)
    df = add_log_lags(df, log_cols)
    df = add_hardware_abnormal_lags(df)
    df["hour_of_day"] = df["time_utc"].dt.hour
    df["day_of_week"] = df["time_utc"].dt.dayofweek
    df = add_trend_zscore_volatility(df)
    df = add_misc_derived_features(df)

    print(f"Rows: {len(df)}  Devices: {df[ENTITY_COL].nunique()}")
    print(f"Columns: {n_cols_start} -> {len(df.columns)} ({len(df.columns) - n_cols_start} secondary features added)")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
