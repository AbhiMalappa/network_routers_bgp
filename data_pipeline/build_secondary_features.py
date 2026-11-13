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

2026-11-13 rewrite: LEVEL_METRICS/SPREAD_STAT_METRICS (the 55 Dynatrace raw
metrics) no longer get one uniform treatment -- an audit found they're not
one shape of data (see docs/derived_feature_metrics.md for the full
reference, TODO_secondary_features.md for the findings/decisions this
rewrite implements):

  Group A (10 cols, continuous but severely outlier-contaminated by a rare
  within-device glitch reading -- traffic bps/min/max/std/range): trend via
  closed-form OLS slope (_slope_prior_48h), self-baseline MODIFIED z-score
  (_mad_zscore_vs_own_history -- median/MAD, not mean/std, computed
  separately for weekday vs. weekend buckets over a rolling 14-day window),
  and MAD-based volatility (_mad_prior_24h/_7d, replacing plain std for the
  same reason).

  Group B (12 cols, zero-inflated counts/snapshots -- interface errors/
  discards, snow_ticket_count): NO trend/z-score/volatility at all (a mean/
  std or even median/MAD statistic collapses when most values are 0). Split
  by shape instead -- 6 genuine counts get SUM lag (how much accumulated),
  5 per-bucket worst/min/max/spread snapshots get rolling-MAX lag (worst
  seen recently, not a meaningless sum-of-maxima), and discards_range gets
  neither (it's exactly discards_max_in_bin - discards_min_in_bin, already
  covered by those two).

  Group C (33 cols, everything else -- CPU, memory, temp, power, bias
  current, peer_count, worst_established_time, etc.): no evidence of
  Group A's outlier problem, kept on the original mean/std z-score
  (_zscore_vs_own_history) and plain std volatility (_std_prior_24h/_7d) --
  only the baseline window itself changes, from expanding to a rolling
  14-day window (same reasoning as Group A: an expanding baseline never
  forgets and can't adapt to a legitimate regime change).

Plus several previously-missing or misnamed features, closed out in the
same audit: valid_flap_count and any_peer_down_samples (renamed
peer_down_sample_count) had NO lag at all despite their siblings
(device_flap_count, num_peers_flapped) already having one; recently_
rebooted_24h/_48h and restart_flag had no lag either; the 4 _stuck_reading
diagnostic flags had no lag; interface_churn_imbalance (down - up) can't
distinguish "nothing happened" from "a lot of symmetric churn", so it now
has a interface_churn_total (down + up) companion.

The 59 Splunk log-derived base columns, the 10 hardware-abnormal flags (+
any_hardware_abnormal rollup), the has_X_data coverage features, and
error_rate/discard_rate's _prior_7d_avg were all audited too and found to
already have the right treatment -- unchanged here.

Input: data/processed/dynatrace_snow_splunk_combined.parquet,
       docs/log_message_text_to_feature_reference.csv (for the list of 59
       Splunk-derived base column names)
Output: data/processed/bgp_combined_data_table.parquet
"""
import os

import numpy as np
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
PRIOR_48H_PERIODS = 48 // BUCKET_HOURS   # 12 buckets -- slope window only
PRIOR_7D_PERIODS = (7 * 24) // BUCKET_HOURS  # 42 buckets
ROLLING_14D_PERIODS = (14 * 24) // BUCKET_HOURS  # 84 buckets -- z-score baseline window only

# Same gap-tolerance reasoning as the old pipeline: a single short reporting
# blip (1-2 buckets) shouldn't poison a full window.
GAP_TOLERANCE_BUCKETS = 2
PRIOR_24H_MIN_PERIODS = PRIOR_24H_PERIODS - GAP_TOLERANCE_BUCKETS  # 4
PRIOR_48H_MIN_PERIODS = PRIOR_48H_PERIODS - GAP_TOLERANCE_BUCKETS  # 10
PRIOR_7D_MIN_PERIODS = PRIOR_7D_PERIODS - GAP_TOLERANCE_BUCKETS    # 40
# z-score baseline floor: same minimum (1 day's worth) the old expanding
# baseline required before producing a value at all -- preserved here even
# though the window itself is now capped at 14 days instead of growing
# forever.
ROLLING_14D_MIN_PERIODS = PRIOR_24H_PERIODS  # 6

ZSCORE_EPSILON = 1e-9
# Standard constant so a MAD-based statistic lands on the same scale a std
# would for ordinary (non-glitchy) data -- same constant implicit in the
# "modified z-score" formula (0.6745 = 1/1.4826).
MAD_SCALE = 1.4826

# Every Dynatrace raw metric considered for trend/z-score/volatility
# treatment -- 55 total. Split into Group A/B/C below; this pair of lists
# is kept as the full reference set, not a treatment group on its own.
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
    "snow_ticket_count",
]
SPREAD_STAT_METRICS = [
    "traffic_in_std", "traffic_out_std", "discards_std",
    "traffic_in_range", "traffic_out_range", "discards_range",
    "power_range_dBm", "power_volatility_std", "power_volatility_range",
    "bias_current_range_mA", "module_temp_range_C", "chassis_temp_range_C",
]

# Group A: severely outlier-contaminated (see module docstring) -- gets
# _slope_prior_48h, _mad_zscore_vs_own_history, _mad_prior_24h/_7d.
GROUP_A_METRICS = [
    "ifcsTrafficIn_bps", "ifcsTrafficOut_bps",
    "traffic_in_min_in_bin", "traffic_in_max_in_bin",
    "traffic_out_min_in_bin", "traffic_out_max_in_bin",
    "traffic_in_std", "traffic_out_std",
    "traffic_in_range", "traffic_out_range",
]
# Group B: zero-inflated -- no trend/z-score/volatility at all. Split by
# shape: genuine counts get SUM lag, per-bucket worst/spread snapshots get
# rolling-MAX lag, discards_range gets neither (see add_group_b_lags()).
GROUP_B_COUNT_METRICS = [
    "total_interface_errors", "total_interface_discards",
    "num_interfaces_with_errors", "num_interfaces_with_discards",
    "abnormal_interface_count", "snow_ticket_count",
]
GROUP_B_WORST_METRICS = [
    "worst_interface_errors", "worst_interface_discards",
    "discards_min_in_bin", "discards_max_in_bin", "discards_std",
]
GROUP_B_METRICS = GROUP_B_COUNT_METRICS + GROUP_B_WORST_METRICS + ["discards_range"]

# Group C: everything else -- no evidence of Group A's problem, kept on the
# original mean/std treatment (only the baseline window changes).
GROUP_C_METRICS = [
    m for m in (LEVEL_METRICS + SPREAD_STAT_METRICS)
    if m not in GROUP_A_METRICS and m not in GROUP_B_METRICS
]

# TREND_METRICS: everyone except Group B (a slope on mostly-zero data with
# an explosive rare tail is as degenerate as mean/std z-scoring it -- see
# TODO_secondary_features.md #0).
TREND_METRICS = GROUP_A_METRICS + GROUP_C_METRICS

# Binary flags + persistence-adjacent counts: prior_24h/_7d SUM lag only.
ABNORMAL_FLAG_COLS = [
    "any_interface_abnormal", "bias_abnormal_any", "temp_abnormal_any",
    "chassis_temp_abnormal_any", "fan_notPresent_any", "fan_abnormal_any",
    "fan_tray_abnormal", "fru_power_abnormal_any", "ps_notPresent_any", "ps_abnormal_any",
]
HARDWARE_FAULT_FLAG_COLS = [c for c in ABNORMAL_FLAG_COLS if c != "any_interface_abnormal"]

# The 4 metric-level stuck-sensor diagnostic flags (see
# build_and_clean_dynatrace_device_data.py's stuck-reading detection). The 4
# _out_of_range flags are deliberately NOT here -- checked and found too
# rare (<=4 devices each) to generalize as a trend feature; left
# diagnostic-only.
STUCK_READING_FLAG_COLS = [
    "bias_current_avg_mA_stuck_reading", "power_avg_dBm_stuck_reading",
    "module_temp_avg_C_stuck_reading", "temp_value_stuck_reading",
]


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
    """device_flap_count, valid_flap_count, num_peers_flapped,
    peer_down_sample_count (counts -> SUM lag) and flap_concentration (a
    ratio -> AVG lag). See module docstring for why these stay ordinary
    features here with no target-based exclusion.

    valid_flap_count and peer_down_sample_count (renamed from
    any_peer_down_samples in main()) previously had NO lag at all despite
    being the same shape of data as their already-lagged siblings -- see
    TODO_secondary_features.md #9/#10."""
    g = df.groupby(ENTITY_COL)
    for metric, out_stem in [
        ("device_flap_count", "flap_count"),
        ("valid_flap_count", "valid_flap_count"),
        ("num_peers_flapped", "num_peers_flapped"),
        ("peer_down_sample_count", "peer_down_sample_count"),
    ]:
        df[f"{out_stem}_prior_24h"] = g[metric].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
        )
        df[f"{out_stem}_prior_7d"] = g[metric].transform(
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


def add_stuck_reading_lags(df: pd.DataFrame) -> pd.DataFrame:
    """SUM lag (prior_24h/_7d) for the 4 metric-level _stuck_reading
    diagnostic flags -- "how often has this sensor been stuck recently".
    See TODO_secondary_features.md #8. astype(int) first (plain bool dtype,
    never null) so shift(1)'s introduced NaN upcasts cleanly to float,
    matching add_coverage_features()'s existing pattern."""
    g = df.groupby(ENTITY_COL)
    lag_cols = {}
    for c in STUCK_READING_FLAG_COLS:
        lag_cols[f"{c}_prior_24h"] = g[c].transform(
            lambda s: s.astype(int).shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
        )
        lag_cols[f"{c}_prior_7d"] = g[c].transform(
            lambda s: s.astype(int).shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).sum()
        )
    return pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)


def add_group_b_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Group B's two lag treatments (see module docstring and
    TODO_secondary_features.md #5):
      - genuine counts -> SUM lag, _prior_24h/_7d, same convention as every
        other count-shaped column in this pipeline.
      - per-bucket worst/min/max/spread snapshots -> rolling MAX lag,
        _max_prior_24h/_7d. Deliberately NOT the plain _prior_24h/_7d name
        the TODO note used as a placeholder example -- that suffix already
        means "SUM" everywhere else in this pipeline (device_flap_count,
        the hardware flags, the 59 log columns, ...), and reusing it here
        for a MAX would silently make the same suffix mean two different
        things depending on which column it's attached to. _max_ makes the
        statistic explicit, same reasoning as the _mad_ prefix elsewhere.
    discards_range gets neither -- see GROUP_B_METRICS above."""
    g = df.groupby(ENTITY_COL)
    lag_cols = {}
    for m in GROUP_B_COUNT_METRICS:
        lag_cols[f"{m}_prior_24h"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
        )
        lag_cols[f"{m}_prior_7d"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).sum()
        )
    for m in GROUP_B_WORST_METRICS:
        lag_cols[f"{m}_max_prior_24h"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).max()
        )
        lag_cols[f"{m}_max_prior_7d"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).max()
        )
    return pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)


def add_slope(df: pd.DataFrame) -> pd.DataFrame:
    """_slope_prior_48h -- replaces _delta_24h (a noisy two-point
    difference) with a linear regression slope (direction + magnitude)
    over the prior 48h. Closed-form OLS slope formula --
    (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2) -- with every sum a plain rolling sum
    of a row-position index masked to align with valid values, NOT a
    per-window regression call, so this stays fully vectorized. Verified
    against np.polyfit on real data (max diff 3.3e-14); see
    TODO_secondary_features.md #0.

    Applies to TREND_METRICS = Group A + Group C only. Group B gets no
    slope -- checked empirically (see module docstring) that it's exactly
    as degenerate as mean/std z-scoring mostly-zero data would be."""
    idx = df.groupby(ENTITY_COL).cumcount().astype(float)
    g = df.groupby(ENTITY_COL)
    slope_cols = {}
    for m in TREND_METRICS:
        y = g[m].transform(lambda s: s.shift(1))
        idx_masked = idx.where(y.notna())

        def roll(s):
            return s.groupby(df[ENTITY_COL]).transform(
                lambda v: v.rolling(PRIOR_48H_PERIODS, min_periods=PRIOR_48H_MIN_PERIODS).sum()
            )

        n = y.groupby(df[ENTITY_COL]).transform(
            lambda v: v.rolling(PRIOR_48H_PERIODS, min_periods=PRIOR_48H_MIN_PERIODS).count()
        )
        Sx, Sxx, Sxy, Sy = roll(idx_masked), roll(idx_masked ** 2), roll(idx_masked * y), roll(y)
        denom = n * Sxx - Sx * Sx
        slope_cols[f"{m}_slope_prior_48h"] = (n * Sxy - Sx * Sy) / denom.where(denom != 0)
    return pd.concat([df, pd.DataFrame(slope_cols, index=df.index)], axis=1)


def add_zscore_group_c(df: pd.DataFrame) -> pd.DataFrame:
    """Group C's _zscore_vs_own_history -- unchanged mean/std statistic,
    unchanged (not stratified by weekday/weekend), but the baseline window
    changes from expanding to a rolling 14-day window -- an expanding
    baseline never forgets and can't adapt to a legitimate regime change
    (traffic growth, a capacity upgrade). Same minimum (6 prior values,
    1 day's worth) the old expanding baseline required before producing a
    value at all. See TODO_secondary_features.md #1/#4."""
    g = df.groupby(ENTITY_COL)
    zscore_cols = {}
    for m in GROUP_C_METRICS:
        shifted = g[m].transform(lambda s: s.shift(1))
        gg = shifted.groupby(df[ENTITY_COL])
        baseline_mean = gg.transform(
            lambda s: s.rolling(ROLLING_14D_PERIODS, min_periods=ROLLING_14D_MIN_PERIODS).mean()
        )
        baseline_std = gg.transform(
            lambda s: s.rolling(ROLLING_14D_PERIODS, min_periods=ROLLING_14D_MIN_PERIODS).std()
        )
        zscore_cols[f"{m}_zscore_vs_own_history"] = (df[m] - baseline_mean) / (baseline_std + ZSCORE_EPSILON)
    return pd.concat([df, pd.DataFrame(zscore_cols, index=df.index)], axis=1)


def add_std_group_c(df: pd.DataFrame) -> pd.DataFrame:
    """Group C's _std_prior_24h/_7d -- unchanged plain rolling std, same
    windows as always. See TODO_secondary_features.md #4."""
    g = df.groupby(ENTITY_COL)
    vol_cols = {}
    for m in GROUP_C_METRICS:
        vol_cols[f"{m}_std_prior_24h"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).std()
        )
        vol_cols[f"{m}_std_prior_7d"] = g[m].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).std()
        )
    return pd.concat([df, pd.DataFrame(vol_cols, index=df.index)], axis=1)


def _stride_rolling_median_mad(values: np.ndarray, window: int, min_periods: int):
    """Rolling median and MAD (median absolute deviation, unscaled) over
    `values`, via numpy.lib.stride_tricks rather than a per-window Python
    callback -- median needs an actual sort, so there's no closed-form
    rolling-sum shortcut the way there is for the OLS slope. Builds every
    device's full set of overlapping windows at once and runs vectorized
    numpy calls across all of them, instead of one Python-level call per
    row -- ~16x faster than .rolling().apply(raw=True) in testing (see
    TODO_secondary_features.md #2), and verified to match a corrected
    .rolling().apply(np.nanmedian) reference exactly (0.0 max diff).

    Correctness note: must use np.nanmedian, not np.median -- np.median on
    any array containing even one NaN returns NaN regardless of how many
    real values are present, which would silently produce NaN wherever
    min_periods should still allow a result (this was an actual bug caught
    during verification, see TODO #2)."""
    n = len(values)
    padded = np.concatenate([np.full(window - 1, np.nan), values])
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)
    valid_counts = np.sum(~np.isnan(windows), axis=1)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(windows, axis=1)
        mad = np.nanmedian(np.abs(windows - med[:, None]), axis=1)
    enough = valid_counts >= min_periods
    return np.where(enough, med, np.nan), np.where(enough, mad, np.nan)


def _per_device_median_mad(df: pd.DataFrame, shifted_series: pd.Series, window: int, min_periods: int):
    """Applies _stride_rolling_median_mad() once per device (the Python-
    level loop this buys back is over ~1,510 devices, not 1.3M rows) on an
    already shift(1)'d series. Returns (median_series, mad_series) aligned
    to df.index."""
    med_parts, mad_parts = [], []
    for _, sub in df.groupby(ENTITY_COL, sort=False):
        vals = shifted_series.loc[sub.index].to_numpy()
        med, mad = _stride_rolling_median_mad(vals, window, min_periods)
        med_parts.append(pd.Series(med, index=sub.index))
        mad_parts.append(pd.Series(mad, index=sub.index))
    return pd.concat(med_parts).sort_index(), pd.concat(mad_parts).sort_index()


def add_mad_zscore_group_a(df: pd.DataFrame) -> pd.DataFrame:
    """Group A's _mad_zscore_vs_own_history -- median/MAD baseline (not
    mean/std: Group A's mean/std is dominated by a small minority of
    devices with a few extreme within-device glitch readings, see module
    docstring), computed SEPARATELY for weekday vs. weekend buckets (Group
    A also has the strongest weekday/weekend gap of any metric checked),
    over a rolling 14-day window (same window as Group C's z-score, see
    TODO_secondary_features.md #1/#2).

    Named differently from Group C's _zscore_vs_own_history on purpose -- a
    median/MAD statistic (the standard term is "modified z-score") is a
    genuinely different formula, not just a more-robust version of the same
    number, so it gets a different name rather than silently meaning two
    different things under one column-name pattern."""
    is_weekend = df["day_of_week"] >= 5
    # Which day-type each SHIFTED (prior) value itself belongs to -- must
    # be shifted the same way `shifted` is, since shifted[p] holds the
    # value from row p-1, not row p. Masking with the unshifted is_weekend
    # would tag each retained value by the CURRENT row's day-type instead
    # of the day-type of the row the value actually came from -- caught
    # via an independent from-scratch spot-check (max diff 0.74 before
    # this fix, 0.0 after).
    # astype("boolean") (pandas nullable dtype) before shifting so the
    # introduced NA at each device's first row doesn't upcast to a plain
    # float Series that `~` can't operate on; fillna(False) afterward is
    # safe because `shifted` is already NaN at those same positions
    # regardless of which mask is applied.
    is_weekend_of_shifted = (
        is_weekend.astype("boolean").groupby(df[ENTITY_COL]).shift(1).fillna(False).astype(bool)
    )
    g = df.groupby(ENTITY_COL)
    zscore_cols = {}
    for m in GROUP_A_METRICS:
        shifted = g[m].transform(lambda s: s.shift(1))
        weekday_only = shifted.where(~is_weekend_of_shifted)
        weekend_only = shifted.where(is_weekend_of_shifted)
        wd_med, wd_mad = _per_device_median_mad(df, weekday_only, ROLLING_14D_PERIODS, ROLLING_14D_MIN_PERIODS)
        we_med, we_mad = _per_device_median_mad(df, weekend_only, ROLLING_14D_PERIODS, ROLLING_14D_MIN_PERIODS)
        baseline_median = we_med.where(is_weekend, wd_med)
        baseline_mad = we_mad.where(is_weekend, wd_mad)
        zscore_cols[f"{m}_mad_zscore_vs_own_history"] = (
            (df[m] - baseline_median) / (MAD_SCALE * baseline_mad + ZSCORE_EPSILON)
        )
    return pd.concat([df, pd.DataFrame(zscore_cols, index=df.index)], axis=1)


def add_mad_volatility_group_a(df: pd.DataFrame) -> pd.DataFrame:
    """Group A's _mad_prior_24h/_7d -- replaces plain rolling std with
    1.4826*MAD, same windows/min_periods as the std version it replaces
    (not weekday-stratified -- that split applies only to the 14-day
    z-score baseline above, not this shorter window). Same 1.4826 scaling
    constant as the z-score's internal MAD, so this feature and that
    denominator are the same quantity, not two things that happen to share
    a name. See TODO_secondary_features.md #2 (RESOLVED)."""
    g = df.groupby(ENTITY_COL)
    mad_cols = {}
    for m in GROUP_A_METRICS:
        shifted = g[m].transform(lambda s: s.shift(1))
        _, mad_24h = _per_device_median_mad(df, shifted, PRIOR_24H_PERIODS, PRIOR_24H_MIN_PERIODS)
        _, mad_7d = _per_device_median_mad(df, shifted, PRIOR_7D_PERIODS, PRIOR_7D_MIN_PERIODS)
        mad_cols[f"{m}_mad_prior_24h"] = mad_24h * MAD_SCALE
        mad_cols[f"{m}_mad_prior_7d"] = mad_7d * MAD_SCALE
    return pd.concat([df, pd.DataFrame(mad_cols, index=df.index)], axis=1)


def add_misc_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Post-reboot risk window (+ SUM lag) and restart_flag's own SUM lag
    (previously missing -- see TODO #6), interface churn imbalance AND
    total (+ lags -- total is new, see TODO #7; needs the log lag columns
    already added by add_log_lags, must run after it), and traffic-
    normalized error/discard rate + their prior_7d avg lag (unchanged)."""
    df["recently_rebooted_24h"] = (df["uptime_hrs"] < 24).astype("Int64")
    df["recently_rebooted_48h"] = (df["uptime_hrs"] < 48).astype("Int64")
    df.loc[df["uptime_hrs"].isna(), ["recently_rebooted_24h", "recently_rebooted_48h"]] = pd.NA

    g = df.groupby(ENTITY_COL)
    reboot_lag_cols = {}
    for c in ["recently_rebooted_24h", "recently_rebooted_48h", "restart_flag"]:
        reboot_lag_cols[f"{c}_prior_24h"] = g[c].transform(
            lambda s: s.shift(1).rolling(PRIOR_24H_PERIODS, min_periods=PRIOR_24H_MIN_PERIODS).sum()
        )
        reboot_lag_cols[f"{c}_prior_7d"] = g[c].transform(
            lambda s: s.shift(1).rolling(PRIOR_7D_PERIODS, min_periods=PRIOR_7D_MIN_PERIODS).sum()
        )
    df = pd.concat([df, pd.DataFrame(reboot_lag_cols, index=df.index)], axis=1)

    churn_cols = {}
    for suffix in ["", "_prior_24h", "_prior_7d"]:
        down_col, up_col = f"interface_link_down_count{suffix}", f"interface_link_up_count{suffix}"
        if down_col in df.columns and up_col in df.columns:
            churn_cols[f"interface_churn_imbalance{suffix}"] = df[down_col] - df[up_col]
            churn_cols[f"interface_churn_total{suffix}"] = df[down_col] + df[up_col]
    df = pd.concat([df, pd.DataFrame(churn_cols, index=df.index)], axis=1)

    total_traffic = df["ifcsTrafficIn_bps"] + df["ifcsTrafficOut_bps"] + 1
    df["error_rate"] = df["total_interface_errors"] / total_traffic
    df["discard_rate"] = df["total_interface_discards"] / total_traffic
    g = df.groupby(ENTITY_COL)  # fresh -- df was just reassigned by the concats above
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
    # any_peer_down_samples reads like a boolean ("was any peer down") but
    # is actually sum(peer_down_samples) across all of a device's peers --
    # a device-level COUNT (query_3_bgp_flap_target.txt:46,51). Renamed so
    # the column states what it is. See TODO_secondary_features.md #10.
    df = df.rename(columns={"any_peer_down_samples": "peer_down_sample_count"})
    n_cols_start = len(df.columns)

    log_cols = load_log_cols()

    df = add_coverage_features(df)
    df = add_persistence_lags(df)
    df = add_log_lags(df, log_cols)
    df = add_hardware_abnormal_lags(df)
    df = add_group_b_lags(df)
    df = add_stuck_reading_lags(df)
    df["hour_of_day"] = df["time_utc"].dt.hour
    df["day_of_week"] = df["time_utc"].dt.day_of_week
    df = add_slope(df)
    df = add_zscore_group_c(df)
    df = add_mad_zscore_group_a(df)
    df = add_std_group_c(df)
    df = add_mad_volatility_group_a(df)
    df = add_misc_derived_features(df)

    print(f"Rows: {len(df)}  Devices: {df[ENTITY_COL].nunique()}")
    print(f"Columns: {n_cols_start} -> {len(df.columns)} ({len(df.columns) - n_cols_start} secondary features added)")

    df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
