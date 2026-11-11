"""Check whether metric VALUES are sane, as opposed to merely present.

`audit_dql_source_coverage.py` checks completeness (is a reading there at
all, via has_X_data). This script checks correctness of readings that ARE
present: physically-impossible values (out of range), internally
inconsistent rows (e.g. min > max within the same bin), and stuck/flatlined
sensors (same non-zero value for an implausibly long run). These are the
"corrupt value" failure modes that a coverage check cannot see, and that
drift monitoring (which compares distributions over time, not individual
readings) is not designed to catch either.

Three checks:
  1. Range sanity -- RANGE_BOUNDS below, drawn from physical/protocol limits
     documented in docs/dql_query_full_list/*.txt and docs/feature_reference.md.
  2. Cross-column consistency -- CONSISTENCY_CHECKS below (e.g. min<=max,
     peers_with_data<=peer_count).
  3. Stuck-value detection -- longest run of an identical non-null, non-zero
     value per device/metric; flags runs longer than --stuck-run-buckets
     (default 30 = 5 days at the 4h bucket grid) as a likely frozen sensor.

Usage:
    python data_pipeline/audit_metric_value_sanity.py
    python data_pipeline/audit_metric_value_sanity.py --input data_pipeline/processed/cleaned_device_level_data.parquet
"""

import argparse
import os

import pandas as pd

# 2026-11-10: moved into data_pipeline/ (QA tool for this pipeline's own
# stage-1 output) and switched from CWD-relative strings to __file__-based
# paths -- the old relative strings only worked when run from the project
# root as cwd.
PIPELINE_DIR = os.path.dirname(__file__)
DEFAULT_INPUT = os.path.join(PIPELINE_DIR, "processed", "cleaned_device_level_data.parquet")
DEFAULT_OUT_PREFIX = os.path.join(PIPELINE_DIR, "processed", "metric_value_sanity")
ENTITY_COL = "dt.entity.network:device"

# metric -> (min, max) inclusive. None means unbounded on that side.
RANGE_BOUNDS = {
    "max_cpu_usage": (0, 100),
    "avg_cpu_usage": (0, 100),
    "memory_utilization_avg": (0, 100),
    "uptime_hrs": (0, None),
    "ifcsTrafficIn_bps": (0, None),
    "ifcsTrafficOut_bps": (0, None),
    "total_interface_errors": (0, None),
    "total_discards": (0, None),
    "interfaces_with_data": (0, None),
    "abnormal_interface_count": (0, None),
    "traffic_in_min_in_bin": (0, None),
    "traffic_in_max_in_bin": (0, None),
    "traffic_in_samples": (0, None),
    "traffic_out_min_in_bin": (0, None),
    "traffic_out_max_in_bin": (0, None),
    "traffic_out_samples": (0, None),
    "discards_min_in_bin": (0, None),
    "discards_max_in_bin": (0, None),
    "discards_samples": (0, None),
    "traffic_in_std": (0, None),
    "traffic_out_std": (0, None),
    "discards_std": (0, None),
    "traffic_in_range": (0, None),
    "traffic_out_range": (0, None),
    "discards_range": (0, None),
    "peer_count": (0, None),
    "peers_with_data": (0, None),
    "device_flap_count": (0, None),  # summed across ALL peers, so NOT capped at 16 -- see peer-scaled check below
    "worst_established_time": (0, None),
    "any_peer_down_samples": (0, None),
    # Optical power: typical transceiver range is roughly -40..+10 dBm.
    "power_worst_dBm": (-40, 10),
    "power_best_dBm": (-40, 10),
    "power_avg_dBm": (-40, 10),
    "tx_power_worst_dBm": (-40, 10),
    "rx_power_worst_dBm": (-40, 10),
    "power_range_dBm": (0, None),
    "power_volatility_std": (0, None),
    "power_volatility_range": (0, None),
    # Bias current: typical laser bias is 0..150 mA.
    "bias_current_worst_mA": (0, 200),
    "bias_current_min_mA": (0, 200),
    "bias_current_avg_mA": (0, 200),
    "bias_current_range_mA": (0, None),
    # Temperatures: physically plausible equipment range.
    "module_temp_worst_C": (-40, 150),
    "module_temp_min_C": (-40, 150),
    "module_temp_avg_C": (-40, 150),
    "chassis_temp_worst_C": (-40, 150),
    "chassis_temp_min_C": (-40, 150),
    "chassis_temp_avg_C": (-40, 150),
    "temp_value": (-40, 150),
    "module_temp_range_C": (0, None),
    "chassis_temp_range_C": (0, None),
    "interface_count": (0, None),
    "valid_flap_count": (0, None),  # see device_flap_count note above
}

# Binary/flag-style columns: only 0/1/NaN are valid.
BINARY_COLS = [
    "restart_flag", "any_interface_abnormal", "bias_abnormal_any", "temp_abnormal_any",
    "chassis_temp_abnormal_any", "fan_notPresent_any", "fan_abnormal_any", "fan_tray_abnormal",
    "fru_power_abnormal_any", "ps_notPresent_any", "ps_abnormal_any",
]

# (name, left_col, op, right_col, right_multiplier) -- flags rows where
# left_col {op} (right_multiplier * right_col) is violated.
CONSISTENCY_CHECKS = [
    ("traffic_in_min_gt_max", "traffic_in_min_in_bin", "<=", "traffic_in_max_in_bin", 1),
    ("traffic_out_min_gt_max", "traffic_out_min_in_bin", "<=", "traffic_out_max_in_bin", 1),
    ("discards_min_gt_max", "discards_min_in_bin", "<=", "discards_max_in_bin", 1),
    ("peers_with_data_gt_peer_count", "peers_with_data", "<=", "peer_count", 1),
    ("valid_flap_count_gt_device_flap_count", "valid_flap_count", "<=", "device_flap_count", 1),
    # device_flap_count is summed across ALL peers (not a per-device cap of
    # 16) -- see docs/leading_indicator_todo.md item 6 -- so the sane upper
    # bound is 16 checks/peer * peer_count, not a flat 16.
    ("device_flap_count_exceeds_peer_scaled_max", "device_flap_count", "<=", "peer_count", 16),
]

# Metrics worth stuck-value checking (continuous, where a long flat nonzero
# run is physically implausible rather than a normal idle/healthy state).
STUCK_VALUE_METRICS = [
    "avg_cpu_usage", "memory_utilization_avg", "ifcsTrafficIn_bps", "ifcsTrafficOut_bps",
    "power_avg_dBm", "bias_current_avg_mA", "module_temp_avg_C", "chassis_temp_avg_C", "temp_value",
]


def check_ranges(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, (lo, hi) in RANGE_BOUNDS.items():
        if col not in df.columns:
            continue
        s = df[col]
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= s < lo
        if hi is not None:
            mask |= s > hi
        n_bad = int(mask.sum())
        if n_bad:
            bad = df.loc[mask, [ENTITY_COL, "time_utc", col]]
            rows.append({
                "check": "range", "column": col, "bound": f"[{lo}, {hi}]",
                "n_violations": n_bad, "n_devices": bad[ENTITY_COL].nunique(),
                "example_value": bad[col].iloc[0], "example_device": bad[ENTITY_COL].iloc[0],
                "example_time": bad["time_utc"].iloc[0],
            })
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        s = df[col].dropna()
        mask = ~s.isin([0, 1])
        n_bad = int(mask.sum())
        if n_bad:
            rows.append({
                "check": "binary", "column": col, "bound": "{0, 1}",
                "n_violations": n_bad, "n_devices": df.loc[s[mask].index, ENTITY_COL].nunique(),
                "example_value": s[mask].iloc[0], "example_device": None, "example_time": None,
            })
    return pd.DataFrame(rows)


def check_consistency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, left, op, right, right_mult in CONSISTENCY_CHECKS:
        if left not in df.columns or right not in df.columns:
            continue
        both_present = df[left].notna() & df[right].notna()
        rhs = right_mult * df[right]
        if op == "<=":
            violated = both_present & (df[left] > rhs)
        else:
            raise ValueError(op)
        n_bad = int(violated.sum())
        if n_bad:
            bad = df.loc[violated, [ENTITY_COL, "time_utc", left, right]]
            rows.append({
                "check": name, "n_violations": n_bad, "n_devices": bad[ENTITY_COL].nunique(),
                "example_device": bad[ENTITY_COL].iloc[0], "example_time": bad["time_utc"].iloc[0],
                f"example_{left}": bad[left].iloc[0], f"example_{right}": bad[right].iloc[0],
            })
    return pd.DataFrame(rows)


def _longest_run(s: pd.Series) -> tuple:
    """Longest run of an identical non-null, non-zero value; returns (length, value)."""
    s = s.dropna()
    s = s[s != 0]
    if s.empty:
        return 0, None
    same_as_prev = s.eq(s.shift())
    run_id = (~same_as_prev).cumsum()
    run_lengths = s.groupby(run_id).size()
    best_run_id = run_lengths.idxmax()
    best_len = run_lengths.max()
    best_value = s[run_id == best_run_id].iloc[0]
    return int(best_len), best_value


def check_stuck_values(df: pd.DataFrame, stuck_run_buckets: int) -> pd.DataFrame:
    rows = []
    for col in STUCK_VALUE_METRICS:
        if col not in df.columns:
            continue
        for entity_id, g in df.sort_values("time_utc").groupby(ENTITY_COL):
            length, value = _longest_run(g[col])
            if length >= stuck_run_buckets:
                rows.append({
                    "column": col, "device": entity_id, "run_length_buckets": length,
                    "run_length_days": round(length * 4 / 24, 1), "stuck_value": value,
                })
    return pd.DataFrame(rows).sort_values("run_length_buckets", ascending=False) if rows else pd.DataFrame(
        columns=["column", "device", "run_length_buckets", "run_length_days", "stuck_value"]
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--stuck-run-buckets", type=int, default=30, help="Flag a stuck value held for at least this many consecutive 4h buckets (default 30 = 5 days)")
    ap.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    args = ap.parse_args()

    df = pd.read_parquet(args.input)

    range_report = check_ranges(df)
    consistency_report = check_consistency(df)
    stuck_report = check_stuck_values(df, args.stuck_run_buckets)

    range_report.to_csv(f"{args.out_prefix}_range_violations.csv", index=False)
    consistency_report.to_csv(f"{args.out_prefix}_consistency_violations.csv", index=False)
    stuck_report.to_csv(f"{args.out_prefix}_stuck_values.csv", index=False)

    print(f"Rows scanned: {len(df)}")
    print(f"\n=== Range violations ===")
    print(range_report.to_string(index=False) if len(range_report) else "none")
    print(f"\n=== Consistency violations ===")
    print(consistency_report.to_string(index=False) if len(consistency_report) else "none")
    print(f"\n=== Stuck values (>= {args.stuck_run_buckets} buckets = {args.stuck_run_buckets*4/24:.1f} days) ===")
    print(stuck_report.head(30).to_string(index=False) if len(stuck_report) else "none")
    print(f"\nFull detail written to {args.out_prefix}_{{range,consistency,stuck}}*.csv")


if __name__ == "__main__":
    main()
