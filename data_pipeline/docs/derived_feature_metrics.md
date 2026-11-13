# Derived Feature Metrics — Reference

Scope: `build_secondary_features.py` (pipeline stage 6). Describes every
derived feature in the final table, what it's computed from, and the exact
formula.

All lag/window features are per-device (grouped by `dt.entity.network:device`)
and strictly backward-looking (the current bucket's own value is excluded via
`shift(1)` before any window/expanding calculation is applied), except
`recently_rebooted_24h/_48h` (a point-in-time flag on the current row).

## Shared constants

- Bucket size: 4h
- "24h" window = 6 buckets; minimum 4 valid buckets required (tolerates up to
  2 missing/gap buckets)
- "48h" window (slope only) = 12 buckets; minimum 10 valid buckets required
- "7d" window = 42 buckets; minimum 40 valid buckets required (same 2-bucket
  gap tolerance)
- "14-day" window (z-score baseline only) = 84 buckets
- Z-score epsilon: 1e-9 (added to a std/MAD denominator to avoid
  divide-by-zero)
- MAD scaling constant: 1.4826 (standard constant so a MAD-based statistic
  lands on the same scale a std would for ordinary, non-glitchy data)

## Metric groups

`LEVEL_METRICS` + `SPREAD_STAT_METRICS` (the Dynatrace raw-metric columns:
CPU, memory, traffic, interface errors/discards, peer_count, optical power,
bias current, module/chassis temp, `snow_ticket_count`, plus the std/range
spread-stat columns) split into three groups by data shape, each with its
own treatment:

- **Group A** (continuous, severely outlier-contaminated): `ifcsTrafficIn_bps`,
  `ifcsTrafficOut_bps`, `traffic_in_min_in_bin`, `traffic_in_max_in_bin`,
  `traffic_out_min_in_bin`, `traffic_out_max_in_bin`, `traffic_in_std`,
  `traffic_out_std`, `traffic_in_range`, `traffic_out_range`.
- **Group B** (zero-inflated counts and per-bucket worst/spread snapshots):
  split further into three treatments, see below.
- **Group C** (everything else in LEVEL_METRICS/SPREAD_STAT_METRICS, 33
  columns — no evidence of Group A's problem, kept on the original mean/std
  treatment): `max_cpu_usage`, `avg_cpu_usage`, `memory_utilization_avg`,
  `interfaces_with_errors_data`, `interfaces_with_discards_data`,
  `interfaces_with_data`, `traffic_in_samples`, `traffic_out_samples`,
  `discards_samples`, `peer_count`, `peers_with_data`,
  `worst_established_time`, `power_worst_dBm`, `power_best_dBm`,
  `power_avg_dBm`, `tx_power_worst_dBm`, `rx_power_worst_dBm`,
  `bias_current_worst_mA`, `bias_current_min_mA`, `bias_current_avg_mA`,
  `module_temp_worst_C`, `module_temp_min_C`, `module_temp_avg_C`,
  `chassis_temp_worst_C`, `chassis_temp_min_C`, `chassis_temp_avg_C`,
  `temp_value`, `power_range_dBm`, `power_volatility_std`,
  `power_volatility_range`, `bias_current_range_mA`, `module_temp_range_C`,
  `chassis_temp_range_C`.

Group B's three sub-treatments:
- **Counts** (6): `total_interface_errors`, `total_interface_discards`,
  `num_interfaces_with_errors`, `num_interfaces_with_discards`,
  `abnormal_interface_count`, `snow_ticket_count`.
- **Worst/spread-of-a-bucket snapshots** (5): `worst_interface_errors`,
  `worst_interface_discards`, `discards_min_in_bin`, `discards_max_in_bin`,
  `discards_std`.
- **Redundant, no lag** (1): `discards_range` (exactly
  `discards_max_in_bin − discards_min_in_bin`, carries no information beyond
  those two columns).

## Definitions

### `_slope_prior_48h`
Linear regression slope (direction + magnitude) over the prior 48h (12
buckets, min 10 valid). Computed via the closed-form OLS slope formula —
`(n·Σxy − Σx·Σy) / (n·Σx² − (Σx)²)`, with every sum a plain rolling sum of a
row-position index masked to align with valid values — not a per-window
regression call, so it stays fully vectorized.
Applies to: Group A and Group C (i.e. every LEVEL_METRICS/SPREAD_STAT_METRICS
column except Group B).

### `_zscore_vs_own_history` (Group C)
`(current value − baseline_mean) / (baseline_std + 1e-9)`, where
`baseline_mean`/`baseline_std` are the mean/std of the prior 14-day rolling
window (not stratified by weekday/weekend).

### `_mad_zscore_vs_own_history` (Group A)
`(current value − baseline_median) / (1.4826 × baseline_MAD + 1e-9)`, where
`baseline_median`/`baseline_MAD` are computed over the prior 14-day rolling
window, **separately for weekday and weekend buckets** (not blended). A
median/MAD statistic (the standard term is "modified z-score") is a
genuinely different formula from a mean/std z-score, not just a more-robust
version of the same number — hence the distinct name.

### `_std_prior_24h` / `_std_prior_7d` (Group C)
Plain rolling **std** of prior values over the trailing 24h (6-bucket, min
4) or 7d (42-bucket, min 40) window.

### `_mad_prior_24h` / `_mad_prior_7d` (Group A)
`1.4826 × median(|window values − median(window values)|)` over the same
24h/7d windows `_std_prior_24h/_7d` uses (not weekday-stratified — that
split applies only to the 14-day z-score baseline, not this shorter
window).

### `_prior_24h` / `_prior_7d` (SUM lag)
Rolling **sum** of prior values over the trailing 24h (6-bucket, min 4) or
7d (42-bucket, min 40) window.
Applies to: Group B's 6 count columns, the 59 Splunk log-derived base
columns, `device_flap_count`, `valid_flap_count`, `num_peers_flapped`,
`peer_down_sample_count`, the 10 hardware abnormal flags +
`any_hardware_abnormal` rollup, `recently_rebooted_24h`/`_48h`,
`restart_flag`, and the 4 `_stuck_reading` flags.

### `_max_prior_24h` / `_max_prior_7d` (rolling MAX lag)
Rolling **max** of prior values, same windows/min_periods as the SUM-lag
version above. Chosen over SUM or mean/median because these columns are
already a max/min/spread snapshot *within* one bucket — summing or
averaging per-bucket extremes across time doesn't have a clean "total" or
"typical" interpretation, while a rolling max preserves the intended
"worst reading seen recently" meaning. Named `_max_prior_24h/_7d`, not the
plain `_prior_24h/_7d` SUM-lag naming — that suffix already means "SUM"
for every other count-shaped column in this pipeline, and reusing it here
would make the same suffix silently mean two different things depending on
which column it's attached to.
Applies to: Group B's 5 worst/spread-snapshot columns.

### `_prior_7d_avg` (AVG lag)
Rolling **mean** of prior values over the trailing 7d (42-bucket, min 40)
window only — no 24h AVG variant exists anywhere in the pipeline.
Applies to: `flap_concentration`, `error_rate`, `discard_rate` — all three
are already-normalized ratios, where SUM/max wouldn't make sense and the
richer zscore/slope treatment doesn't apply either.

### Coverage features (one set per `has_X_data` flag, 14 flags:
`has_health_data`, `has_interface_data`, `has_interface_oper_data`,
`has_traffic_volatility_data`, `has_bgp_data`, `has_optical_power_data`,
`has_optical_power_volatility_data`, `has_optical_bias_data`,
`has_optical_temp_data`, `has_fan_state_data`, `has_device_temp_data`,
`has_fan_tray_data`, `has_fru_power_data`, `has_power_supply_data`)
- `_missing_bins_prior_24h` / `_prior_7d`: count of prior bins in that window
  where `has_X_data` was False. Looser than the SUM lag above — only 1 valid
  bucket required (`min_periods=1`), by design, so a device's very first
  readings still produce a value rather than `NaN`.
- `_bins_observed_lifetime`: cumulative count of ALL prior bins (not
  windowed) where `has_X_data` was True — a running lifetime total.
- `_hours_since_last_real_reading`: hours between the current bucket's
  `time_utc` and the most recent PRIOR bucket where `has_X_data` was True
  (forward-filled per device, then shifted by 1).

### `recently_rebooted_24h` / `_48h`
`uptime_hrs < 24` / `< 48`, evaluated on the **current** row — a
point-in-time flag, not lagged/shifted like everything else. `NA` wherever
`uptime_hrs` is `NA`. Also gets the SUM lag above (how much of the recent
window was spent in a post-reboot state).

### `interface_churn_imbalance` / `interface_churn_total` (+ `_prior_24h` / `_prior_7d`)
`interface_link_down_count{suffix} − interface_link_up_count{suffix}`
(imbalance) and `interface_link_down_count{suffix} + interface_link_up_count{suffix}`
(total), for suffix `""`, `"_prior_24h"`, `"_prior_7d"`. The suffixed
versions combine the already-lagged SUM-lag columns (computed earlier as
part of the 59 Splunk-log-derived columns), not raw values. Kept as a pair
deliberately: imbalance alone can't distinguish "nothing happened" from
"a lot of symmetric churn" (equal up/down counts both give imbalance = 0) —
`interface_churn_total` carries the magnitude that `interface_churn_imbalance`
can't.

### `error_rate` / `discard_rate`
`total_interface_errors / (ifcsTrafficIn_bps + ifcsTrafficOut_bps + 1)`, and
the same with `total_interface_discards`. Point-in-time ratio on the current
row (the `+1` avoids divide-by-zero). Gets the `_prior_7d_avg` lag above.

### `any_hardware_abnormal`
`max(skipna=True)` across the 9 real hardware fault flags (excludes
`any_interface_abnormal`) — equivalent to a logical OR across 0/1 flags.
Gets the SUM lag above, alongside the 10 individual flags: `any_interface_
abnormal`, `bias_abnormal_any`, `temp_abnormal_any`,
`chassis_temp_abnormal_any`, `fan_notPresent_any`, `fan_abnormal_any`,
`fan_tray_abnormal`, `fru_power_abnormal_any`, `ps_notPresent_any`,
`ps_abnormal_any`.

### `_stuck_reading` flags (4: `bias_current_avg_mA`, `power_avg_dBm`,
`module_temp_avg_C`, `temp_value`)
Point-in-time diagnostic flag marking which rows had that metric's raw value
nulled by Stage 1's stuck-sensor detection (≥30 consecutive identical,
non-zero readings — 5 days — on the device's own dense grid). Also gets the
SUM lag above (how often this sensor has been stuck recently).

### `_out_of_range` flags (4: `power_worst_dBm`, `tx_power_worst_dBm`,
`rx_power_worst_dBm`, `power_range_dBm`)
Point-in-time diagnostic flag marking an optical-power floor violation
(< -40 dBm, a "no light" artifact). Diagnostic-only — no lag feature.

## Cross-table: every Dynatrace metric × derived feature

| Metric | `_slope_prior_48h` | z-score variant | std/MAD variant | `_prior_24h/_7d` (SUM) | `_max_prior_24h/_7d` (MAX) | `_prior_7d_avg` | other |
|---|---|---|---|---|---|---|---|
| **Group A** |
| `ifcsTrafficIn_bps` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `ifcsTrafficOut_bps` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_in_min_in_bin` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_in_max_in_bin` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_out_min_in_bin` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_out_max_in_bin` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_in_std` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_out_std` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_in_range` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| `traffic_out_range` | yes | `_mad_zscore_vs_own_history` | `_mad_prior_24h/_7d` | no | no | no | — |
| **Group B -- counts** |
| `total_interface_errors` | no | no | no | yes | no | no | — |
| `total_interface_discards` | no | no | no | yes | no | no | — |
| `num_interfaces_with_errors` | no | no | no | yes | no | no | — |
| `num_interfaces_with_discards` | no | no | no | yes | no | no | — |
| `abnormal_interface_count` | no | no | no | yes | no | no | — |
| `snow_ticket_count` | no | no | no | yes | no | no | — |
| **Group B -- worst/spread-of-a-bucket snapshots** |
| `worst_interface_errors` | no | no | no | no | yes | no | — |
| `worst_interface_discards` | no | no | no | no | yes | no | — |
| `discards_min_in_bin` | no | no | no | no | yes | no | — |
| `discards_max_in_bin` | no | no | no | no | yes | no | — |
| `discards_std` | no | no | no | no | yes | no | — |
| **Group B -- redundant** |
| `discards_range` | no | no | no | no | no | no | no lag at all -- exactly `discards_max_in_bin - discards_min_in_bin` |
| **Group C** |
| `max_cpu_usage` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `avg_cpu_usage` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `memory_utilization_avg` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `interfaces_with_errors_data` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `interfaces_with_discards_data` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `interfaces_with_data` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `traffic_in_samples` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `traffic_out_samples` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `discards_samples` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `peer_count` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `peers_with_data` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `worst_established_time` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | known accepted imperfection -- same shape of skew as Group A on a small minority of devices, left unchanged (low feature importance) |
| `power_worst_dBm` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_out_of_range` flag, see below |
| `power_best_dBm` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `power_avg_dBm` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_stuck_reading` flag, see below |
| `tx_power_worst_dBm` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_out_of_range` flag, see below |
| `rx_power_worst_dBm` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_out_of_range` flag, see below |
| `bias_current_worst_mA` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `bias_current_min_mA` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `bias_current_avg_mA` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_stuck_reading` flag, see below |
| `module_temp_worst_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `module_temp_min_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `module_temp_avg_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_stuck_reading` flag, see below |
| `chassis_temp_worst_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `chassis_temp_min_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `chassis_temp_avg_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `temp_value` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_stuck_reading` flag, see below |
| `power_range_dBm` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | also has `_out_of_range` flag, see below |
| `power_volatility_std` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `power_volatility_range` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `bias_current_range_mA` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `module_temp_range_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| `chassis_temp_range_C` | yes | `_zscore_vs_own_history` | `_std_prior_24h/_7d` | no | no | no | — |
| **BGP flap-related (own query, not in LEVEL_METRICS/SPREAD_STAT_METRICS)** |
| `device_flap_count` | no | no | no | yes | no | no | — |
| `valid_flap_count` | no | no | no | yes | no | no | restart-confirmed flaps zeroed out of device_flap_count |
| `num_peers_flapped` | no | no | no | yes | no | no | — |
| `peer_down_sample_count` | no | no | no | yes | no | no | renamed from `any_peer_down_samples` |
| `flap_concentration` | no | no | no | no | no | yes | — |
| **Hardware abnormal flags (own query)** |
| `any_interface_abnormal` | no | no | no | yes | no | no | — |
| `bias_abnormal_any` | no | no | no | yes | no | no | — |
| `temp_abnormal_any` | no | no | no | yes | no | no | — |
| `chassis_temp_abnormal_any` | no | no | no | yes | no | no | — |
| `fan_notPresent_any` | no | no | no | yes | no | no | — |
| `fan_abnormal_any` | no | no | no | yes | no | no | — |
| `fan_tray_abnormal` | no | no | no | yes | no | no | — |
| `fru_power_abnormal_any` | no | no | no | yes | no | no | — |
| `ps_notPresent_any` | no | no | no | yes | no | no | — |
| `ps_abnormal_any` | no | no | no | yes | no | no | — |
| `any_hardware_abnormal` | no | no | no | yes | no | no | OR-rollup of the 9 above (excludes `any_interface_abnormal`) |
| **Coverage flags (14, own treatment)** |
| `has_health_data` | no | no | no | no | no | no | `_missing_bins_prior_24h/_7d`, `_bins_observed_lifetime`, `_hours_since_last_real_reading` |
| `has_interface_data` | no | no | no | no | no | no | same as above |
| `has_interface_oper_data` | no | no | no | no | no | no | same as above |
| `has_traffic_volatility_data` | no | no | no | no | no | no | same as above |
| `has_bgp_data` | no | no | no | no | no | no | same as above |
| `has_optical_power_data` | no | no | no | no | no | no | same as above |
| `has_optical_power_volatility_data` | no | no | no | no | no | no | same as above |
| `has_optical_bias_data` | no | no | no | no | no | no | same as above |
| `has_optical_temp_data` | no | no | no | no | no | no | same as above |
| `has_fan_state_data` | no | no | no | no | no | no | same as above |
| `has_device_temp_data` | no | no | no | no | no | no | same as above |
| `has_fan_tray_data` | no | no | no | no | no | no | same as above |
| `has_fru_power_data` | no | no | no | no | no | no | same as above |
| `has_power_supply_data` | no | no | no | no | no | no | same as above |
| **Reboot / restart** |
| `uptime_hrs` -> `recently_rebooted_24h` | no | no | no | yes | no | no | point-in-time flag + SUM lag |
| `uptime_hrs` -> `recently_rebooted_48h` | no | no | no | yes | no | no | point-in-time flag + SUM lag |
| `restart_flag` | no | no | no | yes | no | no | — |
| **Interface churn (built from the 59 Splunk log columns' down/up counts)** |
| `interface_churn_imbalance` | no | no | no | n/a (derived at `""`/`_prior_24h`/`_prior_7d` from already-lagged down/up) | no | no | down - up |
| `interface_churn_total` | no | no | no | n/a (derived at `""`/`_prior_24h`/`_prior_7d` from already-lagged down/up) | no | no | down + up |
| **Error/discard rate (derived ratios)** |
| `error_rate` | no | no | no | no | no | yes | errors / traffic |
| `discard_rate` | no | no | no | no | no | yes | discards / traffic |
| **Data-quality diagnostic flags** |
| `bias_current_avg_mA_stuck_reading` | no | no | no | yes | no | no | point-in-time flag + SUM lag |
| `power_avg_dBm_stuck_reading` | no | no | no | yes | no | no | point-in-time flag + SUM lag |
| `module_temp_avg_C_stuck_reading` | no | no | no | yes | no | no | point-in-time flag + SUM lag |
| `temp_value_stuck_reading` | no | no | no | yes | no | no | point-in-time flag + SUM lag |
| `power_worst_dBm_out_of_range` | no | no | no | no | no | no | point-in-time flag, diagnostic-only |
| `tx_power_worst_dBm_out_of_range` | no | no | no | no | no | no | point-in-time flag, diagnostic-only |
| `rx_power_worst_dBm_out_of_range` | no | no | no | no | no | no | point-in-time flag, diagnostic-only |
| `power_range_dBm_out_of_range` | no | no | no | no | no | no | point-in-time flag, diagnostic-only |
| **Static metadata (no derived feature)** |
| `interface_count` | no | no | no | no | no | no | constant per device (0 of 1,510 devices change) -- no derived feature needed |
| **Non-Dynatrace (Splunk-derived, for completeness)** |
| 59 Splunk log-derived base columns | no | no | no | yes | no | no | — |
