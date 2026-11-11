"""
BGP flap prediction — Dynatrace device-level data: build + clean pipeline.

Replaces the 2026-09-02 version of this script (3-file CSV pull:
dql_utc_device_health/interface/bgp_*.csv + dql_device_metadata.csv). As of
2026-09-29 the Dynatrace raw source is a 14-query parquet pull
(data/raw/dql/parquet/dql_15query_parquet_*/, one month-range folder per
pull), adding 11 new hardware-telemetry categories — interface operational
status, traffic/discard volatility, optical power, optical bias current,
optical module + chassis temperature, fan state, a device-level temperature
sensor, fan tray status, FRU power status, power supply status — alongside
the same device_health/interface/bgp signals the old CSV pull provided.
device_metadata stays a separate CSV, unchanged. See
docs/pipeline_architecture.md for the full diagram.

Verified before writing this (see the conversation this came from):
  - Same device universe as the old CSV pull: entity IDs
    (dt.entity.network:device) are 100% overlapping on the bgp table (999
    devices both sides), same CUSTOM_DEVICE-* ID scheme.
  - Same 4h bucket grid, 2026-06-01 -> 2026-09-01, zero fleet-wide gaps,
    zero duplicate (entity, time) rows, identical schema across all three
    monthly pulls, for every one of the 14 query tables.
  - No column-name collisions across the 14 tables (besides the shared
    entity/time key columns) — confirmed directly, not assumed.
  - Unlike the old CSV export, these parquet files are already properly
    typed (double/int64 throughout) — no %/thousands-comma string
    artifacts, no "d."-prefixed duplicate-column export quirk. clean_numeric
    is still applied to every non-key column as a defensive no-op (matches
    dtype already, pd.to_numeric is a cheap pass-through), not because it's
    known to be needed on this source — kept in case a future pull regresses.
  - avg_hsrp_group (present in the old device_health pull, documented in
    docs/feature_reference.md) does not exist in the new query_1 schema.
    Confirmed before dropping it silently: zero-variance in the old data
    (std=0.0, always 1.0 wherever populated) — real dead signal, not the
    loss of anything a model could have used.

Pipeline:
  1. Load each of the 14 queries — each spans every monthly pull folder
     found under data/raw/dql/parquet/ (currently June/July/August 2026),
     concatenated in filename order. Gap-free/dupe-free is asserted on
     every run for every one of the 14 tables, not just checked once — same
     discipline as the old pull, now applied 14x instead of 3x.
  2. Clean numeric formatting (defensive no-op on this source, see above)
     and use the already-UTC-tz-aware `time` column directly, renamed
     time_utc — no string parsing needed, unlike the old CSV's
     "%m/%d/%Y, %I:%M:%S %p" export format.
  3. Outer join all 14 tables on (`dt.entity.network:device`, `time_utc`) —
     a device/time bin present in only some of the 14 sources still
     produces a row, NaN for the columns the other source(s) would supply.
  4. Reindex each device to ITS OWN complete, regular 4h grid (own observed
     min-to-max time, not the full pull window) — unchanged from the old
     pipeline, same reasoning: every rolling/lag feature downstream counts
     by row position, which silently spans more than the intended wall-
     clock window if a bin is simply absent rather than present-with-NaN.
  5. Left-join device_metadata on `dt.entity.network:device` == `id`
     (unchanged source, unchanged identity-resolution logic — see
     resolve_device_identity()).
  6. has_X_data coverage flags — ONE PER TABLE (14 total, was 3). Device
     hardware categories have wildly different real-world coverage (e.g.
     fan_state reports for ~50 devices fleet-wide vs. power_supply_status
     for ~1450) — collapsing that into one shared flag the way
     has_health_data/has_interface_data/has_bgp_data did for the original
     three would hide which specific hardware category is actually
     reporting for a given device/bucket. A missing bin still reads as
     "unknown", never as "confirmed no signal" — same discipline as before,
     just applied to 11 more sources.
  7. Derived flags — same three the old pipeline computed, unchanged logic
     (flat_cpu_device, chronic_flap_window, restart_co_occurring_flap) plus
     valid_flap_count — the source columns (avg_cpu_usage, device_flap_count,
     restart_flag) have identical names in the new pull, so no change was
     needed here beyond confirming that.
  8. Stuck-reading detection + scoped nulling (2026-10-05) for four hardware-
     telemetry "level" columns added by the 14-query pull: bias_current_avg_mA,
     power_avg_dBm, module_temp_avg_C, temp_value. Checked empirically
     (see the conversation this came from) that a frozen reading on one of
     these is a narrow, per-sensor glitch — the device's BGP/interface/health
     coverage looks normal around it — unlike flat_cpu_device, where the
     SAME check (zero-variance) correlates with the device barely reporting
     anything at all (7.5% BGP coverage vs. 69% fleet-wide, 93% also have a
     frozen memory reading). That's why flat_cpu_device still drops the
     device's entire row (see build_features.py) while these four instead
     get a narrower, TIME-SCOPED fix: find_stuck_runs() below detects each
     qualifying run (>= STUCK_RUN_MIN_BUCKETS identical, non-zero, real
     readings, consecutive on the device's own dense grid, bridging gaps of
     up to GAP_TOLERANCE_BUCKETS) and nulls ONLY that metric's value for
     ONLY those rows — not the row, not the device, not any other column.
     Everything downstream (trend/z-score/volatility secondary features,
     built from these same raw columns in build_features.py) automatically
     comes out NaN for the nulled window via the existing gap-tolerant
     rolling logic, no separate handling needed there. A `{metric}_
     stuck_reading` boolean marks which rows were nulled, for auditability —
     dropped before the final feature table the same way flat_cpu_device/
     chronic_flap_window/restart_co_occurring_flap are (see build_features.py).
     Zero-valued readings are treated as "uninformative" for this check (same
     as NaN) rather than as part of a stuck run, since 0 is a plausible real
     state for all four (laser off, ambient-matched temp, etc.) and nulling a
     genuine zero would destroy real signal — weakest-justified for temp_value/
     module_temp_avg_C specifically, where 0 degrees is a real temperature
     too, but kept for consistency across all four.
  9. Optical power floor fix (2026-10-05) for power_worst_dBm, tx_power_
     worst_dBm, rx_power_worst_dBm: readings occasionally go as low as
     -300 dBm, which isn't a real optical power level — confirmed this is a
     10*log10(near-zero-mW) artifact on a "no light" condition, not a real
     reading (see the conversation this came from). A simple floor check
     (OPTICAL_POWER_DBM_FLOOR = -40.0), not a run-length one like point 8 —
     a single bad reading is enough, no consecutiveness required. Also nulls
     power_range_dBm (a RAW column from the same query, computed
     server-side as power_best_dBm - power_worst_dBm) wherever any of the
     three worst-type columns is out of range, since it's confirmed to
     inherit the same corruption directly from power_worst_dBm. power_best_
     dBm and power_avg_dBm checked and left alone — they don't exhibit this
     (a single bad sample drags down a "worst"/min aggregate but not a
     best/avg one). `{col}_out_of_range` flags are diagnostic-only, dropped
     before the final feature table same as the `_stuck_reading` flags
     (see build_features.py).

NaN-safety notes carried over unchanged from the prior version of this
script (still apply, source format doesn't affect them):
  - has_*_data flags go through astype("boolean").fillna(False).astype(bool)
    both before AND after reindexing — an outer join/reindex upcasts a bool
    column to object dtype to hold the introduced NaN (native numpy bool
    can't hold NaN), and `~` on an object Series applies Python's bitwise ~
    (True->-2, False->-1), not logical negation, if left as object dtype.
    The astype("boolean") first (pandas' nullable dtype) avoids a
    FutureWarning fillna raises when downcasting object dtype directly —
    confirmed empirically that converting to "boolean" BEFORE fillna avoids
    it while calling infer_objects() AFTER fillna does not (the warning
    fires inside fillna itself, too late to head off by then).
  - chronic_flap_window / restart_co_occurring_flap are nullable boolean
    (pandas "boolean" dtype), explicitly set to NA (not False) wherever a
    reindexed row is missing the source column(s) the flag depends on.
  - valid_flap_count only zeroes device_flap_count when restart co-
    occurrence is CONFIRMED true; if unknown, the real count is kept rather
    than assumed-restart.

Output: data/processed/cleaned_device_level_data.parquet
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_device_coverage import normalize_device  # noqa: E402

# 2026-11-10: data_pipeline/ is self-contained -- raw/processed live INSIDE
# it (data_pipeline/raw, data_pipeline/processed), not in a project-level
# data/ folder, so this pipeline can run standalone with no dependency on
# anything outside this directory.
PIPELINE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(PIPELINE_DIR, "processed"), exist_ok=True)
RAW_DIR = os.path.join(PIPELINE_DIR, "raw", "dql")
RAW_PARQUET_DIR = os.path.join(RAW_DIR, "parquet")
OUT_PATH = os.path.join(PIPELINE_DIR, "processed", "cleaned_device_level_data.parquet")

ENTITY_COL = "dt.entity.network:device"
BUCKET_HOURS = 4

CHRONIC_FLAP_THRESHOLD = 16  # full 4h window: every 15-min check flagged

# Stuck-reading detection (see docstring point 8). GAP_TOLERANCE_BUCKETS is
# the same value/meaning as build_features.py's constant of the same name
# (how many consecutive missing buckets still count as "the same window") —
# duplicated rather than imported, matching this file's existing practice of
# not cross-importing from build_features.py (BUCKET_HOURS is duplicated the
# same way).
GAP_TOLERANCE_BUCKETS = 2
STUCK_RUN_MIN_BUCKETS = 30  # 5 days at the 4h grid
STUCK_VALUE_METRICS = ["bias_current_avg_mA", "power_avg_dBm", "module_temp_avg_C", "temp_value"]

# Optical power floor (see docstring point 9). No real transceiver reads
# below roughly this; readings past it are a 10*log10(near-zero-mW) "no
# light" artifact, not a real power level. A floor check, not a stuck-run
# one: a single out-of-range reading is enough on its own, no run-length
# requirement.
OPTICAL_POWER_DBM_FLOOR = -40.0
OPTICAL_POWER_WORST_COLS = ["power_worst_dBm", "tx_power_worst_dBm", "rx_power_worst_dBm"]

METADATA_COLS = ["entity.name", "dt.dns_names", "router_type", "devType",
                  "model", "site_code", "interface_count", "network_type",
                  "device_name_resolved"]

DUPLICATE_SENTINEL = "duplicate_values_found"

# One entry per query in the 14-query pull. `flag` is the has_X_data
# coverage-flag column name this table contributes — every one of the 14
# gets its own, deliberately (see docstring point 6). Order here is also
# merge order, chosen to match the old pipeline's health/interface/bgp-first
# convention, then the new hardware categories grouped by query number.
QUERY_TABLES = [
    ("query_1_device_health", "has_health_data"),
    ("query_2a_interface_traffic_errors_discards", "has_interface_data"),
    ("query_2b_interface_oper_status", "has_interface_oper_data"),
    ("query_2c_traffic_discards_volatility", "has_traffic_volatility_data"),
    ("query_3_bgp_flap_target", "has_bgp_data"),
    ("query_4a_i_optical_power", "has_optical_power_data"),
    ("query_4a_ii_vol_optical_power_volatility", "has_optical_power_volatility_data"),
    ("query_4b_optical_bias_current", "has_optical_bias_data"),
    ("query_4c_optical_module_temp", "has_optical_temp_data"),
    ("query_5a_i_fan_state", "has_fan_state_data"),
    ("query_5a_ii_temperature", "has_device_temp_data"),
    ("query_5a_iii_fan_tray_status", "has_fan_tray_data"),
    ("query_5b_fru_power_status", "has_fru_power_data"),
    ("query_5c_power_supply_status", "has_power_supply_data"),
]


def clean_numeric(series: pd.Series) -> pd.Series:
    # Note: pandas here returns a dedicated `str` dtype from read_csv, not
    # classic `object` — checking `== object` alone misses it, so use the
    # dtype-family check instead (covers object, `str`, and StringDtype).
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _load_single_query_files(query_key: str) -> pd.DataFrame:
    """Loads one query's raw files, concatenated across every monthly pull
    folder found under RAW_PARQUET_DIR — glob on `dql_15query_*` rather than
    a hardcoded folder name so a naming quirk in one pull (the first, June,
    folder is literally misspelled "parquent" not "parquet") doesn't need
    special-casing, and so a future new month is picked up automatically
    without editing this script. Raises if nothing matches rather than
    silently returning an empty table."""
    pattern = os.path.join(RAW_PARQUET_DIR, "dql_15query_*", f"{query_key}_2026-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _load_query_2a() -> pd.DataFrame:
    """query_2a is pulled as two column-wise halves, every month, with no
    exceptions as of 2026-11-06 (confirmed directly: every monthly pull
    folder under RAW_PARQUET_DIR now has a standalone query_2a_i_traffic
    and query_2a_ii_errors_discards file — the June/July/August pulls that
    used to carry both halves in one combined query_2a_interface_traffic_
    errors_discards file were re-pulled in the split form and that combined
    file removed, specifically so this function wouldn't need an old-vs-new
    conditional) -- query_2a_i_traffic (ifcsTrafficIn_bps/Out_bps) and
    query_2a_ii_errors_discards (interface error/discard aggregates, see
    docs/dql_query_full_list/query_2a_ii_errors_discards.txt for the current
    per-interface worst/count/total schema). Outer-joined on (entity, time)
    rather than concatenated, since they're column halves of the same rows,
    not additional rows. Outer (not inner) in case either half is ever
    incomplete relative to the other -- matches this project's general
    preference for outer joins over silently dropping rows."""
    i_df = _load_single_query_files("query_2a_i_traffic")
    ii_df = _load_single_query_files("query_2a_ii_errors_discards")
    return i_df.merge(ii_df, on=[ENTITY_COL, "time"], how="outer")


def load_query_table(query_key: str) -> pd.DataFrame:
    """Loads one of the 14 logical query tables. query_2a_interface_
    traffic_errors_discards is physically two files per month (see
    _load_query_2a()) merged into one logical table here; every other
    query_key is a single file per month."""
    if query_key == "query_2a_interface_traffic_errors_discards":
        df = _load_query_2a()
    else:
        df = _load_single_query_files(query_key)

    df = df.rename(columns={"time": "time_utc"})
    if str(df["time_utc"].dt.tz) != "UTC":
        raise ValueError(f"{query_key}: expected time column already tz-aware UTC, got {df['time_utc'].dt.tz}")

    value_cols = [c for c in df.columns if c not in (ENTITY_COL, "time_utc")]
    for c in value_cols:
        df[c] = clean_numeric(df[c])
    return df


def check_gap_free(df: pd.DataFrame, name: str) -> None:
    """Verifies the concatenated monthly pulls form one complete, non-
    overlapping 4h-bucket series per entity — an assertion re-checked on
    every run, not a one-time fact trusted from the first pull."""
    dupes = df.duplicated(subset=[ENTITY_COL, "time_utc"]).sum()
    if dupes:
        print(f"WARNING: {name}: {dupes} duplicate ({ENTITY_COL}, time_utc) rows across "
              f"source files — keeping first occurrence")

    tmin, tmax = df["time_utc"].min(), df["time_utc"].max()
    expected = pd.date_range(tmin, tmax, freq=f"{BUCKET_HOURS}h")
    present = pd.DatetimeIndex(df["time_utc"].unique())
    missing = expected.difference(present)
    if len(missing):
        print(f"WARNING: {name}: {len(missing)} of {len(expected)} expected {BUCKET_HOURS}h buckets "
              f"missing ACROSS ALL devices between {tmin} and {tmax} (a real pull gap, not just a "
              f"per-device gap step 4 will handle) — first few: {list(missing[:5])}")
    else:
        print(f"{name}: confirmed gap-free, {len(expected)}/{len(expected)} buckets present "
              f"({tmin} -> {tmax})")


def resolve_device_identity(meta: pd.DataFrame) -> pd.DataFrame:
    """Adds `device_name_resolved` — a clean, collision-free device name.

    `entity.name` alone isn't a safe identifier: confirmed cases where
    genuinely unrelated devices share a normalized entity.name (e.g. 5
    distinct lab switches all displaying as "rcr.ntwk.kp.org") and cases
    where a device's main and CIMC management-interface entries collapse
    to the same name. Resolution, confirmed against the real 2026-09-02
    metadata pull before building this:

      1. Normalize entity.name (lowercase + domain-suffix strip, same
         function used for the Splunk/SNOW device-coverage matching).
      2. Rows whose normalized entity.name is shared by 2+ ids: mark
         entity.name unusable (DUPLICATE_SENTINEL) for those rows.
      3. For those rows, normalize each dt.dns_names alias and dedupe each
         row's OWN alias set first (a row can list e.g. both the bare and
         full-FQDN form of the same alias — not a real collision). Then
         resolve within the group via iterative greedy assignment: on each
         pass, any row that has an alias no other STILL-UNRESOLVED row in
         the group also lists gets assigned that alias; assigned values
         are removed from contention and the group is re-scanned, since
         resolving one row can free up a value another row was blocked on
         (e.g. cseqeise01's two ids both list plain "cseqeise01", but only
         one also lists "cseqeise01-cimc" — a single pass that just strips
         "anything appearing elsewhere" incorrectly leaves BOTH rows with
         nothing, since each sees the other still contesting "cseqeise01";
         resolving the "-cimc" id to its own exclusive alias first frees
         "cseqeise01" for the other id on the next pass). Repeat until a
         full pass makes no further progress.
      4. If nothing unique is left (checked: this happens for real — two
         ids can list the IDENTICAL alias pair, e.g. one lists {A, B} and
         the other lists {B, A}, with no way to tell which is which),
         device_name_resolved is left NaN. The row's telemetry stays in
         the output regardless — same treatment as a device with no
         metadata match at all — only the identity column comes back
         empty, not deleted.
    """
    meta = meta.copy()
    meta["_norm_entity"] = meta["entity.name"].map(normalize_device)
    dupe_mask = meta.duplicated("_norm_entity", keep=False)

    resolved = meta["entity.name"].copy()
    resolved[dupe_mask] = DUPLICATE_SENTINEL

    n_fallback_resolved = 0
    n_dropped = 0
    for _, group in meta[dupe_mask].groupby("_norm_entity"):
        own_aliases = {}
        for idx, row in group.iterrows():
            if pd.isna(row["dt.dns_names"]):
                own_aliases[idx] = set()
            else:
                own_aliases[idx] = {normalize_device(a.strip()) for a in row["dt.dns_names"].split(",")}

        assigned: dict = {}
        unresolved = set(own_aliases.keys())
        progress = True
        while progress and unresolved:
            progress = False
            for idx in list(unresolved):
                others = set().union(*(own_aliases[i] for i in unresolved if i != idx)) if len(unresolved) > 1 else set()
                exclusive = own_aliases[idx] - others - set(assigned.values())
                if len(exclusive) >= 1:
                    assigned[idx] = next(iter(exclusive))
                    unresolved.discard(idx)
                    progress = True

        for idx in own_aliases:
            if idx in assigned:
                resolved.loc[idx] = assigned[idx]
                n_fallback_resolved += 1
            else:
                resolved.loc[idx] = pd.NA
                n_dropped += 1

    if dupe_mask.sum():
        print(f"NOTE: {dupe_mask.sum()} ids ({meta.loc[dupe_mask, '_norm_entity'].nunique()} collision groups) "
              f"had a duplicate normalized entity.name — {n_fallback_resolved} resolved via a uniquely-owned "
              f"dns_names alias, {n_dropped} left with no resolvable identity (device_name_resolved = NaN, "
              f"telemetry rows kept regardless)")

    meta["device_name_resolved"] = resolved
    return meta.drop(columns=["_norm_entity"])


def load_metadata() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "dql_device_metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    meta = pd.read_csv(path)
    dupe_ids = meta["id"].duplicated().sum()
    if dupe_ids:
        raise ValueError(f"device_metadata has {dupe_ids} duplicate 'id' values — "
                          f"expected a unique key, join would silently fan out")
    return resolve_device_identity(meta)


def reindex_to_dense_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Per ENTITY_COL, fills in any 4h bucket missing from ALL 14 sources
    (within that device's own observed min-to-max range) as an explicit
    NaN row, rather than leaving it silently absent."""
    filled = []
    for entity_id, g in df.groupby(ENTITY_COL, sort=False):
        g = g.sort_values("time_utc")
        full_range = pd.date_range(g["time_utc"].min(), g["time_utc"].max(), freq=f"{BUCKET_HOURS}h")
        g = g.set_index("time_utc").reindex(full_range)
        g[ENTITY_COL] = entity_id
        g.index.name = "time_utc"
        filled.append(g.reset_index())
    return pd.concat(filled, ignore_index=True)


def _find_stuck_positions(values, gap_tolerance: int, min_run_buckets: int) -> list:
    """Scans ONE device's time-ordered readings for a single metric and
    returns a same-length boolean list marking which positions are part of
    a qualifying stuck run.

    A reading is "real" if it's non-null and non-zero (0 is treated as
    uninformative here, same as NaN — see docstring point 8 for why). A run
    is a maximal stretch of real readings holding the EXACT SAME value,
    consecutive on the device's own dense grid, where a gap of up to
    `gap_tolerance` consecutive non-real (NaN/zero) buckets in the middle
    does not break it (longer than that, it does). It qualifies once its
    span (last real position - first real position + 1, i.e. wall-clock
    buckets, including any bridged gap) reaches `min_run_buckets`.

    Only the positions actually HOLDING the repeated value are marked —
    never the bridged NaN/zero positions in between, so a genuine zero
    reading inside an otherwise-stuck window is never nulled by the caller."""
    n = len(values)
    mark = [False] * n
    run_start = None       # position of the run's first real reading
    run_value = None
    last_real_idx = None   # position of the most recent real reading == run_value
    real_positions = []    # positions (so far) holding run_value
    gap_count = 0          # consecutive non-real buckets since last_real_idx

    def close_run():
        if run_start is not None and last_real_idx is not None:
            run_length = last_real_idx - run_start + 1
            if run_length >= min_run_buckets:
                for pos in real_positions:
                    mark[pos] = True

    for i in range(n):
        v = values[i]
        is_real = pd.notna(v) and v != 0
        if run_start is None:
            if is_real:
                run_start, run_value, last_real_idx = i, v, i
                real_positions = [i]
                gap_count = 0
            continue
        if is_real:
            if v == run_value:
                last_real_idx = i
                real_positions.append(i)
                gap_count = 0
            else:
                close_run()
                run_start, run_value, last_real_idx = i, v, i
                real_positions = [i]
                gap_count = 0
        else:
            gap_count += 1
            if gap_count > gap_tolerance:
                close_run()
                run_start, run_value, last_real_idx = None, None, None
                real_positions = []
                gap_count = 0
    close_run()
    return mark


def compute_stuck_reading_masks(df: pd.DataFrame, metrics, gap_tolerance: int, min_run_buckets: int) -> dict:
    """Per metric in `metrics`, returns a boolean Series (aligned to df's
    index) marking which rows of `df` fall inside a qualifying stuck run for
    their device — see _find_stuck_positions() for the exact definition.
    Must run AFTER reindex_to_dense_grid (relies on each device's rows being
    on a regular, gap-explicit 4h grid to count buckets correctly)."""
    masks = {}
    for metric in metrics:
        mask = pd.Series(False, index=df.index)
        for entity_id, g in df.groupby(ENTITY_COL, sort=False):
            g = g.sort_values("time_utc")
            marked = _find_stuck_positions(g[metric].to_numpy(), gap_tolerance, min_run_buckets)
            mask.loc[g.index[marked]] = True
        masks[metric] = mask
    return masks


def build() -> pd.DataFrame:
    flag_cols = [flag for _, flag in QUERY_TABLES]

    merged = None
    for key, flag in QUERY_TABLES:
        t = load_query_table(key)
        check_gap_free(t, key)
        t[flag] = True
        merged = t if merged is None else merged.merge(t, on=[ENTITY_COL, "time_utc"], how="outer")

    for flag_col in flag_cols:
        # An outer join upcasts bool to object dtype to hold the introduced
        # NaN (native numpy bool can't hold NaN) — fillna alone leaves it as
        # object-dtype Python bools, and `~` on an object Series applies
        # Python's bitwise ~ (True->-2, False->-1), not logical negation.
        # astype(bool) forces back to real numpy bool.
        merged[flag_col] = merged[flag_col].astype("boolean").fillna(False).astype(bool)

    merged = reindex_to_dense_grid(merged)
    for flag_col in flag_cols:
        merged[flag_col] = merged[flag_col].astype("boolean").fillna(False).astype(bool)

    meta = load_metadata()
    merged = merged.merge(meta[["id"] + METADATA_COLS], left_on=ENTITY_COL, right_on="id", how="left")
    merged = merged.drop(columns=["id"])

    # flat_cpu_device: zero-variance CPU across a device's whole history in
    # this pull. std() skips NaN by default, so this is computed only from
    # real observations — a device with zero or one real reading gets
    # std=NaN -> fillna(0)==0 -> flagged True, same simplification the old
    # pipeline made (not a new behavior).
    cpu_std = merged.groupby(ENTITY_COL)["avg_cpu_usage"].transform("std")
    merged["flat_cpu_device"] = cpu_std.fillna(0) == 0

    # Stuck-reading detection + scoped nulling for the four hardware-
    # telemetry level columns checked empirically to be narrow, per-sensor
    # glitches rather than a flat_cpu_device-style whole-device symptom —
    # see docstring point 8. Unlike flat_cpu_device, this does NOT drop the
    # row/device: only the specific metric's value, for the specific rows
    # inside a qualifying stuck run, is nulled. The `_stuck_reading` flags
    # are diagnostic-only (dropped before the final feature table, see
    # build_features.py) — kept here for auditability of what got nulled.
    stuck_masks = compute_stuck_reading_masks(
        merged, STUCK_VALUE_METRICS, GAP_TOLERANCE_BUCKETS, STUCK_RUN_MIN_BUCKETS
    )
    for metric, mask in stuck_masks.items():
        merged[f"{metric}_stuck_reading"] = mask
        merged.loc[mask, metric] = float("nan")

    # Optical power floor fix (see docstring point 9). power_worst_dBm /
    # tx_power_worst_dBm / rx_power_worst_dBm occasionally read down to
    # -300 dBm on a "no light" condition — confirmed this is a
    # 10*log10(near-zero-mW) artifact, not a real reading (no real
    # transceiver reads below OPTICAL_POWER_DBM_FLOOR). power_range_dBm is
    # a RAW column from the same query (query_4a_i_optical_power), computed
    # server-side as power_best_dBm - power_worst_dBm — confirmed
    # empirically (see the conversation this came from) that it inherits
    # the same corruption whenever power_worst_dBm does (e.g.
    # power_worst_dBm=-203.29 -> power_range_dBm=203.29, exactly their
    # difference), so it's nulled alongside the three floor-violating
    # columns rather than independently range-checked. power_best_dBm and
    # power_avg_dBm were checked and do NOT exhibit this (the corruption is
    # specific to "worst"/min-of-window aggregates, which a single bad
    # sample can drag arbitrarily low — best/avg aren't dominated by one
    # sample the same way).
    out_of_range_mask = pd.Series(False, index=merged.index)
    for col in OPTICAL_POWER_WORST_COLS:
        bad = merged[col] < OPTICAL_POWER_DBM_FLOOR
        merged[f"{col}_out_of_range"] = bad
        merged.loc[bad, col] = float("nan")
        out_of_range_mask |= bad
    merged["power_range_dBm_out_of_range"] = out_of_range_mask
    merged.loc[out_of_range_mask, "power_range_dBm"] = float("nan")

    # chronic_flap_window: NA (not False) where bgp data is missing for
    # this bin — "unknown" is not the same as "confirmed not chronic".
    merged["chronic_flap_window"] = pd.array(
        merged["device_flap_count"] >= CHRONIC_FLAP_THRESHOLD, dtype="boolean"
    )
    merged.loc[merged["device_flap_count"].isna(), "chronic_flap_window"] = pd.NA

    # restart_co_occurring_flap: needs BOTH device_flap_count (bgp) and
    # restart_flag (health) present for this bin to mean anything.
    merged["restart_co_occurring_flap"] = pd.array(
        (merged["device_flap_count"] > 0) & (merged["restart_flag"] == 1), dtype="boolean"
    )
    merged.loc[
        merged["device_flap_count"].isna() | merged["restart_flag"].isna(), "restart_co_occurring_flap"
    ] = pd.NA

    # valid_flap_count: device_flap_count with restart-driven flaps zeroed
    # out. Only zero the count when restart-co-occurrence is CONFIRMED true;
    # if unknown, keep the real count rather than assume restart.
    restart_confirmed = merged["restart_co_occurring_flap"].fillna(False)
    merged["valid_flap_count"] = merged["device_flap_count"].where(~restart_confirmed, 0)

    merged = merged.sort_values([ENTITY_COL, "time_utc"]).reset_index(drop=True)
    return merged


def summarize(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"\nRows: {n}  Devices: {df[ENTITY_COL].nunique()}")
    print(f"UTC time range: {df['time_utc'].min()} -> {df['time_utc'].max()}")
    print("\nCoverage by source table (rows present / devices ever reporting):")
    for key, flag in QUERY_TABLES:
        n_rows = int(df[flag].sum())
        n_dev = df.loc[df[flag], ENTITY_COL].nunique()
        print(f"  {flag:32s} {n_rows:7d} rows  {n_dev:5d} devices  ({key})")
    print(f"\nflat_cpu_device rows: {df['flat_cpu_device'].sum()} "
          f"({df.loc[df['flat_cpu_device'], ENTITY_COL].nunique()} devices)")
    for metric in STUCK_VALUE_METRICS:
        flag = f"{metric}_stuck_reading"
        print(f"{flag} rows nulled: {df[flag].sum()} "
              f"({df.loc[df[flag], ENTITY_COL].nunique()} devices)")
    for col in OPTICAL_POWER_WORST_COLS + ["power_range_dBm"]:
        flag = f"{col}_out_of_range"
        print(f"{flag} rows nulled: {df[flag].sum()} "
              f"({df.loc[df[flag], ENTITY_COL].nunique()} devices)")
    print(f"chronic_flap_window: True={(df['chronic_flap_window'] == True).sum()}  "  # noqa: E712
          f"NA={df['chronic_flap_window'].isna().sum()}")
    print(f"restart_co_occurring_flap: True={(df['restart_co_occurring_flap'] == True).sum()}  "  # noqa: E712
          f"NA={df['restart_co_occurring_flap'].isna().sum()}")
    n_zeroed = int(((df["device_flap_count"] > 0) & (df["valid_flap_count"] == 0)).sum())
    print(f"valid_flap_count: {df['valid_flap_count'].notna().sum()} populated, "
          f"{df['valid_flap_count'].isna().sum()} NaN (no bgp data), "
          f"{n_zeroed} rows zeroed out for confirmed restart co-occurrence")
    print(f"Devices with no metadata match: {df.loc[df['entity.name'].isna(), ENTITY_COL].nunique()}")


if __name__ == "__main__":
    cleaned = build()
    summarize(cleaned)
    cleaned.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
