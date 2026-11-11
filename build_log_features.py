"""
BGP flap prediction — log feature reshape: group -> pivot.

Takes cleaned_log_data.parquet (long format: one row per device, time bucket,
mnemonic, message_text) and turns it into a wide, device-per-4h-bucket table
of the 59 feature columns settled in docs/log_message_text_to_feature_reference.csv —
that file is the single source of truth for which message_text values are
kept, dropped, and which final column each kept one feeds. This script does
not make any new grouping decisions; it applies the ones already recorded.

Order of operations matters here, and this is deliberately group-then-pivot,
not the reverse: pivoting first would produce ~100 narrow columns that then
need summing back together, versus assigning the final column up front and
letting one groupby+pivot produce the right shape directly. See the
"pivot order" discussion in the conversation this pipeline came out of.

Value used per (device, time, feature_col_name): sum of Splunk's own
log_count field (not a row count) — this preserves the intensity signal
(e.g. the same warning repeating many times within one 4h window is a
stronger signal than it firing once), consistent with device_flap_count
itself being a count, not a boolean.

Three outputs:
  - data/processed/log_features_wide.parquet: (device, time) grain, one
    column per feature. Only contains rows where the device actually had
    log activity in that bucket — does NOT cover the full DQL device/time
    grid. That join happens in build_features.py.
  - data/processed/log_device_coverage.csv: which devices have ANY log
    visibility at all (union of the authoritative device list and devices
    observed with real events — see docs/log_validation_findings.md
    Section 1 for why the union, not just the device list, is used). This
    is what lets build_features.py distinguish "confirmed zero activity"
    (fill 0) from "no visibility into this device" (leave NaN) when it
    joins this table onto the full DQL grid.
  - data/processed/log_device_platform.csv: device -> hardware platform
    (deviceModelName) + device tier (deviceTier), one row per device.

**Platform/tier source changed (2026-09-22).** Previously `deviceModelName`
rode along on every log row (from a Splunk-side lookup baked into the old
SPL), so it was read directly off `cleaned_log_data.parquet`. The corrected
SPL (`docs/spl_to_get_log_data.txt`) no longer carries it — platform is now
a separate CMDB-style export, `data/raw/logs/network_device_metadata.csv`
(fqdn/hostname-keyed, ~3,400 distinct devices, independent of whether a
device has any log rows at all — actually WIDER coverage than the old
per-log-row approach, since it includes devices with metadata but zero log
volume). `build_device_platform()` below reads that file directly instead
of `logs`. Top-N-by-frequency-plus-"Others" binning for the one-hot
`platform_*` features (previously done server-side by Splunk's
`top limit=10 ... useother=true`) also moved — it's now done in
`build_features.py`'s `merge_platform()`, AFTER joining onto the actual
modeled device population, so "top N" reflects devices really in the
feature table rather than a Splunk-wide count decided before anyone knew
which devices would end up there. This file stores the raw, unbinned
`device_platform`/`device_tier` strings only.
"""
import os
import pandas as pd

# 2026-11-10: raw/processed moved to data_pipeline/raw, data_pipeline/
# processed (self-contained pipeline). docs/ stays at the project root --
# it's curated reference material, not pipeline data -- so this is the
# one path here that still needs ROOT_DIR.
PIPELINE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.join(PIPELINE_DIR, "..")
LOG_PATH = os.path.join(PIPELINE_DIR, "processed", "cleaned_log_data.parquet")
REF_PATH = os.path.join(ROOT_DIR, "docs", "log_message_text_to_feature_reference.csv")
COVERAGE_LIST_PATH = os.path.join(PIPELINE_DIR, "raw", "loging_devcies.csv")
METADATA_PATH = os.path.join(PIPELINE_DIR, "raw", "logs", "network_device_metadata.csv")

OUT_WIDE_PATH = os.path.join(PIPELINE_DIR, "processed", "log_features_wide.parquet")
OUT_COVERAGE_PATH = os.path.join(PIPELINE_DIR, "processed", "log_device_coverage.csv")
OUT_PLATFORM_PATH = os.path.join(PIPELINE_DIR, "processed", "log_device_platform.csv")


def build_wide_features(logs: pd.DataFrame, ref: pd.DataFrame, expected_cols: set) -> pd.DataFrame:
    kept_ref = ref[ref["category"] != "dropped"][["mnemonic", "message_text", "feature_col_name"]]

    # Join on a WHITESPACE-NORMALIZED message_text, not the raw column --
    # confirmed 2026-10-01 that Cisco's logged message text silently dropped
    # a trailing space on at least 18 mnemonics starting in September (e.g.
    # FAN_OK: "fan rotating " -> "fan rotating"), which broke this join's
    # exact-string match and silently zeroed 13 of the 59 feature columns
    # for the entire month -- not a data-availability gap (the raw mnemonic
    # events were all still being logged, confirmed directly against
    # cleaned_log_data.parquet), a pure string-matching brittleness bug.
    # Stripped copies are used ONLY as the join key, on both sides, so a
    # trivial logging-format wobble like this can't silently zero a feature
    # again; the reference file's own stored message_text (and `logs`'
    # original column) are left untouched otherwise.
    logs = logs.copy()
    kept_ref = kept_ref.copy()
    logs["_message_text_norm"] = logs["message_text"].str.strip()
    kept_ref["_message_text_norm"] = kept_ref["message_text"].str.strip()
    kept_ref = kept_ref.drop(columns=["message_text"])

    merged = logs.merge(kept_ref, on=["mnemonic", "_message_text_norm"], how="inner")
    dropped_rows = len(logs) - len(merged)
    print(f"Kept {len(merged)} / {len(logs)} log rows ({dropped_rows} dropped per reference file)")

    grouped = merged.groupby(["sys.name", "time_utc", "feature_col_name"])["log_count"].sum().reset_index()
    wide = grouped.pivot(index=["sys.name", "time_utc"], columns="feature_col_name", values="log_count")
    wide = wide.fillna(0).reset_index()
    wide.columns.name = None

    # With 59 categories (up from 31), it's now realistic for a rare
    # category to have zero events across an ENTIRE pull window by chance
    # — the pivot above would simply omit that column rather than produce
    # a real zero, and downstream code would never know the feature was
    # supposed to exist. Add any missing expected column as all-zero
    # (a confirmed-absent category is a legitimate 0, not unknown/NaN —
    # different from the per-device coverage NaN-vs-0 distinction, which
    # is about whether a DEVICE has any log visibility at all, not whether
    # a rare event type happened to fire anywhere in the pull).
    missing = expected_cols - set(wide.columns)
    if missing:
        print(f"NOTE: {len(missing)} expected feature column(s) had zero events in this pull "
              f"(filled as all-zero, not dropped): {sorted(missing)}")
        for c in missing:
            wide[c] = 0.0

    return wide


def build_device_coverage(logs: pd.DataFrame) -> pd.DataFrame:
    coverage_list = pd.read_csv(COVERAGE_LIST_PATH)
    coverage_devices = set(coverage_list["orig_host"].str.lower().unique())
    event_devices = set(logs["sys.name"].str.lower().unique())
    has_coverage = coverage_devices | event_devices
    return pd.DataFrame({"sys_name_lower": sorted(has_coverage), "has_log_coverage": True})


def build_device_platform() -> pd.DataFrame:
    """Reads data/raw/logs/network_device_metadata.csv directly — no longer
    derived from logs (see module docstring). Keyed on fqdn, falling back to
    hostname only when fqdn is missing (331 of 3,757 rows) — both are
    lowercased directly rather than run through normalize_device() here,
    since that function lives in build_device_coverage.py and this file
    doesn't otherwise depend on it; build_features.py's merge_platform()
    re-normalizes this output the same way it does every other Splunk-side
    table, so nothing is lost by keeping this step simple.

    Handles two real data-quality issues found in the raw file: 357 of
    3,757 rows share a duplicate fqdn with another row. 355 of those are
    harmless exact/near-exact repeats (dropped via drop_duplicates). 2
    genuinely disagree on deviceModelName for the same fqdn — left
    unresolved (excluded, not guessed) rather than arbitrarily picking one,
    same discipline as the dns_names ambiguity handling in
    build_device_coverage.py."""
    meta = pd.read_csv(METADATA_PATH)

    key = meta["fqdn"].str.lower()
    key = key.where(key.notna(), meta["hostname"].str.lower())
    meta = meta.assign(sys_name_lower=key).dropna(subset=["sys_name_lower"])

    meta = meta.drop_duplicates(subset=["sys_name_lower", "deviceModelName", "deviceTier"])
    ambiguous = meta["sys_name_lower"][meta["sys_name_lower"].duplicated(keep=False)].unique()
    if len(ambiguous):
        print(f"NOTE: {len(ambiguous)} device(s) have conflicting metadata rows (different "
              f"deviceModelName/deviceTier under the same key) — left unresolved rather than "
              f"guessed: {sorted(ambiguous)[:10]}{'...' if len(ambiguous) > 10 else ''}")
        meta = meta[~meta["sys_name_lower"].isin(ambiguous)]

    return meta[["sys_name_lower", "deviceModelName", "deviceTier"]].rename(
        columns={"deviceModelName": "device_platform", "deviceTier": "device_tier"}
    )


if __name__ == "__main__":
    logs = pd.read_parquet(LOG_PATH)
    ref = pd.read_csv(REF_PATH).fillna({"feature_col_name": ""})

    expected_cols = set(ref.loc[ref["category"] != "dropped", "feature_col_name"].unique())
    # 2026-09-22: expanded from 31 to 59 base feature columns after an
    # exhaustive re-triage of all 526 mnemonics found in a full-index
    # message inventory (data/raw/all_message_text.csv) — see
    # docs/log_mnemonic_triage.md for the full reasoning per new column.
    assert len(expected_cols) == 59, f"Expected 59 feature columns in reference file, found {len(expected_cols)}"

    wide = build_wide_features(logs, ref, expected_cols)
    actual_cols = set(wide.columns) - {"sys.name", "time_utc"}
    assert actual_cols == expected_cols, f"Column mismatch: {actual_cols ^ expected_cols}"

    coverage = build_device_coverage(logs)
    platform = build_device_platform()

    wide.to_parquet(OUT_WIDE_PATH, index=False)
    coverage.to_csv(OUT_COVERAGE_PATH, index=False)
    platform.to_csv(OUT_PLATFORM_PATH, index=False)

    print(f"\nWrote {OUT_WIDE_PATH}: {len(wide)} rows, {wide['sys.name'].nunique()} devices, {len(actual_cols)} feature columns")
    print(f"Wrote {OUT_COVERAGE_PATH}: {len(coverage)} devices with log coverage")
    print(f"Wrote {OUT_PLATFORM_PATH}: {len(platform)} devices, "
          f"{platform['device_platform'].nunique()} distinct platforms, "
          f"{platform['device_tier'].nunique()} distinct tiers")
