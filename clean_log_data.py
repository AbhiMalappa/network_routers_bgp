"""
BGP flap prediction — Splunk Cisco IOS log cleaning pipeline.

Loads the raw splunk_logs_utc_*.csv exports (data/raw/logs/): hourly-bucketed
(4h, in practice — see below) log-event counts by device + mnemonic.

Timestamps carry an explicit +0000 offset in the source (true, unambiguous
UTC — nothing to infer here, unlike the DQL data).

Bucket width is 4h, confirmed by inspection (timestamps only ever land on
UTC hours 0, 4, 8, 12, 16, 20) — same width and, per
docs/bgp_flap_modeling_handoff.md, the same alignment as the dql_utc data,
so the two can be joined on (device, time) directly with no re-bucketing.

Caution baked into this cleaning step, not left for downstream code to
rediscover: several mnemonics that look BGP-related by name are not.
`ADJCHG`/`NBRCHG` are OSPF (their message text uses OSPF FSM states —
loading/full/way down/dead timer — not BGP's Idle/OpenConfirm/Established).
The genuinely BGP-relevant mnemonics found so far are `ADJCHANGE` (real BGP
adjacency state transitions) and `NOTIFICATION` (BGP session-close reason
codes, e.g. bfd down, hold timer expired). This script does not filter to
just those — it keeps all mnemonics — but tags each row with `is_bgp_event`
so downstream feature-building can select deliberately rather than by
guessing from mnemonic names.

**Correction (2026-09-22), found via the `trap_msg` field (Cisco's full
`FACILITY-SEVERITY-MNEMONIC` string, e.g. `BGP-5-ADJCHANGE`) becoming
available in a raw pull for the first time:** `ADJCHANGE` is NOT
exclusively BGP the way the paragraph above assumed — Cisco reuses the same
mnemonic name across protocols. Checked against a fleet-wide unfiltered
message-text inventory (`data/raw/all_message_text.csv`): of all `ADJCHANGE`
log volume, only ~54% carries `facility` `BGP` or `BGP_SESSION`; ~46% is
actually `facility=OSPF` (plus a small `CLNS`/ISIS sliver). `NOTIFICATION`,
by contrast, checked out 100% `facility=BGP` — no correction needed there.

**Practical impact:** the feature table itself was never contaminated —
`build_log_features.py` joins on exact `message_text` content against
`docs/log_message_text_to_feature_reference.csv`, and OSPF's `ADJCHANGE` message text
(`"ospf nbr ethernet went exstart"` etc.) reads nothing like BGP's
(`"neighbor down bfd adjacency down"` etc.) so it was never matched into a
`bgp_*` feature column even under the old mnemonic-only logic — verified
directly, zero OSPF/CLNS-facility rows among the reference file's
`kept_as_target` `ADJCHANGE` entries. What WAS at risk of contamination is
the raw `is_bgp_event` column below, wherever it's consumed directly
(e.g. `build_device_coverage.py`'s BGP-event counts, or any ad-hoc
validation against "genuine BGP log events") — that logic is now
facility-aware, not mnemonic-only, whenever `trap_msg` is present in the
source pull.

Output: data/processed/cleaned_log_data.parquet
"""
import glob
import os
import pandas as pd

# 2026-11-10: raw/processed moved to data_pipeline/raw, data_pipeline/
# processed (self-contained pipeline).
PIPELINE_DIR = os.path.dirname(__file__)
RAW_DIR = os.path.join(PIPELINE_DIR, "raw", "logs")
OUT_PATH = os.path.join(PIPELINE_DIR, "processed", "cleaned_log_data.parquet")

# Verified BGP-specific mnemonics (see docstring). Everything else in the
# pull is a mix of other protocols/subsystems (OSPF, PIM, AAA, hardware, ...).
BGP_MNEMONICS = {"ADJCHANGE", "NOTIFICATION"}

# ADJCHANGE specifically is shared with OSPF/CLNS under other facilities —
# only these facilities are genuine BGP for that mnemonic. NOTIFICATION is
# unambiguously BGP-only regardless of facility (verified: 100% facility=BGP
# in the fleet-wide message inventory), so it isn't restricted here.
BGP_FACILITIES = {"BGP", "BGP_SESSION"}


def parse_trap_msg(trap_msg: pd.Series) -> pd.DataFrame:
    """Splits Cisco's `FACILITY-SEVERITY-MNEMONIC` string (e.g.
    `BGP-5-ADJCHANGE`) into `facility`/`severity` columns. Returns NaN for
    both wherever `trap_msg` itself is missing/malformed — older pulls
    (before this field was added to the Splunk query) won't have it at
    all, and that's handled gracefully, not treated as an error."""
    parts = trap_msg.str.split("-", n=2, expand=True)
    if parts.shape[1] < 3:
        return pd.DataFrame({"facility": pd.NA, "severity": pd.NA}, index=trap_msg.index)
    facility = parts[0]
    severity = pd.to_numeric(parts[1], errors="coerce")
    return pd.DataFrame({"facility": facility, "severity": severity})


def load_raw() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(RAW_DIR, "splunk_logs_utc_*.csv")))
    if not files:
        raise FileNotFoundError(f"No splunk_logs_utc_*.csv files found in {RAW_DIR}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["__source_file"] = os.path.basename(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize message_text whitespace HERE, not left to whatever SPL
    # vintage produced the raw CSV. Found 2026-10-05: the SPL's trim()
    # fix (added 2026-09-30, see docs/spl_to_get_log_data.txt) means any
    # pull from that date forward already arrives pre-trimmed, but a pull
    # from before it (June-August in the pull on disk when this was found)
    # still carries a trailing space on roughly half of UPDOWN's rows (and
    # several other mnemonics) -- a real, confirmed split: 51-53% trailing-
    # whitespace rate for June-August, 0% for September, same mnemonic,
    # same underlying event. Matching the reference CSV's OWN message_text
    # (independently stripped the same way, 2026-10-05) against whichever
    # of these two forms happened to produce a given row corrupts the
    # exact-match join for every pull EXCEPT whichever vintage the
    # reference CSV happened to be captured against -- not a one-time
    # fix, a standing inconsistency across pull vintages that this
    # strip() closes at the source, for every month, past and future.
    df["message_text"] = df["message_text"].astype(str).str.strip()

    df["time_utc"] = pd.to_datetime(df["_time"]).dt.tz_convert("UTC")
    df = df.drop(columns=["_time"])
    df = df.rename(columns={"orig_host": "sys.name"})

    has_trap_msg = "trap_msg" in df.columns
    if has_trap_msg:
        df[["facility", "severity"]] = parse_trap_msg(df["trap_msg"])
    else:
        # Pre-trap_msg pull (older data/raw/logs/splunk_logs_utc_*.csv
        # exports) — no facility info available. Falls back to the old
        # mnemonic-only is_bgp_event below, same behavior as before this
        # correction, with a warning so it's not silently under-strict.
        print("NOTE: 'trap_msg' column not present in this pull — falling back to "
              "mnemonic-only is_bgp_event (cannot exclude OSPF/CLNS ADJCHANGE events). "
              "Re-pull with trap_msg included to get the facility-aware version.")
        df["facility"] = pd.NA
        df["severity"] = pd.NA

    mnemonic_match = df["mnemonic"].isin(BGP_MNEMONICS)
    # ADJCHANGE needs a facility check to exclude OSPF/CLNS (see docstring);
    # NOTIFICATION doesn't (verified BGP-only regardless of facility). A
    # missing facility (no trap_msg) is treated as "unknown, not excluded"
    # — same NaN-is-not-False discipline used elsewhere in this project,
    # so this never gets stricter than the old behavior when facility data
    # simply isn't available yet.
    needs_facility_check = df["mnemonic"] == "ADJCHANGE"
    facility_ok = df["facility"].isin(BGP_FACILITIES) | df["facility"].isna()
    df["is_bgp_event"] = mnemonic_match & (~needs_facility_check | facility_ok)

    dupes = df.duplicated(subset=["sys.name", "time_utc", "mnemonic", "message_text"]).sum()
    if dupes:
        print(f"NOTE: {dupes} duplicate (device, time, mnemonic, message) rows across source files — keeping first")
        df = df.drop_duplicates(subset=["sys.name", "time_utc", "mnemonic", "message_text"], keep="first")

    df = df.sort_values(["sys.name", "time_utc"]).reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame) -> None:
    n = len(df)
    print(f"Rows: {n}  Devices: {df['sys.name'].nunique()}  Distinct mnemonics: {df['mnemonic'].nunique()}")
    print(f"UTC time range: {df['time_utc'].min()} -> {df['time_utc'].max()}")
    bgp = df[df["is_bgp_event"]]
    print(f"Genuine BGP events (ADJCHANGE/NOTIFICATION, facility-checked): {len(bgp)} rows, "
          f"{bgp['sys.name'].nunique()} devices, total log_count={bgp['log_count'].sum()}")
    if df["facility"].notna().any():
        adj = df[df["mnemonic"] == "ADJCHANGE"]
        excluded = adj[~adj["is_bgp_event"]]
        if len(excluded):
            print(f"  ADJCHANGE rows excluded as non-BGP facility (OSPF/CLNS): {len(excluded)} rows, "
                  f"facilities found: {sorted(excluded['facility'].dropna().unique())}")
    else:
        print("  (no trap_msg/facility data in this pull — is_bgp_event is mnemonic-only, unverified for ADJCHANGE)")


if __name__ == "__main__":
    raw = load_raw()
    cleaned = clean(raw)
    summarize(cleaned)
    cleaned.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
