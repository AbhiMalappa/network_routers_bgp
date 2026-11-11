"""
BGP flap prediction -- data pipeline, stage 4: combine Dynatrace + SNOW.

Left-joins classify_snow_with_haiku_batch.py's per-ticket output onto the
Dynatrace device-level table (cleaned_device_level_data.parquet), by device
+ 4h bucket. Adds four columns per (device, time_utc) row:

    haiku_root_cause     -- plain string (e.g. "Carrier / Circuit Issue"),
                            "No Ticket" where nothing matched this bucket.
    haiku_predictability -- plain string (e.g. "Predictable", "Not
                            Predictable", "Known in Advance", "Indeterminate",
                            "N/A (no real issue)"), "No Ticket" where nothing
                            matched. Deliberately NOT collapsed to binary --
                            that's a model-time decision (which labels count
                            as "predictable" for a given model), not a
                            data-building one. Kept as the full category so
                            any future model/filter can pick its own cutoff
                            without re-deriving anything.
    haiku_subcategory    -- plain string, "" where root_cause wasn't Carrier
                            or Hardware (no subcategory exists), "No Ticket"
                            where nothing matched this bucket at all.
    snow_ticket_count    -- int, how many SNOW tickets matched this exact
                            (device, bucket). 0 where none did.

No one-hot encoding here -- deferred to modeling time, if and when needed
(per-root-cause sub-models were mentioned as a possible future use, which
argues for keeping the plain category rather than pre-committing to a
one-hot scheme neither of us has designed yet).

Multiple tickets in one (device, bucket): rare but real. Policy: if ANY
matched ticket has haiku_predictability == "Predictable", use that one
(earliest by opened_at_utc if more than one is). Otherwise use the single
earliest ticket by opened_at_utc ("first ticket"). snow_ticket_count always
reflects the true match count regardless of which one was chosen, so this
collapsing is auditable rather than silently hidden.

Coverage reality check (same caveat as everywhere else in this project SNOW
touches): only a minority of tickets resolve to a fleet device at all
(cmdb_ci -> normalize_device() key), so the overwhelming majority of rows
in the output will be "No Ticket" / 0. This is not a comprehensive SNOW
enrichment of every bucket -- it's exactly the subset that got escalated to
a ticket and resolved to a known device.

Input: data/processed/cleaned_device_level_data.parquet,
       data/processed/snow_bgp_pst_*_with_haiku.csv (most recently modified)
Output: data/processed/dynatrace_snow_combined.parquet
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_device_coverage import normalize_device  # noqa: E402

# 2026-11-10: raw/processed moved to data_pipeline/raw, data_pipeline/
# processed (self-contained pipeline).
PIPELINE_DIR = os.path.dirname(__file__)
DYNATRACE_PATH = os.path.join(PIPELINE_DIR, "processed", "cleaned_device_level_data.parquet")
OUT_PATH = os.path.join(PIPELINE_DIR, "processed", "dynatrace_snow_combined.parquet")

ENTITY_COL = "dt.entity.network:device"
IDENTITY_COL = "device_name_resolved"
BUCKET_HOURS = 4

NO_TICKET = "No Ticket"


def _resolve_snow_csv() -> str:
    pattern = os.path.join(PIPELINE_DIR, "processed", "snow_bgp_pst_*_with_haiku.csv")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            f"No snow_bgp_pst_*_with_haiku.csv found via {pattern} -- run "
            f"classify_snow_with_haiku_batch.py (submit/status/collect) first."
        )
    chosen = files[-1]
    if len(files) > 1:
        print(f"NOTE: {len(files)} snow_bgp_pst_*_with_haiku.csv files found; "
              f"using the most recently modified: {os.path.basename(chosen)}")
    return chosen


def load_snow_per_bucket() -> pd.DataFrame:
    """One row per (normalized device key, 4h bucket) that had >=1 matched
    SNOW ticket -- collisions already resolved per the module docstring's
    policy. Rows are NOT yet filtered to a fleet device here (that happens
    naturally at the merge in main() -- a key with no matching device just
    never finds a home in the left join)."""
    snow_path = _resolve_snow_csv()
    snow = pd.read_csv(snow_path, low_memory=False)
    if "opened_at_utc" not in snow.columns:
        raise ValueError(
            f"{snow_path} has no opened_at_utc column -- re-run "
            f"classify_snow_with_haiku_batch.py's collect to regenerate it "
            f"with the current script (2026-11-09+), or re-apply the one-off "
            f"UTC patch to this file."
        )

    snow = snow[snow["haiku_root_cause"] != "EXCLUDED_NON_BGP"].copy()
    snow["_key"] = snow["cmdb_ci"].map(normalize_device)
    snow["opened_at_utc"] = pd.to_datetime(snow["opened_at_utc"], utc=True)
    snow["time_utc"] = snow["opened_at_utc"].dt.floor(f"{BUCKET_HOURS}h")
    snow = snow.sort_values("opened_at_utc")

    def pick(group: pd.DataFrame) -> pd.Series:
        predictable = group[group["haiku_predictability"] == "Predictable"]
        chosen = predictable.iloc[0] if len(predictable) else group.iloc[0]
        return pd.Series({
            "haiku_root_cause": chosen["haiku_root_cause"],
            "haiku_predictability": chosen["haiku_predictability"],
            "haiku_subcategory": chosen["haiku_subcategory"],
            "snow_ticket_count": len(group),
        })

    per_bucket = snow.groupby(["_key", "time_utc"]).apply(pick, include_groups=False).reset_index()
    n_collisions = (per_bucket["snow_ticket_count"] > 1).sum()
    print(f"SNOW tickets loaded: {len(snow)} (non-excluded). Distinct (cmdb_ci key, bucket) groups "
          f"before the fleet-device join: {len(per_bucket)} ({n_collisions} had >1 ticket -- collision "
          f"policy: prefer a 'Predictable' one, else earliest by opened_at_utc). Most of these keys will "
          f"NOT match a real device (garbage cmdb_ci values like 'Not found' get their own groups here too) "
          f"-- see the real post-join match count printed in main().")
    return per_bucket


def main():
    dt = pd.read_parquet(DYNATRACE_PATH)
    dt["_key"] = dt[IDENTITY_COL].map(normalize_device)

    snow_per_bucket = load_snow_per_bucket()

    out = dt.merge(snow_per_bucket, on=["_key", "time_utc"], how="left")
    out = out.drop(columns=["_key"])

    out["haiku_root_cause"] = out["haiku_root_cause"].fillna(NO_TICKET)
    out["haiku_predictability"] = out["haiku_predictability"].fillna(NO_TICKET)
    out["haiku_subcategory"] = out["haiku_subcategory"].fillna(NO_TICKET)
    out["snow_ticket_count"] = out["snow_ticket_count"].fillna(0).astype(int)

    n_matched_rows = (out["snow_ticket_count"] > 0).sum()
    print(f"\nDynatrace rows: {len(out)}")
    print(f"Rows with >=1 matched SNOW ticket: {n_matched_rows} ({100*n_matched_rows/len(out):.2f}%)")
    print("\nhaiku_root_cause breakdown (matched rows only):")
    print(out.loc[out["snow_ticket_count"] > 0, "haiku_root_cause"].value_counts().to_string())
    print("\nhaiku_predictability breakdown (matched rows only):")
    print(out.loc[out["snow_ticket_count"] > 0, "haiku_predictability"].value_counts().to_string())

    out.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(out)} rows, {len(out.columns)} cols)")


if __name__ == "__main__":
    main()
