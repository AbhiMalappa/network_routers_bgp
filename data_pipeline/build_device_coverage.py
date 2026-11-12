"""
BGP flap prediction — device coverage audit across data sources.

Lists every distinct device seen in any of Dynatrace (DQL), Splunk, or
ServiceNow (SNOW), with a flag and row/event count per source, so coverage
gaps are visible at a glance and can guide where to expand data pulls.
Deliberately device-coverage-first, not BGP-specific — the goal is finding
devices we could pull more data for at all, regardless of whether we've
seen BGP activity from them yet.

Join key: normalized device name, not raw hostname (2026-08-20). Checked
first whether "everything before the first dot" was a safe normalization
(same physical device, different DNS suffix — e.g. Dynatrace's
`ecandcr4.ntwk.kp.org` vs Splunk's `ecandcr4.nndc.kp.org`) — it is NOT
safe in general: Splunk's directory also contains raw IP addresses
(`10.147.53.21` -> first-dot-split gives `"10"`, colliding dozens of
unrelated devices) and interface-qualified hostnames
(`lo0.gse-asr1006-b.ghc.org`, `lo0.wwarfmr1.ghc.org` -> both collapse to
`"lo0"`, merging distinct physical devices). Used instead: strip a KNOWN
domain suffix (see KNOWN_SUFFIXES below, built from suffixes recurring
>=2 times across all sources — one-off multi-label strings are left
alone rather than risk misparsing them as a domain) from the END of the
string. This correctly matches `ecandcr4.ntwk.kp.org` <-> `ecandcr4
.nndc.kp.org` while leaving `lo0.gse-asr1006-b.ghc.org` as
`lo0.gse-asr1006-b` (suffix-stripped but still unique per device) and
leaving bare IP addresses untouched (they don't end in any known
suffix). Verified zero new collisions introduced within any single
source's own device list after this normalization.

Dynatrace directory now carries a `dt.dns_names` column (2026-08-20
update) in addition to `entity.name` — up to 3 comma-separated DNS
aliases per entity (e.g. entity.name `ewaqdcr4-4d23.ntwk.kp.org` has
dns_names `ewaqdcr4.ntwk.kp.org`). `load_dynatrace_directory()` matches
in two tiers, priority order (2026-08-20, simplified from an earlier
version that pooled every entity's dns_names into one global namespace
and had to untangle cross-entity ambiguity after the fact):
  1. Primary: each row's own `entity.name`, normalized, used directly.
  2. Fallback, only for rows whose entity.name doesn't match ANY other
     source: try each `dt.dns_names` alias, normalized, and use it INSTEAD
     of entity.name if exactly one alias matches something in another
     source (Splunk directory/pulled, SNOW, or the pulled fleet).
This is simpler than pooling every alias globally, and it happens to
sidestep the HA/redundant-pair overlap problem that motivated the
earlier approach entirely: `hcabmor1.ntwk.kp.org` and
`hcabmor2.ntwk.kp.org` (a real pair whose dns_names each cross-reference
the other) both match fine on their OWN entity.name against the fleet,
so neither ever reaches the dns_names fallback — the cross-referenced
alias is simply never consulted. Checked empirically against the
2026-08-20 pull: 964 of 2,140 rows have no direct entity.name match
elsewhere; of those, 20 resolve cleanly via exactly one dns_names alias
match (including ewaqdcr4-4d23 -> ewaqdcr4), and exactly 1 is genuinely
ambiguous (`ecobdcr4-AW714`'s dns_names match two DIFFERENT external
devices, `ecobdcr4` and `ecobdcr4-az727`) — left on its own entity.name
rather than guessed, and reported by the loader when it happens.

`device` in the output is the normalized key; `*_raw_names` columns list
every distinct raw hostname per source that normalized to that key
(semicolon-separated), so you can see exactly what matched — this is a
diagnostic join for finding coverage gaps, not a live join key used
anywhere in the actual DQL+Splunk+SNOW feature-building pipeline (a
domain difference occasionally reflects a genuinely different logical
entity, e.g. a management interface vs. the main device, so matches here
should be spot-checked before being treated as confirmed same-device).

Notes baked in from prior investigation:
- Dynatrace AND Splunk each have TWO distinct device lists — a broad
  "directory" (every device the source knows about at all, e.g. from an
  admin-exported catalog) and the narrower set we've actually pulled
  detailed data for.
  - `in_dynatrace_device_directory`: every device Dynatrace reports on,
    per data/raw/dql/dql_device_metadata.csv, expanded via entity.name +
    dt.dns_names — see above.
  - `in_dynatrace_pulled` / `dynatrace_row_count`: devices actually
    present in our processed DQL data
    (data/processed/cleaned_device_level_data.parquet's
    device_name_resolved column) — the current modeled fleet.
  - `in_splunk_device_directory`: every device that logs into Splunk AT
    ALL, per an admin-exported catalog (data/raw/logs/
    all_splunk_logging_devices.csv, 3,823 raw names as of the corrected
    2026-08-20 pull).
  - `in_splunk_logs_pulled` / `splunk_log_row_count`: devices actually
    present in our existing pulled log data
    (data/processed/cleaned_log_data.parquet, 1,016 raw names).
  - `in_splunk_confirmed_bgp_event` / `splunk_confirmed_bgp_event_count`:
    the narrowest cut, genuine ADJCHANGE/NOTIFICATION events only — kept
    as a secondary detail column, not part of `source_count`, per the
    "BGP info is secondary, device coverage is primary" framing this
    file is for.
- SNOW's `cmdb_ci` field mixes real device hostnames with non-device
  placeholder values (carrier/circuit categories like "Circuits Data
  AT&T", "Network Operations Management", "Not found"). Filtered to
  values containing ".kp.org" to keep only genuine device references.

`source_count` counts 3 genuinely distinct sources at their broadest cut
(Dynatrace directory, Splunk directory, SNOW) — the "pulled" and
"confirmed BGP event" columns are detail on their respective source, not
counted separately.

Output: data/processed/device_coverage_by_source.csv
"""
import glob
import os
import re

import pandas as pd

# 2026-11-10: raw/processed moved to data_pipeline/raw, data_pipeline/
# processed (self-contained pipeline, no project-level data/ dependency).
PIPELINE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(PIPELINE_DIR, "processed"), exist_ok=True)
DQL_PATH = os.path.join(PIPELINE_DIR, "processed", "cleaned_device_level_data.parquet")
# 2026-09-02: all_dynatrace_reporting_device.csv was retired — dql_device_
# metadata.csv (built alongside every Dynatrace pull now) is the current
# equivalent, and already has the exact entity.name/dt.dns_names schema
# load_dynatrace_directory() below expects, so no logic change needed there.
DYNATRACE_DIRECTORY_PATH = os.path.join(PIPELINE_DIR, "raw", "dql", "dql_device_metadata.csv")
LOG_PATH = os.path.join(PIPELINE_DIR, "processed", "cleaned_log_data.parquet")
SPLUNK_DIRECTORY_PATH = os.path.join(PIPELINE_DIR, "raw", "logs", "all_splunk_logging_devices.csv")
SNOW_RAW_DIR = os.path.join(PIPELINE_DIR, "raw", "snow")
OUT_PATH = os.path.join(PIPELINE_DIR, "processed", "device_coverage_by_source.csv")

# Recurring (>=2 occurrences) domain suffixes confirmed across Dynatrace
# directory (entity.name + dt.dns_names), Splunk directory, the pulled
# fleet, and SNOW cmdb_ci as of the 2026-08-20 audit. Longest first, so
# e.g. "ntwk.kp.org" strips before the shorter "kp.org" would partially
# match; "dor.ca.kp.org" before "ca.kp.org". Deliberately NOT built from
# single-occurrence suffixes (e.g. "gse-asr1006-b.ghc.org") — those are
# almost certainly a naive mis-split of an interface-qualified hostname,
# not a real shared domain, and including them risks exactly the
# collision this normalization is meant to avoid.
KNOWN_SUFFIXES = sorted([
    "ntwk.kp.org", "pldc.kp.org", "nndc.kp.org", "wcdc.kp.org", "hi.kp.org",
    "ca.kp.org", "test.kp.org", "ssdc.kp.org", "co.kp.org", "ghc.org", "kp.org",
    "crdc.kp.org", "md.kp.org", "ga.kp.org", "or.kp.org", "cin.kp.org",
    "dor.ca.kp.org", "crl.ca.kp.org", "nda.ca.kp.org", "slr.ca.kp.org",
    "l3dc.kp.org", "l3sc.kp.org",
], key=len, reverse=True)

_IP_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")


def normalize_device(name: str) -> str:
    """Lowercased and, when the name ends in a known domain suffix,
    stripped of that suffix. Lowercasing the full result (not just the
    suffix comparison) matters: Dynatrace's dns_names mixes case
    inconsistently even for the same device (e.g. "ecobdcr1-AW713.kp.org"
    vs "ecobdcr1-az715.kp.org" in the same dns_names list) — comparing
    case-sensitively would silently fail to match otherwise-identical
    tokens."""
    if not isinstance(name, str):
        return name
    name = name.lower()
    if _IP_RE.match(name):
        return name
    for suffix in KNOWN_SUFFIXES:
        if name.endswith("." + suffix):
            return name[: -(len(suffix) + 1)]
    return name


def load_snow_device_counts() -> pd.Series:
    # 2026-11-11: SNOW export filename convention moved from incident*.csv
    # to snow_bgp_pst_*.csv some time ago; this glob was never updated to
    # match, so this function had been silently unusable. Same cmdb_ci
    # column in both formats, so no other change needed.
    files = sorted(glob.glob(os.path.join(SNOW_RAW_DIR, "snow_bgp_pst_*.csv")))
    if not files:
        raise FileNotFoundError(f"No snow_bgp_pst_*.csv files found in {SNOW_RAW_DIR}")
    dfs = [pd.read_csv(f, encoding="cp1252", on_bad_lines="warn", low_memory=False, usecols=["cmdb_ci"])
           for f in files]
    snow = pd.concat(dfs, ignore_index=True)
    devices = snow[snow["cmdb_ci"].notna() & snow["cmdb_ci"].str.contains(r"\.kp\.org", case=False, na=False)]
    return devices["cmdb_ci"].value_counts()


def load_device_directory_raw(path: str, col: str) -> pd.Series:
    if not os.path.exists(path):
        return pd.Series(dtype=object)
    directory = pd.read_csv(path)
    return directory[col].dropna()


def load_dynatrace_directory(external_keys: set) -> dict:
    """Returns {normalized_key: entity.name}. See module docstring for the
    two-tier (entity.name first, dns_names fallback) matching strategy.
    `external_keys` is the union of normalized keys already seen in the
    other 3 sources (Splunk directory/pulled, SNOW, pulled fleet) — needed
    up front because tier 2 only fires when tier 1 finds no match there."""
    if not os.path.exists(DYNATRACE_DIRECTORY_PATH):
        return {}
    df = pd.read_csv(DYNATRACE_DIRECTORY_PATH)

    resolved: dict[str, str] = {}
    fallback_resolved = 0
    fallback_ambiguous = []
    for _, row in df.iterrows():
        entity = row["entity.name"]
        own_key = normalize_device(entity)

        if own_key in external_keys:
            resolved[own_key] = entity
            continue

        dns_names = row.get("dt.dns_names")
        alias_matches = set()
        if pd.notna(dns_names):
            alias_keys = {normalize_device(t.strip()) for t in dns_names.split(",") if t.strip()}
            alias_matches = alias_keys & external_keys

        if len(alias_matches) == 1:
            resolved[next(iter(alias_matches))] = entity
            fallback_resolved += 1
        else:
            # No external match at all, or the alias fallback was itself
            # ambiguous (2+ different alias matches, e.g. ecobdcr4-AW714's
            # dns_names matching both ecobdcr4 and ecobdcr4-az727) — keep
            # the entity under its own name rather than guess.
            resolved[own_key] = entity
            if len(alias_matches) > 1:
                fallback_ambiguous.append((entity, alias_matches))

    if fallback_resolved:
        print(f"NOTE: {fallback_resolved} Dynatrace entities matched via a dns_names alias "
              f"fallback (their own entity.name had no match elsewhere).")
    if fallback_ambiguous:
        print(f"NOTE: {len(fallback_ambiguous)} Dynatrace entities had 2+ different dns_names "
              f"aliases each matching a DIFFERENT external device — left under their own "
              f"entity.name rather than guessed. Example: {fallback_ambiguous[0]}")

    return resolved


def raw_names_by_key(raw_series: pd.Series) -> dict:
    """Groups raw hostnames by their normalized key, returning
    {normalized_key: 'raw1;raw2;...'} — used both to build the boolean
    membership sets and to show what actually matched in the output."""
    keys = raw_series.map(normalize_device)
    grouped = raw_series.groupby(keys).apply(lambda s: ";".join(sorted(set(s))))
    return grouped.to_dict()


def build() -> pd.DataFrame:
    # 2026-09-02: cleaned_device_level_data.parquet no longer has sys.name —
    # device_name_resolved is the equivalent (NaN for the small number of
    # devices whose identity couldn't be disambiguated; dropped here since
    # there's nothing to match on for those).
    dql = pd.read_parquet(DQL_PATH, columns=["device_name_resolved"])["device_name_resolved"].dropna()

    logs = pd.read_parquet(LOG_PATH, columns=["sys.name", "is_bgp_event", "log_count"])
    splunk_dir_raw = load_device_directory_raw(SPLUNK_DIRECTORY_PATH, "orig_host")

    snow_raw_counts = load_snow_device_counts()

    dql_norm = dql.map(normalize_device)
    dql_row_count = dql_norm.value_counts()

    log_norm = logs["sys.name"].map(normalize_device)
    log_row_count = log_norm.value_counts()
    bgp_norm = logs.loc[logs["is_bgp_event"], "sys.name"].map(normalize_device)
    bgp_event_sum = logs.loc[logs["is_bgp_event"]].assign(_key=bgp_norm).groupby("_key")["log_count"].sum()
    splunk_dir_names = raw_names_by_key(splunk_dir_raw)

    snow_raw = pd.Series(snow_raw_counts.index)
    snow_norm_keys = snow_raw.map(normalize_device)
    snow_count_by_key = pd.Series(snow_raw_counts.values, index=snow_norm_keys).groupby(level=0).sum()
    snow_names = raw_names_by_key(pd.Series(snow_raw_counts.index))

    external_keys = set(dql_row_count.index) | set(splunk_dir_names) | set(log_row_count.index) | set(snow_names)
    dyn_dir_names = load_dynatrace_directory(external_keys)

    all_keys = sorted(
        set(dyn_dir_names) | set(dql_row_count.index)
        | set(splunk_dir_names) | set(log_row_count.index)
        | set(snow_names)
    )

    rows = []
    for k in all_keys:
        rows.append({
            "device": k,
            "in_dynatrace_device_directory": k in dyn_dir_names,
            "dynatrace_directory_raw_names": dyn_dir_names.get(k, ""),
            "in_dynatrace_pulled": k in dql_row_count.index,
            "dynatrace_row_count": int(dql_row_count.get(k, 0)),
            "in_splunk_device_directory": k in splunk_dir_names,
            "splunk_directory_raw_names": splunk_dir_names.get(k, ""),
            "in_splunk_logs_pulled": k in log_row_count.index,
            "splunk_log_row_count": int(log_row_count.get(k, 0)),
            "in_splunk_confirmed_bgp_event": k in bgp_event_sum.index,
            "splunk_confirmed_bgp_event_count": int(bgp_event_sum.get(k, 0)),
            "in_snow": k in snow_names,
            "snow_raw_names": snow_names.get(k, ""),
            "snow_incident_count": int(snow_count_by_key.get(k, 0)),
        })

    out = pd.DataFrame(rows)
    out["source_count"] = out[
        ["in_dynatrace_device_directory", "in_splunk_device_directory", "in_snow"]
    ].sum(axis=1)
    out = out.sort_values(["source_count", "device"], ascending=[False, True]).reset_index(drop=True)
    return out


def summarize(out: pd.DataFrame) -> None:
    print(f"Total distinct normalized devices across all sources: {len(out)}")
    print()
    print("Source coverage:")
    print(f"  Dynatrace device directory (any):     {out['in_dynatrace_device_directory'].sum()}")
    print(f"  Dynatrace actually pulled (fleet):    {out['in_dynatrace_pulled'].sum()}")
    print(f"  Splunk device directory (any):        {out['in_splunk_device_directory'].sum()}")
    print(f"  Splunk logs actually pulled:          {out['in_splunk_logs_pulled'].sum()}")
    print(f"  Splunk logs, confirmed BGP event:     {out['in_splunk_confirmed_bgp_event'].sum()}")
    print(f"  SNOW (mapped incidents):              {out['in_snow'].sum()}")
    print()
    print("Devices by how many of the 3 primary sources see them (Dynatrace directory/Splunk directory/SNOW):")
    print(out["source_count"].value_counts().sort_index(ascending=False).to_string())
    print()
    print("In Dynatrace directory AND Splunk directory, but DQL not yet pulled (expand pull, no new source needed):",
          ((out["in_dynatrace_device_directory"]) & (~out["in_dynatrace_pulled"])).sum())
    print("In Dynatrace fleet (pulled) but NOT in Splunk directory at all (would need a non-Splunk source):",
          ((out["in_dynatrace_pulled"]) & (~out["in_splunk_device_directory"])).sum())
    print("In Dynatrace fleet AND in Splunk directory, but log data not yet pulled (expand pull, no new source needed):",
          ((out["in_dynatrace_pulled"]) & (out["in_splunk_device_directory"]) & (~out["in_splunk_logs_pulled"])).sum())
    print("In Splunk directory but NOT in Dynatrace directory (fleet-expansion candidates):",
          ((~out["in_dynatrace_device_directory"]) & (out["in_splunk_device_directory"])).sum())


if __name__ == "__main__":
    result = build()
    summarize(result)
    result.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH}")
