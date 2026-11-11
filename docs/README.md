# Network Device & BGP Telemetry — Data Pipeline

A self-contained pipeline that turns raw Dynatrace, Splunk, and ServiceNow
(SNOW) exports into one wide, model-agnostic tabular dataset: one row per
network device per 4-hour time bucket, every feature this project has
established, and **no target column**. The data covers device health
(CPU, memory, uptime, reboots), interface/traffic/error counters, optical
and hardware sensor readings (power, bias current, temperature, fan,
power supply), BGP peer/session state, Splunk syslog event counts, and
SNOW ticket enrichment — general-purpose network device telemetry, not
built around one specific question. It's a reasonable starting point for
BGP flap prediction, hardware failure prediction, or any other
network-reliability question this data can speak to. What a model is
trying to predict — and how — is a decision for a separate, model-specific
step; this pipeline's job ends at "every raw and derived feature a model
could want."

Everything this pipeline needs to run, and everything it produces, lives
inside this folder (`data_pipeline/`). No path here reaches outside it,
except to read `docs/log_message_text_to_feature_reference.csv` — a
curated reference file maintained at the project root, not pipeline data.

---

## 1. What's in this folder

```
data_pipeline/
├── raw/                                  raw exports, dropped in as-is
│   ├── dql/                              Dynatrace Grail (DQL) pulls
│   │   ├── dql_device_metadata.csv
│   │   └── parquet/dql_15query_*/        one folder per monthly pull,
│   │                                     15 query result files each
│   ├── logs/                             Splunk exports
│   │   ├── splunk_logs_utc_*.csv
│   │   ├── network_device_metadata.csv
│   │   └── all_splunk_logging_devices.csv
│   ├── snow/                             ServiceNow ticket exports
│   │   └── snow_bgp_pst_*.csv
│   ├── loging_devcies.csv                log coverage allow-list
│   ├── all_message_text.csv              full Splunk mnemonic/message inventory
│   └── all_message_text_old.csv
│
├── processed/                            every intermediate + the final table
│   ├── cleaned_device_level_data.parquet
│   ├── cleaned_log_data.parquet
│   ├── log_features_wide.parquet / log_device_coverage.csv / log_device_platform.csv
│   ├── {stem}_with_haiku.csv / {stem}_haiku_batch_results.json / {stem}_last_batch_id.txt
│   ├── dynatrace_snow_combined.parquet
│   ├── dynatrace_snow_splunk_combined.parquet
│   ├── bgp_combined_data_table.parquet   <- the final deliverable
│   └── (QA script outputs — see Section 5)
│
├── build_and_clean_dynatrace_device_data.py
├── clean_log_data.py
├── build_log_features.py
├── classify_snow_with_haiku_batch.py
├── combine_dynatrace_snow.py
├── combine_with_splunk.py
├── build_secondary_features.py
├── build_device_coverage.py              shared utility (normalize_device) + its own diagnostic
├── audit_dql_source_coverage.py          QA
├── audit_metric_value_sanity.py          QA
└── audit_log_message_text_coverage.py    QA
```

---

## 2. Pipeline stages

Six stages, run in order. Each reads the previous stage's output and writes
its own into `processed/`.

### Stage 1 — Dynatrace: `build_and_clean_dynatrace_device_data.py`
Loads all 14 logical DQL query tables (one is physically two files —
`query_2a_i_traffic` + `query_2a_ii_errors_discards` — merged here) across
every monthly pull folder under `raw/dql/parquet/`, resolves each device's
identity against `dql_device_metadata.csv`, and applies two data-quality
fixes: nulls stuck/frozen sensor readings (CPU, optical power, module/chassis
temp — the same value repeating for an implausibly long run) and floors
physically-impossible optical power readings. Also computes three
permanent per-row quality flags (`flat_cpu_device`, `chronic_flap_window`,
`restart_co_occurring_flap`) — these stay in the output as ordinary feature
columns; nothing in this pipeline uses them to drop rows.

**Output:** `processed/cleaned_device_level_data.parquet` — one row per
device per 4h bucket, ~90 Dynatrace-side columns, 14 `has_X_data` coverage
flags (one per source query category).

### Stage 2 — Splunk: `clean_log_data.py` → `build_log_features.py`
`clean_log_data.py` parses raw Splunk exports (`raw/logs/splunk_logs_utc_*.csv`)
into a cleaned long-format table (`processed/cleaned_log_data.parquet`):
one row per device, time bucket, mnemonic, and message text.

`build_log_features.py` groups and pivots that into a wide table — 59
feature columns, one per category defined in
`docs/log_message_text_to_feature_reference.csv` (the single source of
truth for which raw `(mnemonic, message_text)` combinations are kept,
dropped, and which column each feeds). Also produces device coverage
(`log_device_coverage.csv` — does this device have ANY log visibility at
all) and device hardware platform/tier (`log_device_platform.csv`, sourced
from `network_device_metadata.csv`, independent of whether a device logged
anything).

**Output:** `processed/log_features_wide.parquet`,
`processed/log_device_coverage.csv`, `processed/log_device_platform.csv`.

### Stage 3 — SNOW: `classify_snow_with_haiku_batch.py`
Classifies every BGP ticket in the most recently modified
`raw/snow/snow_bgp_pst_*.csv` using one Claude Haiku call per ticket (via
the Message Batches API), returning four always-present fields —
`root_cause`, `predictability`, `mechanism`, `evidence` — plus a fifth,
`subcategory`, populated only when `root_cause` is Carrier/Circuit or
Hardware/Device (a finer sub-issue bucket, e.g. "fiber cut" vs "defective
equipment"). Also converts the raw `opened_at` timestamp (local
Pacific time, no embedded UTC offset) to `opened_at_utc`, so it can be
joined against Dynatrace/Splunk data by time.

Three subcommands, each independent so a long-running batch survives the
script exiting in between: `submit` (build + submit the batch), `status`
(poll, non-blocking), `collect` (once finished, write the enriched CSV +
a raw JSON dump). **Costs real API spend — always ask before running
`submit`.**

**Output:** `processed/{input_stem}_with_haiku.csv` — the original ticket
columns plus `haiku_root_cause`, `haiku_predictability`, `haiku_mechanism`,
`haiku_evidence`, `haiku_subcategory`, `opened_at_utc`.

### Stage 4 — Combine Dynatrace + SNOW: `combine_dynatrace_snow.py`
Left-joins the SNOW ticket table onto the Dynatrace device-bucket grid by
device (`normalize_device(cmdb_ci)` matched against `device_name_resolved`)
and 4h bucket (`opened_at_utc` floored to the bucket grid). Adds four
columns per row:

| Column | Type | Notes |
|---|---|---|
| `haiku_root_cause` | string | `"No Ticket"` where nothing matched this bucket |
| `haiku_predictability` | string | Full category (`Predictable`/`Not Predictable`/`Known in Advance`/`Indeterminate`/`N/A (no real issue)`), **not** collapsed to binary — see Section 4 |
| `haiku_subcategory` | string | `""` if root_cause isn't Carrier/Hardware, `"No Ticket"` if nothing matched |
| `snow_ticket_count` | int | How many tickets matched this exact (device, bucket); 0 if none |

When more than one ticket lands in the same (device, bucket): prefer one
classified `Predictable` if any exists (earliest by `opened_at_utc` if
more than one is), otherwise take the single earliest ticket. The true
match count is always in `snow_ticket_count` regardless of which ticket's
classification was kept, so a collision is auditable, not silently hidden.

Most rows will have no match at all — only a minority of SNOW tickets
resolve to a known fleet device (`cmdb_ci` is often blank, a non-device
value, or a device outside the modeled fleet). This is expected: it's a
left join against the sparse subset of real-world incidents that got
escalated to a ticket, not a comprehensive SNOW enrichment of every bucket.

**Output:** `processed/dynatrace_snow_combined.parquet`.

### Stage 5 — Combine with Splunk: `combine_with_splunk.py`
Resolves each Dynatrace device to its Splunk-side key (try the resolved
device name first, fall back to a `dt.dns_names` alias only if exactly one
alias matches — never guess among multiple candidates), then left-joins:

- the 59-column Splunk log feature table (`log_features_wide.parquet`) — a
  device with log coverage but no activity in a given bucket gets `0`
  (confirmed no activity); a device with no log coverage at all, or no
  resolvable Splunk key, gets `NaN` (unknown) — these are not the same thing.
- device platform/tier, one-hot encoded (top-10-by-device-count +
  "Others" binning, computed over devices actually present in this table).
- `router_type`/`network_type` (already on the Dynatrace side from Stage 1's
  metadata join), also one-hot encoded.

Platform/tier/router_type/network_type are deliberately one-hot here —
unlike the SNOW columns, which stay plain strings (see Section 4).

**Output:** `processed/dynatrace_snow_splunk_combined.parquet`.

### Stage 6 — Secondary features: `build_secondary_features.py`
The single place every lag/trend/z-score/volatility feature in this
project gets computed, applied uniformly over the fully combined table —
Dynatrace, SNOW, and Splunk-derived columns all go through the same
rolling-window machinery rather than each source computing its own. Three
treatments, assigned per metric group:

1. **Full treatment** (trend `_delta_24h`, self-baseline z-score
   `_zscore_vs_own_history`, rolling volatility `_std_prior_24h`/`_7d`) —
   every Dynatrace raw level/spread metric (CPU, traffic, interface
   errors/discards, optical power, bias current, module/chassis
   temperature, BGP peer counts, etc.), plus `snow_ticket_count`.
2. **Prior-lag only** (`_prior_24h`/`_7d` SUM, or AVG for a ratio) — the 59
   Splunk log-derived base columns, `device_flap_count`,
   `num_peers_flapped`, `flap_concentration`, the 10 hardware abnormal
   flags, and their `any_hardware_abnormal` rollup. These are either
   already-rare binary flags (trend/z-score of a flag is a weak
   construction) or persistence-style counts where "how much has this
   fired recently" is the useful secondary signal, not a level or
   volatility reading.
3. **Everything else established in this project**: per-source coverage
   features (missing-bins count, lifetime-observed-bins count,
   hours-since-last-real-reading — one triplet per `has_X_data` source),
   post-reboot risk window (`recently_rebooted_24h`/`_48h`), interface
   churn imbalance (down − up, at each lag horizon), and traffic-normalized
   error/discard rate (+ `_prior_7d_avg`).

**Output:** `processed/bgp_combined_data_table.parquet` — the pipeline's
final deliverable.

---

## 3. Running the pipeline

```bash
python3 build_and_clean_dynatrace_device_data.py
python3 clean_log_data.py
python3 build_log_features.py
ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_batch.py submit
ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_batch.py status --batch-id msgbatch_...
ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_batch.py collect --batch-id msgbatch_...
python3 combine_dynatrace_snow.py
python3 combine_with_splunk.py
python3 build_secondary_features.py
```

Stage 1 and Stage 2 (clean + features) are independent of each other and
of Stage 3 — all three can run in parallel. Stage 4 needs Stage 1 + Stage
3's output; Stage 5 needs Stage 4 + Stage 2; Stage 6 needs Stage 5. The
SNOW stage is the only one that costs real money (Claude API spend) and
needs `ANTHROPIC_API_KEY` set — never run `submit` without confirming
first. If `classify_snow_with_haiku_batch.py` hasn't been run at all,
`combine_dynatrace_snow.py` still works — it prints a note and every row
gets `"No Ticket"` / `0`.

Run the relevant QA script after any new raw pull lands — see Section 5.

---

## 4. Design decisions worth knowing before extending this

- **No target, no row filtering, no train/test split.** Every row that
  exists at Stage 1 still exists at Stage 6. A model-specific step,
  separate from this pipeline, decides what the target looks like —
  binary flap-in-next-24h, time-to-next-hardware-failure for a survival
  model, a sequence for an LSTM, or something else entirely — and which
  rows/columns that specific model should use.
- **SNOW's `haiku_predictability` is a plain string, not a binary flag.**
  It has five real categories (`Predictable`, `Not Predictable`, `Known in
  Advance`, `Indeterminate`, `N/A (no real issue)`), plus `"No Ticket"` for
  unmatched buckets. Collapsing this to binary is a modeling decision (which
  labels count as "predictable" for a given use case) — keeping the full
  category means any future model or filter can choose its own cutoff
  without re-deriving anything.
- **`haiku_root_cause`/`haiku_subcategory` are also plain strings, not
  one-hot.** Same reasoning — deferred to modeling time, if and when
  needed (e.g. training a root-cause-specific sub-model is easier off the
  raw string than off a fixed set of dummy columns decided in advance).
- **Platform/tier/router_type/network_type, by contrast, ARE one-hot
  encoded here**, with top-10-by-device-count + "Others" binning for
  platform specifically. This is a deliberate exception, not an
  inconsistency: these are a small number of known, stable, low-
  cardinality categories where encoding is required to represent the
  concept "something that was not in the top 10," not a deferred choice.
- **`device_flap_count` (and its close relatives `num_peers_flapped`,
  `flap_concentration`) are ordinary feature columns here**, with
  `prior_24h`/`_7d` lag but no target-based exclusion — there's no target
  in this pipeline to leak into. Any model-specific step building a
  forward-looking target off this data (flap-based, hardware-failure-based,
  or otherwise) should consider whether a given current-state or
  near-term-history column is too close to whatever it's trying to
  predict, and exclude it (and its lags) from its own feature set if so —
  that's a label-leakage judgment call for that step to make, not this
  pipeline.
- **NaN means "unknown," not "confirmed zero," and that distinction is
  preserved everywhere it matters.** A device with no Splunk log coverage
  at all gets `NaN` on every log-derived column, not `0` — `0` is reserved
  for "this device has coverage and confirmed no activity." The same
  discipline applies to the per-source `has_X_data` coverage features and
  the platform/tier/router_type/network_type `*_known` flags. Every
  rolling/expanding window in Stage 6 uses a `min_periods` floor that
  correctly propagates this: a window spanning genuinely missing data
  produces `NaN`, not a silently-undercounted partial result.

---

## 5. QA scripts

Not part of the required build chain — run them to validate a stage's own
output, typically right after that stage (or after a new raw pull lands).

| Script | Checks | Writes |
|---|---|---|
| `audit_dql_source_coverage.py` | Per (pull, query), flags any day whose device count falls below `--threshold` (default 0.5) of that same query's own median across the pull — catches a query silently losing fleet-wide coverage for a stretch of days without flagging queries that are just inherently low-coverage (e.g. fan/temperature sensors) | `processed/dql_coverage_gaps.csv` |
| `audit_metric_value_sanity.py` | Three checks on Stage 1's output: range sanity (physically-impossible values), cross-column consistency (e.g. min ≤ max, `peers_with_data` ≤ `peer_count`), and stuck-value detection (same non-null, non-zero value for an implausibly long run) | `processed/metric_value_sanity_{range,consistency,stuck}*.csv` |
| `audit_log_message_text_coverage.py` | For every Splunk mnemonic that's already partially kept (at least one non-dropped row in the reference file), what fraction of its real log volume the exact `(mnemonic, message_text)` join in Stage 2 is silently dropping — catches a message-text wording variant for an already-known mnemonic, not just a brand-new mnemonic | `processed/log_message_text_coverage_gaps.csv` |
| `build_device_coverage.py` (run directly) | Per-device coverage across Dynatrace, Splunk, and SNOW — which devices are known to which source at all | `processed/device_coverage_by_source.csv` |

`build_device_coverage.py` also exports `normalize_device()` — lowercase +
strip a list of recurring domain suffixes, IPs left untouched — the shared
hostname-matching utility every cross-source join in this pipeline uses.

---

## 6. Known limitations

- **SNOW coverage is inherently sparse.** Only a minority of tickets
  resolve to a fleet device at all (most either have a blank/non-device
  `cmdb_ci`, or name a device outside the modeled fleet). `snow_ticket_count`
  being 0 for the overwhelming majority of rows is expected, not a bug.
- **`clean_log_data.py` can fail on a raw pull whose `_time` column mixes
  UTC offsets** (e.g. some rows timestamped `+0000`, others `+0100`) —
  `pandas.to_datetime` returns an `object`-dtype result for mixed offsets
  rather than a uniform tz-aware series, which breaks the subsequent
  `.dt.tz_convert("UTC")` call. Check for this if a fresh Splunk pull adds
  rows spanning a DST transition or multiple source timezones.
- **A handful of SNOW tickets are known non-BGP noise** (printer tickets
  that leaked into the export) and are excluded from classification —
  marked `EXCLUDED_NON_BGP` rather than dropped, so row counts stay
  consistent with the raw export.

---

## 7. Dependencies

`pandas`, `numpy`, `anthropic` (Stage 3 only). Python 3.9+.
