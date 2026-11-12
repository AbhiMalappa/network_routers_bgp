"""
Incremental SNOW ticket classifier: only submits tickets to Haiku that are
new or have changed since the last run, and appends/updates results into
one stable, ever-growing output file instead of rebuilding from scratch.

Companion to classify_snow_with_haiku_batch.py, not a replacement for it --
that script still exists for a genuine from-scratch rebuild (e.g. after a
prompt/schema change invalidates every prior classification). This script
is for the normal ongoing workflow: a new raw SNOW pull lands, and only the
tickets that are actually new or updated need to go through Haiku again.

Delta detection: a ticket needs (re)classification if its `number` isn't
in the existing results file at all, OR its `sys_updated_on` in the new
raw pull differs from what's on record for that number. opened_at is
deliberately NOT used for this -- a ticket can sit open for a while and
only get close_notes filled in well after it was opened, so an opened_at
watermark would permanently skip tickets that changed after the first time
we saw them. sys_updated_on reflects the ticket's own last-modified time,
which is exactly what we need.

A ticket that previously errored (ERROR_NO_RESULT -- a Haiku parse failure
or missing batch result) is NOT automatically retried just because it's
still in that state; it's retried only if its sys_updated_on changes, same
policy as every other ticket. An API/parse failure isn't distinguished
from "nothing changed" here -- a deliberate simplicity choice.

Three subcommands, same shape as classify_with_haiku_batch.py:

    ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_incremental.py submit
        -> finds tickets needing (re)classification, submits one batch
           (or prints "nothing to do" and exits if there are none)

    ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_incremental.py status --batch-id msgbatch_...

    ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_incremental.py collect --batch-id msgbatch_...
        -> merges new results into the stable output file: updates rows
           for reclassified tickets, adds rows for new ones, leaves every
           other row untouched

Output: processed/snow_with_haiku.csv -- one stable file, not named after
any particular raw pull. If it doesn't exist yet, the first `submit` run
classifies every ticket in the current raw pull (same as a from-scratch
build); seed it from an existing *_with_haiku.csv first if you want to
avoid re-paying for tickets already classified by the other script.

Per project convention: always ask for approval before actually running
`submit` (or any live API call), even a small test -- do not just execute it.
"""

import argparse
import csv
import glob
import json
import os
import re
import sys

import anthropic
import pandas as pd

EXCLUDED_NON_BGP_TICKETS = {"INC27756341", "INC27596789", "INC27271931"}

PIPELINE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(PIPELINE_DIR, "processed"), exist_ok=True)
RAW_SNOW_DIR = os.path.join(PIPELINE_DIR, "raw", "snow")
PROCESSED_DIR = os.path.join(PIPELINE_DIR, "processed")

MODEL = "claude-haiku-4-5"
COMMENTS_CHAR_CAP = 3000

OUT_CSV = os.path.join(PROCESSED_DIR, "snow_with_haiku.csv")
OUT_JSON = os.path.join(PROCESSED_DIR, "snow_incremental_last_batch_results.json")
BATCH_ID_FILE = os.path.join(PROCESSED_DIR, "snow_incremental_last_batch_id.txt")
PENDING_NUMBERS_FILE = os.path.join(PROCESSED_DIR, "snow_incremental_pending_numbers.json")

ROOT_CAUSES = [
    "Carrier / Circuit Issue",
    "Auto-Resolved (No Human Action)",
    "Other / Unclear",
    "Planned Change / Maintenance",
    "False Alert / False Positive",
    "Hardware / Device Issue",
]
PREDICTABILITY_LABELS = [
    "Predictable",
    "Not Predictable",
    "Known in Advance",
    "Indeterminate",
    "N/A (no real issue)",
]

CARRIER_RC = "Carrier / Circuit Issue"
HARDWARE_RC = "Hardware / Device Issue"

CARRIER_SUBCATEGORIES = [
    "Carrier scheduled/routine maintenance or planned work window",
    "Utility power loss / environmental (wildfire risk, high winds, commercial power)",
    "Fiber cut / physical cable damage (vandalism, construction, train strike, damaged/drop cable, last-mile)",
    "Carrier major/core network outage (no physical detail given)",
    "Defective/faulty equipment (SFP, card, jumper, transport hardware) replaced by carrier",
    "CRC / rising error counters on circuit",
    "BFD adjacency down / circuit issue (generic, no further detail)",
    "Carrier cause unknown / RFO pending / self-healed / transient glitch",
    "Other / unspecified",
]
HARDWARE_SUBCATEGORIES = [
    "Power supply failure",
    "Line/fabric card failure (fc11 cluster)",
    "SFP / transceiver / optical interface failure",
    "Other line/transport card failure",
    "Cable / connector issue",
    "BFD/interface instability, no named physical part",
]
VALID_SUBCATEGORIES_BY_RC = {CARRIER_RC: set(CARRIER_SUBCATEGORIES), HARDWARE_RC: set(HARDWARE_SUBCATEGORIES)}

# Identical prompt to classify_snow_with_haiku_batch.py -- kept in sync
# manually. If you change one, change the other, or the two scripts will
# silently produce different classifications for the same ticket content.
SYSTEM_PROMPT = f"""You classify ServiceNow BGP-outage incident tickets for a telecom network team.

You are given two fields from one ticket: close_notes (the agent's resolution notes) and
comments_and_work_notes (the chronological work-note history, most recent entries first,
possibly truncated). Base your answer only on what these fields actually say.

Return ONLY a JSON object (no markdown fences, no commentary) with exactly these 5 keys:

{{
  "root_cause": one of {json.dumps(ROOT_CAUSES)},
  "predictability": one of {json.dumps(PREDICTABILITY_LABELS)},
  "mechanism": a short (<12 words) free-text label naming the SPECIFIC underlying mechanism,
               e.g. "faulty SFP replaced", "fiber cut from digging", "BigPanda auto-resolved flap",
               "scheduled IOS upgrade", "vandalism damaged cable", "carrier detected errors before fix".
               Name whatever mechanism the text actually describes -- do not limit yourself to the examples above.
  "evidence": one short sentence quoting or closely paraphrasing the specific text that justifies your answer.
  "subcategory": a finer sub-issue bucket, per the rules below. Use "" (empty string) unless root_cause
                 is "Carrier / Circuit Issue" or "Hardware / Device Issue".
}}

Category definitions:
- root_cause:
  - "Carrier / Circuit Issue": a third-party carrier/ISP circuit, fiber, or their equipment caused it.
  - "Auto-Resolved (No Human Action)": notes literally say an automated system (e.g. BigPanda) resolved it with no engineer action.
  - "Planned Change / Maintenance": caused by a scheduled internal change (CHG#) or a scheduled/routine maintenance window.
  - "False Alert / False Positive": an engineer determined the alert didn't reflect a real, sustained outage.
  - "Hardware / Device Issue": a specific internal or carrier-owned hardware component (card, chassis, fan, transceiver) failed.
  - "Other / Unclear": none of the above fit, or no real cause is stated.

- predictability (the bar: would a leading indicator plausibly have given >= 4 hours of warning
  before this alert fired, with enough time for someone to act on it?):
  - "Predictable": yes -- a gradual/degrading precursor is described (e.g. rising errors, a part noted as
    faulty/degrading before replacement, a carrier reporting "at risk of failure" before a scheduled fix).
  - "Not Predictable": no -- sudden physical damage with no plausible precursor (fiber cut, vandalism, a dig,
    a sudden unexplained outage).
  - "Known in Advance": not a forecasting problem at all -- it was already on a schedule (an internal change,
    a routine/scheduled maintenance window) before it happened.
  - "Indeterminate": the text doesn't give enough detail to judge either way.
  - "N/A (no real issue)": root_cause is False Alert -- there's nothing to predict.

How to decide Predictable vs. Not Predictable -- judge primarily from the MECHANISM, not from
whether this specific ticket happens to include a detailed timestamp log:

- Default to "Predictable" whenever a NAMED HARDWARE COMPONENT failed and was replaced or repaired
  -- an SFP/transceiver, a line card, a chassis, a fan, a power supply -- even if the notes are terse
  ("Cause: Faulty SFP, replaced") with no logged flap history. These failure types generically show a
  measurable precursor on real telemetry (optical power, bias current, CPU/memory, fan speed/state,
  error counters) even when this ticket's own notes don't happen to include that data. The absence of
  a pasted-in log in THIS ticket is not evidence the failure was sudden -- it usually just means the
  engineer didn't bother logging it once the diagnosis was obvious.
- Override that default to "Not Predictable" only when the text explicitly frames the cause as sudden,
  external, and physical -- a fiber cut, vandalism, digging, construction damage, a carrier "major
  outage." These stay Not Predictable even if a handful of flap timestamps are logged right around the
  event -- those are symptoms of the sudden break, not an early warning.
- Timestamps are a MODIFIER, not a gate. A flap/BFD-adjacency-loss history for the same peer that spans
  many hours (first event to last event at least ~4 hours apart) IS strong supporting evidence for
  Predictable, even for an otherwise ambiguous cause -- it shows the exact kind of leading indicator a
  flap-count alert would have caught. But a cluster of flaps crammed into a short window (well under
  ~2 hours total span) is NOT meaningful evidence either way -- don't let a few closely-spaced timestamps
  push your answer toward or away from Predictable; fall back to the mechanism-level default above
  instead. Watch in particular for the NOC's own characterization of the flap in the notes ("quick flap,
  recovered", "flapped but quickly recovers", "EEM failover wasn't triggered") -- that explicitly says
  the engineer who read the same logs did not consider it a building problem.
- If no specific hardware component and no sudden-physical-cause language is stated (e.g. "BGP flap,
  cause unknown", "no trouble found", a bare carrier name with no further detail), use "Indeterminate" --
  there's nothing here, mechanism or timestamp, to judge from.

- subcategory (ONLY when root_cause is "Carrier / Circuit Issue" or "Hardware / Device Issue";
  use "" for every other root_cause):

  If root_cause is "Carrier / Circuit Issue", subcategory must be exactly one of:
  {json.dumps(CARRIER_SUBCATEGORIES, indent=2)}

  If root_cause is "Hardware / Device Issue", subcategory must be exactly one of:
  {json.dumps(HARDWARE_SUBCATEGORIES, indent=2)}

  Subcategory guidance:
  - Prefer the most SPECIFIC bucket that genuinely fits (e.g. a named "fc11" card failure goes in the
    fc11 bucket, not the generic "Other line/transport card failure" bucket).
  - "Carrier cause unknown / RFO pending / self-healed / transient glitch" is for tickets where the
    carrier circuit was at fault but the specific physical mechanism is still unresolved, pending,
    self-healed, or explicitly described as transient -- not simply "any circuit issue carrier-side."
    If a specific mechanism IS named (fiber cut, defective equipment, CRC errors, scheduled maintenance,
    major outage, utility power), use that bucket instead even if the text also contains words like
    "RFO" or "investigation."
  - "BFD adjacency down / circuit issue (generic, no further detail)" (carrier) / "BFD/interface
    instability, no named physical part" (hardware) are for BFD/adjacency/interface-flap language with
    NO named physical cause and NO explicit "pending/unknown/self-healed" framing either -- just a bare
    adjacency-down or instability description.
  - "Other / unspecified" (carrier only) is the catch-all when nothing above fits -- use it rather than
    forcing a weak match.

Watch for copy-paste contamination: comments_and_work_notes sometimes contains a block literally marked
"copied from Parent Incident" (or similar). Most of the time this legitimately describes the same shared
circuit/peer issue. But occasionally it describes a DIFFERENT device, peer, or incident entirely (e.g. a stray
CHG# or outage for an unrelated router pasted in during a bulk update). close_notes almost always opens with
its own "Issue: NTWK: <device> ..." line naming the device THIS ticket is actually about -- use that as your
anchor. If a cause, CHG#, or outage you're about to rely on is attached to a different device/peer than the
one named in this ticket's own Issue:/Cause: line, do not use it; prefer "Other / Unclear" / "Indeterminate"
over a confident answer built on content that isn't actually about this ticket.
"""

HAIKU_COLS = ["haiku_root_cause", "haiku_predictability", "haiku_mechanism", "haiku_evidence", "haiku_subcategory"]


def build_user_prompt(close_notes, comments):
    comments_capped = (comments or "")[:COMMENTS_CHAR_CAP]
    return (
        f"close_notes:\n{close_notes or '(blank)'}\n\n"
        f"comments_and_work_notes (first {COMMENTS_CHAR_CAP} chars):\n{comments_capped or '(blank)'}"
    )


def load_raw() -> pd.DataFrame:
    """Raw SNOW exports are NOT one ever-growing cumulative file -- as of
    2026-11-12 they're separate, non-overlapping date-range chunks (e.g.
    snow_bgp_pst_06_1_10_1.csv for June-Sept, snow_bgp_pst_10_1_11_1.csv
    for October, confirmed zero ticket-number overlap between them).
    Every matching file in raw/snow/ is loaded and combined -- picking
    only the most-recently-modified one (the old approach, still used by
    classify_snow_with_haiku_batch.py) would silently see just one chunk.

    If a ticket number ever does appear in more than one chunk (not the
    case today, but not guaranteed to stay that way), the row from
    whichever file has the LATER sys_updated_on wins -- same "most
    current data wins" principle as everything else in this script."""
    paths = sorted(glob.glob(os.path.join(RAW_SNOW_DIR, "snow_bgp_pst_*.csv")))
    if not paths:
        raise FileNotFoundError(f"No snow_bgp_pst_*.csv found in {RAW_SNOW_DIR}")

    dfs = [pd.read_csv(p, encoding="cp1252", low_memory=False) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Raw pull: {len(paths)} file(s) ({', '.join(os.path.basename(p) for p in paths)}), "
          f"{len(df)} rows total")

    dupes = df["number"].duplicated(keep=False).sum()
    if dupes:
        print(f"NOTE: {dupes} row(s) share a ticket number across raw files -- "
              f"keeping the one with the latest sys_updated_on for each.")
        df["_sort_key"] = pd.to_datetime(df["sys_updated_on"], format="mixed")
        df = df.sort_values("_sort_key").drop_duplicates(subset=["number"], keep="last").drop(columns=["_sort_key"])
    return df.reset_index(drop=True)


def load_existing() -> pd.DataFrame:
    if not os.path.exists(OUT_CSV):
        return pd.DataFrame()
    return pd.read_csv(OUT_CSV, low_memory=False)


def compute_opened_at_utc(series: pd.Series) -> pd.Series:
    """Same PST-labeled-but-actually-PDT handling as classify_snow_with_haiku_batch.py."""
    return (
        pd.to_datetime(series, format="mixed")
        .dt.tz_localize("America/Los_Angeles")
        .dt.tz_convert("UTC")
        .astype(str)
    )


def find_tickets_needing_classification(raw: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    """A row from `raw` needs classification if its number isn't in `existing`
    at all, or its sys_updated_on differs from what's on record. Excluded
    tickets never need it regardless. Returns the subset of `raw` rows to
    submit."""
    raw = raw[~raw["number"].isin(EXCLUDED_NON_BGP_TICKETS)].copy()

    if existing.empty:
        print(f"No existing {os.path.basename(OUT_CSV)} found -- treating every ticket as new "
              f"(first run, or seed it from an existing *_with_haiku.csv first to avoid re-paying "
              f"for already-classified tickets).")
        return raw

    recorded_updated_on = dict(zip(existing["number"], existing["sys_updated_on"]))
    raw_updated = pd.to_datetime(raw["sys_updated_on"], format="mixed")

    def needs_it(row, parsed_updated) -> bool:
        number = row["number"]
        if number not in recorded_updated_on:
            return True
        recorded = recorded_updated_on[number]
        if pd.isna(recorded):
            return True
        recorded_parsed = pd.to_datetime(recorded, format="mixed")
        return parsed_updated != recorded_parsed

    mask = [needs_it(row, parsed) for (_, row), parsed in zip(raw.iterrows(), raw_updated)]
    return raw[mask]


def cmd_submit(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    raw = load_raw()
    existing = load_existing()
    todo = find_tickets_needing_classification(raw, existing)

    if todo.empty:
        print("Nothing to do -- every ticket in the raw pull is already classified and unchanged.")
        return

    print(f"{len(todo)} ticket(s) need (re)classification "
          f"({len(raw) - len(todo)} unchanged, skipped).")

    client = anthropic.Anthropic(api_key=api_key)
    requests = []
    for _, row in todo.iterrows():
        user_prompt = build_user_prompt(row.get("close_notes", ""), row.get("comments_and_work_notes", ""))
        requests.append({
            "custom_id": row["number"],
            "params": {
                "model": MODEL,
                "max_tokens": 400,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        })

    batch = client.messages.batches.create(requests=requests)

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)
    with open(PENDING_NUMBERS_FILE, "w") as f:
        json.dump(todo["number"].tolist(), f)

    print(f"Batch submitted. id = {batch.id}")
    print(f"Saved batch id to {BATCH_ID_FILE}")
    print(f"processing_status = {batch.processing_status}")


def cmd_status(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    batch_id = args.batch_id or _read_saved_batch_id()
    batch = client.messages.batches.retrieve(batch_id)
    print(f"batch_id: {batch.id}")
    print(f"processing_status: {batch.processing_status}")
    print(f"request_counts: {batch.request_counts}")
    print(f"created_at: {batch.created_at}")
    print(f"ended_at: {getattr(batch, 'ended_at', None)}")


def cmd_collect(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    batch_id = args.batch_id or _read_saved_batch_id()
    batch = client.messages.batches.retrieve(batch_id)
    if batch.processing_status != "ended":
        print(f"Batch is not finished yet (processing_status={batch.processing_status}). Not collecting.")
        sys.exit(1)

    raw = load_raw()
    raw_by_number = {row["number"]: row for _, row in raw.iterrows()}

    parsed = {}
    errors = []
    subcategory_warnings = 0
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            text = "".join(b.text for b in result.result.message.content if b.type == "text").strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
            try:
                data = json.loads(text)
                root_cause = data.get("root_cause", "")
                subcategory = data.get("subcategory", "")
                valid_subs = VALID_SUBCATEGORIES_BY_RC.get(root_cause)
                if valid_subs is None:
                    if subcategory:
                        subcategory_warnings += 1
                        subcategory = ""
                elif subcategory not in valid_subs:
                    subcategory_warnings += 1
                    subcategory = ""
                data["subcategory"] = subcategory
                parsed[custom_id] = data
            except Exception as e:
                errors.append((custom_id, f"parse error: {e}", text[:200]))
        else:
            errors.append((custom_id, result.result.type, getattr(result.result, "error", None)))

    print(f"Collected {len(parsed)} successful results, {len(errors)} errors.")
    if subcategory_warnings:
        print(f"{subcategory_warnings} results had an invalid/unexpected subcategory for their root_cause "
              f"-- left blank for those rows rather than discarding the rest of the classification.")
    if errors:
        print("Errors:")
        for custom_id, kind, detail in errors[:20]:
            print(f"  {custom_id}: {kind} {detail}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2)
    print(f"Saved raw results to {OUT_JSON}")

    pending_numbers = []
    if os.path.exists(PENDING_NUMBERS_FILE):
        with open(PENDING_NUMBERS_FILE) as f:
            pending_numbers = json.load(f)

    new_rows = []
    for number in pending_numbers:
        if number not in raw_by_number:
            continue  # shouldn't happen -- raw pull used for collect differs from the one used for submit
        row = dict(raw_by_number[number])
        row["opened_at_utc"] = compute_opened_at_utc(pd.Series([row.get("opened_at", "")])).iloc[0]
        if number in EXCLUDED_NON_BGP_TICKETS:
            for col in HAIKU_COLS:
                row[col] = "EXCLUDED_NON_BGP"
        elif number in parsed:
            data = parsed[number]
            row["haiku_root_cause"] = data.get("root_cause", "")
            row["haiku_predictability"] = data.get("predictability", "")
            row["haiku_mechanism"] = data.get("mechanism", "")
            row["haiku_evidence"] = data.get("evidence", "")
            row["haiku_subcategory"] = data.get("subcategory", "")
        else:
            for col in HAIKU_COLS:
                row[col] = "ERROR_NO_RESULT"
        new_rows.append(row)
    new_df = pd.DataFrame(new_rows)

    existing = load_existing()
    if existing.empty:
        merged = new_df
    else:
        # Rows for tickets we just (re)classified replace the old row for
        # that number; every other existing row is left exactly as-is.
        merged = existing[~existing["number"].isin(new_df["number"])]
        merged = pd.concat([merged, new_df], ignore_index=True)

    merged = merged.sort_values("number").reset_index(drop=True)
    merged.to_csv(OUT_CSV, index=False, encoding="utf-8")

    if os.path.exists(PENDING_NUMBERS_FILE):
        os.remove(PENDING_NUMBERS_FILE)

    print(f"Updated {len(new_df)} row(s) ({len(parsed)} classified, "
          f"{sum(1 for n in pending_numbers if n in EXCLUDED_NON_BGP_TICKETS)} excluded, "
          f"{len(new_df) - len(parsed) - sum(1 for n in pending_numbers if n in EXCLUDED_NON_BGP_TICKETS)} errored). "
          f"{OUT_CSV} now has {len(merged)} total rows.")


def _read_saved_batch_id():
    if os.path.exists(BATCH_ID_FILE):
        with open(BATCH_ID_FILE) as f:
            return f.read().strip()
    print("ERROR: no --batch-id given and no saved batch id found.", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="find tickets needing (re)classification and submit a batch")
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser("status", help="check processing status (does not block)")
    p_status.add_argument("--batch-id", default=None)
    p_status.set_defaults(func=cmd_status)

    p_collect = sub.add_parser("collect", help="merge results into the stable output file (only once ended)")
    p_collect.add_argument("--batch-id", default=None)
    p_collect.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
