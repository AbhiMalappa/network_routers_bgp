"""
End-to-end BGP ticket classifier: raw ServiceNow export in, fully enriched
CSV out. One Haiku call per ticket via Claude's Message Batches API (50%
cheaper than synchronous), returning root_cause, predictability, mechanism,
evidence, and -- when root_cause is Carrier or Hardware -- a finer
sub-issue bucket (subcategory), all in a single pass. Re-run this any time
the raw ticket export grows or changes; it always produces a complete,
self-consistent output file from scratch.

(Subcategory used to be a separate second-pass script run only on
Carrier/Hardware tickets after the fact. Folding it into this same call is
better: one script, one batch, no risk of the second pass's input file
getting out of sync with this one -- and the model sees the full
close_notes/comments text when picking a subcategory, not just its own
short mechanism/evidence summary, so it tends to be more accurate too.)

`collect` also adds an `opened_at_utc` column (2026-11-09), inserted right
after `opened_at` -- the raw export's own timestamp has no embedded UTC
offset and is labeled PST but empirically confirmed (see
classify_snow_predictability.py) to actually be PDT (Pacific local time
with DST), so this is tz_localize("America/Los_Angeles") ->
tz_convert("UTC"), not a fixed offset. Needed so this output can be joined
against Dynatrace/Splunk data (both already UTC) by device+time elsewhere
in data_pipeline/.

Paths (2026-11-09): wired into this project's data/raw -> data/processed
convention, same as every other src/ script. Input is picked up by globbing
data/raw/snow/snow_bgp_pst_*.csv (most recently modified) rather than one
fixed filename -- this file gets renamed on every pull, so hardcoding
today's name would go stale on the very next one (same reasoning as
build_and_clean_dynatrace_device_data.py's dql_15query_parquet_* glob).
Output CSV/JSON/batch-id filenames are derived from the input file's own
stem for the same reason, and so that re-running against a later pull
doesn't silently overwrite an in-progress-or-just-collected earlier batch's
files.

Three steps, each its own subcommand so a long-running batch survives this
script exiting in between:

    ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_batch.py submit
        -> builds all requests, submits one batch, prints and saves the batch id

    ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_batch.py status --batch-id msgbatch_...
        -> prints current processing_status and request counts, does not block

    ANTHROPIC_API_KEY=sk-ant-... python3 classify_snow_with_haiku_batch.py collect --batch-id msgbatch_...
        -> only works once status is "ended"; retrieves all results, writes
           the output CSV (original columns + 5 new haiku_* columns) and a
           raw JSON dump

The 3 known non-BGP tickets (printer tickets that leaked into this export --
INC27756341, INC27596789, INC27271931) are excluded from classification and
marked EXCLUDED_NON_BGP in the output CSV instead of being dropped.

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

# 2026-11-10: raw/processed moved to data_pipeline/raw, data_pipeline/
# processed (self-contained pipeline).
PIPELINE_DIR = os.path.dirname(__file__)
RAW_SNOW_DIR = os.path.join(PIPELINE_DIR, "raw", "snow")
PROCESSED_DIR = os.path.join(PIPELINE_DIR, "processed")

MODEL = "claude-haiku-4-5"
COMMENTS_CHAR_CAP = 3000


def _resolve_input_csv() -> str:
    candidates = sorted(
        glob.glob(os.path.join(RAW_SNOW_DIR, "snow_bgp_pst_*.csv")),
        key=os.path.getmtime,
    )
    if not candidates:
        raise FileNotFoundError(f"No snow_bgp_pst_*.csv found in {RAW_SNOW_DIR}")
    chosen = candidates[-1]  # most recently modified
    if len(candidates) > 1:
        print(f"NOTE: {len(candidates)} snow_bgp_pst_*.csv files found in {RAW_SNOW_DIR}; "
              f"using the most recently modified: {os.path.basename(chosen)}")
    return chosen


CSV_PATH = _resolve_input_csv()
_STEM = os.path.basename(CSV_PATH).rsplit(".csv", 1)[0]
BATCH_ID_FILE = os.path.join(PROCESSED_DIR, f"{_STEM}_last_batch_id.txt")
OUT_CSV = os.path.join(PROCESSED_DIR, f"{_STEM}_with_haiku.csv")
OUT_JSON = os.path.join(PROCESSED_DIR, f"{_STEM}_haiku_batch_results.json")

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

# Sub-issue buckets -- only used when root_cause is CARRIER_RC or HARDWARE_RC.
# Keep these strings byte-for-byte identical if you ever touch them: the
# deck-building scripts group on these literal values.
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


def build_user_prompt(close_notes, comments):
    comments_capped = (comments or "")[:COMMENTS_CHAR_CAP]
    return (
        f"close_notes:\n{close_notes or '(blank)'}\n\n"
        f"comments_and_work_notes (first {COMMENTS_CHAR_CAP} chars):\n{comments_capped or '(blank)'}"
    )


def compute_opened_at_utc(rows: list) -> dict:
    """opened_at has no embedded UTC offset -- labeled PST but empirically
    confirmed (classify_snow_predictability.py) to actually be PDT (Pacific
    local time with DST), hence tz_localize("America/Los_Angeles") rather
    than a fixed offset, same as that script. Returns {number: opened_at_utc
    ISO string}, computed once over all rows via pandas rather than a
    per-row Python datetime parse, same as every other opened_at -> UTC
    conversion in this project."""
    numbers = [r["number"] for r in rows]
    opened_at_utc = (
        pd.to_datetime([r.get("opened_at", "") for r in rows], format="mixed")
        .tz_localize("America/Los_Angeles")
        .tz_convert("UTC")
    )
    return dict(zip(numbers, opened_at_utc.astype(str)))


def load_all_rows():
    # cp1252, not UTF-8 -- confirmed repeatedly across every SNOW export
    # this project has handled. utf-8 + errors="replace" wouldn't raise on
    # this file, but would silently mangle any non-ASCII byte (smart
    # quotes, em-dashes -- both seen in real close_notes) into U+FFFD
    # rather than decode it correctly.
    rows = []
    with open(CSV_PATH, newline="", encoding="cp1252") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    return rows, fieldnames


def cmd_submit(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    rows, _ = load_all_rows()
    requests = []
    skipped = 0
    for row in rows:
        number = row.get("number")
        if number in EXCLUDED_NON_BGP_TICKETS:
            skipped += 1
            continue
        user_prompt = build_user_prompt(row.get("close_notes", ""), row.get("comments_and_work_notes", ""))
        requests.append({
            "custom_id": number,
            "params": {
                "model": MODEL,
                "max_tokens": 400,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        })

    print(f"Submitting a batch of {len(requests)} requests ({skipped} excluded non-BGP tickets skipped)...")
    batch = client.messages.batches.create(requests=requests)

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)

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

    rows, fieldnames = load_all_rows()

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
                    # root_cause isn't Carrier/Hardware -- subcategory should be blank.
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

    opened_at_utc_by_number = compute_opened_at_utc(rows)

    haiku_cols = ["haiku_root_cause", "haiku_predictability", "haiku_mechanism", "haiku_evidence", "haiku_subcategory"]
    # opened_at_utc inserted right after opened_at, not appended at the end --
    # matches the one-off patch already applied to the existing _with_haiku.csv.
    out_fieldnames = []
    for c in fieldnames:
        out_fieldnames.append(c)
        if c == "opened_at":
            out_fieldnames.append("opened_at_utc")
    out_fieldnames += haiku_cols
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            number = row["number"]
            out_row = dict(row)
            out_row["opened_at_utc"] = opened_at_utc_by_number.get(number, "")
            if number in EXCLUDED_NON_BGP_TICKETS:
                for col in haiku_cols:
                    out_row[col] = "EXCLUDED_NON_BGP"
            elif number in parsed:
                data = parsed[number]
                out_row["haiku_root_cause"] = data.get("root_cause", "")
                out_row["haiku_predictability"] = data.get("predictability", "")
                out_row["haiku_mechanism"] = data.get("mechanism", "")
                out_row["haiku_evidence"] = data.get("evidence", "")
                out_row["haiku_subcategory"] = data.get("subcategory", "")
            else:
                for col in haiku_cols:
                    out_row[col] = "ERROR_NO_RESULT"
            writer.writerow(out_row)

    print(f"Saved {len(rows)} rows ({len(parsed)} classified, {len(EXCLUDED_NON_BGP_TICKETS)} excluded, "
          f"{len(rows) - len(parsed) - len(EXCLUDED_NON_BGP_TICKETS)} errored) to {OUT_CSV}")


def _read_saved_batch_id():
    if os.path.exists(BATCH_ID_FILE):
        with open(BATCH_ID_FILE) as f:
            return f.read().strip()
    print("ERROR: no --batch-id given and no saved batch id found.", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_submit = sub.add_parser("submit", help="build and submit the batch")
    p_submit.set_defaults(func=cmd_submit)

    p_status = sub.add_parser("status", help="check processing status (does not block)")
    p_status.add_argument("--batch-id", default=None)
    p_status.set_defaults(func=cmd_status)

    p_collect = sub.add_parser("collect", help="retrieve results and write the output CSV (only once ended)")
    p_collect.add_argument("--batch-id", default=None)
    p_collect.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
