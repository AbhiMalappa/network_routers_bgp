"""
BGP flap prediction — audit how much real Splunk log volume is being
silently dropped by the exact (mnemonic, message_text) join in
build_log_features.py, for mnemonics the project already partially keeps.

Why this exists: a brand-new mnemonic (one never seen in docs/log_message_
text_to_feature_reference.csv at all) is the shallow case -- a plain
mnemonic-set diff against the reference file catches it, already handled
elsewhere. This catches the deeper, easier-to-miss case: a mnemonic that IS already
known and partially kept (e.g. ADJCHANGE, BFD_SESS_DESTROYED), where a real
log line's EXACT message_text wording doesn't match any catalogued variant
for that mnemonic. build_wide_features()'s inner join (build_log_features.py)
drops that row silently — same failure mode as an unseen mnemonic, but
invisible to a mnemonic-only diff.

Found and fixed once already (2026-09-30): confirmed via a device_flap_count
cross-validation that ADJCHANGE and BFD_SESS_DESTROYED were silently
dropping 54.8% and 88.5% of their real events respectively — entirely
different message-text wording (e.g. newer 10-Gig interface hardware
phrasing a BFD teardown slightly differently than the catalogued variant),
not a detection problem. This script exists so that's a one-command check
next time, not a multi-hour manual investigation.

What it does NOT catch:
  - Mnemonics marked `category=dropped` in the reference file — those are
    INTENDED to be fully excluded regardless of message_text (e.g. DUP_TIMER,
    confirmed noise). Only "partially kept" mnemonics (at least one non-
    dropped row) are in scope here; a 100% drop rate for a fully-dropped
    mnemonic is correct, not a gap.
  - Brand-new mnemonics never seen in the reference file at all — a plain
    mnemonic-set diff against the reference file catches that case, not
    this script.
  - Message-text TRUNCATION specifically (the SPL-side root cause of some of
    these gaps, see docs/spl_to_get_log_data.txt's 2026-09-30 fix) — this
    script reports the symptom (dropped volume), not that specific cause.
  - Whether a given dropped variant, once triaged, is legitimate to add or
    should stay excluded (e.g. OSPF/CLNS facility contamination for
    ADJCHANGE, or a different-protocol BFD client for BFD_SESS_DESTROYED) —
    that's a human triage step, same discipline as the original mnemonic
    triage. This script finds WHAT'S missing, not whether it should be added.

Output: prints a sorted drop-rate table for every partially-kept mnemonic,
flags any above --threshold (default 10%) as needing re-triage, and writes
a sample of the actual dropped (message_text, facility, count) rows for
each flagged mnemonic to data_pipeline/processed/log_message_text_coverage_gaps.csv
— the same shape of evidence used to do the original ADJCHANGE/
BFD_SESS_DESTROYED triage, ready to review without re-deriving it by hand.
"""
import argparse
import os

import pandas as pd

# 2026-11-11: moved into data_pipeline/ (QA tool for this pipeline's own
# stage-2 output). raw/processed/docs all live inside data_pipeline/ --
# fully self-contained, no path here reaches outside this folder.
PIPELINE_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(PIPELINE_DIR, "processed"), exist_ok=True)
LOG_PATH = os.path.join(PIPELINE_DIR, "processed", "cleaned_log_data.parquet")
REF_PATH = os.path.join(PIPELINE_DIR, "docs", "log_message_text_to_feature_reference.csv")
OUT_PATH = os.path.join(PIPELINE_DIR, "processed", "log_message_text_coverage_gaps.csv")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--threshold", type=float, default=10.0,
                    help="Drop-rate percentage above which a mnemonic is flagged as needing "
                         "re-triage. Default 10.0 (i.e. >10%% of its rows silently dropped).")
    p.add_argument("--samples-per-mnemonic", type=int, default=20,
                    help="Max distinct dropped message_text variants to write per flagged "
                         "mnemonic in the output CSV, sorted by row count descending. Default 20.")
    return p.parse_args()


def compute_drop_rates(logs: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    kept_ref = ref[ref["category"] != "dropped"]
    partially_kept = sorted(kept_ref["mnemonic"].unique())

    rows = []
    for mnem in partially_kept:
        total = logs[logs["mnemonic"] == mnem]
        if len(total) == 0:
            continue
        merged = total.merge(kept_ref[kept_ref["mnemonic"] == mnem][["mnemonic", "message_text"]],
                              on=["mnemonic", "message_text"], how="left", indicator=True)
        dropped = (merged["_merge"] == "left_only").sum()
        rows.append({
            "mnemonic": mnem,
            "total_rows": len(total),
            "dropped_rows": int(dropped),
            "drop_rate_pct": 100 * dropped / len(total),
            "feature_cols": ", ".join(sorted(kept_ref.loc[kept_ref["mnemonic"] == mnem, "feature_col_name"].unique())),
        })
    return pd.DataFrame(rows).sort_values("dropped_rows", ascending=False).reset_index(drop=True)


def dropped_samples(logs: pd.DataFrame, ref: pd.DataFrame, mnemonic: str, top_n: int) -> pd.DataFrame:
    kept_ref = ref[ref["category"] != "dropped"]
    total = logs[logs["mnemonic"] == mnemonic]
    merged = total.merge(kept_ref[kept_ref["mnemonic"] == mnemonic][["mnemonic", "message_text"]],
                          on=["mnemonic", "message_text"], how="left", indicator=True)
    dropped = merged[merged["_merge"] == "left_only"]
    summary = (dropped.groupby(["message_text", "facility"]).size()
               .reset_index(name="n").sort_values("n", ascending=False).head(top_n))
    summary.insert(0, "mnemonic", mnemonic)
    return summary


def main():
    args = parse_args()
    logs = pd.read_parquet(LOG_PATH)
    ref = pd.read_csv(REF_PATH)

    rates = compute_drop_rates(logs, ref)
    print("=== Drop rate by mnemonic (partially-kept mnemonics only — fully-dropped ones excluded) ===")
    print(rates.to_string(index=False))
    print()

    flagged = rates[rates["drop_rate_pct"] > args.threshold]
    print(f"Mnemonics above {args.threshold}% drop rate (flagged for re-triage): {len(flagged)}")
    if flagged.empty:
        print("None — nothing to triage.")
        return

    print(flagged[["mnemonic", "total_rows", "dropped_rows", "drop_rate_pct", "feature_cols"]].to_string(index=False))

    samples = pd.concat(
        [dropped_samples(logs, ref, m, args.samples_per_mnemonic) for m in flagged["mnemonic"]],
        ignore_index=True,
    )
    samples.to_csv(OUT_PATH, index=False)
    print(f"\nWrote {OUT_PATH} ({len(samples)} sample rows across {len(flagged)} flagged mnemonics) "
          f"— same shape of evidence as the original triage, ready to review before adding to "
          f"{REF_PATH}.")


if __name__ == "__main__":
    main()
