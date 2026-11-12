# North Star

**Goal:** Help the network team proactively address router conditions before they cause a
network disruption — i.e. detect at-risk routers early enough so a fix can be applied.

**Approach:** Use device telemetry (Dynatrace), log-derived event signals (Splunk), and
historical ServiceNow ticket outcomes to learn which pre-incident signatures correspond to
disruptions that were predictable in advance, and look for patterns in teh device data that could have serveres as early warning signal.

**Why this is tractable:** every historical incident in the combined dataset
(`data_pipeline/processed/bgp_combined_data_table.parquet`) already carries
ground-truth labels derived from the ticket text:
- `haiku_predictability` — Predictable / Known in Advance / Not Predictable / Indeterminate
  / N/A (no real issue)
- `haiku_root_cause` — Carrier/Circuit Issue, Hardware/Device Issue, False Alert/False
  Positive, Auto-Resolved (No Human Action), Planned Change/Maintenance, Other/Unclear
- `haiku_subcategory` — finer detail (fiber cut, BFD adjacency down, defective SFP, etc.)

`haiku_predictability`, `haiku_root_cause`, and `haiku_subcategory` is all derived from text notes in a service now ticket provided by a human. So, the ServiceNow information is not 100% acurate. 

Scope is root-cause-agnostic: any issue that can be predicted and can be actioned upon. Action can be taken by any team, not limited to network. N/A (no real issue), and False Alert/False Positive are excluded as noise.

**Definition of success:** given recent telemetry + log signals for a router, flag it as
at risk of a disruption, with enough lead time and root-cause context for the network team
to intervene before the disruption happens.

**Stability:** this is the fixed goal for the project. Do not change it unless explicitly asked.
