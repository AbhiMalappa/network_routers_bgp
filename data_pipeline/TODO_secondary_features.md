# Planned changes — build_secondary_features.py

IMPLEMENTED 2026-11-13 (all 11 items below, #0/#1/#2/#3-superseded-by-#5/
#4/#5/#6/#7/#8/#9/#10). Verified against independent from-scratch
recomputation for every new/changed derived feature -- see
docs/derived_feature_metrics.md for the resulting reference doc. One real
bug was caught and fixed during verification, not present in the final
output: add_mad_zscore_group_a()'s weekday/weekend masks were built from
the CURRENT row's day-type instead of the day-type of the (shifted) value
being masked -- fixed by shifting the day-type mask the same way the
values themselves are shifted. Also refined one naming choice beyond this
file's literal wording: Group B's rolling-max lag is named
`_max_prior_24h/_7d`, not the plain `_prior_24h/_7d` this file used as a
placeholder example -- that suffix already means "SUM" everywhere else in
this pipeline, and reusing it for MAX would silently make it mean two
different things depending on which column it's attached to.

Findings and decisions below are from auditing all 54 LEVEL_METRICS/
SPREAD_STAT_METRICS columns that go through `add_trend_zscore_volatility()`
(now split into several functions -- see build_secondary_features.py).

## 0. _delta_24h -> _slope_prior_48h (TREND_METRICS, i.e. all of LEVEL_METRICS
+ SPREAD_STAT_METRICS, EXCEPT Group B below)

Current `_delta_24h` is a raw two-point difference (noisy -- rides on 2
single samples; not normalized across metrics/devices).

Change to: linear regression slope (direction + magnitude) over the prior
48h (12 buckets, min 10 valid -- same 2-bucket gap tolerance as the rest of
the pipeline). Computed via the closed-form OLS slope formula
(`(n*Sxy - Sx*Sy) / (n*Sxx - Sx^2)`, all sums as plain rolling sums of a
masked absolute row-index, NOT a per-window regression call -- keeps it
vectorized). Verified against np.polyfit on real data (max diff 3.3e-14,
30 spot-checks); timed at ~3.5s/metric over the full 1.3M-row table (~3 min
for all applicable metrics).

Considered and rejected: a second-order "slope of slope" (acceleration)
feature. Differencing two already-smoothed slope estimates compounds
estimation noise and needs ~96h of history minimum before it even exists.
No specific precursor mechanism identified yet that would need it -- leave
at a single slope unless one comes up.

Group B exclusion (see #3 below): checked empirically on
total_interface_errors and snow_ticket_count -- slope is exactly 0 for
92.7%/99.1% of rows, with an explosively unstable nonzero tail (slope std
69,272 for total_interface_errors, whose median value is 0). Same
degenerate-constant-plus-wild-tail shape as the z-score problem in #3.
Group B gets NO slope feature -- SUM lag (already decided in #3) is the
right way to express "how much happened recently" for sparse counts.

## 1. Z-score baseline: expanding -> rolling 14-day window (applies to BOTH
Group A and Group C -- the window mechanism is shared; only the statistic
and stratification differ between them, see #2/#4)

Current (`_zscore_vs_own_history`): expanding mean/std over ALL prior
history per device. Problems: baseline never forgets (gets less sensitive
over time), can't adapt to a legitimate regime change (traffic growth,
capacity upgrade).

Change to: rolling 14-day window. (Originally recommended 28-day -- 4 full
weekly cycles vs. 14-day's 2 -- for a more statistically stable mean/std.
Deliberately overridden to 14-day: user's explicit choice, trading some
cycle-count stability for faster adaptation. Known consequence for Group A
specifically: with the weekday/weekend split (#2), 14 days gives ~24
weekend buckets vs. ~48 at 28 days -- the weekend stratum is thinner.)

Both halves of the formula move from expanding to this same rolling-14d
window -- not just baseline_mean: `baseline_std` (Group C) / the analogous
`baseline_mad` (Group A) switch too. There's no version of this where one
half of the pair stays expanding while the other goes rolling.

## 2. Group A (severe outlier contamination) -- median/MAD + weekday split,
renamed column: `_mad_zscore_vs_own_history`

Columns: ifcsTrafficIn_bps, ifcsTrafficOut_bps, traffic_in/out_min_in_bin,
traffic_in/out_max_in_bin, traffic_in/out_std, traffic_in/out_range.

Evidence: mean is 10,000x-75,000x the median; std exceeds the mean itself.
Same columns show the strongest weekday/weekend gap (median ratio 1.4x-4.2x).

Change: baseline = median + MAD (not mean/std), computed separately for
weekday vs. weekend buckets (not blended), over the rolling 14-day window
from #1.

Naming: output column renamed `_mad_zscore_vs_own_history` (not
`_zscore_vs_own_history`) -- a median/MAD statistic is a genuinely different
formula (the standard term is "modified z-score"), not just a more-robust
version of the same number as Group C's mean/std version. Picked "mad" over
"modified" in the name since it states the mechanism directly rather than
requiring the reader to know that term. Group C keeps the unchanged
`_zscore_vs_own_history` name.

RESOLVED: `_std_prior_24h`/`_std_prior_7d` (the standalone volatility
feature, separate from the z-score baseline) -> same fix, same reasoning,
renamed `_mad_prior_24h`/`_mad_prior_7d` for Group A.

Why it needed the same fix: the original Group A evidence was a GLOBAL
mean/median ratio (mixing all devices of very different scale). Checked
PER-DEVICE instead (what the rolling calc actually sees): median per-device
ratio is only 1.42, and 85% of devices are under 2x -- but a real minority,
31 of 1,436 devices (2.2%), have catastrophic WITHIN-device skew (up to
962,707x on the worst one) -- i.e. that device's own history has a few
extreme glitch readings mixed with normal ones. A plain rolling std
explodes for any window containing one of those glitches (std squares
deviations, so one freak value dominates); MAD doesn't, for the same reason
it fixed the z-score baseline.

Formula: `mad_prior_24h/7d = 1.4826 * median(|prior window values -
median(prior window values)|)`, same windows/min_periods as the std version
it replaces (6-bucket/24h min 4, 42-bucket/7d min 40), same 1.4826 scaling
constant used in the z-score's modified-z formula (so this feature and the
z-score's internal MAD denominator are the same quantity, not two things
that happen to share a name).

Performance: no closed-form vectorized trick exists for MAD (needs an
actual sort, unlike the slope's sum-based formula). Naive
`.rolling().apply()`: 142s/metric (74.4s 24h + 67.6s 7d) -> ~24 min for all
10 Group A metrics. Faster alternative found and verified: build each
device's full set of overlapping windows at once via
`numpy.lib.stride_tricks.sliding_window_view`, then compute median/MAD for
all of that device's windows in one vectorized numpy call -- cuts Python-
level iteration from 1.3M (one per row) to 1,510 (one per device). Result:
9.0s/metric (3.2s + 5.8s) -> ~1.5 min for all 10 metrics, ~16x faster.

Correctness bug caught and fixed during verification: the first
`.rolling().apply()` reference implementation used plain `np.median`
instead of `np.nanmedian` inside the window function. `np.median` on any
array containing even one `NaN` returns `NaN` -- even when `min_periods` is
satisfied by the non-null count. This silently produced `NaN` near the
start of every device's series (or after any data gap) where the window
was partially filled, well past the point where pandas' own `min_periods`
logic says a result should exist. Fixed by using `np.nanmedian` throughout;
re-verified exact match (0.0 max diff, 0 nullness disagreements) between
the corrected `.rolling().apply()` reference and the stride-trick version
on a 50-device sample. Worth remembering if this is ever reimplemented --
`np.median`/`np.std` do not skip NaN the way pandas' own `.rolling().mean()`
etc. do; always use the `nan*` numpy variant inside a `raw=True` callback.

## 3. Group B (zero-inflated counts) -- drop z-score, add SUM lag instead

Columns: total_interface_errors, total_interface_discards,
worst_interface_errors, worst_interface_discards, num_interfaces_with_errors,
num_interfaces_with_discards, abnormal_interface_count,
discards_min/max/std/range_in_bin, snow_ticket_count.

Evidence: median = 0 for all of these (most devices/buckets have zero
errors/discards/tickets). Median/MAD doesn't fix this either -- when >50% of
values are 0, MAD is also 0, so the baseline denominator collapses and any
single nonzero reading produces a z-score dominated by the epsilon floor,
not a real comparison.

These are the same shape of data as device_flap_count/num_peers_flapped/the
10 hardware abnormal flags, which the pipeline ALREADY handles correctly:
SUM lag (prior_24h/_7d) only, no z-score ("trend/z-score of a
persistence-adjacent count is a weak construction" -- module's own stated
reasoning, just not applied here yet).

Change: remove these columns from ZSCORE_METRICS/VOLATILITY_METRICS AND from
the new _slope_prior_48h treatment (see #0 -- superseded: originally thought
_delta_24h's replacement would stay a plain difference unaffected by this
issue, but the slope version has the same degenerate-tail problem).

SUPERSEDED by #5 below -- "add them to the SUM-lag treatment instead" was
too blunt. Group B is not one shape of data; see #5 for the actual split.

## 5. Group B is not uniform -- split into SUM lag / rolling-max lag / no lag

Checked the mechanism before applying it blindly (per #3's blanket "add
SUM lag"): Group B's 12 columns are actually three different shapes --
6 genuine counts, 5 per-bucket worst/spread snapshots, and 1 (discards_range)
that turned out to be fully redundant with two already-covered columns.

**Genuine counts (6) -- SUM lag fits, as originally decided in #3:**
total_interface_errors, total_interface_discards, num_interfaces_with_errors,
num_interfaces_with_discards, abnormal_interface_count, snow_ticket_count.
"How many accumulated in the last 24h/7d" -- same mechanism as the existing
hardware-flag/log-column SUM lag.

**"Worst"/min/max-of-a-bucket stats (5) -- SUM lag is a semantic mismatch,
use rolling MAX instead:** worst_interface_errors, worst_interface_discards,
discards_min_in_bin, discards_max_in_bin, discards_std. These are already a
max/min/spread-WITHIN-one-bucket snapshot (worst interface that bucket, or
min/max/std of raw samples in that bin) -- summing per-bucket extremes
across time isn't a "total" of anything meaningful. Rolling MAX was chosen
over rolling mean/median: mean/median would dilute exactly the
peak-severity information these columns exist to isolate, converging them
back toward "typical" -- if an averaged version were wanted, the device's
existing plain-average-error column (which already gets full z-score/slope
treatment) already serves that purpose; an averaged copy of the WORST
column would just be a redundant, noisier version of something that
already exists. Rolling max keeps the intended meaning:
`worst_interface_errors_prior_24h` = "the single worst reading seen in any
of the last 6 buckets" -- a clean "has it gotten bad recently" signal.
Same windows/min_periods as every other prior_24h/_7d lag. discards_std
joins this group for the same reason (it's a per-bucket spread snapshot,
not a count) -- `discards_std_prior_24h/_7d` = "the most volatile single
bucket recently."

**No lag feature needed:** discards_range. Checked empirically (0.0 max
diff across 1,199,669 rows) AND confirmed at the source DQL definition
itself (query_2c_traffic_discards_volatility.txt:61 --
`discards_range = discards_max_in_bin - discards_min_in_bin`) -- it is
EXACTLY redundant with discards_min_in_bin/discards_max_in_bin, both
already in the rolling-max group above. Carries zero information beyond
two columns already being lagged, so it gets no separate lag treatment of
its own.

Source-verified against the original legacy pipeline's own design intent
(xg_boost/src/build_features.py:301-307, 2026-09-30 comment): confirms
worst_interface_errors/discards as "max-across-interfaces 'worst'
readings" and num_interfaces_with_errors/discards as "counts, same
treatment as peers_with_data/interfaces_with_data" -- matches this split
exactly, no surprises.

## 6. recently_rebooted_24h/_48h AND restart_flag -- add SUM lag to both

Current: recently_rebooted_24h/_48h is a point-in-time flag only (uptime_hrs
< 24/48 on the current row), no lag at all.

Checked the mechanism before assuming "just add SUM lag" was enough:
recently_rebooted_24h behaves exactly like the persisting flags already
SUM-lagged elsewhere (ABNORMAL_FLAG_COLS) -- one single reboot event flips
it to 1 and it STAYS 1 for ~6-7 consecutive buckets (until uptime crosses
24h), not just the one bucket the reboot happened in. Confirmed via the
counts: restart_flag (the true one-shot reboot-event indicator) fires 1,020
times total; recently_rebooted_24h is 1 on 6,762 rows -- a 6.63 ratio,
matching the ~6-bucket/24h persistence window almost exactly. Verified
directly on one device's actual reboot: restart_flag=1 at one single
timestamp, recently_rebooted_24h=1 for the following 7 consecutive buckets.

So SUM-lagging recently_rebooted_24h/_48h directly gives "how much of the
recent window was spent in a post-reboot state" (real, but inflated ~6-7x
by persistence, not a count of reboot events). SUM-lagging restart_flag
instead gives the actual "how many distinct reboots in the last 24h/7d" --
a different, cleaner question.

Decision: add SUM lag (prior_24h/_7d, same windows/min_periods as every
other SUM lag) to BOTH -- they answer different questions, not duplicates
of each other. This also closes out restart_flag, one of the previously
-flagged-but-not-yet-reviewed columns from the Dynatrace-origin column
audit (see the "_prior_7d_avg" discussion).

## 4. Group C (CPU, memory, temp, power, bias_current, peer_count,
worst_established_time) -- no change to statistic or stratification

Mean/median ratios are 1.0-2x (not thousands), weekday/weekend ratios sit
around 1.0. No evidence of either problem Group A has, so: keep mean/std
(not median/MAD), keep unstratified (no weekday/weekend split), keep the
`_zscore_vs_own_history` column name.

"No change" is scoped to the statistic/stratification/name only -- Group C
still gets the window-mechanism change from #1 (expanding -> rolling 14d),
same as Group A. It is not fully untouched.

Verified per-device (not just global) on 5 representative Group C metrics,
since the global stats alone were what initially hid Group A's real
problem: max_cpu_usage, module_temp_avg_C, bias_current_avg_mA, and
peer_count are all clean (0-1.3% of devices >2x, 0% >10x). `worst_
established_time` is NOT clean -- 4.3% of devices >2x, 0.2% (2 devices)
>10x, worst case 192.78x -- the same shape of problem as Group A, at
smaller scale, for a structural reason: unlike the other Group C metrics
(all physically bounded -- 0-100% CPU, sensor-range temp, hardware-spec
bias current, small-integer peer counts), BGP session established-time is
an unbounded duration, so it can have the same kind of wide-dynamic-range
outlier vulnerability traffic bps has.

Decision: leave `worst_established_time` as-is anyway (no median/MAD, no
rename) -- user's call: this feature isn't important enough in the feature
set to warrant the special treatment. Documented as a known, accepted
imperfection rather than a verified non-issue, which is the honest
distinction from the other 4 metrics checked. The remaining ~28 Group C
metrics not individually re-checked per-device are assumed clean by the
same bounded-quantity reasoning (CPU/memory/temp/power/bias_current/
interface counts are all scale-limited the way traffic bps and established
_time are not) -- not exhaustively verified one by one.

## 7. interface_churn_imbalance -- add a companion interface_churn_total

Verified the existing mechanism first: interface_churn_imbalance = down -
up matches exactly (0.0 diff) at both the raw and _prior_24h/_7d level, and
NaN propagates correctly (down/up prior_24h are always both-null or
both-non-null together, never one-sided). No bug in what exists.

But checked empirically whether "imbalance" alone can mislead, since a
pure net-difference feature always can: of the rows with nonzero down AND
nonzero up counts (real churn happening), 1,020 of 1,816 (56%) land on
imbalance == 0 -- e.g. 4 down-flaps + 4 up-flaps in a bucket produces the
exact same imbalance=0 as a perfectly quiet bucket. Can't be fixed on
imbalance itself (that's what a net/difference measure IS), and the
magnitude is technically already recoverable (interface_link_down_count/
interface_link_up_count and their own _prior_24h/_7d SUM lags already
exist as separate columns) -- but relying on someone to remember to
recombine two other columns is exactly the kind of thing that gets missed
at model-building time, even though it's trivial to add once, here, at the
data stage.

Decision: add `interface_churn_total` = down + up (the "gross churn" to
imbalance's "net churn"), same suffix pattern ("", "_prior_24h",
"_prior_7d") and same source columns as interface_churn_imbalance already
uses. Sits directly next to imbalance so the magnitude signal isn't
implicit.

## 8. _stuck_reading flags -- add SUM lag to the 4 metric-level ones

Checked frequency before deciding whether a lag was worth adding:
bias_current_avg_mA_stuck_reading fires on 4.135% of rows across 123
distinct devices -- not rare, and given a "stuck" reading requires >=30
consecutive identical buckets (5 days) by its own detection design, a
sensor stuck for that long is plausibly an early symptom of the same
degrading-transceiver mechanism that later shows up as a named hardware
ticket (consistent with snow_bgp_breakdown.pptx's finding that named
hardware failures are 83-100% predictable). The other 3
(power_avg_dBm/module_temp_avg_C/temp_value_stuck_reading) are rarer
(0.44-0.96%, 9-30 devices) but still non-trivial.

Decision: add SUM lag (prior_24h/_7d, same windows/min_periods as every
other SUM lag) to all 4 _stuck_reading flags -- "how often has this
sensor been stuck recently" as its own trend signal. The 4 _out_of_range
flags were checked too but are far rarer (<=0.058% of rows, <=4 devices
each) -- left unchanged, diagnostic-only, as originally designed.

## 9. valid_flap_count -- add SUM lag (currently gets none at all)

valid_flap_count (device_flap_count with restart-confirmed flaps zeroed,
build_and_clean_dynatrace_device_data.py:582) gets ZERO secondary-feature
treatment today -- no hits anywhere in build_secondary_features.py. Its
noisier sibling device_flap_count already gets full SUM lag via
add_persistence_lags(). Checked the two aren't near-duplicates first: 541
of 6,628 rows (8.2%) with real flap activity get zeroed in valid_flap_count
specifically because they're confirmed restart-driven -- a real, distinct
signal, not negligible.

Both columns are being kept (not dropping device_flap_count -- that would
be a model-specific feature-selection call this generic pipeline already
explicitly defers downstream, per this same script's own docstring re:
device_flap_count/num_peers_flapped/flap_concentration).

Decision: add SUM lag (prior_24h/_7d, same windows/min_periods as
device_flap_count's existing treatment) to valid_flap_count.

## 10. any_peer_down_samples -- rename to peer_down_sample_count, add SUM lag

Currently gets zero secondary-feature treatment (no hits in
build_secondary_features.py). Checked the source
(query_3_bgp_flap_target.txt:46,51): it's `sum(peer_down_samples)` across
all of a device's peers -- a device-level COUNT of 15-min samples where a
peer was non-Established, summed. Confirmed on real data: zero-inflated
count (median 0, max 80), correlated with device_flap_count (0.894) but
not identical -- it measures cumulative down-TIME exposure (samples spent
non-Established), not transitions, so it's genuinely distinct information,
not a duplicate.

Naming: `any_peer_down_samples` reads like a boolean ("was any peer down")
but is actually a summed count -- misleading given this project's existing
plain-naming convention (same reasoning as the `_mad_zscore_vs_own_history`
rename). Rename to `peer_down_sample_count`, which states what it is
directly.

Decision: rename any_peer_down_samples -> peer_down_sample_count, and add
SUM lag (prior_24h/_7d, same treatment as its siblings from the same
source query -- device_flap_count, num_peers_flapped). Note: the OLD name
is used downstream in the legacy xg_boost repo (trained models' feature
lists, docs) -- this rename only affects data_pipeline's own output column
name; those legacy artifacts are untouched and out of scope here.
