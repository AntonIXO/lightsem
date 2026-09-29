# Shpongle: baseline06 A/B

**Accepted baseline: C — history enabled and300ms transitions**, selected by the
user after review. The player opens C by default. Original A remains the control.

[Open the synchronized switching player](http://127.0.0.1:8766).
All videos cover the same277.3s at1920×1080/30fps AV1, at most50MB per MP4,
with source-FLAC-derived AAC128k and identical decoded audio across arms.
All four are complete and independently verified: `MANIFEST.json`,
`PROGRESS.json` and `INDEPENDENT_DELIVERY_AUDIT.json`. Actual MP4 sizes are43.03–43.59MB.
The original selected06 media remains untouched.

| Arm | Palette history | Color change | Full video |
| --- | --- | --- | --- |
| A | On | Hard, original06 behavior | [Play A](/root/lightsem/experiments/2026-09-29-baseline/outputs/shpongle_baseline_ab_v2/media/A/A_full_1080p_av1.mp4) |
| B | Off | Hard | [Play B](/root/lightsem/experiments/2026-09-29-baseline/outputs/shpongle_baseline_ab_v2/media/B/B_full_1080p_av1.mp4) |
| C | On | Smooth300ms | [Play C](/root/lightsem/experiments/2026-09-29-baseline/outputs/shpongle_baseline_ab_v2/media/C/C_full_1080p_av1.mp4) |
| D | Off | Smooth300ms | [Play D](/root/lightsem/experiments/2026-09-29-baseline/outputs/shpongle_baseline_ab_v2/media/D/D_full_1080p_av1.mp4) |

**History:** compare A↔B with hard changes, then C↔D with the same fades.
**Transitions:** compare A↔C with history, or B↔D without history.
Compare the same timestamps. Especially useful:1:58–2:20,2:57–3:17,
and4:21–4:37. The intro also differs between history settings. Avoid deciding
history from A↔D or B↔C because both factors change together.

Both history conditions keep identical color-cue opportunities and accents.
Fades start at the color change, last0.3s and retarget continuously if another
change arrives. Non-RGB DMX, movement, dimmers and rendering geometry stay exact.
History means previous generated palette only; audio context is retained.
No palette model/encoder was rerun: saved all-state choices were replayed, with
exact own-state reconstruction required first.

Rate music fit, color continuity, and whether the fade blunts a desired sharp
musical change. Pick a favorite or report ties; a timestamped note is more useful
than inventing a score. The player saves new feedback locally and can exportJSON.
This review is not blinded. C is now the baseline; the other arms remain available
for comparison. See `../../records/USER_BASELINE_C_SELECTION.json`.

Engineering descriptions in `PREPARATION.json` do not establish quality:
palette IDs differ for178.4s between history settings, while maximum per-frame
RGB byte steps fall219→28 with history and146→18 without history when fades are
applied. These are channel-control measurements, not perceived color distance.
