# Shpongle rating set — September 29, 2026

Local review page: http://127.0.0.1:8765

Source: `/root/shpongle.flac` (the existing project recording; there is no
`/root/.shpongle.flac`). Every video covers the same 277.3 seconds, with 8,319
frames at 1920×1080/30fps, AV1 video and the identical decoded AAC soundtrack.
Existing verified renders are reused without re-encoding. New ablations retain
the same camera, optics, exposure, accepted motion, and 469 protected DMX slots.

For direct playback, [VIDEOS.md](VIDEOS.md) links to ten named, byte-identical
review copies in `videos/`. `MANIFEST.json` retains the original audited source
paths and hashes as a historical record.

| ID | Variant | What to compare |
| --- | --- | --- |
| 01 | Original motion-led lighting | Reference before the rhythm-led controller |
| 02 | Rhythm-led companion | Audio rhythm phrases with saved motion style |
| 03 | Audio-only shared model | No visual auxiliary training |
| 04 | Shared model with six visual descriptors | Earlier visual supervision |
| 05 | **All compatible changes** | Rhythm, motion style, brightness dynamics, rich-model colors and accents, palette history |
| 06 | Independent model with rich visual supervision | Alternative accent/color architecture |
| 07 | Shuffled-audio training control | Same playback track; negative-control training recipe |
| 08 | Learned colors only | Rich-model color overlay; companion brightness |
| 09 | Learned accents only | Rich-model accent overlay; companion colors |
| 10 | Combined with palette history disabled | Same rich model and cues, previous palette replaced by its trained null state |

For music fit and overall preference, compare 01→02 and 02→03/04/05/06/07.
For contributions within the combined setup, compare 02/08/09/05. For palette
history, compare 05/10. Keeping playback time while switching helps compare the
same musical moment. The accent-only change is intentionally sparse: the saved
model emitted one added accent near 263.7s; no events were invented for display.

Rate music fit, deliberate design and overall preference from 1–5, add notes,
then export the JSON. Scores stay in local browser storage until exported.
Existing captions can identify variants, so this is not a blinded experiment.
No reported scientific scores are shown beside the rating controls.

Scope is the recent implemented experiments from the committed work: the five
trained model recipes use their pre-existing seed0 selected checkpoints. Seeds
1/2 are numerical replications, not additional mechanism variants. Older model
families and untrained ideas are not silently substituted into this set. Data
audits, memory repair and archive tests do not themselves generate lighting.

“All compatible changes” is the rich shared-model stack, not an invented
ensemble of mutually alternative models. Variant10 is a deterministic full-track
visual ablation, not a replay of the stochastic 64-second palette-history study.
The CPU own-history replay was required to reproduce the saved combined-demo
palette IDs exactly before generating the no-history variant. No audio encoder
forward, retraining, threshold tuning or model promotion was needed.

`PLAN.json` fixes the set; `PREPARATION.json` authenticates the new streams;
`MANIFEST.json` records verified video paths and hashes as the queue completes.
`PROGRESS.json` is the live queue state. Per-video render and audit records are
under `media/08`, `media/09`, and `media/10`.
