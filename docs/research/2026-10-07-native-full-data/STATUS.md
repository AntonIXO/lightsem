# Native full-data expansion status

The repeat admission audit covers 675 catalog recordings and 669 pose caches.
It admits 600 recordings / 994 coach tracks after retaining protected
development groups, quarantines, and the original 80%-coverage 16-second
window gate. There are 272 duplicate-safe Just Dance song groups with paired
waveforms, plus the existing 109 FineDance training groups (128 records).

`FULL_DATA_MANIFEST.json` selects one independently tracked coach per Just
Dance group, choosing the highest audited coverage. The result is 272 JD groups
and 109 FD groups. It is explicitly an admission artifact; no checkpoint has
been trained or promoted from it.

The native speed path remains guarded by raw-motion checks. The stable
logit-space quiet-loss and kinematic tests pass (`4 passed` for the new full
data manifest and quiet-gradient checks); the broader native kinematics suite
also passed in the prior repair verification. The expanded manifest does not
claim that observed Just Dance poses are metric 3D labels.

The current machine has only about 11 GiB free on `/` and 18 GiB on `/mnt`.
Training and extraction are therefore not started: the configured 20 GiB
free-space floor would be violated. Missing-waveform video sources remain
listed for a separately storage-gated audio extraction step. Existing failed
experiments and demos have not been deleted or overwritten. Idle review
servers were stopped to release resources.

Existing two-song engineering demos remain available for review:

- [review index](/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v2/review/index.html)
- [Shpongle schematic](/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v2/media/01/shpongle/schematic.mp4)
- [Cure For Me schematic](/mnt/lightsem-data/experiments/lightsem/body_transfer_20261004_v2/media/01/cure_for_me/schematic.mp4)

They are transfer diagnostics with P2 color and source-body provenance, not
evidence that the expanded training recipe is qualified.
