# Native full-data expansion status

The admission and feature stages completed: 147 duplicate-resolved Just Dance TRAIN groups, 24 protected JD validation groups, and 109 FineDance TRAIN groups. All 171 JD tracks have finite native features. The per-track MuQ cache reused 95 verified historical native streams and GPU-extracted 76 new streams; the merged receipt is `native_dynamics_full_20261007/features/COMPLETE.json`.

Two sequential 60,000-update repaired native joint runs completed. Seed 0 was accidentally removed while preparing a symlinked demo copy; its receipts and evaluation remain preserved. At the user’s explicit request, the exact seed-0 recipe was retrained in `native_dynamics_full_20261007_recovery_seed0_v2`; seed 1 remains in the original full-data root. No third seed was run.

Complete diagnostic demos for both seeds and both songs are in [native_full_complete_demo_20261007](</mnt/lightsem-data/experiments/lightsem/native_full_complete_demo_20261007/README.md>). They use each song’s own generated body, original-speed audio, actual mapped DMX, native 30 Hz motion, and the 90–150 second review window. They are not automatically promoted to the product baseline; paired raw-motion, source-sensitivity, emitted-DMX, and cross-song gates still govern qualification.

The GPU batch-size observation remains recorded: BF16 `fd_batch=16`, `jd_batch=8` used about 2.1 GiB of 12.2 GiB VRAM at roughly 76% utilization. Do not alter this closed recipe; a larger-batch throughput pilot requires a new protocol and seed-matched verification.

The read-only Just Dance admission/extraction audit is recorded in
`JUSTDANCE_ADMISSION_AUDIT_20261007.md` and the versioned artifact root
`/mnt/lightsem-data/experiments/lightsem/native_admission_audit_20261007_v1/`.
It preserves all Stage11 and existing held-out groups, separates masked-window
eligibility from the strict whole-record gate, and does not promote a training
baseline.

The two-manifest admission package is in
`/mnt/lightsem-data/experiments/lightsem/native_admission_package_20261007_v2/`.
Strict TRAIN is 153 canonical groups; the maximum masked-window TRAIN receipt
is 290 groups and remains explicitly unqualified. TEST payloads were not opened
while building this package. Thirteen MuQ streams needed by the maximum
manifest are storage-blocked until extraction can preserve the 20 GiB floor.

Storage maintenance: the sealed old visual-next example tree was moved from
`/root/lightsem/runs/visual_next_20260929` to
`/mnt/lightsem-data/archive/lightsem-runs/visual_next_20260929`. The archive
contains 4,929 files (4,750,872,100 bytes); its path-list receipt is beside the
archive. No native baseline, full-data receipt, or failed campaign artifact was
deleted.

On 2026-10-11 the user explicitly selected the newer RTMPose BF16 full-fit
seed0 checkpoint as the active baseline. The active definition is
`configs/dance/fixed_c.json`; the 2026-10-07 fixed-C definition is preserved as
`configs/dance/fixed_c_20261007.json`. This is a user baseline selection, not a
claim that the missing input-sensitivity gate has passed.
