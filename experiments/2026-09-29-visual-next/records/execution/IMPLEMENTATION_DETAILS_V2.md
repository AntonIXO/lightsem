# Prospective execution details V2

These details precede full H2/H3 extraction, PCA fitting and all new student fits.
They retain the four methods and primary scientific predictions. The user has
made bulk manual source annotation optional; automated exploratory comparisons
and Shpongle deliveries proceed under USER_SCOPE_CORRECTION_20260929.md.

## Resource amendment

Increase the root-filesystem study artifact cap from3GiB to4GiB before the full
matrix. This accommodates the already delivered optional annotation media
(about0.83GiB), both frozen teacher feature banks, all seven projected targets,
the required model checkpoints and compact review videos. The previous3GiB cap
was an engineering allocation, not an accuracy threshold. Keep at least1GiB
free on both filesystems, make no new scientific cache on /mnt, retain the8GiB
RAM and10GiB reserved VRAM limits, and keep one experiment GPU process at a time.
Stage time budgets are not changed by this amendment. The16-frame shared smoke
retains its original frozen3GiB cap. A later corpus manifest must explicitly bind
the4GiB value and its measured runtime allocation.

## Source and feature support

Full extraction uses the original91TRAIN and31historicalDEV families, without
replacement, using each existing64s source window. The12TRAIN annotation/probe
families remain the original selected subset. All source/pose/capture hashes
and canonical split identities must be authenticated before predictions.

Keep common visual auxiliary support across all seven H3 arms: current plus the
previous five10Hz frames must be source-valid. Static controls use that same
mask, so temporal arms do not gain or lose observations relative to controls.
This affects auxiliary support only; main event/palette labels and their masks
remain the original observer targets. Invalid feature targets are zero storage
placeholders with explicit false validity, never supervised examples.

R1 uses retained-background color grids. The DINO patch embeddings use the whole
RGB frame as specified in the original whole-frame resize recipe; people and UI
may therefore be represented in those semantic channels. This is an explicit
limitation, not a claim that DINO only describes LIGHT. The original SAM2 masks
are common across all H3 arms and continue to define R1's explicit color support.
Cache teacher outputs as float16, record conversion roundoff, and compute PCA in
the documented arithmetic below. Normalize the72histogram entries of R0's old
float16 source representation exactly as the original SourceStore loader does.

## Deterministic TRAIN PCA

Every arm keeps96 deployed auxiliary coordinates. R0 uses exact96x96 covariance.
Higher-dimensional arms use a fixed128-column covariance range sketch, seed
20260929, with two extra power iterations. There is no solver/rank/seed selection
from outcomes. Streaming batches contain256rows; BLAS uses two threads. Means,
accumulations and small eigendecompositions use float64, large products use
float32; component signs are fixed by the largest absolute coefficient.

Only91TRAIN families fit means, bases and coordinate standardizers. Use global
population variance and the existing1e-3 standard-deviation floor for numerical
stability, with raw variance/effective-rank diagnostics reported before that floor.
Freeze all seven TRAIN transforms before applying any to DEV. Store projected
targets as float16 with explicit roundoff receipts; training reads float32.

Representation retention and controlled recoloring/layout probes remain required
before deciding whether the original H3 fit gate passed. They are instrument
checks, not independent LIGHT-quality evidence. An approximate PCA must earn the
same retention check; solver approximation is not a reason to waive it. Do not
claim post-standardization rank alone proves absence of collapse.

## Matched student fitting

All seven H3 arms use the original independent three-encoder student with
1284410trainable parameters, same91TRAIN groups,94D audio bus,64s context and
central8s supervision,6000updates, batch4, BF16, AdamW lr3e-4/weight-decay.01,
300step warmup, gradient clip1, auxiliary weight.1 and state dropout.5.

Use paired seeds0/1/2, fixed data-draw and initialization hashes, and identical
original main targets/audio scalers/palette vocabulary. H1 has no independent
pass, so do not turn on its label veto for H3. Override only the normalized visual
auxiliary target and its common validity mask. No DEV access/checkpoint selection
during fitting. Save the final6000-step checkpoint and all training logs; retain
paired untrained initializations for the prescribed controls. Additional cut/mask
and combined demonstration fits need their own prospective target bindings.

External GPU interference is a resource interruption, not a model failure or
scientific refutation. Retain completed results and resume only uncomputed units
with unchanged methods and recorded resource accounting. Never pick a retry based
on an observed score or discard an unfavorable completed family/seed.
