# Next music revision, based on the failed pilot

The present pipeline is implemented and its pilot is complete. It does not earn a
second seed or TEST access. Preserve the incumbent as the usable model baseline;
the native model and its lighting previews remain diagnostic.

The next scientific intervention should address the failures directly:

1. Retain the accepted native representation. None of seven compressed candidates
   passed dynamic fidelity, so another compression sweep is not the next bottleneck.
2. Include every short recording using variable-length valid masks. Exclude padded
   frames from every loss and temporal derivative; do not repeat a short dance to
   make a 32-second target. Report effective song and hold coverage before fitting.
3. Add differentiable forward-kinematic supervision to an explicitly reconstructed
   clean motion estimate. Penalize planted-foot velocity using training reference
   contact labels and score generated feet against their own generated contact
   labels. Keep contact coverage as a separate anti-bypass gate. Validate this loss
   on known sliding and planted-foot counterexamples before any new training.
4. Improve audio conditioning as a distinct ablation: local audio encoding with
   multiscale temporal fusion should be tested against the existing concatenation
   model under identical crop exposure. Preserve donor and contiguous-shift controls.
   A lower marginal distribution score alone cannot support the claim.
5. Record root-velocity bias and its accumulated full-track displacement. The
   current chunk test proves numerical equivalence, not bounded or plausible travel.
   Any root regularizer must preserve real active travel and quiet passages.

Freeze a small seed-0 matrix with the existing native recipe as the matched baseline:
physical loss only, stronger audio conditioning only, and their combination. Commit
the primary, schedule, code, loss weights and all magnitude/quality gates before
results. Do not extend the old 30,000-step runs under their original identities.
This is the next proposed study, not an already executed or validated method.

Keep lighting questions separate. Fixed-role optical calibration passed its ramps.
The new mapper greatly reduces quiet movement, but its active median and p95 fall
slightly below the frozen minima. A future mapper revision can change target gain
or smoothing with the same fixed generator and rerun the full brightness/coherence
ablations; it must retain the original failed result and cannot repair weak music
conditioning by adding direct loudness to the lights.
