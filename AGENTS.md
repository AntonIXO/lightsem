# LightSem experiment and demo checks

## Selected default — 2026-10-11

- The user explicitly selected the RTMPose BF16 full-fit seed0 checkpoint as
  the new motion baseline and requested commit. `configs/dance/fixed_c.json`
  is the active CLI definition. The previous fixed-C definition is preserved
  as `configs/dance/fixed_c_20261007.json`; the historically pinned
  `configs/dance/baseline.json` remains unchanged for completed provenance.
- This baseline is user-selected, but its health evidence remains diagnostic
  where input-sensitivity evidence is missing. Do not call it scientifically
  qualified merely because it is the active baseline.
- Frozen MuQ is required in the baseline and every option combination. D4-max
  (`--d4max`) and external D6 pacing (`--d6`) default OFF and are independently
  opt-in; D6 must also work without D4. Extra legacy D7/D8 overlays are off.
- `lightsem.cli render` uses this native path; `legacy-render` only reproduces
  historical rule-driven studies. P2 color and restored LUFS brightness stay
  fixed; fixture motion caps remain OFF in simulation.
- The selected weights do not include the later quiet-loss gradient fix or
  full Just Dance retraining. Do not label those as trained or qualified.
- The user closed the completed research campaigns. Preserve original failed
  artifacts/verdicts and parked unresolved questions; do not silently resume
  training. Closure: `docs/research/2026-10-05-native-dynamics/CAMPAIGN_CLOSURE.json`.

The user explicitly requires collapse and input-sensitivity checks before model
comparisons or demo delivery (2026-09-30).

## Current product decision — 2026-09-30

- P2 is the explicitly accepted COLOR baseline: independent Rich96 checkpoint,
  palette history enabled, 300 ms transitions, two spatial color groups. Use
  `configs/light_layers/baseline.json`. C is historical. Revalidate relevant
  earlier hypotheses against P2; preserve their original evidence and verdicts.
- The user accepts the demonstrated generated body as the dance source for the
  current task. Focus on body-to-light transfer and arbitrary N outputs, not
  reconstruction, retraining, or requalification of the dancer as a prerequisite.
- Never substitute autonomous sine waves, beat oscillators, canned sweeps,
  random motion or time-driven choreography for failed dance transfer. The user
  calls that cheating: it changes the task while making the lamps look active.
  This explicitly supersedes the September 12 rhythm-companion authorization.
  Report failed transfer and fix it instead. Trigonometry for spatial coordinate
  conversion is fine; an independent temporal motion source is not.
- Every future demo uses BOTH Shpongle and AURORA — Cure For Me. Each song must
  supply its own body trajectory. Never reuse Shpongle's body as Cure For Me's
  dance. Keep the source body visible in transfer diagnostics.
- Support arbitrary output count and capabilities: e.g. 8 moving beams,
  5 simulated laser beams and 50 garland bulbs. Preserve related gestures and
  phrase timing across the installation; do not claim more independent dance
  information merely because outputs were duplicated.
- Test source dependence: a frozen body must not produce autonomous motion;
  changing a limb must affect its assigned outputs; extra outputs must preserve
  source gesture timing; inspect the actual constrained/emitted output.

- Before a large render batch, inspect the actual generated outputs: finite
  values, motion range and speed, inactive time, color-cue count, actual palette
  changes, palette occupancy, and exact duplicate output streams.
- Test response to at least two distinct music inputs when judging music
  conditioning. Hold weights, non-audio inputs, random seed/noise, mapping and
  rendering settings fixed. Recompute the audio-dependent path for each song.
  Never put a new soundtrack over reused DMX and call it a new generation.
- Separate motion, color and brightness diagnostics. Color differences do not
  establish different motion. A moving fallback controller does not establish
  that a learned color head activated.
- Distinguish representation collapse, output inactivity, saturation, duplicate
  outputs, and intentionally fixed controls. Report the component that failed.
  Do not claim representation collapse from a zero event count alone.
- Say which components depend on music and which use a shared procedural
  vocabulary or unconditioned motion prior. Shared style is not evidence that
  audio changes generated body motion.
- Trace saved-body provenance before calling a generator unconditional.
  `prior_native_flow` means the previous audio-conditioned native flow model;
  the accepted saved body was generated with Shpongle. Reusing that body on a
  new song tests the companion controller, not new song-conditioned dance.
- Nonzero movement and different output hashes do not establish choreography.
  Compare after beat/tempo alignment, include different music with similar BPM,
  and inspect phrase development, motif repetition, holds and visible gestures.
  Check loss of gesture information separately at body, mapper and emitted DMX.
- User review on 2026-09-30 rejects R1's two palette changes as output collapse
  for this task, subsequently selects P2 as the color-layout baseline,
  rejects P4's illogical flicker, and rejects the cross-song movement sameness.
  Do not present these rejected results as qualified product alternatives.
  P2 and P4 have identical moving-head RGB and identical non-RGB commands;
  only auxiliary-fixture colors differ between those two variants.
- Do not present failed/inactive or identical outputs as distinct baseline
  improvements. Keep full evidence, but show one representative of exact visual
  duplicates and identify the other runs as diagnostics.
- A user-requested failure diagnostic may still be rendered, clearly labeled.
  Never silently lower thresholds, inject events, change seeds, or replace a
  frozen checkpoint to make a demo look active.
- Store these checks and findings in the project Vibescience vault
  /root/lightsem-vault, not the unrelated default vault. Memory updates are
  already authorized by the user.

## Native dynamics regression — 2026-10-06

- The user now explicitly requests native training repair and verification;
  this authorizes retraining for this task despite the older transfer-only scope.

- The user rejects the native B/C jitter regression. Never treat lower smoothed
  2D speed-profile error, greater speed variation, or more stops as qualified
  dance improvement without raw 3D temporal-quality checks.
- The original dynamics loss measures poses after a 9-tap, sigma-50-ms filter;
  only ~4.4% of an 8-Hz perturbation survives. That measurement cannot certify
  the unfiltered generated body's velocity, acceleration, jerk, support or root
  stability. Check raw generated motion separately on BOTH FD and JD updates.
- Just Dance supplies masked 2D observations, not metric 3D ground truth.
  Keep observational denoising/masks, but do not leave generated depth, root
  height or foot support unregularized merely because JD lacks 3D labels.
- Compare against paired old D7 / continued-training controls on both songs
  and development groups. Freeze raw-body roughness, high-frequency motion,
  root/support, speed-variation and audio-sensitivity gates before scoring.
  Do not accept a stationary or uniformly slowed model as a jitter fix.
- Inspect native outputs BEFORE interpolation, DMX or rendering. Keep raw and
  filtered measurements distinct. Posthoc smoothing/retiming/recomposition
  does not establish that training was repaired. Do not stack D4/D6 on a
  failed native generator and call the generator fixed.
- Real-fixture speed/acceleration caps stay OFF in the current simulation;
  never restore them to conceal bad generated motion. P2 color and restored
  historical brightness remain fixed, with no automatic baseline promotion.
- Evidence: /mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/jitter_audit_v1/ROOT_CAUSE.json

- Verified repair evidence (2026-10-06):
  `/mnt/lightsem-data/experiments/lightsem/native_dynamics_jd_20261005_v1/repair_refine_v2/reports/SELECTION.json`.
  Both seeds pass all original raw-stability/activity/development gates;
  this is recipe-specific evidence, not permission to skip checks in future.
  Inference remains unfiltered native30Hz, with fixture speed limits off.

## Comparison graph readability — 2026-10-06

- Broken-model outliers must not make accepted/reference raw traces unreadable.
  Preserve all-three diagnostics, but also show a direct reference/candidate
  overlay with the same pair-specific scale and a signed difference plot.
  Include every raw peak and disclose the time/scale scope; do not clip or
  smooth away defects. Recompute plotted data from each role's own saved body.
- The repaired C and historical D7 outputs are distinct but strongly correlated.
  Passing jitter/stability gates does not prove that learned musical pacing
  improved. D7 initialization, frozen early blocks and the JD stability prior
  must be disclosed when interpreting resemblance. Historical D7 and current C
  have different noise arrays; their comparison is descriptive, not paired.
- Evidence: native_dynamics_jd_20261005_v1/repair_review_v1/review_3d_v1/
  revisions/direct_comparison_v1/SIMILARITY_AUDIT.json.

## Slow-target coverage and learning gradients — 2026-10-06

- Do not claim the native dynamics recipe trains on all local corpora. Its
  admitted inputs are FineDance TRAIN and the bounded 72-song Just Dance
  TRAIN cohort. More cached videos/poses do not establish usable paired data.
  Preserve existing development partitions, duplicate groups and quarantines.
- Audit actual slow-target frequency, regional mask coverage and censored
  durations before increasing a slow-motion objective. The current strict
  Just Dance whole-quiet proxy covers only ~0.63% of valid frames; this is
  not a count of every true slow gesture or rest in the underlying videos.
- Never clamp a sigmoid probability before quiet-target BCE: sufficiently
  fast predictions then receive zero slowdown gradient. Use stable logit-space
  BCE and verify backpropagation through a genuinely moving trajectory, plus
  the missing-observation case. Do not infer overall zero learning: the other
  speed objectives still have gradients.
- The gradient correction is code-tested only (15 relevant tests pass).
  Displayed repaired-C checkpoints retain their original training recipe;
  new musical effects and raw3D stability require a separately trained and
  verified comparison. Do not claim the current demos include this fix.
- Evidence: native_dynamics_jd_20261005_v1/speed_training_audit_v1/AUDIT.json.

## Merged repository operations — 2026-10-07

`CLAUDE.md` was merged here and removed. This file is the single repository
instruction source. The current native CLI baseline and reusable procedure are
documented in the `$native-dance-light-pipeline` skill at
`/root/.codex/skills/native-dance-light-pipeline/`.

Use the interpreter wrapper that matches the operation:

```bash
bash run.sh <script.py> [args]          # non-Torch audio/renderer work
bash run_torch.sh <script.py> [args]    # Torch/native-dance work
```

Use `uv` for dependency installation; never use system `pip`. Do not invoke a
venv binary directly when the wrapper is available. Use the existing GPU/data
locks and the sequential curator for GPU jobs. Keep `/mnt/lightsem-data` above
the 20 GiB free-space floor. External checkpoints, feature caches, full videos,
and raw waveforms stay on `/mnt`; code, small manifests, receipts, and reports
belong in the repository or the experiment root.

## Artifact discipline

- Do not generate large numbers of redundant JSON audits or reports. Write one
  canonical receipt per stage and update the relevant manifest or delivery
  index with its hash. Create an additional report only when it represents a
  distinct protocol, correction, or independent audit; do not duplicate the
  same facts under new filenames.
- Keep large media, caches, checkpoints, and experiment payloads outside Git.
  Commit code, tests, small manifests, configuration, and concise provenance
  indexes; link to external artifact roots instead of copying them into the
  repository.

The active architecture is:

```text
original song -> frozen MuQ/HTDemucs/BeatThis/All-In-One features
              -> native C planner/flow -> 30 Hz 22-joint body
              -> fixed mapper 01 -> 40 Hz requested/decoded DMX
              -> P2 RGB + restored LUFS brightness -> lit3D video/chart
```

The descriptor/body seam and each body-to-light stage require their own tests.
The old rule-driven path remains available only as `legacy-render`; it must not
be silently selected for native-C work.

## Active full-data branch — 2026-10-07

The full-data admission audit is a separate, unqualified branch. It admits 147
duplicate-resolved Just Dance TRAIN groups, 24 protected JD validation groups,
and 109 FineDance TRAIN groups with finite per-track features. It preserves
coach identity, masks, duplicate groups, showcase quarantine, and the
distinction between global-record coverage and usable masked windows.

The branch's two 60,000-update joint runs finished with finite losses, but seed
0's checkpoint was lost while preparing a symlinked demo copy. Exact recovery
was stopped; seed 1 demos are diagnostics only. Do not promote or call this
branch a trained baseline. Keep the current fixed C/HF16 baseline unchanged.
The status record is `docs/research/2026-10-07-native-full-data/STATUS.md`.

## Durable failure lessons

- Never let broken-role extrema define the y-axis for an accepted/reference
  chart; pair scopes and signed differences must be explicit.
- Filtered 2D dynamics cannot certify raw native 3D quality. Audit raw jerk,
  root, support, and high-frequency residual before interpolation or DMX.
- Stable logit-space BCE is required for quiet-target gradients. The selected
  checkpoint predates this correction, so do not describe it as trained in C.
- Support/contact confidence is not a whole-body pause. Keep regional effort,
  contact, and rest separate.
- Posthoc D4/D6 can create a useful human-check video without repairing native
  generation or proving musical superiority. Preserve shifted, shuffled,
  donor, and loudness controls.
- Do not reuse Shpongle bodies, DMX, or soundtracks for Cure For Me. A new
  soundtrack on reused universes is not a new generated result.
- Different hashes, active lamps, more stops, broader speed variation, or RGB
  changes do not establish choreography. Compare beat-aligned gesture timing,
  phrase development, recurrence, holds, and emitted-light retention.
- Never overwrite sealed receipts. New code, cache, checkpoint, seed, or source
  hashes require a new artifact root and a new protocol identity.
- Disk cleanup must preserve caches needed by the next authorized pipeline run.
  Classify active working caches separately from archival or superseded caches
  before deleting anything; dataset manifests that point at a cache are evidence
  of an active dependency, not permission to remove the cache. Provenance and
  receipt requirements do not imply creating a new JSON file for every cleanup.
  Do not start regeneration or recovery after an accidental deletion unless the
  user explicitly requests it.

## Autonomous training evaluation — 2026-10-08

- The reusable skill is `skills/autonomous-training-evaluation/`. Its protocol
  and metric references define the machine-only audit contract.
- Run CPU checks through `bash run.sh scripts/evaluate_training.py`; use a new
  report path and the manifest schema in `docs/training_health_manifest.schema.json`.
  The audit reads raw native body/stream arrays and refuses to overwrite an
  existing report. Generate a candidate report and a matched baseline report,
  then pass `--compare-baseline` for the paired gate comparison.
- Required checks are finite values and provenance, raw unfiltered motion
  (speed/acceleration/jerk), representation/sample diversity, output inactivity,
  saturation, exact duplicate streams, explicit RGB and brightness diagnostics,
  and at least two independently recomputed song inputs when judging audio
  conditioning. Missing evidence is `unqualified`; zero events are not a
  representation-collapse verdict.
- The single-run statuses are `collapsed`, `failed_safety`, `unqualified`, and
  `diagnostic_only`. A comparison may report qualified improvement or regression
  only from explicit paired gate evidence. Passing the health audit never proves
  choreography, music fit, or artistic/product acceptance.
- Preserve source-body, soundtrack, checkpoint, noise, mapper, renderer, split,
  and code hashes. Keep Shpongle and AURORA — Cure For Me bodies and emitted DMX
  independent. Never substitute oscillators, fallback motion, reused bodies, or
  post-hoc smoothing/D4/D6 for a failed native generator. Keep pair-scoped plots
  with signed candidate-minus-reference differences and all raw peaks.

## Mandatory generation safety gate — 2026-10-11

- Every native generation, including stereo-adapter candidates and every
  review/demo export, must be audited from the saved raw 30 Hz body before
  mapper, interpolation, DMX, or video. Measure speed, acceleration, jerk,
  root/support, finite values, inactive frames, and exact duplicate streams;
  compare the candidate against the selected baseline on the same song and
  clock.
- A candidate with a raw speed/jerk explosion, constant shaking, teleporting,
  nonfinite values, or an unsafe ratio to the selected baseline is rejected
  and quarantined. Do not render it, map it, lower the threshold, smooth it,
  retime it, or hide it with fixture caps. A generation receipt must record
  `rejected_raw_safety` and the measured ratios.
- The check is mandatory even when the change is described as an audio,
  stereo, color, mapper, or presentation experiment. Passing a video smoke or
  decoded DMX audit cannot override a failed raw-body safety gate.
- Use the native-dance-light-pipeline procedure and the autonomous-training
  evaluation schema for every generation. Preserve the failed raw body and
  its provenance in a new artifact root; never present it as a demo.
