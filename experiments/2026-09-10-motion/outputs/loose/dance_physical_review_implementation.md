# Physical revision: matched saved-array visual review

## Status

Added `outputs/review_music_physical_revision.py`,
`tests/test_music_physical_review.py`, and the optional full-world review seam in
`lightsem/experiments/motion_study/lighting_map.py`.

CPU synthetic validation: **17 tests passed**. Tests compare default DMX and every
trace array byte-for-byte with the immutable pre-review training snapshot, for
both direct-audio settings. They verify world-FK bypass, common root orientation,
full-world crop preservation, timeline indices, optical channel isolation,
receipt integrity, evaluator identity checks, and review coverage requirements.
No GPU, new inference, checkpoints, VAL scores, TEST or vault was accessed by this
worker. Old saved NPZ keys/shapes and existing metadata were inspected. This is
implementation evidence, not successful real map/render or visual acceptance.

## Frozen recipe

Default output is
`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/visual_review`.

The freeze pins the visual protocol, script, **separate full native renderer and
mapper source snapshot**, old saved arrays/reports, waveforms, evaluator metadata,
stage02 and optics designs, font, and FFmpeg/FFprobe binaries. The freeze refuses
to run if real evaluator row receipts already exist. Future trained arrays are
not available at freeze. Their exact permitted paths, evaluator identity,
cohort order and draw seeds are pinned by the evaluator protocol. Their hashes
are verified from evaluator envelopes and then bound into each map receipt.

No training, evaluation, optics or historical frozen source is edited. No default
is promoted. The optional seam is the only production edit. `compose` without
`world_joints_override` keeps its original behavior. `compose_world_joints` uses
the same hip/spine `infer_root_rotation` for every world-body arm, including hue.
It never runs FK or integrates root velocity.

### Fixed comparison

All12 exact frozen clips, in their original order, then full Shpongle.
Five columns, always with source-specific labels:

1. Reference native motion.
2. Historical descriptor-to-2D incumbent.
3. Prior native flow.
4. New masked_native, scheduled final30000, full draw7001.
5. New physical, scheduled final30000, full draw7001.

Three rows: body, `original_legacy`, `closer_zero_angular_flux`.
The optical rows copy `legacy_native_legacy` and `closer_zero_angular_flux`
from `optics_full_controls_01/design.json`. Compare arms **within each row**.
Camera, zoom, cone-edge and flux changes between rows are presentation changes.
Each stage panel is rendered at its full640x360 resolution, without resizing.
The montage is3200x1080. Body panels contain fixed front/oblique world projections
at60pixels/m; no per-arm centering or fitted zoom. Large drift can leave the panel;
an off-panel anchor count and world root coordinates disclose this.

Stage02 mapping is identical across comparable world arms: articulated_stage,
anchors7.7/3.2/8, lean0, full-track offline_projection, speed90 and acceleration360.
**These are software simulation settings, not hardware-safe limits.**

### Important historical limitations, discovered from actual old NPZ schemas

- Fixed12 incumbent arrays contain only `times`, `pose8`, `pose32` at40Hz.
  They do **not** contain canonical3D joints or native139. The body panel shows
  both frontal2D decodes, at the historical fixed.45display-units/torso-unit scale.
  Both lighting panels say **UNAVAILABLE**, not a guessed3D lift. This means the
  full requested incumbent-lighting comparison cannot honestly be supplied from
  the frozen assets. It remains an explicit unmet comparison.
- Prior fixed12 native arrays contain only `features`. The map stage computes FK
  **once on the entire saved recording**, then saves `full_world.npz`. Rendering
  slices that saved world array. It never reintegrates native crops.
- Old Shpongle contains `flow0`...`flow5`, regression, pose8 and pose32. Prior flow0
  is the first archived draw. New Shpongle uses the predefined first draw7001.
  All6 evaluator draws in **both** new arms must exist and hash-verify before
  mapping Shpongle. They remain in the evaluator; no best draw is selected.
- Shpongle has no reference motion. Reference body and lights are labelled absent.

### Alignment and audio

The12 clips retain original integer native-frame starts. Each **entire** recording
is mapped first, with its complete loudness curve. Only then are emitted40Hz DMX
frames selected on the native30Hz movie timeline. Full recorded world joints are
sliced without changing their coordinates. No clip-wise loudness normalization,
root integration, root recentering, phase fitting or time warping is used.

Fixed12 loudness uses frozen EBU extraction on each full waveform: momentary400ms,
short-term3s, metadata timestamps at block end PTS+100ms. Shpongle uses its already
frozen full-track loudness NPZ. All world arms must have identical activity arrays.
Audio mux offsets are original0 + clip_start/30, common to all columns. Only the
native final-frame endpoint duration mismatch may be audio-padded; no motion is
padded. FFprobe verifies all decoded video frames,30fps,3200x1080 and audio duration.
This is a timing/count audit, not a perceptual synchronization judgment.

**Direct audio dynamics are enabled and identical in policy across all arms.**
Brightness also retains the mapper's original body-region effects. Movies are
not evidence of music conditioning. The optional disabled-direct-audio rendered
comparison was **not implemented** in this revision. The separate evaluator's
wrong-audio numerical tests, with direct lighting dynamics disabled, remain the
conditioning instrument.

## Commands

From `/root/lightsem`, before any real evaluator output:

```text
CUDA_VISIBLE_DEVICES='' bash run.sh outputs/review_music_physical_revision.py freeze
```

Use the frozen `visual_review/reviewer.py` afterward. Below `$V` denotes the full
visual_review path printed above (shell variable shown only for readability).

```text
CUDA_VISIBLE_DEVICES='' bash run.sh $V/reviewer.py verify
CUDA_VISIBLE_DEVICES='' bash run_torch.sh -u $V/reviewer.py map --budget 1
bash run.sh -u $V/reviewer.py render --budget 1
```

`map` requires saved final full evaluator rows for both arms, but never loads a
checkpoint or runs inference. Missing evaluator prerequisites fail before making
a partial map directory. `render` takes the same exclusive `gpu_lock(NATIVE)` as
training/evaluation. Parent must schedule it only when the GPU is free. One EGL
context is used; all optical-row uniforms are reset from that row's fixed config.
Encoding uses CPU libx264, not a second GPU workload.

Budgets consume the **next missing item in frozen order**. No key/window/draw or
checkpoint-selection option exists. Completed stages hash-verify rather than
rerun. All writes are create-only. A genuinely interrupted stage leaves its
partial directory and blocks overwrite; parent must record the failure and
arrange an explicit new revision rather than silently erase evidence.

Each render writes a full montage, native-width chronological JPEG samples at
every integer second plus the exact final frame, and an authenticated receipt.
No thumbnail is substituted for a stage-pixel instrument. Pixel clipping/optical
quality scores are not claimed by this runner; use the separate optics study.

### Review attestations

Review all samples and preferably the full synchronized movie. Create a new
observer JSON file, then consume the next missing item:

```text
CUDA_VISIBLE_DEVICES='' bash run.sh $V/reviewer.py review --observation /path/new_observation.json
```

Example schema (frame list must include every frozen sample, including599):

```json
{
  "item_id": "holds_083_1200",
  "reviewer": "name and instrument",
  "coverage": "all_frozen_samples",
  "viewed_sample_frames": [0,30,60,90,120,150,180,210,240,270,300,330,360,390,420,450,480,510,540,570,599],
  "notes": "Concrete observations, failures and limits.",
  "comparisons": {
    "body": "Within-body-row comparison; incumbent2D not physically calibrated.",
    "original_legacy": "Within-row old/new native comparison; incumbent lights unavailable.",
    "closer_zero_angular_flux": "Same within-row comparison under fixed closer optics."
  }
}
```

`coverage` may instead be `full_playback`, but the complete sample list is still
required. The receipt clearly marks coverage as observer attestation, not machine
proof that someone watched. It never fabricates an observation or acceptance.

After all13 movies exist:

```text
CUDA_VISIBLE_DEVICES='' bash run.sh $V/reviewer.py report
```

This final create-only report lists render/review completeness and limitations.
Visual gate and human music-fit acceptance remain null. Pipeline qualification,
seed1 authorization and default promotion remain false. A complete sampled review
is not independent human audiovisual acceptance.

## Remaining validation

Parent must run real mapping and rendering after evaluator output becomes
available. CPU synthetic tests do not establish actual CUDA renderer execution,
large full-track solver success, FFmpeg mux success, or subjective quality.
No real movies or reviewed clips were produced by this worker.

## Executed freeze receipt

The worker completed the real metadata/source/input freeze and CPU-only verify,
both exit0, before evaluator row receipts existed. No evaluator result arrays were
opened. Final test rerun:17/17 passed,1.45s including startup.

- Freeze envelope payload SHA256: `48716fc52de9742622fcefb0fc2f4f2ea1ec4aebff2e84a0f9e9c691f9e3b31c`.
- Frozen reviewer SHA256: `eb8ef4dee19c7dacd40a01e39f4075e2f60e07de28e6483795d20643926568d8`.
- Do not run `freeze` again. Continue with the frozen reviewer.
