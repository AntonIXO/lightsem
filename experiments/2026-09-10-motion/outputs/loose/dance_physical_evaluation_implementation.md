# Physical revision: evaluation implementation

## Status and scope

Implemented `outputs/eval_music_physical_revision.py` and
`tests/test_music_physical_evaluation.py`.

**Engineering only. No new VAL scores exist from this worker.**
This worker did not run GPU work, open real model checkpoints, open VAL motion/audio
arrays, change training/design/source snapshots, access TEST, or access a vault.
Only the evaluator, its new test file, and this report were authored.
Existing protocol and review metadata were inspected. No experiment freeze or
real inference was executed by this worker.

CPU validation command:

```text
CUDA_VISIBLE_DEVICES="" bash run_torch.sh -m unittest discover -s tests -p test_music_physical_evaluation.py -v
```

Result: **21 tests passed**, exit 0. The final run took 1.02 seconds including
interpreter startup. Tests use synthetic inputs and tiny CPU models from the
frozen training classes. They do not establish CUDA performance, real checkpoint
load success, trained-model continuity, or physical/music quality.

## Identity and integrity

- Pins the exact physical protocol SHA256
  `c5ea616d907510b92f06383abc7b4e7cb7d6862bc367b5769e838aebf0ed6405`.
- Verifies every frozen training source and every protocol input. It rejects
  mixed live/frozen `lightsem` imports.
- Loads **MusicRevisionModel**, with frozen MusicConfig, arm and physical weights.
  It does not call the old music context or old model loader.
- Requires **both** completed 30000-step training reports before any initial or
  final evaluation. It verifies initial/final file hashes, complete training
  identity, exact step, model config, normalizers and state buffer normalizers.
  All model state tensors must be finite.
- Confirms the initial model content hashes match between training arms.
- Uses immutable atomic JSON envelopes and read-only NPZ artifacts. JSON carries
  its own canonical-payload hash. NPZ hashes are bound into the row receipts.
  Resume verifies identity, exact window/control spec and array hash.
  An orphan NPZ after interruption is accepted only if recomputation matches
  every array and dtype exactly. Completed rows are not regenerated.
- Freezes an evaluator copy and evaluator SHA before results. Source edits after
  this freeze fail identity checks. Do not overwrite the evaluation protocol.
- Real consumed VAL audio/motion files are hash-checked and must have exact aligned
  frame counts and finite contents. No TEST loader exists.
- A nonfinite generated draw writes a permanent failure receipt and blocks that
  stage on restart. It never drops or replaces an individual failed draw.
- Completed training evidence combines frozen per-update `backward_checked`
  enforcement with finite saved logs and final-step receipts. Sparse logs alone
  are **not** described as retained per-update gradient tensors.

## Cohorts, draws and metrics

### Primary

Uses the exact original **119 VAL windows**, unchanged donor/shift starts and
six draw seeds. Each condition receives the same six noise fields and 32 Euler
steps. Runs both arms at initial and final only. FP32 neural/Euler/FK; no autocast
or TF32. All generated draws, FK outputs, summaries and scores must be finite.

Reuses the frozen native physical metric and fixed TRAIN-standardized18-channel
motion summary helpers. Computes correct/donor/shift unbiased energy scores.
Sliding uses **generated contact labels**, not reference labels. It also retains
source and generated sliding, coverage, kinetic energy, travel, penetration,
high-frequency fraction and frozen-distal fraction separately.

Reduction: average draws inside each window, equal windows per recording, equal
recordings per canonical song, then equal canonical songs. Physical ratios are
ratios of these canonical-weighted means, as in the prior frozen gate helper;
they are not means of per-window ratios. Excess means
`max(0, generated canonical mean - source canonical mean)`.

### Supplemental short VAL

Freezes metadata for all3 full short recordings before inference. Uses deterministic
next canonical-distinct VAL donor of sufficient length and a hashed contiguous
start. All six paired draws and 32 steps are retained. This cohort is never merged
into the original119.

A same-record contiguous shift of a **full aligned recording** has no available
support. The protocol explicitly records `shift_start: null` with an unsupported
reason. It does not wrap, pad, repeat, shorten the recording or invent a shift
score. Full physical metrics and saved motion include every real frame. The
unchanged half-second ES helper omits only the final incomplete <15-frame bin;
the receipt reports ES-scored and physical-scored frame counts separately.

Short outputs include all correct/donor native draws and world joints, plus the
reference. All4 initial/final cells are required by the numerical report.

### Full12 continuity and full Shpongle

`full` evaluates complete recordings underlying all12 frozen clips. Both final
arms use seed7001. It generates direct and chunked native motion with identical
full noise, FP32 and32Euler, and reports the maximum absolute native difference.
No per-chunk root integration occurs. World joints/root are computed once from
the full generated sequence and saved. Reviews must **slice saved world joints**,
not reintegrate sliced native velocity. Full reference joints, contacts, root,
times, quality, generated root displacement and maximum excursion are saved.

`shpongle` generates all6 fixed full-track draws, one restartable item per draw,
for each final arm. It saves the same generated arrays and drift diagnostics.
There is no new source motion or conditioning score for this external song.
Shpongle is not the declared12-record continuity gate.

The metadata freeze also pins existing review-array references for the prior
incumbent, prior native flow, and prior full Shpongle delivery. It does not reload
or regenerate these models. A renderer must verify those report/array hashes
before comparison.

## Exact numerical gates

Physical final is compared with the **new masked_native final** baseline, not
with the historical native flow:

| Gate | Requirement |
|---|---|
| Conditioning magnitude | Correct ES improvement >=5% versus **both** donor and shift |
| Conditioning uncertainty | Both paired canonical-song bootstrap95% lower bounds >0 |
| Bootstrap | 10000 song draws, NumPy seed20260910, percentile interval |
| Kinetic ratio | [.5,2] inclusive |
| Root travel ratio | [.25,2] inclusive |
| Contact coverage ratio | [.5,1.5] inclusive |
| Sliding | Absolute excess <=.20m/s **and** <=.8*masked_native excess |
| Floor | p95 penetration excess <=.05m |
| High frequency | Relative-joint >6Hz power-fraction excess <=.05 |
| Continuity | Full12 FP32 maximum absolute error <=5e-5; both final arms checked |

The literal sliding inequality accepts zero/zero, but the report warns that this
is not evidence of a nonzero relative reduction. No threshold is changed.

The final report also includes finite evidence and frozen short/hold coverage.
Visual and human music-fit gates remain **null**, not passed. The script always
leaves pipeline qualification false, selection null, seed1 unauthorized, and
TEST false. It cannot promote a default or declare the full pipeline accepted.

## Parent execution sequence

From `/root/lightsem`, freeze metadata **before any result**:

```text
bash run_torch.sh outputs/eval_music_physical_revision.py freeze
```

Default output:
`/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/music_physical_revision_01/evaluation`

Then use the saved `evaluation/evaluator.py` (or the byte-identical outputs file).
A CPU-only metadata/source verification stage is available:

```text
bash run_torch.sh outputs/eval_music_physical_revision.py verify
```

After both training runs finish, schedule independent segments, for example:

```text
bash run_torch.sh -u outputs/eval_music_physical_revision.py primary --arm masked_native --checkpoint initial --device cuda --budget 10
```

`--budget` consumes the next missing windows/recordings/draws in frozen order.
Reissue the same command until the cohort report exists. There are no index,
window, seed, checkpoint-step, weight, draw-count, solver-step or threshold
selection arguments. No partial cohort aggregate or gates are written.

Required complete matrix:

- `primary`: masked_native and physical, initial and final (4 cells).
- `short`: masked_native and physical, initial and final (4 cells).
- `full`: masked_native and physical, final (2 cells,12 records each).
- `shpongle`: masked_native and physical, final (2 cells,6 full draws each).
- `report`: CPU report after all required cells complete.

The CUDA path uses the same frozen `gpu_lock(NATIVE)` as training. The evaluator
also takes a nonblocking exclusive output lock. Parent must still serialize
renderer and other GPU jobs. Device defaults to CPU; CUDA requires explicit
`--device cuda`. Do not mix RNG devices between cells.

```text
bash run_torch.sh outputs/eval_music_physical_revision.py report
```

## Still required

1. Parent performs the real freeze, verifies it and schedules inference after
   training. No real checkpoint/CUDA execution was preflighted by this worker.
2. Parent verifies GPU process, log and memory externally during real work.
3. Build matched rendering from saved arrays and frozen incumbent references.
   Rendering is intentionally outside this evaluator and was not implemented.
4. Review all12 synced clips against incumbent and both new arms. Review full
   Shpongle. Retain failures. Keep sampled visual evidence distinct from human
   music-fit acceptance and independent lighting-pipeline acceptance.
5. Record measurements and conclusions in the authorized vault through a
   serialized parent-controlled session. This evaluator has no vault access.
