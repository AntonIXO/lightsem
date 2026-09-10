# RTX 5070 iteration-speed pilot

The pilot is complete: training was **5.18–5.35× faster**, but no variant passed
all quality requirements. FP32 remains the default. See the
[full results and confidence intervals](research/2026-09-10-speed/RESULTS.md).

The current Night trainer supports explicit runtime settings without changing
model dimensions, targets, data partitions, or final-step checkpoint selection.
FP32 remains the compatibility default. The four-cell pilot checks `base_once`
and `muq_film` with training seeds 0 and 1 before recommending faster settings.
Its output root is `/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910`.
The frozen September 9 campaign supplies read-only inputs and paired baselines.

## Runtime interface

New experiment protocols can set `training.runtime`:

```json
{"precision":"bf16","optimizer":"adamw","compile":"off","prefetch":2}
```

Precision accepts `fp32`, `tf32`, or `bf16`; optimizer accepts `adamw` or
`adamw_fused`; compilation accepts `off` or `default`; prefetch accepts 0 or 2.
If `training.precision` is also supplied it must agree. Settings are applied
after seeding, because legacy seeding disables both matrix and convolution
TF32. Parameters and Adam states remain FP32 under BF16 autocast. A joint
finite-loss/gradient check runs before every optimizer update. These settings
belong in a newly frozen protocol, never an edited historical protocol.

Checkpoint identities include the complete training configuration. Different
runtime settings cannot silently resume an existing run. Prefetch uses a cloned
producer RNG and saves the consumed sampler cursor; unconsumed crops replay on
resume. Compiled computation receives explicitly generated flow inputs, and
checkpoints retain ordinary model keys without compiler-wrapper prefixes.

## Execution

Use `bash run_torch.sh -m lightsem.experiments.iteration_speed` with:

- `benchmark`: compare real-data precision, optimizer, prefetch, compiler, and
  batch candidates. This performs engineering updates only and saves no model
  weights or quality-selection results.
- `freeze`: verify inputs and seed-matched initial weights; freeze the selected
  runtime, source, four-cell matrix, and quality/resource gates.
- `train --arm base_once --seed 0`: run one frozen cell. Repeat in order for
  `muq_film/0`, `base_once/1`, and `muq_film/1`. `--resume` is explicit recovery,
  not a seed retry or permission to alter the protocol.
- `evaluate --arm base_once --seed 0`: evaluate and render one cell; repeat for
  the same four cells. Evaluation uses FP32, six draws, eight Euler steps, all
  nine controls and the fixed 40-song development cohort.
- `summarize`: compare every cell with its matching archived FP32 baseline.

Run only one GPU command at a time. Existing reports are authenticated; the
benchmark and freeze commands refuse to overwrite completed stages. Runtime
profiles for variants outside this pilot remain unconfirmed.

## Acceptance and storage

Every cell must achieve at least 2x complete training speed, and lose no more
than 3% in native energy score, A-head MSE, or macro event AP. Paired song
bootstrap intervals use 10,000 draws and seed 20260910. A 95% degradation
interval crossing 3% is inconclusive, not a pass. Existing passing motion,
conditioning, brightness, composition, and variation gates must remain passing;
newly constant native channels are rejected. Existing musical-control failures
are not reinterpreted as successes by this efficiency study.

Every song/control is computed. Checkpoints, per-song metrics and output hashes
are retained; full generated arrays and videos are retained for eight fixed
audit clips. Other arrays can be regenerated from the frozen code, checkpoint,
input identities and evaluation seeds. Existing caches are referenced without
copying them. Freeze requires a 1.15 GiB artifact allowance above the historical
15 GiB free-space reserve; compiler caches have dedicated task directories.

BF16 inference is separately available through
`generate_groups(..., precision='bf16')`; its outputs are converted to FP32 at
the NumPy boundary. It is not used for the quality pilot or automatically
recommended from training results. Existing FP32 chunk checks remain intact.

## Evidence behind the candidates

- [PyTorch AMP](https://docs.pytorch.org/docs/2.14/amp.html): operator-specific
  precision for linear/convolution computation with FP32 loss reductions.
- [PyTorch AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html):
  fused and foreach implementations; performance must be measured locally.
- [Pinned asynchronous transfers](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html):
  reusable pinned buffers and a separate stream enable copy/compute overlap.
- [8-bit optimizer limitations](https://huggingface.co/docs/bitsandbytes/en/optimizers):
  savings scale with parameter count. This model has only about 30 MiB of Adam
  moments, so the estimated upper saving is about 22 MiB; 8-bit AdamW is deferred.

Benchmark and scientific results are reported separately in the experiment
directory. Synthetic timing and engineering updates alone do not establish
quality retention or eligibility to change defaults.

## Measured engineering screen

Real training data, resident base tensors and actual auxiliary transfers were
used for three 50-update timing windows after warmup. These are complete update
times, not the earlier synthetic forward/backward measurements.

| Configuration | Base once, ms/update | MuQ FiLM, ms/update |
|---|---:|---:|
| Legacy FP32, batch 32 | 144.69 | 162.05 |
| TF32, batch 32 | 62.36 | 79.87 |
| BF16, batch 32 | 33.47 | 50.00 |
| BF16 + prefetch, batch 32 | 33.74 | 40.78 |
| BF16 + prefetch + compile, batch 32 | 25.95 | 29.69 |
| Same compiled profile, batch 64 | 53.37 | 61.76 |
| Same compiled profile, batch 128 | 108.72 | 131.97 |

Fused AdamW changed BF16 timing by less than 1%, so it was not selected.
Compilation took about 26 seconds per model in the screen and recovered that
cost within the planned run. Neither larger batch improved crops per second.
The selected pilot recipe is BF16, ordinary AdamW, compilation, two-buffer
prefetch and batch 32. Compilation changes stochastic dropout execution, so
quality must still be compared with the paired reference; within-profile
checkpoint/resume replay passed bitwise checks.

The first batch-128 measurements encountered Dynamo's accumulated recompile
limit and ran eagerly. They are preserved and marked inadmissible in
`engineering_before_compiler_repair.json` / `engineering.json`. Only those two
affected engineering cases were repeated with isolated compiler state. The
corrected cases captured two graphs each with zero graph breaks. Runtime
compilation now fails explicitly on a recompile-limit error.

## Replay the archived pilot

The live setup code includes a post-pilot correction for clean benchmark
manifests. To audit the original study, select its immutable source tree:

```bash
bash run_torch.sh - <<'PY'
from pathlib import Path
import sys, runpy
root = Path('/mnt/lightsem-data/experiments/lightsem/iteration_speed_20260910')
sys.path.insert(0, str(root / 'source'))
from lightsem.experiments import stage11_common
stage11_common.WORKSPACE = root / 'source'
runpy.run_path('/root/lightsem/docs/research/2026-09-10-speed/audit.py', run_name='__main__')
PY
```

For the GPU regeneration audit, set `sys.argv = ['audit.py', '--regenerate']`
before `runpy.run_path`. The archive contains the tested code, not a revised
protocol or updated weights. Only one GPU audit/job may run at a time.
