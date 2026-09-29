# Opt-in physical-objective music ablation implementation

2026-09-10. **Engineering implementation only.** No training, GPU execution,
real-record array access, TEST access, checkpoint loading, weight selection, or
vault writes. No improvement claim. Original music model/data/training and all
lighting files are untouched by this work.

## New interface

- `music_revision_model.py`: `MusicRevisionModel`, with required revision
  `arm="masked_native"` or `arm="physical"` and native `MusicConfig(arm="flow")`.
- `music_revision_data.py`: `MusicRevisionResident`, including all nonempty TRAIN
  recordings. Default tensor device is CPU; callers can explicitly request CUDA.
- `tests/test_music_revision.py`: synthetic CPU tests. Small nonzero test weights
  are test fixtures, **not proposed scientific weights**.

Both arms inherit the original layers, dilations, width, initialization,
`vector_field`, noise, draw distribution, chunk logic and `generate` unchanged.
There is no dropout, schedule, conditioning encoder, audio gain or solver change.
Use identical initial state dictionaries and identical sample/draw RNG states.
Neither branch consumes extra random numbers. The data/mask intervention is shared
by both arms, so their difference tests the physical objective, not short-record
inclusion. Comparing either arm with the historical unmasked checkpoint would
also change data exposure and is not that isolated ablation.

### Objective

`masked_native` uses the original normalized-coordinate MSE plus `.05` times the
TRAIN-scaled coordinate-derivative MSE. Reduction pools valid scalar coordinates;
derivatives require both adjacent frames. Empty edge/frame support returns zero.
The physical arm adds separately reported FP32 terms from the existing
`music_physical.py`, evaluated on
`clean_normalized = state + (1-time)*predicted_velocity`:

| Returned key | Meaning / units |
| --- | --- |
| `physical_position` | Mean squared FK joint distance, m² |
| `physical_velocity` | Mean squared FK joint velocity error, (m/s)² |
| `physical_planted_foot_velocity` | Mean generated foot speed squared, selected by detached **reference** contacts at both valid edge endpoints, (m/s)² |
| `physical_root_displacement` | Mean horizontal displacement error from each valid segment origin, m² |
| `physical_floor_penetration` | Mean squared below-floor depth across joints, m² |

`PhysicalWeights` is a frozen dataclass. Every weight defaults to zero; weights
must be finite and nonnegative. Nonzero weights are rejected for `masked_native`.
The caller must preregister and save any scientific weights. Weights are serialized
in `config_dict()`, not hidden in model parameters. Zero-weight terms remain
reported but do not enter total arithmetic or its gradient graph, giving exact
zero-weight parity with the matched baseline. Physical losses may be nonzero at
reference identity (real contact slip or floor penetration). No loss uses generated
contacts to decide which feet are penalized. Root loss preserves reference travel;
it is not a penalty toward zero velocity or an absolute world-position constraint.

Reference motion and masks enter the training loss only. `generate` is inherited
unchanged and accepts audio, RNG/initial-noise and solver/chunk settings, **not**
reference motion, contact labels or validity masks. Generated contact coverage,
generated-contact foot consistency, full-track travel and music-conditioning
controls remain separate evaluation gates.

### Padding and sampler contract

The resident stores each short recording once, then appends zeros to its own
storage region. Longer recordings are not padded or repeated. Motion padding is
zero in physical native coordinates; audio padding is zero after normalization.
`b` is a one-dimensional resident boolean validity tensor. The training wrapper
zeros invalid audio, noise and normalized target before the unchanged convolutional
forward. This also prevents arbitrary/NaN padding values from contaminating nearby
valid predictions. No padding contributes a scored frame or derivative; FK resets
horizontal origins across invalid gaps.

`data.batch(size, frames, numpy_rng)` returns:

- normalized Base92 audio `(B,T,92)`;
- raw native motion/reference `(B,T,139)`;
- boolean valid frames `(B,T)`.

`frames` must equal the resident's fixed crop length. Sampling is uniform canonical
song → uniform recording within song → uniform valid contiguous crop. A short
recording always starts at zero. Equal song weight is a sampling probability, not
a promise that each finite batch contains every song. Losses still pool valid
frames/edges across the batch, so variable valid counts are **not** equivalent to
per-record normalized losses. This preserves original valid-scalar loss semantics.
Sorted keys/groups make replay independent of manifest order. Store the NumPy RNG
state and global Torch RNG state with any future checkpoint. `draw_indices`
reproduces indices exactly; `data.b[indices]` reproduces masks.

The existing `BatchStream` interface fits naturally: its third gather becomes the
boolean mask. CPU `depth=0` is tested. CUDA `depth=2` already gathers `x/y/b` with
one shared index buffer and needs no old-module edit; its pinned-buffer, consumed
RNG and device behavior have **not** been exercised here. Do not claim CUDA
prefetch readiness from the CPU test.

`data.coverage` reports per-record real frames, valid frames per crop, selection
probability, reference-contact frame count, canonical/record totals, padding and
expected valid frames/edges per crop. It makes short recordings visible before
fitting. Reference contact is **not a hold label**: true hold-cohort coverage is
explicitly marked unmeasured. Audit real TRAIN song/hold coverage before fitting;
this implementation deliberately did not read real arrays or invent hold labels.

Padding invariance here means changing padding contents at a fixed crop length
does not change predictions/losses/gradients, and adding masked positions to the
**loss reducer** does not change its scores. It does not claim that an unchanged
biased convolutional network produces identical valid outputs when its input
length is changed. Extra padded hidden sites can alter boundary context. Both
arms use exactly the same fixed crop and boundary convention; no masked layers or
new conditioning architecture were introduced to hide that distinction.

## Compile strategy: preserve safety explicitly

Use `model.loss_function(compile_vector_field=True)` for this opt-in implementation.
It compiles **only** the unchanged neural vector field with `fullgraph=True` and
`dynamic=False`. Input checks, normalization, native loss, clean-estimate checks
and FK loss stay eager. Every validation in `music_physical` still executes.
Normalizer checks also execute after checkpoint loads/buffer mutation. Nothing
suppresses exceptions or falls back silently when compilation fails.

Do **not** use existing `TrainingRuntime.loss_function(model)` unchanged for this
revision: it compiles the entire loss, including GPU boolean checks that cause
scalar synchronization/graph breaks. The intentional eager shell still has GPU
host synchronizations, so performance must be measured later. This is a tested
compile-boundary strategy, not a claim of full-loss graph capture or production
throughput. CPU tests use a counting fullgraph backend to check exact value and
gradient parity, plus `aot_eager` forward/backward in CPU FP32. They also verify
that bad time and mutated zero standard deviation still raise outside the graph.
A CPU BF16 compiled-convolution backward attempt failed with oneDNN's explicit
"DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2" error.
The final neural compile test therefore uses FP32; no backend safety setting was
disabled. The separate physical-loss suite checks BF16-to-FP32 FK gradients, not
BF16 compiled neural backward. Default Inductor/CUDA behavior remains untested.

Minimal opt-in setup (not a training launcher):

```python
config = MusicConfig(arm="flow")  # Keep the frozen native architecture/config.
baseline = MusicRevisionModel(config, motion_norms, arm="masked_native")
physical = MusicRevisionModel(config, motion_norms, arm="physical",
                              weights=PhysicalWeights())  # All zero until frozen.
physical.load_state_dict(baseline.state_dict(), strict=True)
data = MusicRevisionResident(manifest, audio_norms, config.frames, device="cpu")
loss_fn = physical.loss_function(compile_vector_field=False)
audio, reference, valid = data.batch(batch_size, config.frames, numpy_rng)
noise, time = physical.draws(audio)
losses = loss_fn(audio, reference, valid, noise, time)
```

For paired comparison, use the **same batch and draws** for both arms, or restore
identical independent RNG cursors in separate runs. Normalizers remain caller-
supplied frozen TRAIN artifacts; this code does not fit or change them.

## CPU validation

**30/30 tests passed:** 15 new revision tests and 15 existing physical-loss tests.
Command (repository Torch environment; CUDA hidden):

```sh
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 bash run_torch.sh -m unittest discover -s tests -p "test_music_*.py" -v
```

Exit 0. Tests cover all-valid original loss/gradient parity; exact zero-weight
physical/baseline parity; explicit weighted sums; finite gradients; fixed-length
padding-content invariance through a nonzero network; loss-reducer extension
invariance and invalid-edge exclusion; all-invalid/one-frame support; clean
endpoint identity; detached references; inherited audio-only generation; eager
safety around compiled forward; AOT-eager compiled FP32 backward; short-record
coverage without repetition; exact canonical probabilities; deterministic index
and RNG-state replay; and CPU BatchStream paired exposure. Existing physical tests
also cover planted/sliding/contact-free examples, reference-contact anti-bypass,
root drift, floor loss, internal-gap origin resets and degenerate rotations.


The test scope is synthetic mechanics, not physical quality, useful audio
conditioning, real-data near-degenerate gradients, GPU stability or readiness to
train. Before any scientific run: audit real short/hold coverage, freeze new
identities/weights/code/exposure/schedule/gates, schedule GPU preflight separately,
and retain donor/contiguous-shift, generated-contact coverage, active travel,
floor/slip and full-track visual gates. The weak held-out linear summary signal
and failed historical music pilot remain unchanged. No seed-1 or TEST permission
is implied.
