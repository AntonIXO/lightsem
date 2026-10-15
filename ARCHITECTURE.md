# LightSem architecture

Документ описывает исполняемый baseline на **2026-10-15**. Источником истины для runtime является [`configs/dance/fixed_c.json`](configs/dance/fixed_c.json), а для обязательных safety/provenance-правил — [`AGENTS.md`](AGENTS.md).

Текущий baseline имеет статус `user_selected_motion_baseline` для совместимости с loader и отдельный `color_baseline_status=user_selected_gradient_color_baseline`. Это инженерный baseline, выбранный пользователем. Held-out, artistic и physical-fixture qualification остаются отдельными задачами.

## 1. Исполняемый baseline

```text
audio file
  -> pinned MuQ / HTDemucs / BeatThis / All-In-One extraction
  -> native 30 Hz Base92 + 1061-D audio features
  -> authenticated 128-D SpaIn/oBSRNN stereo context
  -> native C planner + flow, 32 solver steps
  -> 30 Hz native 22-joint body
  -> optional D4/D6 post-generation modifier
  -> 40 Hz body geometry resampling
  -> mapper 01 body field
  -> 01_distributed_gain spatial readout
  -> historical LUFS/body dimmer
  -> authenticated P2 RGB seed
  -> source-driven OKLab fixture color field
  -> 40 Hz uint8 multi-universe DMX
  -> decode actual emitted controls
  -> raw/source/DMX audits
  -> 3D body + actual emitted-light video
```

Активные параметры:

| Component | Current value |
|---|---|
| Motion checkpoint | `/mnt/lightsem-data/experiments/lightsem/native_full_native_repair_20261018_v2/training/full_native/final_with_provenance.pt` |
| Motion checkpoint SHA-256 | `75a5969dd0116c19ce5886e9cb51b6250b9238fad86d91baac5f8926d067ba52` |
| Native body | 22 joints, 3D, 30 Hz |
| Lighting clock | 40 Hz |
| Flow steps | 32 |
| Draw seed | 7001; training seed 0 |
| MuQ | `OpenMuQ/MuQ-large-msd-iter`, revision `0562a57814f6f8bbd9fdea0a25921a2fce1a841a`, layer 9 |
| Stereo context | authenticated dual cross-attention, SpaIn/oBSRNN concat128, 128 dimensions |
| Motion mapper | `01` anatomical body field |
| Spatial readout | `01_distributed_gain` |
| Default installation | 8 beams, 5 simulated lasers, 50 bulbs |
| Fixture motion caps | OFF in simulation; raw body and emitted DMX are still audited |
| D4-max | OFF by default |
| D6 | OFF by default |
| Color seed | authenticated P2 RGB, own song |
| Color field | source-driven continuous OKLab field, arbitrary fixture count |
| Brightness | restored historical full-song LUFS/body law, separate from RGB |

The previous P2/native-mapper snapshot is preserved at [`configs/dance/fixed_c_p2_20261015.json`](configs/dance/fixed_c_p2_20261015.json). The promotion is recorded in [`BASELINE_PROMOTION_GRADIENT_20261015.json`](docs/research/2026-10-14-native-jitter-fourier-attention/BASELINE_PROMOTION_GRADIENT_20261015.json).

## 2. Runtime stages

### 2.1 Audio and feature extraction

`lightsem.dance.pipeline.music_input()` binds the input waveform SHA to a complete-song `MusicTimeline` and paired Base92 cache.

Required contracts:

- Original waveform; no donor, shifted, muted or loudness intervention.
- Frozen MuQ model, revision, layer and weight SHA.
- Timeline and Base92 share waveform SHA, timestamps and duration.
- Native body coverage is complete at 30 Hz.
- New audio uses a SHA-keyed cache under `/mnt/lightsem-data/cache/native_c_cli` unless `--audio-cache` overrides it.
- New tracks with the active stereo baseline receive automatic SpaIn/oBSRNN context extraction; the context manifest is bound to the waveform.
- `/mnt/lightsem-data` must remain above the 20 GiB free-space floor.

The active stereo branch is not a free-form embedding fallback. Missing or mismatched context, teacher hash, waveform hash or clock is an error.

### 2.2 Native body generation

`lightsem.dance.pipeline.native_body()` builds:

```text
fine = [Base92_30Hz, audio_features_30Hz]
coarse = 15-frame pooled planner context
stereo = authenticated 128-D context, when enabled
noise = deterministic CPU noise field, seed 7001
```

The native flow generates a 139-value motion state at 30 Hz:

```text
3   root velocity/height channels
132 local rotations: 22 joints × 6D rotation representation
4   contact probabilities
```

Forward kinematics converts the state into `world[T,22,3]`. The body is saved as a `DanceResult` with exact timestamps, metadata, checkpoint identity, audio identity, noise identity and native channels.

No fallback oscillator, beat sweep, random motion, post-hoc smoothing or viewer-fill motion is available in this path.

### 2.3 D4-max and D6

`lightsem.dance.pipeline.modify_body()` applies optional modifiers after native generation:

| Option | Operation | Baseline status |
|---|---|---|
| C | Native C body | Active |
| D4 | Own-song phrase/motif recomposition with D4 bank and audio anchors | Opt-in diagnostic modifier |
| D6 | External learned positive pacing clock, minimum rate 0.25, maximum 1.5 | Opt-in diagnostic modifier |
| D4+D6 | D4 recomposition followed by D6 pacing | Opt-in diagnostic modifier |

D4 and D6 do not retrain native C. D6 is independent of D4. A modifier is not allowed to conceal a raw native failure. Raw body safety is checked before mapping and video.

## 3. Body-to-light transfer

### 3.1 Fixture set

`lightsem.conductor.body_transfer_v2.installation.make_outputs()` creates stable fixture records:

```text
beam/laser: position, pan/tilt range, anatomical role, secondary role,
            spatial slot, anchor, fixed source attribution
bulb:       position, scalar field coordinate, anatomical field role
```

Default output counts are `(beams=8, lasers=5, bulbs=50)`. Numerical constructors accept larger counts, but the 3D renderer accepts at most 16 moving outputs and 128 bulbs.

Extra outputs query the same body field at other fixed fixture locations. They do not add independent dance information.

### 3.2 Mapper 01

`body_transfer_v2.map_body(world40, outputs, "01", 40.)` computes anatomical bone-direction features and maps them to a 22-node field. `installation.adapt_field()` then supplies:

- moving-fixture target points;
- body-derived dimmers;
- source points and field points;
- bulb spatial centers;
- attribution mask from outputs to body joints.

The mapper consumes the saved body only. It has no audio clock and no independent temporal driver.

### 3.3 Distributed readout

The active `lighting.transfer.readout="distributed_gain"` branch is implemented in [`dynamic_dance_combo/campaign.py`](lightsem/experiments/dynamic_dance_combo/campaign.py).

For each moving fixture:

```text
center_i = temporal median of mapper-01 target_i
x_i      = fixed fixture x-coordinate
gain_i   = 1.22 + 0.20 × min(1, |x_i| / 6.5)
target_i = center_i + gain_i × (target_i - center_i)
```

The transform is fixed in fixture space and source-derived in time. It increases spatial/angular spread without adding a clock, phase, noise or independent trajectory. Bulbs are not stretched by this readout.

### 3.4 Body resampling and DMX compilation

Native body geometry is resampled from 30 Hz to the exact 40 Hz lighting clock. Native velocity and rotation channels are not blindly interpolated.

`compile_intent()`:

1. converts target vectors to calibrated pan/tilt angles;
2. applies mechanical projection only when configured;
3. quantizes pan/tilt to coarse/fine 16-bit DMX bytes;
4. writes dimmer and RGB channels;
5. allocates fixtures into 512-channel universes;
6. returns a patch describing universe, address, channel count and capability.

The actual emitted stream is always decoded again by `body_transfer.decode()`. Requested targets are diagnostic inputs; decoded controls are the emitted-light truth.

## 4. Color and brightness

### 4.1 P2 as authenticated color seed

The active color source remains the per-song P2 stream. P2 supplies `left_rgb`, `right_rgb`, palette IDs and color-cue history. The P2 report binds RGB to the original waveform and color configuration hash.

P2 remains the audio-conditioned seed, not the final spatial output. The previous P2-only baseline is preserved as a separate config.

### 4.2 Continuous OKLab fixture field

[`lightsem/dance/color_field.py`](lightsem/dance/color_field.py) generates `fixture_rgb[T,N,3]` for arbitrary `N`.

Pipeline:

```text
P2 left/right RGB + native body
  -> sRGB → linear RGB → OKLab
  -> body/P2 source-flow signal
  -> bounded continuous palette phase
  -> OKLab palette interpolation
  -> body-effort lightness/chroma breath
  -> OKLab chroma-reduction gamut mapping
  -> uint8 sRGB per fixture
```

Active baseline parameters:

```text
palette = [#25105f, #6f1ab6, #e52e88, #ff78a9, #ffd6e7]
spatial_cycles = 0.95
temporal_gain = 6.0
breath_depth = 0.10
```

The field uses fixture spatial coordinates, so fixture count is a query dimension. It does not use fixture-indexed random phases.

The temporal flow signal is source-derived:

```text
speed = mean hand/limb speed from the saved body
p2_delta = consecutive P2 RGB change
drive = clip(0.012 × speed + 0.002 × p2_delta, 0, 0.35)
phase = cumulative drive + signed body articulation term
```

The palette center follows a smooth triangle traversal. It is continuous at palette boundaries and stops advancing when source flow stops. There is no free-running oscillator.

OKLab lightness/chroma breath is a color appearance operation. The emitted dimmer remains the independent LUFS/body channel.

### 4.3 Brightness

`dynamic_dance_combo.brightness.apply_brightness()` computes historical full-song LUFS activity and body accent. It produces per-fixture dimmer values. Color field lightness does not replace dimmer, and dimmer changes do not prove color-head activation.

### 4.4 Color metrics

The color receipt reports:

- finite RGB;
- temporal OKLab velocity;
- spatial OKLab gradient;
- position-sorted spatial gradient;
- unique fixture streams and exact duplicates;
- effective color rank;
- RGB range;
- dimmer/OKLab-lightness correlation;
- arbitrary-N fixture count;
- source-derived color parameters.

The accepted color baseline is engineering-selected. A human preference study and held-out color qualification remain open.

## 5. DMX and renderer

### 5.1 Body-transfer personalities

Moving beam and simulated laser personality:

```text
pan coarse, pan fine, tilt coarse, tilt fine, dimmer, red, green, blue
```

Bulb personality:

```text
dimmer, red, green, blue
```

`patch.json` is mandatory because channel addresses are fixture-specific. A DMX byte difference is not automatically an angular difference; decode 16-bit coarse/fine channels before measuring movement.

### 5.2 Video renderer

`lightsem.dance.render.render_video()` consumes:

- saved native body;
- exact 40 Hz patch and DMX controls;
- decoded directions/RGB/dimmer;
- original-speed audio;
- raw body and emitted-motion chart traces.

The two panes show:

```text
left:  own-song 3D body
right: actual emitted lights from decoded DMX
bottom: raw speed, jerk, root, emitted motor speed, dimmer and modifier rate
```

The renderer does not generate motion. Presentation fill is separate from emitted dimmer. Current color experiments are labeled `OKLab flowing gradient | source-driven color field`; old baseline renders default to `P2`.

### 5.3 Media integrity

Each render must preserve:

- exact frame count and 30 fps PTS;
- original-speed soundtrack and decoded PCM identity;
- body SHA;
- lighting/DMX SHA;
- patch SHA;
- raw trace arrays and chart scope;
- generation and lighting receipts;
- source/audio/checkpoint/config hashes.

Smoke frames are inspected before full encoding. Full renders use project GPU locks and the configured encoder. A failed raw gate rejects the render before mapping/video.

## 6. Safety and diagnostics

### 6.1 Mandatory native generation gate

Before mapper, interpolation, DMX or video, audit the saved raw 30 Hz body:

- finite values;
- speed, acceleration and jerk;
- high-frequency residual;
- root height and support/contact proxies;
- inactive frames;
- exact duplicate trajectories;
- ratio against selected baseline;
- raw source/checkpoint/noise provenance.

Candidates with nonfinite values, shaking, teleporting, raw kinematic explosions or unsafe ratios are quarantined. Smoothing, retiming, fixture caps and D4/D6 cannot rescue them.

### 6.2 Body-to-DMX source dependence

Before presentation, `lighting._source_dependence_audit()` checks:

1. frozen body produces frozen motion controls;
2. assigned joint perturbations change assigned moving fixtures;
3. attribution covers expected joints;
4. moving fixture response exists;
5. arbitrary output counts preserve source timing;
6. actual decoded DMX is finite;
7. exact duplicate streams are reported separately.

Color changes are audited separately from motion. A changing RGB stream does not establish changing motion.

### 6.3 Music conditioning

Music conditioning requires at least two independently recomputed songs with:

- fixed weights;
- fixed output counts and mapping;
- fixed seed/noise policy;
- independent body, color, DMX and soundtrack paths.

A new soundtrack over reused DMX is not a new generation. Shpongle and Cure For Me must retain separate source bodies and source hashes.

### 6.4 Current evidence

The promoted gradient artifact is `/mnt/lightsem-data/experiments/lightsem/color_motion_field_20261015_v9`. Its full-song receipts report source-dependence pass for both tracks. The native CLI was rechecked after promotion:

```text
Shpongle: mapper 01_distributed_gain, source pass, 61 unique RGB streams, software decode pass
Cure For Me: mapper 01_distributed_gain, source pass, 61 unique RGB streams, software decode pass
```

The focused regression suite contains 227 passing tests. This does not establish held-out choreography, human preference, real-laser safety or physical fixture calibration.

## 7. CLI

Run from `/root/lightsem` with the project wrapper:

```bash
export TORCH_ENV=/mnt/lightsem-data/venv-dynamic-structure
```

Default full-song baseline:

```bash
TORCH_ENV="$TORCH_ENV" bash run_torch.sh -m lightsem.cli render \
  --wav /root/shpongle.flac \
  --out runs/shpongle_gradient_full.mp4 \
  --device cuda
```

The same command with Cure For Me uses its own WAV. `--start` and `--duration` crop only the rendered review; body/features/DMX generation remain complete-song.

Optional modifier combinations:

```bash
# D4 only
bash run_torch.sh -m lightsem.cli render --wav "$TRACK" \
  --out runs/track_d4.mp4 --device cuda --d4max

# D6 only
bash run_torch.sh -m lightsem.cli render --wav "$TRACK" \
  --out runs/track_d6.mp4 --device cuda --d6

# D4 + D6
bash run_torch.sh -m lightsem.cli render --wav "$TRACK" \
  --out runs/track_d4_d6.mp4 --device cuda --d4max --d6
```

Body-only generation:

```bash
bash run_torch.sh -m lightsem.cli dance \
  --wav "$TRACK" --out runs/track_body --device cuda
```

Historical rule-driven reproduction is explicitly separate:

```bash
bash run_torch.sh -m lightsem.cli legacy-render --wav "$TRACK" --out runs/legacy.mp4
```

`legacy-render` is not the current native baseline.

## 8. Foundational-model extension point

The current baseline is a composition of a native body model, deterministic fixture readout, continuous color field and separate dimmer. It is not yet one foundational model.

The next trainable architecture should preserve the current contracts while replacing only the creative readout:

```text
audio tokens + body joint tokens + fixture graph tokens
  -> cross-modal latent queries
  -> permutation-equivariant fixture decoder
  -> separate motion / OKLab color / dimmer / accent heads
  -> hard physical safety projection
  -> decoded DMX
```

Required fixture token fields:

```text
position, orientation, fixture type, capability mask,
pan/tilt limits, calibration, current state, perimeter/graph coordinates
```

Required decoder properties:

- arbitrary `N` fixtures;
- permutation equivariance over fixture order;
- body-region attribution;
- source-conditioned residual diversity;
- separate color/lightness/dimmer outputs;
- no free-running temporal source;
- bounded physical state updates before DMX.

Training/evaluation tasks:

1. body/audio/light representation alignment;
2. masked body/audio/DMX reconstruction;
3. fixture permutation and count generalization;
4. continuous OKLab graph-field prediction;
5. brightness/color leakage control;
6. raw motion and emitted-DMX safety;
7. body/audio counterfactual sensitivity;
8. phrase, hold, motif and beat-aligned gesture retention.

The current gradient field is the first production-wired step toward this model. It supplies a continuous arbitrary-N color target and an explicit separation between color and brightness while leaving the native body and hard DMX contracts inspectable.

## 9. Glossary

| Term | Meaning in this repository |
|---|---|
| **Active baseline** | The configuration currently used by `lightsem.cli render`: [`configs/dance/fixed_c.json`](configs/dance/fixed_c.json). It selects native C motion, the distributed spatial readout, the OKLab gradient field, P2 as the authenticated color seed, and separate LUFS/body dimmer. |
| **Native C** | The current audio-conditioned native flow body generator. It produces a complete 30 Hz, 22-joint 3D trajectory before lighting mapping. “C” is a model-generation label, not a DMX protocol. |
| **Native body** | `world[T,22,3]` joint positions plus native channels, contacts, timestamps and provenance. It is the source of motion; it is not a video pose overlay. |
| **MuQ** | Frozen music representation model `OpenMuQ/MuQ-large-msd-iter`. The active recipe pins model ID, revision, layer 9 and weight SHA. |
| **Base92** | Native audio feature vector used by the motion path: 92 channels on the 30 Hz body clock. It is paired to the exact waveform through a hash and timestamp contract. |
| **1061-D audio features** | The larger native planner/flow feature vector used internally with Base92. It contains the frozen audio/source/structure representation required by the native planner. |
| **HTDemucs** | Frozen source-separation stage used during native audio extraction. It separates stems/features; it does not generate lighting. |
| **BeatThis** | Frozen beat/downbeat estimator used as part of the timing representation. Beat probabilities are features, not an autonomous lamp-motion clock. |
| **All-In-One** | Frozen music-structure model used to produce section/structure features for the native extraction recipe. |
| **Stereo context** | Authenticated 128-D SpaIn/oBSRNN context aligned to the song. It is an additional audio representation; it is not a second body or a second motion generator. |
| **SpaIn / oBSRNN** | The two teacher/context streams fused into the active 128-D stereo context. Their hashes and audio binding are recorded in the context manifest. |
| **D4-max / D4** | Optional own-song motif/phrase recomposition applied after native C generation. It changes a saved trajectory; it does not retrain native C. |
| **D6** | Optional external learned pacing clock applied after native C (and optionally after D4). It is bounded by the configured positive rate range and is independent of D4. |
| **P2** | Historical user-selected color seed: palette history, 300 ms causal RGB transitions and two spatial source groups. In the current baseline P2 supplies authenticated left/right RGB inputs to the continuous field. |
| **Rich96 / I_VR_P0** | The historical learned P2 color-model representation/recipe: 96-dimensional visual targets, independent audio branches, visual auxiliary loss and no pose auxiliary. It remains the P2 source model, not the final arbitrary-N fixture decoder. |
| **OKLab** | Perceptual color space used for continuous interpolation and color-field diagnostics. Interpolation occurs in OKLab before gamut mapping to sRGB/DMX RGB. |
| **OKLCH** | Cylindrical form of OKLab, useful for hue/chroma/lightness controls. The current field stores/interpolates OKLab vectors and uses gamut-safe conversion. |
| **RGB** | Red/green/blue color channels written into fixture DMX slots. RGB is separate from dimmer; an RGB change alone does not prove motion activation. |
| **LUFS** | Loudness Units relative to Full Scale, the loudness unit used by the historical full-song brightness law. LUFS drives the dimmer envelope together with body effort; it is not the color model. |
| **Brightness / dimmer** | Per-fixture emitted intensity channel. The current path computes it from historical LUFS/body equations independently of the OKLab color field. |
| **Color field** | `continuous_color_field()` in [`lightsem/dance/color_field.py`](lightsem/dance/color_field.py): an arbitrary-N, spatially continuous RGB generator conditioned on body flow and P2 RGB changes. |
| **Source flow / smooth-triangle phase** | The color field’s temporal phase. It advances from body limb speed, signed articulation and P2 changes; when the source is frozen, it stops. It is not a free-running sine or beat oscillator. |
| **Fixture** | One simulated or physical light output: beam, simulated laser, bulb or another capability-specific device. A fixture has position, type, channels and limits. |
| **Arbitrary-N** | The decoder accepts a requested number `N` of fixtures. More fixtures query the same source field; they do not create `N` independent dance decisions. |
| **Mapper** | Body-to-fixture field that converts the 22-joint body into target points and source attribution. The active mapper ID is `01` (bone-direction field). |
| **01_distributed_gain** | Active spatial readout layered on mapper 01. It expands each moving fixture target around its fixed temporal median using a static position-dependent gain. It changes spatial spread, not temporal source content. |
| **Readout** | The capability-specific conversion from an anatomical field to fixture target points, dimmers and attribution. `native` is the unexpanded mapper output; `distributed_gain` is the current promoted spatial readout. |
| **Pan / tilt** | Horizontal/vertical moving-head angles. Each is encoded as a 16-bit coarse/fine DMX pair and must be decoded before angular metrics are computed. |
| **DMX** | Digital Multiplex, the byte-level lighting-control protocol. In this project a body-transfer frame is one or more 512-channel universes. |
| **Universe** | A 512-channel DMX address space. The body-transfer compiler allocates complete fixtures into successive universes. |
| **Patch** | JSON mapping from fixture identity to universe, channel address, channel width and capability. Without the patch, DMX bytes are ambiguous. |
| **Decoded DMX** | Physical controls reconstructed from emitted bytes: angles, directions, RGB and dimmer. It is the authoritative lighting signal for audits and rendering. |
| **Raw safety** | Audit of the saved native 30 Hz body before interpolation, mapper, DMX or video: finite values, speed, acceleration, jerk, root/support and baseline ratios. |
| **Source dependence** | Frozen-body, limb-perturbation and attribution tests proving that decoded fixture motion responds to the supplied body and does not run autonomously. |
| **Exact duplicate stream** | Two fixtures with byte/decoded streams that are identical over the audited interval. This is reported separately from representation collapse and from intentional fixed controls. |
| **Effective rank** | Diversity diagnostic computed from fixture trajectory/color covariance. A low value means many outputs share a small number of modes; it is not by itself an artistic-quality score. |
| **Jerk** | Third temporal derivative of position/angle. Raw jerk is audited because filtered motion can hide high-frequency native instability. |
| **RMS** | Root-mean-square magnitude, used for energy, speed, residual and error summaries. It is a magnitude statistic, not a semantic choreography metric. |
| **AIST / FD / JD** | AIST++/AIST Dance, FineDance, and Just Dance data families. Their masks, coach identities, duplicate groups and paired/unpaired limits are separate from user-song demos. |
| **Provenance receipt** | JSON record binding paths and SHA-256 hashes for audio, features, checkpoint, body, DMX, patch, code and video. A receipt is evidence for one immutable artifact root. |
| **SHA-256** | Cryptographic content hash used to detect changed files and bind artifacts. A matching hash proves byte identity, not artistic quality. |
| **OCLEP** | No canonical definition or runtime use was found in the current repository, baseline config or architecture code. It must not be expanded as a fact. If `OCLEP` refers to an external paper, service or internal acronym, its source must be added before documenting a meaning. |
| **Native `render`** | Current MuQ → native body → body-to-DMX → decoded-DMX → 3D video command. |
| **`legacy-render`** | Historical rule-driven audio/scene renderer. It is retained for reproducing old studies and is not the active native baseline. |

Abbreviations that describe a historical experiment do not automatically describe the active product path. In particular, P2, D4 and D6 have explicit scopes above; “different hash,” “changed color” or “more output fixtures” is not equivalent to demonstrated new choreography.

## 10. Historical code and non-current paths

- `lightsem/model/light_layers/`: original learned P2 color model and training/evaluation tools.
- `lightsem/conductor/body_transfer.py`: original V1 body transfer and shared DMX compiler.
- `lightsem/conductor/body_transfer_v2/`: 15 deterministic research readouts; active baseline uses mapper 01 plus the explicit distributed readout.
- `lightsem/experiments/motion_study/`: native body representation and flow research.
- `lightsem/experiments/dynamic_dance_combo/`: D4/D6 modifier and lighting campaign code.
- `lightsem/cli.py` `legacy-render`: old rule-driven scene path.
- `lightsem/model/infer.py`, `model/light_trunk.py`, `group_decoder.py`: historical descriptor/light generations; their checkpoints are not interchangeable with native C or P2.

Historical reports remain immutable evidence. A new code, checkpoint, cache, source or config hash requires a new artifact root and receipt.
