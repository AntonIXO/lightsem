# LightSem foundational pipeline architecture

Status: living architecture document, updated 2026-10-21.

## 1. Mission and current claim boundary

LightSem maps music to stage-light outputs. The intended deliverable is an audio-conditioned light controller whose output remains visible in the native renderer, is physically safe, and changes with the correct song.

The current data does **not** identify authored moving-head choreography. Most available corpora contain appearance-level lighting features, RGB/intensity/events, or derived group statistics. They do not contain fixture-resolved authored pan/tilt/DMX aligned to the music. Therefore the pipeline has two explicitly separate modes:

1. **Appearance foundation:** train or adapt audio-to-RGB/brightness/event representations using the canonical BeatSaver contract and open appearance datasets.
2. **Diagnostic procedural motion:** optionally use a declared stage-space motion prior whose speed/turn/phase are audio-conditioned. This is not recovery of authored choreography and cannot be promoted as such without physical motion targets.

Promotion is forbidden until raw safety, rendered motion retention, correct-song conditioning, decoded-DMX provenance and human review all pass.

## 2. System overview

```text
Audio file
  │
  ├── MuQ hidden representation (selected layer / learned adapter)
  ├── Base92 MIR features
  └── optional common MIR features for open-data transfer
          │
          ▼
  Shared temporal audio/context encoder
          │
          ├── LightHead
          │     ├── RGB / hue appearance
          │     ├── brightness / dimmer
          │     └── event / effect logits
          │
          ├── ConductorHead (optional, low-rate)
          │     ├── section / phrase regime
          │     ├── hold / move regime
          │     └── group-level timing
          │
          └── StageMotionHead (only diagnostic until physical targets exist)
                ├── speed / acceleration control
                ├── signed steering / curvature / phase control
                ├── amplitude / hold control
                └── stage-space target or ray representation
                          │
                          ▼
              Fixture patch + physical inverse mapping
                          │
                          ▼
                    Native GPU renderer
                          │
                          ├── rendered beam-locus metrics
                          ├── frame review
                          └── decoded-DMX retention / provenance
```

The critical distinction is between a **control tensor that changes** and a **beam that visibly travels**. Latent rank, RGB variance, native aim deltas and brightness changes do not certify spatial movement.

## 3. Dataset registry

### 3.1 Canonical BeatSaver / LightSem contract

**Role:** primary appearance training contract and canonical LightSem evaluation substrate.

- Contract: `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1/CONTRACT.json`
- Rows: `300`
- Clock: `10 Hz`
- Audio input: `1,116-D = MuQ selected embedding + Base92`
- Split: by `wav_sha256`
- Existing targets/cache fields include:
  - `rgb`
  - `inten`
  - `on`
  - `n_active`
  - `events`
  - effect metadata such as `fx_ring_spin`, `fx_ring_zoom`, `fx_laser_speed`, `fx_boost`
- Missing from the public canonical target contract:
  - authored pan
  - authored tilt
  - fixture-resolved DMX
  - calibrated beam/ray targets

**Use:** RGB/brightness/event/effect appearance learning. Do not manufacture pan/tilt from these fields.

**Current limitation:** the contract is insufficient to supervise physical beam direction.

### 3.2 RPMC-L2

**Role:** open appearance-level music/light pretraining and temporal activity prior.

- Local artifact: `/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5`
- Samples: `699`
- Size: approximately `42.6 GB`
- Released content includes synchronized music features and lighting threshold/HSV tensors.
- Public lighting representation is appearance/threshold based, not fixture-level pan/tilt/DMX.
- The dataset card/release reports an open research license; verify source terms before redistribution or commercial use.

**What enters the new open-data prototype:**

- a common 36-D MIR feature vector;
- appearance-derived lighting activity target;
- a signed group/appearance change target used as a steering/style proxy.

**What it does not provide:** physical beam direction. It can train speed/style/temporal priors only.

### 3.3 Roskilde Cross-Modal Metrics / `songs.pkl`

**Role:** added open appearance/correspondence source and now included in the combined open-data training arm.

- Repository: `https://github.com/MKKeys92/Cross-Modal-Metrics-for-Capturing-Correspondences-in-Stage-Performances`
- Repository license: MIT for code.
- Local dataset: `/mnt/lightsem-data/open_datasets/roskilde_2023/songs.pkl`
- Local metadata: `/mnt/lightsem-data/open_datasets/roskilde_2023/data.csv`
- `songs.pkl` entries: `224`
- `data.csv`: `225` lines including header.
- `songs.pkl` entry structure:
  - `show`
  - `date`
  - `interpret`
  - `song`
  - `audio_features`
  - `lighting_features`
- Example released shapes:
  - `audio_features['mfcc']`: `[T, 20]`
  - `audio_features['chroma_stft']`: `[T, 12]`
  - `audio_features['onset_env']`: `[T, 1]`
  - `audio_features['rms']`: `[T, 1]`
  - `lighting_features`: `[T, 60]`, representing 10 groups × 6 abstracted features.

Direct repository inspection also found:

```python
PASv01.attribute_sizes == (10, 10, 10, 0, 0)
PASv01.pan_idx == []
PASv01.tilt_idx == []
```

The README describes transformations from raw control-like data, but the released `songs.pkl` is an abstracted feature contract. The public object is therefore not a pan/tilt/DMX training source.

**Combined open-data arm:**

```text
RPMC-L2:                  699 samples
Roskilde Cross-Modal:    224 samples
Total:                   923 samples
Seed:                      0
Common MIR input:         36-D
```

Receipt:

`/mnt/lightsem-data/experiments/lightsem/foundation_cyclic_motion_prior_open_combo_20261021_v2/TRAINING.json`

The combined arm currently fails raw jerk in the large stage-target parameterization and was not rendered or promoted.

### 3.4 Independent own-song evaluation assets

These are not training rows. They are locked independent evaluation inputs:

- Shpongle audio: `/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v8/audio_reference/shpongle.m4a`
- Cure For Me audio: `/mnt/lightsem-data/experiments/lightsem/native_portfolio_20261012_v8/audio_reference/cure_for_me.m4a`
- MuQ feature files:
  - `.../audio_pipeline_integration_20261005/features/muq/shpongle.npz`
  - `.../audio_pipeline_integration_20261005/features/muq/cure_for_me.npz`
  - each contains `embeddings [4500,1024]` plus timestamps
- Base92 feature files:
  - `.../native_bf16_v7b/demo_features/features/base92/shpongle.npz`
  - `.../native_bf16_v7b/demo_features/features/base92/cure_for_me.npz`
  - Shpongle `x30`: `[8320,92]`
  - Cure For Me `x30`: `[6050,92]`

Each song must retain independent audio features, prediction arrays, renderer output and provenance. Never reuse Shpongle motion/body/DMX for Cure For Me.

### 3.5 Generated body-to-DMX diagnostic data

- Root: `/mnt/lightsem-data/experiments/lightsem/body_to_dmx_foundation_20261013`
- Contains generated diagnostic body-to-DMX artifacts, including `demo_3b4ab_v1/demo.npz`.
- Used only for an explicitly named auxiliary diagnostic arm.
- It is not authored lighting ground truth and cannot qualify the music-to-light foundation model.

### 3.6 Datasets and leads not admitted to training

- Musilux / AuraLuxMuse: paper-described professional pan/tilt/timecode cues, but public download and dataset license were not verified.
- LightingGen: paper-described DMX corpus, but public download/license was not verified.
- AIST / dancer pose corpora: useful for dance research, but Vibescience records ruled out dancer kinetic energy as dense authored beam supervision at the required resolution.
- HarmonixSet / SALAMI / All-In-One: useful for musical structure or beat/section features, not lighting targets.
- SoniDo / LLark / MERT: architecture and representation references, not LightSem motion labels.

## 4. Feature and target contracts

### 4.1 MuQ + Base92 canonical input

```text
MuQ selected embedding: 1024-D
Base92:                   92-D
Concatenated input:     1116-D
Clock:                    10 Hz
```

The current canonical direct model consumes this stream through an audio/context encoder and fixture-query decoder.

### 4.2 Appearance target contract

Existing direct training losses are implemented in:

`/root/lightsem/scripts/experiments/run_foundation_v3_audio_only.py:loss_parts`

They include RGB, brightness, events and fixture-spread/regularization terms. These targets can make a model colorful and nonconstant while its beam remains static.

### 4.3 Open-data procedural-motion target contract

The combined open-data prototype uses a common 36-D MIR representation:

```text
MFCC[13]
chroma_stft[12]
onset/activity[1]
RMS[1]
MFCC delta[9]
----------------
36 dimensions
```

Derived targets are:

- speed/activity: temporal change in released lighting appearance;
- signed steering proxy: temporal change in group-level intensity/hue appearance.

These labels are explicitly **not** physical beam direction. They are used to test whether a learned speed/turn prior can affect a native renderer at all.

## 5. Current model components

### Existing direct model

- `/root/lightsem/lightsem/experiments/audio_light_foundation/model.py`
- `/root/lightsem/lightsem/experiments/foundation_audio_light/direct_model.py`
- shared temporal audio/context computation;
- fixture-query decoding;
- aim, RGB, brightness and event heads;
- smooth/raw dynamics projection variants;
- `body_teacher` exists in the codebase but is not used as authored beam supervision by the direct loop.

### Cyclic/procedural prototype

- `/root/lightsem/scripts/experiments/run_cyclic_motion_prior.py`
- `SpeedNet` outputs:
  - speed control;
  - signed turn/curvature control.
- Native control law integrates phase and heading.
- Stage-space variant computes a target point, then solves fixture-local pan/tilt residuals around the patch anchor.
- Existing RGB/brightness/event predictions are kept separate from motion.
- The prototype is diagnostic-only and must not be called recovered authored choreography.

## 6. Renderer and evaluation seam

Renderer calibration is frozen in:

`/mnt/lightsem-data/experiments/lightsem/foundation_v3_motion_active_user_visual_audit_20261020_v1/RENDERED_MOTION_GATE.json`

```text
Static control:       0.351248 px centroid-step mean
Known linear sweep:  54.790562 px centroid-step mean
Candidate threshold: 27.570905 px on both songs
```

Raw angle limits:

```text
speed p99:        <= 0.18
acceleration p99: <= 0.50
jerk p99:         <= 1.50
```

A complete arm must run, in order:

1. open-data/training provenance receipt;
2. fresh unfiltered raw angle audit;
3. converted unit-direction derivative audit;
4. static/known-sweep seam controls;
5. independent Shpongle and Cure For Me render;
6. rendered centroid/MAE measurement;
7. actual frame review with `vision_analyze`;
8. correct-audio, shuffled-audio, circular-shift and same-BPM controls;
9. decoded-DMX retention and provenance;
10. human review.

## 7. Known failure modes

- Hard clipping creates jerk spikes.
- Unbounded angle integration sends beams off-stage and produces black output.
- A fixed orbit plus scalar speed modulation is visually almost constant and is not sufficient for the user's idea.
- A signed steering head can change native heading while the renderer still collapses it into a stable fan.
- Appearance-derived speed/steering targets improve above static but do not identify spatial beam semantics.
- Stage-space paths that visibly span the rig can violate raw jerk limits.
- A candidate can pass 2-D angle p99 while converted physical-ray derivatives remain above the same safety budget.
- RGB/brightness changes are not beam travel.
- Native DMX absence blocks promotion even when a video exists.

## 8. Current status

The latest combined open-data train is:

`/mnt/lightsem-data/experiments/lightsem/foundation_cyclic_motion_prior_open_combo_20261021_v2/`

It uses both open lighting datasets:

```text
RPMC-L2 + Roskilde Cross-Modal Metrics
699 + 224 = 923 samples
```

It currently fails raw stage-space jerk before rendering. The earlier v5/v10/v14/v15 artifacts were inspected and remain diagnostic/unqualified; no working demo is claimed.

The foundational architecture is therefore currently split into:

```text
appearance foundation: usable for RGB/brightness/events
open-data motion prior: diagnostic only
physical authored-motion foundation: not yet available
```

## 9. Next honest architectural options

1. Acquire a licensed aligned pan/tilt/DMX subset and train `StageMotionHead` on physical rays.
2. Keep RPMC-L2 + Roskilde for appearance/style pretraining, but add a separate calibrated renderer-aware target contract.
3. Collect a small authored calibration set from a controlled rig with audio, patch, fixture geometry and DMX timecode.
4. If no physical target becomes available, redefine the product explicitly as audio-conditioned procedural stage motion and use a lower claim scope; do not call it authored choreography.

Do not solve the missing spatial target by adding an oscillator, fallback choreography, arbitrary gain, posthoc smoothing, reused body, fixture duplication or latent-rank promotion.
