# Foundation-v2 execution plan — collapse-only

Protocol: `audio-light-foundation-v2-contract-20261016-v1`

The purpose of this protocol is narrow: determine whether the repaired foundation emits non-collapsed, audio-responsive controls. It does not score product quality, mapper behavior, DMX, video, choreography, or artistic fit.

## Frozen data contract

The targetable supervised corpus is the current 300-row BeatSaver intersection with audio, visual event, and authored-light caches. The current disk audit finds 300 Base92-compatible audio caches, 300 visual caches, and 300 light caches. The audio cache is 64 mel channels plus 28 named scalar channels. Exact audio duplicates are grouped: `13bea/6b55` and `3080f/74d5`, leaving 298 unique waveform hashes.

The full input is 1,116 channels: pinned MuQ layer 9 (1,024) concatenated with Base92 (92). MuQ is extracted from the exact WAV at 24 kHz using the pinned `OpenMuQ/MuQ-large-msd-iter` revision `0562a57814f6f8bbd9fdea0a25921a2fce1a841a`, native 25 Hz, 30-second windows, 20-second core, 5-second context. Inference is FP32; saved arrays are float16 with strictly increasing native timestamps. Base92 remains bound to the same WAV hash and retains its raw 40 Hz timestamps. Training alignment is explicit at 10 Hz: MuQ/Base92 are timestamp-aligned and 40 Hz light blocks are averaged into 10 Hz targets. No zero-padding, tail replication, or silent cache reuse is allowed.

The initial v1 inventory recorded 8 old native MuQ caches and 292 rows requiring extraction. Those old caches used a different 8-second overlap-stitch protocol, so they are retained for historical comparison and are not silently reused. A canonical v2 extraction was completed for all 300 WAV rows under one protocol: 300/300 records, 298 unique waveform hashes, all finite 1,024-D layer-9 arrays, and strictly increasing 25-Hz timestamps. The canonical completion receipt is [COMPLETE_V2.json](/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3/COMPLETE_V2.json); the earlier v1 receipt remains preserved beside it. The channel and old-experiment reconciliation is [MUQ_EXTRACTION_CHANNEL_AUDIT.json](/root/lightsem/docs/research/2026-10-16-audio-light-foundation-v2/MUQ_EXTRACTION_CHANNEL_AUDIT.json). The canonical final contract is [CONTRACT.json](/root/lightsem/docs/research/2026-10-16-audio-light-foundation-v2/CONTRACT.json); the frozen initial inventory is preserved as [CONTRACT_INITIAL.json](/root/lightsem/docs/research/2026-10-16-audio-light-foundation-v2/CONTRACT_INITIAL.json) and at its external v1 artifact root.

Just Dance/FineDance native feature rows are not inserted into this supervised authored-light corpus: they have audio/body supervision but no matching authored-light targets. They can be a later unlabeled representation source, but not silently mixed into this collapse-only target run.

## Execution order

1. Materialize a working pinned MuQ loader and verify model id, revision, weight hash, code version, 24 kHz input, layer 9, and 25 Hz timestamps. **Completed.**
2. Extract all 300 rows into a new hash-keyed artifact root. Preserve each WAV, Base92, MuQ, duration, timestamp, and source hash. Do not merge the old 8-row cache because its 8-second boundary protocol is not byte-equivalent to the canonical 30/20/5-second protocol. **Completed; 300/300 receipt present.**
3. Build a duplicate-resolved grouped split by unique audio/song hash. Keep all exact-audio duplicates in one group. Freeze train/dev/test IDs and train-only feature/target normalizers before any model forward. **Completed for the all300 arm: 238 train groups and 60 validation groups.**
4. Run a CPU contract audit: shapes, finite values, monotonic timestamps, duration coverage, MuQ/Base92 waveform identity, target alignment, and no padding. Abort on any missing or changed row. **Completed: 300/300 MuQ rows, finite 1,024-D arrays, monotone timestamps, no missing WAV ids.**
5. Run a one-seed short foundation pilot with the same architecture family and fixed controls. Save raw predictions, latent tensors, split manifest, output hashes, and the exact loss manifest. Do not render or map. **Completed as diagnostic-only all300 target arm and canonical eight-row dance auxiliary arm.**
6. Run collapse-only scoring on both independently recomputed held-out songs and a fixed audio-shuffle control. **Completed; both arms remain output-collapsed and promotion-forbidden.**

## Collapse gate

The candidate is admitted only if all required evidence is present:

- finite outputs and finite latent tensors;
- RGB occupancy/range and variance are above a preregistered floor, with no near-zero head collapse;
- brightness has nonzero variance and is not a constant prior;
- event probability entropy and prior gap are recorded;
- fixture streams are not exact duplicates and have nonzero spread;
- same-song deterministic repeat is identical under fixed seed/noise;
- two songs are recomputed from their own audio paths;
- audio shuffle/swap changes the audio-dependent outputs while all non-audio inputs, weights, seed, and clock remain fixed;
- if the head is output-collapsed but the latent rank is healthy, classify `output_collapsed`; call `representation_collapsed` only with independent latent diversity evidence; missing evidence is `unqualified`.

The existing all300 checkpoint is only a negative control. Its canonical audit is [COLLAPSE_AUDIT.json](/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1/COLLAPSE_AUDIT.json): RGB is below `0.01` for 100% of entries while latent effective rank is about `9.52`, so the measured failure is output collapse, not a representation-collapse claim.

## Explicitly out of scope

No mapper, DMX, video, body inference, D4/D6, post-hoc smoothing, arbitrary gain, product promotion, multi-seed campaign, or artistic conclusion is part of this protocol. Authenticated dance bodies are permitted only in the masked training-time auxiliary teacher branch described above.

## Dance as an auxiliary motion modality

Dance remains part of foundation-v2 as a privileged training signal. It is not an inference input and cannot supply fallback motion. The current targetable corpus has authenticated saved native-C bodies for 8 rows: `1023`, `10dcc`, `1181f`, `11b01`, `1672d`, `210e3`, `21c3a`, and `3b4ab`. On those rows, the teacher can supervise body latent, body-derived fixture aim, speed, and acceleration proxies. The remaining target rows receive no fabricated dance target; their auxiliary loss is masked out.

The paired control is `target_only` on the exact same grouped split, seed, windows, and MuQ+Base92 inputs. A dance arm may be retained only if it does not cause output collapse and does not regress the target heads beyond the preregistered tolerance. Lower dance loss alone is not evidence of a useful light model. Inference remains audio plus fixture metadata only.

## All-dataset roster and source masks

The moved repository data is part of the plan, but each corpus keeps its own contract:

- **BeatSaver authored-light (300 rows):** the primary full MuQ+Base92 light arm. Exact-audio duplicates are grouped; 8 rows have authenticated dance auxiliary bodies.
- **RPMC-L2 compact (140 shards, 379,125 frames):** 512-D cached audio `x` and 6-D authored scene `y`. The compact cache has no source waveform/MuQ binding, so it enters only through an explicit RPMC source adapter and source mask. It is never reported as full MuQ+Base92.
- **AIST++ keypoints3d (1,408 usable clips, 60 music IDs, 30 dancers, 21 choreographies):** dance-only auxiliary source. `keypoints3d` and `keypoints3d_optim` supervise motion representations; there are no authored-light labels.
- **Native JD/FD admissions:** 651 unique JD feature streams, 3,688 admitted JD train segments, and 203 FineDance records. They supply masked dance/audio auxiliary supervision from permitted train partitions. Protected validation/test partitions remain untouched until the collapse audit.
- **Repository audio:** 60 WAV files under `/mnt/lightsem-data/repository_data/wav/wav` are bound by exact hash to the AIST/keypoint records before any MuQ extraction.

The model must carry `source_id`, modality-presence masks, and per-source loss accounting. The collapse receipt reports BeatSaver full-contract, RPMC adapter, AIST dance-only, and JD/FD auxiliary slices separately; a pooled score cannot hide a collapsed source branch.

## Channel and MuQ compatibility rules from old experiments

The old caches are not interchangeable:

- `beatsaver_audio/*.npz` is the canonical 92-D raw bus: 64 mel channels plus 28 named scalar channels at 40 Hz.
- `beatsaver_visual/*.npz::x` is a separate 52-D causal operator bus at 10 Hz. It is useful for legacy operator controls, but it is not Base92 and cannot be concatenated with MuQ under the full contract.
- `native_c_cli/*/features/base92/input.npz::x30` is a 30 Hz timestamp-resampled 92-D bus. On the eight rows shared with the old foundation runs (`1023`, `10dcc`, `1181f`, `11b01`, `1672d`, `210e3`, `21c3a`, `3b4ab`), its channel values match the raw 92-D cache after timestamp interpolation (minimum per-channel correlation `0.999892`).
- The old eight-song foundation used real MuQ layer 9 plus this 92-D Base92 contract. The all300 checkpoint instead appended 1,024 zero channels to raw Base92; that recipe is excluded from v2.

Every artifact and model input must carry `channel_layout`, `source_clock_hz`, `muq_layer`, `muq_timestamp_hash`, and `base92_waveform_sha256`. A 52-D artifact or a zero-padded 1116-D tensor fails contract validation before training.
The detailed compatibility receipt is [CHANNEL_MUQ_AUDIT.json](/root/lightsem/docs/research/2026-10-16-audio-light-foundation-v2/CHANNEL_MUQ_AUDIT.json).

## Current execution state

The full canonical feature contract is now materialized. The prior all300 run remains a negative control because it zero-filled the MuQ half of the interface. A new 2,500-step all300 target arm uses the canonical 1,116-D input and reports healthy temporal latent rank but collapsed RGB/fixture heads on held-out songs; its verdict is `output_collapsed`. A separate 1,000-step continuous-attention diagnostic used the full 1,116-D interface on the eight authenticated dance rows; its raw learned aim also collapsed spatially and exceeded the raw speed/acceleration/jerk safety gate. The resulting 60-second file is therefore a quarantined presentation diagnostic using a fixed spatial anchor plus a bounded learned residual, with no temporal fallback oscillator. The derived controls pass the slow-motion/different-side presentation audit, but they do not override the failed raw-body gate and are not a trained product result: [DEMO.json](/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v3_canonical/DEMO_3B4AB_60S_10HZ_FINAL/DEMO.json). The full-contract arm receipt is [RESULTS.json](/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_all300_canonical_20261016_v1/RESULTS.json), and its canonical collapse audit for the dance arm is [COLLAPSE_AUDIT_CANONICAL.json](/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v3_canonical/COLLAPSE_AUDIT_CANONICAL.json).

## Repair probe — canonical v9

The v9 repair keeps the same canonical MuQ+Base92 input and masked dance
teacher, integrates a fixture-metadata spatial anchor into the model, and
bounds the learned temporal residual at `0.10` under `max_velocity=0.18`,
`max_acceleration=0.5`, `max_jerk=1.5`. On independently recomputed `21c3a`
and `3b4ab`, the raw body passes the finite, spread, side-count, duplicate,
slow-motion, deterministic-repeat, and audio-swap gates: spread p50 is
`0.686–0.724`, side-count minimum is `7`, speed maximum `0.0255`, acceleration
maximum `0.182`, jerk maximum `2.06`, and RGB standard deviation is
`0.116–0.191`. The audio swap changes angle/RGB/brightness/event outputs. A
separate latent representation-collapse measurement is still unqualified, so
this is a diagnostic candidate rather than a promoted foundation checkpoint.

The raw-output demo is
[DEMO.json](/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v9_warm_v7/DEMO_3B4AB_60S_10HZ_FINAL/DEMO.json), with video at
`/mnt/lightsem-data/experiments/lightsem/foundation_audio_light_v2_continuous_20261016_v9_warm_v7/DEMO_3B4AB_60S_10HZ_FINAL/demo.mp4`.
It uses the model's own raw angle output, no external temporal fallback, and
records `native_dmx=false`; promotion remains forbidden until the latent gate,
own-song body provenance, and the full native mapper/DMX gates are run.

The next training run must keep Base92 and MuQ source hashes attached, add the
masked dance teacher only on its eight authenticated rows, add an independent
temporal-latent collapse measurement, and rerun the full gate before any
native mapper, DMX, or product render. RPMC-L2, AIST, and JD/FD remain separate
source-adapter or dance-only branches with per-source collapse receipts; they
are not concatenated into the BeatSaver 1,116-D contract.
