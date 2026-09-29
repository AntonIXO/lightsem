# Four comparison demos — implementation readiness V1

2026-09-28. **Feasible, conditional on a small versioned delivery driver and its preflight. No new training is needed.** This review ran no model, test, extraction, rendering or vault call. It edited no source. It read the existing demo source/bindings, completed metadata, saved NPZ headers and hashes.

## Fixed delivery set and claim

Render these selected seed0 checkpoints in this order, regardless of previous or new numbers:

1. `S_none_P0`
2. `S_V6_P0`
3. `I_VR_P0`
4. `S_VR_P0_Ashuf`

Keep the Sep22 `S_VR_P0` full video as the unchanged fifth anchor. **Do not rerun its inference or render.** This delivers every other tested recipe, not a winner-selection study, new fit, hypothesis closure, clean-LIGHT validation or generalization claim. The accepted movement remains good and unchanged. Learned color/beat behavior must be shown honestly; no new manual cue rule, gain, threshold, palette rescue or motion retiming is allowed.

**Ashuf interpretation:** use the exact same real Shpongle audio94 as the other demos. This is an explicitly named `recipient_audio_delivery` route for the audio-shuffled-trained checkpoint. Its registered primary observer evaluation instead used a tempo-donor route. The new video is not a reproduction of that primary scientific intervention. Do not access a dataset donor, search for a favorable donor, or imply its old AP applies to this real-audio demo.

## Exact checkpoints and bindings

All four checkpoint file SHA256 values were verified read-only in this review. All four completed receipts contain exactly the same `frozen_transforms` object as the existing anchor; not merely matching vocabulary sizes.

Binding file:
`/mnt/lightsem-data/experiments/lightsem/lighting-cue-model-20260922t094347z/observer_evaluation_binding_checkpoints_v1.json`
SHA256 `2334e2c9a8bafc1e5a7a0c0e42184c96d66113d3bfecc3e3b8288c03335aab53`.

Paths below are under the original campaign `observer_fits_v1/`:

| Order | Binding key | Path | SHA256 |
|---|---|---|---|
|1|`00_seed0_S_none_P0_selected`|`01_seed0_S_none_P0/best.pt`|`ef32122167bbc2721a776af9a205fe5bcd5c774e20f588a83242803f2a09c5c9`|
|2|`01_seed0_S_V6_P0_selected`|`02_seed0_S_V6_P0/best.pt`|`0c44d937a9a7b57ad4bd10f62bec970cd6722616ffb12f351c0af764799ce79f`|
|3|`03_seed0_I_VR_P0_selected`|`04_seed0_I_VR_P0/best.pt`|`8de865906cfabb49290971eccf6016ab38edebe885eec7e6369f020fb5544b8a`|
|4|`04_seed0_S_VR_P0_Ashuf_selected`|`05_seed0_S_VR_P0_Ashuf/best.pt`|`9f447f04debb9df2dd985b6fb5f4c21f1698d664eb6b7d1d902b029428b8be74`|

For every arm verify its registered `complete.json` hash, `status=complete`, exact recipe/seed/selected role, `best_weights_sha256`, checkpoint metadata bindings and frozen transforms. Never use `final.pt` or search for a better seed.

## Reuse these immutable inputs, by reference

- Existing plan: `/root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/inference_mapping_plan.json`.
- Existing provenance: `/mnt/lightsem-data/experiments/lightsem/shpongle_layers_demo_20260922T161917Z/provenance_manifest.json`.
- Saved `audio94.npz` in that data directory: SHA256 `224db0481c77677de7159e7c6f88cc42d8d861221c57e20e2b2072760dab6a9f`,6,616,205 bytes. Arrays: raw and normalized float32 `[11092,94]`, canonical float64 clock and original quantized float32 bus clock. **Do not rebuild audio features, rerun BeatThis, or copy this file four times.** Future preflight should verify saved normalization against the frozen normalizer without rewriting it.
- Frozen transforms: original campaign `observer_train_transforms_v1.json`, SHA256 `a4672acf0ff0b95fa3fad2b8a1f481810029f8bde4b30e10bfccfacc94740b39`.
- Common normalization hash `e964b95de95aa32ef861d9593449f860f6d163e53dc615aa6f05e28252df2def`; vocabulary hash `ab368f2288098028cf49180c99348fd751f6a4928b5ba86e07b8bbdf16d72839`.
- Accepted DMX `/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z/streams.npz`, SHA256 `f19c5721f350dbe56cafba4c26eb152fdfab6cb1627b310f16345de795812052`; keys `companion`,`time`. This is the movement baseline, **not** the anchor's already overlaid stream.
- Source soundtrack `/root/shpongle.flac`, SHA256 `95e4388424ef62950bdeae78e47b8e71839ed28c6e01189fe15f9ae6c902b6c0`.
- Unchanged anchor `/root/lightsem/outputs/shpongle_layers_demo_20260922T161917Z/shpongle_experimental_full_1080p_av1.mp4`; recorded SHA256 `a78a64342ecb2db36cc6ea614e62552ba1f334c1b524384ff6614b6a4c5256be`. Existing saved outputs can be used for read-only mapping parity, never to regenerate its model outputs.

All source files listed by the existing inference plan still match their frozen hashes. The companion JSON records source hashes and array-header metadata used in this review.

## Minimal new code

Create new files only under `docs/research/2026-09-28-end-to-end/` (or another explicitly bound versioned source directory). Suggested driver: `run_comparison_demo_v1.py`; independent checks: `test_comparison_demo_v1.py`. Freeze the driver/test hashes before any new model forward. Preserve every original source and artifact.

The new driver needs three bounded functions/phases:

1. **Preflight/bind:** select exactly the four keys above; check all source/input/checkpoint/receipt hashes, common transforms and clocks; construct the aggregate storage ledger and fixed output paths; reject existing outputs. No model forward in preflight.
2. **Infer/map/readback:** one complete-song forward per new arm; reuse the original mapping functions and frozen model helpers. Persist outputs once. Reconstruct/prove mapping without further model forwards. Resume from authenticated saved outputs if rendering is interrupted, never rerun a completed forward silently.
3. **Queue render/audit:** sequentially call the unchanged existing `render_demo.py` on each new `overlay` stream, then its `audit` mode. An aggregate wrapper enforces the new1.5GiB media cap. Use a per-arm media subdirectory so the old helper's512MiB per-directory check does not accidentally count all four; the new aggregate guard must still count all four.

Do **not** directly run original `infer_overlay.py`: its global plan/output, `S_VR_P0` assertions,64MiB local guard and audio copy are hard-coded for the completed anchor. Reuse/import its pure `palette_rgb` logic with hash binding, or copy that exact small function into the new versioned driver with a parity check; never alter the original. Reuse frozen `generated_history`, `candidate_peaks`, `apply_overlay`, `apply_accent_events` and `load_checkpoint` unchanged. The old `verify_readback.py` likewise needs a new adapter because it hard-codes the original directory and duplicated baseline array.

Keep the persisted new schema small: no per-arm audio copy, no per-arm baseline copy, and no redundant `palette_probabilities` array (softmax of saved logits). Keep logits, scores, context, generated/previous IDs, color-cue mask, clocks, full `overlay` DMX and the data needed for exact independent RGB/dimmer reconstruction. Put the immutable baseline reference/hash in the receipt and load that for readback. The old renderer needs only full `[11092,512]` uint8 `overlay`, so this does not require changing it. Avoid duplicating long event lists in JSON; the NPZ is the event evidence, JSON carries bounded counts/hashes.

## Factory and architecture constraints

Use frozen `training.load_checkpoint(path,device='cuda',expected_bindings=...)`. It builds from `reference_model_config` and validates realized config/parameter-match metadata. **Do not call the independent factory using its saved realized width76**: that would match/narrow it a second time.

| Recipe | Architecture | Reference/realized width | Visual auxiliary dimension |
|---|---|---|---|
|S_none_P0|shared|128/128|0|
|S_V6_P0|shared|128/128|6|
|I_VR_P0|three independent encoders|128/76|96|
|S_VR_P0_Ashuf|shared|128/128|96|

All use94 audio inputs, palette32, pose auxiliary0, max_frames40=12000. The11092-frame full song fits this explicit model limit. Independent `context10` has76 columns; do not assert128 for every output. Visual heads exist in the checkpoint but are inactive in `eval()`. No video/pose is needed at inference. Strict state-dict loading must retain the correct0/6/96 auxiliary shape; do not discard heads to load a different architecture.

Preserve original numerical mode: seed0, float32, no autocast, TF32 disabled, cudnn benchmark disabled, `eval()`, no gradients, one complete `[1,11092,94]` input. No chunking or alternate precision rescue. The old full-song shared forward plus rollout took0.188s with135,455,232 bytes peak allocated; that does not measure the independent model. One process/GPU job at a time, cleanly release each model, verify actual VRAM/process/log externally. A mismatch/OOM stops that arm with a receipt; no unregistered architecture/precision change.

## Mapping and protected bytes stay exact

- Accent: unchanged `candidate_peaks`, full `[0,277.3)` region,>=0.5 threshold,0.2s NMS/refractory, fixed0.35 multiplicative gain increment,0.25s raised-cosine width, overlaps take max, only MH/wash dimmers. This is the already frozen display adapter, not a new learned-strength claim.
- Palette: own generated history, null initial state, argmax marks, original strict color threshold and0.5s cue NMS; no gold state. Original HS72 complex mean to HSV/RGB. Baseline RGB remains until the first **actual** color cue; held thereafter. No cue means no fabricated cue.
- Same global palette assignment to MH/wash/projector RGB; no manual fixture-arrangement improvement. Do not tune a bad palette after viewing the result.
- Independently enumerate the rig's43 allowed slots and469 protected slots. Compare all5,202,148 protected bytes for each arm with accepted `companion`. Angle/strobe/gobo/zoom/projector-dimmer/blinder bytes remain exact. Verify baseline-zero dimmers stay zero; exact baseline RGB before first cue and exact dimmers outside pulse support; independent raised-cosine/RGB byte reconstruction.
- Existing renderer's `optical()` zeroes moving-head zoom **in its private render copy**, a frozen convention used by the anchor. Keep that same render-only behavior; it does not modify the emitted DMX artifact or authorize a new zoom intervention.

## Storage:32MiB scientific reserve is feasible

Charge all new derived data/predictions/DMX/receipts/source snapshot/logs against the **original campaign's** remaining63.97MiB. Prefer a new immutable subdirectory within that campaign, for example:
`.../lighting-cue-model-20260922t094347z/comparison_demo_20260928_v1/{01_S_none_P0,02_S_V6_P0,03_I_VR_P0,04_S_VR_P0_Ashuf}/`.

No cap reset. Starting952,976,108 bytes +33,554,432 reserved bytes =986,530,540, below original cap1,020,054,732. Unreserved headroom remains33,524,192 bytes. `/mnt` free1.254GB minus the full32MiB reserve remains about1.220GB, above1GiB. Recheck current counts before launch; charge other concurrent new artifacts too.

Old derived demo total9,576,807 bytes includes6,616,205 reusable audio bytes. Four old-scale nonaudio output sets would total11,842,408 bytes. This is an estimate, not a guarantee of compressibility. The proposed nonduplicating schema provides a stronger bound:

- Header-based arithmetic for four output sets, including full DMX overlays, all model arrays except redundant probabilities, no baseline copy, independent context76 rather than128, and deliberately loose event-array bounds: **32,069,076 raw bytes** before new container/receipt overhead.
- Reserve at most1MiB aggregate bounded metadata/source/logs and128KiB aggregate NPZ/deflate/header overhead: **33,248,724 bytes**, below32MiB by305,708 bytes. This deliberately pessimistic bound does not rely on the old compressibility.
- Build each NPZ in bounded RAM, check its exact serialized length before an exclusive write, and enforce aggregate32MiB plus original0.95GiB plus free-floor checks. Bound metadata/log verbosity; do not retain duplicated JSON event arrays. If schema/size violates the bound, stop before writing, preserving completed outputs; do not expand cap or rerun inference to hunt for smaller outputs.

If the implementer keeps the original duplicate-baseline/probability schema instead, expected storage still fits but the same strict no-compression bound does not. Prefer the small schema above, with independent mapping checks. Checkpoint weights/prototypes/audio stay referenced, not copied.

## Media: separately authorized1.5GiB root reserve is feasible

Proposed root path:
`/root/lightsem/outputs/shpongle_comparison_20260928_v1/{01_S_none_P0,02_S_V6_P0,03_I_VR_P0,04_S_VR_P0_Ashuf}/`.

Four old-scale full MP4s:4×223,607,757=894,431,028 bytes; old full-sheet audit adds about3.5MB per arm. Reserve250–350MB per arm, with a hard aggregate1.5GiB=1,610,612,736 bytes including audit sheets/encode logs/partial media. Root free6.31GB minus the full reservation leaves about4.699GB, above1GiB. Maintain the original helper's400MiB launch reserve and512MiB per-arm cap **and** the aggregate guard. Never delete an old anchor or a failed render automatically.

Use unchanged1920×1080/30fps AV1-NVENC settings, source AAC audio,277.3s/8319frames, fixed camera/exposure/role gains, generic experimental overlay label; recipe names are explicit in filenames and comparison index. No brightness matching by new dimmer rescaling: these are fixed-policy delivery videos, not blinded preference evidence. If a future controlled human comparison needs<=3% encoded luma matching, that is a separate registered material protocol.

Old render time193.53s per full video suggests about12.9min for four renders, plus readback/full-decode/audit overhead. Inference is much smaller than rendering. No need to wait for new fits, annotations, hypothesis closure or original quality gates to deliver these honestly labeled existing-model demos.

## Required checks/binding sequence before execution

1. Freeze comparison plan: four ordered checkpoint keys, recipient-audio route for all, unchanged anchor, exact input/source hashes, requested counts/dtypes, mapping parameters, output directories,32MiB scientific and1.5GiB media reservations. Freeze new driver/test/source hashes and native environment wrapper. These are delivery bindings, not new-fit preregistration or outcome claims.
2. After code is authorized, run only bounded synthetic/software checks in the project environment: accepted-DMX identity;43/469 independent slots; RGB prototype primary; no-cue/first-cue hold; accent overlap/bounds/zero dimmers; generated-state chronology; immutable/no-clobber/aggregate-cap failure controls. Test independent loader config/expected metadata without model forward and reject double narrowing. No sealed TEST access.
3. Future preflight verifies all four completed/checkpoint bindings, saved audio normalization/shape/clocks, exact common transforms, baseline/source identity, fixed mapping helper hashes and the original anchor hashes. No model forward. A source/config mismatch stops before loading GPU data.
4. Authorize sequential four-forward delivery only after those receipts pass. Each arm records `forward_calls=1` for its full audio encoder call, source/checkpoint/audio hashes and numerical mode. Palette-head evaluation inside generated-history is part of that same defined rollout; do not misstate it as one total neural operation.
5. Save once, then independently read back/reconstruct mapping with **zero** model calls. Verify469 protected slots, finite output, state chronology, no invented cues, exact palette/dimmer byte mapping. A saved-output failure is not permission for an automatic second forward.
6. Render each completed verified arm, then full software AV decode,8319-frame/timestamp checks, audio duration/PCM hash comparison to the anchor, immutable artifact receipts and versioned links. Keep poor/flat/no-cue results in the delivered set. No aesthetic success or default promotion is implied.

Run through wrappers, never direct venv executables: inference/checks use `TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh`; existing rendering/audit use `bash run.sh .../render_demo.py`. The renderer already imports `StageRenderer` and canonical frozen helper functions. Do not edit them merely to put the new recipe name on screen.

## Principal source files

- Existing demo `infer_overlay.py` SHA256 `5e868792c94ccdea9519222f15ef8031fd2d4bc32abf650773dc208074d50c3b`.
- Existing demo `render_demo.py` SHA256 `19dd57aec907696e7293587e4bfd49c72a47f884f852f16678748f415e4cba52`.
- Existing demo `verify_readback.py` SHA256 `e7b34db37e9e73de2eb1b61d448bb5d2f4b5ff3b804e6e49a27017b52065c29c`.
- Frozen `training.py` SHA256 `31e58e8c7a10cd80f0d3b5be0e87796e35548f2c2d03705856a0cafa1811802b`; `load_checkpoint` lines1289–1304; `generated_history` lines735–778.
- Frozen `model.py` SHA256 `aa1d63f0cbf5cf270c3c7d676edf3eaf3debeb394220123493a400aa60835529`.
- `run_rhythm_companion.py`: `render_config`, `optical`, `encode_command`, `audit_movie`; SHA256 `eb21814d42a8e76c296193ebb67b5e965e20ff76602cc961ab695bdf7fccdd97`.
- Complete current source hash map and reusable input hashes are in `COMPARISON_DEMO_READINESS_V1.json`.
