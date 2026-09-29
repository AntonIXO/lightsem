# LightSem optics: closer-camera visual audit

## Scope and result

**Bounded AI visual audit completed. No perceptual, human, or music acceptance.**
This audit covers the six moving `closer_` arms. It excludes the static bright foil and the brightness-matched pair. It uses the same frozen stage02 DMX, not newly learned motion.

The narrower angular beams make individual shafts and endpoints distinguishable in these samples. The fixed horizontal bulb row does not move. Beam bodies and endpoints do move. The overall fan-like arrangement remains repetitive. The controls change presentation, not the underlying angle sequence.

## Exact viewed coverage

For **each** of these arms:

- `closer_native_legacy`
- `closer_native_angular_peak`
- `closer_native_angular_flux`
- `closer_zero_legacy`
- `closer_zero_angular_peak`
- `closer_zero_angular_flux`

I viewed all ten chronological sheets: `review_000.jpg`, `030`, `060`, `090`, `120`, `150`, `180`, `210`, `240`, and `270`. Each arm contributes **278 samples at integer seconds 0–277**. The first nine pages have 30 samples; the last has eight. Total: **60 sheets, 1,668 sample tiles**.

The frozen script downsamples 640×360 RGB to **320×180** tiles using `cv2.INTER_AREA`. Each original sheet is 960×1800. `attach_image` further resized every sheet to **640×1200 JPEG quality 82**. Effective viewed tiles were about **213×120**. Small details, low-light shafts, and rapid motion are therefore poorly measured by the sheets.

I also viewed **all 270 extracted native 640×360 PNGs**: 15 chronological frames per fixed two-second interval, three intervals per arm. No attachment resizing was reported for these PNGs. No exposure, gamma, crop, or brightness adjustment was applied.

| Interval | Zero-based source frame indices, inclusive, step 4 | Exact sample times |
|---|---|---|
| [80,82) | 2400,2404,…,2456 | 80 + k×4/30 s, k=0…14 |
| [196,198) | 5880,5884,…,5936 | 196 + k×4/30 s, k=0…14 |
| [270,272) | 8100,8104,…,8156 | 270 + k×4/30 s, k=0…14 |

This is **7.5 Hz sampling**, not continuous playback. Each arm has 278 sheet samples plus 45 fastframe presentations. Three timestamps overlap, so these represent **320 distinct timestamps per arm**, not all 8,319 video frames. No audio was listened to.

## Concrete observations

| Arm | Observed positive visibility | Remaining limitation / negative observation |
|---|---|---|
| closer_native_legacy | Six broad blue/violet beams and side-pool changes are visible after the dim opening. | Diffuse lower edges and central haze merge. Local movements do not make a new layout. |
| closer_native_angular_peak | Straighter bounded edges and separated central oval ends reveal individual shafts. | Dim opening and ending remain hard to inspect. Less surrounding fill is not a quality result. |
| closer_native_angular_flux | Same angular geometry has more visible beam bodies than native constant-peak. | Basic fan and fixed source row remain. Brightness is not matched. |
| closer_zero_legacy | Some narrowing versus native legacy; side endpoint changes remain visible. | Soft boundaries and low spill still blur separation. Zoom-only does not remove legacy softness. |
| closer_zero_angular_peak | Native PNGs reveal very thin shafts and small endpoint shifts. | At sheet scale the opening is nearly just bulb dots. Even native ending frames have very low beam contrast. This is not evidence that angles stopped moving. |
| closer_zero_angular_flux | Thin, bright, discrete shafts and small endpoints are visible, including in the dim ending interval. | Upper shafts have visible stippled/granular texture. Brightness and width both differ from other arms; no preference follows. Overall arrangement remains similar. |

Across all six chronological sequences, seconds 0–59 are dark blue, the 60s rise into clearer blue/violet beams, much of 150–259 is violet, and 260–277 contains strong dimming variations. These are visual observations only, not music alignment findings.

At **80–81.8667 s**, outer shafts and their lower endpoints shift laterally. Their overlaps with adjacent beams change. The central pair remains close to vertical. At **196–197.8667 s**, the center and side endpoints shift. In the final sampled frame, the rightmost endpoint rises and the shaft shortens, separating from its long neighbor. At **270–271.8667 s**, brightness falls and recovers while the outer endpoints drift. These changes are visible in each arm, with very different visibility. None requires movement of the bulb row.

## Separate the factors

- **Camera-only:** all six audited arms use the same fixed closer camera: scale 0.45, position [0,6,11], target [0,2,-3], orbit 0. They provide no internal camera-only contrast. The design's camera-only comparison holds zoom and cone mode fixed against the corresponding `legacy_` arm. Those legacy-camera images were not viewed in this audit. Do not attribute width/brightness contrasts within this group to camera motion.
- **Zoom-only:** compare `native` with `zero` at fixed camera and edge/flux mode. This changes fixture cone zoom, not camera zoom. The angular zero arms have much thinner shafts than their angular native counterparts. Zero legacy retains diffuse edges.
- **Edge-only:** compare `legacy` with `angular_peak` at fixed zoom/camera. Angular boundaries are visibly more bounded and central endpoints more distinct. This also changes displayed support/brightness, so it does not establish preference.
- **Constant-peak versus flux:** compare `angular_peak` with `angular_flux` at fixed zoom/camera. Flux visibly strengthens beam bodies, especially at zero zoom. It does not create a new angle path or move the source row. The registered reference is 14 degrees; this is not hardware-lumen calibration.

Receipt mean display luma outside bulb masks, on the 0–255 scale, is 3.1781 / 2.3136 / 3.1418 for native legacy / peak / flux, and 2.1851 / 1.2651 / 2.9368 for zero legacy / peak / flux. These unequal levels explain why visibility cannot be treated as a brightness-controlled quality ranking. No matched-pair inference is made here.

The **2.3958× raw geometry-proxy result** passes the registered ≥2× proxy threshold against `legacy_native_legacy`. It measures projected midshaft movement relative to half-peak cone radius in the registered interval. It is **not perceptual acceptance** or proof of new learned movement.

## Provenance and limits

All six `render.json` hashes matched their sidecars. All 60 review-page hashes matched the render receipts. All six video SHA256 values matched the receipts. All six existing full media audit receipts passed their stored exact-media checks and their own sidecar hashes. I did not rerender or repeat the full media audit. The render receipt's old pending-review/media labels were not edited.

All six receipts report identical source and arm angle SHA256:
`3148e7fde21b4d91e73884e1f0c8fe6597e88eb701d2d368f2d42ffc2026a8fe`.
Native arms change no DMX bytes. Zero arms change only zero-based zoom channels 9,21,33,45,57,69; their nonallowed bytes remain identical.

CPU extraction used software FFmpeg with `-hwaccel none`, decoder and output `-threads 2`, and `-filter_threads 2`. It sought to each registered interval, selected every fourth decoded frame, and produced 15 images per interval. The exact commands, **all frame indices, timestamps, absolute paths and PNG SHA256 hashes** are in:

- `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/closer_camera/fastframe_index.json`
- PNG path template: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/closer_camera/fastframes/<arm>/<start>_<end>/frame_00.png` through `frame_14.png`.
- Receipt, video, and all review-page paths/hashes: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/closer_camera/verification.json`.
- Existing media receipt verification: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/closer_camera/media_receipts_verified.json`.
- Append-only observation log: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/closer_camera/observer.jsonl`.
- Immutable observation snapshot: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/closer_camera/observer_snapshot_01.json`.
- Extraction script/log: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/closer_camera/extract_fastframes.py` and `extract_fastframes.log`.

The PNGs decode the compressed AV1 videos; the sheets sample pre-encoding RGB. Their color/texture instruments differ. Granularity is observed, not causally assigned to rendering versus codec. Sampling cannot establish continuous smoothness, transient absence, rhythm, AV synchronization, listening quality, music fit, human preference, or product acceptance. No GPU, vault, frozen-output modification, hardware output, or TEST access occurred.
