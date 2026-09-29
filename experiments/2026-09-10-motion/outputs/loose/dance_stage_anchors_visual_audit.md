# Stage geometry02 independent visual/evidence audit

## Scope and comparator

**Left: `stage_geometry_01`. Right: `stage_geometry_02`.** The on-screen labels are “STAGE CHART ORIGINAL ANCHORS” and “STAGE CHART CALIBRATED ANCHORS”. “Original anchors” does **not** mean the originally rejected `surface_revision_02`. The frozen stage02 design names stage01 as its baseline.

This is an unpreregistered descriptive audit of the existing matched `before_after.mp4`. It is supporting evidence, not a vault closure. CPU-only native `bash run.sh` scripts called FFmpeg/ffprobe. No GPU, EGL, NVENC, training, new rendering, production/default changes, vault writes, or TEST access occurred. Existing artifacts were preserved.

## What was actually inspected

- All **10 existing chronological sheets**, `review_000.jpg` through `review_270.jpg`, in row-major order.
- **278 paired 1Hz samples**, seconds 0–277 inclusive: 556 side-images. The final sheet contains eight occupied sample cells; the remaining black cells are padding.
- **10 exact decoded paired stills** at 30, 75, 120, 160, 194, 197, 200, 240, 263 and 277 seconds. Selection used frame index `30 × second`. No exposure boost was applied.
- **All 8,319 encoded video frames** were CPU-decoded and measured. All audio was decoded. This is not continuous human audiovisual watching or listening.

Attachments resized the 1920×1800 sheets to 1200×1125 and 1280×360 stills to 1200×338. Early dark detail and small beam changes are difficult to resolve. One-hertz review cannot establish sub-second smoothness, flicker absence, fast timing, or musical synchronization. No whole-movie aesthetic/music rating is claimed.

## Sampled visual findings

**Timeline.** Both sides are very dark blue in 0–59s. Beams become easier to see in the 60–89s sheet. Blue-purple gives way to a brighter magenta family, most legible in roughly 150–259s. Visibility drops near 263s and the last eight samples are dim again. Angle and gap changes occur, but the same small row of six fixed source bulbs, broad cones, and faint lower central wash dominates.

**Geometry change versus stage01.** Stage02 visibly changes cone slants, overlap, and lower-edge height. At 75 and 120s its central patch ends higher and the outer bundles lean more than on the left. At 194, 197 and 200s, the right has a lifted central patch and dark gaps beneath/between groups. These are visible spatial differences, not only caption changes. They do not form a clearly different stage-wide shape family.

**Separation versus collapse.** Both sides already retain a broad spread in the sampled 194–201s interval. Stage02 does not newly rescue a collapsed stage01 comparator. It preserves separation while redistributing patches. The stronger earlier claim that stage01 reduced collapse versus the faster floor mapper belongs to a different comparison. It must not be credited again as the incremental gain of stage02.

**Motion/shape family.** Across all sheets, stage02 still reads as a recurring crown/row/fan with grouped cones. Some sampled poses show distinct gaps and slants, but this review does not establish a richer gesture vocabulary or recognizable body-driven choreography. This is a statement about sampled visible shape, not proof that motion is absent or a music-fit failure.

**Camera coverage.** This one fixed wide view leaves most of each half black. The lit arrangement remains a small central band, with a faint wash beneath it. The exact stills do not reveal a broad new vertical or floor-depth composition. Wall versus floor is hard to distinguish from the image alone. No alternate camera or head-only render was reviewed; all-camera coverage and head-only/full-scene separation remain unmeasured here.

The pre-existing `measurements.json` reports active70–90s emitted first-hit wall fraction **0.9583%→48.7917%**, and floor **99.0417%→51.2083%**, stage01→stage02. That is a geometric instrument result, not independently reconstructed in this pixel audit. It is compatible with lifted patch edges, but it does not imply that half the image becomes visibly occupied wall area or that the composition now dances.

## Media integrity: complete decode, not perceptual acceptance

- AV1, 1280×360, 30fps; **8,319 decoded frames**, duration **277.300000s**.
- Timestamps strictly increase from 0 to 277.266667s. Adjacent deltas are 0.033333–0.033334s; no interior timestamp gap was found.
- Full video+audio decode and raw RGB decode both exit 0, with empty error logs.
- AAC stereo, 44,100Hz; stream duration **277.292698s**, declared video/audio end difference **7.302ms**.
- **11,942 decoded audio frames**, first timestamp 0; decoded end 277.292697955s. Maximum inter-frame timestamp gap residual is 0.955 microseconds.
- Decoded s16le PCM is **48,914,432 bytes**, SHA-256 `cc282fdc912243ccbb21e7ae7692893a1198193df7deba1313673e6e805a176c`. This matches all three prior audit audio hashes, including stage01. No listening or event-to-music synchronization judgment follows from this match.

The original render receipt says 8,320 generated frames and duration277.306667s. The encoded movie has one fewer frame. Its measured timestamps have no interior gap; this audit does not establish why the generated/encoded counts differ. Do not describe it as 8,320 encoded frames.

## Caption-free decoded ROI

Same fixed ROI as the prior audit: in each 640×360 half, **x=[160,480), y=[100,320)**. It includes the central beams, lower wash and black surround, but excludes captions. Measures use decoded RGB24 before attachment resizing, not JPEG-sheet pixels. Luma is the code-value proxy `0.2126R + 0.7152G + 0.0722B` on 0–255, not calibrated or linear photometry. All-frame weighting and thresholds are identical for left/right.

| Interval | Mean luma stage01→02 | Mean spatial luma SD stage01→02 | ROI area Y≥16 stage01→02 |
|---|---:|---:|---:|
| Full encoded track | 3.570→3.484 | 7.536→7.463 | 6.999%→6.658% |
| 194≤t<201s | 5.718→5.401 | 11.091→10.281 | 12.017%→11.577% |
| First60s | 0.378→0.363 | 1.280→1.269 | 0.034%→0.034% |

Stage02 has slightly lower mean luma and thresholded area than stage01 in this ROI. The visual differences are not purchased by higher mean decoded brightness. Conversely, these numbers do **not** support a claim of larger decoded lit coverage. They are not a measure of choreography or preferred contrast.

Across **every one of the 8,319 frames on both sides**, the ROI fraction with any RGB channel≥250 is zero, and the fraction with all RGB channels≥250 is zero. Thus no decoded near-channel/near-white clipping was found **inside this ROI at this threshold**. This does not rule out pixels elsewhere, pre-encoding clipping, HDR clipping, or geometric beam intersections. Pixel thresholds are descriptive, not acceptance gates.

## Narrow conclusion and remaining gates

Stage02 visibly redistributes already-separated stage01 beams, including lifted patch edges and changed slants. It still shows the recurring row/crown/fan in the inspected samples. Neither the larger geometric wall first-hit fraction nor successful media decoding establishes dancing, musical suitability, or broad visual acceptance.

The 1Hz chronological sampled-review portion is complete. Continuous synchronized audiovisual perception, head-only review, camera-general coverage, fast continuity, music-fit scoring, generator qualification and real-fixture safety are **not** established here. Do not mark the whole visual protocol or broader goal passed from this audit. No default promotion or TEST authorization follows.

## Reproducible artifacts

Root: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_geometry_02/independent_visual_audit/`.

- `review_record.json`: all sheet hashes, sample seconds, notes, exact-still list and scope.
- `measure_existing.py`, `measurement.log`: native CPU procedure and successful receipt.
- `decoded_measurements.json`, `stage_geometry_02_{before,after}_perframe.npy`: every-frame metrics and column definitions.
- `stage_geometry_02_ffprobe.json`, `stage_geometry_02_timestamps.json`, `stage_geometry_02_full_decode.log`: stream, timing and complete decode evidence.
- `stage_geometry_02_030.png` through named `277.png`: ten exact stills listed above.
- `check_audio.py`, `audio_check.json`, `audio_check.log`: full PCM hash and audio timing checks.

Audited MP4 SHA-256: `6df7f38fa79c846d1e8105d1df141f3fbb98710d33e0927ebc21110fecdaf5d0`.
