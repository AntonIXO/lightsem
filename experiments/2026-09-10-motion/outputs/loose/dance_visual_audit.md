# Existing dance-render visual audit

**Scope:** post-result audit of `timing_revision_02`, `actuator_budget_01`, and `stage_geometry_01` only. No new candidate, model run, GPU work, source change, vault write, or future stage02 review. This is supporting evidence, not the system of record or pipeline completion.

## Instrument and limits

- Read `CLAUDE.md` and `docs/research/2026-09-10-motion/DANCE_DIAGNOSIS.md`.
- Used `attach_image` to inspect all 30 supplied sheets in chronological row-major order: 278 samples/video, seconds 0–277, **834 samples total**. The last sheet has eight samples, not 30. Inline attachments were resized; dark detail and small beam differences are hard to resolve.
- Also inspected all 18 exact decoded stills at 75, 160, 194, 197, 200, and 240 seconds (frame index = seconds × 30). No exposure boost was applied.
- Used native `bash run.sh` for CPU FFmpeg/ffprobe decoding, timestamp checks, and decoded-pixel measures on every video frame. Full decoding is **not continuous human audiovisual watching**. One-hertz samples cannot establish smoothness, flicker absence, fast timing, or music fit.
- This is an unpreregistered descriptive review. It does not replace frozen mechanics/bandwidth gates. Aesthetic preference, dance fidelity, human enjoyment, and music-model qualification remain unmeasured. No fixture or safety certification occurred.

## What is visible

**Common timeline.** At 0–59 seconds the stage is extremely dark, with faint blue beams. Visibility grows in the 60–89s sheet. From 90s onward the same six-source blue/purple, then magenta lighting family dominates. The 150–259s sheets are easier to inspect. Near 263s and in the final sheet, the output dims again. A mostly black surrounding image and faint lower-stage wash remain in every comparison.

**Timing02, against surface rev02.** The two sides look close. Beam angles and overlaps change, but the crown/floor-fan remains. Central concentration in the 194–201s interval remains on both sides. This sample review cannot verify the separate signed-lag result or judge whether movement feels better.

**Actuator budget01, against timing02 / simulated 30/80 caps.** The faster simulated 90/360 side shows orientation and lower-stage streak differences across samples. It still repeatedly forms the same crown/fan. The 194, 197, and 200s stills do not show removal of the contraction. Faster allowed mechanics are not a demonstrated aesthetic or music-fit improvement.

**Stage geometry01, against the faster floor mapper.** This is the clearest visible geometry change. The right side more often keeps separated, upright, broadly spaced cones. In the 194, 197, and 200s stills, the left beams converge into a compact bright center; the right retains a broad row. The chronological 180–209s sheet supports this localized difference. However, the output still reads as a repeated crown/row family across the track, with little obvious surface variety from this camera. It does not visibly establish a rich floor/back-wall vocabulary. The right side is less concentrated, not simply brighter: its pixel contrast and mean luma are lower even while more ROI pixels exceed the fixed luma threshold.

The existing `stage_geometry_01/chart_diagnosis.json` reports **raw requested** wall fractions of 0.424% full track and 1.125% active. Those are not decoded pixels or emitted first-hit fractions. They support caution about claiming wall use; this audit does not independently reconstruct ray hits.

## Full decode and media integrity

All three MP4s pass full video+audio FFmpeg decode with exit 0 and empty error logs. All three contain:

- AV1, 1280×360, 30 fps, **8,319 decoded frames**, video duration **277.300000s**.
- Video timestamps 0 to 277.266667s, strictly increasing; adjacent differences 0.033333–0.033334s.
- AAC stereo, 44,100Hz; audio stream start 0, duration **277.292698s**. Video/audio declared end difference is 7.302ms. This establishes container coverage, not perceptual sync to musical events.


Decoded audio checks also match across all three files: identical PCM s16le SHA-256, 48,914,432 bytes each. The 11,942 decoded AAC frames start at 0s; maximum inter-frame timestamp gap residual is under 1 microsecond. The final decoded frame extends to 277.292698s (rounded). Identical audio does not prove alignment of light events to music.

**Receipt discrepancy:** all `render.json` files say `generated_frames=8320` and duration 277.306667s. The encoded files contain one fewer frame. There is no interior timestamp gap in the measured video streams. The cause of the discrepancy was not established here; do not report 8,320 encoded frames.

The first still-extraction attempt failed because this FFmpeg build removed `-vsync`. The audit-only command was changed to `-fps_mode passthrough`; the full retry completed successfully. Both logs are retained.

## Decoded lower-stage pixels

Fixed ROI for each 640×360 half: **x=[160,480), y=[100,320)**. Same ROI, size, thresholds, and all-frame weighting for every file and side. It includes beams, lower-stage light and black surround. Top captions are excluded. Measures use decoded RGB24, with luma proxy `0.2126R + 0.7152G + 0.0722B` on 0–255 code values. This is not linear luminance, calibrated photometry, or an aesthetic score. The threshold Y≥16 is descriptive, not an acceptance threshold.

| Comparison | Mean luma before→after | Mean spatial luma SD before→after | ROI fraction Y≥16 before→after |
|---|---:|---:|---:|
| Timing02 vs surface rev02 | 3.680→3.659 | 8.245→8.240 | 6.565%→6.483% |
| Faster actuator vs timing02 | 3.659→3.716 | 8.241→8.285 | 6.484%→6.685% |
| Stage geometry01 vs faster floor | 3.716→3.570 | 8.285→7.536 | 6.685%→6.999% |

Across **all 8,319 frames of each side**, the ROI fraction with all RGB channels≥250 is zero. The fraction with any RGB channel≥250 is also zero. Thus no decoded near-white/channel clipping at this threshold was found **inside this ROI**. This says nothing about pixels outside it or pre-encoding/HDR clipping.

In **194≤t<201s**, stage geometry01 versus faster floor changes mean luma **5.995→5.719**, spatial SD **14.310→11.090**, and Y≥16 area **9.270%→12.011%**. This supports a less concentrated, more distributed image—not an improvement bought by higher mean brightness. The first 60s after-side mean luma is only 0.357–0.378 versus 4.612–4.831 in 120–180s. Low early visibility is real at this exposure, not proof of absent motion.

The same intermediate candidate appears in adjacent comparisons, but decoded ROI means differ slightly across encodes. Compare each candidate to its own displayed baseline; do not assume pixel-identical cross-file controls.

## Narrow visual verdict and missing evidence

- **Reviewed geometry:** timing02 and actuator budget01 do not remove the crown/floor-fan. Stage geometry01 reduces the sampled 194–201s central contraction relative to the faster floor mapper. This is a directional, local geometry result only.
- **Broad visual acceptance:** not established. Repeated crown/row geometry remains. No claim that the lights now “dance.”
- **Aesthetics and music fit:** unmeasured, not silently passed or failed by a pixel metric. Need continuous synchronized human viewing/listening and a defined, comparator-based rating protocol.
- **Not established here:** sub-second visual continuity/flicker, perceptual music synchronization, actual wall first-hit frequency, all-camera stage coverage, independent pose distinctions, generator generalization, physical fixture tracking/safety, or product readiness.
- **Pipeline status:** not complete. This audit does not authorize TEST use, training, default promotion, hardware output, or closure of broader work.

## Reproducible artifacts

Directory: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/visual_audit/`.

- `review_record.json`: reviewed sheets, hashes, exact-still times, scope and verdicts.
- `decoded_measurements.json`: MP4 hashes, ROI definitions, frame/timestamp checks, full-track and interval metrics.
- `*_ffprobe.json`, `*_timestamps.json`, `*_full_decode.log`: native media evidence.
- `*_perframe.npy`: six metric columns listed in the measurements JSON, one row per frame.
- `*_075.png` through `*_240.png`: the 18 inspected stills.
- `measure_existing.py`, `measurement.log`, `measurement_retry.log`: audit procedure and retained failed/successful receipts.
- `audio_check.json`, `check_audio.py`, `audio_check.log`: additional audio timestamp and decoded-PCM checks.
