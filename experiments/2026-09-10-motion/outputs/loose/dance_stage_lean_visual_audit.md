# Frozen signed-lean render: completion audit

**Planned coverage is complete. The primary signed-lean gate remains failed.**

This is a post-result technical implementation audit, not a new experiment or visual/music acceptance. LEFT is stage02 lean0. RIGHT is stage03 lean2. The same saved old Shpongle flow body, simulator 90°/s and 360°/s² profile, legacy camera and optics remain the comparison context.

## Technical audit

- Full software video + audio decode succeeded, with two-thread options and no GPU use. The decoder was libdav1d for AV1 and native AAC for audio.
- Video: **8,319 frames**, 1280×360 paired canvas, 30 fps. First PTS 0; last frame starts at 277.266667 s. All PTS are contiguous. End is **277.3 s**.
- Audio: 44,100 Hz stereo, **11,943 decoded frames / 12,228,930 samples**. All PTS are contiguous. End is **277.3 s**. The container's 11,944 coded frames include AAC priming (initial padding 1,024 samples); the final decoded frame contains 322 samples. This is not missing video or an audio gap.
- Frame-by-frame SHA-256 records cover both full decoded streams. FFprobe independently counted frames and recorded timestamps. All three audit jobs exited 0; ffprobe emitted no error text.
- This render uses exactly 8,319 whole frames, not the old protocol's 277.306667 s fractional tail. The 0.006667 s trim is disclosed engineering scope, not a new selection or scientific protocol claim.
- MP4 SHA-256: `7870081eeaa9d95e33c7aa73345a4d795c3601208e63e0ddda72d84198277527`.

## Silent visual coverage

All ten chronological sheets, `review_000.jpg` through `review_270.jpg`, were inspected. They contain **278 paired samples**, one per integer second from 0 through 277. Nine sheets contain 30 pairs each; the last contains eight. Its other **22 slots are black layout padding**, not observed video blackout. The 1920×1800 sheets were displayed at 1200×1125, so this review checks coarse composition only.

All **45 fixed native pairs** were also inspected:

| Half-open interval | Decoded frame indices | Coverage |
|---|---|---|
| 80–82 s | 2400, 2404, …, 2456 | 15 pairs |
| 196–198 s | 5880, 5884, …, 5936 | 15 pairs |
| 270–272 s | 8100, 8104, …, 8156 | 15 pairs |

Spacing is four frames, or 7.5 Hz. Each native 640×360 side was kept intact and stacked vertically into a 640×720 PNG: TOP baseline, BOTTOM candidate. No resize or exposure enhancement was used. Together with the sheets, this covers 320 distinct timestamps, not 323: 80, 196 and 270 s overlap.

### Observations

- At 0–59 s, both sides are dark blue and hard to distinguish at sheet scale. From about 60 s, blue/violet shafts brighten. Later sheets retain a purple fan with changes in spacing, slant and overlap; late samples become darker again.
- At 80–82 s, actual lower blue shafts shift slightly between sides and across samples. The central violet structure and broad fan stay similar. No large change in gesture height is apparent.
- At 196–198 s, the brighter shafts show small lateral shifts of lower ends and different slant/overlap more clearly. Stage03 changes these details, but does not establish a distinct new large-scale composition in this window.
- At 270–272 s, small shaft differences remain in the darker blue/violet structure. Low contrast limits the smallest-shift judgments.
- These observations concern **shaft structure and lower ends**, not the stationary fixture bulbs. Motion is not wholly absent. Nor do these samples establish a successful signed-lean fix.
- Overlay labels are intermittently missing or partial. For example, frame 2404 shows only `ST` at the top of the baseline and no candidate label. The side mapping remains fixed. The cause was not investigated; the render is not claimed to be presentation-defect-free.

## Result and limits

The pre-render primary measurement remains **0.4326 → 0.4801 px**, stage02 → stage03, or **1.110×**. It failed both the required 2× gain and 1 px floor. This is geometric projected inner-torso midshaft 1–2 Hz RMS over the active interval, **not actual pixel tracking**. It was read from frozen measurements, not recomputed here.

Silent samples are **not continuous audiovisual playback**, a music-fit judgment, or perceptual qualification. The full decode proves readability and timestamp continuity, not perceived synchronization or original-audio identity. A 1 Hz sheet cannot resolve 1–2 Hz motion. The fixed native samples provide local inspection only. No new physics, model-quality, hardware, or all-motion-lost claim follows.

No GPU, vault, TEST access, or frozen-output edits were used. No gate was relaxed. The parent can close the study with planned render/sample coverage complete and the primary failure unchanged.

## Evidence

- Audit directory: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/stage_lean_visual_audit_01`
- Provenance, exact source/artifact hashes, coverage and limits: `observer.json`
- Full decode hashes: `decoded_framehash.txt`; decode log: `decode.log`
- All frame timestamps/counts: `probe.json`; command record: `commands.json`
- Native frame/time mapping: `native_manifest.json`; inspected images: `stack_001.png`–`stack_045.png`
- Sheet padding check: `sheet_layout.json`
