# Brightness-matched optics and static foil review

## Scope and result

Independent CPU-only, unblinded visual review of **silent sampled frames**. No continuous audiovisual playback, new music-coupling claim, or perceptual/product acceptance.

At the reviewed scales and times, `matched/closer_zero_angular_flux` retains clearer, narrower, separated shafts than `matched/closer_native_legacy`. The legacy arm forms broad, soft, often overlapping wedges. This difference survives the global mean display-RGB control. It is a qualitative observability result, not proof that the sharper style is preferred or realistic.

The unscaled `renders/closer_zero_static_bright_foil` is brighter but does not reproduce the moving arms' changing directions. Brightness alone is therefore not interchangeable with the direction changes visible in these samples. This does not quantify motion quality or establish musical timing.

## Exact visual coverage

- All **10 chronological sheets per arm**, 30 total. Each arm has **278 samples at integer seconds0–277** (834total). Final sheet has8real tiles;22black padding tiles are not video blackouts.
- All **45 native fastframes per arm**,135total: every4th30fps frame in `[80,82)`, `[196,198)`, `[270,272)`. This is7.5Hz sampling,15frames per interval, through81.8667,197.8667,271.8667s.
- There are3overlaps per arm with the1Hz set:80,196,270s. Thus **320unique source frame indices per arm**,960across3arms, not every8319frame visually reviewed.
- Original sheets are960×1800 with320×180 thumbnails, already reduced from640×360. `attach_image` resized each sheet to640×1200 JPEG quality82: effective tiles≈213×120.
- Fastframes came from software-decoded MP4s at native640×360. They were stacked vertically without resampling into640×1080 PNG triptychs (native legacy / flux / foil). All45triptychs were perceived. No attachment resize was reported. No exposure or gamma boost was used.

## What was visible

The0–59s overview is very dim blue/cyan. Narrow flux shafts remain easier to distinguish than legacy haze. From roughly66s the scene becomes brighter blue/violet; later it becomes mainly magenta. All three arms change brightness/color. The foil repeats the same fan geometry.

At80–82s, outer shaft endpoints and small central directions change in both moving arms. Flux boundaries make the changes easier to locate. The foil shares the80s starting pose, then holds its angles. At196–198s the moving arms show clearer changes in shaft direction, apparent length, and lateral position, while the foil stays in its fixed fan. At270–272s, brightness falls then rises; small direction changes remain visible in moving arms. Native legacy stays soft and dim.

Source bulbs are stationary origin dots, not shafts. Their visibility or brightness is not evidence of shaft movement. Angular shafts show granular texture and hard/rounded ends at native size. Better separation does not establish realism or preference.

## Brightness and foil checks

Raw saved per-frame NPZ metrics were independently averaged over8319frames per arm. The metric is Rec709-weighted **display RGB luma in0–255**, excluding top28caption pixels and20px circles around all fixture origins. It is not shaft-only attribution.

| Arm | Original mean | Global RGB scalar | Actual matched/unscaled mean | Match residual |
|---|---:|---:|---:|---:|
| closer_native_legacy |3.178095715|0.924085904|2.963248014|+0.026414564|
| closer_zero_angular_flux |2.936833450|1.0|2.936833450|0|
| static bright foil |5.620365296|unscaled|5.620365296|not matched|

The common target was2.936833450. Nearest-even uint8 rounding leaves legacy≈0.90% above target. There was one global scalar per matched arm, **no framewise scaling**. Receipts report byte-identical raw rerenders before dimming. This controls mean display brightness only, **not lumens, HDR energy, local contrast, color, style, or perceived brightness**.

The foil's renderer gain is4.8 versus2.4:2×gain produces **5.6204 versus2.9368** mean outside bulbs, about1.914×the moving flux arm, not exactly2×display output. Bulb-region means are15.9342 versus7.9470; the foil also has brighter origins. Neither matched arm has raw any-channel≥250 pixels outside bulbs. The foil's mean fraction is1.53e-6 and maximum1.54e-5; white≥250 fraction is zero.

Foil angles were reconstructed from source40Hz row3200 (80s), with zoom bytes zeroed. Reconstructed angle and whole-DMX hashes exactly match the frozen foil receipt. All other source bytes remain unchanged. **Static angles can still vary intensity and color from audio**; this is not a frozen-image or no-audio foil.

## Additional pair-video audit

The later1280×360 `matched/pair/video.mp4` received a new full software2-thread ffmpeg decode and ffprobe inspection. Both counted8319frames with regular30fps PTS and exact277.3s video duration. Decoded audio is contiguous44100Hz stereo,12228930samples/channel,97,831,440bytes as interleaved pcm_f32le. Its SHA256 is `6d5d8917556e5b7247fef054c758e975b9bf9d8ad206dccae755d4b51d794455`, exactly matching both authenticated source-member audio audits. This compares decoded AAC in the source MP4s, not byte equality to the original FLAC.

All commands exited0 with empty error-level logs. Pair-video hash, pair receipt/started sidecars, member receipt hashes, and reviewed-arm artifact hashes passed. Pair SHA256: `9a7e2fa811561fca6aaffcc802398c1bff81b9e96f6e8c09d77b27278856a24a`. The frozen pair receipt remains `PENDING`; the separate audit records completed checks without modifying it. Media integrity does not establish audiovisual synchrony by perception.

## Evidence

- Observer: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/matched_foil/observer.json`
- Exact frame/page coverage and hashes: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/matched_foil/coverage.json`
- New pair decode/probe audit: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_visual_audits_01/matched_foil/pair_audit/audit.json`
- Native extracted frames, unscaled triptychs, decode logs, full probe and framemd5 are retained under the same unique audit directory.

No frozen/render/matched file was edited. No GPU, vault, inference, or TEST access was used. Music and visual acceptance remain unassessed.
