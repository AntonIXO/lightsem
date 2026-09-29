# Independent pose-stage media audit

## Result

**All 14 completed movies pass the measured media checks.** No missing frames, decoded timestamp gaps/overlaps, decode errors or duration overruns were found.
- Twelve reference movies: exactly 600 decoded frames and 20 s each.
- Shpongle original and global brightness-matched movies: exactly 8319 decoded frames and 277.3 s each.
- All movies: one H.264 1920×360 video stream at exactly 30 fps and one AAC 44,100 Hz stereo audio stream.
- Every decoded video PTS advances by exactly 1/30 s. Every decoded audio PTS advances by the preceding frame sample count / 44,100. Both start at zero. Container stream ends and decoded ends equal the registered duration exactly, using rational arithmetic.
- Full independent software RGB24 and native-rate float32 PCM decodes succeeded with no error-level stderr. Streamed byte counts agree with complete decoded frame/sample counts. All 14 decoded RGB SHA256 hashes match the renderer’s recorded decode hashes.

## AAC priming and final samples

All AAC streams have a first packet at PTS −1024/44100 s with skip_samples=1024. Decoded PCM starts at zero; this priming packet is not a missing presentation interval.
- Reference movies: 863 packets, 862 decoded audio frames, 882,000 samples/channel. Final packet/frame contributes 336 samples; discard_padding=688.
- Each Shpongle movie: 11,944 packets, 11,943 decoded audio frames, 12,228,930 samples/channel. Final packet/frame contributes 322 samples; discard_padding=702.
FFmpeg honors this side data. No manual trimming, resampling, silence insertion or tail removal was used in this audit. Decoded audio tail beyond the registered end is exactly zero samples in every movie. The generic PCM helper’s “padding retained” label means no audit-side trim, not that codec discard metadata is ignored.

## Shpongle audio identity

Original versus global brightness-matched decoded PCM is byte-identical: native 44,100 Hz, stereo, interleaved little-endian float32; 97,831,440 bytes each.
`SHA256 6d5d8917556e5b7247fef054c758e975b9bf9d8ad206dccae755d4b51d794455`
Output AAC is lossy. This identity is between the two output AAC decodes, not between AAC and a lossless source. The source files were independently hashed and minimally probed: reference WAV sources are PCM s16le; `/root/shpongle.flac` is FLAC. Shpongle source duration is exactly 12,229,224/44,100 s; the registered 277.3 s render intentionally omits the final 294 source samples (6.667 ms). This is the registered whole-video-frame trim, not decoded packet loss or padding. Registered source offsets and render command bindings match. Audible source-event alignment was not tested.

## Method and evidence

Run from `/root/lightsem`: `CUDA_VISIBLE_DEVICES="" bash run.sh outputs/audit_dance_pose_stage_media.py`. Final source/binding checks: `bash run.sh outputs/finalize_dance_pose_stage_media_audit.py`.
Only generic `digest`, `timeline`, `stream_end`, `q`, `file_command` and streaming `pcm_command` utilities were reused from `outputs/audit_dance_optics_media.py`. Its AV1-specific audit/main was not run. No evaluator, renderer, model, GPU, physical output, vault, TEST, seed1 or default changes.
Each member receipt records exact argv, exit status, stderr, source/video/render/start hashes, raw decoded stream hashes, complete rational timeline checks and evidence-file hashes. Software H.264/AAC decoders are explicit; video uses `-fps_mode passthrough`, PCM has no `-ar`, `-ac`, filter or duration limit. No raw RGB/PCM was written to disk.
Initial ffprobe option order unexpectedly retained extra frame metadata beyond the requested fields. This is a tooling/minimal-probe deviation, not a media failure; evidence is retained. Packet and supplemental source probes limit fields correctly.

Evidence directory: `/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01/observer_audits/media`
- `audit.json`: all 14 machine audits, exact timing, PCM identity, frozen-source/tool hashes and command semantics.
- `supplement.json`: source codecs, map/start/render bindings, executable freeze comparisons and this scope clarification.
- `<movie>/frames.stdout`, `audio_packets.stdout`, `audit.json`, and decode stderr: complete per-frame/packet timing evidence and streamed hashes.

## Scope limits

Each movie shares one saved-body comparison: body panel, same-body articulated-stage baseline and pose-stage candidate, with one shared audio stream. There is no invented incumbent, no Shpongle reference body and no generator comparison.
This is machine media integrity only. It does not establish musicality, music conditioning, preference, perceptual source alignment, decoded clipping, physical safety, hardware qualification or human audiovisual acceptance.

## Storage

Final measured audit size: 44.35 MiB; whole pose-stage study including parallel observer directories: 450.02 MiB; free space: 2.62 GiB. Limits: 128 MiB audit, 1.75 GiB study, at least 1 GiB free. Checks reserve 8 MiB before each audit member; final report/JSON are small. No source or shared main receipt was edited.
