# CPU complete-media audit implementation

## Ready commands

Run from `/root/lightsem`, using the native wrapper:

```text
CUDA_VISIBLE_DEVICES="" bash run.sh -m unittest tests.test_dance_optics_media_audit -v
CUDA_VISIBLE_DEVICES="" bash run.sh outputs/audit_dance_optics_media.py --limit 1
CUDA_VISIBLE_DEVICES="" bash run.sh outputs/audit_dance_optics_media.py --limit 0
```

`--limit 1` audits the next missing **completed** member. It skips originals/matches without both `render.json` and its SHA256 sidecar. `--limit 15` processes up to all currently completed missing members. It never waits for renders. `--limit 0` only writes a fresh coverage receipt. Global `--out` defaults to the registered `optics_full_controls_01` directory. Run only one audit process at a time; no audit process uses the renderer/GPU lock. Capture logs on the data mount, not the nearly full root filesystem.

Each member owns a create-only `media_audit/<group>__<arm>/` directory. A directory already present is never overwritten or resumed. Failed or interrupted audits remain visible and block full coverage; an explicit revised audit location/instrument is required for recovery, not deletion. Coverage snapshots have unique timestamp/UUID names under `media_audit/coverage/`. Receipts and all evidence files are made read-only. Receipt SHA256 sidecars make them tamper-evident, not root-proof.

## Instrument and provenance

- No native LightSem, OpenGL, CUDA, torch, or vault import/call. Only Python standard library, ffmpeg, and ffprobe.
- Authenticates the freeze receipt and every frozen artifact, current registration/design, original DMX, and source audio. Requires the exact 13-arm registered design and fixed two matched members.
- Before decode, authenticates render/start receipts and hashes **every** bound render artifact. Checks arm/settings/freeze identity and matched-original receipt bindings. Repeats those artifact checks after decode. Coverage refresh rechecks the audit evidence and render artifact bindings.
- ffprobe uses `-threads 2` and a software-only AV1/AAC codec whitelist. Complete frame, stream, format and packet JSON are retained. No frame sampling or interval truncation. Frame PTS, durations, audio `nb_samples`, packet side data, and stream integer `duration_ts`/`time_base` are available for inspection.
- Independent ffmpeg video decode explicitly uses `-hwaccel none -c:v libdav1d -threads 2`, without frame-rate conversion. Streamed per-frame `framemd5` retains actual count/PTS/durations and hashes without a raw RGB cache. Output rawvideo encoding is also limited to two threads.
- Independent full AAC decode uses software decoding, two threads, and streamed native-rate/channel `pcm_f32le`. Only its SHA256, byte/sample count and diagnostics are stored. There is no resampling, normalization, silence removal, or truncation.
- Tool versions, commands, exit status and complete error-level stderr are retained. A tool error or changed input is not a pass. A successful audit can report failed exact media checks (such as fewer than8319 actual frames); `status=complete` means the instrument finished, not that the movie meets the registered duration.

## Exact timing and final gates

Generated8319 is carried from the renderer receipt **separately** from container-declared frames, ffprobe actual decoded frames and independent ffmpeg decoded frames. Video timestamps are tested against exactly1/30s, with exact rational arithmetic. Full counts and last-frame end are measured, not inferred from a progress log or container duration. This is presentation-frame counting, not AV1 coded-subframe counting.

Audio PTS continuity uses `current PTS == previous PTS + previous nb_samples/sample_rate` for every adjacent decoded frame. It retains each gap/overlap. Exact stream end and decoded PCM end are separate. AAC priming, edit lists or final-frame padding can make those ends differ. Both are reported without silently dropping samples or claiming the difference is missing music. PCM byte count is cross-checked against ffprobe decoded sample counts.

Coverage always lists all15 expected members. It remains partial until all15 audit receipts authenticate. A full decoded-audio equality comparison is emitted **only after all15 complete decodes**, never inferred from the shared input filename. It compares hash plus PCM byte count/rate/channels/layout. This checks equality among output AAC decodes, not equality with source FLAC. `all_15_exact_media_checks_pass` separately reflects all exact checks; it is null before complete coverage.

Process exit1 indicates failed/partial audits or integrity failures. Exit0 does **not** imply full coverage or that all exact checks passed. Read the coverage receipt and member checks.

## Validation and scope

Ten CPU tests pass with the native wrapper. They cover exact30fps/gaps, variable audio samples, audio gaps/overlaps, missing PTS, integer stream ends, receipt overwrite/tampering, path escape, pending coverage, and an actual short software-encoded AV1/AAC integration test with independent complete decodes and repeatable PCM hash. The synthetic test deliberately labels generated8319 but encodes6 frames; the instrument correctly keeps those counts distinct and fails the8319 check.

No decoded clipping measurement is implemented or claimed. Existing raw display-pixel clipping remains a different instrument. Full successful decoding proves neither perceptual video acceptance nor music acceptance, head-specific motion, or optical flow.

## First registered-media execution

The worker ran `--limit 1` after the CPU tests. `renders/legacy_native_legacy` completed its audit with all exact media checks true. Evidence: `media_audit/renders__legacy_native_legacy/audit.json` under the registered output root. Both complete software decoders counted8319 actual presentation frames, distinct from the8319 generated-frame claim. Video PTS starts0, advances exactly1/30s, and ends277.3s. Audio contains12,228,930 samples/channel at44,100Hz stereo, with no PTS/nb_samples gaps or overlaps. Video stream, audio stream and decoded audio all end exactly277.3s. Full PCM SHA256: `6d5d8917556e5b7247fef054c758e975b9bf9d8ad206dccae755d4b51d794455`.

Coverage at that receipt is **1/15**, not full coverage. The remaining14 have not been audited by this worker. No cross-arm PCM equality or perceptual acceptance follows. First audit took about4.3s and retained metadata/hashes only, not PCM or RGB. Subsequent audits are left to the parent scheduler.
