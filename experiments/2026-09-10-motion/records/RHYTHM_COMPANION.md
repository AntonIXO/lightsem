# Rhythm-led companion lighting

## Purpose

Generate LIGHT. The existing saved dance stays unchanged and is only a diagnostic.
This implementation separates motion-derived spatial style from musical time:

```
saved internal body -> openness / height / asymmetry / lean / facing -> light shape
source audio -> local beat clock + confidence -> 2 / 4 / 8 beat light phrases
source loudness -> unchanged brightness and activity / mechanical speed budget
all controls -> finite floor/wall targets -> quantization-safe angular projection -> DMX
```

`lightsem/experiments/motion_study/companion.py` is an explicit opt-in entry point.
It reuses the exact `pose_stage` photometry and changes only the four angular bytes
per moving head. No new dance draw, model training, motion retiming, DMX hardware,
TEST access, or legacy default change. This is procedural light choreography, not
a newly trained lighting model or proof of learned music conditioning.

## Behavior

- Outer pair: a broad four-beat opening with a depth arc.
- Middle pair: a two-beat response, offset between sides.
- Inner pair: an eight-beat wall sweep; no stationary central pair requirement.
- Body properties shape these gestures but limb velocities do not set their rate.
- Quiet activity or low pulse confidence suppresses autonomous sweeps.
- One clock and one full-record controller state; crop only after mapping.
- Mechanics remain the previous90deg/s and360deg/s2 **simulator** budget. Actual
  fixture feasibility is not certified by these software limits.

The audio tracker uses broadband onset tactus, kick-band phase, and conservative
local pulse/period updates. It handles gradual drift but can still choose a wrong
half/double tempo, follow syncopation, or miss abrupt tempo changes. It is offline,
not a streaming beat/downbeat estimator. No arbitrary 4/4 downbeat is asserted.

## Reproduce this delivery

Artifact directory: `/mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z`.
The preregistered plan, immutable original implementation manifest, explicit
chained tooling revisions, inputs, logs, DMX streams and reports live there.
Original source arrays and recordings are referenced by SHA256 rather than copied.
Historical pose-stage arrays/receipts are never edited.

Commands (each completed stage refuses overwrite; use a new authorized study
and registration for a new attempt):

```bash
bash run.sh -u run_rhythm_companion.py audio --output /mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z
bash run_torch.sh -u run_rhythm_companion.py map --output /mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z
bash run_torch.sh -u run_rhythm_companion.py preview --output /mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z
bash run_torch.sh -u run_rhythm_companion.py audit-preview --output /mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z
bash run_torch.sh -u run_rhythm_companion.py render --output /mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z
bash run_torch.sh -u run_rhythm_companion.py compare --output /mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z
bash run_torch.sh -u run_rhythm_companion.py audit-full --output /mnt/lightsem-data/experiments/lightsem/rhythm_companion_20260912T193657Z
bash run_torch.sh -m unittest discover -s tests -p test_rhythm_companion.py -v
```

Final deliverables:
- `full_candidate_1080p_av1.mp4`: full-screen native1920x1080,30fps,AV1 + source audio.
- `full_comparison_1080p_av1.mp4`: labeled before/after; each native scene reduced
  to960x540 on a1920x1080 canvas. One global downward display-luma match per arm;
  no per-frame exposure tricks. Standalone output is not dimmed to the comparison.
- `full_baseline_1080p_av1.mp4`: native unchanged pose-stage comparator.

## Current measured scope (before full delivery)

The full277.3s saved-body baseline recomputes byte-exactly. Candidate and both
controls keep every non-angle DMX byte unchanged. All emitted rays have finite
floor/wall hits under the geometric instrument (truss occlusion is not modeled).

On fixed70-90s, baseline/candidate median angular speed is20.783/23.757deg/s
(+14.31%). **The preregistered>=25% prediction failed.** Quiet0-20s median is
2.816/2.510deg/s. Candidate maximum83.575deg/s and354.985deg/s2 remain within90/360.
Active p10 hit-x span increases2.632->6.282m; this is a geometric spread statistic,
not visual preference or proof that every beam is separated.

A half-beat clock perturbation changes active emitted angles by5.400deg RMS with
unchanged body and brightness. Holding style at the first frame changes active
raw targets by.254m RMS. Synthetic emitted-DMX90/150bpm tests distinguish movement
periods on a stationary body. Synthetic90/120/150bpm and100->140ramp pulse timing
has median error<10ms vs the registered60ms tolerance. These synthetic tests do
not establish real-track beat accuracy.

The real clock is associated with its own onset evidence at.440 versus.392 for a
half-beat shift in70-90s; this is not an independent beat annotation benchmark.
Full AV media integrity, sampled visual review and human music-fit acceptance are
separate. Numerical source/mechanics passes do not override the missed25% prediction.

System of record: `/root/lightsem-vault`, experiment
`exp-rhythm-companion-20260912t193657z`. Repository prose is a runbook, not a closure.


## Completed delivery — 2026-09-12

All three full-track files above are complete:8319 decoded frames,1920x1080,
30fps,AV1,277.300s video and audio. Software AV decode, every videoPTS, and
contiguous audio packetPTS/durations pass. Each delivers12228930 stereo audio
samples/channel at44100Hz. All three decodedPCM hashes match. Four fixed source
waveform windows have best lag0 within a declared+/-100ms search; source FLAC's
last294samples (6.667ms) are intentionally outside the whole-frame video endpoint.
AAC is lossy; source equality is NOT claimed.

85 focused tests pass. Full native raw baseline/candidate outside-bulb means are
2.877928/2.519113; neither stream has any-channel>=250 outside-bulb clipping.
The comparison globally dims the baseline;277 nominal1Hz decoded samples measure
2.275195/2.230331 meanluma with bulbs/captions excluded, a-1.97% candidate residual.
This is approximate global matching, not exact per-frame or fullpixel equality.

Silent visual review:85candidate nominal1Hz frames on pages0,4,6,10,15,19,22,23;
18dense paired samples at76-78,194-196,264-266s; candidate80s/finalframe and comparison80s
stills. Images were displayed downsampled. The candidate uses more stage space,
paired sweeps and crossings; quiet remains restrained. The2/4/8beat vocabulary
is visibly repetitive. This is NOT continuous audiovisual/human acceptance.

`final_outcome.json` and per-file `*_audit.json` carry the delivery evidence.
The committed>=25% active-speed prediction remains failed (+14.31% measured).
No new changes have been committed and no existing default has been promoted.
