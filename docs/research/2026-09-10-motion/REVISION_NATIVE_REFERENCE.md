# Full native reference after compression failures

This is a documented implementation revision under the accepted failed-gate policy.
Both the original neural codecs and all four linear phase-codec capacities failed
the frozen representation gates. Those results retain their original identities.

The next music experiment uses all 139 native channels at 30 Hz: horizontal root
velocity, root height, 22 continuous six-dimensional local rotations and four
contacts. The generator must produce every channel. Fixed-bone forward kinematics
and one full-trajectory integration convert its output to motion. No reference
pose, root trajectory, or contact sequence is supplied at playback.

This representation uses 4,170 scalar values per second. It is uncompressed and
unlearned. Its expected lossless reconstruction is an engineering prerequisite,
not evidence for the failed learned-compression hypothesis. There is no codec
training seed, optimization budget, or learned-codec superiority claim.

Before music training, an immutable engineering audit runs the same fixed-window,
full-sequence, short-hold, physical and continuity checks. The existing 12-clip
source visual review may transfer only when every VAL feature is exactly preserved
and decoded joints match the reviewed source within 5e-7 metres. This is explicitly
an equivalence check, not a new human rating. All prior thresholds remain unchanged.

The increased output size is a cost of preserving native detail. A matched music
regressor and conditional flow will test whether Base92 predicts useful motion.
Correct-audio, donor and within-track-shift comparisons remain independent of
direct audio brightness/activity. No audio-conditioning or lighting result is
assumed by accepting the native representation.

Artifacts are isolated in
`/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910`.
