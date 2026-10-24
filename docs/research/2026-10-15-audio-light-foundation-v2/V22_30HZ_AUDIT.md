# 30 Hz temporal repair

The 10 Hz v15 demo was visibly too slow. The prior body-to-DMX pipeline used
30 Hz body updates and 40 Hz light controls; the foundation pilot was changed
to train at 30 Hz with the native `x30` audio/body timeline and light targets
resampled from 40 Hz.

The balanced v22 seed-0 fit used 1,600 updates, a 384-frame (12.8 s) window,
training-only dance distillation, unwrapped angular derivatives, temporal
losses, and stronger brightness/event terms. The 90-second export was rendered
from the first 90 seconds of `3b4ab` with 2,700 predictions at 30 Hz and 2,700
video frames at 30 fps. Start, midpoint, and end frames were inspected.

Long-run measurements:

- median aim speed: `0.1906` normalized units/s;
- 90th percentile aim speed: `0.5451`;
- 75.98% of frames exceed `0.1` aim speed;
- brightness exceeds `0.1` on 90.37% of fixture frames;
- all saved predictions are finite and video/audio duration is exactly 90 s.

This is a real temporal improvement over the 10 Hz export, but it remains a
diagnostic. The independent held-out-song comparison still gives aim
correlation `0.9924` between `21c3a` and `3b4ab`, so song-specific movement is
not qualified. There is no native body or DMX emission in this visualizer.

Canonical receipt:
`/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_30hz_20261015_v22_long90b/demo_3b4ab_90s/LONG90_AUDIT.json`.
