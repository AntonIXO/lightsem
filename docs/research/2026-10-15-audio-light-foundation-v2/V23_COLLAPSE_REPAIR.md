# v23 windowed-state collapse repair

The 90-second v22 export was falsely diagnosed from a long-context inference
failure: carrying one recurrent/latent state across the whole 90-second song
made 74.44% of frames fall below mean fixture aim spread `0.2`. The model was
trained on 384-frame (12.8 s) windows, so this was out-of-distribution state
drift rather than a valid long-song inference contract.

v23 resets state at each 384-frame training window. Its raw saved predictions
have per-frame mean aim spread minimum `0.565`, and zero frames below `0.2`.
Median aim speed is `0.324/s`. Frames at 0 s, 45 s and 89.97 s were inspected.

A same-style comparison export uses the native two-view stage layout and the
8/5/50 fixture counts. The eight learned beam slots drive the beams; lasers and
bulbs duplicate those slots for presentation and add no independent motion.
The comparison has no native body and no emitted DMX, and is forbidden from
native promotion.

Artifacts:

- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_30hz_20261015_v23/demo_3b4ab_90s/PREDICTIONS.npz`
- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_30hz_20261015_v23_native_style/shpongle_90s/demo.mp4`
- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_30hz_20261015_v23_native_style/shpongle_90s/COMPARISON.json`
