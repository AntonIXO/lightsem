# Audio-to-light repair and final visual audit

The first demo was rejected after direct inspection: all beams converged and
its mean aim spread was `0.071` against a held-out target of `0.822`. The v5
pairwise-spread repair separated the beams (`0.597`, ratio `0.726`) but did
not add useful temporal aim motion.

The root model was then repaired in three places: dance teacher embeddings are
now actually distilled into the audio latent; angular motion derivatives are
computed after unwrap and normalized instead of producing atan2 wrap spikes;
and RGB targets are clipped to the normalized physical range. A direct
inter-frame objective and an audio residual aim head were added. The best
longer candidate is v15 (seed 0, 1,600 updates).

The v15 videos were inspected at frames 0, 192 and 383 for both held-out
songs. It has finite outputs, visible time-varying aim, green RGB output and
music-dependent brightness. Spatial spread ratios are `0.885` for `21c3a` and
`1.016` for `3b4ab`; aim MSE is `0.0573` and `0.0521` respectively.

It remains diagnostic only. The audio-only aim traces for the two songs have
cross-song correlation `0.9963` and mean absolute difference `0.0309`, so the
movement gate fails: the model repeats nearly the same choreography under two
different soundtracks. Audio shuffle does change aim (`0.0484`) and brightness
(`0.3154`), but that is insufficient to establish song-specific choreography.
There is no native body or DMX in this pilot, so it cannot be promoted to the
native pipeline.

Canonical receipt: `FINAL_AUDIT.json`.
Candidate artifacts:

- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_privileged_dance_20261015_v15/target_plus_dance_aux_spread/model.pt`
- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_privileged_dance_20261015_v15/demo_3b4ab_render/demo.mp4`
- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_privileged_dance_20261015_v15/demo_21c3a_render/demo.mp4`
