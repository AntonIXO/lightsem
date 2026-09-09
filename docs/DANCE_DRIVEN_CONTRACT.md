# Dance-trained music-to-light: binding product requirement

Latest user authorization, September 9–10: implement the [overnight study](NIGHT_CAMPAIGN_20260909.md), with six main variants and two training seeds, about ten hours of useful serial GPU work, expressive dimmers, automatic mirror/independent groups, and an ending-glare correction. Architecture changes and independent collapse checks are authorized. Mechanical acceleration limits and new fixture types are deferred.

**Beat features ARE allowed as model inputs**. The failure to avoid is movement collapsing to BPM, or disappearing altogether. This supersedes the September 8 input ban. Dance supplies training supervision; playback receives music only.

```
Training: aligned music (including beat features) + measured dancer motion → learned controls
Playback: music → learned A/B/C controls → fixture coordinates and DMX → rendered light
```

There is no generated dancer or required pose input during playback. The rig adapter uses fixed coordinate transformations, calibrated ranges and device speed limits. A constant control sequence must produce constant light after actuator settling. The renderer must not invent choreography from time.

Disallowed in this path:

- Preset pulse, chase, strobe, travelling wave, pan/tilt sine or hue-drift generators.
- Event quotas, per-track percentile thresholds forcing flashes, and metronome fallbacks.
- Treating scene-colour statistics, latent variability or changing DMX angles alone as proof of visible dancing light.

The next model restores the full 92-channel music bus at 40 Hz, including pulse clarity and five beat features. It has supervised A controls, learned B event scores and a joint temporal C flow. `metric_level` is allowed in A but does not drive an oscillator in the adapter. Fixed world targets below the truss replace the erroneous wide angular mapping. Static fixture-level visibility calibration is permitted and is frozen before training.

Self-evaluation checks independent axes: lit beam motion, two-second angular travel (small jitter is insufficient), head-only rendered movement, upward light, high-frequency excess, dependence on non-beat audio with beat features held fixed, and event selectivity. High rhythmic periodicity alone is not beat collapse. Known static, brightness-only, BPM-only and jitter controls must fail the appropriate combined gates before candidate training. Optical flow alone can confuse dim-light pulses with motion; it must be paired with actual ray displacement.

The earlier September 9 pilot authorized one seed-0 training. The newly accepted overnight plan separately authorizes twelve main trainings and the reconstruction/representation diagnostics it specifies. Inference ablations reuse the same weights and matched generator noise. Normalizers use TRAIN only; validation uses the existing 40-song VAL partition. Previously opened Stage11 TEST is not fresh confirmation data and is not read by this follow-up. All model training/inference and stage raster rendering use the GPU. CPU prepares data, analyzes saved outputs and produces scientific reports.

The no-clock pilot's distributional result remains evidence about descriptor statistics. The user rejected its visible result, so it is not a product success. Stage11 scene statistics likewise remain distinct from choreography quality. Any new release claim must include actual DMX/video evidence and unchanged precommitted gates. A one-seed prototype does not establish human aesthetic preference or unseen-genre generalization.
