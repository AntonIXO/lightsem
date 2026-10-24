# Rejected demo audit — spatial aim collapse

The first audio-only demo is rejected as a usable stage result. The visual
failure was visible in the montage and is confirmed from the saved 10 Hz
predictions, not inferred from a screenshot alone.

| Quantity | target body-derived proxy | target-only model | dance-aux model |
|---|---:|---:|---:|
| mean pairwise aim spread | 0.80–0.89 | 0.0138 | 0.0711 |
| p95 pairwise aim spread | 0.95–1.04 | 0.0194 | 0.0801 |
| mean per-frame aim span | 1.61–1.71 | 0.0285 | 0.1497 |

The RGB and brightness streams are finite and non-duplicate, so this is a
motion/aim spatial collapse rather than an RGB duplicate-stream verdict. The
demo is preserved as a rejected diagnostic; no DMX or product claim is made.

Evidence roots:

- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_privileged_dance_20261015_v4/demo_3b4ab_v3`
- `/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_privileged_dance_20261015_v4/demo_3b4ab_compare`

The repair must be measured against the same held-out targets and must pass a
frozen output-spread gate before a replacement demo is rendered.
