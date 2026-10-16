# SSL latent light adapter integration

This experiment adds an opt-in `--ssl-light` branch to the selected native-C
baseline. The native body, MuQ/Base92 audio features, mapper-01 patch, P2 RGB,
and restored LUFS dimmer stay fixed. The SSL model reads the causal semantic
context and a fitted per-fixture linear decoder emits bounded pan/tilt residuals
for the first eight moving fixtures. It does not replace the body or use raw
DMX as an SSL input.

The adapter runs before video rendering and rejects nonfinite output, a changed
RGB/dimmer stream, a body-insensitive latent, or a candidate/reference raw
speed or acceleration ratio above 1.25. The baseline remains unchanged unless
`--ssl-light` is explicitly supplied.

The saved full-song integration receipt is
`/mnt/lightsem-data/experiments/lightsem/body_light_ssl_baseline_integration_20261016_v1/INTEGRATION.json`.
Both Shpongle and AURORA — Cure For Me passed the raw gate. The saved native
CLI run now uses `/root/shpongle.wav` with its matching cached MuQ/Base92
features. The full Shpongle baseline and SSL videos are sealed in the receipt;
the side-by-side is labeled BEFORE/AFTER. The SSL decoder used CPU because a
shared WhisperX service held the GPU lock. The data-level run still covers both
songs using saved full-song native body/DMX artifacts.

This remains `diagnostic_only`; no baseline promotion or artistic-quality claim
is made by the engineering gate.
