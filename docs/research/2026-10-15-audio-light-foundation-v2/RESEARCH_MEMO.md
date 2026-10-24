# Audio-to-light foundation v2 — 2026-10-15

This is a new, diagnostic one-seed experiment. The target is **audio → learned
stage-light controls**. Dance is a training-only privileged modality used to
teach music-sensitive kinematic structure; it is not required at inference and
it is not the product output.

## What the literature actually establishes

The paper search was intentionally retrieval-only. Perplexity was asked to find
primary papers and official artifacts, not to design this experiment. The arXiv
MCP was then used to verify the records and read methods sections.

- [Generalized distillation / privileged information](https://arxiv.org/abs/1511.03643)
  provides the formal teacher-at-training, student-at-deployment setup.
- [Audio representation learning by distilling video](https://arxiv.org/abs/2302.02845)
  is the closest operational precedent: a video teacher supplies intermediate
  embeddings while the audio student remains usable without video. Its tasks
  are speaker/emotion recognition, not continuous control.
- [ForkMerge](https://arxiv.org/abs/2301.12618) shows that auxiliary tasks can
  cause weak or strong negative transfer; task weight must be judged on target
  validation, not auxiliary loss or gradient cosine alone.
- [Whisper-GPT](https://arxiv.org/abs/2412.11449) provides a verified
  continuous-spectrogram plus discrete-acoustic-token precedent with causal
  75-Hz alignment. It does not validate a lighting target.
- [Skip-BART](https://arxiv.org/abs/2506.01482) is the strongest verified
  audio-to-light precedent, but its targets are HSV hue/value distributions.
  The paper explicitly does not establish pan/tilt, velocity, lasers, arbitrary
  fixture inventories, or physical safety.
- [SeqLight](https://arxiv.org/abs/2605.03660) extends this to multi-light HSV
  decomposition with imitation/RL on RPMC-L2. It still does not provide raw
  DMX, moving-head dynamics, or laser safety.
- [MATHDance](https://arxiv.org/abs/2505.14222) and [TokenDance](https://arxiv.org/abs/2603.27314)
  show dance tokenization and MuQ/music conditioning, but dance codes are not
  lighting controls and must be treated as auxiliary supervision only.

The newly returned [AuraLuxMuse](https://arxiv.org/abs/2610.11792) was recorded
as an unverified research lead: the metadata describes a paired Musilux lighting
dataset and LAMP, but release, license, code and reproducibility were not
independently established. It is excluded from the first run.

## Experiment decision

The primary model has an audio encoder and streaming latent context. At
training time only, a body teacher produces a dance latent and kinematic proxy
targets. At inference time, only audio features and fixture metadata are given.
The model has separate learned heads for:

```text
fixture aim/motion, angular speed, angular acceleration,
RGB color, brightness/dimmer, cue/event probability
```

The heads share the audio latent but are scored separately. There is no
post-hoc motion wrapper, sine oscillator, beat sweep, D4/D6 recomposition, or
fallback dance. Color and brightness are learned outputs, while P2 remains the
fixed comparison target for the primary light stream.

## Data contract

The first pilot uses the eight prepared rows for which all required artifacts
co-exist: pinned MuQ layer 9, Base92, saved native-C body, BeatSaver authored
light groups, RGB/intensity and event targets. Six songs are training and two
are held out by song. The authored light targets are the primary supervision;
body-derived fixture motion is an explicitly labeled auxiliary proxy because
the public target has no physical pan/tilt labels.

This is a mechanism pilot, not a foundation qualification. It cannot establish
generalization from eight songs, cannot validate commercial DMX rights, and
cannot claim that dance knowledge improves lighting until the target-only and
auxiliary arms are paired on the same split.

## Required interpretation

The auxiliary arm earns credit only if it improves held-out light metrics or
learned motion-proxy sensitivity without regressing color/brightness/event
metrics. If it only lowers dance loss, that is not a target success. If it
hurts target validation, the result is negative transfer and the dance branch
must be disabled or reweighted in a new protocol.
