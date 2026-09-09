# Diagnosis and design, 2026-09-09

The no-clock restricted mapper has nonzero lit ray speed (median 11.12 degrees/s), yet the full Shpongle video has visible normalized geometry change on only 0.00541 of sampled frame pairs. The later angular mapper increases speed but sends 0.4602 of RGB-weighted moving-head light upward. These diagnose geometry/visibility failures; they do not prove removing beat inputs caused the failure. The old pilot also changed target rate, architecture and head factorization, so beat removal was never isolated.

Stage9/10 in PLAN and Vibescience exposed weak evaluation: beat-synchronous flashing inflated Light-BAS; a per-song constant decoder retained scene statistics; a critic separated beat-trigger negatives without improving the show. Stage11's positive scene-statistics result did not validate choreography. The September 8 no-clock experiment improved a descriptor-distribution/audio comparison but was rejected visually. Those proxies are insufficient as product gates.

The next intervention is one integrated prototype, not causal attribution to beat restoration: full 92-channel/40Hz input; separate A regression and B events; temporal C flow; sequence-level clock-block dropout; a corrected world-coordinate rig map; independent emitted-light diagnostics. Clock dropout is an unproven anti-shortcut hypothesis. Same-beat/different-nonbeat-audio inference is the falsifier. No learned critic, renderer oscillator or human-preference claim is added.

Primary sources checked after the Perplexity research pass (the returned pass was generic; its suggestion that beat removal caused static light was rejected):

| Primary source | Relevance and limit |
|---|---|
| [Conformer](https://arxiv.org/abs/2005.08100) | Convolution plus attention; existing encoder family. Does not establish light quality. |
| [Listen, Denoise, Action](https://arxiv.org/abs/2211.09707) | Audio-conditioned movement sequences; supports sequence modelling, not exact choreography recovery. |
| [Classifier-Free Guidance](https://arxiv.org/abs/2207.12598) | Conditioning dropout is established; selective beat dropout remains our hypothesis. No guidance sampling added. |
| [RAFT](https://arxiv.org/abs/2003.12039) | Image motion is observable separately from commands; our lightweight observer uses Farneback, not RAFT weights. |
| [TTUR/FID](https://arxiv.org/abs/1706.08500) | Distributional comparison cannot establish input dependence or temporal movement by itself. |
| [Wasserstein GAN](https://arxiv.org/abs/1701.07875) | Distribution distances; marginal similarity can coexist with wrong conditioning. No GAN added. |
| [Mixture Density Networks](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/bishop-ncrg-94-004.pdf) | Conditional multimodality; a new MDN family was unnecessary for this one follow-up. |
| [AI Choreographer](https://arxiv.org/abs/2101.08779) | Music-conditioned dance and temporal context; dance evaluation does not transfer automatically to lighting. |
| [Bailando](https://arxiv.org/abs/2203.13055) | Learned dance primitives; no quantized token machinery added in this prototype. |
| [EDGE](https://arxiv.org/abs/2211.10658) | Controllable music-to-dance generation; motivates separate tests of variation and conditioning. |

Dense flow follows the [OpenCV documentation](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html). Frame brightness normalization suppresses ideal global gain changes, and an RGB dark floor prevents background codec values counting as light. Real rendered pulses can still change apparent geometry through tonemapping; optical flow is therefore always paired with actual lit-ray movement. Head-only renders prevent a projector satisfying the moving-head gate. Two-second span and spectral excess reject tiny fast jitter.

TRAIN-only engineering calibration exposed insufficient beam visibility even under true motion. The final static adapter raises moving-head baseline, reduces saturation, broadens beam angle, and uses haze 0.25 in the fixed renderer. These are visibility calibration, not learned choreography. Comparisons of video gates must use the same renderer setting for both candidates. Initial and final calibration artifacts are preserved separately.

This is one training seed on TRAIN/VAL. Thresholds and output selection are frozen before candidate inference. Failed gates stay failed; a directional Vibescience verdict alone does not unlock TEST or establish aesthetic success.
