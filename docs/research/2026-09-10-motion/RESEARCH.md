# Motion representation and lighting dynamics

The current 128/256 pose-probe labels denote decoder hidden widths, not richer
descriptions: both receive the same 21 continuous and three event channels. The
learned pose24 encoder is framewise, while its decoder already has about 3.1 s of
temporal context at 40 Hz. Better reconstruction against imperfect detected poses
does not establish better dance semantics. Raw pose sampling is 15–30 Hz, not the
10 Hz used in an earlier visual sampling stage.

Across 883 coaches, the median fraction of joint-coordinate gaps longer than
0.4 seconds is 0.07987; 643 exceed 5%. Those aggregate numbers do not make every
body segment unusable. Preprocessing fills long gaps even when diagnostic flags
record them. The existing eight-panel video also masks generated predictions with
the source detector's confidence mask, independently creating apparent flashing.
New diagnostics separate finite predictions, uncertain reference joints, and video.

The current Shpongle render derives brightness from sampled motion descriptors,
not a measured audio envelope. Its shader scales scatter and bulb glow by
`1/(1+0.42*max(sum(intensities)-1,0))`, compressing visible response as the rig gets
brighter. Median emitted pan/tilt vector speed is about 40.9 degrees/s in seconds
0–20 and 51.0 in seconds 70–90. These are motor-coordinate rates, not 3D ray rates.
Isolated fixed-geometry ramps must distinguish renderer compression from mapping.

## Primary-source evidence and decisions

| Source | Evidence used | Decision and limitation |
|---|---|---|
| [FineDance, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_FineDance_A_Fine-grained_Choreography_Dataset_for_3D_Full_Body_Dance_ICCV_2023_paper.pdf), [public release](https://github.com/li-ronghui/FineDance) | Optical motion capture with aligned music; public subset 7.7 h, full paper dataset larger. | Primary representation source; verify actual release and preserve noncommercial research terms. |
| [AIST++ official downloads](https://google.github.io/aistplusplus_dataset/download.html) | 60 Hz optimized 3D joints with temporal/kinematic constraints; explicit ignore list. | Separate multi-view-reconstructed quality reference. Prior LightSem nulls preclude treating its musical loops as proof of EDM dynamics. |
| [LODGE, CVPR 2024](https://arxiv.org/abs/2403.10518), [implementation](https://github.com/li-ronghui/LODGE) | Global/local motion context, articulated rotations and contacts. Released preprocessing removes some monotonous intros and low-variance training clips. | Borrow continuity ideas; preserve quiet motion. Released inference has dataset-specific identifiers/genres, so it is not a drop-in arbitrary-music baseline. |
| [MoMask, CVPR 2024](https://arxiv.org/abs/2312.00063) | Temporal motion codec with reduced temporal rate and residual quantization. | Supports sequence compression; begin with continuous latent to separate temporal structure from codebook optimization. Text conditioning is not audio evidence. |
| [DART, ICLR 2025](https://arxiv.org/abs/2410.05260) | Motion primitives conditioned on history support continuous controllable generation. | Continuity reference, not an off-the-shelf music model. |
| [MDM, ICLR 2023](https://arxiv.org/abs/2209.14916) | Motion prediction permits direct geometric and contact losses. | Score articulated positions/velocities/contacts; do not replace motion fidelity with embedding distance alone. |
| [Zhou et al., CVPR 2019](https://arxiv.org/abs/1812.07035) | Continuous six-dimensional rotation representation for learned 3D rotation. | Use 6D rotations with orthonormal projection and fixed-skeleton FK; test conventions and round trips. |
| [GVHMR, SIGGRAPH Asia 2024](https://arxiv.org/abs/2409.06662) | Gravity/world-aware temporal monocular motion recovery. | Later bounded comparison on existing recordings; estimated motion, additional model assets, not mocap ground truth. |
| [WHAM, CVPR 2024](https://arxiv.org/abs/2312.07531) | World-grounded motion recovery and foot-contact treatment. | Supports evaluating world travel, jitter and sliding; defer full-corpus extraction. |
| [InfiniteDance, 2026](https://arxiv.org/abs/2603.13375) | Large monocularly reconstructed dance corpus and motion restoration. | Do not scale pseudo-ground-truth before clean-source representation passes. |
| [EBU Tech 3341](https://tech.ebu.ch/docs/tech/tech3341.pdf) | Momentary 400 ms and short-term 3 s loudness measurements. | Direct audio dynamics preserve within-track contrast; DMX mapping is a separately tested design decision. |
| [Foote novelty / AudioLabs FMP](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S4_NoveltySegmentation.html) | Structural novelty detects sustained changes in musical content. | Slow global composition and palette should follow structural evidence, not every limb fluctuation. |

Primary-source inspection supports a representation-first experiment, not a claim
that 64 channels or flow matching will succeed. Current hardware speed results
support faster matched training, not universal quality retention: the historical
quality-confidence intervals remain unchanged. Early exploration uses seed0;
six inference draws measure sample variation, and are distinct from training seeds.

## Reproducibility and research limitations

No new research answer came from Perplexity: its Deep Research quota was exhausted.
The recommendations above were checked against original papers, official releases
and source code. Dataset metadata, source snapshot, split membership, runtime,
normalizers, predictions and independent gates are frozen before scientific runs.
The source-data/codec gate precedes music training; audio conditioning is assessed
before a direct loudness path could conceal its failure. Prior experiments remain
historical records with their computed verdicts, including negative results.
