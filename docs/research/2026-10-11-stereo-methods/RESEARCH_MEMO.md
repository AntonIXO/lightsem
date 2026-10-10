# Stereo spatial methods — fresh independent review, 2026-10-11

This pass used four independent research agents plus a direct primary-source
search. Perplexity Deep Research was attempted but the account reported zero
remaining Deep Research queries; the fallback used arXiv/official repository
searches. No claim below treats a stereo-master cue as physical azimuth.

## What was actually trained

The existing pilot trained descriptor R1/R2 native adapters and three
source-aware semantic recipes. This pass added and trained a fourth method:

**Stem-JEPA-lite:** a source-query model receives the mixture descriptor and a
four-way source query, then predicts the queried source's local activity,
onsets, stereo image, and held-out-source embedding. It trained for 6,000
updates, seed 0, on MUSDB TRAIN targets and was audited on DEV.

The result was finite but failed admission: event F1 `0.289` versus the
mixture-only DSP reference `0.467`, LR-swap activity error `4.60`, and no
native adapter was trained from it.

## Original methods and applicability

| Method | Original contribution | LightSem fit | Decision |
| --- | --- | --- | --- |
| [Stem-JEPA](https://arxiv.org/abs/2408.02514), [official code](https://github.com/SonyCSLParis/Stem-JEPA) | Predict a missing musical stem embedding from an aligned mix/context | Directly matches source-to-source reasoning; requires aligned multitracks | Implemented a compact source-query probe; failed current timing/LR gates |
| [SpaIn-Net](https://arxiv.org/abs/2202.07523), [code](https://github.com/darius522/spain-net) | Source separation conditioned on known panning angle | Useful controlled re-pan benchmark; static synthetic panning is not mastered stereo truth | Next counterfactual, not yet trained |
| [StereoCRW](https://arxiv.org/abs/2204.12489), [code](https://github.com/IFICL/stereocrw) | Self-supervised inter-channel delay correspondence | Good ITD auxiliary; simulated/binaural physical-delay contract differs from music masters | Audit/auxiliary only |
| [SAR-SSL](https://arxiv.org/abs/2312.00476), [code](https://github.com/Audio-WestlakeU/SAR-SSL) | Cross-channel masked reconstruction for spatial factors | Static-room assumptions mismatch mastered songs | Do not use as direct teacher |
| [BS-RoFormer](https://arxiv.org/abs/2309.02612), [implementation](https://github.com/asriverwang/BS-RoFormer) | Strong band-split music separation | Viable stereo offline teacher, but first-party checkpoint/license was not verified | Hold pending checkpoint verification |
| oBSRNN-SIMO, [code](https://github.com/magronp/bsrnn), [weights](https://zenodo.org/records/17516442) | Stereo 4-stem separation with explicit stereo handling | Best verified research-scope teacher candidate; stream low-rate descriptors only | Candidate next teacher smoke, not run in this pass |
| [COCOLA](https://arxiv.org/abs/2404.16969), [code](https://github.com/gladia-research-group/cocola) | Cross-stem harmonic/rhythmic coherence | Useful matched compatibility baseline, not sufficient spatial representation | Candidate baseline |
| MSCOL, [paper](https://arxiv.org/abs/2302.07077), [code](https://github.com/cgaroufis/MSCOL_SMC23) | Contrastive mixture/source compatibility with source presence | Low-cost source-existence baseline | Candidate baseline |
| [GRAM](https://arxiv.org/abs/2602.03307), [official repos](https://github.com/labhamlet/GRAM-T) | Masked spatial multichannel reconstruction | Binaural/FOA channel assumptions do not transfer to mastered stereo | Architecture inspiration only |

## Independent counterfactual recommendation

Before another native run, use the existing MUSDB stem rows to generate
controlled counterfactuals and recompute descriptors from waveforms:

1. Exact L/R swap and dual-mono collapse.
2. Source removal (`mixture - stem`) with residual/hash checks.
3. Equal-power re-pan of one stem at known positions.
4. Gain/mute sweeps and source swaps.
5. Off-grid delays and deterministic 3:2 / 4:3 polyrhythmic gates.

Score source-specific activity drop, pan slope/sign, ILD/width/coherence error,
onset and lag error, off-target timing stability, and LR equivariance. Use
paired song-level confidence intervals; do not pool frames as independent.

The one-row smoke already shows why this is needed: source removal was not
source-specific for drums/other, and a controlled drum re-pan produced a
predicted pan slope of only `0.053` versus target slope `1.0`.

## Teacher decision

oBSRNN-SIMO is the only current teacher candidate with a verified stereo
contract, public checkpoint, and usable research license. Its ~1.2 GB checkpoint
can be profiled on 10-second chunks, retaining only source×L/R low-rate
descriptors. Do not cache full stereo stems: the 20 GiB floor would be
threatened. HTDemucs remains a continuity control because the existing cache
averages stems to mono and its weight license is unresolved; BS-RoFormer lacks a
verified first-party checkpoint in this workspace.

No method in this pass earned native admission. A successful future arm must
pass representation gates before any body/DMX generation.
