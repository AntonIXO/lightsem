# Stereo and spatial audio reading for LightSem — 2026-10-04

Research notes; no models were trained or compared and no demo was produced. Historical negative results are scoped to their actual setup, not exclusions of entire method families. Recommendations below are proposed experiments, not established LightSem gains.

## Conclusion

The most direct next spatial target is **source-specific stereo image over time**: apparent left/right balance, width, cross-channel coherence/phase, and confidence, synchronized with each source's activity and rhythmic events. Preserve both input channels and both channels of every separated source. A mono cache cannot recover the discarded stereo information.

A studio stereo master is not a calibrated microphone array. Panning, doubling, phase processing and artificial reverberation can construct a spatial impression without a unique physical source location. Do not label inferred stereo pan as physical azimuth, elevation or distance. Binaural and first-order Ambisonics (FOA) results remain valuable, but they require separate data assumptions and evaluation paths.

## Primary papers read

### SpaIn-Net — strongest music-specific implementation lead

[Petermann and Kim, 2022, arXiv:2202.07523](https://arxiv.org/abs/2202.07523). Read §§2.1–2.3, 3.1–3.3 and 4 through arXiv MCP `read_paper_section` (local outline ID 3, “Methodology”; this cached extraction contains the dataset and results sections too). Official arXiv HTML independently corroborated these sections.

The model conditions XUMX source-separation streams on supplied source panning angles, testing raw angles, positional embeddings, concatenation/addition and AdaIN. Its stereo magnitude input retains channel identity. Slakh mono stems receive known constant-power panning, including a difficult two-guitar mixture. The paper reports better separation with spatial conditioning; same-class sources particularly benefit. However, angles are auxiliary inputs, not discovered instrument positions; sources are static; synthetic panning does not reproduce all professional production effects. The simplest raw-angle conditioning is competitive. This supports a feasible controlled benchmark: preserve source identity while independently changing its pan. It does not establish automatic music spatial understanding or rhythmic reasoning.

[Official code](https://github.com/darius522/spain-net): MIT license, repository available, last push reported by GitHub API 2022-08-21. Root README warns of refactoring; `code/README.md` contains almost no instructions. No released pretrained checkpoint was found in the inspected tree/README. Budget for reimplementation or code repair rather than assuming immediate inference.

### StereoCRW — useful correspondence objective, not a complete stereo scene parser

[Chen, Fouhey and Owens, ECCV 2022, arXiv:2204.12489](https://arxiv.org/abs/2204.12489). Read §3.1–3.3 and “Limitations” through arXiv MCP outline IDs 4 and 7; official HTML checked §4.1 as well.

StereoCRW learns channel-to-channel temporal correspondence by maximizing return probability in a contrastive random walk. Its ResNet embeddings retain magnitude and phase; learned correspondence produces interaural delay. Audio models train on FMA/FAIR-Play, while evaluation includes simulated microphone signals and binaural recordings. The visually conditioned branch addresses a selected speaker in a mixture, using face features. These are short-time delay representations, not instrument-separated pan trajectories. Channel-wise volume rescaling is deliberately an augmentation: that makes the delay objective complementary to, and insufficient for, a branch that must retain level/panning differences. On studio music, nonzero delay may reflect a production effect rather than physical direction. Reuse its correspondence objective as an ablation against GCC-PHAT and explicit stereo features, retaining a separate ILD branch.

[Official code](https://github.com/IFICL/stereocrw): MIT; training, evaluation, simulation, and [checkpoint download script](https://github.com/IFICL/stereocrw/blob/master/scripts/download_models.sh) available. Script lists Dropbox checkpoints for StereoCRW, MonoCLR, ZeroNCE and AV-MonoCLR. Their current download integrity and inference compatibility were not tested. FMA audio licensing is per track; the repository's MIT code license is not an audio license.

### Upmixing via style transfer — valuable factorization/data idea

[Yang et al., ICASSP 2022, arXiv:2203.12053](https://arxiv.org/abs/2203.12053). Read §§2–3 via MCP outline ID 2 (embedded under “Introduction”), and §§4.1–4.5 via ID 3.

A modified VAE learns spatial configuration separately from music content. Critically, the encoder takes **five-channel** audio; the decoder also receives stereo content whose stem positions have been rearranged independently. Thus stereo panning cannot trivially substitute for the supplied spatial code. Training uses synthesized MUSDB18 surround mixes; an additional internal surround dataset supports blind upmixing. Spatial evaluation includes interchannel-level-distribution distance (WILD) and source angle estimates. It is not an off-the-shelf stereo encoder. The useful LightSem idea is controlled factorization: the same content with different spatial layouts and different content with the same layout. Its random blind upmixing path must not be repurposed as evidence of sound-driven lamp motion.

An official public checkpoint or source repository was not located in the inspected paper/author materials. The internal surround corpus is not available through the paper. A new stereo-input adaptation is a distinct hypothesis, with reconstruction and latent-use checks required.

### GRAM — broad pretrained spatial representations worth testing with domain controls

[Yuksel et al., arXiv:2506.00934](https://arxiv.org/abs/2506.00934). Read §§3.1–3.3 and 4.1 through MCP outline ID 3 (“Related Work” in the imperfect cached outline); checked official HTML too. API metadata did not resolve the cached arXiv version, so no version-specific claim is made here.

GRAM uses masked multichannel reconstruction. The binaural path preserves level differences; the Ambisonics path adds intensity vectors. Its pretraining spatializes AudioSet through simulated room/HRTF responses, with noise, and evaluates general audio tasks plus simulated and real SELD. Temporal representations have an 80 ms frame rate in the described SELD setup. This is a useful comparison to generic mono encoders, particularly because spatial cues are explicit reconstruction targets. It does not show robust source-wise spatial understanding of professionally mastered music. FOA intensity vectors require W/X/Y/Z-like soundfield channels; duplicating stereo cannot create them. Use timestamp embeddings, with source-removal and LR-swap probes, rather than a single pooled track embedding.

[Official code](https://github.com/labhamlet/GRAM-T): CC-BY-4.0, latest push reported by GitHub API 2026-02-24. The [inference wrapper](https://github.com/labhamlet/GRAM-T/blob/master/huggingface/gramt_binaural.py) points to `labhamlet/gramt-binaural`; its Hugging Face page was inaccessible and direct model API returned HTTP 401 on this check. Therefore checkpoint accessibility and its separate license are **unverified**, despite the available wrapper. NatHEAR README includes placeholder links; do not describe the complete package as reproduced or immediately runnable.

### SAR-SSL — transferable reconstruction design, different acoustic regime

[Yang and Li, TASLP 2024, arXiv:2312.00476v2](https://arxiv.org/abs/2312.00476). Read official arXiv HTML §§III and IV-A1–IV-A2 (not MCP in this pass).

SAR-SSL masks frames in one channel and reconstructs complex STFT values from separate spatial and spectral encoders, using a decoder without temporal interactions to pressure the encoders to carry information. It estimates TDOA, reverberation and related room parameters after transfer. Its formulation assumes one static source and two static microphones with time-invariant room responses. This is not the mixture of independently processed, moving/panned instruments in a stereo master. The decomposition and cross-channel masking are useful ideas to reimplement on controlled music remixes, but static-room targets should not be claimed as ground truth for arbitrary songs.

[Official code](https://github.com/Audio-WestlakeU/SAR-SSL): MIT, last push reported 2024-10-11. `code_v1` is documented as matching paper results, whereas `code` has results marked as under testing. README lists checkpoint filenames, but no download was verified. Pretraining depends on WSJ0 plus simulated/real room datasets, each with separate access terms.

## Broader families retained for the candidate map

- Classical pan/delay masking and clustering: DUET, ADRess, PROJECT, multichannel NMF and primary/ambient decomposition. Discovered in the primary papers' related work; originals were not independently deep-read here. They remain useful inexpensive baselines and potential pseudo-label sources, with assumptions about source overlap and panning explicitly tested.
- Audio-visual spatial correspondence: [Telling Left from Right, arXiv:2006.06175](https://arxiv.org/abs/2006.06175), and [audio-visual spatial alignment, arXiv:2206.00970](https://arxiv.org/abs/2206.00970). Discovered and abstract-screened. Useful if genuine synchronized spatial video/360-degree capture is available; commercial music videos need not visually show the source layout of the master.
- Modern spatially informed band-split separation: [Yang et al., 2024 author PDF](https://xianruiwang.github.io/files/mss_2024.pdf), “Stereophonic Music Source Separation with Spatially-Informed Bridging Band-Split Network.” Screened source/method excerpts: uses panning-angle prior and chooses the left channel as separation reference. A candidate architecture extension, not verified stereo-source-image output.
- Binaural/FOA scene foundation models, array SELD and room-acoustic SSL remain separate test families. Their data path may become directly relevant to a future live-capture mode.

## Concrete LightSem proposal — inference from the literature

1. Keep stereo waveform input and aligned complex STFT or explicit magnitude/phase features. Avoid independent per-channel normalization that erases level differences. Use both L/R and optional mid/side views; evaluate the mono frontend as a deliberate ablation.
2. Preserve the stereo source image from a separator. Cache `source × channel × time`, original mixture residual, activity and separator confidence. A source image can be wide or diffuse; do not reduce every instrument to one point immediately. Teacher separation is uncertain pseudo-ground-truth, not a label oracle.
3. Train or probe a temporal spatial branch with source slots: pan/ILD per band, width/side energy, coherence/phase-delay distributions, activity and confidence. Source appearance/content and rhythmic onset/phrase representations remain separate factors with interaction layers. Pan changes should alter spatial factors while preserving event timing; timing changes should alter rhythm factors without inventing spatial motion.
4. Generate controlled stem remixes for exact pan/width/delay labels, including two sources of the same instrument, shared-pan sources, sparse holds, overlaps, panning automation and frequency-dependent effects. Split by composition/source recording before generating variants. Keep synthetic and real-master results separate.
5. Compare explicit DSP, classical separation, existing stereo separators plus descriptors, StereoCRW-inspired delay loss, spatial masked reconstruction (SAR-SSL/GRAM ideas), and spatial/content factorization. A scoped prior MuQ/MERT or separation failure does not forbid these tests; it increases the value of changing channel preservation, source representation and targeted metrics explicitly.

## Counterfactual and collapse gates before downstream claims

| Intervention | Required spatial behavior | Required invariant or distinct failure diagnosis |
| --- | --- | --- |
| Swap L and R | Pan/ILD sign reverses; delay sign reverses; spatial source image mirrors | Instrument identity, source event times, phrase boundaries and appropriate width magnitudes persist |
| Re-pan one known stem, same content/levels elsewhere | Only that source's spatial factor moves reliably | Untouched source factors and all onset timestamps remain stable |
| Collapse input to dual mono | Confident side-specific image disappears | Content/rhythm can remain; this is loss of spatial information, not total representation collapse |
| Add width/delay/reverb without changing notes | Relevant width/coherence/phase factors change | Do not call the effect a physical moving object; rhythm onset robustness measured separately |
| Remove/freeze one stem | Its activity/events disappear or hold as appropriate | Other source events survive; no extra latent motion invented by time alone |
| Replace one source with same-BPM different rhythm | Source rhythm and cross-source interaction features change | A global tempo token alone cannot pass |
| Duplicate a source slot | Diagnostics identify shared content/duplicate trajectory | Extra slots do not count as extra independent sources |

Measure finite values, silent-frame behavior, active-slot counts, feature variance/effective rank, source assignment stability, exact duplicate streams, pan error on controlled mixtures, ILD/IPD/coherence error, separation quality and spatial distortion separately. Record source-removal selectivity and LR equivariance error. Compare learned branches to explicit features and frozen/random/shuffled controls before training a downstream controller.

No light-output benefit follows automatically from an encoder probe. Any eventual comparison must inspect representation → body/mapper → constrained emitted DMX, preserving the accepted P2 color baseline and the accepted source-body policy. Shpongle and AURORA — Cure For Me require separately recomputed audio paths and their own body trajectories; shared controller/seed/rendering settings remain fixed. Include similar-BPM different songs and tempo-aligned comparisons. Never substitute independent choreography for failed transfer.

## Data and verification notes

- [Slakh2100 official release](https://zenodo.org/records/4599666): Zenodo API confirms CC-BY-4.0. Exact stems/MIDI and controllable remixes make it suitable for the first supervised spatial intervention set; synthetic timbre and simple panning remain domain gaps. Use the cleaned split/duplicate guidance from the release.
- [MUSDB18 official release](https://zenodo.org/records/1117372): stereo stems and mixes, educational-use terms with commercial permission required from rights holders. It supports natural stereo reconstruction checks, but `other` combines many sources.
- [FMA official repository](https://github.com/mdeff/fma): dataset audio distributed under each artist's selected license; metadata CC-BY-4.0 and repository code MIT. Suitable for unlabeled stereo correspondence with per-track channel/quality/license checks.
- arXiv MCP abstract/search calls returned HTTP 429. Direct `download_paper` subsequently succeeded through the HTML endpoint for the four main papers. Local `get_paper_outline` and `read_paper_section` returned success. Metadata fields (authors/date/version) were incomplete, and section outlines sometimes swallowed later sections; actual section text and official HTML were inspected rather than relying on outline labels. SAR-SSL was read from official HTML only.
- Repository/API checks establish what is visible, not successful installation, weight download or reproduction. No training, separator inference, empirical counterfactual test or LightSem benchmark was run in this research pass.
