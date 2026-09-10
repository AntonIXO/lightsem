# Overnight expression and representation study

**Completed September 10.** All 12 main trainings, 12 pose probes, deliveries and independent audits finished. Recorded GPU-stage time: 9.609 h; main optimizer time: 7.724 h. No new configuration passed all gates. The incumbent-plus-adapter demonstration reduced ending border luminance by 73.57%, but expressive brightness coverage reached only 3/8 clips against the required 6/8. See [full results and deliverables](research/2026-09-10-night/RESULTS.md). The original authorization and methods follow.

The accepted task is to improve moving-head brightness and group composition, reduce the washed-out Shpongle ending, and spend about ten hours of useful serial GPU work testing audio representations and conditional generation. Mechanical acceleration limits and new fixture types are deferred. The baseline was committed as `36ed994` before implementation changes.

The run root is `/mnt/lightsem-data/experiments/lightsem/night_expression_20260909`. The system of record is `/root/lightsem-vault`. The original beat-conditioned experiment and its failed musical-control gates remain unchanged.

## Questions and evidence

There are three different questions: what the audio representation contains, what a trained decoder can recover from it, and what the emitted light actually does. A good probe does not establish that a generator uses its input. Visible movement does not establish musical control. Better reconstruction of pose from a pose-conditioned code does not establish that the code is predictable from audio.

The current inputs contain64log-mel channels and28scalars. The explicit beat channels come from a constant-tempo fit. The bar-phase input groups successive beats by four; it is not an independently observed meter annotation. The motion targets are also partly dependent: native `symmetry=1-abs(laterality)` before resampling, `section_energy` is smoothed quantity of motion, and `metric_level` uses the audio beat period. These channels cannot independently certify rhythm understanding.

TRAIN contains 419 canonical songs and 798 coach sequences. Development VAL contains 40 songs and 85 coaches. The previous Stage 11 test has already been used and is not accessed here. Raw detected COCO-17 positions and confidence masks support 2Dreconstruction diagnostics; they do not supply 3Dground truth or reliable physical contact labels.

## Main factorial comparison

| Input | Single-entry temporal flow | Repeated audio FiLM |
|---|---|---|
| Existing92features | `base_once` | `base_film` |
| Existing92plus local Fourier features | `fourier_once` | `fourier_film` |
| Existing92plus frozen MuQ | `muq_once` | `muq_film` |

Every cell has training seeds 0 and 1, equal sample exposure, equal final-step selection, the same A 4/B 3/C 17 targets and losses, and matched shared-module initialization. The FiLM modules use a 32-dimensional bottleneck and are initially identity transformations. Their additional parameters are reported, not hidden as a perfectly size-matched intervention.

MuQ is the released `OpenMuQ/MuQ-large-msd-iter`, revision `0562a57814f6f8bbd9fdea0a25921a2fce1a841a`. The encoder runs FP32 in an isolated environment. Native25Hzfeatures retain timestamps; interpolation to40Hzdoes not create additional timing information. Layers3/6/9/12are compared using canonical-song folds within TRAIN. Layer9was selected by this rule; all four native motion-probe R²values were negative. This selection is not evidence of successful transfer.

The Fourier block uses the named `flux_lf`, `flux_mid`, and `flux_hf` channels, windows2and8seconds, and16frequencies spanning0.5–8Hz. It exposes local magnitude and complex coordinates plus activity summaries. It is an inductive representation of existing envelope information, not a new source of waveform information. A separate complex-STFT novelty diagnostic examines phase information absent from the magnitude bus.

## Output adapter

The new adapter groups six heads into three mirror pairs, uses independent whole-sequence conditional draws, and mixes mirror and independent targets according to predicted side-activity balance. It uses fixed world-coordinate geometry, not a learned per-fixture DMX head. The automatic blend is a policy on motion descriptors, not a classifier trained on lighting-design labels.

Head brightness uses predicted activity and event evidence, with no constant dimmer floor. Entire groups can become dark. Static role gains reduce the background wash and white-light contributions. No song-position input, ending fade, event quota, or time-based oscillator is introduced. The existing 180°/s pan and 120°/s tilt caps remain; real actuator feasibility is not claimed.

Brightness acceptance reads encoded DMX bytes. Mirror validation checks both desired targets and emitted rays. The Shpongle comparison uses the same camera, exposure, shader time, rendering dimensions and codec on both sides. Role-isolated ending clips diagnose the washed-out image; their contributions are not additive because the renderer normalizes aggregate light.

## Diagnostics and acceptance

The reconstruction suite trains descriptor decoders of widths 128 and 256, a learned 24-dimensional pose code, and direct audio-to-pose probes using Base/Fourier/MuQ. All use confidence-masked 2Dtargets and final scheduled weights. Pose reconstruction is kept separate from audio-only inference; generated dancers are not required in the lighting playback path.

Native generation uses an ensemble energy score on two-second C trajectories, matched-noise audio controls, alternate-noise controls, closest-tempo donors, a within-song four-bar shift, zero conditioning, and beat-only conditioning. Separate auxiliary-only and core-only swaps expose which input the model uses. Holding the six explicit rhythm channels does not hold all rhythm encoded by MuQ; hybrid-input controls are labelled accordingly.

Every checkpoint receives native dispersion/rank diagnostics, actual lit-ray measurements, head-only video analysis, encoded dimmer checks, mirror checks and unrelated-group correlation. Candidate acceptance requires the mandatory tests in both training seeds. Selection then uses native energy score, never the size of a swap response or the appearance of one Shpongle excerpt. Two seeds do not provide a reliable general estimate of recipe-level seed variance.

The annotated rhythm benchmark uses358GuitarSet microphone excerpts after the two predefined timing exclusions. Zenodo downloads repeatedly timed out. The transport fallback is pinned `taohu/guitarset`; all360original microphone WAV files and360original JAMS files were verified against the packaged mirdata0.3.8per-file MD5index. JAMS formatting was restored before checking its original checksum. The audio and annotation contents were not changed.

The tested PLP recipe did not beat the constant grid on this corpus: mean beat F 1 was 0.4884 versus 0.5822 and CMLt was 0.1259 versus 0.4683. This is a negative result for that recipe on GuitarSet, not a universal rejection of local periodicity. Synthetic drift, syncopation, swing, meter and silence cases remain separate engineering tests. The main Fourier arms do not replace the beat tracker.

## Provenance and correction

An implementation error initially addressed mel columns3/5/7rather than the named flux columns67/69/71. The incomplete `audio_fourier` reconstruction probe was interrupted before final evaluation. Invalid inputs, source and run files are retained, with byte-verified compressed storage. Corrected TRAIN/VALfeatures were rebuilt, named-channel regression tests added, and diagnostic protocolv2frozen before resuming.

Four completed seed 0 probes—descriptor 128, descriptor 256, pose_code 24 and audio_base—did not consume the faulty features. Their inputs and relevant executable definitions were independently compared and found unchanged. All four were retained without outcome selection. The invalid Fourier run is excluded from scientific aggregates. No main network had been trained on the faulty features.

The main protocol freezes code, tests, inputs, normalizers, feature choice, seeds, controls, gates and final-step selection. Its common step budget is set using engineering step times and the remaining compute allocation before main training. CPU analysis and idle waiting are not represented as GPU computation. A persistent serial pipeline writes explicit stage artifacts and stops on technical failures rather than substituting models, seeds or CPU inference.

## Primary literature

- Li et al., **MERT: Acoustic Music Understanding Model with Large-Scale Self-supervised Training**, ICLR2024. Acoustic and CQTteacher objectives produce reusable music representations; its MIR results do not establish LightSem motion quality. [Paper](https://arxiv.org/html/2306.00107v4).
- Zhu et al., **MuQ: Self-Supervised Music Representation Learning with Mel Residual Vector Quantization**,2025. The paper examines Mel-RVQtargets, scaling and layer-specific behavior. The public MSD checkpoint differs from the large-data paper model. [Paper](https://arxiv.org/pdf/2501.01108), [official release](https://huggingface.co/OpenMuQ/MuQ-large-msd-iter).
- Won, Hung and Le, **A Foundation Model for Music Informatics**, ICASSP2024. Token-level tasks expose differences hidden by sequence-level tagging; context length and architecture matter. [Paper](https://arxiv.org/pdf/2311.03318).
- Zhou, Zhu and Chen, **Layer-wise Investigation of Large-Scale Self-Supervised Music Representation Models**,2025. Task-appropriate intermediate layers often outperform all-layer mixing; this study is not a direct beat-tracking evaluation. [Paper](https://arxiv.org/html/2505.16306v1).
- Kanatas et al., **What Makes a Good Layer?**,2026. Intrinsic geometric measures can guide some tasks but are not universal representation-quality scores. [Paper](https://arxiv.org/html/2608.14819v1).
- Chen et al., **BEATs: Audio Pre-Training with Acoustic Tokenizers**, ICML2023. A general acoustic representation family, not a beat tracker despite its name. [Paper](https://arxiv.org/abs/2212.09058).
- Wu et al., **Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation**,2023/2024. Audio-text semantic alignment does not itself demonstrate frame-local rhythmic precision. [Paper](https://arxiv.org/html/2211.06687v4).
- Cramer et al., **Look, Listen and Learn More: Design Choices for Deep Audio Embeddings**, ICASSP2019. OpenL3 supports local sliding windows; the one-second aggregation is a limitation for fine timing, not a categorical inability to encode local audio. [Paper](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/cramer_looklistenlearnmore_icassp_2019.pdf).
- Wang, Wang and Cai, **MotionBeat**,2025. Motion-aligned contrastive objectives and structured rhythm tests are relevant research ideas. The paper pools within beats and uses3Dmotion; usable released weights were not verified. [Paper](https://arxiv.org/html/2510.13244v1).
- Grosche, Müller and Kurth, **Cyclic Tempogram—A Mid-level Tempo Representation for Music Signals**, ICASSP2010. Cyclic folding handles tempo-octave ambiguity but discards absolute octave information. [Paper](https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf).
- Grosche and Müller, **Extracting Predominant Local Pulse Information from Music Recordings**,2011. Local periodicity is a distinct mechanism from a single global grid. Its benefit is tested rather than assumed. [Author reference](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C6/C6S3_PredominantLocalPulse.html).
- Duxbury et al., **Complex Domain Onset Detection for Musical Signals**, DAFx2003. Joint magnitude/phase novelty motivates a genuinely phase-sensitive alternative to magnitude-only caches. [Paper](https://dafx.de/paper-archive/2003/pdfs/dafx81.pdf).
- FitzGerald, **Harmonic/Percussive Separation Using Median Filtering**, DAFx2010. Spectrogram structure supports a harmonic/percussive decomposition; it is not an oracle instrument separation. [Paper](https://www.audiolabs-erlangen.de/resources/aps-w23/papers/2010_FitzGerald_HarmonicPercussiveSep_DAFx.pdf).
- Foscarin, Schlüter and Widmer, **Beat This! Accurate Beat Tracking Without DBN Postprocessing**, ISMIR2024. Strong F1and continuity are distinct; neither another model's output nor detected footfalls replace independent beat annotation. [Paper](https://arxiv.org/pdf/2407.21658).
- Tseng, Castellon and Liu, **EDGE: Editable Dance Generation From Music**, CVPR2023. Conditional diffusion, music cross-attention and multiple quality checks are relevant; optimizing beat alignment alone is insufficient. [Paper](https://arxiv.org/html/2211.10658v1).
- Alexanderson et al., **Listen, Denoise, Action!**, SIGGRAPH2023. Audio-conditioned motion is naturally probabilistic rather than a unique recovered choreography. [Paper](https://arxiv.org/html/2211.09707v2).
- Perez et al., **FiLM: Visual Reasoning with a General Conditioning Layer**, AAAI2018. Feature-wise affine conditioning is a tested architecture mechanism; transfer to this flow is a hypothesis. [Paper](https://arxiv.org/pdf/1709.07871).

MERT and MuQ checkpoint licenses are distinct from their code licenses. The downloaded checkpoints are recorded as research inputs with CC-BY-NC 4.0 metadata; this study makes no commercial deployment claim.

The additional MERT development control initially exposed an HF/Torch weight-normalization naming incompatibility. Those unscored features were archived. The corrected loader maps the legacy positional-convolution keys and requires strict loading plus equality of every checkpoint tensor. MuQ passed a separate complete checkpoint-key and tensor-equality audit. Neither incident is treated as a scientific negative result.
