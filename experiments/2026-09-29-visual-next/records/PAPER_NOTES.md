# Verified paper reading notes

All10 Perplexity recommendations and7 additional primary papers were fetched. Exact PDFs/extracted text and SHA256 are in PAPERS.json. Read scopes name inspected sections, not an exhaustive claim to have read every appendix. No model weights were downloaded.

## DINOv3

[Primary source](https://arxiv.org/abs/2508.10104v1) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/dinov3.pdf) · [Official code/project](https://github.com/facebookresearch/dinov3)

**Read:** Abstract/introduction; Gram anchoring Sec4.2; distilled family; dense evaluation Sec6.1; official release/preprocessing instructions.

Dense patch features are more suitable than one global semantic embedding for retaining local layout. The paper separates global discrimination from local consistency using Gram anchoring; this is a reason to inspect patch features, not to train a new foundation model here. The official21M ViT-S/16 is a bounded first candidate. Exact color fidelity and music-to-light transfer are unmeasured. Weights require an access process; verify access and checksum before execution. Preserve explicit HS/color targets and test low-semantic backgrounds, geometry and color perturbations.

## VideoPrism: A Foundational Visual Encoder for Video Understanding

[Primary source](https://arxiv.org/abs/2402.13217v3) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/videoprism.pdf) · [Official code/project](https://github.com/google-deepmind/videoprism)

**Read:** Abstract; architecture and two-stage training Sec2.2–2.3; ablations and limitations Sec3.5–3.6; official code availability.

A frozen video encoder retains spatiotemporal outputs and combines semantic global/local distillation. It provides a useful temporal-representation comparator in principle. Its paper explicitly limits long-video understanding: the encoder consumes16 sampled frames. The official implementation currently uses JAX/Flax and lists PyTorch support as a TODO, so it is deferred from the first PyTorch pilot. Large and partly private pretraining data leave overlap and transfer questions; no stage-light or precise chromatic timing result is established.

## V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

[Primary source](https://arxiv.org/abs/2603.14482) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/vjepa21.pdf) · [Official code/project](https://github.com/facebookresearch/vjepa2)

**Read:** Abstract; Sec2.3 dense and deep supervision; Sec3.10 distilled models/Table11; conclusion; official80M Base384 release.

Supervising visible as well as masked tokens and intermediate layers improves dense video features alongside global tasks. The released80M ViT-B/16 provides a specific future temporal teacher. Use a frozen model if pursued; do not port its large-scale pretraining recipe into this project. Keep explicit chromatic targets because video understanding scores do not establish color faithfulness. Published robotics/action gains are not LightSem effect sizes. Compare ordered clips with order-destroyed controls, and measure actual VRAM and feature cache cost first.

## Putting the Object Back into Video Object Segmentation

[Primary source](https://arxiv.org/abs/2310.12982) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/cutie.pdf) · [Official code/project](https://github.com/hkchengrex/Cutie)

**Read:** Abstract; object transformer and memory Sec3; runtime/ablation discussion; Sec4.3 and supplementB failure cases; official release.

Object-level memory complements local pixel matching to improve coherent video object masks. This supports testing propagation from fixed SAM2 initialization against independent per-frame SAM2, with identical initial prompts. The paper documents failures for similar objects occluding or approaching one another; coach identity and foreground leakage therefore need independent masks. A segmentation benchmark gain is not a color-observer gain. Test retained-background HS error directly, not just mask smoothness or IoU, and reset at declared shot boundaries.

## RAFT: Recurrent All-Pairs Field Transforms for Optical Flow

[Primary source](https://arxiv.org/abs/2003.12039) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/raft.pdf) · [Official code/project](https://github.com/princeton-vl/RAFT)

**Read:** Abstract; feature/correlation/update methods Sec3.1–3.3; timing/parameter comparison Sec4.4.

All-pairs correlation and iterative updates offer dense motion estimates; the paper includes a small roughly1M-parameter variant. Flow is useful as a diagnostic for moving background regions or mask consistency, not as unquestioned ground truth. Light flashes, color changes, blank stage areas and cuts may violate its transfer assumptions; this is a LightSem-specific concern, not a result measured by the paper. Include static-image color-change controls and known translations before using flow to veto labels. No first-pilot motion-controller replacement is proposed.

## Regularized Discrete Optimal Transport

[Primary source](https://arxiv.org/abs/1307.5551) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/regularized_ot.pdf) · [Official code/project](https://arxiv.org/abs/1307.5551)

**Read:** Abstract; relaxed transport and graph regularization Sec3; color transfer Sec4.1–4.2; algorithm/scaling discussion.

Relaxed mass conservation and graph regularization help transfer multimodal color distributions without amplifying noise. This motivates a constrained fixture-assignment oracle and distributional objectives rather than collapsing a palette to its mean RGB. The paper is image recoloring, not a physical lighting controller: fixture gamut, overlap, emitted luminance and temporal transitions must be constrained independently. Histogram matching alone ignores spatial layout. Keep a regional held-out error and palette-permutation control; an oracle improvement establishes output capacity only.

## Palette-based Photo Recoloring

[Primary source](https://pixl.cs.princeton.edu/pubs/chang_2015_ppr/index.php) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/palette_recolor.pdf) · [Official code/project](https://pixl.cs.princeton.edu/pubs/chang_2015_ppr/index.php)

**Read:** Official author page; abstract; palette construction Sec3.2; gamut/luminance goals Sec3.3–3.4; user-study discussion; video limitation Sec5.1.

The method uses a small editable palette, perceptual color space and gamut-aware transformations with monotonic luminance. This suggests retaining multiple palette components and testing their allocation to fixtures. Its video extension uses one frame palette throughout a sequence and explicitly notes long-sequence limitations. Image-space recoloring and its user study do not validate DMX rendering or music fit. The discussion notes that simple color distances can misrank visible artifacts; retain human inspection and temporal/brightness guards alongside numerical reconstruction.

## DreamSim: Learning New Dimensions of Human Visual Similarity using Synthetic Data

[Primary source](https://arxiv.org/abs/2306.09344) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/dreamsim.pdf) · [Official code/project](https://github.com/ssundaram21/dreamsim)

**Read:** Abstract; metric training Sec4.2; attribute sensitivity Sec5.2; foreground/background discussion; supplementC limitations.

Human similarity supervision makes the embedding sensitive to color/layout beyond purely semantic baselines. However, its analysis emphasizes foreground objects over backgrounds and its synthetic dataset is object-centric; this is a poor match to person-masked lighting backgrounds. Therefore it is a possible secondary diagnostic only after controlled-light perturbation validation, not a primary lighting-quality metric. Do not evaluate a feature teacher with the same embedding and call agreement independent evidence. Actual music-fit/design preference remains a separate human endpoint.

## INSID3: Training-Free In-Context Segmentation with DINOv3

[Primary source](https://arxiv.org/abs/2603.28480) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/insid3.pdf) · [Official code/project](https://github.com/visinf/INSID3)

**Read:** Abstract; Sec3.1 positional subspace; Sec4.2 bias analysis; supplementF limitations.

The paper shows spurious cross-image matches at similar coordinates in low-semantic regions and suppresses a positional subspace estimated from noise. This directly motivates dark/flat-background and location controls for DINOv3 in LightSem. Do not automatically remove positional information: it may be useful for within-scene spatial composition, and the paper distinguishes those uses. The method needs a reference mask and handles one concept at a time, so it is not an immediately justified YOLO replacement for multiple moving coaches.

## TransNet V2: An effective deep network architecture for fast shot transition detection

[Primary source](https://arxiv.org/abs/2008.04838v1) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/transnetv2.pdf) · [Official code/project](https://github.com/soCzech/TransNetV2)

**Read:** Four-page paper: introduction, architecture, synthetic/real training, comparison procedure and limitations; official code.

The detector combines temporal convolutions and learned/RGB-histogram similarities at48x27 resolution. It evaluates centered windows and uses a fixed0.5 threshold; no live-causal claim follows. The paper warns about false hits from abrupt camera/environment/object changes and missed transitions. Use it as an edit candidate detector, not truth: a color flash may be a useful light event. Validate filtering against independent content-event annotations, preserve recall/coverage, and retain cut-coincident events as a separate reported category. Its headline benchmark results do not predict LightSem effect size.

## Tracking Anything with Decoupled Video Segmentation

[Primary source](https://arxiv.org/abs/2309.03903) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/deva.pdf) · [Official code/project](https://github.com/hkchengrex/Tracking-Anything-with-DEVA)

**Read:** Abstract/introduction; in-clip consensus and propagation; Sec4.6 limitations.

Image segmentation and temporal propagation can be decoupled, allowing current image masks to be reused. The paper uses bidirectional consensus and notes delayed discovery of new objects between image-model updates. This motivates an independent mask-stability arm, not detector replacement. It also means future-aware propagation cannot be silently used in a supposedly causal visual forecast. For the first test, bounded Cutie propagation from fixed anchors is simpler; DEVA remains a supported alternative under a distinct recipe, with fresh-object and persistent-false-positive controls.

## Mask-Free Video Instance Segmentation

[Primary source](https://arxiv.org/abs/2303.15904) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/maskfreevis.pdf) · [Official code/project](https://github.com/SysCV/MaskFreeVis)

**Read:** Abstract; Temporal KNN-patch Loss Sec3.1.1–3.1.2; correspondence/ablation discussion.

Temporal KNN patch correspondences provide mask-consistency supervision with zero, one or several matches, handling ambiguous/no correspondences better than an unconditional one-to-one assumption. Mask-free does not mean annotation-free: the system uses bounding boxes. Its consistency idea motivates a diagnostic only; training a new segmentation model is outside the first pilot. Appearance changes and repeated graphics can invalidate correspondences, so independent foreground masks and retained-background color error remain necessary.

## A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation

[Primary source](https://openaccess.thecvf.com/content_cvpr_2016/html/Perazzi_A_Benchmark_Dataset_CVPR_2016_paper.html) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/davis2016.pdf) · [Official code/project](https://github.com/davisvideochallenge/davis-matlab)

**Read:** Abstract; Sec4.1 region, boundary and temporal metrics; validation and decay results.

Region IoU, boundary quality and temporal stability measure different failure modes. The paper cautions that occlusions and strong deformations can be mistaken for instability. Thus smoother masks alone cannot qualify LightSem targets. Report independent human-mask IoU/boundaries and errors in the retained color distribution, grouped by source-song family. Borrow the evaluation distinctions without claiming that DAVIS performance transfers to stylized choreography backgrounds or downloading an unrelated dataset.

## DINOv2: Learning Robust Visual Features without Supervision

[Primary source](https://arxiv.org/abs/2304.07193) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/dinov2.pdf) · [Official code/project](https://github.com/facebookresearch/dinov2)

**Read:** Abstract; data curation Sec3; DINO/iBOT training Sec4; model distillation and ablation discussion.

Released frozen patch features and distilled small models provide a practical baseline for adding visual structure without foundation-model training. Image/global and patch objectives do not guarantee exact chromatic information. Start with ViT-S/14 plus explicit color/layout channels, retain a dimension-matched random-capacity control, and evaluate independent event targets rather than reconstructing supplied statistics. Public pretraining can overlap source imagery; canonical grouping does not prove absence of pretraining exposure.

## VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

[Primary source](https://arxiv.org/abs/2203.12602) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/videomae.pdf) · [Official code/project](https://github.com/MCG-NJU/VideoMAE)

**Read:** Abstract; temporal redundancy/leakage discussion Sec3.2; tube masking; domain-shift and conclusion discussion.

High-ratio tube masking makes reconstruction less susceptible to copying redundant neighboring frames. The paper emphasizes data quality/domain shift. This motivates order-destroying controls and temporal leakage checks, not a claim that masking inherently preserves color or predicts music-driven light. We defer training this model and use a small released frozen temporal candidate only in a preregistered qualification stage; verify the actual checkpoint terms and runtime before any later acquisition.

## ColorVideoVDP: A visual difference predictor for image, video and display distortions

[Primary source](https://arxiv.org/abs/2401.11485) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/colorvideovdp.pdf) · [Official code/project](https://github.com/gfxdisp/ColorVideoVDP)

**Read:** Abstract; display model and opponent/temporal processing Sec3.1–3.2; stated limitations.

This full-reference metric models color and luminance distortions over space/time under specified display/viewing conditions. It is useful for matched-scene encoding or renderer-difference diagnostics when those conditions are explicit. Comparing unrelated source-video and stage compositions is not a valid substitute for perceived design quality. The paper lacks higher-level saliency/annoyance modeling and does not establish music fit. Keep it secondary; a favorable score cannot promote a new lighting baseline.

## Synchformer: Efficient Synchronization from Sparse Cues

[Primary source](https://arxiv.org/abs/2401.16423) · [Downloaded PDF](/root/lightsem/runs/visual_next_20260929/papers/synchformer.pdf) · [Official code/project](https://github.com/v-iashin/Synchformer)

**Read:** Abstract; segment-level architecture/multimodal pretraining Sec3; offset/synchronizability evaluation discussion.

A model trained for natural audiovisual events predicts offset from sparse short segments and separately considers whether a clip can be synchronized at all. This informs a transfer-check design, not a primary stage-light score: abstract illumination is not necessarily a visible physical sound source. Require known shifts, metronome/rate-matched and donor controls before any use. Do not infer designed appearance or genuine music conditioning from an unqualified synchronization model.
