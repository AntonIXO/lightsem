# LightSem visual-target campaign: primary-source dossier

Read-only audit. No training, vault calls, production edits, project imports, or extraction runs.
Sources were retrieved on 2026-09-22. Full snapshots and source hashes are in
`/tmp/lightsem_visual_sources/` and the accompanying JSON.

## Executive findings

1. **The local six-vector is not the Skip-BART label.** RPMC-L2 stores HSV
   histograms. Skip-BART predicts one modal hue and mean value per frame, with
   saturation fixed at 255. LightSem instead reduces histograms to two circular
   moments, mean saturation, mean value, value p90, and hue entropy concentration.
2. **`light["0"]` denotes a V-threshold variant, not a hue rotation.** The
   `extract_rpmc.py` comment “un-rotated light track” is misleading.
3. **Current Just Dance hue summaries include achromatic pixels.** There is no
   saturation/value validity condition. OpenCV assigns hue zero to black and
   achromatic pixels. Their prevalence is not measured here.
4. **Moving dilated rectangles can change the label without changing the show.**
   They delete real background between limbs. Missed boxes leave coaches visible.
   SAM 2 can be a better mask instrument, but cannot reveal occluded background.
5. **Source grouping matters more than raw frame count.** The paper claims
   held-out shows, but the official RPMC-L2 card supplies no predefined splits and
   identifies samples by hashes. A random hash split cannot reproduce the claim.
6. **Do not oversell Skip-BART's rhythm result.** In v2, change-magnitude
   correlations are hue **0.0088 versus 0.0050** for its emotion-rule baseline,
   and value **0.0294 versus 0.0058**. The table is scaled by 100. The no-skip hue
   ablation is 0.0115. No trivial shift/constant null or target spread is supplied.
7. **Vision stays on the training side.** Hold audio songs, source groups and
   inference input fixed. Compare mask quality and target richness separately.

## 1. Skip-BART: the current paper, not just the original preprint

- Primary: https://arxiv.org/html/2506.01482v2 (2025-12-31).
- Earlier version also read fully: https://arxiv.org/html/2506.01482v1 (2025-06-02).
- Official code snapshot: https://github.com/RS2002/Skip-BART/tree/88e56d1c07460276273fbbed777d248239262ced
- Full v1/v2 text, appendices, tables and captions were read. Official cards and
  `dataset.py`, `model.py`, `generate.py` were read; `finetune.py` was checked for
  target rounding and configuration. No checkpoint was run.

### Exact extraction and labels

The paper converts video frames to HSV. H has 180 OpenCV bins; V has 256.
It constructs histograms after excluding pixels below a value threshold. The
release supplies threshold variants 0, 30, 60, 90, 120, 150, 180, 210 and 240.
Hue is the mode, not a circular mean. Value is a rounded histogram-weighted mean.
Output saturation is fixed to 255. No human, UI or spatial mask is described.

**Paper and code differ:** official `dataset.py` selects H at threshold 60,
replaces H-bin 0 by the integer average of bins 1 and 179, and takes argmax. It
selects V at threshold 0, discards V-bin 0, and uses `sum(n * count[n])/sum(count)`.
Empty V counts return zero. `finetune.py` rounds the resulting value and casts it
to integer. v2 Eq. 11 applies the same threshold to H and V; Eq. 12 prints an
`n+1` weighted mean, which could yield 256 despite the stated 0–255 vocabulary.
A faithful replication must resolve these differences instead of silently
combining them.

The labels are **camera-derived appearance proxies**, not console commands from
an engineer. A livehouse frame mixes reflected light, illuminated people, haze,
camera exposure, surfaces and stage graphics. It does not identify fixture
positions or control signals. The appendix's von Mises mixture decomposition
(EM, BIC component count) is a promising circular-palette representation shown
on one frame, **not an evaluated multi-light generator**.

### Model, objective, and temporal memory

- Audio: frozen OpenL3 512-dimensional vectors; 1-second windows, 0.1-second hop.
- Backbone: bidirectional audio encoder plus autoregressive lighting decoder.
  Prior lighting tokens provide output history. An aligned audio embedding is
  concatenated with lighting embeddings at the decoder input.
- Transfer: PianoBART plus DARE task-weight merging and LoRA tuning.
- Audio-only pretraining: all-token MSE + masked-token MSE + adversarial realism
  cross-entropy. Separate discriminator real/fake loss. Mask ratio is sampled
  from 15–30%; replacement embeddings use Gaussian statistics from the sequence.
  v2 gives loss weights 0.8, 0.2 and 0.1.
- Lighting training: next-token hue CE and value CE, with task weights inversely
  proportional to the preceding epoch's validation accuracy.
- Inference: autoregressive stochastic temperature sampling with a circular hue
  step limit and a linear intensity step limit. These are not beat locks or
  certified flash-safety controls.
- Maximum context: **1,024 frames = 102.4 seconds**. Longer training clips are
  cropped. Official generation truncates to the first 1,024 audio embeddings.
  “Entire piece of music” therefore does not prove arbitrary whole-song memory,
  repeated-section recall, or cross-window consistency.

Paper Table 4 reports 240M total / 19M trainable parameters and hidden size 2048;
current code and model card use hidden width 1024. Use the actual checkpoint
configuration if replication becomes necessary.

### Population and performance limits

699 clips came from 35 performances, December 2020–July 2024, across rock, punk,
metal and core styles, with four capture devices. Clips range from 20 seconds to
about five minutes. Paper split is 8:1:1, with test shows excluded from the other
sets. Audio pretraining uses **training and validation** audio, but not test.
That is not the same protocol as strict train-only pretraining.

v2 hue MAE is 28.72 bins versus 43.43 for its rule baseline; V MAE is 51.27 versus
86.55. Hue RMSE is 36.13 versus 48.67; V RMSE is 60.74 versus 93.39. These are
priced against a DEAM emotion predictor plus a prescribed emotion-to-color
mapping, not against LightSem's beat trigger or current controller. No target
spread, constant predictor, circular-shift null, or strong deterministic timing
controller is supplied. Tiny change correlations do not establish faithful
rhythmic tracking.

The in-domain human study has 38 raters but only **three 20-second excerpts**,
shown as one animated color block with audio. Overall scores are Skip-BART
4.35/7 versus human-derived GT 4.51/7 and rule baseline 2.67/7. The GT comparison
has corrected p=0.724: **failure to reject is not equivalence**. Skip versus its
no-skip ablation is not significant overall (p=0.152), and the no-skip model's
rhythm mean is slightly higher (4.54 versus 4.50). GT is higher on surprise.

v2 adds 30 raters on three Suno-generated tracks (folk, R&B, jazz), with no
human-designed lighting GT. Skip-BART scores 4.34 versus rule 3.34 overall. This
is not broad cross-genre evidence on real recordings. The authors themselves
report excessive local fluctuations and poor long-term rhythmic stability.

### Availability and license

The HF model card declares Apache-2.0 for the model. The inspected GitHub tree
has **no LICENSE file**. Do not assume the model-card label grants a separate
blanket code license. The HF usage snippet names `Skip_BART.from_pretrained`,
whereas actual code defines `ML_BART` and loads state dictionaries. Generation
also contains likely batch/time indexing errors and saves embeddings under the
`*_gen.npy` path. These are static observations, not tested failures. Available
checkpoints are not the same thing as a verified turnkey baseline.

## 2. Official RPMC-L2 card

https://huggingface.co/datasets/RS2002/RPMC-L2/blob/1b6893810d18bde66c92aad2a6d2aef7c1012977/README.md

The whole card was read. It declares **CC-BY-4.0**. It supplies music features
(OpenL3, mel, chroma, etc.) and `light/<threshold>` arrays of shape `(F,3,256)`.
It does not list motion, human masks, semantic cue labels, fixture coordinates,
DMX, or spatial annotations.

The card explicitly says **no predefined splits**. Samples are identified by
hashes in the merged HDF5. Public loader code divides a file list by proportion,
with optional shuffle; it has no show-group field. No hash-to-show manifest was
found in the inspected card or code tree. This does not prove the paper leaked;
it means its grouped split cannot be recovered from those artifacts alone.

The paper says raw video was withheld for copyright reasons. The feature
dataset license does not clear all third-party music/video uses, and certainly
does not license Just Dance recordings.

## 3. Palmer et al. 2013: useful dimensions, not a lighting lookup table

- https://pmc.ncbi.nlm.nih.gov/articles/PMC3670360/
- https://doi.org/10.1073/pnas.1212562110
- Author copy: https://palmerlab.berkeley.edu/pdf/PalmerSchlossXuPrado-Leon(2013).pdf

The full main article, all three experiments, Materials and Methods, discussion,
references and equation images were read. The separate supplement was not
retrievable: publisher 403, PMC download challenge, and EuropePMC's API reports
that the article is not open access. Exact supplementary color coordinates and
musical work IDs are therefore **not verified** here.

This is a psychophysical study, not a learned generator. Experiment 1 used 18
50-second classical orchestral excerpts: Bach/Mozart/Brahms × slow/medium/fast ×
major/minor. The excerpts differ in more than those factors. Participants ranked
five best and five worst colors from 37 calibrated color samples: eight hues ×
four saturation/lightness cuts, plus white, black and three grays. Music and
colors received separate emotion ratings; colors received separate appearance
ratings. The music looped during selection. There are no frame labels, masks,
lighting transitions, trained parameters, or temporal memory.

The exact association summary is:

`C = (5*c1 + 4*c2 + 3*c3 + 2*c4 + c5)/15`

`I = (5*i1 + 4*i2 + 3*i3 + 2*i4 + i5)/15`

`MCA = C - I`

There were 48 US and 49 Mexican participants. Faster/major excerpts tended to
match lighter, yellower, more saturated colors than slower/minor excerpts.
Emotion correlations across **18 stimulus means** were high (for example,
US happy/sad 0.97 and Mexico 0.97), but there is no held-out predictor, target
spread, median-predictor error or lighting-quality null. These numbers do not
estimate music-to-light model accuracy. Experiments 2 and 3 used emotionally
expressive faces (24 and 16 US participants) as converging association evidence.

The authors explicitly say their correlational results do not establish causal
mediation or its direction and do not exclude direct associations. Results in
US/Mexican listeners to Western classical music do not establish universal
cross-cultural mappings, nor mandatory emotion-to-hue rules in concerts.
Colored-square matching is not the same measurement as human preference for a
moving multi-fixture show. The useful transfer is to keep **saturation and
lightness distinct from hue**, not to force one palette from tempo or emotion.

No reuse license for all stimuli/data was identified. Free PMC reading is not
an open-data or software license. Music and NimStim face images have independent
rights. No pretrained model is supplied.

## 4. SAM 2: selected extraction paper

**Ravi et al., arXiv2408.00714v2 (2024-10-28).** The mask-paper worker read all
42 pages, appendices A–H, tables, model/data cards and references, and visually
inspected Figures 1–17. No paper sections were left unread. v2 uses **SAM 2.1**
checkpoints while calling them SAM 2 for brevity. Its full dossier is embedded
in the JSON; detailed notes are `/tmp/lightsem_sam2_dossier.md`.

- Primary: https://arxiv.org/abs/2408.00714v2
- Official source: https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4

### Objective, labels, and temporal memory

SAM 2 performs promptable video segmentation, not person classification. A box,
positive/negative clicks, or a mask selects an object or part, which is linked
across frames as a masklet. SA-V masks are **category-free**: no semantic person
or action labels. Masks are annotated at 6 FPS; released videos are 24 FPS.

Training loss is `20*focal(mask) + Dice(mask) + L1(predicted IoU, true IoU) +
CE(object present)`. The lowest-loss multimask candidate gets segmentation
supervision, while all IoU predictions are supervised. On absent-object frames,
the presence head is trained but mask output is not. This supports abstention
signals; it does not guarantee a correct absence decision.

Hiera encodes each frame. Four memory-attention transformer layers condition
current features on past spatial masks/features and object pointers. A SAM-like
prompt decoder emits mask candidates, IoU estimates and presence. Memory combines
recent FIFO frames and prompted frames. The paper default is six past frames;
Small config uses `num_maskmem: 7` including conditioning. This is **not a limit
on the entire interactive video state**. Multiple coaches share image features
but have independent object memories; there is no learned inter-object reasoning.

### Training and benchmark provenance

Released checkpoints use SA-1B images plus **manual SA-V** and licensed Internal
video. The final model is not trained on every automatically labelled mask in
the SA-V release. SA-V contains 50.9k videos, 196 hours and 642.6k masklets, of
which 190.9k are human-corrected manual masklets. Internal video is separate and
not the public SA-V release. Manual labels are model-assisted and can have errors.

Many paper benchmark rows add DAVIS/MOSE/YouTubeVOS to training. Table 4's most
relevant **first-frame box** result is J&F **74.4 versus SAM+Cutie 69.4** (versus
SAM+XMem++ 67.6), averaged over 17 datasets. That is expanded-data Hiera-B+, not
released Small and not Just Dance. Interactive correction benchmarks use oracle
error locations from ground truth; automatic extraction has no such oracle.

Frozen **SAM 2.1 Hiera Small** is the initial candidate; Tiny is the cheaper
fallback. Small versus Tiny: 46M/38.9M parameters; SA-V J&F 76.6/76.5, MOSE
73.5/71.8, LVOSv2 78.3/77.3. These released-weight benchmarks use first-frame masks,
not boxes. Their 84.8/91.2 FPS are compiled A100 measurements, not an RTX 5070
prediction. No local model was downloaded or run.

### Mask quality and runtime limits

Prompt with existing validated full-body pose boxes, assign object IDs, propagate
within shots, and refresh at cuts/re-entry. Keep UI masks. SAM is **not a semantic
person detector**: it can choose clothing, a limb or a whole person. Existing
pose boxes select the intended coach but are not pixelwise truth. An automatic
all-object mask generator cannot tell which objects should be excluded.

The paper reports shot-change failures, long-occlusion problems, similar-object
confusion, thin fast details and long-video/crowded-scene errors. A predicted
empty mask can mean true absence or failure. Stylized/neon coaches, costume,
composited glow and graphic afterimages need local gold-mask QA. Decide whether
props, reflections, shadows and afterimages count as coach before scoring.

This is binary visible-object segmentation, not alpha matting or background
reconstruction. A small validated boundary margin may be necessary. Never fill
masked pixels with black and include them in the histogram. Better silhouettes
can still flicker and create false palette dynamics through changing support.

Python >=3.10, torch >=2.5.1 and torchvision >=0.20.1 are documented minima;
they do not themselves guarantee Blackwell support. The optional CUDA extension
can be disabled with `SAM2_BUILD_CUDA=0`; core PyTorch is still required. Use the
project environment and uv after recovery, not the kernel/system environment.

**Memory trap:** the official loader stores 1024² RGB float32 frames, or
12 MiB/frame: approximately **21.1 GiB per minute at 30 FPS**, before masks and
states. This is arithmetic from source, not a local measurement. CPU offload
moves the cost to RAM; bounded shot chunks and released state are still needed.
Use original video pixel coordinates for default normalized box prompts; convert
OpenCV BGR to RGB; preserve timestamps and mask resolution. Defer compilation
until correctness and memory have been tested on a short shot.

Official code/checkpoints are Apache-2.0; SA-V data are CC-BY-4.0 under download
terms. Neither license grants rights to Just Dance media.

## 5. Static code review: exact seam to repair

- `build_justdance_visual.py:39` masks four fixed UI rectangles and finite coach
  boxes padded 15% on each side. Frames are resized to half resolution. It makes
  three independent HSV histograms and invokes `extract_rpmc.reduce_light`.
  `K`, `C` and `stride_check` are unused by the scene function.
- `extract_rpmc.py:29` computes first circular hue moments, not modal hue.
  The six-vector is invariant to pixel permutation. It cannot distinguish a
  red-left/blue-right layout from blue-left/red-right. Opposed hue modes can
  cancel even when both colors are saturated. Entropy does not recover the modes.
  Value p90 measures a value quantile, not a spatial spotlight.
- Legacy extraction has no minimum visible area. An all-zero histogram yields
  hue concentration 1 and the other summaries 0, which can look valid. This is
  a static failure path, not a measured frequency.
- Stage11 `resize_mask` uses original-image dimensions correctly. It decodes an
  explicit 10 Hz grid, applies the preceding pose timestamp, and uses the active
  span offset. It tests a recording-wide 15% padding policy, then no padding if
  needed; it fails below 100 visible pixels. These improvements should remain.
- Stage11 still uses box masks and the same six summaries. It stores H, S and V
  marginals, but no joint color histogram or region structure. It is impossible
  to recover hue conditioned on saturation/value or image regions from these
  marginals alone. Rich targets require source re-decoding after data recovery.
- Stage11 normalizes mean V and V p90 independently over the entire active
  recording before cropping. They are relative intensity targets, not raw
  photometry. Preserve the project rule against regressing absolute light levels.
- The scene model is a rectified flow trained with velocity MSE on six channels.
  Both arms have an offline audio Conformer; only the denoiser is pointwise or
  temporal. Its audio radius is 138 frames (13.8 seconds each side); each temporal
  velocity evaluation has radius 126 frames. Multiple Euler steps propagate
  further. None of this restores palette/spatial information already discarded.
  There is no explicit repeated-section motif memory.

Moving masks change the observed background population even if segmentation is
perfect. Do not confuse newly visible colored pixels with a designed cue. Also
measure UI leakage, exposure, cuts, mask coverage and regional validity.

## 6. Bounded H-V plan; proposal only, not preregistered or run

First audit the extraction on the same fixed development material:

| Mask | Existing six summaries | Richer palette + region summaries |
|---|---|---|
| Current Stage11 box + UI | Baseline | Representation-only change |
| Frozen SAM2 human + same UI | Mask-only change | Combined change |

Freeze audio, timestamps, canonical song groups, valid-frame policy, intensity
normalization and source exposure. Use hand-labelled masks from a stratified
subset before judging learned lighting. Include fast motion, multiple coaches,
entry/exit, occlusion, edits, low light and desaturated scenes. Report human IoU,
boundary recall, leaked coach color, background removed, valid region coverage,
and the resulting palette/intensity target error against the box baseline.
Abstain on unreliable masks instead of hallucinating hidden background.

The richer target should retain multiple circular hue modes/distributions,
chromatic-valid mass, relative intensity and fixed screen-region composition.
Keep achromatic intensity separate. If palette slots use mixtures, handle slot
permutations and temporal identity. Do not claim screen regions correspond to
physical fixtures. Exact bins/regions/thresholds must be frozen on development
material, not selected by final lighting ratings.

After extraction is fixed, run the parent-requested **pose auxiliary on/off ×
visual supervision on/off** factorial on the **same audio songs and groups**.
The visual arm must not get extra songs. Keep seeds, budget and architecture
capacity controlled. All four arms emit lights from music only. Pose is training
supervision, not a runtime dependency or a dancer-improvement objective.

Use the incumbent controller and beat trigger as lighting comparators, plus
appropriate shift/identity controls. Evaluate emitted light with audio. Preserve
brightness matching where needed. Measure palette richness, spatial layout,
transition timing, persistence and repeated-section coherence separately. Better
mask IoU or histogram reconstruction is not product acceptance.

## 7. Optional pretrained extraction tools; no new workstream

- **Torchvision Mask R-CNN ResNet50 FPN v2 / COCO_V1:** semantic `person` class
  masks without prompts; no video memory. 46.36M parameters and 177.2 MB weights.
  Useful only if SAM box prompts are unreliable. Photo-trained person detection
  may miss stylized coaches. Torchvision code is BSD-3-Clause; dataset/image rights
  are separate. No local runtime test was made.
- **DINOv2 ViT-S/14:** frozen 21M-parameter patch features as an optional diagnostic
  or teacher, not a runtime input. Code/base weights are Apache-2.0. Generic
  semantics are not guaranteed to preserve lighting color or brightness.
  Transformer attention mixes human/UI information into nominal background
  tokens; masked pooling alone does not fully isolate background. Keep this out
  of the first mask/representation test unless the simpler targets fail.
- **Skip-BART checkpoint:** a possible music-only principal-color baseline, with
  a 102.4-second cap and the source inconsistencies above. It is not a pretrained
  multi-fixture lighting controller. No need to download it for mask QC.

No GPU runs, installs, checkpoint downloads or extraction can start while the
parent-reported data mount is missing and torch is unavailable. Do not build
replacement caches into an empty mount path. Root has about 9 GB free.

## Supplementary reading already completed: Glow with the Flow

https://arxiv.org/html/2602.08838v1 (2026-02-09), already cited in
`docs/research/2026-09-08/report-source.md`. Full text and tables were read before
the parent specified SAM2 as the extraction primary. It adds no campaign branch.

The system uses pretrained HTDemucs, All-In-One structure/beat analysis and
LarsNet kick/snare analysis, then rule-based editable light objects. No new
end-to-end audio-to-light model is trained. Its palette is clustered from video
frames in the hottest audio section: primary, secondary and background colors,
plus soft primary/secondary variants. Clustering details, stride and HSV
softening coefficients are not fully specified. No person masking is described.

A global palette and section-energy level select sustained layers; kicks/snares
trigger timed events. This is song/section/event structure, not learned motif
memory. Music-only use needs an independently supplied palette; copying its
video input into inference would violate LightSem's requirement.

32 raters judged 15 paired 10-second contexts from only three music videos in an
automotive visualization. AI versus human means were emotion 7.00/7.11, rhythm
7.27/7.20 and chromatic match 7.27/7.00. Corrected tests were nonsignificant, not
equivalence tests. Participants still described generated designs as repetitive,
predictable and too explicit; generated palettes were visibly less rich. Thus
high synchronization and congruence are compatible with poor artistic variety.
No measured authoring-time saving, released code/checkpoint, or general
multi-fixture concert acceptance was established by this paper.
