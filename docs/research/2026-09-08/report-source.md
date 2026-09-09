# LightSem: where the research should go next
**Research review for Anton · 8 September 2026**

## The decision

Continue with an explicit whole-song lighting plan and a small, actively queried control space. First repair the evaluation materials. Keep a distributional lighting decoder as a gated research branch.

The repository has useful audio descriptors, a working conductor and renderer, and a first preference pilot. Its negative results are substantial, but their scope is narrower than “audio cannot generate designed lighting.” The failed planner did not test explicit motif reuse; the group decoder did not reproduce the main mechanisms of Skip-BART or SeqLight. The comparison on page 2 explains why.

Three research directions have the clearest connection to the actual product:

- **Remember musical returns.** Give recurring sections related palettes, spatial patterns and movement motifs, with controlled variation. Test this directly against equally varied lighting without recurrence memory. [MusicFrameworks](https://www.cs.cmu.edu/~rbd/papers/frameworks-ismir2021.pdf) motivates the representation; its lighting transfer remains a hypothesis.
- **Spend human attention on informative choices.** Optimize three or four interpretable controls with active comparisons or synchronized line search. [qEUBO](https://proceedings.mlr.press/v206/astudillo23a.html) and [Sequential Line Search](https://koyama.xyz/project/sequential_line_search/) provide implementable mechanisms.
- **Preserve richer scene targets.** Test colour distributions and temporally consistent fixture assignment before investing in a complete generative training stack. [SeqLight](https://arxiv.org/html/2605.03660v1) makes this distinction concrete.

## A useful new warning from the literature

The 2026 paper [Glow with the Flow](https://arxiv.org/html/2602.08838v1) reports that lighting can receive strong synchronization and congruence ratings while viewers still find it repetitive or too literal. Its generated palettes also lose richness. That closely matches the failure mode LightSem needs to measure: synchronization and aesthetic quality are separate outcomes.

This is a research recommendation, not evidence of a new improvement. No new lighting model, rendering campaign or GPU experiment was run for this review.

## Scope and reading route

This review targets **pre-rendered audio → stage lighting**. Whole-song context is allowed. Dance is training supervision; no generated dancer is required at inference. The current Just Dance extraction is outside this review.

Read pages 2–6 for the evidence, 7–9 for six discriminating experiments, and 10 for the implementation order. The reference notes distinguish full-paper verification from abstract-only and repository-only evidence. Availability was checked as of the report date.

<!-- PAGEBREAK -->
# What the project already establishes

## The actual implementation

The learned demo path is `render_song_demo.py`: audio features → Conformer descriptors/events/style → ConductorHybrid → DMX → renderer. The generic CLI render path remains an older rule-based route. Stage 9's GroupEngine is an experimental alternative, so comparisons must identify the entry point and configuration explicitly. [Continuation audit](../../CONTINUATION_AUDIT_2026-09-07.md)

The model supplies useful rhythmic and movement descriptors. Much of the visible composition still comes from conductor rules and fixture assignment. The new manual ratings concern nine **ConductorHybrid** controls; they do not rate GroupEngine's wave or colour-spread controls. [Ratings audit](../../HL_RATINGS_2026-09-08.md)

| Local result | What it establishes | What remains open |
| --- | --- | --- |
| Stage 9 scene regression collapsed | Six scene moments and the implemented training recipe failed | Histogram targets, autoregressive light tokens, persistent fixture assignment |
| Offline bar planner had approximately zero R² | Eight per-bar knob targets were poorly predicted | Explicit recurrence IDs, motif memory and section-linked variation |
| Stems/MERT and model scaling gave little useful gain | Repeating those recipes is low priority | A different target or decision representation |
| Cue-held output improved beat alignment but failed the aesthetic gate | Better synchronization can be insufficient | Spatially distributed motion and slower compositional structure |
| Operator with true history degraded on its own history | The tested history model had a rollout problem | Other generation recipes, evaluated with their own generated state |

These are findings from the [local continuation audit](../../CONTINUATION_AUDIT_2026-09-07.md), not a new replication. In code, `group_decoder.py` emits parallel parameters for five roles and reduces the scene to six moments; `train_group_decoder.py` applies normalized scene MSE and hazard BCE. That differs materially from the primary methods reviewed next.

## The 39 ratings are a pilot

All 39 ratings come from Anton across 29 registered song groups. Grouped held-out accuracy is **56.4% for music fit** and **59.0% for designed appearance**; both conditional uncertainty intervals include chance. The first equals its majority-side accuracy baseline. Appearance log loss is 0.739, worse than the uninformative 0.693. These measure preference prediction, not a generated show winning against the default. [Ratings audit](../../HL_RATINGS_2026-09-08.md)

Saved-video image luma differs by a median **16.1%** within pairs, despite a 0.67% median difference in the DMX matching statistic. A blinder is squared by the metric but linear in the renderer. This is a confirmed mismatch, although it does not explain every difference. Image luma is not calibrated physical luminance.

Keep the original pilot immutable. The H-L confirmation gate remains unmet, and a fresh fitted-versus-default preference evaluation has not happened. Canonical audio identities also need checking: matching titles alone do not prove duplicate recordings.

<!-- PAGEBREAK -->
# Direct lighting research: useful distinctions

## Skip-BART · existing paper, worth revisiting accurately

**Zhao et al., 2025 / ICLR 2026 repository designation.** The method combines an offline music encoder with autoregressive discrete hue/value generation, aligned audio/light embedding skips, PianoBART transfer and constrained sampling. Its RPMC labels come from videos: 699 samples from 35 performances, concentrated in rock, punk, metal and core. These are camera-derived colours, not console-authored DMX. [Full paper](https://arxiv.org/html/2506.01482v2)

The [official repository](https://github.com/RS2002/Skip-BART) links [weights](https://huggingface.co/RS2002/Skip-BART) and [processed data](https://huggingface.co/datasets/RS2002/RPMC-L2). Model and dataset cards declare Apache 2.0 and CC BY 4.0 respectively; a separate repository license was not located. Its primary-colour output and fixed saturation limit direct transfer. A nonsignificant difference from humans does not establish equivalence.

## SeqLight · existing paper, a materially different decoder

**Zhao et al., May 2026 preprint.** SeqLight predicts full hue/value histograms with KL loss, then decomposes them into individual lights using a goal-conditioned policy. HER-generated demonstrations, behavior cloning, AIRL and GRPO replace professional per-fixture demonstrations; temporal constraints discourage discontinuities. [Full paper](https://arxiv.org/html/2605.03660v1)

The study covers 30 participants and six pieces, including three generated out-of-domain songs. That is limited generalization evidence. Code exists, but trained weights and a repository license were not verified; the [README](https://github.com/RS2002/SeqLight) remains incomplete. Start with the representation and assignment questions, not an assumed ready-to-run full checkpoint.

## Glow with the Flow · new, close to the renderer problem

**Robinson et al., PacificVis Notes 2026.** Pretrained audio analysis and source-video palettes feed rules that generate sustained layers, spatial modulation and discrete events in editable light objects. It is not a newly trained audio-to-light generator. Its 32-person study used 15 contexts from three videos in an automotive visualization; corrected nonsignificant tests establish neither equivalence nor authoring time savings. Code/data were not located. [Full paper](https://arxiv.org/html/2602.08838v1)

Borrow the editable representation and separation of slow layers from fast events. Audio-only inference needs an independently supplied palette.

## Two leads with unresolved assets

**AuraLuxMuse / LightMuse:** the [source repository](https://github.com/joeyjoestar111/AuraLuxMuse) handles fixture attributes and cue retrieval, and contains generated grandMA2 XML. Those files are model outputs, not verified authored training examples. No primary paper, source training corpus, weights or license was verified. Treat it as a code reference for cue representations.

**LightingGen:** Wang et al., IEEE TMM 2026, is a verified publication, **LightingGen: A DMX Based Generation Method for Entertainment Stage Lighting**. [Publisher DOI](https://doi.org/10.1109/TMM.2026.3664978) and publisher-deposited metadata establish the record. Full methods and code/data were inaccessible or not located. Dataset hours and named components appearing only in secondary summaries are excluded here.

<!-- PAGEBREAK -->
# Whole-song structure: remember, then vary

## The representation is the important new experiment

`lightsem/audio/structure.py` already computes recurrence-based clustering, but uses it to produce boundaries. Its later labels are energy heuristics. The generator does not retain section identity to recall a lighting motif. `train_planner.py` instead regressed per-bar controls. An explicit recurrence experiment therefore changes the mechanism.

A proposed intermediate representation is:

> **Audio timeline → section identity + position within section → motif identity + controlled variation → continuous fixture trajectories → DMX**

For example, A–B–A′ could recall a related palette and wave pattern when A returns, while changing density or phase. This is a proposed lighting design, not a learned correspondence established by a paper. Keep colour diversity and motif identity separate: more frequent random colours need not produce more coherent variety.

## Papers that make the distinction useful

**MusicFrameworks — Dai et al., ISMIR 2021.** A hierarchical symbolic melody representation separates structural repetition and rhythmic/melodic controls. The transferable idea is to represent reuse explicitly. It does not establish lighting performance; an official implementation was not located in the bounded search. [Author PDF](https://www.cs.cmu.edu/~rbd/papers/frameworks-ismir2021.pdf)

**Whole-Song Hierarchical Generation — Wang, Min and Xia, ICLR 2024.** Four levels connect global form to local symbolic accompaniment, using cascaded diffusion and external controls. The paper links code/checkpoints, but their contents and terms were not audited. This supports a hierarchy as an architectural precedent; it does not guarantee literal chorus-motif reuse or justify porting a music diffusion model. [Proceedings and paper](https://proceedings.iclr.cc/paper_files/paper/2024/hash/39d239a30bb40536a5e4e78b5780ba82-Abstract-Conference.html)

**Repetition-Structure Inference with Formal Prototypes — Finkensiep et al., ISMIR 2023.** Formal templates constrain a repetition grammar. The proof of concept studies 298 melodies of 8–24 notes; it is not an audio segmenter. Its [Julia code](https://github.com/DCMLab/form-repetition-ismir23) uses an ILP solver, defaulting to Gurobi. Exact inference is expensive. Borrow the interpretation of repeated sections before considering this specialized implementation. [Paper](https://archives.ismir.net/ismir2023/paper/000045.pdf)

## Analysis front ends already in the vault

**SongFormer — Hao et al., 2025; revised April 2026.** Short/long-context music features and heterogeneous supervision support boundary and function prediction. The [official repository](https://github.com/ASLP-lab/SongFormer) has training/inference code and checkpoint links; code is CC BY 4.0, while checkpoint terms require separate review. Its pop-oriented taxonomy has no dedicated EDM drop class. Profile resources before scheduling it on the 12 GB GPU. [Full paper](https://arxiv.org/html/2510.02797v3)

**EDMFormer — Sajeer et al., March 2026 preprint.** EDM-specific adaptation and the 98-track EDM-98 corpus are relevant, but the small, singly annotated corpus limits conclusions. An actual code/data download was not located. Keep this as a watch item, not an immediate dependency. [Full paper](https://arxiv.org/html/2603.08759v1)

<!-- PAGEBREAK -->
# Learn from fewer, better human decisions

## Start with a small, observable control space

The immediate task is choosing settings Anton likes across songs. It does not require a neural reward model. Freeze brightness initially and expose three or four controls with clear visual effects: wave phase spread, motion rate, colour-transition duration and colour-change frequency. The existing 39 ratings cannot supervise controls that were absent from their renders.

**Sequential Line Search — Koyama et al., SIGGRAPH 2017.** Bayesian optimization proposes a line through a larger design space; people select a preferred point. [Official MIT C++/Python code](https://github.com/yuki-koyama/sequential-line-search) makes it an approachable starting point. A synchronized loop and slider may suit time-varying lighting, but that adaptation is untested. A line-search task is not equivalent to one binary vote. [Project](https://koyama.xyz/project/sequential_line_search/)

**qEUBO — Astudillo et al., AISTATS 2023.** Select comparisons by expected utility of their best option. Its optimality statements have assumptions; they do not promise sample efficiency on this project. The [current BoTorch tutorial](https://botorch.org/docs/tutorials/preference_bo) supplies PairwiseGP and analytic EUBO, with CPU operation and MIT library licensing. This is the most direct active-pair baseline. [Paper record](https://proceedings.mlr.press/v206/astudillo23a.html)

**BOPE — Lin et al., AISTATS 2022.** Separate parameters → measurable outcomes from outcomes → human utility. This could connect candidate selection to useful renderer diagnostics. However, a few scalar summaries may omit what matters in a video. Original MIT replication code is archived; a [maintained official tutorial](https://botorch.org/docs/tutorials/bope) remains available. [Paper](https://arxiv.org/abs/2203.11382)

## Useful extensions, with narrower evidence

**Sequential Gallery — Koyama, Sato and Goto, SIGGRAPH 2020.** Plane-search galleries have [MIT code](https://github.com/yuki-koyama/sequential-gallery). A preliminary six-person study averaged 5.36 subtasks to satisfaction. Each subtask involved a gallery, not a single A/B judgement; viewing temporal variants may be much slower. Compare quality per minute, not nominal task count. [Author paper](https://koyama.xyz/project/sequential_gallery/download/author-version.pdf)

**Constrained PBO — Iwai et al., IJCAI 2025.** Preference acquisition incorporates feasibility. Eleven professional designers performed tasks with 50 comparisons in 6D/12D spaces; the constraint was predicted CTR, not deployed CTR. No official code release was verified. LightSem can initially reject measured luminance or clipping violations directly. [Proceedings paper](https://www.ijcai.org/proceedings/2025/0463.pdf)

**Consecutive PBO — Erarslan et al., November 2025 preprint.** Candidate reuse and explicit indifference are attractive. Full acquisition is costly: a reported 6D synthetic case took 345±34 seconds per iteration. A downloadable official code archive was not verified. Borrow ties and reuse before the full optimizer. [Full paper](https://arxiv.org/html/2511.05163v1)

**Preferential Multi-Objective BO — Astudillo et al., 2024/2025.** Separate objective-specific choices can explore trade-offs, but the inspected version evaluates synthetic/simulated tasks and no code was verified. Since the two local questions agree on 33/39 pairs, a full Pareto-search system is premature. [Readable paper](https://arxiv.org/pdf/2406.14699)

<!-- PAGEBREAK -->
# Timing, metrics and usable data

## Beat tracking is a bounded benchmark opportunity

**Beat This! — Foscarin, Schlüter and Widmer, ISMIR 2024.** The transformer predicts beats/downbeats without mandatory DBN postprocessing. Strong F1 results coexist with weaker continuity measures in the paper. The [official repository](https://github.com/CPJKU/beat_this) provides MIT code/weights, a smaller model and CPU inference. It explicitly warns about overlap with pretrained training data. [Full paper](https://arxiv.org/html/2407.21658v1)

LightSem's audited constant-grid fit matches mapper BPM exactly on only about 62–65% of the checked material; octave and 2:3 alternatives matter. Benchmark a replacement against independently verified beat/downbeat timing and continuity. A higher Light-BAS score alone cannot establish that the grid improved. Mapper BPM is also a reference requiring validation, not unquestionable musical ground truth. [Local audit](../../CONTINUATION_AUDIT_2026-09-07.md)

## Measure several kinds of correspondence

**Cross-Modal Metrics — Kohl, Wursthorn and Weiß, ACM Multimedia 2025.** The publisher abstract distinguishes beat/change concurrence, loudness/brightness correlation and musical/lighting structural correspondence using control parameters. It reports distinguishing real shows from random sequences and examining four styles. Full formulas, implementation and public data were not verified because the full paper was inaccessible. [Publisher](https://doi.org/10.1145/3746027.3755488)

The project-specific implication is to keep a diagnostic vector: timing, simultaneous fixture activation, spatial motion, colour occupancy/change, section contrast and recurrence. These proposed diagnostics still need validation against known failure cases and human responses. No new composite aesthetic score is justified.

A useful negative-control set includes a constant scene, synchronized whole-rig flashing, a travelling wave, time-shifted events and section-shuffled motifs. A metric that ranks every synchronized flash above a preferred wave should expose its limitation instead of becoming the sole optimization target.

## Raveform is already a practical asset

**Kim et al., TISMIR 2026.** [Raveform](https://raveform.netlify.app/) includes 1,423 tracks with expert tempo, beat/downbeat and functional-structure annotations, with an eight-fold evaluation schema. Its collection supplies track links and annotations; it does not establish redistribution rights for the music or provide lighting controls. Dataset-specific licensing remains an asset-check gap. [Publication](https://doi.org/10.5334/tismir.288)

The local annotation bundle already exists under `/mnt/lightsem-data/raveform/raveform`. Check actual audio availability and canonical identity before using a frozen held-out subset. This can support structure and beat evaluation without new lighting labels.

## What additional Just Dance processing changes

More processed videos may improve coverage and target reliability. They do not automatically fix a collapsed target representation or validate a conductor. After extraction and downstream feature joins finish, freeze the manifest, deduplicate song identities, measure usable duration and target quality, and assess which descriptor gaps the added material addresses. Training on the new corpus belongs to a separate experiment.

<!-- PAGEBREAK -->
# Six experiments that can change the decision
**All budgets and thresholds below are proposed design choices. They have not been registered or run.**

## 1. Repair the comparison materials and validate diagnostics

**Question:** can a fresh pair differ in its intended treatment without unintended brightness or rendering changes?

Correct the blinder metric/renderer mismatch in a versioned metric. Verify it with analytically known dimmer sweeps. Reprocess saved videos on CPU, preserve historical scores, and measure new candidate renders with a fixed camera, exposure and image region.

For non-brightness experiments, choose a prospective image-luma tolerance, initially 3%, and also inspect clipping and distributions. This is an engineering starting point, not a perceptual equality standard. A brightness treatment must remain a separate study; normalizing it away would erase the intervention.

**Gate:** pass controlled renderer checks and material matching before new aesthetic comparisons. Probe the proposed timing/spatial/colour metrics with the negative controls on page 6. Metric validity is distinct from whether any candidate looks better. Existing-video analysis needs no GPU; newly constructed controls require rendering.

## 2. Benchmark beat/downbeat analysis before changing it

**Question:** does Beat This improve the actual timing reference used by the conductor?

Freeze 20 development and 40 evaluation songs with independent beat/downbeat annotations and verified identities, using locally available Raveform material where suitable. Audit pretrained overlap first. Compare the current grid fitter, frozen Beat This outputs, and the existing analysis baseline at the same offline context budget. Tune only on development songs.

**Primary:** song-macro beat F1; separately report downbeat F1, continuity, octave/2:3 errors and runtime. As a provisional advancement threshold, seek at least +5 percentage points beat F1 with an interval above zero, and no material continuity/downbeat deterioration under predeclared margins. Forty songs may leave small effects unresolved.

**Interpretation:** an improved timing benchmark earns a conductor ablation, not an aesthetic improvement claim. CPU inference is supported; local throughput has not been measured.

<!-- PAGEBREAK -->
# Experiments · recurrence and preferences

## 3. Isolate recurrence memory from ordinary variation

**Question:** does recalling a motif when musical material returns improve whole-song design?

Freeze 12 development and 20 evaluation songs. Compare current composition, a varied section controller without recurrence memory, and the same controller with recurrence-linked motifs. Match palette inventory, change count and average image luma across the latter two arms. Use identical descriptors, timing, rig and seeds.

Start with manually checked section identities to test the representation; then replace them with automatic identities. Show enough of both section occurrences to judge a return. Arbitrary isolated 15-second clips cannot test this hypothesis.

**Primary:** held-out designed-appearance preference; secondary music fit, perceived repetition and recurrence consistency. Use 40 locked comparisons, two contrasts per song, with song-level uncertainty. A provisional 65% win rate and an interval above chance justify further confirmation; a point estimate alone does not.

**Null:** recurrence adds no benefit over equally varied composition. This inexpensive representation test should precede a new full-song neural planner.

## 4. Compare active preference acquisition with random sampling

**Question:** at an equal number of pair queries, does EUBO choose a better setting than random pairs? Viewing time is a separate measured cost; this design does not equalize human effort.

Freeze three or four observable wave/colour controls and brightness. Give EUBO and Sobol/random acquisition 20 newly rated training pairs each, with the same development-song schedule, candidate bounds and rendering budget. Use a declared low-dimensional utility model; do not silently add a large song embedding to a 40-pair study.

Freeze the two recommended settings. On 20 unused songs, compare the active recommendation with both the current default and the random-acquisition recommendation: 40 locked pairs. Add eight side-swapped repeat or identical-render checks. **Total: 88 pair presentations**, with response counts and viewing time recorded separately.

**Primary:** designed-appearance win rate by song, with music fit as a separate gate. Target 65% wins and a confidence bound above chance for each prespecified comparison before claiming success; account for testing both contrasts. This small pilot detects only large gains.

Record “no meaningful preference” and “cannot judge” separately. For a simple binary model, ties can initially be abstentions; report their frequency and repeat stability. A later tie-likelihood study must compare calibrated alternatives. Fitting/acquisition can use CPU; new videos and human decisions are still required.

<!-- PAGEBREAK -->
# Experiments · scene representation and authored looks

## 5. Separate scene representation from fixture assignment

**Question:** does the six-moment target limit scene reconstruction, and does fixture assignment limit temporal consistency? These oracle contrasts cannot explain training collapse by themselves.

First run an oracle rendering check with real target histograms. Compare six-moment reconstruction versus H/V distribution preservation, crossed with independent-frame versus temporally constrained fixture assignment. Freeze the scene source, fixture geometry and intensity scale.

**Primary diagnostics:** distribution reconstruction, colour occupancy, frame-to-frame fixture changes and persistence. Use held-out human comparisons to assess whether any reconstruction gain survives rendering. Define reconstruction and continuity tolerances on development data; do not choose them after inspecting evaluation results.

Only if the oracle check passes should a small audio-conditioned experiment compare the incumbent with a histogram-based target, under identical data splits, capacity budget and rollout evaluation. Reserve the GPU until extraction finishes. A faithful Skip-BART baseline is a separate implementation task; do not label this simplified ablation a SeqLight reproduction.

**Null:** a richer target changes statistics without improving visible design. In that case, stop before the costly RL/decomposition stack.

## 6. Test an authored-look ceiling before collecting a corpus

**Question:** can the current renderer produce the kind of lighting Anton wants when timing and motif choices are supplied well?

Build a compact, original bank of looks with documented palette, movement, spatial spread and transition controls. On development songs, supply a manually reviewed cue timeline; on 20 held-out songs compare a similarly specified oracle plan with the current automatic plan. Keep renderer capability and timing sources explicit.

**Interpretation:** if an independently reviewed reference plan is faithfully rendered yet judged poor, inspect renderer primitives. Otherwise, an unsuccessful manual plan is inconclusive. If the oracle wins, the planner/selection gap becomes concrete. This does not establish that audio can learn the oracle policy.

A cue-retrieval baseline can follow using separately authored development examples. The AuraLuxMuse source is a representation reference; its generated XML is not a substitute for an authored corpus. This experiment requires new authoring and rating work, even if it requires no neural training.

<!-- PAGEBREAK -->
# Recommended continuation and limits

## Order of work

**First, make the next comparison trustworthy.** Version the brightness diagnostic, verify canonical song identities, and freeze a clean evaluation manifest. Keep the original 39 ratings and videos as historical pilot evidence.

**Then test the two nearest product changes.** Run the recurrence-memory experiment and a small active preference study as separate interventions. Freeze descriptors, beat analysis and renderer within each comparison.

**Run the timing benchmark alongside CPU work.** A clean timing win can later enter its own rendering ablation. Newly extracted Just Dance material should enter training only after its manifest and downstream features are complete and audited.

**Escalate conditionally.** A successful renderer oracle earns planner work. A successful histogram oracle earns a controlled decoder study. Neither success automatically justifies implementing an entire diffusion or RL pipeline.

## What this search changed

The strongest additions to the existing vault are the human-in-the-loop optimization methods, editable light-object design, formal recurrence representations, Beat This and the control-parameter metric paper. Skip-BART, SeqLight, SongFormer and EDMFormer were already present; the contribution here is correcting their relationship to the local code and checking availability.

Fourteen new paper records were selected for the project vault. LightingGen and AuraLuxMuse remain documented leads with incomplete evidence. This review does not establish a new openly usable authored DMX corpus.

## Evidence limits and source reconciliation

Perplexity was used successfully for discovery, followed by primary papers, official repositories/model cards and local code. Its summary overstated SeqLight weight availability and blurred hierarchical generation with guaranteed motif reuse. Those claims were corrected. It also resurfaced Witek's groove paper, already in the vault; that was not counted as a new discovery.

Direct studies use few songs, camera-derived targets, narrow genres or ambient environments. Preference optimization has not established transfer to dynamic lighting. Several works are preprints. “Not located” describes this search, not proof that an asset does not exist.

Local metrics come from the continuation and ratings audits. Existing validation material has identity/split caveats, so old results should not be promoted into new locked-test evidence. The ratings cover one rater and older controls. More observations, more questions and repeated views do not automatically create more independent songs.

Searches covered direct lighting/DMX, musical form, active preferences, beat tracking and EDM annotations. Follow-ups resolved availability contradictions. Discovery stopped when mechanisms had primary support or bounded evidence gaps and further searching was unlikely to change the experiment order.

**Reading priority:** Glow with the Flow → Sequential Line Search / qEUBO → MusicFrameworks → SeqLight's representation and decomposition sections → Beat This. Use the remaining papers to refine a successful pilot rather than expanding the first implementation.

<!-- PAGEBREAK -->
# Reference notes · lighting and musical structure

**Skip-BART.** Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang. *Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task?* 2025, revised December 2025; ICLR 2026 designation in official repository. [Primary paper](https://arxiv.org/html/2506.01482v2) · [Code](https://github.com/RS2002/Skip-BART). Full methods and artifact cards checked.

**SeqLight.** Zijian Zhao, Dian Jin, Zijing Zhou, Xiaoyu Zhang. *Stage Light is Sequence²: Multi-Light Control via Imitation Learning.* May 2026 preprint. [Primary paper](https://arxiv.org/html/2605.03660v1) · [Code](https://github.com/RS2002/SeqLight). Full methods checked; trained weights/license not verified.

**Glow with the Flow.** Frederic Anthony Robinson, Vishnu Raj, David Cooper, Fan Du, David Gunawan. *Glow with the Flow: AI-Assisted Creation of Ambient Lightscapes for Music Videos.* PacificVis Notes 2026. [Primary paper](https://arxiv.org/html/2602.08838v1). Full methods/evaluation checked; no code/data located.

**Cross-Modal Metrics.** Michael Kohl, Tobias Wursthorn, Christof Weiß. *Cross-Modal Metrics for Capturing Correspondences Between Music Audio and Stage Lighting Signals.* ACM Multimedia 2025, 528–534. [Publisher](https://doi.org/10.1145/3746027.3755488). Abstract verified; full PDF blocked.

**LightingGen.** Tianhao Wang, Yujian Jiang, Wei Jiang, Xiangzhong Zhou, Xuzeng Guan. *LightingGen: A DMX Based Generation Method for Entertainment Stage Lighting.* IEEE TMM 28, 5216–5226, 2026. [Publisher](https://doi.org/10.1109/TMM.2026.3664978). Bibliographic metadata verified; methods/assets unresolved.

**AuraLuxMuse / LightMuse.** Joey Deng, repository owner `joeyjoestar111`. [Source repository](https://github.com/joeyjoestar111/AuraLuxMuse), inspected September 2026. Code observations only; no verified primary paper or authored corpus.

**MusicFrameworks.** Shuqi Dai, Zeyu Jin, Celso Gomes, Roger B. Dannenberg. *Controllable deep melody generation via hierarchical music structure representation.* ISMIR 2021. [Author PDF](https://www.cs.cmu.edu/~rbd/papers/frameworks-ismir2021.pdf). Full primary paper checked.

**Whole-Song Hierarchical Generation.** Ziyu Wang, Lejun Min, Gus Xia. *Whole-Song Hierarchical Generation of Symbolic Music Using Cascaded Diffusion Models.* ICLR 2024. [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/39d239a30bb40536a5e4e78b5780ba82-Abstract-Conference.html). Full paper checked; linked code/checkpoints not independently audited.

**Formal Prototypes.** Christoph Finkensiep, Matthieu Haeberle, Friedrich Eisenbrand, Markus Neuwirth, Martin Rohrmeier. *Repetition-Structure Inference with Formal Prototypes.* ISMIR 2023. [Paper](https://archives.ismir.net/ismir2023/paper/000045.pdf) · [Code](https://github.com/DCMLab/form-repetition-ismir23). Full methods and specialized implementation requirements checked.

**SongFormer.** Chunbo Hao, Ruibin Yuan, Jixun Yao, Qixin Deng, Xinyi Bai, Yanbo Wang, Wei Xue, Lei Xie. *SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision.* 2025; revised April 2026. [Paper](https://arxiv.org/html/2510.02797v3) · [Code](https://github.com/ASLP-lab/SongFormer). Full paper and repository checked.

**EDMFormer.** Sahal Sajeer, Krish Patel, Oscar Chung, Joel Song Bae. *EDMFormer: Genre-Specific Self-Supervised Learning for Music Structure Segmentation.* March 2026 preprint. [Paper](https://arxiv.org/html/2603.08759v1). Full paper checked; downloadable assets not located.

<!-- PAGEBREAK -->
# Reference notes · preferences, timing and local evidence

**Sequential Line Search.** Yuki Koyama, Issei Sato, Daisuke Sakamoto, Takeo Igarashi. *Sequential Line Search for Efficient Visual Design Optimization by Crowds.* SIGGRAPH 2017. [Author project](https://koyama.xyz/project/sequential_line_search/) · [Code](https://github.com/yuki-koyama/sequential-line-search). Primary method and MIT implementation checked.

**qEUBO.** Raul Astudillo, Zhiyuan Jerry Lin, Eytan Bakshy, Peter I. Frazier. *qEUBO: A Decision-Theoretic Acquisition Function for Preferential Bayesian Optimization.* AISTATS 2023, PMLR 206. [Proceedings](https://proceedings.mlr.press/v206/astudillo23a.html) · [Official tutorial](https://botorch.org/docs/tutorials/preference_bo). Abstract and implementation checked; full PDF exceeded browser size limit.

**BOPE.** Zhiyuan Jerry Lin, Raul Astudillo, Peter I. Frazier, Eytan Bakshy. *Preference Exploration for Efficient Bayesian Optimization with Multiple Outcomes.* AISTATS 2022. [Paper](https://arxiv.org/abs/2203.11382) · [Official tutorial](https://botorch.org/docs/tutorials/bope). Primary method and release status checked.

**Sequential Gallery.** Yuki Koyama, Issei Sato, Masataka Goto. *Sequential Gallery for Interactive Visual Design Optimization.* SIGGRAPH / TOG 2020. [Author PDF](https://koyama.xyz/project/sequential_gallery/download/author-version.pdf) · [Code](https://github.com/yuki-koyama/sequential-gallery). Full paper and MIT release checked.

**Constrained PBO.** Koki Iwai, Yusuke Kumagae, Yuki Koyama, Masahiro Hamasaki, Masataka Goto. *Constrained Preferential Bayesian Optimization and Its Application in Banner Ad Design.* IJCAI 2025. [Proceedings PDF](https://www.ijcai.org/proceedings/2025/0463.pdf). Full paper checked; code not verified.

**Consecutive PBO.** Aras Erarslan, Carlos Sevilla Salcedo, Ville Tanskanen, Anni Nisov, Eero Päiväkumpu, Heikki Aisala, Kaisu Honkapää, Arto Klami, Petrus Mikkola. *Consecutive Preferential Bayesian Optimization.* November 2025 preprint. [Full text](https://arxiv.org/html/2511.05163v1). Methods/runtime checked; code archive not verified.

**Preferential Multi-Objective BO.** Raul Astudillo, Kejun Li, Maegan Tucker, Chu Xin Cheng, Aaron D. Ames, Yisong Yue. *Preferential Multi-Objective Bayesian Optimization.* 2024 preprint; TMLR record 2025. [Readable paper](https://arxiv.org/pdf/2406.14699) · [TMLR record](https://openreview.net/forum?id=mjsoESaWDH). Readable version checked; accepted full text blocked.

**Beat This!** Francesco Foscarin, Jan Schlüter, Gerhard Widmer. *Beat This! Accurate Beat Tracking Without DBN Postprocessing.* ISMIR 2024. [Paper](https://arxiv.org/html/2407.21658v1) · [Code/models](https://github.com/CPJKU/beat_this). Full paper and official assets checked.

**Raveform.** Taejun Kim, Jongsoo Kim, Hyungyu Kim, Juhan Nam. *Raveform: A Dataset of Metrical and Functional Structure Annotations for EDM Tracks in DJ Mixes.* TISMIR 9(1), 131–143, 2026. [Publication](https://doi.org/10.5334/tismir.288) · [Official data/schema](https://raveform.netlify.app/). Metadata, schema and local annotation presence checked.

**Local sources.** [Continuation audit, 7 September](../../CONTINUATION_AUDIT_2026-09-07.md); [manual ratings audit, 8 September](../../HL_RATINGS_2026-09-08.md); associated saved analysis protocol, fitted predictions and material audit. Code baseline: commit `c274c23` plus audited working-tree changes. Original ratings SHA-256: `def1baf86ef8d0d924d88d532352994e5902a0f1204d4a90d3ee37e16894501e`.

All online availability statements describe the September 2026 inspection. Local document links require access to this project workspace.
