# Direction 3 — transformer / multimodal methods (Perplexity reasoning pass, 2026-09-04)

Project brief attached: `lightsem_brief.md` (copied below the answer). Answer verbatim.

A contrastive critic on this stack will mostly rediscover **beat+loudness**; that is already your strongest coupling and already your failure mode (model Light-BAS +0.135 vs coach +0.087 vs beat-trigger +0.092).  Steal Lodge-style two-level planning onto the conductor, not an AR light LM that regresses HSV.[1]

## Principles that bind the design

MAS.S60 / FactorCL: InfoNCE on paired $$(A,L)$$ maximises $$I(A;L)$$, i.e. redundant information. Unique LD choices and synergistic looks (blackout-then-strobe only if drop **and** vocal gap) are invisible to a pairwise critic.  Additive fusion $$C=f_A(A)+f_L(L)$$ cannot express that XOR; tensor/gated/cross-attention can, but only if you prove it.  Before believing any fusion critic, fit its nearest additive model (unimodal scores + late sum) and demand a ranking change; if BoN order is unchanged, you learned RMS and onsets.  Cross-modal transfer from dance→light via audio moves only what dance and light share with audio. Your ceiling is already measured: inter-coach $$r$$ 0.22–0.42, block A $$R^2=-0.22$$, Oslo energy $$r=+0.335$$.  Segment before aligning: contrast **bars** (or 2 s windows), never 40 fps HSV frames, or persistence and look-ahead leak will fake alignment the way H-D leaked under a symmetric window.[1][2][3][4]

Implication: $$C(A,L)$$ is a **filter** for shared timing/energy/look-change rate. It is not a lighting designer, and it must not be allowed to become the training objective until it loses to metronomic and loudness hacks.

## Critic: BoN first, RL later

Train $$C$$ as a dual encoder on **bar-synchronous** tokens: audio = your 92-ch bus (optionally frozen MERT later); light = HSV histogram + look-change flag + sparsity/strobe density, **not** raw dimmer. Positives: real RPMC-L2 pairs, Just Dance background visuals (event AP 3.17× vs 1.48× shift — your best audio–light coupling), any console logs.  Negatives must be three families, not two: (i) circular time-shift ≥ 1–2 bars; (ii) other-song light on this audio; (iii) **beat-trigger / metronomic light on this audio**. Without (iii), InfoNCE’s cheapest solution is a beat detector, and you will reward-hack into the mode you already occupy.[1][5]

Do not use a single scalar. Keep a factorized head: $$C_R$$ (redundant: beat, RMS) and $$C_U$$ (unique: look-change timing, palette jump, section contrast). At inference, rank with $$C_U - \lambda C_R$$ or require $$C_U$$ to beat the additive projection. That is FactorCL applied to a reranker, not a slogan.[2]

**(a) Best-of-N.** Sample $$N\in\{8,16\}$$ from the rectified-flow head only (block C). Run the **conductor**, not the EGL renderer, to get light features. Pick $$\arg\max_i C_U(A, L_i)$$. Cost is N forward passes of a tiny critic plus N cheap rule-table evals. Report vs random-of-N and vs the current single sample, both causal and offline. If all N samples sit in the metronomic mode, BoN cannot save you — diversity of 7.4 vs real 6.3 does not mean the critic’s preferred axis is populated.[1]

**(b) Reward.** Only after BoN beats random-of-N on a human pair test. Then DPO/pairwise on the **conductor knobs** first (10–20 scalars), not the flow network. Flow-head RL on $$C$$ will exploit $$C_R$$: more on-beat onsets, more dimmer-on-RMS. Bailando’s actor-critic used a beat-align reward on purpose; you would be copying their objective onto a product that is already too metronomic.  Preference fine-tune of flow is the last step, with a frozen critic and a penalty for BAS above the coach’s +0.087.[6]

Failure modes to instrument **before** any reward loop: (1) critic accuracy real-vs-shift high, real-vs-beat-trigger at chance → loudness/beat hack; (2) $$C$$ correlates with RMS more than with look-change flags → same; (3) additive projection matches $$C$$ rankings → no synergy; (4) Just Dance visuals are a rhythm-game show, not a concert LD — RPMC-L2 and JD must be reported separately or you will overfit game palettes.

## What to steal, what not to replace

| Borrow | From | Use here |
| --- | --- | --- |
| Discrete look codebook + GPT over units | Bailando choreographic memory [6] | Tokenise **looks** (palette, fixture role, strobe density), not DMX |
| Beat critic as a **constraint**, not the objective | Bailando actor-critic [6] | Cap BAS; do not maximise it |
| CFG + section inpainting | EDGE [7][8] | Edit a drop/breakdown in the pre-rendered pass |
| Coarse primitives → local diffusion | Lodge / Lodge++ [9][10] | Section plan **onto conductor knobs**, then existing flow |
| Causal latent AR | MotionStreamer [11] | Live $$L=0$$ only |
| Scale / in-the-wild music | InfiniteDance 100 h [12] | Motion pretrain signal, not inference dancer |
| Skip connection = explicit frame align | Skip-BART [5] | If you ever AR-decode light, inject $$x_{t}$$ into $$y_t$$; do not hope attention learns 1:1 |
| Global HSV then allocate fixtures | SeqLight [13] | That is direction 4, not a critic |

AR masked-token “light LM” pretrained on logs **does not beat** regress+hazard+flow as a replacement. Skip-BART autoregresses absolute hue/value on 699 RPMC-L2 clips; you already ruled out regressing absolute light level; persistence won; they needed skip connections because the decoder would not lock rhythm; training was 2×4090+A100, not 12 GB.  Discrete tokens **are** right for look-changes — that is what your operator head already does, and it is the strongest audio–light number you have (events AP lift +0.31). Keep the three-head for motion statistics with mixed support (rates, hazards, stochastic style). Add a small look-token model as a **planner**, not a decoder to DMX.[1][5]

Diffusion forcing (independent per-token noise) is useful for a pre-rendered show: denoise section-plan tokens at low noise and frame-level flow at high noise, or infill a breakdown without regenerating the verse.  It is not a new product head. The structure that matches “whole track known, H-C refuted, conductor is the bottleneck” is Lodge’s two-level: section planner **writes conductor knobs** (scene, palette, intensity, strobe density), frame model stays as is. Feeding section channels into the per-frame trunk already hurt BAS (−0.028) with a shuffled-donor control that explained most of the FID drop.  Do not repeat H-C. Test the planner.[1][14]

## Frozen encoders and few hundred prefs

At ~30 h paired and one 12 GB GPU, a frozen MERT / MuQ / CLAP encoder is only worth an **offline dump** to the second disk, concatenated as extra channels. You cannot fine-tune 330 M. Frozen music SSL transfers redundant MIR (beat, timbre, coarse structure), not LD intent.  H-C already showed extra structure channels can hurt. The ablation you named is the right one: same width, random projection of log-mel (or of the 92-ch bus), **plus** the 92-ch bus alone. If MERT ≉ random projection on operator-head event AP, Oslo $$r$$, and human “fits the music”, keep the dump. If it does, you paid for width. Extract once; never put Jukebox-class models in the training loop (EDGE needed frozen Jukebox 5B; you do not have that GPU).[7][15]

Conductor from pairwise prefs: Bradley–Terry on knobs, $$P(i\succ j)=\sigma(w^\top(\phi_i-\phi_j))$$, $$\phi$$ = the rule-table knobs (or conductor-v1 outputs), regularised toward current defaults. Two questions dissociate in your n=10 test (“looks like an LD” vs “fits the music”); fit them separately.  Detectable effect sizes: a large gap like 8–1 vs beat-trigger is visible at ~30 decided pairs; a 60% true win rate needs on the order of **200** decided pairs to separate from 50%; a 55% gap needs ~1000.  For $$K\approx 8$$–12 linear knobs, 200–400 pairwise ratings can recover main effects; interactions and a learned conductor v1 need 500+ and will still be a pilot. Do not spend those ratings on DPO of the flow head.[1][16]

## Four model-side experiments

1. **Bar-level factorized critic as BoN-8 over flow samples.** Falsify: critic distinguishes real vs time-shift but not real vs beat-trigger, **or** additive projection preserves BoN ranking, **or** BoN-8 does not beat random-of-8 on held-out human “fits the music.” Must beat: random pick among the same N samples, and the 30-line beat-trigger.

2. **Section planner → conductor knobs (Lodge), audio-only, no per-frame section channels.** Falsify: paired Light-BAS worse than current (H-C style regression), **or** shuffled-section-plan control matches the real plan on FID/human LD ratings. Must beat: current hand-written conductor, and the already-refuted H-C concat.[1][9]

3. **Frozen MERT/MuQ vs same-width random projection vs 92-ch bus, operator-head event AP + Oslo $$r$$.** Falsify: pretrained encoder ≰ random projection on those two metrics (song-level split). Must beat: 92-ch bus and the random-projection encoder.

4. **Bradley–Terry linear fit of ≤12 conductor knobs from pairwise prefs, two questions separate.** Falsify: held-out preference acc $$\le 60\%$$ at $$N=200$$, **or** learned $$w$$ increases BAS toward metronomic, **or** new human batch does not prefer fitted knobs over hand-written defaults. Must beat: default rule table, random knobs, beat-trigger.

Do not run AR light-LM replacement or flow-head RL until (1) and (2) have failed or succeeded. Those two are where the product actually lives.

## Sources

[1] lightsem_brief.md | https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/16322705/7dcfd527-1462-49f7-b179-3c37730369af/lightsem_brief.md?AWSAccessKeyId=ASIA2F3E
[2] Factorized Contrastive Learning:Going Beyond Multi-view ... - arXiv | https://arxiv.org/html/2306.05268v2
[3] Lec5 - Fusion | PDF | Errors And Residuals | Regression Analysis | https://www.scribd.com/document/913449311/lec5-fusion
[4] The More, the Merrier: Contrastive Fusion for Higher-Order Multimodal Alignment | https://arxiv.org/abs/2511.21331
[5] [Paper Note] Automatic Stage Lighting Control: Is it a Rule ... | https://en.papernotes.org/ICLR2026/audio_speech/automatic_stage_lighting_control_is_it_a_rule-driven_process_or_generative_task/
[6] Paper page - Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory | https://huggingface.co/papers/2203.13055
[7] ZeyuLing/Motius-EDGE-AISTPP · Hugging Face | https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP
[8] Edge Dance - LogicWeb | https://www.logicweb.com/ai/edge-dance/
[9] Lodge++: High-Quality and Long Dance Generation With ... | https://pubmed.ncbi.nlm.nih.gov/41052184/
[10] Drop the Beat Representation for Dance Generation with Mamba ... | https://arxiv.org/html/2603.08023v1
[11] MotionStreamer: Streaming Motion Generation via Diffusion- ... | https://dblp.uni-trier.de/rec/journals/corr/abs-2503-15451.html
[12] InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization | https://www.alphaxiv.org/abs/2603.13375
[13] Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning | https://arxiv.org/abs/2605.03660v1
[14] Diffusion Forcing - Harold Benoit | https://haroldbenoit.com/notes/ml/diffusion-models/frameworks---theory/diffusion-forcing
[15] MuQ-Eval: An Open-Source Per-Sample Quality Metric for AI Music Generation Evaluation | https://arxiv.org/abs/2603.22677
[16] Pairwise Win Rate Calculator with Elo and Bradley-Terry ... | https://latenteval.ai/tools/eval/pairwise-win-rate-calculator
[17] A systematic review of molecular representation learning foundation ... | https://pmc.ncbi.nlm.nih.gov/articles/PMC12784970/
[18] Multi-modal contrastive drug synergy prediction model ... | https://pmc.ncbi.nlm.nih.gov/articles/PMC12465304/
[19] https://nucleus.iaea.org/sites/naipc/ididas/Lists/... | https://nucleus.iaea.org/sites/naipc/ididas/Lists/References/DispForm.aspx?ID=4694
[20] Multisensory Neuromorphic Devices: From Physics to Integration | https://pmc.ncbi.nlm.nih.gov/articles/PMC12791111/
[21] A Systematic Review of Contrastive Learning in Medical AI - PMC | https://pmc.ncbi.nlm.nih.gov/articles/PMC12938576/
[22] Orthogonalized Multimodal Contrastive Learning with ... | https://arxiv.org/html/2602.14983v1
[23] Strategic Multimodal Representation Alignment for Optimal ... - arXiv | https://arxiv.org/html/2511.12121v4
[24] huuuuuuuuu/InfiniteDance · Hugging Face | https://huggingface.co/huuuuuuuuu/InfiniteDance
[25] AI Dance Generator from Music, Built by K-pop Dancers | mvnt | https://mvnt.studio/
[26] Build software better, together | https://ithub.global.ssl.fastly.net/topics/light-control
[27] ICLR 2025: Exploring Alignment in Multimodal Contrastive ... | https://www.studocu.vn/vn/document/hanoi-university-of-science/tin-dai-cuong/iclr-2025-exploring-alignment-in-multimodal-contrastive-learning/151692726
[28] Diffusion Forcing Planner: History-Annealed Planning with Time ... | https://arxiv.org/abs/2606.11019
[29] Partial Information Decomposition via Normalizing Flows in Latent ... | https://arxiv.org/html/2510.04417v1
[30] C05 Modeling: Multimodal AI | https://computing.mit.edu/cross-cutting/common-ground-for-computing-education/common-ground-subjects/c05-modeling-multimodal-ai/
[31] SynIB: Informational Bottleneck for Maximizing Synergy in Multimodal Learning | https://arxiv.org/abs/2606.09853
[32] Plan, Verify and Fill: A Structured Parallel Decoding Approach for Diffusion Language Models | https://arxiv.org/html/2601.12247v2
[33] Information-Theoretic Decomposition for Multimodal Interaction Learning | https://arxiv.org/abs/2606.11614v1
[34] TuneJury/tunejury · Hugging Face | https://huggingface.co/TuneJury/tunejury
[35] zhudi2825/MuQ-Eval-A1 | https://huggingface.co/zhudi2825/MuQ-Eval-A1
[36] lec4 - multimodal | PDF | Information | Cognitive Science | https://www.scribd.com/document/913451870/lec4-multimodal
[37] Improving Diffusion Language Model Reasoning via ... | https://www.alphaxiv.org/abs/2603.13243
[38] Diffusion Forcing: Next-token Prediction Meets Full- ... | https://www.swiftscholar.net/paper/6942b0afebe97c515a7cd787
[39] Explainable multimodal feature fusion networks for Parkinson's ... | https://pmc.ncbi.nlm.nih.gov/articles/PMC12982457/
[40] Fusion-driven multimodal learning for biomedical time series ... | https://pmc.ncbi.nlm.nih.gov/articles/PMC12484004/
[41] Multimodal data fusion in neuroscience: promises, challenges ... | https://pmc.ncbi.nlm.nih.gov/articles/PMC13061299/
[42] Hidden state-based fusion network for multimodal sentiment analysis | https://academic.oup.com/comjnl/advance-article/doi/10.1093/comjnl/bxag062/8713340
[43] Benchmarking Multimodal Deep Fusion Strategies for ... - PMC | https://pmc.ncbi.nlm.nih.gov/articles/PMC13114584/
[44] a systematic review of architectures, multimodal fusion and critical ... | https://link.springer.com/article/10.1186/s40708-025-00286-7
[45] Multimodal fusion strategies for survival prediction in breast cancer | https://pmc.ncbi.nlm.nih.gov/articles/PMC12595345/
[46] IndiSeek learns information-guided disentangled representations | https://arxiv.org/html/2509.21584v2
[47] Multimodal fusion of complementary material representations for ... | https://www.cell.com/cell-reports-physical-science/fulltext/S2666-3864(26)00424-8
[48] A Unified Information Bottleneck Framework for Multimodal Biomedical Machine Learning | https://www.mdpi.com/1099-4300/28/4/445
[49] Artificial Intelligence-Driven Multimodal Sensor Fusion for Complex Market Systems via Federated Transformer-B | https://www.mdpi.com/1424-8220/26/8/2418
[50] 3DWaFusion: Three-Dimensional Multiscale Wavelet Convolutional Neural Network for Multimodal Medical Image Fus | https://www.mdpi.com/1424-8220/26/12/3784
[51] OpenEnvision/Awesome-Multimodal-Modeling - GitHub | https://github.com/OpenEnvision/Awesome-Multimodal-Modeling
[52] A Comprehensive Survey of Fusion Models in Machine Learning ... | https://medium.com/@nirvana.elahi/a-comprehensive-survey-of-fusion-models-in-machine-learning-architectures-mechanisms-and-bffdbaf260e0
[53] Multimodal fusion and explainability of artificial intelligence ... | https://pmc.ncbi.nlm.nih.gov/articles/PMC12876535/
[54] Multi-agent task and motion planning trends analysis: a survey | https://link.springer.com/article/10.1007/s10462-026-11588-5?error=cookies_not_supported&amp;code=c9e8dd47-593d-4151-9442-4cf8544d7358
[55] CustomDance: Customized 3D Dance Generation with Coarse-to ... | https://arxiv.org/html/2608.06722v2
[56] Robust Multimodal Learning With Heterogeneous and ... | https://escholarship.org/uc/item/5kj741gh
[57] Rethinking Multimodal Fusion for Time Series: Auxiliary Modalities Need Constrained Fusion | https://arxiv.org/abs/2603.22372
[58] zirobtc/motion-2 · Hugging Face | https://huggingface.co/zirobtc/motion-2
[59] ARDY: Autoregressive Diffusion with Hybrid Representation for ... | https://www.alphaxiv.org/abs/2607.08741
[60] Hierarchical Planning with Latent World Models - alphaXiv | https://www.alphaxiv.org/overview/2604.03208
[61] [PDF] Minimax Hypothesis Testing for the Bradley-Terry-Luce Model - arXiv | https://arxiv.org/pdf/2410.08360.pdf
[62] Change Point Detection in Pairwise Comparison Data with Covariates | https://www.tandfonline.com/doi/full/10.1080/10618600.2025.2580478
[63] Computational Approach for Automated Generation of ... | https://www.syncform.studio/research.html/
[64] Bradley-Terry Pairwise Ranking Calculator - MetricGate | https://metricgate.com/docs/bradley-terry-pairwise-ranking/
[65] Bradley-Terry Model: Converting Preferences to Rankings - Interactive | https://mbrenndoerfer.com/writing/bradley-terry-model-pairwise-preferences-rankings
[66] Bradley-Terry Model - Wayne's Talk | https://waynestalk.com/en/bradley-terry-model-en/
[67] Team Strength Model Calculator - MetricGate | https://metricgate.com/docs/team-strength-model/
[68] Bradley-Terry Model | yale-nlp/SciArena | DeepWiki | https://deepwiki.com/yale-nlp/SciArena/4.2.1-bradley-terry-model
[69] Bradley–Terry model | https://handwiki.org/wiki/Bradley%E2%80%93Terry_model
[70] BradleyTerry Class | lmarena/arena-rank | DeepWiki | https://deepwiki.com/lmarena/arena-rank/5.1.1-bradleyterry-class
[71] Li Siyao | https://www.alphaxiv.org/@li-siyao
