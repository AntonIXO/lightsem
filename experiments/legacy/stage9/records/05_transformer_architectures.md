# Transformer architectures across the related fields (Perplexity reasoning pass, 2026-09-04, late)

Project brief attached plus an addendum describing the Stage 9 plan and the Beat Saber schema findings. Answer verbatim.

arXiv abstracts pulled and fed to the pass: 2311.13687 (beat-aligned spectrogram-to-sequence chart
generation), 2512.18804 (TempoMoE: tempo/beat experts), 2506.07199 (approximately equivariant flow matching for
symmetric synthesizer parameter spaces), 2603.22316 (ST-GDance++ group choreography), 2605.01197 (MG-Former
conducting gestures), 2511.08040 (MEGAMI: permutation-equivariant effect embeddings for automatic mixing),
2607.09581 (Wan-Dancer hierarchical minute-scale), 2604.17656 (Video-Robin autoregressive diffusion planning).

Stay off 40 fps and off AR hue/value. The papers that actually fit LightSem are **beat-aligned chart transformers**, **SongFormer-style section heads**, and **SeqLight’s look-then-assign split** — not Wan-Dancer, not Skip-BART’s LM, not fixture-type MoE.[1][2][3][4][5]

## What actually transfers

Dance papers are mostly the wrong object. They buy long-range structure with models you cannot train: Wan-Dancer is a 14B DiT, global 38-keyframe plan then local 5 s refinement, time-mapped RoPE, rectified flow, 128×A100. Steal only the *split* (slow plan, fast fill), not the stack. Lodge++ is the same split at motion scale: coarse primitives, then parallel local diffusion. CustomDance’s MLLM anchors and ST-GDance++’s group diffusion are the same idea at a budget you do not have.[6][7]

What *does* transfer at 30–50 h / 12 GB:

| Field | Design that carries the result | Transfer to LightSem |
|---|---|---|
| Dance (small data) | TokenDance: FSQ tokens, BiMamba local–global–local, CE, 240-frame crops, d=512. Music split into acoustic vs semantic codebooks. Transformer ablation is worse (FID_k 61 vs 47). [8] | Discrete *effect ids* + beat-rate envelopes. Do not FSQ DMX. Do not replace the frozen 9M Conformer. |
| Dance rhythm | TempoMoE: experts indexed by tempo bins + multi-scale beat experts, music-driven routing, no genre labels. MambaDance: Gaussian beat cue inside a Mamba diffusion. [9][10] | Legitimate **rate/phase** routing for the decoder. Not a fixture-type MoE in the trunk. |
| Conducting | MG-Former: audio encoder + AR pose decoder, reconstruction + retrieval alignment. Long-range structure + beat sync, not per-frame kinematics. [11] | Closest dance analog to a lighting conductor. Alignment loss belongs on the **critic**, not as extra bus channels (H-C already killed that). [1] |
| Charts | Beat-aligned spectrogram-to-sequence: hop = 1/48 beat, time token + action token, 3-layer Transformer d=256. Unaligned training collapses on 12th/24th/32nd notes. ITGPT is the same family. STRUM is a CRNN/ensemble transcription pipeline — ignore it. [4][12][13] | **This is the Beat Saber grammar.** Beat-relative positions, sparse events, tiny Transformer. |
| Lighting | Skip-BART: BART, OpenL3 @ 10 Hz, **discrete hue/value bins**, skip-concat for frame alignment, AR CE, RSTC with Δhue/Δvalue clamps. SeqLight: same backbone predicts **HSV histograms**, then a separate IL decomposer assigns fixtures. [2][3] | Steal skip-alignment, histogram target, RSTC smoothness. Do **not** steal AR absolute hue/value — you already ruled that out. [1] |
| Symmetry | Hayes: point regression fails under permutation even with invariant losses; need a generative model of equivalent modes + approx-equivariant flow. MEGAMI: permutation-equivariant effect embeddings, not per-track regression. [14][15] | Do not regress per-fixture DMX. Generate a look; let the engine assign. |
| Structure | SongFormer: 4-layer Transformer d=512 @ ~8.3 Hz, SSL fusion, **source embedding**, boundary + function heads. 512 Harmonix songs already beat Gemini on HR.5F. All-In-One: neighborhood attention on demixed stems, joint beat/downbeat/sections. [5][16] | Planner. Audio-only. Writes knobs, not per-frame labels. |
| Motion gen (heavy) | OmniMotion: continuous masked-AR + DiT, AdaLN + cross-attn. OmniMotion-X: AR diffusion transformer. FlowerDance: MeanFlow + BiMamba, few-step. Video-Robin: AR diffusion *planning*. [17][18][19] | Planner hierarchy and cheap few-step sampling only. Not the 12 GB training recipe. |

Conditioning that survives your ablations: **AdaLN/FiLM from slow planner knobs**, plus a skip/concat of *time-aligned* audio features into the decoder (Skip-BART’s actual ablation win). Cross-attention is only worth it from a short planner token stream (tens of section tokens), not from a 40 fps structure bus. H-C already showed frame-wise section channels hurt Light-BAS (−0.028) while a shuffled donor still moved FID — that is leakage, not structure.[1][2]

Objective that survives: **not** AR light LM on hue/value; **not** RL on the flow head before the critic. Histogram / event-box / InfoNCE only.[1]

## Event-box, not 40 fps

Train the decoder as a **beat-grid group-effect model**. Do not train a lighting-chart Transformer as the sole head, and do not stay at 40 fps.

Beat Saber v3 is almost the same vocabulary you wrote down: group id, distribution/phase-spread, colour/brightness/strobe on an exact BPM grid, 2–10k events per 3–4 min song. Chart generation is the paper that solved this object: sparse tokens on a beat grid, tiny encoder–decoder, alignment is load-bearing. Unaligned 4-beat crops fail; aligned ones jump F1 on 12th notes from 9% to 59%.[4]

40 fps is a **renderer** rate. Your own H-B result already said the largest lever is a one-pole smoother, not more temporal resolution. A 4 min track at 40 fps is ~9.6k frames of a continuous field you have no unique target for. The same track on 8th notes at 120 BPM is ~2k steps — the same order as a Beat Saber lightshow, and the same order the 3×256 chart Transformer was trained on.[1][4]

Keep the histogram renderer for RPMC-L2 and Just Dance. Those corpora have no event boxes. Skip-BART/SeqLight already showed the learnable target there is a **frame-level HSV histogram**, not a fixture vector. SeqLight had to invent a second stage because many fixture assignments mix to the same histogram. That is your mixture-of-light problem, not a reason to AR-decode 512 DMX channels.[2][3]

Practical split:

- **Beat Saber (research):** discrete event-box CE (effect id, group, beat-time token, palette/brightness bins, phase-spread, density). Shared type-masked head.
- **RPMC-L2 / Just Dance:** same head’s continuous channels (intensity envelope, palette, rate-in-beats, phase spread, density, look-change hazard) trained through the differentiable HSV renderer.
- **DMX:** deterministic effects engine. Never a network output.

A pure ‘chart Transformer’ will overfit Beat Saber’s 5-lane + Chroma dialect and then fail on a moving-head truss. Vivify is already unparseable; treat that as a warning about asset-specific grammars. ~85% usable maps is enough for an auxiliary CE, not for the product head.

## Symmetry and experts

SeqLight’s hierarchy is the better fit. Hayes is the right *diagnosis*, the wrong *prescription* for this GPU.

Hayes: under permutation symmetry, regression to a point estimate is the wrong statistical object, even with an invariant loss. Equivalent assignments are modes of a distribution; you need a generative model, and if the symmetry is clean, an equivariant flow. MEGAMI says the same thing for mix knobs: permutation-equivariant embeddings, not per-track regression.[14][15]

Your fixtures are **not** Surge-XT oscillators. Washes, movers, strobes, LED bars are typed, the patch changes, and you already committed to random-rig dropout plus a deterministic expander. Building an equivariant CNF over fixture permutations is how you spend the quarter training a model that still cannot transfer to a new club. SeqLight’s move is the one that matches the Stage 9 plan: generate the **look** (histogram / group-effect), then decompose. Copy that split. Do **not** copy their AIRL+GRPO decomposer — you already forbade RL on the generative head before the critic gate, and their IL stack is a second research project.[1][3]

Tempo/beat experts are a legitimate routing axis; fixture-type experts in the trunk are not. Tempo is in the audio, stable across corpora (TempoMoE’s 60–200 BPM claim), and it should change **rate, phase-spread, density**, not which MLP sees the mel. Fixture type is a property of the patch, which you drop out and then expand with an engine. Putting a type MoE inside the frozen Conformer is exactly the thing you ruled out, and it would leak rig geometry into the only shared audio trunk you have. If you want a cheap TempoMoE, put a 4-bin tempo gate **on the decoder head** (slow/medium/fast/double-time), FiLM’d by the planner’s rate knob. That is routing on music, not on hardware.[1][9]

## Mamba, masked-AR, flow

Do not replace the frozen Conformer with Mamba. TokenDance’s Transformer loss is real on ~8 h of **tokenised motion**, not on your 92-ch bus. Your trunk already has depthwise conv (local, beat-scale) plus a 4 s window composing to ~26 s. The Mamba pitch is long 40 fps sequences; if you drop the decoder to beat-grid, the sequence is a few thousand tokens and a 3–4 layer Transformer is the chart-paper result, not a compromise. Bidirectional Mamba is only interesting if you later train a **full-song** decoder at 8th-note rate and the 12 GB Transformer OOMs — measure that before rewriting the trunk.[1][8]

Block C’s rectified flow should not become the group-effects decoder. FlowerDance’s MeanFlow is the only flow paper worth stealing: few-step sampling so best-of-N is cheap. OmniMotion’s continuous masked-AR + DiT is two stacked generative models with AdaLN/cross-attn; that is a motion foundation-model recipe, not a 9M frozen trunk plus a head. If block C still exists as residual motion descriptors, leave the flow, do not RL it, and do not feed it to lights until the critic can rank. For the new decoder, prefer **parallel** heads (hazard + categorical effect id + bounded regression) over any AR hue loop. Skip-BART’s own limitation section admits local overshoot; RSTC is a clamp, not a solution.[1][2][17][19]

Masked-AR vs flow for the new decoder: neither, as the primary. Masked-AR is justified when you have a continuous autoencoder over a motion manifold (OmniMotion). You have a **small discrete effect vocabulary** plus a few bounded continuous operators. That is classification + regression on a beat grid. Use flow/MeanFlow only if the continuous operator vector (rate, phase, density, envelope) is multi-modal **after** conditioning on audio+plan — test residual error vs a Gaussian head first. If NLL of a 2-step MeanFlow does not beat L2+circular palette loss on held-out songs, kill it.

## Recommended stack

Frozen 9M Conformer stays. Trainable mass ~7M. Everything audio-only on offline arms. Song splits. 12 GB, hours.

**Planner.** 4-layer Transformer, d=256, 4 heads, pre-norm, RoPE on bar index, ~3.2M. Token rate: **1 / bar** (bar-pool the existing 92-ch bus + operator-head features; do not run SongFormer’s MuQ+MusicFM fusion on this GPU). Conditioning: concat of bar audio; source embedding if you mix Raveform / JD / Beat Saber density stats (SongFormer’s actual trick for dirty labels). Output: 8 knobs per section — intensity, palette (circular 2D), density, look-change rate, strobe permission, contrast, hold-vs-chase, plus a section-boundary logit. Loss: BCE+TV on boundaries, CE/focal on Raveform roles, L2 on knobs distilled from corpus statistics (HSV energy, event density). **Not** injected as per-frame bus channels. Offline only; quote L=0 as “copy last bar’s knobs”. Training: ~3–6 h, bs=8 songs truncated to 420 s equivalent in bars.[1][5]

Falsify: shuffled-section-plan control (you already specified this). If FID_light and section-contrast do not beat shuffle **and** beat a 3-tercile energy baseline (your own loud/quiet thirds were only 1.5×), the planner is a prior over duration, not music. Must beat: constant knobs; energy-quantile knobs.

**Decoder.** Type-masked head on frozen trunk + frozen operator head. 3 layers, d=256, 4 heads, **beat-relative** RoPE (chart paper: time token on a 1/4-beat or 1/8-beat grid; Wan-Dancer’s time-mapped RoPE idea without the DiT). ~2.5M. Token rate: **4–8 Hz** (16th or 8th), never 40 fps. Conditioning: **AdaLN from planner knobs** (slow) + Skip-BART-style skip of the aligned audio frame into the head (fast). Outputs per fixture-group: intensity envelope, palette, effect id, rate-in-beats, phase spread, density, look-change hazard. Loss: renderer KL/Wasserstein on HSV histograms (RPMC-L2, JD background); CE on Beat Saber event-box fields; BCE on look-change vs operator-head events; random-rig dropout. Optional 4-bin tempo gate on this head only. Inference: sample effect ids with temperature + RSTC-style clamps on Δintensity and Δpalette; engine expands to 40 fps DMX.[1][2][4]

Falsify: (a) histogram loss down, Light-BAS not above beat-trigger +0.092 and not above your current +0.135; (b) Beat Saber CE high, DMX look does not beat the **engine-only** baseline (planner knobs + beat-trigger effects, no learned effect id); (c) offline L=whole-track vs L=0 — if the gap is ~0, as in H-A, stop claiming look-ahead. Must beat: 30-line beat-trigger; persistence; operator-head events driving a fixed chase.

**Critic.** Bar-synchronous dual encoder, 2 layers d=128, ~1.5M. Audio bar tokens vs decoder tokens (not DMX, not pixels). InfoNCE with your three negatives: time-shift, other-song, beat-trigger light. Use as **best-of-8 reranker only**. No reward, no PPO. Training: ~2–4 h.[1]

Falsify: cannot put real above all three negatives on held-out songs (AUROC ~0.5); or N=8 equals N=1 on FID_light **and** on a 10-pair human “looks engineered” test. Must beat: random pick among 8; ranking by RMS energy correlation.

If this stack cannot beat beat-trigger + operator-head on Light-BAS with CIs, the next model is not a bigger Transformer. It is a better **target** (more Beat Saber, real xLights if you ever get it) or an admission that authored light is only recoverable at section scale, and the decoder should stay a typed effects engine with a learned look-change hazard.

## Sources

[1] lightsem_brief.md | https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/16322705/42d424d9-e895-4125-90f9-01954a00b49a/lightsem_brief.md?AWSAccessKeyId=ASIA2F3E
[2] Stage Light is Sequence2: Multi-Light Control via Imitation Learning | https://arxiv.org/html/2605.03660
[3] Beat-Aligned Spectrogram-to-Sequence Generation of Rhythm ... | https://arxiv.org/html/2311.13687
[4] SongFormer: Scaling Music Structure Analysis with Heterogeneous Supervision | https://arxiv.org/html/2510.02797
[5] Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation | https://arxiv.org/html/2607.09581
[6] Lodge++: High-Quality and Long Dance Generation With ... | https://pubmed.ncbi.nlm.nih.gov/41052184/
[7] TokenDance: Token-to-Token Music-to-Dance Generation with Bidirectional Mamba | https://arxiv.org/html/2603.27314
[8] Tempo as the Stable Cue: Hierarchical Mixture of Tempo and Beat Experts for Music to 3D Dance Generation | https://arxiv.org/abs/2512.18804
[9] Not Like Transformers: Drop the Beat Representation for Dance Generation with Mamba-Based Diffusion Model | https://arxiv.org/abs/2603.08023
[10] MG-Former: A Transformer-Based Framework for Music-Driven 3D Conducting Gesture Generation | https://arxiv.org/abs/2605.01197v1
[11] ITGPT: A Transformer Based Architecture for the Generation ... - arXiv | https://arxiv.org/abs/2607.14148
[12] STRUM: A Spectral Transcription and Rhythm Understanding Model ... | https://arxiv.org/abs/2605.12135
[13] Audio synthesizer inversion in symmetric parameter ... | https://arxiv.org/abs/2506.07199
[14] Automatic Music Mixing using a Generative Model of Effect Embeddings | https://arxiv.org/abs/2511.08040
[15] nielsr/arxiv-chandra-ocr-full-markdown-20260406 | https://huggingface.co/buckets/nielsr/arxiv-chandra-ocr-full-markdown-20260406/tree/2307/2307.16425.md
[16] OmniMotion: Multimodal Motion Generation with Continuous ... - arXiv | https://arxiv.org/abs/2510.14954
[17] OmniMotion-X: Versatile Multimodal Whole-Body Motion Generation | https://www.arxiv.org/abs/2510.19789
[18] FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation | https://arxiv.org/abs/2511.21029
[19] Automatic Stage Lighting Control: Is it a Rule-Driven ... | https://iclr.cc/virtual/2026/poster/10008746
[20] Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task? | https://arxiv.org/pdf/2506.01482.pdf
[21] Token-to-Token Music-to-Dance Generation with Bidirectional Mamba | https://arxiv.org/abs/2603.27314
[22] Wan-Dancer: A Hierarchical Framework for Minute-scale Coherent Music-to-Dance Generation | https://arxiv.org/abs/2607.09581v3
[23] Update README.md · Wan-AI/Wan-Dancer-14B at 85ce88d | https://huggingface.co/Wan-AI/Wan-Dancer-14B/commit/85ce88dd8d025459dcf0fe93982d6da8b9002957
[24] [Paper Note] Automatic Stage Lighting Control: Is it a Rule ... | https://en.papernotes.org/ICLR2026/audio_speech/automatic_stage_lighting_control_is_it_a_rule-driven_process_or_generative_task/
[25] Wan-AI/Wan-Dancer-14B - Hugging Face | https://huggingface.co/Wan-AI/Wan-Dancer-14B
[26] Comfy-Org/Wan-Dancer - Hugging Face | https://huggingface.co/Comfy-Org/Wan-Dancer
[27] Wan-Dancer: A Hierarchical Framework for Minute-scale ... | https://www.alphaxiv.org/replicate/2607.09581
[28] 一张照片跳到160 秒：万相团队开源Wan-Dancer-14B 本地部署 ... | https://blog.mushroom.cv/blog/wan-dancer-14b-one-photo-dance-video-generation/
[29] What is Wan-Dancer-14B? The Music-to-Dance AI Model ... | https://wan-dancer.com/what-is-wan-dancer
[30] What Is Wan-Dancer? Music-to-Dance Video Explained | https://wavespeed.ai/blog/ai-models/what-is-wan-dancer/
[31] Wan2.2 Dancer: Minute-Scale Music-to-Dance Video Generation ... | https://comfyui-wiki.com/en/models/wan/wan-dancer
[32] Wan-Video/Wan-Dancer — GitHub trending stats &amp; insights | https://trendshift.io/repositories/81501
[33] TokenDance：通过协同KV缓存共享扩展多智能体大语言模型服务 | https://www.alphaxiv.org/zh/abs/2604.03143
[34] ModelScope on X: "Wan-Dancer is here! A music-driven ... | https://x.com/ModelScope2022/status/2076662931866177759
[35] [PDF] Physically Optimized Text-guided Whole-Body Locomotion for ... | https://arxiv.org/pdf/2603.17927.pdf
[36] [PDF] Understand Before You Imitate for Video to Humanoid Locomotion | https://arxiv.org/pdf/2512.23649.pdf
[37] Strum - Play Guitar - App Store - Apple | https://apps.apple.com/za/app/strum-play-guitar/id6749384181
[38] Strum - Play Guitar - App Store - Apple | https://apps.apple.com/de/app/strum-play-guitar/id6749384181
[39] Strum - Play Guitar 앱 - App Store | https://apps.apple.com/kr/app/strum-play-guitar/id6749384181
[40] Програма Strum - Play Guitar - App Store - Apple | https://apps.apple.com/ua/app/strum-play-guitar/id6749384181?l=uk
[41] shenyiyang177/strum · Hugging Face | https://huggingface.co/shenyiyang177/strum
[42] [논문 리뷰] OmniMotion: Multimodal Motion Generation with ... | https://www.themoonlight.io/ko/review/omnimotion-multimodal-motion-generation-with-continuous-masked-autoregression
[43] MG-Former: A Transformer-Based Framework for Music-Driven 3D ... | https://nanless.github.io/audio-paper-digest-blog/posts/2026-05-05-mg-former-a-transformer-based-framework-for-music/
[44] Auto-Chart | OCTAVE | https://octavestudio.tools/guide/auto-chart
[45] STRUM: A Spectral Transcription and Rhythm Understanding ... | https://nanless.github.io/audio-paper-digest-blog/posts/2026-05-13-strum-a-spectral-transcription-and-rhythm/
[46] SongFormer: Scaling Music Structure Analysis with ... | https://arxiv.org/html/2510.02797v1
[47] Intelligent Music Segmentation and Structure Analysis Using Self ... | https://jase.tku.edu.tw/jase/?tkuisotope=jase-202609-32-068
[48] modal-audio-analysis/research.md at main | https://github.com/nsthorat/modal-audio-analysis/blob/main/research.md
[49] ASLP-lab/SongFormer | https://huggingface.co/ASLP-lab/SongFormer
[50] osheroff/SongFormer · Hugging Face | https://huggingface.co/osheroff/SongFormer
[51] README.md · ASLP-lab/SongFormer at main | https://huggingface.co/ASLP-lab/SongFormer/blob/main/README.md
[52] Paper page - To catch a chorus, verse, intro, or anything else: Analyzing a song with structural functions | https://huggingface.co/papers/2205.14700
[53] MOSS-Music is an open-source music understanding ... | https://github.com/OpenMOSS/MOSS-Music
[54] all-in-one-mlx 1.0.2 on PyPI - Libraries.io | https://libraries.io/pypi/all-in-one-mlx
[55] [Literature Review] SongPrep: A Preprocessing Framework ... | https://www.themoonlight.io/en/review/songprep-a-preprocessing-framework-and-end-to-end-model-for-full-song-structure-parsing-and-lyrics-transcription
[56] all-in-one-fix 2.0.4 on PyPI | https://libraries.io/pypi/all-in-one-fix
[57] AI Song Structure Analysis: Intro, Verse, Chorus - MusicTech Lab | https://musictechlab.io/blog/music-data/automatic-song-structure-analysis-how-ai-detects-intro-verse-chorus
