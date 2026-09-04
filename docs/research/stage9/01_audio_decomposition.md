# Direction 1 — audio decomposition (Perplexity reasoning pass, 2026-09-04)

Project brief attached: `lightsem_brief.md` (copied below the answer). Answer verbatim.

The 92-channel bus already has Burger’s mix-level low/high flux, RMS, onset and beat phase. The only families worth adding are ones that **disambiguate sources** the mix collapses. Do not expect this direction to fix frame-wise coach $$R^2$$ or close light FID 36.5 → 8.1; those are corpus and conductor problems.

## What is already in the bus

Burger 2013 is the prior you are citing for body-part split: pulse clarity → whole-body/feet; 50–100 Hz flux → head speed $$r=0.73$$; 6.4–12.8 kHz flux → hands/head $$r=0.65$$; percussiveness → upper body.  Your bus already injects those two flux bands plus mix RMS and half-wave onset. Adding mix-level centroid, roughness, or another flux band is a re-encode of log-mel.[1]

H-C already killed song-structure as a **per-frame input**: BAS −0.028, FID 36.5 → 27.6 vs shuffled-donor 33.1. Feeding verse/chorus/drop channels into the Conformer is repeating a failed experiment. Offline look-ahead also did nothing. Any new family that is mostly “section energy” will fail the same way.

Inter-coach agreement is $$r=0.22$$–$$0.42$$. That is the ceiling of any dance-trained model. Extra audio channels cannot invent a mapping the coaches themselves do not share.

## Candidate families

| Family | Prior evidence that it is *not* in the 92-ch mix | Expected lift (honest) | Audit before training | RTX 5070 / ~30 h audio | Leakage / artefacts |
|---|---|---|---|---|---|
| **HTDemucs 4-stem + per-stem RMS, onset, beat/downbeat** | All-In-One’s SOTA beat/downbeat/section model is fed **Demucs spectrograms**, not the mix; tasks help each other. [2] Lighting operators cue kick vs snare vs vocal. Mix flux smears them. | RPMC-L2 event AP: **+0.03 to +0.08** over 0.31 if look-changes are source-specific. Coach $$R^2$$: **~0 frame-wise still**; maybe +0.03–0.06 at 2 s on hand-speed vs footfall. FID: **36.5 → ~32–34** if events clean up; will not approach 8.1. | Beat-sync ridge/HGB: current 92 vs 92+{4×RMS, 4×onset, 4×flux}. Targets: RPMC-L2 look-change, JD hand-speed vs footfall, Oslo 2 s energy. Kill if ΔAP &lt; 0.02 and Δ$$R^2$$ &lt; 0.02 vs mix. | HTDemucs ~4–8× RT on GPU, ~5 s/song on a 3090 TRT path. [3][4] **~4–8 h**, ~6–8 GB. Fits. | Mastered-mix bleed → false drum onsets; pumping; stereo leftovers. Demucs windows are **non-causal (~8 s chunks)**. Quote a chunked/causal counterpart. Do **not** start with BS/Mel-RoFormer: extra SDR does not matter until bleed is shown to cap onset AP. RoFormer is slower, VRAM-hungry, OOM-prone on 12 GB. [5][6] |
| **Vocal activity + lyric-boundary** | Just Dance coaches are lyric-synchronous by design; LDs punch looks on vocal entries. Acoustic vocal envelope ≠ mix RMS when the mix is guitar- or kick-heavy. | RPMC-L2: small, **+0.01–0.04**, metal/livehouse less lyric-driven. JD visual events: larger. Oslo EDM: near **zero**. Coach $$R^2$$: maybe phrase-scale, not 40 fps. | From the **vocal stem only**: RMS, VAD, onset, optional Whisper word-onset rate. Compare drums-only vs vocals-only vs both against mix. If vocals lose to drums on look-change AP, drop lyrics. | Stem is free after Demucs. VAD is CPU minutes. Whisper timestamps **~3–8 h**. | Whisper is non-causal and hallucinates; multilingual JD. Lyrics are **text**, not lighting physics — word identity will overfit song identity. Use **onset/activity**, not token embeddings. |
| **Chord / key / harmonic tension** | Rule-based ASLC used emotion/chord tables; Skip-BART (trained on **your** RPMC-L2) shows that paradigm is weak vs generative audio→HSV, and uses OpenL3 not chords. [7] Burger never needed harmony for movement. [1] | Hue histograms: **noise-level**. Motion $$R^2$$: **none**. Event AP: **none** (chords sit on bars, looks sit on transients). | Chordino/BTC + key + a tension proxy, beat-sync, ridge vs 92-ch on hue hist + events. Expect a null. | CPU **&lt;2 h**. | Key/chord models are mix-confused on EDM/DJ. Circular hue vs discrete key is the mapping the project already failed at absolute level. |
| **Structure / drop (All-In-One, SongFormer, MSAF, EDMFormer)** | SongFormer is SOTA MSA. [8] EDMFormer exists because pop MSA **misses drops**. [9] **Your H-C already refuted this as an input channel.** All-In-One’s useful part is **beat/downbeat on stems**, not the verse label. | Per-frame: **≤0 or negative BAS**, FID fake-gain like 36.5→27.6 with shuffled 33.1. Drop detectors may help **Oslo energy contrast** (you only have 1.5× loud vs quiet thirds) if used as a **conductor planner**, not a bus channel. | Replicate H-C with a **shuffled-donor** control. For drops: binary drop-onset vs mix flux/RMS predicting RPMC-L2 events and Oslo speed. If drop ≈ loudness jump, kill. | All-In-One ≈ Demucs + small transformer: **6–12 h**. SongFormer **1–3 h**. MSAF CPU, worse. EDMFormer: 98-song toy set — do not train it. | Full-track, non-causal. Section labels leak song identity. **Must quote L = 0**. Do not feed time-to-drop; that is the H-D leak pattern. |
| **Dynamics / LRA** | Short-term LUFS / LRA / crest factor. Project already **ruled out absolute level**. JD pop is brickwalled (LRA often ~5–8 LU). Mix RMS is already there. | Event AP: **~0** on JD, maybe +0.01 on Oslo. $$R^2$$: **0**. FID: **0**. | ITU BS.1770 short-term + LRA + crest, residualized against RMS. If partial $$R^2$$ &lt; 0.01, skip. | CPU **&lt;1 h**. | Mastering compression erases the feature. Non-causal integrated LUFS is illegal without a causal loudness meter. |
| **Timbre descriptors** | log-mel **is** timbre. Percussiveness in Burger is onset-slope, close to your half-wave onset. [1] | **Redundant.** Any lift will vanish after residualizing on log-mel. | Ridge of {centroid, flatness, roughness, MFCC} after regressing out log-mel. Kill if leftover $$R^2$$ &lt; 0.01. | CPU minutes. | None interesting. |
| **Frozen embeddings: MERT-v1-330M, MuQ, MusicFM, CLAP, Jukebox** | EDGE conditions on **Jukebox layer 66, 4800-D @ 30 fps**. [10] Masked FMs have an acoustic→semantic depth trend; **single well-chosen layers beat naive all-layer fusion**; rhythm peaks later (~64% depth), semantic mid (~53%), tonal early (~30%). [11] MERT-330M: mid layers 10–13 used in practice; early 3–6 timbre/rhythm, layer 23 melody. [12][13] MuQ late 9–12 beat MERT mid on a perceptual probe. [14] Skip-BART on RPMC-L2 used **OpenL3**, not these. [7] TokenDance: swapping MFCC for MERT/Jukebox is not a free win. [15] CLAP is clip-level. | Event AP: **+0.00 to +0.05** after residualizing song identity; raw numbers will look better and be fake. Coach $$R^2$$: **will not go positive frame-wise**. FID: extra variance, same H-C trap. Jukebox/CLAP: **do not run**. | (1) Song-split ridge/HGB of PCA-32/64 of **one layer group** vs 92-ch. (2) Residualize against song-mean embedding. (3) Identity leak: kNN on embeddings recovering held-out song ID. If leak high and residual lift ~0, the embedding is a **song hash**. Probe MERT ~L15–16 (rhythm) and ~L10–13 (semantic) separately; MuQ L9–12; **not** last layer. | MERT/MuQ/MusicFM: **2–6 h**, ~4–6 GB, 75-ish Hz then resample to 40 fps. CLAP: cheap and **wrong timescale**. Jukebox 5B: **days, won’t fit 12 GB** as a feature job. | Pretraining corpora contain JD/pop hits → **song identity leakage** even with song-level splits. 1024–4800-D into a 9 M Conformer **overfits**. Non-causal windows (5–30 s). Always report L = 0 chunked embeddings. |
| **Per-stem Burger flux (head/hands)** | Mix low-flux→head, high-flux→hands is **already in the bus**. The missing piece is **which source** produced the flux: kick vs hat vs vocal sibilance vs bass. [1] Conductor already maps torso/head → washes, hands → strobes. | This is how stems pay rent on **block C** (hands) vs bounce. Expect the only real shot at moving hand-speed off noise. Still bounded by inter-coach $$r$$. FID: small. | After stems: 2-band flux on **drums / other / vocals** separately. Predict JD head vs hand speed (and conductor’s two groups). Compare to mix 2-band flux. Kill if no split gain. | **~0** extra after Demucs (CPU flux). | Same stem bleed: hat energy in “other”, kick in bass. If drums-high flux ≈ mix-high flux, you gained nothing. |

## Ranked five, and the experiment that kills each

Add nothing to the Conformer until the audit beats the 92-ch baseline on a **song-split**, beat-synchronous ridge/HGB, with a **shuffled-donor** and an **L = 0** counterpart. Training the 9 M model is how you launder a null.

1. **HTDemucs 4-stem envelopes + onsets + 2-band flux** (skip RoFormer until bleed is proven).  
   **Decider:** Δ event-AP on RPMC-L2 and Δ 2 s $$R^2$$ on JD hand-speed vs footfall, 92 vs 92+stems. Threshold: +0.02 AP **and** a source-specific split (drums→foot/head, hats/vocals→hands). Else dead.

2. **Per-stem Burger flux as an explicit head/hands prior** (almost free given 1).  
   **Decider:** partial $$R^2$$ of drums-low vs drums-high vs vocal-high on head vs hand descriptors, after mix 2-band flux is in the model. If the stem split adds nothing over mix flux, Burger is already fully used.

3. **Vocal activity / phrase onset** (not lyric tokens).  
   **Decider:** vocals-only vs drums-only vs mix on look-change AP and JD visual events. Keep only if vocals win on a subset the drums lose (entries, breakdowns). Oslo as negative control.

4. **Downbeat / bar-phase only** from All-In-One (or Beat This!), **not** section labels. Current bus has kick-flux beat phase; 70% of coaches bounce 1×/2× the beat, so **bar** is the open question for metric-level and wash/pan-at-the-bar.  
   **Decider:** sin/cos bar-phase vs current beat-phase predicting metric-level and bounce period. Kill if Δ$$R^2$$ &lt; 0.02. Do **not** re-import verse/chorus.

5. **One frozen embedding, PCA’d to 32–64 D:** MuQ L9–12 **or** MERT-330M L10–13 (semantic/hue) **and** a separate MERT ~L15–16 (rhythm). Not Jukebox, not CLAP, not all layers.  
   **Decider:** residual lift after 92-ch **and** after subtracting song-mean embedding; plus song-ID kNN leak. If residual ΔAP &lt; 0.02, it is a hash of the catalogue.

Do not add chords, LRA, mix timbre, MSAF, SongFormer labels, EDMFormer, or Mel-Band RoFormer in this round. The FID gap is a **conductor + target** problem; H-B (smoothing) already moved more than any feature bus will. Direction 1 can buy cleaner **events** and a **head/hands split**. It cannot buy a dancer the coaches themselves do not agree on.

## Sources

[1] Frontiers | Influences of Rhythm- and Timbre-Related Musical Features on Characteristics of Music-Induced Move | https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2013.00183/full
[2] nielsr/arxiv-chandra-ocr-full-markdown-20260406 | https://huggingface.co/buckets/nielsr/arxiv-chandra-ocr-full-markdown-20260406/tree/2307/2307.16425.md
[3] README.md · MansfieldPlumbing/Demucs_v4_TRT at main | https://huggingface.co/MansfieldPlumbing/Demucs_v4_TRT/blob/main/README.md
[4] Demucs: Meta's AI-Powered Music Source Separation | https://runthisai.com/de/blog/demucs-guide
[5] htdemucs vs BS-RoFormer vs Spleeter: A 2026 Audio Source ... | https://aistemsplitter.org/blog/htdemucs-vs-bs-roformer-vs-spleeter-2026-benchmark
[6] BS-RoFormer / Mel-Band RoFormer: 2026's Best Source ... | https://tomodahinata.com/en/blog/bs-roformer-mel-band-roformer-vocal-separation-guide
[7] [Paper Note] Automatic Stage Lighting Control: Is it a Rule ... | https://en.papernotes.org/ICLR2026/audio_speech/automatic_stage_lighting_control_is_it_a_rule-driven_process_or_generative_task/
[8] SongFormer: Scaling Music Structure Analysis with ... | https://arxiv.org/abs/2510.02797
[9] EDMFormer: Genre-Specific Self-Supervised Learning for Music Structure Segmentation | https://arxiv.org/abs/2603.08759
[10] ZeyuLing/Motius-EDGE-AISTPP · Hugging Face | https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP
[11] What Makes a Good Layer? Assessing the Layer-Wise Intrinsic ... | https://arxiv.org/html/2608.14819
[12] [PDF] Cross-Cultural Music Similarity: Bridging Human Perception, Signal ... | https://transactions.ismir.net/articles/341/files/6a5f581690920.pdf
[13] README.md · amaai-lab/merit at main | https://huggingface.co/amaai-lab/merit/blame/main/README.md
[14] Audio Foundation Models Outperform Symbolic ... | https://arxiv.org/html/2601.19029v1
[15] Token-to-Token Music-to-Dance Generation with Bidirectional Mamba | https://arxiv.org/html/2603.27314v1
[16] lightsem_brief.md | https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/16322705/0c37ac58-686c-4aae-b3f7-e9c6929696c1/lightsem_brief.md?AWSAccessKeyId=ASIA2F3E
[17] A music source separation method integrating time–frequency ... | https://pmc.ncbi.nlm.nih.gov/articles/PMC12533092/
[18] Music-Source-Separation-Training (MSST) | https://arxiv.org/html/2607.23395v1
[19] Adopting State-of-the-Art Pretrained Audio Representations for Music Recommender Systems | https://arxiv.org/abs/2604.23077
[20] Interpretable and Perceptually-Aligned Music Similarity with Pretrained Embeddings | https://www.arxiv.org/abs/2601.19109
[21] MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation... | https://openreview.net/forum?id=17O8DqToyr
[22] Music Source Separation Training - Context7 | https://context7.com/zfturbo/music-source-separation-training
[23] Beyond Genre: Diagnosing Bias in Music Embeddings Using Concept Activation Vectors | https://arxiv.org/abs/2509.24482
[24] AEmotionStudio/roformer-models - Hugging Face | https://huggingface.co/AEmotionStudio/roformer-models
[25] Music Source Separation and Restoration models for MSR ... - GitHub | https://github.com/CPJKU/music-source-restoration
[26] TuneJury/tunejury · Hugging Face | https://huggingface.co/TuneJury/tunejury
[27] Demucs Projects | https://aitinkerers.org/technologies/demucs
[28] Exploratory Study on Body Movement: Headphones vs. ... | https://www.studocu.com/en-ca/document/university-of-ottawa/workshop-in-essay-writing/exploratory-study-on-body-movement-headphones-vs-speakers-fpsy-11-698/1487
[29] Music Source Separation download | https://sourceforge.net/projects/music-source-separation.mirror/
[30] [Literature Review] Beyond Genre: Diagnosing Bias in Music Embeddings Using Concept Activation Vectors | https://www.themoonlight.io/en/review/beyond-genre-diagnosing-bias-in-music-embeddings-using-concept-activation-vectors
[31] Automatic Stage Lighting Control: Is it a Rule-Driven ... | https://iclr.cc/virtual/2026/poster/10008746
[32] Temporally Defined Brain Network Activation Associated ... | https://onlinelibrary.wiley.com/doi/pdf/10.1002/hbm.70478
[33] Ubiquitous predictive processing in the spectral domain of ... | https://elifesciences.org/reviewed-preprints/109053
[34] Stage Light is Sequence$^2$: Multi-Light Control via Imitation Learning | https://arxiv.org/abs/2605.03660v1
[35] AN ATLAS OF GALAXY SPECTRAL ENERGY ... | https://arxiv.org/html/1312.3029v1
[36] Sensitivity analysis of Burgers' equation with shocks - arXiv | https://arxiv.org/html/1708.04332v4
[37] Spotify Audio Features Scraper | https://apify.com/musicae/spotify-audio-features-scraper
[38] [論文評述] Adopting State-of-the-Art Pretrained Audio ... - Moonlight | https://www.themoonlight.io/tw/review/adopting-state-of-the-art-pretrained-audio-representations-for-music-recommender-systems
[39] Neuroscientists find greater complexity in how we perceive motion | http://www.sciencedaily.com/releases/2011/12/111205140404.htm
[40] SC Edge Full Manual | PDF | Bluetooth | Battery Charger | https://www.scribd.com/document/1010064720/SC-Edge-Full-Manual
[41] What We Think before a Voluntary Movement | https://www.academia.edu/75671871/What_We_Think_before_a_Voluntary_Movement
[42] Build software better, together | https://ithub.global.ssl.fastly.net/topics/light-control
[43] ASLP-lab/SongFormer | https://huggingface.co/ASLP-lab/SongFormer
[44] README.md · ASLP-lab/SongFormer at main | https://huggingface.co/spaces/ASLP-lab/SongFormer/blob/main/README.md
[45] osheroff/SongFormer · Hugging Face | https://huggingface.co/osheroff/SongFormer
[46] README.md · ASLP-lab/SongFormer at ... | https://huggingface.co/spaces/ASLP-lab/SongFormer/blob/5a5fa7dd3312b57bb931f3966e4d6556959beb92/README.md
[47] ASLP-lab/SongFormDB · Datasets at Hugging Face | https://huggingface.co/datasets/ASLP-lab/SongFormDB
[48] README.md · monteslu/htdemucs-web-onnx at main | https://huggingface.co/monteslu/htdemucs-web-onnx/blob/main/README.md
[49] monteslu/htdemucs-web-onnx - Hugging Face | https://huggingface.co/monteslu/htdemucs-web-onnx
[50] ASLP-lab/SongFormer | https://github.com/ASLP-lab/SongFormer/
[51] SongFormer Structure Analysis | multimodal-art-projection/AutoMV | DeepWiki | https://deepwiki.com/multimodal-art-projection/AutoMV/3.1.1-songformer-structure-analysis
[52] Music Preprocessing Pipeline | multimodal-art-projection/AutoMV | DeepWiki | https://deepwiki.com/multimodal-art-projection/AutoMV/3.1-music-preprocessing-pipeline
[53] Music Structure Analysis (MSA) | https://www.emergentmind.com/topics/music-structure-analysis-msa
[54] A multimodal graph-based music auto-tagging framework | https://link.springer.com/article/10.1007/s10799-026-00496-3
[55] Towards Unified Music Emotion Recognition Across Dimensional and Categorical Models | https://link.springer.com/chapter/10.1007/978-3-032-31654-7_8?error=cookies_not_supported&amp;code=fc699e24-d43d-4a21-9c43-2ad2bf17d6e2
[56] Scaling Music Understanding in Audio Language Models - arXiv | https://arxiv.org/html/2511.10289v1
[57] Raveform: A Dataset of Metrical and Functional Structure... | https://reference-global.com/article/10.5334/tismir.288
[58] Update README.md · amaai-lab/merit at 2fc4641 | https://huggingface.co/amaai-lab/merit/commit/2fc4641b6699143ed00eaf59c233bb629541ee17
[59] README.md · treadon/banger-scorer at main | https://huggingface.co/treadon/banger-scorer/blob/main/README.md
[60] kckDeepak/AI-DJ-Mixing-System - GitHub | https://github.com/kckDeepak/AI-DJ-Mixing-System
[61] all-in-one-mps 1.0.0 on PyPI - Libraries.io | https://libraries.io/pypi/all-in-one-mps
[62] All-In-One 音乐结构分析器：从音频解析到段落识别的全流程指南 | https://blog.gitcode.com/9d3636637dabff3f59f1bc138a3ffe77.html
[63] 5步精通音乐结构分析：从环境搭建到实战应用的全流程指南 | https://blog.gitcode.com/8fe7c9629c27169d596df42c56ef64aa.html
[64] all-in-one-fix 2.0.4 on PyPI | https://libraries.io/pypi/all-in-one-fix
[65] Is there a TouchDesigner project for detecting builds, drops and energy? | https://www.reddit.com/r/TouchDesigner/comments/1vh1zzw/is_there_a_touchdesigner_project_for_detecting/
[66] Cross-modal deep generative models reveal the cortical ... - PMC | https://pmc.ncbi.nlm.nih.gov/articles/PMC12627580/
[67] [PDF] Academic Text-to-Music Grand Challenge: Datasets, Baselines, and ... | https://arxiv.org/pdf/2605.21538.pdf
[68] melband-roformer-infer | https://www.piwheels.org/project/melband-roformer-infer/
[69] chenmozhijin/BSRoformer-GGUF · Hugging Face | https://huggingface.co/chenmozhijin/BSRoformer-GGUF
[70] README.md · musetric/vocal-separation-roformer-onnx at main | https://huggingface.co/musetric/vocal-separation-roformer-onnx/blob/main/README.md
[71] mlboydaisuke/MelBandRoformer-Vocal-CoreAI | https://huggingface.co/mlboydaisuke/MelBandRoformer-Vocal-CoreAI
[72] amaai-lab/merit - Hugging Face | https://huggingface.co/amaai-lab/merit
[73] README.md · Atrac/MERT-v1-330M-finetuned-gtzan at main | https://huggingface.co/Atrac/MERT-v1-330M-finetuned-gtzan/blob/main/README.md
[74] GitHub - chenmozhijin/BSRoformer.cpp: GGML-based C++ inference for BS Roformer/Mel-Band-Roformer vocal separat | https://ithub.global.ssl.fastly.net/chenmozhijin/BSRoformer.cpp
[75] RVC-Boss/GPT-SoVITS - Shadowgraph | https://shadowgraph.io/RVC-Boss/GPT-SoVITS
[76] Calculate Neural Network Memory Use | https://cal3.calculator.city/calculate-neural-network-memory-use/
[77] Edge Dance - LogicWeb | https://www.logicweb.com/ai/edge-dance/
[78] Model-Specific Configurations | ZFTurbo/Music-Source ... | https://deepwiki.com/ZFTurbo/Music-Source-Separation-Training/11.2-model-specific-configurations
[79] Edit Loudness Meter settings - Adobe Help Center | https://helpx.adobe.com/ca/premiere/desktop/add-audio-effects/apply-audio-effects/edit-loudness-meter-settings.html
[80] Benchmarking multi-output models for continuous music emotion ... | https://link.springer.com/article/10.1007/s00521-026-12370-y?error=cookies_not_supported&amp;code=a96836be-0fbc-4717-a18d-3b04c3857e15
[81] Cross-Cultural Music Similarity: Bridging Human Perception ... | https://transactions.ismir.net/articles/10.5334/tismir.341
[82] 论文解读 -- MERT: ACOUSTIC MUSIC UNDERSTANDING MODEL WITH LARGE-SCALE SELF-SUPERVISED TRAINING | https://blog.csdn.net/qq_34941290/article/details/153203833
[83] What Makes a Good Layer? Assessing the Layer-Wise ... | https://www.alphaxiv.org/abs/2608.14819
[84] APEX: Large-scale Multi-task Aesthetic-Informed Popularity ... | https://www.alphaxiv.org/abs/2605.03395
[85] 2024.findings-naacl.231 | PDF | Data Compression | https://www.scribd.com/document/910477943/2024-findings-naacl-231
[86] LUFS Audio Volume Guide for Podcasts, Videos, Lessons and Music | https://www.audio-editor.online/blog/lufs-audio-volume-guide
[87] docs/sota-research-2026.md - wanghai-tech / hikoon-ACR | https://gitlab.hikoon.com/wanghai-tech/hikoon-ACR/blob/60e0f9e3fa475fba593139908637bb31ab72c3f0/docs/sota-research-2026.md
[88] What are LUFS? Complete Guide for Music, Podcasts &amp; Broadcast | https://nugenaudio.com/what-are-lufs/
[89] P5-7: CultureMERT: Continual Pre-Training for Cross-Cultural Music ... | https://ismir2025program.ismir.net/poster_210.html
[90] Target LUFS for YouTube, TikTok, and Spotify (2025) | https://clickyapps.com/creator/video/guides/lufs-targets-2025
[91] LUFS Loudness Calculator - Free Filmmaking Calculator | https://www.toolsforfilm.com/tools/lufs-loudness
[92] circular economy strategies for fibrous polymer composites | https://www.sciencedirect.com/science/article/pii/S277239762600016X
[93] madmom-infer · PyPI | https://pypi.org/project/madmom-infer/
[94] Twenty-Five Years of MIR Research: Achievements, Practices, Evaluations, and Future Challenges | https://arxiv.org/abs/2511.07205
[95] midashenglm | https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/midashenglm/
[96] Advances in Building Energy Management: A Comprehensive Review | https://www.mdpi.com/2075-5309/15/23/4237
[97] Hi! I'm Virtual Riot and I just put out my new Album "Burning Out"! AMA | https://www.reddit.com/r/EDM/comments/1snbf8j/hi_im_virtual_riot_and_i_just_put_out_my_new/
[98] How to Build Tension and Drops in Edm - musicproductionwiki.com | https://musicproductionwiki.com/articles/how-to-build-tension-and-drops-in-edm
[99] Command Line Interface | https://lib.rs/crates/beat-this
[100] Analyzing EM/IR in IC Design Layouts for Reliability | https://studylib.net/doc/28308498/analyzing-em%EF%BC%8Fir-in-ic-design-layouts-to-ensure-reliabilit...
