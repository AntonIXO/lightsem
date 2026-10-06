# Source separation as supervision for LightSem audio understanding

Research date: 2026-10-04. This is a literature/reproducibility assessment, not a local model comparison. No model weights or datasets were downloaded, no inference/training ran, and no LightSem experiment verdict is changed. Papers were retrieved/read through arXiv MCP; official repositories, model pages and dataset releases were independently inspected. See `separation_datasets.md` for the data audit.

## Main recommendation and scope

Keep a broad candidate set: coarse stem separators, query-conditioned separation, discriminative complex masks, generative separation and compact conditioned U-Nets. LightSem's earlier failure with htdemucs-derived mono envelope features applies to that representation/integration/run, not to all separation approaches. A warranted revisit would retain stereo source images, expose event-level and source-specific latent trajectories, supervise source absence, and measure source removal/time-shift interventions before downstream choreography tests. A higher separator SDR alone cannot establish that this revisit worked.

The useful role of a separator is an **offline teacher or auxiliary reconstruction task**. It can provide candidate isolated waveforms, source activity and source-conditioned features. These are noisy pseudo-labels; true multitracks/MIDI remain preferable for teaching source identity and precise timing. The compact LightSem encoder should not need to regenerate studio-quality waveforms at runtime merely to identify who played what and when. That is a proposed design, not a result from these papers.

## Paper reading evidence

### Hybrid Transformers for Music Source Separation — Rouard, Massa, Défossez (2022/ICASSP 2023)

[Paper](https://arxiv.org/abs/2211.08553), [author repository](https://github.com/facebookresearch/demucs), [maintainer fork](https://github.com/adefossez/demucs). Read via arXiv MCP `download_paper`: abstract, architecture, dataset construction, experimental setup and Table 1.

The hybrid time/spectrogram U-Net replaces inner layers with alternating within-domain self-attention and cross-domain attention. Table 1 reports 7.52 dB MUSDB18-HQ SDR without extra data, 8.80 with 800 extra songs, 9.00 after source-specific fine-tuning and 9.20 for the sparse fine-tuned variant. This illustrates that architecture and training-data changes cannot be conflated. Four output groups are vocals/drums/bass/other; a group is not an individual source instance.

The official release provides `htdemucs`, `htdemucs_ft` and experimental `htdemucs_6s` (adds piano/guitar); it explicitly warns about piano artifacts. The paper's sparse 9.20 dB model is not supplied in this release. API output can preserve stereo, whereas converting stems to mono later destroys this information. Official CLI output rescaling may change relative stem levels, so a teacher pipeline must retain a documented shared gain and inspect reconstruction residuals. The repository is archived; use a pinned revision. MIT repository license is verified; record the exact upstream weight distribution/terms separately rather than extrapolating to training-data rights.

**LightSem role:** retain as a continuity control and usable coarse teacher, especially to distinguish “feature bottleneck failed” from “separator failed.” Six stems do not solve individual drums or long-tail instrumentation.

### Music Source Separation with Band-Split RoPE Transformer — Lu et al. (2023/ICASSP 2024)

[Paper](https://arxiv.org/abs/2309.02612). Read via arXiv MCP `download_paper` and cached continuation: system/method sections 2–3, all experiments/results in section 4, tables 1–2 and conclusion.

BS-RoFormer splits the complex stereo spectrogram into 62 uneven non-overlapping bands, alternates time and band attention with RoPE, and predicts complex masks. Training combines waveform MAE with multiresolution complex-STFT MAE. The paper reports 9.80 dB median SDR on MUSDB18-HQ without extra data for the six-block variant; this is distinct from 9.97 **global** SDR on the SDX23 private test and 11.99 median MUSDB18-HQ SDR for the twelve-block system using extra data. These numbers are not directly interchangeable.

The winning configuration has 93.4M parameters **per separator** and trains three separate models for vocals, bass and drums; “other” is the subtraction residual. Training each took four weeks on 16 A100-80GB GPUs. Even the 72.2M six-block model used 16 V100-32GB GPUs. Random stem mixtures improve separation but deliberately break musical relationships; a rhythm-interaction learner must also see coherent same-song multitracks.

**LightSem role:** candidate stronger offline coarse-stem teacher and reusable time×band latent architecture. Keep source reconstruction/event accuracy separate from downstream relevance. It does not by itself identify kick versus snare, interactions, or phrase intent. Public implementations/checkpoints are not automatically the paper's exact system.

### Banquet: A Stem-Agnostic Single-Decoder System for Music Source Separation Beyond Four Stems — Watcharasupat, Lerch (2024)

[Paper](https://arxiv.org/abs/2406.18747), [author code](https://github.com/kwatcharasupat/query-bandit), [official weights](https://zenodo.org/records/13694558). Read via arXiv MCP through methods, full setup and results including section 5.4; the directly inspected HTML identifies v2, 26 August 2024.

Banquet produces a shared time×band mixture embedding; a frozen instrument-recognition PaSST encodes an audio example, then FiLM conditions a shared decoder to estimate that source's complex mask. It supports multichannel mixtures; only its query is downmixed. This is the closest direct recipe here for query-addressable source representations. The separator has 24.9M trainable parameters, excluding the fully frozen query extractor. Encoder pretraining is important; training from scratch is weaker. The published experiments used an H100-80GB, six-second crops and an overlap-heavy inference schedule.

Crucial limits: guitar/piano scores remain modest; fine-grained same-instrument timbre distinctions are weak. In the `q:all` extension, newly added long-tail stems are all below 5 dB SNR, with frequent nearly silent outputs. Authors restarted collapsed runs; no trainable-encoder + augmentation + balanced-sampling run finished without collapse. This is not a reliable universal instrument oracle. Its architecture still merits testing with better labels, explicit absence supervision and failure accounting.

**LightSem role:** study/distill source-conditioned latent trajectories; first validate common instrument families and absent queries. Multiple queries yielding the same audio do not demonstrate multiple independent sources.

### QSCNet: A Conditioned UNet for Music Source Separation — O'Hanlon et al. (2025)

[Paper](https://arxiv.org/abs/2512.15532), [full text](https://arxiv.org/html/2512.15532v1). Read methods and results via arXiv MCP `download_paper` and cached continuation; primary HTML independently inspected.

QSCNet revisits U-Nets rather than accepting a family-level exclusion. It inserts a single PaSST-conditioned FiLM **before** SCNet's dual-path recurrent neck. On the MoisesDB 144/48/48 split it reports Avg5 SNR 8.5 dB versus Banquet's 6.9, with a 10.2M separator (roughly 40% of Banquet's parameters). However, training data sampling/augmentation and setup differ, so this does not isolate conditioning location as the sole cause. The fixed-output SCNet6 scores 9.2, showing an efficiency/flexibility tradeoff. Published training uses one A100-80GB. No author QSCNet code/checkpoint was verified in this survey. Treat it as a compact reimplementation candidate, not an immediately available teacher. Its stereo complex-input design is relevant; paper comparisons do not establish LightSem event fidelity or local 12GB training feasibility.

### SAM Audio: Segment Anything in Audio — Shi et al. (2025)

[Paper](https://arxiv.org/abs/2512.18099), [official code](https://github.com/facebookresearch/sam-audio), [small model](https://huggingface.co/facebook/sam-audio-small), [processor source](https://github.com/facebookresearch/sam-audio/blob/main/sam_audio/processor.py), [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE). Read via arXiv MCP `download_paper`, `get_paper_outline` and `read_paper_section` (Approach, Experimental Setup, Results); official processor and primary HTML independently inspected.

SAM Audio uses flow matching with a diffusion transformer and text/visual/span prompts to produce target and residual. Small/base/large have 500M/1B/3B parameters **excluding** text/visual encoders and the audio codec. Evaluation usually uses the large model, sixteen ODE steps and eight-candidate reranking. Its professional instrument comparison uses ten-second MUSDB clips for vocals/drums/bass; favorable subjective scores are not proof of all-instrument transcription or timing accuracy. Paper latency is measured on A100, not RTX 5070.

A particularly transferable ingredient is its auxiliary audio-representation alignment: intermediate prompt-conditioned states project to a frozen audio-event detector's framewise target embedding and use cosine matching. This directly supports considering event-level teacher targets for LightSem, although SAM disables the auxiliary loss during clean-data fine-tuning; it is not evidence the loss is universally beneficial. Its 25 Hz DAC-VAE latents describe codec content, not automatically instrument identity or exact percussive-onset truth.

The official `batch_audio` processor averages input channels (`wav.mean(0)`): this release is **not a stereo source-image teacher**, even when given a stereo file. Weights require gated access and use the custom SAM License, not Apache/MIT. Its plausible role is broader semantic/event pseudo-labeling, with fidelity and absent-target checks before use. Do not train LightSem to copy plausible invented events or treat a text prompt as evidence the source exists.

## Reproducibility and adoption matrix

| Candidate | Available material checked | License/provenance distinction | Proposed local action; unverified limits |
|---|---|---|---|
| HT-Demucs | Author code, explicit model names and download path | MIT repository; proprietary extra training material; exact selected-weight provenance must be retained | Cache stereo estimates with one shared scale; known baseline control. Chunked inference is documented, but no RTX 5070 benchmark performed. |
| BS/Mel-RoFormer | [lucidrains implementation](https://github.com/lucidrains/BS-RoFormer), [ZFTurbo training/weights registry](https://github.com/ZFTurbo/Music-Source-Separation-Training), [Ju-Chiang Wang's repository](https://github.com/asriverwang/BS-RoFormer) | lucidrains MIT is implementation permission. Community checkpoint rights/training provenance are per asset. Author-associated repository inspected has source/inference code but no verified released paper checkpoint/license file. | Select one named, hash-pinned checkpoint only after checking training provenance and its actual output targets; do not equate a vocal-only checkpoint with four-stem separation. Offline profiling first. |
| Banquet | Author training/inference code, split files, Zenodo `ev-pre-aug.ckpt` recommended by README | Code MIT; **official weights CC BY-NC-SA 4.0**, verified from Zenodo record API. PaSST dependency/data rights separate. | Best query-separation reproduction starting point. Test smaller inference batch on 12GB; README's batch 12 RTX 4090 example does not establish 5070 fit. No broad pseudo-label batch before long-tail/silence audit. |
| QSCNet | Primary paper; SCNet architecture family | No official QSCNet code/weight license verified | Promising small reimplementation branch after independent common-source benchmark; do not promise full-paper reproduction on 12GB. |
| SAM Audio | Official inference/evaluation code, gated model pages | SAM License explicitly covers model weights/software; gates and dependencies apply | Optional semantic teacher, mono only in inspected release. Small-model inference memory/latency and event faithfulness untested locally. |

Banquet Zenodo API inspected: `https://zenodo.org/api/records/13694558`, `metadata.license.id = cc-by-nc-sa-4.0`. API GitHub profile `asriverwang` identifies Ju-Chiang Wang; this supports author association, not an inferred weight license. All release availability statements are dated observations, not a guarantee of future downloadability.

## Compact encoder and evaluation proposal

This section is LightSem-specific inference from the literature. Preserve original stereo mixture and two-channel source estimates. Cache teacher outputs once with checkpoint hashes, code revision, resampling/window settings, query exemplars, seed and shared gain. For 12GB hardware, train a compact time-frequency student on short examples or cached latents; first measure peak memory and step time. A parameter count alone does not predict memory. Teacher feature matching should be weighted by independently validated target confidence and mixed with real multitrack supervision.

Suggested outputs are time-varying source slots carrying identity/activity, event onset/offset and accent, envelope/register, stereo balance/width, and confidence. Add a slower context path for recurring motifs, phrase changes and relationships between source event streams. Do not collapse the source axis or time axis into one pooled embedding. A single source can drive many lights without becoming many independent sources.

Use coherent multitracks to teach interplay; use isolated-source removal, insertion, delay, amplitude and panning interventions to test causal specificity. The rest of the mixture stays identical. Compare per-source activity and onset performance, absent-source false positives, duplicated/silent outputs, reconstruction residual, stereo-image error and representation rank/variance. Measure teacher instability separately from student collapse. Compare frozen weights/noise/settings on distinct tracks, including similar-BPM music, and after beat/tempo alignment. New soundtrack over old trajectories is not new conditioning.

At the downstream stage preserve the accepted body source and P2 color baseline; do not require dancer retraining to test the sound encoder. Diagnose loss of event/gesture information at audio encoder, body conditioning, mapper and emitted fixture controls separately. Any future demonstration needs both Shpongle and AURORA—Cure For Me with their own body trajectories; actual lamp motion/color/brightness and source dependence must pass the existing project gates.

## Retrieval status

Completed arXiv MCP paper-body reading: `2211.08553`, `2309.02612`, `2406.18747`, `2512.15532`, `2512.18099`. Metadata search/get_abstract requests encountered upstream timeout/HTTP429; paper-body HTML retrieval succeeded and cached continuations/section reads were available. SAM's MCP outline is coarse/mis-segmented in places, so section IDs and content were checked against primary HTML. No claim of local experimental verification follows from paper retrieval.
