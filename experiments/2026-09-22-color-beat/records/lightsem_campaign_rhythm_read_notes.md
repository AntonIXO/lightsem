# LightSem rhythm paper reading notes

Scope: primary full-text/source review only. No model runs, GPU work, training, installs, datasets, or vault calls. Retrieved 2026-09-22T09:50:15.567442+00:00. Full arXiv texts read, including AIST++ appendices; All-In-One PDF Figure 3 read for ablation numbers. Repository implementations were inspected, not executed.

Dossier: `/tmp/lightsem_campaign_rhythm_papers.json`.
Original full texts, PDFs, metadata and inspected code: `/tmp/lightsem_campaign_rhythm_sources/`. JSON contains file SHA256 hashes and pinned code revisions.

## Campaign decisions supported by the reading

1. **Lighting-accent policy is not beat tracking.** A tracker supplies the musical clock. A light policy chooses which beats/subdivisions/phrases to express, what to suppress, and attack/decay, intensity and spatial/color changes. Neither tracker training corpus labels nor dance minima are lighting-accent labels. Keep timing-instrument evaluation separate from emitted-light acceptance.
2. **Keep the dancer unchanged.** Direct audio timing can drive LIGHT while internal motion provides spatial style. None of these papers authorizes retraining/retiming the dancer as the lighting solution.
3. **Output precision is not label accuracy.** Beat This outputs every20ms; All-In-One every10ms. Both papers use beat F1 with +/-70ms matching. All-In-One section boundaries use +/-0.5s. AIST++60FPS is a time grid, not annotated beat precision. Preserve annotation uncertainty, PCM time origin, resampling, and render/DMX timing separately.
4. **No clean benchmark claim without training-overlap audit.** Beat This final*/small* trained on GuitarSet comping and Harmonix. All-In-One default harmonix-all averages all8 CV models; every Harmonix song trains6 components. Current AIO source makes fold IDs from sorted available MP3 names modulo8: a partial/re-versioned local collection gives the wrong folds. Need original manifest+OOF model or external, rights-cleared, independently annotated songs. JustDance is not a named training corpus, but song/recording overlap is unverified.
5. **Treat raw activations carefully.** Beat This intentionally drives overconfident predictions. A high sigmoid output is not calibrated uncertainty or a safety/fallback guarantee. All-In-One functional labels are sections, not lighting salience or a learned drop detector.

## Beat This! — arXiv2407.21658v1 / ISMIR2024

Original: https://arxiv.org/html/2407.21658v1
Code: https://github.com/CPJKU/beat_this
Revision: b95c8ab0c58c2d9fcfd40508ae8dffbc05ac4f5c; inspected package1.1.0.

- Full-context offline30s model. 22.05kHz mono,128 mel bins,1024 window,441hop. Convolution + alternating frequency/time partial transformers; then6 temporal transformer layers,512 hidden/16heads.
- Two linear heads: beat logit sums beat+downbeat logits. This encourages hierarchy but does not guarantee it. Minimal local maxima + threshold0.5; downbeats snap to nearest beat. Current release pads/discards6-frame edges and overlaps chunks; paper describes nonoverlapping30s excerpts. Pin source.
- Positive-class weighting addresses sparse events. Max-pooled7-frame predictions make BCE shift-tolerant; negative loss ignored within6frames of marks. Pitch, tempo, local-shuffle masking augmentations. No constant-tempo auxiliary head.
- **Published GTZAN comparison:** Beat This beat/downbeat F1 **89.1+/-0.3 / 78.3+/-0.4% versus Hung et al. SpecTNT-TCN+DBN88.7 /75.6%**. But beat CMLt/AMLt **79.8/89.8 versus81.2/92.0%**, and downbeat CMLt/AMLt **67.3/79.1 versus71.5/88.1%**. F1 improves; continuity does not. Hung result was not reproduced and no significance test was possible.
- Matched-Hung-data variant has downbeat F1 **75.5+/-0.5 versus Hung75.6%**. Do not repeat a blanket claim that both metrics improve under matched training data.
- Loss ablation: full versus no-shift-tolerance weighted BCE gives beat F1 **92.6 versus91.2%** and downbeat **85.4 versus82.2%** on the single validation split, not GTZAN.
- Data:18 groups,4556 training/validation tracks; GTZAN993 test-only tracks. Includes GuitarSet comping, Harmonix, classical, jazz, solo instruments. Annotation ambiguities and clear GTZAN defects are acknowledged. v1.0 annotation release also warns of bad RWC conversions.
- Interface: `File2Beats`, `Audio2Beats`, `Audio2Frames` (50Hz logits), `load_model`. final0/1/2 ~78MB; small0/1/2 ~8.1MB; single_final and fold0..7 available. CPU inference interface exists; no native environment validation done.
- Dependencies:torch>=2,torchaudio,numpy,einops,rotary-embedding-torch,soxr; optional madmom DBN. **Code and weights MIT explicitly.** PaperCC-BY4.0. Audio rights remain separate.

## All-In-One — arXiv2307.16425v1 / WASPAA2023

Original: https://arxiv.org/html/2307.16425v1
Code: https://github.com/mir-aidj/all-in-one
Revision: 18e78903c0365147a2c5d4e5e57ebf88cb7d800e; package1.1.0. Weights: https://huggingface.co/taejunkim/allinone at 379e5fd010b3fdd0ee8381ff8cbcfa51d70b5c19.

- Demucs separates bass/drums/other/vocals. Shared conv frontend,11 modules alternating two dilated temporal branches and local instrumental attention. C24, about300K task parameters **excluding Demucs**. Long-context offline system; branch receptive fields reported ~41/82s.
- Four learned heads: beat, downbeat, section boundary,10-class function. **BPM is derived from beat intervals, not a fifth learned head.** Current code uses BCE/BCE/BCE/CE, task weights1/3/15/0.1 and masked losses. Beat labels widen +/-1frame, sections +/-2;100Hz is representation resolution, not annotation accuracy.
- DBN produces beat/bar positions; current code restricts bars to3 or4beats. Section postprocessing uses24s local normalization/peak picking. Functional labels=start,end,intro,outro,break,bridge,inst,solo,verse,chorus.
- **Published matched-size comparison:** on Harmonix8folds, All-In-One versus authors' demixed TCN-Large(301K) gives beat F1 **.958 versus.953**, downbeat **.915 versus.901**, sectionHR.5F **.660 versus.626**, labelPWF **.738 versus.717**, labelSf **.769 versus.746**. Prior SpecTNT-TCN has beatCMLt **.939 versus All-In-One.913**, so not every metric improves.
- **Negative transfer matters:** full downbeatF1 **.915 versus.916 without label head**, while sectionF1 falls **.660 to.624**. Removing BOTH beat/downbeat heads raises labelPWF **.738 to.744** but sectionF1 falls **.660 to.653**. Do not assert all heads mutually improve; the latter does not isolate downbeat-only interference.
- Only Harmonix supervised data, no augmentation or extra labeled datasets. Paper split6train/1val/1test. Released default ensemble is not OOF. JustDance exact-song overlap unknown. GuitarSet not a named supervised training set, but Demucs pretraining exposure unaudited.
- `allin1.analyze(..., include_activations=True, include_embeddings=True)` returns event seconds, bar positions, segments;100Hz raw arrays and `[4,T,24]` embeddings (extra8-model axis for ensemble).
- Dependencies:NATTEN matching PyTorch/CUDA,Demucs,madmom,librosa,numpy,hydra/omegaconf,HF hub,ffmpeg. Native torch2.14/cu130/RTX5070 compatibility untested. Use one canonical WAV: README warns MP3 decoder offsets ~20-40ms.
- **CodeMIT and HF weight cardMIT.** PaperCC-BY4.0. Underlying audio/dependency licenses remain separate.

## AIST++ / FACT — arXiv2101.08779v3 / ICCV2021

Original: https://arxiv.org/html/2101.08779v3
Project: https://google.github.io/aichoreographer/
Dataset/API: https://google.github.io/aistplusplus_dataset/ and https://github.com/google/aistplusplus_api
Model: https://github.com/google-research/mint
API revision:2dd7b3e946b794fd0081c98e2e2433545abf8b87; FACT revision:b8f8bdfbbe3fbfa67831a2ef9bcf71a6a9e74552.

- AIST++ is reconstructed multi-view motion, not mocap or human beat annotations. 1408sequences,60songs,30dancers,10genres,5.2h;9camera views are not9 independent motions. Basic dances repeat the same choreography across6songs/tempi. Split both music AND choreography, not random clips.
- Exact60FPS annotations; some raw videos differ. Bundle adjustment/triangulation/SMPL fitting uses temporal/bone constraints. Reprojection agrees with a detector, not independent3D truth. Main-text PCKh98.7 differs from appendix98.4; avoid using either as verified ground truth.
- FACT:120motion seed frames+240audio frames;2-layer modality encoders and12-layer deep full-attention cross-modal block,800hidden/10heads. Future20-frame L2 supervision; keep first next frame autoregressively. Deep fusion/future-N results concern motion stability, not learned beat tracking.
- Audio35dims=envelope+20MFCC+12chroma+onset peaks+librosa beats. Official preprocessing seeds librosa `start_bpm` from the AIST filename and uses`tightness=100`. A dataset-BPM-assisted audio feature is not a human beat label.
- **BeatAlign is one-way motion-velocity-minimum proximity to librosa beats.** Gaussian sigma3frames at60Hz=50ms. It is not a hard70ms F1, not position minima, not a ground-truth annotation, not beat recall or bar correctness, and not lighting quality.
- FACT BeatAlign **.241 versus Dancenet.232**, **random-paired real motion/music.213**, and **real-paired.292**. These are empirical comparators, not universal floor/ceiling. FACT kinetic/geometric motionFID **35.35/12.40 versus Dancenet69.18/17.76**; no lighting claim follows.
- **Perception contradicts a broad metric claim:** FACT wins only25.4% against random-paired REAL motion/music despite BeatAlign.241 versus.213. Authors call this expressiveness bias; the measured fact is preference/metric divergence, not that users were wrong.
- Limits:deterministic, no floor physics, foot sliding/floating; genre/choreography confounding remains relevant. Official inspected score script contains FID but no BeatAlign reproduction; do not invent an exact smoothing implementation.
- API loads SMPL arrays/scale/translation and COCO joints. Legacy TensorFlow/orbit/protobuf/SMPL pipeline and official Drive checkpoint exist. No use needed for a direct LIGHT timing campaign. Paper/preprocessing219-dim motion versus score script225-dim convention needs adapter review if ever reused.
- **PaperCC-BY4.0; API+FACT codeApache2.0.** Explicit Drive-weight license and separate AIST++ annotation license were not found in inspected pages. Original AIST media is **academic-research-only, no unauthorized redistribution; commercial/non-research requires written consent**. SMPL has separate terms. Code license does not grant music/video/model-data rights.

## Remaining gates

- Freeze exact checkpoint, source, waveform and annotation hashes before measurement.
- Resolve supervised song overlap and source-separator exposure at the claim scope required. Until then report contaminated corpus results as descriptive.
- Independently annotate musical timing and, if training an accent policy, annotate/define the DIFFERENT light-accent target. Do not silently substitute pose minima.
- No experiments are preregistered by this file; parent alone owns vault writes and campaign authorization.
