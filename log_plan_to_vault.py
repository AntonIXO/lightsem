#!/usr/bin/env python3
"""Write the LightSem v2 ("dancing light") plan into the LIGHTSEM vault.

Companion to PLAN.md (2026-09-03). Registers: tags, papers, diagnostics, a new
problem, runbooks (one per stage, self-contained prompts with gates), a
synthesis, and an update on the original problem pointing here.

Runs over MCP stdio with VIBESCIENCE_VAULT pinned to /root/lightsem-vault via
log_to_vault.Client, so nothing can land in the optiHealth vault. Re-runnable:
the server rejects duplicate ids, which is reported, not fatal.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client, PROBLEM as OLD_PROBLEM  # noqa: E402

NEW_PROBLEM = "dancing-light-generative-20260903"


def snapshot(path: str) -> int:
    n = 0
    for _, _, files in os.walk(path):
        n += len([f for f in files if f.endswith(".md")])
    return n


def call(c: Client, name: str, args: dict) -> dict | str:
    out = c.call(name, args)
    short = json.dumps(out, ensure_ascii=False)[:220] if not isinstance(out, str) else out[:220]
    print(f"  {name:22s} {args.get('id', args.get('title', args.get('name', '')))!s:48.48s} -> {short}")
    return out


# --------------------------------------------------------------------------- tags
TOPIC_TAGS = [
    ("generative-modelling",
     "Treating the target as a distribution to SAMPLE from (diffusion, flow matching, "
     "diffusion forcing, VQ+AR) rather than a function to regress. Use when p(y|x) is "
     "multi-modal and per-frame MSE collapses to the conditional mean.",
     ["generation", "diffusion", "flow-matching", "diffusion-forcing", "sampling", "generative"]),
    ("music-to-dance",
     "Music-conditioned dance/motion generation (EDGE, FlowerDance, InfiniteDance, "
     "DiscoForcing) and its datasets (AIST++, FineDance, InfiniteDance). In LightSem the "
     "generated dancer is a LATENT intermediate: present at train time, sampled at test time.",
     ["dance-generation", "latent-dancer", "choreography-generation", "m2d", "music2dance"]),
    ("real-time",
     "Strictly causal, bounded-latency streaming inference (audio-in to DMX-out). Covers "
     "online beat tracking, causal audio encoders, diffusion forcing, teacher->causal-student "
     "distillation, lookahead/delay-line trade-offs.",
     ["streaming", "causal", "latency", "online", "live"]),
    ("evaluation-metric",
     "Design and validation of metrics: Light-BAS (beat alignment of light kinematics), "
     "FID over light features, diversity, collapse rate, human pairwise preference, and the "
     "nulls (metronome, circular shift, split-half) each one must be priced against.",
     ["metrics", "beat-alignment", "fid", "human-eval", "null-model"]),
    ("motion-to-light",
     "Mapping body kinematics (speed, jerk, spread, verticality, laterality, Laban effort) to "
     "fixture parameters (dimmer, pan/tilt, zoom, strobe, colour). The conductor layer.",
     ["pose-to-dmx", "laban", "kinematics-to-light", "conductor", "mapping"]),
]
PROBLEM_TAGS = [
    ("conditional-mean-collapse",
     "Using a regression/MSE objective where p(target|input) is multi-modal: the optimal "
     "predictor is the conditional mean, a blurred low-energy average no real sample looks "
     "like, so held-out R2 is ~0 even when every draw from the true conditional would be "
     "acceptable. Diagnosed by an inter-annotator/inter-performer ceiling near zero at the "
     "target resolution. Remedy is a generative objective and distributional evaluation, not "
     "a different regression target.",
     ["mode-averaging", "regression-on-multimodal-target", "mse-collapse", "mean-prediction"]),
    ("metric-gaming",
     "A trivial or degenerate policy attains the metric's optimum (a metronome maximises "
     "beat alignment; persistence maximises level R2; jitter aligns to beats). The metric "
     "then cannot separate a working system from a trivial one. Guard: report the trivial "
     "policy's score alongside, use the metric as a constraint, carry the claim on a "
     "distributional or human metric.",
     ["metronome-baseline", "bas-saturation", "degenerate-optimum", "goodhart"]),
]

# --------------------------------------------------------------------------- papers
PAPERS = [
    ("Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task? (Skip-BART)",
     "arXiv:2506.01482",
     ["First end-to-end audio->light generation trained on real lighting-engineer data (RPMC-L2, 699 songs, HSV histograms per frame, CC-BY-4.0).",
      "BART over OpenL3 -> per-frame hue (180 bins) + value; saturation fixed 100%; PianoBART transfer; RSTC sampling.",
      "Human eval: p<0.001 vs rule-based, p=0.72 vs human engineers. Offline, single primary light, no kinematics."],
     ["audio-to-light", "generative-modelling"]),
    ("Stage Light is Sequence^2: Multi-Light Control via Imitation Learning (SeqLight)",
     "arXiv:2605.03660",
     ["Skip-BART adapted to predict the FULL hue/value distribution (KL), then a goal-conditioned MDP decomposes it into per-light hue/value via BC -> AIRL -> GRPO with HER-style goals from mixed light.",
      "Decomposition is venue-specific and trained WITHOUT engineer demonstrations; last-frame-constrained sampling for smoothness.",
      "Still hue/value only: no pan/tilt/zoom, no motion. Preference +16.4% in-domain, +13.5% out-of-domain."],
     ["audio-to-light", "generative-modelling"]),
    ("EDGE: Editable Dance Generation From Music",
     "arXiv:2211.10658",
     ["Transformer diffusion on SMPL conditioned on Jukebox features; editing via inpainting/joint conditioning.",
      "Reference open-weight music->dance generator; Jukebox features are heavy for a 12 GB GPU."],
     ["music-to-dance", "generative-modelling"]),
    ("FlowerDance: MeanFlow for Efficient and Refined 3D Dance Generation",
     "arXiv:2511.21029",
     ["MeanFlow + BiMamba + channel-level fusion; physical-consistency constraint required for convergence.",
      "FineDance: BAS 0.232 vs GT 0.201, FID_k 29.7, FSR 0.147 (GT 0.216); AIST++: FID_k 20.5, BAS 0.227.",
      "2008 FPS for a 1024-frame clip; 10 ODE steps near-SOTA, 5 steps jittery; authors note BAS is gameable by jitter."],
     ["music-to-dance", "generative-modelling", "evaluation-metric"]),
    ("InfiniteDance: Scalable 3D Dance Generation Towards in-the-wild Generalization",
     "arXiv:2603.13375",
     ["100.69 h music-dance pairs, 30 genres, SMPL-X, pseudo-GT from video repaired by a Foot Restoration Diffusion Model.",
      "ChoreoLLaMA: LLaMA-3.2-1B over MuQ music tokens (1024-d @30 fps) + RVQ-VAE dance tokens, RAG reference dance, cadence-MoE; weights (2.15 GB) and data on HF huuuuuuuuu/InfiniteDance, non-commercial academic.",
      "Documents that AIST++/FineDance-trained generators collapse on unseen in-the-wild music."],
     ["music-to-dance", "generative-modelling"]),
    ("DiscoForcing: Real-Time Audio-Driven Character Control with Diffusion Forcing",
     "arXiv:2605.28491",
     ["Strictly causal, bounded-latency streaming dance from live audio: dilated causal conv -> RVQ rhythm codes + periodic-phase (FFT) features from a sliding buffer.",
      "272-d root-relative incremental motion -> motion VAE latent -> diffusion forcing with hybrid noise schedules, FIFO denoising window, temporal guidance against stale history.",
      "Deployed to a Unity avatar and a Unitree G1 via ROS2; zero-lookahead baselines lose beat synchronisation."],
     ["music-to-dance", "real-time", "generative-modelling"]),
    ("MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space",
     "arXiv:2503.15451",
     ["Continuous causal latent space + probabilistic AR model for streaming motion; avoids discretisation loss and error accumulation of GPT-style tokens."],
     ["real-time", "generative-modelling"]),
    ("MotionBeat: Motion-Aligned Music Representation via Embodied Contrastive Learning",
     "arXiv:2510.13244",
     ["Beat-synchronous audio/motion tokens, embodied contrastive loss with tempo-aware and beat-jitter negatives, Soft-DTW/EMD rhythm alignment, bar-equivariant phase rotations.",
      "Beats Jukebox/CLAP/wav2vec as the conditioning encoder for dance generation (BAS 0.25 vs 0.24); transfers to beat tracking and tagging."],
     ["music-to-dance", "cross-modal"]),
    ("DanceCamera3D: 3D Camera Movement Synthesis with Music and Dance",
     "arXiv:2403.13667",
     ["DCM dataset (108 seqs, 3.2 h) of paired dance + camera + music; transformer diffusion generates camera position/direction/FOV.",
      "Dance is the STRONG condition and music the WEAK one (separate CFG weights); body-attention loss keeps the dancer framed.",
      "Metrics: FID_k/Dist_k on velocity-acceleration features, shot-feature FID, dancer-missing rate, user study. A moving-head light is a camera run in reverse."],
     ["motion-to-light", "music-to-dance", "evaluation-metric"]),
    ("Music-to-Dance Generation via Atomic Movements",
     "arXiv:2607.13978",
     ["Choreography as a planned sequence of atomic movements (type, duration, timing) then a transition-aware completion stage; interpretable and editable.",
      "Analog for a cue-list-first conductor: plan looks/cues symbolically, then render continuous parameters."],
     ["music-to-dance", "generative-modelling"]),
    ("Tempo as the Stable Cue: Hierarchical Mixture of Tempo and Beat Experts for Music to 3D Dance Generation",
     "arXiv:2512.18804",
     ["Tempo (60-200 BPM) is the cue that stays consistent across datasets/genres; tempo-routed experts improve rhythm alignment without genre labels.",
      "Mitigation candidate for the club/EDM genre gap."],
     ["music-to-dance"]),
    ("EDMFormer: Genre-Specific Self-Supervised Learning for Music Structure Segmentation (EDM-98)",
     "arXiv:2603.08759",
     ["98 professionally annotated EDM tracks with build-up/drop/breakdown taxonomy; transformer over SSL embeddings improves drop/build-up boundaries.",
      "EDM-CUE (HF disco-eth/edm-cue) is metadata-only: 4,710 tracks, 21k cue points, no audio."],
     ["audio-to-light"]),
]

# --------------------------------------------------------------------------- diagnostics
DIAGNOSTICS = [
    ("light-bas", "light_beat_alignment_score", "score", "higher_better",
     "Beat Alignment Score computed on LIGHT kinematics: kinematic beats = local minima of "
     "fixture-parameter velocity (pan/tilt/zoom/dimmer envelope) matched to music beats with "
     "the dance-BAS Gaussian kernel. Always reported next to the metronome (upper) and "
     "circular-shift (lower) nulls; a metronome maximises it, so it is a constraint, not a headline."),
    ("fid-light-kinetic", "fid_light_kinetic", "fid", "lower_better",
     "Frechet distance between distributions of light kinetic features (velocity/acceleration "
     "of fixture parameters over 2.5 s windows) for a generated show vs the reference set of "
     "light mapped from REAL dance motion through the same conductor. Nulls: random-shift "
     "motion vs real (must be far larger), split-half real vs real (resolution floor)."),
    ("light-diversity-kinetic", "light_diversity_kinetic", "dist", "neutral",
     "Mean pairwise Euclidean distance of light kinetic features across generated shows "
     "(Div_k analog). Compared to the real-dance-mapped value; a collapse shows as a large drop "
     "that FID alone can hide."),
    ("edm-collapse-rate", "edm_collapse_rate", "fraction", "lower_better",
     "Fraction of frames in which the latent dancer's MDV speed falls below the 5th percentile "
     "of real dance MDV speed, on out-of-distribution (club/EDM) music. Detects the in-the-wild "
     "collapse InfiniteDance documents for AIST++/FineDance-trained generators."),
    ("human-preference-winrate", "human_pairwise_winrate", "fraction", "higher_better",
     "Pairwise blind preference 'which light dances better with the music' on identical "
     "renders (same rig, renderer, audio), >=20 raters, bootstrap 95% CI over tracks x raters. "
     "The only headline metric; every other diagnostic is a proxy."),
    ("audio-to-dmx-latency-ms", "audio_to_dmx_latency_ms", "ms", "lower_better",
     "Wall-clock from an audio sample entering the causal front-end to the DMX frame that "
     "reacts to it, measured on the deployment box with a click-track impulse; reported "
     "together with any deliberate lookahead/delay-line L so latency = pipeline + L."),
]

# --------------------------------------------------------------------------- problem
PROBLEM_DESC = """LightSem v2 (Anton, 2026-09-03). Founding idea: learn from paired (music, dance video) how a body answers music; at inference remove the body, keep the sound, and let the LIGHT move the way the dancer would (speed -> brightness, laterality -> which fixtures, spread -> beam width, verticality -> tilt, sharp accents -> strobe/colour snaps).

WHY THIS IS A NEW PROBLEM AND NOT A RETRY. The original problem [[audio-to-light-semantic-mapping-20260807]] tested the idea as REGRESSION to one dancer and refuted it: inter-dancer agreement r=0.024 at 0.1 s, held-out audio->KE R2 ~ 0, motion->sound->light pivot bound <= 0, audio->light LEVEL marginal -0.005 vs persistence. Those rule-outs are scoped to regression / dense supervision and stand. The GENERATIVE formulation was never trained end-to-end or scored on generation ('Path B: hallucinate a latent dancer' in the old problem) and is therefore untested, not refuted. The vault's own hygiene rule applies.

MECHANISM. p(motion | music) is multi-modal, so the MSE-optimal predictor is the conditional mean (problem tag conditional-mean-collapse). The music-to-dance literature sidesteps this by SAMPLING a dance and evaluating distributions: FlowerDance (arXiv 2511.21029) reaches BAS 0.232 vs ground truth 0.201 on FineDance; InfiniteDance (2603.13375) scales to 100.69 h / 30 genres; DiscoForcing (2605.28491) does it strictly causally in real time. Cross-modal transfer moves only redundant information (MAS.S60 lecture 08): the SHARED part of audio/motion (tempo, phase, phrase, energy envelope, timbral section changes) is predicted; the motion-UNIQUE part (which limb, direction, size) is sampled. Evaluation must be distributional and priced against a metronome (problem tag metric-gaming).

ARCHITECTURE (see /root/lightsem/PLAN.md section 4). A0 causal/offline audio front-end. HEAD A latent dancer: offline music->dance generator (InfiniteDance/ChoreoLLaMA, FlowerDance, EDGE) -> ~24-d Motion Descriptor Vector (MDV: speed, jerk, spread, verticality, laterality, travel, symmetry, contacts, Laban Weight/Time/Space/Flow proxies); online causal student audio->MDV trained with diffusion forcing on ~120 h of licensed 3D dance (AIST++, FineDance, InfiniteDance), dancer present at train time only. HEAD B operator head on RPMC-L2 (699 songs): timbre/spectral-driven scene-change probability (the validated z=+2.8..5 signal) + hue/value distribution (KL, Skip-BART/SeqLight style). HEAD C deterministic beat-locked micro layer. CONDUCTOR: v0 rule table MDV->fixtures with multiplicative gates (strobe = jerk x hit/drop), slew limits; v1 learned (SeqLight-style decomposition, preference fine-tuning), audited against its nearest additive projection. Optional corps de ballet: K latent dancers -> fixture groups. Output DMX 512ch@40fps -> Art-Net / moderngl renderer / BlenderDMX.

STAGES (runbooks rb-ls2-00..06): 0 oracle real-dance->light + Light-BAS/FID_light metrics (kills the pivot in a day if real dance does not read as dancing); 1 offline latent dancer vs real-mapped reference incl. EDM collapse rate; 2 operator head at 699 songs; 3 conductor v0 + blind human A/B vs beat-trigger, Skip-BART/SeqLight, real-dance upper reference; 4 causal student, lookahead sweep, audio->DMX latency < 100 ms, Art-Net on hardware; 5 learned conductor / group of dancers; 6 EDM anticipation only if EDM lighting ground truth is obtained.

DATA/INFRA. Bulk data on /mnt/lightsem-data (325 GB free; root disk 89% full): AIST++ SMPL motions.zip downloaded 2026-09-03; full RPMC-L2 42.5 GB, InfiniteDance 15-27 GB, FineDance 7.7 h public, MuQ/MERT/ChoreoLLaMA checkpoints planned. No YouTube scraping (closed dead end). InfiniteDance is non-commercial academic."""

SYNTHESIS = """STATE OF KNOWLEDGE 2026-09-03 (no experiments run yet under this problem).

What is established (inherited from audio-to-light-semantic-mapping-20260807): regression from audio to one dancer's motion at beat resolution is ill-posed (inter-dancer r 0.024 @0.1 s, 0.13 @2 s, 0.50 @8 s); audio->light LEVEL is owned by persistence (R2_within +0.708, audio marginal -0.005); audio->light EVENT is real and timbre-driven (marginal AP lift +0.09, z +2.83; MFCC z +4.98, spectral z +4.21, energy z +0.66) but reactive (lift decays ~99% by 4 s); the sound-pivot chain motion->sound->light is bounded <= 0; AIST audio has no build-up/drop structure; a metronome-priced null is mandatory for any timing metric.

What is the live claim: the founding idea is well-posed as GENERATION. Sampling a dancer conditioned on music reproduces real dancers' beat alignment (FlowerDance BAS 0.232 vs GT 0.201) and is deployable causally (DiscoForcing). The light inherits kinematics from a sampled dancer (unique part) and timing/palette/cues from audio (shared part, plus RPMC-L2 operator data). Nobody has published a learned KINEMATIC layer for stage lighting (Skip-BART/SeqLight are hue/value only); the closest analog is music->camera (DanceCamera3D), where dance is the strong condition and music the weak one.

Decisive next measurement: Stage 0 — map REAL AIST++ SMPL motion to light through the conductor v0 rule table, render, compute Light-BAS against metronome/circular-shift nulls and FID_light split-half resolution, and look at it. If real dance mapped to light does not read as dancing, the pivot is dead before any ML; if it does, Stage 1 compares generated dancers to that reference distributionally (collapse rate on EDM is the risk to watch).

Biggest risks: (1) the mapping is the art and may not read; (2) generators collapse on club/EDM music; (3) causality costs beat sync; (4) metric gaming (BAS/FID-on-levels); (5) InfiniteDance licence is non-commercial.

Do not: regress KE; regress absolute levels; claim anticipation on RPMC-L2; scrape YouTube; report BAS without the metronome value beside it."""

# --------------------------------------------------------------------------- runbooks
RUNBOOKS = [
    ("rb-ls2-infra", 0,
     """INFRA for LightSem v2 (context: /root/lightsem/PLAN.md section 7; root disk 89% full, /mnt has 325 GB).
1. Create a dedicated torch venv on /mnt: `UV_CACHE_DIR=/mnt/lightsem-data/uv-cache uv venv /mnt/lightsem-data/venv-torch --python 3.12` then install torch (cu130, sm_120 for the RTX 5070), numpy, librosa, h5py, smplx, huggingface_hub. Stop borrowing /root/optiHealth-EiV/.venv via run_gpu.sh (works today; couples two separate projects). Add a `run_torch.sh` wrapper (never invoke the venv python binary directly from the shell tool: 'embedded null byte' pitfall).
2. Download to /mnt/lightsem-data/: full RPMC-L2 (HF RS2002/RPMC-L2, data/RPMC_L2_parts/ 4 parts ~42.5 GB; CC-BY-4.0); InfiniteDance data+weights (HF huuuuuuuuu/InfiniteDance, 15-27 GB + 2.15 GB checkpoint; non-commercial academic); FineDance public 7.7 h (Google Drive, manual); MuQ and MERT-v1-95M checkpoints. AIST++ motions.zip (SMPL) is already at /mnt/lightsem-data/aistpp/.
3. Verify each download by checksum/size and by loading one sample; record paths in /mnt/lightsem-data/README.md.
Gate: one SMPL sequence from motions.zip forward-kinematics to joints that match the keypoints3d_optim file for the same sequence to < 5 cm mean error after alignment.""",
     ["torch venv on /mnt loads on GPU", "FK of motions.zip matches keypoints3d_optim within 5 cm"]),
    ("rb-ls2-00-oracle-dancer", 1,
     """STAGE 0 — ORACLE DANCER AND LIGHT METRICS (context: PLAN.md section 5, Stage 0). Question: does REAL dance, mapped to light, read as 'the light is dancing'? If not, the whole latent-dancer pivot dies here, before any ML.
Background you need: the vault ruled out regressing dancer kinetic energy as DMX supervision (inter-dancer r 0.024 @0.1 s) and audio->light levels (persistence R2 +0.708). This stage does not regress anything: it maps real motion deterministically to light and builds the metrics every later stage uses.
Build: (a) lightsem/motion/: SMPL forward kinematics from /mnt/lightsem-data/aistpp/motions.zip, Motion Descriptor Vector (MDV, ~24-d, root-relative, mirror-aware): speed (global, upper, lower), acceleration, jerk, spatial spread (bounding volume, limb extension), verticality (COM height, jump), laterality (L/R weight, lateral travel), travel direction and turn rate, symmetry, foot contacts, Laban proxies (Weight ~ |accel|, Time ~ jerk/velocity, Space ~ path curvature, Flow ~ velocity smoothness). Unit-test against the keypoints3d path. (b) lightsem/conductor/v0.py: rule table speed->dimmer, laterality->L/R fixture groups, lateral travel->pan sweep, verticality->tilt, spread->zoom, turn rate->pan chase, jerk AND hit->strobe/colour snap, Weight->saturation, Flow->crossfade time; per-fixture slew limits. (c) lightsem/eval/: Light-BAS (kinematic beats of light = local minima of fixture-parameter velocity / dimmer peaks vs music beats, dance-BAS kernel), FID_light-k/-g over 2.5 s windows of velocity/acceleration features, Div_light, plus metronome and circular-shift nulls and a split-half real-vs-real resolution floor.
Run: 5 AIST++ sequences (different genres) real motion -> light -> MP4 via the existing renderer, and the same 5 through the existing structure->scene rules.
PREREGISTER before running (propose_hypothesis with these gates): G1 Light-BAS(real-dance light) >= 0.9 x Light-BAS(metronome at the same rate); G2 FID_light(split-half real) x 3 < FID_light(circular-shift motion vs real); G3 blind look by Anton: >= 4/5 real-dance renders judged 'dancing' vs <= 2/5 for the rule-based ones. Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic.
If G3 fails, iterate the mapping (sparser channels, Laban smoothing, slew limits) and re-run; do not proceed to Stage 1 on a failed G3.""",
     ["G1 Light-BAS(real) >= 0.9 x metronome", "G2 split-half FID x3 < shift-null FID", "G3 >= 4/5 real-dance renders read as dancing"]),
    ("rb-ls2-01-latent-dancer-offline", 2,
     """STAGE 1 — OFFLINE LATENT DANCER (context: PLAN.md Stage 1; requires rb-ls2-00 gates passed). Question: does a SAMPLED dancer, mapped through the same conductor v0, land in the same light-feature distribution as REAL dancers, including on club/EDM music?
Setup: install InfiniteDance/ChoreoLLaMA (HF huuuuuuuuu/InfiniteDance, LLaMA-3.2-1B backbone, MuQ features) in the /mnt torch venv; fallbacks EDGE (Jukebox may exceed 12 GB; use cached features if provided) and FlowerDance when code is public. Generate for the 40 AIST++ test music clips and for 10 club/EDM tracks Anton owns (no scraping). Map to light with conductor v0. Reference distribution = light mapped from REAL AIST++ motion (Stage 0).
PREREGISTER gates: G1 Light-BAS(generated) >= 0.9 x Light-BAS(real-mapped), reported next to the metronome value; G2 FID_light(gen vs real) < 0.5 x FID_light(circular-shift vs real); G3 edm-collapse-rate < 0.10 on the 10 EDM tracks (fraction of frames with MDV speed below the real 5th percentile); G4 Div_light(gen) >= 0.7 x Div_light(real). Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic, edm-collapse-rate.
Interpretation rules: failure of G3 alone = genre gap (try tempo-routed experts arXiv 2512.18804, InfiniteDance RAG reference, conditioning on beat/phase features only) — not refutation of the pivot. Failure of G1/G2 on AIST++ itself = the generator is worse than the literature claims on our mapping; check the mapping's sensitivity before blaming the model.""",
     ["G1 BAS(gen) >= 0.9 x BAS(real)", "G2 FID(gen) < 0.5 x FID(shift-null)", "G3 EDM collapse rate < 0.10", "G4 Div(gen) >= 0.7 x Div(real)"]),
    ("rb-ls2-02-operator-head-699", 3,
     """STAGE 2 — OPERATOR HEAD AT FULL RPMC-L2 SCALE (context: PLAN.md Stage 2; independent of Stage 1, can run in parallel). Background: on 140 songs, audio->lighting-EVENT (scene change in next 1 s) had marginal AP lift +0.0909 over a rhythm-of-past-changes baseline, z=+2.83 vs circular-shift null; MFCC z=+4.98 and spectral z=+4.21 carry it, energy z=+0.66 does not; openl3 was beaten by 10 spectral descriptors; the effect is reactive (lift ~0 by 4 s). Audio->LEVEL is refuted (persistence R2 +0.708); do not regress levels.
Do: download full RPMC-L2 (699 songs, 42.5 GB) to /mnt/lightsem-data/rpmc_l2; extract per song the spectral + MFCC families, openl3 (control), the 6-d scene vector AND the raw hue (180-bin) / value histograms. Train a small causal model (causal conv/patched transformer, < 3 M params) with two heads: p(scene change in next 0.2/0.5/1.0 s) and per-frame hue/value distributions (KL, per-song centred, Skip-BART/SeqLight style). 5-fold CV by song; 40-rep circular-shift null per horizon; per-family ablation as before.
PREREGISTER gates: G1 marginal AP lift at 1.0 s >= +0.09 with z > 3 on 699 songs; G2 hue-distribution KL on held-out songs < per-song-mean baseline KL; G3 (guard) absolute marginal lift at 4 s reported — anticipation may only be claimed if it exceeds +0.05 with a CI excluding 0. Diagnostics: event-ap-lift-audio-marginal, event-ap-lift-vs-horizon (existing) plus a new hue-KL diagnostic to register at run time.
Output: a checkpoint the conductor consumes (cue probability + hue/value target) and the per-family attribution table.""",
     ["G1 marginal AP lift @1 s >= +0.09, z > 3, 699 songs", "G2 held-out hue KL < per-song-mean baseline", "G3 no anticipation claim unless lift@4 s > +0.05 with CI excluding 0"]),
    ("rb-ls2-03-conductor-ab", 4,
     """STAGE 3 — CONDUCTOR v0 END-TO-END + BLIND HUMAN A/B (context: PLAN.md Stage 3 and section 6; requires rb-ls2-01 and rb-ls2-02). This is the first result that may be CALLED 'dancing light'.
Build: fuse HEAD A (latent dancer -> MDV), HEAD B (cue probability, hue/value target, structure), HEAD C (beat-locked micro layer) through conductor v0 into DMX; arbitration: HEAD B changes the look and palette, HEAD A drives kinematics/intensity contour, HEAD C accents; multiplicative gate for strobe (jerk x hit/drop). Render 20 tracks (10 dance genres, 10 club/EDM) in four conditions with the SAME rig, renderer and audio: (i) HEAD C only (beat-trigger, SoundSwitch-like), (ii) hue/value-only reproduction of Skip-BART/SeqLight behaviour (no kinematics), (iii) LightSem v2 (A+B+C), (iv) light mapped from REAL dance for the dance tracks (upper reference; not available for EDM).
Study: >= 20 raters, pairwise blind 'which light dances better with the music', order and side randomised, bootstrap 95% CI over tracks x raters. Diagnostic: human-preference-winrate.
PREREGISTER gates: G1 v2 vs (i) win rate > 0.60 with CI excluding 0.5; G2 v2 vs (ii) > 0.50 with CI excluding 0.5; G3 v2 vs (iv) not significantly below 0.40. Also report Light-BAS and FID_light for every condition next to the metronome null so the proxies can be calibrated against the human result.
Failure modes to name in the close: if (i) wins, kinematics hurt (too busy) -> sparsify mapping; if (ii) ties, kinematics add nothing on that renderer -> check haze/beam visibility before concluding.""",
     ["G1 win rate vs beat-trigger > 0.60, CI excludes 0.5", "G2 win rate vs hue/value-only > 0.50, CI excludes 0.5", "G3 not significantly below 0.40 vs real-dance reference"]),
    ("rb-ls2-04-causal-student-latency", 5,
     """STAGE 4 — CAUSAL STUDENT AND LIVE LATENCY (context: PLAN.md Stage 4 and DiscoForcing arXiv 2605.28491; requires rb-ls2-01). Goal: the dancer becomes a TRAIN-TIME-ONLY modality (Anton's phrasing: train on motion, remove it at inference) and the system runs strictly causally under 100 ms.
Data: MDV corpus from AIST++ (5.2 h), FineDance public (7.7 h), InfiniteDance (100.7 h), all licensed research data; split by song and dancer.
Model: causal audio front-end (online beat tracker BeatNet/zero-latency PLP; sliding buffer -> dilated causal conv -> RVQ rhythm codes + periodic phase features; causal spectral/MFCC) -> small (~5 M) generative model over MDV sequences trained with diffusion forcing / flow matching, FIFO denoising window at inference; optional distillation from the offline teacher on unlabeled music (MotionStream-style self-forcing). Baselines that MUST be run: (a) causal AR REGRESSION to MDV (expected to collapse — this documents that the generative objective is doing the work; register its Div_light), (b) offline teacher + MDV (ceiling).
Sweep lookahead L in {0, 100, 250, 500} ms implemented as an audio delay line. Measure audio-in -> DMX-out latency with a click-track impulse on this box. Send Art-Net to a real node or BlenderDMX and verify the wire format (currently untested).
PREREGISTER gates: G1 student Light-BAS >= 0.9 x teacher at the chosen L; G2 FID_light(student) <= 1.5 x FID_light(teacher); G3 audio-to-dmx-latency-ms < 100 + L; G4 regression baseline Div_light < 0.5 x real (collapse documented); G5 Art-Net frames decoded correctly by an independent receiver. Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic, audio-to-dmx-latency-ms.""",
     ["G1 student BAS >= 0.9 x teacher", "G2 student FID <= 1.5 x teacher", "G3 latency < 100 ms + L", "G4 regression baseline collapses (Div < 0.5 x real)", "G5 Art-Net verified on hardware/BlenderDMX"]),
    ("rb-ls2-05-learned-conductor", 6,
     """STAGE 5 — LEARNED CONDUCTOR AND CORPS DE BALLET (context: PLAN.md Stage 5; optional research stage after rb-ls2-03 passes). Two ideas, ONE hypothesis each (never bundle): (A) SeqLight-style goal-conditioned decomposition of HEAD B's hue/value distribution across the rig's fixtures plus a small policy (MDV, cue, structure, beat phase) -> parameters, trained by imitation of conductor v0 + real-dance-mapped light and fine-tuned on Stage 3/4 preference pairs (GRPO/DPO). Audit every learned fusion term by projecting to the nearest additive model (Hessel & Lee); only credit multiplicative terms that survive. (B) K latent dancers (different seeds/genres or a group-choreography model, arXiv 2603.22316 / 2506.18671) assigned to fixture groups (left truss, centre, right truss, floor).
PREREGISTER gates per hypothesis: G1 beats conductor v0 in a fresh blind A/B with win rate CI excluding 0.5 (human-preference-winrate); G2 for (A) the additive projection loses >= 2 points of win rate, else ship the additive model; G3 for (B) Div_light rises >= 20% without FID_light worsening > 20%.""",
     ["G1 beats conductor v0 in blind A/B, CI excludes 0.5", "G2 additive projection loses >= 2 pts else ship additive", "G3 group version: Div +20% with FID not worse than +20%"]),
    ("rb-ls2-06-edm-anticipation", 7,
     """STAGE 6 — EDM ANTICIPATION, CONDITIONAL ON DATA (context: PLAN.md Stage 6). The claim 'light that LEADS the build-up by 4-8 s' is refuted on RPMC-L2 (rock/punk/metal; lift ~0 by 4 s) and validated only on synthetic EDM (precision 1.0, lead 2.78 +- 0.08 s, z +2.4 vs random). It remains UNTESTED on real EDM lighting ground truth, and no such corpus exists publicly (EDM-98 has structure labels only; EDM-CUE is metadata only, no audio).
Preconditions: obtain either (a) a small EDM show corpus with a lighting operator (grandMA/Chamsys show files or DMX capture + audio, >= 20 tracks), or (b) at minimum EDM-98 audio + annotations to validate the structure labeller. Without (a), do NOT reopen the anticipation hypothesis; the tension/time-to-drop layer ships as a feature labelled 'heuristic, validated on synthetic EDM'.
If (a) exists: PREREGISTER absolute marginal lift at horizons 2/4/8 s over the rhythm baseline with circular-shift nulls; gate = lift at 4 s > +0.05 with CI excluding 0 (effect size, never z alone — the horizon-sweep script's z-based verdict was wrong once already).""",
     ["precondition: real EDM lighting ground truth (>= 20 tracks) or EDM-98 audio", "gate: absolute lift @4 s > +0.05 with CI excluding 0"]),
]


def main() -> int:
    other_vault = "/root/vibescience-vault"
    before_other = snapshot(other_vault)
    before_ours = snapshot("/root/lightsem-vault")
    c = Client()
    info = c.initialize()
    assert info["name"] == "vibescience-lightsem", info
    print("connected:", info)

    print("\n== tags")
    for tid, desc, aliases in TOPIC_TAGS:
        call(c, "register_tag", {"id": tid, "axis": "topic", "description": desc, "aliases": aliases})
    for tid, desc, aliases in PROBLEM_TAGS:
        call(c, "register_tag", {"id": tid, "axis": "problem", "description": desc, "aliases": aliases})

    print("\n== diagnostics")
    for did, name, unit, direction, desc in DIAGNOSTICS:
        call(c, "register_diagnostic", {"id": did, "name": name, "unit": unit, "direction": direction,
                                        "description": desc, "topic_tags": ["evaluation-metric"]})

    print("\n== papers")
    paper_ids = []
    for title, arx, claims, tags in PAPERS:
        out = call(c, "add_paper", {"title": title, "arxiv_id_or_url": arx, "key_claims": claims, "topic_tags": tags})
        pid = out.get("paper_id") or out.get("id") if isinstance(out, dict) else None
        if pid:
            paper_ids.append(pid)
    print("  paper ids:", paper_ids)

    print("\n== problem")
    call(c, "create_problem", {
        "id": NEW_PROBLEM,
        "title": "Dancing light: audio -> sampled latent dancer -> DMX as a GENERATIVE task (LightSem v2)",
        "description": PROBLEM_DESC,
        "topic_tags": ["audio-to-light", "music-to-dance", "motion-to-light", "generative-modelling",
                       "real-time", "evaluation-metric", "cross-modal", "pose"],
        "problem_tags": ["conditional-mean-collapse", "metric-gaming", "ill-posed-ground-truth", "matched-baseline"],
        "paper_refs": [p[1] for p in PAPERS],
    })
    for pid in paper_ids:
        call(c, "link_paper", {"entity_id": NEW_PROBLEM, "paper_id": pid})

    print("\n== runbooks")
    for rid, prio, prompt, gates in RUNBOOKS:
        call(c, "register_runbook", {"id": rid, "problem_id": NEW_PROBLEM, "prompt": prompt,
                                     "priority": prio, "gates": gates,
                                     "topic_tags": ["audio-to-light", "music-to-dance"]})

    print("\n== synthesis + old-problem update")
    call(c, "synthesize_problem", {"problem_id": NEW_PROBLEM, "synthesis": SYNTHESIS})
    call(c, "update_problem", {"id": OLD_PROBLEM, "add_update": (
        "2026-09-03: Path B ('hallucinate a latent dancer at inference') was never trained end-to-end or "
        "scored on generation, so it is UNTESTED, not refuted. It is now the main line of LightSem v2 under "
        f"problem [[{NEW_PROBLEM}]] with a generative objective (sampling a dancer, distributional evaluation "
        "priced against a metronome) — see /root/lightsem/PLAN.md. The rule-outs here remain binding for "
        "REGRESSION / dense supervision / sound-pivot formulations and for absolute light levels.")})

    c.close()
    after_other = snapshot(other_vault)
    after_ours = snapshot("/root/lightsem-vault")
    print(f"\nlightsem-vault md files: {before_ours} -> {after_ours}")
    print(f"optiHealth vault md files: {before_other} -> {after_other}  (must be unchanged)")
    assert before_other == after_other, "WROTE INTO THE WRONG VAULT"
    return 0


if __name__ == "__main__":
    sys.exit(main())
