#!/usr/bin/env python3
"""Rev. 3 (2026-09-04) of the LightSem v2 plan in the LIGHTSEM vault.

Integrates the Perplexity reasoning passes run with full project context on
2026-09-04 (audio->light design; music-induced-movement science; causal model
and latency; the dataset audit pass degenerated and was replaced by direct
source checks) and Anton's note that off-the-shelf pose models can extract the
motion data from video. Additive: updates the problem, synthesis and runbook
prompts; adds papers, one tag, one diagnostic and two runbooks; reprioritises.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client  # noqa: E402

P = "dancing-light-generative-20260903"


def snapshot(path):
    return sum(len([f for f in fs if f.endswith(".md")]) for _, _, fs in os.walk(path))


def call(c, name, args):
    out = c.call(name, args)
    short = json.dumps(out, ensure_ascii=False)[:160] if not isinstance(out, str) else out[:160]
    key = args.get("id") or args.get("runbook_id") or args.get("title") or args.get("problem_id") or ""
    print(f"  {name:20s} {str(key)[:44]:44s} -> {short}")
    return out


DESC = """LightSem v2, rev. 3 (2026-09-04). TASK: a causal SOUND -> LIGHT model for concerts and festivals, TRAINED ON DANCE. Paired (music, dance motion) -> motion descriptors -> light targets on a canonical rig (fixed mapping) -> train audio -> light directly. At inference: sound in, DMX out; no dancer, pose or dance generator. Motion comes from mocap datasets OR is extracted from video with existing pose models (YOLO-pose / RTMPose / ViTPose 2D; GVHMR / SMPLest-X 3D) — the dancer is a training-time-only modality either way (Anton, 2026-09-04).

WHAT AUDIO DETERMINES IN A DANCING BODY (music-induced-movement science, integrated 2026-09-04). Burger et al. 2013 (N=60, 30 clips; doi:10.3389/fpsyg.2013.00183): pulse clarity -> CoM speed / shoulder wiggle / quantity of motion / hips / feet r=0.67/0.67/0.62/0.58/0.55; low-frequency flux (kick/bass) -> head speed r=0.73; high-frequency flux -> hands r=0.65 (large cultural residual); percussiveness -> head/shoulders 0.67/0.61; tempo n.s. for amplitude. Burger 2014 (doi:10.3389/fnhum.2014.00903) and Toiviainen 2010 (doi:10.1525/mp.2010.28.1.59): vertical bounce locks to the tactus, horizontal sway/torso to 2-beat and bar; faster tempo shifts energy up the metric hierarchy (bar-level r up to 0.86); PERIOD is a group fact, PHASE is a person fact. Van Dyck 2013 / Cameron 2022: bass raises quantity of motion, hips, torso. Solberg & Jensenius 2017 (club EDM): drops surge QoM and vertical head motion, breakdowns collapse it. Bigand 2024: music drives head bobs and vertical bounce; the partner drives hands and lateral motion. Reading of our numbers: r=0.50 at 8 s IS the section-energy channel; r=0.024 at 0.1 s means instantaneous kinetic-energy AMPLITUDE is music-independent (phase/style scatter). A generative objective escapes the conditional-mean collapse for diversity, but cannot invent a coupling absent from p(y|audio): a flow model of 10 Hz KE is a dance prior with a music watermark.

TARGETS, THREE BLOCKS, THREE OBJECTIVES. Block A shared skeleton (deterministic, MSE/L1 + periodicity head): quantity of motion; CoM/head/foot speed; vertical bounce amplitude and period at the tactus; embodied metrical level; Laban Weight (bass, pulse clarity, RMS, drop contrast) and Time (percussiveness); footfall rate near the beat; section energy contrast. Block B events (point process, AP/F1 at 0.2 and 1 s): footfalls, jumps, hits from dance; scene changes and palette shifts from RPMC-L2. Block C style residual (sampled; conditional rectified flow on the residual after subtracting A): hands, laterality sign and phase, travel direction, symmetry, spread residual, Laban Space/Flow, jerk series; conditioned on a genre token; sampled once per 'dancer' or dropped. Noise (no loss): exact trajectories, gesture identity, microtiming. Never average 30 dancers into one vector; average only shared scalars per bar.

RIG MAPPING FOLLOWS THE BODY: QoM/Weight -> master intensity, haze, saturation; vertical bounce at tactus -> dimmer/tilt bounce on the beat; mediolateral sway at 2-beat/bar -> pan sweep period (direction sampled); head/torso <- LF flux -> big slow fixtures; hands <- HF flux -> small fast fixtures (LED bars, strobes); footfalls/jumps/hits -> strobe bursts, tilt snaps; scene change/palette (RPMC-L2) -> look/hue; spread residual -> zoom; Laban Time -> crossfade time. Canonical rig; rig adaptor to venue patch.

MODEL AND DEPLOYMENT: causal Conformer trunk (6-8 blocks, d 256-384, 15-40 M params, 40 fps, KV cache 3-4 s; Alexanderson 2023 arXiv 2211.09707) with heads A (regression), B (hazard), C (CFM 1-4 Euler steps; MotionStreamer 2503.15451: MSE head FID 59.2 vs diffusion head 10.7 on the same trunk). No VQ on a 24-d bus. Conditioning: causal log-mel/MFCC + RMS + LF/HF flux + onset + online beat phase (PLP zero-latency TISMIR 2024 11.6 ms; BeatNet ~20 ms); never a bidirectional Jukebox/MERT/MuQ at inference. TWO-PATH DEPLOYMENT: ITU-R BT.1359-1 detectability is +45 ms when sound leads light (flash-tone JND 20-70 ms), so transients (onset/beat -> strobe/hit) run on a 10-20 ms DSP path and the learned model paints the slower channels (budget 12+8+5+20-40 ms ~ 45-65 ms); a 100-250 ms PA delay line is viable in the house, never in monitors. Causality verified by future-perturbation and gradient probes; latency by click-track + photodiode, p50/p99.

DATA (research licences; no YouTube scraping): AIST++ (annotations CC-BY-4.0) + AIST multi-view video (research DB, reachable) for validating the video->pose path against mocap; FineDance 7.7 h public (research); InfiniteDance 100.7 h (non-commercial academic); OpenDanceSet 100 h (poses + URLs); PopDanceSet, Let's Dance, BRACE for the 2D path; Oslo club-dancing mocap to DJ-mixed EDM (Solberg & Jensenius, Zenodo 21945706 / 21810782 / 21945721, CC-BY-4.0, head + wrist markers, music withheld but break/build/drop timing given) — block A on concert music; Oslo standstill set (Zenodo 21964686, CC-BY-4.0); RPMC-L2 699 songs (CC-BY-4.0) for HEAD B; own event recordings (consent) as the only commercial-clean path. 2D pose normalised by torso length supplies speed, planar laterality, spread, hands-vs-head split and periodicity; not jumps, contacts or Weight.

STAGES (runbooks): 0a descriptors + targets + metrics + oracle renders; 0b PREDICTABILITY AUDIT (per descriptor, per timescale: inter-dancer r, audio R2 vs a dancer-identity baseline, period locking) which assigns channels to blocks A/B/C/noise; 0c late-lights A/B (0/40/80/120 ms) and the category-error test (oracle dance-derived DMX vs a 30-line beat-trigger, brightness-matched); 1 train the three-block model on AIST++ + FineDance with splits by track, genre AND dancer; 2 scale data via the video->descriptor pipeline + Oslo EDM mocap, and HEAD B on 699 RPMC-L2 songs; 3 conductor + blind brightness-matched human A/B with two questions ('on the music?', 'the lighting engineer?'); 4 live two-path deployment, lookahead sweep, latency and causality tests, Art-Net on hardware; 5 learned conductor; 6 EDM anticipation only with real EDM lighting ground truth."""

SYNTHESIS = """STATE OF KNOWLEDGE 2026-09-04 rev. 3 (no experiments run yet under this problem).

TASK: causal sound -> light for concerts/festivals, trained on dance (mocap or video-extracted pose); no dancer at inference.

Established (inherited): regression audio -> one dancer's motion at beat resolution is ill-posed (inter-dancer r 0.024 @0.1 s); audio -> light LEVEL is owned by persistence; audio -> light EVENT is real and timbre-driven (AP lift +0.09, z +2.83) but reactive; sound-pivot chain <= 0; AIST audio lacks EDM structure; timing metrics need a metronome null.

New (literature, 2026-09-04): audio determines a shared energy/periodicity skeleton of dance — quantity of motion and CoM/feet/hips speed from pulse clarity (r ~0.55-0.67), head speed from low-frequency flux (r 0.73), hands from high-frequency flux (r 0.65, large residual), vertical bounce period-locked to the tactus and horizontal sway to the bar, energy surge at drops — plus a distribution over style (phase, side, hands, path). Period is shared, phase is personal. Therefore: predict the skeleton (block A, MSE), model events as a point process (block B), sample the style residual (block C, flow), and put nothing else in the loss. A flow model of 10 Hz kinetic energy would be a dance prior with a music watermark, not a music->light model.

Latency (ITU-R BT.1359-1: +45 ms detectability when sound leads; flash-tone JND 20-70 ms) forces a two-path system: transients on a 10-20 ms DSP path, learned model on the slow channels.

Decisive next measurements, in order: 0a build descriptors/targets/metrics and render real dance -> light; 0b the predictability audit that assigns every descriptor to A/B/C/noise (gate: QoM, vertical bounce, head speed and metrical level beat a dancer-identity baseline with audio R2 > 0.3 at 2-8 s, else only events survive from dance); 0c late-lights A/B and the category-error test (oracle dance-derived DMX vs a 30-line beat-trigger, brightness-matched — if the beat-trigger wins, block C leaves the product).

Risks: category error; sampling a prior; late lights; studio-dance music vs concert music (mitigation: Oslo EDM mocap + video->pose on concert footage); metric gaming; licences (InfiniteDance/OpenDanceSet research-only; Oslo CC-BY-4.0 but music separate).

Do not: regress 10 Hz kinetic energy; regress absolute levels; claim anticipation on RPMC-L2; scrape YouTube; put a dancer/pose/dance generator in the inference path; report BAS without the metronome value; use bidirectional audio encoders in the live path."""

PAPERS = [
    ("Burger, Thompson, Luck, Saarikallio, Toiviainen 2013 — Influences of rhythm- and timbre-related musical features on characteristics of music-induced movement",
     "https://doi.org/10.3389/fpsyg.2013.00183",
     ["N=60 free dancers, 30 pop clips; stimulus-level correlations: pulse clarity -> CoM speed 0.67, shoulder wiggle 0.67, QoM 0.62, hips 0.58, feet 0.55; LF flux 50-100 Hz -> head speed 0.73; HF flux -> hands 0.65; percussiveness -> head 0.67; tempo n.s. for amplitude.",
      "Defines block A of the LightSem target: the channels audio determines across people."],
     ["music-induced-movement", "pose"]),
    ("Burger et al. 2014 — Hunting for the beat in the body: period and phase locking in music-induced movement",
     "https://doi.org/10.3389/fnhum.2014.00903",
     ["Vertical (SI) movement period-locks to the tactus; horizontal (ML/AP) to tactus and bar; LF flux drives tactus and bar locking (ML 0.63); faster tempo shifts energy up the metric hierarchy (bar-level ML r 0.86).",
      "Period is shared across people, phase is not — phase must be sampled, not regressed."],
     ["music-induced-movement", "pose"]),
    ("Toiviainen, Luck, Thompson 2010 — Embodied meter: hierarchical eigenmodes in music-induced movement",
     "https://doi.org/10.1525/mp.2010.28.1.59",
     ["Eigenmovements of free dance: mode 1 vertical bounce/knee flexion at the beat; higher modes mediolateral sway and torso rotation at 2- and 4-beat levels; core body encodes the meter.",
      "Rig mapping: vertical channels bounce at the beat, horizontal sweeps run at half-bar/bar."],
     ["music-induced-movement", "pose"]),
    ("Van Dyck et al. 2013 — The impact of the bass drum on human dance movement",
     "https://doi.org/10.1525/mp.2013.30.4.349",
     ["Louder bass drum increases spontaneous dance movement, especially hips/torso: bass is a quantity-of-motion gain, not a pose."],
     ["music-induced-movement"]),
    ("Solberg & Jensenius 2017 — Group behaviour and interpersonal synchronization to electronic dance music (club dancers, breaks and drops)",
     "https://doi.org/10.1177/1029864917713785",
     ["Club dancers to DJ-mixed EDM: breakdown collapses quantity of motion, drop surges QoM and vertical head motion — a group-level structural event. Primary data on Zenodo 21945706/21810782/21945721 (CC-BY-4.0; head/wrist markers; music withheld).",
      "The 2016 group-QoM series was partly confounded by marker visibility; trust the 2017 head-marker + wrist-accelerometer design."],
     ["music-induced-movement", "audio-to-light"]),
    ("Witek et al. 2014 — Syncopation, body-movement and pleasure in groove music",
     "https://doi.org/10.1371/journal.pone.0094446",
     ["Medium syncopation maximises the urge to move (quadratic R2 0.35) and pleasure (0.43); groove is an inverted-U gain on the shared movement channels, not a new kinematic axis."],
     ["music-induced-movement"]),
    ("Bigand et al. 2024 — The geometry of interpersonal synchrony in human dance",
     "https://doi.org/10.1016/j.cub.2024.05.055",
     ["70 lay dyads, 15 principal dance moves = 95% of kinematics; music drives anteroposterior head bobs, the partner drives hands and full-body lateral motion; only vertical bounce is hybrid and beat-locked."],
     ["music-induced-movement", "pose"]),
    ("Alexanderson, Nagy, Beskow, Henter 2023 — Listen, Denoise, Action! Audio-driven motion synthesis with diffusion models",
     "arXiv:2211.09707",
     ["Speech/music -> motion is one-to-many; MSE gives underarticulated averaged output; DiffWave-Conformer diffusion is the rebuttal. Conformer replaces dilated convs (local velocity + long attention).",
      "Backbone reference for the LightSem trunk."],
     ["generative-modelling", "real-time"]),
    ("Chen et al. 2024 — Diffusion Forcing: next-token prediction meets full-sequence diffusion",
     "arXiv:2407.01392",
     ["Per-token independent noise levels; causal sequence diffusion usable as AR or full-sequence; the training recipe DiscoForcing builds on."],
     ["generative-modelling", "real-time"]),
    ("Real-time beat tracking with zero latency and PLP (TISMIR 2024)",
     "https://doi.org/10.5334/tismir.189",
     ["11.6 ms frame latency at 512/44.1k, F1 74.7%; 1-frame lookahead gives 0 ms claimed system latency; kernel uses ~6 s of past; future is extrapolated. Beat phase feature for the live path."],
     ["real-time", "audio-to-light"]),
    ("ITU-R BT.1359-1 — Relative timing of sound and vision for broadcasting",
     "https://www.itu.int/rec/R-REC-BT.1359",
     ["Detectability thresholds: sound leading picture +45 ms, sound lagging -125 ms; acceptability +90 / -185 ms. Light lagging sound = sound leading = the harsh side.",
      "Basis for the two-path deployment: transients on a 10-20 ms DSP path."],
     ["real-time", "evaluation-metric"]),
    ("Castro et al. 2018 — Let's Dance: learning from online dance videos",
     "arXiv:1801.07388",
     ["~1000 clips, 10 genres, pose tracks released for research; honest about motion-vs-appearance. Candidate for the 2D video->descriptor path."],
     ["pose", "music-to-dance"]),
    ("Moltisanti et al. 2022 — BRACE: the breakdancing competition dataset for dance motion synthesis",
     "arXiv:2207.10120",
     ["Pose from competition video (Red Bull BC One); licence tied to competition footage; useful for the 2D descriptor path, research only."],
     ["pose", "music-to-dance"]),
    ("OpenDance: multimodal controllable 3D dance generation with large-scale Internet data (OpenDanceSet)",
     "arXiv:2506.07565",
     ["100.26 h, 14 genres, 147 subjects, 3D motion + 2D keypoints + trajectories + text; derived poses and source URLs are released, not pixels; ToS falls on the user."],
     ["music-to-dance", "pose"]),
]

DIAG = ("descriptor-audio-r2-over-identity", "descriptor_audio_r2_over_dancer_identity", "R2 delta", "higher_better",
        "Per motion descriptor and per timescale (0.1/0.5/2/8 s): held-out-track R2 of an audio-feature model MINUS the R2 of a dancer-identity baseline (one-hot dancer / per-dancer mean), on AIST++ and FineDance. Positive = audio carries information beyond who is dancing; the channel may enter the deterministic block A. Near zero with small inter-dancer r = sample it (block C) or drop it. Reported together with inter-dancer r and the period-locking strength of the channel at tactus/2-beat/bar.")

RB_NEW = [
    ("rb-ls2-00b-predictability-audit", 2,
     """STAGE 0b — PREDICTABILITY AUDIT: which motion descriptors does audio determine? (context: PLAN.md rev. 3 section 2-3 and Stage 0b; requires rb-ls2-00 descriptors). This audit assigns every descriptor to block A (deterministic), B (event), C (sampled) or noise BEFORE any training, so the model never regresses what is personal.
Background: prior probes measured only kinetic energy (inter-dancer r 0.024 @0.1 s, 0.13 @2 s, 0.50 @8 s; held-out audio R2 ~0). Music-induced-movement science predicts which channels ARE shared: pulse clarity -> quantity of motion / CoM speed / feet / hips (r 0.55-0.67), low-frequency flux -> head speed (r 0.73), high-frequency flux -> hands (0.65, large residual), vertical bounce period-locked to the tactus, horizontal sway to 2-beat/bar, energy contrast at drops (Burger 2013/2014, Toiviainen 2010, Solberg & Jensenius 2017). Period is shared; phase is personal.
Do, on AIST++ (30 dancers x 60 tracks) AND FineDance (27 dancers): for every descriptor (QoM; CoM/head/hand/foot speed; vertical bounce amplitude; laterality amplitude and sign; travel; spread; symmetry; jerk; Laban Weight/Time/Space/Flow proxies; footfall rate) at 0.1/0.5/2/8 s smoothing: (i) inter-dancer r for same music; (ii) held-out-TRACK R2 of a ridge/GBM on causal audio features (log-mel, RMS, LF/HF flux, onset, pulse clarity, beat phase); (iii) R2 of a dancer-identity baseline (one-hot dancer, per-dancer mean); (iv) period-locking strength of the vertical channel at the tactus and of horizontal channels at 2-beat/bar (autocorrelation peak vs a shuffled-beat null). Split by track; GroupKFold by music; bootstrap CIs over tracks.
Assign: block A if audio R2 beats identity by > 0.1 with CI excluding 0 at some timescale; block B if the channel is event-like (footfalls, jumps) and its event times beat a rate-matched random null; block C if inter-dancer r is small but the audio-conditioned spread/scale is stable; else noise.
PREREGISTER gate (propose_hypothesis): at least QoM, vertical bounce amplitude, head speed and the metrical-level indicator beat the identity baseline with audio R2 > 0.3 at 2-8 s, and vertical period-locking at the tactus exceeds the shuffled-beat null with z > 3. If no channel passes, the dance-derived skeleton is dead and only events survive from dance. Diagnostic: descriptor-audio-r2-over-identity (record per channel in config_note/key_numbers).""",
     ["QoM, vertical bounce, head speed, metrical level beat dancer identity with audio R2 > 0.3 at 2-8 s", "vertical period-locking at tactus beats shuffled-beat null z > 3"]),
    ("rb-ls2-00c-late-lights-and-category-test", 3,
     """STAGE 0c — TWO CHEAP HUMAN TESTS THAT SHAPE THE ARCHITECTURE (context: PLAN.md rev. 3 sections 4, 6 Stage 0c, 8; requires rb-ls2-00 renders). Both are blind, on screen + headphones, with the existing renderer.
(i) LATE LIGHTS. Take the working beat-trigger layer (HEAD C rules) and delay its output by 0 / 40 / 80 / 120 ms relative to the audio. Pairwise A/B, >= 5 raters, 20 clips of 8 s: 'which is on the music?'. Background: ITU-R BT.1359-1 detectability +45 ms when sound leads; flash-tone JND 20-70 ms. Gate G1: report the delay at which the win rate vs 0 ms drops below 0.4 with CI; if 80 ms already loses, every transient (strobe/hit/snap) must run on a 10-20 ms DSP path and the learned model may only paint channels slower than ~150 ms.
(ii) CATEGORY ERROR (dance != light). 20 tracks (10 dance genres, 10 club/EDM): oracle dance-derived DMX (real AIST++/FineDance motion through conductor v0) vs a 30-line beat-trigger (onset -> strobe, RMS -> dimmer, MFCC jump -> colour). Brightness-match every pair (mean and 95th-percentile LUT); >= 5 raters; two questions: 'which is on the music?' and 'which looks like a lighting engineer?'. Gate G2: oracle wins > 0.5 with CI excluding 0.5 on at least one question; if the beat-trigger wins both, dance keeps only blocks A/B (energy + events) and block C (sampled style residual) leaves the product; if the oracle wins, proceed to rb-ls2-01. Diagnostic: human-preference-winrate.""",
     ["G1 delay at which beat-trigger loses is reported; transient path architecture decided", "G2 oracle dance-derived DMX beats beat-trigger on >= 1 question with CI excluding 0.5, else block C is dropped"]),
]

RB_UPDATES = {
    "rb-ls2-00-oracle-dancer": (1, """STAGE 0a — DESCRIPTORS, LIGHT TARGETS, METRICS, ORACLE RENDERS (context: PLAN.md rev. 3 sections 3, 5, 6 Stage 0a). Dance is the training set; no dancer at inference. This stage builds the descriptor extractor (mocap AND video paths), the fixed motion->light mapping, the metrics, and renders REAL dance mapped to light.
Build: (a) lightsem/motion/: SMPL forward kinematics from /mnt/lightsem-data/aistpp/motions.zip; a 2D pose path (YOLO11-pose or RTMPose, torso-length normalised, multi-person/occluded frames rejected) run on AIST multi-view video (research licence; https://storage.repository.aist.go.jp/1095/v1.0.0/video/10M/ reachable) and validated against the mocap descriptors (gate: speed/laterality/spread descriptors from 2D within r > 0.8 of mocap on 50 clips); descriptor blocks per PLAN.md section 3: A shared skeleton (QoM, CoM/head/foot speed, vertical bounce amplitude + period at tactus, metrical level, Laban Weight/Time proxies, footfall rate, section energy), B events (footfalls, jumps, hits), C style residual (hands, laterality sign/phase, travel, symmetry, spread residual, Space/Flow, jerk); all root-relative, mirror-aware, 40 fps, beat-synchronised; never average dancers except shared per-bar scalars. (b) lightsem/conductor/v0.py: the body-following rig table (QoM/Weight -> intensity/haze/saturation; vertical bounce -> dimmer/tilt bounce at the beat; sway at 2-beat/bar -> pan sweep; head/torso <- LF flux -> big slow fixtures; hands <- HF flux -> small fast fixtures; footfalls/jumps -> strobe/tilt snaps; spread -> zoom; Laban Time -> crossfade) on a canonical rig with slew limits. (c) lightsem/eval/: Light-BAS reported as delta vs metronome (plus circular-shift null), FID_light on DELTA features (onsets, colour jumps, velocities; never levels), Div_light among samples of one track, split-half floor.
Run: targets for all 1408 AIST++ clips; render 5 clips (different genres) real dance -> light -> MP4, and the same 5 through the old structure->scene rules.
PREREGISTER gates: G1 Light-BAS(real-dance light) >= 0.9 x metronome; G2 FID_light(split-half real) x 3 < FID_light(circular-shift vs real); G3 2D-pose descriptors match mocap r > 0.8; G4 Anton's blind look: >= 4/5 real-dance renders read as dancing (if not, fix the mapping, not the idea). Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic."""),
    "rb-ls2-01-latent-dancer-offline": (4, """STAGE 1 — TRAIN THE THREE-BLOCK AUDIO -> LIGHT MODEL ON AIST++ + FineDance (context: PLAN.md rev. 3 sections 3-4, Stage 1; requires rb-ls2-00, 00b, 00c). No dancer, pose or dance generator at inference.
Targets from rb-ls2-00 with block assignment from rb-ls2-00b. Trunk: causal Conformer, 6-8 blocks, d_model 256-384, 15-40 M params, 40 fps, KV cache 3-4 s. Heads: A shared skeleton (MSE/L1 on z-scored features + periodicity head), B events (Bernoulli/hazard, AP/F1 at 0.2 and 1 s), C style residual (conditional rectified flow / CFM on the residual after subtracting A, 1-4 Euler steps, genre token). Aux: beat-gated L1 on hit channels, velocity penalty on wash channels. No VQ tokenizer. Conditioning (causal only): 40-64 log-mel or MFCC + RMS + LF and HF spectral flux + onset + online beat phase (PLP/BeatNet: sin/cos phi, IBI, stability); no bidirectional Jukebox/MERT/MuQ.
Baselines that MUST run: (a) MSE on all 24 dims with the same trunk (expected: Div_light < 0.5 x real — the conditional-mean collapse, documented); (b) HEAD C beat-trigger rules; (c) the Stage-F macro-encoder harness.
Splits: held-out TRACK, held-out GENRE and held-out DANCER (AIST++ same-song-different-dancer is a choreography leak). Also run on 10 club/EDM tracks Anton owns for the collapse rate.
PREREGISTER gates: G1 block A held-out R2 >= 0.8 x the rb-ls2-00b audit ceiling per channel; G2 block B AP at 0.2 s beats the rhythm-of-past-events baseline with z > 3 vs circular shift; G3 Light-BAS(model) >= 0.9 x real-mapped, reported next to the metronome value; G4 FID_light(delta, model vs held-out real) < 0.5 x shift null; G5 Div_light(model) >= 0.7 x real while the MSE baseline's Div < 0.5 x real; G6 edm-collapse-rate < 0.10 (AIST++/FineDance alone will probably fail G6 — that is the case for Stage 2, not a stop). Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic, edm-collapse-rate, descriptor-audio-r2-over-identity."""),
    "rb-ls2-02-operator-head-699": (5, """STAGE 2 — SCALE THE DANCE DATA (VIDEO -> DESCRIPTORS) AND ADD THE OPERATOR HEAD (context: PLAN.md rev. 3 section 5 and Stage 2; requires rb-ls2-01). Two halves, two hypotheses.
(A) DATA SCALE VIA POSE EXTRACTION (Anton's YOLO idea) + Oslo EDM mocap. Run the validated 2D/3D pose -> descriptor pipeline (YOLO11-pose/RTMPose 2D; GVHMR/SMPLest-X 3D where jumps/contacts are needed) over legally usable dance video: InfiniteDance (HF huuuuuuuuu/InfiniteDance, non-commercial academic, ships SMPL-X + MuQ), OpenDanceSet (poses + URLs), PopDanceSet, Let's Dance, BRACE, and own event footage (consent). Add the Oslo club-dancing mocap (Zenodo 21945706, 21810782, 21945721; CC-BY-4.0; head/wrist markers; DJ-mixed EDM; music withheld — source the tracks, align on the provided break/build/drop timing) for block A on concert music. Target ~100-200 h. No YouTube scraping. Retrain. Gates G1: edm-collapse-rate < 0.10 on the 10 club/EDM tracks and FID_light on them within 1.5x the AIST++ held-out value; G2 no regression of Stage-1 gates; G3 the Oslo drop/break QoM contrast is reproduced by block A on held-out mixes (drop QoM surge > breakdown by the same sign with CI excluding 0).
(B) OPERATOR HEAD on full RPMC-L2 (699 songs, 42.5 GB, HF RS2002/RPMC-L2/data/RPMC_L2_parts/ to /mnt/lightsem-data/rpmc_l2). Background: on 140 songs marginal AP lift +0.0909, z +2.83; MFCC z +4.98, spectral z +4.21, energy z +0.66; reactive. Train the block-B/hue-value head on the shared trunk: p(scene change in 0.2/0.5/1.0 s) and hue (180 bins)/value distributions (KL, per-song centred). 5-fold by song, 40-rep circular-shift null, per-family ablation. Gates G4: marginal AP lift at 1.0 s >= +0.09 with z > 3 on 699 songs; G5 held-out hue KL < per-song-mean baseline; G6 (guard) absolute lift at 4 s reported; anticipation claimable only if > +0.05 with CI excluding 0."""),
    "rb-ls2-03-conductor-ab": (6, """STAGE 3 — CONDUCTOR v0 END-TO-END + BLIND BRIGHTNESS-MATCHED HUMAN A/B (context: PLAN.md rev. 3 sections 4, 6 Stage 3, 7; requires rb-ls2-01 and rb-ls2-02). First result that may be CALLED 'dancing light'.
Build: trunk heads A/B/C + HEAD B operator outputs + HEAD C transient rules through conductor v0 (body-following rig table; multiplicative strobe gate accent x hit/drop; slew limits) into DMX on the canonical rig, then the rig adaptor to the club rig in lightsem/dmx. Two-path timing even offline: transients from the DSP path, slow channels from the model.
Render 20 tracks (10 dance genres, 10 club/EDM) in four conditions with the SAME rig, renderer and audio: (i) beat-trigger only (HEAD C), (ii) hue/value-only Skip-BART/SeqLight reproduction (no kinematics), (iii) LightSem v2 (A+B+C), (iv) light derived from REAL dance (upper reference; dance tracks only).
Study: >= 20 raters; pairwise blind; BRIGHTNESS-MATCH every pair (mean and 95th-percentile LUT); 4-8 s clips; include a too-bright foil; order/side randomised; two forced choices: 'which is on the music?' and 'which looks like a lighting engineer?'; bootstrap 95% CI over tracks x raters. Diagnostic: human-preference-winrate.
PREREGISTER gates: G1 v2 vs (i) win rate > 0.60 with CI excluding 0.5 on 'on the music'; G2 v2 vs (ii) > 0.50 with CI excluding 0.5 on 'lighting engineer'; G3 v2 vs (iv) not significantly below 0.40; G4 the too-bright foil does not beat v2 (raters-prefer-brightness check). Report Light-BAS delta vs metronome and FID_light(delta) per condition to calibrate the proxies against the human result. Name the failure mode in the close: if (i) wins, kinematics are too busy -> sparsify; if (ii) ties, check beam visibility/haze before concluding kinematics add nothing."""),
    "rb-ls2-04-causal-student-latency": (7, """STAGE 4 — LIVE: TWO-PATH DEPLOYMENT, LATENCY, CAUSALITY, REAL RIGS (context: PLAN.md rev. 3 section 4 and Stage 4; requires rb-ls2-01). The model is already causal; this stage proves it under concert/festival constraints.
Background: ITU-R BT.1359-1 detectability +45 ms when sound leads light; flash-tone JND 20-70 ms; Art-Net/sACN sub-10 ms on a managed LAN; fixture internal smoothing 5-50 ms; online beat trackers: PLP 11.6 ms, BeatNet ~20 ms, ~6 s of past needed for stable phase; a 100-250 ms PA delay line is viable in the house feed, never in monitors.
Do: (1) transient path: onset/beat-phase -> strobe/hit/snap channels on a DSP path, target <= 20 ms; (2) learned path: trunk + heads at 40 fps, FP16, 1-2 flow steps, KV cache; (3) lookahead sweep L in {0, 40, 80, 250} ms as an audio delay line; (4) causality verified mechanically: future-perturbation (splice a click / +6 dB at t+50/100/250 ms; output at t bit-identical, fp16 max |dy| < 1e-4), gradient probe dy_t/daudio_{t+k} = 0, shuffled-future buffer; (5) end-to-end latency with a click-track impulse: scope on audio-in, photodiode on a fixture (or LED), Art-Net timestamps; report p50/p99 over 60 s at 40 fps and at 120 BPM; (6) Art-Net/sACN to a real node or BlenderDMX, verified by an independent receiver; (7) rig adaptor for a multi-universe festival patch in BlenderDMX.
PREREGISTER gates: G1 transient path <= 20 ms p99; G2 learned path <= 65 ms p50 audio-in -> DMX-out at L = 0 (12 ms audio block + 8 ms model + 5 ms net + fixture); G3 Light-BAS delta vs metronome at L = 0 >= 0.9 x offline; G4 causality tests pass bit-identically; G5 Art-Net frames decoded by an independent receiver; G6 the MSE baseline still shows Div_light < 0.5 x real live. Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic, audio-to-dmx-latency-ms."""),
    "rb-ls2-05-learned-conductor": (8, None),
    "rb-ls2-06-edm-anticipation": (9, None),
}


def main():
    other = "/root/vibescience-vault"
    b_other, b_ours = snapshot(other), snapshot("/root/lightsem-vault")
    c = Client()
    info = c.initialize()
    assert info["name"] == "vibescience-lightsem", info
    print("connected:", info)

    print("\n== tag")
    call(c, "register_tag", {"id": "music-induced-movement", "axis": "topic",
                             "description": "Empirical music-psychology literature on how audio features drive spontaneous body movement across many people (Burger/Toiviainen/Luck Jyvaskyla mocap, Solberg & Jensenius club EDM, groove research). Source of the shared-vs-personal split that defines LightSem's target blocks.",
                             "aliases": ["embodied-meter", "groove", "music-psychology", "spontaneous-dance", "jyvaskyla"]})

    print("\n== diagnostic")
    did, name, unit, direction, desc = DIAG
    call(c, "register_diagnostic", {"id": did, "name": name, "unit": unit, "direction": direction,
                                    "description": desc, "topic_tags": ["evaluation-metric", "music-induced-movement"]})

    print("\n== papers")
    pids = []
    for title, ref, claims, tags in PAPERS:
        out = call(c, "add_paper", {"title": title, "arxiv_id_or_url": ref, "key_claims": claims, "topic_tags": tags})
        if isinstance(out, dict) and (out.get("id") or out.get("paper_id")):
            pids.append(out.get("id") or out.get("paper_id"))
    for pid in pids:
        call(c, "link_paper", {"entity_id": P, "paper_id": pid})

    print("\n== problem")
    call(c, "update_problem", {"id": P, "description": DESC,
                               "topic_tags": ["audio-to-light", "music-to-dance", "motion-to-light", "generative-modelling",
                                              "real-time", "evaluation-metric", "cross-modal", "pose", "music-induced-movement"],
                               "add_update": ("2026-09-04 rev. 3: integrated four Perplexity reasoning passes (audio->light design; music-induced-movement "
                                              "science; causal model + latency; dataset audit degenerated, replaced by direct source checks) and Anton's "
                                              "note on extracting motion from video with pose models. Targets split into deterministic skeleton / event "
                                              "point-process / sampled style residual; rig mapping follows the body; two-path deployment (transients on a "
                                              "10-20 ms DSP path) because ITU-R BT.1359-1 detectability is +45 ms when sound leads. New runbooks 00b "
                                              "(predictability audit) and 00c (late-lights + category-error tests) run BEFORE any training.")})
    call(c, "synthesize_problem", {"problem_id": P, "synthesis": SYNTHESIS})

    print("\n== runbooks")
    for rid, prio, prompt, gates in RB_NEW:
        call(c, "register_runbook", {"id": rid, "problem_id": P, "prompt": prompt, "priority": prio, "gates": gates,
                                     "topic_tags": ["audio-to-light", "music-induced-movement"]})
    for rid, (prio, prompt) in RB_UPDATES.items():
        args = {"runbook_id": rid, "priority": prio, "note": "rev. 3 2026-09-04: three-block targets, two-path latency, video->pose data path"}
        if prompt:
            args["prompt"] = prompt
        call(c, "update_runbook", args)

    c.close()
    a_other, a_ours = snapshot(other), snapshot("/root/lightsem-vault")
    print(f"\nlightsem-vault md files: {b_ours} -> {a_ours}")
    print(f"optiHealth vault md files: {b_other} -> {a_other}  (must be unchanged)")
    assert b_other == a_other, "WROTE INTO THE WRONG VAULT"
    return 0


if __name__ == "__main__":
    sys.exit(main())
