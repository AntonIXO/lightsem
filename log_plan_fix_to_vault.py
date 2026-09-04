#!/usr/bin/env python3
"""Rev. 2 correction of the LightSem v2 plan in the LIGHTSEM vault (2026-09-03).

Anton's correction: dance datasets are the TRAINING SET for a sound->light model
for concerts/festivals. No music-to-dance generation at inference. This script
rewrites the problem description/title, the synthesis, and the runbook prompts
that had a 'latent dancer' generator in the inference path. Additive: nothing
is deleted; the rev. 1 text is superseded by update, and the change is dated.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client  # noqa: E402

NEW_PROBLEM = "dancing-light-generative-20260903"


def snapshot(path: str) -> int:
    return sum(len([f for f in fs if f.endswith(".md")]) for _, _, fs in os.walk(path))


def call(c, name, args):
    out = c.call(name, args)
    short = json.dumps(out, ensure_ascii=False)[:200] if not isinstance(out, str) else out[:200]
    print(f"  {name:20s} {args.get('id', args.get('runbook_id', args.get('problem_id', '')))!s:40.40s} -> {short}")
    return out


DESC = """LightSem v2, rev. 2 (Anton, 2026-09-03, corrected). TASK: a SOUND -> LIGHT model for concerts and festivals, TRAINED ON DANCE. Paired (music, 3D dance motion) datasets (AIST++, FineDance, InfiniteDance, PopDanceSet, ~120 h, licensed) are the training set; the dancer's motion is ground truth for how a body answers this music. Offline, once: motion -> Motion Descriptor Vector (speed, jerk, spread, verticality, laterality, travel, Laban effort proxies) -> LIGHT TARGETS through a fixed mapping on a canonical rig (speed -> brightness, side -> fixture group, spread -> beam width, jumps -> tilt, accents -> strobe/colour snap). Then train AUDIO -> LIGHT directly and causally. At inference there is no dancer, no pose, no dance generator: sound in, DMX out. Rev. 1 of this problem had a music-to-dance generator in the inference path; that is withdrawn (kept only as an optional offline pseudo-labelling teacher, PLAN.md section 8).

WHY THIS IS NOT THE RULED-OUT ATTEMPT. The Aug-2026 rule-outs in [[audio-to-light-semantic-mapping-20260807]] trained a REGRESSOR to one dancer's kinetic energy at beat resolution: held-out R2 ~ 0, inter-dancer r = 0.024 at 0.1 s. p(motion | music) is multi-modal (30 AIST++ dancers give 30 valid answers to one bar), so the MSE-optimal predictor is the conditional mean, a blurred average no dancer performs (problem tag conditional-mean-collapse). The music-to-dance literature solved exactly this with GENERATIVE objectives evaluated distributionally (FlowerDance BAS 0.232 vs GT 0.201, arXiv 2511.21029). Applied to audio -> light targets, the same objective samples one valid light choreography per performance, which is what a light show is. Four changes vs the ruled-out attempt: (1) target = designed low-dim light control vector, not raw KE; (2) objective = flow matching / diffusion forcing, regression kept only as the baseline that must collapse; (3) ~120 h / 30+ genres instead of 5.2 h; (4) metrics = FID / diversity / beat alignment of light against HELD-OUT dance-derived light, priced against metronome and shift nulls, human A/B as headline. The shared audio/motion information (tempo, phase, phrase, energy, timbral changes) is predicted; the dancer-unique detail is sampled (MAS.S60 lecture 08: transfer moves only redundant information).

ARCHITECTURE (PLAN.md section 3). A0 causal audio front-end (online beat/downbeat, bar phase, spectral+MFCC families, MuQ/MERT embedding, DiscoForcing-style RVQ rhythm codes + periodic phase). HEAD A dance-trained light head: causal generative model audio -> 24-d MDV-space light control sequence, ~5-20 M params. HEAD B operator head on RPMC-L2 (699 songs): timbre-driven scene-change probability (validated z +2.8..5) + hue/value distributions (KL). HEAD C deterministic beat-locked micro layer. CONDUCTOR v0 rule table with multiplicative strobe gate and slew limits, v1 learned (SeqLight-style decomposition, preference fine-tuning, additive-projection audit); rig adaptor canonical rig -> venue patch, multi-universe, Art-Net/sACN, BlenderDMX.

STAGES (runbooks rb-ls2-infra, 00..06): 0 build targets + Light-BAS/FID metrics + renders of REAL dance mapped to light (if that does not read as dancing, fix the mapping before training); 1 train audio->light on AIST++ vs regression and rule baselines; 2 scale to ~120 h + RPMC-L2 699 operator head; 3 conductor + blind human A/B vs beat-trigger / Skip-BART / real-dance upper reference; 4 live: latency < 100 ms + L, online tracking, Art-Net on hardware, festival rig adaptor; 5 learned conductor; 6 EDM anticipation only with real EDM lighting ground truth.

DATA/INFRA: bulk data on /mnt/lightsem-data (325 GB free; root disk 89% full). AIST++ SMPL motions.zip downloaded 2026-09-03. No YouTube scraping. InfiniteDance is non-commercial academic."""

SYNTHESIS = """STATE OF KNOWLEDGE 2026-09-03 rev. 2 (no experiments run yet under this problem).

TASK: sound -> light for concerts/festivals, trained on dance datasets; no dancer/pose/dance generator at inference.

Established (inherited): regression audio -> one dancer's motion at beat resolution is ill-posed (inter-dancer r 0.024 @0.1 s; held-out R2 ~ 0); audio -> light LEVEL is owned by persistence (+0.708 vs audio -0.005); audio -> light EVENT is real and timbre-driven (AP lift +0.09, z +2.83; MFCC z +4.98, spectral z +4.21, energy z +0.66) but reactive; sound-pivot chain bounded <= 0; AIST audio lacks EDM structure; timing metrics need a metronome null.

Live claim: training audio -> light on dance-derived targets is viable IF the objective is generative and the evaluation distributional. Evidence by analogy from music-to-dance: sampled sequences keep real dancers' beat alignment (BAS 0.232 vs 0.201) although any per-frame regression to one dancer would score ~0. No learned KINEMATIC layer for stage lighting exists in the literature (Skip-BART/SeqLight are hue/value only); closest analog is music -> camera (DanceCamera3D), where dance is the strong condition and music the weak one.

Decisive next measurements: Stage 0 - map REAL AIST++ motion to light through the rule table, render, compute Light-BAS vs metronome/shift nulls and the split-half FID floor, and look at it (fixes the mapping before any training). Stage 1 - train the generative audio -> light head on AIST++ next to a regression baseline and the beat-trigger rules; the regression baseline must show diversity collapse while the generative head matches held-out dance-derived light distributionally.

Risks: (1) the motion -> light table may not read as dancing; (2) dataset music is studio dance, product music is club/EDM/rock (collapse rate is the watch metric; InfiniteDance's 30 genres and tempo-routed experts are the mitigation); (3) causality costs beat sync (lookahead sweep in Stage 4); (4) metric gaming; (5) InfiniteDance licence is non-commercial.

Do not: regress KE; regress absolute levels; claim anticipation on RPMC-L2; scrape YouTube; put a dance generator in the inference path; report BAS without the metronome value beside it."""

RUNBOOK_UPDATES = {
    "rb-ls2-00-oracle-dancer": """STAGE 0 — BUILD THE DANCE-DERIVED LIGHT DATASET AND THE METRICS (context: PLAN.md rev. 2, sections 3.2-3.3, 5 Stage 0). Task framing: dance datasets are the TRAINING SET for a sound->light model; there is no dancer at inference. This stage builds the targets and checks that REAL dance mapped to light reads as 'the light is dancing'. If it does not, the MAPPING is wrong (not the idea): iterate the rule table before any training.
Background: the vault ruled out regressing dancer kinetic energy as DMX supervision (inter-dancer r 0.024 @0.1 s) and audio->light levels (persistence R2 +0.708). Nothing is regressed here.
Build: (a) lightsem/motion/: SMPL forward kinematics from /mnt/lightsem-data/aistpp/motions.zip; Motion Descriptor Vector (MDV, ~24-d, root-relative, mirror-aware, beat-synchronised): speed (global, upper, lower), acceleration, jerk, spatial spread, verticality (COM height, jumps), laterality (L/R weight, lateral travel), travel direction and turn rate, symmetry, foot contacts, Laban proxies (Weight ~ |accel|, Time ~ jerk/velocity, Space ~ path curvature, Flow ~ velocity smoothness); unit-test against keypoints3d. (b) lightsem/conductor/v0.py: rule table on a CANONICAL rig (6 heads L/C/R x 2 trusses, washes, blinders, strobe): speed->dimmer, laterality->L/R groups, lateral travel->pan sweep, verticality->tilt, spread->zoom, turn rate->pan chase, jerk AND accent->strobe/colour snap, Weight->saturation, Flow->crossfade; per-fixture slew limits. Store per clip: MDV sequence + canonical-rig light targets. (c) lightsem/eval/: Light-BAS (kinematic beats of light = local minima of fixture-parameter velocity / dimmer peaks vs music beats, dance-BAS kernel), FID_light-k/-g over 2.5 s windows of velocity/acceleration features, Div_light, metronome and circular-shift nulls, split-half real-vs-real resolution floor.
Run: targets for all 1408 AIST++ clips; render 5 clips (different genres) real dance -> light -> MP4 with the existing renderer, and the same 5 through the old structure->scene rules.
PREREGISTER (propose_hypothesis with gates): G1 Light-BAS(real-dance light) >= 0.9 x Light-BAS(metronome at the same rate); G2 FID_light(split-half real) x 3 < FID_light(circular-shift motion vs real); G3 Anton's blind look: >= 4/5 real-dance renders judged 'dancing' vs <= 2/5 rule-based. Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic.""",
    "rb-ls2-01-latent-dancer-offline": """STAGE 1 — TRAIN AUDIO -> LIGHT ON AIST++ (context: PLAN.md rev. 2 section 3.4 and Stage 1; requires rb-ls2-00 gates). The product model: a CAUSAL GENERATIVE model from audio features to the 24-d MDV-space light control sequence, trained on the dance-derived targets from Stage 0. No dancer, pose or dance generator at inference.
Why generative: the Aug-2026 regression to one dancer's KE collapsed to the conditional mean (held-out R2 ~ 0) because p(motion|music) is multi-modal; the same target with a sampling objective is what the music-to-dance literature uses successfully (FlowerDance BAS 0.232 vs GT 0.201). Keep multi-dancer clips of the same music as separate samples.
Model: A0 features (beat/downbeat + bar phase, spectral centroid/bandwidth/rolloff/contrast + MFCC, MuQ or MERT embedding; all computed causally) -> diffusion forcing / flow matching over MDV sequences with a FIFO denoising window (DiscoForcing arXiv 2605.28491 recipe), 5-20 M params, split by song AND dancer. Baselines that MUST run: (a) causal AR REGRESSION to the same target (expected to collapse; record its Div_light), (b) HEAD C beat-trigger rules, (c) the Stage-F macro encoder harness for continuity.
Score on held-out dance-derived light (unseen songs and dancers); also run on 10 club/EDM tracks Anton owns (no scraping) for the collapse rate.
PREREGISTER gates: G1 Light-BAS(model) >= 0.9 x Light-BAS(held-out real-dance light), reported next to the metronome value; G2 FID_light(model vs held-out real) < 0.5 x FID_light(circular-shift vs real); G3 Div_light(model) >= 0.7 x real while regression baseline Div_light < 0.5 x real; G4 edm-collapse-rate < 0.10 on the EDM tracks — AIST++ alone will probably fail G4; that is the case for Stage 2, not a stop. Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic, edm-collapse-rate.
Interpretation: G1-G3 failing on AIST++ itself = the objective/target design is wrong, investigate before scaling data; only G4 failing = genre gap.""",
    "rb-ls2-02-operator-head-699": """STAGE 2 — SCALE THE DANCE DATA TO ~120 H AND ADD THE OPERATOR HEAD (context: PLAN.md rev. 2 Stage 2; requires rb-ls2-01). Two halves, two hypotheses (never bundle).
(A) DATA SCALE. Add FineDance public 7.7 h (SMPL-X, Google Drive manual), InfiniteDance 100.7 h / 30 genres (HF huuuuuuuuu/InfiniteDance, non-commercial academic; ships MuQ features), PopDanceSet 3.6 h; retarget to a common skeleton; MDV + canonical-rig targets on /mnt/lightsem-data; retrain HEAD A. Gate G1: edm-collapse-rate < 0.10 on the 10 club/EDM tracks and FID_light on them within 1.5x the AIST++ held-out value; G2 no regression of Stage-1 gates on AIST++.
(B) OPERATOR HEAD on full RPMC-L2 (699 songs, 42.5 GB, 4 split parts at HF RS2002/RPMC-L2/data/RPMC_L2_parts/, download to /mnt/lightsem-data/rpmc_l2). Background: on 140 songs audio->scene-change (next 1 s) had marginal AP lift +0.0909 over a rhythm-of-past-changes baseline, z +2.83 vs circular-shift null; MFCC z +4.98, spectral z +4.21, energy z +0.66; reactive (lift ~0 by 4 s). Audio->LEVEL is refuted; do not regress levels. Train a small causal head (shared A0 backbone) for p(scene change in 0.2/0.5/1.0 s) and per-frame hue (180 bins) / value distributions (KL, per-song centred). 5-fold by song, 40-rep circular-shift null per horizon, per-family ablation. Gates G3: marginal AP lift at 1.0 s >= +0.09 with z > 3 on 699 songs; G4: held-out hue KL < per-song-mean baseline; G5 (guard): absolute lift at 4 s reported, anticipation claimable only if > +0.05 with CI excluding 0. Register a hue-KL diagnostic at run time.""",
    "rb-ls2-03-conductor-ab": """STAGE 3 — CONDUCTOR v0 END-TO-END + BLIND HUMAN A/B (context: PLAN.md rev. 2 Stage 3 and section 4; requires rb-ls2-01 and rb-ls2-02). First result that may be CALLED 'dancing light'.
Build: fuse HEAD A (audio -> MDV-space light control, dance-trained), HEAD B (cue probability, hue/value target, structure), HEAD C (beat-locked micro layer) through conductor v0 into DMX for the canonical rig, then the rig adaptor to the club rig in lightsem/dmx. Arbitration: HEAD B changes look and palette, HEAD A drives kinematics/intensity contour, HEAD C accents; multiplicative strobe gate (accent x hit/drop); slew limits.
Render 20 tracks (10 dance genres, 10 club/EDM) in four conditions with the SAME rig, renderer and audio: (i) HEAD C only (beat-trigger, SoundSwitch-like), (ii) hue/value-only Skip-BART/SeqLight reproduction (no kinematics), (iii) LightSem v2 (A+B+C), (iv) light derived from REAL dance (upper reference; dance tracks only).
Study: >= 20 raters, pairwise blind 'which light dances better with the music', order/side randomised, bootstrap 95% CI over tracks x raters. Diagnostic: human-preference-winrate.
PREREGISTER gates: G1 v2 vs (i) win rate > 0.60 with CI excluding 0.5; G2 v2 vs (ii) > 0.50 with CI excluding 0.5; G3 v2 vs (iv) not significantly below 0.40. Report Light-BAS and FID_light per condition next to the metronome null to calibrate the proxies against the human result.
Name the failure mode in the close: if (i) wins, kinematics are too busy -> sparsify the mapping; if (ii) ties, check beam visibility/haze in the renderer before concluding kinematics add nothing.""",
    "rb-ls2-04-causal-student-latency": """STAGE 4 — LIVE: LATENCY, ONLINE TRACKING, REAL RIGS (context: PLAN.md rev. 2 Stage 4; requires rb-ls2-01). The model from Stage 1/2 is already causal; this stage proves it under concert/festival constraints.
Do: (1) strictly causal A0 path end-to-end (online beat tracker BeatNet / zero-latency PLP; sliding-buffer features; no full-track access); (2) lookahead sweep L in {0, 100, 250, 500} ms implemented as an audio delay line; (3) audio-in -> DMX-out latency measured with a click-track impulse on this box; (4) Art-Net/sACN to a real node or BlenderDMX, wire format verified by an independent receiver (currently untested); (5) rig adaptor from the canonical rig to a multi-universe festival patch (GDTF-like personalities), verified in BlenderDMX.
PREREGISTER gates: G1 Light-BAS at L = 0 >= 0.9 x the offline (full-track features) value, reported with the metronome null; G2 FID_light at L = 0 <= 1.5 x offline; G3 audio-to-dmx-latency-ms < 100 + L; G4 Art-Net frames decoded correctly by an independent receiver; G5 the causal regression baseline still shows Div_light < 0.5 x real (the collapse is documented in the live setting too). Diagnostics: light-bas, fid-light-kinetic, light-diversity-kinetic, audio-to-dmx-latency-ms.""",
    "rb-ls2-05-learned-conductor": """STAGE 5 — LEARNED CONDUCTOR AND PREFERENCE FINE-TUNING (context: PLAN.md rev. 2 Stage 5; optional research stage after rb-ls2-03 passes). One hypothesis per idea, never bundled: (A) SeqLight-style goal-conditioned decomposition of HEAD B's hue/value distribution across the rig's fixtures plus a small policy (HEAD A control vector, cue, structure, beat phase) -> fixture parameters, trained by imitation of conductor v0 + real-dance-derived light and fine-tuned on Stage 3/4 preference pairs (GRPO/DPO); audit every learned fusion term by projecting to the nearest additive model (Hessel & Lee), credit only multiplicative terms that survive. (B) multi-group choreography: independent samples of HEAD A (different seeds / genre tokens) assigned to fixture groups (left truss, centre, right truss, floor) so a festival rig gets several coordinated 'voices'.
PREREGISTER gates per hypothesis: G1 beats conductor v0 in a fresh blind A/B with win-rate CI excluding 0.5 (human-preference-winrate); G2 for (A) the additive projection loses >= 2 points of win rate, else ship the additive model; G3 for (B) Div_light rises >= 20% without FID_light worsening > 20%.""",
}


def main() -> int:
    other = "/root/vibescience-vault"
    b_other, b_ours = snapshot(other), snapshot("/root/lightsem-vault")
    c = Client()
    info = c.initialize()
    assert info["name"] == "vibescience-lightsem", info
    print("connected:", info)

    print("\n== problem rewrite")
    call(c, "update_problem", {
        "id": NEW_PROBLEM,
        "title": "Sound -> light for concerts/festivals, TRAINED ON DANCE datasets (LightSem v2): generative audio -> light on dance-derived targets",
        "description": DESC,
        "add_update": ("2026-09-03 rev. 2 (Anton's correction): dance datasets are the TRAINING SET for a "
                       "sound->light model; no music-to-dance generation at inference. Rev. 1's 'latent dancer' "
                       "inference path is withdrawn; generators survive only as an optional offline pseudo-labelling "
                       "teacher (PLAN.md section 8). Runbooks rb-ls2-00..05 rewritten accordingly."),
    })
    call(c, "synthesize_problem", {"problem_id": NEW_PROBLEM, "synthesis": SYNTHESIS})

    print("\n== runbook rewrites")
    for rid, prompt in RUNBOOK_UPDATES.items():
        call(c, "update_runbook", {"runbook_id": rid, "prompt": prompt,
                                   "note": "rev. 2 2026-09-03: dance = training set; no dance generator at inference"})

    c.close()
    a_other, a_ours = snapshot(other), snapshot("/root/lightsem-vault")
    print(f"\nlightsem-vault md files: {b_ours} -> {a_ours}")
    print(f"optiHealth vault md files: {b_other} -> {a_other}  (must be unchanged)")
    assert b_other == a_other, "WROTE INTO THE WRONG VAULT"
    return 0


if __name__ == "__main__":
    sys.exit(main())
