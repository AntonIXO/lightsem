#!/usr/bin/env python
"""Queue Stage 9 (PLAN.md rev. 8, section 6.9) in the vault: two tags, three diagnostics, four
papers, one runbook, seven hypotheses, one problem update. Nothing is measured here.

    bash run.sh scripts/vault/queue_stage9_to_vault.py
"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client

P = "dancing-light-generative-20260903"
VAULT = Path("/root/lightsem-vault")
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


def soft(r, what):
    """Registrations that may already exist: print and continue."""
    if not (isinstance(r, dict) and r.get("ok") is not False):
        print(f"  [{what}] not created:", str(r)[:160])
    return r


# ---- tags (registered vocabulary; topic axis) -------------------------------------------
for tid, desc in [
        ("source-separation", "stems, per-stem features and their use as conditioning"),
        ("learned-conductor", "replacing hand-written descriptor-to-fixture rules with a learned decoder")]:
    if not (VAULT / "tags" / f"{tid}.md").exists():
        soft(c.call("register_tag", {"id": tid, "axis": "topic", "description": desc}), f"tag {tid}")

# ---- diagnostics --------------------------------------------------------------------------
for d in [
    {"id": "critic-pair-accuracy", "name": "critic_accuracy_real_vs_beat_trigger",
     "unit": "fraction", "direction": "higher_better",
     "topic_tags": ["evaluation-metric", "audio-to-light", "cross-modal"],
     "description": ("Accuracy of the audio-light critic at telling the real authored light for a "
                     "song from beat-trigger light rendered on the same audio, on held-out songs. "
                     "Real-vs-time-shift accuracy is reported next to it but does not count: a "
                     "critic that separates shift but not beat-trigger has learned RMS and onsets.")},
    {"id": "rig-transfer-hist-distance", "name": "hsv_histogram_distance_unseen_rig_vs_canonical",
     "unit": "distance", "direction": "lower_better",
     "topic_tags": ["evaluation-metric", "motion-to-light", "learned-conductor"],
     "description": ("Per-frame HSV-histogram distance between a show rendered on an unseen patch "
                     "(different N, no strobe, extra LED bars) and the same show on the canonical "
                     "rig, priced against channel-copy / drop-extras remapping.")},
    {"id": "preference-heldout-accuracy", "name": "bradley_terry_heldout_pair_accuracy",
     "unit": "fraction", "direction": "higher_better",
     "topic_tags": ["evaluation-metric", "audio-to-light", "learned-conductor"],
     "description": ("Held-out accuracy of a Bradley-Terry model over conductor knobs at predicting "
                     "which of two brightness-matched renders a rater preferred; fitted separately "
                     "for 'fits the music' and 'looks like an engineer did it'; chance is 0.5.")}]:
    if not (VAULT / "diagnostics" / f"{d['id']}.md").exists():
        ok(c.call("register_diagnostic", d))

# ---- papers -------------------------------------------------------------------------------
PAPERS = [
    ("Automatic Stage Lighting Control: Is it a Rule-Driven Process or Generative Task? (Skip-BART, RPMC-L2)",
     "arXiv:2506.01482",
     ["ASLC framed as generation, not classification: BART over OpenL3 audio at 10 Hz emits hue and value "
      "autoregressively with a music-to-light skip connection; saturation fixed at 100%",
      "RPMC-L2: 699 clips from 35 livehouse shows, 2020-2024, released as processed features (CC-BY-4.0 on HF), no video",
      "38 raters: Skip-BART 4.35 vs ground truth 4.51 vs rule-based 2.67 on a 7-point scale; GT vs Skip-BART p=0.72",
      "training: 240M params, 19M trainable under LoRA, ~18 GB, two 4090s and an A100; ~15 h pretrain + 1.5 h fine-tune"],
     ["audio-to-light", "generative-modelling"]),
    ("Stage Light is Sequence^2: Multi-Light Control via Imitation Learning (SeqLight)",
     "arXiv:2605.03660",
     ["two-stage design: Skip-BART predicts the full hue/value distribution of all lights (venue-independent), "
      "then a goal-conditioned policy (BC -> AIRL -> PPO/GRPO fine-tune, HER-style goals from mixed light) "
      "decomposes it onto the venue's lights with no per-fixture demonstrations",
      "trained on RPMC-L2 only; eight simulated point lights; fixed light order; hue and value only, no pan/tilt/gobo",
      "30 raters: SeqLight 4.54 vs Skip-BART 3.90 vs rule-based 2.70 in-domain; 3.94 vs 3.47 vs 2.70 on Suno OOD songs",
      "the closest published system to LightSem Stage 9's group-effect decoder; per-fixture DMX is not published"],
     ["audio-to-light", "generative-modelling", "learned-conductor"]),
    ("Factorized Contrastive Learning: Going Beyond Multi-view Redundancy (FactorCL)",
     "arXiv:2306.05268",
     ["contrastive objectives capture only the information shared across views; task-relevant unique information "
      "needs factorised shared/unique representations",
      "basis for Stage 9's two-head critic (redundant vs unique) and for the beat-trigger negative family"],
     ["cross-modal", "evaluation-metric"]),
    ("Lodge: A Coarse to Fine Diffusion Network for Long Dance Generation Guided by the Characteristic Dance Primitives",
     "arXiv:2403.10518",
     ["two-level generation: sparse characteristic primitives planned globally, local diffusion fills in",
      "the structure Stage 9 borrows for the section planner writing conductor knobs while the frame model stays fixed"],
     ["music-to-dance", "generative-modelling"]),
]
for title, arx, claims, tags in PAPERS:
    stem = title.split(" (")[0].split(":")[0].lower()
    hits = [f for f in (VAULT / "papers").glob("*.md") if arx.split(":")[1] in f.read_text()]
    if hits:
        print("paper exists:", hits[0].name); continue
    r = c.call("add_paper", {"title": title, "arxiv_id_or_url": arx, "key_claims": claims, "topic_tags": tags})
    pid = r.get("paper_id") or r.get("id") if isinstance(r, dict) else None
    print("paper:", pid or r)
    if pid:
        soft(c.call("link_paper", {"entity_id": P, "paper_id": pid}), "link")

# ---- runbook ------------------------------------------------------------------------------
if not (VAULT / "runbooks" / "rb-ls2-09-designed-light.md").exists():
    ok(c.call("register_runbook", {
        "id": "rb-ls2-09-designed-light", "problem_id": P, "priority": 1,
        "prompt": (
            "STAGE 9 -- FROM DECENT TO DESIGNED: authored light as the target, the conductor as the model "
            "(PLAN.md rev. 8 section 6.9). Diagnosis after Stage 8 (held-out JD: BAS +0.135 vs coach +0.087 "
            "vs beat trigger +0.092; FID 36.5 vs floor 8.1; block A R2 -0.22; inter-coach r 0.22-0.42): the "
            "bottleneck is the target (dance is a weak proxy, coaches disagree), the hand-written conductor "
            "(owns every artistic decision) and the absence of human preference -- not the trunk, causality "
            "or the amount of dance. Five outside passes (one deep research, four reasoning, full brief "
            "attached) and two papers read in full (Skip-BART arXiv:2506.01482, SeqLight arXiv:2605.03660) "
            "agree.\n"
            "Targets, one format (group-level event stream + per-group intensity/colour envelopes at 40 fps): "
            "RPMC-L2 699/699 (cache/rpmc_all, CC-BY-4.0, 10 Hz OpenL3 + scene vector, NO waveform); Just "
            "Dance visual (332 captures, wav on disk; exp-016 lift 3.17x vs shift 1.48x); xLights free .xsq "
            "sequences (~556 free, ~25-40 h, effects on groups at 25 ms, audio NOT included, research only, "
            "only where the song is on disk); Beat Saber lightshow maps (BeatSaver API ?tags=lightshow, v3 "
            "lightColorEventBoxGroups = group events with phase-spread filters, exact BPM grid, audio in zip, "
            "research only). Console show files (grandMA XML + MVR) exist only by contract: business "
            "development, not scraping. Not targets: Rock Band VENUE, lasers, pyro, fountains, SoundSwitch.\n"
            "Steps and gates: 9.2 corpus predictability gate (H-H) before any training -- AP_audio - AP_shift "
            ">= +0.15 AND AP_audio >= AP_beat-trigger + 0.05 on events and on group-envelope R2, causal and "
            "offline together. 9.3 group-effects decoder (H-I, Anton's mixture of light in SeqLight's form): "
            "shared 0.5-2 M head with type masks on the frozen trunk + operator head emitting per-group "
            "intensity envelope, palette, effect id, rate in beats, phase spread, density, look-change hazard; "
            "deterministic effects engine binds groups to any patch (sorted by type and stage x) and keeps the "
            "H-B one-pole; loss on a rig-invariant observable via a cheap differentiable histogram renderer "
            "against RPMC-L2 / JD visual / gated corpora; random-rig dropout; no conductor-DMX distillation; "
            "no pan/tilt learning. Gate G9.3 (Value- and onset-matched): FID vs conductor beats a shuffled-"
            "donor control by > 2x; BAS not below the conductor without raising onset density; brightness-"
            "matched A/B 'fits the music' CI excluding 50% at n >= 20; rig transfer beats channel-copy. Miss "
            "any two: the conductor was not the bottleneck. 9.4 section planner writing conductor knobs (H-J), "
            "audio-only, one bar ahead allowed, vs a shuffled-section-plan control and a causal planner; "
            "time-to-drop never an input. 9.5 factorised critic (H-K): bar-synchronous dual encoder, negatives "
            "= time-shift, other-song AND beat-trigger light; heads C_R/C_U; best-of-8 over flow samples "
            "through the conductor; gate = real-vs-beat-trigger accuracy above chance, additive projection "
            "changes the ranking, best-of-8 beats random-of-8 on human 'fits the music'; only then a reward, "
            "and then on conductor knobs, never the flow head. 9.6 Bradley-Terry over <= 12 knobs from "
            "200-400 brightness-matched pairs (H-L), two questions fitted separately; gate held-out accuracy "
            "> 60% at n=200, no BAS drift toward the metronome. 9.7 audio decomposition (H-G) gated by "
            "song-split beat-synchronous audits against the 92-ch bus with shuffled donor and L=0 "
            "counterpart: HTDemucs 4-stem envelopes/onsets/flux (threshold +0.02 AP on JD visual with a "
            "source-specific head/hands split), per-stem Burger flux, vocal activity (no lyrics), bar phase "
            "(no section labels), one frozen embedding layer group vs a same-width random projection and a "
            "song-ID leak probe (H-M). Not in Stage 9: AR light LM decoder, RL/DPO on the flow head before "
            "G9.5, type-MoE in the trunk, section/chord/LRA/timbre bus channels, Jukebox/CLAP, fixture-token "
            "decoder before console data, more dance as a first lever, product training on any of the research "
            "corpora, scraping. Oslo stays the concert test. Order: week 1 H-G audit || H-H pull+audit; week 2 "
            "H-I MVP; week 3 H-J, H-K; open: H-L (needs raters), H-M, console partnership."),
        "gates": [
            "G9.2 corpus gate: AP_audio - AP_shift >= +0.15 and AP_audio >= AP_beat-trigger + 0.05 on events and group-envelope R2, causal and offline",
            "G9.3 decoder gate (Value- and onset-matched): FID vs conductor beats shuffled donor by > 2x; BAS not below conductor without more onsets; A/B 'fits the music' CI excludes 50% at n >= 20; rig transfer beats channel-copy",
            "G9.4 planner gate: paired BAS not below no-planner; FID and 'feels structured' beat the shuffled-section plan and the causal planner",
            "G9.5 critic gate: real-vs-beat-trigger accuracy above chance; additive projection changes best-of-8 order; best-of-8 beats random-of-8 on human 'fits the music'",
            "G9.6 preference gate: held-out Bradley-Terry accuracy > 60% at n = 200; fitted knobs do not raise BAS toward the metronome; fresh batch prefers them to defaults",
            "G9.7 feature gate: each family beats the 92-ch bus by >= +0.02 AP (events) or >= +0.02 R2 on a song split with shuffled donor and L = 0 counterpart, else it stays out of the trunk"],
        "topic_tags": ["audio-to-light", "learned-conductor", "source-separation", "cross-modal",
                       "generative-modelling", "evaluation-metric", "offline-vs-online"]}))

# ---- hypotheses ---------------------------------------------------------------------------
H = []


def hyp(tag, **kw):
    r = ok(c.call("propose_hypothesis", dict(problem_id=P, **kw)))
    hid = r.get("id") or r.get("hypothesis_id")
    print(f"{tag}: {hid}"); H.append((tag, hid))


hyp("H-G", statement=(
    "Source-separated stems add lighting-relevant information the 92-channel mix bus collapses: "
    "HTDemucs drums/bass/vocals/other envelopes, onsets and 2-band flux raise held-out event AP on the "
    "Just Dance visual target by at least +0.02 over the bus with the time-shift control flat, and split "
    "the coach's head (drums-low) from hands (hats/vocal-high) at 2 s where mix flux cannot; vocal "
    "activity and bar phase are tested the same way; chords, loudness range and timbre are not."),
    predicted_effects=[
        {"diagnostic_id": "event-ap-lift-by-feature-family", "direction": "up",
         "magnitude_note": ">= +0.02 AP over the 92-ch bus on JD visual events, song split, shift control flat; per-stem flux gives a source-specific head/hands split"},
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "direction": "up",
         "magnitude_note": "+0.03 to +0.06 R2 at 2 s on hand speed vs footfall from the stem split; frame-wise block A stays ~0 (the corpus ceiling)"}],
    rationale=(
        "Burger 2013's mix-level low/high flux is already in the bus; what the mix collapses is WHICH source "
        "produced the flux (kick vs hat vs vocal sibilance). All-In-One feeds Demucs spectrograms for beat/"
        "downbeat/section; lighting operators cue kick vs snare vs vocal entries. Section labels as bus "
        "channels already failed (H-C), so only source-disambiguating families are tried. RPMC-L2 has no "
        "waveform, so the arms are priced on Just Dance visual, the coach split, the Oslo mixes and, after "
        "G9.2, Beat Saber."),
    plan=(
        "rb-ls2-09 step 9.7. Demucs htdemucs on the 28 h of JD wav (4-8 h on the 5070, chunked = the L=0 "
        "counterpart); per-stem RMS, onset, 2-band flux, vocal activity, downbeat phase; song-split beat-"
        "synchronous ridge / gradient boosting against the 92-ch bus; shuffled-donor and time-shift controls; "
        "nothing enters the Conformer until the audit beats the bus. Frozen MuQ/MERT layer dump (H-M) only "
        "against a same-width random projection and a song-ID kNN leak probe."),
    gates=[{"id": "G9.7-stems", "description": ">= +0.02 event AP over the 92-ch bus on JD visual, song split, shift control flat"},
           {"id": "G9.7-split", "description": "per-stem flux gives a source-specific head/hands split not available from mix flux"},
           {"id": "G9.7-causal", "description": "chunked (L = 0) stems reported next to whole-track stems"}],
    topic_tags=["source-separation", "audio-to-light", "cross-modal", "music-induced-movement"],
    problem_tags=["matched-baseline", "leaked-future"])

hyp("H-H", statement=(
    "Authored hobby/game light shows are lighting corpora, not tempo corpora: on xLights free .xsq "
    "sequences (where the song is on disk) and Beat Saber lightshow maps, audio predicts the group-level "
    "event stream and envelopes on held-out shows with AP_audio - AP_time-shift >= +0.15 and "
    "AP_audio >= AP_beat-trigger + 0.05, causal and offline. A corpus that fails stays out of every loss."),
    predicted_effects=[
        {"diagnostic_id": "event-ap-lift-audio-marginal", "direction": "up",
         "magnitude_note": "AP_audio - AP_shift >= +0.15 and >= beat-trigger + 0.05 on held-out shows; RPMC-L2 gave +0.31 ridge / 1.81x with hazard; JD visual 3.17x vs 1.48x"}],
    rationale=(
        "The deep-research pass found no public, commercially clean corpus of professional DMX programming; "
        "RPMC-L2 (699, mixed HSV at 10 Hz) is the only public stage-domain set. xLights sequences are the "
        "authored object itself (effects on groups at 25 ms, ~25-40 h free) and Beat Saber v3 maps carry "
        "group-level colour/rotation event boxes with phase-spread filters on an exact BPM grid (one map: "
        "348 boxes, 9082 events, 21 groups). Both could be beat-trigger-shaped rather than designed, which "
        "is what the gate decides. Research use only; no product training."),
    plan=(
        "rb-ls2-09 step 9.2. Pull ~300 top-rated BeatSaver lightshow maps and the free xLights sequences "
        "whose songs are on disk; convert to the Stage 9 format (group events + per-group envelopes at 40 "
        "fps; environment/model -> fixture role tables); rb-ls2-00b audit split by show; time-shift null in "
        "{+-2, +-4, +-8} s; conductor/beat_trigger.py as the baseline; Beat Saber gets a group-count control."),
    gates=[{"id": "G9.2-shift", "description": "AP_audio - AP_shift >= +0.15 on events and group-envelope R2"},
           {"id": "G9.2-bt", "description": "AP_audio >= AP_beat-trigger + 0.05"},
           {"id": "G9.2-L0", "description": "causal and offline reported together"}],
    topic_tags=["audio-to-light", "evaluation-metric", "cross-modal"],
    problem_tags=["matched-baseline", "ill-posed-ground-truth"])

hyp("H-I", statement=(
    "A learned group-effects decoder (Anton's mixture of light: a shared type-masked head on the frozen "
    "trunk and operator head emitting per-group intensity envelope, palette, effect id, rate in beats, "
    "phase spread and density, expanded to DMX by a deterministic effects engine) trained on authored "
    "light through a differentiable histogram renderer with random-rig dropout beats the hand-written "
    "conductor on Value- and onset-matched held-out songs: FID vs conductor beats a shuffled-donor control "
    "by > 2x, BAS is not lower without more onsets, raters prefer it on 'fits the music' with a CI "
    "excluding 50% at n >= 20, and it transfers to an unseen patch better than channel-copy."),
    predicted_effects=[
        {"diagnostic_id": "fid-light-kinetic", "direction": "down",
         "magnitude_note": "FID_light vs real below the conductor's by more than 2x the shuffled-donor gap; current 36.5 vs floor 8.1"},
        {"diagnostic_id": "human-preference-winrate", "direction": "up",
         "magnitude_note": "'fits the music' win rate vs the conductor with CI excluding 0.5 at n >= 20 brightness-matched pairs"},
        {"diagnostic_id": "rig-transfer-hist-distance", "direction": "down",
         "magnitude_note": "below channel-copy / drop-extras remapping on an unseen patch (different N, no strobe, extra LED bars)"},
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "none",
         "magnitude_note": "not below the conductor at matched onset density; BAS bought by flashing every beat is a fail"}],
    rationale=(
        "The conductor owns every artistic decision and the model only steers 30 channels that audio does "
        "not predict (block A R2 -0.22). Of the three mixture-of-light architectures, group effects + engine "
        "is how shows are programmed, is permutation- and mirror-safe by construction, and is the only one "
        "trainable from corpora without per-fixture DMX; SeqLight (arXiv:2605.03660) makes the same split "
        "(global HSV distribution, then venue-specific decomposition) and wins its human study 4.54 vs 3.90 "
        "vs 2.70. Type is an output mask, not an expert (MAS.S60: partitioned experts under-share)."),
    plan=(
        "rb-ls2-09 step 9.3. Freeze Conformer + operator head; 0.5-2 M causal head -> group-effect vector "
        "at 40 fps; reuse the expander with fixtures sorted by (type, x) and the fitted one-pole; torch "
        "histogram renderer (additive coloured blobs; spots as 2-D Gaussians); targets RPMC-L2 + JD visual "
        "(+ G9.2 corpora); per-type losses (wash: hue+value histogram, never absolute level; strobe: onset "
        "times and duty cycle); aux look-change CE; random-rig dropout; no conductor-DMX distillation; no "
        "pan/tilt learning. Eval on held-out songs with Value matched and onset density capped; split FID "
        "(hue / onset / full); shuffled-donor control; brightness-matched A/B with a 'too metronomic' "
        "question; rig-transfer test; causal and offline."),
    gates=[{"id": "G9.3-fid", "description": "FID vs conductor beats the shuffled-donor control by > 2x, Value- and onset-matched"},
           {"id": "G9.3-bas", "description": "Light-BAS not below the conductor without raising onset density"},
           {"id": "G9.3-ab", "description": "brightness-matched A/B 'fits the music' win-rate CI excludes 50% at n >= 20"},
           {"id": "G9.3-rig", "description": "unseen-patch HSV-histogram distance below channel-copy"}],
    topic_tags=["learned-conductor", "motion-to-light", "audio-to-light", "generative-modelling", "evaluation-metric"],
    problem_tags=["matched-baseline", "metric-gaming", "conditional-mean-collapse"])

hyp("H-J", statement=(
    "Section-level intent belongs on the conductor, not in the per-frame model: an audio-only offline "
    "planner that writes scene family, palette, intensity class, strobe density and transition type per "
    "section onto the conductor / engine knobs (one bar ahead allowed) beats the no-planner model and a "
    "shuffled-section-plan control on FID_light and on the human 'feels intentionally structured' "
    "question, with paired Light-BAS not lower. H-C refuted section channels as per-frame input; this is "
    "the untested lever."),
    predicted_effects=[
        {"diagnostic_id": "fid-light-kinetic", "direction": "down",
         "magnitude_note": "below the no-planner model AND below the shuffled-section-plan control (H-C: 36.5 -> 27.6 with shuffle at 33.1 was the trap)"},
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "none",
         "magnitude_note": "paired BAS not below the no-planner model (H-C lost -0.028)"},
        {"diagnostic_id": "human-preference-winrate", "direction": "up",
         "magnitude_note": "'feels intentionally structured' win rate vs no-planner with CI excluding 0.5"}],
    rationale=(
        "The largest measured effect in the project's data is the drop/breakdown movement contrast "
        "(1.68-2.53x on Oslo), and the product is pre-rendered so the planner may look ahead. Lodge-style "
        "two-level generation plans sparse global primitives and fills locally; here the primitives are "
        "looks and the filler is the existing frame model. Feeding the same information as per-frame "
        "channels hurt (H-C), which is a fact about where the information belongs, not about whether it "
        "exists. Raveform's expert DJ-mix structure supplies EDM section roles as conditioning."),
    plan=(
        "rb-ls2-09 step 9.4. Planner over bars/sections from the offline bus (Laplacian segmentation + EDM "
        "role labeller, Raveform roles where available); outputs = knobs of the conductor (now) or the 9.3 "
        "engine (later); frame model untouched; controls: shuffled-section plan (real plan on wrong "
        "sections), causal planner, no planner; time-to-drop never an input; held-out songs; FID, paired "
        "BAS, human A/B with the 'structured' question."),
    gates=[{"id": "G9.4-shuffle", "description": "FID and the 'structured' question beat the shuffled-section-plan control"},
           {"id": "G9.4-bas", "description": "paired Light-BAS not below the no-planner model"},
           {"id": "G9.4-causal", "description": "offline planner reported next to a causal planner"}],
    topic_tags=["audio-to-light", "learned-conductor", "offline-vs-online", "generative-modelling"],
    problem_tags=["leaked-future", "matched-baseline"])

hyp("H-K", statement=(
    "A factorised audio-light critic trained with time-shift, other-song AND beat-trigger negatives on "
    "bar-synchronous tokens learns something beyond beat and loudness: it separates real authored light "
    "from beat-trigger light on held-out songs above chance, its best-of-8 ranking over rectified-flow "
    "samples changes when the critic is projected to its nearest additive model, and best-of-8 beats "
    "random-of-8 on human 'fits the music'. Without the beat-trigger negative it collapses to a beat "
    "detector and rewards the mode the model already occupies."),
    predicted_effects=[
        {"diagnostic_id": "critic-pair-accuracy", "direction": "up",
         "magnitude_note": "real-vs-beat-trigger accuracy clearly above 0.5 on held-out songs; real-vs-shift reported alongside"},
        {"diagnostic_id": "human-preference-winrate", "direction": "up",
         "magnitude_note": "best-of-8 vs random-of-8 'fits the music' win rate with CI excluding 0.5"},
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "none",
         "magnitude_note": "best-of-8 must not raise BAS above the coach's +0.087 band; a critic that maximises BAS has hacked C_R"}],
    rationale=(
        "Contrastive alignment provably captures shared information only (MAS.S60 lecture 06, FactorCL); "
        "on this data that is beat + RMS, which is already the model's failure mode (BAS +0.135 vs coach "
        "+0.087). A two-head critic (redundant C_R, unique C_U) ranked by C_U - lambda C_R, with the "
        "additive-projection test from lecture 07, is the reranker form of FactorCL. Best-of-N through the "
        "conductor costs N tiny forward passes; reward use waits for the gate."),
    plan=(
        "rb-ls2-09 step 9.5. Dual encoder on bar tokens (audio = 92-ch bus; light = HSV histogram + look-"
        "change flag + strobe density); positives RPMC-L2, JD visual, G9.2 corpora; three negative families; "
        "instrument: real-vs-shift and real-vs-beat-trigger accuracy, correlation of C with RMS vs with "
        "look-change flags, additive projection; best-of-8 over block C samples scored through the "
        "conductor; human pairs best-of-8 vs random-of-8 on held-out songs; RPMC-L2 and JD reported separately."),
    gates=[{"id": "G9.5-bt", "description": "real-vs-beat-trigger accuracy above chance on held-out songs"},
           {"id": "G9.5-additive", "description": "additive projection changes the best-of-8 order"},
           {"id": "G9.5-bon", "description": "best-of-8 beats random-of-8 on human 'fits the music'"}],
    topic_tags=["cross-modal", "evaluation-metric", "audio-to-light", "generative-modelling"],
    problem_tags=["metric-gaming", "matched-baseline"])

hyp("H-L", statement=(
    "Two to four hundred brightness-matched pairwise ratings on held-out songs are enough to fit a "
    "Bradley-Terry model over at most 12 conductor knobs, separately for 'fits the music' and 'looks "
    "like an engineer did it', with held-out pair accuracy above 60% at n = 200, without pushing "
    "Light-BAS toward the metronome, and a fresh batch of raters prefers the fitted knobs to the defaults."),
    predicted_effects=[
        {"diagnostic_id": "preference-heldout-accuracy", "direction": "up",
         "magnitude_note": "> 0.60 at n = 200 decided pairs for at least one of the two questions; chance 0.5"},
        {"diagnostic_id": "human-preference-winrate", "direction": "up",
         "magnitude_note": "fitted knobs vs defaults on a fresh batch, CI excluding 0.5"},
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "none",
         "magnitude_note": "fitted knobs do not raise BAS above the current +0.135"}],
    rationale=(
        "No human preference sits anywhere in the loop; the 10-pair category test showed the two questions "
        "dissociate (8-1-1 vs a tie). Detectability: an 8-1 gap shows at ~30 decided pairs, a 60% true win "
        "rate needs ~200 to separate from 50%, 55% needs ~1000; K ~ 8-12 linear knobs recover main effects "
        "from 200-400 pairs. This is the cheapest way to put a human in the loop and the ratings are not "
        "spent on DPO of the flow head."),
    plan=(
        "rb-ls2-09 step 9.6. Generate knob-perturbed brightness-matched pairs with the render_song_demo "
        "path on held-out songs (over-bright foils kept); collect 200-400 ratings on the two questions; "
        "Bradley-Terry with ridge toward defaults; held-out accuracy; BAS of the fitted knobs; fresh-batch "
        "A/B fitted vs defaults."),
    gates=[{"id": "G9.6-acc", "description": "held-out Bradley-Terry accuracy > 60% at n = 200"},
           {"id": "G9.6-bas", "description": "fitted knobs do not raise BAS toward the metronome"},
           {"id": "G9.6-fresh", "description": "fresh raters prefer fitted knobs to defaults, CI excluding 50%"}],
    topic_tags=["learned-conductor", "evaluation-metric", "audio-to-light"],
    problem_tags=["metric-gaming"])

hyp("H-M", statement=(
    "One frozen music-foundation layer group (MuQ layers 9-12 or MERT-330M layers 10-13, plus MERT "
    "~15-16 for rhythm), PCA'd to 32-64 d and dumped offline, carries lighting-relevant information the "
    "92-channel bus lacks: it beats a same-width random projection of the bus on JD visual event AP and on "
    "Oslo crowd-speed correlation after the song-mean embedding is subtracted, and a song-ID kNN probe "
    "shows the gain is not a hash of the catalogue. Expected to fail; lowest priority."),
    predicted_effects=[
        {"diagnostic_id": "event-ap-lift-by-feature-family", "direction": "up",
         "magnitude_note": "residual >= +0.02 AP over the bus AND over the random projection after song-mean subtraction; else it is a song hash"}],
    rationale=(
        "Frozen SSL transfers redundant MIR (beat, timbre, coarse structure), not designer intent; H-C "
        "showed extra channels can hurt; pretraining corpora contain these pop hits (song-identity leak); "
        "1024-4800 d into a 9 M Conformer overfits; Jukebox does not fit 12 GB and CLAP is clip-level. "
        "Skip-BART used OpenL3, not these. Kept only because it is cheap to dump once."),
    plan=(
        "rb-ls2-09 step 9.7 item 5. Extract once on the JD wavs (2-6 h, 4-6 GB), resample to 40 fps, PCA "
        "32-64; song-split ridge vs the bus and vs a random projection of the same width; subtract the "
        "song-mean embedding; kNN song-ID leak probe; L = 0 chunked embeddings reported."),
    gates=[{"id": "G9.7-emb", "description": "residual >= +0.02 AP over the bus and over the random projection after song-mean subtraction"},
           {"id": "G9.7-leak", "description": "song-ID kNN probe does not explain the gain"}],
    topic_tags=["audio-to-light", "cross-modal", "evaluation-metric"],
    problem_tags=["leaked-future", "matched-baseline"])

# ---- problem update -----------------------------------------------------------------------
ok(c.call("update_problem", {"id": P, "add_update": (
    "2026-09-04 (evening) Stage 9 queued (rb-ls2-09-designed-light, PLAN.md rev. 8 section 6.9). Anton's "
    "reading of the shipped Just Dance model: decent, not human. Five outside passes with the full brief "
    "(Perplexity deep research x1, reasoning x4: audio decomposition, datasets in two halves, multimodal "
    "methods, mixture of light) plus Skip-BART (arXiv:2506.01482) and SeqLight (arXiv:2605.03660) read in "
    "full agree on the diagnosis: the bottleneck is the target (dance is a weak proxy; coaches agree at r "
    "0.22-0.42), the hand-written conductor, and the absence of human preference -- not the trunk, "
    "causality or the amount of dance. Deep-research verdict: 'another music-to-motion transformer will "
    "not close the gap; the project needs authored lighting intent, a learned hierarchical planner, and a "
    "preference model that tells deliberate composition from synchronised motion'. Datasets: no public, "
    "commercially clean corpus of professional DMX programming exists; RPMC-L2 (all 699 now cached, "
    "CC-BY-4.0, no waveform) is the only public stage-domain set; authored light exists at scale in xLights "
    "free .xsq sequences (~556, ~25-40 h, audio not included) and Beat Saber lightshow maps (v3 group "
    "event boxes with phase-spread filters, verified on the API); console show files only by contract. "
    "Seven hypotheses queued: H-G stems (audit-gated), H-H corpus predictability gate, H-I group-effects "
    "decoder (the mixture of light, SeqLight's split), H-J section planner on the knobs, H-K factorised "
    "critic with beat-trigger negatives as best-of-N, H-L Bradley-Terry knobs from 200-400 pairs, H-M "
    "frozen encoder vs random projection. Rule-outs added: AR light LM decoder, RL on the flow head before "
    "the critic gate, type-MoE in the trunk, section/chord/LRA/timbre bus channels, Jukebox/CLAP, "
    "fixture-token decoder before console data, product training on research corpora.")}))

lv = len(list(VAULT.rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print("hypotheses:", H)
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov, "optiHealth vault must not change"
c.close()
