#!/usr/bin/env python
"""Queue Stage 8 (the Just Dance corpus) in the vault: one runbook, two hypotheses,
the one peer-reviewed paper built on this material. Nothing is measured here.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))


def ok(r):
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


PAPER_SLUG = "switch-justdance-a-music-to-dance-corpus-from-nintendo-switch-just-dance-gameplay-via-gvhmr"
already = (Path("/root/lightsem-vault/papers") / f"{PAPER_SLUG}.md").exists()
paper = {"paper_id": PAPER_SLUG} if already else c.call("add_paper", {
    "title": "Switch-JustDance: a music-to-dance corpus from Nintendo Switch Just Dance gameplay via GVHMR",
    "arxiv_id_or_url": "arXiv:2511.17925",
    "key_claims": [
        "HDMI capture of Just Dance gameplay reconstructed to SMPL with GVHMR at about 200 ms per frame; resampled to 30 Hz",
        "causal smoothing adds phase lag to the reconstructed motion",
        "the game scores one Joy-Con, so player motion can cheat with the upper body; the coach video is the choreography, not the player",
        "no detector-failure analysis for stylised colour, UI occlusion or multi-coach maps is published"],
    "topic_tags": ["music-to-dance", "pose"]})
pid = paper.get("paper_id") or paper.get("id") if isinstance(paper, dict) else None
print("paper:", pid, "(existing)" if already else paper if not pid else "")
if pid and not already:
    c.call("link_paper", {"entity_id": P, "paper_id": pid})

ok(c.call("register_runbook", {
    "id": "rb-ls2-08-justdance", "problem_id": P, "priority": 2,
    "prompt": (
        "STAGE 8 -- JUST DANCE CORPUS: coach motion, designed background visuals, full-song "
        "structure (PLAN.md rev. 7 section 6.8). Corpus: 338 YouTube gameplay captures of Just Dance "
        "1-2020, 22.0 h, full commercial songs (median 236 s), one filmed and stylised coach per "
        "routine (5 multi-coach files), 720p/1080p, coach video at 30 fps whatever the capture rate, "
        "audio on all files (6 silent), some tracks Ubisoft in-house covers. Manifest at "
        "/mnt/lightsem-data/cache/justdance_manifest.json. RESEARCH ONLY (YouTube-sourced, Ubisoft "
        "choreography and art); usable for a paper, not a product training set.\n"
        "Why it is different from AIST++: full songs with verses, choruses and drops, and choreography "
        "made to that song section by section -- a designed response to structure, which is what "
        "H-C (rb-ls2-07) needs and AIST++'s structure-less loops cannot give. Why it might not be: "
        "if between-song movement is explained by genre or the wiki's Difficulty/Effort rating rather "
        "than by audio, it trains a genre prior like AIST++ did.\n"
        "Three signals, tested in order: (1) coach motion -> block A/C targets through the 2D pose "
        "path (vertical/energy family only: vert_bounce_amp, vert_period_s, com_speed, qom, "
        "head_speed, spread; hip_yaw_speed and footfall_rate masked from the loss); (2) background "
        "visuals with coach and UI masked -> per-frame brightness and HSV histograms in RPMC-L2's "
        "format, scene-change and palette events, the operator-head test unchanged (Ubisoft's video "
        "artists hand-animate the lighting to the music; Gold Move flashes mark chorus/drop starts in "
        "JD2015-2018); (3) whole-song structure labels for H-C, with the coach's chorus/verse energy "
        "contrast and the background's brightness contrast as two independent readings of the "
        "sections.\n"
        "Steps: 8.1 usability manifest (drop 6 silent files; song-key dedupe -- 15 duplicate groups, "
        "30 files; per-file fps, start_s, coverage, n_coaches; resample 60 fps captures to 30; mask "
        "pictogram strip, score band, lyrics, star meter before detection; Gold Move frames "
        "low-confidence; multi-coach files skipped). 8.2 predictability audit on held-out SONGS, the "
        "rb-ls2-00b recipe unchanged, plus genre/effort partialled out and chorus/verse ratio. "
        "8.3 sync check: vertical-bounce phase against the beat grid should sit at one phase across "
        "files; a fitted offset must transfer to a channel it was not fitted on. 8.4 visual-show test "
        "on held-out songs. 8.5 train the offline trunk on AIST++ + Just Dance with a corpus token, "
        "Oslo held out entirely as the concert-domain test. Cost: pose at 72 frames/s on the RTX 5070 "
        "at imgsz 960 -> 1 h screening pass, ~9 h full extraction; neither starts without Anton's "
        "word. Split of labour: the parallel session owns video -> descriptors; this session owns the "
        "audio bus on the captures, structure labels, the audit, the trunk and the vault."),
    "gates": [
        "G8.1 >= 250 usable single-coach songs with coverage >= 0.8 after masking",
        "G8.2 held-out-song audio R2 >= +0.20 on the vertical/energy family at 2 s; genre/effort-partialled correlation not within +/-0.1 of zero; chorus/verse movement ratio CI clear of 1",
        "G8.4 visual-show event AP lift >= +0.09, z > 3, shift control peaking at 0",
        "G8.5 held-out-song block A R2 above the AIST++-only model AND Oslo shared-movement R2 not below it"],
    "topic_tags": ["music-to-dance", "pose", "audio-to-light", "music-induced-movement"]}))

h1 = ok(c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "Just Dance coach motion is predictable from audio on held-out songs in a way AIST++ "
        "motion was not, because the choreography is made to the song's structure rather than "
        "improvised to a structure-less loop: held-out-song R2 for the vertical/energy family "
        "reaches +0.20 or more at 2 s and survives partialling out genre and the wiki's Effort "
        "rating. If it collapses under genre as AIST++ did, the coach motion is a genre prior and "
        "only the background visuals and the structure labels survive from the corpus."),
    "predicted_effects": [
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "direction": "up",
         "magnitude_note": "held-out-song audio R2 >= +0.20 at 2 s, against AIST++'s <= 0.21 collapsing to +/-0.21 under genre and Oslo's +0.30"}],
    "rationale": (
        "Stage 0b: AIST++ movement is explained by genre (0.65-0.96 of between-track variance) and "
        "audio-movement correlations collapse when genre and tempo are partialled out; Oslo free "
        "dancing to EDM gives +0.30 on held-out tracks because the music has structure and the "
        "dancers respond to it. Just Dance choreography is designed section by section to full "
        "commercial songs, so the structure response is built in; but it is one performer per song "
        "and the song's genre/difficulty could still explain the between-song variance."),
    "plan": (
        "rb-ls2-08 step 8.2. Descriptors from the coach through pose2d (masked UI, 30 fps), audio "
        "bus (causal and offline) and beat grid on the capture audio, split by song key. Ridge "
        "audio -> movement at 0.5/2/8 s, R2 on held-out songs against a song-identity baseline; "
        "same with genre and Effort partialled out; chorus/verse movement ratio with bootstrap CI."),
    "gates": [
        {"id": "G8.2-r2", "description": "held-out-song audio R2 >= +0.20 on the vertical/energy family at 2 s"},
        {"id": "G8.2-genre", "description": "genre/effort-partialled audio-movement correlation not within +/-0.1 of zero"},
        {"id": "G8.2-contrast", "description": "chorus/verse movement ratio with CI clear of 1"}],
    "topic_tags": ["music-to-dance", "music-induced-movement", "pose", "cross-modal"],
    "problem_tags": ["ill-posed-ground-truth"]}))
print("H-E:", h1.get("id") or h1.get("hypothesis_id"))

h2 = ok(c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "The Just Dance background, with coach and UI masked, is a designed light show whose "
        "scene changes and palette shifts are predictable from audio in the same way a human "
        "operator's are on RPMC-L2: audio's marginal lift over the rhythm-of-past-changes "
        "baseline clears +0.09 with z > 3 on held-out songs, and the time-shift control peaks at 0."),
    "predicted_effects": [
        {"diagnostic_id": "event-ap-lift-audio-marginal", "direction": "up",
         "magnitude_note": ">= +0.09 lift, z > 3, on held-out songs; RPMC-L2 gave +0.3145"}],
    "rationale": (
        "Ubisoft's video artists hand-animate the background lighting to the music and Gold Move "
        "flashes in JD2015-2018 tend to land at the start of a chorus or drop. That is a designed "
        "audio -> visual mapping in the same data format as RPMC-L2's HSV histograms, 22 h of it "
        "on pop/EDM/latin, next to 699 concert shows. If it holds, HEAD B has a second corpus; "
        "if it fails, the backgrounds are decorative and stay out of the loss."),
    "plan": (
        "rb-ls2-08 step 8.4. Mask coach (pose bounding box dilated) and the four UI regions; "
        "per-frame mean brightness and HSV histograms at 30 fps in RPMC-L2's layout; the existing "
        "operator-head event detector and rhythm baseline unchanged; held-out songs; time-shift "
        "control +/-30 s within song."),
    "gates": [
        {"id": "G8.4-lift", "description": "event AP lift >= +0.09 with z > 3 on held-out songs"},
        {"id": "G8.4-hue", "description": "hue KL to the true palette below the per-song mean baseline"},
        {"id": "G8.4-shift", "description": "time-shift control peaks within 2 s of zero"}],
    "topic_tags": ["audio-to-light", "evaluation-metric", "cross-modal"],
    "problem_tags": ["matched-baseline"]}))
print("H-F:", h2.get("id") or h2.get("hypothesis_id"))

c.call("update_problem", {"problem_id": P, "add_update": (
    "2026-09-04 Stage 8 queued (rb-ls2-08): Anton downloaded 338 Just Dance gameplay captures "
    "(22.0 h, full songs, one stylised filmed coach, research-only). Three signals: coach motion "
    "for block A/C, hand-animated background lighting as a designed show for HEAD B, full-song "
    "structure for H-C. Decision measurement is the held-out-song predictability audit; if it "
    "collapses under genre as AIST++ did, only the visuals and structure labels survive.")})

lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov, "optiHealth vault must not change"
c.close()
