#!/usr/bin/env python
"""Re-post the problem updates that update_problem(problem_id=...) silently
dropped (it wants `id`; the validation error came back as a string and the
calls were unchecked). Chronological, each dated. One-off, 2026-09-04."""
import sys
from pathlib import Path
sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"
UPDATES = [
    "2026-09-04 Stage 0b resolved in two halves (exp-002, exp-003). AIST++ cannot supply block A: genre "
    "explains 0.65-0.96 of between-track movement variance and Burger 2013 does not replicate, several "
    "correlations invert. Free dancing to EDM can: held-out-track R2 +0.30 for the shared movement curve on "
    "Oslo, drop/breakdown ratio 1.7-2.5x with Cohen's d up to 2.55. The binding constraint moves from 'is "
    "there a signal' to 'how much free-dance data can we get'.",
    "2026-09-04 implementation pass complete (exp-004..exp-008): descriptor extractor (mocap and video), "
    "constant-tempo beat grid, canonical-rig conductors, evaluation suite with nulls, 9.3 M causal Conformer "
    "with three heads, streaming inference, blind A/B harness; 14 of 14 verification checks. Stage 1 light "
    "gates fail (+0.017 vs real-mapped +0.216); operator head passes (+0.31 lift, z 3.66); latency budget met.",
    "2026-09-04 repair pass (exp-009): the Stage 1 audio bus read up to 375 ms of the future through librosa "
    "defaults; fixed, retrained, held-out block A R2 +0.0438 -> +0.0192, which is the size of the artefact. "
    "Event-rate calibration did not raise Light-BAS (predicted mechanism wrong). The surplus light events come "
    "from the sampled channels' 11-452x corpus HF energy. Product pivoted to the offline pre-rendered path "
    "(rb-ls2-07, PLAN rev. 6).",
    "2026-09-04 Stage 0c category test (exp-010): dance-derived hybrid light beats the beat trigger 8-1 "
    "(1 equal) on 'looks like a lighting engineer', ties 5-4 on 'on the music'; one rater, 10 brightness-matched "
    "AIST++ pairs. Category error rejected directionally; block C stays. Rater's tell for 'not an engineer' was "
    "change rate, the same defect the model carries.",
    "2026-09-04 Stage 8 queued (rb-ls2-08): 338 Just Dance gameplay captures (22.0 h, full songs, one filmed "
    "stylised coach or several -- each coach its own channel; research-only). Three signals: coach motion for "
    "block A/C, hand-animated background lighting as a designed show for HEAD B, full-song structure for H-C. "
    "Decision measurement is the held-out-song predictability audit. Extraction launched 15:00 with YOLO26m-pose "
    "at 960 after a six-clip trial (solos and multi-coach at >= 0.998 coverage per coach).",
    "2026-09-04 Stage 7 (rb-ls2-07), three of four hypotheses tested. H-A (exp-012): the look-ahead ceiling fails "
    "both gates -- with the whole track visible, held-out block A R2 on AIST++ is -0.009 against the causal +0.019 "
    "and the light gates do not move; the corpus binds, not the model. H-B (exp-011): spectral matching is the "
    "largest lever found -- a causal one-pole per channel fitted to the corpus above-6 Hz energy takes model "
    "Light-BAS from +0.018 to +0.293 (real-mapped +0.211), FID 67 -> 20, light events 86 -> 35 per clip against "
    "30 real, diversity not falling; the 12 Hz filter the hypothesis named does nothing, the mismatch sits near "
    "0.2 Hz; zero-phase at matched cutoffs scores the same, so the offline path removes lag, not jitter. H-D "
    "(exp-013): audio alone does not anticipate an operator's scene change 4 s ahead (offline margin -0.02, CI "
    "spanning zero); a 10x first result was a leak through target-derived hazard features under a symmetric "
    "window, caught by the shift control. H-C waits for the Just Dance corpus.",
]
c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))
for u in UPDATES:
    r = c.call("update_problem", {"id": P, "add_update": u})
    assert isinstance(r, dict) and r.get("ok") is not False, r
    print("posted:", u[:60])
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md"))); assert ov2 == ov
c.close()
