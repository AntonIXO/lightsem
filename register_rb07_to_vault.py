#!/usr/bin/env python
"""Register runbook rb-ls2-07 for real.

queue_offline_pivot_to_vault.py tried on 2026-09-04 with `title` and `note`; the server
returned a validation error as a STRING, the script's ok() helper only checked for
{"ok": False}, and the failure passed silently. PLAN.md section 6.7 and the four H-A..H-D
hypotheses have referred to a runbook that did not exist since then.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from log_to_vault import Client

P = "dancing-light-generative-20260903"
RID = "rb-ls2-07-offline-prerendered"
if (Path("/root/lightsem-vault/runbooks") / f"{RID}.md").exists():
    print("already registered"); sys.exit(0)

c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
ov = len(list(Path("/root/vibescience-vault").rglob("*.md")))
r = c.call("register_runbook", {
    "id": RID, "problem_id": P, "priority": 1,
    "prompt": (
        "STAGE 7 -- OFFLINE PRE-RENDERED SHOW: the look-ahead axis as a designed variable "
        "(PLAN.md rev. 6 sections 0.1 and 6.7). 2026-09-04, Anton's call: the product is a "
        "pre-rendered show for a known track, not a live desk. Consequences, in order.\n"
        "1. Look-ahead is legal and must be labelled. Every number carries its position on the "
        "axis: L=0 causal, L=250 ms (a house delay line; IMAG already forces 50-150 ms), or "
        "whole track. A score is comparable only to a score at the same L; an offline number "
        "quoted without its L=0 counterpart on the same splits is not a result.\n"
        "2. extract_bus(causal=False) is the offline feature path and already exists.\n"
        "3. The five ORACLE_CHANNELS stop being a debt; a whole-track comb beat grid is correct "
        "offline and flags['oracle_beat'] marks a property, not a compromise.\n"
        "4. Stage 4's latency budget is no longer a product gate; it stays as the L=0 "
        "characterisation.\n"
        "5. Four hypotheses, in this order, do not reorder: H-A look-ahead ceiling (same 9.3 M "
        "Conformer, mask and causal padding off, causal=False bus; gates: offline block A R2 >= "
        "+0.04, offline Light-BAS >= +0.105, both next to their L=0 values); H-B zero-phase "
        "spectral matching (no retraining; diversity must not fall); H-C whole-track structure "
        "conditioning with a label-shuffle control; H-D operator head with anticipation (AP lift "
        "at +4 s > +0.05, CI excluding zero, shift control at 0). H-A decides whether the other "
        "three are worth running: if an offline model with the whole track cannot clear +0.105, "
        "the corpus binds and the next move is data (Stage 8), not architecture."),
    "gates": [
        "G7-A offline block A R2 >= +0.04 and offline Light-BAS >= +0.105, each quoted next to its L=0 counterpart",
        "G7-B light-event-rate-ratio within 1.3 of real-mapped, no channel above 2x corpus HF energy, FID below and diversity not below the unfiltered arm",
        "G7-C structure-conditioned Light-BAS above unconditioned with paired-bootstrap CI excluding zero; shuffled labels destroy the gain",
        "G7-D event AP lift at +4 s > +0.05 with CI excluding zero; offline lift above the causal +0.3145; shift control within 2 s of zero"],
    "topic_tags": ["audio-to-light", "offline-vs-online", "real-time", "generative-modelling"]})
assert isinstance(r, dict) and r.get("ok") is not False, r
print("registered:", r.get("id") or r)
lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
ov2 = len(list(Path("/root/vibescience-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {ov}->{ov2}")
assert ov2 == ov
c.close()
