"""Correct two auto-computed verdicts that are misleading.

close_experiment() COMPUTES the verdict from predicted_effects vs recorded
measurements. Both experiments got verdict='supports' because the diagnostic I
recorded moved 'up' relative to its null -- but in both cases the HYPOTHESIS AS
STATED was refuted:

  1. "energy carries the signal" -> energy z=+0.66, NOT significant. The
     diagnostic went up only because energy's raw lift exceeds its null mean
     slightly. The hypothesis is refuted; other families carry it.
  2. "the coupling is anticipatory" -> effect size decays to ~0 by 4 s and goes
     negative at 8 s. Significant z, null effect. Refuted as an ANTICIPATION
     claim.

This is a real trap in the tool, not a mistake to hide: a single scalar
diagnostic cannot express "the number moved but the claim failed". The
mechanism that exists for exactly this is GATES -- a failing blocking gate
downgrades supports -> directional_only. I did not preregister gates on these
two, so nothing could downgrade them.

Fix: record the gates that SHOULD have been preregistered, as failed, with the
evidence. Also update_problem so the problem statement carries the corrected
reading rather than leaving 'supports' as the last word.
"""
import json

from log_to_vault import PROBLEM, Client

fam = json.load(open("data/cache/family_attribution.json"))
hz = json.load(open("data/cache/horizon_sweep.json"))
F, R = fam["families"], hz["rows"]

c = Client()
try:
    c.initialize()
    # update_problem takes id + description, and description REPLACES rather
    # than appends -- so read the current text and extend it.
    cur = c.call("get_problem", {"id": PROBLEM})
    desc = cur.get("description", "") if isinstance(cur, dict) else ""
    print(f"current description: {len(desc)} chars")

    addendum = f"""

---
CORRECTED READING of two auto-computed 'supports' verdicts
(experiments '...carried-by-loudness-energy...' and '...is-anticipatory...').

Both verdicts are ARTEFACTS of recording a single scalar diagnostic without a
preregistered gate. close_experiment computes the verdict from predicted_effect
direction vs measurement, so a number that moved up reads as 'supports' even
when the stated claim failed. Treat both as REFUTED:

1. "Energy carries the audio->lighting-event signal" -- REFUTED.
   energy z={F['energy']['z']:+.2f} (p={F['energy']['p']:.3f}), the ONLY family that fails its null.
   Carried instead by timbre/MFCC z={F['timbre']['z']:+.2f} and spectral z={F['spectral']['z']:+.2f}.
   openl3 (512-d, learned) z={F['openl3']['z']:+.2f} is BEATEN by 10 hand-picked spectral
   descriptors. Consequence: threshold rigs key on the wrong feature, an L3
   model has real headroom, and the encoder should be built over spectral/
   timbral descriptors rather than fine-tuned from openl3.

2. "The coupling is anticipatory" -- REFUTED as an anticipation claim.
   marginal lift {R['0.2s']['marginal']:+.4f} (0.2s) -> {R['1.0s']['marginal']:+.4f} (1.0s) -> {R['4.0s']['marginal']:+.4f} (4.0s) -> {R['8.0s']['marginal']:+.4f} (8.0s).
   ~99% decay by 4 s, negative by 8 s. z stays >2 only because the null is
   itself negative and tightens as the base rate grows. Peak lift is at the
   SHORTEST horizon -- the signature of DETECTION, not prediction.
   Consequence: the 'light that leads the build-up' pitch is NOT supported on
   this corpus at 4-8 s lead. What is supported is better detection at ~1 s
   latency using timbral rather than loudness features.

STANDING RESULT (unaffected): audio->lighting-EVENT coupling is real,
z=+2.83 p=0.000 at 1.0 s vs a circular-shift null, 140 songs / 1376 events,
while audio->lighting-LEVEL is refuted (marginal -0.005 vs persistence).

PROCESS LESSON: preregister a GATE for every claim, not just a diagnostic
direction. A failing blocking gate downgrades supports -> directional_only,
which is exactly the mechanism these two needed and did not have."""

    r = c.call("update_problem", {"id": PROBLEM, "description": desc + addendum})
    print("update_problem ->", json.dumps(r)[:200])
finally:
    c.close()
