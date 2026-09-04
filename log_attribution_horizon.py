"""Log the attribution + horizon results. Both refine the event finding.

Reports the effect-size collapse honestly: z stays high at long horizons only
because the permutation null is itself negative and tight, while the absolute
marginal lift decays to ~0. High z with a null effect size is not a usable
anticipation window, and is recorded as such.
"""
import json

from log_to_vault import PROBLEM, Client

fam = json.load(open("data/cache/family_attribution.json"))
hz = json.load(open("data/cache/horizon_sweep.json"))
F, R = fam["families"], hz["rows"]

c = Client()
try:
    c.initialize()

    # ---------------------------------------------------------------- 1
    c.call("register_diagnostic", {
        "name": "event_ap_lift_by_feature_family",
        "description": (
            "Marginal lift in per-song AP over the rhythm baseline for predicting a "
            "lighting scene change, computed separately per interpretable audio feature "
            "family (energy / spectral / harmonic / timbre) and for the openl3 learned "
            "embedding. Each family judged against its OWN circular-shift null, because "
            "families differ in dimensionality and therefore in overfitting bias."),
        "unit": "lift", "direction": "higher_better",
    })

    h = c.call("propose_hypothesis", {
        "problem_id": PROBLEM,
        "statement": (
            "The audio->lighting-event signal is carried by LOUDNESS/ENERGY. If true, "
            "the industry's L2 threshold triggers are already at the ceiling on this "
            "data and a 'semantic' L3 model buys nothing."
        ),
        "rationale": (
            "The ICLR 2026 RPMC paper frames the field as 'rule-driven process or "
            "generative task'. Energy is what commercial FFT-threshold rigs "
            "(Resolume, SoundSwitch) already key on. If energy alone reproduces the "
            "event signal, there is no headroom to claim and the project should stop."
        ),
        "plan": (
            "Split audio into interpretable families computed from the RPMC feature "
            "groups: energy (mel-dB loudness + spectral flux), spectral "
            "(centroid/bandwidth/rolloff/contrast), harmonic (chroma-CENS + chroma "
            "change), timbre (MFCC), plus openl3 as the learned-embedding reference. "
            "Identical treatment for each: same causal multi-scale pooling + change "
            "features, same rhythm baseline, same 5-fold CV by song, and a per-family "
            "circular-shift null (25 reps) so differing dimensionality cannot be "
            "mistaken for signal. 140 songs."
        ),
        "predicted_effects": [
            {"diagnostic_id": "event-ap-lift-by-feature-family", "direction": "up",
             "magnitude_note": "If the energy family alone reaches z>2 vs its null and "
                               "no non-energy family does, the hypothesis is supported "
                               "and the project has no L3 headroom."},
        ],
        "topic_tags": ["audio-to-light"],
    })
    hid = h["id"]
    e = c.call("start_experiment", {
        "hypothesis_id": hid,
        "config_note": ("probe_feature_families.py on 140 songs, 5-fold CV by song, "
                        "25 permutation reps per family, horizon 1.0 s. Features via "
                        "fetch_interp_feats.py (HTTP range reads, 2366 MB, nothing "
                        "downloaded)."),
    })
    eid = e["id"]
    c.call("record_diagnostics", {
        "experiment_id": eid,
        "measurements": [
            {"diagnostic_id": "event-ap-lift-by-feature-family",
             "before": F["energy"]["null_mean"], "after": F["energy"]["lift"]},
        ],
    })
    r = c.call("close_experiment", {
        "experiment_id": eid,
        "notes": f"""REFUTED, and the refutation is GOOD NEWS for the project.

Energy is the ONLY family that fails to beat its own null. Marginal lift over
the rhythm baseline ({fam['rhythm_baseline']:.4f}x), 140 songs, horizon 1.0 s:

  family      dim     lift      marg        null            z       p
  energy        2   {F['energy']['lift']:.4f}   {F['energy']['marginal']:+.4f}   {F['energy']['null_mean']:.4f}+-{F['energy']['null_sd']:.4f}   {F['energy']['z']:+.2f}   {F['energy']['p']:.3f}
  harmonic      2   {F['harmonic']['lift']:.4f}   {F['harmonic']['marginal']:+.4f}   {F['harmonic']['null_mean']:.4f}+-{F['harmonic']['null_sd']:.4f}   {F['harmonic']['z']:+.2f}   {F['harmonic']['p']:.3f}
  openl3      512   {F['openl3']['lift']:.4f}   {F['openl3']['marginal']:+.4f}   {F['openl3']['null_mean']:.4f}+-{F['openl3']['null_sd']:.4f}   {F['openl3']['z']:+.2f}   {F['openl3']['p']:.3f}
  spectral     10   {F['spectral']['lift']:.4f}   {F['spectral']['marginal']:+.4f}   {F['spectral']['null_mean']:.4f}+-{F['spectral']['null_sd']:.4f}   {F['spectral']['z']:+.2f}   {F['spectral']['p']:.3f}
  timbre      128   {F['timbre']['lift']:.4f}   {F['timbre']['marginal']:+.4f}   {F['timbre']['null_mean']:.4f}+-{F['timbre']['null_sd']:.4f}   {F['timbre']['z']:+.2f}   {F['timbre']['p']:.3f}

THREE FINDINGS:

1. ENERGY IS NOT SIGNIFICANT (z={F['energy']['z']:+.2f}, p={F['energy']['p']:.3f}). Loudness does NOT
   explain when a lighting operator changes the scene. This directly contradicts
   the assumption behind FFT-threshold rigs. The industry's L2 approach is not
   at the ceiling here -- it is keying on the wrong feature.

2. TIMBRE (MFCC, z={F['timbre']['z']:+.2f}) AND SPECTRAL (z={F['spectral']['z']:+.2f}) CARRY THE SIGNAL.
   Operators respond to a change in the CHARACTER of the sound -- distortion
   coming in, a section changing texture -- not to it getting louder. On
   Rock/Punk/Metal livehouse material that is musically sensible: these genres
   are loud throughout, so loudness is uninformative while timbre is not.
   This is the specific, defensible claim that an L3 semantic model has
   headroom, and it NAMES the features that model must encode.

3. openl3 (z={F['openl3']['z']:+.2f}) is BEATEN by 10 hand-picked spectral descriptors
   (z={F['spectral']['z']:+.2f}) despite having 512 dimensions. The learned general-purpose
   embedding is not the right representation for this task. A bespoke encoder
   over spectral/timbral descriptors is better motivated than fine-tuning
   openl3 -- and it is far cheaper.

CAVEAT: 'timbre' here is MFCC. RPMC-L2 ships no zero-crossing-rate despite an
earlier schema note claiming it, so no separate percussive family was testable.
MFCC and the spectral descriptors are correlated; this probe establishes that
NON-ENERGY spectral character carries the signal, not which of the two is
primary.""",
    })
    print("attribution ->", json.dumps(r)[:260])

    # ---------------------------------------------------------------- 2
    c.call("register_diagnostic", {
        "name": "event_ap_lift_vs_horizon",
        "description": (
            "Marginal lift over the rhythm baseline for predicting a lighting scene "
            "change within the next H seconds, swept over H. Each horizon has its own "
            "circular-shift null because the base rate grows with H. Distinguishes an "
            "anticipatory signal (survives at multi-second H) from a reactive one "
            "(only significant when the event is imminent)."),
        "unit": "lift", "direction": "higher_better",
    })
    h2 = c.call("propose_hypothesis", {
        "problem_id": PROBLEM,
        "statement": (
            "The audio->lighting-event coupling is ANTICIPATORY: audio predicts an "
            "event several seconds before it happens, so a rig could LEAD the music "
            "rather than react to it. This is the core L3 product claim."
        ),
        "rationale": (
            "A single significant horizon is compatible with anticipation, with pure "
            "reaction (which threshold triggers already do, offering no headroom), and "
            "with an unstable estimator. Only the SHAPE of lift versus horizon "
            "separates them."
        ),
        "plan": ("Sweep horizon over 0.2/0.5/1.0/2.0/4.0/8.0 s. Per-horizon "
                 "circular-shift null, 20 reps, 5-fold CV by song, 140 songs, "
                 "1376 events. Report both z AND absolute marginal lift, since a "
                 "significant z with a null effect size is not a usable window."),
        "predicted_effects": [
            {"diagnostic_id": "event-ap-lift-vs-horizon", "direction": "up",
             "magnitude_note": "Anticipation requires a materially POSITIVE marginal "
                               "lift at horizons >= 4 s, not merely z>2."},
        ],
        "topic_tags": ["audio-to-light"],
    })
    hid2 = h2["id"]
    e2 = c.call("start_experiment", {
        "hypothesis_id": hid2,
        "config_note": "probe_horizon_sweep.py, 140 songs, 1376 events, 20 perms/horizon.",
    })
    eid2 = e2["id"]
    c.call("record_diagnostics", {
        "experiment_id": eid2,
        "measurements": [
            {"diagnostic_id": "event-ap-lift-vs-horizon",
             "before": R["4.0s"]["null_mean"], "after": R["4.0s"]["marginal"]},
        ],
    })
    r2 = c.call("close_experiment", {
        "experiment_id": eid2,
        "notes": f"""MIXED -- and the honest reading is REACTIVE, not anticipatory.

  horizon   base    marginal lift        null              z       p
    0.2s   {R['0.2s']['base_rate']:.3f}     {R['0.2s']['marginal']:+.4f}      {R['0.2s']['null_mean']:+.4f}+-{R['0.2s']['null_sd']:.4f}    {R['0.2s']['z']:+.2f}   {R['0.2s']['p']:.3f}
    0.5s   {R['0.5s']['base_rate']:.3f}     {R['0.5s']['marginal']:+.4f}      {R['0.5s']['null_mean']:+.4f}+-{R['0.5s']['null_sd']:.4f}    {R['0.5s']['z']:+.2f}   {R['0.5s']['p']:.3f}
    1.0s   {R['1.0s']['base_rate']:.3f}     {R['1.0s']['marginal']:+.4f}      {R['1.0s']['null_mean']:+.4f}+-{R['1.0s']['null_sd']:.4f}    {R['1.0s']['z']:+.2f}   {R['1.0s']['p']:.3f}
    2.0s   {R['2.0s']['base_rate']:.3f}     {R['2.0s']['marginal']:+.4f}      {R['2.0s']['null_mean']:+.4f}+-{R['2.0s']['null_sd']:.4f}    {R['2.0s']['z']:+.2f}   {R['2.0s']['p']:.3f}
    4.0s   {R['4.0s']['base_rate']:.3f}     {R['4.0s']['marginal']:+.4f}      {R['4.0s']['null_mean']:+.4f}+-{R['4.0s']['null_sd']:.4f}    {R['4.0s']['z']:+.2f}   {R['4.0s']['p']:.3f}
    8.0s   {R['8.0s']['base_rate']:.3f}     {R['8.0s']['marginal']:+.4f}      {R['8.0s']['null_mean']:+.4f}+-{R['8.0s']['null_sd']:.4f}    {R['8.0s']['z']:+.2f}   {R['8.0s']['p']:.3f}

The script's own automated verdict said 'ANTICIPATORY' because z stays above 2
at 4 s and 8 s. THAT VERDICT IS REJECTED. Read the effect size, not the z:

  marginal lift {R['0.2s']['marginal']:+.4f} (0.2s) -> {R['1.0s']['marginal']:+.4f} (1.0s) -> {R['4.0s']['marginal']:+.4f} (4.0s) -> {R['8.0s']['marginal']:+.4f} (8.0s)

The effect DECAYS BY ~99% BY 4 s AND GOES NEGATIVE BY 8 s. z stays high only
because the permutation null is itself negative and its spread tightens as the
base rate grows, so 'not as bad as the null' registers as significant even when
the absolute lift is zero. High z with a null effect size is not a usable
anticipation window.

The z-based rule in probe_horizon_sweep.py is therefore WRONG and should gate
on absolute marginal lift instead. Recorded rather than silently patched, since
the flawed rule produced a claim that had to be overturned by hand.

WHAT THIS MEANS: the coupling is real (permutation-validated at every horizon)
but SHORT-RANGE. Audio tells you an event is happening within ~1-2 s, not that
one is coming in 4-8 s. Peak lift is at the SHORTEST horizon tested (0.2 s,
{R['0.2s']['marginal']:+.4f}x), which is the signature of detection, not prediction.

CONSEQUENCE FOR THE PRODUCT: the original L3 pitch -- light that LEADS a
build-up -- is NOT supported by this data at the 4-8 s lead times that pitch
implies. What IS supported is smarter DETECTION: reacting to timbral change
rather than loudness, at ~1 s latency. That is a real improvement over
threshold triggers (which key on the feature that this data says does not
work) but it is a weaker claim than anticipation and must not be sold as one.

The anticipation the DSP baseline demonstrated on synthetic EDM (2.78 s lead)
does not transfer here. RPMC-L2 is Rock/Punk/Metal livehouse material, which
lacks the long telegraphed build-ups that make EDM anticipation possible. The
lead-time claim may be genre-specific and remains untested on EDM with real
lighting ground truth.""",
    })
    print("horizon ->", json.dumps(r2)[:260])
finally:
    c.close()
