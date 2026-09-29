#!/usr/bin/env python
"""Queue the offline / pre-rendered pivot: two diagnostics and four hypotheses.

Anton's call on 2026-09-04: the product is a PRE-RENDERED show, not a live desk.
The track is known before the show, so look-ahead is no longer a leak to be
hunted -- it is an axis to be designed on and reported. Nothing here is run;
this writes the queue so the next session starts from a preregistered plan
rather than from whichever idea is nearest.

The causal path is not abandoned. It becomes the labelled L=0 end of the
look-ahead axis, and every offline number has to be quoted next to it, because
the size of that gap is the only honest statement of what the pre-rendering
buys.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/root/lightsem")
from scripts.vault.log_to_vault import Client

P = "dancing-light-generative-20260903"


def ok(r):
    # A validation error comes back as a STRING, not as {"ok": False}. The first
    # version of this helper only checked the dict case and let the
    # register_runbook call below fail silently (its `title` field is not
    # accepted); rb-ls2-07 was registered a second time by scripts/vault/register_rb07_to_vault.py.
    assert isinstance(r, dict) and r.get("ok") is not False, r
    return r


c = Client(); assert c.initialize()["name"] == "vibescience-lightsem"
before = len(list(Path("/root/vibescience-vault").rglob("*.md")))

# ---------------------------------------------------------------- diagnostics
# Both of these would have caught a failure that took a day to find by hand.

ok(c.call("register_diagnostic", {
    "id": "light-event-rate-ratio",
    "name": "light_event_rate_over_real_mapped",
    "unit": "ratio",
    "direction": "neutral",
    "topic_tags": ["evaluation-metric", "audio-to-light"],
    "description": (
        "Visible light events per minute in the rendered DMX, divided by the same count "
        "for light mapped from the real dance on the same clips. Direction is neutral "
        "because 1.0 is the target and both sides of it are failures: too few events is "
        "dead light, too many is a train that cannot beat its own null.\n\n"
        "Registered because Light-BAS cannot be read without it. A dense event train has a "
        "HIGH circular-shift null by construction, so a model can place every flash "
        "correctly and still score at its own null purely by flashing too often. On "
        "2026-09-04 the model sat at 2.9 (85.8 events per clip against 30.1) while its "
        "Light-BAS delta was +0.018, and the density was invisible in every number being "
        "reported until the per-clip counts were read side by side.")}))

ok(c.call("register_diagnostic", {
    "id": "descriptor-hf-energy-ratio",
    "name": "descriptor_above_6hz_energy_over_corpus",
    "unit": "ratio",
    "direction": "neutral",
    "topic_tags": ["evaluation-metric", "generative-modelling", "motion-to-light"],
    "description": (
        "Share of a predicted descriptor channel's variance above 6 Hz, divided by the "
        "share the real corpus channel has, averaged over held-out clips. 1.0 is the "
        "target; above 1 is jitter the body never produced.\n\n"
        "The targets are mocap through a 12 Hz zero-phase Butterworth and are physically "
        "smooth. A flow head sampled per frame is not, even conditioned on its own previous "
        "output: measured 2026-09-04 the sampled block ran 11x to 452x the corpus value "
        "(section_energy 116x, space_eff 202x, weight 343x, hand_spread 452x) while the "
        "regressed block sat at 0.3-5x. That excess is what crosses the light's visibility "
        "threshold and drives light-event-rate-ratio.")}))

# ----------------------------------------------------------------- hypotheses
# Ordered. H-A first, because if it fails the other three are not worth running.

hA = ok(c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "Most of the gap between the model's light (+0.018 against its null) and light "
        "mapped from the real dance (+0.211) is caused by the causality constraint, not by "
        "the corpus. The same trunk with unmasked attention over the whole track will close "
        "at least half of it."),
    "predicted_effects": [
        {"diagnostic_id": "descriptor-audio-r2-over-identity", "direction": "up",
         "magnitude_note": "held-out block A R2 at least doubles, +0.0192 -> >= +0.04"},
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "up",
         "magnitude_note": "model delta >= 0.5 x real-mapped, i.e. >= +0.105"}],
    "rationale": (
        "This is a ceiling measurement and it has to come first, because it decides whether "
        "the next month is spent on architecture or on data. Stage 0b already showed the "
        "corpus is weak: genre alone explains 0.65-0.96 of between-track movement variance "
        "in AIST++, and removing the accidental 375 ms look-ahead cost held-out R2 "
        "+0.0438 -> +0.0192, which says look-ahead was worth a lot on this data. If an "
        "offline model with the entire track still cannot beat +0.105, then nothing about "
        "the model is the binding constraint and the answer is more free-dance-to-EDM data, "
        "not a better head."),
    "plan": (
        "One training run. Same 9.3 M Conformer, same 749k frames, same splits, same 12k "
        "steps, same seed; the only change is that the attention mask and the causal "
        "convolution padding are removed, and extract_bus is called with causal=False so "
        "the features are the centred, offline-correct versions librosa gives by default. "
        "Score on the identical Stage 1 protocol and quote the causal number beside it as "
        "the look-ahead tax. Sweep L in {0, 250 ms, whole track} if the endpoints differ, "
        "since 250 ms is buyable with a house delay line and the rest is not."),
    "gates": [
        {"id": "G-ceiling-r2", "description":
         "offline held-out block A R2 >= +0.04 (double the causal value)"},
        {"id": "G-ceiling-bas", "description":
         "offline model Light-BAS delta >= +0.105, half of real-mapped"},
        {"id": "G-tax-reported", "description":
         "the causal and offline numbers are quoted together on the same splits; an "
         "offline number alone is not a result"}],
    "topic_tags": ["audio-to-light", "offline-vs-online", "generative-modelling"],
    "problem_tags": ["matched-baseline"],
}))
print("H-A", hA.get("id"))

hB = ok(c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "The model's light flashes three times too often because its sampled descriptors "
        "carry 11x to 452x the corpus's above-6-Hz energy. Filtering the model's output "
        "with the same zero-phase 12 Hz Butterworth the targets were built with will bring "
        "the light event rate to the corpus value and lower FID, without costing diversity."),
    "predicted_effects": [
        {"diagnostic_id": "light-event-rate-ratio", "direction": "down",
         "magnitude_note": "2.9 -> within 1.3 of the real-mapped rate"},
        {"diagnostic_id": "descriptor-hf-energy-ratio", "direction": "down",
         "magnitude_note": "worst channel 452x -> within 2x of corpus"},
        {"diagnostic_id": "fid-light-kinetic", "direction": "down",
         "magnitude_note": "below the 67.29 the causal model reaches"},
        {"diagnostic_id": "light-diversity-kinetic", "direction": "up",
         "magnitude_note": "must not fall; killing a common noise floor should make clips "
                           "more distinguishable, not less"}],
    "rationale": (
        "Measured, not guessed: the regressed block sits at 0.3-5x the corpus HF energy and "
        "the sampled block at 11-452x, so this is a property of per-frame sampling and not "
        "of the trunk. ar_flow was already enabled, so conditioning on the previous output "
        "is not sufficient. A causal one-pole was written and is the online fallback, but "
        "offline the exact filter the targets were made with is available, and matching the "
        "target's own construction is a stronger argument than fitting a time constant.\n"
        "The trap this must not fall into: smoothing trivially reduces the event rate, and "
        "an over-smoothed model would score well on rate while producing dead light. That "
        "is why diversity is a predicted effect with a direction, not a footnote."),
    "plan": (
        "No retraining. Filter each predicted channel with scipy filtfilt at the corpus's "
        "own cutoff, re-render, and re-run the Stage 1 light protocol. Compare three arms "
        "on identical descriptors: unfiltered, causal one-pole fitted per channel "
        "(calibrate_output_smoothing, already implemented), and zero-phase. Report the "
        "causal arm as the online cost of the same fix."),
    "gates": [
        {"id": "G-smooth-rate", "description":
         "light-event-rate-ratio within 1.3 of the real-mapped rate"},
        {"id": "G-smooth-hf", "description":
         "no channel above 2x the corpus above-6-Hz energy"},
        {"id": "G-smooth-fid", "description": "FID below the unfiltered arm"},
        {"id": "G-smooth-div", "description":
         "diversity not below the unfiltered arm; a rate win bought with dead light fails"}],
    "topic_tags": ["audio-to-light", "generative-modelling", "offline-vs-online"],
    "problem_tags": ["conditional-mean-collapse", "metric-gaming"],
}))
print("H-B", hB.get("id"))

hC = ok(c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "Conditioning on whole-track structure -- section label and time to the next drop "
        "-- is worth more than anything else the offline setting buys, because structure is "
        "where the largest measured effect in this project's data lives."),
    "predicted_effects": [
        {"diagnostic_id": "light-bas-minus-shift-null", "direction": "up",
         "magnitude_note": "CI excluding the same model without structure conditioning"},
        {"diagnostic_id": "fid-light-kinetic", "direction": "down",
         "magnitude_note": "closer to real-mapped than the unconditioned model"}],
    "rationale": (
        "Solberg and Jensenius measured movement at the drop at 1.68x to 2.53x the "
        "breakdown level with Cohen's d up to +2.55, and we reproduced that on the Oslo "
        "corpus. It is the largest effect anywhere in this project, several times bigger "
        "than any frame-level audio-movement correlation, and it is exactly the thing a "
        "causal model cannot use: knowing a drop is 8 bars away requires hearing the build "
        "finish. A human operator has the setlist and does exactly this.\n"
        "It is also the honest reason to prefer pre-rendering, as opposed to preferring it "
        "because the numbers happen to be higher."),
    "plan": (
        "Add section label (intro / build / drop / breakdown / core) and signed seconds to "
        "the next drop as conditioning channels. Labels come from whole-track Laplacian "
        "segmentation plus the EDM role labeller already in lightsem/audio/, validated "
        "against the Oslo record's own hand-marked breakdown/buildup/drop times on the two "
        "mixes -- the only place in this corpus where the truth is written down. Retrain "
        "the offline trunk with and without those channels, everything else identical."),
    "gates": [
        {"id": "G-struct-lift", "description":
         "Light-BAS with structure conditioning above the same model without it, CI "
         "excluding zero on a paired bootstrap"},
        {"id": "G-struct-contrast", "description":
         "rendered light intensity ratio drop/breakdown within 1.3x of the corpus 1.68-2.53"},
        {"id": "G-struct-shuffle", "description":
         "SHUFFLE CONTROL: section labels permuted across tracks must destroy the gain. If "
         "a shuffled label helps as much, the model is using the channel as a bias term "
         "and the structure claim is void"}],
    "topic_tags": ["audio-to-light", "offline-vs-online", "music-induced-movement"],
    "problem_tags": ["matched-baseline", "metric-gaming"],
}))
print("H-C", hC.get("id"))

hD = ok(c.call("propose_hypothesis", {
    "problem_id": P,
    "statement": (
        "The operator head, which is already the strongest signal in the project at +0.3145 "
        "marginal lift, gains more from look-ahead than the dance head does, because "
        "anticipating a build is what an operator is actually doing."),
    "predicted_effects": [
        {"diagnostic_id": "event-ap-lift-audio-marginal", "direction": "up",
         "magnitude_note": "above the causal head's +0.3145 on the same 699 shows"},
        {"diagnostic_id": "event-ap-lift-vs-horizon", "direction": "up",
         "magnitude_note": "lift at a +4 s horizon > +0.05 with a CI excluding zero"}],
    "rationale": (
        "PLAN.md section 9 rules out claiming anticipation on RPMC-L2, and that rule was "
        "written for a live product where anticipation would have to be inferred rather "
        "than known. Pre-rendering removes the reason for the rule but not the standard: "
        "the preregistered number stays exactly as it was, lift at 4 s above +0.05 with a "
        "CI excluding zero, and the shift control still has to peak at 0 s.\n"
        "This is the cheapest of the four -- the features are extracted, the causal head is "
        "trained, and the change is the attention mask."),
    "plan": (
        "Re-extract the RPMC features without the causal hold, train the same 3.55 M head "
        "with unmasked attention, and score the horizon curve against the causal head on "
        "the same 86-show test split. Keep the rhythm-of-past-changes baseline and the "
        "time-shift control unchanged."),
    "gates": [
        {"id": "G-antic-4s", "description":
         "event AP lift at +4 s > +0.05 with CI excluding zero"},
        {"id": "G-antic-beats-causal", "description":
         "offline marginal lift above the causal head's +0.3145 on the same split"},
        {"id": "G-antic-shift", "description":
         "time-shift control still peaks within 2 s of zero"}],
    "topic_tags": ["audio-to-light", "offline-vs-online", "evaluation-metric"],
    "problem_tags": ["matched-baseline"],
}))
print("H-D", hD.get("id"))

ok(c.call("register_runbook", {
    "id": "rb-ls2-07-offline-prerendered",
    "title": "Offline pre-rendered show: the look-ahead axis as a designed variable",
    "note": (
        "2026-09-04. Anton's call: the product is a pre-rendered show for a known track, "
        "not a live desk. Consequences, in the order they matter.\n"
        "1. Look-ahead is legal and must be labelled. Every number carries its position on "
        "the axis: L=0 causal, L=250 ms (buyable with a house delay line, IMAG already "
        "forces 50-150 ms), or whole-track. A score is comparable only to a score at the "
        "same L. The causal path stays alive as the L=0 end, not as a deprecated branch.\n"
        "2. extract_bus(causal=False) is the offline feature path and already exists -- the "
        "flag was kept when the bus was made causal, so the centred STFT, the centred HPSS "
        "median and the interpolating resample are one argument away.\n"
        "3. The five ORACLE_CHANNELS stop being a debt. A whole-track comb beat grid is "
        "simply correct offline, and the oracle_beat flag now marks a property rather than "
        "a compromise.\n"
        "4. Stage 4's latency budget is no longer a product gate. It stays as the L=0 "
        "characterisation and for any live-augmented variant.\n"
        "5. Order: H-A (ceiling) decides whether H-B, H-C and H-D are worth running. Do not "
        "reorder for convenience -- if the ceiling says the corpus binds, the right next "
        "move is data, and three architecture experiments would just spend a week "
        "confirming it slowly."),
}))

after = len(list(Path("/root/vibescience-vault").rglob("*.md")))
lv = len(list(Path("/root/lightsem-vault").rglob("*.md")))
print(f"lightsem {lv} md; optiHealth {before}->{after}")
assert after == before, "optiHealth vault must not change"
c.close()
