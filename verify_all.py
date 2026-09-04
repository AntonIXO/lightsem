#!/usr/bin/env python
"""End-to-end verification of the pieces every result depends on.

Not a smoke test. Each check is aimed at a specific way a number in this project
could be wrong without anything crashing: a beat grid off by an octave or half a
beat, a DMX round trip that loses a channel, a descriptor that is not invariant
to the things it claims to be invariant to, a metric that does not separate
signal from its own null, a model that can see the future.

    bash run_torch.sh verify_all.py              # everything
    bash run_torch.sh verify_all.py causal      # only checks matching "causal"
"""
from __future__ import annotations

import sys
import traceback

import numpy as np

PASS, FAIL, SKIP = [], [], []
ONLY = [a for a in sys.argv[1:] if not a.startswith("-")]


def check(name):
    def deco(fn):
        if ONLY and not any(o.lower() in name.lower() for o in ONLY):
            SKIP.append(name)
            return fn
        try:
            detail = fn()
            PASS.append((name, detail))
            print(f"  PASS  {name}: {detail}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        except Exception as e:                        # noqa: BLE001
            FAIL.append((name, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
        return fn
    return deco


print("=== 1. beat grid ===")


@check("comb grid recovers a known tempo and phase")
def _():
    from lightsem.audio.beatgrid import fit_constant_grid
    fps, dur = 86.13, 30.0
    for bpm, phase in ((120.0, 0.13), (90.0, 0.4), (128.0, 0.0)):
        n = int(dur * fps)
        onset = np.zeros(n)
        period = 60.0 / bpm
        for k in range(int(dur / period)):
            i = int(round((phase + k * period) * fps))
            if 0 <= i < n:
                onset[i] = 1.0 if k % 2 == 0 else 0.6      # weaker off-beats
        onset += 0.05 * np.random.default_rng(0).standard_normal(n)
        g = fit_constant_grid(onset, fps)
        assert abs(g["bpm"] - bpm) < 2.0, f"bpm {g['bpm']:.1f} != {bpm}"
        err = abs(((g["phase_s"] - phase + period / 2) % period) - period / 2)
        assert err < 0.03, f"phase off by {err*1000:.0f} ms at {bpm} bpm"
    return "120/90/128 BPM recovered within 2 BPM and 30 ms"


@check("tempo prior does not invent a tempo from noise")
def _():
    from lightsem.audio.beatgrid import fit_constant_grid
    rng = np.random.default_rng(1)
    g = fit_constant_grid(np.abs(rng.standard_normal(2600)), 86.13)
    assert g["comb_score"] < 1.6, f"comb score {g['comb_score']:.2f} on noise"
    return f"noise gives comb score {g['comb_score']:.2f} (a real track scores 2-5)"


print("\n=== 2. skeleton and descriptors ===")


def _synthetic_dancer(T=600, fps=60.0, bounce_hz=2.0, scale=1.0, shift=0.0):
    t = np.arange(T) / fps
    P = np.zeros((T, 17, 3))
    base = np.array([
        [0, 1.75, 0], [.03, 1.78, 0], [-.03, 1.78, 0], [.07, 1.77, 0], [-.07, 1.77, 0],
        [.2, 1.5, 0], [-.2, 1.5, 0], [.35, 1.2, 0], [-.35, 1.2, 0], [.4, .95, 0],
        [-.4, .95, 0], [.12, 1.0, 0], [-.12, 1.0, 0], [.13, .55, 0], [-.13, .55, 0],
        [.14, .1, 0], [-.14, .1, 0]])
    bob = 0.06 * np.sin(2 * np.pi * bounce_hz * t)
    sway = 0.10 * np.sin(2 * np.pi * bounce_hz / 4 * t)
    for j in range(17):
        P[:, j, 0] = base[j, 0] + sway
        P[:, j, 1] = base[j, 1] + bob
        P[:, j, 2] = base[j, 2]
    P[:, 9, 0] += 0.25 * np.sin(2 * np.pi * bounce_hz * 2 * t)
    P[:, 10, 0] -= 0.25 * np.sin(2 * np.pi * bounce_hz * 2 * t)
    return P * scale + shift


@check("descriptors are invariant to body size and world position")
def _():
    from lightsem.motion.descriptors import extract
    from lightsem.motion.skeleton import prepare
    a = extract(prepare(_synthetic_dancer(), 60.0), tempo=120.0)
    b = extract(prepare(_synthetic_dancer(scale=1.6, shift=3.0), 60.0), tempo=120.0)
    worst, worst_ch = 0.0, ""
    for ch in ("qom", "head_speed", "hand_speed", "spread", "vert_bounce_amp"):
        d = abs(a.channels[ch].mean() - b.channels[ch].mean()) / (abs(a.channels[ch].mean()) + 1e-9)
        if d > worst:
            worst, worst_ch = d, ch
    assert worst < 0.02, f"{worst_ch} changed by {worst*100:.1f}% under scale+shift"
    return f"worst channel drift {worst*100:.2f}% ({worst_ch}) under 1.6x size and 3 m shift"


@check("mirroring flips laterality and leaves scalars alone")
def _():
    from lightsem.motion.descriptors import extract
    from lightsem.motion.skeleton import mirror, prepare
    sk = prepare(_synthetic_dancer(), 60.0)
    a, b = extract(sk, tempo=120.0), extract(mirror(sk), tempo=120.0)
    d = abs(a.channels["qom"].mean() - b.channels["qom"].mean()) / abs(a.channels["qom"].mean())
    assert d < 0.02, f"qom changed {d*100:.1f}% under mirroring"
    la, lb = a.channels["lateral_pos"], b.channels["lateral_pos"]
    r = np.corrcoef(la, lb)[0, 1]
    assert r < -0.9, f"lateral_pos correlation under mirror is {r:+.2f}, expected about -1"
    return f"qom drift {d*100:.2f}%, lateral_pos correlation {r:+.3f}"


@check("bounce period is recovered from a known oscillation")
def _():
    from lightsem.motion.descriptors import extract
    from lightsem.motion.skeleton import prepare
    for hz in (1.5, 2.0, 3.0):
        d = extract(prepare(_synthetic_dancer(bounce_hz=hz), 60.0), tempo=60.0 * hz)
        per = np.median(d.channels["vert_period_s"])
        assert abs(per - 1.0 / hz) < 0.05, f"period {per:.3f} != {1/hz:.3f} at {hz} Hz"
    return "1.5/2.0/3.0 Hz recovered within 50 ms"


print("\n=== 3. DMX ===")


@check("encode -> decode round trip is exact to DMX quantisation")
def _():
    from lightsem.conductor.rig import canonical_rig
    from lightsem.dmx.decode import decode_stream
    from lightsem.dmx.encode import pack_universe
    rig = canonical_rig()
    rng = np.random.default_rng(0)
    T, nf = 200, len(rig.fixtures)
    P = {k: rng.random((T, nf)) for k in
         ("dimmer", "strobe", "hue", "sat", "val", "pan", "tilt", "zoom")}
    P["gobo"] = np.zeros((T, nf))
    dec = decode_stream(rig, pack_universe(rig, P))
    # Only compare parameters a fixture actually has: a wash has no pan channel,
    # so the decoder reports 0.5 for it and comparing that to a random request
    # tests the test, not the code.
    has = {"moving_head": ("dimmer", "pan", "tilt", "zoom"),
           "wash": ("dimmer", "zoom"), "blinder": ("dimmer",),
           "strobe": ("dimmer",), "projector": ("dimmer", "zoom", "pan", "tilt")}
    worst, where = 0.0, ""
    for i, f in enumerate(rig.fixtures):
        for k in has[f.kind]:
            e = float(np.abs(P[k][:, i] - dec[k][:, i]).max())
            if e > worst:
                worst, where = e, f"{f.kind}.{k}"
    assert worst <= 1.5 / 255, f"round-trip error {worst:.4f} on {where}"
    return f"max error {worst:.5f} on {where} (one DMX step is {1/255:.5f})"


@check("slew limiting actually bounds the rate")
def _():
    from lightsem.dmx.encode import slew_limit
    x = np.zeros((100, 3))
    x[50:] = 1.0
    y = slew_limit(x, 40.0, 4.0)
    rate = np.abs(np.diff(y, axis=0)).max() * 40.0
    assert rate <= 4.01, f"rate {rate:.2f} exceeds the 4.0/s limit"
    return f"step input limited to {rate:.2f}/s against a 4.0/s limit"


print("\n=== 4. metrics ===")


@check("Light-BAS: metronome scores 1, random sits at its own null")
def _():
    from lightsem.eval.bas import bas_report
    beats = np.arange(0, 20, 0.5)
    rep = bas_report(beats, beats, 20.0)
    assert rep["bas"] > 0.999, f"metronome scores {rep['bas']:.3f}"
    rng = np.random.default_rng(0)
    rnd = np.sort(rng.uniform(0, 20, 40))
    rep2 = bas_report(rnd, beats, 20.0, rng=rng)
    assert abs(rep2["bas"] - rep2["null_mean"]) < 3 * rep2["null_sd"], \
        f"random events {rep2['bas']:.3f} vs null {rep2['null_mean']:.3f}"
    return f"metronome {rep['bas']:.3f}; random {rep2['bas']:.3f} vs null {rep2['null_mean']:.3f}"


@check("FID_light separates a set from a structure-destroyed version of itself")
def _():
    from lightsem.conductor.beat_trigger import render_beat_trigger
    from lightsem.conductor.rig import canonical_rig
    from lightsem.eval.fid import FeatureScaler, frechet, split_half_floor
    from lightsem.eval.light_features import block_shuffle, delta_features
    from lightsem.motion.io import CACHE_ROOT, load_bus, slice_bus
    rig = canonical_rig()
    rng = np.random.default_rng(0)
    # Use the dance-derived conductor, which is what the experiment measures:
    # the beat trigger is so stereotyped that shuffling it barely moves its
    # features, which tests the baseline's monotony rather than the metric.
    from lightsem.conductor.hybrid import ConductorHybrid
    from lightsem.motion.io import load_descriptors
    import json
    cond = ConductorHybrid()
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    seen, picks = set(), []
    for r in idx:
        if r["music_id"] in seen or r["duration_s"] < 9:
            continue
        seen.add(r["music_id"])
        picks.append(r)
        if len(picks) >= 24:
            break
    real, null = [], []
    for r in picks:
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        bus = slice_bus(load_bus(CACHE_ROOT / "audio" / f"{r['music_id']}.npz"),
                        d.duration_s)
        fr = cond.render(d, bus).frames
        real.append(delta_features(fr, rig, 40.0))
        null.append(delta_features(block_shuffle(fr, 40.0, 1.0, rng), rig, 40.0))
    sc = FeatureScaler(np.array(real))
    floor = split_half_floor(sc(np.array(real)), rng=rng)["mean"]
    d = frechet(sc(np.array(real)), sc(np.array(null)))
    assert d > 3 * floor, f"FID to the null {d:.1f} is not above 3x the floor {floor:.1f}"
    return f"floor {floor:.1f}, null {d:.1f} ({d/max(floor,1e-9):.0f}x)"


@check("brightness matcher equalises two very different streams")
def _():
    from lightsem.conductor.rig import canonical_rig
    from lightsem.eval.ab import match_brightness, scale_brightness
    from lightsem.conductor.beat_trigger import render_beat_trigger
    from lightsem.motion.io import CACHE_ROOT, load_bus, slice_bus
    rig = canonical_rig()
    bus = slice_bus(load_bus(CACHE_ROOT / "audio" / "mBR0.npz"), 10.0)
    a = render_beat_trigger(bus).frames
    b = scale_brightness(a, rig, 0.35)
    _, _, m = match_brightness(a, b, rig)
    assert m["matched"], f"did not converge: {m}"
    return f"3x brightness difference matched to {abs(m['combined_a']-m['combined_b'])/m['combined_a']*100:.1f}%"


print("\n=== 5. model ===")


@check("trained model cannot see the future")
def _():
    import torch
    from lightsem.model.infer import LightModel
    from lightsem.model.light_trunk import causality_probe
    lm = LightModel()
    x = torch.randn(2, 300, lm.cfg.in_dim, device=lm.device)
    leaks = [causality_probe(lm.model, x, c) for c in (40, 120, 250)]
    assert max(leaks) == 0.0, f"future leaked into the past: {leaks}"
    return f"leak exactly 0.0 at cuts of 40/120/250 frames"


@check("inference is chunk-invariant (streaming equals batch)")
def _():
    import torch
    from lightsem.audio.online import extract_bus
    from lightsem.model.infer import LightModel
    lm = LightModel()
    # 50 s is longer than the 26 s receptive field, so the short-chunk run has to
    # re-prime from real history rather than always seeing the whole clip.
    bus = extract_bus("/root/lightsem/data/wav/wav/mBR0.wav", duration_s=50.0)
    a = lm.descriptors(bus, seed=0, chunk=2400)
    b = lm.descriptors(bus, seed=0, chunk=200)
    worst = max(float(np.abs(a.channels[k] - b.channels[k]).max())
                for k in lm.block_a)
    # float32 matmuls reduce in a different order at different batch shapes, so
    # exact equality is not available; 5e-3 is 0.3 % of one target IQR, while the
    # bug this catches (priming with one attention window instead of the full
    # receptive field) produced 0.79.
    assert worst < 5e-3, f"block A differs by {worst:.5f} between chunk sizes"
    return (f"block A identical to {worst:.2e} across 200 vs 2400 frame chunks "
            f"(receptive field {lm.receptive_field()} frames = "
            f"{lm.receptive_field()/40:.0f} s)")


@check("style head samples differ, and differ more than block A does")
def _():
    from lightsem.audio.online import extract_bus
    from lightsem.model.infer import LightModel
    lm = LightModel()
    bus = extract_bus("/root/lightsem/data/wav/wav/mBR0.wav", duration_s=12.0)
    a = lm.descriptors(bus, seed=0)
    b = lm.descriptors(bus, seed=1)
    da = np.mean([np.abs(a.channels[k] - b.channels[k]).mean() for k in lm.block_a])
    dc = np.mean([np.abs(a.channels[k] - b.channels[k]).mean() for k in lm.block_c])
    assert da < 1e-6, f"block A is not deterministic across seeds ({da:.2e})"
    assert dc > 1e-3, f"block C does not vary across seeds ({dc:.2e})"
    return f"block A varies {da:.1e} across seeds, block C {dc:.3f}"


print("\n=== 6. vault isolation ===")


@check("the optiHealth vault is untouched")
def _():
    from pathlib import Path
    n = len(list(Path("/root/vibescience-vault").rglob("*.md")))
    assert n == 688, f"optiHealth vault has {n} md files, expected 688"
    return f"{n} md files, unchanged"


print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed"
      + (f", {len(SKIP)} skipped by filter {ONLY}" if SKIP else ""))
if FAIL:
    for n, e in FAIL:
        print(f"  FAILED: {n} — {e}")
sys.exit(1 if FAIL else 0)
