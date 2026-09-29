#!/usr/bin/env python
"""Stage 4 (software half) — latency and causality of the deployed path.

    bash run_torch.sh scripts/legacy_stages/run_stage4.py

Measures what can be measured without a rig: the audio-in to DMX-out latency of
each stage as p50/p99 over many blocks, and the causality of the trained model
by mechanical test rather than by assertion. The remaining half — a photodiode
against a click track, and Art-Net on a real node — needs hardware and is not
faked here.

The budget to hit (PLAN.md section 4): transient path <= 20 ms, learned path
<= 65 ms audio-in to DMX-out, causality bit-identical.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from lightsem.audio.online import extract_bus
from lightsem.conductor.hybrid import ConductorHybrid
from lightsem.conductor.rig import canonical_rig
from lightsem.dmx.encode import pack_universe
from lightsem.model.infer import LightModel
from lightsem.model.light_trunk import causality_probe

OUT = Path("/root/lightsem/experiments/legacy/stages/outputs/stage4")
FPS = 40.0


def pct(x):
    a = np.asarray(x) * 1000.0
    return {"p50_ms": float(np.percentile(a, 50)), "p99_ms": float(np.percentile(a, 99)),
            "mean_ms": float(a.mean()), "n": int(a.size)}


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    lm = LightModel()
    rig = canonical_rig()
    cond = ConductorHybrid()
    res = {"device": dev, "params_m": lm.model.n_params() / 1e6}

    # --- causality: perturb the future, the past must not move ---------------
    X = torch.randn(4, 400, lm.cfg.in_dim, device=dev)
    leaks = [causality_probe(lm.model, X, c) for c in (50, 100, 200, 300)]
    res["causality"] = {"max_leak": float(max(leaks)), "per_cut": leaks,
                        "bit_identical": bool(max(leaks) == 0.0)}
    print(f"causality leak (must be 0): {max(leaks)}")

    # --- model step latency, streaming with the FULL primed context ----------
    # Not one attention window: windows compose across layers, so the trunk's
    # receptive field is n_blocks x (attn_window - 1 + conv_kernel - 1) + 1 =
    # 1039 frames, 26 s at 40 fps. Timing a 160-frame context would have
    # understated the deployed cost by the same factor that made the chunked
    # inference wrong.
    ctx = lm.receptive_field()
    res["receptive_field_frames"] = ctx
    res["receptive_field_s"] = ctx / FPS
    print(f"receptive field {ctx} frames = {ctx/FPS:.1f} s of audio context")
    for chunk in (1, 4, 8, 20):
        times = []
        with torch.no_grad():
            x = torch.randn(1, ctx + chunk, lm.cfg.in_dim, device=dev)
            for _ in range(5):
                lm.model(x)
            if dev == "cuda":
                torch.cuda.synchronize()
            for _ in range(60):
                t0 = time.perf_counter()
                lm.model(x)
                if dev == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        res[f"model_chunk_{chunk}f"] = pct(times)
        print(f"  model forward, {chunk} new frame(s) ({chunk/FPS*1000:.0f} ms of audio): "
              f"p50 {res[f'model_chunk_{chunk}f']['p50_ms']:.2f} ms  "
              f"p99 {res[f'model_chunk_{chunk}f']['p99_ms']:.2f} ms")

    # --- conductor + DMX packing per frame -----------------------------------
    wav = "/root/lightsem/data/wav/wav/mBR0.wav"
    bus = extract_bus(wav, duration_s=20.0)
    d = lm.descriptors(bus, seed=0)
    # Both calls are whole-clip vectorised, so a per-frame number is amortised
    # by construction. Repeating the clip is what makes the p99 an observation
    # instead of a copy of the p50: the spread across repeats is real, the
    # spread inside a clip is not measurable from here.
    t_cond, t_pack = [], []
    for _ in range(20):
        t0 = time.perf_counter(); P = cond.params(d, bus)
        t_cond.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); frames = pack_universe(rig, P)
        t_pack.append(time.perf_counter() - t0)
    n = frames.shape[0]
    res["conductor_per_frame"] = pct([t / n for t in t_cond]) | {"amortised": True}
    res["dmx_pack_per_frame"] = pct([t / n for t in t_pack]) | {"amortised": True}
    print(f"  conductor {res['conductor_per_frame']['p50_ms']:.3f} ms/frame "
          f"(p99 {res['conductor_per_frame']['p99_ms']:.3f}), DMX pack "
          f"{res['dmx_pack_per_frame']['p50_ms']:.3f} ms/frame "
          f"(p99 {res['dmx_pack_per_frame']['p99_ms']:.3f}), amortised over "
          f"{n} frames x 20 repeats")

    # --- transient path at its own, shorter hop -------------------------------
    # The learned path can afford a 23 ms analysis hop; the transient path cannot,
    # because ITU-R BT.1359-1 puts audible sound-before-light at +45 ms and the
    # hop alone would spend half of that. Onset detection needs no mel bank and
    # no model, so it runs on a 256-sample hop (5.8 ms) as a separate DSP stage.
    hop_fast = 256
    fast_block_ms = 1000.0 * hop_fast / 44100
    y = np.random.randn(44100).astype(np.float32)
    times = []
    for _ in range(200):
        blk = y[:hop_fast * 2]
        t0 = time.perf_counter()
        spec = np.abs(np.fft.rfft(blk * np.hanning(len(blk))))
        flux = float(np.maximum(np.diff(np.log1p(spec)), 0).sum())
        _ = flux > 1.0
        times.append(time.perf_counter() - t0)
    res["transient_dsp_stage"] = pct(times)
    res["transient_path_fast_ms"] = (fast_block_ms + res["transient_dsp_stage"]["p99_ms"]
                                     + 5.0)
    print(f"  transient DSP at hop {hop_fast} ({fast_block_ms:.1f} ms block): "
          f"p99 {res['transient_dsp_stage']['p99_ms']:.3f} ms compute -> path "
          f"{res['transient_path_fast_ms']:.1f} ms")

    # --- end-to-end budget ----------------------------------------------------
    audio_block_ms = 1000.0 * 1024 / 44100          # the STFT hop
    model_ms = res["model_chunk_1f"]["p50_ms"]
    model_ms99 = res["model_chunk_1f"]["p99_ms"]
    cond_ms = res["conductor_per_frame"]["p50_ms"] + res["dmx_pack_per_frame"]["p50_ms"]
    net_ms, fixture_ms = 5.0, 30.0
    res["budget"] = {
        "audio_block_ms": audio_block_ms, "model_p50_ms": model_ms,
        "model_p99_ms": model_ms99, "conductor_ms": cond_ms,
        "network_ms_assumed": net_ms, "fixture_ms_assumed": fixture_ms,
        "learned_path_p50_ms": audio_block_ms + model_ms + cond_ms + net_ms + fixture_ms,
        "learned_path_p99_ms": audio_block_ms + model_ms99 + cond_ms + net_ms + fixture_ms,
        "transient_path_ms": audio_block_ms + cond_ms + net_ms,
        "transient_path_fast_ms": res["transient_path_fast_ms"],
        "budget_learned_ms": 65.0, "budget_transient_ms": 20.0,
    }
    b = res["budget"]
    res["gates"] = {
        "transient_path_le_20ms": {
            "passed": bool(b["transient_path_fast_ms"] <= 20.0),
            "evidence": f"{b['transient_path_fast_ms']:.1f} ms on its own 5.8 ms hop "
                        f"(a 20 ms budget). Sharing the learned path's 23.2 ms analysis "
                        f"hop costs {b['transient_path_ms']:.1f} ms and misses, which is "
                        f"the measurement that forces the two paths apart"},
        "learned_path_le_65ms": {
            "passed": bool(b["learned_path_p99_ms"] <= 65.0),
            "evidence": f"p50 {b['learned_path_p50_ms']:.1f} ms, p99 "
                        f"{b['learned_path_p99_ms']:.1f} ms, of a 65 ms budget "
                        f"(fixture {fixture_ms:.0f} ms assumed, not measured)"},
        "causality_bit_identical": {
            "passed": res["causality"]["bit_identical"],
            "evidence": f"max |dy| over cuts at 50/100/200/300 frames = "
                        f"{res['causality']['max_leak']}"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["budget"], indent=1))
    print(json.dumps(res["gates"], indent=1))


if __name__ == "__main__":
    main()
