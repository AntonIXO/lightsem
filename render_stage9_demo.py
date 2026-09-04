#!/usr/bin/env python
"""Stage 9 demos on any song: every arm of this stage rendered to MP4 side by side with the
shipped model and the beat trigger, so the numbers in outputs/stage9 can be watched.

  bash run_torch.sh render_stage9_demo.py ~/shpongle.flac [--out DIR] [--max-s N] [--arms a,b,c]

Arms (outputs/stage9/demo_songs/<name>__<arm>.mp4):
  beat_trigger   conductor/beat_trigger.py
  conductor      the shipped model (light_trunk_jd_none_smooth) through the hybrid conductor
  decoder_v1     group decoder v1 (regressed parameters; collapsed to a static look + chase)
  decoder_v2     group decoder v2 (sampled look, rectified flow) + effects engine
  decoder_v2_plan  v2 with the offline section planner's knobs written on top
  engine_only    v2's slow knobs driving beat-trigger effects (the 9.9 engine-only baseline)
  best_of_8      the shipped model's 8 flow seeds, the critic's pick (needs critic_v1.pt)
A JSON next to the videos carries Light-BAS minus null and events per clip for every arm.
"""
from __future__ import annotations

import argparse, json, re, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/lightsem")
_argv, sys.argv = sys.argv, sys.argv[:1]
from extract_rpmc_causal import features_from_mel, SR, HOP, FPS_OUT  # noqa: E402
sys.argv = _argv
from lightsem.audio.online import causal_resample, ema, extract_bus  # noqa: E402
from lightsem.conductor.beat_trigger import render_beat_trigger  # noqa: E402
from lightsem.conductor.hybrid import ConductorHybrid  # noqa: E402
from lightsem.conductor.rig import canonical_rig  # noqa: E402
from lightsem.eval.bas import bas_report  # noqa: E402
from lightsem.eval.light_features import level_features, light_onsets  # noqa: E402
from lightsem.model.group_decoder import GroupEngine, slow_knobs  # noqa: E402
from lightsem.model.infer import LightModel  # noqa: E402
from lightsem.sim.renderer import RenderConfig, StageRenderer  # noqa: E402
from run_stage9_decoder_eval import CKPT, Decoder, HZ  # noqa: E402

OUT = Path("/root/lightsem/outputs/stage9/demo_songs")


def operator_features(wav: Path, dur: float) -> np.ndarray:
    """The 52-d causal 10 Hz operator features (build_justdance_visual / build_beatsaver recipe)."""
    import librosa
    au, _ = librosa.load(str(wav), sr=SR)
    mel = librosa.feature.melspectrogram(y=au, sr=SR, n_fft=2048, hop_length=HOP, n_mels=128)
    ch, grid = features_from_mel(librosa.power_to_db(mel).astype(np.float32))
    fps_in = SR / HOP; dur10 = min(dur, mel.shape[1] / fps_in)
    n_out = int(round(dur10 * FPS_OUT)); t_out = np.arange(n_out) / FPS_OUT; t_in = np.arange(mel.shape[1]) / fps_in
    cols = []
    for name, v in ch.items():
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        for tau in (0.2, 1.0, 4.0):
            cols.append(causal_resample(ema(v, fps_in, tau), t_in, t_out))
    per = grid["period_s"]; ph = np.mod((t_out - grid["phase_s"]) / per, 1.0)
    cols += [np.sin(2 * np.pi * ph), np.cos(2 * np.pi * ph), (1 - ph) ** 2.5, np.full(n_out, per)]
    return np.stack(cols, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-s", type=float, default=0.0)
    ap.add_argument("--arms", default="beat_trigger,conductor,decoder_v1,decoder_v2,decoder_v2_plan,engine_only,best_of_8")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    arms = a.arms.split(",")
    ren = StageRenderer(canonical_rig(), RenderConfig(width=854, height=480, fps=30))
    rig = canonical_rig(); cond = ConductorHybrid(); eng = GroupEngine(rig)
    lm = LightModel(CKPT / "light_trunk_jd_none_smooth.pt")
    dec1 = Decoder(CKPT / "group_decoder_v1.pt", device) if (CKPT / "group_decoder_v1.pt").exists() else None
    dec2 = Decoder(CKPT / "group_decoder_v2.pt", device) if (CKPT / "group_decoder_v2.pt").exists() else None
    planner = None
    if (CKPT / "planner_v1.pt").exists():
        from run_stage9_planner_eval import PlannerRunner, apply_plan
        planner = PlannerRunner(CKPT / "planner_v1.pt", device)
    critic = None
    if (CKPT / "critic_v1.pt").exists() and "best_of_8" in arms:
        critic = torch.load(CKPT / "critic_v1.pt", map_location=device, weights_only=False)
    for src in a.audio:
        src = Path(src); name = re.sub(r"[^A-Za-z0-9]+", "_", src.stem).strip("_")
        wav = out / f"{name}.wav"
        cmd = ["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1", "-ar", "44100", "-y"] + (["-t", str(a.max_s)] if a.max_s > 0 else [])
        subprocess.run(cmd + [str(wav)], check=True)
        t0 = time.time(); bus = extract_bus(wav); dur = bus.duration_s
        x = operator_features(wav, dur); t10 = np.arange(len(x)) / HZ
        print(f"{name}: {dur:.0f} s, tempo {bus.tempo:.1f} bpm, bus + features in {time.time()-t0:.0f}s", flush=True)
        renders = {}
        if "beat_trigger" in arms:
            renders["beat_trigger"] = render_beat_trigger(bus).frames
        if "conductor" in arms:
            renders["conductor"] = cond.render(lm.descriptors(bus, seed=0), bus).frames
        ref_mean = level_features(renders.get("conductor", render_beat_trigger(bus).frames), rig)["mean"]

        def engine(p):
            fr = eng.render(p, t10, bus.beats, bus.channels["bar_phase"], dur).frames
            g = float(np.clip(ref_mean / max(level_features(fr, rig)["mean"], 1e-3), 0.3, 3.0))
            return eng.render(p, t10, bus.beats, bus.channels["bar_phase"], dur, gain=g).frames if abs(g - 1) > 0.05 else fr
        if dec1 and "decoder_v1" in arms:
            renders["decoder_v1"] = engine(dec1.params(x, "rpmc"))
        if dec2:
            p2 = dec2.params(x, "rpmc", seed=0)
            if "decoder_v2" in arms:
                renders["decoder_v2"] = engine(p2)
            if "engine_only" in arms:
                renders["engine_only"] = engine(slow_knobs(p2))
            if planner and "decoder_v2_plan" in arms:
                phase = np.mod(np.arctan2(x[:, 48], x[:, 49]) / (2 * np.pi), 1.0)
                knobs, bar = planner.knobs(x, phase)
                renders["decoder_v2_plan"] = engine(apply_plan(p2, knobs, bar))
        if critic is not None:
            from train_critic import Critic, LIGHT_DIM, RED_AUDIO, RED_LIGHT, dmx_scene, light_tokens, WIN
            from train_planner import bar_pool, bars_of
            phase = np.mod(np.arctan2(x[:, 48], x[:, 49]) / (2 * np.pi), 1.0); bar = bars_of(phase); ta = bar_pool(x, bar)
            amu, asd = torch.tensor(critic["amu"], device=device), torch.tensor(critic["asd"], device=device)
            lmu, lsd = torch.tensor(critic["lmu"], device=device), torch.tensor(critic["lsd"], device=device)
            red_a = torch.tensor([j for j in range(ta.shape[1]) if j % 52 in RED_AUDIO], device=device); red_l = torch.tensor(RED_LIGHT, device=device)
            cU = Critic(ta.shape[1], LIGHT_DIM).to(device).eval(); cU.load_state_dict(critic["U"])
            cR = Critic(len(red_a), len(RED_LIGHT)).to(device).eval(); cR.load_state_dict(critic["R"])

            @torch.no_grad()
            def score_long(ta_, tl_):
                n = min(len(ta_), len(tl_)); starts = list(range(0, max(n - WIN, 0) + 1, WIN // 2)) or [0]
                pad = lambda v, i: v[i:i + WIN] if i + WIN <= n else np.pad(v[i:n], ((0, i + WIN - n), (0, 0)))  # noqa: E731
                A_ = (torch.from_numpy(np.stack([pad(ta_, i) for i in starts])).to(device) - amu) / asd
                L_ = (torch.from_numpy(np.stack([pad(tl_, i) for i in starts])).to(device) - lmu) / lsd
                return float((cU.score(A_, L_) - 0.5 * cR.score(A_[..., red_a], L_[..., red_l])).mean())
            cands, scores = [], []
            for s_ in range(a.seeds):
                fr = cond.render(lm.descriptors(bus, seed=s_), bus).frames; cands.append(fr)
                y = dmx_scene(fr, rig); n = min(len(y), len(bar)); scores.append(score_long(ta, light_tokens(y[:n], bar[:n])))
            best = int(np.argmax(scores)); renders["best_of_8"] = cands[best]
            print(f"  critic scores over {a.seeds} seeds: {np.round(scores, 3).tolist()} -> seed {best}", flush=True)
        stats = {}
        for k, frames in renders.items():
            ev, _ = light_onsets(frames, rig, 40.0); r = bas_report(ev, bus.beats, dur)
            stats[k] = {"bas_minus_null": r["bas"] - r["null_mean"], "events": int(r["n_events"]), "brightness": level_features(frames, rig)["mean"]}
            path = out / f"{name}__{k}.mp4"; t1 = time.time()
            ren.render_video(frames, 40.0, path, audio_path=str(wav), duration_s=dur, progress_every=0)
            print(f"  wrote {path.name} ({time.time()-t1:.0f}s)  BAS-null {stats[k]['bas_minus_null']:+.3f} events {stats[k]['events']}", flush=True)
        (out / f"{name}__stage9_stats.json").write_text(json.dumps({"tempo": bus.tempo, "duration_s": dur, "arms": stats}, indent=1))


if __name__ == "__main__":
    main()
