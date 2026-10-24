#!/usr/bin/env python3
"""Prototype: learn audio-conditioned speed, then drive bounded symmetric stage orbits."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import subprocess
from pathlib import Path

import h5py
import librosa
import numpy as np
import torch
from torch import nn


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    return ((x - x.mean(0, keepdims=True)) / np.maximum(x.std(0, keepdims=True), 1e-5)).astype(np.float32)


def interp_time(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) == n:
        return np.asarray(x, np.float32)
    old = np.linspace(0.0, 1.0, len(x), dtype=np.float64)
    new = np.linspace(0.0, 1.0, n, dtype=np.float64)
    return np.stack([np.interp(new, old, x[:, j]) for j in range(x.shape[1])], 1).astype(np.float32)


def rpmc_sample(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    m = group["music"]
    mfcc = np.asarray(m["mfcc"][:13], np.float32)
    chroma = np.asarray(m["chroma_stft"][:12], np.float32)
    onset_proxy = np.asarray(m["spectral_centroids"][:], np.float32)
    mel = np.asarray(m["mel_spectrogram"][:], np.float32)
    rms = np.log1p(np.maximum(mel, 0.0)).mean(0, keepdims=True).astype(np.float32)
    delta = np.diff(mfcc[:9], axis=1, prepend=mfcc[:9, :1]).astype(np.float32)
    feat = np.concatenate([mfcc, chroma, onset_proxy, rms, delta], 0).T
    light_curves = []
    for key in group["light"].keys():
        h = np.asarray(group["light"][key][:, 0, :], np.float32)
        v = np.asarray(group["light"][key][:, 2, :], np.float32)
        mass = np.maximum(v.sum(1), 1.0)
        bins = np.arange(v.shape[1], dtype=np.float32)
        mean = (v * bins[None]).sum(1) / mass / 255.0
        spread = np.sqrt(np.maximum((v * (bins[None] / 255.0 - mean[:, None]) ** 2).sum(1) / mass, 0.0))
        hbins = np.arange(h.shape[1], dtype=np.float32) * (2.0 * np.pi / 180.0)
        hm = np.maximum(h[:, :180].sum(1), 1.0)
        hue_x = (h[:, :180] * np.cos(hbins[:180])[None]).sum(1) / hm
        hue_y = (h[:, :180] * np.sin(hbins[:180])[None]).sum(1) / hm
        hue = np.unwrap(np.arctan2(hue_y, hue_x)).astype(np.float32)
        light_curves.append(np.stack([mean, spread, hue], 1))
    light = np.mean([interp_time(x, max(len(x) for x in light_curves)) for x in light_curves], 0)
    n = max(32, min(len(feat) // 7, len(light) // 3))
    feat = interp_time(feat, n)
    light = interp_time(light, n)
    activity = np.abs(np.diff(light[:, 0], prepend=light[0, 0])) + 0.5 * np.abs(np.diff(light[:, 1], prepend=light[0, 1]))
    scale = max(float(np.quantile(activity, 0.95)), 1e-4)
    speed_target = np.clip(activity / scale, 0.0, 1.0)
    turn_delta = np.diff(light[:, 2], prepend=light[0, 2])
    turn_scale = max(float(np.quantile(np.abs(turn_delta), 0.95)), 1e-4)
    turn_target = np.tanh(turn_delta / turn_scale).astype(np.float32)
    return zscore(np.nan_to_num(feat)), np.stack([speed_target, turn_target], 1).astype(np.float32)


def roskilde_sample(item: dict) -> tuple[np.ndarray, np.ndarray]:
    a = item["audio_features"]
    mfcc = np.asarray(a["mfcc"], np.float32)[:, :13]
    chroma = np.asarray(a["chroma_stft"], np.float32)[:, :12]
    onset = np.asarray(a["onset_env"], np.float32).reshape(-1, 1)
    rms = np.asarray(a["rms"], np.float32).reshape(-1, 1)
    delta = np.asarray(a["mfcc_delta"], np.float32)[:, :9]
    feat = np.concatenate([mfcc, chroma, onset, rms, delta], 1)
    light = np.asarray(item["lighting_features"], np.float32)
    n = min(len(feat), len(light))
    feat, light = feat[:n], light[:n]
    groups = light.reshape(n, 10, 6)
    activity = np.mean(np.abs(np.diff(light, axis=0, prepend=light[:1])), axis=1)
    scale = max(float(np.quantile(activity, 0.95)), 1e-4)
    speed_target = np.clip(activity / scale, 0.0, 1.0)
    intensity = groups[:, :, 0]
    left_right = intensity[:, 5:].mean(1) - intensity[:, :5].mean(1)
    turn_delta = np.diff(left_right, prepend=left_right[:1])
    turn_scale = max(float(np.quantile(np.abs(turn_delta), 0.95)), 1e-4)
    turn_target = np.tanh(turn_delta / turn_scale).astype(np.float32)
    return zscore(np.nan_to_num(feat)), np.stack([speed_target, turn_target], 1).astype(np.float32)


class SpeedNet(nn.Module):
    def __init__(self, dim: int = 36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, 64, 5, padding=2), nn.GELU(), nn.GroupNorm(8, 64),
            nn.Conv1d(64, 64, 5, padding=2), nn.GELU(), nn.GroupNorm(8, 64),
            nn.Conv1d(64, 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x).transpose(1, 2)
        return torch.stack([torch.sigmoid(raw[..., 0]), torch.tanh(raw[..., 1])], -1)


def own_features(audio_path: Path) -> np.ndarray:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio_path), "-f", "f32le", "-ac", "1", "-ar", "22050", "pipe:1"],
        check=True,
        stdout=subprocess.PIPE,
    )
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    sr = 22050
    hop = 2205
    n_fft = 4096
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, center=True))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop).T
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr).T[:, :12]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop).reshape(-1, 1)
    rms = librosa.feature.rms(S=stft, frame_length=n_fft).T
    delta = np.diff(mfcc[:, :9], axis=0, prepend=mfcc[:1, :9])
    n = min(len(mfcc), len(chroma), len(onset), len(rms), len(delta))
    return zscore(np.concatenate([mfcc[:n], chroma[:n], onset[:n], rms[:n], delta[:n]], 1))


def generate_stage_target_aim(controls: np.ndarray, patch: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controls = np.asarray(controls, np.float32)
    t = len(controls)
    speed = controls[:, 0].clip(0.0, 1.0)
    turn = controls[:, 1].clip(-1.0, 1.0)
    # Audio controls phase rate, heading curvature and target-plane amplitude.
    rate = 0.05 + 0.05 * speed
    heading = 0.85 * np.tanh(np.cumsum(turn * 0.03)).astype(np.float32)
    offsets = np.asarray([0.0, np.pi, np.pi / 2, 3 * np.pi / 2, np.pi / 4, 5 * np.pi / 4, 3 * np.pi / 4, 7 * np.pi / 4], np.float32)
    phase = np.cumsum(rate * 0.1).astype(np.float32) + heading
    amplitude = 0.8 + 0.2 * speed
    positions = np.asarray([row["position"] for row in patch], np.float32)
    anchors = np.asarray([row["anchor"] for row in patch], np.float32)
    base = anchors - positions
    base_pan = np.arctan2(base[:, 0], -base[:, 2])
    base_tilt = np.arctan2(base[:, 1], np.linalg.norm(base[:, (0, 2)], axis=1))
    aim = np.zeros((t, 8, 2), np.float32)
    for i in range(8):
        phase_i = phase + offsets[i]
        dx = (3.0 * amplitude * np.cos(phase_i + heading)).astype(np.float32)
        dz = (2.0 * amplitude * np.sin(phase_i + heading)).astype(np.float32)
        q = np.zeros((t, 3), np.float32)
        q[:, 0] = dx
        q[:, 1] = 0.55
        q[:, 2] = -3.7 + dz
        ray = q - positions[i]
        target_pan = np.arctan2(ray[:, 0], -ray[:, 2])
        target_tilt = np.arctan2(ray[:, 1], np.linalg.norm(ray[:, (0, 2)], axis=1))
        aim[:, i, 0] = (target_pan - base_pan[i]) / np.pi
        aim[:, i, 1] = (target_tilt - base_tilt[i]) / (np.pi / 2.0)
    return aim, heading, amplitude


def load_appearance(path: Path, n: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        out = {k: np.asarray(z[k], np.float32) for k in ("rgb", "brightness", "event")}
    for k, x in out.items():
        if len(x) != n:
            old = np.linspace(0.0, 1.0, len(x))
            new = np.linspace(0.0, 1.0, n)
            flat = x.reshape(len(x), -1)
            aligned = np.stack([np.interp(new, old, flat[:, j]) for j in range(flat.shape[1])], 1)
            out[k] = aligned.reshape((n,) + x.shape[1:]).astype(np.float32)
    return out


def raw_stats(aim: np.ndarray) -> dict[str, float]:
    v = np.diff(aim, axis=0) * 10.0
    a = np.diff(aim, n=2, axis=0) * 100.0
    j = np.diff(aim, n=3, axis=0) * 1000.0
    vals = {}
    for name, x in (("speed", v), ("acceleration", a), ("jerk", j)):
        vals[name + "_p99"] = float(np.quantile(np.linalg.norm(x, axis=-1), 0.99))
    vals["raw_safety_pass"] = bool(vals["speed_p99"] <= .18 and vals["acceleration_p99"] <= .5 and vals["jerk_p99"] <= 1.5)
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rpmc", type=Path, default=Path("/mnt/lightsem-data/rpmc_l2/RPMC_L2.h5"))
    ap.add_argument("--steps", type=int, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shpongle-audio", type=Path, required=True)
    ap.add_argument("--cure-audio", type=Path, required=True)
    ap.add_argument("--shpongle-appearance", type=Path, required=True)
    ap.add_argument("--cure-appearance", type=Path, required=True)
    ap.add_argument("--patch", type=Path, required=True)
    ap.add_argument("--roskilde", type=Path, default=None)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    source_counts = {"rpmc_l2": 0, "roskilde_cross_modal": 0}
    with h5py.File(args.rpmc, "r") as f:
        for name in f.keys():
            try:
                samples.append(rpmc_sample(f[name]))
                source_counts["rpmc_l2"] += 1
            except Exception:
                continue
    if args.roskilde is not None:
        with args.roskilde.open("rb") as f:
            roskilde_items = pickle.load(f)
        for item in roskilde_items:
            try:
                samples.append(roskilde_sample(item))
                source_counts["roskilde_cross_modal"] += 1
            except Exception:
                continue
        del roskilde_items
    if len(samples) < 100:
        raise RuntimeError(f"too few open-data samples: {len(samples)}")
    dim = samples[0][0].shape[1]
    model = SpeedNet(dim)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history=[]
    model.train()
    for step in range(args.steps):
        chosen = random.randrange(len(samples)); feat, target = samples[chosen]
        if len(feat) < 128: continue
        start = random.randrange(len(feat) - 128 + 1)
        x = torch.from_numpy(feat[start:start+128].T[None])
        y = torch.from_numpy(target[start:start+128][None])
        pred = model(x)
        loss = torch.nn.functional.smooth_l1_loss(pred[..., 0], y[..., 0]) + 5.0 * torch.nn.functional.smooth_l1_loss(pred[..., 1], y[..., 1])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 0 or (step + 1) % max(1, args.steps // 6) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach())})
    model.eval()
    patch = json.loads(args.patch.read_text())
    learned_patch = [row for row in patch if row["kind"] == "beam"]
    if len(learned_patch) != 8:
        raise ValueError(f"expected 8 beam patch rows, found {len(learned_patch)}")
    args.out.mkdir(parents=True, exist_ok=False)
    ckpt = args.out / "speed_model.pt"
    torch.save({"state_dict": model.state_dict(), "dim": dim, "seed": args.seed, "source_samples": len(samples)}, ckpt)
    song_cfg = {
        "shpongle": (args.shpongle_audio, args.shpongle_appearance),
        "cure_for_me": (args.cure_audio, args.cure_appearance),
    }
    reports={}
    for name, (audio_path, appearance_path) in song_cfg.items():
        feat = own_features(audio_path)
        with torch.no_grad():
            controls = model(torch.from_numpy(feat.T[None])).numpy()[0]
        aim, heading, amplitude = generate_stage_target_aim(controls, learned_patch)
        appearance = load_appearance(appearance_path, len(controls))
        pred_path = args.out / f"PREDICTIONS_{name}.npz"
        np.savez_compressed(pred_path, time=np.arange(len(controls), dtype=np.float64)/10.0, aim=aim, anchor=np.zeros((1,8,2),np.float32), speed_modulation=controls[:, 0], direction_modulation=controls[:, 1], heading=heading, stage_amplitude=amplitude, audio_latent=feat, **appearance)
        reports[name] = {"audio": str(audio_path.resolve()), "audio_sha256": sha(audio_path), "appearance": str(appearance_path.resolve()), "appearance_sha256": sha(appearance_path), "prediction": str(pred_path.resolve()), "prediction_sha256": sha(pred_path), "frames": len(controls), "duration_s": len(controls)/10.0, "speed_modulation_mean": float(controls[:, 0].mean()), "speed_modulation_std": float(controls[:, 0].std()), "direction_modulation_mean": float(controls[:, 1].mean()), "direction_modulation_std": float(controls[:, 1].std()), **raw_stats(aim)}
    training = {"schema":"lightsem.foundation_cyclic_motion_prior.training.v1","protocol_id":"foundation-cyclic-motion-prior-open-data-combo-20261021-v1","status":"diagnostic_candidate","seed":args.seed,"rpmc_path":str(args.rpmc.resolve()),"rpmc_sha256":sha(args.rpmc),"roskilde_path":str(args.roskilde.resolve()) if args.roskilde else None,"roskilde_sha256":sha(args.roskilde) if args.roskilde else None,"source_counts":source_counts,"open_data_samples":len(samples),"feature_dim":dim,"checkpoint":str(ckpt.resolve()),"checkpoint_sha256":sha(ckpt),"history":history,"songs":reports,"target_semantics":"RPMC-L2 and Roskilde public lighting abstractions supervise appearance-derived speed and group-level steering only; no pan/tilt or authored spatial target","promotion":"forbidden until rendered motion, correct-song controls, native DMX retention and human review"}
    (args.out/"TRAINING.json").write_text(json.dumps(training,indent=2,sort_keys=True)+"\n")
    (args.out/"PROTOCOL.json").write_text(json.dumps({"goal":"Test whether a controller trained on the combined open RPMC-L2 plus Roskilde Cross-Modal Metrics public appearance abstractions can drive bounded symmetric stage-space steering that survives the native renderer on both own songs.","one_seed":args.seed,"base_motion":"bounded symmetric stage-aware cyclic prior with audio-conditioned speed and group-level steering","open_data_sources":["RPMC-L2","Cross-Modal Metrics Roskilde 2023 songs.pkl"],"audio_conditioning":"common 36-D MIR features trained on both open sources then applied to independent own-song audio","no_authored_motion_claim":True,"no_posthoc_rescue":True,"raw_limits":{"speed_p99":.18,"acceleration_p99":.5,"jerk_p99":1.5},"render_gate":{"static_px":.35124809486952,"known_sweep_px":54.790561525842435,"candidate_min_px":27.570904810355977}},indent=2,sort_keys=True)+"\n")
    (args.out/"MMCA.json").write_text(json.dumps({"status":"pending","note":"Must run correct-song, shifted-audio and same-BPM controls after raw/rendered gates."},indent=2)+"\n")
    print(json.dumps(training,indent=2,sort_keys=True))

if __name__ == "__main__":
    main()
