#!/usr/bin/env python
"""Read-only source-dependence, fusion and collapse audit for a pilot model."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig
from lightsem.conductor.body_to_dmx import scene_to_dmx
from lightsem.dmx import club_rig
from lightsem.dmx.decode import decode_stream, emitted_light
from scripts.light_layers.train_body_to_dmx import load_pairs


def effective_rank(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float); x = x - x.mean(0, keepdims=True)
    s = np.linalg.svd(x, compute_uv=False)
    p = (s * s) / max(float((s * s).sum()), 1e-12)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    return float(np.exp(entropy)), entropy


def cosine_mean(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
    if len(x) > 1200: x = x[np.linspace(0, len(x) - 1, 1200).astype(int)]
    return float((x @ x.T)[np.triu_indices(len(x), 1)].mean())


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, int); scores = np.asarray(scores, float)
    positives = int(labels.sum())
    if positives == 0: return float("nan")
    order = np.argsort(-scores, kind="mergesort"); hits = labels[order].cumsum()
    precision = hits / np.arange(1, len(labels) + 1)
    return float((precision * labels[order]).sum() / positives)


def scene_to_metrics(scene, body_world, event, angles=None):
    rig = club_rig(); stream = scene_to_dmx(scene, body_world, rig, event, angles=angles); decoded = decode_stream(rig, stream)
    light = emitted_light(decoded); heads = np.array([i for i, f in enumerate(rig.fixtures) if f.kind == "moving_head"])
    pan, tilt = decoded["pan"][:, heads], decoded["tilt"][:, heads]
    bins = np.unique(np.stack([np.floor(pan * 12), np.floor(tilt * 8)], -1).reshape(-1, 2), axis=0)
    chunks = [stream[:, f.address - 1:f.address - 1 + f.n_channels].tobytes() for f in rig.fixtures]
    duplicates = sum(chunks[i] == chunks[j] for i in range(len(chunks)) for j in range(i))
    return {"dmx_sha256": hashlib.sha256(stream.tobytes()).hexdigest(), "finite": bool(np.isfinite(stream).all()),
            "inactive_fraction": float(np.mean(light.mean(1) < .02)), "brightness_mean": float(light.mean()),
            "pan_span_deg": ((pan.max(0) - pan.min(0)) * 540).tolist(), "tilt_span_deg": ((tilt.max(0) - tilt.min(0)) * 270).tolist(),
            "angular_occupancy_bins": int(len(bins)), "exact_duplicate_fixture_streams": int(duplicates)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=Path, required=True); ap.add_argument("--body-root", type=Path, required=True)
    ap.add_argument("--ids", required=True); ap.add_argument("--cache", type=Path, default=Path("/mnt/lightsem-data/cache")); ap.add_argument("--manifest", type=Path, default=Path("/mnt/lightsem-data/beatsaver/manifest.json")); ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    state = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False); cfg = BodyToDMXConfig(**state["config"])
    model = BodyToDMX(cfg); model.load_state_dict(state["model"], strict=True); model.eval()
    if isinstance(state["stats"], dict):
        bm, bs, am, ass = [torch.tensor(state["stats"][k]) for k in ("body_mean", "body_std", "audio_mean", "audio_std")]
    else:
        bm, bs, am, ass = [torch.tensor(v) for v in state["stats"]]
    pairs = load_pairs(args.body_root, args.ids.split(","), args.cache, args.manifest)
    if len(pairs) < 2: raise RuntimeError("need at least two complete pairs")
    rows = []; all_h = []; all_scene = []; all_add = []
    for pi, p in enumerate(pairs):
        body = torch.tensor(p["body"]); audio = torch.tensor(p["audio"]); n = len(body)
        with torch.no_grad():
            z = model(((body - bm) / bs)[None], ((audio - am) / ass)[None]); za = model(((body - bm) / bs)[None], ((audio - am) / ass)[None], additive=True)
        scene = z["scene"][0].numpy(); add_scene = za["scene"][0].numpy(); angles = torch.sigmoid(z["angle_logit"])[0].numpy().reshape(n, 6, 2); h = z["h"][0].numpy(); all_h.append(h); all_scene.append(scene); all_add.append(add_scene)
        # Frozen-body intervention: keep the first body feature at every frame.
        frozen = body[:1].repeat(n, 1)
        with torch.no_grad(): frozen_out = model(((frozen - bm) / bs)[None], ((audio - am) / ass)[None]); frozen_scene = frozen_out["scene"][0].numpy(); frozen_angles = torch.sigmoid(frozen_out["angle_logit"])[0].numpy().reshape(n, 6, 2)
        # Limb intervention: modify the first hand/foot-associated feature block
        # in the explicit body vector; it is a sensitivity probe, not a semantic label.
        limb = body.clone(); limb[:, 66 + 18:66 + 24] += 0.35
        with torch.no_grad(): limb_out = model(((limb - bm) / bs)[None], ((audio - am) / ass)[None]); limb_scene = limb_out["scene"][0].numpy(); limb_angles = torch.sigmoid(limb_out["angle_logit"])[0].numpy().reshape(n, 6, 2)
        other = pairs[(pi + 1) % len(pairs)]; m = min(n, len(other["audio"]))
        with torch.no_grad(): swap_out = model(((body[:m] - bm) / bs)[None], ((torch.tensor(other["audio"][:m]) - am) / ass)[None]); swap_scene = swap_out["scene"][0].numpy(); swap_angles = torch.sigmoid(swap_out["angle_logit"])[0].numpy().reshape(m, 6, 2)
        def bounded(s):
            q = s.copy(); q[:, :2] = np.tanh(q[:, :2]); q[:, 2:] = 1 / (1 + np.exp(-q[:, 2:])); return q
        # World body is not needed for source tests; use normalized body motion
        # as a safe angular proxy only for the emitted-stream audit.
        # The first 66 body features are root-relative xyz positions.  Keep all
        # axes in the proxy; dropping y/z would manufacture zero tilt coverage.
        world_proxy = body[:, :66].numpy().reshape(n, 22, 3).astype(np.float32)
        metrics = scene_to_metrics(bounded(scene), world_proxy, 1 / (1 + np.exp(-z["event_logit"][0, :, 0].numpy())), angles=angles)
        event_scores = torch.sigmoid(z["event_logit"])[0, :, 0].numpy()
        shifted_scores = torch.sigmoid(model(((body - bm) / bs)[None], ((torch.roll(audio, shifts=20, dims=0) - am) / ass)[None])["event_logit"])[0, :, 0].detach().numpy()
        rows.append({"id": p["id"], **metrics, "event_ap": average_precision(p["event"], event_scores), "event_ap_shifted_audio": average_precision(p["event"], shifted_scores),
                     "frozen_body_scene_delta": float(np.mean(np.abs(scene - frozen_scene))), "frozen_body_angle_delta": float(np.mean(np.abs(angles - frozen_angles))),
                     "limb_perturb_scene_delta": float(np.mean(np.abs(scene - limb_scene))), "limb_perturb_angle_delta": float(np.mean(np.abs(angles - limb_angles))),
                     "audio_swap_scene_delta": float(np.mean(np.abs(scene[:m] - swap_scene))), "audio_swap_angle_delta": float(np.mean(np.abs(angles[:m] - swap_angles))),
                     "full_vs_additive_scene_delta": float(np.mean(np.abs(scene - add_scene))), "latent_effective_rank": effective_rank(h)[0],
                     "latent_spectral_entropy": effective_rank(h)[1], "latent_mean_pairwise_cosine": cosine_mean(h)})
    hr, he = effective_rank(np.concatenate(all_h)); sr, se = effective_rank(np.concatenate(all_scene)); ar, ae = effective_rank(np.concatenate(all_add))
    report = {"schema_version": 1, "status": "diagnostic_only", "reason": "unseen-song bodies lack paired active-baseline safety evidence; public targets do not supervise physical angles", "model": str(args.run / "model.pt"), "model_sha256": hashlib.sha256((args.run / "model.pt").read_bytes()).hexdigest(), "pairs": rows,
              "aggregate": {"latent_effective_rank": hr, "latent_spectral_entropy": he, "scene_effective_rank": sr, "scene_spectral_entropy": se, "additive_scene_effective_rank": ar, "additive_scene_spectral_entropy": ae,
                             "mean_event_ap": float(np.nanmean([r["event_ap"] for r in rows])), "mean_event_ap_shifted_audio": float(np.nanmean([r["event_ap_shifted_audio"] for r in rows])),
                             "mean_frozen_body_delta": float(np.mean([r["frozen_body_scene_delta"] for r in rows])), "mean_frozen_body_angle_delta": float(np.mean([r["frozen_body_angle_delta"] for r in rows])), "mean_limb_delta": float(np.mean([r["limb_perturb_scene_delta"] for r in rows])), "mean_limb_angle_delta": float(np.mean([r["limb_perturb_angle_delta"] for r in rows])), "mean_audio_swap_delta": float(np.mean([r["audio_swap_scene_delta"] for r in rows])), "mean_audio_swap_angle_delta": float(np.mean([r["audio_swap_angle_delta"] for r in rows])), "mean_full_additive_delta": float(np.mean([r["full_vs_additive_scene_delta"] for r in rows]))}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
