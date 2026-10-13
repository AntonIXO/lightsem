#!/usr/bin/env python
"""Generate a quarantined body-to-DMX diagnostic package and preview plot."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig
from lightsem.conductor.body_to_dmx import scene_to_dmx
from lightsem.dmx import club_rig
from lightsem.dmx.decode import decode_stream, emitted_light
from lightsem.experiments.native_repair.assessment import raw_metrics
from scripts.light_layers.train_body_to_dmx import body_features


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--duration", type=float, default=45.)
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    state = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    cfg = BodyToDMXConfig(**state["config"]); model = BodyToDMX(cfg); model.load_state_dict(state["model"], strict=True); model.eval()
    stats = state["stats"]
    if isinstance(stats, dict):
        bm, bs, am, ass = [torch.tensor(stats[k]) for k in ("body_mean", "body_std", "audio_mean", "audio_std")]
    else:
        bm, bs, am, ass = [torch.tensor(v) for v in stats]
    with np.load(args.target, allow_pickle=False) as z:
        audio, target_scene, target_event = z["x"].astype(np.float32), z["y"].astype(np.float32), z["ev_any"].astype(np.float32)
    n = min(len(audio), max(1, int(args.duration * 10)))
    times = np.arange(n, dtype=np.float64) / 10.
    body, _ = body_features(args.body, times); audio = audio[:n]; target_scene = target_scene[:n]; target_event = target_event[:n]
    with np.load(args.body, allow_pickle=False) as z:
        native_t, world = z["time"].astype(float), z["world"].astype(np.float32)
    world10 = np.stack([np.interp(times, native_t, world[:, j, k]) for j in range(22) for k in range(3)], 1).reshape(n, 22, 3)
    with torch.no_grad():
        pred = model(((torch.tensor(body) - bm) / bs)[None], ((torch.tensor(audio) - am) / ass)[None])
    scene = pred["scene"][0].numpy(); scene[:, :2] = np.tanh(scene[:, :2]); scene[:, 2:] = 1 / (1 + np.exp(-scene[:, 2:]))
    event = torch.sigmoid(pred["event_logit"])[0, :, 0].numpy()
    angles = torch.sigmoid(pred["angle_logit"])[0].numpy().reshape(n, 6, 2)
    rig = club_rig(); dmx = scene_to_dmx(scene, world10, rig, event, angles=angles); decoded = decode_stream(rig, dmx); light = emitted_light(decoded)
    heads = np.array([i for i, f in enumerate(rig.fixtures) if f.kind == "moving_head"])
    report = {"schema_version": 1, "status": "diagnostic_only", "reason": "new-song source body lacks paired active-baseline safety comparison; physical angle targets are weak body pseudo-targets", "song_target": str(args.target), "song_target_sha256": sha(args.target), "body": str(args.body), "body_sha256": sha(args.body), "model": str(args.run / "model.pt"), "model_sha256": sha(args.run / "model.pt"), "frames": int(n), "fps": 10., "dmx_sha256": hashlib.sha256(dmx.tobytes()).hexdigest(), "finite_dmx": bool(np.isfinite(dmx).all()), "raw_body": raw_metrics(world[:max(2, min(len(world), int(args.duration * 30))) ]), "brightness_mean": float(light.mean()), "inactive_fraction": float(np.mean(light.mean(1) < .02)), "event_rate": float(np.mean(event > .5)), "pan_span_deg": ((decoded["pan"][:, heads].max(0) - decoded["pan"][:, heads].min(0)) * 540).tolist(), "tilt_span_deg": ((decoded["tilt"][:, heads].max(0) - decoded["tilt"][:, heads].min(0)) * 270).tolist(), "target_scene_mae": float(np.mean(np.abs(scene - target_scene))), "target_event_brier": float(np.mean((event - target_event) ** 2))}
    np.savez_compressed(out / "demo.npz", time=times.astype(np.float32), world=world10, body_features=body, scene=scene, event=event, angles=angles, dmx=dmx, dimmer=decoded["dimmer"], hue=decoded["hue"], sat=decoded["sat"], value=decoded["val"], target_scene=target_scene, target_event=target_event)
    (out / "DEMO.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(times, np.linalg.norm(world10[:, 4] - world10[:, 0], axis=1), label="hand/root distance")
    axes[0].plot(times, np.linalg.norm(world10[:, 7] - world10[:, 0], axis=1), label="hand/root distance 2", alpha=.75); axes[0].set_ylabel("body m"); axes[0].legend(loc="upper right")
    for i in range(6): axes[1].plot(times, decoded["pan"][:, heads[i]] * 540 - 270, alpha=.8); axes[1].plot(times, decoded["tilt"][:, heads[i]] * 270 - 135, alpha=.35)
    axes[1].set_ylabel("angle deg"); axes[1].set_title("moving heads: solid pan, faint tilt")
    axes[2].plot(times, light.mean(1), label="emitted brightness"); axes[2].plot(times, event, label="event probability", alpha=.8); axes[2].set_ylabel("level"); axes[2].legend(loc="upper right")
    hue = (np.arctan2(scene[:, 1], scene[:, 0]) / (2 * np.pi)) % 1.; axes[3].plot(times, hue, label="scene hue"); axes[3].plot(times, scene[:, 3], label="scene value"); axes[3].set_ylabel("scene"); axes[3].set_xlabel("seconds"); axes[3].legend(loc="upper right")
    fig.suptitle("LightSem body-to-DMX diagnostic demo — quarantined")
    fig.tight_layout(); fig.savefig(out / "preview.png", dpi=140); plt.close(fig)
    print(json.dumps({"out": str(out), "status": report["status"], "dmx_sha256": report["dmx_sha256"], "preview": str(out / "preview.png")}, indent=2))


if __name__ == "__main__": main()
