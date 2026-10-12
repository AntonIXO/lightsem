#!/usr/bin/env python
"""Decode one body-to-DMX pilot output without rendering video."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch

from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig
from lightsem.conductor.body_to_dmx import scene_to_dmx
from lightsem.dmx import club_rig
from lightsem.dmx.decode import decode_stream, emitted_light
from scripts.light_layers.train_body_to_dmx import body_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--body", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    state = torch.load(args.run / "model.pt", map_location="cpu", weights_only=False)
    cfg = BodyToDMXConfig(**state["config"]); model = BodyToDMX(cfg); model.load_state_dict(state["model"]); model.eval()
    bm, bs, am, ass = [torch.tensor(v) for v in state["stats"]]
    with np.load(args.target, allow_pickle=False) as z:
        audio, target, event_target = z["x"].astype(np.float32), z["y"].astype(np.float32), z["ev_any"].astype(np.float32)
    times = np.arange(len(audio), dtype=np.float64) / 10.
    body, body_meta = body_features(args.body, times); n = min(len(body), len(audio))
    body, audio, target, event_target = body[:n], audio[:n], target[:n], event_target[:n]
    with np.load(args.body, allow_pickle=False) as z:
        native_t, world = z["time"].astype(float), z["world"].astype(np.float32)
    world10 = np.stack([np.interp(times[:n], native_t, world[:, j, k]) for j in range(22) for k in range(3)], 1).reshape(n, 22, 3)
    with torch.no_grad():
        pred = model(((torch.tensor(body) - bm) / bs)[None], ((torch.tensor(audio) - am) / ass)[None])
        scene = pred["scene"][0].numpy(); event = torch.sigmoid(pred["event_logit"][0, :, 0]).numpy()
        angles = torch.sigmoid(pred["angle_logit"])[0].numpy().reshape(n, 6, 2)
    scene[:, :2] = np.tanh(scene[:, :2]); scene[:, 2:] = 1. / (1. + np.exp(-scene[:, 2:]))
    rig = club_rig(); stream = scene_to_dmx(scene, world10, rig, event, angles=angles); decoded = decode_stream(rig, stream); light = emitted_light(decoded)
    spans = []
    duplicate = 0
    for i, f in enumerate(rig.fixtures):
        a = f.address - 1; spans.append(stream[:, a:a + f.n_channels].tobytes())
        for j in range(i): duplicate += int(spans[i] == spans[j])
    heads = np.array([i for i, f in enumerate(rig.fixtures) if f.kind == "moving_head"])
    pan, tilt = decoded["pan"][:, heads], decoded["tilt"][:, heads]
    bins = np.unique(np.stack([np.floor(pan * 12), np.floor(tilt * 8)], -1).reshape(-1, 2), axis=0)
    report = {"schema_version": 1, "status": "diagnostic_only", "reason": "source-body pairing is unqualified and no active-baseline safety comparison exists; no video rendered",
              "body": str(args.body), "body_sha256": hashlib.sha256(args.body.read_bytes()).hexdigest(), "target": str(args.target),
              "model": str(args.run / "model.pt"), "dmx_sha256": hashlib.sha256(stream.tobytes()).hexdigest(), "shape": list(stream.shape),
              "finite": bool(np.isfinite(stream).all()), "inactive_fraction": float(np.mean(light.mean(1) < .02)),
              "exact_duplicate_fixture_streams": duplicate, "pan_span_deg": ((pan.max(0) - pan.min(0)) * 540).tolist(),
              "tilt_span_deg": ((tilt.max(0) - tilt.min(0)) * 270).tolist(), "angular_occupancy_bins": int(len(bins)),
              "brightness_mean": float(light.mean()), "event_rate": float(np.mean(event > .5)),
              "target_scene_mae": float(np.mean(np.abs(scene - target))), "target_event_brier": float(np.mean((event - event_target) ** 2)),
              "raw_body": body_meta["raw_metrics"]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists(): raise FileExistsError(args.out)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
