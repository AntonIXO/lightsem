#!/usr/bin/env python
"""Train and audit the body-to-DMX multimodal pilot.

This is deliberately a small, one-seed research pilot.  BeatSaver's group
target supervises observable scene/event channels; it does not pretend to
supervise physical pan/tilt.  The script refuses to overwrite an artifact root
and emits one canonical JSON receipt plus a checkpoint and prediction arrays.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random, subprocess, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lightsem.model.body_to_dmx import BodyToDMX, BodyToDMXConfig, causality_probe, masked_loss
from lightsem.experiments.native_repair.assessment import raw_metrics
from lightsem.conductor.body_transfer_v2.common import prepare_features
from lightsem.conductor.body_transfer_v2.composition import laban_instantaneous_shape_effort


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def body_features(path: Path, times: np.ndarray) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as z:
        t = np.asarray(z["time"], float); world = np.asarray(z["world"], np.float32)
        contacts = np.asarray(z["contacts"], np.float32)
    if world.ndim != 3 or world.shape[1:] != (22, 3) or not np.isfinite(world).all():
        raise ValueError(f"invalid body {path}")
    rel = world - world[:, :1]
    vel = np.vstack([np.zeros((1, 22, 3), np.float32), np.diff(rel, axis=0) * 30.0])
    acc = np.vstack([np.zeros((1, 22, 3), np.float32), np.diff(vel, axis=0) * 30.0])
    # Reuse the sealed V2 Laban implementation from the prior transfer study;
    # these are shape/effort proxies, never an autonomous temporal controller.
    feat = prepare_features(world, fps=30.0)
    laban_result = laban_instantaneous_shape_effort(feat)
    lp = np.asarray(laban_result.points, np.float32)
    activity = np.asarray(laban_result.activity, np.float32)
    laban = np.stack([
        np.mean(np.abs(lp[:, :, 0]), 1), np.mean(np.abs(lp[:, :, 1]), 1),
        np.mean(np.abs(lp[:, :, 2]), 1), np.std(lp, axis=(1, 2)),
        np.mean(activity, 1), np.quantile(activity, .9, axis=1),
        np.linalg.norm(lp, axis=-1).mean(1), np.linalg.norm(vel, axis=-1).mean(1),
    ], 1).astype(np.float32)
    raw = np.concatenate([rel.reshape(len(rel), -1), vel.reshape(len(vel), -1), contacts, laban], 1).astype(np.float32)
    out = np.stack([np.interp(times, t, raw[:, i]) for i in range(raw.shape[1])], 1)
    return out.astype(np.float32), {"raw_metrics": raw_metrics(world), "body_sha256": sha(path)}


def pseudo_angle_targets(path: Path, times: np.ndarray, n_heads: int = 6) -> np.ndarray:
    """Weak body-geometry target for the canonical moving-head angle head.

    This target is deliberately labelled pseudo supervision: it is derived from
    source joints, not from authored physical DMX.  It trains a source-dependent
    angle interface while the authored-light corpus supervises scene/value/colour
    heads.  The target is not evidence of a human lighting operator.
    """
    with np.load(path, allow_pickle=False) as z:
        t = np.asarray(z["time"], float); world = np.asarray(z["world"], np.float32)
    rel = world - world[:, :1]
    sampled = np.stack([np.interp(times, t, rel[:, j, k]) for j in range(22) for k in range(3)], 1).reshape(len(times), 22, 3)
    out = np.full((len(times), n_heads, 2), .5, np.float32)
    for i in range(n_heads):
        j, k = (2 * i) % 22, (2 * i + 1) % 22
        out[:, i, 0] = np.clip(.5 + .10 * np.tanh(sampled[:, j, 0] + .5 * sampled[:, k, 0]), 0., 1.)
        out[:, i, 1] = np.clip(.5 + .10 * np.tanh(sampled[:, j, 1] + .5 * sampled[:, k, 1]), 0., 1.)
    return out.reshape(len(times), -1).astype(np.float32)


def load_pairs(body_root: Path, ids: list[str], cache: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    pairs = []
    for mid in ids:
        row = manifest[mid]
        key = row.get("song_key")
        if not key:
            import re
            s = re.sub(r"[^a-z0-9]", "", f"{row.get('song') or ''}{row.get('artist') or ''}".lower())
            key = hashlib.sha1(s.encode()).hexdigest()[:8]
        stem = f"{key}={mid}"
        bp = body_root / f"body_{mid}" / "body.npz"
        tp = cache / "beatsaver_visual" / f"{stem}.npz"
        if not bp.is_file() or not tp.is_file():
            continue
        with np.load(tp, allow_pickle=False) as z:
            x, y, ev = np.asarray(z["x"], np.float32), np.asarray(z["y"], np.float32), np.asarray(z["ev_any"], np.float32)
        if x.ndim != 2 or y.shape != (len(x), 6) or ev.shape != (len(x),):
            raise ValueError(f"invalid target {tp}")
        times = np.arange(len(x), dtype=np.float64) / 10.0
        b, bm = body_features(bp, times)
        angle = pseudo_angle_targets(bp, times)
        n = min(len(b), len(x)); b, x, y, ev = b[:n], x[:n], y[:n], ev[:n]
        if n < 64 or not np.isfinite(np.concatenate([b, x, y, ev[:, None]], 1)).all():
            continue
        pairs.append({"id": mid, "body": b, "audio": x, "scene": y, "event": ev, "angle": angle[:n],
                      "body_path": str(bp), "target_path": str(tp), **bm})
    return pairs


def normalize(train):
    b = np.concatenate([p["body"] for p in train]); a = np.concatenate([p["audio"] for p in train])
    bm, bs = b.mean(0), b.std(0); am, ass = a.mean(0), a.std(0)
    bs[bs < 1e-5] = 1.; ass[ass < 1e-5] = 1.
    return (bm.astype(np.float32), bs.astype(np.float32), am.astype(np.float32), ass.astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("/mnt/lightsem-data/beatsaver/manifest.json"))
    ap.add_argument("--cache", type=Path, default=Path("/mnt/lightsem-data/cache"))
    ap.add_argument("--ids", default="", help="comma-separated map ids; default all available body pairs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = ap.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    ids = args.ids.split(",") if args.ids else sorted(p.name[len("body_"):] for p in args.body_root.glob("body_*") if (p / "body.npz").exists())
    pairs = load_pairs(args.body_root, ids, args.cache, args.manifest)
    if len(pairs) < 2: raise RuntimeError(f"need at least 2 complete body/light pairs, got {len(pairs)}")
    pairs.sort(key=lambda p: p["id"])
    # Keep at least two song groups in the development split; a one-song
    # validation number is too unstable to support a cross-song claim.
    n_val = max(2, int(np.ceil(len(pairs) * 0.2))); train, val = pairs[:-n_val], pairs[-n_val:]
    stats = normalize(train)
    bm, bs, am, ass = [torch.tensor(v, device=args.device) for v in stats]
    audio_dim = int(train[0]["audio"].shape[1])
    if any(p["audio"].shape[1] != audio_dim for p in pairs):
        raise ValueError("audio feature widths differ across targets")
    cfg = BodyToDMXConfig(audio_dim=audio_dim); model = BodyToDMX(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    history = []
    for step in range(args.steps):
        p = train[step % len(train)]
        n = len(p["body"]); start = 0 if n <= args.window else random.randrange(0, n - args.window)
        sl = slice(start, min(start + args.window, n))
        body = torch.from_numpy(p["body"][sl]).to(args.device); audio = torch.from_numpy(p["audio"][sl]).to(args.device)
        scene = torch.from_numpy(p["scene"][sl]).to(args.device); event = torch.from_numpy(p["event"][sl]).to(args.device); angle = torch.from_numpy(p["angle"][sl]).to(args.device)
        pred = model((body[None] - bm) / bs, (audio[None] - am) / ass)
        losses = masked_loss(pred, scene[None], event[None], angle=angle[None]); opt.zero_grad(set_to_none=True); losses["total"].backward(); opt.step()
        if step % max(1, args.steps // 20) == 0 or step == args.steps - 1:
            history.append({"step": step, **{k: float(v.detach().cpu()) for k, v in losses.items()}})
    model.eval()
    def evaluate(rows, tag):
        scene_err, event_scores, angle_scores, ys, ps = [], [], [], [], []
        with torch.no_grad():
            for p in rows:
                body = torch.from_numpy(p["body"]).to(args.device); audio = torch.from_numpy(p["audio"]).to(args.device)
                z = model((body[None] - bm) / bs, (audio[None] - am) / ass)
                q = z["scene"][0].cpu().numpy(); e = torch.sigmoid(z["event_logit"])[0, :, 0].cpu().numpy(); a = torch.sigmoid(z["angle_logit"])[0].cpu().numpy()
                scene_err.append(float(np.mean(np.abs(q - p["scene"])))); event_scores.append(float(np.mean((e - p["event"]) ** 2))); angle_scores.append(float(np.mean(np.abs(a - p["angle"]))))
                ys.append(p["scene"]); ps.append(q)
        return {"songs": len(rows), "scene_mae": float(np.mean(scene_err)), "event_brier": float(np.mean(event_scores)), "angle_mae": float(np.mean(angle_scores)),
                "prediction_effective_rank": float(np.linalg.matrix_rank(np.concatenate(ps), tol=1e-3)),
                "target_effective_rank": float(np.linalg.matrix_rank(np.concatenate(ys), tol=1e-3))}
    report = {"schema_version": 1, "status": "diagnostic_only", "reason": "unconditioned fixed-C source bodies lack paired active stereo-baseline comparison",
              "seed": args.seed, "config": cfg.__dict__, "params": model.n_params(), "pairs": [p["id"] for p in pairs],
              "train_ids": [p["id"] for p in train], "val_ids": [p["id"] for p in val], "history": history,
              "train": evaluate(train, "train"), "val": evaluate(val, "val"),
              "causality_max_past_change": causality_probe(model, torch.zeros(1, min(64, args.window), cfg.body_dim, device=args.device), torch.zeros(1, min(64, args.window), cfg.audio_dim, device=args.device), min(32, args.window - 1)),
              "source_bodies": [{"id": p["id"], "path": p["body_path"], "sha256": p["body_sha256"], "raw_metrics": p["raw_metrics"]} for p in pairs],
              "targets": [{"id": p["id"], "path": p["target_path"], "sha256": sha(Path(p["target_path"]))} for p in pairs]}
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "stats": [v.tolist() for v in stats], "seed": args.seed}, out / "model.pt")
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "status": report["status"], "pairs": len(pairs), "val": report["val"]}, indent=2))


if __name__ == "__main__":
    main()
