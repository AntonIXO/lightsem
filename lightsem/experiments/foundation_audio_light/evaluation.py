"""Machine checks for the continuous-attention pilot."""

import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_body_metrics(world, fps=30.0):
    x = np.asarray(world, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (22, 3):
        raise ValueError("expected T,22,3 world body")
    finite = bool(np.isfinite(x).all())
    velocity = np.diff(x, axis=0, prepend=x[:1]) * fps
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1]) * fps
    jerk = np.diff(acceleration, axis=0, prepend=acceleration[:1]) * fps
    root = x[:, 0]
    speed = np.linalg.norm(velocity, axis=-1)
    return {
        "finite": finite,
        "frames": int(len(x)),
        "speed_p99": float(np.quantile(speed, .99)),
        "acceleration_p99": float(np.quantile(np.linalg.norm(acceleration, axis=-1), .99)),
        "jerk_p99": float(np.quantile(np.linalg.norm(jerk, axis=-1), .99)),
        "root_jerk_p99": float(np.quantile(np.linalg.norm(np.diff(np.diff(np.diff(root, axis=0), axis=0), axis=0) * fps**3, axis=-1), .99)) if len(x) >= 4 else 0.0,
    }


@torch.no_grad()
def permutation_error(model, audio, body, fixtures, perm):
    a = model(audio, body, fixtures)
    b = model(audio, body, fixtures[:, perm])
    inv = torch.argsort(perm)
    return float((a["brightness"] - b["brightness"][:, :, inv]).abs().max().cpu())


def write_receipt(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
