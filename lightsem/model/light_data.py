"""Training tensors for the audio -> descriptor model, and the three splits.

Splits are by TRACK, by GENRE and by DANCER, kept separate rather than merged
into one "test" number. AIST++ contains the same choreography performed by many
dancers to the same track, so a random split lets the model memorise a
choreography and report the memory as skill. Stage 0b then showed the corpus's
between-track movement variance is 65-96 % genre, which makes the unseen-genre
split the only one that can be called generalisation at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..conductor.scalers import load_scalers
from ..motion.io import CACHE_ROOT, load_bus, load_descriptors

TENSOR = CACHE_ROOT / "train_tensor.npz"

AUDIO_SCALARS = (
    "rms", "rms_db", "flux", "flux_lf", "flux_lmid", "flux_mid", "flux_hmid",
    "flux_hf", "energy_lf", "energy_lmid", "energy_mid", "energy_hmid", "energy_hf",
    "onset", "onset_ema", "flux_ema", "flux_lf_ema", "flux_hf_ema", "centroid",
    "flatness", "rolloff", "percussive_ratio", "pulse_clarity",
    "beat_phase_sin", "beat_phase_cos", "beat_accent", "bar_phase", "beat_period_s",
)

BLOCK_A = ("qom", "com_speed", "head_speed", "foot_speed", "hip_yaw_speed",
           "vert_bounce_amp", "weight", "time_eff", "spread", "section_energy",
           "footfall_rate", "vert_period_s", "metric_level")
BLOCK_B = ("footfall", "jump", "hit")
BLOCK_C = ("hand_speed", "hand_spread", "lateral_pos", "travel_speed", "symmetry",
           "laterality", "space_eff", "flow_eff", "jerk", "vert_bounce")

#: v2 split, drawn from measurement rather than from the literature.
#: Stage 1 measured each channel's held-out-track R2; only these six are
#: predictable at all (0.015 to 0.220). Regressing the rest does not produce a
#: cautious estimate of them, it produces a flat line, and a flat line drives a
#: light that flashes constantly and evenly — which is exactly what sank the
#: Stage 1 light gates. Everything unpredictable therefore moves to the sampled
#: head, where an unpredictable quantity belongs.
BLOCK_A_V2 = ("footfall_rate", "metric_level", "vert_period_s", "vert_bounce_amp",
              "com_speed", "hip_yaw_speed")
BLOCK_C_V2 = tuple([c for c in BLOCK_A if c not in BLOCK_A_V2] + list(BLOCK_C))

N_MELS = 64


@dataclass
class Splits:
    train: np.ndarray
    val_track: np.ndarray
    test_genre: np.ndarray
    test_dancer: np.ndarray


def make_splits(meta: list[dict], seed: int = 0) -> Splits:
    rng = np.random.default_rng(seed)
    genres = sorted({m["genre"] for m in meta})
    dancers = sorted({m["dancer"] for m in meta})
    tracks = sorted({m["music_id"] for m in meta})
    test_genres = set(rng.permutation(genres)[:2].tolist())
    rest_d = [d for d in dancers]
    test_dancers = set(rng.permutation(rest_d)[:6].tolist())
    val_tracks = set(
        [t for t in rng.permutation(tracks).tolist()
         if not any(m["music_id"] == t and m["genre"] in test_genres for m in meta)][:8])
    idx = {k: [] for k in ("train", "val_track", "test_genre", "test_dancer")}
    for i, m in enumerate(meta):
        if m["genre"] in test_genres:
            idx["test_genre"].append(i)
        elif m["dancer"] in test_dancers:
            idx["test_dancer"].append(i)
        elif m["music_id"] in val_tracks:
            idx["val_track"].append(i)
        else:
            idx["train"].append(i)
    return Splits(**{k: np.array(v, dtype=int) for k, v in idx.items()})


def build_tensor() -> None:
    idx = json.loads((CACHE_ROOT / "motion_index.json").read_text())
    sc = load_scalers()
    bus_cache: dict[str, object] = {}
    X_all, A_all, B_all, C_all, clip_id, meta = [], [], [], [], [], []
    for i, r in enumerate(idx):
        d = load_descriptors(CACHE_ROOT / "motion" / f"{r['name']}.npz")
        mid = r["music_id"]
        if mid not in bus_cache:
            bus_cache[mid] = load_bus(CACHE_ROOT / "audio" / f"{mid}.npz")
        bus = bus_cache[mid]
        T = min(len(d.times), len(bus.times))
        if T < 120:
            continue
        mel = bus.mel[:, :T].T
        sca = np.stack([bus.channels[k][:T] for k in AUDIO_SCALARS], 1)
        X = np.concatenate([mel, sca], 1).astype(np.float32)

        def z(name, v):
            p = sc.get(name, {"median": 0.0, "iqr": 1.0})
            return (v - p["median"]) / (1.5 * p["iqr"])

        A = np.stack([z(k, d.channels[k][:T]) for k in BLOCK_A], 1).astype(np.float32)
        C = np.stack([z(k, d.channels[k][:T]) for k in BLOCK_C], 1).astype(np.float32)
        B = np.zeros((T, len(BLOCK_B)), dtype=np.float32)
        for bi, ev in enumerate(BLOCK_B):
            t = np.asarray(d.events.get(ev, []))
            if t.size:
                k = np.clip(np.round(t * d.fps).astype(int), 0, T - 1)
                B[k, bi] = 1.0
        X_all.append(X); A_all.append(A); B_all.append(B); C_all.append(C)
        clip_id.append(np.full(T, len(meta), dtype=np.int32))
        meta.append({"name": r["name"], "music_id": mid, "dancer": r["dancer"],
                     "genre": r["genre"], "choreo_id": r["choreo_id"], "T": int(T)})
    X = np.concatenate(X_all); A = np.concatenate(A_all)
    B = np.concatenate(B_all); C = np.concatenate(C_all)
    cid = np.concatenate(clip_id)
    np.savez(TENSOR, X=X, A=A, B=B, C=C, clip_id=cid,
             meta=json.dumps(meta), feat_dim=X.shape[1],
             block_a=np.array(BLOCK_A), block_b=np.array(BLOCK_B),
             block_c=np.array(BLOCK_C))
    print(f"X {X.shape} A {A.shape} B {B.shape} (pos rate "
          f"{B.mean(0).round(4).tolist()}) C {C.shape} clips {len(meta)} -> {TENSOR}")


def load_tensor():
    z = np.load(TENSOR, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return z, meta


if __name__ == "__main__":
    build_tensor()
