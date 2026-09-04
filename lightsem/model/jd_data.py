"""Training tensor for the Just Dance corpus, and its splits by song.

Same layout as `light_data.build_tensor` (X = mel + AUDIO_SCALARS from the
bus, A/B/C from descriptors z-scored with the corpus scalers, clip ids) so
the same trainer and evaluator consume it. Differences that matter:

* the bus is sliced to the coach's active span (`start_s` from the index),
  because the capture opens on menus and closes on a results screen;
* every coach of a multi-coach routine is its own clip with the same
  `music_id`, as several AIST++ dancers share a track;
* splits are by `song_key`, never by file: fifteen songs exist in several
  versions and a version in test with its twin in train is a leak.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..conductor.scalers import load_scalers
from ..motion.io import CACHE_ROOT, load_bus, load_descriptors, slice_bus
from .light_data import AUDIO_SCALARS, BLOCK_A, BLOCK_B, BLOCK_C, Splits

JD_TENSOR = CACHE_ROOT / "train_tensor_jd.npz"
JD_STRUCT = CACHE_ROOT / "justdance_structure"
#: H-C conditioning channels, appended to X after the bus features: role
#: one-hot (intro, breakdown, build-up, drop, outro), section energy, and the
#: signed seconds to the next energy rise and to the next drop squashed by
#: tanh(x / 30). Whole-track information a causal model cannot have.
STRUCT_DIM = 8


def structure_channels(yt: str, offset_s: float, T: int, fps: float = 40.0) -> np.ndarray:
    z = np.load(JD_STRUCT / f"{yt}.npz", allow_pickle=False)
    i0 = int(round(offset_s * fps))
    role = z["role"][i0:i0 + T]; energy = z["energy"][i0:i0 + T]
    rise = z["to_rise_s"][i0:i0 + T]; drop = z["to_drop_s"][i0:i0 + T]
    n = len(role)
    out = np.zeros((T, STRUCT_DIM), dtype=np.float32)
    out[np.arange(n), np.clip(role, 0, 4)] = 1.0
    out[:n, 5] = np.nan_to_num(energy, nan=0.0)
    out[:n, 6] = np.tanh(np.nan_to_num(rise, nan=0.0) / 30.0)
    out[:n, 7] = np.tanh(np.nan_to_num(drop, nan=0.0) / 30.0)
    return out
JD_INDEX = CACHE_ROOT / "justdance_motion_index.json"
JD_MOTION = CACHE_ROOT / "justdance_motion"
JD_AUDIO = CACHE_ROOT / "justdance_audio"


def make_jd_splits(meta: list[dict], seed: int = 0, val_frac: float = 0.1,
                   test_frac: float = 0.15) -> Splits:
    """By song key. `test_genre` and `test_dancer` are empty: one performer
    per song, and no genre label the corpus itself provides."""
    rng = np.random.default_rng(seed)
    keys = sorted({m["song_key"] for m in meta})
    perm = rng.permutation(keys).tolist()
    n_test = max(1, int(round(test_frac * len(keys))))
    n_val = max(1, int(round(val_frac * len(keys))))
    test, val = set(perm[:n_test]), set(perm[n_test:n_test + n_val])
    idx = {"train": [], "val_track": [], "test_genre": [], "test_dancer": []}
    for i, m in enumerate(meta):
        if m["song_key"] in test:
            idx["test_dancer"].append(i)      # the held-out-song test set
        elif m["song_key"] in val:
            idx["val_track"].append(i)
        else:
            idx["train"].append(i)
    return Splits(**{k: np.array(v, dtype=int) for k, v in idx.items()})


def build_jd_tensor(audio_dir: Path | None = None, out: Path | None = None,
                    min_cov: float = 0.8, structure: str = "none", seed: int = 0) -> None:
    """`structure`: "none", "real" (H-C conditioning) or "shuffled" (the
    control: every clip gets another song's structure track, so a gain that
    survives it is a bias term, not structure)."""
    audio_dir = Path(audio_dir) if audio_dir else JD_AUDIO
    out = Path(out) if out else JD_TENSOR
    idx = json.loads(JD_INDEX.read_text())
    sc = load_scalers()
    bus_cache: dict[str, object] = {}
    X_all, A_all, B_all, C_all, clip_id, meta = [], [], [], [], [], []
    rows = [r for r in idx if r.get("coverage", 0.0) >= min_cov]
    if structure == "shuffled":
        rng = np.random.default_rng(seed)
        others = [r["music_id"] for r in rows]
        perm = rng.permutation(len(rows))
        donor = {rows[i]["name"]: others[perm[i]] for i in range(len(rows))}
        for i in range(len(rows)):                      # never a song's own track
            if donor[rows[i]["name"]] == rows[i]["music_id"]:
                j = (i + 1) % len(rows); donor[rows[i]["name"]] = rows[j]["music_id"]
    for r in rows:
        if r.get("coverage", 0.0) < min_cov:
            continue
        d = load_descriptors(JD_MOTION / f"{r['name']}.npz")
        mid = r["music_id"]
        if mid not in bus_cache:
            bus_cache[mid] = load_bus(audio_dir / f"{mid}.npz")
        bus = slice_bus(bus_cache[mid], duration_s=float(r["duration_s"]),
                        offset_s=float(r.get("start_s", 0.0)))
        T = min(len(d.times), len(bus.times))
        if T < 120:
            continue
        mel = bus.mel[:, :T].T
        sca = np.stack([bus.channels[k][:T] for k in AUDIO_SCALARS], 1)
        X = np.concatenate([mel, sca], 1).astype(np.float32)
        if structure != "none":
            src = mid if structure == "real" else donor[r["name"]]
            off = float(r.get("start_s", 0.0)) if structure == "real" else 0.0
            X = np.concatenate([X, structure_channels(src, off, T)], 1).astype(np.float32)

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
                     "genre": "jd", "choreo_id": r["choreo_id"], "song_key": r["song_key"],
                     "n_coaches": r.get("n_coaches", 1), "coach_index": r.get("coach_index", 0),
                     "corpus": "jd", "structure": structure, "T": int(T)})
    if not meta:
        raise SystemExit("no Just Dance clips with the required coverage")
    X = np.concatenate(X_all); A = np.concatenate(A_all)
    B = np.concatenate(B_all); C = np.concatenate(C_all)
    cid = np.concatenate(clip_id)
    np.savez(out, X=X, A=A, B=B, C=C, clip_id=cid, meta=json.dumps(meta), feat_dim=X.shape[1],
             structure=structure,
             block_a=np.array(BLOCK_A), block_b=np.array(BLOCK_B), block_c=np.array(BLOCK_C))
    print(f"X {X.shape} A {A.shape} B {B.shape} (pos rate {B.mean(0).round(4).tolist()}) "
          f"C {C.shape} clips {len(meta)} songs {len({m['song_key'] for m in meta})} -> {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--min-cov", type=float, default=0.8)
    ap.add_argument("--structure", default="none", choices=["none", "real", "shuffled"])
    a = ap.parse_args()
    build_jd_tensor(audio_dir=a.audio_dir or None, out=a.out or None, min_cov=a.min_cov,
                    structure=a.structure)
