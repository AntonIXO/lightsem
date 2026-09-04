"""If audio->lighting-events is real, WHICH audio property carries it?

Run only if probe_event_cv.py clears its permutation null. A z-score says
something is there; it does not say what, and "openl3 embeddings predict it"
is not a finding anyone can build a rig on.

openl3 is a 512-d learned embedding -- opaque. So test interpretable audio
descriptors computed from the SAME extracted features, and ask which family
carries the event signal:

  ENERGY    - RMS / loudness envelope and its derivative. If this is the whole
              story, the ICLR paper's "rule-driven" claim stands and no learned
              model is needed: a loudness change detector suffices.
  SPECTRAL  - centroid/bandwidth/rolloff (brightness of timbre). Drops and
              breakdowns change timbre, not just level.
  HARMONIC  - chroma change (key/chord movement). Section boundaries.
  PERCUSSIVE- onset/zero-crossing rate. Rhythmic density.

Each family is given the SAME rhythm-baseline treatment and the same
permutation null as the main probe, so the comparison is apples-to-apples.

WHY THIS MATTERS FOR THE PROJECT: if energy alone carries it, the honest
conclusion is that the industry's L2 threshold triggers are already near the
ceiling and L3 has no headroom on this data. If spectral/harmonic families add
over energy, that is the specific, defensible claim that a semantic model is
worth building -- and it names which features it must encode.

NOTE: this probe reads the ORIGINAL RPMC feature groups over HTTP range reads
(lightsem/data/rpmc_remote.py), because extract_rpmc.py kept only openl3 and
discarded the interpretable descriptors.
"""
from __future__ import annotations

import json

import numpy as np

FAMILIES = {
    "energy": ["mel_spectrogram_db"],
    "spectral": ["spectral_centroids", "spectral_bandwidth", "spectral_rolloff",
                 "spectral_contrast"],
    "harmonic": ["chroma_cens"],
    # NOTE: RPMC-L2 has NO zero_crossing_rate despite an earlier schema note
    # claiming it. Verified layout (probe_h5_layout.py) has exactly:
    # chroma_{cens,cqt,stft}, cqt, mel_spectrogram{,_db}, mfcc, openl3,
    # spectral_{bandwidth,centroids,contrast,rolloff}.
    # Use MFCC as the timbre/percussive-texture family instead.
    "timbre": ["mfcc"],
}


def summarise(name, arr):
    """Reduce a (D,L) feature block to a few interpretable channels."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if name == "mel_spectrogram_db":
        # loudness envelope + spectral flux (positive change only)
        loud = a.mean(0, keepdims=True)
        flux = np.maximum(np.diff(a, axis=1, prepend=a[:, :1]), 0).mean(0, keepdims=True)
        return np.concatenate([loud, flux], 0)
    if name == "chroma_cens":
        # chroma change: cosine distance between consecutive frames
        n = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-8)
        d = 1.0 - (n[:, 1:] * n[:, :-1]).sum(0)
        d = np.concatenate([d[:1], d])[None, :]
        return np.concatenate([a.mean(0, keepdims=True), d], 0)
    return a


def to_10hz(a: np.ndarray, n_out: int) -> np.ndarray:
    """Resample a (D,L) block at the native 43.07 Hz feature rate onto the
    10 Hz grid used everywhere else, by averaging within each output bin.

    Averaging (not sampling) is deliberate: these are the descriptors whose
    CHANGE we care about, and point-sampling a 43 Hz signal at 10 Hz aliases.
    """
    D, L = a.shape
    edges = np.linspace(0, L, n_out + 1).astype(int)
    out = np.empty((D, n_out), np.float32)
    cs = np.concatenate([np.zeros((D, 1), np.float64), np.cumsum(a, 1)], 1)
    for i in range(n_out):
        lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
        out[:, i] = (cs[:, hi] - cs[:, lo]) / (hi - lo)
    return out


def main():
    import glob
    import os

    import h5py  # noqa: F401

    from lightsem.data.rpmc_remote import open_h5

    # only songs we already have targets for, so features and y join cleanly
    have = {os.path.basename(p)[:-4]
            for p in glob.glob("data/cache/rpmc_train/*.npz")}
    f, raw = open_h5(verbose=False)
    keys = [k for k in f.keys() if k in have]
    # ALL songs with targets. The 30-song run gave 274 events vs the main
    # probe's 1376 and every family landed inside its null -- that is a power
    # problem, not an answer, so match the main probe's event count.
    print(f"songs with existing targets: {len(have)}; reading {len(keys)}")

    store, meta = {}, {}
    for i, k in enumerate(keys):
        try:
            m = f[k]["music"]
            # y length on the 10 Hz grid, matching lightsem.model.data
            y = np.load(f"data/cache/rpmc_train/{k}.npz")["y"]
            n_out = len(y) // 3
            if n_out < 160:
                print(f"  [{i+1}/{len(keys)}] {k[:16]} skip (short)")
                continue
            per = {}
            for fam, names in FAMILIES.items():
                blocks = [summarise(nm, m[nm][:]) for nm in names if nm in m]
                if not blocks:
                    continue
                L = min(b.shape[1] for b in blocks)
                per[fam] = to_10hz(
                    np.concatenate([b[:, :L] for b in blocks], 0), n_out)
            store[k] = per
            meta[k] = {a: list(v.shape) for a, v in per.items()}
            print(f"  [{i+1}/{len(keys)}] {k[:16]}  "
                  + " ".join(f"{a}:{v.shape[0]}x{v.shape[1]}"
                             for a, v in per.items())
                  + f"   net {raw.bytes_fetched/1e6:.0f} MB")
        except Exception as e:                                  # noqa: BLE001
            print(f"  [{i+1}/{len(keys)}] {k[:16]} FAILED: {type(e).__name__}: {e}")

    print(f"\nsongs ok: {len(store)}   total network: {raw.bytes_fetched/1e6:.1f} MB")
    np.savez_compressed(
        "data/cache/interp_feats.npz",
        **{f"{k}__{fam}": v for k, d in store.items() for fam, v in d.items()})
    json.dump(meta, open("data/cache/interp_feats_meta.json", "w"), indent=2)
    print("wrote data/cache/interp_feats.npz + _meta.json")


if __name__ == "__main__":
    main()
