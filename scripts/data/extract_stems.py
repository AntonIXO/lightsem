#!/usr/bin/env python
"""H-G, step 1: HTDemucs 4-stem separation of the Just Dance wavs (PLAN.md 9.7).

Two modes, both saved as mono 22.05 kHz int16 per stem so the audit can iterate on
features without re-running the separator:

  whole   apply_model on the full track (demucs' own overlap-add; ~7.8 s of look-ahead
          inside each segment) -> cache/justdance_stems/<yt>.npz
  causal  the L = 0 counterpart the gate G9.7-causal asks for: trailing windows of one
          model segment (7.8 s) with hop 1.95 s, only the LAST hop of each window kept,
          per-window normalisation, so no output sample sees the future
          -> cache/justdance_stems_causal/<yt>.npz

Sharded so several processes share the GPU:

  bash run_torch.sh scripts/data/extract_stems.py --mode whole  --shard 0 --nshards 4
  bash run_torch.sh scripts/data/extract_stems.py --mode causal --shard 0 --nshards 4 --limit 80
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

CACHE = Path("/mnt/lightsem-data/cache")
WAV = CACHE / "justdance_wav"
OUT = {"whole": CACHE / "justdance_stems", "causal": CACHE / "justdance_stems_causal"}
SR_OUT = 22050


def load_stereo(path: Path, sr: int) -> torch.Tensor:
    x, r = sf.read(str(path), dtype="float32", always_2d=True)     # (T, C)
    x = torch.from_numpy(x.T)                                        # (C, T)
    if r != sr:
        import julius
        x = julius.resample_frac(x, r, sr)
    if x.shape[0] == 1:
        x = x.repeat(2, 1)
    return x


def to_mono_int16(y: torch.Tensor, sr_in: int) -> np.ndarray:
    """(S, C, T) float at sr_in -> (S, T') int16 mono at SR_OUT."""
    import julius
    m = y.mean(1)
    m = julius.resample_frac(m, sr_in, SR_OUT)
    return (m.clamp(-1, 1) * 32767).round().to(torch.int16).cpu().numpy()


def separate_whole(model, wav: torch.Tensor, dev: str) -> torch.Tensor:
    from demucs.apply import apply_model
    ref = wav.mean(0)
    mu, sd = ref.mean(), ref.std() + 1e-8
    x = ((wav - mu) / sd)[None].to(dev)
    with torch.no_grad():
        out = apply_model(model, x, shifts=0, split=True, overlap=0.25, progress=False, device=dev)[0]
    return (out * sd + mu).cpu()                                     # (S, C, T)


def separate_causal(model, wav: torch.Tensor, dev: str, seg_s: float, hop_s: float, batch: int) -> torch.Tensor:
    from demucs.apply import apply_model
    sr = model.samplerate
    W, H = int(round(seg_s * sr)), int(round(hop_s * sr))
    C, T = wav.shape
    pad = torch.cat([torch.zeros(C, W - H), wav], 1)                 # first hop sees only itself + zeros
    n_hops = int(np.ceil(T / H))
    out = torch.zeros(len(model.sources), C, n_hops * H)
    wins, ends = [], []
    def flush():
        if not wins:
            return
        xb = torch.stack(wins).to(dev)                               # (B, C, W)
        ref = xb.mean(1, keepdim=True)                               # (B, 1, W)
        mu = ref.mean(-1, keepdim=True); sd = ref.std(-1, keepdim=True) + 1e-8
        with torch.no_grad():
            yb = apply_model(model, (xb - mu) / sd, shifts=0, split=False, progress=False, device=dev)
        yb = (yb * sd[:, None] + mu[:, None]).cpu()                  # (B, S, C, W)
        for y, e in zip(yb, ends):
            out[:, :, e - H:e] = y[:, :, W - H:W]
        wins.clear(); ends.clear()
    for k in range(n_hops):
        e = (k + 1) * H                                              # end in the un-padded clock
        seg = pad[:, e + (W - H) - W: e + (W - H)]                   # last W samples ending at un-padded e
        if seg.shape[1] < W:
            seg = torch.cat([seg, torch.zeros(C, W - seg.shape[1])], 1)
        wins.append(seg); ends.append(e)
        if len(wins) == batch:
            flush()
    flush()
    return out[:, :, :T]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("whole", "causal"), default="whole")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hop", type=float, default=1.95)
    a = ap.parse_args()
    out_dir = OUT[a.mode]; out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(WAV.glob("*.wav"))
    if a.limit:
        files = files[:a.limit]
    files = files[a.shard::a.nshards]
    dev = "cuda"
    from demucs.pretrained import get_model
    model = get_model("htdemucs"); model.to(dev); model.eval()
    seg_s = float(model.models[0].segment) if hasattr(model, "models") else float(model.segment)
    print(f"{a.mode} shard {a.shard}/{a.nshards}: {len(files)} files, segment {seg_s:.2f}s, sources {model.sources}", flush=True)
    t0 = time.time(); n_done = 0; audio_s = 0.0
    for i, f in enumerate(files):
        out = out_dir / f"{f.stem}.npz"
        if out.exists():
            continue
        t1 = time.time()
        wav = load_stereo(f, model.samplerate)
        if a.mode == "whole":
            y = separate_whole(model, wav, dev)
        else:
            y = separate_causal(model, wav, dev, seg_s, a.hop, a.batch)
        stems = to_mono_int16(y, model.samplerate)
        np.savez(out, sr=SR_OUT, sources=np.array(model.sources), stems=stems,
                 mode=a.mode, seg_s=seg_s, hop_s=a.hop if a.mode == "causal" else 0.0)
        dur = wav.shape[1] / model.samplerate; audio_s += dur; n_done += 1
        print(f"[{i+1}/{len(files)}] {f.stem} {dur:.0f}s in {time.time()-t1:.1f}s "
              f"({audio_s/(time.time()-t0):.1f} s audio/s cumulative)", flush=True)
    print(f"DONE {n_done} files, {audio_s/3600:.2f} h audio in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
