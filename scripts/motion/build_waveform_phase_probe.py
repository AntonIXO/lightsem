#!/usr/bin/env python3
"""Extract a CPU-only complex-STFT phase-prediction probe.

This is a frozen feature diagnostic, not a generator intervention.  It uses
the complex-domain prediction residual from consecutive STFT frames and keeps
bandwise energy/validity so weak-bin phase noise cannot masquerade as a strong
musical event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract(audio: Path, body: Path, out: Path) -> dict:
    if out.exists():
        raise FileExistsError(out)
    waveform, sr = sf.read(audio, always_2d=True)
    waveform = waveform.mean(1).astype(np.float32)
    n_fft = 2048
    hop = int(round(sr / 30.0))
    noverlap = n_fft - hop
    freq, time, z = stft(waveform, fs=sr, window="hann", nperseg=n_fft,
                         noverlap=noverlap, nfft=n_fft, boundary=None, padded=False)
    z = z.T.astype(np.complex64)
    magnitude = np.abs(z)
    phase = np.angle(z)
    predicted = magnitude[1:-1] * np.exp(1j * (2.0 * phase[1:-1] - phase[:-2])) if len(z) >= 3 else z[:0]
    residual = np.abs(z[2:] - predicted)
    phase_error = np.angle(np.exp(1j * (phase[2:] - (2.0 * phase[1:-1] - phase[:-2]))))
    valid = magnitude[2:] > np.maximum(np.quantile(magnitude[2:], .2, axis=0, keepdims=True), 1e-7)
    bands = ((20., 200.), (200., 1000.), (1000., 4000.), (4000., 10000.))
    features = []
    names = []
    residual_time = time[2:]
    for lo, hi in bands:
        mask = (freq >= lo) & (freq < hi)
        if not mask.any():
            raise ValueError(f"STFT band has no bins: {(lo, hi)}")
        energy = magnitude[2:, mask].mean(1)
        residual_value = residual[:, mask].mean(1)
        weighted_phase = (np.abs(phase_error[:, mask]) * valid[:, mask]).sum(1) / valid[:, mask].sum(1).clip(min=1)
        valid_fraction = valid[:, mask].mean(1)
        features.extend((energy, residual_value, weighted_phase, valid_fraction))
        names.extend((f"band_{int(lo)}_{int(hi)}_energy", f"band_{int(lo)}_{int(hi)}_complex_residual",
                      f"band_{int(lo)}_{int(hi)}_phase_error", f"band_{int(lo)}_{int(hi)}_valid_fraction"))
    features = np.stack(features, 1).astype(np.float32)
    with np.load(body, allow_pickle=False) as values:
        body_time = values["time"].astype(np.float64)
        world = values["world"].astype(np.float32)
    root_relative = world - world[:, :1]
    speed = np.linalg.norm(np.diff(root_relative, axis=0, prepend=root_relative[:1]) * 30., axis=-1).mean(1)
    probe = np.stack([np.interp(body_time, residual_time, features[:, i]) for i in range(features.shape[1])], 1)
    residual_interp = np.stack([np.interp(body_time, residual_time, residual[:, (freq >= lo) & (freq < hi)].mean(1))
                                for lo, hi in bands], 1)
    corr = [float(np.corrcoef(probe[:, i], speed)[0, 1]) if np.std(probe[:, i]) > 1e-9 and np.std(speed) > 1e-9 else None
            for i in range(probe.shape[1])]
    out.mkdir(parents=True, exist_ok=False)
    arrays = out / "phase_probe.npz"
    np.savez_compressed(arrays, time=body_time, features=probe, names=np.asarray(names), band_residual=residual_interp,
                        body_speed=speed)
    report = {
        "schema_version": 1,
        "status": "diagnostic_only",
        "audio": str(audio), "audio_sha256": sha(audio),
        "body": str(body), "body_sha256": sha(body),
        "sample_rate": int(sr), "n_fft": n_fft, "hop_samples": hop,
        "feature_clock_hz": 30.0, "bands_hz": bands,
        "features": names, "feature_shape": list(probe.shape),
        "phase_prediction": "X[t-1] magnitude with linearly extrapolated phase from t-2,t-1",
        "body_speed_correlation": corr,
        "scope": "CPU frozen waveform probe; phase residual is not an authored event or choreography label",
    }
    (out / "REPORT.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "REPORT.json.sha256.json").write_text(json.dumps({"sha256": sha(out / "REPORT.json")}) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--body", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(extract(args.audio, args.body, args.out), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
