from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np


REVISION = "0562a57814f6f8bbd9fdea0a25921a2fce1a841a"
WEIGHTS_SHA256 = "273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6"


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_layers(model, waveform, layers=(9,)):
    """Extract one or more hidden layers under the canonical 30/20/5 contract."""
    import torch

    layers = tuple(sorted({int(layer) for layer in layers}))
    if not layers or min(layers) < 0:
        raise ValueError("layers must contain nonnegative indices")
    sample_rate = 24000
    native_rate = 25
    core = 20 * sample_rate
    context = 5 * sample_rate
    size = 30 * sample_rate
    waveform = np.asarray(waveform, np.float32)
    padded = np.pad(waveform, (context, size), mode="reflect")
    frame_count = int(np.ceil(len(waveform) / sample_rate * native_rate))
    chunks = {layer: [] for layer in layers}
    with torch.inference_mode():
        for start in range(0, len(waveform), core):
            segment = padded[start : start + size]
            audio = torch.from_numpy(segment)[None].cuda()
            output = model(audio, output_hidden_states=True)
            take = min(20 * native_rate, frame_count - start // (sample_rate // native_rate))
            for layer in layers:
                values = output.hidden_states[layer][0, 5 * native_rate : 5 * native_rate + take]
                if values.ndim != 2 or values.shape[0] != take or not torch.isfinite(values).all():
                    raise FloatingPointError(f"bad MuQ chunk {start} layer {layer} {tuple(values.shape)}")
                chunks[layer].append(values.cpu().numpy().astype(np.float16))
    arrays = {layer: np.concatenate(values, 0)[:frame_count] for layer, values in chunks.items()}
    timestamps = np.arange(frame_count, dtype=np.float64) / native_rate
    return arrays, timestamps


def extract(model, waveform):
    arrays, timestamps = extract_layers(model, waveform, (9,))
    return arrays[9], timestamps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--layers", default="9", help="comma-separated hidden-state indices")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_muq_20261016_v3"),
    )
    args = parser.parse_args()
    layers = tuple(int(value) for value in args.layers.split(",") if value.strip())
    root = Path("/mnt/lightsem-data/experiments/lightsem/audio_light_foundation_v2_contract_20261016_v1")
    contract = json.loads((root / "CONTRACT.json").read_text())
    rows = list(contract["rows"])
    if args.limit is not None:
        rows = rows[: args.limit]
    free = shutil.disk_usage("/mnt/lightsem-data").free
    estimate = sum(int(max(1, row["duration_s"] or 180) * 25 * 1024 * 2) for row in rows)
    print(json.dumps({"rows": len(rows), "estimated_bytes": estimate, "free_gib": free / 2**30, "floor_gib": 20}, indent=2))
    if free - estimate < 20 * 2**30:
        raise RuntimeError("storage floor")
    args.out.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    import soxr
    from muq import MuQ

    model = MuQ.from_pretrained("/root/lightsem-audio-study/model_links/muq", local_files_only=True).cuda().float().eval()
    started = time.time()
    records = []
    for index, row in enumerate(rows, 1):
        wav = Path(row["wav_path"])
        output = args.out / f"{row['id']}.npz"
        partial = args.out / f"{row['id']}.npz.partial"
        if output.exists():
            print("skip existing", row["id"])
            continue
        waveform, sample_rate = sf.read(wav, dtype="float32", always_2d=True)
        waveform = waveform.mean(1)
        if sample_rate != 24000:
            waveform = soxr.resample(waveform, sample_rate, 24000)
        layer_arrays, timestamps = extract_layers(model, waveform, layers)
        selected = layer_arrays.get(9, next(iter(layer_arrays.values())))
        payload = {
            "muq": selected,
            "muq_time": timestamps,
            "wav_sha256": np.array(sha(wav)),
            "model_revision": np.array(REVISION),
            "layer": np.array(9, dtype=np.int32),
        }
        if len(layers) > 1:
            for layer, values in layer_arrays.items():
                payload[f"muq_layer_{layer}"] = values
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **payload)
        partial.replace(output)
        records.append({
            "id": row["id"], "wav": str(wav), "wav_sha256": sha(wav),
            "path": str(output), "sha256": sha(output), "frames": len(selected),
            "native_hz": 25, "layers": list(layers), "duration_s": len(waveform) / 24000,
        })
        print(f"{index}/{len(rows)} {row['id']} frames={len(selected)} layers={layers} elapsed={time.time() - started:.1f}s", flush=True)
    receipt = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "audio-light-foundation-v2-muq-20261016-v2",
        "model_id": "OpenMuQ/MuQ-large-msd-iter",
        "revision": REVISION,
        "layer": 9,
        "layers": list(layers),
        "weights_sha256": WEIGHTS_SHA256,
        "sample_rate": 24000,
        "native_hz": 25,
        "saved_dtype": "float16",
        "chunk_seconds": 30,
        "core_seconds": 20,
        "context_seconds": 5,
        "records": records,
        "elapsed_s": time.time() - started,
        "source_contract": "/root/lightsem/docs/research/2026-10-16-audio-light-foundation-v2/CONTRACT.json",
        "source_contract_sha256": sha("/root/lightsem/docs/research/2026-10-16-audio-light-foundation-v2/CONTRACT.json"),
    }
    (args.out / "COMPLETE_V2.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

