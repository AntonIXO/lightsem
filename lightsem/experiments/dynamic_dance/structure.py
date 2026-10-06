"""Pinned official All-In-One functional music structure analysis.

The neural model and postprocessor come from mir-aidj/all-in-one.  No
functional labels are inferred by a substitute heuristic.  The public helpers
use an isolated interpreter so this legacy model's dependencies do not alter
the MuQ/Beat This environment.  Supplied HTDemucs stems avoid a second demix.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

CODE_REVISION = "18e78903c0365147a2c5d4e5e57ebf88cb7d800e"
WEIGHT_REVISION = "379e5fd010b3fdd0ee8381ff8cbcfa51d70b5c19"
SOURCE_DIRECTORY = Path("/mnt/lightsem-data/tools/dynamic-allinone")
WEIGHT_DIRECTORY = Path("/mnt/lightsem-data/models/dynamic-allinone")
DEFAULT_PYTHON = Path("/mnt/lightsem-data/venv-dynamic-structure/bin/python")
SOURCE_ORDER = ("drums", "bass", "other", "vocals")
OFFICIAL_SOURCE_ORDER = ("bass", "drums", "other", "vocals")
MIN_FREE_BYTES = 20 * 1024**3
WEIGHTS = {
    "harmonix-fold0-0vra4ys2.pth": "0db596dfb0995f41d62f6267d76a9d54c046f1649bd35e1dbeca0c5f9a7b8acd",
    "harmonix-fold1-3ozjhtsj.pth": "607b4864eaabccd4f6951410d5ea6676666ad1cfa6ce2e18bc99f3824b77c3b7",
    "harmonix-fold2-gmgo0nsy.pth": "67c28ec7ca612d365e5fcb152f6530eb53e34c4303940afb30fd17a295c958a4",
    "harmonix-fold3-i92b7m8p.pth": "446d05672b2669cdb7d9b8bbbc228a4f3592484335a7995cca89a744e3950ddc",
    "harmonix-fold4-1bql5qo0.pth": "96327222ab9b33c005388b788ba602f529aa03cc4dde1c859c011f1004490493",
    "harmonix-fold5-x4z5zeef.pth": "c87f162a1090290bcbb863688746784919cf912c56744fb3d81a1a7dba3b114e",
    "harmonix-fold6-x7t226rq.pth": "13e7e639bf30fcd3fa049ad863088773a4e57b06d7c68581385b3ef625c60688",
    "harmonix-fold7-qwwskhg6.pth": "b888b7fba95efa3d5a37ca013fe7efe2741621ea9ca335e6b2723e4fd0fade57",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _require_disk(path: Path, additional_bytes: int = 0) -> None:
    free = shutil.disk_usage(path).free
    if free - additional_bytes < MIN_FREE_BYTES:
        raise RuntimeError(
            f"Structure analysis would breach the 20 GiB disk floor: "
            f"{free} free, {additional_bytes} projected additional bytes"
        )


def _official_receipt() -> dict[str, Any]:
    """Fail closed if installed inference source differs from the frozen checkout."""
    import allin1

    observed = subprocess.check_output(
        ["git", "-C", str(SOURCE_DIRECTORY), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != CODE_REVISION:
        raise RuntimeError(f"All-In-One checkout changed: {observed}")
    subprocess.run(
        ["git", "-C", str(SOURCE_DIRECTORY), "diff", "--exit-code", "HEAD", "--", "src/allin1"],
        check=True, stdout=subprocess.DEVNULL,
    )
    installed = Path(allin1.__file__).parent
    hashes = {}
    for source in sorted((SOURCE_DIRECTORY / "src/allin1").rglob("*.py")):
        relative = source.relative_to(SOURCE_DIRECTORY / "src/allin1")
        target = installed / relative
        if not target.is_file() or _sha256(source) != _sha256(target):
            raise RuntimeError(f"Installed official All-In-One source differs: {relative}")
        hashes[str(relative)] = _sha256(target)
    return {
        "repository": "https://github.com/mir-aidj/all-in-one",
        "code_revision": CODE_REVISION,
        "installed_source_sha256": hashes,
        "version": importlib.metadata.version("allin1"),
    }


def _load_model(device: str):
    """Instantiate the official ensemble with content-verified frozen weights."""
    import torch
    from allin1.models.allinone import AllInOne
    from allin1.models.ensemble import Ensemble
    from omegaconf import OmegaConf

    receipt = _official_receipt()
    compatibility_path = DEFAULT_PYTHON.parent.parent / "natten_compatibility_receipt.json"
    compatibility = json.loads(compatibility_path.read_text()) if compatibility_path.is_file() else None
    if device.startswith("cuda") and compatibility and compatibility.get("cpu_only"):
        raise RuntimeError("This pinned NATTEN build is CPU-only; explicitly request device='cpu'")
    models, weight_receipts = [], []
    for filename, expected_hash in WEIGHTS.items():
        path = WEIGHT_DIRECTORY / filename
        if not path.is_file():
            raise FileNotFoundError(f"Pinned All-In-One weight missing: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Pinned All-In-One weight hash mismatch: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = AllInOne(OmegaConf.create(checkpoint["config"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()
        models.append(model)
        weight_receipts.append({"path": str(path), "sha256": actual_hash})
    receipt.update({
        "model": "harmonix-all",
        "weight_repository": "https://huggingface.co/taejunkim/allinone",
        "weight_revision": WEIGHT_REVISION,
        "weights": weight_receipts,
        "device": device,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("allin1", "torch", "natten", "madmom", "numpy", "omegaconf")
        },
    })
    if compatibility is not None:
        receipt["natten_build_compatibility"] = compatibility
        receipt["natten_build_compatibility_sha256"] = _sha256(compatibility_path)
    return Ensemble(models).eval(), receipt


def _extract_in_process(
    audio_path: Path,
    output_directory: Path,
    device: str,
    stems_path: Path | None,
    stem_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    import soundfile as sf
    import torch
    from allin1.demix import demix
    from allin1.helpers import run_inference, save_results
    from allin1.spectrogram import extract_spectrograms

    # Threads are intentionally bounded independently of the GPU curator.
    torch.set_num_threads(int(os.environ.get("LIGHTSEM_STRUCTURE_THREADS", "4")))
    model, model_receipt = _load_model(device)
    with tempfile.TemporaryDirectory(prefix="structure-work-", dir=output_directory) as temporary:
        work = Path(temporary)
        if stems_path is None:
            paths = demix([audio_path], work / "demix", device)
            separation_receipt = {
                "kind": "official_allin1_demucs_subprocess",
                "model": "htdemucs",
                "demucs_version": importlib.metadata.version("demucs"),
            }
        else:
            stems = np.load(stems_path, mmap_mode="r", allow_pickle=False)
            _validate_stems(stems)
            audio_info = sf.info(audio_path)
            if abs(stems.shape[-1] / 44100 - audio_info.duration) > 0.1:
                raise ValueError("Provided stems do not cover the complete source audio duration")
            _require_disk(output_directory, stems.nbytes + 256 * 1024**2)
            source_directory = work / audio_path.stem
            source_directory.mkdir()
            stem_hashes = {}
            for index, name in enumerate(SOURCE_ORDER):
                path = source_directory / f"{name}.wav"
                sf.write(path, stems[index].T, 44100, subtype="FLOAT")
                stem_hashes[name] = _sha256(path)
            paths = [source_directory]
            separation_receipt = {
                "kind": "supplied_demucs_stems",
                "input_order": list(SOURCE_ORDER),
                "official_spectrogram_order": list(OFFICIAL_SOURCE_ORDER),
                "sample_rate_hz": 44100,
                "samples": int(stems.shape[-1]),
                "transient_float_wav_sha256": stem_hashes,
                "source_receipt": stem_provenance,
            }
        specs = extract_spectrograms(paths, work / "spec", multiprocess=False)
        with torch.inference_mode():
            result = run_inference(
                path=audio_path,
                spec_path=specs[0],
                model=model,
                device=device,
                include_activations=True,
                include_embeddings=False,
            )
        if not result.segments:
            raise RuntimeError("Official All-In-One returned no functional sections")
        if not all(np.isfinite(value).all() for value in result.activations.values()):
            raise RuntimeError("Official All-In-One activations contain non-finite values")
        raw_directory = output_directory / "raw"
        save_results(result, raw_directory)
        raw_json = raw_directory / f"{audio_path.stem}.json"
        raw_activations = raw_json.with_suffix(".activ.npz")
        payload = {
            "status": "complete",
            "audio_path": str(audio_path),
            "audio_sha256": _sha256(audio_path),
            "sections": [
                {"start": float(s.start), "end": float(s.end), "label": s.label,
                 "confidence": None}
                for s in result.segments
            ],
            "uncertainty": "Official labels are model estimates; calibrated section confidence is unavailable.",
            "native_hop_seconds": float(model.cfg.hop_size / model.cfg.sample_rate),
            "beats": list(map(float, result.beats)),
            "downbeats": list(map(float, result.downbeats)),
            "bpm": None if result.bpm is None else float(result.bpm),
            "model": model_receipt,
            "separation": separation_receipt,
            "raw_artifacts": [
                {"path": str(p), "sha256": _sha256(p)}
                for p in (raw_json, raw_activations)
            ],
            "temporary_stems_retained": False,
        }
    _json(output_directory / "structure.json", payload)
    return payload


def _validate_stems(stems: np.ndarray) -> None:
    if stems.ndim != 3 or stems.shape[:2] != (4, 2) or stems.shape[-1] == 0:
        raise ValueError("Expected four stereo stems [4,2,N] at 44100 Hz: drums,bass,other,vocals")
    if not np.issubdtype(stems.dtype, np.floating):
        raise ValueError("Stems must be floating-point audio")
    for stem in stems:
        if not np.isfinite(stem).all():
            raise ValueError("Stem audio contains non-finite values")


def _run_isolated(
    audio_path: str | Path,
    output_directory: str | Path,
    device: str,
    stems: np.ndarray | None,
    stem_provenance: dict[str, Any] | None,
    python_executable: str | Path | None,
) -> dict[str, Any]:
    audio_path = Path(audio_path).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = output_directory / "structure.json"
    if output_file.exists():
        raise FileExistsError(f"Preserving existing structure evidence: {output_file}")
    projected = 256 * 1024**2 if stems is None else 2 * stems.nbytes + 256 * 1024**2
    _require_disk(output_directory, projected)
    python = Path(python_executable or DEFAULT_PYTHON)
    if not python.is_file():
        raise FileNotFoundError(f"All-In-One dependency interpreter unavailable: {python}")
    with tempfile.TemporaryDirectory(prefix="structure-input-", dir=output_directory) as temporary:
        temporary_path = Path(temporary)
        command = [str(python), str(Path(__file__).resolve()), "--audio", str(audio_path),
                   "--output", str(output_directory), "--device", device]
        if stems is not None:
            stems_file = temporary_path / "stems.npy"
            np.save(stems_file, np.asarray(stems, dtype=np.float32), allow_pickle=False)
            command.extend(["--stems", str(stems_file)])
        if stem_provenance is not None:
            provenance_file = temporary_path / "stem_provenance.json"
            _json(provenance_file, stem_provenance)
            command.extend(["--stem-provenance", str(provenance_file)])
        log_path = output_directory / "structure.log"
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        if device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        with log_path.open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if completed.returncode:
            if not (output_directory / "structure_failure.json").exists():
                _json(output_directory / "structure_failure.json", {
                    "status": "failed", "returncode": completed.returncode, "log": str(log_path),
                    "audio_path": str(audio_path), "device": device,
                })
            raise RuntimeError(f"Official All-In-One extraction failed; inspect {log_path}")
    return json.loads(output_file.read_text())


def extract_structure(
    audio_path: str | Path,
    output_directory: str | Path,
    device: str = "cuda",
    *,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze an audio file with official separation and the pinned ensemble.

    Prefer ``extract_from_stems`` when the campaign already computed pinned
    HTDemucs sources. Device failure is explicit; there is no automatic CPU or
    heuristic fallback. Run CUDA calls only under the campaign's GPU lock.
    """
    return _run_isolated(audio_path, output_directory, device, None, None, python_executable)


def extract_from_stems(
    audio_path: str | Path,
    stems: np.ndarray,
    output_directory: str | Path,
    device: str = "cuda",
    *,
    stem_provenance: dict[str, Any] | None = None,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze supplied [drums,bass,other,vocals] stereo 44100 Hz sources.

    Stem separation is not rerun. Temporary float audio and spectrograms are
    deleted after success or failure; their source receipts remain. Arrays must
    cover the complete audio on its original timeline.
    """
    stems = np.asarray(stems)
    _validate_stems(stems)
    return _run_isolated(audio_path, output_directory, device, stems, stem_provenance, python_executable)


def smoke_receipt() -> dict[str, Any]:
    """CPU-only dependency and weight-loading check; performs no inference."""
    model, receipt = _load_model("cpu")
    receipt["parameters_per_fold"] = [sum(p.numel() for p in m.parameters()) for m in model.models]
    receipt["inference_performed"] = False
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stems", type=Path)
    parser.add_argument("--stem-provenance", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        if args.smoke:
            _json(args.output / "structure_setup_smoke.json", smoke_receipt())
        else:
            if args.audio is None:
                parser.error("--audio is required for extraction")
            provenance = json.loads(args.stem_provenance.read_text()) if args.stem_provenance else None
            _extract_in_process(args.audio, args.output, args.device, args.stems, provenance)
    except Exception as error:
        _json(args.output / "structure_failure.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "audio_path": str(args.audio), "device": args.device,
            "no_fallback_labels": True,
        })
        raise


if __name__ == "__main__":
    main()
