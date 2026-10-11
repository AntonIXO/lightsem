"""Strict, streamed oBSRNN-SIMO source-teacher extraction.

The teacher is an offline research component.  It keeps the original mastered
left/right mix and emits source-specific stereo-image descriptors; it never
materializes a full-song stem cache.  The checkpoint and source checkout are
content-pinned because the upstream loader otherwise silently creates a random
model when a checkpoint is missing and uses ``strict=False``.

The model's checkpoint order is ``vocals, bass, drums, other``.  The LightSem
source contract is ``drums, bass, vocals, other`` and all public extraction
outputs use that order explicitly.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import Callable, Iterator, Mapping
import zipfile

import numpy as np


SOURCE_COMMIT = "94adc8c184ac9b1d2e496fcb6e78cfe3fb78fc9c"
EXPECTED_ZIP_MD5 = "817e8ce4bcd17c38bd3812167e97db58"
EXPECTED_CHECKPOINT_SHA256 = "fd23279e43ade16b9c30b466c978d73f879c2cfb8d79917e6d880d8ed8df51e8"
DEFAULT_SOURCE_ROOT = Path("/mnt/lightsem-data/tools/bsrnn")
DEFAULT_CHECKPOINT = Path(
    "/mnt/lightsem-data/checkpoints/three_spatial_methods/"
    "simo-bsrnn-opt/simo-bsrnn-opt/separator.ckpt"
)
CHECKPOINT_TARGET_ORDER = ("vocals", "bass", "drums", "other")
SOURCE_ORDER = ("drums", "bass", "vocals", "other")
SAMPLE_RATE = 44_100
DESCRIPTOR_DIM = 16

# These names are the stable fallback descriptor contract.  A caller may pass
# the shared project descriptor function to ``extract_streamed``; it must return
# the same width and timestamps (see DescriptorFn below).
DESCRIPTOR_NAMES = (
    "log_energy", "flux", "ild_db", "width_fraction", "pan_balance",
    "coherence", "phase_sin", "phase_cos",
    "flux_band_low", "flux_band_mid", "flux_band_high", "flux_band_hf",
    "energy_delta_band_low", "energy_delta_band_mid",
    "energy_delta_band_high", "energy_delta_band_hf",
)
DescriptorFn = Callable[[np.ndarray, int, float], tuple[np.ndarray, np.ndarray]]


class BSRNNTeacherError(RuntimeError):
    """Raised when a source, checkpoint, or output violates the teacher contract."""


def _digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    return _digest(path, "sha256")


def _git_commit(source_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={source_root}", "-C", str(source_root),
             "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BSRNNTeacherError(f"Cannot verify pinned BSRNN checkout: {source_root}") from exc
    return result.stdout.strip()


def _resolve_checkpoint(checkpoint: str | Path) -> tuple[Path, Path | None]:
    """Resolve an extracted checkpoint and optionally its containing archive.

    We intentionally do not extract an unverified archive in this loader.  A
    caller may extract the pinned ZIP once, after which this function binds both
    the archive MD5 and the extracted checkpoint SHA-256 when the ZIP is passed.
    """
    path = Path(checkpoint)
    archive: Path | None = None
    if path.is_dir():
        candidates = sorted(path.rglob("separator.ckpt"))
        if len(candidates) != 1:
            raise BSRNNTeacherError(f"Expected one separator.ckpt below {path}, found {len(candidates)}")
        path = candidates[0]
    elif path.suffix.lower() == ".zip":
        archive = path
        if not path.exists():
            raise FileNotFoundError(path)
        if _digest(path, "md5") != EXPECTED_ZIP_MD5:
            raise BSRNNTeacherError(f"Checkpoint ZIP MD5 mismatch: {path}")
        with zipfile.ZipFile(path) as archive_file:
            members = [m for m in archive_file.namelist() if m.endswith("/separator.ckpt")]
        if len(members) != 1:
            raise BSRNNTeacherError(f"Expected one separator.ckpt in {path}, found {len(members)}")
        member = Path(members[0])
        # The canonical extraction path used by the pinned artifact is the ZIP
        # basename with the archive's top-level directory retained.
        extracted = path.with_suffix("") / member
        if not extracted.exists():
            raise BSRNNTeacherError(
                f"Pinned ZIP is present but not extracted at {extracted}; "
                "extract it explicitly before loading"
            )
        path = extracted
    if path.name != "separator.ckpt" or not path.is_file():
        raise BSRNNTeacherError(f"Expected extracted separator.ckpt, got {path}")
    return path, archive


def _namespace(value: Mapping[str, object]) -> SimpleNamespace:
    """Convert checkpoint dictionaries into the attribute config expected upstream."""
    return SimpleNamespace(**{
        str(k): _namespace(v) if isinstance(v, Mapping) else v
        for k, v in value.items()
    })


@contextmanager
def _bsrnn_import_path(source_root: Path) -> Iterator[None]:
    root = str(source_root.resolve())
    old = list(sys.path)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        yield
    finally:
        sys.path[:] = old


def _build_model(hparams: Mapping[str, object], source_root: Path):
    """Instantiate the exact upstream class from checkpoint hyperparameters."""
    import torch

    required = ("cfg_optim", "cfg_scheduler", "cfg_eval", "targets", "sample_rate",
                "n_fft", "n_hop", "nb_channels", "feature_dim", "joint_bandsplit",
                "num_repeat", "fac_sep", "time_layer", "band_layer", "bidirectional",
                "fac_mask", "mask_ctxt", "subtract_last_trgt", "n_heads", "group_num",
                "n_att_head", "attn_enc_dim", "stereo", "fac_tac", "act_tac", "eps")
    missing = [name for name in required if name not in hparams]
    if missing:
        raise BSRNNTeacherError(f"Checkpoint hyperparameters missing {missing}")
    with _bsrnn_import_path(source_root):
        # The model class imports training/evaluation helpers at module import
        # time.  Those helpers pull in the optional MUSDB/museval toolchains,
        # which are neither needed nor allowed for offline inference.  Install
        # narrow import stubs before loading the original class; no model code
        # or state is replaced by these stubs.
        if "helpers.data" not in sys.modules:
            data_stub = types.ModuleType("helpers.data")
            data_stub.rec_audio = lambda *args, **kwargs: None
            sys.modules["helpers.data"] = data_stub
        if "helpers.eval" not in sys.modules:
            eval_stub = types.ModuleType("helpers.eval")
            eval_stub.compute_sdr = lambda *args, **kwargs: (None, None)
            sys.modules["helpers.eval"] = eval_stub
        module = importlib.import_module("models.bsrnn")
        model_cls = module.BSRNN
        cfg_optim = _namespace(hparams["cfg_optim"])
        cfg_scheduler = _namespace(hparams["cfg_scheduler"])
        eval_values = dict(hparams["cfg_eval"])
        # The checkpoint predates the current class's sdr_win/sdr_hop names.
        # They are irrelevant for inference but required during construction.
        eval_values.setdefault("sdr_win", eval_values.get("win_dur", 1.0))
        eval_values.setdefault("sdr_hop", eval_values.get("hop_dur", 1.0))
        cfg_eval = _namespace(eval_values)
        model_kwargs = {name: hparams[name] for name in required
                        if name not in ("cfg_optim", "cfg_scheduler", "cfg_eval", "targets")}
        model = model_cls(cfg_optim, cfg_scheduler, cfg_eval,
                          targets=list(hparams["targets"]), **model_kwargs)
    if not isinstance(model, torch.nn.Module):
        raise BSRNNTeacherError("Pinned BSRNN class did not produce a torch module")
    return model


@dataclass
class BSRNNTeacher:
    """Loaded, hash-bound oBSRNN-SIMO model and its provenance."""

    model: object
    device: object
    checkpoint: Path
    checkpoint_sha256: str
    source_root: Path
    source_commit: str
    checkpoint_target_order: tuple[str, ...] = CHECKPOINT_TARGET_ORDER
    source_order: tuple[str, ...] = SOURCE_ORDER
    sample_rate: int = SAMPLE_RATE

    def infer(self, mix: "object", *, peak: float | None = None) -> "object":
        """Infer four stereo stems from ``mix`` shaped ``[2,N]`` or ``[1,2,N]``.

        The upstream inference normalizes the entire mixture by one absolute peak
        shared by both channels.  ``peak`` lets streamed extraction use one global
        peak for every chunk; per-channel or per-chunk normalization is rejected.
        """
        import torch

        x = torch.as_tensor(mix, dtype=torch.float32)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        if x.ndim != 3 or x.shape[0] != 1 or x.shape[1] != 2 or x.shape[2] < 8:
            raise ValueError("Expected mix [2,N] or [1,2,N]")
        if not torch.isfinite(x).all():
            raise ValueError("Mixture contains non-finite values")
        global_peak = float(x.abs().max().item()) if peak is None else float(peak)
        if not np.isfinite(global_peak) or global_peak <= 0.0:
            raise ValueError("Mixture peak must be finite and positive")
        x = (x / global_peak).to(self.device)
        with torch.inference_mode():
            output = self.model(x)["waveforms"]
        if output.ndim != 4 or tuple(output.shape[:3]) != (1, 4, 2):
            raise BSRNNTeacherError(f"Unexpected BSRNN output shape {tuple(output.shape)}")
        output = output.detach().to("cpu") * global_peak
        # Checkpoint target order -> common LightSem source order.
        indices = [self.checkpoint_target_order.index(name) for name in self.source_order]
        output = output[:, indices]
        if not torch.isfinite(output).all():
            raise BSRNNTeacherError("BSRNN emitted non-finite source waveform")
        return output[0]


def load_teacher(checkpoint: str | Path = DEFAULT_CHECKPOINT, device: str | object = "cpu",
                 *, source_root: str | Path = DEFAULT_SOURCE_ROOT) -> BSRNNTeacher:
    """Load the pinned teacher with strict hyperparameter and state-key checks."""
    import torch

    source_root = Path(source_root)
    if _git_commit(source_root) != SOURCE_COMMIT:
        raise BSRNNTeacherError(f"BSRNN source commit mismatch at {source_root}")
    ckpt_path, archive = _resolve_checkpoint(checkpoint)
    digest = _sha256(ckpt_path)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise BSRNNTeacherError(f"Checkpoint SHA-256 mismatch: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or "state_dict" not in raw or "hyper_parameters" not in raw:
        raise BSRNNTeacherError("Checkpoint lacks Lightning state_dict/hyper_parameters")
    hparams = raw["hyper_parameters"]
    if tuple(hparams.get("targets", ())) != CHECKPOINT_TARGET_ORDER:
        raise BSRNNTeacherError(f"Unexpected checkpoint target order: {hparams.get('targets')}")
    checks = {"sample_rate": SAMPLE_RATE, "n_fft": 2048, "n_hop": 512,
              "nb_channels": 2, "stereo": "tac", "feature_dim": 128,
              "num_repeat": 12, "joint_bandsplit": True}
    for key, expected in checks.items():
        if hparams.get(key) != expected:
            raise BSRNNTeacherError(f"Checkpoint hyperparameter {key}={hparams.get(key)!r}, expected {expected!r}")
    model = _build_model(hparams, source_root)
    model_keys = set(model.state_dict())
    checkpoint_state = raw["state_dict"]
    checkpoint_keys = set(checkpoint_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing or unexpected:
        raise BSRNNTeacherError(f"Strict state-key mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}")
    for key, value in checkpoint_state.items():
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(model.state_dict()[key].shape):
            raise BSRNNTeacherError(f"State tensor shape mismatch for {key}")
    result = model.load_state_dict(checkpoint_state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise BSRNNTeacherError(f"Strict state load reported {result}")
    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise BSRNNTeacherError("CUDA requested but unavailable; refusing CPU fallback")
    model.to(target_device).eval()
    return BSRNNTeacher(model=model, device=target_device, checkpoint=ckpt_path,
                        checkpoint_sha256=digest, source_root=source_root,
                        source_commit=SOURCE_COMMIT)


def _default_descriptors(waveforms: np.ndarray, sample_rate: int,
                         fps: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Build the 16D source stereo-image fallback from ``extract_stereo_v1``.

    The shared pilot descriptor function can replace this through
    ``extract_streamed(..., descriptor_fn=...)``.  Values are source-local
    stereo-image observations, never physical azimuth/elevation labels.
    """
    from lightsem.audio.stereo_v1 import extract_stereo_v1, FEATURE_NAMES

    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 3 or waveforms.shape[1] != 2:
        raise ValueError("Expected source waveforms [S,2,N]")
    results = []
    time = None
    for source in waveforms:
        item = extract_stereo_v1(source, sample_rate, fps=fps,
                                 spatial_fps=fps, relation_fps=max(1.0, fps / 20.0))
        if time is None:
            time = item.time_40.astype(np.float64)
        elif len(time) != len(item.time_40) or not np.allclose(time, item.time_40):
            raise BSRNNTeacherError("Source descriptor timestamps disagree")
        names = {name: i for i, name in enumerate(FEATURE_NAMES)}
        f = item.features_40
        energy = .5 * (f[:, names["left_log_rms"]] + f[:, names["right_log_rms"]])
        left_flux = f[:, [names[f"flux_left_band_{i}"] for i in range(5)]]
        right_flux = f[:, [names[f"flux_right_band_{i}"] for i in range(5)]]
        flux_bands = .5 * (left_flux + right_flux)
        delta = f[:, [names[f"energy_delta_band_{i}"] for i in range(5)]]
        # Preserve low/mid/high/high-frequency contrast while keeping exactly
        # 16 dimensions for the four-source 64D context seam.
        flux4 = np.stack((flux_bands[:, 0], flux_bands[:, 1], flux_bands[:, 2],
                          flux_bands[:, 3:].mean(1)), axis=1)
        delta4 = np.stack((delta[:, 0], delta[:, 1], delta[:, 2],
                           delta[:, 3:].mean(1)), axis=1)
        results.append(np.column_stack((
            energy, flux_bands.mean(1),
            f[:, names["ild_db"]], f[:, names["width_fraction"]],
            f[:, names["pan_balance"]], f[:, names["coherence"]],
            f[:, names["phase_sin"]], f[:, names["phase_cos"]], flux4, delta4,
        )).astype(np.float32))
    out = np.stack(results, axis=0)
    if out.shape[-1] != DESCRIPTOR_DIM or not np.isfinite(out).all():
        raise BSRNNTeacherError("Fallback descriptors are non-finite or wrong width")
    return np.asarray(time), out


def context_64(descriptors: np.ndarray) -> np.ndarray:
    """Flatten ``[S,T,16]`` common-order features to ``[T,64]``."""
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(SOURCE_ORDER) or values.shape[2] != DESCRIPTOR_DIM:
        raise ValueError(f"Expected [{len(SOURCE_ORDER)},T,{DESCRIPTOR_DIM}] descriptors")
    if not np.isfinite(values).all():
        raise ValueError("Descriptors contain non-finite values")
    return values.transpose(1, 0, 2).reshape(values.shape[1], len(SOURCE_ORDER) * DESCRIPTOR_DIM)


@dataclass
class TeacherFeatures:
    time_40: np.ndarray
    descriptors: np.ndarray
    metadata: dict

    @property
    def context_64(self) -> np.ndarray:
        return context_64(self.descriptors)


def _audio_info(path: Path):
    try:
        import soundfile as sf
    except ImportError as exc:
        raise BSRNNTeacherError("soundfile is required for streamed extraction") from exc
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE or info.channels != 2:
        raise BSRNNTeacherError(
            f"Mastered-stereo contract requires 44.1kHz 2ch input; got {info.samplerate}Hz/{info.channels}ch"
        )
    if info.frames <= 0:
        raise BSRNNTeacherError("Audio file contains no frames")
    return sf, info


def _global_peak(path: Path, sf, blocksize: int = 1 << 20) -> float:
    peak = 0.0
    with sf.SoundFile(str(path), "r") as handle:
        for block in handle.blocks(blocksize=blocksize, dtype="float32", always_2d=True):
            if block.size:
                peak = max(peak, float(np.max(np.abs(block))))
    if not np.isfinite(peak) or peak <= 0.0:
        raise BSRNNTeacherError("Input has no finite nonzero global peak")
    return peak


def _read_window(path: Path, sf, start: int, length: int, total: int) -> np.ndarray:
    read_start = max(0, start)
    lead = read_start - start
    readable = max(0, min(length - lead, total - read_start))
    with sf.SoundFile(str(path), "r") as handle:
        handle.seek(read_start)
        block = handle.read(readable, dtype="float32", always_2d=True)
    value = np.zeros((length, 2), dtype=np.float32)
    if len(block):
        value[lead:lead + len(block)] = block
    return value.T


def extract_streamed(audio_path: str | Path, teacher: BSRNNTeacher, *,
                     split: str = "train", chunk_seconds: float = 10.0,
                     hop_seconds: float = 9.0, descriptor_fn: DescriptorFn | None = None,
                     fps: float = 40.0) -> TeacherFeatures:
    """Extract source stereo descriptors with bounded model/waveform memory.

    ``split='test'`` (and aliases) is rejected to keep the teacher prep TRAIN
    only; ``showcase`` is allowed for the explicitly requested demos.  The model
    sees 10-second windows with a centered 0.5-second halo by default, while the
    returned arrays contain only the core 9-second regions.
    """
    if str(split).lower() in {"test", "val", "validation", "heldout", "held-out"}:
        raise BSRNNTeacherError(f"Teacher extraction is TRAIN/showcase-only, got split={split!r}")
    if chunk_seconds <= hop_seconds or fps <= 0:
        raise ValueError("Require chunk_seconds > hop_seconds and positive fps")
    sf, info = _audio_info(Path(audio_path))
    chunk = int(round(chunk_seconds * SAMPLE_RATE))
    hop = int(round(hop_seconds * SAMPLE_RATE))
    if chunk < 8 or hop < 1 or (chunk - hop) % 2:
        raise ValueError("Chunk/ hop must yield an integral centered halo")
    halo = (chunk - hop) // 2
    peak = _global_peak(Path(audio_path), sf)
    descriptor_fn = descriptor_fn or _default_descriptors
    times_out: list[np.ndarray] = []
    descriptors_out: list[np.ndarray] = []
    for core_start in range(0, info.frames, hop):
        core_len = min(hop, info.frames - core_start)
        window_start = core_start - halo
        waveform = _read_window(Path(audio_path), sf, window_start, chunk, info.frames)
        estimates = teacher.infer(waveform, peak=peak).numpy()
        local_time, values = descriptor_fn(estimates, SAMPLE_RATE, fps)
        local_time = np.asarray(local_time, dtype=np.float64)
        values = np.asarray(values, dtype=np.float32)
        if values.shape[0] != len(SOURCE_ORDER) or values.ndim != 3 or values.shape[2] != DESCRIPTOR_DIM:
            raise BSRNNTeacherError(f"Descriptor function returned {values.shape}, expected [4,T,16]")
        if len(local_time) != values.shape[1] or np.any(np.diff(local_time) <= 0):
            raise BSRNNTeacherError("Descriptor timestamps are invalid")
        absolute = window_start / SAMPLE_RATE + local_time
        lo, hi = core_start / SAMPLE_RATE, (core_start + core_len) / SAMPLE_RATE
        keep = (absolute >= lo - 1e-7) & (absolute < hi - 1e-7)
        if keep.any():
            times_out.append(absolute[keep])
            descriptors_out.append(values[:, keep])
    if not times_out:
        raise BSRNNTeacherError("No descriptor frames were emitted")
    times = np.concatenate(times_out)
    values = np.concatenate(descriptors_out, axis=1)
    # Boundary timestamps can coincide due to frame centering.  Keep the first
    # source estimate and require strict monotonicity for downstream resampling.
    keep = np.concatenate(([True], np.diff(times) > 1e-8))
    times, values = times[keep], values[:, keep]
    if not np.isfinite(times).all() or not np.isfinite(values).all():
        raise BSRNNTeacherError("Streamed teacher output contains non-finite values")
    metadata = {
        "schema_version": 1,
        "teacher": "oBSRNN-SIMO",
        "source_root_commit": teacher.source_commit,
        "checkpoint_sha256": teacher.checkpoint_sha256,
        "checkpoint_target_order": list(CHECKPOINT_TARGET_ORDER),
        "source_order": list(SOURCE_ORDER),
        "sample_rate_hz": SAMPLE_RATE,
        "input_path": str(Path(audio_path)),
        "input_frames": int(info.frames),
        "chunk_seconds": float(chunk_seconds),
        "hop_seconds": float(hop_seconds),
        "halo_seconds": float(halo / SAMPLE_RATE),
        "fps": float(fps),
        "global_peak": float(peak),
        "split": str(split),
        "normalization": "one global abs peak shared by both mastered L/R channels and all chunks",
        "stereo_semantics": "source stereo image; not physical 3D location",
        "descriptor_names": list(DESCRIPTOR_NAMES),
    }
    return TeacherFeatures(times.astype(np.float64), values.astype(np.float32), metadata)


def spatial_retention_smoke(waveforms: np.ndarray, sample_rate: int = SAMPLE_RATE,
                            descriptor_fn: DescriptorFn | None = None) -> dict:
    """Check finite, source-specific stereo-image retention under L/R swap."""
    descriptor_fn = descriptor_fn or _default_descriptors
    values = np.asarray(waveforms, dtype=np.float32)
    if values.ndim != 3 or values.shape[0] != len(SOURCE_ORDER) or values.shape[1] != 2:
        raise ValueError("Expected source waveforms [4,2,N]")
    time, original = descriptor_fn(values, sample_rate, 40.0)
    swapped_time, swapped = descriptor_fn(values[:, ::-1], sample_rate, 40.0)
    if len(time) != len(swapped_time) or not np.allclose(time, swapped_time):
        raise BSRNNTeacherError("L/R smoke timestamps disagree")
    if not np.isfinite(original).all() or not np.isfinite(swapped).all():
        return {"finite": False, "pass": False, "reason": "nonfinite"}
    odd = (2, 4, 6)  # ILD, pan, phase_sin
    even = (3, 5, 7)  # width, coherence, phase_cos
    odd_error = float(np.max(np.abs(original[:, :, odd] + swapped[:, :, odd])))
    even_error = float(np.max(np.abs(original[:, :, even] - swapped[:, :, even])))
    ranges = np.ptp(original[:, :, (2, 3, 4, 5, 6, 7)], axis=1)
    source_distinct = bool(np.any(ranges > 1e-7, axis=1).all())
    return {
        "finite": True,
        "pass": bool(odd_error < 1e-4 and even_error < 1e-4 and source_distinct),
        "lr_swap_odd_max_abs_error": odd_error,
        "lr_swap_even_max_abs_error": even_error,
        "source_spatial_ranges": ranges.astype(float).tolist(),
        "source_spatially_nonconstant": bool(source_distinct),
        "spatial_semantics": "stereo image descriptors; not physical 3D location",
    }


__all__ = [
    "BSRNNTeacher", "BSRNNTeacherError", "DESCRIPTOR_DIM", "DESCRIPTOR_NAMES",
    "DescriptorFn", "SOURCE_ORDER", "CHECKPOINT_TARGET_ORDER", "TeacherFeatures",
    "context_64", "load_teacher", "extract_streamed", "spatial_retention_smoke",
]
