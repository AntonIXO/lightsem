"""MuQ contract and trainable-scope helpers.

This module deliberately keeps model loading optional.  Cached feature
experiments can validate provenance and train downstream adapters without
claiming that MuQ itself was fine-tuned.  Actual backbone fine-tuning requires
passing a live MuQ module to :func:`configure_muq_scope`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MuQContract:
    model_id: str = "OpenMuQ/MuQ-large-msd-iter"
    revision: str = "0562a57814f6f8bbd9fdea0a25921a2fce1a841a"
    selected_layer: int = 9
    sample_rate_hz: int = 24000
    native_hz: int = 25
    dimension: int = 1024
    base92_dimension: int = 92


def validate_muq_features(features: Any, *, contract: MuQContract = MuQContract(),
                          name: str = "muq") -> np.ndarray:
    arr = np.asarray(features)
    if arr.ndim != 2 or arr.shape[1] != contract.dimension:
        raise ValueError(f"{name} must have shape (T,{contract.dimension})")
    if arr.shape[0] < 1 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite frames")
    return arr.astype(np.float32, copy=False)


def validate_base92(features: Any, *, contract: MuQContract = MuQContract()) -> np.ndarray:
    arr = np.asarray(features)
    if arr.ndim != 2 or arr.shape[1] != contract.base92_dimension:
        raise ValueError(f"Base92 must have shape (T,{contract.base92_dimension})")
    if arr.shape[0] < 1 or not np.isfinite(arr).all():
        raise ValueError("Base92 must contain finite frames")
    return arr.astype(np.float32, copy=False)


def load_layer_bundle(path: str) -> dict[str, np.ndarray]:
    """Load a canonical NPZ with ``muq`` or ``muq_layer_<n>`` arrays."""
    with np.load(path, allow_pickle=False) as data:
        out = {}
        for key in data.files:
            if key == "muq" or key.startswith("muq_layer_"):
                out[key] = np.asarray(data[key])
        if not out:
            raise ValueError(f"{path} contains no MuQ layer arrays")
        return out


def load_live_muq(checkpoint: str, *, device: str = "cpu", local_files_only: bool = True):
    """Load a live MuQ backbone for an explicitly opted-in fine-tune arm.

    The dependency is imported lazily so CPU layer-selection tools do not need
    Torch or the MuQ package.  Callers must record checkpoint/revision/weight
    hashes separately in their experiment receipt.
    """
    import torch
    from muq import MuQ

    model = MuQ.from_pretrained(checkpoint, local_files_only=local_files_only)
    return model.to(torch.device(device)).float()


def configure_muq_scope(module: Any, scope: str, *, last_blocks: int = 2) -> dict[str, Any]:
    """Freeze all MuQ parameters, or unfreeze an explicit final block scope."""
    from torch import nn
    if scope not in {"frozen", "last_blocks", "full"}:
        raise ValueError("scope must be frozen, last_blocks or full")
    for parameter in module.parameters():
        parameter.requires_grad_(scope == "full")
    selected: list[str] = []
    if scope == "last_blocks":
        blocks = None
        for candidate in ("layers", "blocks", "encoder_layers"):
            value = getattr(module, candidate, None)
            if isinstance(value, (nn.ModuleList, nn.Sequential)):
                blocks = value
                break
        if blocks is None:
            raise ValueError("could not find MuQ block list for last_blocks scope")
        if last_blocks < 1 or last_blocks > len(blocks):
            raise ValueError("last_blocks is outside the model block range")
        for index in range(len(blocks) - last_blocks, len(blocks)):
            for name, parameter in blocks[index].named_parameters(prefix=f"blocks.{index}"):
                parameter.requires_grad_(True)
                selected.append(name)
    report = {
        "scope": scope,
        "selected_parameter_names": selected,
        "total_parameters": int(sum(p.numel() for p in module.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in module.parameters() if p.requires_grad)),
    }
    report["trainable_fraction"] = report["trainable_parameters"] / max(report["total_parameters"], 1)
    return report

