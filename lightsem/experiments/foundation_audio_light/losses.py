"""Masked, separately logged pilot losses."""

import torch
import torch.nn.functional as F


def _masked(value, mask):
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def foundation_losses(pred, target, valid):
    """Return named losses; no missing target is silently treated as zero."""
    out = {}
    if "scene" in pred and "scene" in target:
        out["scene"] = _masked((pred["scene"] - target["scene"]).square(), valid[..., None])
    if "event" in pred and "event" in target:
        out["event"] = _masked(F.binary_cross_entropy_with_logits(pred["event"], target["event"], reduction="none"), valid[..., None])
    if "rgb" in pred and "rgb" in target:
        out["color_rgb"] = _masked((pred["rgb"] - target["rgb"]).abs(), valid[..., None, None])
    if "brightness" in pred and "brightness" in target:
        out["brightness"] = _masked((pred["brightness"] - target["brightness"]).abs(), valid[..., None, None])
    if not out:
        raise ValueError("no overlapping masked targets")
    out["total"] = sum(out.values()) / len(out)
    return out
