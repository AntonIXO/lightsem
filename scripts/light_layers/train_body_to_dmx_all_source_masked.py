"""Committed all-source entrypoint and task-mask contract for old BodyToDMX.

The implementation is deliberately a thin wrapper around the validated
multisource trainer.  The source masks are the contract: BeatSaver supervises
scene/event/weak-angle, RPMC supervises only its six-dimensional scene target,
and Roskilde supervises only its abstract [T,60] head.  Neither open source is
allowed to supervise body, physical angle, or native DMX motion.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.light_layers.train_body_to_dmx_multisource_roskilde import main


SOURCE_TASK_MASKS = {
    "beatsaver": {"scene": True, "event": True, "angle": True, "motion": True, "roskilde60": False},
    "rpmc": {"scene": True, "event": False, "angle": False, "motion": False, "roskilde60": False},
    "roskilde": {"scene": False, "event": False, "angle": False, "motion": False, "roskilde60": True},
}


def validate_source_masks(masks: dict) -> None:
    required = set(SOURCE_TASK_MASKS)
    if set(masks) != required:
        raise ValueError(f"source mask keys must be {sorted(required)}")
    for source, expected in SOURCE_TASK_MASKS.items():
        if masks[source] != expected:
            raise ValueError(f"invalid task mask for {source}: {masks[source]!r}")
    for source in ("rpmc", "roskilde"):
        if masks[source]["motion"] or masks[source]["angle"] or masks[source]["event"]:
            raise ValueError(f"open source {source} cannot supervise physical motion/angle/event")


def masked_legacy_loss(pred: dict, targets: dict, masks: dict) -> dict[str, torch.Tensor]:
    """Compute old losses with either source masks or temporal target masks."""
    # Unit/integration callers may pass per-frame masks; source training passes
    # the explicit SOURCE_TASK_MASKS contract above.
    if isinstance(masks.get("scene"), torch.Tensor):
        scene_mask = masks["scene"].bool()
        event_mask = masks.get("event", torch.zeros_like(scene_mask)).bool()
        angle_mask = masks.get("angle", torch.zeros_like(scene_mask)).bool()
        scene_values = pred["scene"][scene_mask.unsqueeze(-1).expand_as(pred["scene"])]
        scene_targets = targets["scene"][scene_mask.unsqueeze(-1).expand_as(targets["scene"])]
        scene_loss = F.smooth_l1_loss(scene_values, scene_targets) if scene_values.numel() else pred["scene"].sum() * 0.0
        event_values = pred["event_logit"].squeeze(-1)[event_mask]
        event_targets = targets["event"].float()[event_mask]
        event_loss = F.binary_cross_entropy_with_logits(event_values, event_targets) if event_values.numel() else pred["event_logit"].sum() * 0.0
        angle_values = pred["angle_logit"][angle_mask]
        angle_targets = targets["angle"].float()[angle_mask]
        angle_loss = F.mse_loss(torch.sigmoid(angle_values), angle_targets) if angle_values.numel() else pred["angle_logit"].sum() * 0.0
        return {"scene": scene_loss, "event": event_loss, "angle": angle_loss, "total": scene_loss + 0.25 * event_loss + 0.5 * angle_loss}
    validate_source_masks(masks)
    scene = pred["scene"]
    zero = scene.sum() * 0.0
    scene_loss = F.smooth_l1_loss(scene, targets["scene"]) if masks["scene"] else zero
    event_loss = F.binary_cross_entropy_with_logits(pred["event_logit"].squeeze(-1), targets["event"].float()) if masks["event"] else zero
    angle_loss = F.mse_loss(torch.sigmoid(pred["angle_logit"]), targets["angle"].float()) if masks["angle"] else zero
    roskilde_loss = F.smooth_l1_loss(pred["scene60"], targets["roskilde60"]) if masks["roskilde60"] else zero
    return {"scene": scene_loss, "event": event_loss, "angle": angle_loss, "roskilde60": roskilde_loss, "total": scene_loss + 0.25 * event_loss + 0.5 * angle_loss + roskilde_loss}


if __name__ == "__main__":
    main()
