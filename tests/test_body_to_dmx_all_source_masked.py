import torch

from scripts.light_layers.train_body_to_dmx_all_source_masked import (
    SOURCE_TASK_MASKS,
    masked_legacy_loss,
    validate_source_masks,
)


def test_open_sources_cannot_supervise_physical_motion_or_angle():
    validate_source_masks(SOURCE_TASK_MASKS)
    assert SOURCE_TASK_MASKS["rpmc"]["scene"]
    assert not SOURCE_TASK_MASKS["rpmc"]["motion"]
    assert not SOURCE_TASK_MASKS["rpmc"]["angle"]
    assert SOURCE_TASK_MASKS["roskilde"]["roskilde60"]
    assert not SOURCE_TASK_MASKS["roskilde"]["motion"]
    assert not SOURCE_TASK_MASKS["roskilde"]["angle"]


def test_masked_legacy_loss_drops_unobserved_event_and_angle():
    pred = {
        "scene": torch.zeros(1, 3, 6, requires_grad=True),
        "event_logit": torch.zeros(1, 3, 1, requires_grad=True),
        "angle_logit": torch.zeros(1, 3, 12, requires_grad=True),
    }
    targets = {
        "scene": torch.ones(1, 3, 6),
        "event": torch.ones(1, 3),
        "angle": torch.ones(1, 3, 12),
    }
    masks = {"scene": torch.ones(1, 3, dtype=torch.bool),
             "event": torch.zeros(1, 3, dtype=torch.bool),
             "angle": torch.zeros(1, 3, dtype=torch.bool)}
    losses = masked_legacy_loss(pred, targets, masks)
    assert losses["event"].item() == 0.0
    assert losses["angle"].item() == 0.0
    losses["total"].backward()
    assert pred["event_logit"].grad is not None
    assert torch.equal(pred["event_logit"].grad, torch.zeros_like(pred["event_logit"].grad))
    assert torch.equal(pred["angle_logit"].grad, torch.zeros_like(pred["angle_logit"].grad))
