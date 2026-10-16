"""Named losses and anti-collapse diagnostics for the controlled SSL loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ControlledSSLLossConfig:
    masked_prediction: float = 1.0
    future_prediction: float = 1.0
    variance: float = 0.25
    covariance: float = 0.04
    variance_floor: float = 0.5


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded[..., None]
    return torch.where(expanded, value, torch.zeros_like(value)).sum() / expanded.sum().clamp_min(1)


def _vicreg_terms(value: torch.Tensor, valid: torch.Tensor, floor: float) -> tuple[torch.Tensor, torch.Tensor]:
    x = value[valid]
    if x.shape[0] < 2:
        zero = value.new_zeros(())
        return zero, zero
    centered = x - x.mean(0, keepdim=True)
    std = torch.sqrt(centered.var(0, unbiased=False) + 1e-04)
    variance = torch.relu(floor - std).mean()
    cov = centered.T @ centered / max(x.shape[0] - 1, 1)
    offdiag = cov - torch.diag(torch.diagonal(cov))
    covariance = offdiag.square().mean()
    return variance, covariance


def controlled_ssl_loss(outputs: dict[str, torch.Tensor], config: ControlledSSLLossConfig = ControlledSSLLossConfig()) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the total and separately named terms.

    The target encoder is detached by the model.  Missing masked or future
    frames contribute zero, while a completely missing valid sequence is
    rejected upstream.  The variance/covariance terms are explicit so a low
    prediction loss cannot be mistaken for a useful representation.
    """
    valid = outputs["valid"]
    mask = outputs["mask"] & valid
    if not bool(mask.any()):
        raise ValueError("controlled SSL requires at least one masked valid frame")
    predicted = F.normalize(outputs["predicted"], dim=-1)
    target = F.normalize(outputs["target"].detach(), dim=-1)
    masked_error = 1.0 - (predicted * target).sum(-1)
    masked_prediction = _masked_mean(masked_error, mask)

    future_valid = valid[:, 1:] & valid[:, :-1]
    future_predicted = F.normalize(outputs["future_predicted"], dim=-1)
    future_target = F.normalize(outputs["future_target"].detach(), dim=-1)
    future_error = 1.0 - (future_predicted * future_target).sum(-1)
    future_prediction = _masked_mean(future_error, future_valid) if bool(future_valid.any()) else future_error.new_zeros(())

    variance, covariance = _vicreg_terms(outputs["context"], valid, config.variance_floor)
    total = (config.masked_prediction * masked_prediction
             + config.future_prediction * future_prediction
             + config.variance * variance
             + config.covariance * covariance)
    terms = {
        "masked_prediction": masked_prediction,
        "future_prediction": future_prediction,
        "variance": variance,
        "covariance": covariance,
        "total": total,
    }
    if not all(torch.isfinite(value) for value in terms.values()):
        raise FloatingPointError("controlled SSL produced a non-finite loss")
    return total, terms
