"""One-step trainer and input-sensitivity audit for body-light SSL."""

from __future__ import annotations

from dataclasses import asdict

import torch

from .losses import ControlledSSLLossConfig, controlled_ssl_loss
from .model import ControlledJepaConfig, ControlledJepaModel, make_mask


class ControlledSSLLoop:
    """A deterministic single-seed training loop with an EMA target encoder."""

    def __init__(self, config: ControlledJepaConfig, *, seed: int = 0,
                 learning_rate: float = 3e-4,
                 loss_config: ControlledSSLLossConfig = ControlledSSLLossConfig()):
        if seed < 0:
            raise ValueError("seed must be non-negative")
        torch.manual_seed(seed)
        self.config = config
        self.loss_config = loss_config
        self.model = ControlledJepaModel(config)
        self.optimizer = torch.optim.AdamW(self.model.context_encoder.parameters(), lr=learning_rate, weight_decay=1e-2)
        self.optimizer.add_param_group({"params": self.model.predictor.parameters()})
        self.optimizer.add_param_group({"params": self.model.future_predictor.parameters()})
        self.optimizer.add_param_group({"params": [self.model.mask_token]})
        self.mask_generator = torch.Generator(device="cpu").manual_seed(seed + 17)
        self.steps = 0

    def step(self, audio: torch.Tensor, body: torch.Tensor, light: torch.Tensor,
             timestamps: torch.Tensor, valid: torch.Tensor | None = None,
             mask: torch.Tensor | None = None) -> dict[str, float]:
        self.model.train()
        if valid is None:
            valid = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
        if mask is None:
            mask = make_mask(valid, self.config.mask_ratio, generator=self.mask_generator)
        outputs = self.model(audio, body, light, timestamps, valid, mask)
        loss, terms = controlled_ssl_loss(outputs, self.loss_config)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0).detach())
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError("controlled SSL gradient norm is non-finite")
        self.optimizer.step()
        self.model.ema_update()
        self.steps += 1
        return {"step": float(self.steps), "mask_fraction": float(mask.float().mean()),
                "gradient_norm": gradient_norm, **{name: float(value.detach()) for name, value in terms.items()}}

    def state_dict(self) -> dict:
        return {"config": self.model.config_dict(), "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(), "steps": self.steps,
                "loss_config": asdict(self.loss_config)}


@torch.no_grad()
def audit_input_sensitivity(model: ControlledJepaModel, audio: torch.Tensor, body: torch.Tensor,
                            light: torch.Tensor, timestamps: torch.Tensor,
                            valid: torch.Tensor | None = None) -> dict[str, float]:
    """Measure representation response to body/light interventions.

    This is a diagnostic, not a qualification gate by itself.  A body or light
    intervention must be paired with the native raw-body safety and source-
    dependence audits before any downstream policy or demo is considered.
    """
    model.eval()
    if valid is None:
        valid = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
    base_mask = torch.zeros_like(valid)
    base = model(audio, body, light, timestamps, valid, base_mask)
    body_alt = body.clone()
    body_alt[..., :min(3, body_alt.shape[-1])] += 0.5
    body_changed = model(audio, body_alt, light, timestamps, valid, base_mask)
    light_alt = light.roll(shifts=1, dims=1)
    light_changed = model(audio, body, light_alt, timestamps, valid, base_mask)
    body_delta = (body_changed["context"] - base["context"]).abs()[valid].mean()
    light_delta = (light_changed["future_predicted"] - base["future_predicted"]).abs()[valid[:, 1:]].mean()
    embeddings = base["context"][valid]
    centered = embeddings - embeddings.mean(0, keepdim=True)
    singular = torch.linalg.svdvals(centered) if embeddings.shape[0] > 1 else embeddings.new_zeros((1,))
    probabilities = singular.square() / singular.square().sum().clamp_min(1e-12)
    effective_rank = torch.exp(-(probabilities[probabilities > 0] * torch.log(probabilities[probabilities > 0])).sum())
    return {
        "body_context_delta": float(body_delta),
        "light_future_delta": float(light_delta),
        "embedding_std": float(embeddings.std()),
        "effective_rank": float(effective_rank),
        "finite": float(torch.isfinite(embeddings).all()),
    }
