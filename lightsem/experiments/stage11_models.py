"""Matched pointwise and temporal scene-flow models for Stage 11.

Inputs are externally normalized audio ``(B, T, 52)`` and scene targets
``(B, T, 6)`` at 10 Hz. No normalization is learned or recomputed here: the
training/checkpoint interface must pin the training-set transform. Both arms
use the same offline Conformer; only the scene/noise denoiser differs.

Sampling never consumes reference scenes. Generate one noise tensor for the
whole song, then integrate a global Euler state. Chunked sampling evaluates
that same global velocity field in pieces at *each* integration step; it does
not independently generate windows or blend their predictions.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from typing import Iterator, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lightsem.model.light_trunk import ConformerBlock, TrunkConfig


@dataclass(frozen=True)
class SceneFlowConfig:
    flow_kind: str = "temporal"
    in_dim: int = 52
    scene_dim: int = 6
    d_model: int = 192
    n_blocks: int = 3
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 15
    attn_window: int = 80
    dropout: float = 0.1
    flow_kernel: int = 5
    flow_dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    def __post_init__(self) -> None:
        # JSON checkpoints store tuples as lists.
        object.__setattr__(self, "flow_dilations", tuple(self.flow_dilations))
        if self.flow_kind not in {"pointwise", "temporal"}:
            raise ValueError("flow_kind must be 'pointwise' or 'temporal'")
        for name in ("in_dim", "scene_dim", "d_model", "n_blocks", "n_heads", "ff_mult"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.n_heads or self.d_model % 2:
            raise ValueError("d_model must be even and divisible by n_heads")
        if self.conv_kernel < 1 or self.conv_kernel % 2 != 1:
            raise ValueError("conv_kernel must be a positive odd integer")
        if self.flow_kernel < 1 or self.flow_kernel % 2 != 1:
            raise ValueError("flow_kernel must be a positive odd integer")
        if self.attn_window < 2 or self.attn_window % 2:
            raise ValueError("attn_window must be a positive even integer >= 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not self.flow_dilations or any(
            not isinstance(d, int) or isinstance(d, bool) or d < 1
            for d in self.flow_dilations
        ):
            raise ValueError("flow_dilations must contain positive integers")

    @property
    def encoder_radius(self) -> int:
        """Exact symmetric context of the existing Conformer implementation."""
        # Its attention mask uses strict inequalities, hence half-window - 1.
        return self.n_blocks * (
            self.attn_window // 2 - 1 + self.conv_kernel // 2
        )

    @property
    def velocity_radius(self) -> int:
        if self.flow_kind == "pointwise":
            return 0
        return (self.flow_kernel // 2) * sum(self.flow_dilations)


class _SceneResidual(nn.Module):
    def __init__(self, cfg: SceneFlowConfig, dilation: int):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.temporal = cfg.flow_kind == "temporal"
        if self.temporal:
            self.net = nn.Conv1d(
                cfg.d_model,
                cfg.d_model,
                cfg.flow_kernel,
                dilation=dilation,
                padding=(cfg.flow_kernel // 2) * dilation,
            )
        else:
            hidden = cfg.d_model * 5 // 2
            self.net = nn.Sequential(
                nn.Linear(cfg.d_model, hidden),
                nn.SiLU(),
                nn.Linear(hidden, cfg.d_model),
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(x)
        if self.temporal:
            y = self.net(y.transpose(1, 2)).transpose(1, 2)
        else:
            y = self.net(y)
        return x + F.silu(y)


def _normal_like(reference: Tensor, generator: torch.Generator | None) -> Tensor:
    # A CPU generator is useful when one frozen noise field must be shared
    # across CPU/GPU evaluation. Drawing on its own device avoids implicit use
    # of a different/global RNG when the model lives elsewhere.
    device = reference.device if generator is None else generator.device
    return torch.randn(
        reference.shape, dtype=reference.dtype, device=device, generator=generator
    ).to(reference.device)


@contextmanager
def _evaluation_mode(module: nn.Module) -> Iterator[None]:
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, mode in modes:
            child.training = mode


class SceneFlowModel(nn.Module):
    """Audio-conditioned flow; shared modules initialize identically by seed.

    ``velocity(x, z, t)`` and ``forward(x, z, t)`` predict velocities. ``t``
    may be a scalar, ``(B,)``, ``(B, 1, 1)`` or ``(B, T, 1)``; training loss
    permits only a single flow time per sequence. ``encode_audio`` and
    ``velocity_from_h`` expose encoder reuse for inference.

    Default parameter counts: temporal 3,727,494; pointwise 3,730,374. Reset
    the initialization seed before constructing each arm. Shared modules are
    constructed before the arm-specific denoiser, so their parameters match.
    """

    def __init__(
        self,
        flow_kind: str | None = None,
        *,
        dim: int | None = None,
        config: SceneFlowConfig | None = None,
    ):
        super().__init__()
        cfg = config or SceneFlowConfig()
        overrides = {}
        if flow_kind is not None:
            overrides["flow_kind"] = flow_kind
        if dim is not None:
            overrides["d_model"] = dim
        self.cfg = replace(cfg, **overrides) if overrides else cfg
        cfg = self.cfg
        trunk_cfg = TrunkConfig(
            in_dim=cfg.in_dim,
            d_model=cfg.d_model,
            n_blocks=cfg.n_blocks,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            conv_kernel=cfg.conv_kernel,
            attn_window=cfg.attn_window,
            offline=True,
            dropout=cfg.dropout,
        )
        self.audio_in = nn.Linear(cfg.in_dim, cfg.d_model)
        self.encoder = nn.ModuleList(
            [ConformerBlock(trunk_cfg) for _ in range(cfg.n_blocks)]
        )
        self.flow_in = nn.Linear(cfg.d_model + cfg.scene_dim + 1, cfg.d_model)
        self.flow_out = nn.Linear(cfg.d_model, cfg.scene_dim)
        self.denoiser = nn.ModuleList(
            [_SceneResidual(cfg, dilation) for dilation in cfg.flow_dilations]
        )

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    @classmethod
    def from_config(cls, config: Mapping | SceneFlowConfig) -> "SceneFlowModel":
        return cls(config=config if isinstance(config, SceneFlowConfig)
                   else SceneFlowConfig(**dict(config)))

    def n_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _check_sequence(self, x: Tensor, channels: int, name: str) -> None:
        if x.ndim != 3 or x.shape[-1] != channels or min(x.shape[:2]) < 1:
            raise ValueError(f"{name} must have nonempty shape (B, T, {channels})")
        if not x.is_floating_point():
            raise ValueError(f"{name} must be floating point")

    def encode_audio(self, x: Tensor) -> Tensor:
        self._check_sequence(x, self.cfg.in_dim, "x")
        h = self.audio_in(x)
        for block in self.encoder:
            h = block(h)
        return h

    @staticmethod
    def _flow_time(t: Tensor | float, z: Tensor, *, per_sequence: bool = False) -> Tensor:
        t = torch.as_tensor(t, dtype=z.dtype, device=z.device)
        if t.ndim == 0:
            t = t.reshape(1, 1, 1)
        elif t.ndim == 1:
            t = t[:, None, None]
        if t.ndim != 3 or t.shape[2] != 1 or t.shape[0] not in (1, z.shape[0]):
            raise ValueError("t must be scalar, (B,), (B, 1, 1), or (B, T, 1)")
        if t.shape[1] not in (1, z.shape[1]):
            raise ValueError("t time dimension must be 1 or match the sequence")
        if per_sequence and t.shape[1] != 1:
            raise ValueError("training requires one flow time per sequence")
        return t.expand(z.shape[0], z.shape[1], 1)

    def velocity_from_h(self, h: Tensor, z: Tensor, t: Tensor | float) -> Tensor:
        self._check_sequence(h, self.cfg.d_model, "h")
        self._check_sequence(z, self.cfg.scene_dim, "z")
        if h.shape[:2] != z.shape[:2]:
            raise ValueError("h and z must have matching batch/time dimensions")
        x = self.flow_in(torch.cat((h, z, self._flow_time(t, z)), dim=-1))
        for block in self.denoiser:
            x = block(x)
        return self.flow_out(x)

    def velocity(self, x: Tensor, z: Tensor, t: Tensor | float) -> Tensor:
        return self.velocity_from_h(self.encode_audio(x), z, t)

    def forward(self, x: Tensor, z: Tensor, t: Tensor | float) -> Tensor:
        return self.velocity(x, z, t)

    def loss(
        self,
        x: Tensor,
        y: Tensor,
        generator: torch.Generator | None = None,
        *,
        noise: Tensor | None = None,
        t: Tensor | float | None = None,
        noise_generator: torch.Generator | None = None,
        time_generator: torch.Generator | None = None,
    ) -> Tensor:
        """Conditional flow matching with one uniform time per whole crop.

        ``generator`` drives both draws by default. Separate noise/time
        generators are also accepted for independently checkpointed streams.
        Encoder dropout uses Torch's ordinary training RNG, which must also
        be checkpointed by the trainer.
        """
        self._check_sequence(y, self.cfg.scene_dim, "y")
        if x.shape[:2] != y.shape[:2]:
            raise ValueError("x and y must have matching batch/time dimensions")
        if noise is None:
            noise = _normal_like(y, noise_generator if noise_generator is not None else generator)
        elif noise.shape != y.shape or noise.device != y.device or noise.dtype != y.dtype:
            raise ValueError("noise must match y shape, device and dtype")
        if t is None:
            rng = time_generator if time_generator is not None else generator
            t = torch.rand(
                (y.shape[0], 1, 1),
                dtype=y.dtype,
                device=y.device if rng is None else rng.device,
                generator=rng,
            ).to(y.device)
        tt = self._flow_time(t, y, per_sequence=True)
        z = (1 - tt) * noise + tt * y
        return F.mse_loss(self.velocity(x, z, tt), y - noise)

    @staticmethod
    def _check_steps(steps: int) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
            raise ValueError("steps must be a positive integer")

    def _initial_state(
        self, x: Tensor, noise: Tensor | None, generator: torch.Generator | None
    ) -> Tensor:
        expected = (*x.shape[:2], self.cfg.scene_dim)
        if noise is not None:
            if tuple(noise.shape) != expected or noise.dtype != x.dtype:
                raise ValueError("noise must match the song batch/time, scene dimension and dtype")
            return noise.to(x.device).clone()
        return _normal_like(x.new_empty(expected), generator)

    @torch.no_grad()
    def sample(
        self,
        x: Tensor,
        steps: int = 8,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Full-song Euler sample on the input/model device; restore train mode."""
        self._check_steps(steps)
        self._check_sequence(x, self.cfg.in_dim, "x")
        with _evaluation_mode(self):
            h = self.encode_audio(x)
            z = self._initial_state(x, noise, generator)
            for step in range(steps):
                z = z + self.velocity_from_h(h, z, step / steps) / steps
        return z

    @torch.no_grad()
    def encode_audio_chunked(
        self,
        x: Tensor,
        chunk_size: int = 600,
        *,
        output_device: torch.device | str | None = None,
    ) -> Tensor:
        """Exact local encoder evaluation with its composed symmetric halo.

        Accepts CPU input even for a GPU model. Only one input chunk plus its
        context is moved to the model device. Output defaults to the input
        device; use CPU output to bound GPU memory for long recordings.
        """
        self._check_sequence(x, self.cfg.in_dim, "x")
        self._check_steps(chunk_size)
        device = self.audio_in.weight.device
        out = torch.empty(
            (*x.shape[:2], self.cfg.d_model), dtype=x.dtype,
            device=x.device if output_device is None else output_device,
        )
        radius = self.cfg.encoder_radius
        with _evaluation_mode(self):
            for start in range(0, x.shape[1], chunk_size):
                end = min(start + chunk_size, x.shape[1])
                lo, hi = max(0, start - radius), min(x.shape[1], end + radius)
                h = self.encode_audio(x[:, lo:hi].to(device))
                out[:, start:end] = h[:, start - lo:end - lo].to(out.device)
        return out

    @torch.no_grad()
    def sample_chunked(
        self,
        x: Tensor,
        steps: int = 8,
        chunk_size: int = 600,
        generator: torch.Generator | None = None,
        noise: Tensor | None = None,
        *,
        output_device: torch.device | str | None = None,
    ) -> Tensor:
        """Integrate one global noise field with exact per-step halo evaluation.

        ``z`` is never updated in place while visiting chunks: every chunk in
        a step observes the same pre-update state. The temporal halo is 126
        frames per velocity evaluation, not a guessed 600-frame overlap. An
        independently sampled eight-step window would instead need 1008
        frames of noise context, plus 138 frames of encoder context.

        When input/output are CPU and the model is on GPU, global encoder and
        scene states stay on CPU and only the current windows occupy GPU.
        For bitwise reuse of initial conditions across hardware, supply the
        same explicit ``noise`` or a CPU generator with its frozen state.
        Floating-point kernel differences may produce rounding-level changes.
        """
        self._check_steps(steps)
        self._check_steps(chunk_size)
        self._check_sequence(x, self.cfg.in_dim, "x")
        device = self.audio_in.weight.device
        storage = x.device if output_device is None else torch.device(output_device)
        with _evaluation_mode(self):
            h = self.encode_audio_chunked(x, chunk_size, output_device=storage)
            z = self._initial_state(x, noise, generator).to(storage)
            radius = self.cfg.velocity_radius
            for step in range(steps):
                next_z = torch.empty_like(z)
                for start in range(0, x.shape[1], chunk_size):
                    end = min(start + chunk_size, x.shape[1])
                    lo, hi = max(0, start - radius), min(x.shape[1], end + radius)
                    velocity = self.velocity_from_h(
                        h[:, lo:hi].to(device), z[:, lo:hi].to(device), step / steps
                    )
                    core = velocity[:, start - lo:end - lo].to(storage)
                    next_z[:, start:end] = z[:, start:end] + core / steps
                z = next_z
        return z
