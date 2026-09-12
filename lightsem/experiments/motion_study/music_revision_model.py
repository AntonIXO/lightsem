"""Opt-in matched flow objectives; no training launcher or architecture changes.

Compile only vector_field via loss_function(). The loss wrapper and FK stay eager
so music_physical's safety checks remain active. GPU synchronization is deliberate;
throughput and CUDA compilation are NOT validated by the CPU engineering tests.
"""
from dataclasses import asdict, dataclass
from functools import partial
import math

import torch

from .music_model import MusicModel
from .music_physical import clean_estimate, physical_losses


@dataclass(frozen=True)
class PhysicalWeights:
    """Caller-frozen weights, in physical_losses' units. No recommended setting."""
    position: float = 0.
    velocity: float = 0.
    planted_foot_velocity: float = 0.
    root_displacement: float = 0.
    floor_penetration: float = 0.

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError('Physical weights must be finite and nonnegative')


def masked_native_losses(prediction, desired, delta_scale, valid):
    """Original coordinate and derivative metric, pooling only valid scalars.

    Both endpoints must be valid for an edge. Empty support gives differentiable
    zero. Sanitize before subtraction/squaring so even NaN padding has no gradient.
    Shape/type checks are performed by the public model loss wrapper.
    """
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        prediction = torch.where(valid[..., None], prediction.float(), 0.)
        desired = torch.where(valid[..., None], desired.float(), 0.)
        error = prediction-desired
        native = error.square().sum() / (valid.sum()*error.shape[-1]).clamp_min(1)
        edge = valid[:, :-1] & valid[:, 1:]
        delta = torch.where(edge[..., None], torch.diff(error, dim=1), 0.) / delta_scale
        velocity = delta.square().sum() / (edge.sum()*error.shape[-1]).clamp_min(1)
        return {'total': native+.05*velocity, 'native': native, 'velocity': velocity}


class MusicRevisionModel(MusicModel):
    """Same native flow network, draws and audio-only generation for both arms.

    valid is a loss/data mask, never an inference conditioning input. Invalid
    audio and normalized state are zeroed before the unchanged vector field.
    Both arms use this same boundary convention and the same caller-owned draws.
    """
    def __init__(self, config, normalizers, *, arm, weights=PhysicalWeights()):
        if config.arm != 'flow':
            raise ValueError('Revision tests the physical objective on flow only')
        if arm not in ('masked_native', 'physical'):
            raise ValueError('Unknown revision arm')
        if not isinstance(weights, PhysicalWeights):
            raise TypeError('Supply fixed PhysicalWeights, not an implicit mapping')
        if arm == 'masked_native' and any(asdict(weights).values()):
            raise ValueError('Masked-native arm must have zero physical weights')
        super().__init__(config, normalizers)
        self.revision_arm, self.physical_weights = arm, weights
        self._validate_normalizers()

    def _validate_normalizers(self):
        for name in ('mean', 'std', 'delta_scale'):
            value = getattr(self, name)
            if value.shape != (139,) or not bool(torch.isfinite(value).all()):
                raise ValueError('Expected finite native139 normalizers')
            if name != 'mean' and not bool((value > 0).all()):
                raise ValueError('Normalizer scales must be positive')

    def _validate_inputs(self, audio, target, valid, noise, time):
        if target.ndim != 3 or target.shape[-1] != 139 or min(target.shape[:2]) < 1:
            raise ValueError('Expected nonempty (B,T,139) target')
        if audio.shape != (*target.shape[:2], 92) or noise.shape != target.shape:
            raise ValueError('Audio/noise dimensions do not match target')
        if valid.shape != target.shape[:2] or valid.dtype != torch.bool:
            raise ValueError('Expected boolean (B,T) valid frame mask')
        if time.shape != (target.shape[0],):
            raise ValueError('Expected (B,) flow time')
        if any(v.device != self.mean.device for v in (audio, target, valid, noise, time)):
            raise ValueError('Model and all loss inputs must share a device')
        if not bool(torch.isfinite(time).all() & (time >= 0).all() & (time <= 1).all()):
            raise ValueError('Time must be finite and in [0,1]')
        for value in (audio, target, noise):
            if not bool(torch.where(valid[..., None], torch.isfinite(value), True).all()):
                raise ValueError('Nonfinite valid training input')
        # Deliberately rechecked after checkpoint loads or buffer mutation.
        self._validate_normalizers()

    def losses_from_draws(self, audio, target, valid, noise, time):
        return self._losses_from_draws(audio, target, valid, noise, time,
                                      vector_field=self.vector_field)

    def _losses_from_draws(self, audio, target, valid, noise, time, *, vector_field):
        self._validate_inputs(audio, target, valid, noise, time)
        # No padded values can leak into adjacent valid predictions through convs.
        audio = torch.where(valid[..., None], audio, 0.)
        reference = torch.where(valid[..., None], target.detach().float(), 0.)
        normalized = torch.where(valid[..., None], (reference-self.mean)/self.std, 0.)
        noise = torch.where(valid[..., None], noise.float(), 0.)
        state = (1-time[:, None, None])*noise + time[:, None, None]*normalized
        desired = normalized-noise
        prediction = vector_field(audio, state, time)
        losses = masked_native_losses(prediction, desired, self.delta_scale, valid)
        if self.revision_arm == 'physical':
            physical = physical_losses(clean_estimate(state, prediction, time), reference,
                                       self.mean, self.std, valid)
            # Zero weights do not attach unused FK graphs or change total arithmetic.
            for name, weight in asdict(self.physical_weights).items():
                if weight:
                    losses['total'] = losses['total'] + weight*physical[name]
            losses.update({'physical_'+name: value for name, value in physical.items()})
        return losses

    def loss_function(self, *, compile_vector_field=False, backend=None):
        """Eager safety/loss shell + optional fullgraph compiled neural forward.

        Do NOT use TrainingRuntime.loss_function(model) for this revision: it
        compiles the entire validated loss and introduces scalar graph breaks.
        This explicit split preserves all checks (including GPU bool syncs).
        backend is for engineering tests; production default is Torch Inductor.
        No fallback or safety suppression is installed if compilation fails.
        """
        fn = self.vector_field
        if compile_vector_field:
            options = {} if backend is None else {'backend': backend}
            fn = torch.compile(fn, fullgraph=True, dynamic=False, **options)
        elif backend is not None:
            raise ValueError('backend requires compile_vector_field=True')
        return partial(self._losses_from_draws, vector_field=fn)

    def config_dict(self):
        return {**super().config_dict(), 'revision_arm': self.revision_arm,
                'physical_weights': asdict(self.physical_weights),
                'loss': 'valid native MSE + .05 valid-edge derivative MSE; physical arm adds caller-fixed clean-estimate FK terms',
                'masking': 'zero audio/normalized state on invalid frames; no invalid frame or cross-boundary derivative scored',
                'compile_strategy': 'vector_field only, fullgraph; eager validated loss and FK',
                'generation': 'unchanged MusicModel.generate; audio, RNG/state and solver settings only'}
