"""D6 effort planning and D7/D8 native-flow conditioning.

Planning labels are kinematic proxies, not annotations of musical meaning.
The flow adapters are pointwise: the incumbent's exact finite halo is unchanged.
No motion, target contacts or target plan is supplied at generation time.
"""
from dataclasses import asdict, dataclass
from functools import partial
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..motion_study.geometry import joints_from_features
from ..motion_study.music_model import MusicConfig, MusicModel
from ..motion_study.music_revision_model import PhysicalWeights, masked_native_losses
from ..motion_study.music_physical import clean_estimate, physical_losses

REGION_NAMES = ('support', 'torso', 'head', 'left_arm', 'right_arm')
REGION_JOINTS = ((0, 1, 2, 4, 5, 7, 8, 10, 11), (3, 6, 9, 12), (15,), (13, 16, 18, 20), (14, 17, 19, 21))
PLAN_CHANNELS = 34  # log_speed5 + support4 + hold5 + duration5x4
DURATION_BINS_SECONDS = (.5, 1., 2.)
DEFAULT_PHYSICAL = PhysicalWeights(position=.5, velocity=.05, planted_foot_velocity=1.,
                                   root_displacement=.25, floor_penetration=10.)


def pool_half_seconds(value, valid=None):
    """Mean every 15 native30Hz frames, retaining a final partial bin."""
    if value.ndim != 3:
        raise ValueError('Expected B,T,C')
    if valid is None:
        valid = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
    if valid.shape != value.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Expected boolean valid mask')
    pad = (-value.shape[1]) % 15
    value = torch.where(valid[..., None], value, 0.)
    value = F.pad(value, (0, 0, 0, pad))
    mask = F.pad(valid, (0, pad))
    count = mask.reshape(mask.shape[0], -1, 15).sum(-1)
    mean = value.reshape(value.shape[0], -1, 15, value.shape[-1]).sum(-2) / count[..., None].clamp_min(1)
    return mean, count > 0


@torch.no_grad()
def kinematic_plan_targets(native):
    """Full recording -> 2Hz proxy targets; root integrated once.

    Hold = regional speed <=.05m/s; duration is remaining time in a regional
    speed<=.2m/s state (or its complement), binned at .5/1/2 seconds. Endpoint
    duration is right-censored, and its loss mask explicitly excludes that run.
    """
    native = torch.as_tensor(native, dtype=torch.float32)
    if native.ndim != 2 or native.shape[-1] != 139 or not len(native) or not torch.isfinite(native).all():
        raise ValueError('Expected finite nonempty native139 recording')
    joints = joints_from_features(native)
    speed = torch.linalg.vector_norm(torch.diff(joints, dim=0) * 30, dim=-1)
    # One-frame recordings carry no observed velocity; speed loss is masked.
    speed = torch.cat((speed[:1], speed), 0) if len(native) > 1 else native.new_zeros((1, 22))
    regional = torch.stack([speed[:, list(ix)].mean(-1) for ix in REGION_JOINTS], -1)
    pooled, valid = pool_half_seconds(regional[None])
    support, _ = pool_half_seconds(native[None, :, 135:].clamp(0, 1))
    regional = pooled[0]
    states = regional <= .2
    duration = torch.zeros_like(regional, dtype=torch.long)
    duration_valid = torch.zeros_like(states)
    for channel in range(5):
        boundaries = torch.nonzero(states[1:, channel] != states[:-1, channel]).flatten().tolist()
        start = 0
        for end in [i + 1 for i in boundaries] + [len(states)]:
            remaining = torch.arange(end-start, 0, -1, dtype=torch.float32) * .5
            duration[start:end, channel] = torch.bucketize(remaining, torch.tensor(DURATION_BINS_SECONDS), right=False)
            if end < len(states):
                duration_valid[start:end, channel] = True
            start = end
    return {'log_speed': torch.log1p(regional), 'support': support[0],
            'hold': (regional <= .05).float(), 'duration': duration,
            'duration_valid': duration_valid, 'valid': valid[0] & (len(native) > 1)}


@dataclass(frozen=True)
class PlannerConfig:
    input_dim: int
    width: int = 96
    dilations: tuple = (1, 2, 4, 8)
    heads: int = 4


class EffortPlanner(nn.Module):
    """One jointly trained local/global planner; local output is the D6 arm.

    D8 uses the full-song branch, D8_flat uses the identical local branch.
    Both heads have the same supervision; global context is masked content
    attention, so recurring audio can exchange information without a clock.
    """
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError('Width must divide attention heads')
        self.config = config
        self.input = nn.Linear(config.input_dim, config.width)
        self.blocks = nn.ModuleList(nn.Conv1d(config.width, config.width, 3, padding=d, dilation=d)
                                    for d in config.dilations)
        self.norm = nn.LayerNorm(config.width)
        self.attention = nn.MultiheadAttention(config.width, config.heads, batch_first=True)
        self.global_mix = nn.Linear(config.width, config.width)
        self.output = nn.Linear(config.width, PLAN_CHANNELS)

    def forward(self, audio, valid):
        if audio.ndim != 3 or valid.shape != audio.shape[:2]:
            raise ValueError('Expected B,T,F audio and B,T valid')
        x = F.silu(self.input(torch.where(valid[..., None], audio, 0.)))
        x = torch.where(valid[..., None], x, 0.)
        for block in self.blocks:
            x = x + .2 * block(F.silu(x).transpose(1, 2)).transpose(1, 2)
            x = torch.where(valid[..., None], x, 0.)
        local = self.output(x)
        query = self.norm(x)
        # Require a nonempty recording; fully padded synthetic rows cannot make
        # all-masked softmax NaN. They remain zero in outputs and losses.
        attention_mask = valid.clone()
        attention_mask[:, 0] = True
        context, _ = self.attention(query, query, query, key_padding_mask=~attention_mask, need_weights=False)
        global_plan = self.output(x + .2 * self.global_mix(context))
        return {'local': torch.where(valid[..., None], local, 0.),
                'global': torch.where(valid[..., None], global_plan, 0.)}

    def config_dict(self):
        return {**asdict(self.config), 'regions': REGION_NAMES, 'fps': 2,
                'target_scope': 'motion-derived proxies; no human semantic labels',
                'local_halo_bins': sum(self.config.dilations),
                'global_context': 'masked full-recording content attention; no target-motion conditioning'}


def unpack_plan(raw):
    if raw.shape[-1] != PLAN_CHANNELS:
        raise ValueError('Expected 34 planning channels')
    return {'log_speed': F.softplus(raw[..., :5]), 'support': raw[..., 5:9].sigmoid(),
            'hold': raw[..., 9:14].sigmoid(),
            'duration': raw[..., 14:].reshape(*raw.shape[:-1], 5, 4).softmax(-1)}


def plan_condition(raw):
    parts = unpack_plan(raw)
    return torch.cat((parts['log_speed'], parts['support'], parts['hold'],
                      parts['duration'].flatten(-2)), -1)


def planner_losses(predictions, targets, valid):
    """Both branches scored equally; no loss on padding or censored durations."""
    valid = valid & targets['valid']
    mask = valid[..., None]
    # Sanitize before nonlinear arithmetic, not only its final reduction.
    # torch.where after a NaN square/BCE can still propagate NaN gradients.
    target_speed = torch.where(mask, targets['log_speed'], 0.)
    target_support = torch.where(mask, targets['support'], 0.)
    target_hold = torch.where(mask, targets['hold'], 0.)
    duration_mask = mask & targets['duration_valid']
    target_duration = torch.where(duration_mask, targets['duration'], 0)
    result = {}
    for branch, raw in predictions.items():
        raw = torch.where(mask, raw, 0.)
        values = unpack_plan(raw)
        def mean(value, mask=mask):
            mask = mask.expand_as(value)
            return torch.where(mask, value, 0.).sum() / mask.sum().clamp_min(1)
        speed = mean((values['log_speed'] - target_speed).square())
        support = mean(F.binary_cross_entropy_with_logits(raw[..., 5:9], target_support, reduction='none'))
        hold = mean(F.binary_cross_entropy_with_logits(raw[..., 9:14], target_hold, reduction='none'))
        ce = F.cross_entropy(raw[..., 14:].reshape(-1, 4), target_duration.flatten(), reduction='none').reshape(*raw.shape[:2], 5)
        duration = mean(ce, duration_mask)
        result.update({branch + '_speed': speed, branch + '_support': support,
                       branch + '_hold': hold, branch + '_duration': duration})
    result['total'] = sum(result.values()) / len(predictions)
    return result


@dataclass(frozen=True)
class DynamicFlowConfig:
    extra_dim: int
    arm: str = 'D7'
    adapter_width: int = 32
    adapter_scale: float = .2
    source_dropout: float = .2
    plan_dim: int = PLAN_CHANNELS

    def __post_init__(self):
        if self.arm not in ('continued', 'D7', 'D8', 'D8_flat'):
            raise ValueError('Unregistered dynamic flow arm')
        if self.extra_dim < 1 or self.plan_dim != PLAN_CHANNELS:
            raise ValueError('Require explicit audio and plan dimensions')
        if not 0 <= self.source_dropout < 1 or not 0 < self.adapter_scale <= 1:
            raise ValueError('Invalid adapter/dropout bounds')


class DynamicFlow(MusicModel):
    """Warmstarted flow with zero-initialized repeated bounded adapters.

    Packed conditioning = normalized Base92, normalized extra audio, plan34.
    Pointwise projections preserve the exact incumbent halo. D8's whole-song
    planning is computed ONCE before chunking; never inside the vector field.
    """
    def __init__(self, config, normalizers, dynamic, physical_weights=DEFAULT_PHYSICAL):
        if config.arm != 'flow':
            raise ValueError('Dynamic campaign requires native flow')
        super().__init__(config, normalizers)
        self.dynamic = dynamic
        self.physical_weights = physical_weights
        self.audio_projection = nn.Linear(dynamic.extra_dim, dynamic.adapter_width)
        self.plan_projection = nn.Linear(dynamic.plan_dim, dynamic.adapter_width)
        self.adapters = nn.ModuleList(nn.Linear(dynamic.adapter_width, config.width) for _ in self.blocks)
        for layer in self.adapters:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def conditioning_dim(self):
        return 92 + self.dynamic.extra_dim + self.dynamic.plan_dim

    def load_incumbent(self, state):
        """Only added adapter/projection keys may be missing, never a base key."""
        unexpected_base = set(state) - set(super().state_dict())
        if unexpected_base:
            raise ValueError('Unexpected incumbent keys: ' + repr(sorted(unexpected_base)))
        result = self.load_state_dict(state, strict=False)
        allowed = ('audio_projection.', 'plan_projection.', 'adapters.')
        if result.unexpected_keys or any(not key.startswith(allowed) for key in result.missing_keys):
            raise ValueError('Incumbent architecture mismatch')

    def vector_field(self, audio, state, t):
        phase = t[:, None] * self.frequencies[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), -1))
        x = self.input(torch.cat((audio[..., :92], state), -1).transpose(1, 2))
        extra = audio[..., 92:92+self.dynamic.extra_dim]
        conditioning = F.silu(self.audio_projection(extra))
        if self.dynamic.arm in ('D8', 'D8_flat'):
            conditioning = conditioning + F.silu(self.plan_projection(audio[..., -self.dynamic.plan_dim:]))
        for block, adapter in zip(self.blocks, self.adapters):
            if self.dynamic.arm != 'continued':
                x = x + self.dynamic.adapter_scale * torch.tanh(adapter(conditioning)).transpose(1, 2)
            x = block(x, time)
        return self.output(F.silu(x)).transpose(1, 2).float()

    def losses_from_draws(self, audio, target, valid, noise, time, *, vector_field=None):
        if audio.shape != (*target.shape[:2], self.conditioning_dim) or target.shape != noise.shape or target.shape[-1] != 139:
            raise ValueError('Training dimensions do not match')
        if valid.shape != target.shape[:2] or valid.dtype != torch.bool or time.shape != (len(target),):
            raise ValueError('Invalid masks or flow times')
        if not bool(torch.isfinite(time).all() & (time >= 0).all() & (time <= 1).all()):
            raise ValueError('Invalid flow times')
        for value in (audio, target, noise):
            if not bool(torch.where(valid[..., None], torch.isfinite(value), True).all()):
                raise ValueError('Nonfinite valid training inputs')
        if not bool(torch.isfinite(self.mean).all() & torch.isfinite(self.std).all() & (self.std > 0).all() & torch.isfinite(self.delta_scale).all() & (self.delta_scale > 0).all()):
            raise ValueError('Invalid native normalizers')
        audio = torch.where(valid[..., None], audio, 0.)
        reference = torch.where(valid[..., None], target.detach().float(), 0.)
        normalized = torch.where(valid[..., None], (reference-self.mean)/self.std, 0.)
        noise = torch.where(valid[..., None], noise.float(), 0.)
        state = (1-time[:, None, None])*noise + time[:, None, None]*normalized
        desired = normalized-noise
        prediction = (self.vector_field if vector_field is None else vector_field)(audio, state, time)
        losses = masked_native_losses(prediction, desired, self.delta_scale, valid)
        physical = physical_losses(clean_estimate(state, prediction, time), reference,
                                   self.mean, self.std, valid)
        for name, weight in asdict(self.physical_weights).items():
            if weight:
                losses['total'] = losses['total'] + weight*physical[name]
        losses.update({'physical_'+name: value for name, value in physical.items()})
        return losses

    def loss_function(self, compile_vector_field=False, backend=None):
        fn = self.vector_field
        if compile_vector_field:
            fn = torch.compile(fn, fullgraph=True, dynamic=False, **({} if backend is None else {'backend': backend}))
        return partial(self.losses_from_draws, vector_field=fn)

    def config_dict(self):
        return {**super().config_dict(), 'dynamic': asdict(self.dynamic),
                'physical_weights': asdict(self.physical_weights),
                'adapter_halo': 0, 'conditioning_dim': self.conditioning_dim,
                'conditioning': 'Base92 + raw frozen audio projection + separately precomputed plan',
                'source_dropout': 'caller-owned named-source mask; paired RNG independent of noise/crops'}


def drop_named_sources(audio, source_indices, keep):
    """Zero normalized named-source descriptors; no hidden RNG inside model.

    keep B,4 values are caller-owned and identical across paired flow arms.
    This is a training feature-dropout intervention, not waveform source removal.
    """
    result = audio.clone()
    if keep.shape != (len(audio), 4) or len(source_indices) != 4:
        raise ValueError('Need explicit four-source mask')
    for source, indices in enumerate(source_indices):
        result[..., list(indices)] *= keep[:, source, None, None]
    return result
