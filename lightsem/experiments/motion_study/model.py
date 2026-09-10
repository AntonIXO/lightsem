"""Continuous temporal motion compression; geometry losses stay FP32."""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import DIM, FPS, BODY_HEIGHT, FEET, joints_from_features


@dataclass(frozen=True)
class CodecConfig:
    arm: str = 'temporal64'
    width: int = 256
    frames: int = 480

    @property
    def latent(self):
        return {'frame24': 24, 'temporal64': 64, 'temporal128': 128}[self.arm]

    @property
    def stride(self):
        return 1 if self.arm == 'frame24' else 4

    def __post_init__(self):
        if self.arm not in ('frame24', 'temporal64', 'temporal128') or self.frames % 4:
            raise ValueError('Unsupported codec configuration')


class Residual(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
                                 nn.SiLU(), nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation))

    def forward(self, x):
        return x + self.net(F.silu(x))*.2


class MotionCodec(nn.Module):
    def __init__(self, config, normalizers):
        super().__init__()
        self.config = config
        width, latent = config.width, config.latent
        self.register_buffer('mean', torch.tensor(normalizers['mean'], dtype=torch.float32))
        self.register_buffer('std', torch.tensor(normalizers['std'], dtype=torch.float32))
        self.register_buffer('velocity_scale', torch.tensor(normalizers['velocity_rms'], dtype=torch.float32))
        self.register_buffer('acceleration_scale', torch.tensor(normalizers['acceleration_rms'], dtype=torch.float32))
        if config.stride == 1:
            self.encoder = nn.Sequential(nn.Conv1d(DIM, width, 1), nn.SiLU(), nn.Conv1d(width, latent, 1))
        else:
            self.encoder = nn.Sequential(nn.Conv1d(DIM, width, 5, padding=2), nn.SiLU(),
                nn.Conv1d(width, width, 4, stride=2, padding=1), nn.SiLU(),
                nn.Conv1d(width, width, 4, stride=2, padding=1),
                *(Residual(width, d) for d in (1, 2, 4, 8, 16)), nn.SiLU(), nn.Conv1d(width, latent, 1))
        self.decoder_core = nn.Sequential(nn.Conv1d(latent, width, 1),
            *(Residual(width, d) for d in (1, 2, 4, 8, 16)))
        self.up1 = nn.Conv1d(width, width, 5, padding=2) if config.stride == 4 else nn.Identity()
        self.up2 = nn.Conv1d(width, width, 5, padding=2) if config.stride == 4 else nn.Identity()
        self.output = nn.Conv1d(width, DIM, 1)

    def config_dict(self):
        return {**asdict(self.config), 'latent_channels': self.config.latent,
                'latent_fps': FPS/self.config.stride, 'native_fps': FPS,
                'loss_weights': {'features': 1., 'position': 10., 'velocity': 1., 'acceleration': .25,
                                 'root': 1., 'contacts': .25, 'contact_geometry': 1.},
                'rotations': 'continuous6D-rows-with-SO3-projection', 'root': 'vx_vz_height-integrate-once'}

    def encode(self, features):
        x = ((features-self.mean)/self.std).transpose(1, 2)
        padding = (-x.shape[-1]) % self.config.stride
        if padding:
            x = F.pad(x, (0, padding), mode='replicate')
        return self.encoder(x).transpose(1, 2)

    def decode(self, latent):
        x = self.decoder_core(latent.transpose(1, 2))
        if self.config.stride == 4:
            x = self.up1(F.interpolate(F.silu(x), scale_factor=2., mode='linear', align_corners=False))
            x = self.up2(F.interpolate(F.silu(x), scale_factor=2., mode='linear', align_corners=False))
        raw = self.output(F.silu(x)).transpose(1, 2).float()*self.std+self.mean
        return torch.cat((raw[..., :135], torch.sigmoid(raw[..., 135:])), -1)

    def forward(self, features):
        return self.decode(self.encode(features))[:, :features.shape[1]]

    def forward_chunked(self, features, chunk_frames=960):
        if chunk_frames % 4:
            raise ValueError('Chunk size must preserve the latent phase')
        # Bounds exceed the composed encoder/decoder receptive field. Outputs
        # remain velocities; integration happens once after concatenation.
        halo = 640
        outputs = []
        for start in range(0, features.shape[1], chunk_frames):
            end = min(start+chunk_frames, features.shape[1])
            lo, hi = max(0, start-halo), min(features.shape[1], end+halo)
            local = self(features[:, lo:hi])
            outputs.append(local[:, start-lo:end-lo])
        return torch.cat(outputs, dim=1)

    def losses_from_draws(self, features, world_joints, unused):
        prediction = self(features)
        with torch.autocast(device_type=features.device.type, enabled=False):
            prediction = prediction.float()
            target = features.float()
            joints = joints_from_features(prediction)
            reference = world_joints.float()
            origin = reference[:, :1, :1] * reference.new_tensor([1, 0, 1])
            reference = reference-origin
            feature_loss = ((prediction-target)/self.std).square().mean()
            position_loss = ((joints-reference)/BODY_HEIGHT).square().sum(-1).mean()
            velocity_error = (torch.diff(joints, dim=1)-torch.diff(reference, dim=1))*FPS
            velocity_loss = velocity_error.square().sum(-1).mean()/self.velocity_scale.square().clamp_min(1e-8)
            acceleration_error = torch.diff(velocity_error, dim=1)*FPS
            acceleration_loss = acceleration_error.square().sum(-1).mean()/self.acceleration_scale.square().clamp_min(1e-8)
            root_loss = ((joints[:, :, 0]-reference[:, :, 0])/BODY_HEIGHT).square().sum(-1).mean()
            contacts = F.binary_cross_entropy(prediction[..., 135:].clamp(1e-6, 1-1e-6), target[..., 135:])
            # A contact classifier cannot substitute for physically correct feet.
            # Match source contact geometry/velocity, never erase real foot motion.
            contact_mask = target[..., 135:]
            foot_error = ((joints-reference)[:, :, FEET]/BODY_HEIGHT).square().sum(-1)
            foot_position = (foot_error*contact_mask).sum()/contact_mask.sum().clamp_min(1)
            consecutive = contact_mask[:, 1:]*contact_mask[:, :-1]
            foot_velocity_error = velocity_error[:, :, FEET].square().sum(-1)/self.velocity_scale.square().clamp_min(1e-8)
            foot_velocity = (foot_velocity_error*consecutive).sum()/consecutive.sum().clamp_min(1)
            contact_geometry = 25*foot_position+foot_velocity
            total = feature_loss + 10*position_loss + velocity_loss + .25*acceleration_loss + root_loss + .25*contacts + contact_geometry
        return {'total': total, 'features': feature_loss, 'position': position_loss,
                'velocity': velocity_loss, 'acceleration': acceleration_loss, 'root': root_loss,
                'contacts': contacts, 'contact_geometry': contact_geometry}
