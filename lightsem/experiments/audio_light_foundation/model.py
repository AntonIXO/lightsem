"""Audio-only learned stage-light controller with training-only dance teacher."""

from dataclasses import dataclass, asdict
import torch
from torch import nn
import torch.nn.functional as F
from ..foundation_audio_light.audio_encoder import AudioEncoder, AudioEncoderConfig
from ..foundation_audio_light.long_context import LongContextMemory, LongContextConfig
from ..foundation_audio_light.fixture_queries import FixtureQueryDecoder


@dataclass(frozen=True)
class AudioToLightConfig:
    audio_dim: int=1116
    body_dim: int=66
    width: int=96
    heads: int=4
    local_window: int=32
    latent_slots: int=8
    fixtures: int=8
    local_rate_hz: float=10.0
    adapter_kind: str="none"
    adapter_bottleneck: int=32
    adapter_heads: int=4
    adapter_local_window: int=32


class AudioToLightFoundation(nn.Module):
    """Primary audio→light model; body is never an inference input."""
    def __init__(self, config=AudioToLightConfig()):
        super().__init__(); self.config=config
        self.audio=AudioEncoder(AudioEncoderConfig(
            config.audio_dim,config.width,local_rate_hz=config.local_rate_hz,
            adapter_kind=config.adapter_kind,
            adapter_bottleneck=config.adapter_bottleneck,
            adapter_heads=config.adapter_heads,
            adapter_local_window=config.adapter_local_window,
        ))
        self.context=LongContextMemory(LongContextConfig(config.width,config.heads,config.local_window,config.latent_slots))
        self.fixtures=FixtureQueryDecoder(8,config.width,config.heads,config.latent_slots)
        self.body_teacher=nn.Sequential(nn.LayerNorm(config.body_dim),nn.Linear(config.body_dim,config.width),nn.SiLU(),nn.Linear(config.width,config.width))
        self.body_target=nn.Linear(config.width,config.body_dim)
        # Do not normalize each fixture token immediately before the heads.
        # That erased much of the time-varying amplitude carried by the
        # recurrent audio context and let a static fixture fan minimize the
        # auxiliary motion loss.  The backbone and fixture decoder still
        # retain their own normalization; these heads preserve its temporal
        # scale for target matching.
        self.aim=nn.Sequential(nn.Linear(config.width,config.width),nn.SiLU(),nn.Linear(config.width,2))
        # A direct audio residual keeps the temporal signal from being
        # attenuated by the static fixture query.  It is shared across
        # fixtures and added to the equivariant fixture-specific head, so it
        # cannot encode fixture order or create an independent clock.
        self.aim_audio=nn.Sequential(nn.Linear(config.width,config.width),nn.SiLU(),nn.Linear(config.width,2))
        self.speed=nn.Sequential(nn.Linear(config.width,config.width//2),nn.SiLU(),nn.Linear(config.width//2,1))
        self.acceleration=nn.Sequential(nn.Linear(config.width,config.width//2),nn.SiLU(),nn.Linear(config.width//2,1))
        self.rgb=nn.Sequential(nn.Linear(config.width,config.width),nn.SiLU(),nn.Linear(config.width,3))
        self.rgb_audio=nn.Sequential(nn.Linear(config.width,config.width),nn.SiLU(),nn.Linear(config.width,3))
        self.brightness=nn.Sequential(nn.Linear(config.width,config.width//2),nn.SiLU(),nn.Linear(config.width//2,1))
        self.brightness_audio=nn.Sequential(nn.Linear(config.width,config.width//2),nn.SiLU(),nn.Linear(config.width//2,1))
        self.event=nn.Linear(config.width,1)

    def _decode(self, hidden, fixtures, latent_seed):
        # Reusing the final latent state for every historical frame breaks
        # full-vs-streaming equivalence. The recurrent latent still conditions
        # ``hidden`` inside LongContextMemory; fixture queries use its learned
        # seed as a time-independent context.
        fixture_tokens=self.fixtures(hidden,fixtures,latent_seed)
        aim_logits=self.aim(fixture_tokens)+self.aim_audio(hidden)[:,:,None,:]
        rgb_logits=self.rgb(fixture_tokens)+self.rgb_audio(hidden)[:,:,None,:]
        brightness_logits=self.brightness(fixture_tokens)+self.brightness_audio(hidden)[:,:,None,:]
        out={'aim':aim_logits.tanh(),'speed':F.softplus(self.speed(fixture_tokens)),
             'acceleration':self.acceleration(fixture_tokens).tanh(),'rgb':rgb_logits.sigmoid(),
             'rgb_logits':rgb_logits,
             'brightness':brightness_logits.sigmoid(),'event':self.event(hidden)}
        return out

    def forward(self,audio,fixtures,timestamps=None,valid=None,state=None,duration_seconds=None):
        tokens=self.audio(audio,timestamps=timestamps,valid=valid,duration_seconds=duration_seconds)
        hidden,new_state=self.context(tokens,valid,state,return_state=True)
        latent_seed=self.context.latent_seed[None].expand(hidden.shape[0],-1,-1)
        out=self._decode(hidden,fixtures,latent_seed)
        out['audio_latent']=hidden
        out['state']=new_state
        return out

    def teacher_forward(self,body,fixtures):
        """Body-to-light privileged teacher used only while fitting audio."""
        hidden=self.body_teacher(body)
        latent=hidden.mean(1,keepdim=True)
        return self._decode(hidden,fixtures,latent)

    def dance_target(self,body):
        # This branch is only called during training/evaluation of the
        # privileged target.  It is never passed to forward().
        return self.body_target(self.body_teacher(body))

    def config_dict(self):
        return {'architecture':'audio_tokens+local_attention+GRUCell_state+Perceiver_latents+learned_light_heads',**asdict(self.config),'body_inference_input':False}
