"""Matched native-motion regression and conditional flow with one global state."""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .native_codec import NativeCodec


@dataclass(frozen=True)
class MusicConfig:
    arm: str = 'flow'
    width: int = 256
    frames: int = 960
    dilations: tuple = (1,2,4,8,16,32,64,112)
    steps: int = 32

    def __post_init__(self):
        if self.arm not in ('regression','flow'):
            raise ValueError('Unknown music model')


class ConditionalBlock(nn.Module):
    def __init__(self,width,dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.conv2 = nn.Conv1d(width,width,3,padding=dilation,dilation=dilation)
        self.time = nn.Linear(width,width)

    def forward(self,x,time):
        y = self.conv1(F.silu(x+self.time(time)[:,:,None]))
        return x+.2*self.conv2(F.silu(y))


class MusicModel(nn.Module):
    def __init__(self,config,normalizers):
        super().__init__()
        self.config = config
        width = config.width
        self.register_buffer('mean',torch.tensor(normalizers['mean'],dtype=torch.float32))
        self.register_buffer('std',torch.tensor(normalizers['std'],dtype=torch.float32))
        self.register_buffer('delta_scale',torch.tensor(normalizers['delta_scale'],dtype=torch.float32))
        self.register_buffer('frequencies',torch.exp(torch.linspace(0,math.log(1000),16)))
        grid = torch.arange(-12,13,dtype=torch.float32)
        kernel = torch.exp(-.5*(grid/3).square())
        # Unit pointwise variance, with stationary random padding at track ends.
        self.register_buffer('noise_kernel',(kernel/kernel.square().sum().sqrt())[None,None])
        self.time = nn.Sequential(nn.Linear(32,width),nn.SiLU(),nn.Linear(width,width))
        self.input = nn.Conv1d(92+139,width,1)
        self.blocks = nn.ModuleList(ConditionalBlock(width,d) for d in config.dilations)
        self.output = nn.Conv1d(width,139,1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.codec = NativeCodec()

    @property
    def halo(self):
        return 2*sum(self.config.dilations)

    def vector_field(self,audio,state,t):
        phase = t[:,None]*self.frequencies[None]
        time = self.time(torch.cat((phase.sin(),phase.cos()),dim=-1))
        x = self.input(torch.cat((audio,state),dim=-1).transpose(1,2))
        for block in self.blocks:
            x = block(x,time)
        return self.output(F.silu(x)).transpose(1,2).float()

    def vector_field_chunked(self,audio,state,t,core_frames=960):
        result = []
        for start in range(0,audio.shape[1],core_frames):
            end = min(start+core_frames,audio.shape[1])
            lo,hi = max(0,start-self.halo),min(audio.shape[1],end+self.halo)
            value = self.vector_field(audio[:,lo:hi],state[:,lo:hi],t)
            result.append(value[:,start-lo:end-lo])
        return torch.cat(result,dim=1)

    def noise(self,batch,frames,generator=None,device=None):
        device = self.mean.device if device is None else device
        # A single stationary field is generated before any chunking. Native
        # channels remain independent; temporal correlation is fixed, not fitted.
        raw = torch.randn(batch*139,1,frames+24,device=device,generator=generator)
        return F.conv1d(raw,self.noise_kernel).reshape(batch,139,frames).transpose(1,2)

    def draws(self,audio):
        return self.noise(audio.shape[0],audio.shape[1]),torch.rand(audio.shape[0],device=audio.device)

    def losses_from_draws(self,audio,target,unused,noise,time):
        target = (target.float()-self.mean)/self.std
        if self.config.arm=='flow':
            state = (1-time[:,None,None])*noise+time[:,None,None]*target
            desired = target-noise
        else:
            state = torch.zeros_like(target)
            time = torch.ones_like(time)
            desired = target
        prediction = self.vector_field(audio,state,time)
        with torch.autocast(device_type=audio.device.type,enabled=False):
            error = prediction.float()-desired.float()
            native = error.square().mean()
            # A fixed quadratic temporal metric retains the conditional-mean
            # optimum of flow matching. It matches real derivatives, not zero.
            velocity = (torch.diff(error,dim=1)/self.delta_scale).square().mean()
            total = native+.05*velocity
        return {'total':total,'native':native,'velocity':velocity}

    @torch.inference_mode()
    def generate(self,audio,seed=0,steps=None,chunked=True,initial_noise=None):
        fn = self.vector_field_chunked if chunked else self.vector_field
        if self.config.arm=='regression':
            state = audio.new_zeros((*audio.shape[:2],139))
            normalized = fn(audio,state,audio.new_ones(audio.shape[0]))
        else:
            generator = torch.Generator(device=audio.device).manual_seed(seed)
            normalized = self.noise(*audio.shape[:2],generator=generator) if initial_noise is None else initial_noise.clone()
            steps = self.config.steps if steps is None else steps
            if steps<1:
                raise ValueError('At least one ODE step')
            # Every step updates ONE full-sequence state. No independent chunk
            # noise, restarts, trajectory blending or root integration occurs here.
            for index in range(steps):
                time = audio.new_full((audio.shape[0],),index/steps)
                normalized = normalized+fn(audio,normalized,time)/steps
        return self.codec.decode(normalized.float()*self.std+self.mean)

    def config_dict(self):
        return {**asdict(self.config),'motion_channels':139,'motion_fps':30,'receptive_field_frames':2*self.halo+1,
                'noise':'stationary Gaussian temporal kernel sigma3frames, support25, unit pointwise variance; one global field',
                'loss':'normalized native MSE + .05 TRAIN-scaled derivative MSE',
                'decoder':self.codec.config_dict(),'inference':'FP32 Euler, full state updated at every step; exact receptive-field halo'}
