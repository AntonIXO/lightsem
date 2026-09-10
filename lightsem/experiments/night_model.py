"""Matched audio feature x conditioning factorial; A/B/C output schema retained."""
import torch
from torch import nn
from torch.nn import functional as F
from .beat_conditioned_model import BeatConditionedModel

class NightModel(BeatConditionedModel):
    def __init__(self, family='base', architecture='once', aux_dim=0):
        if family not in ('base','fourier','muq') or architecture not in ('once','film'):
            raise ValueError('Unknown factorial cell')
        super().__init__()  # all incumbent modules have the same initialization by seed
        self.family=family;self.architecture=architecture;self.aux_dim=int(aux_dim)
        # Keep optional-module initialization from changing the shared training RNG stream.
        with torch.random.fork_rng(devices=[]):
            initial_seed=torch.initial_seed()
            torch.default_generator.manual_seed(271828+initial_seed)
            self.aux=nn.Sequential(nn.Linear(aux_dim,64),nn.SiLU(),nn.Linear(64,192,bias=False)) if aux_dim else None
            torch.default_generator.manual_seed(314159+initial_seed)
            self.film=nn.ModuleList([nn.Sequential(nn.Linear(192,32),nn.SiLU(),nn.Linear(32,384))
                                    for _ in self.denoiser]) if architecture=='film' else None
            if self.film is not None:
                for layer in self.film:
                    nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)

    def config_dict(self):
        return {**super().config_dict(),'family':self.family,'architecture':self.architecture,'aux_dim':self.aux_dim}

    def encode_audio(self,x):
        if x.ndim!=3 or x.shape[-1]!=92+self.aux_dim:raise ValueError('Wrong audio schema')
        h=self.audio_in(x[:,:,:92])
        if self.aux is not None:h=h+self.aux(x[:,:,92:])
        for block in self.encoder:h=block(h)
        return h

    def velocity_from_h(self,h,z,t):
        x=self.flow_in(torch.cat((h,z,self._flow_time(t,z)),dim=-1))
        for i,block in enumerate(self.denoiser):
            if self.film is None:x=block(x)
            else:
                gamma,beta=self.film[i](h).chunk(2,dim=-1)
                y=block.norm(x)*(1+gamma)+beta
                y=block.net(y.transpose(1,2)).transpose(1,2)
                x=x+F.silu(y)
        return self.flow_out(x)

    def encode_audio_chunked(self,x,chunk_size=640,output_device='cuda'):
        # Existing exact encoder context; input width may exceed the old config.in_dim.
        if self.audio_in.weight.device.type!='cuda':raise RuntimeError('GPU inference required')
        radius=self.cfg.encoder_radius;out=torch.empty((*x.shape[:2],192),device=output_device,dtype=x.dtype)
        for start in range(0,x.shape[1],chunk_size):
            end=min(start+chunk_size,x.shape[1]);lo=max(0,start-radius);hi=min(x.shape[1],end+radius)
            h=self.encode_audio(x[:,lo:hi].cuda());out[:,start:end]=h[:,start-lo:end-lo].to(output_device)
        return out

    def loss_inputs(self,x,y,generator,beat_dropout=.25):
        keep=(torch.rand((len(x),1,1),device=x.device,generator=generator)>=beat_dropout)
        # The five EXPLICIT clock channels, not every column after 87.
        x=torch.cat((x[:,:,:87],x[:,:,87:92]*keep,x[:,:,92:]),dim=2)
        z=torch.randn(y[:,:,4:].shape,device=x.device,generator=generator)
        t=torch.rand((len(x),1,1),device=x.device,generator=generator)
        return x,z,t

    def losses_from_draws(self,x,y,b,pos_weight,z,t):
        h=self.encode_audio(x);a=F.mse_loss(self.a_head(h),y[:,:,:4])
        events=F.binary_cross_entropy_with_logits(self.b_head(h),b,pos_weight=pos_weight)
        flow=F.mse_loss(self.velocity_from_h(h,(1-t)*z+t*y[:,:,4:],t),y[:,:,4:]-z)
        return {'a':a,'b':events,'c_flow':flow,'total':a+.3*events+flow}

    def losses(self,x,y,b,pos_weight,*,generator,beat_dropout=.25):
        x,z,t=self.loss_inputs(x,y,generator,beat_dropout)
        return self.losses_from_draws(x,y,b,pos_weight,z,t)
