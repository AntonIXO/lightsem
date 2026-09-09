"""Shared Conformer, supervised A/B, and a joint temporal C flow."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .stage11_models import SceneFlowModel, SceneFlowConfig

def model_config():
    return SceneFlowConfig(in_dim=92,scene_dim=17,d_model=192,n_blocks=3,n_heads=4,
                           attn_window=160,conv_kernel=15,dropout=.1)

class BeatConditionedModel(SceneFlowModel):
    def __init__(self):
        super().__init__(config=model_config())
        self.a_head=nn.Linear(192,4)
        self.b_head=nn.Linear(192,3)

    def losses(self,x,y,b,pos_weight,*,generator,beat_dropout=.25):
        # Drop the entire five-channel clock block for a quarter of crops.
        # Normalized zero is TRAIN mean; pulse clarity remains in audio.
        keep=(torch.rand((len(x),1,1),device=x.device,generator=generator)>=beat_dropout)
        x=torch.cat((x[:,:,:87],x[:,:,87:]*keep),dim=2)
        h=self.encode_audio(x)
        a=F.mse_loss(self.a_head(h),y[:,:,:4])
        events=F.binary_cross_entropy_with_logits(self.b_head(h),b,pos_weight=pos_weight)
        z=torch.randn(y[:,:,4:].shape,device=x.device,generator=generator)
        t=torch.rand((len(x),1,1),device=x.device,generator=generator)
        target=y[:,:,4:]-z
        velocity=self.velocity_from_h(h,(1-t)*z+t*y[:,:,4:],t)
        flow=F.mse_loss(velocity,target)
        return {'a':a,'b':events,'c_flow':flow,'total':a+.3*events+flow}

    @torch.no_grad()
    def generate(self,x,noise,steps=8,chunk_size=640):
        if self.audio_in.weight.device.type!='cuda':raise RuntimeError('GPU inference required')
        self.eval();x=x.cuda();h=self.encode_audio_chunked(x,chunk_size,output_device='cuda')
        z=noise.cuda();radius=self.cfg.velocity_radius
        for step in range(steps):
            next_z=torch.empty_like(z)
            for start in range(0,x.shape[1],chunk_size):
                end=min(start+chunk_size,x.shape[1]);lo=max(0,start-radius);hi=min(x.shape[1],end+radius)
                v=self.velocity_from_h(h[:,lo:hi],z[:,lo:hi],step/steps)
                next_z[:,start:end]=z[:,start:end]+v[:,start-lo:end-lo]/steps
            z=next_z
        return torch.cat((self.a_head(h),z),2),torch.sigmoid(self.b_head(h))

def noise_for(length,seed=101):
    return torch.randn((1,length,17),generator=torch.Generator(device='cpu').manual_seed(seed))
