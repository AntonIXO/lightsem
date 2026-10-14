"""Native learned dynamics: data-supervised planning inside the motion flow.

No playback clock, trajectory recomposition, source-motion input or tempo rule.
Just Dance gradients act on generated samples; only paired 3D data get CFM.
"""
from dataclasses import dataclass,asdict
import torch
from torch import nn
import torch.nn.functional as F
from ..dynamic_dance.learning import DynamicFlow,DynamicFlowConfig
from ..motion_study.music_model import MusicConfig
from .kinematics import PLAN_DIM,masked_mean
from .local_audio_fusion import LocalAudioFusion,masked_energy_quantile
MOTIF_DIM=32
CONDITION_DIM=PLAN_DIM+MOTIF_DIM

@dataclass(frozen=True)
class Config:
    arm:str='joint'
    extra_dim:int=1061
    stereo_dim:int=0
    planner_width:int=96
    planner_heads:int=4
    frames:int=480
    steps:int=32
    gradient_steps:int=4
    freeze_early_blocks:int=6
    adapter_scale:float=.2
    stereo_adapter_scale:float|None=None
    stereo_plan_scale:float|None=None
    quiet_adapter:bool=False
    quiet_adapter_scale:float=.2
    quiet_energy_gate:bool=False
    quiet_energy_threshold:float=0.0
    quiet_energy_temperature:float=.5
    stereo_fusion:str='concat'
    audio_fusion:str='none'
    audio_cross_width:int=64
    audio_cross_heads:int=4
    audio_cross_scale:float=.2
    audio_energy_gate:bool=False
    audio_energy_threshold:float=0.0
    audio_energy_temperature:float=.5
    audio_energy_relative:bool=False
    audio_energy_quantile:float=.3
    audio_energy_smooth_frames:int=1
    temporal_adapter:bool=False
    temporal_adapter_scale:float=.1
    temporal_adapter_hidden:int=64
    temporal_energy_gate:bool=False
    temporal_energy_threshold:float=0.0
    temporal_energy_temperature:float=.5
    noise_energy_gate:bool=False
    noise_energy_threshold:float=0.0
    noise_energy_temperature:float=.5
    noise_energy_max_reduction:float=.5
    def __post_init__(self):
        if self.arm not in ('continued','dynamics','joint'):raise ValueError('Unknown registered arm')
        if self.steps!=32 or not 1<=self.gradient_steps<=self.steps:raise ValueError('Declared32-step native solver required')
        if self.extra_dim<1 or self.stereo_dim<0 or self.adapter_scale<=0 or (self.stereo_adapter_scale is not None and self.stereo_adapter_scale<=0) or (self.stereo_plan_scale is not None and self.stereo_plan_scale<=0):raise ValueError('Invalid audio dimensions/scales')
        if self.stereo_fusion not in ('concat','gated_residual','role_specialized','dual_cross_attention'):raise ValueError('Unknown stereo fusion')
        if self.stereo_fusion!='concat' and self.stereo_dim!=128:raise ValueError('Learned fusion requires concat128 context')
        if self.quiet_adapter_scale<=0 or self.quiet_adapter_scale>1:raise ValueError('Invalid quiet adapter scale')
        if self.quiet_energy_temperature<=0:raise ValueError('Invalid quiet energy temperature')
        if self.audio_fusion not in ('none','attention','film','hybrid'):raise ValueError('Unknown local audio fusion')
        if self.audio_cross_width<8 or self.audio_cross_heads<1 or self.audio_cross_width%self.audio_cross_heads!=0:raise ValueError('Invalid audio cross-attention dimensions')
        if self.audio_cross_scale<=0 or self.audio_cross_scale>1:raise ValueError('Invalid audio cross-attention scale')
        if self.audio_energy_temperature<=0:raise ValueError('Invalid audio energy temperature')
        if not 0.0<self.audio_energy_quantile<1.0:raise ValueError('Invalid audio energy quantile')
        if self.audio_energy_smooth_frames<1 or self.audio_energy_smooth_frames%2==0:raise ValueError('Energy smoothing width must be odd and positive')
        if self.temporal_adapter_scale<=0 or self.temporal_adapter_scale>1:raise ValueError('Invalid temporal adapter scale')
        if self.temporal_adapter_hidden<4:raise ValueError('Temporal adapter hidden width must be >=4')
        if self.temporal_energy_gate and self.extra_dim<=1040:raise ValueError('Temporal energy gate requires the native mixture-RMS channel')
        if self.temporal_energy_temperature<=0:raise ValueError('Invalid temporal energy temperature')
        if self.noise_energy_temperature<=0 or self.noise_energy_max_reduction<=0 or self.noise_energy_max_reduction>1:raise ValueError('Invalid noise energy gate')
        if self.noise_energy_gate and self.extra_dim<=1040:raise ValueError('Noise energy gate requires the native mixture-RMS channel')


class Planner(nn.Module):
    def __init__(self,dimension,width=96,heads=4):
        super().__init__();self.input=nn.Linear(dimension,width)
        self.blocks=nn.ModuleList(nn.Conv1d(width,width,3,padding=d,dilation=d) for d in (1,2,4,8))
        self.norm=nn.LayerNorm(width);self.attention=nn.MultiheadAttention(width,heads,batch_first=True)
        self.mix=nn.Linear(width,width);self.output=nn.Linear(width,PLAN_DIM);self.recurrence=nn.Linear(width,32)
    def forward(self,audio,valid):
        x=F.silu(self.input(torch.where(valid[...,None],audio,0.)))
        x=torch.where(valid[...,None],x,0.)
        for block in self.blocks:x=torch.where(valid[...,None],x+.2*block(F.silu(x).transpose(1,2)).transpose(1,2),0.)
        q=self.norm(x);vm=valid.clone();vm[:,0]=True
        context,_=self.attention(q,q,q,key_padding_mask=~vm,need_weights=False)
        h=torch.where(valid[...,None],x+.2*self.mix(context),0.)
        raw=self.output(h).float();u=raw.reshape(*raw.shape[:-1],6,8)
        quantiles=F.softplus(u[...,:3]).cumsum(-1)
        packed=torch.cat((quantiles,u[...,3:4].sigmoid(),u[...,4:].softmax(-1)),-1).flatten(-2)
        packed=torch.where(valid[...,None],packed,0.)
        embedding=torch.where(valid[...,None],F.normalize(self.recurrence(h).float(),dim=-1,eps=1e-6),0.)
        return {'raw':raw,'packed':packed,'embedding':embedding,'conditioning':torch.cat((packed,embedding),-1)}


def plan_at_frames(plan,coarse_valid,frame_indices):
    position=(frame_indices.float()-7.)/15.;lo0=position.floor().long();hi0=lo0+1;weight=(position-lo0.float()).clamp(0,1)
    last=coarse_valid.sum(1).clamp_min(1)-1
    lo=torch.minimum(lo0.clamp_min(0),last[:,None]);hi=torch.minimum(hi0.clamp_min(0),last[:,None])
    a=plan.gather(1,lo[...,None].expand(-1,-1,plan.shape[-1]));b=plan.gather(1,hi[...,None].expand(-1,-1,plan.shape[-1]))
    return a+(b-a)*weight[...,None]


class Flow(DynamicFlow):
    def __init__(self,base_config,normalizers,config):
        super().__init__(base_config,normalizers,DynamicFlowConfig(config.extra_dim,arm='D7'))
        self.native_config=config
        self.native_projection=nn.Linear(CONDITION_DIM,32)
        self.native_adapters=nn.ModuleList(nn.Linear(32,base_config.width) for _ in self.blocks)
        for layer in self.native_adapters:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.stereo_projection=None
        self.stereo_adapters=nn.ModuleList()
        self.stereo_gate=None
        self.stereo_cross_fast=None
        self.stereo_cross_slow=None
        self.quiet_projection=None
        self.quiet_adapters=nn.ModuleList()
        self.local_audio=None
        self.temporal_adapter=None
        if config.stereo_dim:
            projection_dim=config.stereo_dim
            if config.stereo_fusion=='gated_residual':
                self.stereo_gate=nn.Sequential(nn.Linear(config.stereo_dim,64),nn.SiLU(),nn.Linear(64,64))
                nn.init.zeros_(self.stereo_gate[-1].weight);nn.init.zeros_(self.stereo_gate[-1].bias)
                projection_dim=64
            elif config.stereo_fusion=='role_specialized':
                projection_dim=64
            elif config.stereo_fusion=='dual_cross_attention':
                # Keep the two learned streams separate, then let each stream
                # attend to the other before the shared adapters.  Zeroing
                # the attention output projections makes the new branch an
                # exact concat control at initialization.
                self.stereo_cross_fast=nn.MultiheadAttention(64,4,batch_first=True)
                self.stereo_cross_slow=nn.MultiheadAttention(64,4,batch_first=True)
                for attention in (self.stereo_cross_fast,self.stereo_cross_slow):
                    nn.init.zeros_(attention.out_proj.weight)
                    nn.init.zeros_(attention.out_proj.bias)
                projection_dim=128
            self.stereo_projection=nn.Linear(projection_dim,32)
            self.stereo_adapters=nn.ModuleList(nn.Linear(32,base_config.width) for _ in self.blocks)
            for layer in self.stereo_adapters:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        if config.quiet_adapter:
            self.quiet_projection=nn.Linear(config.extra_dim,32)
            self.quiet_adapters=nn.ModuleList(nn.Linear(32,base_config.width) for _ in self.blocks)
            for layer in self.quiet_adapters:
                nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        if config.audio_fusion!='none':
            self.local_audio=LocalAudioFusion(config.extra_dim,base_config.width,
                config.audio_cross_width,config.audio_cross_heads,config.audio_fusion,
                energy_gate=config.audio_energy_gate,
                energy_threshold=config.audio_energy_threshold,
                energy_temperature=config.audio_energy_temperature,
                energy_relative=config.audio_energy_relative,
                energy_quantile=config.audio_energy_quantile,
                energy_smooth_frames=config.audio_energy_smooth_frames)
        if config.temporal_adapter:
            # A channel-mixing temporal residual gives the Fourier/jerk
            # objective enough capacity to separate articulated modes. Only
            # the final projection is zero-init, preserving the incumbent
            # exactly before training; this is native vector-field behavior,
            # never post-hoc body smoothing.
            hidden=min(config.temporal_adapter_hidden,base_config.width)
            self.temporal_adapter=nn.Sequential(
                nn.Conv1d(base_config.width,hidden,1),nn.SiLU(),
                nn.Conv1d(hidden,hidden,9,padding=4,groups=hidden),nn.SiLU(),
                nn.Conv1d(hidden,base_config.width,1))
            nn.init.zeros_(self.temporal_adapter[-1].weight);nn.init.zeros_(self.temporal_adapter[-1].bias)
    def flow_stereo(self,value):
        if self.native_config.stereo_fusion=='gated_residual':
            fast, slow = value[...,:64], value[...,64:]
            return fast + torch.tanh(self.stereo_gate(value))*slow
        if self.native_config.stereo_fusion=='role_specialized':
            return value[...,:64]
        if self.native_config.stereo_fusion=='dual_cross_attention':
            fast,slow=value[...,:64],value[...,64:]
            fast_from_slow,_=self.stereo_cross_fast(fast,slow,slow,need_weights=False)
            slow_from_fast,_=self.stereo_cross_slow(slow,fast,fast,need_weights=False)
            return torch.cat((fast+fast_from_slow,slow+slow_from_fast),-1)
        return value
    def planner_stereo(self,value):
        if self.native_config.stereo_fusion=='role_specialized':
            return value[...,64:]
        return self.flow_stereo(value)
    @property
    def halo(self):
        # The 9-tap depthwise kernel adds four frames on each side of the incumbent
        # receptive field. Chunked inference must include the same neighbors.
        return super().halo + (4 if self.native_config.temporal_adapter else 0)
    @property
    def conditioning_dim(self):return 92+self.dynamic.extra_dim+CONDITION_DIM+self.native_config.stereo_dim
    def prepare_conditioning(self,audio):
        if audio.ndim!=3 or audio.shape[-1]!=self.conditioning_dim:raise ValueError('Wrong native conditioning width')
        extra=audio[...,92:92+self.dynamic.extra_dim]
        conditioning=F.silu(self.audio_projection(extra))
        native_start=92+self.dynamic.extra_dim
        native=F.silu(self.native_projection(audio[...,native_start:native_start+CONDITION_DIM])) if self.native_config.arm=='joint' else None
        stereo_start=native_start+CONDITION_DIM
        stereo=audio[...,stereo_start:stereo_start+self.native_config.stereo_dim] if self.native_config.stereo_dim else None
        biases=[]
        for index,(adapter,native_adapter) in enumerate(zip(self.adapters,self.native_adapters)):
            bias=self.dynamic.adapter_scale*torch.tanh(adapter(conditioning))
            if native is not None:bias=bias+self.native_config.adapter_scale*torch.tanh(native_adapter(native))
            if stereo is not None:
                if stereo.shape[-1]!=self.native_config.stereo_dim:raise ValueError('Stereo native context width mismatch')
                stereo_hidden=F.silu(self.stereo_projection(self.flow_stereo(stereo)))
                stereo_scale=self.native_config.adapter_scale if self.native_config.stereo_adapter_scale is None else self.native_config.stereo_adapter_scale
                bias=bias+stereo_scale*torch.tanh(self.stereo_adapters[index](stereo_hidden))
            if self.quiet_projection is not None:
                quiet_hidden=F.silu(self.quiet_projection(extra))
                if self.native_config.quiet_energy_gate:
                    energy=extra[...,1040:1041]
                    quiet_gate=torch.sigmoid((self.native_config.quiet_energy_threshold-energy)/self.native_config.quiet_energy_temperature)
                    quiet_hidden=quiet_hidden*quiet_gate
                bias=bias+self.native_config.quiet_adapter_scale*torch.tanh(self.quiet_adapters[index](quiet_hidden))
            biases.append(bias.transpose(1,2))
        return audio[...,:92],torch.stack(biases,1)
    def prepare_audio_context(self,audio,valid=None,*,energy_reference=None):
        if self.local_audio is None:return None
        return self.local_audio.prepare(audio[...,92:92+self.dynamic.extra_dim],valid,energy_reference=energy_reference)
    def prepared_vector_field(self,base,biases,state,t,audio_context=None,temporal_energy=None):
        phase=t[:,None]*self.frequencies[None];time=self.time(torch.cat((phase.sin(),phase.cos()),-1))
        x=self.input(torch.cat((base,state),-1).transpose(1,2))
        for i,block in enumerate(self.blocks):x=block(x+biases[:,i],time)
        if self.temporal_adapter is not None:
            temporal_residual=torch.tanh(self.temporal_adapter(x))
            if self.native_config.temporal_energy_gate:
                if temporal_energy is None:
                    raise ValueError('Temporal energy context required')
                gate=torch.sigmoid((self.native_config.temporal_energy_threshold-temporal_energy)/self.native_config.temporal_energy_temperature)
                temporal_residual=temporal_residual*gate[:,None]
            x=x+self.native_config.temporal_adapter_scale*temporal_residual
        if self.local_audio is not None:
            if audio_context is None:raise ValueError('Local audio context required; silent bypass forbidden')
            x=x+self.native_config.audio_cross_scale*self.local_audio(x.transpose(1,2),audio_context).transpose(1,2)
        return self.output(F.silu(x)).transpose(1,2).float()
    def vector_field(self,audio,state,t):
        return self.prepared_vector_field(*self.prepare_conditioning(audio),state,t,self.prepare_audio_context(audio))

    def config_dict(self):
        return {**super().config_dict(),'native_control_channels':80,'legacy34_plan_projection_unused':True,'conditioning':'Base92+1061frozenaudio+48learneddynamics+32continuousmotif+optionalStereoV1+optionalQuietAdapter+optionalLocalAudioFusion+optionalTemporalAdapter; onlyjointconsumescontrol','stereo_dim':self.native_config.stereo_dim,'stereo_fusion':self.native_config.stereo_fusion,'quiet_adapter':self.native_config.quiet_adapter,'audio_fusion':self.native_config.audio_fusion,'temporal_adapter':self.native_config.temporal_adapter,'audio_context':'fine native audio with +/-60,+/-30,+/-15,+/-5,0 frame offsets; no playback clock','retiming':False}



class NativeDynamicsModel(nn.Module):
    def __init__(self,base_config,normalizers,config=Config()):
        super().__init__();self.config=config;self.flow=Flow(base_config,normalizers,config);self.planner=Planner(config.extra_dim,config.planner_width,config.planner_heads)
        self.noise_energy_scale=nn.Parameter(torch.zeros(())) if config.noise_energy_gate else None
        self.stereo_plan=None
        if config.stereo_dim:
            plan_dim=64 if config.stereo_fusion in ('gated_residual','role_specialized') else config.stereo_dim
            self.stereo_plan=nn.Linear(plan_dim,CONDITION_DIM)
            nn.init.zeros_(self.stereo_plan.weight);nn.init.zeros_(self.stereo_plan.bias)
        # All arms retain identical pretrained early3D filters. Candidatelearns
        # its new adapters at every block, plus shared lateblocks/audioadapters.
        for block in self.flow.blocks[:config.freeze_early_blocks]:
            for p in block.parameters():p.requires_grad_(False)
        for p in self.flow.plan_projection.parameters():p.requires_grad_(False) # unused legacy34-channel slots
        if config.arm!='joint':
            for module in (self.flow.native_projection,self.flow.native_adapters):
                for p in module.parameters():p.requires_grad_(False)
            if self.flow.stereo_projection is not None:
                for module in (self.flow.stereo_projection,self.flow.stereo_adapters):
                    for p in module.parameters():p.requires_grad_(False)
    def warmstart(self,d7_state):
        result=self.flow.load_state_dict(d7_state,strict=False)
        allowed=('native_projection.','native_adapters.')
        if self.config.stereo_dim:allowed=allowed+('stereo_projection.','stereo_adapters.')
        if result.unexpected_keys or any(not k.startswith(allowed) for k in result.missing_keys):raise ValueError('D7 architecture/state mismatch')
    def condition(self,fine_audio,coarse_audio,coarse_valid,frame_indices,stereo_context=None,coarse_stereo=None):
        if fine_audio.shape[-1]!=92+self.config.extra_dim:raise ValueError('Wrong audio schema')
        if self.config.stereo_dim:
            if stereo_context is None or coarse_stereo is None:raise ValueError('Stereo context is required by this model arm')
            if stereo_context.shape[:2]!=fine_audio.shape[:2] or stereo_context.shape[-1]!=self.config.stereo_dim:raise ValueError('Wrong fine stereo schema')
            if coarse_stereo.shape[:2]!=coarse_audio.shape[:2] or coarse_stereo.shape[-1]!=self.config.stereo_dim:raise ValueError('Wrong coarse stereo schema')
            if not torch.isfinite(stereo_context).all() or not torch.isfinite(coarse_stereo).all():raise ValueError('Nonfinite stereo context')
        elif stereo_context is not None or coarse_stereo is not None:
            raise ValueError('Stereo context supplied to a non-stereo model')
        # Observerplanner receives identical supervision in allarms. Only joint
        # connects its outputs to the decoder; this controls label exposure.
        plan=self.planner(coarse_audio,coarse_valid)
        if self.stereo_plan is not None:
            plan_scale=1. if self.config.stereo_plan_scale is None else self.config.stereo_plan_scale
            plan['conditioning']=plan['conditioning']+plan_scale*self.stereo_plan(self.flow.planner_stereo(coarse_stereo))
        values=plan_at_frames(plan['conditioning'],coarse_valid,frame_indices)
        result=torch.cat((fine_audio,values),-1)
        if stereo_context is not None:result=torch.cat((result,stereo_context),-1)
        return result,plan
    def rollout(self,conditioning,noise,*,gradient_steps=None,field=None,teacher=None,audio_valid=None,energy_reference=None):
        """Exact32forwardsteps; gradients only through finaldeclaredsteps.

        The detached prefix uses the CURRENT model, not fabricated3Dtargets.
        Training/evaluation share the forwardsolver; truncation affects only
        derivative estimation. A frozen D7 field is an explicit soft prior.
        """
        steps=self.config.steps;grad_steps=self.config.gradient_steps if gradient_steps is None else gradient_steps
        fn=self.flow.prepared_vector_field if field is None else field;base_audio,biases=self.flow.prepare_conditioning(conditioning);state=self._gate_noise(conditioning,noise);prior=state.sum()*0
        audio_context=self.flow.prepare_audio_context(conditioning,audio_valid,energy_reference=energy_reference)
        temporal_energy=conditioning[...,92+1040] if self.config.temporal_energy_gate else None
        context_kwargs={} if audio_context is None else {'audio_context':audio_context}
        if temporal_energy is not None: context_kwargs['temporal_energy']=temporal_energy
        detached_kwargs={} if audio_context is None else {'audio_context':tuple(x.detach() if torch.is_tensor(x) else None for x in audio_context)}
        if temporal_energy is not None: detached_kwargs['temporal_energy']=temporal_energy.detach()
        teacher_audio=None
        if teacher is not None:
            teacher_audio=torch.cat((conditioning[...,:92+self.config.extra_dim].detach(),conditioning.new_zeros((*conditioning.shape[:2],34))),-1)
        for i in range(steps):
            t=state.new_full((len(state),),i/steps)
            if i<steps-grad_steps:
                with torch.no_grad():state=state+fn(base_audio.detach(),biases.detach(),state,t,**detached_kwargs)/steps
            else:
                velocity=fn(base_audio,biases,state,t,**context_kwargs)
                if teacher is not None:
                    with torch.no_grad():base=teacher.vector_field(teacher_audio,state.detach(),t)
                    prior=prior+(velocity-base).square().mean()/max(grad_steps,1)
                state=state+velocity/steps
        native=self.flow.codec.decode(state.float()*self.flow.std+self.flow.mean)
        return native,prior
    def _gate_noise(self,conditioning,noise):
        if not self.config.noise_energy_gate:
            return noise.clone()
        energy=conditioning[...,92+1040]
        gate=torch.sigmoid((self.config.noise_energy_threshold-energy)/self.config.noise_energy_temperature)
        # The centered sigmoid is exactly zero at the zero-init parameter. It
        # can learn a bounded reduction (or increase) while preserving the
        # fixed noise draw and all active audio timing.
        reduction=self.config.noise_energy_max_reduction*(torch.sigmoid(self.noise_energy_scale)-.5)
        return noise*(1.-reduction*gate[...,None])
    @torch.inference_mode()
    def generate(self,fine_audio,coarse_audio,coarse_valid,frame_indices,noise,*,plan_override=None,core_frames=960,stereo_context=None,coarse_stereo=None,local_context_override=None,energy_reference=None):
        conditioning,plan=self.condition(fine_audio,coarse_audio,coarse_valid,frame_indices,stereo_context=stereo_context,coarse_stereo=coarse_stereo)
        if plan_override is not None:
            if self.config.arm!='joint':raise ValueError('Plan intervention requires active joint architecture')
            values=plan_at_frames(plan_override,coarse_valid,frame_indices)
            conditioning=torch.cat((fine_audio,values),-1)
            if stereo_context is not None:conditioning=torch.cat((conditioning,stereo_context),-1)
        base,biases=self.flow.prepare_conditioning(conditioning);state=self._gate_noise(conditioning,noise);halo=self.flow.halo
        if self.config.audio_energy_relative and energy_reference is None:
            energy_reference=masked_energy_quantile(fine_audio[...,92+1040],torch.ones_like(frame_indices,dtype=torch.bool),self.config.audio_energy_quantile)
        audio_context=self.flow.prepare_audio_context(conditioning,energy_reference=energy_reference)
        if local_context_override is not None:
            if self.flow.local_audio is None:raise ValueError('Local audio intervention on a model without local fusion')
            audio_context=local_context_override
        for i in range(self.config.steps):
            time=state.new_full((len(state),),i/self.config.steps);parts=[]
            for start in range(0,state.shape[1],core_frames):
                end=min(start+core_frames,state.shape[1]);lo=max(0,start-halo);hi=min(state.shape[1],end+halo)
                context_kwargs={} if audio_context is None else {'audio_context':tuple(x[:,lo:hi] if torch.is_tensor(x) else None for x in audio_context)}
                if self.config.temporal_energy_gate:
                    context_kwargs['temporal_energy']=conditioning[:,lo:hi,92+1040]
                velocity=self.flow.prepared_vector_field(base[:,lo:hi],biases[:,:,:,lo:hi],state[:,lo:hi],time,**context_kwargs)
                parts.append(velocity[:,start-lo:end-lo])
            state=state+torch.cat(parts,1)/self.config.steps
        return self.flow.codec.decode(state.float()*self.flow.std+self.flow.mean),plan
    def config_dict(self):
        return {**asdict(self.config),'planner_hz':2,'planning_channels':48,'continuous_motif_channels':32,'total_control_channels':80,'channel_schema':'6regions*(ordered log-speed q10/q50/q90,low-speedfraction,4censored low/high-bout duration probabilities)','condition':'audioonly;motion-derivedtargets used inlosses only; allarms train the same observerplanner, only joint consumes it','whole_articulation_low_speed':'allobserved root-relative joints quiet; lower-bodyquietdoesnot stoparms; roottranslation/contact supervised separately bypaired3D','time_axis':'original30Hz songtime, no retiming/recomposition/floor','weak_supervision':'JD generated32-step outputs, backpropagatefinal4steps; no JD3D CFM targets','flow':self.flow.config_dict()}


def planner_loss(plan,targets,validity,pose_context,pose_valid):
    """Motion-only supervision of the integrated planner; no tempo heuristics."""
    if plan is None:return targets.sum()*0,{}
    raw=plan['raw'].reshape(*targets.shape[:2],6,8);pred=plan['packed'].reshape_as(raw);target=targets.reshape_as(raw)
    quant=masked_mean(F.smooth_l1_loss(pred[...,:3],target[...,:3],reduction='none',beta=.1),validity[...,0,None])
    low=masked_mean(F.binary_cross_entropy_with_logits(raw[...,3],target[...,3],reduction='none'),validity[...,1])
    ce=-(target[...,4:]*raw[...,4:].log_softmax(-1)).sum(-1);duration=masked_mean(ce,validity[...,2])
    # Bounded64sample whole-song recurrence; labels are coachposes, not MuQ.
    idx=torch.linspace(0,targets.shape[1]-1,min(64,targets.shape[1]),device=targets.device).long().unique()
    embedding=plan['embedding'][:,idx];reference=F.normalize(pose_context[:,idx].float(),dim=-1,eps=1e-6);good=pose_valid[:,idx]
    pairs=good[:,:,None]&good[:,None,:]&(torch.abs(idx[:,None]-idx[None,:])>=4)[None]
    recurrence=masked_mean((embedding@embedding.transpose(1,2)-reference@reference.transpose(1,2)).square(),pairs)
    return quant+.25*low+.1*duration+.1*recurrence,{'planner_quantile':quant,'planner_low':low,'planner_duration':duration,'planner_recurrence':recurrence}
