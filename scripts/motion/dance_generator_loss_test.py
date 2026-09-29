import json
from pathlib import Path
import torch
from lightsem.experiments.motion_study.geometry import joints_from_features,FEET
from lightsem.experiments.motion_study.music_model import MusicModel,MusicConfig
torch.set_num_threads(4)
root=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
norms=json.loads((root/'data/motion_normalizers.json').read_text())
m=MusicModel(MusicConfig(),norms)
y=torch.zeros(1,60,139);y[...,2]=1.;y[...,3:135]=torch.tensor([1.,0,0,0,1.,0]).repeat(22);y[...,135:]=1
ref=joints_from_features(y).detach()
bias=torch.tensor(.1,requires_grad=True)
pred=y+torch.nn.functional.pad(bias.expand(1,60,1),(0,138))
feet=joints_from_features(pred)[...,FEET,:]
mask=(y[:,1:,135:]*y[:,:-1,135:]).detach()
foot_speed_sq=torch.diff(feet,dim=1).square().sum(-1)*30**2
support=(foot_speed_sq*mask).sum()/mask.sum().clamp_min(1)
support.backward()
assert abs(float(support)-.01)<1e-6
assert abs(float(bias.grad)-.2)<1e-5
zero=(torch.diff(ref[...,FEET,:],dim=1).square().sum(-1)*30**2).mean()
assert float(zero)==0.
n=(y-m.mean)/m.std;noise=m.noise(1,60,generator=torch.Generator().manual_seed(811))
roundtrips=[]
for t in [0.,.5,.99,1.]:
 state=(1-t)*noise+t*n;v=n-noise
 clean=state+(1-t)*v
 error=float((clean-n).abs().max()); assert error<1e-5
 roundtrips.append({'t':t,'clean_normalized_max_abs':error})
result={'support_mse_static':float(zero),'support_mse_slide_0.1_m_s':float(support.detach()),'gradient_wrt_vx':float(bias.grad),'clean_estimate_roundtrips':roundtrips,'scope':'synthetic CPU unit tests for proposed loss path, not model training or validation'}
Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/dance_generator_loss_test.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
