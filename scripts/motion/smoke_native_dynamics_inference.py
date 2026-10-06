"""Engineering-only device/solver smoke; no real-song scores or demo outputs."""
from pathlib import Path
import argparse,hashlib
import torch
from lightsem.experiments.native_dynamics.data import ROOT,write
from lightsem.experiments.native_dynamics.evaluation import setup,load_final,cpu_noise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--guarded',action='store_true');a=p.parse_args()
    if not a.guarded:raise ValueError('GPU guard required')
    from scripts.motion.native_dynamics_guard import authenticate_guarded_job
    authenticate_guarded_job(Path(__file__));setup()
    model,norms,identity=load_final(a.root,'joint',0,'cpu');expected=model.flow.noise(1,64,generator=torch.Generator(device='cpu').manual_seed(7001),device='cpu')
    model=model.to('cuda');noise=cpu_noise(model.flow,64,7001);assert torch.equal(noise,expected)
    audio=torch.zeros(1,64,1153,device='cuda');coarse=torch.zeros(1,1024,1061,device='cuda');valid=torch.zeros(1,1024,device='cuda',dtype=torch.bool);valid[:,:5]=True
    native,plan=model.generate(audio,coarse,valid,torch.arange(64,device='cuda')[None],noise.to('cuda'))
    assert native.shape==(1,64,139) and torch.isfinite(native).all()
    assert torch.isfinite(plan['conditioning']).all()
    write(a.root/'preflight/inference_gpu_v4/SMOKE.json',{'passed':True,'same_CPU_noise_with_CPU_and_GPU_model_kernel':True,'noise_sha256':hashlib.sha256(noise.numpy().tobytes()).hexdigest(),'shape':list(native.shape),'checkpoint':identity,'scope':'Synthetic zero-feature device/solver engineering test only; no real-song outcomes or replacement dance persisted.'})
    print('INFERENCE_DEVICE_SMOKE_PASSED')
