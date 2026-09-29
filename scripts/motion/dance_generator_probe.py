import json, hashlib
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.motion_study.music_model import MusicModel,MusicConfig
from lightsem.experiments.motion_study.music_data import motion_summary
from lightsem.experiments.motion_study.music_evaluation import physical
from lightsem.experiments.motion_study.geometry import joints_from_features

torch.set_num_threads(4)
root=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
out=Path('/root/lightsem/experiments/2026-09-10-motion/outputs/loose/dance_generator_probe.json')
manifest=json.loads((root/'data/audio_manifest.json').read_text())
norms=json.loads((root/'data/motion_normalizers.json').read_text())
an=json.loads((root/'data/audio_normalizers.json').read_text())
results={'scope':'unpreregistered CPU diagnostic; 4 sorted TRAIN and 4 sorted VAL records, first 960 frames; not corpus conditioning score', 'rows':[]}
def load(e):
    with np.load(e['audio_path']) as f: x=(f['x'][:960]-np.asarray(an['mean'],np.float32))/np.asarray(an['std'],np.float32)
    with np.load(e['path']) as f: y=f['features'][:960]
    return torch.from_numpy(x)[None],torch.from_numpy(y)[None]
def metric(m,error):
    native=error.square().mean()
    velocity=(torch.diff(error,dim=1)/m.delta_scale).square().mean()
    return {'native':float(native),'velocity':float(velocity),'total':float(native+.05*velocity)}
with torch.inference_mode():
  for arm in ['flow','regression']:
    path=root/f'music/runs/{arm}/seed_0/final.pt'
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=MusicModel(MusicConfig(arm=arm),norms).eval();m.load_state_dict(ck['model'])
    print('loaded',arm,flush=True)
    for partition in ['train','val']:
      entries=sorted([e for e in manifest['entries'] if e['partition']==partition and e['n_frames']>=960],key=lambda e:e['key'])[:4]
      for i,e in enumerate(entries):
        x,y=load(e); donor=next(other for other in entries if other['canonical_id']!=e['canonical_id']); xd,_=load(donor)
        target=(y-m.mean)/m.std
        noise=m.noise(1,960,generator=torch.Generator().manual_seed(811+i))
        for t in ([0.,.1,.5,.9,1.] if arm=='flow' else [1.]):
          tm=torch.tensor([t]); state=(1-t)*noise+t*target if arm=='flow' else torch.zeros_like(target)
          desired=target-noise if arm=='flow' else target
          preds={c:m.vector_field(a,state,tm) for c,a in [('correct',x),('donor',xd),('zero',torch.zeros_like(x))]}
          row={'arm':arm,'partition':partition,'key':e['key'],'t':t,'errors':{c:metric(m,v-desired) for c,v in preds.items()},
            'audio_change_rms':float((x-xd).square().mean().sqrt()),
            'prediction_change_rms':float((preds['correct']-preds['donor']).square().mean().sqrt()),
            'prediction_rms':float(preds['correct'].square().mean().sqrt())}
          results['rows'].append(row)
        print(arm,partition,e['key'],flush=True)
    if arm=='flow':
      # Exact quadratic-loss counterexample, no fitted model involved.
      e=next(e for e in manifest['entries'] if e['partition']=='train' and e['n_frames']>=960)
      _,y=load(e); biased=y.clone();biased[...,0]+=.1
      errors=(biased-y)/m.std
      p0,_=physical(y);p1,_=physical(biased)
      rot=y.clone();rot[...,3:135]*=2
      results['counterexamples']={'key':e['key'],
        'constant_root_vx_plus_0.1_m_s':{'coordinate_loss':metric(m,errors),'last_root_displacement_m':float((joints_from_features(biased)-joints_from_features(y))[0,-1,0,0]),'source_physical':p0,'biased_physical':p1},
        'rotation6d_scale2_same_geometry':{'coordinate_loss':metric(m,(rot-y)/m.std),'joint_max_abs_m':float((joints_from_features(rot)-joints_from_features(y)).abs().max())}}
      results['noise_scales']={'std_quantiles':torch.quantile(m.std,torch.tensor([0.,.1,.5,.9,1.])).tolist(),
        'delta_scale_quantiles':torch.quantile(m.delta_scale,torch.tensor([0.,.1,.5,.9,1.])).tolist(),
        'zero_output_noise_loss':metric(m,noise),'zero_output_target_loss':metric(m,(y-m.mean)/m.std)}
    out.write_text(json.dumps(results,indent=2))
print('DONE',out,flush=True)
