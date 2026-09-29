"""Restore the authenticated 92-channel music bus and 40 Hz A/B/C targets."""
from collections import Counter
from pathlib import Path
import argparse
import numpy as np
from scipy.ndimage import maximum_filter1d
from ..dance_no_clock.dance_no_clock_data import hierarchical_weights, FEATURE_NAMES
from ..stage11.stage11_training import A_NAMES, B_NAMES, C_NAMES
from ..stage11.stage11_common import authenticated_json, read_authenticated, digest

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/beat_conditioned_20260909')
SOURCE=Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908')
NAMES=list(A_NAMES+C_NAMES)
X_NAMES=list(FEATURE_NAMES)+['pulse_clarity','beat_phase_sin','beat_phase_cos','beat_accent','bar_phase','beat_period_s']

def event_targets(b):
    return maximum_filter1d(np.asarray(b,np.float32),size=5,axis=0,mode='constant')

def prepare(root=ROOT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/'data_manifest.json').exists():
        return read_authenticated(root/'data_manifest.json')
    old=read_authenticated(SOURCE/'training_data.json')
    source_norms=read_authenticated(SOURCE/'normalizers.json')['motion']
    assert source_norms['a_names']+source_norms['c_names']==NAMES
    entries=[dict(e,fps=40.) for e in old['motion'] if e['partition'] in ('train','val')]
    train=[e for e in entries if e['partition']=='train']
    songs={p:{e['canonical_id'] for e in entries if e['partition']==p} for p in ('train','val')}
    assert not songs['train']&songs['val']
    weights=hierarchical_weights(train)
    xm=np.zeros(92);x2=xm.copy();ym=np.zeros(21);y2=ym.copy();bm=np.zeros(3)
    for e in entries:
        if digest(e['path'])!=e['sha256']:raise RuntimeError('Tensor drift: '+e['key'])
        if e['partition']!='train':continue
        with np.load(e['path'],allow_pickle=False) as z:
            x,y,b=z['x'].astype(float),z['y'].astype(float),event_targets(z['b'])
        if x.shape!=(e['n_frames'],92) or y.shape!=(len(x),21) or b.shape!=(len(x),3):raise ValueError('Schema')
        if not all(np.isfinite(v).all() for v in (x,y,b)):raise ValueError('Nonfinite tensor')
        w=weights[e['key']];xm+=w*x.mean(0);x2+=w*np.square(x).mean(0)
        ym+=w*y.mean(0);y2+=w*np.square(y).mean(0);bm+=w*b.mean(0)
    norms={'feature_names':X_NAMES,'target_names':NAMES,'event_names':list(B_NAMES),
           'x_mean':xm.tolist(),'x_std':np.sqrt(np.maximum(x2-xm*xm,1e-12)).tolist(),
           'y_center':ym.tolist(),'y_scale':np.sqrt(np.maximum(y2-ym*ym,1e-6)).tolist(),
           'event_prevalence':bm.tolist(),'pos_weight':((1-bm)/np.maximum(bm,1e-6)).tolist(),
           'event_tolerance_frames':2,'fit_partition':'train',
           'weighting':'equal canonical song / recording / coach / frame'}
    authenticated_json(root/'normalizers.json',norms)
    manifest={'entries':entries,'normalizers_sha256':digest(root/'normalizers.json'),
              'feature_names':X_NAMES,'target_names':NAMES,'event_names':list(B_NAMES),
              'source_index_sha256':digest(SOURCE/'training_data.json'),
              'counts':{p:{'songs':len(songs[p]),'coaches':sum(e['partition']==p for e in entries),
                          'frames':sum(e['n_frames'] for e in entries if e['partition']==p)} for p in songs},
              'fps':40.,'test_tensors_read':False,'beat_inputs':True,'target_decimation':False}
    authenticated_json(root/'data_manifest.json',manifest)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    print(prepare(p.parse_args().root)['counts'])
