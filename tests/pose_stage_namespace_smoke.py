"""Fresh-process synthetic source-namespace and full mapping smoke. No corpus IO."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

out=Path(sys.argv[1]);repo=Path('/root/lightsem')
spec=importlib.util.spec_from_file_location('eval_synthetic',repo/'scripts/motion/eval_pose_stage_study.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
f=e.loadjson(out/'freeze.json');source=out/'source'
# An old helper has a source activation function. Loading it MUST NOT call it.
helper=e.load_helper(out,'review')
bindings=e.activate_source(source,f['files'])
assert all(Path(v['path']).is_relative_to(source) for v in bindings.values())
assert 'lightsem.experiments.motion_study.pose_stage' in bindings
try:e.activate_source(source,f['files'])
except ValueError:pass
else:raise AssertionError('Mixed namespace was accepted')
if len(sys.argv)>2 and sys.argv[2]=='namespace_only':
    print('NAMESPACE_OK');sys.exit(0)
import torch
from lightsem.experiments.motion_study.geometry import forward_kinematics
from lightsem.experiments.motion_study.lighting_map import LightingConfig,compose_world_joints
from lightsem.conductor.rig import canonical_rig
from lightsem.sim.renderer import RenderConfig
n=600;rot=torch.eye(3,dtype=torch.float64).expand(n,22,3,3)
root=torch.tensor([[0.,.95,0.]],dtype=torch.float64).expand(n,3)
world=forward_kinematics(rot,root).numpy();world_before=world.copy()
plan=json.loads((repo/'experiments/2026-09-10-motion/outputs/loose/pose_stage_study_plan.json').read_text())
curve={'time':np.array([0.,20.]),'momentary':np.array([-20.,-20.]),'short_term':np.array([-20.,-20.])}
baseline,_,_=compose_world_joints(world,curve,20.,config=LightingConfig(**plan['baseline']),dynamics=True)
e.savenpz(out/'world.npz',reference=world,times=np.arange(n)/30)
e.savenpz(out/'baseline.npz',reference=baseline,time=np.arange(800)/40)
e.savenpz(out/'loudness.npz',**curve)
item={'id':'synthetic','key':'synthetic','category':'travel','start':0,'frames':600,'full_frames':n,'source_arm':'reference'}
ctx={'freeze':{'sources':{'synthetic':{'world':str(out/'world.npz'),'dmx':str(out/'baseline.npz'),'loudness':str(out/'loudness.npz'),'map':'synthetic'}}},
    'freeze_sha256':e.digest(out/'freeze.json'),'plan':plan,'rig':canonical_rig(),'cfg':RenderConfig(**plan['renderer']['renderer']),
    'helpers':{'optics':e.load_helper(out,'optics')},'namespace_bindings':bindings}
row=e.map_item(out,ctx,item);target,cached=e.mapped(out,ctx,item)
assert row==cached and row['baseline_recomputation_exact'] and row['nonangle_equal']
assert np.array_equal(e.arrays(target/'full_world.npz')['world'],world_before)
assert e.arrays(target/'full_dmx.npz')['candidate'].shape==(800,512)
assert e.map_item(out,ctx,item)==row  # completed cache, no overwrite or remap
p=target/'full_dmx.npz';p.chmod(0o644);p.write_bytes(p.read_bytes()+b'corrupt')
try:e.mapped(out,ctx,item)
except ValueError:pass
else:raise AssertionError('Corrupt mapped cache accepted')
print('SYNTHETIC_FULL_MAP_AND_NAMESPACE_OK')
