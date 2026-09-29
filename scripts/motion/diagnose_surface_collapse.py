import json
from pathlib import Path
import numpy as np
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.motion_study.spatial_observer import *
from lightsem.experiments.stage11.stage11_common import authenticated_json,digest
out=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/surface_revision_01');old=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910/delivery/final_diagnostic');rig=canonical_rig()
rows={}
for name,dmx in controls(rig).items():
 s=spatial_series(dmx,rig);r=summary(s,0,len(dmx)/40);rows[name]={'summary':r,'gates':gates(r)}
 print(name,json.dumps(rows[name]),flush=True)
authenticated_json(out/'control_calibration.json',{'thresholds':THRESHOLDS,'rows':rows,'observer_sha256':digest('lightsem/experiments/motion_study/spatial_observer.py')})
rows={}
for name in ['hybrid','incumbent','independent','brightness_off','hybrid_legacy']:
 with np.load(old/'lighting'/f'{name}.npz') as z:d=z['dmx']
 s=spatial_series(d,rig)
 np.savez_compressed(out/f'old_{name}_spatial.npz',**s)
 rows[name]={}
 for label,(lo,hi) in {'full':(0,len(d)/40),'quiet':(0,20),'entry':(45,75),'active':(70,90),'ending':(250,277)}.items():
  r=summary(s,lo,hi);rows[name][label]={'summary':r,'gates':gates(r,active=label!='quiet')}
 print(name,json.dumps(rows[name]),flush=True)
authenticated_json(out/'old_spatial.json',rows)
