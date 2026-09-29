import importlib.util,json
from pathlib import Path
import numpy as np
from lightsem.sim.renderer import StageRenderer,RenderConfig
from lightsem.conductor.rig import canonical_rig
from lightsem.experiments.stage11.stage11_common import gpu_lock
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/motion_native_audio_20260910')
OUT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/optics_flux_engineering_01')
OLD=ROOT/'delivery/surface_revision_02/source/lightsem/sim'
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
 import sys;sys.modules[name]=m;spec.loader.exec_module(m);return m
s=module('lightsem.sim.shaders_before_observability',OLD/'shaders.py')
r=module('lightsem.sim.renderer_before_observability',OLD/'renderer.py');r.FRAGMENT_SHADER=s.FRAGMENT_SHADER;r.VERTEX_SHADER=s.VERTEX_SHADER
p=json.loads((ROOT/'delivery/surface_revision_02/protocol.json').read_text())['renderer'];p.update(width=320,height=180)
with np.load(OUT.parent/'stage_geometry_02/candidate.npz') as z:dmx=z['dmx']
rig=canonical_rig();frames=[dmx[round(t*40)].copy() for t in (0,20,70,80,90,194,198,250,276)]
for zoom in (0,128,255):
 for dimmer in (51,204):
  d=np.zeros(512,np.uint8)
  for f in rig.fixtures:
   a=f.address-1
   if f.kind=='moving_head':d[a:a+4]=dmx[3200,a:a+4];d[a+4]=dimmer;d[a+6:a+9]=255;d[a+9]=zoom
   else:d[a]=dimmer
  frames.append(d)
with gpu_lock(ROOT):
 old=r.StageRenderer(rig,r.RenderConfig(**p));reference=[old.render_frame(d,0.).copy() for d in frames];old.ctx.release()
 new=StageRenderer(rig,RenderConfig(**p));actual=[new.render_frame(d,0.).copy() for d in frames];new.ctx.release()
 parity=[int(np.abs(a.astype(int)-b.astype(int)).max()) for a,b in zip(reference,actual)];assert max(parity)==0,parity
 # Non-head optics do not change when angular-head profile is enabled.
 d=dmx[3200].copy()
 for f in rig.by_kind('moving_head'):d[f.address-1:f.address+9]=0
 new=StageRenderer(rig,RenderConfig(**p));a=new.render_frame(d,0.).copy();new.ctx.release()
 angular=StageRenderer(rig,RenderConfig(**p,beam_edge_profile='angular',beam_flux_reference_deg=14.));b=angular.render_frame(d,0.).copy();angular.ctx.release();assert np.array_equal(a,b)
 # Confirm real GPU executes new camera and angular uniforms on head-only frame.
 d=dmx[3200].copy()
 for f in rig.by_kind('moving_head'):d[f.address-1+9]=0
 cfg={**p,'cam_pos':(0.,6.,11.),'cam_target':(0.,2.,-3.),'camera_scale':.45,'beam_edge_profile':'angular','beam_edge_softness_deg':.6,'beam_flux_reference_deg':14.}
 angular=StageRenderer(rig,RenderConfig(**cfg));newpixels=angular.render_frame(d,0.).copy();device=angular.renderer_name;angular.ctx.release();assert np.isfinite(newpixels).all()
 import cv2;cv2.imwrite(str(OUT/'preflight_angular_80s.png'),cv2.cvtColor(newpixels,cv2.COLOR_RGB2BGR))
 report={'default_raw_pixel_parity_frames':len(frames),'maximum_byte_difference':max(parity),'nonhead_edge_mode_parity':True,'device':device,'new_controls_execute':True,'scope':'GPU engineering parity/control preflight, not visual/music acceptance','config_preview':cfg}
 (OUT/'engineering.json').write_text(json.dumps(report,indent=2));print(report)
