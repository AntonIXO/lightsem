#!/usr/bin/env python3
"""Read-only output verification. Never calls the model."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
from lightsem.conductor.rig import canonical_rig
HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'inference_mapping_plan.json').read_text())
root=Path(plan['output_dir'])
m=json.loads((root/'provenance_manifest.json').read_text())
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
for v in m['artifacts'].values(): assert digest(v['path'])==v['sha256']
with np.load(root/'overlay_dmx.npz') as z: o={k:z[k] for k in z.files}
with np.load(root/'model_outputs.npz') as z: p={k:z[k] for k in z.files}
with np.load(plan['baseline']['path']) as z:
    assert np.array_equal(z['companion'],o['baseline'])
    assert np.array_equal(z['time'],o['time'])
# Independently enumerate writable fields from rig patch, not saved masks.
allowed=np.zeros(512,bool); rgbslots=[]; dimslots=[]; angles=[]
for f in canonical_rig().fixtures:
    a=f.address-1
    if f.kind=='moving_head':
        rgbslots.extend([a+6,a+7,a+8]);dimslots.append(a+4);angles.extend([a+i for i in range(4)])
    if f.kind=='wash':rgbslots.extend([a+2,a+3,a+4]);dimslots.append(a)
    if f.kind=='projector':rgbslots.extend([a+2,a+3,a+4])
allowed[rgbslots+dimslots]=True
assert np.array_equal(allowed,o['allowed_slot_mask'])
assert np.array_equal(o['overlay'][:,~allowed],o['baseline'][:,~allowed])
assert np.array_equal(o['overlay'][:,angles],o['baseline'][:,angles])
assert np.array_equal(o['palette_ids40'],np.repeat(p['generated_palette_ids'],4))
assert np.array_equal(o['color_cue_times_s'],p['time10_s'][p['color_cues']])
active=o['color_active40']
color=np.floor(o['palette_rgb'][o['palette_ids40']]*255+.5).astype(np.uint8)
for i in range(0,len(rgbslots),3):
    assert np.array_equal(o['overlay'][np.ix_(active,rgbslots[i:i+3])],color[active])
    assert np.array_equal(o['overlay'][np.ix_(~active,rgbslots[i:i+3])],o['baseline'][np.ix_(~active,rgbslots[i:i+3])])
# Reconstruct fixed raised-cosine pulses independently from persisted event times.
pulse=np.zeros(len(o['time']))
for t in o['accent_times_s']:
    distance=np.abs(o['time']-t)/.125
    pulse=np.maximum(pulse,np.where(distance<1,.35*(.5+.5*np.cos(np.pi*distance)),0.))
expected=np.floor(np.minimum(o['baseline'][:,dimslots]*(1+pulse[:,None]),255)+.5).astype(np.uint8)
assert np.array_equal(expected,o['overlay'][:,dimslots])
assert np.allclose(pulse,o['accent_pulse40'],rtol=0,atol=2e-8)
assert np.isfinite(p['palette_logits']).all() and np.isfinite(p['context10']).all()
# Model state changes only on registered generated cues (initialization excluded).
changed=np.diff(p['generated_palette_ids'])!=0
assert np.all(p['color_cues'][1:][changed])
assert p['previous_ids'][0]==32
assert np.array_equal(p['previous_ids'][1:],p['generated_palette_ids'][:-1])
receipt={'status':'passed','model_forward_calls':0,'reloaded_artifact_hashes_verified':True,
         'saved_baseline_equals_accepted_companion':True,'independent_slot_mask_exact':True,
         'independent_protected_bytes_exact':True,'independent_RGB_track_reconstruction_exact':True,
         'independent_accent_dimmer_reconstruction_exact':True,'generated_state_hold_and_previous_ids_exact':True,
         'first_actual_color_cue_s':float(o['color_cue_times_s'][0]) if len(o['color_cue_times_s']) else None,
         'artifacts':{'overlay_dmx_sha256':digest(root/'overlay_dmx.npz'),'model_outputs_sha256':digest(root/'model_outputs.npz')},
         'helper_sha256':digest(__file__),'mnt_free_bytes':shutil.disk_usage(root).free,
         'new_artifact_bytes':sum(p.stat().st_size for p in root.iterdir() if p.is_file())}
assert receipt['mnt_free_bytes']>=1024**3 and receipt['new_artifact_bytes']<=64*1024**2
with (root/'readback_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
