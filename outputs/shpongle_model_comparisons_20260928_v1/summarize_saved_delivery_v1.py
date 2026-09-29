"""Post-generation engineering counts only. No prediction/model/quality scoring."""
from pathlib import Path
import json, hashlib
import numpy as np
root=Path('/root/lightsem/outputs/shpongle_model_comparisons_20260928_v1')
proof=json.loads((root/'PARENT_AUDIT_READBACK_V1.json').read_text())
assert proof['status']=='passed' and proof['videos_completed'] is False
rows=[]
for arm in proof['arms']:
    path=Path(arm['model_outputs']['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==arm['model_outputs']['sha256']
    with np.load(path,allow_pickle=False) as z:
        ids=z['generated_palette_ids'];cues=z['color_cues']
    changed=np.flatnonzero(np.diff(ids)!=0)+1
    overlay=Path(arm['overlay']['path'])
    assert hashlib.sha256(overlay.read_bytes()).hexdigest()==arm['overlay']['sha256']
    with np.load(overlay,allow_pickle=False) as z:
        accents=z['accent_times_s'];active=z['color_active40'];rgb=z['palette_rgb']
    active_indices=np.flatnonzero(active)
    rows.append({'recipe':arm['recipe'],'inputs':{'model_outputs':arm['model_outputs'],'overlay':arm['overlay']},
        'initial_palette_id':int(ids[0]),'unique_palette_ids':np.unique(ids).tolist(),
        'subsequent_palette_id_changes':len(changed),'palette_change_times_s':(changed/10).tolist(),
        'color_cue_opportunities':int(cues.sum()),'accent_events':len(accents),'accent_times_s':accents.tolist(),
        'first_learned_color_active_s':float(active_indices[0]/40) if len(active_indices) else None,
        'palette_rgb_for_used_ids':{str(int(i)):rgb[i].tolist() for i in np.unique(ids)}})
out={'status':'descriptive_only','post_generation':True,'new_model_calls':0,'new_fits':0,'new_hypothesis_test':False,
 'scope':'Engineering summary of authenticated completed Shpongle artifacts, not a preregistered scientific endpoint. Cue opportunities are not token changes; token changes are not guaranteed visible transitions. Counts do not measure beat accuracy, quality or generalization. No outputs or selection are changed. Ashuf used recipient audio.', 'arms':rows}
with (root/'SAVED_DELIVERY_DESCRIPTIVES_V1.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps([{k:r[k] for k in ('recipe','color_cue_opportunities','subsequent_palette_id_changes','unique_palette_ids','accent_events')} for r in rows],indent=2))
