"""Read-only saved-artifact support audit. No inference, labels, AP, W1, calibration or bootstrap runs."""
from pathlib import Path
import hashlib,json,math
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1')
RP=ROOT/'palette_secondary_report_v1.json'
EXPECTED='d25c8edbf229869fd8b3b8a7e0428bba09326703216d5583c1554efafc0336f6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(r):
    p=Path(r['path'])
    assert sha(p)==r['sha256'],str(p)
    return p
assert sha(RP)==EXPECTED
report=json.loads(RP.read_text());combined=json.loads(checked(report['predictions']).read_text())
base=json.loads(checked(report['base_registration']).read_text());manifest=json.loads(checked(base['manifest']).read_text())
meta={r['recording_id']:r for r in manifest['records']}
rows={(r['role'],r['seed'],r['recording_id'],r['draw']):r for r in report['rows']}
floors={(r['role'],r['seed'],r['recording_id']):r for r in report['floors']}
errors=[];count=0;cells=[]
def check(ok,msg):
    global count
    count+=1
    if not ok:errors.append(msg)
for entry in combined['records']:
    key=entry['role'],entry['seed'],entry['recording_id'];rid=entry['recording_id']
    check(meta[rid]['cuts_status']=='unannotated','Cannot assume known-cut maskzero:'+rid)
    with np.load(checked(entry['original_prediction']),allow_pickle=False) as z:
        requested=z['model__requested_times10_s'].copy();source=z['model__source_times10_s'].copy()
        valid=z['observer__color_valid'].copy()
        if 'observer__frame_valid' in z:valid &=z['observer__frame_valid']
        gaps=np.flatnonzero(np.diff(source)>.2+1e-8);valid[gaps]=False;valid[gaps+1]=False
        target_mask=z['target__color_mask10'].copy();old_cues=np.unique(np.r_[0,np.flatnonzero(z['model__generated_color_cues'])]).astype(int)
        probability=z['model__color'].copy()
    with np.load(checked(entry['artifact']),allow_pickle=False) as z:
        cues=z['cues'].astype(int)
        check(np.array_equal(cues,old_cues),'Cues changed:'+str(key))
        inter=valid[:-1]&valid[1:];inter[cues[1:]-1]=False
        prefix=np.r_[0,np.cumsum(~valid)];starts=np.arange(560)
        lag=int(np.sum(prefix[starts+81]==prefix[starts]))
        unsupported=sum(not valid[a:(cues[j+1] if j+1<len(cues) else 640)].any() for j,a in enumerate(cues))
        intervals=[];center=requested[0]+28;end=requested[0]+36
        for i in np.flatnonzero(target_mask):
            stop=source[i+1] if i+1<len(source) else source[i]+.1
            check(stop-source[i]<=.2+1e-8,'Eligible timinglabel bridges gap:'+str(key))
            a,b=max(center,source[i]),min(end,stop)
            if a<b:intervals.append((a,b))
        duration=sum(b-a for a,b in intervals)
        for draw in (101,102,103,104):
            row=rows[(*key,draw)]
            check(np.array_equal(z[f'{draw}__uniform_noise'],np.random.default_rng(draw).random(len(cues))),'Noise changed:'+str(key))
            for name,c in row['conditions'].items():
                ids=z[f'{draw}__{name}'];held=ids[cues][np.searchsorted(cues,np.arange(640),side='right')-1]
                check(np.array_equal(ids,held),'Not held:'+str(key)+name)
                for summary_name in ('generated','reference'):
                    s=c[summary_name]
                    check(abs(s['observed_valid_time_fraction']-valid.mean())<1e-12,'Valid fraction:'+str(key))
                    check(abs(s['valid_duration_s']-valid.sum()*.1)<1e-12,'Valid duration:'+str(key))
                    check(s['eight_second_pair_count']==lag,'8s support count:'+str(key))
                    check(s['intercue_pair_count']==int(inter.sum()),'Adjacent support count:'+str(key))
                check(c['unsupported_interval_count']==unsupported,'Unsupported cue interval count:'+str(key))
                check(abs(c['timing']['eligible_duration_s']-duration)<1e-9,'Timing support duration:'+str(key))
                at=cues[cues>0];at=at[(ids[at]!=ids[at-1])&valid[at]&valid[at-1]]
                at=at[[any(a<=requested[i]<b for a,b in intervals) for i in at]] if len(at) else at
                check(c['timing']['candidate_count']==len(at),'Actual emitted candidate count:'+str(key)+name)
                check(c['timing']['emitted_count']==int(np.sum(probability[at]>=.5)),'Actual deployed cue count:'+str(key)+name)
            check(np.all(z[f'{draw}__persistence']==z[f'{draw}__own_state'][0]),'Persistence changed:'+str(key))
        for f in floors[key].values():
            if not isinstance(f,dict):continue
            check(f['generated']['eight_second_pair_count']==lag and f['reference']['eight_second_pair_count']==lag,'Floor support:'+str(key))
    cells.append(dict(role=key[0],seed=key[1],recording_id=rid,valid_fraction=float(valid.mean()),valid_duration_s=float(valid.sum()*.1),
        eight_second_pair_count=lag,intercue_pair_count=int(inter.sum()),unsupported_interval_count=unsupported,
        cue_count=len(cues),center_timing_duration_s=duration,mask_sha256=hashlib.sha256(valid.tobytes()).hexdigest()))
output=HERE/'PALETTE_SAVED_SUPPORT_AUDIT_V1.json'
result=dict(scope='Read-only existing prediction masks/cues/counts/arithmetic;no rawsource inputs or inference/AP/W1/bootstrap/calibration rerun',
    status='passed' if not errors else 'failed',report={'path':str(RP),'sha256':EXPECTED},source={'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
    checked_cells=len(cells),checks=count,errors=errors,cells=cells)
with output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in result.items() if k!='cells'},indent=2))
