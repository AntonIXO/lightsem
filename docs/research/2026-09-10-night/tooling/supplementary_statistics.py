"""Post-hoc explanatory statistics from saved predictions; no fitted model or selection.

CPU only. Original primary metrics and frozen acceptance rules remain authoritative.
"""
from pathlib import Path
import argparse, itertools, json, sys
import numpy as np
from scipy.ndimage import uniform_filter1d, binary_dilation
from scipy.signal import find_peaks

ROOT = Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0, str(ROOT/'source'))
from lightsem.experiments.night_common import BASE, base_context
from lightsem.experiments.stage11_common import read_authenticated, authenticated_json, digest

def ci(x):
    x=np.asarray(x,float)
    if not len(x) or not np.isfinite(x).all(): raise ValueError('Nonempty finite values required')
    rng=np.random.default_rng(19413)
    bs=x[rng.integers(len(x),size=(10000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'canonical_songs':len(x)}

def checked(path, sha):
    if digest(path)!=sha: raise RuntimeError('Input drift: '+str(path))

def masked_joint_error(p,t,m):
    valid=np.asarray(m,bool).reshape(-1,17,2).all(2)
    d=np.linalg.norm((p-t).reshape(-1,17,2),axis=2)
    return float(d[valid].mean()) if valid.any() else None

def pose():
    inputs=read_authenticated(BASE/'evaluation/input_manifest.json')['rows']
    bykey={r['entry']['key']:r for r in inputs}
    prepared=read_authenticated(ROOT/'evaluation/input_manifest.json')['rows']
    if len(bykey)!=40: raise ValueError('Expected forty distinct cases')
    rows=[]; files={}; names=['descriptor128','descriptor256','pose_code24','audio_base','audio_fourier','audio_muq']
    for name,seed in itertools.product(names,[0,1]):
        ep=ROOT/'diagnostics/pose_probes'/name/f'seed_{seed}'/'evaluation.json'
        report=read_authenticated(ep); files[str(ep)]=digest(ep)
        if len(report['rows'])!=40 or {r['key'] for r in report['rows']}!=set(bykey): raise ValueError('Pose membership')
        saved={r['key']:r for r in report['rows']}
        for r in report['rows']:
            checked(r['path'],r['sha256']); source=bykey[r['key']]
            checked(source['path'],source['sha256'])
            with np.load(source['path']) as z:
                events=z['b'].max(1)>.5
                # Separate annotated movement-event context from audio-onset context.
                from lightsem.experiments.beat_conditioned_data import X_NAMES
                onset=z['x'][:,X_NAMES.index('onset')]
                peaks=find_peaks(onset,distance=10,prominence=max(float(np.std(onset))*.5,1e-8))[0]
            event_context=binary_dilation(events,iterations=20)
            onset_context=np.zeros(len(events),bool);onset_context[peaks]=True
            onset_context=binary_dilation(onset_context,iterations=10)
            with np.load(r['path']) as z:
                truth=z['truth'];mask=z['mask']; modes={k:z[k] for k in z.files if k not in ('truth','mask')}
            donor_key=prepared['base/'+r['key']]['donor_key'];donor_row=saved[donor_key]
            checked(donor_row['path'],donor_row['sha256'])
            with np.load(donor_row['path']) as z:
                common=mask&z['mask'];donor_control={'key':donor_key,'canonical_id':donor_row['canonical_id'],
                    'saved_donor_prediction_error_tl':masked_joint_error(z['full'],truth,common),
                    'donor_observed_pose_error_tl':masked_joint_error(z['truth'],truth,common),
                    'matched_full_error_tl':masked_joint_error(modes['full'],truth,common),'common_valid_coordinates':int(common.sum())}
            if donor_row['canonical_id']==r['canonical_id']:raise ValueError('Donor must be another canonical song')
            if not np.isfinite(truth).all() or not all(np.isfinite(p).all() for p in modes.values()): raise ValueError('Pose nonfinite')
            extra={}
            for mode,pred in modes.items():
                ms={}
                for seconds in (.1,.5,2.,8.):
                    size=int(seconds*40)|1
                    support=uniform_filter1d(mask.astype(float),size,axis=0,mode='constant')
                    tt=uniform_filter1d(truth*mask,size,axis=0,mode='constant')/np.maximum(support,1e-9)
                    pp=uniform_filter1d(pred*mask,size,axis=0,mode='constant')/np.maximum(support,1e-9)
                    ms[str(seconds)]=masked_joint_error(pp,tt,support>=.8)
                segments={}
                for label,select in [('motion_event_context',event_context),('outside_motion_event_context',~event_context),('audio_onset_context',onset_context),('outside_audio_onset_context',~onset_context)]:
                    mm=mask&select[:,None]
                    segments[label]={'joint_error_tl':masked_joint_error(pred,truth,mm),'frames':int(select.sum()),'valid_coordinates':int(mm.sum())}
                extra[mode]={'smoothed_joint_error_tl':ms,'segments':segments}
            persistence={}
            if name=='descriptor128' and seed==0:
                for seconds in (.1,.5,2.,8.):
                    lag=round(seconds*40)
                    persistence[str(seconds)]=masked_joint_error(truth[:-lag],truth[lag:],mask[:-lag]&mask[lag:])
            rows.append({'name':name,'seed':seed,'key':r['key'],'canonical_id':r['canonical_id'],'primary_errors':r['errors'],'extra':extra,'oracle_persistence':persistence,'donor_control':donor_control})
        print(json.dumps({'stage':'supplement_pose','name':name,'seed':seed}),flush=True)
    summary={}
    for name,seed in itertools.product(names,[0,1]):
        rs=[r for r in rows if r['name']==name and r['seed']==seed]
        summary[f'{name}/seed_{seed}']={mode:{metric:ci([r['primary_errors'][mode][metric] for r in rs]) for metric in ('joint_error_tl','velocity_error_tl_s','root_error_tl','bone_length_error_tl')} for mode in rs[0]['primary_errors']}
    paired={}
    for seed in (0,1):
        a={r['key']:r for r in rows if r['name']=='descriptor256' and r['seed']==seed}
        b={r['key']:r for r in rows if r['name']=='pose_code24' and r['seed']==seed}
        paired[str(seed)]=ci([(a[k]['primary_errors']['full']['joint_error_tl']-b[k]['primary_errors']['full']['joint_error_tl'])/a[k]['primary_errors']['full']['joint_error_tl'] for k in sorted(a)])
    output={'post_hoc':True,'method':'Canonical-song bootstrap within each seed; smoothing .1/.5/2/8s, >=80% mask support; movement labels dilated .5s, onset peaks dilated .25s; target-past persistence is an oracle control, unavailable to an audio-only system. Foreign-pose and foreign-prediction controls reuse the fixed closest-tempo donor and saved outputs; common confidence support, no refit/inference.',
            'rows':rows,'summary':summary,'relative_pose_code_improvement_over_descriptor256':paired,'source_sha256':digest(__file__),'input_reports':files}
    authenticated_json(ROOT/'diagnostics/supplementary_pose.json',output)

def embeddings():
    folder=ROOT/'diagnostics/embedding_controls'; original=read_authenticated(folder/'report.json'); output={}; inputs={}; baseline_join=None
    for family in original['results']:
        path=folder/(family+'_predictions.npz');inputs[str(path)]=digest(path)
        with np.load(path) as z:p=z['prediction'].astype(float);t=z['truth'].astype(float);keys=z['canonical_ids'];fold=z['fold']
        join=(t,keys,fold)
        if baseline_join is None:baseline_join=join
        elif not all(np.array_equal(x,y) for x,y in zip(join,baseline_join)):raise ValueError('Unmatched feature-family evaluation rows')
        if not np.isfinite(p).all():raise ValueError('Nonfinite prediction')
        baseline=np.empty_like(t);scale=np.empty_like(t)
        for f in range(5):
            tr=fold!=f;te=~tr
            baseline[te]=t[tr].mean(0);scale[te]=np.maximum(t[tr].std(0,ddof=1),1e-5)
        rows=[]
        for key in sorted(set(keys)):
            ix=keys==key;a=t[ix];b=p[ix];s=scale[ix];var=a.var(0);ok=var>1e-8
            mse=np.square((a-b)/s).mean(0);base_mse=np.square((a-baseline[ix])/s).mean(0)
            centered=np.square((a-a.mean(0))-(b-b.mean(0))).mean(0)
            rows.append({'canonical_id':str(key),'fold':int(fold[ix][0]),'standardized_mse_by_channel':mse.tolist(),'train_mean_standardized_mse_by_channel':base_mse.tolist(),
                'within_song_variance_by_channel':var.tolist(),'centered_standardized_mse_by_channel':(centered/s[0]**2).tolist(),
                'raw_r2':float((1-np.square(a-b).mean(0)[ok]/var[ok]).mean()),'centered_r2':float((1-centered[ok]/var[ok]).mean()),
                'standardized_mse':float(mse.mean()),'train_mean_standardized_mse':float(base_mse.mean()),'gain_over_train_mean':float((base_mse.mean()-mse.mean())/base_mse.mean())})
        output[family]={'rows':rows,'summary':{k:ci([r[k] for r in rows]) for k in ('raw_r2','centered_r2','standardized_mse','train_mean_standardized_mse','gain_over_train_mean')}}
    authenticated_json(ROOT/'diagnostics/supplementary_embeddings.json',{'post_hoc':True,'results':output,'targets':original['targets'],'input_files':inputs,'source_sha256':digest(__file__),
        'method':'Re-express saved 80-song TRAIN out-of-fold predictions. Scale and constant baseline use training folds only. Raw macro song R2 is preserved; tiny within-song variance can dominate it. No model refit or candidate selection.'})

def coaches():
    manifest,norm=base_context(); groups={};pr=read_authenticated(ROOT/'features/pose/report.json')
    pp={r['key']:r for r in pr['rows']};rows=[]
    for e in manifest['entries']:
        if e['partition']=='val':groups.setdefault(e['recording_id'],[]).append(e)
    for recording,entries in groups.items():
        for a,b in itertools.combinations(sorted(entries,key=lambda x:x['key']),2):
            if a['canonical_id']!=b['canonical_id'] or abs(a['offset_s']-b['offset_s'])>1e-6 or a['n_frames']!=b['n_frames']:raise ValueError('Unaligned coach pair')
            cached=[]
            for e in (a,b):
                p=pp[e['key']];checked(p['path'],p['sha256']);checked(e['path'],e['sha256'])
                with np.load(p['path']) as z:pose=z['pose'];mask=z['mask']
                with np.load(e['path']) as z:y=(z['y']-norm['y_center'])/norm['y_scale']
                cached.append((pose,mask,y))
            x,z=cached
            rows.append({'recording_id':recording,'canonical_id':a['canonical_id'],'keys':[a['key'],b['key']],
                'pose_difference_tl':masked_joint_error(x[0],z[0],x[1]&z[1]),'standardized_descriptor_mse':float(np.square(x[2]-z[2]).mean()),'frames':a['n_frames']})
    macro={}
    for r in rows:macro.setdefault(r['canonical_id'],[]).append(r)
    if not rows:raise ValueError('Empty same-recording coach pairs')
    summary={k:ci([np.mean([r[k] for r in rs]) for rs in macro.values()]) for k in ('pose_difference_tl','standardized_descriptor_mse')}
    authenticated_json(ROOT/'diagnostics/supplementary_coaches.json',{'post_hoc':True,'rows':rows,'summary':summary,'source_sha256':digest(__file__),
        'method':'Only simultaneous coaches from the same recording and exact same active audio span. Mean-root-x-normalized detected 2D poses. Pair means are averaged within canonical song before bootstrap. Pose detection, viewing geometry, body differences and genuine choreography differences all contribute; this is not an irreducible-error estimate.'})

def rhythm():
    import mir_eval
    report=read_authenticated(ROOT/'rhythm/report.json');rows=[]
    for r in report['rows']:
        checked(r['path'],r['sha256'])
        with np.load(r['path']) as z:
            truth=mir_eval.beat.trim_beats(z['downbeats'])
            scores={k:float(mir_eval.beat.f_measure(truth,mir_eval.beat.trim_beats(z[k]))) for k in ('constant_downbeats','input_bar_events')}
        tokens=r['key'].split('_');style=''.join(c for c in tokens[1].split('-')[0] if not c.isdigit())
        rows.append({'key':r['key'],'player':r['player'],'lead_sheet':r['lead_sheet'],'style':style,'version':tokens[-1],
                     'progression':tokens[1].split('-')[0][-1],'downbeat_f1':scores,'beat_scores':r['scores']})
    groups={}
    for axis in ('player','style','version','progression'):
        groups[axis]={}
        for value in sorted({r[axis] for r in rows}):
            rs=[r for r in rows if r[axis]==value]
            groups[axis][value]={'clips':len(rs),'beat_f1':{m:float(np.mean([r['beat_scores'][m]['f1'] for r in rs])) for m in ('constant_grid','plp')},
                'downbeat_f1':{m:float(np.mean([r['downbeat_f1'][m] for r in rs])) for m in ('constant_downbeats','input_bar_events')}}
    authenticated_json(ROOT/'diagnostics/supplementary_rhythm.json',{'post_hoc':True,'rows':rows,'groups':groups,'source_sha256':digest(__file__),
        'method':'No refit. Same358files and first5s trim. Additional downbeat F1 for constant-grid downbeats and actual bar-phase resets; subgroup means are descriptive, not independent new tests.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['pose','embeddings','coaches','rhythm']);args=ap.parse_args()
    globals()[args.stage]()
