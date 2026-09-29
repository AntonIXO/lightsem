"""Predeclared supplementary MuQ centering controls, using all four final MuQ cells.

These out-of-distribution input edits explain local use; they never change main gates.
"""
from pathlib import Path
import argparse,json,sys,time,gc
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import authenticated_json,read_authenticated,digest,verify_artifacts,artifact_manifest,gpu_lock
from lightsem.experiments.night_common import BASE
OUT=ROOT/'diagnostics/muq_local_controls'
MODES=('aux_centered','aux_mean_only','aux_centered_donor')

def freeze():
    OUT.mkdir(exist_ok=True)
    source=read_authenticated(ROOT/'evaluation/input_manifest.json');files={str(Path(__file__)):digest(__file__),
        str(ROOT/'protocol.json'):digest(ROOT/'protocol.json'),str(BASE/'evaluation/input_manifest.json'):digest(BASE/'evaluation/input_manifest.json'),
        str(ROOT/'features/muq/training_array.json'):digest(ROOT/'features/muq/training_array.json')}
    for key,r in source['rows'].items():
        if key.startswith('muq/'):
            for mode in ('full','donor_full'):files[r['files'][mode]['path']]=r['files'][mode]['sha256']
    protocol={'files':files,'arms':['muq_once','muq_film'],'seeds':[0,1],'modes':list(MODES),'draw_seeds':list(range(101,107)),
        'normalization':'Use exact training FP16-storage-to-FP32-standardization first. Center each1024channel over the fixed60s excerpt. Mean-only repeats this excerpt mean; centered-donor subtracts its own excerpt mean. Base92channels remain the correct original audio in all three modes.',
        'interpretation':'Supplementary mechanism analysis, prepared before any main candidate inference. Input centering is distribution-shifting, not an independently trained centered-embedding arm. Joint score and matched-noise centered-donor tests assess local contributions conditional on the unchanged base features. No new acceptance threshold or selection.'}
    path=OUT/'protocol.json'
    if path.exists():
        if read_authenticated(path)!=protocol:raise RuntimeError('MuQ local-control protocol changed')
    else:authenticated_json(path,protocol)

def run():
    import torch
    from lightsem.experiments.night_evaluation import load_cell,normalized_input,generate_groups
    from lightsem.experiments.night_metrics import energy_score,alignment_error
    from lightsem.experiments.beat_conditioned_metrics import bootstrap_mean
    from sklearn.metrics import average_precision_score
    p=read_authenticated(OUT/'protocol.json')
    for path,sha in p['files'].items():
        if digest(path)!=sha:raise RuntimeError('Local-control input drift: '+path)
    if not read_authenticated(ROOT/'pipeline_state.json').get('gpu_pipeline_complete'):raise RuntimeError('Main GPU pipeline must finish first')
    rows=read_authenticated(BASE/'evaluation/input_manifest.json')['rows'];prepared=read_authenticated(ROOT/'evaluation/input_manifest.json')['rows']
    norm=read_authenticated(ROOT/'features/muq/training_array.json');reports=[]
    for seed in p['seeds']:
        for arm in p['arms']:
            model,checkpoint,main,n=load_cell(arm,seed,ROOT);results=[];folder=OUT/arm/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
            for src in rows:
                key=src['entry']['key'];rp=folder/(key+'.json')
                if rp.exists():
                    r=read_authenticated(rp)
                    if r['checkpoint_sha256']!=checkpoint or r['protocol_sha256']!=digest(OUT/'protocol.json'):raise RuntimeError('Saved local inference drift')
                    verify_artifacts(r['artifacts']);results.append(r);continue
                with np.load(src['path']) as z:truth=(z['y'].astype(np.float32)-np.asarray(n['y_center'],np.float32))/np.asarray(n['y_scale'],np.float32);b=z['b']
                pins=prepared['muq/'+key]['files']
                with np.load(pins['full']['path']) as z:full=normalized_input(z['raw_audio'],z['auxiliary'],n,norm)
                with np.load(pins['donor_full']['path']) as z:donor=normalized_input(z['raw_audio'],z['auxiliary'],n,norm)
                center=full[:,:,92:].mean(1,keepdim=True);donor_center=donor[:,:,92:].mean(1,keepdim=True)
                variants={}
                for mode in MODES:
                    x=full.clone()
                    if mode=='aux_centered':x[:,:,92:]-=center
                    elif mode=='aux_mean_only':x[:,:,92:]=center
                    else:x[:,:,92:]=donor[:,:,92:]-donor_center
                    if not torch.equal(x[:,:,:92],full[:,:,:92]):raise ValueError('Base audio changed')
                    if mode!='aux_mean_only' and x[:,:,92:].mean(1).abs().max()>2e-5:raise ValueError('Centering failed')
                    variants[mode]=x
                result={};paths=[];begin=time.monotonic()
                for mode,x in variants.items():
                    y,events=generate_groups(model,x,seeds=tuple(p['draw_seeds']))
                    result[mode]={'energy_score':energy_score(y[:,:,4:],truth[:,4:]),'alignment_error':alignment_error(y.mean(0),truth,n['target_names']),
                        'event_ap':[float(average_precision_score(b[:,j],events[:,j])) if 0<b[:,j].sum()<len(b) else None for j in range(3)]}
                    path=folder/(key+'_'+mode+'.npz');np.savez_compressed(path,standardized=y,event_scores=events,noise_seeds=p['draw_seeds']);paths.append(path)
                r={'key':key,'canonical_id':src['entry']['canonical_id'],'arm':arm,'seed':seed,'checkpoint_sha256':checkpoint,'results':result,
                    'gpu_task_seconds':time.monotonic()-begin,'artifacts':artifact_manifest(paths),
                    'protocol_sha256':digest(OUT/'protocol.json')}
                authenticated_json(rp,r);results.append(r)
            stats={}
            for control in ('aux_mean_only','aux_centered_donor'):
                stats[control]=bootstrap_mean([(r['results'][control]['energy_score']-r['results']['aux_centered']['energy_score'])/max(r['results'][control]['energy_score'],1e-12) for r in results])
            report={'arm':arm,'seed':seed,'rows':results,'centered_energy_advantages':stats,'gpu_task_seconds':sum(r['gpu_task_seconds'] for r in results),'checkpoint_sha256':checkpoint}
            authenticated_json(folder/'report.json',report);reports.append(report)
            print(json.dumps({'stage':'muq_local_controls','arm':arm,'seed':seed,'advantages':stats}),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
    authenticated_json(OUT/'report.json',{'cells':reports,'gpu_task_seconds':sum(r['gpu_task_seconds'] for r in reports),'protocol_sha256':digest(OUT/'protocol.json'),'interpretation':p['interpretation']})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:
        with gpu_lock(ROOT):run()
