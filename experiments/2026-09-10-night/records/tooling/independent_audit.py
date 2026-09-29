"""Independent read-only audit of frozen training, joins, scores and emitted bytes.

No model is executed here. Candidate metrics are rederived from saved outputs.
"""
from pathlib import Path
import argparse, itertools, json, math, sys, time
import numpy as np
from scipy.spatial.distance import cdist, pdist

ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.stage11_common import read_authenticated,authenticated_json,digest,verify_artifacts,json_digest
from lightsem.experiments.night_common import BASE,base_context,ARMS,SEEDS

def require(value, reason):
    if not value:raise AssertionError(reason)

def close(a,b,label,atol=2e-6):
    if a is None or b is None:require(a is None and b is None,label)
    else:require(np.allclose(a,b,rtol=1e-6,atol=atol,equal_nan=False),label+f': {a} != {b}')

def bootstrap(x):
    x=np.asarray(x,float);require(len(x)>0 and np.isfinite(x).all(),'Invalid bootstrap input')
    rng=np.random.default_rng(991);bs=np.mean(x[rng.integers(len(x),size=(10000,len(x)))],axis=1)
    return {'mean':float(x.mean()),'ci95':np.quantile(bs,[.025,.975]).tolist(),'songs':len(x),'resamples':10000}

def score(draws,truth):
    # Independent vectorization of the unbiased ensemble estimator.
    require(draws.shape==(6,2400,17) and truth.shape==(2400,17),'Energy input shape')
    blocks=draws.astype(float).reshape(6,30,1360).transpose(1,0,2)
    target=truth.astype(float).reshape(30,1360)
    values=[cdist(x,y[None]).mean()-pdist(x).sum()/30 for x,y in zip(blocks,target)]
    return float(np.mean(values)/math.sqrt(1360))

def alignment(pred,truth,names):
    channels=[names.index(k) for k in ('com_speed','vert_bounce_amp')]
    # Accumulate in original float32, matching the definition's valid moving mean.
    delta=[]
    for j in channels:
        a=np.array([pred[k:k+40,j].mean() for k in range(len(pred)-39)])
        b=np.array([truth[k:k+40,j].mean() for k in range(len(truth)-39)])
        delta.append(np.square(a-b))
    return float(np.mean(delta))

def byte_composition(dmx,rig):
    heads=sorted(rig.by_kind('moving_head'),key=lambda h:(h.position[0],h.name))
    dim=np.stack([dmx[:,h.address+3]/255. for h in heads],axis=1)
    spans=np.percentile(dim,95,axis=0)-np.percentile(dim,5,axis=0)
    longest=[];velocity=[]
    for j,h in enumerate(heads):
        runs=[len(list(group)) for value,group in itertools.groupby(dim[:,j]==0) if value]
        longest.append(max(runs,default=0))
        start=h.address-1
        words=dmx[:,start:start+4].astype(np.uint32)
        pt=np.column_stack((words[:,0]*256+words[:,1],words[:,2]*256+words[:,3]))
        pt=pt/65535*np.array([h.pan_range_deg,h.tilt_range_deg])
        velocity.append(np.diff(pt,axis=0).ravel())
    corr=np.corrcoef(velocity);values=[abs(corr[i,j]) for i,j in itertools.combinations(range(6),2) if i+j!=5]
    return {'spans':spans,'longest':longest,'correlation':float(np.nanmedian(values)) if np.isfinite(values).any() else None}

def output_gates(values,videos,thresholds):
    t=thresholds
    def supported(key):return [r[key] for r in values if r[key] is not None]
    alignment=bootstrap(supported('alignment_advantage'));local=bootstrap(supported('local_alignment_advantage'));event=bootstrap(supported('event_ap_advantage'))
    physical=[];brightness=[]
    for v in videos:
        b=v['beam'];s=v['screen'];c=v['composition'];off=v['off_intervals']
        physical.append(b['lit_motion_fraction']>=t['lit_motion_min'] and b['span_2s_median_deg']>=t['span_min_deg'] and b['upward_light_fraction'] is not None and
            b['upward_light_fraction']<=t['upward_max'] and b['direction_hf_above4hz_fraction']<=t['hf_max'] and s['visible_motion_frame_fraction']>=t['visible_motion_min'] and
            s['mean_brightness']>=t['screen_brightness_min'] and s['support_fraction']>=t['screen_support_min'])
        brightness.append(c['heads_with_span_ge_point4']>=t['heads_with_dimmer_span_min'] and off['heads_with_off_at_least_150ms']>=t['off_heads_min'])
    gates={'conditioning':np.mean([r['controlled'] for r in values])>=t['controlled_song_fraction_min'],
        'native_alignment':alignment['mean']>=t['alignment_advantage_min'] and alignment['ci95'][0]>0,
        'local_alignment':local['mean']>=t['local_alignment_advantage_min'] and local['ci95'][0]>0,
        'event_selectivity':event['mean']>=t['event_ap_advantage_min'] and event['ci95'][0]>0,
        'visible_motion':len(physical)==8 and np.mean(physical)>=t['valid_clip_fraction_min'],
        'brightness':len(brightness)==8 and np.mean(brightness)>=t['valid_clip_fraction_min'],
        'no_fast_jitter':np.mean([r['beam']['direction_hf_above4hz_fraction'] for r in values])<=t['hf_max'],
        'native_temporal_variation':np.mean([r['native_dispersion']['fraction_ratio_in_point25_to2'] or 0 for r in values])>=t['native_supported_fraction_min'],
        'native_draw_variation':np.mean([r['native_dispersion']['mean_within_audio_draw_std'] for r in values])>=t['native_draw_std_min'],
        'group_independence':np.mean([r['independent']['unrelated_velocity_abs_correlation'] is not None and r['independent']['unrelated_velocity_abs_correlation']<t['unrelated_velocity_abs_correlation_max'] for r in videos])>=t['valid_clip_fraction_min'],
        'mirror_geometry':all(r['mirror_target_error_m']<t['mirror_target_error_max_m'] and r['mirror_emitted_ray_error_deg']<t['mirror_ray_error_max_deg'] for r in videos),
        'integrity':True}
    return {k:bool(v) for k,v in gates.items()}

def preflight():
    p=read_authenticated(ROOT/'protocol.json');f=read_authenticated(ROOT/'frozen_sources.json')
    require(p['frozen'] and p['source_sha256']==f['source_sha256'],'Frozen protocol/source mismatch')
    require(p['arms']==list(ARMS) and p['seeds']==list(SEEDS) and p['main_model_count']==12,'Frozen factorial membership')
    require(p['training']['steps']==15200 and p['test_access'] is False,'Fixed final step and no TEST')
    for path,sha in p['input_artifacts'].items():require(digest(path)==sha,'Changed input '+path)
    for rel,sha in f['files'].items():require(digest(ROOT/'source'/rel)==sha,'Changed frozen source '+rel)
    require(json_digest(f['files'])==f['source_sha256'],'Source manifest digest')
    m,n=base_context();require(all(e['partition'] in ('train','val') for e in m['entries']),'TEST data present')
    require(n['feature_names'][67]=='flux_lf' and n['feature_names'][69]=='flux_mid' and n['feature_names'][71]=='flux_hf','Fourier named columns')
    inputs=read_authenticated(ROOT/'evaluation/input_manifest.json');base=read_authenticated(BASE/'evaluation/input_manifest.json')['rows']
    require(len(base)==40 and len({r['entry']['canonical_id'] for r in base})==40,'Canonical VAL membership')
    require(len(inputs['rows'])==120,'Prepared family membership')
    report={'passed':True,'source_sha256':f['source_sha256'],'input_artifacts':len(p['input_artifacts']),'source_files':len(f['files']),
        'protocol_sha256':digest(ROOT/'protocol.json'),'auditor_sha256':digest(__file__),'test_access':False}
    authenticated_json(ROOT/'diagnostics/independent_input_audit.json',report)

def training(complete=False):
    import torch
    p=read_authenticated(ROOT/'protocol.json');rows=[];initial_by_seed={};reports=[]
    for seed,arm in itertools.product(SEEDS,ARMS):
        run=ROOT/'runs'/arm/f'seed_{seed}';rp=run/'report.json'
        if not rp.exists():
            require(not complete,'Missing completed cell '+str(run));continue
        r=read_authenticated(rp);verify_artifacts(r['artifacts']);reports.append(str(rp))
        before=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
        after=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
        latest=torch.load(run/'latest.pt',map_location='cpu',weights_only=False)
        identity=after['identity'];require(identity==before['identity']==latest['identity'],'Checkpoint identity')
        # JSON serializes the checkpoint's dilation tuple as an array.
        require(json_digest(identity)==json_digest(r['identity']),'Report/checkpoint semantic identity')
        require(identity['arm']==arm and identity['seed']==seed and identity['protocol_sha256']==digest(ROOT/'protocol.json'),'Cell identity')
        require(identity['source_sha256']==p['source_sha256'],'Cell source')
        require(before['step']==0 and after['step']==latest['step']==r['steps']==15200,'Scheduled training steps')
        require(after['normalizers']==before['normalizers']==latest['normalizers']==read_authenticated(BASE/'normalizers.json'),'Normalizer identity')
        require(set(before['model'])==set(after['model'])==set(latest['model']),'Model state membership')
        changed=[];unchanged=[]
        for k,v in after['model'].items():
            require(torch.isfinite(v).all().item(),'Nonfinite checkpoint '+k)
            require(torch.equal(v,latest['model'][k]),'Latest/final mismatch '+k)
            (unchanged if torch.equal(v,before['model'][k]) else changed).append(k)
        require(len(changed)>0,'Untrained model')
        require(len(after['optimizer']['state'])>0,'Empty optimizer')
        optimizer_steps={int(s['step']) for s in after['optimizer']['state'].values()}
        require(optimizer_steps=={15200},'Optimizer exposure')
        require(after['scheduler']['last_epoch']==15200,'Scheduler step')
        log=[json.loads(x) for x in (run/'training.jsonl').read_text().splitlines()]
        require([x['step'] for x in log]==[1]+list(range(100,15201,100)),'Missing/duplicate training log steps')
        require(all(x['arm']==arm and x['seed']==seed for x in log),'Log identity')
        require(all(np.isfinite([x[k] for k in ('a','b','c_flow','total','grad','lr','training_seconds')]).all() for x in log),'Nonfinite training log')
        require(all(x['training_seconds']<y['training_seconds'] for x,y in zip(log,log[1:])),'Nonmonotone compute log')
        close(log[-1]['training_seconds'],after['training_seconds'],'Compute total')
        if seed not in initial_by_seed:initial_by_seed[seed]=before['model']
        shared=initial_by_seed[seed]
        matching=[k for k in shared if k in before['model'] and shared[k].shape==before['model'][k].shape]
        require(all(torch.equal(shared[k],before['model'][k]) for k in matching),'Shared initialization changed')
        rows.append({'arm':arm,'seed':seed,'steps':after['step'],'optimizer_states':len(after['optimizer']['state']),
            'changed_tensors':len(changed),'unchanged_tensors':unchanged,'shared_initial_tensors':len(matching),'training_seconds':r['training_seconds'],
            'final_sha256':digest(run/'final.pt'),'training_log_rows':len(log)})
        del before,after,latest
    require(not complete or len(rows)==12,'Incomplete matrix')
    authenticated_json(ROOT/'diagnostics/independent_training_audit.json',{'status':'passed' if rows else 'pending','passed_for_completed_cells':True if rows else None,'complete':len(rows)==12,
        'rows':rows,'auditor_sha256':digest(__file__),'protocol_sha256':digest(ROOT/'protocol.json')})

def poses():
    import ast,torch
    sources=read_authenticated(BASE/'evaluation/input_manifest.json')['rows'];source={r['entry']['key']:r for r in sources}
    protocol=read_authenticated(ROOT/'diagnostics/protocol.json');rows=[]
    for seed,name in itertools.product(SEEDS,['descriptor128','descriptor256','pose_code24','audio_base','audio_fourier','audio_muq']):
        folder=ROOT/'diagnostics/pose_probes'/name/f'seed_{seed}';r=read_authenticated(folder/'report.json');ev=read_authenticated(folder/'evaluation.json')
        verify_artifacts(r['artifacts']);before=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False);after=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        require(before['name']==after['name']==r['name']==name and before['seed']==after['seed']==r['seed']==seed,'Pose checkpoint identity')
        require(before['step']==0 and after['steps']==r['steps']==12000,'Pose final step')
        require(set(before['model'])==set(after['model']) and any(not torch.equal(before['model'][k],v) for k,v in after['model'].items()),'Pose model changed')
        require(all(torch.isfinite(v).all().item() for v in after['model'].values()),'Pose finite checkpoint')
        log=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        require([r['step'] for r in log]==[1]+list(range(1000,12001,1000)),'Pose scheduled log')
        require(all(np.isfinite([r[k] for k in ('mse','velocity_mse','gpu_seconds')]).all() for r in log),'Finite pose training')
        require(len(ev['rows'])==40 and {r['key'] for r in ev['rows']}==set(source),'Pose cohort membership')
        errors=[];valid_counts=[]
        for item in ev['rows']:
            key=item['key'];src=source[key]
            require(digest(item['path'])==item['sha256'],'Saved pose prediction drift')
            with np.load(ROOT/'features/pose'/(key+'.npz')) as z:truth=z['pose'][src['start_frame']:src['start_frame']+2400];mask=z['mask'][src['start_frame']:src['start_frame']+2400]
            with np.load(item['path']) as z:
                require(np.array_equal(truth,z['truth']) and np.array_equal(mask,z['mask']),'Pose target/mask join')
                for mode in [k for k in z.files if k not in ('truth','mask')]:
                    pred=z[mode];valid=mask.reshape(2400,17,2).all(2)
                    require(np.isfinite(pred).all() and valid.any(),'Finite supported pose output')
                    error=np.linalg.norm(pred.reshape(2400,17,2)-truth.reshape(2400,17,2),axis=-1)[valid].mean()
                    close(float(error),item['errors'][mode]['joint_error_tl'],'Recomputed joint error')
                    require(int(valid.sum())==item['errors'][mode]['valid_joints'],'Joint support count')
                    if mode=='full':errors.append(float(error));valid_counts.append(int(valid.sum()))
            require(item['canonical_id']==src['entry']['canonical_id'],'Pose canonical join')
        rows.append({'name':name,'seed':seed,'steps':12000,'cases':40,'macro_joint_error_tl':float(np.mean(errors)),'valid_joints':sum(valid_counts)})
        adopted=protocol.get('adopted_unchanged_reports',{}).get(str(folder/'report.json'))
        if adopted:require(digest(folder/'report.json')==adopted,'Adopted completed probe identity')
        else:require(r['protocol_sha256']==digest(ROOT/'diagnostics/protocol.json'),'Pose protocol identity')
        del before,after
    checks={}
    for module,names in [('night_probes.py',['PoseDecoder','input_for','load_pose_training','evaluate_probe']),('night_training.py',['NightResident'])]:
        trees=[]
        for version in ('pose_study','pose_study_v2'):
            path=ROOT/'diagnostics'/version/'source/lightsem/experiments'/module
            tree=ast.parse(path.read_text());trees.append({node.name:ast.dump(node,include_attributes=False) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))})
        for name in names:checks[module+':'+name]=trees[0][name]==trees[1][name]
    require(all(checks.values()),'Unchanged-probe code adoption')
    authenticated_json(ROOT/'diagnostics/independent_pose_audit.json',{'passed':True,'rows':rows,'adoption_ast_checks':checks,'auditor_sha256':digest(__file__),
        'method':'All12final probe checkpoints and logs; exact raw-prepared pose/confidence joins for40VALcases; independently recomputed joint errors; AST equality for all adopted completed probe definitions.'})

def rhythm():
    import hashlib,mir_eval
    source=read_authenticated(ROOT/'rhythm/guitarset/source.json');report=read_authenticated(ROOT/'rhythm/report.json');manifest=read_authenticated(ROOT/'rhythm/input_manifest.json')
    wavs={p.stem.removesuffix('_mic'):p for p in (ROOT/'rhythm/guitarset').rglob('*_mic.wav')}
    jams={p.stem:p for p in (ROOT/'rhythm/guitarset').rglob('*.jams')}
    originals=source['verified_original_files'];require(len(originals)==360 and len({r['key'] for r in originals})==360,'Original corpus membership')
    for r in originals:
        require(hashlib.md5(wavs[r['key']].read_bytes()).hexdigest()==r['audio_md5'],'Original WAV integrity')
        require(hashlib.md5(jams[r['key']].read_bytes()).hexdigest()==r['jams_md5'],'Original JAMS integrity')
    excluded={'04_BN3-154-E_comp','04_Jazz1-200-B_comp'};require({r['key'] for r in report['rows']}==set(wavs)-excluded,'Fixed rhythm exclusions')
    require(len(report['rows'])==358 and {r['key'] for r in report['rows']}=={r['key'] for r in manifest['rows']},'Rhythm output membership')
    deltas={};continuity={}
    for r in report['rows']:
        require(digest(r['path'])==r['sha256'],'Rhythm prepared drift')
        with np.load(r['path']) as z:
            truth=mir_eval.beat.trim_beats(z['beats'])
            ann=json.loads(jams[r['key']].read_text());beat_ann=next(a for a in ann['annotations'] if a['namespace']=='beat_position')
            require(np.array_equal(z['beats'],np.array([v['time'] for v in beat_ann['data']])),'Real annotation event join')
            for mode in ('constant_grid','plp'):
                pred=mir_eval.beat.trim_beats(z[mode]);c=mir_eval.beat.continuity(truth,pred)
                close(float(mir_eval.beat.f_measure(truth,pred)),r['scores'][mode]['f1'],'Recomputed rhythm F1')
                close(float(c[1]),r['scores'][mode]['cmlt'],'Recomputed continuity')
        deltas.setdefault(r['lead_sheet'],[]).append(r['scores']['plp']['f1']-r['scores']['constant_grid']['f1'])
        continuity.setdefault(r['lead_sheet'],[]).append(r['scores']['plp']['cmlt']-r['scores']['constant_grid']['cmlt'])
    require(len(deltas)==30,'Lead-sheet independence units')
    delta=bootstrap([np.mean(v) for v in deltas.values()]);cont=bootstrap([np.mean(v) for v in continuity.values()])
    close(delta['mean'],report['f1_delta']['mean'],'Rhythm paired bootstrap mean');close(delta['ci95'],report['f1_delta']['ci95'],'Rhythm paired bootstrap interval')
    require(report['gates']=={'beat_f1':delta['mean']>=.05,'continuity':cont['mean']>=-.02,'real_annotations':True,'integrity':True},'Rhythm frozen gates')
    authenticated_json(ROOT/'diagnostics/independent_rhythm_audit.json',{'passed':True,'original_files':720,'scored_clips':358,'lead_sheets':30,
        'f1_delta':delta,'cmlt_delta':cont,'gates':report['gates'],'auditor_sha256':digest(__file__)})

def evaluation():
    from dataclasses import replace
    from lightsem.experiments.night_render import compose_scene,ExpressionConfig
    from lightsem.conductor.rig import canonical_rig
    from sklearn.metrics import average_precision_score
    p=read_authenticated(ROOT/'protocol.json');m,n=base_context();source=read_authenticated(BASE/'evaluation/input_manifest.json')['rows']
    prepared=read_authenticated(ROOT/'evaluation/input_manifest.json')['rows'];decision=read_authenticated(ROOT/'evaluation/report.json')
    cfg=ExpressionConfig(**read_authenticated(ROOT/'expression_config.json'));rig=canonical_rig();cells=[]
    for seed,arm in itertools.product(SEEDS,ARMS):
        family=arm.split('_')[0];folder=ROOT/'evaluation'/arm/f'seed_{seed}'
        inf=read_authenticated(folder/'inference_report.json');video=read_authenticated(folder/'video_report.json');summary=read_authenticated(folder/'summary.json')
        require(len(inf['rows'])==40 and len(video['rows'])==8,'Output membership count')
        require({r['key'] for r in inf['rows']}=={r['entry']['key'] for r in source},'Output case membership')
        require({r['key'] for r in video['rows']}==set(p['evaluation']['render_entries']),'Video case membership')
        saved={r['key']:r for r in inf['rows']};video_by_key={r['key']:r for r in video['rows']};checked_rows=[];max_es_error=0.;remaps=0
        for src in source:
            key=src['entry']['key'];r=saved[key];require(r['checkpoint_sha256']==inf['checkpoint_sha256']==digest(ROOT/'runs'/arm/f'seed_{seed}'/'final.pt'),'Evaluated checkpoint')
            require(r['input_sha256']==src['sha256'] and r['canonical_id']==src['entry']['canonical_id'],'Target join')
            require(r['arm']==arm and r['seed']==seed and r['protocol_sha256']==digest(ROOT/'protocol.json'),'Inference identity')
            verify_artifacts(r['artifacts']);inp=prepared[family+'/'+key]
            require(r['conditioning_inputs']==inp['files'] and r['donor_key']==inp['donor_key'] and r['shift_frames']==inp['shift_frames'],'Conditioning join')
            with np.load(src['path']) as z:truth=(z['y'].astype(np.float32)-np.asarray(n['y_center'],np.float32))/np.asarray(n['y_scale'],np.float32);event_truth=z['b']
            errors={};aps={}
            for mode in p['evaluation']['modes']:
                with np.load(folder/key/(mode+'.npz')) as z:draws=z['standardized'];events=z['event_scores'];dmx=z['dmx'];seeds=z['noise_seeds']
                expected=np.arange(101,107)+(1000 if mode=='noise2' else 0)
                require(np.array_equal(seeds,expected),'Noise membership')
                require(draws.shape==(6,2400,21) and events.shape==(2400,3) and np.isfinite(draws).all() and np.isfinite(events).all(),'Finite inference shapes')
                require(dmx.shape==(2400,512) and dmx.dtype==np.uint8,'Emitted DMX shape/type')
                recomputed=score(draws[:,:,4:],truth[:,4:]);stored=r['metrics']['energy_score'][mode]
                close(recomputed,stored,'Energy score',atol=1e-9);max_es_error=max(max_es_error,abs(recomputed-stored))
                errors[mode]=alignment(draws.mean(0),truth,n['target_names']);close(errors[mode],r['metrics']['alignment_errors'][mode],'Alignment')
                aps[mode]=[float(average_precision_score(event_truth[:,j],events[:,j])) if 0<event_truth[:,j].sum()<len(event_truth) else None for j in range(3)]
                for a,b in zip(aps[mode],r['metrics']['event_ap'][mode]):close(a,b,'Event AP')
                # Independently rerun the deterministic adapter, using saved draws only.
                frames,trace,mapping=compose_scene(draws,n['target_names'],60.,event_scores=events,config=cfg,rig=rig)
                require(np.array_equal(frames,dmx),'Emitted adapter bytes differ');remaps+=1
                if mode=='full':
                    xx=draws[:,:,4:].astype(float);yy=truth[:,4:].astype(float);refstd=yy.std(0);supported=refstd>1e-4
                    ratio=xx.std(1).mean(0)[supported]/refstd[supported];disp=r['metrics']['native_dispersion']
                    close(ratio,disp['time_std_ratio'],'Native temporal variation')
                    close(float(np.mean((ratio>=.25)&(ratio<=2))),disp['fraction_ratio_in_point25_to2'],'Native ratio support')
                    close(float(xx.std(0).mean()),disp['mean_within_audio_draw_std'],'Native draw variation')
                    if key in video_by_key:
                        v=video_by_key[key];verify_artifacts(v['artifacts'])
                        clip,_,_=compose_scene(draws[:,:800],n['target_names'],20.,event_scores=events[:800],config=cfg,rig=rig)
                        bc=byte_composition(clip,rig)
                        close(bc['spans'],v['composition']['head_dimmer_p95_p05'],'Encoded dimmer span')
                        require(bc['longest']==v['off_intervals']['longest_off_frames'],'Encoded off intervals')
                        require(int((bc['spans']>=.4).sum())==v['composition']['heads_with_span_ge_point4'],'Brightness qualifying heads')
                        require(sum(n>=6 for n in bc['longest'])==v['off_intervals']['heads_with_off_at_least_150ms'],'Off qualifying heads')
                        independent,_,_=compose_scene(draws[:,:800],n['target_names'],20.,event_scores=events[:800],config=replace(cfg,composition='independent'),rig=rig)
                        close(byte_composition(independent,rig)['correlation'],v['independent']['unrelated_velocity_abs_correlation'],'Independent group correlation')
            aa=(errors['donor_clock_fixed']-errors['full'])/errors['donor_clock_fixed'] if errors['donor_clock_fixed']>1e-12 else None
            la=(errors['shift_four_bars']-errors['full'])/errors['shift_four_bars'] if errors['shift_four_bars']>1e-12 else None
            ap=[a-b for a,b in zip(aps['full'],aps['beat_only6']) if a is not None]
            ea=float(np.mean(ap)) if ap else None
            close(aa,r['metrics']['alignment_advantage'],'Alignment advantage');close(la,r['metrics']['local_alignment_advantage'],'Local advantage');close(ea,r['metrics']['event_ap_advantage'],'Event advantage')
            checked_rows.append(r)
        close(np.mean([r['metrics']['energy_score']['full'] for r in checked_rows]),summary['mean_energy_score'],'Mean ES')
        gates=output_gates([r['metrics'] for r in checked_rows],video['rows'],p['thresholds'])
        require(gates==summary['gates'] and all(gates.values())==summary['all_gates_pass'],'Independent gate decision')
        for key,metric in [('alignment','alignment_advantage'),('local_alignment','local_alignment_advantage'),('event_ap_advantage','event_ap_advantage')]:
            values=[r['metrics'][metric] for r in checked_rows if r['metrics'][metric] is not None]
            b=bootstrap(values);close(b['mean'],summary[key]['mean'],'Bootstrap mean');close(b['ci95'],summary[key]['ci95'],'Bootstrap interval')
        cells.append({'arm':arm,'seed':seed,'cases':len(checked_rows),'modes_per_case':9,'dmx_remaps_exact':remaps,'max_energy_score_error':max_es_error,'checkpoint_sha256':inf['checkpoint_sha256']})
        print(json.dumps({'stage':'independent_evaluation_audit','arm':arm,'seed':seed,'cases':40}),flush=True)
    eligible=[a for a in ARMS if all(read_authenticated(ROOT/'evaluation'/a/f'seed_{s}'/'summary.json')['all_gates_pass'] for s in SEEDS)]
    require(eligible==decision['eligible_arms'],'Selection eligible membership')
    chosen=min(eligible,key=lambda a:np.mean([read_authenticated(ROOT/'evaluation'/a/f'seed_{s}'/'summary.json')['mean_energy_score'] for s in SEEDS])) if eligible else None
    require(chosen==decision['chosen_arm'],'Selection policy')
    authenticated_json(ROOT/'diagnostics/independent_evaluation_audit.json',{'passed':True,'cells':cells,'chosen_arm':chosen,'auditor_sha256':digest(__file__),
        'method':'Independent cdist/pdist energy score; explicit moving means; target/checkpoint/noise/input joins; bootstrap reproduction; deterministic DMX byte reconstruction. No model inference and no gate changes.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inputs','training','evaluation','poses','rhythm']);ap.add_argument('--complete',action='store_true');a=ap.parse_args()
    if a.stage=='inputs':preflight()
    elif a.stage=='training':training(a.complete)
    elif a.stage=='evaluation':evaluation()
    elif a.stage=='poses':poses()
    else:rhythm()
