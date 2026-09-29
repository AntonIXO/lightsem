"""Opt-in execution continuation only. No import-time data or scientific runs."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys, time, zipfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE_PATH=HERE/'run_palette_history_secondary_v1.py'
BASE_SHA='95d0f59160439ee153ef532235c25320870f113eda7a4d930438811d12ce5dbb'
BASE_REG_SHA='42f3a1faa773603ee5e55fb34c461264cbdc834c58c33d9deb7611f97ad8c153'
MAP_SHA='f761df59931d83ef432359dc340cc093abf72f22ba09a0162429a804bd65df03'
AMENDMENT_SHA='9e329c257e894df4e0bdbbe8506e1a4566a16468f10a4c5d0f5c016dfb0a840b'
PARTIAL_SHA='67e49eafa0331d06d553090d7e4559ebecba07e315ada2bb77825192ba6ccca3'
ID='palette-history-continuation-20260922-v1'
PROTOCOL=HERE/'PALETTE_CONTINUATION_PROTOCOL_V1.json'
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1')


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}


def checked(value):
    require(isinstance(value,dict) and {'path','sha256'}<=value.keys(),'Missing path/hash reference')
    path=Path(value['path'])
    require(path.is_file() and sha(path)==value['sha256'],'Changed referenced artifact: '+str(path))
    return path


def array_hash(a):
    a=np.ascontiguousarray(a)
    h=hashlib.sha256();h.update(str(a.shape).encode());h.update(str(a.dtype).encode());h.update(a.tobytes())
    return h.hexdigest()


def cell_key(row):return row['role'],row['seed'],row['recording_id']


def base_module():
    require(sha(BASE_PATH)==BASE_SHA,'Frozen base driver changed')
    name='palette_continuation_base_v1'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,BASE_PATH)
        mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name]


def read_npz(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        require(len(names)==len(set(names)) and z.testzip() is None,'Corrupt/duplicate NPZ members')
        require(all(n.endswith('.npy') and '/' not in n for n in names),'Unexpected NPZ member')
    with np.load(path,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
    require(all(not x.dtype.hasobject for x in values.values()),'Object arrays not allowed')
    return values


def expected_cells(receipts,eventreg):
    result=[]
    for receipt in sorted(receipts,key=lambda r:(('selected','final').index(r['checkpoint_role']),r['seed'])):
        for old in receipt['records']:
            role,seed,rid=receipt['checkpoint_role'],receipt['seed'],old['recording_id']
            result.append(dict(role=role,seed=seed,recording_id=rid,canonical_id=old['canonical_id'],
                original_prediction=old['artifact'],target_sha256=old['target_sha256'],
                donor_id=eventreg['donor_map'][rid],checkpoint=receipt['checkpoint'],
                filename=f'{role}_s{seed}_{hashlib.sha256(rid.encode()).hexdigest()[:20]}.npz'))
    require(len(result)==186 and len({cell_key(r) for r in result})==186,'Expected186 unique cells')
    return result


def metadata(cell,cues,b,producer=None):
    result=dict(role=cell['role'],seed=cell['seed'],recording_id=cell['recording_id'],
        original_prediction_sha256=cell['original_prediction']['sha256'],
        checkpoint_sha256=cell['checkpoint']['sha256'],donor_id=cell['donor_id'],
        cues_sha256=b.tr.json_hash(cues.tolist()))
    if producer is not None:result['execution_producer']=producer
    return result


def validate_bundle(values,cues,expected_metadata,b):
    keys={'cues','metadata_json'}|{f'{d}__{n}' for d in b.DRAWS for n in (*b.CONDITIONS,'uniform_noise')}
    require(set(values)==keys,'Wrong prediction archive keys')
    require(values['cues'].dtype==np.uint16 and np.array_equal(values['cues'],cues),'Cue dtype/inventory changed')
    b.validate_cues(cues)
    require(values['metadata_json'].ndim==0 and values['metadata_json'].dtype.kind=='U'
            and json.loads(str(values['metadata_json']))==expected_metadata,'Prediction metadata changed')
    for draw in b.DRAWS:
        branches=b.validate_saved_draw(values,draw,cues)
        for ids in branches.values():
            held=ids[cues][np.searchsorted(cues,np.arange(640),side='right')-1]
            require(np.array_equal(ids,held),'Palette changes outside fixed cues')
    return True


def validate_partition(partial,expected,base_reg,calibration):
    require(partial.get('status')=='interrupted_not_complete' and partial.get('audit_status')=='all_101_passed',
            'Partial audit did not pass')
    require(partial['base_registration']['sha256']==base_reg['sha256'] and partial['base_driver']['sha256']==BASE_SHA
            and partial['calibration']['sha256']==calibration['sha256'],'Partial producer/calibration changed')
    require((partial['expected_cells'],partial['present_cells'],partial['missing_cells'])==(186,101,85),'Partial counts changed')
    require(len(partial['complete_inventory'])==186 and len(partial['validated_present'])==101
            and len(partial['missing_inventory'])==85,'Partial row counts changed')
    fields=('role','seed','recording_id','canonical_id','filename','original_prediction','target_sha256','donor_id')
    for i,(row,cell) in enumerate(zip(partial['complete_inventory'],expected)):
        require(all(row[k]==cell[k] for k in fields) and row['checkpoint_sha256']==cell['checkpoint']['sha256']
                and row['present']==(i<101),'Partial expected inventory differs')
    for rows,cells in [(partial['validated_present'],expected[:101]),(partial['missing_inventory'],expected[101:])]:
        require([cell_key(r) for r in rows]==[cell_key(c) for c in cells],'Partial partition differs')
    require(Path(partial['original_directory']).resolve()==ROOT/'predictions_v1','Wrong original directory')
    require(not (ROOT/'predictions_v1/complete.json').exists(),'Original interrupted directory has complete marker')
    names={p.name for p in (ROOT/'predictions_v1').iterdir()}
    require(names=={c['filename'] for c in expected[:101]},'Original partial directory has missing/extra entries')


def proof_flags(proof,head):
    if proof.get('status')=='diagnostic_complete':
        require(head=='donor','First fixture proof only permits failed donor head')
        require(proof['comparisons']['donor_cpu_repeat']['exact_equal'] is True,'CPU repeat not exact')
        cpu=proof['comparisons'][head]['cpu_vs_saved']
        gpu=proof['comparisons'][head]['cuda_vs_saved']
        cues=proof['cues'][head]
        return (cpu['allclose_1e5'] is False and gpu['exact_equal'] is True and gpu['all_finite'] is True
                and cues['cpu_matches_saved'] is True and cues['cuda_matches_saved'] is True)
    require(proof.get('status')=='strict_roundoff_proved','Unknown proof status')
    return (head in proof['original_guard_failed_conditions'] and proof['cpu_repeat_exact'][head] is True
        and proof['cpu_vs_saved'][head]['allclose_1e5'] is False
        and proof['cuda_vs_saved'][head]['exact_equal'] is True and proof['cuda_vs_saved'][head]['all_finite'] is True
        and proof['cpu_cues_exact'][head] is True and proof['cuda_cues_exact'][head] is True)


def cues_from_logits(logits,b):
    probability=torch.sigmoid(torch.from_numpy(np.asarray(logits)[None]).float()).numpy()
    return np.unique(np.r_[0,b.tr._cue_indices(probability,.5)[0]]).astype(np.int64)


def load_proofs(mapping,expected,base_reg,b):
    require(mapping.get('status')=='PROPOSAL_ONLY_NOT_AUTHORIZED' and mapping['map_count']==6
            and len(mapping['exceptions'])==6 and mapping['no_resumed_science'] is True,'Wrong fixed proof map')
    require(mapping['registration']==base_reg and mapping['original_driver']==ref(BASE_PATH),'Proof base source changed')
    checked(mapping['preflight_plan'])
    normal={cell_key(c):c for c in expected[101:]};result={};cell_seen=set()
    for entry in mapping['exceptions']:
        key=cell_key(entry);require(key in normal and key not in cell_seen,'Unknown/duplicate exception cell');cell_seen.add(key)
        cell=normal[key]
        require(entry['order_index']==next(i for i,c in enumerate(expected) if cell_key(c)==key)
                and entry['checkpoint']==cell['checkpoint'] and entry['original_prediction']==cell['original_prediction']
                and entry['donor_id']==cell['donor_id'],'Proof cell binding changed')
        proof=json.loads(checked(entry['proof']).read_text())
        require(proof['checkpoint']==entry['checkpoint'] and proof['original_prediction']==entry['original_prediction']
                and proof['input_hashes']==entry['input_hashes'],'Proof evidence mismatch')
        sources=proof.get('sources',proof.get('source_bindings'))
        require(set(sources)=={cell['recording_id'],cell['donor_id']},'Proof source inventory changed')
        for source in sources.values():
            for reference in source.values():checked(reference)
        for h in ('model','donor'):
            if proof.get('status')=='diagnostic_complete':
                gpu=proof['comparisons'][h]['cuda_vs_saved']
                require(proof['cues'][h]['cpu_matches_saved'] is True and proof['cues'][h]['cuda_matches_saved'] is True,
                        'Both fixture cue inventories need exact CPU/GPU proof')
            else:
                gpu=proof['cuda_vs_saved'][h]
                require(proof['cpu_cues_exact'][h] is True and proof['cuda_cues_exact'][h] is True,
                        'Both fixture cue inventories need exact CPU/GPU proof')
            require(gpu['exact_equal'] is True and gpu['all_finite'] is True,
                    'Both GPU heads must reproduce saved vectors bitwise')
        vectors=read_npz(checked(entry['cpu_vectors']));saved=read_npz(checked(cell['original_prediction']))
        heads=(['donor'] if proof.get('status')=='diagnostic_complete' else proof['original_guard_failed_conditions'])
        require(len(heads)==len(set(heads)) and heads and set(heads)<={'model','donor'},'Bad exception heads')
        for head in heads:
            require(proof_flags(proof,head),'GPU bitwise/exact-cue proof missing')
            vector=vectors[entry['vector_keys'][head]];reference=saved[head+'__color_logit']
            require(vector.dtype==np.float32 and vector.shape==(640,) and np.isfinite(vector).all()
                    and array_hash(vector)==entry['cpu_color_logit_hashes'][head],'CPU proof vector changed')
            require(reference.dtype==np.float32 and reference.shape==(640,) and np.isfinite(reference).all(),'Saved vector invalid')
            cues=b.cue_indices(saved[head+'__generated_color_cues'])
            require(np.array_equal(cues_from_logits(vector,b),cues),'Proof vector changes cues')
            result[(*key,head)]=dict(entry=entry,head=head,cpu_vector_hash=array_hash(vector),
                saved_vector_hash=array_hash(reference),cue_hash=b.tr.json_hash(cues.tolist()),
                proof=entry['proof'],gpu_bitwise_proved=True)
    require(len(result)==6,'Exactly six failed-head exceptions are required')
    return result


def replay_guard(actual,saved,*,cell,head,inputs,proofs,cues,b):
    actual,saved=np.asarray(actual),np.asarray(saved)
    require(actual.dtype==saved.dtype==np.float32 and actual.shape==saved.shape==(640,)
            and np.isfinite(actual).all() and np.isfinite(saved).all(),'Replay requires finite FP32[640]')
    close=bool(np.allclose(actual,saved,rtol=1e-5,atol=1e-5))
    key=(*cell_key(cell),head);proof=proofs.get(key)
    if proof is None:
        require(close,'Unregistered replay difference: '+str(key))
        return {'mode':'original_allclose_1e-5','head':head,'cpu_vector_sha256':array_hash(actual)}
    entry=proof['entry']
    require(entry['checkpoint']==cell['checkpoint'] and entry['original_prediction']==cell['original_prediction']
            and entry['donor_id']==cell['donor_id'] and entry['input_hashes']==inputs,'Replay proof cell/input drift')
    require(proof['gpu_bitwise_proved'] is True and array_hash(actual)==proof['cpu_vector_hash']
            and array_hash(saved)==proof['saved_vector_hash'],'Replay CPU/saved vector not exact proof')
    require(np.array_equal(cues_from_logits(actual,b),cues)
            and b.tr.json_hash(cues.tolist())==proof['cue_hash'],'Replay exact cues changed')
    return {'mode':'exact_registered_cpu_vector_with_gpu_bitwise_proof','head':head,
            'cpu_vector_sha256':array_hash(actual),'proof':proof['proof'],'original_allclose':close}


def producer_refs(reg_path,reg):
    return {'v1_partial':{'registration':reg['base_registration'],'source':reg['base_driver']},
            'continuation_v1':{'registration':ref(reg_path),'source':ref(__file__)}}


def validate_native_continuation_receipt(native,reg):
    require(isinstance(native,dict) and native.get('id')=='dancing-light-generative-20260903'
            and isinstance(native.get('updates'),list),'Native continuation receipt is not a successful Problem update')
    names=('protocol','wrapper','tests','base_registration','calibration','partial_receipt','exception_map','amendment')
    tokens=[ID,reg['vault_experiment_id'],'RETROSPECTIVE V1 EXECUTION DEVIATION',reg['execution_deviation']['artifact']['sha256']]+[reg[name]['sha256'] for name in names]
    require(any(isinstance(u,dict) and isinstance(u.get('text'),str)
                and all(token in u['text'] for token in tokens) for u in native['updates']),
            'Native update does not bind this complete continuation source/inventory')


def load_authorized(path,phase):
    reg=json.loads(Path(path).read_text())
    require(reg.get('continuation_id')==ID and reg.get('authorized') is True
            and reg.get('registered_before_continuation') is True and phase in reg.get('authorized_phases',[]),
            'Continuation authorization missing')
    require(reg['base_registration']['sha256']==BASE_REG_SHA and reg['base_driver']['sha256']==BASE_SHA
            and reg['exception_map']['sha256']==MAP_SHA and reg['partial_receipt']['sha256']==PARTIAL_SHA,
            'Base/partial/proof inventory changed')
    require(checked(reg['protocol']).resolve()==PROTOCOL.resolve() and checked(reg['wrapper']).resolve()==Path(__file__).resolve(),
            'Continuation protocol/wrapper binding mismatch')
    require(reg['amendment']['sha256']==AMENDMENT_SHA,'Wrong execution amendment')
    amendment=json.loads(checked(reg['amendment']).read_text())
    require(amendment['base_registration']==reg['base_registration'] and amendment['base_driver']==reg['base_driver']
            and amendment['calibration']==reg['calibration'] and amendment['partial_reuse']==reg['partial_receipt']
            and amendment['exact_exception_map']==reg['exception_map'] and amendment['expected_exception_count']==6,
            'Amendment antecedents differ')
    checked(amendment['failure']);checked(amendment['head_preflight_complete'])
    disclosure=reg.get('execution_deviation',{})
    require(disclosure.get('disclosure_type')=='RETROSPECTIVE V1 EXECUTION DEVIATION'
            and disclosure.get('fully_prospective_vault_compliance') is False
            and disclosure.get('v1_source_test_binding_vault_registered_before_101') is False
            and disclosure.get('v1_calibration_hash_vault_registered_before_101') is False
            and disclosure.get('v1_definition_start_experiment_succeeded') is True,
            'Missing or softened V1 execution-deviation disclosure')
    deviation=json.loads(checked(disclosure['artifact']).read_text())
    require(disclosure['artifact']['sha256']=='df755a9bc9746e74f4270721f36df197dad1142793af70bf68346be241c56205'
            and deviation.get('experiment_id')==reg['vault_experiment_id']
            and deviation.get('original_start_experiment_succeeded') is True
            and deviation.get('pre_calibration_executable_source_vault_update_succeeded') is False
            and deviation.get('pre_prediction_calibration_vault_update_succeeded') is False
            and deviation.get('late_recording_is_not_preregistration') is True,
            'Execution-deviation artifact changed or softened')
    for reference in deviation['affected_receipts'].values():checked(reference)
    checked(reg['tests']);cpu=json.loads(checked(reg['cpu_test_receipt']).read_text())
    require(cpu.get('status')=='passed' and cpu.get('synthetic_only') is True and cpu.get('real_data_access') is False
            and cpu.get('tests_failed')==0 and cpu.get('tests_passed',0)>0
            and cpu['wrapper_sha256']==reg['wrapper']['sha256'] and cpu['tests_sha256']==reg['tests']['sha256']
            and cpu['protocol_sha256']==reg['protocol']['sha256'] and cpu['amendment_sha256']==AMENDMENT_SHA,'Continuation CPU test receipt mismatch')
    checked(cpu['test_log']);checked(cpu['junit'])
    vault=json.loads(checked(reg['vault_receipt']).read_text())
    require(vault.get('status')=='registered' and vault.get('continuation_id')==ID
            and vault.get('experiment_id')==reg['vault_experiment_id'],'Continuation vault identity mismatch')
    for name in ('protocol','wrapper','base_registration','calibration','partial_receipt','exception_map','amendment'):
        require(vault[name+'_sha256']==reg[name]['sha256'],'Vault binding mismatch: '+name)
    native=json.loads(checked(vault['native_vault_receipt']).read_text())
    validate_native_continuation_receipt(native,reg)
    checked(reg['base_driver']);base_path=checked(reg['base_registration'])
    b=base_module();loaded=b.load_authorized(base_path,'predict' if phase=='resume-predict' else 'score')
    base_reg,eventreg,manifest,transforms,store,baselines,receipts,train,dev=loaded
    require(reg['base_registration']==ref(base_path),'Noncanonical base registration ref')
    calibration_path=checked(reg['calibration'])
    calibration_receipt=b.check_receipt(calibration_path,base_path)
    cal=b.pm.PaletteCalibration(**calibration_receipt['calibration'])
    b.pm._validate_calibration(cal,b.pm.palette_prototypes_hash(transforms['palette_prototypes']))
    b.validate_calibration_inventory(cal,train)
    partial=json.loads(checked(reg['partial_receipt']).read_text())
    expected=expected_cells(receipts,eventreg)
    validate_partition(partial,expected,reg['base_registration'],reg['calibration'])
    proofs=load_proofs(json.loads(checked(reg['exception_map']).read_text()),expected,reg['base_registration'],b)
    require(sorted([list(k) for k in proofs])==sorted(reg['exception_head_keys']),'Registered head-exception inventory differs')
    return b,reg,loaded,cal,partial,expected,proofs


def source_cell(cell,store,transforms,eventreg,b,cache):
    rid=cell['recording_id']
    if rid not in cache:
        record=store.visual(rid);label=b.ev.freeze_labels(record,transforms)
        require(label.content_hash()==eventreg['target_hashes'][rid],'Frozen target changed')
        cache[rid]=(record,label)
    record,label=cache[rid]
    require(label.content_hash()==cell['target_sha256'],'Cell target changed')
    saved=read_npz(checked(cell['original_prediction']))
    for branch in ('model','donor'):
        require(np.array_equal(saved[branch+'__requested_times10_s'],record.times)
                and np.array_equal(saved[branch+'__source_times10_s'],label.source_times10_s),'Original clock changed')
    return record,label,saved,b.cue_indices(saved['model__generated_color_cues'])


def reused_entries(partial,expected,store,transforms,eventreg,b,cache):
    entries=[]
    for item,cell in zip(partial['validated_present'],expected[:101]):
        require(cell_key(item)==cell_key(cell),'Partial ordered cell changed')
        p=checked(item['artifact'])
        require(p.resolve()==ROOT/'predictions_v1'/cell['filename'],'Reused artifact path changed')
        record,label,saved,cues=source_cell(cell,store,transforms,eventreg,b,cache)
        values=read_npz(p);validate_bundle(values,cues,metadata(cell,cues,b),b)
        entries.append(make_entry(cell,ref(p),cues,b,'v1_partial'))
    return entries


def make_entry(cell,artifact,cues,b,producer):
    return {**{k:cell[k] for k in ('role','seed','recording_id','canonical_id','original_prediction','target_sha256','donor_id')},
        'artifact':artifact,'cues_sha256':b.tr.json_hash(cues.tolist()),'producer':producer}


def validate_combined(receipt,reg_path,reg,expected,partial,b):
    require(receipt.get('status')=='complete' and receipt.get('continuation_id')==ID
            and receipt['registration_sha256']==sha(reg_path),'Not a complete continuation receipt')
    for key in ('base_registration','base_driver','calibration','partial_receipt','exception_map','amendment'):
        require(receipt[key]==reg[key],'Combined lineage changed: '+key)
    producers=producer_refs(reg_path,reg)
    require(receipt.get('execution_deviation')==reg['execution_deviation']
            and receipt.get('fully_prospective_vault_compliance') is False,'Combined execution deviation concealed')
    require(receipt['execution_producers']==producers,'Combined execution producers changed')
    entries=receipt['records'];keys=[cell_key(e) for e in entries]
    require(len(entries)==186 and len(set(keys))==186 and set(keys)=={cell_key(e) for e in expected},'Combined union not exact186')
    original={cell_key(e):e for e in partial['validated_present']}
    expected_map={cell_key(e):e for e in expected}
    resumed_root=Path(reg['prediction_output']).resolve()
    for entry in entries:
        key=cell_key(entry);cell=expected_map[key]
        for name in ('canonical_id','original_prediction','target_sha256','donor_id'):
            require(entry[name]==cell[name],'Combined cell provenance changed')
        producer='v1_partial' if key in original else 'continuation_v1'
        require(entry['producer']==producer and entry['execution_producer']==producers[producer],'Per-entry producer changed')
        p=checked(entry['artifact'])
        if producer=='v1_partial':
            require(entry['artifact']=={'path':original[key]['artifact']['path'],'sha256':original[key]['artifact']['sha256']},'Reused bytes/path replaced')
        else:require(p.resolve()==resumed_root/cell['filename'],'Resumed artifact path changed')
    return entries


def execute(args):
    started=time.monotonic()
    b,reg,loaded,cal,partial,expected,proofs=load_authorized(args.registration,args.phase)
    _,eventreg,manifest,transforms,store,baselines,receipts,train,dev=loaded
    output=Path(args.output).resolve()
    require(output.is_relative_to(ROOT) and not output.exists(),'Exclusive bounded output required')
    registered_output=reg['prediction_output'] if args.phase=='resume-predict' else reg['score_output']
    require(str(output)==registered_output,'Output differs from continuation registration')
    b.space(ROOT,b.RESERVE)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.get_num_threads()==2,'CPU execution threads changed')
    cache={};producers=producer_refs(args.registration,reg)
    common=dict(continuation_id=ID,registration_sha256=sha(args.registration),base_registration=reg['base_registration'],
        base_driver=reg['base_driver'],calibration=reg['calibration'],partial_receipt=reg['partial_receipt'],
        exception_map=reg['exception_map'],amendment=reg['amendment'],execution_producers=producers,
        execution_deviation=reg['execution_deviation'],fully_prospective_vault_compliance=False,
        scope=json.loads(b.PROTOCOL_PATH.read_text())['study_scope'])
    if args.phase=='resume-predict':
        entries=reused_entries(partial,expected,store,transforms,eventreg,b,cache)
        output.mkdir(parents=True,exist_ok=False)
        model=None;loaded_checkpoint=None
        for cell in expected[101:]:
            b.deadline(started)
            checkpoint=cell['checkpoint']
            if loaded_checkpoint!=checkpoint['sha256']:
                if model is not None:del model
                b.ev.validate_training_receipt(checkpoint,eventreg)
                model,_=b.ev.load_bound_model(checkpoint,transforms,transforms_sha256=loaded[0]['transforms']['sha256'],
                    expected_bindings=checkpoint['bindings'],device='cpu')
                loaded_checkpoint=checkpoint['sha256']
            record,label,saved,cues=source_cell(cell,store,transforms,eventreg,b,cache)
            rid,donor=cell['recording_id'],cell['donor_id']
            b.ev.verify_donor_tempos(store,[rid,donor],eventreg['donor_tempos_bpm'])
            x=b.ev.audio_window(store,record,transforms)
            raw_donor=b.ev.audio_window(store,store.visual(donor),transforms)
            dx=b.ev.nonrhythm_donor(x,raw_donor)
            require(np.array_equal(dx[:,:86],raw_donor[:,:86]) and np.array_equal(dx[:,86:],x[:,86:]),'Frozen donor intervention changed')
            inputs={name:array_hash(value) for name,value in [('recipient_input',x),('donor_original_input',raw_donor),('donor_intervened_input',dx)]}
            with torch.no_grad(),torch.autocast(device_type='cpu',enabled=False):
                own=model(torch.from_numpy(x[None]));don=model(torch.from_numpy(dx[None]))
            guards={}
            for head,result in [('model',own),('donor',don)]:
                actual=result['chromatic_change_logit'][0].float().numpy()
                head_cues=b.cue_indices(saved[head+'__generated_color_cues'])
                if any((*cell_key(cell),h) in proofs for h in ('model','donor')):
                    require(np.array_equal(cues_from_logits(actual,b),head_cues),'Exception fixture changed a cue inventory')
                guards[head]=replay_guard(actual,saved[head+'__color_logit'],cell=cell,head=head,inputs=inputs,
                    proofs=proofs,cues=head_cues,b=b)
            require(np.array_equal(cues_from_logits(own['chromatic_change_logit'][0].float().numpy(),b),cues),'Recipient exact cues changed')
            draws=b.paired_conditions(model,own['context10'],don['context10'],cues)
            values={'cues':cues.astype(np.uint16),'metadata_json':np.array(json.dumps(metadata(cell,cues,b,producers['continuation_v1']),sort_keys=True))}
            for draw in b.DRAWS:
                for name in b.CONDITIONS:values[f'{draw}__{name}']=draws[draw][name].astype(np.uint8)
                values[f'{draw}__uniform_noise']=draws[draw]['uniform_noise']
            validate_bundle(values,cues,metadata(cell,cues,b,producers['continuation_v1']),b)
            path=output/cell['filename'];b.write_npz(path,values,ROOT)
            entry=make_entry(cell,ref(path),cues,b,'continuation_v1');entry['replay_guards']=guards
            entries.append(entry)
        for entry in entries:entry['execution_producer']=producers[entry['producer']]
        require({p.name for p in output.iterdir()}=={c['filename'] for c in expected[101:]},'Unexpected continuation outputs')
        complete={**common,'status':'complete','records':entries,'reused_count':101,'new_count':85,
            'model_forward_calls':170,'scientific_device':'cpu','gpu_used':False,'elapsed_s':time.monotonic()-started}
        validate_partition(partial,expected,reg['base_registration'],reg['calibration'])
        validate_combined(complete,args.registration,reg,expected,partial,b)
        b.deadline(started);b.write_json(output/'combined_complete.json',complete,ROOT)
        return
    # A new lineage-aware scorer uses original V1 methods, never rewrites V1 receipts.
    prediction=json.loads(Path(args.predictions).read_text())
    require(Path(args.predictions).resolve()==Path(reg['prediction_output'])/'combined_complete.json','Wrong combined receipt path')
    entries=validate_combined(prediction,args.registration,reg,expected,partial,b)
    expected_map={cell_key(c):c for c in expected};prototypes=np.asarray(transforms['palette_prototypes'],np.float64)
    rows=[];floors=[]
    for entry in entries:
        b.deadline(started);cell=expected_map[cell_key(entry)]
        record,label,saved,cues=source_cell(cell,store,transforms,eventreg,b,cache)
        require(entry['cues_sha256']==b.tr.json_hash(cues.tolist()),'Combined cue binding changed')
        values=read_npz(checked(entry['artifact']))
        producer=None if entry['producer']=='v1_partial' else producers['continuation_v1']
        validate_bundle(values,cues,metadata(cell,cues,b,producer),b)
        reference=b.reference_track(record,cues,prototypes)
        floor_self=b.compact_score(b.pm.compare_palette_tracks(reference,reference,prototypes,cal),reference)
        cueids=reference.token_ids[np.searchsorted(cues,np.arange(640),side='right')-1]
        quantized=b.generated_track(reference,cueids,prototypes)
        floor_quant=b.compact_score(b.pm.compare_palette_tracks(quantized,reference,prototypes,cal),reference)
        floors.append(dict(role=entry['role'],seed=entry['seed'],recording_id=entry['recording_id'],self_floor=floor_self,quantization_floor=floor_quant))
        for draw in b.DRAWS:
            conditions=b.validate_saved_draw(values,draw,cues)
            scored=b.score_draw(reference,conditions,prototypes,cal,label,cues,saved['model__color'])
            rows.append(dict(role=entry['role'],seed=entry['seed'],draw=draw,recording_id=entry['recording_id'],
                canonical_id=entry['canonical_id'],conditions=scored))
    summary=b.summarize(rows,dev);b.deadline(started)
    b.write_json(output,{**common,**summary,'status':'complete','rows':rows,'floors':floors,
        'predictions':ref(args.predictions),'elapsed_s':time.monotonic()-started},ROOT)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='phase',required=True)
    sub.add_parser('schema')
    for phase in ('resume-predict','score'):
        s=sub.add_parser(phase);s.add_argument('--registration',required=True);s.add_argument('--output',required=True)
        if phase=='score':s.add_argument('--predictions',required=True)
    args=p.parse_args()
    if args.phase=='schema':print(json.dumps({'continuation_id':ID,'protocol':ref(PROTOCOL),'wrapper':ref(__file__),'amendment':ref(HERE/'PALETTE_REPLAY_EXECUTION_AMENDMENT_V1.json')},indent=2));return
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        execute(args)


if __name__=='__main__':main()
