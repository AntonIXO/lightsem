"""Synthetic CPU continuation-contract tests. No real source arrays or outcomes."""
from copy import deepcopy
import importlib.util,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

P=Path(__file__).with_name('run_palette_history_continuation_v1.py')
spec=importlib.util.spec_from_file_location('continuation_under_test',P)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
b=m.base_module()


def cell(i=0):
    return dict(role='final',seed=0,recording_id=f'r{i}',canonical_id=f's{i}',
        original_prediction={'path':f'/old/{i}.npz','sha256':'a'*64},target_sha256='t'*64,
        donor_id=f'd{i}',checkpoint={'sha256':'c'*64},filename=f'f{i}.npz')


def guarded():
    c=cell();saved=np.full(640,-1.,np.float32);actual=saved.copy();actual[10]+=1e-3
    cues=np.array([0]);inputs={'recipient_input':'rx','donor_original_input':'dy','donor_intervened_input':'dx'}
    e={'checkpoint':c['checkpoint'],'original_prediction':c['original_prediction'],'donor_id':c['donor_id'],'input_hashes':inputs}
    proof={'entry':e,'cpu_vector_hash':m.array_hash(actual),'saved_vector_hash':m.array_hash(saved),
        'cue_hash':b.tr.json_hash(cues.tolist()),'gpu_bitwise_proved':True,'proof':{'path':'/proof','sha256':'f'*64}}
    return c,actual,saved,cues,inputs,{(*m.cell_key(c),'model'):proof}


def guard_call(parts):
    c,a,s,q,x,p=parts
    return m.replay_guard(a,s,cell=c,head='model',inputs=x,proofs=p,cues=q,b=b)


def test_original_tolerance_unchanged_without_exception():
    c,a,s,q,x,p=guarded()
    with pytest.raises(ValueError,match='Unregistered'):guard_call((c,a,s,q,x,{}))
    a=s.copy();a[10]+=1e-6
    assert guard_call((c,a,s,q,x,{}))['mode']=='original_allclose_1e-5'


def test_exact_proved_vector_only_not_larger_epsilon():
    parts=guarded()
    assert guard_call(parts)['mode'].startswith('exact_registered')
    parts[1][10]=np.nextafter(parts[1][10],np.float32(0))
    with pytest.raises(ValueError,match='not exact proof'):guard_call(parts)


@pytest.mark.parametrize('bad',['input','checkpoint','source','donor','head','role','seed','recording','saved','cue','proof'])
def test_proof_drift_fails_closed(bad):
    c,a,s,q,x,p=guarded();key=(*m.cell_key(c),'model')
    if bad=='input':x=dict(x,recipient_input='changed')
    if bad=='checkpoint':c=dict(c,checkpoint={'sha256':'d'*64})
    if bad=='source':c=dict(c,original_prediction={'path':'/x','sha256':'e'*64})
    if bad=='donor':c=dict(c,donor_id='changed')
    if bad=='head':p={(*m.cell_key(c),'donor'):p[key]}
    if bad=='role':c=dict(c,role='selected')
    if bad=='seed':c=dict(c,seed=1)
    if bad=='recording':c=dict(c,recording_id='changed')
    if bad=='saved':s=s.copy();s[0]+=.01
    if bad=='cue':q=np.array([0,10])
    if bad=='proof':p[key]['gpu_bitwise_proved']=False
    with pytest.raises(ValueError):guard_call((c,a,s,q,x,p))


@pytest.mark.parametrize('which',['nan','inf','shape','dtype'])
def test_nonfinite_bad_vector_rejected(which):
    c,a,s,q,x,p=guarded()
    if which=='nan':a[0]=np.nan
    if which=='inf':a[0]=np.inf
    if which=='shape':a=a[:-1]
    if which=='dtype':a=a.astype(np.float64)
    with pytest.raises(ValueError,match='FP32'):guard_call((c,a,s,q,x,p))


def test_proof_flags_require_actual_failed_head_and_bitwise_repeat():
    p={'status':'strict_roundoff_proved','original_guard_failed_conditions':['donor'],
       'cpu_repeat_exact':{'model':True,'donor':True},'cpu_vs_saved':{'model':{'allclose_1e5':True},'donor':{'allclose_1e5':False}},
       'cuda_vs_saved':{h:{'exact_equal':True,'all_finite':True} for h in ['model','donor']},
       'cpu_cues_exact':{'model':True,'donor':True},'cuda_cues_exact':{'model':True,'donor':True}}
    assert m.proof_flags(p,'donor');assert not m.proof_flags(p,'model')
    p['cpu_repeat_exact']['donor']=False;assert not m.proof_flags(p,'donor')
    first={'status':'diagnostic_complete'}
    with pytest.raises(ValueError,match='only permits'):m.proof_flags(first,'model')


def bundle(producer=None):
    c=cell();q=np.array([0,100,500]);meta=m.metadata(c,q,b,producer)
    v={'cues':q.astype(np.uint16),'metadata_json':np.array(json.dumps(meta))}
    for d in b.DRAWS:
        ids=np.zeros(640,np.uint8);ids[100:500]=1;ids[500:]=2
        for n in b.CONDITIONS:v[f'{d}__{n}']=np.zeros(640,np.uint8) if n=='persistence' else ids.copy()
        v[f'{d}__uniform_noise']=np.random.default_rng(d).random(len(q))
    return v,q,meta


def test_bundle_exact_noise_state_and_new_producer():
    producer={'registration':{'path':'/new','sha256':'n'*64},'source':{'path':'/wrapper','sha256':'w'*64}}
    v,q,meta=bundle(producer)
    assert m.validate_bundle(v,q,meta,b)
    with pytest.raises(ValueError,match='metadata'):m.validate_bundle(v,q,m.metadata(cell(),q,b),b)
    v['101__own_state'][1]=4
    with pytest.raises(ValueError,match='outside'):m.validate_bundle(v,q,meta,b)


@pytest.mark.parametrize('bad',['noise','range','dtype','initial','persistence','extra','cues'])
def test_bundle_content_mutations_rejected(bad):
    v,q,meta=bundle()
    if bad=='noise':v['101__uniform_noise'][0]=.99
    if bad=='range':v['101__own_state'][0]=32
    if bad=='dtype':v['101__own_state']=v['101__own_state'].astype(np.int64)
    if bad=='initial':v['101__null_state'][0]=1
    if bad=='persistence':v['101__persistence'][500:]=1
    if bad=='extra':v['extra']=np.array(0)
    if bad=='cues':v['cues']=v['cues'].astype(np.int64)
    with pytest.raises((ValueError,b.tr.ContractError)):m.validate_bundle(v,q,meta,b)


def synthetic_lineage(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();old=root/'predictions_v1';old.mkdir();new=root/'resumed';new.mkdir()
    monkeypatch.setattr(m,'ROOT',root)
    regpath=tmp_path/'newreg.json';regpath.write_text('{}')
    reg={'base_registration':{'path':'/oldreg','sha256':m.BASE_REG_SHA},'base_driver':{'path':str(m.BASE_PATH),'sha256':m.BASE_SHA},
         'calibration':{'path':'/oldcal','sha256':'k'*64},'partial_receipt':{'path':'/partial','sha256':m.PARTIAL_SHA},
         'exception_map':{'path':'/map','sha256':m.MAP_SHA},'amendment':{'path':'/amend','sha256':m.AMENDMENT_SHA},
         'prediction_output':str(new),'execution_deviation':{'disclosure_type':'RETROSPECTIVE V1 EXECUTION DEVIATION'}}
    expected=[cell(i) for i in range(186)];producers=m.producer_refs(regpath,reg);present=[];entries=[];inventory=[]
    for i,c in enumerate(expected):
        path=(old if i<101 else new)/c['filename'];path.write_bytes(str(i).encode())
        producer='v1_partial' if i<101 else 'continuation_v1'
        e=m.make_entry(c,m.ref(path),np.array([0]),b,producer);e['execution_producer']=producers[producer]
        entries.append(e)
        inv={k:c[k] for k in ('role','seed','recording_id','canonical_id','filename','original_prediction','target_sha256','donor_id')}
        inv.update(checkpoint_sha256=c['checkpoint']['sha256'],present=i<101);inventory.append(inv)
        if i<101:present.append({**inv,'artifact':m.ref(path)})
    partial={'status':'interrupted_not_complete','audit_status':'all_101_passed','base_registration':reg['base_registration'],
        'base_driver':reg['base_driver'],'calibration':reg['calibration'],'expected_cells':186,'present_cells':101,'missing_cells':85,
        'complete_inventory':inventory,'validated_present':present,'missing_inventory':inventory[101:],'original_directory':str(old)}
    receipt={**{k:reg[k] for k in ('base_registration','base_driver','calibration','partial_receipt','exception_map','amendment')},
        'status':'complete','continuation_id':m.ID,'registration_sha256':m.sha(regpath),'execution_producers':producers,'records':entries,
        'execution_deviation':reg['execution_deviation'],'fully_prospective_vault_compliance':False}
    return receipt,regpath,reg,expected,partial


def test_disjoint186_union_and_original_interrupted_marker(tmp_path,monkeypatch):
    r,rp,g,e,p=synthetic_lineage(tmp_path,monkeypatch)
    m.validate_partition(p,e,g['base_registration'],g['calibration'])
    assert len(m.validate_combined(r,rp,g,e,p,b))==186
    assert not (m.ROOT/'predictions_v1/complete.json').exists()
    (m.ROOT/'predictions_v1/complete.json').write_text('{}')
    with pytest.raises(ValueError,match='complete marker'):m.validate_partition(p,e,g['base_registration'],g['calibration'])


@pytest.mark.parametrize('bad',['missing','duplicate','producer','base_calibration','registration','artifact','source','deviation'])
def test_combined_lineage_mutations_rejected(tmp_path,monkeypatch,bad):
    r,rp,g,e,p=synthetic_lineage(tmp_path,monkeypatch)
    if bad=='missing':r['records'].pop()
    if bad=='duplicate':r['records'][-1]=r['records'][0]
    if bad=='producer':r['records'][0]['producer']='continuation_v1'
    if bad=='base_calibration':r['calibration']={'path':'/newcal','sha256':'x'*64}
    if bad=='registration':r['registration_sha256']=m.BASE_REG_SHA
    if bad=='artifact':r['records'][0]['artifact']['sha256']='x'*64
    if bad=='source':r['records'][-1]['original_prediction']={'path':'/oops','sha256':'x'*64}
    if bad=='deviation':r['fully_prospective_vault_compliance']=True
    with pytest.raises(ValueError):m.validate_combined(r,rp,g,e,p,b)


def test_unauthorized_fails_before_any_dataset_access(tmp_path,monkeypatch):
    path=tmp_path/'reg.json';path.write_text(json.dumps({'continuation_id':m.ID,'authorized':False}))
    monkeypatch.setattr(m,'checked',lambda *a:pytest.fail('Artifact opened before authorization'))
    monkeypatch.setattr(m,'base_module',lambda:pytest.fail('Base data loader reached'))
    with pytest.raises(ValueError,match='authorization'):m.load_authorized(path,'resume-predict')


def test_archive_roundtrip_and_corruption(tmp_path):
    path=tmp_path/'a.npz';v,_,_=bundle()
    np.savez_compressed(path,**v)
    assert set(m.read_npz(path))==set(v)
    path.write_bytes(b'not a zip')
    with pytest.raises(Exception):m.read_npz(path)


@pytest.mark.parametrize('bad',['error_string','wrong_problem','missing_binding','split_updates'])
def test_native_receipt_cannot_be_error_or_unrelated_update(bad):
    names=('protocol','wrapper','tests','base_registration','calibration','partial_receipt','exception_map','amendment')
    reg={'vault_experiment_id':'exp-synthetic','execution_deviation':{'artifact':{'sha256':'d'*64}},**{name:{'sha256':(str(i)*64)} for i,name in enumerate(names)}}
    tokens=[m.ID,reg['vault_experiment_id'],'RETROSPECTIVE V1 EXECUTION DEVIATION',reg['execution_deviation']['artifact']['sha256']]+[reg[name]['sha256'] for name in names]
    native={'id':'dancing-light-generative-20260903','updates':[{'text':' '.join(tokens)}]}
    m.validate_native_continuation_receipt(native,reg)
    if bad=='error_string':native='validation error problem_id rejected'
    if bad=='wrong_problem':native['id']='wrong'
    if bad=='missing_binding':native['updates'][0]['text']=' '.join(tokens[:-1])
    if bad=='split_updates':native['updates']=[{'text':token} for token in tokens]
    with pytest.raises(ValueError):m.validate_native_continuation_receipt(native,reg)


def test_amendment_and_exact_proof_map_static_hashes():
    assert m.AMENDMENT_SHA=='9e329c257e894df4e0bdbbe8506e1a4566a16468f10a4c5d0f5c016dfb0a840b'
    assert m.MAP_SHA=='f761df59931d83ef432359dc340cc093abf72f22ba09a0162429a804bd65df03'
    assert m.PARTIAL_SHA=='67e49eafa0331d06d553090d7e4559ebecba07e315ada2bb77825192ba6ccca3'
    # Hash is dtype/shape/content sensitive and matches exact diagnostic contract.
    import hashlib
    a=np.arange(6,dtype=np.float32).reshape(2,3)
    wanted=hashlib.sha256(str(a.shape).encode()+str(a.dtype).encode()+a.tobytes()).hexdigest()
    assert m.array_hash(a)==wanted
    assert m.array_hash(a)!=m.array_hash(a.astype(np.float64))
    assert m.array_hash(a)!=m.array_hash(a.reshape(3,2))


def test_execution_deviation_artifact_is_explicitly_bound():
    text=P.read_text()
    assert 'df755a9bc9746e74f4270721f36df197dad1142793af70bf68346be241c56205' in text
    assert "fully_prospective_vault_compliance=False" in text
