#!/usr/bin/env python3
"""Independent read-only validation of interrupted V1 predictions; never infer/score."""
from pathlib import Path
import hashlib, importlib.util, json, shutil, sys, time, zipfile
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/palette-history-secondary-20260922-v1')
PARTIAL=ROOT/'predictions_v1'
REG=HERE/'REGISTRATION_AUTHORIZED_V1.json'
DRIVER=HERE/'run_palette_history_secondary_v1.py'
FAILURE=HERE/'PREDICTION_FAILURE_V1.json'
CAL=ROOT/'calibration_v1.json'
RECEIPT=ROOT/'PARTIAL_INTERRUPTED_RECEIPT_V1.json'
EXPECTED_REG='42f3a1faa773603ee5e55fb34c461264cbdc834c58c33d9deb7611f97ad8c153'
EXPECTED_DRIVER='95d0f59160439ee153ef532235c25320870f113eda7a4d930438811d12ce5dbb'


def require(condition,message):
    if not condition:raise RuntimeError(message)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def ref(p):return {'path':str(p),'sha256':sha(p),'bytes':Path(p).stat().st_size}


def array_sha(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read_npz(p):
    with zipfile.ZipFile(p) as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)),'Duplicate archive member: '+str(p))
        require(archive.testzip() is None,'CRC failure: '+str(p))
        require(all(n.endswith('.npy') and '/' not in n for n in names),'Unexpected archive member')
    with np.load(p,allow_pickle=False) as z:
        result={k:z[k].copy() for k in z.files}
    require(all(not a.dtype.hasobject for a in result.values()),'Object data not permitted')
    return result


def guard(reserve):
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    free=shutil.disk_usage(ROOT).free
    require(used+reserve<=64*2**20 and free-reserve>=2**30,'Storage reserve/budget failed')
    return {'new_study_bytes':used,'mnt_free_bytes':free}


def write_receipt(path,value):
    require(path.parent==ROOT and not path.exists(),'Exclusive new receipt required')
    data=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    guard(len(data))
    with path.open('xb') as f:f.write(data)


def forbidden(*args,**kwargs):raise RuntimeError('FORBIDDEN inference/rollout/scientific-score call')


def main():
    started=time.monotonic()
    require(sha(REG)==EXPECTED_REG and sha(DRIVER)==EXPECTED_DRIVER,'Base registration/source changed')
    require(not RECEIPT.exists() and not (PARTIAL/'complete.json').exists(),'Old completion or new receipt already exists')
    guard(2*2**20)
    # Import frozen adapter for its authorization and data-source checks only.
    spec=importlib.util.spec_from_file_location('palette_v1_audit_base',DRIVER)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    torch.set_num_threads(2)
    # Tripwires make an accidental model/metric call fail before it can run.
    torch.nn.Module._call_impl=forbidden
    m.ev.palette_rollouts=forbidden;m.tr.generated_history=forbidden
    m.pm.fit_palette_scales=forbidden;m.pm.compare_palette_tracks=forbidden
    m.pm.summarize_palette_track=forbidden
    m.metrics.visual_event_ap=forbidden;m.metrics.paired_song_bootstrap=forbidden
    m.tr.load_checkpoint=forbidden;m.ev.load_bound_model=forbidden
    reg,eventreg,manifest,transforms,store,baselines,receipts,train,dev=m.load_authorized(REG,'predict')
    store.max_ram_bytes=128*2**20
    fail=json.loads(FAILURE.read_text())
    for label,p in [('source',DRIVER),('registration',REG),('calibration',CAL)]:
        require(fail[label]['path']==str(p) and fail[label]['sha256']==sha(p),'Failure binding changed: '+label)
    m.ev.checked_ref(fail['log'])
    require(fail['partial_npz_count']==101 and fail['complete_receipt_exists'] is False
            and fail['scientific_score_run'] is False,'Failure scope changed')
    calibration=m.check_receipt(CAL,REG)
    cal=m.pm.PaletteCalibration(**calibration['calibration'])
    m.pm._validate_calibration(cal,m.pm.palette_prototypes_hash(transforms['palette_prototypes']))
    m.validate_calibration_inventory(cal,train)
    require(set(calibration['cue_hashes'])=={r['recording_id'] for r in train},'Calibration cue inventory changed')
    original_snapshot={p.name:ref(p) for p in PARTIAL.iterdir() if p.is_file()}
    require(len(list(PARTIAL.iterdir()))==101 and len(original_snapshot)==101
            and all(n.endswith('.npz') for n in original_snapshot),'Unexpected partial entries')
    expected=[];seen=set();cell_sources={}
    rows_by_id={r['recording_id']:r for r in manifest['records']}
    for receipt in sorted(receipts,key=lambda r:(m.ROLES.index(r['checkpoint_role']),r['seed'])):
        checkpoint=receipt['checkpoint']
        m.ev.checked_ref(checkpoint)
        m.ev.validate_training_receipt(checkpoint,eventreg)
        for old in receipt['records']:
            role,seed,rid=receipt['checkpoint_role'],receipt['seed'],old['recording_id']
            key=(role,seed,rid)
            require(key not in seen,'Repeated expected cell');seen.add(key)
            donor=eventreg['donor_map'][rid]
            require(old['source_row']==rows_by_id[rid] and old['donor_source_row']==rows_by_id[donor],
                    'Source row not equal frozen manifest')
            require(donor!=rid and rows_by_id[donor]['canonical_id']!=rows_by_id[rid]['canonical_id']
                    and rows_by_id[donor]['partition']==rows_by_id[rid]['partition'],'Invalid donor source')
            filename=f'{role}_s{seed}_{hashlib.sha256(rid.encode()).hexdigest()[:20]}.npz'
            cell=dict(role=role,seed=seed,recording_id=rid,canonical_id=old['canonical_id'],filename=filename,
                original_prediction=old['artifact'],target_sha256=old['target_sha256'],donor_id=donor,
                checkpoint_sha256=checkpoint['sha256'],present=filename in original_snapshot)
            expected.append(cell);cell_sources[key]=(receipt,old)
    require(len(expected)==186 and sum(c['present'] for c in expected)==101,'Expected inventory mismatch')
    require(set(original_snapshot)=={c['filename'] for c in expected[:101]},'Partial outputs not exact deterministic prefix')
    # All31 recipient records occur among first93 selected cells. Check source buses
    # and donor tempo bindings, without model calls or derived feature writes.
    source_records={};labels={};source_checks={}
    for rid in sorted({c['recording_id'] for c in expected[:101]} | {c['donor_id'] for c in expected[:101]}):
        row=rows_by_id[rid]
        checked_sources={k:ref(store.checked(row[k])) for k in ('audio','rhythm','visual','visual_complete')}
        record=store.visual(rid)
        label=m.ev.freeze_labels(record,transforms)
        require(label.content_hash()==eventreg['target_hashes'][rid],'Current target differs: '+rid)
        m.ev.verify_donor_tempos(store,[rid],eventreg['donor_tempos_bpm'])
        source_records[rid]=record;labels[rid]=label
        source_checks[rid]=dict(canonical_id=record.canonical_id,partition=record.partition,
            target_sha256=label.content_hash(),inputs=checked_sources,
            requested_clock_sha256=array_sha(label.requested_times10_s),
            source_pts_clock_sha256=array_sha(label.source_times10_s),
            requested_start_s=float(label.requested_times10_s[0]),
            requested_stop_exclusive_s=float(label.requested_times10_s[0]+64.),
            source_pts_start_s=float(label.source_times10_s[0]),source_pts_last_s=float(label.source_times10_s[-1]),
            source_requested_max_abs_difference_s=float(np.max(np.abs(label.source_times10_s-label.requested_times10_s))),
            donor_tempo_exact=True)
    checked=[]
    wanted_keys={'cues','metadata_json'}|{f'{d}__{k}' for d in m.DRAWS for k in (*m.CONDITIONS,'uniform_noise')}
    for index,cell in enumerate(expected[:101]):
        role,seed,rid=cell['role'],cell['seed'],cell['recording_id']
        receipt,old=cell_sources[(role,seed,rid)]
        p=PARTIAL/cell['filename'];values=read_npz(p)
        require(set(values)==wanted_keys,'Partial NPZ keyset changed: '+str(p))
        original=read_npz(m.ev.checked_ref(old['artifact']))
        record,label=source_records[rid],labels[rid]
        require(label.content_hash()==old['target_sha256'],'Target receipt drift')
        for branch in ('model','donor'):
            require(np.array_equal(original[branch+'__requested_times10_s'],label.requested_times10_s)
                    and np.array_equal(original[branch+'__source_times10_s'],label.source_times10_s),
                    'Requested/source clock mismatch')
            require(np.array_equal(original[branch+'__requested_times40_s'],label.requested_times10_s[0]+np.arange(2560)/40),
                    '40Hz source clock mismatch')
        require(np.array_equal(original['target__color_mask10'],label.color_mask)
                and np.array_equal(original['target__accent_mask10'],label.accent_mask10),'Frozen target support mismatch')
        require(np.array_equal(original['observer__color_valid'],record.color_valid)
                and np.array_equal(original['observer__value_valid'],record.value_valid),'Observer support mismatch')
        if record.frame_valid is not None:
            require(np.array_equal(original['observer__frame_valid'],record.frame_valid),'Frame validity mismatch')
        mask=original['model__generated_color_cues']
        require(mask.dtype==np.bool_ and mask.shape==(640,),'Source cue mask schema')
        expected_cues=np.unique(np.r_[0,np.flatnonzero(mask)]).astype(np.int64)
        cues=values['cues']
        require(cues.dtype==np.uint16 and np.array_equal(cues,expected_cues),'Cue dtype/inventory mismatch')
        require(cues.ndim==1 and len(cues)>0 and cues[0]==0 and np.all(np.diff(cues.astype(int))>0)
                and cues[-1]<640,'Malformed cues')
        cues_hash=m.tr.json_hash(cues.tolist())
        meta=values['metadata_json']
        require(meta.ndim==0 and meta.dtype.kind=='U','Metadata must be scalar Unicode JSON')
        expected_meta=dict(role=role,seed=seed,recording_id=rid,
            original_prediction_sha256=old['artifact']['sha256'],checkpoint_sha256=receipt['checkpoint']['sha256'],
            donor_id=cell['donor_id'],cues_sha256=cues_hash)
        require(json.loads(str(meta))==expected_meta,'Metadata provenance mismatch')
        for draw in m.DRAWS:
            noise=values[f'{draw}__uniform_noise']
            require(noise.dtype==np.float64 and noise.shape==(len(cues),) and np.isfinite(noise).all()
                    and np.array_equal(noise,np.random.default_rng(draw).random(len(cues))),'Draw noise mismatch')
            branches={}
            for condition in m.CONDITIONS:
                ids=values[f'{draw}__{condition}']
                require(ids.dtype==np.uint8 and ids.shape==(640,) and np.all(ids<32),'ID schema/range mismatch')
                held=ids[cues][np.searchsorted(cues,np.arange(640),side='right')-1]
                require(np.array_equal(ids,held),'State changes outside recipient cue inventory')
                branches[condition]=ids
            require(branches['own_state'][0]==branches['null_state'][0]
                    and np.all(branches['persistence']==branches['own_state'][0]),'Paired initialization/persistence mismatch')
        item={**cell,'artifact':original_snapshot[cell['filename']], 'cues_sha256':cues_hash,
              'checks':dict(full_archive_crc_and_members=True,metadata=True,source_and_clocks=True,
                 target_and_support=True,all4_draws_noise_exact=True,all4_conditions_held_state=True,
                 paired_initial_and_persistence=True),
              'produced_under_registration_sha256':EXPECTED_REG,'produced_by_driver_sha256':EXPECTED_DRIVER}
        checked.append(item)
        if (index+1)%25==0:print(f'Validated {index+1}/101 partial cells; no inference/scoring',flush=True)
    after={p.name:ref(p) for p in PARTIAL.iterdir() if p.is_file()}
    require(after==original_snapshot and not (PARTIAL/'complete.json').exists(),'Partial directory changed during audit')
    require(sha(REG)==EXPECTED_REG and sha(DRIVER)==EXPECTED_DRIVER,'Base changed during audit')
    for source,expected_hash in reg['frozen_sources'].items():require(sha(source)==expected_hash,'Frozen source changed')
    result=dict(schema_version=1,status='interrupted_not_complete',audit_status='all_101_passed',
        protocol_id=reg['protocol_id'],authorization='Explicit parent request for read-only interrupted-artifact validation; no resume authorization',
        base_registration=ref(REG),base_driver=ref(DRIVER),calibration=ref(CAL),failure_receipt=ref(FAILURE),
        failure_log=fail['log'],validator=ref(__file__),
        command='PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= TORCH_ENV=/root/lightsem/.venv/torch-access bash run_torch.sh -u '+str(Path(__file__).resolve()),
        event_prediction_receipts=reg['event_predictions'],frozen_sources=reg['frozen_sources'],
        input_bindings={k:reg[k] for k in ('protocol','clarification','event_registration','manifest','transforms','baselines')},
        calibration_content_sha256=cal.content_sha256,
        expected_cells=186,present_cells=101,missing_cells=85,exact_original_execution_prefix=True,
        complete_inventory=expected,validated_present=checked,missing_inventory=expected[101:],
        first_missing_cell=expected[101],source_records=source_checks,
        original_directory=str(PARTIAL),original_directory_unchanged=True,original_complete_receipt_created=False,
        partial_inventory_sha256=m.tr.json_hash(original_snapshot),elapsed_s=time.monotonic()-started,
        execution=dict(device='cpu',torch_version=torch.__version__,numpy_version=np.__version__,
            model_forward_calls=0,rollout_calls=0,metric_outcomes_computed=False,training_calls=0,
            target_regeneration_only_for_hash_validation=True,
            inference_and_score_tripwires_enabled=True),
        limits='Structural/content/source-lineage validation only. Does not independently reproduce logits or sampled tokens, approve tolerance changes, or imply scientific success.',
        storage=guard(2*2**20))
    write_receipt(RECEIPT,result)
    print(json.dumps({'status':result['status'],'audit_status':result['audit_status'],
        'present':101,'missing':85,'receipt':ref(RECEIPT),'storage':guard(0)}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        failure={'status':'validation_failed','error':f'{type(error).__name__}: {error}',
            'validator':ref(__file__),'base_registration':ref(REG),'no_model_forward_or_scoring_requested':True}
        failed=ROOT/'PARTIAL_VALIDATION_FAILURE_V1.json'
        if not failed.exists():write_receipt(failed,failure)
        raise
