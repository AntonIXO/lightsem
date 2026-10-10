"""Build an immutable RTMPose all-partition fit manifest; never rebuild YOLO.

Native raw PTS and predictions remain in the authenticated ragged cache. Training
samples are a separate, masked 30 Hz view bound to the decoded audio clock.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from lightsem.experiments.native_dynamics.full_fit_admission import (
    iter_native_tracks, native_metadata, sha, verify_normalization, window_union,
)
from lightsem.experiments.native_dynamics.rtmpose_cache import masked_track


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with Path(str(path)+'.sha256.json').open('x') as stream:
        json.dump({'sha256':sha(path)}, stream)


def build(*, stage_manifest, audio_manifest, ids_file, pose_dir, output_root,
          legacy_map=Path('/mnt/lightsem-data/cache/justdance_songs.json'), allow_incomplete=False):
    root, pose_dir = Path(output_root), Path(pose_dir)
    if (root/'data/MANIFEST.json').exists() or (root/'data/poses').exists():
        raise FileExistsError('Use a new admission output root; completed poses/manifests are immutable')
    stage, audio = (json.loads(Path(p).read_text()) for p in (stage_manifest,audio_manifest))
    ids = sorted({s.strip() for s in Path(ids_file).read_text().splitlines()
                  if s.strip() and not s.lstrip().startswith('#')})
    ar = {r['recording_id']:r for r in audio['rows']}
    sr = {r['recording_id']:r for r in stage['recordings']}
    legacy = {r['yt_id']:r for r in json.loads(Path(legacy_map).read_text())['rows']}
    missing_audio = set(ids)-ar.keys()
    missing_pose = [rid for rid in ids if not (pose_dir/f'{rid}.npz').is_file()]
    if missing_audio or (missing_pose and not allow_incomplete):
        raise ValueError(f'Incomplete paired inputs: audio={sorted(missing_audio)}, pose={len(missing_pose)}')
    # Use known legacy links to canonical Stage11 families. Unresolved extras
    # get recording-specific identities with an explicit provisional status.
    links = defaultdict(set)
    for rid, row in legacy.items():
        if rid in sr and row.get('song_key'):
            links[row['song_key']].add(sr[rid]['canonical_id'])
    records = {}
    for rid in ids:
        if rid in sr:
            records[rid] = dict(sr[rid], source_partition=sr[rid]['partition'])
        else:
            song = legacy.get(rid,{})
            linked = links.get(song.get('song_key'),set())
            cid = next(iter(linked)) if len(linked)==1 else 'jd-unresolved-'+rid
            records[rid] = dict(recording_id=rid, canonical_id=cid,
                identity_status='legacy_link_to_stage11' if len(linked)==1 else 'provisional_recording_identity',
                source_partition='legacy_unassigned', source_path=ar[rid]['source_path'],
                source_sha256=ar[rid]['source_sha256'], title_normalized=song.get('title_norm',rid))
    groups = defaultdict(list)
    for rid, rec in records.items(): groups[rec['canonical_id']].append(rid)
    pose_out = root/'data/poses'
    pose_out.mkdir(parents=True)
    entries, exclusions, audits = [], [], []
    recording_observed, recording_usable = {}, {}
    for rid in ids:
        rec, wave = records[rid], ar[rid]
        path = pose_dir/f'{rid}.npz'
        if not path.exists():
            exclusions.append({'recording_id':rid,'reason':'pose_extraction_pending'})
            continue
        binding = verify_normalization(wave)
        with np.load(path, allow_pickle=False) as z:
            raw = {name:z[name] for name in z.files}
        meta = native_metadata(raw)
        # Authenticate local source and both modalities, not only a historical
        # path or cached source name.
        source_hash = sha(wave['source_path'])
        metadata_hash = meta.get('source_sha256')
        if (source_hash != wave['source_sha256'] or source_hash != metadata_hash or
                source_hash != rec['source_sha256']):
            raise ValueError(f'Changed paired video source: {rid}')
        if 'source_path' in meta:
            try:
                metadata_source = Path(meta['source_path']).resolve()
                paired_source = Path(wave['source_path']).resolve()
            except (TypeError, ValueError, OSError) as error:
                raise ValueError(f'Invalid source_path metadata: {rid}') from error
            if metadata_source != paired_source:
                raise ValueError(f'Pose/source path mismatch: {rid}')
        if wave['classification']=='silent':
            exclusions.append({'recording_id':rid,'reason':'fresh_waveform_silent'})
            continue
        if meta.get('audio_start_s') is None:
            raise ValueError(f'Unknown audio clock origin: {rid}')
        times = np.asarray(raw['t'],np.float64)-float(meta['audio_start_s'])
        duration = float(wave['duration_s'])
        length = int(np.ceil(duration*30))
        recording_observed[rid] = np.zeros(length,bool)
        recording_usable[rid] = np.zeros(length,bool)
        raw_hash = sha(path)
        for tid, k, confidence, box in iter_native_tracks(raw):
            audit = {'recording_id':rid,'track_id':tid,'group':rec['canonical_id'],
                     'source_partition':rec['source_partition']}
            try:
                # k/box already refer to ONE track; do not slice a coach axis.
                observation, quality = masked_track(k,box,times,duration,float(meta['height']),
                                                     confidence=confidence, confidence_gate=False)
            except ValueError as error:
                audit.update(status='excluded',reason=str(error))
                exclusions.append(audit); audits.append(audit)
                continue
            good = observation['mask'].sum(1)>=10
            recording_observed[rid] |= good
            union = window_union(observation['starts'],len(good))
            usable = good & union
            quality.update(usable_masked_seconds=float(usable.sum()/30),
                           window_union_seconds=float(union.sum()/30),
                           audio_origin_s=float(meta['audio_start_s']),
                           pose_audio_end_difference_s=float(times[-1]+np.median(np.diff(times))-duration))
            # A paired track is admissible only when its observed endpoint
            # binds to the decoded audio clock.  Keep the raw audit row, but do
            # not let a materially truncated or overlong pose stream enter the
            # training manifest.
            if abs(quality['pose_audio_end_difference_s']) > 0.25:
                audit.update(status='excluded', reason='pose/audio duration binding mismatch', quality=quality)
                exclusions.append(audit); audits.append(audit)
                continue
            audit['quality'] = quality
            if not len(observation['starts']):
                audit.update(status='excluded',reason='no_valid_masked_window')
                exclusions.append(audit); audits.append(audit)
                continue
            recording_usable[rid] |= usable
            key = f'{rid}_t{tid}'
            cache = pose_out/f'{key}.npz'
            with cache.open('xb') as stream: np.savez_compressed(stream,**observation)
            detected = np.isfinite(box).all(1)
            scores = confidence[detected]
            entry = {'key':key,'feature_key':f'jd_{rid}','partition':'train',
                'source_partition':rec['source_partition'],'domain':'jd','recording_id':rid,
                'coach_index':tid,'track_id':tid,'coach_id':f'rtmpose:{rid}:track:{tid}',
                'coach_identity_status':'within-recording track segment; no asserted match to legacy coach indices',
                'group':rec['canonical_id'],'canonical_id':rec['canonical_id'],
                'identity_status':rec.get('identity_status','unspecified'),
                'duplicate_members':groups[rec['canonical_id']], 'cross_corpus_links':[],
                'legacy_map_present':rid in legacy,'title':rec.get('title_normalized',''),
                'pose_cache':str(cache),'pose_cache_sha256':sha(cache),
                'raw_pose_path':str(path),'raw_pose_sha256':raw_hash,
                'wav_path':wave['wav_path'],'wav_sha256':wave['wav_sha256'],
                'source_path':wave['source_path'],'source_sha256':source_hash,
                'normalization_receipt':wave.get('normalization_receipt'),'normalization_binding':binding,
                'fresh_audio_classification':wave['classification'],'duration_s':duration,
                'frames':length,'quality':quality,'raw_score_mean':float(scores.mean()),
                'raw_score_joint_p10':float(np.quantile(scores,.1)),
                'mask_contract':'finite estimates; detector boxes; hip anchor; finite-to-finite jump; interpolation<=60ms; RTMPose scores metadata only; no confidence gate or forward fill',
                'native_timestamps_preserved_in_raw_cache':True,
                'training_timestamps':'30 Hz audio-relative clock; raw video PTS minus audio stream origin',
                'timeline_path':str(root/'features/timelines'/f'jd_{rid}.npz')}
            entries.append(entry)
            audit.update(status='admitted',key=key)
            audits.append(audit)
    if not entries:
        raise ValueError('No eligible RTMPose tracks; do not advance an empty manifest')
    strict = [r for r in entries if r['quality']['coverage']>=.8]
    nonsilent = [r for rid,r in ar.items() if rid in ids and r['classification']!='silent']
    coverage = {
        'requested_recordings':len(ids),'completed_pose_recordings':len(ids)-len(missing_pose),
        'audio_source_minutes':sum(ar[r]['duration_s'] for r in ids)/60,
        'non_silent_audio_minutes':sum(r['duration_s'] for r in nonsilent)/60,
        'recording_union_observed_minutes':sum(x.sum() for x in recording_observed.values())/1800,
        'recording_union_usable_window_minutes':sum(x.sum() for x in recording_usable.values())/1800,
        'coach_observed_minutes':sum(r['quality']['valid_observation_seconds'] for r in entries)/60,
        'coach_usable_window_minutes':sum(r['quality']['usable_masked_seconds'] for r in entries)/60,
        'coach_usable_minutes_after_whole_record_80pct':sum(r['quality']['usable_masked_seconds'] for r in strict)/60,
        'whole_record_gate_usable_minutes_lost':sum(r['quality']['usable_masked_seconds'] for r in entries if r not in strict)/60,
        'admitted_recordings':len({r['recording_id'] for r in entries}),
        'admitted_groups':len({r['group'] for r in entries}),'admitted_track_segments':len(entries),
        'units':'Recording unions count simultaneous coaches once; coach-minutes keep people separate; overlapping windows counted once.'}
    report = {'schema_version':2,'status':'partial_diagnostic' if missing_pose else 'full_fit_rtmpose_no_split_unqualified',
        'no_validation_split':True,'requested_ids':ids,'missing_pose_ids':missing_pose,
        'entries':entries,'exclusions':exclusions,'coverage':coverage,
        'source_partition_policy':'All fit; original partition retained as metadata. Historical manifests unchanged.',
        'stage_manifest_sha256':sha(stage_manifest),'audio_manifest_sha256':sha(audio_manifest),
        'ids_file_sha256':sha(ids_file),'builder_sha256':sha(__file__),
        'raw_cache_run_receipt':str(pose_dir/'RUN.json'),
        'raw_cache_run_sha256':sha(pose_dir/'RUN.json') if (pose_dir/'RUN.json').exists() else None,
        'mask_code_sha256':sha(Path(__file__).parents[2]/'lightsem/experiments/native_dynamics/rtmpose_cache.py')}
    write_new(root/'data/MANIFEST.json',report)
    write_new(root/'ADMISSION_PACKAGE/RECORDING_COACH_EXCLUSIONS.json',{'rows':audits,'recording_exclusions':[r for r in exclusions if 'track_id' not in r]})
    write_new(root/'ADMISSION_PACKAGE/COVERAGE.json',coverage)
    write_new(root/'ADMISSION_PACKAGE/strict_qualification_manifest.json',{
        'status':'rtmpose_whole_record_comparator_not_legacy_qualification','entries':strict,
        'note':'Changed estimator/score semantics; >=80% is a coverage comparator, not qualification under the historical YOLO gates.'})
    write_new(root/'ADMISSION_PACKAGE/max_masked_window_train_manifest.json',report)
    protected = []
    for rid in ids:
        rec = records[rid]
        if rec.get('source_partition') in ('val','test'):
            protected.append({'recording_id':rid,'canonical_id':rec['canonical_id'],
                'source_partition':rec['source_partition'],'protected_partition':rec['source_partition'],
                'source_path':rec['source_path'],'source_sha256':rec['source_sha256'],
                'payload_opened_this_run':True,'coaches':rec.get('coaches',[])})
    protected_groups={r['canonical_id'] for r in protected}
    train_groups={r['canonical_id'] for r in entries}
    write_new(root/'ADMISSION_PACKAGE/protected_val_test_manifest.json',{
        'status':'historical_partition_metadata_only','test_payload_opened_this_run':True,
        'protected_group_count':len(protected_groups),'test_group_count':sum(r['protected_partition']=='test' for r in protected),
        'entries':protected,'note':'Included in fit only because the user explicitly requested no train/validation/test distinction. Historical payloads and manifests were not modified.'})
    write_new(root/'ADMISSION_PACKAGE/LEAKAGE_AUDIT.json',{
        'status':'partition_boundary_not_used_by_explicit_full_fit','train_group_count':len(train_groups),
        'historical_protected_group_count':len(protected_groups),
        'overlap_intentional_for_full_fit':sorted(train_groups & protected_groups),
        'test_payload_opened_this_run':True,'historical_manifests_modified':False})
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ('stage-manifest','audio-manifest','ids-file','pose-dir','output-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--allow-incomplete',action='store_true',help='Diagnostic preview only; cannot authorize training')
    args=p.parse_args()
    print(json.dumps(build(**vars(args))['coverage'],indent=2))


if __name__=='__main__': main()
