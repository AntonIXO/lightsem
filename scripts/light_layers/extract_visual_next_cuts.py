#!/usr/bin/env python3
"""Frozen H1 exploratory cut/observer traces; no fabricated independent labels."""
from __future__ import annotations
import argparse,json,resource,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from lightsem.experiments.visual_next.models import load_transnet,transnet_predict
from lightsem.experiments.visual_next.instruments import edit_veto
from lightsem.model.light_layers.evaluation import candidate_peaks
from lightsem.model.light_layers.visual import sha256,write_arrays,write_json
from lightsem.experiments.stage11.stage11_common import gpu_lock


def native_frames(row):
    path=row['source_video']['path']
    raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_streams','-show_frames','-show_entries',
        'stream=start_time:frame=best_effort_timestamp_time','-of','json',path],timeout=240)
    probe=json.loads(raw);origin=float(probe['streams'][0].get('start_time',0))
    timestamps=np.asarray([float(f['best_effort_timestamp_time'])-origin for f in probe['frames']])
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps)<=0):
        raise RuntimeError('Native source PTS must increase strictly')
    use=np.flatnonzero((timestamps>=row['start_s'])&(timestamps<row['start_s']+row['duration_s']))
    if len(use)<100 or np.any(np.diff(use)!=1):raise RuntimeError('Unsupported native window')
    # Decode exact original frame indices, without rate conversion or approximate seeking.
    command=['ffmpeg','-v','error','-nostdin','-threads','2','-noautorotate','-i',path,
        '-an','-sn','-dn','-vf',f'select=between(n\\,{use[0]}\\,{use[-1]}),scale=48:27',
        '-frames:v',str(len(use)),'-fps_mode','passthrough','-threads','1',
        '-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    decoded=subprocess.check_output(command,timeout=240)
    if len(decoded)!=len(use)*27*48*3:raise RuntimeError('Native decoded frame count mismatch')
    return np.frombuffer(decoded,np.uint8).reshape(-1,27,48,3),timestamps[use],use


def positive_intervals(times,positive):
    bounds=np.flatnonzero(np.diff(np.r_[False,positive,False]))
    return np.asarray([[times[a],times[b-1]] for a,b in zip(bounds[::2],bounds[1::2])]).reshape(-1,2)


def observation_intervals(times,valid):
    intervals=[]
    for i in np.flatnonzero(valid):
        stop=times[i+1] if i+1<len(times) else times[i]+.1
        if stop-times[i]>.2+1e-8:continue
        if intervals and abs(intervals[-1][1]-times[i])<1e-9:intervals[-1][1]=stop
        else:intervals.append([float(times[i]),float(stop)])
    return np.asarray(intervals).reshape(-1,2)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',type=Path,required=True)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    root=Path(freeze['output']);root.mkdir(exist_ok=False)
    report_rows=[];started=time.monotonic();torch.set_num_threads(2)
    threshold=freeze['observer_novelty_threshold']
    assert 0<threshold<1
    with gpu_lock():
        torch.cuda.init();torch.cuda.reset_peak_memory_stats();model=load_transnet()
        for number,row in enumerate(freeze['records']):
            if time.monotonic()-started>freeze['max_seconds']:raise RuntimeError('Frozen stage runtime cap')
            if any(shutil.disk_usage(p).free<2**30 for p in (root,'/mnt')):raise RuntimeError('Disk floor')
            unit_started=time.monotonic();folder=root/row['canonical_id'];folder.mkdir()
            frames,times,indices=native_frames(row)
            predictions=transnet_predict(model,frames)
            assert predictions.shape==(len(times),2) and np.isfinite(predictions).all()
            edits=positive_intervals(times,predictions[:,0]>.5)
            many=positive_intervals(times,predictions[:,1]>.5)
            sample=Path(row['capture_receipt']['path']).parent/'sam2.npz'
            with np.load(sample,allow_pickle=False) as z:
                observer_times=z['novelty_times_s'].copy()
                novelty=z['novelty_scores'].astype(float)
                available=z['novelty_eligible'].copy()
            intervals=observation_intervals(observer_times,available)
            # Frozen TRAIN-only scale maps original novelty threshold to .5;
            # unbounded confidence preserves ranking, without clipping-induced ties.
            candidate_times,confidence=candidate_peaks(observer_times,novelty*.5/threshold,intervals)
            keep=edit_veto(candidate_times,edits,margin=.2)
            emitted=confidence>=.5
            output={'native_source_times_s':times,'native_frame_indices':indices,
                'transnet_single_many':predictions,'edit_intervals_s':edits,
                'secondary_many_hot_intervals_s':many,'candidate_times_s':candidate_times,
                'candidate_confidence':confidence,'A1_keep':keep,
                'observer_sample_times_s':observer_times,'observer_available':available}
            write_arrays(folder/'traces.npz',output)
            entry={'canonical_id':row['canonical_id'],'recording_id':row['recording_id'],
                'role':row['role'],'source_sha256':row['source_video']['sha256'],
                'native_frames':len(times),'single_head_edit_intervals':len(edits),
                'many_hot_intervals_secondary':len(many),'candidate_count_all_scores':len(candidate_times),
                'A0_emitted':int(emitted.sum()),'A1_emitted':int((emitted&keep).sum()),
                'vetoed_at_frozen_threshold':int((emitted&~keep).sum()),
                'observer_available_seconds':float(np.diff(intervals,axis=1).sum()),
                'reference_denominator_changed':False,
                'independent_content_F1':None,'independent_content_recall':None,
                'independent_false_cues_per_minute':None,
                'seconds':time.monotonic()-unit_started,
                'arrays_sha256':sha256(folder/'traces.npz')}
            write_json(folder/'REPORT.json',entry);report_rows.append(entry)
            print('CUT_TRACE',number+1,len(freeze['records']),row['canonical_id'],
                  'A0',entry['A0_emitted'],'A1',entry['A1_emitted'],flush=True)
            if torch.cuda.max_memory_reserved()>10*2**30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:
                raise RuntimeError('Resource memory cap')
            del frames,predictions,output
        peak=torch.cuda.max_memory_reserved()
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    write_json(root/'REPORT.json',{'status':'exploratory_cut_traces_complete',
        'scope':'Observer intervention traces only; vetoed candidates are not known false; independent efficacy unmeasured',
        'freeze_sha256':sha256(args.freeze),'elapsed_s':time.monotonic()-started,
        'peak_reserved_bytes':peak,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'rows':report_rows,'original_hypothesis_closed':False,'test_access':False,
        'content_F1_and_recall_magnitude_gates':'unmeasured_missing_independent_annotation'})


if __name__=='__main__':main()
