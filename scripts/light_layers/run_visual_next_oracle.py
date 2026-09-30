#!/usr/bin/env python3
"""H4 equal-information RGB capacity oracle on the fixed C110–174s trajectory."""
from __future__ import annotations
import argparse,json,resource,time,shutil,subprocess
from pathlib import Path
import numpy as np
from lightsem.conductor.rig import canonical_rig
from lightsem.sim.renderer import StageRenderer
from lightsem.experiments.stage11.stage11_common import gpu_lock
from lightsem.experiments.visual_next.observation import FrameObservation
from lightsem.experiments.visual_next.oracle import (GROUP_ORDER,ColorState,assignments,
    region_histograms_from_grid,regional_cost,choose_assignment,palette_quantization_reference,oracle_gate_report)
from lightsem.experiments.visual_next.palettes import fixture_groups,group_rgb_mask,prototype_representatives
from lightsem.experiments.visual_next.instruments import paired_summary
from lightsem.model.light_layers.visual import sha256,write_json,write_arrays
from scripts.motion.run_rhythm_companion import render_config,optical


class ObserverRenderer:
    def __init__(self,base,groups):
        self.base=base;self.groups=groups;self.rig=canonical_rig()
        self.renderer=StageRenderer(self.rig,render_config())
        self.rgb_mask=group_rgb_mask(groups);self.cache={};self.visibility={};self.calls=0
        self.peak_device_used_bytes=0
    def close(self):self.renderer.ctx.release()
    def universe(self,index,colors):
        frame=self.base[index].copy()
        for fixture in self.groups:
            frame[fixture['rgb_slots_zero_based']]=colors[GROUP_ORDER.index(fixture['o4'])]
        if not np.array_equal(frame[~self.rgb_mask],self.base[index,~self.rgb_mask]):raise RuntimeError('Non-RGB changed')
        return frame
    def image(self,index,colors):
        frame=self.universe(index,colors)
        image=self.renderer.render_frame(optical(frame[None],self.rig)[0],0.)
        self.calls+=1
        if image.shape!=(1080,1920,3):raise RuntimeError('Native renderer geometry drift')
        return image[::3,::3].copy()
    def descriptor(self,index,colors):
        byte=np.asarray(colors,np.uint8);key=(int(index),byte.tobytes())
        if key in self.cache:return self.cache[key]
        if index not in self.visibility:
            neutral=self.image(index,np.full((4,3),255,np.uint8))
            self.visibility[index]=neutral.max(-1)>=21
        image=self.image(index,byte);mask=self.visibility[index]
        observer=FrameObservation(image);hist=[]
        for r,c in ((0,0),(0,320),(180,0),(180,320)):
            region=np.zeros(mask.shape,bool);region[r:r+180,c:c+320]=mask[r:r+180,c:c+320]
            hist.append(observer.target(region).hs_histogram)
        luma=image[:,:,0]*.2126+image[:,:,1]*.7152+image[:,:,2]*.0722
        # Full fixed pixel lattice, identical across every arm; not candidate support.
        result=(np.asarray(hist,np.float32),float(luma.mean()),float((image.max(-1)>=250).mean()))
        self.cache[key]=result
        if len(self.cache)%256==0:self.check_resources()
        return result
    def check_resources(self):
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])*2**20
        self.peak_device_used_bytes=max(self.peak_device_used_bytes,used)
        if used>10*2**30 or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>8*2**30:raise RuntimeError('Oracle memory allowance')
        root=Path('/root/lightsem/runs/visual_next_20260929')
        if any(shutil.disk_usage(p).free<2**30 for p in (root,'/mnt')):raise RuntimeError('Free-space floor')
        total=sum(p.stat().st_size for d in root.iterdir() if d.is_dir() and d.name!='papers' for p in d.rglob('*') if p.is_file())
        if total>4*2**30:raise RuntimeError('Study4GiB artifact allowance')
    def save_cache(self,path):
        entries=sorted(self.cache.items(),key=lambda x:x[0])
        write_arrays(path,{'frame_indices':np.asarray([k[0] for k,v in entries],np.int32),
            'colors':np.asarray([np.frombuffer(k[1],np.uint8).reshape(4,3) for k,v in entries]),
            'histograms':np.asarray([v[0] for k,v in entries]),
            'luma':np.asarray([v[1] for k,v in entries]),'clipping':np.asarray([v[2] for k,v in entries])})


def setup(freeze):
    with np.load(freeze['baseline_stream']['path'],allow_pickle=False) as z:base=z['candidate'].copy()
    with np.load(freeze['original_overlay']['path'],allow_pickle=False) as z:
        raw=z['overlay'].copy();ids=z['palette_ids40'].copy();active=z['color_active40'].copy();rgb=z['palette_rgb'].copy()
    prototypes=np.asarray(json.loads(Path(freeze['baseline_fit_report']['path']).read_text())['frozen_transforms']['palette_prototypes'])
    groups=fixture_groups(canonical_rig())
    if groups!=freeze['fixture_groups']:raise RuntimeError('Fixture partition drift')
    palette=[]
    for index,p in enumerate(prototypes):
        modes=prototype_representatives(p)
        palette.append(np.floor(np.asarray([rgb[index]]+[m['rgb'] for m in modes])*255+.5).astype(np.uint8))
    palette=np.asarray(palette)
    changes=np.flatnonzero(active & (np.r_[True,~active[:-1]]|np.r_[True,ids[1:]!=ids[:-1]]))
    start=110*40;stop=174*40
    if not active[start:stop].all():raise RuntimeError('Fixed context requires active palette state')
    first=int(changes[np.searchsorted(changes,start,side='right')-1])
    before=changes[changes<first]
    if first==0 or (len(before) and first-before[-1]<12):raise RuntimeError('Initial inherited fade not settled; explicit state reconstruction required')
    cues=np.r_[first,changes[(changes>first)&(changes<stop)]].astype(int)
    initial=[]
    for group in GROUP_ORDER:
        fs=[f for f in groups if f['o4']==group]
        colors=[base[first-1,f['rgb_slots_zero_based']] for f in fs]
        if not all(np.array_equal(colors[0],c) for c in colors) and first/40+.3>110:
            raise RuntimeError('Unequal inherited fixture fades enter the scored context')
        # Different pre-adoption fixture colors are harmless only if their fade
        # finishes before110s. No output before the fixed context is inferred.
        initial.append(colors[0]/255)
    initial=ColorState.constant(np.asarray(initial),first/40)
    return base,raw,ids,palette,groups,cues,initial


def state_schedule(choices,cues,initial,ids,palette):
    state=initial;result=[]
    for cue,choice in zip(cues,choices):
        state=state.adopt(palette[ids[cue],choice]/255,cue/40);result.append(state)
    return result


def observe_schedule(renderer,times,cues,states):
    result=[]
    for time_s in times:
        event=int(np.searchsorted(cues,time_s*40,side='right')-1)
        rgb=np.floor(np.clip(states[event].at([time_s])[0],0,1)*255+.5).astype(np.uint8)
        result.append(renderer.descriptor(round(time_s*40),rgb))
    return np.asarray([x[0] for x in result]),np.asarray([x[1] for x in result]),np.asarray([x[2] for x in result])


def optimize(renderer,arm,ref,valid,cues,initial,ids,palette,deadline):
    candidates=assignments(arm);state=initial;previous=np.zeros(4,int);decisions=[];objectives=[]
    times=110+np.arange(64)
    for j,cue in enumerate(cues):
        end=cues[j+1]/40 if j+1<len(cues) else 174.
        at=np.flatnonzero((times>=cue/40)&(times<end))
        states=[state.adopt(palette[ids[cue],choice]/255,cue/40) for choice in candidates]
        if len(at):
            prediction=[]
            for candidate in states:
                observations=[]
                for t in times[at]:
                    rgb=np.floor(np.clip(candidate.at([t])[0],0,1)*255+.5).astype(np.uint8)
                    observations.append(renderer.descriptor(int(t*40),rgb)[0])
                    if time.monotonic()>deadline:raise RuntimeError('H4 runtime cap')
                prediction.append(observations)
            costs=regional_cost(ref[at],np.asarray(prediction),valid[at])
        else:costs=np.full(len(candidates),np.nan)
        choice,objective,index=choose_assignment(costs,candidates,previous)
        decisions.append(choice);objectives.append({'cue_frame':int(cue),'candidate_costs':costs,
            'selected_index':index,'penalized_objective':objective,'optimization_samples':at})
        state=states[index];previous=choice
    return np.asarray(decisions),objectives


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',required=True,type=Path)
    args=ap.parse_args();freeze=json.loads(args.freeze.read_text())
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    out=Path(freeze['output']);out.mkdir(exist_ok=False);started=time.monotonic();deadline=started+freeze['max_seconds']
    base,raw,ids,palette,groups,cues,initial=setup(freeze)
    write_arrays(out/'SCHEDULE_INPUTS.npz',{'cue_indices':cues,'palette_candidates':palette,'palette_ids40':ids})
    # Independently verify that choosing C's target everywhere reproduces C bytes.
    same=np.zeros((len(cues),4),int);same_states=state_schedule(same,cues,initial,ids,palette)
    for index in range(110*40,174*40):
        event=np.searchsorted(cues,index,side='right')-1
        colors=np.floor(same_states[event].at([index/40])[0]*255+.5).astype(np.uint8)
        for f in groups:
            assert np.array_equal(colors[GROUP_ORDER.index(f['o4'])],base[index,f['rgb_slots_zero_based']])
    records=[];optimization_times=110+np.arange(64);scoring_times=110+.5+np.arange(64)
    with gpu_lock():
        renderer=ObserverRenderer(base,groups)
        try:
            if freeze['mode']=='smoke':
                samples=110+np.arange(16)*4
                rows=[]
                for i,t in enumerate(samples):
                    event=np.searchsorted(cues,t*40,side='right')-1
                    choice=np.asarray([(i+j)%4 for j in range(4)])
                    color=palette[ids[cues[event]],choice]
                    rows.append(renderer.descriptor(round(t*40),color))
                write_arrays(out/'SMOKE.npz',{'histograms':np.asarray([r[0] for r in rows]),
                    'luma':np.asarray([r[1] for r in rows]),'clipping':np.asarray([r[2] for r in rows]),'times':samples})
                result={'status':'oracle_renderer_smoke_complete','samples':16,'actual_renderer_calls':renderer.calls,
                    'seconds':time.monotonic()-started,'C_replay_exact':True,'optimization_performed':False,
                    'candidate_cache_entries':len(renderer.cache),'renderer':renderer.renderer.renderer_name}
            else:
                references={};validity={};all_decisions={};reference_support=[]
                for row in freeze['families']:
                    cid=row['canonical_id']
                    with np.load(row['teacher_raw']['path']) as z:
                        histogram,valid=region_histograms_from_grid(z['R1'],row['geometry_wh'])
                        valid &= z['frame_valid'][:,None]
                    references[cid]=histogram;validity[cid]=valid
                    reference_support.append({'canonical_id':cid,'scoring_frame_coverage':float(valid[5::10].any(1).mean()),
                                              'valid_scoring_regions':int(valid[5::10].sum())})
                write_json(out/'REFERENCE_SUPPORT.json',{'families':reference_support,'frozen_before_optimization':True,
                    'source':'quantized automatic SAM2-retained R1 grid; no manual truth'})
                for row in freeze['families']:
                    cid=row['canonical_id'];histogram=references[cid];valid=validity[cid]
                    all_decisions[cid]={}
                    for arm in ('O0','O2','O4'):
                        decisions,trace=optimize(renderer,arm,histogram[::10],valid[::10],cues,initial,ids,palette,deadline)
                        all_decisions[cid][arm]=decisions
                        # Costs can be undefined only when the same source interval
                        # is unavailable for every candidate. Save explicit masks.
                        packed={f'cost_{j}':np.nan_to_num(x['candidate_costs']) for j,x in enumerate(trace)}
                        packed.update({f'cost_valid_{j}':np.isfinite(x['candidate_costs']) for j,x in enumerate(trace)})
                        packed['decisions']=decisions
                        write_arrays(out/(cid+'_'+arm+'_optimization.npz'),packed)
                    print('ORACLE_OPTIMIZED',cid,'render_cache',len(renderer.cache),flush=True)
                for row in freeze['families']:
                    cid=row['canonical_id'];donor=freeze['donors'][cid];reference=references[cid][5::10];valid=validity[cid][5::10]
                    decisions=all_decisions[cid]
                    arms={**decisions,'permuted':np.roll(decisions['O4'],1,axis=1),'donor':all_decisions[donor]['O4'],'C':same}
                    scores={};arrays={}
                    for arm,choice in arms.items():
                        states=state_schedule(choice,cues,initial,ids,palette)
                        pred,luma,clipping=observe_schedule(renderer,scoring_times,cues,states)
                        error=regional_cost(reference,pred[None],valid)[0]
                        scores[arm]={'error':float(error) if np.isfinite(error) else None,
                            'mean_luma':float(luma.mean()),'clipping':float(clipping.mean())}
                        arrays[arm+'_histogram']=pred;arrays[arm+'_luma']=luma;arrays[arm+'_clipping']=clipping;arrays[arm+'_choices']=choice
                    arrays.update(reference=reference,reference_valid=valid)
                    quantized=np.stack([palette_quantization_reference(reference[i],palette[ids[round(t*40)]])
                                        for i,t in enumerate(scoring_times)])
                    quantized_error=regional_cost(reference,quantized[None],valid)[0]
                    arrays['palette_only_quantization']=quantized
                    path=out/(cid+'_SCORING.npz');write_arrays(path,arrays)
                    records.append({'canonical_id':cid,'donor':donor,'scores':scores,'valid_regions':int(valid.sum()),
                        'palette_quantization_error':float(quantized_error) if np.isfinite(quantized_error) else None,
                        'source_frame_coverage':float(valid.any(1).mean()),'scoring_sha256':sha256(path)})
                    if time.monotonic()>deadline:raise RuntimeError('H4 runtime cap')
                renderer.save_cache(out/'RENDER_CACHE.npz')
                usable=[r for r in records if r['source_frame_coverage']>=.8 and r['scores']['O0']['error'] is not None and r['scores']['O4']['error'] is not None]
                summary=paired_summary([r['scores']['O0']['error'] for r in usable],[r['scores']['O4']['error'] for r in usable],direction='down') if len(usable)>=2 else None
                result={'status':'exploratory_fixture_oracle_complete','rows':records,'paired_summary':summary,
                    'gate_report':oracle_gate_report(records,summary),
                    'palette_quantization_scope':'Descriptive palette-only reference; never subtracted from error; not a universal lower bound because render mixing/tonemapping creates intermediate colors.',
                    'elapsed_s':time.monotonic()-started,'actual_renderer_calls':renderer.calls,'cache_entries':len(renderer.cache),
                    'C_replay_exact':True,'non_RGB_slots_preserved':479,'candidate_cache_sha256':sha256(out/'RENDER_CACHE.npz'),
                    'scope':'Output capacity conditional on automatic source histograms and one fixedC trajectory; no audio predictability, independent mask truth or aesthetic claim.'}
            renderer.check_resources()
            result['peak_whole_device_used_bytes_sampled']=renderer.peak_device_used_bytes
            result['GPU_memory_scope']='NVML whole-device samples, including desktop; GL allocation not tracked by Torch reserved-memory counters.'
        finally:renderer.close()
    for p,d in freeze['files'].items():assert sha256(p)==d,p
    result.update(freeze_sha256=sha256(args.freeze),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  test_access=False,baseline_C_changed=False)
    write_json(out/'REPORT.json',result)


if __name__=='__main__':main()
