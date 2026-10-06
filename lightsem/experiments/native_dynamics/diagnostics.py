"""Read-only body/requested/emitted audit for native learned dynamics."""
from pathlib import Path
import numpy as np
from .data import ROOT,read,write,sha
from .showcase import ALIASES,CONDITIONS
from ..dynamic_dance_combo import diagnostics as old
from ..dynamic_dance import diagnostics as d
from ..dynamic_dance import transfer_audit as ta
from ..dynamic_dance.contracts import DanceResult,MusicTimeline
from lightsem.conductor.body_transfer import decode


def compact_pair(root,method,track,pins):
    folder=root/'control_emission'/method/track['id'];bodypath=root/'bodies'/method/track['id']/'body.npz'
    report=read(folder/'REPORT.json');patch=read(folder/'patch.json');body=DanceResult.load(bodypath)
    for key,p in [('body_sha256',bodypath),('emission_sha256',folder/'emission.npz'),('patch_sha256',folder/'patch.json')]:
        if sha(p)!=report[key]:raise ValueError('Changed compact source/control')
        pins[str(p)]=sha(p)
    with np.load(folder/'emission.npz') as z:arrays={k:z[k].copy() for k in z.files}
    controls=decode(arrays['universes'],patch);moving=np.asarray([p['kind']!='bulb' for p in patch]);ranges=np.asarray([[p['pan_range_deg'],p['tilt_range_deg']] for p in patch if p['kind']!='bulb'])
    return {'id':method,'body':body,'arrays':arrays,'controls':controls,'moving':moving,'ranges':ranges,'tick':float(ranges.max()/65535),'patch':patch,'requested':arrays['requested'],'report':report}


def source_response(original,other):
    a,b=original['body'].metadata,other['body'].metadata
    keys=('checkpoint','noise_sha256','draw_seed','training_seed','precision')
    same={k:a[k]==b[k] for k in keys}
    if not all(same.values()):raise ValueError('Audio sensitivity inputs not held fixed')
    result=old._difference(original,other)
    result['fixed_generation_inputs']=same
    result['analyzed_waveform_changed']=a['audio']['binding']['analyzed_waveform_sha256']!=b['audio']['binding']['analyzed_waveform_sha256']
    result['global_brightness_and_colors']='Original-song historical envelope/P2 held fixed; only body path intervened; body accents may differ.'
    return result


def audit_song(root,song):
    root=Path(root);plan=read(root/'PLAN.json');track=next(t for t in plan['tracks'] if t['id']==song);music=MusicTimeline.load(track['music_timeline']);release=read(root/'records/audio_intervals'/f'{song}.release.json');validation=read(root/'reports/VALIDATION.json')
    pins={str(root/'PLAN.json'):sha(root/'PLAN.json'),str(root/'reports/VALIDATION.json'):sha(root/'reports/VALIDATION.json')};all_reports=[]
    for seed in (0,1):
        pairs={}
        for arm,alias in ALIASES.items():
            method=alias+('' if seed==0 else '_seed1');pair,absent=old._load_pair(root,method,track,plan,pins)
            if absent:raise ValueError(absent)
            pairs[arm]=pair
        ref=pairs['continued'];scores={arm:old._scores(pair,ref,music,release,plan,track['duration_s']) for arm,pair in pairs.items()}
        for arm,base in [('dynamics','continued'),('joint','dynamics')]:
            pair=pairs[arm];baseline=pairs[base];c=scores[arm];b=scores[base]
            g=d.comparison_gates(c['body'],b['body'],candidate_emission=c['emission'],baseline_emission=b['emission'])['gates'];ga=d.comparison_gates(c['body'],scores['continued']['body'],candidate_emission=c['emission'],baseline_emission=scores['continued']['emission'])['gates']
            keys=old.PHYSICAL_GATES;fail=[k for k in keys if g[k]['status']!='pass'];original_fail=[k for k in keys if ga[k]['status']!='pass']
            for label,cs,bs,target in [('incremental',c,b,fail),('vs_continued',c,scores['continued'],original_fail)]:
                if cs['body']['contacts']['geometric_support_fraction']<bs['body']['contacts']['geometric_support_fraction']-.05:target.append('support_coverage')
                if cs['body']['geometry']['floor_penetration_m']['p95']>bs['body']['geometry']['floor_penetration_m']['p95']+.02:target.append('floor_penetration')
            differences=old._difference(pair,baseline);controls={}
            if seed==0:
                for condition in CONDITIONS[1:]+(('plan_shift',) if arm=='joint' else ()):
                    other=compact_pair(root,ALIASES[arm]+'__'+condition,track,pins)
                    reference=compact_pair(root,ALIASES[arm]+'__reconstruction',track,pins) if condition in ('mute_drums','mute_vocals') else pair
                    controls[condition]=source_response(reference,other)
                repeat=controls['repeat']['emitted_motor_rms_deg'];bodyrepeat=controls['repeat']['root_relative_joint_rms_m']
                for condition,value in controls.items():
                    value['body_above_repeat_noise']=value['root_relative_joint_rms_m']>max(.0001,5*bodyrepeat)
                    value['emission_above_repeat_noise']=value['emitted_motor_rms_deg']>=max(pair['tick'],5*repeat)
            timing={'correct':timing_scores(pair,release,track)}
            if seed==0:
                for condition in ('shift','donor')+(('plan_shift',) if arm=='joint' else ()):
                    timing[condition]=timing_scores(compact_pair(root,ALIASES[arm]+'__'+condition,track,pins),release,track)
            v=validation['comparisons'][arm]['seeds'][seed]
            report={'method':ALIASES[arm]+('' if seed==0 else '_seed1'),'song':song,'arm':arm,'training_seed':seed,'comparator':ALIASES[base],'scientific_comparator':base,'display_reference_id':'D7_reference','candidate':c,'baseline':b,'original_continued_reference':scores['continued'],'physical_gates':{k:g[k] for k in keys},'original_reference_physical_gates':{k:ga[k] for k in keys},'physical_failure_reasons':fail+['vs_continued:'+x for x in original_fail],'physical_engineering_status':'pass' if not fail and not original_fail else 'fail','primary':{'name':'JD native speed profile validation','magnitude_gate':('unqualified' if not v['gates']['complete_registered_coverage'] else 'pass' if v['gates']['magnitude'] else 'fail'),'available_observation_result':v['metrics']['profile_error'],'registered_coverage_complete':v['gates']['complete_registered_coverage'],'scope':'Showcase recurrence/release measures descriptive only. Incomplete observational coverage prevents full primary qualification.'},'differences':differences,'controls':controls,'original_audio_timing_proxies':timing,'inputs':dict(pins),'artistic_acceptance':'pending_user_review','product_baseline_changed':False}
            write(root/'reports/comparisons'/f"{report['method']}_{song}.json",report);all_reports.append(report)
    from ..d4max_d8.source_checks import source_checks
    source=source_checks(root,song)
    write(root/'reports'/f'{song}_NUMERICAL_ADMISSION.json',{'comparisons':[{'method':r['method'],'physical_engineering_status':r['physical_engineering_status'],'failure_reasons':r['physical_failure_reasons']} for r in all_reports],'finite_actual_bodies_and_DMX':True,'source_controls':source,'render_policy':'Render finite failures as explicitly labeled diagnostics, never substitute motion.'})
    return all_reports


def timing_scores(pair,release,track):
    events=release.get('attack_times')
    if events is None:return {'status':'unmeasurable_no_audio_anchors'}
    body=pair['body'];series=d.dynamics_series(body.world,body.time);angles=pair['controls']['angles_deg'][:,pair['moving']]
    peaks=d.gesture_events(series['relative_speed_m_s'],body.time)
    return {'body':d.event_alignment(peaks,events,start_s=0.,end_s=track['duration_s']),
            'requested':ta.motor_event_alignment(pair['requested'],pair['arrays']['time'],events,start_s=0.,end_s=track['duration_s']),
            'emitted':ta.motor_event_alignment(angles,pair['arrays']['time'],events,start_s=0.,end_s=track['duration_s']),
            'scope':'Alignment to frozen original-song audio attacks; descriptive conditioning probe, not independent artistic evidence.'}


def cross_song(root=ROOT):
    root=Path(root);plan=read(root/'PLAN.json');tracks=plan['tracks'];music=[MusicTimeline.load(t['music_timeline']) for t in tracks];rows=[]
    for seed in (0,1):
        for arm,alias in ALIASES.items():
            method=alias+('' if seed==0 else '_seed1');paths=[root/'bodies'/method/t['id']/'body.npz' for t in tracks];bodies=[DanceResult.load(p) for p in paths]
            n=min(len(b.world) for b in bodies);a,b=bodies
            rows.append({'arm':arm,'seed':seed,'exact_same_prefix_body':bool(np.array_equal(a.world[:n],b.world[:n])),
                'body_prefix_root_relative_rms_m':float(np.sqrt(np.mean(((a.world[:n]-a.world[:n,:1])-(b.world[:n]-b.world[:n,:1]))**2))),
                'tempo_aligned':d.tempo_aligned_comparison(a.world,a.time,music[0].beats,b.world,b.time,music[1].beats),
                'inputs':{str(p):sha(p) for p in paths},'scope':'Different full songs, identical model and explicit maximum-length noise prefix; own recomputed audio paths. Difference is sensitivity, not choreography quality.'})
    write(root/'reports/CROSS_SONG.json',{'rows':rows});return rows
