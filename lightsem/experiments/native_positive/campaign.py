"""Fixed native-C library plus historical D4-max and positive D6; no new training."""
from pathlib import Path
from dataclasses import fields
import numpy as np
from ..dynamic_dance.contracts import DanceResult,DancePlan,MusicTimeline
from ..d4max_d8.motifs import Config,build_bank,compose
from ..d4max.pacing import apply_d6
from ..dynamic_dance_combo.campaign import read,save,sha,arrays,_save_composed,map_one
from ..dynamic_dance.diagnostics import body_dynamics,dynamics_series,recurrence_alignment


def metrics(body,music):
    d=body_dynamics(body.world,body.time,contacts=body.contacts)
    v=dynamics_series(body.world,body.time)['world_speed_m_s']
    return {'physical':d,'speed_p90_minus_p10':float(np.quantile(v,.9)-np.quantile(v,.1)),
            'finite':bool(np.isfinite(body.world).all()),
            'musical_recurrence_proxy':recurrence_alignment(body.world,body.time,music.muq,music.muq_time)}


def generate(root,song):
    root=Path(root);plan=read(root/'PLAN.json');t=next(x for x in plan['tracks'] if x['id']==song)
    music=MusicTimeline.load(t['music_timeline']);source_path=root/'bodies/D7_reference'/song/'body.npz';source=DanceResult.load(source_path)
    cfg=Config(**{k:v for k,v in plan['d4max'].items() if k in {f.name for f in fields(Config)}})
    bank=build_bank(source.world,music,cfg)
    dst=root/'banks'/song;dst.mkdir(parents=True,exist_ok=False)
    arrays(dst/'features.npz',features=bank['features'],labels=bank['labels'],medoids=bank['medoids'],**bank['feature_reference'])
    save(dst/'BANK.json',{k:v for k,v in bank.items() if k not in ('features','labels','medoids','feature_reference')})
    with np.load(t['local_plan_telemetry']) as z:telemetry={k:z[k].copy() for k in z.files}
    composed,dp,_=compose(source.world,music,cfg,bank=bank,target_music=music)
    result,planned,proposals=apply_d6(composed,dp,telemetry,music,minimum_rate=.25)
    composed_path=_save_composed(root,'D4_only',song,composed,dp,source_path,{'motion_origin':'Own-song native C motifs recomposed by D4-max','newly_generated_motion':False,'d6_enabled':False,'source_order_preserved':False})
    extra={'components':['native_joint_C_including_D8_hierarchy','D4_max','D6_positive_anchored'],'motion_origin':'Own-song native C fragments recomposed with MuQ and source descriptors, then paced by frozen D6','newly_generated_motion':False,'d6_enabled':True,'source_order_preserved':True,'fragment_clock_domain':'D4_only source time; original library jumps retained separately','d4_plan':str(composed_path.parent/'plan.npz'),'d4_plan_sha256':sha(composed_path.parent/'plan.npz'),'generated_source':str(source_path),'generated_source_sha256':sha(source_path),'planner_telemetry':t['local_plan_telemetry'],'planner_telemetry_sha256':sha(t['local_plan_telemetry']),'generation_provenance':source.metadata,'no_additional_legacy_D8_residual':True}
    p=_save_composed(root,'C4',song,result,planned,composed_path,extra)
    arrays(p.parent/'rate_proposals.npz',time=source.time,desired_rates=proposals)
    rows={'native_C':metrics(source,music),'historical_positive_C4':metrics(DanceResult.load(root/'bodies/C3'/song/'body.npz'),music),'D4_only':metrics(composed,music),'hybrid':metrics(result,music)}
    controls={}
    shifted,sp,_=apply_d6(composed,dp,telemetry,music,control='shift',minimum_rate=.25);controls['shifted_D6']=shifted
    shuffled,sdp,_=compose(source.world,music,cfg,bank=bank,target_music=music,control='shuffle');controls['shuffled_D4'],_,_=apply_d6(shuffled,sdp,telemetry,music,minimum_rate=.25)
    binding=t['waveform_controls']['donor'];donor=MusicTimeline.load(binding['music_timeline'])
    with np.load(binding['local_plan_telemetry']) as z:dt={k:z[k].copy() for k in z.files}
    dc,dcp,_=compose(source.world,music,cfg,bank=bank,target_music=donor);controls['donor_query'],_,_=apply_d6(dc,dcp,dt,donor,minimum_rate=.25)
    for key,value in controls.items():rows[key]=metrics(value,music)
    # Save actual control trajectories compactly for reproducibility and comparisons.
    arrays(root/'controls'/f'{song}.npz',time=source.time,**{key:value.world for key,value in controls.items()})
    sensitivity={key:float(np.sqrt(np.mean((value.world-result.world)**2))) for key,value in controls.items()}
    repeat,rp,_=apply_d6(composed,dp,telemetry,music,minimum_rate=.25)
    assert np.array_equal(repeat.world,result.world) and np.array_equal(rp.source_time,planned.source_time)
    save(root/'reports'/f'{song}_BODY.json',{'metrics':rows,'control_body_RMS_difference_m':sensitivity,'repeat_exact':True,'pacing':planned.metadata['pacing'],'source_body_sha256':sha(source_path),'body_sha256':sha(p),'source_banks_identify_own_song':True,'no_neural_collapse_claim':True})
    map_one(root,song,'C4')
    return rows

if __name__=='__main__':
    import sys
    generate(Path(sys.argv[1]),sys.argv[2])
