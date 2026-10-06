"""Fixed V3-01 body-only mapping with exact per-song P2 RGB and emitted audits."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .campaign import array_sha, read, sha, verify_files, write_new
from .contracts import DanceResult
from lightsem.conductor.body_transfer import TransferConfig, sample_body, compile_intent, decode
from lightsem.conductor.body_transfer_v2 import make_outputs, map_body
from lightsem.conductor.body_transfer_v2.diagnostics import measure


def implementation_pins():
    import lightsem.conductor.body_transfer as transfer
    import lightsem.conductor.body_transfer_v2 as mapper
    from . import retiming, diagnostics
    paths=[Path(__file__),Path(transfer.__file__),Path(retiming.__file__),Path(diagnostics.__file__)]
    paths.extend(Path(mapper.__file__).parent.glob('*.py'))
    return {str(p):sha(p) for p in sorted(set(paths))}


def lighting_body(world):
    from .retiming import resample_body
    times=np.arange(round(len(world)/30.*40.))/40.
    source=np.minimum(times,(len(world)-1)/30.)
    return times,resample_body(world,source,30.)


def baseline_body(track):
    import torch
    from lightsem.experiments.motion_study.geometry import contacts_from_joints
    with np.load(track['body'], allow_pickle=False) as data:
        world, times = data['world'].copy(), data['time'].copy()
    if world.shape != (5400, 22, 3) or not np.isfinite(world).all():
        raise ValueError('Expected authenticated per-song 180-second body')
    contacts=contacts_from_joints(torch.from_numpy(world)).numpy()
    return DanceResult(time=times, world=world, contacts=contacts,
                       source_time=times.copy(), metadata={
                           'method': 'baseline', 'song': track['id'],
                           'source_body': track['body'], 'source_sha256': sha(track['body']),
                           'provenance': read(track['body_provenance']),
                           'scope': 'Accepted audio-conditioned body, not an unconditional motion source.'}).validate()


def p2_colors(track):
    with np.load(track['baseline_timeline'], allow_pickle=False) as data:
        universes = data['universes'].copy()
    patch = read(track['baseline_patch'])
    result = []
    for left in (True, False):
        group = [p for p in patch if (p['position'][0] < 0) == left]
        moving = next(p for p in group if p['kind'] != 'bulb')
        slot = moving['address'] - 1 + 5
        rgb = universes[:, moving['universe'], slot:slot+3]
        for fixture in group:
            offset = 1 if fixture['kind'] == 'bulb' else 5
            start = fixture['address']-1+offset
            if not np.array_equal(rgb, universes[:,fixture['universe'],start:start+3]):
                raise ValueError('Protected P2 is not the declared two-group stream')
        result.append(rgb.copy())
    return result


def compile_body(world, left, right, plan, counts=None):
    """Caller supplies actual 40 Hz body; no audio enters this function."""
    from scripts.motion.run_body_portfolio import hard_checks
    outputs = make_outputs(**(plan['counts'] if counts is None else counts))
    config = TransferConfig(**plan['transfer'])
    intent = map_body(world, outputs, '01', config.fps)
    dmx, patch, intent = compile_intent(intent, outputs, left, right, config)
    hard = hard_checks(dmx, patch, intent, config.fps)
    decoded = decode(dmx, patch)
    for index, fixture in enumerate(patch):
        expected = left if fixture['position'][0] < 0 else right
        actual = np.rint(decoded['rgb'][:,index]*255).astype(np.uint8)
        if not np.array_equal(actual, expected):
            raise RuntimeError('Mapping changed frozen P2 RGB')
    return dmx, patch, intent, hard


def map_result(root, plan, method, track):
    from scripts.motion.run_body_portfolio import angular_ticks
    root = Path(root)
    verify_files(plan['files'])
    folder = root / 'timelines' / method / track['id']
    result_path = root / 'bodies' / method / track['id'] / 'body.npz'
    result = DanceResult.load(result_path)
    plan_path = result_path.parent/'plan.npz'
    pins = {'plan_sha256': sha(root/'PLAN.json'), 'body_sha256': sha(result_path),
            'body_metadata_sha256': sha(result_path.with_suffix('.json')),
            'P2_timeline_sha256': sha(track['baseline_timeline']),
            'P2_patch_sha256': sha(track['baseline_patch']),
            'implementation':implementation_pins(),
            'dance_plan':({str(p):sha(p) for p in (plan_path,plan_path.with_suffix('.json'))}
                          if plan_path.exists() else None)}
    if folder.exists():
        report = read(folder/'report.json')
        if report['inputs'] != pins:
            raise RuntimeError('Completed mapping inputs changed')
        verify_files({str(folder/'timeline.npz'): report['timeline_sha256'],
                      str(folder/'patch.json'): report['patch_sha256']})
        return report
    if result.world.shape != (5400,22,3) or not np.allclose(result.time,np.arange(5400)/30,atol=1e-5):
        raise ValueError('All comparisons require the same 180-second, 30 Hz body clock')
    times, world = lighting_body(result.world)
    left,right = p2_colors(track)
    dmx,patch,intent,hard = compile_body(world,left,right,plan)
    numeric = measure(world,intent,dmx,patch,40.,60.,120.)
    from .diagnostics import body_dynamics
    original = baseline_body(track)
    body_report = body_dynamics(result.world,result.time,contacts=result.contacts,
                               reference_body=original.world)
    arrays = {'time':times,'body':world,'universes':dmx,'targets':intent['targets'],
              'source_points':intent['source_points'],'field_points':intent['field_points'],
              'dimmer':intent['dimmer']}
    if result.source_time is not None:
        source = np.asarray(result.source_time)
        if source.ndim == 1:
            resampled = np.interp(times,result.time,source)
        else:
            resampled = np.stack([np.interp(times,result.time,source[:,i])
                                  for i in range(source.shape[1])],axis=1)
        arrays['source_time'] = resampled
        arrays['playback_rate'] = np.gradient(resampled,1/40,axis=0)
    if plan_path.exists():
        from .contracts import DancePlan
        dance_plan = DancePlan.load(plan_path)
        if result.metadata.get('method')=='D4' or method.split('__')[0]=='D4':
            local_rates=np.asarray(dance_plan.effort)
            arrays['playback_rate']=np.stack([np.interp(times,dance_plan.time,local_rates[:,i])
                                              for i in range(local_rates.shape[1])],axis=1)
        markers = []
        for entry in dance_plan.anchors + dance_plan.events:
            value = entry.get('time',entry.get('target_time',entry.get('time_s')))
            if isinstance(value,(float,int)) and np.isfinite(value):
                markers.append(float(value))
        arrays['intervention_time'] = np.asarray(sorted(set(markers)),np.float64)
    folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'timeline.npz',**arrays)
    write_new(folder/'patch.json',patch)
    report = {'algorithm':method,'track':track['id'],
              'status':'diagnostic','diagnostic_reasons':['Musical superiority and human acceptance are not established by mapping validity.'],
              'inputs':pins,'body_sha256':sha(result_path),
              'timeline_sha256':sha(folder/'timeline.npz'),'patch_sha256':sha(folder/'patch.json'),
              'emitted_DMX_sha256':array_sha(dmx),'emitted_motion_sha256':array_sha(angular_ticks(dmx,patch)),
              'hard_invariants_passed':True,'P2_RGB_exact':True,'hard_checks':hard,
              'diagnostics':numeric,'body_diagnostics':body_report,
              'frames':len(world),'method':'Fixed V3-01 bone directions',
              'scope':'Actual emitted outputs; color, brightness and geometric motion measured separately.'}
    write_new(folder/'report.json',report)
    return report


def source_preflight(root,plan,track):
    from scripts.motion.run_body_portfolio import angular_ticks,intervention,REGIONS
    path = Path(root)/'preflight'/('mapper_'+track['id']+'.json')
    pins={'plan_sha256':sha(Path(root)/'PLAN.json'),'source_body_sha256':sha(track['body']),
          'P2_sha256':sha(track['baseline_timeline']),'P2_patch_sha256':sha(track['baseline_patch']),
          'implementation':implementation_pins()}
    if path.exists():
        prior=read(path)
        if prior.get('inputs')!=pins or prior.get('status')!='passed':
            raise RuntimeError('Changed or failed source preflight must not be reused')
        return prior
    body = baseline_body(track).world
    pose = body[len(body)//2]
    frames=101
    stationary=np.repeat(pose[None],frames,axis=0)
    colors=[np.repeat(rgb[3600:3601],frames,axis=0) for rgb in p2_colors(track)]
    cases=[]
    for counts in ({'beams':1,'lasers':0,'bulbs':0},plan['counts'],
                   {'beams':23,'lasers':11,'bulbs':83}):
        dmx,patch,_,hard=compile_body(stationary,*colors,plan,counts)
        ticks=angular_ticks(dmx,patch)
        exact=bool(np.all(dmx==dmx[:1]))
        if not exact:
            raise RuntimeError('Stationary body creates autonomous emitted output')
        row={'counts':counts,'stationary_exact':exact,'hard_checks':hard,'regions':{}}
        for region in REGIONS:
            changed=intervention(pose,region,frames,12.)
            other,opatch,_,limits=compile_body(changed,*colors,plan,counts)
            delta=np.abs(angular_ticks(other,opatch)-ticks)
            names=[p['name'] for p in patch if p['kind']!='bulb']
            response=np.any(delta>=1,axis=(0,2))
            row['regions'][region]={'response_outputs':[n for n,yes in zip(names,response) if yes],
                                     'max_ticks':int(delta.max()),'hard_checks':limits}
        cases.append(row)
    full=cases[1]
    if any(r['max_ticks']<1 for r in full['regions'].values()):
        raise RuntimeError('Required installation misses an anatomical source response')
    report={'track':track['id'],'status':'passed','cases':cases,'inputs':pins,
            'scope':'Synthetic source interventions are tests only, never demo motion. Sparse rigs need not cover every body region.'}
    write_new(path,report)
    return report
