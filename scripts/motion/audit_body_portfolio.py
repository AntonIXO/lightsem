"""Independent artifact audit and source-attribution sidecars for the portfolio.

Re-evaluating the frozen deterministic mapper here verifies saved intent; it
does not generate or replace DMX, retry a method, or alter any outcome.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise RuntimeError('Audit changed: '+str(path))
        return
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def audit(root):
    plan = read(root/'PLAN.json')
    source = Path(plan['source_root'])
    sys.path.insert(0,str(source))
    from lightsem.conductor.body_transfer_v2 import map_body, make_outputs
    from lightsem.conductor.body_transfer import sample_body
    for path, expected in plan['files'].items():
        if digest(path) != expected:
            raise RuntimeError('Frozen input drift: '+path)
    rows = []
    for track in plan['tracks']:
        with np.load(track['body']) as data:
            times,body = sample_body(data['world'])
        with np.load(track['baseline_timeline']) as data:
            old = data['universes'].copy()
        old_patch = read(track['baseline_patch'])
        rgb = {}
        for side in (True,False):
            p = next(p for p in old_patch if p['kind']!='bulb' and (p['position'][0]<0)==side)
            a = p['address']-1+5
            rgb[side] = old[:,p['universe'],a:a+3]
        for algorithm in plan['algorithms']:
            folder = root/'timelines'/algorithm['id']/track['id']
            report,patch = read(folder/'report.json'),read(folder/'patch.json')
            if digest(folder/'timeline.npz') != report['timeline_sha256'] or digest(folder/'patch.json') != report['patch_sha256']:
                raise RuntimeError('Saved output hash mismatch')
            intent = map_body(body,make_outputs(**plan['counts']),algorithm['id'])
            with np.load(folder/'timeline.npz') as data:
                for key in ('targets','dimmer','field_points','source_points'):
                    if not np.array_equal(data[key],intent[key]):
                        raise RuntimeError('Saved intent cannot be reproduced: '+key)
                for key in ('field_absolute_points','shared_stage_points','shared_role_mask','shared_similarity_scale'):
                    if key in intent and (key not in data or not np.array_equal(data[key],intent[key])):
                        raise RuntimeError('Saved shared structure cannot be reproduced: '+key)
                if not np.array_equal(data['body'],body) or not np.array_equal(data['time'],times):
                    raise RuntimeError('Saved body/timestamps changed')
                dmx = data['universes'].copy()
            angles = []
            occupied = set()
            for fixture in patch:
                a,u = fixture['address']-1,fixture['universe']
                width = fixture['n_channels']
                for slot in range(a,a+width):
                    if not 0<=slot<512 or (u,slot) in occupied:
                        raise RuntimeError('Overlapping/out-of-universe fixture allocation')
                    occupied.add((u,slot))
                channels = dmx[:,u,a:a+width].astype(np.int64)
                offset = 0
                if fixture['kind']!='bulb':
                    ticks = np.stack((channels[:,0]*256+channels[:,1],channels[:,2]*256+channels[:,3]),axis=1)
                    value = (ticks/65535.-.5)*np.array([fixture['pan_range_deg'],fixture['tilt_range_deg']])
                    angles.append(value)
                    offset = 4
                if not np.array_equal(channels[:,offset+1:offset+4],rgb[fixture['position'][0]<0]):
                    raise RuntimeError('P2 bytes changed')
            angle = np.stack(angles,axis=1)
            vmax = float(np.linalg.norm(np.diff(angle,axis=0)*40,axis=-1).max())
            amax = float(np.linalg.norm(np.diff(angle,n=2,axis=0)*1600,axis=-1).max())
            if vmax>90+1e-6 or amax>360+1e-6 or not np.isfinite(angle).all():
                raise RuntimeError('Independently decoded mechanics failed')
            preflight=read(root/'preflight'/algorithm['id']/(track['id']+'.json'))
            if not preflight['frozen_body_DMX_exact'] or not report['hard_invariants_passed']:
                raise RuntimeError('Hard gate failed')
            attribution={'algorithm':algorithm['id'],'track':track['id'],
                'source_manifest_sha256':digest(root/'source_manifest.json'),
                'timeline_sha256':report['timeline_sha256'],
                'method':intent['method'],'outputs':[{'name':p['name'],'source_joints':a}
                    for p,a in zip(patch,intent['attribution'])],
                'scope':'Dependency support, not a claim of equal sensitivity to every listed joint.'}
            save(folder/'ATTRIBUTION.json',attribution)
            rows.append({'algorithm':algorithm['id'],'track':track['id'],'intent_exact':True,
                'source_body_exact':True,'P2_RGB_exact':True,'fixtures':len(patch),
                'frames':len(dmx),'maximum_speed_deg_s':vmax,'maximum_acceleration_deg_s2':amax,
                'timeline_sha256':report['timeline_sha256'],'patch_sha256':report['patch_sha256'],
                'report_sha256':digest(folder/'report.json'),'attribution_sha256':digest(folder/'ATTRIBUTION.json'),
                'frozen_source_exact':preflight['frozen_body_DMX_exact'],
                'source_coverage':preflight['emitted_source_region_coverage']})
            print('VERIFIED',algorithm['id'],track['id'],flush=True)
    if len(rows)!=30 or len({(r['algorithm'],r['track']) for r in rows})!=30:
        raise RuntimeError('Incomplete comparison matrix')
    result={'status':'verified_30_emitted_timelines','plan_sha256':digest(root/'PLAN.json'),
        'source_manifest_sha256':digest(root/'source_manifest.json'),
        'audit_source':str(Path(__file__).resolve()),'audit_source_sha256':digest(__file__),
        'frozen_input_count':len(plan['files']),'rows':rows,
        'scope':'Independent byte decoding and deterministic intent reproduction; no artistic acceptance.'}
    save(root/'records/PRIMARY_NUMERIC_VERIFICATION.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    audit(args.root.resolve())
