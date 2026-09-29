"""Preregistered optical ramps with a fixed camera and emitted DMX."""
import argparse
import importlib.util
import json
import sys
import types
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio_prepare import ROOT
from ...conductor.rig import canonical_rig
from ...sim.renderer import StageRenderer,RenderConfig
from ..stage11.stage11_common import read_authenticated,authenticated_json,digest,freeze_sources,verify_sources,gpu_lock

OUT = ROOT/'exposure'
ROLE_REVISION = False


def freeze():
    out = OUT
    if (out/'protocol.json').exists():
        return read_authenticated(out/'protocol.json')
    snapshot = freeze_sources(out)
    legacy = ROOT/'music/source/lightsem/sim'
    p = {'source_sha256':snapshot['source_sha256'],
        'legacy_sources':{str(legacy/name):digest(legacy/name) for name in ('renderer.py','shaders.py')},
        'levels':np.linspace(0,1,21).tolist(),'scenes':['single_head','all_heads','whole_rig'],
        'scales':[.08,.12,.16,.20,.24],
        'render':{'width':854,'height':480,'fps':30.,'haze':.25,'cam_orbit':0.,'intensity_gain':2.4},
        'primary':'whole-rig bulb-region luminance gain L(.8)-L(0) over L(.2)-L(0)',
        'gates':{'default_pixel_max_abs':0,'monotonic_tolerance':1e-4,'response_min':2.,
                 'bulb_luma_at_point1_min':.01,'whole_frame_clipped_white_max':.01,
                 'bulb_clipped_white_max':.10,'nonbulb_luma_above_dark_at_point8_min':.002},
        'selection':'largest fixed scale passing every ramp gate across all three scenes; none if no scale qualifies',
        'frozen_time':0.,'geometry':'static fixture pan/tilt and white color; DMX dimmers only vary',
        'test_access':False,'model_outputs_used':False}
    if ROLE_REVISION:
        p['scales'] = [.4]
        p['fixed_role_gains'] = [1.,.3,.03,.2]
        p['selection'] = 'Fixed primary scale0.4 and role gains1/.3/.03/.2; qualification still requires every original gate.'
        p['parent_protocol_sha256'] = digest(ROOT/'exposure/protocol.json')
    authenticated_json(out/'protocol.json',p)
    return p


def context():
    out = OUT
    p = read_authenticated(out/'protocol.json')
    snapshot = verify_sources(out)
    if not Path(__file__).resolve().is_relative_to(Path(snapshot['snapshot']).resolve()):
        raise ValueError('Use the frozen exposure launcher')
    for path,sha in p['legacy_sources'].items():
        if digest(path)!=sha:
            raise ValueError('Legacy renderer source changed')
    receipt = read_authenticated(out/'experiment.json')
    if receipt['protocol_sha256']!=digest(out/'protocol.json'):
        raise ValueError('Exposure experiment not started')
    return out,p


def old_renderer():
    """Import the actual historical renderer/shader under an isolated package."""
    import lightsem.dmx.fixtures as fixtures
    prefix = '_lightsem_exposure_baseline'
    for suffix in ('','.sim','.dmx'):
        module = types.ModuleType(prefix+suffix)
        module.__path__ = []
        sys.modules[module.__name__] = module
    sys.modules[prefix+'.dmx.fixtures'] = fixtures
    root = ROOT/'music/source/lightsem/sim'
    for name in ('shaders','renderer'):
        spec = importlib.util.spec_from_file_location(prefix+'.sim.'+name,root/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def masks(rig,cfg,selected):
    position = np.asarray(cfg.cam_pos)
    forward = np.asarray(cfg.cam_target)-position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward,[0,1,0]);right /= np.linalg.norm(right)
    up = np.cross(right,forward)
    yy,xx = np.mgrid[:cfg.height,:cfg.width]
    result = []
    for index in selected:
        delta = np.asarray(rig.fixtures[index].position)-position
        depth = float(delta@forward)
        if depth<=0:
            raise ValueError('Fixture outside forward camera volume')
        u = (delta@right)/depth/.85/(cfg.width/cfg.height)
        v = (delta@up)/depth/.85
        x,y = (u+1)*cfg.width/2,(1-v)*cfg.height/2
        radius = max(3.,.24*cfg.height/(1.7*depth))
        mask = (xx-x)**2+(yy-y)**2<=radius**2
        if not mask.any():
            raise ValueError('Fixture bulb is outside screen')
        result.append(mask)
    return result


def metrics(image,regions):
    x = image.astype(np.float64)/255
    y = x@np.asarray([.2126,.7152,.0722])
    union = np.logical_or.reduce(regions)
    clipped = x.min(-1)>=.98
    return {'bulb_luma':float(np.mean([y[m].mean() for m in regions])),
            'bulb_clipped':float(np.mean([clipped[m].mean() for m in regions])),
            'frame_clipped':float(clipped.mean()),'nonbulb_luma':float(y[~union].mean()),
            'frame_luma':float(y.mean())}


def run():
    from PIL import Image,ImageDraw
    out,p = context()
    if (out/'report.json').exists():
        return read_authenticated(out/'report.json')
    rig = canonical_rig()
    cfg = RenderConfig(**p['render'])
    new = StageRenderer(rig,cfg)
    old_module = old_renderer()
    old = old_module.StageRenderer(rig,old_module.RenderConfig(**p['render']))
    head_indices = [i for i,f in enumerate(rig.fixtures) if f.kind=='moving_head']
    scenes = {'single_head':head_indices[:1],'all_heads':head_indices,'whole_rig':list(range(len(rig.fixtures)))}
    arrays,rows,parity = {},[],0
    for name,selected in scenes.items():
        regions = masks(rig,cfg,selected)
        for f in rig.fixtures:
            f.state.pan=.5;f.state.tilt=.43;f.state.zoom=.25;f.state.saturation=0.;f.state.strobe=0.
        for level in p['levels']:
            for i,f in enumerate(rig.fixtures):
                f.state.dimmer=level if i in selected else 0.
            frame = rig.universe()
            with old.ctx:
                old.fbo.use()
                previous = old.render_frame(frame,0.)
            with new.ctx:
                new.fbo.use()
                default = new.render_frame(frame,0.)
            parity = max(parity,int(np.abs(previous.astype(int)-default.astype(int)).max()))
            rows.append({'scene':name,'profile':'legacy','scale':None,'level':level,'metrics':metrics(previous,regions)})
            if abs(level-.1)<1e-8 or abs(level-.8)<1e-8:
                arrays[(name,'legacy',level)] = previous
            for scale in p['scales']:
                new.cfg.exposure_profile='fixed'
                new.cfg.fixed_exposure_scale=scale
                new.cfg.fixed_role_gains=p.get('fixed_role_gains',[1.,1.,1.,1.])
                with new.ctx:
                    new.fbo.use()
                    image = new.render_frame(frame,0.)
                rows.append({'scene':name,'profile':'fixed','scale':scale,'level':level,'metrics':metrics(image,regions)})
                if abs(level-.1)<1e-8 or abs(level-.8)<1e-8:
                    arrays[(name,str(scale),level)] = image
            new.cfg.exposure_profile='legacy'
        print(json.dumps({'scene':name,'default_parity_max':parity}),flush=True)
    new.ctx.release();old.ctx.release()
    g = p['gates']
    candidates = []
    def value(group,level,metric):
        return min(group,key=lambda r:abs(r['level']-level))['metrics'][metric]
    for scale in p['scales']:
        by_scene = {}
        for scene in scenes:
            selected = sorted([r for r in rows if r['scale']==scale and r['scene']==scene],key=lambda r:r['level'])
            response = (value(selected,.8,'bulb_luma')-value(selected,0,'bulb_luma'))/max(value(selected,.2,'bulb_luma')-value(selected,0,'bulb_luma'),1e-9)
            gates = {'ramp_monotonicity':all(b['metrics']['bulb_luma']-a['metrics']['bulb_luma']>=-g['monotonic_tolerance'] for a,b in zip(selected,selected[1:])),
                'dynamic_range':response>=g['response_min'],
                'quiet_visibility':value(selected,.1,'bulb_luma')>=g['bulb_luma_at_point1_min'],
                'clipping':max(r['metrics']['frame_clipped'] for r in selected)<=g['whole_frame_clipped_white_max'] and max(r['metrics']['bulb_clipped'] for r in selected)<=g['bulb_clipped_white_max'],
                'beam_visibility':value(selected,.8,'nonbulb_luma')-value(selected,0,'nonbulb_luma')>=g['nonbulb_luma_above_dark_at_point8_min']}
            by_scene[scene] = {'gates':gates,'response':response,'passed':all(gates.values())}
        candidates.append({'scale':scale,'scenes':by_scene,'passed':all(v['passed'] for v in by_scene.values()) and parity<=g['default_pixel_max_abs']})
    eligible = [r for r in candidates if r['passed']]
    selected = max(eligible,key=lambda r:r['scale'])['scale'] if eligible else None
    # Preserve every candidate curve and image; selection never hides a failure.
    image_paths = []
    for scene in scenes:
        grid = Image.new('RGB',(854*2,510*(1+len(p['scales']))),'white')
        pen = ImageDraw.Draw(grid)
        for row,profile in enumerate(['legacy']+[str(s) for s in p['scales']]):
            for col,level in enumerate((.1,.8)):
                pen.text((col*854+8,row*510+5),f'{scene} / {profile} / dimmer {level}',fill='black')
                grid.paste(Image.fromarray(arrays[(scene,profile,level)]),(col*854,row*510+25))
        path = out/f'{scene}_all_scales.png'
        grid.save(path);image_paths.append(str(path))
    legacy_rows = [r for r in rows if r['scene']=='whole_rig' and r['profile']=='legacy']
    before = (value(legacy_rows,.8,'bulb_luma')-value(legacy_rows,0,'bulb_luma'))/max(value(legacy_rows,.2,'bulb_luma')-value(legacy_rows,0,'bulb_luma'),1e-9)
    result = {'rows':rows,'candidates':candidates,'selected_scale':selected,'default_pixel_max_abs':parity,
              'legacy_response':before,'images':{p:digest(p) for p in image_paths},'protocol_sha256':digest(out/'protocol.json')}
    authenticated_json(out/'report.json',result)
    print(json.dumps({'selected_scale':selected,'default_pixel_max_abs':parity}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run'])
    args=parser.parse_args()
    if args.stage=='freeze':freeze()
    else:
        with gpu_lock(ROOT):run()
