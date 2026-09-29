"""CPU-only synthetic tests. Never create EGL contexts or load study DMX."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import unittest
import tempfile
import math

PATH=Path(__file__).parents[1]/'scripts/motion/run_dance_optics_full.py'
spec=importlib.util.spec_from_file_location('dance_optics_full',PATH)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)


def cfg(**kw):
    d=dict(cam_pos=(0.,0.,10.),cam_target=(0.,0.,0.),camera_scale=1.,width=640,height=360,
           beam_edge_profile='legacy',beam_edge_softness_deg=.6)
    return SimpleNamespace(**(d|kw))


def rig():
    heads=[SimpleNamespace(address=1+12*i,name=f'mh{i+1}',kind='moving_head',position=(i-2.5,6.,-1.),
        pan_range_deg=540.,tilt_range_deg=270.,beam_min_deg=6.,beam_max_deg=32.) for i in range(6)]
    wash=SimpleNamespace(address=100,name='wash',kind='wash',position=(0.,4.,-1.))
    return SimpleNamespace(fixtures=heads+[wash],by_kind=lambda k: heads if k=='moving_head' else [wash])


def test_no_native_project_imports():
    # Safe even when unrelated suite tests already imported native modules.
    import ast
    nodes=ast.parse(PATH.read_text()).body
    imports=[n.module for n in nodes if isinstance(n,ast.ImportFrom)]
    assert not any(name and name.startswith('lightsem') for name in imports)


def test_dmx_parity_and_foil_scope():
    source=np.random.default_rng(23).integers(0,256,(4000,512),dtype=np.uint8);r=rig()
    saved=source.copy()
    for zoom in ('native','zero'):
        d,p=m.transform_dmx(source,r,{'zoom':zoom,'frozen_angles':False})
        assert p['source_angle_sha256']==p['arm_angle_sha256']
        allowed=[f.address+8 for f in r.by_kind('moving_head')]
        keep=np.ones(512,bool);keep[allowed]=False
        assert np.array_equal(d[:,keep],source[:,keep])
        if zoom=='zero':assert np.all(d[:,allowed]==0)
    foil,p=m.transform_dmx(source,r,{'zoom':'zero','frozen_angles':True,'freeze_time_s':80.})
    for f in r.by_kind('moving_head'):
        a=f.address-1
        assert np.array_equal(foil[:,a:a+4],np.broadcast_to(source[3200,a:a+4],(4000,4)))
        assert np.array_equal(foil[:,a+4:a+9],source[:,a+4:a+9])
    assert np.array_equal(source,saved)
    assert p['nonallowed_bytes_identical']


def test_bad_dmx_and_foil_fail():
    with unittest.TestCase().assertRaises(ValueError):m.transform_dmx(np.zeros((3,512)),rig(),{'zoom':'native'})
    with unittest.TestCase().assertRaises(ValueError):m.transform_dmx(np.zeros((3,512),np.uint8),rig(),{'zoom':'zero','frozen_angles':True,'freeze_time_s':80.})


def test_projection_pixel_center_and_scale():
    p=m.project(np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]),cfg())
    np.testing.assert_allclose(p,[[319.5,179.5],[337.5,179.5],[319.5,161.5]])
    q=m.project(np.array([[1.,0.,0.]]),cfg(camera_scale=.5))
    assert q[0,0]-319.5==36
    assert np.isnan(m.project([[0,0,11]],cfg())).all()


def test_first_hits_finite_surfaces_and_truss():
    p=np.array([[0,6,-1],[0,3,0],[0,5,0],[20,3,0]],float)
    v=np.array([[0,-1,0],[0,0,-1],[0,0,-1],[0,-1,0]],float)
    hit,kind,distance=m.first_hits(v,p)
    assert kind.tolist()==[0,1,1,-1]
    np.testing.assert_allclose(hit[:3],[[0,0,-1],[0,3,-6],[0,5,-6]])
    assert np.isnan(hit[3]).all()
    # Upward emitted axis reaches underside of horizontal truss first.
    hit,kind,distance=m.first_hits(np.array([[0,1,0.]]),np.array([[0,6,-1.]]))
    assert kind[0]==2
    np.testing.assert_allclose(distance,[.14],atol=1e-12)
    # Upright intersects before wall, including parallel slab dimensions.
    hit,kind,distance=m.first_hits(np.array([[0,0,-1.]]),np.array([[-6,3,0.]]))
    assert kind[0]==3
    np.testing.assert_allclose(distance,[.86])


def test_relative_half_peak_is_not_absolute_half():
    angle=m.half_peak_angle(np.deg2rad(np.array([3.])), 'legacy')[0]
    assert math.isclose(np.rad2deg(angle),7.2126,abs_tol=.002)
    absolute=np.arccos(np.cos(np.deg2rad(3))-.0054)
    assert angle>absolute
    lo=np.cos(np.deg2rad(3))-.018;hi=np.cos(np.deg2rad(3))+.0072
    smooth=lambda x: np.clip((x-lo)/(hi-lo),0,1)**2*(3-2*np.clip((x-lo)/(hi-lo),0,1))
    assert math.isclose(smooth(np.cos(angle)),.5*smooth(1.),abs_tol=1e-12)


def test_angular_half_peak_and_flux_width_invariance():
    nominal=np.deg2rad(np.array([3.,7.,16.]))
    angle=m.half_peak_angle(nominal,'angular',.6)
    expected=np.arccos((np.cos(nominal-.6*np.pi/180)+np.cos(nominal+.6*np.pi/180))/2)
    np.testing.assert_allclose(angle,expected,atol=1e-12)
    # Flux isn't an input: positive multiplicative normalization cancels at half peak.
    assert 'flux' not in m.half_peak_angle.__code__.co_varnames


def test_circle_polygon_radius():
    radius=m.circle_radius(np.array([[0.,0.,0.]]),np.array([[0.,0.,-1.]]),np.array([1.]),cfg())
    # 64-side polygon is slightly smaller than the continuous projected disk.
    expected=18*np.sqrt(64*np.sin(2*np.pi/64)/(2*np.pi))
    assert math.isclose(radius[0],expected,abs_tol=1e-10)


def test_welch_vector_rms_and_static():
    t=np.arange(800)/40
    x=np.stack((2*np.sin(2*np.pi*1.5*t),np.zeros_like(t)),-1)[:,None,:]
    assert math.isclose(m.band_rms(x)[0],np.sqrt(2),rel_tol=.002)
    assert m.band_rms(np.ones_like(x))[0]<1e-12
    with unittest.TestCase().assertRaises(ValueError):m.band_rms(x*np.nan)


def test_raw_pixel_masks_and_two_clipping_definitions():
    r=rig();c=cfg(cam_pos=(0,6,11),cam_target=(0,2,-3),camera_scale=.45)
    masks,xy=m.pixel_masks(r,c)
    assert not masks['outside_caption'][:28].any()
    assert np.array_equal(masks['outside_bulbs']|masks['bulbs_only'],masks['outside_caption'])
    assert not (masks['outside_bulbs']&masks['bulbs_only']).any()
    image=np.zeros((360,640,3),np.uint8)
    y,x=np.argwhere(masks['outside_bulbs'])[0]
    image[y,x]=[250,0,0]
    y2,x2=np.argwhere(masks['outside_bulbs'])[1];image[y2,x2]=[250,250,250]
    image[:28]=255
    stats=m.pixel_stats(image,masks);n=masks['outside_bulbs'].sum()
    assert math.isclose(stats['outside_bulbs_white_ge250'],1/n)
    assert math.isclose(stats['outside_bulbs_anychannel_ge250'],2/n)
    assert math.isclose(stats['outside_bulbs_support_gt5'],2/n)


def test_brightness_single_global_scalar_and_rounding():
    scales=m.brightness_scalars([20.,5.]);np.testing.assert_equal(scales,[.25,1.])
    x=np.array([[[0,1,2],[5,100,255]],[[30,90,200],[20,40,60]]],np.uint8)
    y=m.dim_rgb(x,scales[0]);np.testing.assert_equal(y,np.rint(x*.25).astype(np.uint8))
    assert np.max(abs(y-x*.25))<=.5
    with unittest.TestCase().assertRaises(ValueError):m.brightness_scalars([0,1])
    with unittest.TestCase().assertRaises(ValueError):m.dim_rgb(x,1.1)


def test_decimal_frame_count_no_float_extra_frame():
    assert m.frame_count(277.3,30.)==8319
    assert m.frame_count(.051,30)==2


def test_receipts_create_only_and_detect_tamper():
    temporary=tempfile.TemporaryDirectory()
    tmp_path=Path(temporary.name)
    p=tmp_path/'x.json';m.write_json(p,{'hello':'world'})
    assert m.authenticated(p)=={'hello':'world'}
    with unittest.TestCase().assertRaises(FileExistsError):m.write_json(p,{'no':'overwrite'})
    p.chmod(0o644);p.write_text('{}')
    with unittest.TestCase().assertRaises(RuntimeError):m.authenticated(p)
    temporary.cleanup()


def test_preflight_gates():
    e={'maximum_byte_difference':0,'nonhead_edge_mode_parity':True,'new_controls_execute':True,
       'config_preview':{'beam_flux_reference_deg':14.}}
    m.validate_engineering(e)
    for broken in [e|{'maximum_byte_difference':1},e|{'nonhead_edge_mode_parity':False},e|{'config_preview':{}}]:
        with unittest.TestCase().assertRaises(ValueError):m.validate_engineering(broken)


def test_factorial_all_single_factor_contrasts():
    results={}
    for c,z,e in m.itertools.product(('legacy','closer'),('native','zero'),('legacy','angular_peak','angular_flux')):
        score=(1 if c=='legacy' else 2)*(1 if z=='native' else 3)*(1 if e=='legacy' else 4)
        results[f'{c}_{z}_{e}']={'active':{'valid':True,'primary':score,
            'heads':[{'name':str(i),'motion_radius_ratio':score} for i in range(6)]}}
    contrast=m.contrasts(results)
    assert len(contrast)==24 # camera6 + zoom6 + cone(3pairs*4contexts)
    assert all(len(v['per_head'])==6 for v in contrast)
    assert all(v['ratio']==2 for v in contrast if v['factor']=='camera')



def test_synthetic_whole_geometry_and_invalid_fail_closed():
    r=rig();dmx=np.zeros((800,512),np.uint8)
    time=np.arange(800)/40
    for i,f in enumerate(r.by_kind('moving_head')):
        a=f.address-1
        ticks=np.rint(32768+50*np.sin(2*np.pi*1.5*time)).astype(int)
        dmx[:,a]=ticks>>8;dmx[:,a+1]=ticks&255;dmx[:,a+2]=128
        dmx[:,a+9]=64
    c=cfg(cam_pos=(0,6,11),cam_target=(0,2,-3),camera_scale=.45)
    result,series=m.geometry_arm(dmx,r,c,{'active':[0,20]})
    assert result['active']['valid']
    assert result['active']['primary']>0
    assert len(result['active']['heads'])==6
    assert series['midshaft_px'].shape==(800,6,2)
    static=np.broadcast_to(dmx[0],dmx.shape)
    result,_=m.geometry_arm(static,r,c,{'active':[0,20]})
    assert result['active']['primary']<1e-12
    # Reverse camera means every projected point is behind it: fail, don't drop.
    result,_=m.geometry_arm(dmx,r,cfg(cam_target=(0,0,20)),{'active':[0,20]})
    assert not result['active']['valid']
    assert result['active']['primary'] is None


def test_engineering_binding_requires_inputs_and_rejects_drift():
    old=m.REPO
    with tempfile.TemporaryDirectory() as folder:
        base=Path(folder);(base/'lightsem').mkdir()
        py=base/'lightsem/a.py';js=base/'lightsem/a.json';dmx=base/'input.npz'
        py.write_text('x=1');js.write_text('{}');dmx.write_bytes(b'fixed_input')
        m.REPO=base
        try:
            e={'before_after_source_parity':True,'source_hashes':{str(p):m.digest(p) for p in (py,js,dmx)}}
            d={'source':str(dmx)};m.validate_engineering_binding(e,d)
            bad={'before_after_source_parity':True,'source_hashes':{str(py):m.digest(py)}}
            with unittest.TestCase().assertRaises(ValueError):m.validate_engineering_binding(bad,d)
            with unittest.TestCase().assertRaises(ValueError):m.validate_engineering_binding({},d)
            py.write_text('x=2')
            with unittest.TestCase().assertRaises(ValueError):m.validate_engineering_binding(e,d)
        finally:m.REPO=old


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(v) for k,v in sorted(globals().items()) if k.startswith('test_') and callable(v))


if __name__=='__main__':unittest.main()
