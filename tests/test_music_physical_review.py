"""CPU synthetic tests only. No checkpoints, VAL arrays, inference or GPU."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path('/root/lightsem')
spec=importlib.util.spec_from_file_location('physical_visual',ROOT/'scripts/motion/review_music_physical_revision.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
from lightsem.experiments.motion_study import lighting_map as lm
from lightsem.experiments.motion_study.geometry import joints_from_features
from lightsem.experiments.motion_study.articulated_stage import infer_root_rotation


def synthetic(n=32):
    f=np.zeros((n,139),np.float32);f[:,2]=.9
    f[:,3:135]=np.tile([1.,0.,0.,0.,1.,0.],22)
    f[:,0]=.12;f[:,1]=-.03
    # Root rotates; test inference beyond the neutral pose.
    theta=np.linspace(-.4,.4,n);f[:,3:9]=np.stack((np.cos(theta),np.zeros(n),-np.sin(theta),np.zeros(n),np.ones(n),np.zeros(n)),1)
    curve={'time':np.array([0.,n/30]),'momentary':np.array([-20.,-18.]),'short_term':np.array([-21.,-19.])}
    return f,curve

class WorldSeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Immutable pre-review training snapshot, CPU source only (not a model).
        # Its original relative import needs the companion module from that
        # same snapshot after the live package was reorganized.
        import sys
        companion='lightsem.experiments.beat_conditioned_render'
        companion_path=v.EXPERIMENT/'source/lightsem/experiments/beat_conditioned_render.py'
        companion_spec=importlib.util.spec_from_file_location(companion,companion_path)
        historical=importlib.util.module_from_spec(companion_spec)
        sys.modules[companion]=historical
        companion_spec.loader.exec_module(historical)
        path=v.EXPERIMENT/'source/lightsem/experiments/motion_study/lighting_map.py'
        name='lightsem.experiments.motion_study._pre_review_lighting_map'
        spec=importlib.util.spec_from_file_location(name,path)
        try:
            cls.old=importlib.util.module_from_spec(spec);sys.modules[name]=cls.old;spec.loader.exec_module(cls.old)
        finally:
            del sys.modules[companion]

    def test_default_dmx_and_trace_byte_parity(self):
        f,c=synthetic()
        for dynamics in (True,False):
            new=lm.compose(f,c,len(f)/30,dynamics=dynamics)
            old=self.old.compose(f,c,len(f)/30,dynamics=dynamics)
            np.testing.assert_array_equal(new[0],old[0])
            for key in old[1]:np.testing.assert_array_equal(new[1][key],old[1][key])

    def test_world_helper_avoids_fk_and_preserves_input(self):
        f,c=synthetic();world=joints_from_features(torch.from_numpy(f)).numpy();before=world.copy()
        with patch.object(lm,'joints_from_features',side_effect=AssertionError('FK forbidden')):
            a=lm.compose_world_joints(world,c,len(f)/30)
        np.testing.assert_array_equal(world,before);self.assertEqual(a[0].shape,(round(len(f)/30*40),512))

    def test_world_inferred_orientation_near_native(self):
        f,c=synthetic();world=joints_from_features(torch.from_numpy(f)).numpy()
        a=lm.compose(f,c,len(f)/30);b=lm.compose_world_joints(world,c,len(f)/30)
        # Inferred root frame float rounding can change final 8bit rounding.
        np.testing.assert_allclose(a[1]['palette_hue'],b[1]['palette_hue'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(a[1]['raw_targets'],b[1]['raw_targets'],atol=1e-8,rtol=0)

    def test_same_world_identical_for_all_arms(self):
        f,c=synthetic();world=joints_from_features(torch.from_numpy(f)).numpy()
        a=lm.compose_world_joints(world,c,len(f)/30)
        b=lm.compose(f*7,c,len(f)/30,world_joints_override=world)
        np.testing.assert_array_equal(a[0],b[0])
        for k in a[1]:np.testing.assert_array_equal(a[1][k],b[1][k])

    def test_world_validation(self):
        f,c=synthetic();w=joints_from_features(torch.from_numpy(f)).numpy()
        for invalid in (w[:1],np.zeros((32,17,2)),w*np.nan):
            with self.assertRaises(ValueError):lm.compose_world_joints(invalid,c,len(f)/30)
        with self.assertRaises(ValueError):lm.compose(f,c,len(f)/30,world_joints_override=w[:-1])
        with self.assertRaises(ValueError):lm.compose(f,c,len(f)/30,world_joints_override=w,independent_features=np.stack([f]*6))
        with self.assertRaises(ValueError):lm.compose_world_joints(np.zeros_like(w),c,len(f)/30)

    def test_slice_preserves_accumulated_root_exactly(self):
        f,_=synthetic(100);world=joints_from_features(torch.from_numpy(f)).numpy()
        clip=v.full_slice(world,60,30)
        np.testing.assert_array_equal(clip,world[60:90]);self.assertNotEqual(float(clip[0,0,0]),0.)
        self.assertFalse(np.array_equal(clip,joints_from_features(torch.from_numpy(f[60:90])).numpy()))

    def test_bad_crop_rejected(self):
        w=np.zeros((10,22,3))
        for start,n in ((-1,2),(9,2),(0,0),(0,11)):
            with self.assertRaises(ValueError):v.full_slice(w,start,n)

class ReviewTests(unittest.TestCase):
    def test_receipt_integrity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.json';v.savejson(p,{'x':1});self.assertEqual(v.loadjson(p),{'x':1})
            with self.assertRaises(FileExistsError):v.savejson(p,{'x':1})
            e=json.loads(p.read_text());e['payload']['x']=2;p.chmod(0o644);p.write_text(json.dumps(e))
            with self.assertRaises(ValueError):v.loadjson(p)

    def test_npz_create_only_and_finite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.npz';v.savenpz(p,x=np.arange(3));np.testing.assert_array_equal(v.arrays(p)['x'],np.arange(3))
            with self.assertRaises(FileExistsError):v.savenpz(p,x=np.arange(3))
            with self.assertRaises(ValueError):v.savenpz(Path(d)/'nan.npz',x=np.array([np.nan]))

    def test_dmx_indices_global_not_clip_local(self):
        idx=v.dmx_indices(1200,600,10000)
        self.assertEqual(idx[0],1600);self.assertEqual(idx[-1],2399)
        with self.assertRaises(ValueError):v.dmx_indices(1200,600,2000)

    def test_sampling_retains_exact_tail(self):
        self.assertEqual(v.sample_indices(61),[0,30,60]);self.assertEqual(v.sample_indices(60),[0,30,59])
        self.assertEqual(v.sample_indices(8319)[-1],8318)

    def test_optics_changes_only_head_zoom(self):
        from lightsem.conductor.rig import canonical_rig
        rig=canonical_rig();x=np.random.default_rng(3).integers(0,256,(5,512),dtype=np.uint8)
        y=v.optical_dmx(x,rig,{'zoom':'zero','frozen_angles':False});allow=np.zeros(512,bool)
        for f in rig.by_kind('moving_head'):allow[f.address-1+9]=True
        np.testing.assert_array_equal(y[:,~allow],x[:,~allow]);self.assertTrue((y[:,allow]==0).all())
        np.testing.assert_array_equal(v.optical_dmx(x,rig,{'zoom':'native','frozen_angles':False}),x)

    def test_labels_not_inventing_reference(self):
        self.assertIn('NO source',v.labels('shpongle')['reference'])
        self.assertIn('2D',v.labels('083')['old_incumbent'])
        self.assertIn('7001',v.labels('083')['new_physical'])

    def test_world_shapes(self):
        v.validate_worlds({'reference':np.zeros((5,22,3))},5)
        with self.assertRaises(ValueError):v.validate_worlds({'reference':np.zeros((4,22,3))},5)

    def test_evaluator_hash_identity_and_draw(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'000.json';npz=p.with_suffix('.npz');v.savenpz(npz,joints=np.zeros((2,22,3)))
            q={'protocol_sha256':'p','source_sha256':'s','evaluator_sha256':'e'}
            identity={'arm':'physical','checkpoint':'final','stage':'full','seed':0,'steps':32,'draws':6,
                'precision':'float32','tf32':False,'evaluation_sha256':v.jhash(q),**q}
            spec={'key':'083','draw_seeds':[7001]}
            row={'identity':identity,'spec':spec,'finite_all_draws':True,'draw_seed':7001,
                'arrays_path':str(npz),'arrays_sha256':v.digest(npz)};v.savejson(p,row)
            self.assertEqual(v.verify_evaluator_row(p,q,'physical','full',spec),row)
            with self.assertRaises(ValueError):v.verify_evaluator_row(p,q,'masked_native','full',spec)
            with self.assertRaises(ValueError):v.verify_evaluator_row(p,q,'physical','full',{'key':'083','draw_seeds':[111730]})
            npz.chmod(0o644);npz.write_bytes(b'bad')
            with self.assertRaises(ValueError):v.verify_evaluator_row(p,q,'physical','full',spec)

    def test_no_result_model_import_in_visual_script(self):
        text=(ROOT/'scripts/motion/review_music_physical_revision.py').read_text()
        self.assertNotIn('torch.load(',text);self.assertNotIn('.generate(',text);self.assertNotIn('cuda()',text)

    def test_review_requires_next_missing_and_all_rows(self):
        # No file/video access: test attestation rejection with mocked authenticated renderer.
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);i={'id':'first','frames':60};obs=out/'obs.json'
            o={'item_id':'second','reviewer':'CPU test','notes':'n','coverage':'all_frozen_samples',
               'viewed_sample_frames':[0,30,59],'comparisons':{k:'n' for k in v.ROWS}}
            obs.write_text(json.dumps(o))
            with patch.object(v,'verify_render',return_value=(out,{})):
                with self.assertRaises(ValueError):v.review(out,{'items':[i]},obs)
                o['item_id']='first';o['comparisons'].pop('body');obs.write_text(json.dumps(o))
                with self.assertRaises(ValueError):v.review(out,{'items':[i]},obs)

if __name__=='__main__':unittest.main()
