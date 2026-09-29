"""Synthetic CPU-only IO/geometry/stage tests. Never opens saved bodies."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pose_eval',ROOT/'scripts/motion/eval_pose_stage_study.py')
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)

class IOTests(unittest.TestCase):
    def test_receipt_hash_create_only_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'receipt.json';e.savejson(p,{'one':1});self.assertEqual(e.loadjson(p),{'one':1})
            with self.assertRaises(FileExistsError):e.savejson(p,{'one':2})
            p.chmod(0o644);p.write_text('{"payload":{"one":2},"sha256":"wrong"}')
            with self.assertRaises(ValueError):e.loadjson(p)
    def test_wrong_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x';p.write_bytes(b'original');h=e.digest(p);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):e.checked(p,h)
    def test_registration_required_and_exact(self):
        plan={'hypothesis_id':'h','experiment_id':'e'}
        with patch.object(e,'SUPPORTED_PLAN_HASH',e.jhash(plan)):
            with self.assertRaises(ValueError):e.validate_registration(plan,{})
            reg={'registered_by':'parent','registered_before_candidate_execution':True,'hypothesis':{'id':'h','plan':plan},'experiment':{'id':'e','config_note':plan}}
            e.validate_registration(plan,reg)
            reg['experiment']['config_note']={'changed':True}
            with self.assertRaises(ValueError):e.validate_registration(plan,reg)
    def test_incomplete_cache_and_wrong_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);item={'id':'synthetic'};ctx={'freeze_sha256':'f'}
            (out/'mapped/synthetic').mkdir(parents=True)
            with self.assertRaises(FileNotFoundError):e.mapped(out,ctx,item)
            e.savejson(out/'mapped/synthetic/map.json',{'item':item,'freeze_sha256':'wrong'})
            with self.assertRaises(ValueError):e.mapped(out,ctx,item)
    def test_full_coverage(self):
        w=np.ones((60,22,3));d=np.zeros((80,512),np.uint8);t=np.arange(60)/30;dt=np.arange(80)/40
        e.validate_arrays(w,d,t,dt,60)
        for z in [(w[:-1],d,t,dt),(w,d[:-1],t,dt),(w,d,t+.1,dt),(w,d.astype(float),t,dt)]:
            with self.assertRaises(ValueError):e.validate_arrays(*z,60)
        w[0,0,0]=np.nan
        with self.assertRaises(ValueError):e.validate_arrays(w,d,t,dt,60)
    def test_parity_never_repaired(self):
        a=np.zeros((10,512),np.uint8);e.verify_parity(a,a.copy());b=a.copy();b[2,30]=1
        with self.assertRaises(ValueError):e.verify_parity(a,b)
        self.assertEqual(int(b[2,30]),1)
    def test_missing_freeze_before_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):e.context(Path(tmp),activate=True)
    def test_storage_reserve_total_cap_and_bookkeeping(self):
        state={'filesystem_free_bytes':e.FREE_RESERVE_BYTES+e.BOOKKEEPING_MARGIN_BYTES+100,
            'study_artifact_bytes':0,'remaining_study_budget_bytes':e.STUDY_LIMIT_BYTES}
        with patch.object(e,'storage_state',return_value=state):
            e.check_storage(Path('/synthetic'),100)
            with self.assertRaises(e.ResourceBlocked):e.check_storage(Path('/synthetic'),101)
            e.check_storage(Path('/synthetic'),101,bookkeeping=True)
        state['filesystem_free_bytes']=10*e.FREE_RESERVE_BYTES
        state['study_artifact_bytes']=e.STUDY_LIMIT_BYTES-e.BOOKKEEPING_MARGIN_BYTES-100
        with patch.object(e,'storage_state',return_value=state):
            e.check_storage(Path('/synthetic'),100)
            with self.assertRaises(e.ResourceBlocked):e.check_storage(Path('/synthetic'),101)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            with e.lock(out),patch.object(e,'storage_state',return_value=state):
                with self.assertRaises(e.ResourceBlocked):e.writebytes(out/'blocked.bin',b'x'*101)
                self.assertFalse((out/'blocked.bin').exists())
                # Failure bookkeeping may consume the reserved operating margin,
                # never the filesystem free reserve or total-artifact hard cap.
                e.savejson(out/'failed.json',{'small':'receipt'})
    def test_resource_preflight_retains_blocked_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            state={'filesystem_free_bytes':e.FREE_RESERVE_BYTES+1024**2,'study_artifact_bytes':0,'remaining_study_budget_bytes':e.STUDY_LIMIT_BYTES}
            with patch.object(e,'storage_state',return_value=state):
                with self.assertRaises(e.ResourceBlocked):e.resource_preflight(out,'native',2*1024**2)
            r=e.loadjson(out/'resource_preflight/native_000.json')
            self.assertEqual(r['status'],'resource_blocked');self.assertFalse(r['estimate_is_hard_bound'])
    def test_lock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            with e.lock(Path(tmp)):
                with self.assertRaises(BlockingIOError):
                    with e.lock(Path(tmp)):pass

class NamespaceTests(unittest.TestCase):
    def test_fresh_frozen_namespace_and_synthetic_full_map(self):
        import os,shutil,subprocess
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);files={}
            for p in (ROOT/'lightsem').rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json'):
                    dest=out/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
                    files['source/'+str(p.relative_to(ROOT))]=e.digest(dest)
            for name,filename in [('review','review_music_physical_support_revision.py'),('optics','run_dance_optics_full.py')]:
                dest=out/'frozen/source_helpers'/f'{name}.py';dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'scripts/motion'/filename,dest)
                files[f'source_helpers/{name}.py']=e.digest(dest)
            e.savejson(out/'freeze.json',{'files':files})
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
            result=subprocess.run(['bash',str(ROOT/'run_torch.sh'),str(ROOT/'tests/pose_stage_namespace_smoke.py'),str(out)],capture_output=True,text=True,env=env,timeout=45)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('SYNTHETIC_FULL_MAP_AND_NAMESPACE_OK',result.stdout)
            p=out/'source/lightsem/experiments/motion_study/pose_stage.py';p.write_text(p.read_text()+'\n# synthetic source corruption\n')
            bad=subprocess.run(['bash',str(ROOT/'run_torch.sh'),str(ROOT/'tests/pose_stage_namespace_smoke.py'),str(out),'namespace_only'],capture_output=True,text=True,env=env,timeout=45)
            self.assertNotEqual(bad.returncode,0);self.assertIn('Hash changed:',bad.stderr)


class GeometryTests(unittest.TestCase):
    def test_broadband_is_new_not_selected_subband(self):
        t=np.arange(800)/40;x=np.zeros((800,6,2));x[:,:,0]=np.sin(2*np.pi*.7*t)[:,None]+np.sin(2*np.pi*3*t)[:,None]
        self.assertGreater(e.band_rms(x,(.5,4)).mean(),.9)
        self.assertLess(e.band_rms(x,(1,2)).mean(),.02)
        with self.assertRaises(ValueError):e.band_rms(x[:399],(.5,4))
        x[0,0,0]=np.nan
        with self.assertRaises(ValueError):e.band_rms(x,(.5,4))
    def test_exact_edge_band_power_partition(self):
        t=np.arange(800)/40;x=np.zeros((800,6,2))
        x[:,:,0]=(np.sin(2*np.pi*.7*t)+np.sin(2*np.pi*1.0*t)+np.sin(2*np.pi*2*t)+np.sin(2*np.pi*4*t))[:,None]
        power=e.band_rms(x,(.5,4))**2
        parts=sum(e.band_rms(x,band)**2 for band in ((.5,1),(1,2),(2,4)))
        np.testing.assert_allclose(power,parts,rtol=2e-15,atol=2e-15)
        off_grid=e.band_rms(x,(.53,3.97))**2
        divided=e.band_rms(x,(.53,1.13))**2+e.band_rms(x,(1.13,3.97))**2
        np.testing.assert_allclose(off_grid,divided,rtol=2e-15,atol=2e-15)
    def test_cloud_not_pair_merge(self):
        xy=np.broadcast_to(np.array([[-20,0],[-20,0],[0,20],[0,20],[20,0],[20,0.]]),(800,6,2)).copy()
        radii=np.ones((800,6));r,ratio,spread=e.cloud(xy,radii)
        self.assertTrue(r['pass']);self.assertEqual(r['fraction_below'],0)
        r,_,_=e.cloud(np.zeros_like(xy),radii);self.assertFalse(r['pass']);self.assertEqual(r['longest_below_s'],20)
    def test_cloud_runs_and_invalid(self):
        xy=np.broadcast_to(np.column_stack((np.arange(6)*20,np.zeros(6))),(800,6,2)).copy();radii=np.ones((800,6))
        xy[:8]=0;r,_,_=e.cloud(xy,radii);self.assertTrue(r['pass']);self.assertEqual(r['fraction_below'],.01)
        xy[8]=0;r,_,_=e.cloud(xy,radii);self.assertFalse(r['pass'])
        xy[0,0,0]=np.nan;r,_,_=e.cloud(xy,radii);self.assertFalse(r['pass']);self.assertEqual(r['invalid_frames'],1)
    def test_collapse_uses_provided_first_hit_radii(self):
        xy=np.broadcast_to(np.column_stack((np.arange(6),np.zeros(6))),(800,6,2));radii=np.full((800,6),2.)
        a,_,_=e.cloud(xy,radii);b,_,_=e.cloud(xy,radii/2)
        self.assertFalse(a['pass']);self.assertTrue(b['pass'])
    def test_hull_zero_without_jitter(self):
        self.assertEqual(e.hull_area([[0,0],[0,0],[1,1],[2,2]]),0)
        self.assertEqual(e.hull_area([[0,0],[2,0],[2,3],[0,3],[1,1]]),6)
    def test_nonangle_and_exact_decode(self):
        heads=[SimpleNamespace(address=1+16*i,pan_range_deg=540,tilt_range_deg=270) for i in range(6)]
        rig=SimpleNamespace(by_kind=lambda k:heads);a=np.zeros((20,512),np.uint8);b=a.copy();b[:,0]=2
        self.assertTrue(e.nonangle_equal(a,b,rig));b[:,9]=1;self.assertFalse(e.nonangle_equal(a,b,rig))
        mech,_,_=e.mechanics(a,rig);self.assertTrue(mech['pass'])
        a[1,0]=255;mech,_,_=e.mechanics(a,rig);self.assertFalse(mech['pass'])
    def test_frozen_optics_geometry_on_synthetic_only(self):
        # Load helper SOURCE bytes only. This test does not load any study arrays.
        spec=importlib.util.spec_from_file_location('optics_synthetic',ROOT/'scripts/motion/run_dance_optics_full.py');o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o)
        cfg=SimpleNamespace(cam_pos=[0,6,11],cam_target=[0,2,-3],camera_scale=.45,height=360,width=640,beam_edge_profile='angular',beam_edge_softness_deg=.6)
        heads=[SimpleNamespace(address=1+16*i,pan_range_deg=540,tilt_range_deg=270,beam_min_deg=2,position=np.array([i-2.5,6.,-1.])) for i in range(6)]
        rig=SimpleNamespace(by_kind=lambda k:heads)
        a=np.zeros((800,512),np.uint8)
        for f in heads:
            ticks=np.array([32768,30000]);i=f.address-1;a[:,i:i+4]=[ticks[0]>>8,ticks[0]&255,ticks[1]>>8,ticks[1]&255]
        report,z=e.geometry(a,rig,cfg,o,(0,20))
        self.assertIn('first_hit_radius_px',z);self.assertIn('midshaft_radius_px',z)
        self.assertEqual(len(report['blocks10s']),2)
        self.assertEqual(set(report['bands']),{'[0.5, 4]','[0.5, 1]','[1, 2]','[2, 4]'})
        self.assertFalse(np.array_equal(z['first_hit_radius_px'],z['midshaft_radius_px']))
        radii=np.broadcast_to(np.arange(1,7,dtype=float),(800,6));rms=np.arange(6,0,-1,dtype=float)
        with patch.object(o,'circle_radius',return_value=radii),patch.object(e,'band_rms',return_value=rms):
            unequal,_=e.geometry(a,rig,cfg,o,(0,20))
        self.assertAlmostEqual(unequal['primary'],float(np.mean(rms/radii[0])))
        self.assertNotAlmostEqual(unequal['primary'],float(rms.mean()/radii.mean()))
        with self.assertRaises(ValueError):e.geometry(a,rig,cfg,o,(10,30))
        for f in heads:
            i=f.address-1;t=54613;a[:,i+2:i+4]=[t>>8,t&255]
        with np.errstate(invalid='ignore',divide='ignore'):
            missing,_=e.geometry(a,rig,cfg,o,(0,20))
        self.assertIsNone(missing['primary']);self.assertFalse(missing['collapse']['pass'])
        self.assertEqual(missing['collapse']['status'],'unmeasurable_coverage_not_collapse')

class StageTests(unittest.TestCase):
    def plan(self):
        return {'items':[{'id':'shpongle_prior_flow0','key':'shpongle','start':0,'frames':8319,'full_frames':8319}]+[{'id':i,'key':str(k),'start':k*30,'frames':600,'full_frames':8000} for k,i in enumerate(['holds_083_1200_reference','travel_027_4200_reference','coordination_057_1200_reference']+[f'other_{j}' for j in range(9)])]}
    def test_original_map_exception_survives_receipt_write_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=ValueError('original synthetic failure')
            with patch.object(e,'_map_item',side_effect=original),patch.object(e,'savejson',side_effect=OSError('disk full')):
                with self.assertRaises(ValueError) as result:e.map_item(Path(tmp),{'freeze_sha256':'f'},{'id':'synthetic'})
            self.assertIs(result.exception,original)
            self.assertIn('disk full',original.__notes__[0])
    def test_native_first_even_if_shpongle_last_in_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);p=self.plan();p['items']=p['items'][1:]+p['items'][:1];ctx={'plan':p,'freeze_sha256':'f'};calls=[]
            def fake_map(out,ctx,item):
                calls.append(item['id']);e.savejson(out/'mapped'/item['id']/'map.json',{'synthetic':True})
                return {'item':item,'pass':True}
            with patch.object(e,'map_item',side_effect=fake_map):e.evaluate_stage(out,ctx,'native')
            self.assertEqual(calls,['shpongle_prior_flow0'])
    def test_quiet_coordinate_speed_cap(self):
        c={'geometry_valid':True,'mechanics':{'max_speed_deg_s':0.,'max_acceleration_deg_s2':0.},'collapse':{'valid_all':True,'fraction_below':0.,'longest_below_s':0.},'full_valid_head_fraction':1.,'primary':3.,'surface_fractions':{'0':.5,'1':.5},'quiet':{'coordinate_speed_median_all_heads_deg_s':16.3488}}
        b=dict(c,primary=1.)
        self.assertTrue(e.map_gates({'candidate':c,'baseline':b},{'key':'shpongle'},True)['quiet'])
        c['quiet']['coordinate_speed_median_all_heads_deg_s']+=.0001
        self.assertFalse(e.map_gates({'candidate':c,'baseline':b},{'key':'shpongle'},True)['quiet'])
    def test_oracle_holds_not_rewarded(self):
        p={'oracle_transfer_selection':{'categories':['travel','coordination'],'minimum_improved':6,'total':8}}
        rows=[{'item':{'category':category},'reports':{'candidate':{'primary':candidate},'baseline':{'primary':1.}}} for category,candidate in [('holds',100.)]*4+[('travel',2.)]*4+[('coordination',2.)]*2+[('coordination',.5)]*2]
        self.assertEqual(e.oracle_wins(rows,p),6)
        rows[4]['reports']['candidate']['primary']=1.
        self.assertEqual(e.oracle_wins(rows,p),5)
        with self.assertRaises(ValueError):e.oracle_wins(rows[:-1],p)
    def test_map_gate_thresholds_not_cached_pass(self):
        c={'geometry_valid':True,'mechanics':{'max_speed_deg_s':91.,'max_acceleration_deg_s2':0.,'pass':True},'collapse':{'valid_all':True,'fraction_below':0.,'longest_below_s':0.,'pass':True},'full_valid_head_fraction':1.}
        result=e.map_gates({'candidate':c,'baseline':c},{'key':'synthetic'},True)
        self.assertFalse(result['mechanics']);self.assertEqual(set(result),{'measurement_coverage','mechanics','valid','nonangle','collapse'})
        with self.assertRaises(ValueError):e.map_gates({'candidate':c,'baseline':{}},{'key':'synthetic'},1)
    def test_failure_capable_numeric_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);plan=self.plan();plan['claim_scope']='synthetic only';ctx={'plan':plan,'freeze_sha256':'f'}
            item=plan['items'][0]
            def partial_failure(out,ctx,item,state):
                state['phase']='candidate_geometry';state['completed_gates']={'nonangle':True,'mechanics':False}
                raise ValueError('synthetic later geometry failure')
            with patch.object(e,'_map_item',side_effect=partial_failure):
                with self.assertRaises(ValueError):e.map_item(out,ctx,item)
            r=e.numerical_report(out,ctx)
            self.assertFalse(r['all_numeric_gates_pass']);self.assertFalse(r['all_maps_complete'])
            self.assertEqual(r['rows'][0]['status'],'failed');self.assertEqual(len(r['unassessed_items']),13)
            self.assertEqual(r['rows'][0]['gates'],{'nonangle':True,'mechanics':False})
            self.assertEqual(r['rows'][0]['failure']['operation']['phase'],'candidate_geometry')
            self.assertEqual(r['rows'][1]['status'],'unrun')
            with self.assertRaises(FileExistsError):e.numerical_report(out,ctx)
    def test_native_fail_schedule_and_no_reference_run(self):
        ctx={'plan':self.plan()}
        with patch.object(e,'stage_receipt',return_value={'pass':False}) as check:
            result=e.render_schedule(Path('/synthetic'),ctx)
            self.assertEqual([x['start'] for x in result],[0,2100,5700]);self.assertTrue(all(x['frames']==600 for x in result));self.assertEqual(check.call_count,1)
            with self.assertRaises(ValueError):e.evaluate_stage(Path('/synthetic'),ctx,'references')
    def test_reference_fail_and_success_schedules(self):
        ctx={'plan':self.plan()}
        with patch.object(e,'stage_receipt',side_effect=[{'pass':True},{'pass':False}]):
            result=e.render_schedule(Path('/synthetic'),ctx);self.assertEqual(len(result),4);self.assertFalse(any(x['matched'] for x in result))
        with patch.object(e,'stage_receipt',return_value={'pass':True}):
            result=e.render_schedule(Path('/synthetic'),ctx);self.assertEqual(len(result),14);self.assertEqual(sum(x['matched'] for x in result),1);self.assertEqual(result[-1]['frames'],8319)

if __name__=='__main__':unittest.main()
