"""Recovery and material-identity tests for the actual campaign trainer."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from lightsem.experiments import stage11_training as tr
from lightsem.experiments.stage11_common import atomic_json,authenticated_json,digest,read_authenticated
from lightsem.experiments.stage11_models import SceneFlowModel,SceneFlowConfig


class TrainingIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        cfg={'family':'scene','steps':5,'batch':2,'frames':8,'lr':.0003,'resident_gpu':False,'log_every':5,'checkpoint_every_steps':2}
        atomic_json(self.root/'protocol.json',{'training_seeds':[0],'arms':{'S_temporal':cfg}})
        entries=[]
        for i in range(2):
            path=self.root/f'{i}.npz';rng=np.random.default_rng(i)
            np.savez(path,x=rng.normal(size=(100,52)).astype('f'),y=rng.normal(size=(100,6)).astype('f'))
            entries.append({'key':str(i),'canonical_id':str(i),'recording_id':str(i),'partition':'train',
                            'is_old':True,'path':str(path),'n_frames':100,'sha256':digest(path)})
        norm={'x_mean':[0.]*52,'x_std':[1.]*52,'y_center':[0.]*6,'y_scale':[1.]*6}
        authenticated_json(self.root/'normalizers.json',{'scene':norm,'scene_metrics':{}})
        authenticated_json(self.root/'training_data.json',{'scene':entries,'normalizers_sha256':digest(self.root/'normalizers.json')})
        self.small=SceneFlowConfig(in_dim=52,scene_dim=6,d_model=16,n_blocks=1,n_heads=4,
            ff_mult=2,conv_kernel=3,attn_window=8,flow_dilations=(1,2))
        self.rows=[{'canonical_id':str(i),'metrics':{'temporal_error':.4}} for i in range(2)]

    def tearDown(self):self.tmp.cleanup()

    def run_small(self,**kw):
        with patch.object(tr,'_model',side_effect=lambda *args:SceneFlowModel(config=self.small)), \
             patch.object(torch.cuda,'is_available',return_value=False):
            return tr.train_run(self.root,'S_temporal',0,engineering_steps=5,engineering_device='cpu',**kw)

    def test_pending_final_validation_is_completed_after_interruption(self):
        with patch.object(tr,'evaluate_scene',side_effect=RuntimeError('external interruption')):
            with self.assertRaisesRegex(RuntimeError,'external interruption'):self.run_small()
        directory=self.root/'engineering_runs/S_temporal/seed_0'
        interrupted=torch.load(directory/'latest.pt',weights_only=False)
        self.assertEqual(interrupted['pending_validation'],5)
        self.assertFalse((directory/'best.pt').exists())
        with patch.object(tr,'evaluate_scene',return_value=self.rows) as evaluate:
            report=self.run_small(resume=True)
        self.assertEqual(evaluate.call_count,1)
        self.assertEqual(report['steps'],5)
        final=torch.load(directory/'latest.pt',weights_only=False)
        self.assertIsNone(final['pending_validation'])
        for key,value in interrupted['model'].items():
            torch.testing.assert_close(final['model'][key],value,rtol=0,atol=0)
        self.assertEqual(read_authenticated(directory/'report.json')['status'],'completed')

    def test_best_publication_interruption_reuses_completed_capture(self):
        real_write=tr.atomic_torch
        def fail_after_best(path,blob):
            real_write(path,blob)
            if Path(path).name=='best.pt':raise RuntimeError('crashed after publishing best')
        with patch.object(tr,'evaluate_scene',return_value=self.rows),patch.object(tr,'atomic_torch',side_effect=fail_after_best):
            with self.assertRaisesRegex(RuntimeError,'publishing best'):self.run_small()
        with patch.object(tr,'evaluate_scene',side_effect=AssertionError('must not re-evaluate')):
            report=self.run_small(resume=True)
        self.assertEqual(report['best_step'],5)
        self.assertEqual(report['best_validation_score'],.4)

    def test_scientific_cpu_override_is_rejected(self):
        # The guard is intentionally tested before normal scientific inputs
        # would be loaded through the real integrity boundary.
        with patch.object(tr,'verify_training_inputs',return_value=read_authenticated(self.root/'training_data.json')), \
             patch.object(tr,'verify_sources',return_value={}):
            with self.assertRaisesRegex(ValueError,'cannot enter scientific'):
                tr.train_run(self.root,'S_temporal',0,engineering_device='cpu')

    def test_resumed_optimizer_and_rng_reproduce_uninterrupted_weights(self):
        with patch.object(tr,'evaluate_scene',return_value=self.rows):
            self.run_small()
        directory=self.root/'engineering_runs/S_temporal/seed_0'
        expected=torch.load(directory/'latest.pt',weights_only=False)
        import shutil
        shutil.move(str(directory),str(directory.with_name('uninterrupted_reference')))
        real_batch=tr.ResidentDataset.batch;calls=0
        def interrupted_batch(instance,*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==3:raise RuntimeError('external stop after checkpoint')
            return real_batch(instance,*args,**kwargs)
        with patch.object(tr.ResidentDataset,'batch',new=interrupted_batch):
            with self.assertRaisesRegex(RuntimeError,'external stop'):self.run_small()
        paused=torch.load(directory/'latest.pt',weights_only=False)
        self.assertEqual(paused['step'],2)
        with patch.object(tr,'evaluate_scene',return_value=self.rows):self.run_small(resume=True)
        actual=torch.load(directory/'latest.pt',weights_only=False)
        self.assertEqual(actual['scheduler'],expected['scheduler'])
        for key,value in expected['model'].items():
            torch.testing.assert_close(actual['model'][key],value,rtol=0,atol=0)
        self.assertEqual(actual['rng']['numpy_batch'],expected['rng']['numpy_batch'])
        torch.testing.assert_close(actual['flow_rng'],expected['flow_rng'],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
