import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from lightsem.experiments.training_runtime import TrainingRuntime,BatchStream,backward_checked,runtime_loss
from lightsem.experiments.night.night_model import NightModel
from lightsem.experiments.night.night_training import NightResident,run_cell
from lightsem.experiments.stage11.stage11_common import seed_all,read_authenticated
from lightsem.model.light_trunk import CausalWindowAttention


class MaskTests(unittest.TestCase):
    def test_mask_boundaries_cache_and_legacy_state(self):
        for offline in (False,True):
            model=CausalWindowAttention(16,2,8,0.,offline)
            mask=model.window_mask(17,torch.device('cpu'))
            for q in range(17):
                for k in range(17):
                    expected=abs(q-k)<4 if offline else 0<=q-k<8
                    self.assertEqual(bool(mask[0,0,q,k]),expected)
            self.assertIs(mask,model.window_mask(17,torch.device('cpu')))
            self.assertNotIn('_window_mask',model.state_dict())
            self.assertEqual(model.window_mask(9,torch.device('cpu')).shape[-1],9)
            clone=CausalWindowAttention(16,2,8,0.,offline);clone.load_state_dict(model.state_dict(),strict=True)

    def test_invalid_runtime_is_rejected(self):
        for kwargs in ({'precision':'fp8'},{'prefetch':1},{'optimizer':'adam8bit'},{'compile':'anything'}):
            with self.assertRaises(ValueError):TrainingRuntime(**kwargs)


class TinyData:
    draw_indices=NightResident.draw_indices
    batch=NightResident.batch
    def __init__(self):
        self.groups=[[[0,1]],[[2]]];self.lengths=np.array([90,80,100]);self.starts=np.array([0,90,170])
        values=np.random.default_rng(1).normal(size=(270,116)).astype(np.float32)
        self.x=torch.from_numpy(values[:,:92].copy()).cuda();self.y=torch.from_numpy(values[:,92:113].copy()).cuda()
        self.b=torch.from_numpy((values[:,113:]>0).astype(np.float32)).cuda()
        self.host_aux=np.random.default_rng(2).normal(size=(270,4)).astype(np.float16)
        self.aux_mean=torch.zeros(4,device='cuda');self.aux_std=torch.ones(4,device='cuda');self.aux_dim=4
    def set_family(self,family):
        self.host_aux=None;self.aux_dim=0


@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class RuntimeTests(unittest.TestCase):
    def test_seed_override_and_finite_bf16(self):
        seed_all(0);r=TrainingRuntime.legacy('tf32');r.apply()
        self.assertTrue(torch.backends.cudnn.allow_tf32)
        self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
        seed_all(0);r=TrainingRuntime.legacy('bf16');r.apply()
        m=NightModel().cuda();x=torch.randn(2,32,92,device='cuda');y=torch.randn(2,32,21,device='cuda')
        loss=runtime_loss(m,r.loss_function(m),r,x,y,torch.zeros(2,32,3,device='cuda'),torch.ones(3,device='cuda'),torch.Generator(device='cuda').manual_seed(1))
        backward_checked(loss,m)
        self.assertTrue(all(p.dtype==torch.float32 for p in m.parameters()))
        self.assertEqual(loss['total'].dtype,torch.float32)

    def test_legacy_loss_and_rng_unchanged(self):
        seed_all(23);m=NightModel().cuda().train()
        x=torch.randn(2,32,92,device='cuda');y=torch.randn(2,32,21,device='cuda');b=torch.zeros(2,32,3,device='cuda');pos=torch.ones(3,device='cuda')
        state=torch.cuda.get_rng_state();g=torch.Generator(device='cuda').manual_seed(7)
        keep=torch.rand((len(x),1,1),device='cuda',generator=g)>=.25
        xx=torch.cat((x[:,:,:87],x[:,:,87:92]*keep,x[:,:,92:]),dim=2)
        h=m.encode_audio(xx);a=F.mse_loss(m.a_head(h),y[:,:,:4]);events=F.binary_cross_entropy_with_logits(m.b_head(h),b,pos_weight=pos)
        z=torch.randn(y[:,:,4:].shape,device='cuda',generator=g);t=torch.rand((len(x),1,1),device='cuda',generator=g)
        c=F.mse_loss(m.velocity_from_h(h,(1-t)*z+t*y[:,:,4:],t),y[:,:,4:]-z)
        final_global=torch.cuda.get_rng_state();final_flow=g.get_state()
        torch.cuda.set_rng_state(state);g.manual_seed(7)
        loss=m.losses(x,y,b,pos,generator=g)
        self.assertTrue(torch.equal(loss['total'],a+.3*events+c))
        self.assertTrue(torch.equal(final_global,torch.cuda.get_rng_state()))
        self.assertTrue(torch.equal(final_flow,g.get_state()))

    def test_prefetch_matches_sync_and_resume_cursor(self):
        data=TinyData();r1=np.random.default_rng(9);r2=np.random.default_rng(9)
        with BatchStream(data,32,32,r2,2) as stream:
            for _ in range(3):
                expected=data.batch(32,32,r1);actual=stream.next()
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))
                self.assertEqual(r1.bit_generator.state,r2.bit_generator.state)
        cursor=deepcopy(r2.bit_generator.state)
        resumed=np.random.default_rng();resumed.bit_generator.state=cursor
        with BatchStream(data,32,32,resumed,2) as stream:
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(data.batch(32,32,r1),stream.next())))

    def test_batch64_preserves_two_legacy_crop_blocks(self):
        data=TinyData();r1=np.random.default_rng(0);r2=np.random.default_rng(0)
        a=np.concatenate([data.draw_indices(32,32,r1),data.draw_indices(32,32,r1)])
        self.assertTrue(np.array_equal(a,data.draw_indices(64,32,r2)))

    def test_nonfinite_loss_never_updates_optimizer(self):
        model=torch.nn.Linear(2,1).cuda();before=deepcopy(model.state_dict())
        optimizer=torch.optim.AdamW(model.parameters())
        total=model(torch.ones(1,2,device='cuda')).sum()
        # Auxiliary NaN with finite total/grad must still fail before the update.
        with self.assertRaises(FloatingPointError):
            backward_checked({'total':total,'other':total.detach()*float('nan')},model)
            optimizer.step()
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_actual_trainer_resume_and_runtime_identity(self):
        cfg={'steps':4,'warmup':2,'batch':2,'frames':32,'lr':3e-4,'beat_dropout':.25,'checkpoint_interval':2,
             'runtime':TrainingRuntime(precision='bf16',optimizer='adamw_fused',prefetch=2,
                                       compile='default' if os.environ.get('LIGHTSEM_TEST_COMPILE') else 'off').as_dict()}
        def context(root):return {'training':cfg}, {}, {'pos_weight':[1.,1.,1.]}, {'protocol_sha256':'fixed','source_sha256':'fixed'}
        with tempfile.TemporaryDirectory() as temp, patch('lightsem.experiments.night.night_training.study_context',side_effect=context):
            root=Path(temp);full=root/'full';partial=root/'partial'
            run_cell('base_once',0,TinyData(),full)
            original=BatchStream.next;count=0
            def interrupt(stream):
                nonlocal count
                count+=1
                if count==3:raise RuntimeError('external interruption')
                return original(stream)
            with patch.object(BatchStream,'next',interrupt),self.assertRaisesRegex(RuntimeError,'external interruption'):
                run_cell('base_once',0,TinyData(),partial)
            cfg['runtime']['precision']='tf32'
            with self.assertRaisesRegex(RuntimeError,'Resume identity differs'):
                run_cell('base_once',0,TinyData(),partial,resume=True)
            cfg['runtime']['precision']='bf16'
            run_cell('base_once',0,TinyData(),partial,resume=True)
            a=torch.load(full/'runs/base_once/seed_0/final.pt',weights_only=False)
            b=torch.load(partial/'runs/base_once/seed_0/final.pt',weights_only=False)
            self.assertTrue(all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()))
            self.assertTrue(torch.equal(a['flow_rng'],b['flow_rng']))
            self.assertEqual(a['rng']['numpy_batch'],b['rng']['numpy_batch'])

    def test_inference_reuse_and_bf16_numpy_boundary(self):
        from lightsem.experiments.night.night_evaluation import generate_groups
        seed_all(5);m=NightModel().cuda().eval();x=torch.randn(1,1600,92,device='cuda')
        with torch.no_grad():
            h=m.encode_audio_chunked(x,640)
            self.assertLess(float((h-m.encode_audio(x)).abs().max()),2e-5)
        a,b=generate_groups(m,x,seeds=(101,102),steps=2)
        c,d=generate_groups(m,x,seeds=(101,102),steps=2,encoded=h)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
        a,b=generate_groups(m,x,seeds=(101,102),steps=2,precision='bf16')
        self.assertEqual(a.dtype,np.float32);self.assertEqual(b.dtype,np.float32)
        self.assertTrue(np.isfinite(a).all() and np.isfinite(b).all())


if __name__=='__main__':unittest.main()
