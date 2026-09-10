"""Discarded real-TRAIN timing runs before music-protocol freeze."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audio_prepare import ROOT
from .music_model import MusicModel,MusicConfig
from .music_data import MusicResident
from ..training_runtime import TrainingRuntime,BatchStream,backward_checked
from ..stage11_common import read_authenticated,authenticated_json,digest,seed_all,gpu_lock


def main():
    path = ROOT/'music/engineering.json'
    if path.exists():
        return read_authenticated(path)
    if not read_authenticated(ROOT/'codec/final_acceptance.json')['audio_training_authorized']:
        raise ValueError('Representation has not qualified')
    data = MusicResident(read_authenticated(ROOT/'data/audio_manifest.json'),read_authenticated(ROOT/'data/audio_normalizers.json'))
    norms = read_authenticated(ROOT/'data/motion_normalizers.json')
    runtime = TrainingRuntime()
    rows = []
    for arm in ('regression','flow'):
        for batch in (16,32):
            seed_all(0)
            runtime.apply()
            model = MusicModel(MusicConfig(arm=arm),norms).cuda()
            optimizer = runtime.make_optimizer(model,3e-4)
            fn = runtime.loss_function(model)
            rng = np.random.default_rng(10000)
            times = []
            torch.cuda.reset_peak_memory_stats()
            with BatchStream(data,batch,960,rng,runtime.prefetch) as batches:
                for step in range(32):
                    torch.cuda.synchronize()
                    begin = time.monotonic()
                    x,y,b = batches.next()
                    noise,t = model.draws(x)
                    optimizer.zero_grad(set_to_none=True)
                    with runtime.autocast():
                        losses = fn(x,y,b,noise,t)
                    backward_checked(losses,model)
                    optimizer.step()
                    torch.cuda.synchronize()
                    times.append(time.monotonic()-begin)
                    if step==0:
                        print(json.dumps({'arm':arm,'batch':batch,'compile_seconds':times[-1]}),flush=True)
            model.eval()
            sample = data.batch(1,2300,np.random.default_rng(11000))[0]
            with torch.inference_mode():
                # Full ODE, nonzero fitted engineering weights and internal seams.
                start = time.monotonic()
                full = model.generate(sample,seed=17,chunked=False)
                chunk = model.generate(sample,seed=17,chunked=True)
                torch.cuda.synchronize()
                infer = time.monotonic()-start
                error = float((full-chunk).abs().max())
            if error>5e-5:
                raise ValueError(f'Global integration differs at chunks: {arm}: {error}')
            row = {'arm':arm,'batch':batch,'update_seconds_median':float(np.median(times[8:])),
                   'compile_seconds':times[0],'engineering_updates':32,
                   'parameters':sum(p.numel() for p in model.parameters()),'chunk_max_abs':error,
                   'full_and_chunk_76s_generation_seconds':infer,'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                   'runtime':runtime.as_dict(),'scientific_weights_saved':False}
            rows.append(row)
            print(json.dumps(row),flush=True)
            del model,optimizer,fn,x,y,b,noise,t,losses,full,chunk,sample
            gc.collect()
            torch.cuda.empty_cache()
    scores = {batch:sum(r['update_seconds_median']/batch for r in rows if r['batch']==batch) for batch in (16,32)}
    result = {'rows':rows,'selected_batch':min(scores,key=lambda b:(scores[b],b)),
        'selection':'lowest summed seconds per crop across both arms, no VAL outcomes',
        'model_source_sha256':digest(Path(__file__).with_name('music_model.py')),
        'data_source_sha256':digest(Path(__file__).with_name('music_data.py')),
        'test_source_sha256':digest('/root/lightsem/tests/test_motion_study.py'),
        'source_sha256':digest(__file__),'gpu':torch.cuda.get_device_name()}
    authenticated_json(path,result)
    return result


if __name__=='__main__':
    with gpu_lock(ROOT):
        main()
