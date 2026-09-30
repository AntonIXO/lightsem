"""Convert the pinned official SavedModel and verify on seeded synthetic inputs.

CPU engineering check only. No enrolled source data or scientific outcomes.
"""
from pathlib import Path
import json
import os
import sys
import time

ROOT=Path('/root/lightsem/runs/visual_next_20260929')
RECORDS=Path('/root/lightsem/experiments/2026-09-29-visual-next/records/execution')
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
sys.path.insert(0,str(ROOT/'conversion_deps'))
sys.path.insert(0,str(ROOT/'vendor/transnetv2/inference-pytorch'))

import numpy as np
import tensorflow as tf
import torch
from convert_weights import convert_weights
from lightsem.model.light_layers.visual import sha256, write_json


def main():
    torch.set_num_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    acquisition=json.loads((RECORDS/'MODEL_ACQUISITION.json').read_text())
    row=acquisition['models']['transnetv2']
    for file in row['tensorflow_weights']:
        assert sha256(file['path'])==file['sha256']
    for rel,digest in row['source_files'].items():
        assert sha256(ROOT/'vendor/transnetv2'/rel)==digest
    started=time.monotonic()
    model,tf_model=convert_weights(str(ROOT/'weights/transnetv2-tensorflow'))
    model.eval()
    rng=np.random.default_rng(20260929)
    cases=[np.zeros((1,100,27,48,3),np.uint8),rng.integers(0,256,(1,100,27,48,3),dtype=np.uint8)]
    deltas=[]
    for frames in cases:
        with torch.inference_mode():a,b=model(torch.from_numpy(frames))
        x,y=tf_model(tf.cast(frames,tf.float32))
        for left,right in ((a.numpy(),x.numpy()),(b['many_hot'].numpy(),y['many_hot'].numpy())):
            error=float(np.max(np.abs(left-right)));deltas.append(error)
            # Conversion arithmetic across frameworks; not a scientific threshold.
            assert np.allclose(left,right,rtol=1e-4,atol=1e-4),error
    dest=ROOT/'weights/transnetv2-pytorch-official.pth'
    with dest.open('xb') as f:torch.save(model.state_dict(),f)
    write_json(RECORDS/'TRANSNET_CONVERSION.json',{'status':'converted_and_synthetic_equivalence_verified',
        'tensorflow':tf.__version__,'torch':torch.__version__,'max_logit_abs_error_by_output':deltas,
        'cases':['all_black_100_frames','seeded_uniform_uint8_100_frames'],
        'elapsed_s':time.monotonic()-started,'weights':{'path':str(dest),'sha256':sha256(dest)},
        'converter_sha256':sha256(ROOT/'vendor/transnetv2/inference-pytorch/convert_weights.py'),
        'source_data_read':False,'device':'cpu','rtol':1e-4,'atol':1e-4})
    print('CONVERSION_VERIFIED',deltas,flush=True)


if __name__=='__main__':main()
