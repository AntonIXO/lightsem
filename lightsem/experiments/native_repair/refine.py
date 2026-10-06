"""Matched short continuations of a completed repair checkpoint.

Same parent, data, draws and fresh optimizer per seed. Only the declared raw HF weight
changes. No output filtering, architecture change or seed choice.
"""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from ..native_dynamics import training as original
from ..native_dynamics.data import read,sha
from .training import batch_losses
from .losses import WEIGHTS

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--recipe',choices=['continued','strong'],required=True);p.add_argument('--guarded',action='store_true');a=p.parse_args()
    from scripts.motion.native_dynamics_guard import authenticate_guarded_job
    authenticate_guarded_job(Path(__file__).parents[3]/'scripts/motion/refine_native_repair.py')
    protocol=read(a.root/'TRAINING_PROTOCOL.json')
    if protocol['recipe']!=a.recipe or not 1<=protocol['training']['steps']<=30000:raise ValueError('Wrong frozen refinement')
    if a.parent.resolve()!=Path(protocol['parent_root']).resolve():raise ValueError('Wrong parent')
    if set(WEIGHTS)!=set(protocol['repair_weights']):raise ValueError('Repair weight schema drift')
    WEIGHTS.update(protocol['repair_weights'])
    original_load=original.load_initial
    def load_parent(arm,seed,device='cuda'):
        model,teacher,state,norms,teacher_identity=original_load(arm,seed,device)
        path=a.parent/'learning/runs/joint'/f'seed_{seed}'/'final.pt';report=read(path.parent/'REPORT.json')
        if report['steps']!=protocol.get('parent_phase_steps',30000) or report['artifacts'][str(path)]!=sha(path):raise ValueError('Unqualified parent identity')
        parent=torch.load(path,map_location='cpu',weights_only=False)
        if original.asdict(model.config)!=parent['config']:raise ValueError('Architecture change forbidden')
        for name in ('audio_normalizers','extra_normalizers'):
            for key in ('mean','std'):
                if not np.array_equal(state[name][key],parent[name][key]):raise ValueError('Normalizer drift')
        model.load_state_dict(parent['model'],strict=True)
        return model,teacher,state,norms,{'checkpoint':str(path),'sha256':sha(path),'parent_updates':protocol.get('parent_total_updates',30000),'teacher_D7':teacher_identity,'continuation':'Fresh matched optimizer/RNG; matched fresh sample order forbothrecipes. Not pretending an exactoptimizerresume.'}
    original.load_initial=load_parent;original.batch_losses=batch_losses
    report=original.train(a.root,'joint',a.seed)
    print(json.dumps({'status':report['status'],'recipe':a.recipe,'seed':a.seed,'phase_steps':report['steps'],'total_updates':protocol.get('parent_total_updates',30000)+report['steps']}))
if __name__=='__main__':main()
