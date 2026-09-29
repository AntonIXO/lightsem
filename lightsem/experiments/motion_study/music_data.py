"""Paired native data and a fixed, lighting-relevant motion scoring space."""
from collections import Counter, defaultdict

import numpy as np
import torch

from . import ROOT as SOURCE
from .audio_prepare import ROOT
from .geometry import joints_from_features
from .training import MotionResident
from ..stage11.stage11_common import read_authenticated, authenticated_json, digest

SUMMARY_NAMES = ('root_speed','root_height','body_speed','upper_speed','lower_speed',
    'left_hand_speed','right_hand_speed','left_foot_speed','right_foot_speed',
    'hand_spread','hand_height','foot_spread','torso_lean','arm_extension',
    'contact_fraction','vertical_speed','upper_energy','lower_energy')


def motion_summary(features,joints=None):
    """18 fixed physical channels, then nonoverlapping half-second means.

    This is the probabilistic energy-score space, not a fitted loss head. It
    measures native geometry and dynamics used by the lighting mapper, without
    access to audio, source identity, or any prediction-dependent normalization.
    """
    joints = joints_from_features(features) if joints is None else joints
    delta = torch.diff(joints,dim=-3)*30
    velocity = torch.cat((torch.zeros_like(joints[...,:1,:,:]),delta),dim=-3)
    speed = torch.linalg.vector_norm(velocity,dim=-1)
    upper = speed[...,12:]
    lower = speed[...,[1,2,4,5,7,8,10,11]]
    spine = joints[...,12,:]-joints[...,0,:]
    value = torch.stack((
        torch.linalg.vector_norm(features[...,:2],dim=-1),features[...,2],speed.mean(-1),
        upper.mean(-1),lower.mean(-1),speed[...,20],speed[...,21],
        speed[...,[7,10]].mean(-1),speed[...,[8,11]].mean(-1),
        torch.linalg.vector_norm(joints[...,20,:]-joints[...,21,:],dim=-1),
        joints[...,[20,21],1].mean(-1),
        torch.linalg.vector_norm(joints[...,10,:]-joints[...,11,:],dim=-1),
        torch.linalg.vector_norm(spine[...,[0,2]],dim=-1)/torch.linalg.vector_norm(spine,dim=-1).clamp_min(1e-6),
        (torch.linalg.vector_norm(joints[...,20,:]-joints[...,16,:],dim=-1)+
         torch.linalg.vector_norm(joints[...,21,:]-joints[...,17,:],dim=-1))/2,
        features[...,135:].clamp(0,1).mean(-1),velocity[...,1].abs().mean(-1),
        upper.square().mean(-1),lower.square().mean(-1)),dim=-1)
    logarithmic = value.new_tensor([1,0,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,1]).bool()
    value = torch.where(logarithmic,torch.log1p(value.clamp_min(0)),value)
    frames = value.shape[-2]//15*15
    return value[...,:frames,:].reshape(*value.shape[:-2],frames//15,15,len(SUMMARY_NAMES)).mean(-2)


def energy_score(draws,target):
    """Unbiased sample energy score, Euclidean RMS across the fixed vector."""
    draws = draws.reshape(draws.shape[0],-1).double()
    target = target.reshape(1,-1).double()
    first = torch.linalg.vector_norm(draws-target,dim=-1).mean()/draws.shape[1]**.5
    if len(draws)==1:
        return float(first)
    distance = torch.cdist(draws,draws)/draws.shape[1]**.5
    return float(first-.5*distance.sum()/(len(draws)*(len(draws)-1)))


def prepare_targets():
    torch.set_num_threads(4)
    path = ROOT/'data/motion_normalizers.json'
    if path.exists():
        return read_authenticated(path)
    manifest = read_authenticated(ROOT/'data/audio_manifest.json')
    base = read_authenticated(SOURCE/'data/normalizers.json')
    train = [e for e in manifest['entries'] if e['partition']=='train']
    counts = Counter(e['canonical_id'] for e in train)
    second = np.zeros(139)
    sm,ss = np.zeros(len(SUMMARY_NAMES)),np.zeros(len(SUMMARY_NAMES))
    for e in train:
        if digest(e['path'])!=e['sha256']:
            raise ValueError('Motion changed')
        with np.load(e['path']) as data:
            raw = data['features']
        weight = 1/(len(counts)*counts[e['canonical_id']])
        delta = np.diff(raw.astype(np.float64),axis=0)/np.asarray(base['std'])
        second += weight*(delta*delta).mean(0)
        with torch.inference_mode():
            summary = motion_summary(torch.from_numpy(raw)).numpy().astype(np.float64)
        sm += weight*summary.mean(0)
        ss += weight*(summary*summary).mean(0)
    result = {**base,'delta_scale':np.sqrt(np.maximum(second,.05**2)).tolist(),
        'summary':{'names':list(SUMMARY_NAMES),'mean':sm.tolist(),
                   'std':np.sqrt(np.maximum(ss-sm*sm,.01**2)).tolist(),
                   'fps':2,'fit':'equal canonical/record/half-second, audio-eligible TRAIN only'},
        'audio_manifest_sha256':digest(ROOT/'data/audio_manifest.json'),
        'native_normalizers_sha256':digest(SOURCE/'data/normalizers.json'),
        'source_sha256':digest(__file__)}
    authenticated_json(path,result)
    return result


class MusicResident(MotionResident):
    def __init__(self,manifest,audio_norms,frames=960):
        self.entries = [e for e in manifest['entries'] if e['partition']=='train' and e['n_frames']>=frames]
        self.host_aux,self.aux_dim = None,0
        arrays,targets,starts,lengths = [],[],[],[]
        grouped = defaultdict(list)
        cursor = 0
        for i,e in enumerate(self.entries):
            if digest(e['audio_path'])!=e['audio_sha256'] or digest(e['path'])!=e['sha256']:
                raise ValueError('Paired TRAIN artifact changed')
            with np.load(e['audio_path']) as data:
                arrays.append(((data['x']-np.asarray(audio_norms['mean'],np.float32))/np.asarray(audio_norms['std'],np.float32)).astype(np.float32))
            with np.load(e['path']) as data:
                targets.append(data['features'])
            if len(arrays[-1])!=len(targets[-1]):
                raise ValueError('Audio/motion frame alignment mismatch')
            starts.append(cursor)
            lengths.append(len(targets[-1]))
            cursor += lengths[-1]
            grouped[e['canonical_id']].append(i)
        self.groups = [grouped[k] for k in sorted(grouped)]
        self.starts,self.lengths = np.asarray(starts),np.asarray(lengths)
        self.x = torch.from_numpy(np.concatenate(arrays)).cuda()
        self.y = torch.from_numpy(np.concatenate(targets)).cuda()
        self.b = torch.zeros((len(self.x),1),device='cuda')
        if not self.groups:
            raise ValueError('No eligible paired training crops')


if __name__=='__main__':
    prepare_targets()
