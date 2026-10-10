from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from lightsem.experiments.native_dynamics.full_fit_admission import (
    iter_native_tracks, native_metadata,
)
from lightsem.experiments.native_dynamics.rtmpose_cache import masked_track,sha


def _recording_id(path, metadata):
    """Resolve the ID without assuming the retired nested source schema."""
    for key in ('recording_id', 'source_id'):
        if metadata.get(key):
            return str(metadata[key])
    source = metadata.get('source')
    if isinstance(source, dict) and source.get('id'):
        return str(source['id'])
    # The cache filename is the authenticated recording key used by the
    # extraction queue.  ``source_path`` is a human/video path and may not
    # contain that key.
    if path.stem:
        return path.stem
    source_path = metadata.get('source_path')
    if source_path:
        return Path(str(source_path)).stem
    return path.stem

def audit(npz):
 with np.load(npz,allow_pickle=False) as z:
  raw={name:z[name] for name in z.files}
  meta=native_metadata(raw)
  t=np.asarray(raw['t'],np.float64); n=len(t)
  if 'height' not in meta or 'width' not in meta:
   raise ValueError('Native pose metadata requires height and width')
  H=float(meta['height']); W=float(meta['width']); duration=float(t[-1]-t[0])
  if duration <= 0:
   raise ValueError('Native pose timestamps have zero duration')
 rows=[]
 for tid,k,c,b in iter_native_tracks(raw):
  present=np.isfinite(k).all((1,2)) & np.isfinite(b).all(1)
  if present.sum()<30:continue
  try:
   out,met=masked_track(k,b,t,duration,H,confidence=c,confidence_gate=False)
  except Exception as e:continue
  rows.append({'track_id':int(tid),'raw_present_fraction':float(present.mean()),**met,'eligible_windows':int(len(out['starts'])),'max_gap_s':float(np.max(np.diff(t))),'finite_to_finite_jump_edges':int(met['joint_jump_edges'])})
 usable=[r for r in rows if r['eligible_windows']]
 ids=np.asarray(raw['track_id'])
 return {'source':str(npz),'source_sha256':sha(npz),'recording_id':_recording_id(Path(npz),meta),'source_path':meta.get('source_path'),'source_metadata_sha256':meta.get('source_sha256'),'n_source_frames':n,'metadata_n_frames':meta.get('n_frames'),'n_detection_rows':int(len(ids)),'n_raw_tracks':int(np.unique(ids[ids>=0]).size),'n_audited_tracks':len(rows),'n_usable_tracks':len(usable),'valid_16s_windows':sum(r['eligible_windows'] for r in rows),'usable_minutes':sum(r['valid_observation_seconds'] for r in usable)/60,'track_rows':rows,'contract':'ragged detections joined by frame_index to exact source timestamps; no fill/merge; repository masked_track, confidence_gate=False'}

def main(a):
 root=Path(a.root);out=[]
 for cand in a.candidates.split(','):
  for p in sorted((root/'raw'/cand).glob('*.npz')):
   out.append({'candidate':cand,**audit(p)})
 (root/'reports').mkdir(exist_ok=True);path=root/'reports'/'POSE_AUDIT.json';path.write_text(json.dumps({'schema_version':1,'status':'complete','rows':out,'protocol_sha256':sha(root/'PROTOCOL.json')},indent=2)+'\n');print(path)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--candidates',required=True);main(p.parse_args())
