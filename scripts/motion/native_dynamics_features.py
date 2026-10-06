"""Prepare real whole-song audio observations for audited Just Dance records."""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
from lightsem.experiments.native_dynamics.data import read,write,sha,ROOT
from lightsem.experiments.dynamic_dance.audio import AudioAnalyzer,load_waveform,extract_base92,guard_disk

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--guarded',action='store_true');a=p.parse_args()
 if not a.guarded:raise ValueError('Frozen GPUguard required')
 from scripts.motion.native_dynamics_guard import authenticate_guarded_job
 receipt=authenticate_guarded_job(__file__)
 manifest=read(a.root/'data/MANIFEST.json');analyzer=AudioAnalyzer('cuda','cpu');reports=[];begin=time.monotonic()
 if (a.root/'features/COMPLETE.json').exists():raise FileExistsError('Feature preparation complete; no replay')
 for row in manifest['entries']:
  guard_disk(a.root,required_bytes=256*2**20)
  entry={'key':row['feature_key'],'partition':row['partition'],'wav_path':row['wav_path'],'wav_sha256':row['wav_sha256'],'timeline_path':row['timeline_path']}
  wave=load_waveform(row['wav_path']);time30=np.arange(row['frames'])/30.;waveid=hashlib.sha256(np.ascontiguousarray(wave).tobytes()).hexdigest()
  base=extract_base92(wave,time30,entry,a.root,waveid,None);del wave
  entry.update(base92_path=base['path'],base92_sha256=base['npz_sha256'],base92_provenance='Audited Just Dance capture; Base92 recomputed from this exact original waveform; no intervention')
  report=analyzer.extract(entry,a.root)
  reports.append({'key':row['key'],'partition':row['partition'],'timeline_path':report['timeline_path'],'timeline_sha256':report['timeline_sha256'],'base92_path':base['path'],'base92_sha256':base['npz_sha256'],'seconds':report['elapsed_s']})
  print('NATIVE_JD_FEATURE',len(reports),len(manifest['entries']),row['key'],report['elapsed_s'],flush=True)
 write(a.root/'features/COMPLETE.json',{'entries':reports,'data_manifest_sha256':sha(a.root/'data/MANIFEST.json'),'elapsed_s':time.monotonic()-begin,'guard_job':receipt['identity']['job_id'],'source':sha(__file__),'no_training':True,'no_TEST':True,'models':'FrozenMuQ layer9,pinnedHTDemucs,BeatThis,AllInOne; same analyzedwaveform for Base92and1061features.'})
if __name__=='__main__':main()
