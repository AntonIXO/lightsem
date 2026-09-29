#!/usr/bin/env python3
"""Verify delivered media, timestamp coverage and source-audio alignment."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt',type=Path)
    p.add_argument('--binding',type=Path,help='Also authenticate the exact exported timeline and patch')
    args=p.parse_args()
    receipt=json.loads(args.receipt.read_text())
    def file_sha(path):
        h=hashlib.sha256()
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    assert file_sha(receipt['source_audio'])==receipt['source_audio_sha256'], 'Source audio hash mismatch'
    if args.binding:
        assert file_sha(args.binding/'timeline.npz')==receipt['binding_sha256'], 'Binding hash mismatch'
        assert file_sha(args.binding/'fixture_patch.json')==receipt['patch_sha256'], 'Patch hash mismatch'
    video=Path(receipt['video'])
    if hashlib.sha256(video.read_bytes()).hexdigest()!=receipt['sha256']:
        raise ValueError('Video hash mismatch')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)]))
    vs=next(s for s in probe['streams'] if s['codec_type']=='video')
    au=next(s for s in probe['streams'] if s['codec_type']=='audio')
    assert [vs['width'],vs['height']]==receipt['resolution']
    assert int(vs['nb_frames'])==receipt['frames']
    assert abs(float(au['duration'])-receipt['duration'])<.06
    assert abs(float(vs['start_time']))<1e-6 and abs(float(au['start_time']))<1e-6
    subprocess.run(['ffmpeg','-hide_banner','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    times=subprocess.check_output(['ffprobe','-v','error','-select_streams','v','-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',str(video)],text=True)
    pts=np.array([float(line.strip().split(',')[0]) for line in times.splitlines() if line.strip()])
    assert len(pts)==receipt['frames']
    assert np.max(np.abs(pts-np.arange(len(pts))/receipt['fps']))<1e-5
    def pcm(path,ss=None):
        cmd=['ffmpeg','-v','error']
        if ss is not None:cmd+=['-ss',str(ss)]
        cmd+=['-i',str(path),'-t',str(receipt['duration']),'-ac','1','-ar','8000','-f','f32le','pipe:1']
        return np.frombuffer(subprocess.check_output(cmd),dtype='<f4')
    original=pcm(receipt['source_audio'],receipt['start'])
    encoded=pcm(video)
    n=min(len(original),len(encoded)); original=original[:n]; encoded=encoded[:n]
    correlation=float(np.corrcoef(original,encoded)[0,1])
    assert correlation>.98, f'Audio not aligned to declared source offset: {correlation}'
    audit={'status':'verified','video':str(video),'frames':len(pts),'resolution':receipt['resolution'],
        'codec':vs['codec_name'],'audio_codec':au['codec_name'],'duration':receipt['duration'],
        'uniform_frame_timestamps':True,'full_software_decode':True,
        'audio_zero_lag_correlation':correlation,'audio_comparison_rate':8000,
        'audio_source_offset_s':receipt['start'],'bytes':video.stat().st_size,
        'source_audio_hash_verified':True,'binding_hashes_verified':bool(args.binding),
        'sha256':receipt['sha256'],'scope':'Integrity and synchronization; not a photometric or aesthetic certification'}
    target=args.receipt.with_name(args.receipt.stem.replace('_render','_audit')+'.json')
    target.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
