"""Measured EBU momentary/short-term loudness and fixed playback timestamps."""
import re
import subprocess
from pathlib import Path

import numpy as np

from .audio_prepare import ROOT
from ..night_common import BASE
from ..stage11_common import read_authenticated,authenticated_json,digest


def loudness(audio_path,output):
    audio_path,output = Path(audio_path),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    receipt = output/'loudness.json'
    if receipt.exists():
        report = read_authenticated(receipt)
        if digest(audio_path)!=report['audio_sha256'] or digest(report['path'])!=report['sha256']:
            raise ValueError('Loudness input/artifact changed')
        return report
    metadata = output/'ebur128.txt'
    if not metadata.exists():
        temporary=metadata.with_suffix('.partial')
        command = ['ffmpeg','-hide_banner','-v','error','-nostdin','-i',str(audio_path),
                   '-af',f'ebur128=metadata=1,ametadata=print:file={temporary}',
                   '-f','null','-']
        subprocess.run(command,check=True,capture_output=True)
        temporary.replace(metadata)
    rows,row = [],None
    for line in metadata.read_text().splitlines():
        if line.startswith('frame:'):
            if row is not None:
                rows.append(row)
            match = re.search(r'pts_time:([\d.eE+-]+)',line)
            if not match:
                raise ValueError('Missing loudness timestamp')
            # Metadata frames contain100ms of audio and have start PTS. Expose
            # a measurement at the end of that block, without advancing it.
            row = {'time':float(match.group(1))+.1}
        elif line.startswith('lavfi.r128.M='):
            row['momentary'] = float(line.split('=',1)[1])
        elif line.startswith('lavfi.r128.S='):
            row['short_term'] = float(line.split('=',1)[1])
    if row is not None:
        rows.append(row)
    if not rows or any(set(r)!= {'time','momentary','short_term'} for r in rows):
        raise ValueError('Incomplete EBU loudness metadata')
    arrays = {key:np.asarray([r[key] for r in rows],np.float64) for key in rows[0]}
    if not all(np.isfinite(v).all() for v in arrays.values()) or np.any(np.diff(arrays['time'])<=0):
        raise ValueError('Invalid loudness curve')
    duration=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=noprint_wrappers=1:nokey=1',str(audio_path)],text=True))
    if abs(arrays['time'][-1]-duration)>.11:raise ValueError('Loudness metadata does not cover the full track')
    path = output/'loudness.npz'
    if path.exists():
        with np.load(path) as old:
            if any(not np.array_equal(old[k],v) for k,v in arrays.items()):
                raise ValueError('Interrupted loudness replay differs')
    else:
        np.savez_compressed(path,**arrays)
    report = {'path':str(path),'sha256':digest(path),'audio_path':str(audio_path),'audio_sha256':digest(audio_path),
        'metadata_sha256':digest(metadata),'frames':len(rows),'measurement':'FFmpeg ebur128 K-weighted LUFS,400ms momentary and3s short-term,100ms metadata frames',
        'timestamp':'metadata PTS +100ms, end of analysis block','source_sha256':digest(__file__),
        'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'documentation':'https://ffmpeg.org/ffmpeg-filters.html#ebur128'}
    authenticated_json(receipt,report)
    return report


def prepare_shpongle():
    meta = read_authenticated(BASE/'inference/shpongle_input.json')
    audio = read_authenticated(BASE/'protocol.json')['demo']['audio_path']
    if digest(audio)!=meta['source_audio_sha256'] or digest(meta['features_path'])!=meta['features_sha256']:
        raise ValueError('Authorized Shpongle source changed')
    out = ROOT/'data/shpongle'
    out.mkdir(parents=True,exist_ok=True)
    with np.load(meta['features_path']) as data:
        raw = data['x']
    times = np.arange(round(meta['source_duration_s']*30))/30
    aligned = np.stack([np.interp(times,np.arange(len(raw))/40,raw[:,i]) for i in range(92)],axis=1).astype(np.float32)
    path = out/'audio30.npz'
    if path.exists():
        with np.load(path) as old:
            if not np.array_equal(old['x'],aligned):
                raise ValueError('Shpongle feature replay differs')
    else:
        np.savez_compressed(path,x=aligned)
    curves = loudness(audio,out)
    report = {'path':str(path),'sha256':digest(path),'audio_path':audio,'source':meta,'loudness':curves,
        'frames':len(aligned),'fps':30,'source_sha256':digest(__file__),
        'alignment':'fixed interpolation of verified Base92 40Hz cache to30Hz; no new audio model input'}
    authenticated_json(out/'input.json',report)
    return report


if __name__=='__main__':
    prepare_shpongle()
