"""Independent annotated rhythm benchmark and separately labelled synthetic stress tests."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,zipfile,time
import numpy as np
from scipy.signal import find_peaks
from .night_common import ROOT
from ..stage11.stage11_common import authenticated_json,read_authenticated,digest

FILES={'annotation.zip':'b39b78e63d3446f2e54ddb7a54df9b10','audio_mono-mic.zip':'275966d6610ac34999b58426beb119c3'}
EXCLUDED=('04_BN3-154-E_comp','04_Jazz1-200-B_comp')

def download_mirror(root=ROOT):
    """Transport fallback, accepted only when every original file MD5 matches."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    import importlib.util
    root=Path(root);folder=root/'rhythm/guitarset';folder.mkdir(parents=True,exist_ok=True)
    index_path=Path(importlib.util.find_spec('mirdata').origin).parent/'datasets/indexes/guitarset_index_1.1.0.json'
    index=json.loads(index_path.read_text())['tracks'];verified=[]
    revision='f0eb62f128c93a7e64adba2a0f1d77a7edcac90c';parquets=[]
    for i in range(5):
        path=hf_hub_download('taohu/guitarset',f'data/train-{i:05d}-of-00005.parquet',repo_type='dataset',revision=revision,
                             cache_dir=str(root/'rhythm/mirror_cache'))
        parquets.append({'path':path,'sha256':digest(path)})
        table=pq.read_table(path,columns=['track_id','jams','audio_mic'])
        for row in table.to_pylist():
            key=row['track_id'];expected=index[key];audio=row['audio_mic']['bytes']
            if hashlib.md5(audio).hexdigest()!=expected['audio_mic'][1]:raise RuntimeError('Mirror audio is not original: '+key)
            parsed=json.loads(row['jams']);annotation=json.dumps(parsed,indent=2,ensure_ascii=True).encode()
            if hashlib.md5(annotation).hexdigest()!=expected['jams'][1]:
                annotation=json.dumps(parsed,indent=2,ensure_ascii=False).encode()
            if hashlib.md5(annotation).hexdigest()!=expected['jams'][1]:raise RuntimeError('Mirror annotation differs: '+key)
            for category,content in [('audio_mic',audio),('jams',annotation)]:
                target=folder/expected[category][0]
                if not target.resolve().is_relative_to(folder.resolve()):raise RuntimeError('Unsafe indexed path')
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists():target.write_bytes(content)
                if hashlib.md5(target.read_bytes()).hexdigest()!=expected[category][1]:raise RuntimeError('Existing original file differs')
            verified.append({'key':key,'audio_md5':expected['audio_mic'][1],'jams_md5':expected['jams'][1]})
        print(json.dumps({'stage':'guitarset_verified_mirror','files_verified':2*len(verified)}),flush=True)
    if len(verified)!=360 or {r['key'] for r in verified}!=set(index):raise RuntimeError('Incomplete mirrored corpus')
    authenticated_json(folder/'source.json',{'source':'https://zenodo.org/records/3371780','version':'1.1.0',
        'transport':'taohu/guitarset','transport_revision':revision,'parquets':parquets,'verified_original_files':verified,
        'index_source':'mirdata==0.3.8 packaged original per-file checksums','index_sha256':digest(index_path),
        'annotation_serialization':'JSON decoded and original indent2 serialization restored, then original MD5 verified',
        'excluded':list(EXCLUDED),'exclusion_source':'https://github.com/marl/GuitarSet/issues/5'})

def download(root=ROOT):
    root=Path(root);folder=root/'rhythm/guitarset';folder.mkdir(parents=True,exist_ok=True)
    for name,expected in FILES.items():
        path=folder/name
        if not path.exists():
            temporary=path.with_suffix('.partial')
            urls=['https://zenodo.org/api/records/3371780/files/'+name+'/content',
                  'https://zenodo.org/records/3371780/files/'+name+'?download=1']
            error=None
            for url in urls:
                try:
                    with urllib.request.urlopen(url,timeout=90) as src,temporary.open('wb') as dest:
                        while chunk:=src.read(8*1024*1024):dest.write(chunk)
                    error=None;break
                except Exception as exc:error=exc
            if error is not None:raise error
            temporary.rename(path)
        if hashlib.md5(path.read_bytes()).hexdigest()!=expected:raise RuntimeError('GuitarSet checksum mismatch')
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                target=(folder/item.filename).resolve()
                if not target.is_relative_to(folder.resolve()):raise RuntimeError('Unsafe archive path')
            z.extractall(folder)
    authenticated_json(folder/'source.json',{'source':'https://zenodo.org/records/3371780','version':'1.1.0','archives':FILES,
        'excluded':list(EXCLUDED),'exclusion_source':'https://github.com/marl/GuitarSet/issues/5'})

def prepare(root=ROOT):
    import soundfile as sf,librosa
    from lightsem.audio.online import extract_bus
    from .night_features import fourier_features
    root=Path(root);folder=root/'rhythm/guitarset';out=root/'rhythm/features';out.mkdir(exist_ok=True)
    annotations={p.stem:p for p in folder.rglob('*.jams')};rows=[];begin=time.monotonic()
    for wav in sorted(folder.rglob('*_mic.wav')):
        if wav.name.startswith('.') or '__MACOSX' in wav.parts:continue
        key=wav.stem.removesuffix('_mic')
        if key in EXCLUDED:continue
        if key not in annotations:raise ValueError('Missing rhythm annotation')
        path=out/(key+'.npz');rp=out/(key+'.json')
        if rp.exists():rows.append(read_authenticated(rp));continue
        data=json.loads(annotations[key].read_text());ann=next(a for a in data['annotations'] if a['namespace']=='beat_position')
        beats=np.asarray([v['time'] for v in ann['data']],float)
        positions=np.asarray([v['value']['position'] for v in ann['data']],int)
        audio,sr=sf.read(wav,dtype='float32');audio=audio.mean(1) if audio.ndim==2 else audio
        bus=extract_bus(y=audio,sr=sr,fps_out=40.,causal=False)
        from ..beat_conditioned.beat_conditioned_data import X_NAMES
        from lightsem.model.light_data import AUDIO_SCALARS
        x=np.concatenate((bus.mel.T,np.stack([bus.channels[k] for k in AUDIO_SCALARS],1)),1).astype(np.float32)
        pulse=librosa.beat.plp(onset_envelope=bus.channels['onset'],sr=40,hop_length=1,win_length=321,tempo_min=30,tempo_max=240)
        peaks=find_peaks(pulse,distance=5,prominence=.1)[0]/40.
        target=np.zeros((len(x),2),np.float32)
        for j,tt in enumerate((beats,beats[positions==1])):
            ii=np.rint(tt*40).astype(int);ii=ii[(ii>=0)&(ii<len(x))];target[ii,j]=1.
        resets=np.flatnonzero(np.diff(bus.channels['bar_phase'])<-.5)+1
        np.savez_compressed(path,x=x,fourier=fourier_features(x),target=target,beats=beats,downbeats=beats[positions==1],
                            constant_grid=bus.beats,constant_downbeats=bus.downbeats,input_bar_events=resets/40.,plp=peaks,pulse=pulse)
        row={'key':key,'player':key.split('_')[0],'lead_sheet':key.split('_')[1],'path':str(path),'sha256':digest(path),
             'audio_sha256':digest(wav),'annotation_sha256':digest(annotations[key]),'seconds':len(audio)/sr}
        authenticated_json(rp,row);rows.append(row)
        if len(rows)%30==0:print(json.dumps({'stage':'rhythm_prepare','clips':len(rows),'elapsed_s':time.monotonic()-begin}),flush=True)
    if len(rows)!=358:raise RuntimeError(f'Expected358annotated clips, got{len(rows)}')
    authenticated_json(root/'rhythm/input_manifest.json',{'rows':rows,'exclusions':list(EXCLUDED),'frames_hz':40,
        'source_sha256':digest(__file__),'limitations':'Guitar performances with accompaniment-based meter; not an EDM or arbitrary-meter generalization claim'})

def evaluate(root=ROOT):
    import mir_eval
    from ..beat_conditioned.beat_conditioned_metrics import bootstrap_mean
    root=Path(root);manifest=read_authenticated(root/'rhythm/input_manifest.json');rows=[]
    for row in manifest['rows']:
        if digest(row['path'])!=row['sha256']:raise RuntimeError('Rhythm input changed')
        with np.load(row['path']) as z:
            truth=mir_eval.beat.trim_beats(z['beats']);scores={}
            for mode in ('constant_grid','plp'):
                pred=mir_eval.beat.trim_beats(z[mode]);c=mir_eval.beat.continuity(truth,pred)
                scores[mode]={'f1':float(mir_eval.beat.f_measure(truth,pred)),'cmlt':float(c[1]),'amlt':float(c[3]),
                              'reference_events':len(truth),'estimated_events':len(pred)}
        rows.append({**row,'scores':scores})
    grouped={}
    for row in rows:grouped.setdefault(row['lead_sheet'],[]).append(row)
    delta=[np.mean([r['scores']['plp']['f1']-r['scores']['constant_grid']['f1'] for r in g]) for g in grouped.values()]
    continuity=[np.mean([r['scores']['plp']['cmlt']-r['scores']['constant_grid']['cmlt'] for r in g]) for g in grouped.values()]
    result={'rows':rows,'f1_delta':bootstrap_mean(delta),'cmlt_delta':bootstrap_mean(continuity),
            'means':{m:{k:float(np.mean([r['scores'][m][k] for r in rows])) for k in ('f1','cmlt','amlt')} for m in ('constant_grid','plp')},
            'gates':{'beat_f1':float(np.mean(delta))>=.05,'continuity':float(np.mean(continuity))>=-.02,
                     'real_annotations':len(rows)==358,'integrity':True},
            'independence_unit':'lead sheet; performer-stratified results retained','metric':'mir_eval beat F1 default70ms and continuity; first5s trimmed',
            'source_sha256':digest(__file__)}
    authenticated_json(root/'rhythm/report.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['download','mirror','prepare','evaluate']);a=p.parse_args()
    {'download':download,'mirror':download_mirror,'prepare':prepare,'evaluate':evaluate}[a.stage]()
