"""Independent CPU media audit. Does not import evaluator, renderer or model."""
from pathlib import Path
from fractions import Fraction
import hashlib, json, os, shutil, sys, time, subprocess
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
from scripts.motion.audit_dance_optics_media import digest, timeline, stream_end, q, file_command, pcm_command
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/dance_diagnosis_20260911/pose_stage_01')
OUT=ROOT/'observer_audits/media'
os.environ['CUDA_VISIBLE_DEVICES']=''
def save(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
def auth(p):
    d=json.loads(p.read_text()); h=hashlib.sha256(json.dumps(d['payload'],sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if h!=d['sha256']: raise ValueError('Payload checksum '+str(p))
    return d['payload']
def storage():
    def size(p): return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    r={'audit_bytes':size(OUT),'study_bytes':size(ROOT),'free_bytes':shutil.disk_usage(ROOT).free,'unix':time.time()}
    if r['audit_bytes']+8*2**20>128*2**20 or r['study_bytes']+8*2**20>1.75*2**30 or r['free_bytes']-8*2**20<2**30: raise RuntimeError('Disk reserve '+str(r))
    return r

def main():
    initial=storage(); freeze=auth(ROOT/'freeze.json'); plan=json.loads((ROOT/'frozen/plan.json').read_text())
    reg=json.loads((ROOT/'registration.json').read_text())
    assert json.loads(reg['hypothesis']['plan'])==plan==json.loads(reg['experiment']['config_note'])
    frozen={k:{'expected':h,'actual':digest(ROOT/'frozen'/k)} for k,h in freeze['files'].items()}
    assert all(x['expected']==x['actual'] for x in frozen.values())
    common={'schema':1,'scope':'Exact media integrity and AAC decoded identity only; no musicality, source alignment or human AV acceptance.',
       'started_unix':time.time(),'registration_sha256':digest(ROOT/'registration.json'),'freeze_sha256':digest(ROOT/'freeze.json'),
       'plan_sha256':digest(ROOT/'frozen/plan.json'),'frozen_files_verified':frozen,'storage_initial':initial,
       'tools':{str(x):digest(x) for x in [Path('/usr/bin/ffmpeg'),Path('/usr/bin/ffprobe'),Path(__file__),Path(__file__).with_name('audit_dance_optics_media.py'),Path('/root/lightsem/run.sh')]},
       'runtime':{'executable':sys.executable,'version':sys.version,'CUDA_VISIBLE_DEVICES':os.environ['CUDA_VISIBLE_DEVICES']},
       'reused_utilities':['digest','timeline','stream_end','q','file_command','pcm_command'],
       'no_main_evaluator_renderer_import':True,'raw_RGB_PCM_disk_cache':False,'human_AV_acceptance':None}
    for tool in ['ffmpeg','ffprobe']:
        common[tool+'_version_command']=file_command(['/usr/bin/'+tool,'-version'],OUT,tool+'_version')
    rows=[]
    expected={i['id']+'_planned':i for i in plan['items']}
    expected['shpongle_prior_flow0_global_matched']=next(i for i in plan['items'] if i['key']=='shpongle')
    videos={x.parent.name:x for x in (ROOT/'renders').glob('*/matched.mp4')}
    assert set(videos)==set(expected) and len(videos)==14
    for name in sorted(expected):
      folder=OUT/name; folder.mkdir(exist_ok=False)
      row={'id':name,'status':'failed','checks':{},'commands':{},'storage_before':storage()}
      try:
        video=videos[name]; rpath=video.parent/'render.json'; receipt=auth(rpath); started=auth(video.parent/'started.json'); item=expected[name]
        n=item['frames']; duration=Fraction(n,30)
        prov={'video_sha256':digest(video),'render_receipt_sha256':digest(rpath),'started_receipt_sha256':digest(video.parent/'started.json'),
              'source_audio_sha256':digest(receipt['audio_path']),'source_audio_path':receipt['audio_path'],'freeze_sha256':digest(ROOT/'freeze.json')}
        row['provenance']=prov
        assert prov['video_sha256']==receipt['files']['matched.mp4']
        assert prov['source_audio_sha256']==receipt['audio_sha256']
        assert receipt['freeze_sha256']==started['freeze_sha256']==prov['freeze_sha256']
        assert receipt['entry']['frames']==n and receipt['entry']['id']==name and receipt['item']==item
        assert receipt['audio_trim_start_s']==item['start']/30+item['audio_offset_s']
        assert receipt['audio_duration_s']==float(duration)
        row['receipt_context']={k:receipt[k] for k in ['entry','layout','caption_provenance','audio_trim_start_s','audio_duration_s','audio_padding','pose_padding','ffmpeg_command']}
        row['expected']={'frames':n,'fps':30,'width':1920,'height':360,'duration':q(duration)}
        base=['/usr/bin/ffprobe','-v','error','-threads','2','-codec_whitelist','h264,aac']
        fields='stream=index,codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,r_frame_rate,time_base,start_pts,duration_ts,nb_frames,nb_read_frames,sample_rate,channels,channel_layout,initial_padding:format=duration,start_time:frame=media_type,stream_index,pts,duration,pkt_duration,nb_samples'
        cmd=base+['-show_entries',fields,'-show_frames','-count_frames','-of','json',str(video)]
        rec=file_command(cmd,folder,'frames'); row['commands']['frames']=rec
        assert rec['returncode']==0
        data=json.loads((folder/rec['stdout']).read_text()); streams=data['streams']
        assert len(streams)==2
        v=next(s for s in streams if s['codec_type']=='video'); a=next(s for s in streams if s['codec_type']=='audio')
        row['streams']=streams; row['format']=data['format']
        vt=timeline([f for f in data['frames'] if f['media_type']=='video'],v,'video')
        at=timeline([f for f in data['frames'] if f['media_type']=='audio'],a,'audio')
        row['video_timeline']=vt; row['audio_timeline']=at
        row['stream_timing']={'video':stream_end(v),'audio':stream_end(a)}
        cmd=base+['-select_streams','a:0','-show_packets','-show_entries','packet=pts,dts,duration:packet_side_data=side_data_type,skip_samples,discard_padding,skip_reason,discard_reason','-of','json',str(video)]
        rec=file_command(cmd,folder,'audio_packets'); row['commands']['audio_packets']=rec
        assert rec['returncode']==0
        packets=json.loads((folder/rec['stdout']).read_text())['packets']
        missing=[i for i,x in enumerate(packets) if 'pts' not in x or 'duration' not in x]
        discontinuities=[i for i in range(1,len(packets)) if all('pts' in packets[j] and 'duration' in packets[j] for j in [i-1,i]) and packets[i]['pts']!=packets[i-1]['pts']+packets[i-1]['duration']]
        row['audio_packets']={'count':len(packets),'first':packets[0],'last':packets[-1],'missing_timing':missing,'discontinuities':discontinuities,
            'side_data':[{'index':i,**x} for i,x in enumerate(packets) if x.get('side_data_list')]}
        cmd=['/usr/bin/ffmpeg','-v','error','-nostdin','-xerror','-err_detect','explode','-hwaccel','none','-threads','2','-c:v','h264','-i',str(video),'-map','0:v:0','-an','-fps_mode','passthrough','-pix_fmt','rgb24','-c:v','rawvideo','-threads:v','1','-f','rawvideo','pipe:1']
        rgbdir=folder/'rgb'; rgbdir.mkdir()
        rgb=pcm_command(cmd,rgbdir); rgb['representation']='RGB24 1920x360, all decoded frames, passthrough no fps synthesis; streaming SHA256 and byte count; no raw disk file'
        row['rgb']=rgb; row['commands']['rgb']=rgb
        cmd=['/usr/bin/ffmpeg','-v','error','-nostdin','-xerror','-err_detect','explode','-hwaccel','none','-threads','2','-c:a','aac','-i',str(video),'-map','0:a:0','-vn','-c:a','pcm_f32le','-threads:a','1','-f','f32le','pipe:1']
        pcm=pcm_command(cmd,folder); row['pcm']=pcm; row['commands']['pcm']=pcm
        pcm.update(sample_rate=int(a['sample_rate']),channels=a['channels'],channel_layout=a.get('channel_layout'))
        pcm['sample_count_per_channel']=pcm['bytes']//(4*a['channels'])
        row['duration_differences']={k:q(Fraction(x['rational_seconds'])-duration) for k,x in [('video_decoded',vt['decoded_end']),('audio_decoded',at['decoded_end']),('video_stream',row['stream_timing']['video']['end']),('audio_stream',row['stream_timing']['audio']['end'])]}
        row['decoded_audio_tail_samples']=int((Fraction(at['decoded_end']['rational_seconds'])-duration)*int(a['sample_rate']))
        row['checks']={
          'one_h264_one_aac':v['codec_name']=='h264' and a['codec_name']=='aac',
          'geometry':(v['width'],v['height'])==(1920,360),
          'declared_rates_30fps':Fraction(v['avg_frame_rate'])==Fraction(v['r_frame_rate'])==30,
          'all_frame_counts_exact':vt['decoded_frame_count']==int(v['nb_frames'])==int(v['nb_read_frames'])==n and rgb['bytes']==n*1920*360*3,
          'every_video_PTS_contiguous_30fps':vt['regular_30fps'],
          'video_first_PTS_zero':vt['first_pts']==q(0),
          'video_decoded_end_exact':vt['decoded_end']==q(duration),
          'video_stream_end_exact':row['stream_timing']['video']['end']==q(duration),
          'audio_stream_end_exact':row['stream_timing']['audio']['end']==q(duration),
          'audio_first_PTS_zero':at['first_pts']==q(0),
          'every_audio_PTS_nb_samples_contiguous':at['pts_nb_samples_contiguous'],
          'every_audio_packet_PTS_duration_contiguous':not missing and not discontinuities,
          'PCM_bytes_match_all_decoded_samples':pcm['bytes']==at['sample_count_per_channel']*a['channels']*4,
          'all_commands_succeeded':all(x['returncode']==0 for x in row['commands'].values()),
          'no_error_level_messages':all(not x['stderr_nonempty'] for x in row['commands'].values()),
          'decoded_RGB_matches_renderer_decode':rgb['sha256']==receipt['media']['decoded_rgb_sha256'],
          'bound_files_unchanged':digest(video)==prov['video_sha256'] and digest(rpath)==prov['render_receipt_sha256'] and digest(receipt['audio_path'])==prov['source_audio_sha256']}
        row['audio_decoded_end_exact']=at['decoded_end']==q(duration)
        row['status']='complete'; row['all_integrity_checks_pass']=all(row['checks'].values())
      except Exception as e:
        row['error']={'type':type(e).__name__,'message':str(e)}
      row['artifacts']={str(x.relative_to(folder)):digest(x) for x in folder.rglob('*') if x.is_file()}
      save(folder/'audit.json',row); rows.append(row)
      print(json.dumps({'id':name,'status':row['status'],'failed_checks':[k for k,v in row['checks'].items() if not v],'tail_samples':row.get('decoded_audio_tail_samples'),'error':row.get('error')}),flush=True)
    common['members']=rows; common['completed']=sum(x['status']=='complete' for x in rows)
    pair=[x for x in rows if x['id'].startswith('shpongle_')]
    if all(x['status']=='complete' for x in pair) and len(pair)==2:
      common['shpongle_original_vs_global_matched_PCM_identity']={'equal':len({tuple(x['pcm'][k] for k in ['sha256','bytes','sample_rate','channels','channel_layout']) for x in pair})==1,
       'members':{x['id']:x['pcm'] for x in pair},'scope':'Decoded output AAC equality in native-rate native-channel interleaved float32 PCM. NOT source lossless identity or musical alignment.'}
    common['storage_final']=storage(); common['completed_unix']=time.time()
    common['all_14_integrity_checks_pass']=len(rows)==14 and all(x.get('all_integrity_checks_pass',False) for x in rows)
    common['all_14_audio_decoded_ends_exact']=len(rows)==14 and all(x.get('audio_decoded_end_exact',False) for x in rows)
    save(OUT/'audit.json',common)
    print('COMPLETE '+str(OUT/'audit.json'),flush=True)
if __name__=='__main__': main()
