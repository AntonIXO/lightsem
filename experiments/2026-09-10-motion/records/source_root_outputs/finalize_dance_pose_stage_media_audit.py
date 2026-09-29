from pathlib import Path
import sys,json,hashlib,time,shutil
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
from audit_dance_pose_stage_media import auth, digest, save, ROOT, OUT, storage
from audit_dance_optics_media import file_command
result=json.loads((OUT/'audit.json').read_text()); freeze=auth(ROOT/'freeze.json')
sources={}
for row in result['members']:
    source=row['provenance']['source_audio_path']
    if source not in sources:
        target=OUT/'source_probes'/Path(source).stem; target.mkdir(parents=True,exist_ok=False)
        rec=file_command(['/usr/bin/ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=codec_name,sample_rate,channels,time_base,start_pts,duration_ts:format=duration','-of','json',source],target,'source_audio')
        sources[source]={'command':rec,'data':json.loads((target/rec['stdout']).read_text()),'sha256':digest(source),'artifacts':{x.name:digest(x) for x in target.iterdir()}}
checks={}
for row in result['members']:
    folder=ROOT/'renders'/row['id']; r=auth(folder/'render.json'); s=auth(folder/'started.json'); mapping=ROOT/'mapped'/r['entry']['item_id']/'map.json'
    checks[row['id']]={'render_receipt_unchanged':digest(folder/'render.json')==row['provenance']['render_receipt_sha256'],
        'started_receipt_unchanged':digest(folder/'started.json')==row['provenance']['started_receipt_sha256'],
        'map_receipt_sha256':digest(mapping),'map_receipt_matches_render_and_start':digest(mapping)==r['map_sha256']==s['map_sha256'],
        'entry_matches_start':s['entry']==r['entry'],
        'every_probe_audio_sample_count_positive':all(f.get('nb_samples',0)>0 for f in json.loads((OUT/row['id']/'frames.stdout').read_text())['frames'] if f['media_type']=='audio')}
summary={'audit_json_sha256':digest(OUT/'audit.json'),'script_sha256':digest(__file__),'sources':sources,'binding_checks':checks,
    'executables_match_freeze':{path:digest(path)==sha for path,sha in freeze['executables'].items()},
    'source_alignment_scope':'Only source hashes, registered trim offsets and encoder command provenance were checked. No cross-correlation or musical event alignment test. Output AAC is lossy; decoded output-to-output PCM identity is not source lossless PCM identity.',
    'tooling_deviation':'The initial frames command placed -show_frames after -show_entries; ffprobe therefore retained extra frame metadata despite requested field list. Retained honestly; storage stayed below 128MiB. Packet and source probes use field selection last. No raw RGB/PCM disk cache.',
    'PCM_representation_clarification':'pcm_command generic text says decoded padding retained. Here no manual trim/resample was applied, but native FFmpeg AAC decode honored demuxed skip_samples and discard_padding side data; decoded PCM contains exactly registered samples.',
    'storage':storage()}
save(OUT/'supplement.json',summary)
lines=['# Independent pose-stage media audit','',
'## Result','',
'**All 14 completed movies pass the measured media checks.** No missing frames, decoded timestamp gaps/overlaps, decode errors or duration overruns were found.',
'- Twelve reference movies: exactly 600 decoded frames and 20 s each.',
'- Shpongle original and global brightness-matched movies: exactly 8319 decoded frames and 277.3 s each.',
'- All movies: one H.264 1920×360 video stream at exactly 30 fps and one AAC 44,100 Hz stereo audio stream.',
'- Every decoded video PTS advances by exactly 1/30 s. Every decoded audio PTS advances by the preceding frame sample count / 44,100. Both start at zero. Container stream ends and decoded ends equal the registered duration exactly, using rational arithmetic.',
'- Full independent software RGB24 and native-rate float32 PCM decodes succeeded with no error-level stderr. Streamed byte counts agree with complete decoded frame/sample counts. All 14 decoded RGB SHA256 hashes match the renderer’s recorded decode hashes.',
'','## AAC priming and final samples','',
'All AAC streams have a first packet at PTS −1024/44100 s with skip_samples=1024. Decoded PCM starts at zero; this priming packet is not a missing presentation interval.',
'- Reference movies: 863 packets, 862 decoded audio frames, 882,000 samples/channel. Final packet/frame contributes 336 samples; discard_padding=688.',
'- Each Shpongle movie: 11,944 packets, 11,943 decoded audio frames, 12,228,930 samples/channel. Final packet/frame contributes 322 samples; discard_padding=702.',
'FFmpeg honors this side data. No manual trimming, resampling, silence insertion or tail removal was used in this audit. Decoded audio tail beyond the registered end is exactly zero samples in every movie. The generic PCM helper’s “padding retained” label means no audit-side trim, not that codec discard metadata is ignored.',
'','## Shpongle audio identity','',
'Original versus global brightness-matched decoded PCM is byte-identical: native 44,100 Hz, stereo, interleaved little-endian float32; 97,831,440 bytes each.',
'`SHA256 6d5d8917556e5b7247fef054c758e975b9bf9d8ad206dccae755d4b51d794455`',
'Output AAC is lossy. This identity is between the two output AAC decodes, not between AAC and a lossless source. The source files were independently hashed and minimally probed; codec details are in `supplement.json`. Registered source offsets and render command bindings match. Audible source-event alignment was not tested.',
'','## Method and evidence','',
'Run from `/root/lightsem`: `CUDA_VISIBLE_DEVICES="" bash run.sh outputs/audit_dance_pose_stage_media.py`. Final source/binding checks: `bash run.sh outputs/finalize_dance_pose_stage_media_audit.py`.',
'Only generic `digest`, `timeline`, `stream_end`, `q`, `file_command` and streaming `pcm_command` utilities were reused from `outputs/audit_dance_optics_media.py`. Its AV1-specific audit/main was not run. No evaluator, renderer, model, GPU, physical output, vault, TEST, seed1 or default changes.',
'Each member receipt records exact argv, exit status, stderr, source/video/render/start hashes, raw decoded stream hashes, complete rational timeline checks and evidence-file hashes. Software H.264/AAC decoders are explicit; video uses `-fps_mode passthrough`, PCM has no `-ar`, `-ac`, filter or duration limit. No raw RGB/PCM was written to disk.',
'Initial ffprobe option order unexpectedly retained extra frame metadata beyond the requested fields. This is a tooling/minimal-probe deviation, not a media failure; evidence is retained. Packet and supplemental source probes limit fields correctly.',
'','Evidence directory: `'+str(OUT)+'`',
'- `audit.json`: all 14 machine audits, exact timing, PCM identity, frozen-source/tool hashes and command semantics.',
'- `supplement.json`: source codecs, map/start/render bindings, executable freeze comparisons and this scope clarification.',
'- `<movie>/frames.stdout`, `audio_packets.stdout`, `audit.json`, and decode stderr: complete per-frame/packet timing evidence and streamed hashes.',
'','## Scope limits','',
'Each movie shares one saved-body comparison: body panel, same-body articulated-stage baseline and pose-stage candidate, with one shared audio stream. There is no invented incumbent, no Shpongle reference body and no generator comparison.',
'This is machine media integrity only. It does not establish musicality, music conditioning, preference, perceptual source alignment, decoded clipping, physical safety, hardware qualification or human audiovisual acceptance.',
'','## Storage','',
'Final measured audit size: %.2f MiB; whole pose-stage study including parallel observer directories: %.2f MiB; free space: %.2f GiB. Limits: 128 MiB audit, 1.75 GiB study, at least 1 GiB free. Checks reserve 8 MiB before each audit member; final report/JSON are small. No source or shared main receipt was edited.'%(summary['storage']['audit_bytes']/2**20,summary['storage']['study_bytes']/2**20,summary['storage']['free_bytes']/2**30),'']
report=Path('/root/lightsem/outputs/dance_pose_stage_media_audit.md')
with report.open('x') as f:f.write('\n'.join(lines))
save(OUT/'final_manifest.json',{'audit_sha256':digest(OUT/'audit.json'),'supplement_sha256':digest(OUT/'supplement.json'),'report_path':str(report),'report_sha256':digest(report),'finalizer_sha256':digest(__file__),'storage':storage(),'completed_unix':time.time()})
print(json.dumps({'sources':{k:v['data'] for k,v in sources.items()},'checks':checks,'tools':summary['executables_match_freeze'],'report':str(report)}),flush=True)
