"""GPU-render fixed 2D pose comparisons from saved VAL probe outputs."""
from pathlib import Path
import argparse,json,math,subprocess,sys,time
import numpy as np
ROOT=Path('/mnt/lightsem-data/experiments/lightsem/night_expression_20260909')
sys.path.insert(0,str(ROOT/'source'))
from lightsem.experiments.night_common import BASE,STAGE
from lightsem.experiments.stage11_common import authenticated_json,read_authenticated,digest,gpu_lock
OUT=ROOT/'delivery/pose'
PANELS=[('Detected 2D pose','descriptor256','truth'),('Descriptors, width 128','descriptor128','full'),
        ('Descriptors, width 256','descriptor256','full'),('Pose-conditioned code 24','pose_code24','full'),
        ('Audio base -> pose','audio_base','full'),('Audio Fourier -> pose','audio_fourier','full'),
        ('Audio MuQ -> pose','audio_muq','full'),('Incumbent audio -> descriptors -> pose','descriptor256','incumbent_audio_to_descriptors')]
VERTEX='''#version 330
in vec2 pos; void main(){gl_Position=vec4(pos,0,1);}
'''
FRAGMENT='''#version 330
uniform vec2 joints[136]; uniform float valid[136]; uniform sampler2D labels;
out vec4 color;
const ivec2 edges[18]=ivec2[](ivec2(0,1),ivec2(0,2),ivec2(1,3),ivec2(2,4),ivec2(0,5),ivec2(0,6),ivec2(5,6),ivec2(5,7),ivec2(7,9),ivec2(6,8),ivec2(8,10),ivec2(5,11),ivec2(6,12),ivec2(11,12),ivec2(11,13),ivec2(13,15),ivec2(12,14),ivec2(14,16));
float lineDistance(vec2 p,vec2 a,vec2 b){vec2 d=b-a;return length(p-a-clamp(dot(p-a,d)/max(dot(d,d),1e-9),0.,1.)*d);}
void main(){
 vec2 px=gl_FragCoord.xy;int col=int(px.x/480.);int row=int(px.y/540.);int panel=(1-row)*4+col;
 vec2 q=mod(px,vec2(480,540));vec2 world=vec2((q.x-240.)/48.,(q.y-42.)/76.);
 vec3 bg=vec3(.985);if(q.x<1.||q.y<1.)bg=vec3(.76);
 if(abs(world.y)<.008&&q.y<480.)bg=vec3(.86);
 float d=1000.;
 for(int i=0;i<18;i++){int a=panel*17+edges[i].x;int b=panel*17+edges[i].y;if(valid[a]>.5&&valid[b]>.5)d=min(d,lineDistance(world,joints[a],joints[b]));}
 vec3 ink=panel==0?vec3(.13):vec3(.10,.32,.49);float alpha=1.-smoothstep(.021,.037,d);
 for(int j=0;j<17;j++){int k=panel*17+j;if(valid[k]>.5)alpha=max(alpha,1.-smoothstep(.037,.052,length(world-joints[k])));}
 if(q.y>475.)alpha=0.;
 vec3 result=mix(bg,ink,alpha);vec4 label=texture(labels,vec2(px.x/1920.,1.-px.y/1080.));
 color=vec4(mix(result,label.rgb,label.a),1.);
}
'''

def freeze():
    OUT.mkdir(exist_ok=True)
    rows=read_authenticated(BASE/'evaluation/input_manifest.json')['rows'][:3]
    files={str(Path(__file__)):digest(__file__)};selected=[]
    for row in rows:
        key=row['entry']['key'];panel_paths={}
        for title,probe,mode in PANELS:
            ep=ROOT/'diagnostics/pose_probes'/probe/'seed_0'/'evaluation.json';r=read_authenticated(ep)
            selected_row=next(v for v in r['rows'] if v['key']==key);path=selected_row['path']
            if digest(path)!=selected_row['sha256']:raise ValueError('Pose input changed')
            files[path]=selected_row['sha256'];panel_paths[probe]=path
        selected.append({'source':row,'panel_paths':panel_paths})
    protocol={'selected':selected,'files':files,'seed':0,'seconds':20,'width':1920,'height':1080,'fps':30,
        'selection':'First three canonical cases in the pre-existing fixed40VALcohort; first20seconds of each saved60second excerpt. Seed0 fixed, no quality selection.',
        'interpretation':'Detected2Dposes in torso lengths; confidence masks shared across comparisons. Pose-conditioned reconstruction is not audio inference. Fixed orthographic scale and camera; no fitted per-panel scale.'}
    path=OUT/'protocol.json'
    if path.exists():
        if read_authenticated(path)!=protocol:raise RuntimeError('Pose video protocol drift')
    else:authenticated_json(path,protocol)

def run():
    import moderngl
    from PIL import Image,ImageDraw,ImageFont
    p=read_authenticated(OUT/'protocol.json')
    for path,sha in p['files'].items():
        if digest(path)!=sha:raise RuntimeError('Pose movie input drift')
    if not read_authenticated(ROOT/'pipeline_state.json').get('gpu_pipeline_complete'):raise RuntimeError('Main GPU pipeline must finish first')
    from lightsem.experiments.stage11_data import load_manifest
    records={r['recording_id']:r for r in load_manifest(STAGE)['recordings']}
    ctx=moderngl.create_context(standalone=True,backend='egl',device_index=0)
    name=ctx.info['GL_RENDERER']
    if 'nvidia' not in name.lower():raise RuntimeError('NVIDIA raster required')
    program=ctx.program(vertex_shader=VERTEX,fragment_shader=FRAGMENT)
    vbo=ctx.buffer(np.array([-1,-1,3,-1,-1,3],np.float32).tobytes());vao=ctx.simple_vertex_array(program,vbo,'pos')
    fbo=ctx.simple_framebuffer((1920,1080),components=4);fbo.use()
    atlas=Image.new('RGBA',(1920,1080),(0,0,0,0));draw=ImageDraw.Draw(atlas)
    font=ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf',18)
    for i,(title,_,_) in enumerate(PANELS):draw.text((480*(i%4)+16,540*(i//4)+20),title,font=font,fill=(25,25,25,255))
    texture=ctx.texture(atlas.size,4,atlas.tobytes());texture.use(0);program['labels'].value=0
    results=[]
    for row in p['selected']:
        src=row['source'];key=src['entry']['key'];rp=OUT/(key+'.json')
        if rp.exists():
            old=read_authenticated(rp)
            if digest(old['path'])!=old['sha256'] or old['protocol_sha256']!=digest(OUT/'protocol.json'):raise ValueError('Pose video drift')
            results.append(old);continue
        poses=[];masks=[]
        for title,probe,mode in PANELS:
            with np.load(row['panel_paths'][probe]) as z:poses.append(z[mode][:800].reshape(800,17,2));masks.append(z['mask'][:800].reshape(800,17,2).all(2))
        pose=np.stack(poses,1);mask=np.stack(masks,1).astype(np.float32)
        audio=records[src['entry']['recording_id']]['wav_path'];offset=src['entry']['offset_s']+src['start_frame']/40
        video=OUT/(key+'.mp4')
        cmd=['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','pipe:0',
            '-ss',str(offset),'-i',audio,'-map','0:v:0','-map','1:a:0','-c:v','av1_nvenc','-preset','p5','-cq','25','-b:v','0',
            '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-t','20','-shortest',str(video)]
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE);begin=time.monotonic()
        try:
            for frame in range(600):
                idx=round(frame*40/30);program['joints'].write(pose[idx].astype(np.float32).tobytes());program['valid'].write(mask[idx].tobytes())
                vao.render(moderngl.TRIANGLES);rgb=np.frombuffer(fbo.read(components=3,alignment=1),np.uint8).reshape(1080,1920,3)[::-1]
                process.stdin.write(rgb.tobytes())
        finally:
            process.stdin.close();error=process.stderr.read().decode();rc=process.wait();process.stderr.close()
        if rc:raise RuntimeError(error)
        r={'key':key,'path':str(video),'sha256':digest(video),'renderer':name,'gpu_task_seconds':time.monotonic()-begin,'frames':600,
            'audio_offset_s':offset,'audio_path':audio,'audio_sha256':digest(audio),'protocol_sha256':digest(OUT/'protocol.json'),'argv':cmd}
        authenticated_json(rp,r);results.append(r)
        print(json.dumps({'stage':'pose_video','key':key,'gpu_task_seconds':r['gpu_task_seconds']}),flush=True)
    ctx.release()
    authenticated_json(OUT/'report.json',{'rows':results,'gpu_task_seconds':sum(r['gpu_task_seconds'] for r in results),'protocol_sha256':digest(OUT/'protocol.json')})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','run']);a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:
        with gpu_lock(ROOT):run()
