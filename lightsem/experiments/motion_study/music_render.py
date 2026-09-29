"""Four-column review: clean source, two native generators, historical 2D path."""
import subprocess
import time
from pathlib import Path

import numpy as np

from .render import VERTEX,camera_project
from ..stage11.stage11_common import authenticated_json,digest

FRAGMENT = '''#version 330
uniform vec2 joints[176]; uniform float contact[32]; uniform sampler2D labels;
uniform float scale; uniform vec2 resolution; out vec4 color;
uniform int has_video; uniform sampler2D source_video;
const int parents[22]=int[](-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19);
const int feet[4]=int[](7,8,10,11);
const ivec2 edges[18]=ivec2[](ivec2(0,1),ivec2(0,2),ivec2(1,3),ivec2(2,4),ivec2(0,5),ivec2(0,6),ivec2(5,6),ivec2(5,7),ivec2(7,9),ivec2(6,8),ivec2(8,10),ivec2(5,11),ivec2(6,12),ivec2(11,12),ivec2(11,13),ivec2(13,15),ivec2(12,14),ivec2(14,16));
float dl(vec2 p,vec2 a,vec2 b){vec2 d=b-a;return length(p-a-clamp(dot(p-a,d)/max(dot(d,d),1e-10),0.,1.)*d);}
void main(){
 vec2 px=gl_FragCoord.xy;int model=int(px.x/640.);int view=1-int(px.y/480.);int panel=view*4+model;
 vec2 q=mod(px,vec2(640,480));vec2 world=vec2((q.x-320.)/scale,(q.y-75.)/scale);
 vec3 result=vec3(.985);result=mix(result,vec3(.70),1.-smoothstep(.3/scale,1.1/scale,abs(world.y)));
 if(model==0 && has_video==1){result=texture(source_video,vec2(q.x/640.,1.-q.y/480.)).rgb;}
 else if(model==3){
  for(int i=0;i<18;i++){float a=1.-smoothstep(.008,.014,dl(world,joints[panel*22+edges[i].x],joints[panel*22+edges[i].y]));result=mix(result,vec3(.08,.35,.57),a);}
 }else{
  for(int j=1;j<22;j++){float a=1.-smoothstep(.008,.014,dl(world,joints[panel*22+j],joints[panel*22+parents[j]]));vec3 ink=j==1||j==2||j==3||j==6||j==9?vec3(.16):vec3(.08,.35,.57);result=mix(result,ink,a);}
  for(int k=0;k<4;k++){float a=1.-smoothstep(.017,.025,length(world-joints[panel*22+feet[k]]));result=mix(result,contact[panel*4+k]>.5?vec3(.82,.18,.13):vec3(.20,.58,.30),a);}
 }
 if(q.x<1.||q.y<1.)result=vec3(.75);
 vec4 label=texture(labels,vec2(px.x/resolution.x,1.-px.y/resolution.y));color=vec4(mix(result,label.rgb,label.a),1.);
}'''


def render(path,models,incumbent8,incumbent32,audio_path,audio_offset,title,*,source_video=None,fixed_camera=False):
    import moderngl
    from PIL import Image,ImageDraw,ImageFont
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if len(models)!=3:
        raise ValueError('Expected source, regression, flow')
    frames = len(models[0][1])
    for _,j,c in models:
        if j.shape!=(frames,22,3) or c.shape!=(frames,4) or not np.isfinite(j).all():
            raise ValueError('Invalid native review sequence')
    for pose in (incumbent8,incumbent32):
        if pose.shape!=(frames,17,2) or not np.isfinite(pose).all():
            raise ValueError('Invalid historical2D pose; never mask finite predictions')
    origin = models[0][1][0,0]*np.array([1,0,1])
    pose = np.zeros((frames,8,22,2),np.float32)
    contacts = np.zeros((frames,8,4),np.float32)
    centers = []
    for view in range(2):
        source = camera_project(models[0][1]-origin,view)
        center = float((source[...,0].min()+source[...,0].max())/2)
        centers.append(center)
        for i,(_,j,c) in enumerate(models):
            value = camera_project(j-origin,view)
            value[...,0] -= center
            pose[:,view*4+i] = value
            contacts[:,view*4+i] = c
        # Fixed display conversion from historical torso-length units. This is
        # not3D calibration and never enters native quantitative comparisons.
        pose[:,view*4+3,:17] = (incumbent8 if view==0 else incumbent32)*.45
    extent = max(1.8,*(float(np.abs(pose[:,view*4,:,0]).max())+.2 for view in range(2)))
    top = max(2.4,*(float(pose[:,view*4,:,1].max())+.25 for view in range(2)))
    scale = min(290/extent,360/top)
    if fixed_camera:
        scale=90.
        for view in range(2):
            for i,(_,j,c) in enumerate(models):
                pose[:,view*4+i]=camera_project(j,view)
        centers=[0.,0.]
    width,height = 2560,960
    ctx = moderngl.create_context(standalone=True,backend='egl',device_index=0)
    if 'nvidia' not in ctx.info['GL_RENDERER'].lower():
        raise RuntimeError('NVIDIA rendering required')
    program = ctx.program(vertex_shader=VERTEX,fragment_shader=FRAGMENT)
    vbo = ctx.buffer(np.asarray([-1,-1,3,-1,-1,3],np.float32).tobytes())
    vao = ctx.simple_vertex_array(program,vbo,'pos')
    fbo = ctx.simple_framebuffer((width,height),components=4)
    fbo.use()
    program['scale'].value = scale
    program['resolution'].value = (width,height)
    atlas = Image.new('RGBA',(width,height),(0,0,0,0))
    pen = ImageDraw.Draw(atlas)
    font = ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf',17)
    for view in range(2):
        for i in range(4):
            x,y = i*640,view*480
            view_label='same source view' if source_video is not None and i==0 else ('front' if view==0 else 'oblique')
            name = (models[i][0]+' / '+view_label) if i<3 else 'Incumbent 2D / '+('8 steps' if view==0 else '32 steps')
            pen.rectangle((x+1,y+1,x+639,y+57),fill=(250,250,250,250))
            pen.text((x+10,y+7),name,font=font,fill=(20,20,20,255))
            pen.text((x+10,y+32),title if i<3 else 'Frontal only; fixed torso-length display scale',font=font,fill=(80,80,80,255))
    texture = ctx.texture(atlas.size,4,atlas.tobytes())
    texture.use(0)
    program['labels'].value = 0
    program['has_video'].value=int(source_video is not None)
    video_texture=ctx.texture((640,480),3)
    video_texture.use(1);program['source_video'].value=1
    source=None
    if source_video is not None:
        source=subprocess.Popen(['ffmpeg','-v','error','-nostdin','-ss',str(audio_offset),'-i',str(source_video),'-an','-t',str(frames/30),'-vf','fps=30,scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    command = ['ffmpeg','-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
               '-r','30','-i','pipe:0','-ss',str(audio_offset),'-i',str(audio_path),'-map','0:v:0','-map','1:a:0',
               '-c:v','av1_nvenc','-preset','p5','-cq','25','-b:v','0','-pix_fmt','yuv420p',
               '-c:a','aac','-b:a','192k','-t',str(frames/30),'-shortest',str(path)]
    encoder = subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    start = time.monotonic()
    selected = set(np.linspace(0,frames-1,9).round().astype(int))
    shots = []
    try:
        for frame in range(frames):
            if source is not None:
                pixels=source.stdout.read(640*480*3)
                if len(pixels)!=640*480*3:raise RuntimeError('Source video ended early')
                video_texture.write(pixels)
            program['joints'].write(pose[frame].tobytes())
            program['contact'].write(contacts[frame].tobytes())
            vao.render(moderngl.TRIANGLES)
            rgb = np.frombuffer(fbo.read(components=3,alignment=1),np.uint8).reshape(height,width,3)[::-1]
            encoder.stdin.write(rgb.tobytes())
            if frame in selected:
                shot = Image.fromarray(rgb)
                shot.thumbnail((960,480))
                shots.append((frame,shot))
    finally:
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        code = encoder.wait()
        if source is not None:
            source.stdout.close();source_error=source.stderr.read().decode();source_code=source.wait()
            if source_code:raise RuntimeError(source_error)
    if code:
        raise RuntimeError(error)
    tile_w,tile_h = shots[0][1].width,shots[0][1].height+30
    sheet = Image.new('RGB',(3*tile_w,3*tile_h),'white')
    pen = ImageDraw.Draw(sheet)
    for i,(frame,shot) in enumerate(shots):
        x,y = i%3*tile_w,i//3*tile_h
        sheet.paste(shot,(x,y+25))
        pen.text((x+10,y+3),f'{audio_offset+frame/30:.2f}s',fill=(20,20,20),font=font)
    sheet.save(path.with_suffix('.png'))
    report = {'path':str(path),'sha256':digest(path),'frames':frames,'fps':30,'seconds':time.monotonic()-start,
        'source_sha256':digest(__file__),'renderer':ctx.info['GL_RENDERER'],'scale':scale,'camera_centers':centers,
        'audio_path':str(audio_path),'audio_offset_s':audio_offset,'command':command,
        'models':[v[0] for v in models]+['incumbent2D8steps','incumbent2D32steps'],
        'source_video':str(source_video) if source_video else None,'fixed_camera':fixed_camera,
        'historical_display':'fixed.45display units per torso length; frontal only; no physical3D comparability claim'}
    ctx.release()
    authenticated_json(path.with_suffix('.json'),report)
    return report
