"""GPU skeleton review, fixed front/oblique cameras and synchronized original audio."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import ROOT
from .geometry import FPS, PARENTS, FEET
from ..stage11.stage11_common import authenticated_json, read_authenticated, digest, gpu_lock

VERTEX = '#version 330\nin vec2 pos; void main(){gl_Position=vec4(pos,0,1);}'
FRAGMENT = '''#version 330
uniform vec2 joints[132]; uniform float contact[24]; uniform sampler2D labels;
uniform int nmodels; uniform float scale; uniform vec2 resolution;
out vec4 color;
const int parents[22]=int[](-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19);
const int feet[4]=int[](7,8,10,11);
float distanceLine(vec2 p,vec2 a,vec2 b){vec2 d=b-a;return length(p-a-clamp(dot(p-a,d)/max(dot(d,d),1e-10),0.,1.)*d);}
void main(){
 vec2 px=gl_FragCoord.xy;int model=int(px.x/640.);int view=1-int(px.y/480.);int panel=view*nmodels+model;
 vec2 q=mod(px,vec2(640,480));vec2 world=vec2((q.x-320.)/scale,(q.y-75.)/scale);
 vec3 result=vec3(.985);
 result=mix(result,vec3(.70),1.-smoothstep(.3/scale,1.1/scale,abs(world.y)));
 if(abs(fract(world.x*.5+.5)-.5)<.003)result=min(result,vec3(.94));
 for(int j=1;j<22;j++){
  float alpha=1.-smoothstep(.008,.014,distanceLine(world,joints[panel*22+j],joints[panel*22+parents[j]]));
  vec3 ink=j==1||j==2||j==3||j==6||j==9?vec3(.16):vec3(.08,.35,.57);
  if(j==10||j==11)ink=vec3(.65,.32,.10);
  result=mix(result,ink,alpha);
 }
 for(int k=0;k<4;k++){
  float alpha=1.-smoothstep(.017,.025,length(world-joints[panel*22+feet[k]]));
  result=mix(result,contact[panel*4+k]>.5?vec3(.82,.18,.13):vec3(.20,.58,.30),alpha);
 }
 if(q.x<1.||q.y<1.)result=vec3(.75);
 vec4 label=texture(labels,vec2(px.x/resolution.x,1.-px.y/resolution.y));
 color=vec4(mix(result,label.rgb,label.a),1.);
}'''


def camera_project(joints, view):
    if view == 0:
        return joints[..., [0, 1]]
    # Fixed world camera. Depth remains visible; no framewise recentering.
    return np.stack((.80*joints[..., 0]+.60*joints[..., 2], joints[..., 1]), -1)


def render_comparison(path, models, audio_path, audio_offset=0., title='', snapshots=True):
    import moderngl
    from PIL import Image, ImageDraw, ImageFont
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not 1 <= len(models) <= 3:
        raise ValueError('One to three models per comparison')
    frames = len(models[0][1])
    for _, joints, contacts in models:
        if joints.shape != (frames, 22, 3) or contacts.shape != (frames, 4) or not np.isfinite(joints).all():
            raise ValueError('Invalid comparison motion')
    # ONE source-derived translation shared by every panel, fixed for the clip.
    origin = models[0][1][0, 0]*np.array([1, 0, 1])
    projected, contact = [], []
    for view in range(2):
        for _, joints, contacts in models:
            projected.append(camera_project(joints-origin, view))
            contact.append(contacts)
    camera_centers = []
    for view in range(2):
        source = projected[view*len(models)]
        center = float((source[..., 0].max()+source[..., 0].min())/2)
        camera_centers.append(center)
        for model in range(len(models)):
            projected[view*len(models)+model][..., 0] -= center
    extent = max(1.8, *(float(np.abs(projected[i*len(models)][..., 0]).max())+.2 for i in range(2)))
    top = max(2.4, *(float(projected[i*len(models)][..., 1].max())+.25 for i in range(2)))
    scale = min(290/extent, 360/top)
    pose = np.stack(projected, 1).astype(np.float32)
    contact = np.stack(contact, 1).astype(np.float32)
    width, height = 640*len(models), 960
    ctx = moderngl.create_context(standalone=True, backend='egl', device_index=0)
    if 'nvidia' not in ctx.info['GL_RENDERER'].lower():
        raise RuntimeError('NVIDIA raster required')
    program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
    vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
    vao = ctx.simple_vertex_array(program, vbo, 'pos')
    fbo = ctx.simple_framebuffer((width, height), components=4)
    fbo.use()
    program['nmodels'].value = len(models)
    program['scale'].value = scale
    program['resolution'].value = (width, height)
    atlas = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    font = ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf', 17)
    for view in range(2):
        for index, (name, _, _) in enumerate(models):
            x, y = index*640, view*480
            draw.rectangle((x+1, y+1, x+639, y+57), fill=(250, 250, 250, 250))
            draw.text((x+12, y+8), name+' / '+('front' if view == 0 else 'oblique'), font=font, fill=(20, 20, 20, 255))
            draw.text((x+12, y+32), title, font=font, fill=(90, 90, 90, 255))
    labels = ctx.texture(atlas.size, 4, atlas.tobytes())
    labels.use(0)
    program['labels'].value = 0
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{width}x{height}', '-r', str(FPS), '-i', 'pipe:0', '-ss', str(audio_offset), '-i', str(audio_path),
               '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'av1_nvenc', '-preset', 'p5', '-cq', '25', '-b:v', '0',
               '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-t', str(frames/FPS), '-shortest', str(path)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    started, shots = time.monotonic(), []
    selected = set(np.linspace(0, frames-1, 9).round().astype(int))
    try:
        for frame in range(frames):
            # Fixed GLSL maximum arrays are zero-filled beyond the actual panel count.
            joints = np.zeros((132, 2), np.float32)
            contacts = np.zeros(24, np.float32)
            joints[:pose.shape[1]*22] = pose[frame].reshape(-1, 2)
            contacts[:contact.shape[1]*4] = contact[frame].reshape(-1)
            program['joints'].write(joints.tobytes())
            program['contact'].write(contacts.tobytes())
            vao.render(moderngl.TRIANGLES)
            rgb = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(height, width, 3)[::-1]
            encoder.stdin.write(rgb.tobytes())
            if snapshots and frame in selected:
                shot = Image.fromarray(rgb)
                shot.thumbnail((640, 480))
                shots.append((frame, shot))
    finally:
        encoder.stdin.close()
        error = encoder.stderr.read().decode()
        code = encoder.wait()
    if code:
        raise RuntimeError(error)
    if shots:
        tile_width, tile_height = shots[0][1].width, shots[0][1].height+30
        sheet = Image.new('RGB', (tile_width*3, tile_height*3), 'white')
        pen = ImageDraw.Draw(sheet)
        for i, (frame, shot) in enumerate(shots):
            x, y = i % 3*tile_width, i//3*tile_height
            sheet.paste(shot, (x, y+25))
            pen.text((x+10, y+3), f'{audio_offset+frame/FPS:.2f}s', fill=(20, 20, 20), font=font)
        sheet.save(path.with_suffix('.png'))
    report = {'path': str(path), 'sha256': digest(path), 'renderer': ctx.info['GL_RENDERER'],
              'frames': frames, 'fps': FPS, 'seconds': time.monotonic()-started, 'camera_scale': scale,
              'source_origin': origin.tolist(), 'camera_horizontal_centers': camera_centers,
              'audio_path': str(audio_path), 'audio_offset_s': audio_offset,
              'models': [model[0] for model in models], 'command': command, 'source_sha256': digest(__file__)}
    ctx.release()
    authenticated_json(path.with_suffix('.json'), report)
    return report


def source_review(root=ROOT):
    root = Path(root)
    manifest = read_authenticated(root/'data/manifest.json')
    entries = {e['key']: e for e in manifest['entries']}
    out = root/'delivery/source'
    out.mkdir(parents=True, exist_ok=True)
    protocol = {'manifest_sha256': digest(root/'data/manifest.json'), 'source_sha256': digest(__file__),
                'clips': manifest['review_clips'], 'fixed_views': ['front', 'oblique'],
                'template': 'fixed canonical22; source rotations, no body mesh; feet red when contact label positive'}
    if (out/'protocol.json').exists():
        if read_authenticated(out/'protocol.json') != protocol:
            raise ValueError('Source-review protocol drift')
    else:
        authenticated_json(out/'protocol.json', protocol)
    results = []
    for clip in manifest['review_clips']:
        entry = entries[clip['key']]
        path = out/f'{clip["category"]}_{clip["key"]}_{clip["start"]}.mp4'
        if path.with_suffix('.json').exists():
            report = read_authenticated(path.with_suffix('.json'))
            if digest(path) != report['sha256']:
                raise ValueError('Source review artifact changed')
        else:
            if digest(entry['path']) != entry['sha256']:
                raise ValueError('Prepared source changed')
            with np.load(entry['path']) as value:
                sl = slice(clip['start'], clip['start']+clip['frames'])
                joints, contacts = value['joints'][sl], value['contacts'][sl]
            report = render_comparison(path, [('FineDance source', joints, contacts)], entry['wav_path'],
                                       clip['start']/FPS, entry['key']+' / '+clip['category'])
        results.append(report)
        print(json.dumps({'stage': 'source_review', 'key': clip['key'], 'path': str(path)}), flush=True)
    authenticated_json(out/'report.json', {'rows': results, 'protocol_sha256': digest(out/'protocol.json')})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    with gpu_lock(args.root):
        source_review(args.root)
