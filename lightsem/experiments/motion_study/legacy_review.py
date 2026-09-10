"""Rerender frozen legacy predictions with raw video and separate uncertainty.

The archived script, protocol, arrays and movies are never edited. The source
confidence mask affects only reference ink opacity; predictions use finiteness.
"""
from pathlib import Path
import argparse
import json
import subprocess
import time

import numpy as np

from . import ROOT
from ..night_common import ROOT as NIGHT, STAGE
from ..stage11_common import authenticated_json, read_authenticated, digest, gpu_lock
from ..stage11_data import load_manifest

PANELS = [
    ('Detected pose: gray = uncertain', 'descriptor256', 'truth'),
    ('Descriptors, decoder width 128', 'descriptor128', 'full'),
    ('Descriptors, decoder width 256', 'descriptor256', 'full'),
    ('Pose-conditioned code 24', 'pose_code24', 'full'),
    ('Audio base to pose', 'audio_base', 'full'),
    ('Audio Fourier to pose', 'audio_fourier', 'full'),
    ('Audio MuQ to pose', 'audio_muq', 'full'),
    ('Incumbent audio to descriptors to pose', 'descriptor256', 'incumbent_audio_to_descriptors')]
VERTEX = '#version 330\nin vec2 pos; void main(){gl_Position=vec4(pos,0,1);}'
FRAGMENT = '''#version 330
uniform vec2 joints[136]; uniform float confidence[136]; uniform sampler2D labels;
uniform sampler2D source_video; out vec4 color;
const ivec2 edges[18]=ivec2[](ivec2(0,1),ivec2(0,2),ivec2(1,3),ivec2(2,4),ivec2(0,5),ivec2(0,6),ivec2(5,6),ivec2(5,7),ivec2(7,9),ivec2(6,8),ivec2(8,10),ivec2(5,11),ivec2(6,12),ivec2(11,12),ivec2(11,13),ivec2(13,15),ivec2(12,14),ivec2(14,16));
float distanceLine(vec2 p,vec2 a,vec2 b){vec2 d=b-a;return length(p-a-clamp(dot(p-a,d)/max(dot(d,d),1e-9),0.,1.)*d);}
void main(){
 vec2 px=gl_FragCoord.xy;int col=int(px.x/640.);int row=2-int(px.y/360.);int panel=row*3+col;
 vec2 q=mod(px,vec2(640,360));vec2 world=vec2((q.x-320.)/64.,(q.y-26.)/48.);
 vec3 result=vec3(.985);
 if(panel==0){result=texture(source_video,vec2(q.x/640.,1.-q.y/360.)).rgb;}
 else {
  if(abs(world.y)<.012)result=vec3(.85);
  int base=(panel-1)*17;
  for(int i=0;i<18;i++){
   int a=base+edges[i].x;int b=base+edges[i].y;float c=min(confidence[a],confidence[b]);
   if(c<0.)continue;
   float alpha=1.-smoothstep(.022,.045,distanceLine(world,joints[a],joints[b]));
   vec3 ink=(panel==1)?mix(vec3(.70),vec3(.12),c):vec3(.10,.32,.49);
   result=mix(result,ink,alpha);
  }
  for(int j=0;j<17;j++){
   int k=base+j;if(confidence[k]<0.)continue;
   float alpha=1.-smoothstep(.03,.052,length(world-joints[k]));
   vec3 ink=(panel==1)?mix(vec3(.70),vec3(.12),confidence[k]):vec3(.10,.32,.49);
   result=mix(result,ink,alpha);
  }
 }
 if(q.x<1.||q.y<1.)result=vec3(.7);
 vec4 label=texture(labels,vec2(px.x/1920.,1.-px.y/1080.));
 color=vec4(mix(result,label.rgb,label.a),1.);
}'''


def visibility(pose, source_mask, is_reference):
    finite = np.isfinite(pose).all(-1)
    confidence = source_mask.astype(np.float32) if is_reference else np.ones_like(source_mask, dtype=np.float32)
    return np.where(finite, confidence, -1).astype(np.float32)


def run(root=ROOT):
    import moderngl
    from PIL import Image, ImageDraw, ImageFont
    root = Path(root)
    out = root/'delivery/legacy_pose'
    out.mkdir(parents=True, exist_ok=True)
    previous = read_authenticated(NIGHT/'delivery/pose/protocol.json')
    for path, sha in previous['files'].items():
        if digest(path) != sha:
            raise ValueError('Archived diagnostic input changed: ' + path)
    records = {r['recording_id']: r for r in load_manifest(STAGE)['recordings']}
    protocol = {'source_protocol_sha256': digest(NIGHT/'delivery/pose/protocol.json'),
                'source_sha256': digest(__file__), 'panels': PANELS, 'fps': 30,
                'frames': 600, 'width': 1920, 'height': 1080,
                'predicted_visibility': 'finite coordinates; never source confidence',
                'reference_uncertainty': 'uncertain finite joints/edges gray; missing coordinates hidden',
                'camera': 'fixed across all frames/panels; no per-frame centering or scaling'}
    if (out/'protocol.json').exists():
        if read_authenticated(out/'protocol.json') != protocol:
            raise ValueError('Review protocol changed; use new output root')
    else:
        authenticated_json(out/'protocol.json', protocol)
    ctx = moderngl.create_context(standalone=True, backend='egl', device_index=0)
    if 'nvidia' not in ctx.info['GL_RENDERER'].lower():
        raise RuntimeError('NVIDIA raster required')
    program = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT)
    vbo = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes())
    vao = ctx.simple_vertex_array(program, vbo, 'pos')
    fbo = ctx.simple_framebuffer((1920, 1080), components=4)
    fbo.use()
    atlas = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    font = ImageFont.truetype('/usr/share/fonts/TTF/DejaVuSans.ttf', 18)
    for i, title in enumerate(['Source recording'] + [p[0] for p in PANELS]):
        x, y = 640*(i % 3), 360*(i//3)
        draw.rectangle((x+1, y+1, x+639, y+33), fill=(250, 250, 250, 245))
        draw.text((x+12, y+8), title, font=font, fill=(25, 25, 25, 255))
    labels = ctx.texture(atlas.size, 4, atlas.tobytes())
    labels.use(0)
    program['labels'].value = 0
    source_texture = ctx.texture((640, 360), 3)
    source_texture.use(1)
    program['source_video'].value = 1
    results = []
    for row in previous['selected']:
        entry = row['source']['entry']
        key = entry['key']
        report = out/f'{key}.json'
        if report.exists():
            result = read_authenticated(report)
            if digest(result['path']) != result['sha256']:
                raise ValueError('Rendered review drift')
            results.append(result)
            continue
        record = records[entry['recording_id']]
        if digest(record['source_path']) != record['source_sha256']:
            raise ValueError('Raw source recording changed')
        poses, masks = [], []
        for _, probe, mode in PANELS:
            with np.load(row['panel_paths'][probe]) as value:
                pose = value[mode][:800].reshape(800, 17, 2)
                mask = value['mask'][:800].reshape(800, 17, 2).all(-1)
            masks.append(visibility(pose, mask, mode == 'truth'))
            poses.append(np.nan_to_num(pose))
        poses, masks = np.stack(poses, 1), np.stack(masks, 1)
        offset = entry['offset_s'] + row['source']['start_frame']/40
        source_command = ['ffmpeg', '-v', 'error', '-nostdin', '-ss', str(offset), '-i', record['source_path'],
                          '-an', '-t', '20', '-vf', 'fps=30,scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        video = out/f'{key}.mp4'
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', '1920x1080', '-r', '30', '-i', 'pipe:0', '-ss', str(offset), '-i', record['wav_path'],
                   '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'av1_nvenc', '-preset', 'p5', '-cq', '25', '-b:v', '0',
                   '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-t', '20', '-shortest', str(video)]
        source = subprocess.Popen(source_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        started = time.monotonic()
        try:
            for frame in range(600):
                pixels = source.stdout.read(640*360*3)
                if len(pixels) != 640*360*3:
                    raise RuntimeError('Raw source ended before the frozen excerpt')
                source_texture.write(pixels)
                index = round(frame*40/30)
                program['joints'].write(poses[index].astype(np.float32).tobytes())
                program['confidence'].write(masks[index].tobytes())
                vao.render(moderngl.TRIANGLES)
                rgb = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(1080, 1920, 3)[::-1]
                encoder.stdin.write(rgb.tobytes())
                if frame == 300:
                    Image.fromarray(rgb).save(out/f'{key}_10s.png')
        finally:
            source.stdout.close()
            encoder.stdin.close()
            source_error, encoder_error = source.stderr.read().decode(), encoder.stderr.read().decode()
            source_code, encoder_code = source.wait(), encoder.wait()
        if source_code or encoder_code:
            raise RuntimeError(source_error + '\n' + encoder_error)
        result = {'key': key, 'path': str(video), 'sha256': digest(video), 'source_path': record['source_path'],
                  'source_sha256': record['source_sha256'], 'audio_offset_s': offset, 'frames': 600,
                  'protocol_sha256': digest(out/'protocol.json'), 'seconds': time.monotonic()-started,
                  'renderer': ctx.info['GL_RENDERER'], 'commands': [source_command, command]}
        authenticated_json(report, result)
        results.append(result)
        print(json.dumps({'stage': 'legacy_review', 'key': key, 'seconds': result['seconds']}), flush=True)
    ctx.release()
    authenticated_json(out/'report.json', {'rows': results, 'protocol_sha256': digest(out/'protocol.json')})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    with gpu_lock(args.root):
        run(args.root)
