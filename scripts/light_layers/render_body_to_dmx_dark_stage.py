#!/usr/bin/env python
"""Render body-to-DMX output through LightSem's canonical dark 3D stage.

This intentionally reuses BodyStageRenderer, the 63-fixture patch, the fixed
camera/configuration and the project encoder/audit helpers from the historical
body-transfer demo. The generated body/DMX remains diagnostic-only.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lightsem.conductor.body_transfer import decode
from lightsem.conductor.body_to_dmx import expand_canonical_angles
from lightsem.dmx import club_rig
from lightsem.dmx.decode import decode_stream
from lightsem.dance.pipeline import accelerator_lock
from lightsem.sim.body_stage_renderer import BodyStageRenderer
from scripts.motion.run_rhythm_companion import render_config, encode_command, audit_movie
from scripts.rendering.render_body_transfer import PARENTS


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def interp(t0, values, t):
    values = np.asarray(values)
    return np.stack([np.interp(t, t0, values[:, i]) for i in range(values.shape[1])], 1)


def interp_circular(t0, hue, t):
    s = interp(t0, np.sin(2 * np.pi * hue), t); c = interp(t0, np.cos(2 * np.pi * hue), t)
    return np.arctan2(s, c) / (2 * np.pi) % 1.


def hsv_rgb(h, s, v):
    h = (h % 1.) * 6.; i = np.floor(h).astype(int); f = h - i
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return np.stack([np.choose(i % 6, [v, q, p, p, t, v]), np.choose(i % 6, [t, v, v, q, p, p]), np.choose(i % 6, [p, p, t, v, v, q])], -1)


def angle_directions(pan, tilt):
    yaw = (pan - .5) * np.pi
    pitch = (tilt - .5) * (np.pi / 2.)
    return np.stack([np.sin(yaw) * np.cos(pitch), -np.cos(yaw) * np.cos(pitch), np.sin(pitch)], -1).astype(np.float32)


def decorate(rgb, pose, title, seconds):
    image = Image.fromarray(rgb); draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    draw.text((30, 22), title + ' · body-to-DMX · diagnostic only', font=font, fill=(221, 229, 241))
    draw.text((1700, 25), f'{seconds:06.2f} s', font=small, fill=(159, 174, 194))
    draw.rounded_rectangle((22, 726, 270, 1055), radius=10, fill=(5, 8, 12), outline=(27, 34, 44))
    draw.text((40, 740), 'Source body', font=small, fill=(151, 171, 191))
    pose = pose - pose[0] * np.array([1., 0., 1.])
    xy = np.column_stack((146 + (pose[:, 0] * .8 + pose[:, 2] * .6) * 112, 1030 - pose[:, 1] * 112))
    for j, parent in enumerate(PARENTS):
        if parent < 0: continue
        ink = (110, 164, 199) if j % 2 == 0 else (188, 140, 210)
        draw.line([tuple(xy[parent]), tuple(xy[j])], fill=ink, width=3)
    for x, y in xy: draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(205, 215, 228))
    return np.asarray(image)


def build_controls(demo: Path, patch: list[dict], t40: np.ndarray):
    with np.load(demo / 'demo.npz', allow_pickle=False) as z:
        t10, world10, dmx10, scene = z['time'].astype(float), z['world'].astype(np.float32), z['dmx'].astype(np.uint8), z['scene'].astype(np.float32)
    source_rig = club_rig(); decoded = decode_stream(source_rig, dmx10)
    source_heads = np.array([i for i, f in enumerate(source_rig.fixtures) if f.kind == 'moving_head'])
    pan6, tilt6 = decoded['pan'][:, source_heads], decoded['tilt'][:, source_heads]
    angles6 = np.stack([pan6, tilt6], -1)
    angles13_10 = expand_canonical_angles(angles6, 13)
    angles13 = np.stack([interp(t10, angles13_10[:, :, k], t40) for k in range(2)], -1)
    directions13 = angle_directions(angles13[:, :, 0], angles13[:, :, 1])
    h6, s6, v6 = decoded['hue'][:, source_heads], decoded['sat'][:, source_heads], decoded['val'][:, source_heads]
    h13 = np.stack([interp_circular(t10, h6[:, [j % 6]], t40)[:, 0] for j in range(13)], 1)
    s13 = np.stack([interp(t10, s6[:, [j % 6]], t40)[:, 0] for j in range(13)], 1)
    v13 = np.stack([interp(t10, v6[:, [j % 6]], t40)[:, 0] for j in range(13)], 1)
    dim13 = np.stack([interp(t10, decoded['dimmer'][:, [j % 6]], t40)[:, 0] for j in range(13)], 1)
    rgb13 = hsv_rgb(h13, np.clip(s13, 0, 1), np.clip(v13, 0, 1))
    scene_h = np.arctan2(scene[:, 1], scene[:, 0]) / (2 * np.pi) % 1.; scene_v = np.clip(scene[:, 3], 0, 1)
    bulb_h = interp_circular(t10, scene_h[:, None], t40)[:, 0]; bulb_v = np.interp(t40, t10, scene_v)
    directions = np.zeros((len(t40), len(patch), 3), np.float32); rgb = np.zeros((len(t40), len(patch), 3), np.float32); dim = np.zeros((len(t40), len(patch)), np.float32)
    moving = [i for i, p in enumerate(patch) if p['kind'] != 'bulb']
    bulbs = [i for i, p in enumerate(patch) if p['kind'] == 'bulb']
    directions[:, moving] = directions13
    rgb[:, moving] = rgb13
    dim[:, moving] = dim13
    bulb_rgb = hsv_rgb(np.mod(bulb_h[:, None] + np.linspace(-.06, .06, len(bulbs))[None], 1.), np.full((len(t40), len(bulbs)), .9), np.ones((len(t40), len(bulbs))))
    rgb[:, bulbs] = bulb_rgb
    dim[:, bulbs] = bulb_v[:, None] * .55
    body40 = np.stack([np.stack([np.interp(t40, t10, world10[:, j, k]) for k in range(3)], 1) for j in range(22)], 1)
    return {'directions': directions, 'rgb': rgb, 'dimmer': dim, 'body': body40, 'dmx_sha256': hashlib.sha256(dmx10.tobytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('stage', choices=('smoke', 'render')); ap.add_argument('--demo', type=Path, required=True); ap.add_argument('--audio', type=Path, required=True); ap.add_argument('--patch', type=Path, default=Path('/root/lightsem/runs/body_transfer_20260930_v1/shpongle/patch.json')); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--full', action='store_true'); args = ap.parse_args()
    demo, audio, patch_path, root = args.demo.resolve(), args.audio.resolve(), args.patch.resolve(), args.out.resolve()
    if root.exists() and any(root.iterdir()): raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True); meta = json.loads((demo / 'DEMO.json').read_text());
    if meta.get('status') != 'diagnostic_only' or not meta.get('raw_body', {}).get('finite') or not meta.get('finite_dmx'): raise RuntimeError('Demo safety/provenance gate failed')
    patch = json.loads(patch_path.read_text()); shutil.copy2(patch_path, root / 'patch.json');
    # Canonical stage cadence is 40 Hz; source model DMX is retained in demo.npz.
    with np.load(demo / 'demo.npz', allow_pickle=False) as z: duration = float(z['time'][-1]); n10 = len(z['time'])
    n40 = int(round(duration * 40.)); t40 = np.arange(n40) / 40.; controls = build_controls(demo, patch, t40)
    np.savez_compressed(root / 'controls40.npz', time=t40, directions=controls['directions'], rgb=controls['rgb'], dimmer=controls['dimmer'])
    cfg = render_config(); cfg.camera_scale = .62
    smoke_ticks = [0, min(n40 - 1, 2400), min(n40 - 1, 4800), n40 - 1]
    if args.stage == 'smoke':
        with accelerator_lock():
            renderer = BodyStageRenderer(patch, cfg)
            try:
                rows=[]
                for tick in smoke_ticks:
                    image = renderer.render_controls(controls['directions'][tick], controls['rgb'][tick], controls['dimmer'][tick]); path = root / f'smoke_{tick:06d}.png'; Image.fromarray(decorate(image, controls['body'][min(tick, len(controls['body'])-1)], '3b4ab canonical dark stage', tick/40)).save(path); rows.append({'tick':tick,'path':str(path),'sha256':sha(path),'max_pixel':int(image.max())})
            finally: renderer.ctx.release()
        (root / 'SMOKE.json').write_text(json.dumps({'status':'passed','renderer':'BodyStageRenderer','frames':rows,'source_dmx_sha256':controls['dmx_sha256']},indent=2)+'\n'); print(json.dumps({'status':'smoke_passed','out':str(root)},indent=2)); return
    from scripts.motion.run_rhythm_companion import encode_command
    n30 = int(round(duration * 30.)); indices = np.rint(np.arange(n30) * 40 / 30).astype(int); indices = np.clip(indices, 0, n40 - 1)
    movie = root / 'body_to_dmx_canonical_dark_stage.mp4'; cmd = encode_command(movie, {'audio':str(audio)}, cfg.width, cfg.height, 0, n30); started=time.monotonic()
    with accelerator_lock():
        renderer=BodyStageRenderer(patch,cfg); encoder=None
        try:
            with (root/'encode.log').open('xb') as err:
                encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
                for frame, tick in enumerate(indices):
                    image=renderer.render_controls(controls['directions'][tick],controls['rgb'][tick],controls['dimmer'][tick]); image=decorate(image,controls['body'][tick], '3b4ab canonical dark stage', frame/30); encoder.stdin.write(image.tobytes())
                    if frame in (0, n30//2, n30-1): Image.fromarray(image).save(root/f'frame_{frame:06d}.png')
                    if frame % 600 == 0: print('CANONICAL_RENDER',frame,n30,flush=True)
                encoder.stdin.close()
                if encoder.wait()!=0: raise RuntimeError((root/'encode.log').read_text())
        finally:
            if encoder is not None and encoder.poll() is None: encoder.kill(); encoder.wait()
            renderer.ctx.release()
    render_report={'frames':n30,'source_control_frames_40hz':n40,'duration_s':n30/30,'video':str(movie),'video_sha256':sha(movie),'patch_sha256':sha(patch_path),'audio_sha256':sha(audio),'source_dmx_sha256':controls['dmx_sha256'],'renderer':'BodyStageRenderer','config':{'width':cfg.width,'height':cfg.height,'fps':cfg.fps,'camera_scale':cfg.camera_scale},'elapsed_s':time.monotonic()-started,'status':'diagnostic_only','reason':meta['reason']}
    movie.with_name(movie.stem+'_render.json').write_text(json.dumps(render_report,indent=2)+'\n'); audit=audit_movie(root,movie); audit['status']='diagnostic_only'; audit['source_dmx_sha256']=controls['dmx_sha256']; (root/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n'); (root/'DELIVERY.json').write_text(json.dumps({'status':'diagnostic_only','movie':render_report,'audit':audit},indent=2)+'\n'); print(json.dumps({'movie':str(movie),'audit':str(root/'AUDIT.json'),'status':'diagnostic_only'},indent=2))


if __name__ == '__main__': main()
