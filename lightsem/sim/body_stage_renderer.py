"""The existing dark volumetric stage with decoded body-transfer controls.

The stage geometry, haze, beam optics and tonemap come from the old shader.
Separate emissive bulbs avoid turning a garland into fifty floodlights. Thin
laser segments are integrated analytically to avoid raymarch sampling flicker.
All time-dependent directions, colors and levels come from supplied DMX.
"""
from __future__ import annotations

import math
import moderngl
import numpy as np

from .renderer import RenderConfig, StageRenderer
from .shaders import FRAGMENT_SHADER, VERTEX_SHADER

MAX_MOVING = 16
MAX_BULBS = 128


def stage_shader():
    text = FRAGMENT_SHADER
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Legacy scene shader changed: ' + old[:50])
        text = text.replace(old, new)
    replace('// ---------- scene SDF ----------', '''
uniform int u_n_bulbs;
uniform vec3 u_bulb_pos[128];
uniform vec3 u_bulb_col[128];
uniform float u_bulb_int[128];
uniform float u_laser_length[MAX_LIGHTS];

// ---------- scene SDF ----------''')
    replace('if (u_lkind[i] == 2) cone = 1.0;', '''
    // A laser's narrow surface spot has fixed optics, never a clock pattern.
    if (u_lkind[i] == 4) {
        float radial = length(toP - u_ldir[i] * dot(toP, u_ldir[i]));
        cone = cosA > 0.0 ? exp(-0.5 * radial * radial / (0.035 * 0.035)) * 20.0 : 0.0;
    }
    if (u_lkind[i] == 2) cone = 1.0;''')
    replace('vec3 lc = lightAt(i, p, a);\n            if (a <= 0.0) continue;\n            scatter',
            'if (u_lkind[i] == 4) continue; // integrated below, not sparsely sampled\n            vec3 lc = lightAt(i, p, a);\n            if (a <= 0.0) continue;\n            scatter')
    replace('col += scatter * 3.2 * norm;', '''
    // Analytic integral of a finite Gaussian laser shaft along the camera ray.
    // The radius/gain are fixed renderer optics shared by both songs.
    for (int i = 0; i < MAX_LIGHTS; i++) {
        if (i >= u_n_lights) break;
        if (u_lkind[i] != 4) continue;
        vec3 w = ro - u_lpos[i];
        float b = dot(rd, u_ldir[i]);
        float a = dot(rd, w);
        float c = dot(u_ldir[i], w);
        float denom = max(1.0 - b*b, 0.0001);
        float along = clamp((c - b*a) / denom, 0.0, u_laser_length[i]);
        vec3 center = u_lpos[i] + along * u_ldir[i];
        float cameraAlong = dot(center - ro, rd);
        if (cameraAlong <= 0.0 || cameraAlong > tMax) continue;
        float radial = length(ro + cameraAlong * rd - center);
        float gaussian = exp(-0.5 * radial*radial / (0.035 * 0.035));
        float integral = 2.506628 * 0.035 / sqrt(denom);
        float dens = u_haze * exp(-max(center.y, 0.0) * 0.16);
        dens *= 0.85 + 0.15 * sin(center.x * 0.4 + center.z * 0.3);
        float fall = 1.0 / (1.0 + 0.06 * along * along);
        scatter += u_lcol[i] * u_lint[i] * gaussian * integral * dens * fall * 20.0;
    }
    col += scatter * 3.2 * norm;''')
    replace('// Filmic-ish tonemap.', '''
    // Individual emissive garland lamps, with a small fixed optical halo.
    // They do not illuminate the whole haze volume as moving heads would.
    for (int i = 0; i < 128; i++) {
        if (i >= u_n_bulbs) break;
        vec3 toL = u_bulb_pos[i] - ro;
        float proj = dot(toL, rd);
        if (proj <= 0.0 || (tHit > 0.0 && proj > tHit + 0.08)) continue;
        float d = length(toL - rd * proj);
        float glow = exp(-d*d*400.0) * 0.85 + exp(-d*d*70.0) * 0.06;
        col += u_bulb_col[i] * u_bulb_int[i] * glow;
    }

    // Filmic-ish tonemap.''')
    return text


class BodyStageRenderer(StageRenderer):
    """Read decoded virtual outputs without mapping them onto the old 14 rig slots."""
    def __init__(self, patch, cfg: RenderConfig):
        self.cfg = cfg
        self.patch = patch
        self.moving = np.array([p['kind'] != 'bulb' for p in patch])
        if sum(self.moving) > MAX_MOVING or sum(~self.moving) > MAX_BULBS:
            raise ValueError('This stage renderer supports 16 moving outputs and 128 bulbs')
        if any(p['kind'] not in ('beam', 'laser', 'bulb') for p in patch):
            raise ValueError('Unknown virtual fixture kind')
        self.positions = np.array([p['position'] for p in patch], dtype='f4')
        self.ctx = moderngl.create_context(standalone=True, backend='egl', device_index=cfg.gpu_device)
        name = self.ctx.info['GL_RENDERER'].lower()
        if cfg.require_gpu and any(x in name for x in ('llvmpipe', 'softpipe', 'swiftshader', 'software')):
            self.ctx.release()
            raise RuntimeError('Hardware renderer required: ' + name)
        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=stage_shader())
        self.vbo = self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype='f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, 'in_pos')
        self.fbo = self.ctx.simple_framebuffer((cfg.width, cfg.height), components=4)
        self.fbo.use()
        for key, value in {
            'u_res': (float(cfg.width), float(cfg.height)), 'u_time': 0.,
            'u_cam_pos': cfg.cam_pos, 'u_cam_target': cfg.cam_target,
            'u_camera_scale': cfg.camera_scale, 'u_haze': cfg.haze,
            'u_fixed_exposure': 1, 'u_exposure_scale': cfg.fixed_exposure_scale,
            'u_beam_edge_profile': 1, 'u_beam_edge_softness_deg': cfg.beam_edge_softness_deg,
            'u_beam_flux_reference_deg': cfg.beam_flux_reference_deg,
            'u_n_lights': int(self.moving.sum()), 'u_n_bulbs': int((~self.moving).sum()),
        }.items():
            self._require_uniform(key, value)
        kinds = np.array([0 if p['kind'] == 'beam' else 4 for p in patch if p['kind'] != 'bulb'], dtype='i4')
        self._array('u_lkind', kinds, MAX_MOVING)
        self._array('u_lpos', self.positions[self.moving], MAX_MOVING)
        self._array('u_lcos', np.full(len(kinds), math.cos(math.radians(3.)), dtype='f4'), MAX_MOVING)
        self._array('u_bulb_pos', self.positions[~self.moving], MAX_BULBS)

    def _array(self, name, values, maximum):
        values = np.asarray(values)
        padded = np.zeros((maximum,) + values.shape[1:], dtype=values.dtype)
        padded[:len(values)] = values
        member = self.prog.get(name, None)
        if member is None:
            raise RuntimeError('Missing shader array: ' + name)
        member.write(padded.tobytes())

    def render_controls(self, directions, rgb, dimmer):
        directions, rgb, dimmer = (np.asarray(x, dtype='f4') for x in (directions, rgb, dimmer))
        n = len(self.patch)
        if directions.shape != (n, 3) or rgb.shape != (n, 3) or dimmer.shape != (n,):
            raise ValueError('Decoded control shape does not match patch')
        if not all(np.isfinite(x).all() for x in (directions, rgb, dimmer)):
            raise ValueError('Nonfinite decoded control')
        moving_dirs = directions[self.moving]
        origins = self.positions[self.moving]
        lengths = np.full(len(origins), 45., dtype='f4')
        for axis, plane in ((1, 0.), (2, -6.)):
            with np.errstate(divide='ignore', invalid='ignore'):
                distance = (plane - origins[:, axis]) / moving_dirs[:, axis]
            lengths = np.minimum(lengths, np.where(distance > 0, distance, 45.))
        self._array('u_laser_length', lengths.astype('f4'), MAX_MOVING)
        self._array('u_ldir', moving_dirs, MAX_MOVING)
        self._array('u_lcol', rgb[self.moving], MAX_MOVING)
        self._array('u_lint', dimmer[self.moving] * self.cfg.intensity_gain, MAX_MOVING)
        self._array('u_bulb_col', rgb[~self.moving], MAX_BULBS)
        self._array('u_bulb_int', dimmer[~self.moving] * self.cfg.intensity_gain, MAX_BULBS)
        self.ctx.clear(0., 0., 0., 1.)
        self.vao.render(moderngl.TRIANGLES)
        data = self.fbo.read(components=3, alignment=1)
        return np.flipud(np.frombuffer(data, dtype=np.uint8).reshape(self.cfg.height, self.cfg.width, 3))
