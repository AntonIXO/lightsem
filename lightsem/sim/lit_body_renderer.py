"""Opaque source-body geometry in the existing decoded-DMX dark stage.

The 22 joints are y-up metres. ``body_scale`` and ``body_offset`` are fixed
scene transforms: there is no camera following, recentering, time input, pose
fitting, beam re-aiming or brightness substitution. Bones are analytic capsules
and the head is a sphere. Body intersections run once per camera ray after a
bounding-box rejection, outside the stage/haze march loops.

The body uses the stage's existing cone, falloff, RGB and dimmer calculation.
A separately declared, constant neutral fill makes an unilluminated source
readable. Garland bulbs remain emissive objects, not invented floodlights.
Camera-ray occlusion is supported; cast shadows and interreflection are not.
"""
from __future__ import annotations

import math

import moderngl
import numpy as np

from .body_stage_renderer import BodyStageRenderer, MAX_BULBS, MAX_MOVING, stage_shader
from .renderer import RenderConfig
from .shaders import VERTEX_SHADER

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
# Joint-indexed capsule radii; a zero-length neck-to-head capsule is not needed:
# an additional sphere at joint 15 gives the head its fixed anatomical size.
BONE_RADII = np.array([.065, .065, .065, .070, .050, .050, .075, .043,
                      .043, .080, .040, .040, .052, .055, .055, .045,
                      .047, .047, .038, .038, .032, .032], dtype='f4')
HEAD_RADIUS = .105


def transform_body(body_pose, *, scale=1.0, offset=(0., 0., 0.)):
    """Apply only the declared fixed similarity; preserve source translation."""
    pose = np.asarray(body_pose, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)
    if pose.shape != (22, 3) or not np.isfinite(pose).all():
        raise ValueError('Body must contain 22 finite y-up XYZ joints')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Body scale must be finite and positive')
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Body offset must contain three finite coordinates')
    world = pose * scale + offset
    if not np.isfinite(world).all() or np.max(np.abs(world)) > np.finfo('f4').max:
        raise ValueError('Body transform exceeds finite shader coordinates')
    return world.astype('f4')


def ray_capsule_distance(origin, direction, a, b, radius):
    """CPU reference for geometry diagnostics; closest forward surface or inf.

    A normalized ray is required. Both roots and cap hemispheres matter when
    the camera starts inside a bone. Degenerate bones are full spheres.
    """
    ro, rd, pa, pb = (np.asarray(x, dtype=float) for x in (origin, direction, a, b))
    if any(x.shape != (3,) or not np.isfinite(x).all() for x in (ro, rd, pa, pb)):
        raise ValueError('Ray and capsule coordinates must be finite XYZ vectors')
    if not np.isclose(np.dot(rd, rd), 1., rtol=0, atol=1e-6):
        raise ValueError('Ray direction must be normalized')
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('Capsule radius must be positive')
    ba, oa = pb - pa, ro - pa
    baba, bard, baoa = np.dot(ba, ba), np.dot(ba, rd), np.dot(ba, oa)
    candidates = []
    if baba > 1e-10:
        aa = baba - bard * bard
        bb = baba * np.dot(rd, oa) - baoa * bard
        cc = baba * np.dot(oa, oa) - baoa * baoa - radius * radius * baba
        disc = bb * bb - aa * cc
        if aa > 1e-10 * baba and disc >= 0:
            for t in ((-bb - math.sqrt(disc)) / aa, (-bb + math.sqrt(disc)) / aa):
                y = baoa + t * bard
                if t > 1e-5 and 0 <= y <= baba:
                    candidates.append(t)
    for end, center in enumerate((pa, pb)):
        oc = ro - center
        sphere_b = np.dot(rd, oc)
        disc = sphere_b * sphere_b - np.dot(oc, oc) + radius * radius
        if disc < 0:
            continue
        for t in (-sphere_b - math.sqrt(disc), -sphere_b + math.sqrt(disc)):
            y = baoa + t * bard
            if t > 1e-5 and (baba <= 1e-10 or (y <= 0 if end == 0 else y >= baba)):
                candidates.append(t)
    return min(candidates, default=float('inf'))


BODY_GEOMETRY_GLSL = r'''
uniform vec3 u_body_joints[22];
uniform float u_body_radii[22];
uniform float u_body_head_radius;
uniform vec3 u_body_bounds_min;
uniform vec3 u_body_bounds_max;
uniform float u_body_neutral_fill;
uniform int u_body_diagnostic; // 0 scene, 1 visible-body mask, 2 direct body light
const int BODY_PARENTS[22] = int[22](-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19);

bool bodyBounds(vec3 ro, vec3 rd) {
    float nearT = 0.0, farT = 60.0;
    for (int axis = 0; axis < 3; axis++) {
        if (abs(rd[axis]) < 1e-8) {
            if (ro[axis] < u_body_bounds_min[axis] || ro[axis] > u_body_bounds_max[axis]) return false;
        } else {
            float a = (u_body_bounds_min[axis] - ro[axis]) / rd[axis];
            float b = (u_body_bounds_max[axis] - ro[axis]) / rd[axis];
            nearT = max(nearT, min(a,b));
            farT = min(farT, max(a,b));
        }
    }
    return farT >= nearT;
}

void acceptBodyHit(float t, vec3 normal, inout float nearest, inout vec3 hitNormal) {
    if (t > 1e-5 && t < nearest) {
        nearest = t;
        hitNormal = normal;
    }
}

void hitCapsule(vec3 ro, vec3 rd, vec3 pa, vec3 pb, float radius,
                inout float nearest, inout vec3 hitNormal) {
    vec3 ba = pb - pa, oa = ro - pa;
    float baba = dot(ba,ba), bard = dot(ba,rd), baoa = dot(ba,oa);
    if (baba > 1e-10) {
        float aa = baba - bard*bard;
        float bb = baba*dot(rd,oa) - baoa*bard;
        float cc = baba*dot(oa,oa) - baoa*baoa - radius*radius*baba;
        float disc = bb*bb - aa*cc;
        if (aa > 1e-10*baba && disc >= 0.0) {
            float root = sqrt(disc);
            for (int side = 0; side < 2; side++) {
                float t = (-bb + (side == 0 ? -root : root)) / aa;
                float y = baoa + t*bard;
                if (y >= 0.0 && y <= baba && t > 1e-5 && t < nearest) {
                    vec3 n = normalize(oa + rd*t - ba*(y/baba));
                    acceptBodyHit(t,n,nearest,hitNormal);
                }
            }
        }
    }
    for (int end = 0; end < 2; end++) {
        vec3 center = end == 0 ? pa : pb;
        vec3 oc = ro-center;
        float bb = dot(rd,oc);
        float disc = bb*bb-dot(oc,oc)+radius*radius;
        if (disc < 0.0) continue;
        float root = sqrt(disc);
        for (int side = 0; side < 2; side++) {
            float t = -bb + (side == 0 ? -root : root);
            float y = baoa + t*bard;
            bool cap = baba <= 1e-10 || (end == 0 ? y <= 0.0 : y >= baba);
            if (cap && t > 1e-5 && t < nearest) {
                acceptBodyHit(t,normalize(ro+rd*t-center),nearest,hitNormal);
            }
        }
    }
}

float intersectBody(vec3 ro, vec3 rd, out vec3 normal) {
    normal = vec3(0.0,1.0,0.0);
    float nearest = 1e20;
    if (!bodyBounds(ro,rd)) return nearest;
    for (int j = 1; j < 22; j++) {
        hitCapsule(ro,rd,u_body_joints[BODY_PARENTS[j]],u_body_joints[j],u_body_radii[j],nearest,normal);
    }
    hitCapsule(ro,rd,u_body_joints[15],u_body_joints[15],u_body_head_radius,nearest,normal);
    return nearest;
}
'''


def lit_body_shader():
    """Add analytic body surfaces; retain legacy decoded-DMX optical equations."""
    text = stage_shader()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Stage shader changed: ' + old[:65])
        text = text.replace(old, new)

    replace('// ---------- scene SDF ----------', BODY_GEOMETRY_GLSL + '\n// ---------- scene SDF ----------')
    replace('    if (tHit > 0.0) {\n        vec3 p = ro + rd * tHit;', '''
    vec3 bodyNormal;
    float bodyHit = intersectBody(ro,rd,bodyNormal);
    if (bodyHit < 60.0 && (tHit < 0.0 || bodyHit < tHit)) { tHit = bodyHit; mat = 3; }
    if (u_body_diagnostic == 1) {
        f_color = vec4(mat == 3 ? vec3(1.0) : vec3(0.0),1.0);
        return;
    }
    if (u_body_diagnostic == 2 && mat != 3) { f_color = vec4(0.0,0.0,0.0,1.0); return; }
    if (tHit > 0.0) {
        vec3 p = ro + rd * tHit;''')
    replace('        vec3 n = normalize(vec3(', '        vec3 n = mat == 3 ? bodyNormal : normalize(vec3(')
    replace('        vec3 albedo = (mat == 0) ?', '        vec3 albedo = mat == 3 ? vec3(0.68) : (mat == 0) ?')
    replace('                                     // washed the whole frame to flat grey)\n        for (int i = 0; i < MAX_LIGHTS; i++) {', '''
                                     // washed the whole frame to flat grey)
        if (mat == 3) {
            // Constant neutral presentation fill, independent of generated RGB/dimmer.
            surf = vec3(u_body_neutral_fill) * (0.70 + 0.30*max(n.y,0.0));
            if (u_body_diagnostic == 2) surf = vec3(0.0);
        }
        for (int i = 0; i < MAX_LIGHTS; i++) {''')
    replace('        col += surf;', '''
        col += surf;
        if (u_body_diagnostic == 2) {
            // Same exposure/tonemap, direct body illumination only: no haze or fill.
            if (u_fixed_exposure != 0) col *= u_exposure_scale;
            col = max(col,vec3(0.0));
            col = (col*(2.51*col+0.03))/(col*(2.43*col+0.59)+0.14);
            f_color = vec4(pow(clamp(col,0.0,1.0),vec3(0.85)),1.0);
            return;
        }''')
    # A body is opaque to visible fixtures as well as surface/haze intersections.
    replace('        if (proj <= 0.0) continue;',
            '        if (proj <= 0.0 || (mat == 3 && proj > tHit)) continue;')
    replace('proj > tHit + 0.08', 'proj > tHit + (mat == 3 ? 0.0 : 0.08)')
    return text


class LitBodyStageRenderer(BodyStageRenderer):
    """Actual stage controls and a metric, opaque source skeleton in one scene.

    Construct independent close/wide camera instances and supply the same pose
    and controls. Each render activates its own context/FBO, so alternating
    instances cannot accidentally draw through the other camera's framebuffer.
    """

    def __init__(self, patch, cfg: RenderConfig, *, body_offset=(0., 0., 0.),
                 body_scale=1.0, neutral_fill=.12):
        transform_body(np.zeros((22, 3)), scale=body_scale, offset=body_offset)
        if not math.isfinite(neutral_fill) or not 0 <= neutral_fill <= 1:
            raise ValueError('Neutral fill must be finite in [0,1]')
        self.cfg, self.patch = cfg, patch
        self.body_offset = tuple(float(x) for x in body_offset)
        self.body_scale, self.neutral_fill = float(body_scale), float(neutral_fill)
        self.moving = np.array([p['kind'] != 'bulb' for p in patch], dtype=bool)
        if self.moving.sum() > MAX_MOVING or (~self.moving).sum() > MAX_BULBS:
            raise ValueError('This stage renderer supports 16 moving outputs and 128 bulbs')
        if any(p['kind'] not in ('beam', 'laser', 'bulb') for p in patch):
            raise ValueError('Unknown virtual fixture kind')
        self.positions = np.asarray([p['position'] for p in patch], dtype='f4').reshape(-1, 3)
        if not np.isfinite(self.positions).all():
            raise ValueError('Nonfinite fixture position')
        self.ctx = moderngl.create_context(standalone=True, backend='egl', device_index=cfg.gpu_device)
        try:
            name = self.ctx.info['GL_RENDERER'].lower()
            if cfg.require_gpu and any(x in name for x in ('llvmpipe', 'softpipe', 'swiftshader', 'software')):
                raise RuntimeError('Hardware renderer required: ' + name)
            self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=lit_body_shader())
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
                'u_body_head_radius': HEAD_RADIUS*self.body_scale,
                'u_body_neutral_fill': self.neutral_fill, 'u_body_diagnostic': 0,
            }.items():
                self._require_uniform(key, value)
            kinds = np.array([0 if p['kind'] == 'beam' else 4 for p in patch if p['kind'] != 'bulb'], dtype='i4')
            self._array('u_lkind', kinds, MAX_MOVING)
            self._array('u_lpos', self.positions[self.moving], MAX_MOVING)
            self._array('u_lcos', np.full(len(kinds), math.cos(math.radians(3.)), dtype='f4'), MAX_MOVING)
            self._array('u_bulb_pos', self.positions[~self.moving], MAX_BULBS)
            self._array('u_body_radii', BONE_RADII*self.body_scale, 22)
        except BaseException:
            self.ctx.release()
            raise

    def body_world(self, body_pose):
        return transform_body(body_pose, scale=self.body_scale, offset=self.body_offset)

    def presentation_config(self):
        return {'body_scale': self.body_scale, 'body_offset': list(self.body_offset),
                'coordinate_system': 'source XYZ, y-up metres; fixed similarity only; no recentering',
                'neutral_fill_radiance': self.neutral_fill, 'neutral_fill_scope': 'constant presentation-only body fill',
                'bone_radii_m': (BONE_RADII*self.body_scale).tolist(),
                'head_radius_m': HEAD_RADIUS*self.body_scale,
                'shader_time': 0., 'cast_shadows': False, 'interreflection': False,
                'garland': 'emissive bulbs only; no invented body illumination'}

    def render_controls(self, directions, rgb, dimmer, body_pose, *, diagnostic='scene'):
        modes = {'scene': 0, 'body_mask': 1, 'direct_body_light': 2}
        if diagnostic not in modes:
            raise ValueError('Unknown body diagnostic mode')
        world = self.body_world(body_pose)
        margin = HEAD_RADIUS*self.body_scale
        with self.ctx:
            self.fbo.use()
            self._array('u_body_joints', world, 22)
            self._require_uniform('u_body_bounds_min', tuple(float(x) for x in world.min(0)-margin))
            self._require_uniform('u_body_bounds_max', tuple(float(x) for x in world.max(0)+margin))
            self._require_uniform('u_body_diagnostic', modes[diagnostic])
            return super().render_controls(directions, rgb, dimmer)

    def render_diagnostics(self, directions, rgb, dimmer, body_pose):
        """Four explicit controls; no GPU use until the caller invokes this."""
        actual = self.render_controls(directions, rgb, dimmer, body_pose)
        off = self.render_controls(directions, rgb, np.zeros_like(dimmer), body_pose)
        mask = self.render_controls(directions, rgb, dimmer, body_pose, diagnostic='body_mask')
        direct = self.render_controls(directions, rgb, dimmer, body_pose, diagnostic='direct_body_light')
        visible = mask[..., 0] > 127
        y, x = np.where(visible)
        pixels = int(visible.sum())
        metrics = {
            'body_visible_pixels': pixels, 'body_visible_fraction': float(visible.mean()),
            'body_bbox_xyxy': [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if pixels else None,
            'body_lights_off_mean_255': float(off[visible].mean()) if pixels else None,
            'body_actual_mean_255': float(actual[visible].mean()) if pixels else None,
            'body_direct_light_mean_255': float(direct[visible].mean()) if pixels else None,
            'body_direct_lit_pixels_above_1': int((visible & (direct.max(2) > 1)).sum()),
            'body_direct_lit_fraction_above_1': float((direct[visible].max(1) > 1).mean()) if pixels else None,
            'direct_measurement_scope': 'body surface only; excludes haze, neutral fill and bulb glow',
            'neutral_fill_radiance': self.neutral_fill,
        }
        return {'actual': actual, 'lights_off': off, 'body_mask': mask,
                'direct_body_light': direct, 'metrics': metrics}

    def close(self):
        self.ctx.release()
