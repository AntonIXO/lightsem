"""GLSL shaders for the volumetric stage renderer.

The whole stage is drawn by RAYMARCHING a full-screen quad: for each pixel we
march a ray from the camera through the scene volume, accumulating

  * in-scattering from every light cone it passes through (this is what makes
    a beam visible in haze — the visual core of a lighting sim), and
  * surface hits against the floor, back wall and truss.

A cone is evaluated analytically per sample: the sample's angle to the
fixture axis is compared against the fixture's beam half-angle, with a soft
edge, inverse-square-ish distance falloff and a haze density term.
"""

VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330

#define MAX_LIGHTS 16

in vec2 v_uv;
out vec4 f_color;

uniform vec2  u_res;
uniform float u_time;
uniform vec3  u_cam_pos;
uniform vec3  u_cam_target;
uniform float u_camera_scale;
uniform int   u_beam_edge_profile;  // 0 legacy, 1 angular (moving heads only)
uniform float u_beam_edge_softness_deg;
uniform float u_beam_flux_reference_deg; // 0 keeps peak; >0 matches a legacy cone integral
uniform float u_haze;          // global haze density
uniform int   u_n_lights;
uniform int   u_fixed_exposure;
uniform float u_exposure_scale;

uniform vec3  u_lpos[MAX_LIGHTS];    // fixture position
uniform vec3  u_ldir[MAX_LIGHTS];    // normalised beam axis
uniform vec3  u_lcol[MAX_LIGHTS];    // rgb * intensity
uniform float u_lcos[MAX_LIGHTS];    // cos(beam half-angle)
uniform float u_lint[MAX_LIGHTS];    // 0..1 dimmer
uniform int   u_lkind[MAX_LIGHTS];   // 0 beam, 1 wash, 2 blinder, 3 projector

// ---------- scene SDF ----------
// Stage box: floor y=0, back wall z=-6, side walls +-9, ceiling y=8.
float sdFloor(vec3 p)    { return p.y; }
float sdBackWall(vec3 p) { return p.z + 6.0; }

float sdBox(vec3 p, vec3 b) {
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// Truss: horizontal bar at y=6.3 plus two vertical uprights.
float sdTruss(vec3 p) {
    float bar = sdBox(p - vec3(0.0, 6.3, -1.0), vec3(6.2, 0.16, 0.16));
    float l   = sdBox(p - vec3(-6.0, 3.2, -1.0), vec3(0.14, 3.2, 0.14));
    float r   = sdBox(p - vec3( 6.0, 3.2, -1.0), vec3(0.14, 3.2, 0.14));
    return min(bar, min(l, r));
}

float sdScene(vec3 p, out int mat) {
    // Bound the room. Infinite floor/back-wall planes filled the entire frame
    // with ambient grey and buried the beams -- the first render was a grey
    // fog with no visible cones. Keep surfaces finite so most of the screen
    // is BLACK and the volumetric scattering can read against it.
    float d = 1e9; mat = 0;

    if (abs(p.x) < 11.0 && p.z > -7.0 && p.z < 9.0) {
        float f = sdFloor(p);
        if (f < d) { d = f; mat = 0; }
    }
    if (abs(p.x) < 9.0 && p.y < 9.0) {
        float w = sdBackWall(p);
        if (w < d) { d = w; mat = 1; }
    }
    float t = sdTruss(p);      if (t < d) { d = t; mat = 2; }
    return d;
}

// Procedural gobo pattern for the projector, sampled in wall space.
float gobo(vec2 uv, int idx, float t) {
    if (idx <= 0) return 1.0;
    if (idx == 1) {                       // concentric rings
        float r = length(uv);
        return 0.5 + 0.5 * sin(18.0 * r - 2.0 * t);
    }
    if (idx == 2) {                       // radial spokes
        float a = atan(uv.y, uv.x);
        return 0.5 + 0.5 * sin(8.0 * a + t);
    }
    if (idx == 3) {                       // grid
        vec2 g = abs(fract(uv * 4.0 + 0.5) - 0.5);
        return smoothstep(0.02, 0.12, min(g.x, g.y));
    }
    // idx >= 4: rotating starburst
    float a = atan(uv.y, uv.x);
    float r = length(uv);
    return 0.5 + 0.5 * sin(12.0 * a + 6.0 * r - 3.0 * t);
}

// Cone in-scattering contribution at point p for light i.
vec3 lightAt(int i, vec3 p, out float atten) {
    vec3  toP  = p - u_lpos[i];
    float dist = length(toP);
    atten = 0.0;
    if (dist < 1e-4) return vec3(0.0);
    vec3 dir = toP / dist;

    float cosA = dot(dir, normalize(u_ldir[i]));
    float edge = u_lcos[i];

    // Soft cone edge; wash fixtures get a much softer falloff.
    // Tight edges are what make a beam read as a SHAFT rather than a glow.
    float softness = (u_lkind[i] == 1) ? 0.16 : 0.018;
    float cone = smoothstep(edge - softness, edge + softness * 0.4, cosA);
    // Opt-in angular edge. Leave the legacy expression above unchanged;
    // washes, blinders and projectors retain their original optics.
    if (u_beam_edge_profile == 1 && u_lkind[i] == 0) {
        float nominal_half = acos(clamp(edge, -1.0, 1.0));
        float softness_rad = radians(u_beam_edge_softness_deg);
        float outer_cos = cos(nominal_half + softness_rad);
        float inner_cos = cos(max(nominal_half - softness_rad, 0.0));
        cone = smoothstep(outer_cos, inner_cos, clamp(cosA, -1.0, 1.0));
        if (u_beam_flux_reference_deg > 0.0) {
            // Integral over cos(theta); 2*pi cancels in the ratio. Preserve
            // integrated cone weight, NOT peak radiance. Bulb glow below is
            // deliberately unchanged. This is a renderer model, not lumens.
            float ref_edge = cos(radians(u_beam_flux_reference_deg * 0.5));
            float lower = ref_edge - 0.018;
            float upper = ref_edge + 0.018 * 0.4;
            float w = upper - lower;
            float q = clamp((1.0 - lower) / w, 0.0, 1.0);
            float ref_integral = w * (q*q*q - 0.5*q*q*q*q) + max(1.0-upper, 0.0);
            float angular_integral = 1.0 - 0.5*(inner_cos + outer_cos);
            cone *= ref_integral / max(angular_integral, 1e-8);
        }
    }
    if (u_lkind[i] == 2) cone = 1.0;   // blinders are omni-ish flood

    float fall = 1.0 / (1.0 + 0.06 * dist * dist);
    // Blinders/strobes are floods, but giving them a TRUE omni response in
    // the volumetric pass filled the whole room with milky fog and crushed
    // the beam contrast. Restrict them to a wide forward lobe instead.
    if (u_lkind[i] == 2) {
        cone = smoothstep(-0.15, 0.55, cosA);
        fall = 1.0 / (1.0 + 0.14 * dist * dist);
    }
    atten = cone * fall * u_lint[i];
    return u_lcol[i] * atten;
}

void main() {
    vec2 uv = (gl_FragCoord.xy / u_res) * 2.0 - 1.0;
    uv.x *= u_res.x / u_res.y;

    // Camera basis.
    vec3 fwd   = normalize(u_cam_target - u_cam_pos);
    vec3 right = normalize(cross(fwd, vec3(0.0, 1.0, 0.0)));
    vec3 up    = cross(right, fwd);
    vec3 rd    = normalize(fwd + uv.x * right * u_camera_scale + uv.y * up * u_camera_scale);
    vec3 ro    = u_cam_pos;

    vec3 col = vec3(0.0);

    // ---------- surface pass: sphere-trace to the nearest solid ----------
    float t = 0.0;
    float tHit = -1.0;
    int   mat  = -1;
    for (int s = 0; s < 96; s++) {
        vec3 p = ro + rd * t;
        int m;
        float d = sdScene(p, m);
        if (d < 0.01) { tHit = t; mat = m; break; }
        t += max(d * 0.9, 0.02);
        if (t > 60.0) break;
    }

    if (tHit > 0.0) {
        vec3 p = ro + rd * tHit;
        // Normal by central differences.
        int mm;
        vec2 e = vec2(0.002, 0.0);
        vec3 n = normalize(vec3(
            sdScene(p + e.xyy, mm) - sdScene(p - e.xyy, mm),
            sdScene(p + e.yxy, mm) - sdScene(p - e.yxy, mm),
            sdScene(p + e.yyx, mm) - sdScene(p - e.yyx, mm)));

        vec3 albedo = (mat == 0) ? vec3(0.055, 0.055, 0.062)
                    : (mat == 1) ? vec3(0.075, 0.075, 0.085)
                                 : vec3(0.12, 0.12, 0.13);

        vec3 surf = albedo * 0.02;   // ambient (kept very low: high ambient
                                     // washed the whole frame to flat grey)
        for (int i = 0; i < MAX_LIGHTS; i++) {
            if (i >= u_n_lights) break;
            float a;
            vec3 lc = lightAt(i, p, a);
            if (a <= 0.0) continue;
            vec3 ldir = normalize(u_lpos[i] - p);
            float ndl = max(dot(n, ldir), 0.0);
            vec3 contrib = lc * ndl * albedo * 5.0;
            // Projector paints a gobo texture on the back wall.
            if (u_lkind[i] == 3 && mat == 1) {
                float g = gobo(p.xy * vec2(0.22, 0.22) - vec2(0.0, 0.6),
                               4, u_time);
                contrib *= (0.35 + 1.65 * g);
            }
            surf += contrib;
        }
        col += surf;
    }

    // ---------- volumetric pass: march the haze ----------
    float tMax = (tHit > 0.0) ? min(tHit, 45.0) : 45.0;
    const int STEPS = 64;
    float dt = tMax / float(STEPS);
    // Dithered start to break up banding.
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    float tv = jitter * dt;

    vec3 scatter = vec3(0.0);
    for (int s = 0; s < STEPS; s++) {
        vec3 p = ro + rd * tv;
        tv += dt;
        if (tv > tMax) break;
        // Haze is denser low over the stage, with slow drift.
        float dens = u_haze * exp(-max(p.y, 0.0) * 0.16);
        dens *= 0.85 + 0.15 * sin(p.x * 0.4 + p.z * 0.3 + u_time * 0.25);
        if (dens <= 0.0) continue;
        for (int i = 0; i < MAX_LIGHTS; i++) {
            if (i >= u_n_lights) break;
            float a;
            vec3 lc = lightAt(i, p, a);
            if (a <= 0.0) continue;
            scatter += lc * dens * dt;
        }
    }
    // Scatter gain. NOTE: total in-scattering scales with the NUMBER of lit
    // fixtures, so a drop (6 heads + 4 washes + blinders at full) pushed
    // ~10x the intro's energy through the tonemap and rendered as a pure
    // white-out. Normalising by a soft function of the lit count keeps a
    // drop bright and saturated without losing every beam edge.
    float litSum = 0.0;
    for (int i = 0; i < MAX_LIGHTS; i++) {
        if (i >= u_n_lights) break;
        litSum += u_lint[i];
    }
    float norm = 1.0 / (1.0 + 0.42 * max(litSum - 1.0, 0.0));
    // Opt-in calibration profile: radiance remains proportional to emitted
    // dimmers. Its exposure is constant across every frame and fixture count.
    if (u_fixed_exposure != 0) norm = 1.0;
    col += scatter * 3.2 * norm;

    // Fixture "bulb" glow so the rig reads as physical hardware.
    for (int i = 0; i < MAX_LIGHTS; i++) {
        if (i >= u_n_lights) break;
        vec3 toL = u_lpos[i] - ro;
        float proj = dot(toL, rd);
        if (proj <= 0.0) continue;
        float d = length(toL - rd * proj);
        float glow = exp(-d * d * 22.0) * u_lint[i];
        col += u_lcol[i] * glow * 0.85 * norm;
    }

    // Filmic-ish tonemap. A plain pow(1/2.2) gamma lifted the near-blacks to
    // mid grey and the whole frame read as fog; ACES-style keeps the room
    // black while letting beams bloom.
    if (u_fixed_exposure != 0) col *= u_exposure_scale;
    col = max(col, 0.0);
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    col = clamp(col, 0.0, 1.0);
    col = pow(col, vec3(0.85));
    f_color = vec4(col, 1.0);
}
"""
