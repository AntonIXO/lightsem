"""An empty, moonless festival venue built at real-world scale.

Architecture is deliberately passive. Only the returned garland materials can
emit, and these also start black. The scene controller supplies every photon.
The neutral textiles, metalwork and wet ground therefore respond to the show
rather than baking a coloured illustration into the venue.
"""

from __future__ import annotations

import math
import random

import bpy
from mathutils import Vector

from lightsem.festival.geometry import _Mesh, _collection, _circle, _stage, _lighting_rig


TAU = math.tau


def _material(name, color, roughness=.5, metallic=0., *, texture="plain"):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = (*color, 1)
    nodes, links = m.node_tree.nodes, m.node_tree.links
    p = nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = roughness
    p.inputs["Metallic"].default_value = metallic
    p.inputs["Emission Strength"].default_value = 0
    m["passive_architecture"] = True
    if texture == "plain":
        return m
    geometry = nodes.new("ShaderNodeNewGeometry")
    noise = nodes.new("ShaderNodeTexNoise")
    links.new(geometry.outputs["Position"], noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = {"cloth": 5., "metal": 38., "foil": 90., "soil": 2.1}.get(texture, 20.)
    noise.inputs["Detail"].default_value = 5
    variation = nodes.new("ShaderNodeMixRGB")
    variation.blend_type = "MULTIPLY"
    variation.inputs[0].default_value = .35
    variation.inputs[1].default_value = (*color, 1)
    links.new(noise.outputs["Color"], variation.inputs[2])
    links.new(variation.outputs[0], p.inputs["Base Color"])
    rough = nodes.new("ShaderNodeMapRange")
    rough.inputs["From Min"].default_value = .15
    rough.inputs["From Max"].default_value = .85
    rough.inputs["To Min"].default_value = max(.08, roughness - .18)
    rough.inputs["To Max"].default_value = min(.98, roughness + .13)
    links.new(noise.outputs["Fac"], rough.inputs["Value"])
    links.new(rough.outputs[0], p.inputs["Roughness"])
    fine = nodes.new("ShaderNodeTexNoise")
    links.new(geometry.outputs["Position"], fine.inputs["Vector"])
    fine.inputs["Scale"].default_value = {"cloth": 420., "metal": 760., "foil": 210., "soil": 58.}.get(texture, 100.)
    fine.inputs["Detail"].default_value = 3.
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .32 if texture != "soil" else .60
    bump.inputs["Distance"].default_value = {"cloth": .0013, "metal": .00035, "foil": .009, "soil": .036}.get(texture, .001)
    links.new(fine.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], p.inputs["Normal"])
    if texture == "cloth":
        p.inputs["Sheen Weight"].default_value = .22
        p.inputs["Sheen Roughness"].default_value = .75
        warp = nodes.new("ShaderNodeTexWave")
        warp.bands_direction = "X"
        warp.inputs["Scale"].default_value = 280
        warp.inputs["Distortion"].default_value = 1.0
        links.new(geometry.outputs["Position"], warp.inputs["Vector"])
        weft = nodes.new("ShaderNodeTexWave")
        weft.bands_direction = "Y"
        weft.inputs["Scale"].default_value = 280
        links.new(geometry.outputs["Position"], weft.inputs["Vector"])
        weave = nodes.new("ShaderNodeMath")
        weave.operation = "MULTIPLY"
        links.new(warp.outputs["Fac"], weave.inputs[0])
        links.new(weft.outputs["Fac"], weave.inputs[1])
        links.new(weave.outputs[0], bump.inputs["Height"])
    if texture == "soil":
        p.inputs["Coat Weight"].default_value = .24
        p.inputs["Coat Roughness"].default_value = .18
        larger_bump = nodes.new("ShaderNodeBump")
        larger_bump.inputs["Strength"].default_value = .35
        larger_bump.inputs["Distance"].default_value = .11
        links.new(noise.outputs["Fac"], larger_bump.inputs["Height"])
        links.new(bump.outputs["Normal"], larger_bump.inputs["Normal"])
        links.new(larger_bump.outputs["Normal"], p.inputs["Normal"])
    return m


def _rounded(obj, width=.018, segments=3):
    modifier = obj.modifiers.new("Rounded manufactured edges", "BEVEL")
    modifier.width = width
    modifier.segments = segments
    modifier.limit_method = "ANGLE"
    modifier.angle_limit = .6
    modifier.affect = "EDGES"
    normal = obj.modifiers.new("Face weighted corner normals", "WEIGHTED_NORMAL")
    normal.keep_sharp = True


def _surface(name, function, nu, nv, collection, materials, material_index=0):
    verts = [function(i / nu, j / nv) for i in range(nu + 1) for j in range(nv + 1)]
    faces = []
    for i in range(nu):
        for j in range(nv):
            a = i * (nv + 1) + j
            faces.append((a, a + nv + 1, a + nv + 2, a + 1))
    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    for material in materials:
        mesh.materials.append(material)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        polygon.material_index = material_index
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    return obj


def _terrain(collection, materials, seed):
    soil, puddle, grit = materials

    def ground(u, v):
        x, y = (u - .5) * 92, (v - .5) * 100
        # Centimetre scale undulation; no mountainous noise or mirror-flat plane.
        z = -.08 + .028 * math.sin(x * .9 + .6 * math.sin(y)) * math.sin(y * .69)
        z += .014 * math.sin(x * 2.3 + y * 1.4) + .007 * math.cos(y * 4.5 - x * 3.4)
        return x, y, z

    _surface("Rain_dark_compacted_earth", ground, 168, 182, collection, [soil])
    rng = random.Random(seed)
    water = _Mesh()
    # Irregular shallow puddles interrupt the rough earth; no full-floor mirror.
    for cx, cy, radius in ((-7, 9, 2.1), (5, 12, 2.0), (1, 5, 1.25), (-2, 17, 1.45),
                           (9, 1, 1.8), (-8, -4, 1.5), (5, -6, 1.3), (-12, 16, 1.2)):
        points = []
        for j in range(65):
            angle = j * TAU / 64
            r = radius * (1 + .13 * math.sin(angle * 5 + cx) + .085 * math.cos(angle * 7))
            points.append((cx + r * math.cos(angle), cy + .57 * r * math.sin(angle), -.047))
        for j in range(64):
            water.face(((cx, cy, -.047), points[j], points[j + 1]))
    water.object("Shallow_irregular_rain_puddles", [puddle], collection)
    stones = _Mesh()
    for _ in range(370):
        x, y = rng.uniform(-20, 20), rng.uniform(-9, 27)
        size = rng.uniform(.012, .035)
        stones.ellipsoid((x, y, -.031), (size, size * 1.4, size * .6), 0, 8, 4)
    stones.object("Small_ground_grit", [grit], collection, smooth=True)


def _canopy(collection, fabrics, rope, cable, metal):
    seams, weaves, rigging, poles = _Mesh(), _Mesh(), _Mesh(), _Mesh()
    count = 12
    center = Vector((0, 5, 12.6))

    def sail(section, u, v):
        # Bowed triangular tensile membrane. Each side scallops inward under
        # tension, leaving deliberate long openings between sail panels.
        span = .90 - .17 * math.sin(math.pi * u)
        angle = (section + .5 + (v - .5) * span) * TAU / count
        radius = 2.0 + 21.0 * u
        z = 12.55 - 3.25 * u - 1.15 * math.sin(math.pi * u)
        z += .70 * math.sin(math.pi * v) * math.sin(math.pi * u)
        z += .025 * math.sin(v * 46 * math.pi + u * 4) * math.sin(math.pi * u)
        return (radius * math.cos(angle), 5 + radius * math.sin(angle), z)

    endpoints = []
    for section in range(count):
        obj = _surface(f"Tensioned_canvas_sail_{section + 1:02d}",
                       lambda u, v, s=section: sail(s, u, v), 56, 32,
                       collection, fabrics, section % len(fabrics))
        thickness = obj.modifiers.new("Woven canvas thickness", "SOLIDIFY")
        thickness.thickness = .0015
        thickness.offset = 0
        # Double sewn seams and sewn-in tension ropes follow the true surface.
        for v in (0, 1):
            path = [sail(section, i / 72, v) for i in range(73)]
            seams.tube(path, .014, 0, 8)
            inner = .011 if v == 0 else .989
            seams.tube([sail(section, i / 72, inner) for i in range(73)], .005, 0, 6)
        for u in (0, 1, .82):
            seams.tube([sail(section, u, i / 60) for i in range(61)], .010, 0, 6)
        # A disciplined macrame band in front of the cloth supplies real parallax
        # and detail without the former full-roof tangle of neon polylines.
        for line in range(38):
            v = (line + .5) / 38
            path = []
            for j in range(33):
                u = .64 + j / 32 * .18
                p = Vector(sail(section, u, v))
                p.z -= .055 + .010 * math.sin(j * math.pi / 4 + line * math.pi)
                path.append(p)
            weaves.tube(path, .008, line % 2, 6)
        for line in range(12):
            u = .64 + (line + .5) / 12 * .18
            path = []
            for j in range(97):
                v = j / 96
                p = Vector(sail(section, u, v))
                p.z -= .055 + .012 * math.cos(v * 38 * TAU + line * math.pi)
                path.append(p)
            weaves.tube(path, .008, line % 2, 6)
        angle = (section + .5) * TAU / count
        anchor = Vector((23.3 * math.cos(angle), 5 + 23.3 * math.sin(angle), 9.75))
        endpoints.append(tuple(anchor))
        poles.tube(((anchor.x, anchor.y, .05), anchor), .095, 0, 16, end_radius=.068)
        poles.box((anchor.x, anchor.y, .05), (.9, .9, .10))
        rigging.tube((anchor, Vector(sail(section, 1, .5))), .012, 0, 6)
        # The crown is entirely suspended from perimeter masts.
        rigging.tube((anchor, center), .012, 0, 6)
        outer = Vector((anchor.x * 1.15, 5 + (anchor.y - 5) * 1.15, .04))
        rigging.tube((anchor, outer), .008, 0, 6)
    for r, z in ((2.0, 12.6), (1.9, 12.42), (1.72, 12.25)):
        seams.tube(_circle((0, 5, z), r, steps=128), .026, 0, 10)
    seams.object("Canvas_reinforced_hems_and_suspended_crown", [rope], collection, smooth=True)
    weaves.object("Organized_handwoven_canopy_border", [rope, fabrics[1]], collection, smooth=True)
    rigging.object("Crown_tension_cables_no_central_mast", [cable], collection, smooth=True)
    poles.object("Perimeter_canopy_masts", [metal], collection, smooth=True)
    return endpoints


def _portals(collection, foil, rope, metal, cable):
    rigging, braid, ribs = _Mesh(), _Mesh(), _Mesh()
    rings = []
    for index, (y, z, radius, thickness) in enumerate(((-5.1, 7.6, 2.3, .20), (1.1, 8.1, 2.95, .245),
                                                     (7.5, 8.8, 3.55, .29))):
        def torus(u, v):
            a, b = TAU * u, TAU * v
            crinkle = .006 * math.sin(a * 93 + b * 17) + .004 * math.sin(a * 157 - b * 13)
            tube = thickness + crinkle
            r = radius + tube * math.cos(b)
            return (r * math.cos(a), y + tube * math.sin(b), z + r * math.sin(a))
        _surface(f"Crinkled_silver_textile_portal_{index + 1}", torus, 256, 40,
                 collection, [foil])
        # Counterwound cords physically lift off the silver textile skin.
        for handedness in (-1, 1):
            for strand in range(14):
                path = []
                for j in range(641):
                    a = TAU * j / 640
                    b = handedness * a * 19 + strand * TAU / 14
                    r = radius + (thickness + .011) * math.cos(b)
                    path.append((r * math.cos(a), y + (thickness + .011) * math.sin(b), z + r * math.sin(a)))
                braid.tube(path, .006, index % 2, 5)
        for segment in range(24):
            a = segment * TAU / 24
            path = []
            for j in range(25):
                b = TAU * j / 24
                r = radius + (thickness + .008) * math.cos(b)
                path.append((r * math.cos(a), y + (thickness + .008) * math.sin(b), z + r * math.sin(a)))
            ribs.tube(path, .009, 0, 6)
        for side in (-1, 1):
            x = side * radius * .65
            lower = z + radius * math.sqrt(1 - .65 ** 2)
            rigging.tube(((x, y, lower), (x, y, 12.35)), .006, 0, 6)
            rigging.tube(((x, y, 12.35), (side * 10.9, y, 11.9)), .008, 0, 6)
        rings.append({"center": (0, y, z), "radius": radius})
    braid.object("Portals_counterwound_braided_rope_skin", [rope, metal], collection, smooth=True)
    ribs.object("Portals_stitched_transverse_bindings", [rope], collection, smooth=True)
    rigging.object("Portals_steel_suspension", [cable], collection, smooth=True)
    return rings


def _petals(collection, fabrics, rope, metal):
    edges, frames = _Mesh(), _Mesh()
    # Smooth stretched membranes, deliberately outside the screen rectangles.
    for sculpture, (x, y, sign) in enumerate(((-15.2, -5., 1), (15.2, -5., -1),
                                             (-18.1, 9.5, 1), (18.1, 9.5, -1))):
        for blade in range(9):
            angle = math.radians(-57 + blade * 114 / 8)
            reach = 5.0 + .65 * math.cos(angle * 1.8)
            axis = Vector((math.sin(angle), 0, math.cos(angle)))
            across = Vector((math.cos(angle), 0, -math.sin(angle)))
            base = Vector((x, y, .24))

            def petal(u, v):
                width = .43 * math.sin(math.pi * u) ** .75
                p = base + axis * (reach * u) + across * ((v - .5) * 2 * width)
                p.y += sign * (.7 * math.sin(math.pi * u) + .60 * u * u)
                p.y += .20 * math.sin(math.pi * v) * math.sin(math.pi * u)
                p.y += .008 * math.sin(u * 22 * math.pi) * math.sin(math.pi * v)
                return tuple(p)

            obj = _surface(f"Stretched_silk_leaf_{sculpture + 1}_{blade + 1}", petal, 44, 12,
                           collection, fabrics, (blade // 3 + sculpture) % len(fabrics))
            modifier = obj.modifiers.new("Hemmed membrane thickness", "SOLIDIFY")
            modifier.thickness = .001
            for v in (0, 1):
                edges.tube([petal(i / 60, v) for i in range(61)], .008, 0, 6)
            frames.tube([petal(i / 60, .5) for i in range(61)], .015, 0, 8)
        frames.tube(((x, y, .04), (x, y, 1.6)), .060, 0, 12)
        frames.box((x, y, .05), (.8, .8, .1))
    edges.object("Textile_leaf_sewn_rope_edges", [rope], collection, smooth=True)
    frames.object("Textile_leaf_spring_frames", [metal], collection, smooth=True)


def _garlands(collection, routes, materials, metal, cable):
    wires, sockets, attachments = _Mesh(), _Mesh(), _Mesh()
    objects = []
    for circuit, (route, material) in enumerate(zip(routes, materials)):
        start, end = Vector(route["position"]), Vector(route["target"])
        delta = end - start
        side = delta.cross(Vector((0, 0, 1)))
        side = side.normalized() if side.length > 1e-5 else Vector((1, 0, 0))
        sag = min(.6, delta.length * .03, max(0, min(start.z, end.z) - .06) * .5)
        bulbs = _Mesh()

        def path(t, strand):
            p = start.lerp(end, t)
            envelope = math.sin(math.pi * t)
            p.z -= sag * envelope
            p += side * strand * .12 * envelope
            return p

        for strand in (-1, 1):
            wires.tube([path(j / 96, strand) for j in range(97)], .005, 0, 6)
            for j in range(40):
                p = path((j + .5) / 40, strand)
                drop = .055 + .045 * ((j + circuit) % 3)
                tip = p - Vector((0, 0, min(drop, max(0, p.z - .05))))
                wires.tube((p, tip), .003, 0, 5)
                sockets.tube((tip, tip - Vector((0, 0, .012))), .012, 0, 8)
                bulbs.ellipsoid(tip - Vector((0, 0, .029)), (.022, .022, .030), 0, 10, 7)
        obj = bulbs.object(f"Garland_{circuit + 1:02d}", [material], collection, smooth=True)
        obj["dmx_circuit"] = circuit + 1
        obj["fixture_name"] = route["name"]
        obj["fixture_type"] = "RGB fairy-light circuit"
        obj["route_start"], obj["route_end"] = list(start), list(end)
        obj["bulb_count"] = 80
        objects.append(obj)
        attachments.box(start + Vector((0, 0, .04)), (.08, .065, .08))
        # Radial endpoints hang from the tensile roof. A crown shared by all
        # strings does not need a mast in the camera's view of the stage.
        for endpoint in (start, end):
            if (endpoint - Vector((0, 5, 12))).length < 1:
                wires.tube((endpoint, (endpoint.x, endpoint.y, 12.55)), .006, 0, 6)
            else:
                radial = Vector((endpoint.x, endpoint.y - 5, 0))
                radius = radial.length
                upper = endpoint + Vector((0, 0, max(.5, 12.55 - .16 * radius - endpoint.z)))
                wires.tube((endpoint, upper), .005, 0, 6)
    wires.object("Garland_physical_route_and_suspension_wires", [cable], collection, smooth=True)
    sockets.object("Garland_individual_black_lamp_sockets", [metal], collection, smooth=True)
    attachments.object("Garland_patch_junction_boxes", [metal], collection)
    return objects


def _stage_details(collection, fabric, metal, cable):
    def skirt(u, v):
        x = -11.5 + 23 * u
        return (x, -11.075 + .04 * math.sin(x * 14) + .012 * math.sin(x * 27), .03 + .91 * v)
    _surface("Pleated_black_stage_skirt", skirt, 460, 5, collection, [fabric])
    bolts = _Mesh()
    for x in (-5.65, 5.65):
        for j in range(18):
            z = 3 + j * .42
            bolts.ellipsoid((x, -14.83, z), (.018, .012, .018), 0, 8, 5)
    bolts.object("Screen_frame_fasteners", [metal], collection, smooth=True)
    cable_runs = _Mesh()
    for side in (-1, 1):
        for loop in range(5):
            cable_runs.tube(_circle((side * 11.7, -9.3, .02 + loop * .012), .28 + loop * .023,
                                   steps=64), .012, 0, 7)
    cable_runs.object("Neatly_coiled_stage_power_cables", [cable], collection, smooth=True)


def _wash_supports(collection, fixtures, metal, silver, cable):
    """Physical telescopic stands follow every wash in the actual fixture patch."""
    poles, hardware, wires = _Mesh(), _Mesh(), _Mesh()
    installed = []
    for fixture in fixtures:
        if fixture["kind"] != "wash":
            continue
        x, y, z = map(float, fixture["position"])
        # Mount behind the lens, preserving the optical aperture and pan/tilt
        # clearance. A narrow weighted stand occupies only its patched position.
        sy = y + .24
        height = max(.10, z - .17)
        split = min(2.6, height * .63)
        poles.tube(((x, sy, .08), (x, sy, split)), .055, 0, 16)
        poles.tube(((x, sy, split - .16), (x, sy, height)), .039, 0, 16)
        hardware.box((x, sy, .03), (.64, .64, .06), 0)
        hardware.box((x, sy, .083), (.48, .48, .045), 0)
        for angle in (math.pi / 6, math.pi * 5 / 6, math.pi * 1.5):
            foot = (x + .48 * math.cos(angle), sy + .48 * math.sin(angle), .055)
            poles.tube(((x, sy, .73), foot), .022, 0, 10)
            hardware.box((foot[0], foot[1], .022), (.13, .13, .044), 0)
        # Height collar, clamp screw, and compact U yoke.
        poles.tube(((x, sy, split - .07), (x, sy, split + .055)), .068, 0, 16)
        hardware.tube(((x + .062, sy, split), (x + .12, sy, split)), .009, 1, 8)
        hardware.tube(((x + .12, sy - .04, split), (x + .12, sy + .04, split)), .011, 0, 8)
        poles.tube(((x - .26, sy, height), (x + .26, sy, height)), .025, 0, 12)
        for side in (-1, 1):
            poles.tube(((x + side * .26, sy, height), (x + side * .26, sy, z + .06)), .023, 0, 12)
        wires.tube(((x + .10, y + .14, z + .09), (x + .15, sy + .1, z - .25),
                    (x + .065, sy + .01, split), (x + .065, sy + .01, .08),
                    (x + (.75 if x >= 0 else -.75), sy + .2, .015)), .008, 0, 7)
        for level in range(max(1, int(height))):
            h = .5 + level
            if h < height:
                hardware.tube(_circle((x, sy, h), .058, steps=20), .003, 0, 5)
        installed.append(list(fixture["position"]))
    if installed:
        poles.object("Patched_wash_telescopic_steel_stands_and_yokes", [metal], collection, smooth=True)
        plate = hardware.object("Patched_wash_ballast_plates_feet_clamps", [metal, silver], collection, smooth=True)
        _rounded(plate, .008, 3)
        wires.object("Patched_wash_stand_power_and_DMX_cables", [cable], collection, smooth=True)
    return installed


def build_environment(patch, seed=4107):
    """Create passive, empty architecture and the patch's controlled garlands."""
    collections = {name: _collection("NightFestival_" + name) for name in
                   ("Architecture", "Canopy", "Sculptures", "Ground", "Garlands")}
    metal = _material("Night_black_anodized_aluminium", (.022, .024, .027), .37, .82, texture="metal")
    silver = _material("Night_brushed_aluminium", (.53, .55, .56), .33, .9, texture="metal")
    foil = _material("Night_crinkled_silver_woven_laminate", (.63, .64, .62), .29, .87, texture="foil")
    deck = _material("Night_stage_phenolic_plywood", (.018, .017, .016), .68, texture="cloth")
    rubber = _material("Night_rubber_and_speaker_enclosures", (.008, .009, .011), .67, texture="metal")
    grille = _material("Night_speaker_perforated_steel", (.030, .031, .032), .51, .68, texture="metal")
    cable = _material("Night_black_cable_jacket", (.006, .0065, .007), .55)
    rope = _material("Night_warm_unbleached_braided_rope", (.39, .34, .25), .88, texture="cloth")
    fabrics = [_material("Night_canvas_" + name, color, .78, texture="cloth") for name, color in
               (("warm_ivory", (.48, .44, .35)), ("unbleached_flax", (.29, .26, .21)),
                ("pearl", (.52, .51, .47)), ("charcoal", (.075, .068, .060)))]
    blackcloth = _material("Night_black_stage_drape", (.008, .008, .01), .91, texture="cloth")
    soil = _material("Night_rain_dark_earth", (.045, .037, .029), .42, texture="soil")
    puddle = _material("Night_shallow_rainwater", (.020, .019, .016), .095)
    pp = puddle.node_tree.nodes.get("Principled BSDF")
    pp.inputs["IOR"].default_value = 1.333
    pp.inputs["Coat Weight"].default_value = .85
    pp.inputs["Coat Roughness"].default_value = .075
    grit = _material("Night_small_gravel", (.08, .071, .058), .87, texture="soil")
    garland_routes = [f for f in patch["fixtures"] if f["kind"] == "garland"]
    garland_materials = []
    for i, fixture in enumerate(garland_routes):
        m = _material(f"Garland_{i + 1:02d}", (.18, .16, .13), .27)
        m["passive_architecture"] = False
        m["dmx_circuit"], m["fixture_name"] = i + 1, fixture["name"]
        m["control"] = "RGB + dimmer; zero light when DMX is black"
        garland_materials.append(m)
    screens = _stage(collections["Architecture"], [metal, deck, rubber, grille, cable, silver, metal])
    mounts = [f["position"] for f in patch["fixtures"] if f["kind"] == "moving_head"]
    _lighting_rig(collections["Architecture"], mounts, metal, silver, cable)
    wash_mounts = _wash_supports(collections["Architecture"], patch["fixtures"], metal, silver, cable)
    for obj in list(collections["Architecture"].objects):
        if obj.type == "MESH" and not obj.modifiers and any(word in obj.name for word in ("Stage_deck", "Stage_grilles", "clamps")):
            _rounded(obj, .018, 3)
    _stage_details(collections["Architecture"], blackcloth, silver, cable)
    canopy_poles = _canopy(collections["Canopy"], fabrics, rope, cable, metal)
    rings = _portals(collections["Sculptures"], foil, rope, silver, cable)
    _petals(collections["Sculptures"], fabrics, rope, metal)
    garland_objects = _garlands(collections["Garlands"], garland_routes, garland_materials, metal, cable)
    _terrain(collections["Ground"], [soil, puddle, grit], seed)
    return {
        "collections": collections,
        "garland_materials": garland_materials,
        "garland_route_objects": garland_objects,
        "decorative_materials": [metal, silver, foil, rope, *fabrics],
        "screen_placements": screens,
        "canopy_poles": canopy_poles,
        "portal_rings": rings,
        "crowd_count": 0,
        "installed_moving_head_mounts": mounts,
        "installed_wash_mounts": wash_mounts,
        "suggested_camera": {"position": (2.0, 18., 2.35), "target": (0, -5., 6.2), "lens": 29.0},
        "description": "Empty moonless canvas festival with physical fabric, braided portals and damp earth",
    }
