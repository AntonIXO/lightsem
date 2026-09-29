"""Procedural architecture for the LightSem festival visualizer.

This module deliberately has no show timing, renderer, or DMX dependency.  All
positions are metres, Z is up, and the stage is at negative Y.  ``build_environment``
only creates geometry and returns the emissive materials that a show can control.
The projection surfaces and moving lights belong to the scene/fixture builder.

Large structures are batched meshes; crowd meshes are shared by linked instances.
There are no external assets, downloads, or Blender add-on dependencies.
"""

from __future__ import annotations

import math
import random

import bpy
from mathutils import Vector


_TAU = math.tau


def _material(name, color, *, metallic=0.0, roughness=0.55, emission=0.0):
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*color[:3], 1.0)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color[:3], 1.0)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Emission Color"].default_value = (*color[:3], 1.0)
    shader.inputs["Emission Strength"].default_value = emission
    return material


def _collection(name):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


class _Mesh:
    """Small mesh batcher, including tubes with stable local cross sections."""

    def __init__(self):
        self.vertices = []
        self.faces = []
        self.material_ids = []

    def face(self, vertices, material=0):
        offset = len(self.vertices)
        self.vertices.extend(tuple(v) for v in vertices)
        self.faces.append(tuple(range(offset, offset + len(vertices))))
        self.material_ids.append(material)

    def box(self, center, size, material=0):
        cx, cy, cz = center
        sx, sy, sz = (v / 2.0 for v in size)
        offset = len(self.vertices)
        self.vertices.extend(
            (cx + x * sx, cy + y * sy, cz + z * sz)
            for x, y, z in (
                (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
            )
        )
        for face in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                     (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            self.faces.append(tuple(offset + i for i in face))
            self.material_ids.append(material)

    def tube(self, points, radius, material=0, sides=5, *, end_radius=None):
        points = [Vector(p) for p in points]
        if len(points) < 2:
            return
        offset = len(self.vertices)
        previous_normal = None
        for index, point in enumerate(points):
            tangent = points[min(index + 1, len(points) - 1)] - points[max(0, index - 1)]
            if tangent.length < 1e-8:
                tangent = Vector((0, 0, 1))
            tangent.normalize()
            guide = Vector((0, 0, 1)) if abs(tangent.z) < 0.92 else Vector((1, 0, 0))
            normal = tangent.cross(guide).normalized()
            if previous_normal is not None and normal.dot(previous_normal) < 0:
                normal = -normal
            binormal = tangent.cross(normal).normalized()
            previous_normal = normal
            r = radius if end_radius is None else radius + (end_radius - radius) * index / (len(points) - 1)
            for side in range(sides):
                angle = side * _TAU / sides
                v = point + r * (math.cos(angle) * normal + math.sin(angle) * binormal)
                self.vertices.append(tuple(v))
        for ring in range(len(points) - 1):
            for side in range(sides):
                a = offset + ring * sides + side
                b = offset + ring * sides + (side + 1) % sides
                self.faces.append((a, b, b + sides, a + sides))
                self.material_ids.append(material)
        self.faces.append(tuple(offset + i for i in reversed(range(sides))))
        self.material_ids.append(material)
        last = offset + (len(points) - 1) * sides
        self.faces.append(tuple(last + i for i in range(sides)))
        self.material_ids.append(material)

    def ellipsoid(self, center, radii, material=0, segments=10, rings=6):
        offset = len(self.vertices)
        cx, cy, cz = center
        rx, ry, rz = radii
        for ring in range(rings + 1):
            phi = math.pi * ring / rings
            for segment in range(segments):
                theta = segment * _TAU / segments
                self.vertices.append((cx + rx * math.sin(phi) * math.cos(theta),
                                      cy + ry * math.sin(phi) * math.sin(theta),
                                      cz + rz * math.cos(phi)))
        for ring in range(rings):
            for segment in range(segments):
                a = offset + ring * segments + segment
                b = offset + ring * segments + (segment + 1) % segments
                self.faces.append((a, a + segments, b + segments, b))
                self.material_ids.append(material)

    def object(self, name, materials, collection, smooth=False):
        mesh = bpy.data.meshes.new(name + "_Mesh")
        mesh.from_pydata(self.vertices, [], self.faces)
        mesh.update()
        for material in materials:
            mesh.materials.append(material)
        for polygon, material_id in zip(mesh.polygons, self.material_ids):
            polygon.material_index = material_id
            polygon.use_smooth = smooth
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        return obj


def _circle(center, radius, *, plane="xy", steps=80, phase=0):
    cx, cy, cz = center
    result = []
    for index in range(steps + 1):
        angle = phase + index * _TAU / steps
        a, b = radius * math.cos(angle), radius * math.sin(angle)
        result.append((cx + a, cy + b, cz) if plane == "xy" else (cx + a, cy, cz + b))
    return result


def _truss(batch, start, finish, width=0.30, material=0):
    start, finish = Vector(start), Vector(finish)
    direction = (finish - start).normalized()
    length = (finish - start).length
    guide = Vector((0, 0, 1)) if abs(direction.z) < 0.9 else Vector((1, 0, 0))
    u = direction.cross(guide).normalized() * width / 2
    v = direction.cross(u).normalized() * width / 2
    corners = [u + v, u - v, -u - v, -u + v]
    for corner in corners:
        batch.tube((start + corner, finish + corner), 0.036, material, 6)
    bays = max(1, round(length / 0.65))
    for bay in range(bays):
        p = start.lerp(finish, bay / bays)
        q = start.lerp(finish, (bay + 1) / bays)
        for side in range(4):
            adjacent = (side + 1) % 4
            batch.tube((p + corners[side], p + corners[adjacent]), 0.022, material, 5)
            batch.tube((p + corners[side], q + corners[adjacent]), 0.016, material, 4)


def _stage(collection, materials):
    metal, deck, dark, grille, cable, silver, trim = materials
    body, rig, detail, emissive = _Mesh(), _Mesh(), _Mesh(), _Mesh()
    body.box((0, -14, 0.48), (23.0, 5.7, 0.96), 0)
    body.box((0, -13.98, 1.01), (23.15, 5.8, 0.10), 1)
    for step in range(3):
        body.box((0, -10.85 + step * .38, .16 * (3 - step)), (9.0, .55, .32 * (3 - step)), 0)
        emissive.tube(((-4.45, -10.58 + step * .38, .32 * (3 - step) + .015),
                       (4.45, -10.58 + step * .38, .32 * (3 - step) + .015)), .014, 0)
    # Wing risers and a projecting control/DJ booth. Its front is left open for a screen.
    for sign in (-1, 1):
        body.box((sign * 9.6, -14.3, 1.35), (2.8, 3.5, .55), 0)
        body.box((sign * 3.10, -12.08, 2.08), (.14, .65, 1.9), 0)
    body.box((0, -12.35, 3.02), (6.35, 1.45, .16), 1)
    body.box((0, -12.80, 1.7), (6.15, .11, 1.2), 0)
    # Mixer/decks, small controls and two monitor wedges.
    for x in (-1.45, 0, 1.45):
        body.box((x, -12.15, 3.14), (1.12, .67, .10), 2)
        detail.ellipsoid((x, -12.17, 3.21), (.22, .22, .035), 1, 16, 4)
        for i in range(6):
            detail.box((x - .40 + i * .14, -11.94, 3.215), (.035, .04, .045), 1)
    for sign in (-1, 1):
        body.box((sign * 5.4, -11.65, 1.27), (1.1, .7, .45), 2)
    # Outer portal and longitudinal top rails: fixture mounts remain unobstructed.
    for x in (-12.0, 12.0):
        _truss(rig, (x, -12.0, .05), (x, -12.0, 11.4), .42)
        _truss(rig, (x, -16.7, .05), (x, -16.7, 11.4), .42)
        _truss(rig, (x, -16.7, 11.4), (x, -12.0, 11.4), .35)
        detail.box((x, -12, .12), (1.1, 1.1, .24), 0)
    _truss(rig, (-12, -12, 11.4), (12, -12, 11.4), .42)
    _truss(rig, (-12, -16.7, 11.4), (12, -16.7, 11.4), .42)
    for x in (-7.5, 0, 7.5):
        _truss(rig, (x, -16.7, 11.4), (x, -12, 11.4), .28)
    # Main screen frame, exactly outside the reserved 11 x 7 metre rectangle.
    for x in (-5.72, 5.72):
        _truss(rig, (x, -15.05, 2.80), (x, -15.05, 10.25), .20)
    for z in (2.80, 10.25):
        _truss(rig, (-5.72, -15.05, z), (5.72, -15.05, z), .20)
    for sign in (-1, 1):
        x = sign * 9.0
        for dx in (-2.16, 2.16):
            detail.tube(((x + dx, -13.2, 2.82), (x + dx, -13.2, 8.17)), .055, 1, 8)
        for z in (2.82, 8.17):
            detail.tube(((x - 2.16, -13.2, z), (x + 2.16, -13.2, z)), .055, 1, 8)
        # Suspended line arrays live outside the side screens, facing the audience.
        for index in range(7):
            sx = sign * 12.78
            z = 7.7 - index * .48
            sy = -12.0 + .032 * index * index
            body.box((sx, sy, z), (1.03, .78, .43), 2)
            detail.box((sx, sy + .405, z), (.91, .028, .34), 2)
            for stripe in range(8):
                detail.tube(((sx - .42, sy + .425, z - .145 + stripe * .041),
                             (sx + .42, sy + .425, z - .145 + stripe * .041)), .006, 1, 4)
        detail.tube(((sign * 12.78, -12, 8.0), (sign * 12, -12, 11.2)), .015, 3)
        for index in range(3):
            sx = sign * (7.5 + index * 1.2)
            body.box((sx, -10.83, .55), (1.10, 1.03, 1.08), 2)
            detail.box((sx, -10.301, .55), (.94, .03, .88), 2)
            for z in (.29, .78):
                detail.tube(_circle((sx, -10.275, z), .19, plane="xz", steps=24), .025, 0, 6)
    # Patch racks and coiled power runs ground the installation in a real event.
    for sign in (-1, 1):
        body.box((sign * 10.85, -9.7, .70), (.7, .65, 1.40), 2)
        for z in (.35, .65, .95, 1.2):
            detail.box((sign * 10.85, -9.36, z), (.56, .024, .07), 1)
        for offset in range(4):
            detail.tube(((sign * 10.85, -9.6, .035 + offset * .007),
                         (sign * 12.3, -9.3, .035), (sign * 13.5, -6, .035),
                         (sign * 16.4, 1.8, .035)), .014, 3, 5)
    body.object("Stage_deck_booth_speakers", [metal, deck, dark], collection)
    rig.object("Aluminium_stage_truss", [silver], collection, smooth=True)
    detail.object("Stage_grilles_hardware_and_cables", [metal, silver, grille, cable], collection)
    emissive.object("Stage_step_edge_strips", [trim], collection, smooth=True)
    return {"main": {"center": (0, -15, 6.5), "size": (11, 7)},
            "left": {"center": (-9, -13, 5.5), "size": (4, 5)},
            "right": {"center": (9, -13, 5.5), "size": (4, 5)},
            "fascia": {"center": (0, -11.99, 2.2), "size": (6, 1.6)}}


def _canopy(collection, cloth, cord, structural, garlands, seed):
    rng = random.Random(seed)
    threads, ribs, poles, knotwork = _Mesh(), _Mesh(), _Mesh(), _Mesh()
    circuit_count = len(garlands)
    bulbs = [_Mesh() for _ in range(circuit_count)]

    def bulb(legacy_circuit, position, radii):
        # Retain the original twelve-circuit wiring for the default venue. Custom
        # counts get independent angular sectors, never shared material aliases.
        if not circuit_count:
            return
        if circuit_count == 12:
            circuit = legacy_circuit
        else:
            angle = math.atan2(position[1] - 3, position[0]) % _TAU
            circuit = min(circuit_count - 1, int(angle / _TAU * circuit_count))
        bulbs[circuit].ellipsoid(position, radii, 0, 6, 4)
    # String-art roof: radial ruled surfaces have both warp and crossing bias cords.
    # A sparse mesh leaves every beam and projection visible through the architecture.
    count = 14
    origin_y, radius = 3.0, 22.0

    def surface(section, u, v):
        angle = (section + v) * _TAU / count
        radial = 1.12 + (radius - 1.12) * u
        z = 12.1 - 4.05 * u - 1.18 * math.sin(math.pi * u) + .48 * math.sin(math.pi * v) * u
        return (radial * math.cos(angle), origin_y + radial * math.sin(angle), z)

    pole_positions = []
    for section in range(count):
        outer = surface(section, 1, 0)
        pole_positions.append(outer)
        x, y, z = outer
        poles.tube(((x, y, .12), (x * .998, y, z + .5)), .115, 0, 10, end_radius=.075)
        poles.box((x, y, .09), (.8, .8, .18), 0)
        guy = (x * 1.13, origin_y + (y - origin_y) * 1.13, .08)
        poles.tube(((x, y, z), guy), .012, 1, 4)
        # Structural spokes are grouped double cords, following the roof sag.
        rib = [surface(section, i / 30, 0) for i in range(31)]
        ribs.tube(rib, .033, section % len(cord), 6)
        ribs.tube([(a, b, c - .10) for a, b, c in rib], .014, (section + 1) % len(cord), 5)
        # Radial warp, broad enough to register in the aerial view.
        for lane in range(1, 19):
            v = lane / 19
            threads.tube([surface(section, i / 24, v) for i in range(25)],
                         .010 if lane % 3 else .014, (section + lane // 4) % len(cord), 4)
        # Counter-running diagonals weave into diamonds. The taper near the crown is natural.
        for offset in range(-17, 33):
            for handedness in (-1, 1):
                path = []
                for index in range(61):
                    u = index / 60
                    v = offset / 18 + handedness * 1.85 * u
                    if .02 <= v <= .98:
                        path.append(surface(section, u, v))
                if len(path) >= 2:
                    threads.tube(path, .009, (section + offset // 4 + (handedness > 0)) % len(cord), 4)
        # Three concentric zig-zag ornament bands make the roof legible as an installation.
        for band, radial in enumerate((.38, .66, .9)):
            path = [surface(section, radial + .045 * math.sin(6 * math.pi * i / 60), i / 60)
                    for i in range(61)]
            ribs.tube(path, .022, (section + band + 2) % len(cord), 5)
        # Garland circuit assignment follows the wiring sectors, independent of RGB.
        circuit = section % 12
        arc = []
        for index in range(81):
            v = index / 80
            px, py, pz = surface(section, .91, v)
            arc.append((px, py, pz - .7 * math.sin(math.pi * v)))
        knotwork.tube(arc, .015, 0, 5)
        for index in range(1, 80, 2):
            px, py, pz = arc[index]
            drop = .09 + .18 * ((index % 7) / 7)
            knotwork.tube(((px, py, pz), (px, py, pz - drop)), .006, 0, 4)
            bulb(circuit, (px, py, pz - drop), (.034, .034, .050))
        # Fairy-star lines across alternate seams descend into the room.
        for index in range(2, 46):
            u = index / 48
            px, py, pz = surface(section, u, .05)
            bulb(circuit, (px, py, pz - .16), (.029, .029, .037))
    threads.object("Woven_diamond_canopy_14_sails", cord, collection, smooth=True)
    ribs.object("Canopy_radial_ribs_and_zigzag_bands", cord, collection, smooth=True)
    poles.object("Canopy_masts_anchors_and_guy_wires", structural, collection, smooth=True)
    # Central crown: interlaced suspended spirals around a layered circular aperture.
    crown = _Mesh()
    for index, (r, z) in enumerate(((1.25, 12.15), (1.5, 11.85), (1.13, 11.50), (.75, 11.15))):
        crown.tube(_circle((0, 3, z), r, steps=80), .045, index % len(cord), 8)
    for strand in range(28):
        path = []
        for index in range(33):
            t = index / 32
            angle = strand * _TAU / 28 + t * 1.6
            r = 1.24 * (1 - .48 * t) + .15 * math.sin(math.pi * t)
            path.append((r * math.cos(angle), 3 + r * math.sin(angle), 12.12 - 1.6 * t))
        crown.tube(path, .017, strand % len(cord), 5)
    crown.object("Suspended_woven_crown", cord, collection, smooth=True)
    # Garlands along two side paths, plus hanging illuminated points in the roof.
    for sign in (-1, 1):
        for bay in range(5):
            path = []
            y0 = -8 + bay * 6
            for i in range(41):
                t = i / 40
                path.append((sign * (17.0 + .5 * math.sin(t * math.pi)),
                             y0 + t * 6, 5.6 - 1.25 * math.sin(t * math.pi)))
            knotwork.tube(path, .014, 0, 5)
            for i in range(2, 40, 2):
                px, py, pz = path[i]
                circuit = (bay + (6 if sign > 0 else 0)) % 12
                bulb(circuit, (px, py, pz - .07), (.045, .045, .065))
            poles_side = _Mesh()
            poles_side.tube(((sign * 17, y0, 0), (sign * 17, y0, 5.7)), .055, 0, 6)
            poles_side.object(f"Garland_path_post_{sign}_{bay}", structural, collection, smooth=True)
    for circuit in range(12):
        # Additional small pendant clusters prevent the canopy from reading as a solid sheet.
        for index in range(14):
            angle = rng.uniform(0, _TAU)
            r = rng.uniform(5, 18)
            p = (math.cos(angle) * r, 3 + math.sin(angle) * r,
                 10.8 - .18 * r - rng.uniform(.2, .8))
            bulb(circuit, p, (.024, .024, .034))
    for circuit in range(circuit_count):
        # For very dense custom patches, preserve a physical receiver for every
        # addressed circuit even when a narrow sector misses existing fairy bulbs.
        if not bulbs[circuit].vertices:
            angle = (circuit + .5) / circuit_count * _TAU
            p = (18 * math.cos(angle), 3 + 18 * math.sin(angle), 7.15)
            bulbs[circuit].ellipsoid(p, (.034, .034, .05), 0, 6, 4)
        obj = bulbs[circuit].object(f"Garland_{circuit + 1:02d}", [garlands[circuit]], collection, smooth=True)
        obj["dmx_circuit"] = circuit + 1
        obj["fixture_type"] = "RGB fairy-light circuit"
    knotwork.object("Garland_suspension_cables", [structural[1]], collection, smooth=True)
    return pole_positions


def _lighting_rig(collection, positions, metal, silver, cable):
    """Build support rails and grounded towers around the actual fixture patch.

    Co-planar rows share a rail; each light has a short suspension at its own X.
    The fixture position itself stays free for the scene builder's head and lens.
    """
    if positions is None:
        positions = [(x, y, z) for y, z in ((-11, 9.5), (-2, 10.5), (7, 10.5), (16, 9.5))
                     for x in (-10, -6, -2, 2, 6, 10)]
    rows = {}
    for position in positions:
        if len(position) != 3 or not all(math.isfinite(float(v)) for v in position):
            raise ValueError("Every fixture mount must contain three finite coordinates")
        x, y, z = map(float, position)
        rows.setdefault((round(y, 3), round(z, 3)), []).append(x)
    truss, hardware, wires = _Mesh(), _Mesh(), _Mesh()
    for (y, z), xs in sorted(rows.items()):
        left, right = min(xs) - .9, max(xs) + .9
        rail_z = z + .67
        _truss(truss, (left, y, rail_z), (right, y, rail_z), .28)
        for x in (left, right):
            _truss(truss, (x, y, .12), (x, y, rail_z), .27)
            hardware.box((x, y, .08), (.95, .95, .16), 0)
            # Low ballast plates support the tower without intrusive long outriggers.
            hardware.box((x, y, .22), (.70, .70, .16), 0)
        for x in xs:
            hardware.tube(((x, y, rail_z - .10), (x, y, z + .20)), .025, 1, 6)
            hardware.box((x, y, z + .18), (.25, .11, .06), 0)
            wires.tube(((x, y + .15, z + .14), (x + .13, y + .16, z + .35),
                        (x + .18, y + .17, rail_z), (left, y + .17, rail_z)), .010, 0, 4)
        wires.tube(((left, y + .17, rail_z), (left + .14, y + .17, .05)), .014, 0, 5)
    if rows:
        truss.object("DMX_patch_fixture_rails_and_grounded_towers", [silver], collection, smooth=True)
        hardware.object("DMX_patch_fixture_clamps_and_ballast", [metal, silver], collection, smooth=True)
        wires.object("DMX_patch_fixture_power_and_signal_runs", [cable], collection, smooth=True)
    return positions


def _routed_garlands(collection, routes, materials, metal, cable):
    """Give each patched string output its own physical position-to-target span."""
    wires, supports = _Mesh(), _Mesh()
    endpoints = {}
    objects = []
    for circuit, (route, material) in enumerate(zip(routes, materials)):
        start, end = Vector(route["position"]), Vector(route["target"])
        delta = end - start
        side = delta.cross(Vector((0, 0, 1)))
        if side.length < 1e-6:
            side = Vector((1, 0, 0))
        side.normalize()
        sag = min(.85, delta.length * .045, max(0, min(start.z, end.z) - .08) * .5)
        bulbs = _Mesh()

        def point(t, strand):
            p = start.lerp(end, t)
            envelope = math.sin(math.pi * t)
            p.z -= sag * envelope
            # Both strands meet their true endpoints. The open weave in between
            # reads as a double fairy-light garland rather than a solid LED bar.
            p += side * (strand * .20 * envelope)
            return p

        for strand in (-1, 1):
            path = [point(i / 96, strand) for i in range(97)]
            wires.tube(path, .011, 0, 5)
            for index in range(1, 41):
                p = point(index / 41, strand)
                drop = .065 + .055 * ((index + circuit) % 4)
                tip = p - Vector((0, 0, min(drop, max(0, p.z - .06))))
                wires.tube((p, tip), .005, 0, 4)
                bulbs.ellipsoid(tip, (.034, .034, .048), 0, 6, 4)
        # Dimmer-addressable geometry and material remain in a strict 1:1 mapping.
        obj = bulbs.object(f"Garland_{circuit + 1:02d}", [material], collection, smooth=True)
        obj["dmx_circuit"] = circuit + 1
        obj["fixture_type"] = "RGB fairy-light circuit"
        obj["route_start"] = list(start)
        obj["route_end"] = list(end)
        obj["bulb_count"] = 80
        objects.append(obj)
        for endpoint in (start, end):
            key = (round(endpoint.x, 4), round(endpoint.y, 4))
            endpoints[key] = max(endpoints.get(key, 0), endpoint.z)
        supports.box(start - Vector((0, 0, .19)), (.17, .13, .23), 0)
    # A common crown endpoint shares a single mast; translated routes retain
    # their own endpoint anchors and never float when a patch moves a circuit.
    for (x, y), height in endpoints.items():
        if height <= .05:
            continue
        supports.tube(((x, y, .08), (x, y, height + .13)), .078, 0, 8, end_radius=.042)
        supports.box((x, y, .07), (.67, .67, .14), 0)
        supports.box((x, y, .18), (.43, .43, .10), 0)
    if routes:
        wires.object("Patched_garland_route_cables", [cable], collection, smooth=True)
        supports.object("Patched_garland_endpoint_masts_and_junctions", [metal], collection, smooth=True)
    return objects


def _portal_rings(collection, cord, metal):
    loops, suspension, inlay = _Mesh(), _Mesh(), _Mesh()
    rings = []
    for index, (y, radius) in enumerate(((-5, 2.4), (1, 3.1), (7, 4.0))):
        center = (0, y, 7.5)
        rings.append({"center": center, "radius": radius})
        for offset in (-.10, .10):
            loops.tube(_circle((0, y + offset, 7.5), radius, plane="xz", steps=128), .075, index % len(cord), 8)
        for rib in range(120):
            angle = rib * _TAU / 120
            p = (radius * math.cos(angle), y - .11, 7.5 + radius * math.sin(angle))
            q = ((radius + .075 * math.cos(6 * angle)) * math.cos(angle + .016), y + .11,
                 7.5 + (radius + .075 * math.cos(6 * angle)) * math.sin(angle + .016))
            inlay.tube((p, q), .013, (rib // 15 + index) % len(cord), 4)
        for side in (-1, 1):
            x = side * radius * .60
            bottom = 7.5 + radius * .80
            suspension.tube(((x, y, bottom), (x, y, max(11.9, bottom + .3))), .009, 0, 4)
    loops.object("Three_suspended_psychedelic_portals", cord, collection, smooth=True)
    inlay.object("Portal_woven_transverse_details", cord, collection, smooth=True)
    suspension.object("Portal_rigging", [metal], collection, smooth=True)
    return rings


def _petal_fans(collection, fabrics, metal, cord):
    petals, edges, framework = _Mesh(), _Mesh(), _Mesh()
    locations = [(-14.8, -5.0, 1), (14.8, -5.0, -1), (-17.5, 9.0, 1), (17.5, 9.0, -1)]
    for fan_index, (x, y, sign) in enumerate(locations):
        base = Vector((x, y, .45))
        # A shallow curved fan of twelve split-ridge textile petals, naturally faceted.
        for blade in range(12):
            angle = math.radians(-67 + blade * 134 / 11)
            reach = 5.6 + .75 * math.cos(angle * 2) + .25 * math.sin(blade * 1.7)
            dx = math.sin(angle) * reach
            z = math.cos(angle) * reach
            depth = .9 * math.sin(angle * 1.5) * sign
            tip = base + Vector((dx, depth, z))
            waist = base + Vector((dx * .58, depth * .58 - .32, z * .58))
            width = .40 + .12 * math.cos(angle)
            left = waist + Vector((-width * math.cos(angle), .08, width * math.sin(angle)))
            right = waist + Vector((width * math.cos(angle), .08, -width * math.sin(angle)))
            ridge = waist + Vector((0, -.38, .08))
            material = (blade + fan_index * 2) % len(fabrics)
            for triangle in ((base, left, ridge), (base, ridge, right), (left, tip, ridge), (ridge, tip, right)):
                petals.face(triangle, material)
            edges.tube((base, left, tip, right, base), .013, (blade + fan_index) % len(cord), 4)
            framework.tube((base, ridge, tip), .022, 0, 5)
        framework.tube(((x, y, 0), (x, y, 2.0)), .10, 0, 8)
        framework.box((x, y, .10), (1.3, 1.3, .2), 0)
    petals.object("Sculptural_folded_textile_petal_fans", fabrics, collection)
    edges.object("Petal_fan_fluorescent_seams", cord, collection, smooth=True)
    framework.object("Petal_sculpture_frames", [metal], collection, smooth=True)
    # Organic woven spires at the wings, with no solid surfaces in front of screens.
    spires, spire_ribs = _Mesh(), _Mesh()
    for sign in (-1, 1):
        x, y = sign * 15.4, -12.4
        for strand in range(20):
            path = []
            for index in range(41):
                t = index / 40
                angle = strand * _TAU / 20 + t * 2.3
                r = 1.75 * (1 - t) ** .8 + .10
                path.append((x + r * math.cos(angle) + sign * .8 * t * t,
                             y + r * math.sin(angle), .10 + 9.3 * t))
            spires.tube(path, .023, strand % len(cord), 5)
        for level in range(1, 15):
            t = level / 15
            r = 1.75 * (1 - t) ** .8 + .10
            spire_ribs.tube(_circle((x + sign * .8 * t * t, y, .1 + 9.3 * t), r, steps=48),
                           .018, (level // 3) % len(cord), 5)
    spires.object("Organic_woven_stage_spires", cord, collection, smooth=True)
    spire_ribs.object("Organic_spire_latitude_rings", cord, collection, smooth=True)


def _person_mesh(index, materials, collection, rng):
    """A clothed human archetype, facing negative Y, with a distinct dancing pose."""
    body = _Mesh()
    skin, shirt, trousers, hair, shoes = 0, 1, 2, 3, 4
    # A tapered elliptical torso and hips avoid the cone silhouettes of test scenes.
    body.tube(((0, 0, .90), (0, -.012, 1.35)), .185, shirt, 10, end_radius=.215)
    body.ellipsoid((0, 0, .88), (.19, .125, .17), trousers, 10, 5)
    neck = (0, -.012, 1.465)
    body.tube(((0, -.015, 1.31), neck), .075, skin, 8)
    body.ellipsoid((0, -.024, 1.62), (.115, .115, .15), skin, 12, 8)
    body.ellipsoid((0, .008, 1.68), (.118, .104, .101), hair, 12, 6)
    # Nose, only a few centimetres, makes the crowd's facing direction readable.
    body.ellipsoid((0, -.133, 1.615), (.022, .037, .032), skin, 8, 4)
    if index % 5 == 0:
        body.ellipsoid((0, .085, 1.52), (.12, .065, .22), hair, 10, 6)
    stride = .04 + .14 * rng.random()
    for sign in (-1, 1):
        hip = (sign * .095, .0, .88)
        knee = (sign * (.11 + stride * .4), sign * stride, .47)
        ankle = (sign * (.13 + stride * .6), sign * stride * .7, .09)
        body.tube((hip, knee), .087, trousers, 8, end_radius=.064)
        body.tube((knee, ankle), .066, trousers, 8, end_radius=.046)
        body.ellipsoid((ankle[0], ankle[1] - .055, .07), (.075, .15, .07), shoes, 10, 5)
        shoulder = Vector((sign * .185, -.009, 1.32))
        pose = (index + (1 if sign > 0 else 3)) % 7
        if pose in (0, 1):
            elbow = Vector((sign * .37, -.015, 1.65 + pose * .12))
            hand = Vector((sign * (.29 + .16 * pose), -.055, 1.99 + pose * .07))
        elif pose == 2:
            elbow = Vector((sign * .42, -.055, 1.30))
            hand = Vector((sign * .47, -.12, 1.62))
        elif pose == 3:
            elbow = Vector((sign * .32, -.10, 1.08))
            hand = Vector((sign * .40, -.31, 1.31))
        else:
            elbow = Vector((sign * (.27 + .015 * pose), -.03, 1.05))
            hand = Vector((sign * .29, -.09 - .025 * pose, .80 + .03 * pose))
        sleeve_end = shoulder.lerp(elbow, .44)
        body.tube((shoulder, sleeve_end), .082, shirt, 8, end_radius=.064)
        body.tube((sleeve_end, elbow, hand), .049, skin, 8, end_radius=.037)
        body.ellipsoid(hand, (.043, .043, .07), skin, 8, 5)
    # One mesh per pose/material archetype; hundreds of spectators share these meshes.
    obj = body.object(f"Crowd_archetype_{index:02d}", materials, collection, smooth=True)
    return obj


def _crowd(collection, palette, seed):
    rng = random.Random(seed)
    archetypes = []
    for index in range(18):
        materials = [palette["skins"][index % len(palette["skins"])],
                     palette["shirts"][index % len(palette["shirts"])],
                     palette["pants"][index % len(palette["pants"])],
                     palette["hair"][index % len(palette["hair"])], palette["shoes"]]
        archetypes.append(_person_mesh(index, materials, collection, rng))
    positions = []
    # Leave a few organically winding aisles and the camera foreground clear.
    attempts = 0
    while len(positions) < 176 and attempts < 6000:
        attempts += 1
        x, y = rng.uniform(-12.7, 12.7), rng.uniform(-7.8, 23.0)
        if y > 13 and abs(x - 1.8 * math.sin(y * .23)) < 1.8:
            continue
        if y < -3 and abs(x) < 2.0:
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < .91 ** 2 for px, py in positions):
            continue
        positions.append((x, y))
    for index, (x, y) in enumerate(positions):
        template = archetypes[index % len(archetypes)]
        if index < len(archetypes):
            obj = template
        else:
            obj = bpy.data.objects.new(f"Dancer_{index:03d}", template.data)
            collection.objects.link(obj)
        height = rng.uniform(.91, 1.12)
        width = rng.uniform(.87, 1.16)
        obj.scale = (width, width, height)
        obj.rotation_euler = (rng.uniform(-.035, .035), rng.uniform(-.065, .065), rng.uniform(-.52, .52))
        obj.location = (x, y, .01)
        obj["visual_role"] = "audience scale reference"
    return len(positions)


def _landscape(collection, materials, seed):
    rng = random.Random(seed)
    ground, path, bark, foliage, stone = materials
    floor = _Mesh()
    floor.box((0, 1, -.15), (160, 160, .26), 0)
    floor.object("Festival_ground", [ground], collection)
    paths, trees, leaves, rocks = _Mesh(), _Mesh(), _Mesh(), _Mesh()
    for sign in (-1, 1):
        # Broad, muted gravel side paths have a physical edge, without a bright drawn grid.
        paths.box((sign * 18.1, 4, -.003), (2.8, 43, .02), 0)
        for y in range(-15, 25, 3):
            rocks.ellipsoid((sign * 19.7 + rng.uniform(-.25, .25), y, .12),
                            (.24, .32, .18), 0, 7, 4)
    for index in range(40):
        angle = index * _TAU / 40 + rng.uniform(-.04, .04)
        radius = rng.uniform(27.5, 42)
        x, y = math.cos(angle) * radius, 3 + math.sin(angle) * radius
        height = rng.uniform(6.5, 13.5)
        lean = rng.uniform(-.7, .7)
        trees.tube(((x, y, 0), (x + lean * .4, y, height * .53),
                    (x + lean, y + .25, height)), rng.uniform(.18, .34), 0, 8, end_radius=.055)
        for branch in range(7):
            a = angle + branch * 2.4
            z = height * (.35 + branch * .073)
            reach = rng.uniform(1.4, 3.0) * (1.15 - branch * .075)
            tip = (x + math.cos(a) * reach + lean * .6, y + math.sin(a) * reach, z + .9)
            trees.tube(((x + lean * .5, y, z), tip), .09, 0, 6, end_radius=.015)
            leaves.ellipsoid(tip, (reach * .70, reach * .67, 1.1), branch % len(foliage), 8, 5)
        leaves.ellipsoid((x + lean, y + .25, height - .3), (1.4, 1.3, 1.65), index % len(foliage), 8, 5)
    paths.object("Muted_gravel_walkways", [path], collection)
    trees.object("Woodland_tree_trunks_and_branches", [bark], collection, smooth=True)
    leaves.object("Woodland_faceted_foliage", foliage, collection)
    rocks.object("Walkway_edge_stones", [stone], collection)


def build_environment(seed=4107, garland_count=12, fixture_mounts=None, garland_routes=None):
    """Build the complete unlit venue and return objects/materials for show control.

    The ``garland_materials`` start at emission strength zero. Animate their
    Principled BSDF ``Emission Color`` and ``Emission Strength`` from the patch.
    Decorative cords use only a dim 0.12-strength pigment glow by default; the
    renderer may dim these together for a complete electronic blackout. Set
    ``garland_count`` to any nonnegative integer for distinct physical circuits.
    ``fixture_mounts`` contains the actual moving-head XYZ positions from the
    patch, or an empty sequence to omit their support towers. ``garland_routes``
    optionally supplies one ``position``/``target`` dict per garland circuit;
    these become physical strings instead of the default roof-sector circuits.
    """
    if isinstance(garland_count, bool) or not isinstance(garland_count, int) or garland_count < 0:
        raise ValueError("garland_count must be a nonnegative integer")
    if garland_routes is not None:
        if len(garland_routes) != garland_count:
            raise ValueError("garland_routes must contain exactly garland_count routes")
        for route in garland_routes:
            for key in ("position", "target"):
                if key not in route or len(route[key]) != 3 or not all(
                        math.isfinite(float(v)) for v in route[key]):
                    raise ValueError("Every garland route needs finite XYZ position and target")
            if (Vector(route["position"]) - Vector(route["target"])).length < 1e-6:
                raise ValueError("Garland position and target must differ")
    collections = {name: _collection("Festival_" + name) for name in
                   ("Architecture", "Canopy", "Sculptures", "Crowd", "Landscape")}
    metal = _material("Venue_powder_coated_charcoal", (.019, .026, .036), metallic=.60, roughness=.40)
    silver = _material("Venue_brushed_aluminium", (.24, .29, .34), metallic=.82, roughness=.29)
    deck = _material("Venue_stage_deck", (.027, .028, .036), roughness=.65)
    rubber = _material("Venue_speaker_black", (.009, .012, .018), roughness=.73)
    grille = _material("Venue_speaker_grille", (.029, .037, .041), metallic=.65, roughness=.55)
    cable = _material("Venue_cable", (.008, .01, .015), roughness=.52)
    trim = _material("Venue_edge_marker", (.05, .24, .32), emission=.12)
    colors = [(.035, .40, .50), (.35, .045, .45), (.53, .04, .21),
              (.085, .27, .58), (.44, .15, .045), (.07, .44, .25)]
    cords = [_material(f"Woven_pigment_{index:02d}", color, roughness=.56, emission=.12)
             for index, color in enumerate(colors)]
    fabrics = [_material(f"Textile_petal_{index:02d}", tuple(c * .60 for c in color), roughness=.73)
               for index, color in enumerate(colors)]
    garlands = [_material(f"Garland_{index + 1:02d}", (1, .63, .23), roughness=.32, emission=0)
                for index in range(garland_count)]
    for index, material in enumerate(garlands):
        material["dmx_circuit"] = index + 1
        material["control"] = "RGB + dimmer"
    screens = _stage(collections["Architecture"], [metal, deck, rubber, grille, cable, silver, trim])
    rig_positions = _lighting_rig(collections["Architecture"], fixture_mounts, metal, silver, cable)
    pole_positions = _canopy(collections["Canopy"], fabrics, cords, [metal, cable],
                             garlands if garland_routes is None else [], seed)
    garland_objects = None
    if garland_routes is not None:
        garland_objects = _routed_garlands(collections["Canopy"], garland_routes, garlands, metal, cable)
    rings = _portal_rings(collections["Sculptures"], cords, silver)
    _petal_fans(collections["Sculptures"], fabrics, metal, cords)
    skins = [_material(f"Audience_skin_{i}", color, roughness=.72) for i, color in enumerate(
        ((.29, .15, .09), (.49, .28, .17), (.66, .40, .27), (.23, .10, .065), (.75, .53, .38)))]
    shirts = [_material(f"Audience_clothing_{i}", color, roughness=.84) for i, color in enumerate(
        ((.02, .025, .04), (.055, .025, .07), (.018, .085, .075), (.17, .045, .026),
         (.22, .19, .12), (.08, .13, .17), (.035, .03, .04), (.15, .035, .10), (.16, .16, .18)))]
    pants = [_material(f"Audience_trousers_{i}", color, roughness=.85) for i, color in enumerate(
        ((.02, .028, .045), (.023, .021, .027), (.073, .067, .06)))]
    hair = [_material(f"Audience_hair_{i}", color, roughness=.80) for i, color in enumerate(
        ((.012, .008, .009), (.05, .022, .014), (.18, .11, .04)))]
    crowd_count = _crowd(collections["Crowd"], {"skins": skins, "shirts": shirts, "pants": pants,
                        "hair": hair, "shoes": rubber}, seed + 1)
    ground = _material("Venue_damp_ground", (.018, .023, .026), metallic=.18, roughness=.31)
    # A procedural fine bump provides wet-earth texture without image dependencies.
    nodes = ground.node_tree.nodes
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 7.0
    noise.inputs["Detail"].default_value = 3.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = .22
    bump.inputs["Distance"].default_value = .08
    ground.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
    ground.node_tree.links.new(bump.outputs["Normal"], nodes.get("Principled BSDF").inputs["Normal"])
    path = _material("Venue_gravel_path", (.042, .044, .041), roughness=.84)
    bark = _material("Woodland_bark", (.06, .034, .021), roughness=.92)
    foliage = [_material(f"Woodland_leaf_{i}", c, roughness=.89) for i, c in enumerate(
        ((.012, .036, .018), (.016, .045, .028), (.025, .047, .019)))]
    stone = _material("Venue_path_stones", (.08, .083, .085), roughness=.86)
    _landscape(collections["Landscape"], (ground, path, bark, foliage, stone), seed + 2)
    return {
        "collections": collections,
        "garland_materials": garlands,
        "garland_route_objects": garland_objects,
        "decorative_materials": cords + [trim],
        "screen_placements": screens,
        "canopy_poles": pole_positions,
        "portal_rings": rings,
        "crowd_count": crowd_count,
        "installed_moving_head_mounts": rig_positions,
        "fixture_mounts": {
            "stage_truss": [(x, -11.95, 10.95) for x in (-10, -6, -2, 2, 6, 10)],
            "stage_floor": [(x, -10.8, 1.30) for x in (-10, -6, 6, 10)],
            "side_masts": [(-16.8, y, 5.4) for y in (-5, 7, 19)] +
                           [(16.8, y, 5.4) for y in (-5, 7, 19)],
            "laser_positions": [(-10.8, -11.8, 8.7), (10.8, -11.8, 8.7)],
        },
    }
