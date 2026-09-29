#!/usr/bin/env python3
"""Inspect a loaded festival .blend without rendering or saving it.

Run after the .blend argument on Blender's command line. ``--blackout`` applies
zero dimmers in memory, then also checks that controlled emitters turn off.
``--black-image`` additionally checks an already-rendered blackout PNG; this is
the image-space test that can expose compositor or shader contributions which
structural inspection misses. This script never starts a GPU render.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from lightsem.festival.patch import load_timeline, sample_timeline


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def scalar(socket, visited=None):
    """Resolve simple constant scalar graphs; unknown is deliberately not zero."""
    if not socket.is_linked:
        value = getattr(socket, 'default_value', None)
        return float(value) if isinstance(value, (float, int)) else None
    visited = set() if visited is None else visited
    source = socket.links[0].from_socket
    key = source.as_pointer()
    if key in visited:
        return None
    visited = visited | {key}
    node = source.node
    if node.type == 'VALUE':
        return float(source.default_value)
    if node.type != 'MATH':
        return None
    values = [scalar(item, visited) for item in node.inputs[:2]]
    a, b = values
    # DMX intensity often multiplies a procedural pattern that is nonconstant.
    if node.operation == 'MULTIPLY' and 0.0 in values:
        return 0.0
    if a is None or b is None:
        return None
    operations = {
        'MULTIPLY': lambda: a * b,
        'ADD': lambda: a + b,
        'SUBTRACT': lambda: a - b,
        'DIVIDE': lambda: a / b if b else 0.0,
        'MAXIMUM': lambda: max(a, b),
        'MINIMUM': lambda: min(a, b),
        'ABSOLUTE': lambda: abs(a),
        'POWER': lambda: a ** b,
    }
    if node.operation not in operations:
        return None
    try:
        value = float(operations[node.operation]())
        return min(1.0, max(0.0, value)) if node.use_clamp else value
    except (ValueError, OverflowError, TypeError):
        return None


def emission_strengths(tree, path=''):
    """Find possible surface/volume emitters, including nested node groups."""
    for node in tree.nodes:
        name = path + node.name
        if node.type in {'BSDF_PRINCIPLED', 'PRINCIPLED_VOLUME'}:
            socket = node.inputs.get('Emission Strength')
            if socket is not None:
                yield name, scalar(socket)
        elif node.type in {'EMISSION', 'BACKGROUND'}:
            yield name, scalar(node.inputs['Strength'])
        elif node.type == 'GROUP' and node.node_tree:
            yield from emission_strengths(node.node_tree, name + '/')


def runtime_emitters(runtime):
    """Use recovered runtime references, not merely arbitrary object labels."""
    lights, materials = set(), set()

    def collect(value):
        if isinstance(value, bpy.types.Object):
            if value.type == 'LIGHT':
                lights.add(value.as_pointer())
        elif isinstance(value, bpy.types.Material):
            materials.add(value.as_pointer())
        elif isinstance(value, dict):
            for key, child in value.items():
                if key not in {'fixture', 'scene', 'environment', 'camera'}:
                    collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    collect(runtime)
    return lights, materials


def image_stats(path, threshold):
    image = bpy.data.images.load(str(path.resolve()), check_existing=False)
    try:
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        if not pixels.size or image.channels < 3:
            raise ValueError('Expected a nonempty RGB or RGBA image')
        pixels = pixels.reshape(-1, image.channels)[:, :3]
        if not np.isfinite(pixels).all():
            raise ValueError('Image contains nonfinite channel values')
        maximum = float(pixels.max())
        return {
            'path': str(path.resolve()), 'sha256': sha(path),
            'resolution': list(image.size), 'max_rgb': maximum,
            'mean_rgb': float(pixels.mean()),
            'nonzero_channels': int(np.count_nonzero(pixels)),
            'threshold': threshold, 'passes': maximum <= threshold,
        }
    finally:
        bpy.data.images.remove(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-module', help='Default: saved scene module or realistic_scene')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--blackout', action='store_true', help='Apply zero dimmers in memory; never saves .blend')
    parser.add_argument('--require-blackout', action='store_true', help='Check saved/current emitters are already off')
    parser.add_argument('--black-image', type=Path, action='append', default=[])
    parser.add_argument('--black-threshold', type=float, default=0.0,
                        help='Allowed RGB maximum in Blender image pixels; default exact zero')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if not math.isfinite(args.black_threshold) or args.black_threshold < 0:
        raise ValueError('Black threshold must be finite and nonnegative')
    scene = bpy.context.scene
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    module_name = args.scene_module or scene.get('LightSem scene module', 'lightsem.festival.realistic_scene')
    module = importlib.import_module(module_name)
    patch = json.loads(scene['Fixture patch'])
    runtime = module.recover_runtime(patch)
    controlled_lights, controlled_materials = runtime_emitters(runtime)
    binding = Path(scene['Source binding'])
    binding_patch = json.loads((binding / 'fixture_patch.json').read_text())
    check(patch == binding_patch, 'Embedded fixture patch differs from authenticated binding')
    timeline = load_timeline(binding)
    timeline_sha = sha(binding / 'timeline.npz')
    check(timeline_sha == scene['Source DMX sha256'], 'Scene source DMX hash differs from binding')
    audio = patch['timeline']['audio']
    check(sha(audio['path']) == audio['sha256'], 'Source audio hash mismatch')
    check(len(runtime['fixtures']) == len(patch['fixtures']), 'Not every patched fixture was recovered')
    if args.blackout:
        t = float(scene['Source start seconds']) + (scene.frame_current - scene.frame_start) / (
            scene.render.fps / scene.render.fps_base)
        controls = sample_timeline(timeline, t)
        module.apply_dmx(runtime, controls, t, blackout=True)
    require_blackout = args.blackout or args.require_blackout

    crowd_pattern = re.compile(r'crowd|audience|people|person|human', re.IGNORECASE)
    used_materials = {}
    for obj in scene.objects:
        check(not crowd_pattern.search(obj.name), f'Crowd/person object remains: {obj.name}')
        if obj.data:
            check(not crowd_pattern.search(obj.data.name), f'Crowd/person object data remains: {obj.data.name}')
        for slot in obj.material_slots:
            if slot.material:
                used_materials[slot.material.as_pointer()] = slot.material
    for mat in used_materials.values():
        check(not crowd_pattern.search(mat.name), f'Crowd/person material still in use: {mat.name}')

    world_strengths = []
    if scene.world:
        check(scene.world.use_nodes, 'World must use nodes with an explicit zero-strength Background')
        if scene.world.use_nodes:
            world_strengths = list(emission_strengths(scene.world.node_tree))
            check(bool(world_strengths), 'World has no inspectable background/emission shader')
            for name, strength in world_strengths:
                check(strength == 0.0, f'World emission is not provably zero: {name}={strength}')
    lights = []
    for obj in scene.objects:
        if obj.type != 'LIGHT':
            continue
        controlled = obj.as_pointer() in controlled_lights
        lights.append({'name': obj.name, 'controlled': controlled, 'watts': obj.data.energy})
        if not controlled or require_blackout:
            check(obj.data.energy == 0.0, f'{"Blackout" if controlled else "Uncontrolled"} light emits: {obj.name}')
    material_emission = []
    for pointer, mat in used_materials.items():
        controlled = pointer in controlled_materials
        if not mat.use_nodes:
            continue
        for name, strength in emission_strengths(mat.node_tree):
            if strength != 0.0:
                material_emission.append({'material': mat.name, 'node': name,
                                          'controlled': controlled, 'strength': strength})
            if not controlled or require_blackout:
                check(strength == 0.0,
                      f'{"Blackout" if controlled else "Uncontrolled"} material emission not zero: {mat.name}/{name}={strength}')
    images = [image_stats(path, args.black_threshold) for path in args.black_image]
    for result in images:
        check(result['passes'], f'Blackout image contains visible signal: {result["path"]} max={result["max_rgb"]}')

    report = {
        'status': 'verified' if not errors else 'failed',
        'blend': bpy.data.filepath,
        'blend_sha256': sha(bpy.data.filepath) if bpy.data.filepath else None,
        'scene_module': module_name, 'blender': bpy.app.version_string,
        'engine': scene.render.engine, 'fixture_count': len(patch['fixtures']),
        'scene_object_count': len(scene.objects),
        'controlled_light_count': len(controlled_lights),
        'controlled_material_count': len(controlled_materials),
        'world_strengths': world_strengths, 'lights': lights,
        'nonzero_or_procedural_material_emission': material_emission,
        'blackout_applied_in_memory': args.blackout,
        'blackout_emitters_checked': require_blackout,
        'blackout_images': images, 'errors': errors,
        'source_binding': str(binding.resolve()), 'timeline_sha256': timeline_sha,
        'fixture_patch_sha256': sha(binding / 'fixture_patch.json'),
        'source_audio_sha256': audio['sha256'],
        'scope': 'Scene structure, zero ambient emission, DMX/audio provenance and optional blackout pixels; not aesthetic certification',
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError(f'{len(errors)} dark-scene verification failures; see {args.report}')


if __name__ == '__main__':
    main()
