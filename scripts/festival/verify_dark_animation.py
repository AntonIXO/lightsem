#!/usr/bin/env python3
"""Verify a saved native DMX animation and packed soundtrack without rendering.

Load the completed .blend before this script on Blender's command line. Five
frames are read from native keyframes, then compared with the authenticated DMX
at their source timestamps. Applying reference DMX changes only in-memory data;
this script never renders or saves the loaded .blend.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import traceback

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


def capture(runtime):
    values = {}

    def put(name, value):
        values[name] = [float(value)] if isinstance(value, (int, float, bool)) else list(value)

    for ref in runtime['fixtures']:
        name, kind = ref['fixture']['name'], ref['kind']
        if 'light' in ref:
            obj = ref['light']
            for attribute in ('energy', 'color', 'spot_size'):
                put(name + '.light.' + attribute, getattr(obj.data, attribute))
            for attribute in ('location', 'rotation_euler', 'scale'):
                put(name + '.light.' + attribute, getattr(obj, attribute))
        for role in ('lens', 'material'):
            if role not in ref:
                continue
            nodes = ref[role].node_tree.nodes
            emitter = nodes.get('DMX Emission')
            if emitter:
                color, strength = emitter.inputs['Color'], emitter.inputs['Strength']
            else:
                shader = nodes.get('Principled BSDF')
                if shader is None:
                    raise ValueError(f'Unrecognized controlled emission shader: {name}/{role}')
                color, strength = shader.inputs['Emission Color'], shader.inputs['Emission Strength']
            put(name + '.' + role + '.emission_color', color.default_value)
            put(name + '.' + role + '.emission_strength', strength.default_value)
        if kind == 'projector':
            nodes = ref['light'].data.node_tree.nodes
            for node_name in ('DMX Color', 'DMX Zoom', 'DMX Pattern', 'Media Clock'):
                put(name + '.projector.' + node_name, nodes[node_name].outputs[0].default_value)
        if 'fan' in ref:
            for attribute in ('location', 'rotation_euler', 'scale', 'hide_render', 'hide_viewport'):
                put(name + '.fan.' + attribute, getattr(ref['fan'], attribute))
    return values


def check_audio(scene, source, fps, origin, check):
    editor = scene.sequence_editor
    strips = [] if editor is None else [strip for strip in editor.strips_all if strip.type == 'SOUND']
    check(len(strips) == 1, f'Expected one source soundtrack, found {len(strips)}')
    results = []
    for strip in strips:
        packed = strip.sound.packed_file
        packed_hash = hashlib.sha256(packed.data).hexdigest() if packed else None
        check(packed_hash == source['sha256'], f'Packed audio missing or hash mismatch: {strip.name}')
        check(not strip.mute and strip.volume > 0, f'Source soundtrack is muted: {strip.name}')
        check(strip.frame_final_start == scene.frame_start, 'Audio does not start at the first rendered frame')
        check(strip.frame_final_end == scene.frame_end + 1, 'Audio does not cover the complete rendered interval')
        check(strip.frame_final_duration == scene.frame_end - scene.frame_start + 1, 'Audio duration differs from scene duration')
        check(strip.animation_offset_start == 0, 'Unexpected additional audio animation offset')
        check(strip.sound_offset == 0, 'Unexpected additional audio time offset')
        native_source_offset = (scene.frame_start - strip.frame_start) / fps
        rounding_tolerance = .5 / fps + 1e-8
        check(abs(native_source_offset - origin) <= rounding_tolerance,
              'Packed audio frame offset is not aligned with the source epoch')
        check(abs(strip.frame_offset_start / fps - origin) <= rounding_tolerance,
              'Audio start trim differs from the declared source offset')
        results.append({
            'name': strip.name, 'packed_bytes': packed.size if packed else 0,
            'packed_sha256': packed_hash, 'expected_sha256': source['sha256'],
            'frame_start': strip.frame_start, 'frame_offset_start': strip.frame_offset_start,
            'visible_frame_start': strip.frame_final_start, 'visible_frame_end_exclusive': strip.frame_final_end,
            'visible_frame_count': strip.frame_final_duration,
            'native_source_offset_s': native_source_offset, 'declared_source_offset_s': origin,
            'offset_error_s': native_source_offset - origin,
            'allowed_native_frame_rounding_s': rounding_tolerance,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Absolute and relative tolerances for serialized floating point properties')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    if not bpy.data.filepath:
        raise ValueError('Load a completed .blend before running this script')
    blend = Path(bpy.data.filepath)
    original_hash = sha(blend)
    errors = []
    report = {
        'status': 'failed', 'blend': str(blend), 'blend_sha256': original_hash,
        'blender': bpy.app.version_string, 'rendered': False, 'saved_scene': False,
        'tolerance_absolute': args.tolerance, 'tolerance_relative': args.tolerance,
        'frames_checked': [], 'errors': errors,
        'scope': 'Native animation serialization against authenticated DMX and packed source audio; no rendering',
    }

    def check(condition, message):
        if not condition:
            errors.append(message)

    try:
        scene = bpy.context.scene
        module_name = scene['LightSem scene module']
        module = importlib.import_module(module_name)
        patch = json.loads(scene['Fixture patch'])
        binding = Path(scene['Source binding'])
        check(patch == json.loads((binding / 'fixture_patch.json').read_text()),
              'Embedded fixture patch differs from authenticated binding')
        timeline = load_timeline(binding)
        timeline_hash = sha(binding / 'timeline.npz')
        check(timeline_hash == scene['Source DMX sha256'], 'Source DMX hash differs from saved scene')
        source_audio = patch['timeline']['audio']
        check(sha(source_audio['path']) == source_audio['sha256'], 'External source audio hash mismatch')
        fps = scene.render.fps / scene.render.fps_base
        origin = float(scene['Source start seconds'])
        runtime = module.recover_runtime(patch)
        check(len(runtime['fixtures']) == len(patch['fixtures']), 'Not every patched fixture recovered')
        check(len(bpy.data.actions) > 0, 'Saved scene contains no native animation actions')
        report.update({
            'scene_module': module_name, 'binding': str(binding),
            'source_timeline_sha256': timeline_hash,
            'fixture_patch_sha256': sha(binding / 'fixture_patch.json'),
            'source_audio_sha256': source_audio['sha256'],
            'source_start_seconds': origin, 'fps': fps,
            'frame_start': scene.frame_start, 'frame_end': scene.frame_end,
            'fixture_count': len(runtime['fixtures']), 'native_action_count': len(bpy.data.actions),
        })
        frames = sorted({round(scene.frame_start + (scene.frame_end - scene.frame_start) * fraction)
                         for fraction in (0, .25, .5, .75, 1)})
        for frame in frames:
            scene.frame_set(frame)
            bpy.context.view_layer.update()
            native = capture(runtime)
            source_time = origin + (frame - scene.frame_start) / fps
            controls = sample_timeline(timeline, source_time)
            module.apply_dmx(runtime, controls, source_time, frame=frame, bake=False)
            expected = capture(runtime)
            mismatches = {}
            maximum = 0.0
            for name, value in native.items():
                a, b = np.asarray(value, dtype=float), np.asarray(expected[name], dtype=float)
                delta = float(np.max(np.abs(a - b)))
                maximum = max(maximum, delta)
                if not np.allclose(a, b, atol=args.tolerance, rtol=args.tolerance):
                    mismatches[name] = {'native': value, 'from_source_dmx': expected[name],
                                        'max_absolute_difference': delta}
            check(not mismatches, f'Frame {frame}: {len(mismatches)} native properties differ from source DMX')
            report['frames_checked'].append({
                'frame': frame, 'source_time_s': source_time,
                'source_index': int(controls['source_index']), 'properties_checked': len(native),
                'max_absolute_difference': maximum, 'mismatches': mismatches,
                'passes': not mismatches,
            })
        report['packed_audio'] = check_audio(scene, source_audio, fps, origin, check)
    except BaseException:
        report['exception'] = traceback.format_exc()
        errors.append('Validation raised an exception; see exception field')
    finally:
        report['saved_blend_unchanged'] = sha(blend) == original_hash
        check(report['saved_blend_unchanged'], 'Loaded .blend changed on disk during validation')
        report['status'] = 'verified' if not errors else 'failed'
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError(f'Native animation validation failed; see {args.report}')


if __name__ == '__main__':
    main()
