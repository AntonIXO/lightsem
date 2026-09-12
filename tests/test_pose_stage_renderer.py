"""CPU-only synthetic tests. No real study assets, freeze, EGL, models or vault.
Run: bash run.sh -m unittest discover -s tests -p test_pose_stage_renderer.py -v
"""
import builtins
from contextlib import nullcontext
import hashlib
from unittest.mock import Mock, patch
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
import wave
import numpy as np
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

r = module('synthetic_pose_renderer', ROOT/'outputs/render_pose_stage_study.py')
optics = module('synthetic_optics_helper', ROOT/'outputs/run_dance_optics_full.py')

class CPUHelpers:
    @staticmethod
    def dmx_indices(start, n, count):
        indices = np.rint(np.arange(start, start+n)/30*40).astype(int)
        r.require(indices.max() < count, 'DMX bounds')
        return indices
    @staticmethod
    def optical_dmx(dmx, rig, plan):
        return dmx.copy()
    @staticmethod
    def body_image(arm, world, poses, index, font):
        return Image.new('RGB', (640, 360), (248, 248, 248))

class FakeRenderer:
    renderer_name = 'synthetic CPU constant pixels; NO EGL'
    def __init__(self, delta=0):
        self.delta = delta
    def render_frame(self, dmx, t):
        return np.full((360, 640, 3), int(dmx[0])+self.delta, dtype=np.uint8)

class FakeEval:
    def __init__(self, out):
        self.out = out
    @staticmethod
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    @staticmethod
    def arrays(path):
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    @staticmethod
    def savejson(path, obj):
        with Path(path).open('x') as f:
            json.dump(obj, f, allow_nan=False)
    @staticmethod
    def loadjson(path):
        return json.loads(Path(path).read_text())
    @staticmethod
    def savenpz(path, **arrays):
        with Path(path).open('xb') as f:
            np.savez_compressed(f, **arrays)
    def checked(self, path, sha):
        r.require(self.digest(path) == sha, 'Changed test artifact')
    def mapped(self, out, ctx, item):
        return out/'mapped', {}

class PureTests(unittest.TestCase):
    def setUp(self):
        self.plan = {'fixed_fast_samples': {'reference_local_frames': [270, 274, 278, 326],
            'shpongle_intervals_s': [[0, 2], [80, 82], [196, 198], [270, 272]], 'stride_frames': 4}}
    def test_run_review_report_route_without_project_or_gpu_import(self):
        original_import = builtins.__import__
        def no_project_import(name, *args, **kwargs):
            if name == 'lightsem' or name.startswith('lightsem.'):
                raise AssertionError('CPU review/report must not import GPU project code')
            return original_import(name, *args, **kwargs)
        for stage in ('review', 'report'):
            for native_pass in (False, True):
                with self.subTest(stage=stage, native_pass=native_pass):
                    out = Path('/synthetic-uncreated-pose-stage-test')
                    context = {'freeze': {'files': {'render_pose_stage_study.py': 'synthetic'}}}
                    schedule = [{'id': 'synthetic', 'item_id': 'synthetic', 'start': 0, 'frames': 6, 'matched': False}]
                    ev = SimpleNamespace(lock=Mock(return_value=nullcontext()),
                        context=Mock(return_value=context), checked=Mock(),
                        render_schedule=Mock(return_value=schedule),
                        stage_receipt=Mock(return_value={'pass': native_pass}), digest=Mock(return_value='synthetic'))
                    with patch.object(r, 'record_review') as review, patch.object(r, 'report') as report, \
                            patch('builtins.__import__', side_effect=no_project_import):
                        r.run(out, 1, ev, stage, Path('observation.json'))
                    ev.context.assert_called_once_with(out, activate=False)
                    if stage == 'review':
                        review.assert_called_once_with(out, context, schedule, Path('observation.json'), ev)
                        report.assert_not_called()
                    else:
                        report.assert_called_once_with(out, context, schedule, ev)
                        review.assert_not_called()
                    expected = [str(out/'native/stage.json')]
                    if native_pass:
                        expected.append(str(out/'references/stage.json'))
                    self.assertEqual(list(context['render_authorization']['stage_receipts']), expected)

    def test_terminal_report_preserves_failure_and_missing_coverage(self):
        with tempfile.TemporaryDirectory(prefix='pose-stage-report-cpu-') as tmp:
            out = Path(tmp)
            ev = FakeEval(out)
            (out/'freeze.json').write_text('{}')
            schedule = [{'id': name, 'item_id': 'synthetic', 'start': 0, 'frames': 6, 'matched': False}
                        for name in ('completed', 'failed', 'incomplete', 'unrun')]
            ctx = {'render_authorization': {'schedule': schedule, 'stage_receipts': {}}}
            for entry in schedule[:3]:
                target = out/'renders'/entry['id']
                target.mkdir(parents=True)
                ev.savejson(target/'started.json', {'entry': entry, 'freeze_sha256': ev.digest(out/'freeze.json'),
                    'render_authorization': ctx['render_authorization']})
            (out/'renders/completed/render.json').write_text('{}')
            failed = schedule[1]
            ev.savejson(out/'renders/failed/failed.json', {'entry': failed, 'freeze_sha256': ev.digest(out/'freeze.json'),
                'render_authorization': ctx['render_authorization'], 'phase': 'software_media_audit',
                'gates': {'candidate_pixels': True, 'complete_software_media': False},
                'error_type': 'ValueError', 'error': 'Synthetic full decode failure'})
            (out/'renders/incomplete/partial.bin').write_bytes(b'partial evidence')
            with patch.object(r, 'verified_render', return_value=(out/'renders/completed',
                    {'gates': {'candidate_pixels': False, 'complete_software_media': True}})):
                r.report(out, ctx, schedule, ev)
                receipt = ev.loadjson(out/'render_report.json')
                self.assertEqual(receipt['coverage_counts'], {'completed': 1, 'failed': 1, 'resource_blocked': 0, 'incomplete': 1, 'unrun': 1})
                self.assertFalse(receipt['all_approved_entries_rendered'])
                self.assertFalse(receipt['all_candidate_pixel_gates_pass'])
                self.assertFalse(receipt['all_software_media_gates_pass'])
                self.assertFalse(receipt['all_independent_sampled_observations_present'])
                self.assertIsNone(receipt['rows'][2]['gates']['candidate_pixels'])
                self.assertIsNone(receipt['rows'][3]['gates']['complete_software_media'])
                self.assertIn('partial.bin', receipt['rows'][2]['observed_artifact_hashes'])
                self.assertIsNone(receipt['human_music_fit_acceptance'])
                with self.assertRaises(FileExistsError):
                    r.report(out, ctx, schedule, ev)
            missing = out/'all-missing'
            missing.mkdir(); (missing/'freeze.json').write_text('{}')
            r.report(missing, ctx, schedule, FakeEval(missing))
            receipt = ev.loadjson(missing/'render_report.json')
            self.assertIsNone(receipt['all_candidate_pixel_gates_pass'])
            self.assertIsNone(receipt['all_software_media_gates_pass'])
            self.assertFalse(receipt['all_approved_entries_rendered'])

    @patch.object(r, 'study_artifact_bytes', return_value=0)
    def test_storage_preflight_and_runtime_reserve_boundaries(self, _study_size):
        item = {'id': 'synthetic', 'key': '083', 'start': 0}
        entry = {'id': 'synthetic', 'item_id': 'synthetic', 'start': 0, 'frames': 600, 'matched': False}
        plan = {**self.plan, 'items': [item]}
        with patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*2**30)):
            estimate = r.storage_preflight(Path('/synthetic'), plan, [entry])
        self.assertEqual(estimate['entries'][0]['retained_silent_video_planning_bytes'], 0)
        self.assertEqual(estimate['entries'][0]['retained_muxed_video_planning_bytes'], estimate['entries'][0]['video_planning_bytes'])
        self.assertEqual(estimate['video_planning_bytes_per_second'], 2*2**20)
        self.assertFalse(estimate['estimate_is_crf_hard_bound'])
        self.assertTrue(estimate['runtime_reserve_guard_required'])
        need = estimate['estimated_batch_bytes']+r.STORAGE_RESERVE_BYTES+r.BOOKKEEPING_MARGIN_BYTES
        with patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=need-1)):
            with self.assertRaises(r.ResourceBlocked) as caught:
                r.storage_preflight(Path('/synthetic'), plan, [entry])
        self.assertFalse(caught.exception.observation['scientific_refutation'])
        self.assertFalse(caught.exception.observation['estimate_is_crf_hard_bound'])
        runtime_need = r.STORAGE_RESERVE_BYTES+r.STREAM_WRITE_HEADROOM_BYTES+r.BOOKKEEPING_MARGIN_BYTES
        with patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=runtime_need)):
            r.storage_guard(Path('/synthetic'), 'streamed_video_frame')
        with patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=runtime_need-1)):
            with self.assertRaises(r.ResourceBlocked):
                r.storage_guard(Path('/synthetic'), 'streamed_video_frame')
        maximum_current = r.STUDY_ARTIFACT_LIMIT_BYTES-r.STREAM_WRITE_HEADROOM_BYTES-r.BOOKKEEPING_MARGIN_BYTES
        with patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*2**30)):
            with patch.object(r, 'study_artifact_bytes', return_value=maximum_current):
                r.storage_guard(Path('/synthetic'), 'streamed_video_frame')
            with patch.object(r, 'study_artifact_bytes', return_value=maximum_current+1):
                with self.assertRaises(r.ResourceBlocked) as budget_failure:
                    r.storage_guard(Path('/synthetic'), 'streamed_video_frame')
        self.assertEqual(budget_failure.exception.observation['study_artifact_limit_bytes'], 1792*2**20)
        self.assertEqual(r.STORAGE_RESERVE_BYTES, 2**30)

    def test_storage_size_includes_all_study_files(self):
        with tempfile.TemporaryDirectory(prefix='pose-stage-size-cpu-') as tmp:
            out = Path(tmp)
            (out/'freeze.json').write_text('{}')
            paths = ['frozen/source.py', 'mapped/full_world.npz', 'renders/partial.mp4',
                     'renders/silent.mp4', 'renders/muxed.mp4', 'renders/sample.png', 'render_report.json']
            for index, name in enumerate(paths):
                path = out/name; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'x'*(index+1))
            self.assertEqual(r.study_artifact_bytes(out/'renders'), 2+sum(range(1, len(paths)+1)))

    def test_preflight_block_retained_and_reported_not_scientific_failure(self):
        with tempfile.TemporaryDirectory(prefix='pose-stage-resource-cpu-') as tmp:
            out = Path(tmp); ev = FakeEval(out)
            (out/'freeze.json').write_text('{}'); (out/'mapped').mkdir()
            (out/'mapped/map.json').write_text('{}')
            entry = {'id': 'blocked', 'item_id': 'synthetic', 'start': 0, 'frames': 6, 'matched': False}
            ctx = {'plan': {'items': [{'id': 'synthetic'}]},
                   'render_authorization': {'schedule': [entry], 'stage_receipts': {}}}
            error = r.ResourceBlocked(out, 'storage_preflight', 100, 1000)
            r.record_storage_block(out, ctx, entry, error, ev)
            failed = ev.loadjson(out/'renders/blocked/failed.json')
            self.assertEqual(failed['outcome'], 'resource_blocked')
            self.assertIsNone(failed['gates']['candidate_pixels'])
            r.report(out, ctx, [entry], ev)
            receipt = ev.loadjson(out/'render_report.json')
            self.assertEqual(receipt['coverage_counts']['resource_blocked'], 1)
            self.assertFalse(receipt['all_approved_entries_rendered'])
            self.assertIsNone(receipt['all_candidate_pixel_gates_pass'])
            self.assertIsNone(receipt['all_software_media_gates_pass'])
            other = dict(entry, id='cannot-save-failure')
            with patch.object(ev, 'savejson', side_effect=OSError('synthetic receipt storage error')):
                r.record_storage_block(out, ctx, other, error, ev)
            self.assertTrue(any('synthetic receipt storage error' in note for note in error.__notes__))

    def test_run_preflight_blocks_before_gpu_import(self):
        out = Path('/synthetic-uncreated-pose-stage-resource-test')
        context = {'freeze': {'files': {'render_pose_stage_study.py': 'synthetic'}}, 'plan': {'items': []}}
        entry = {'id': 'synthetic', 'item_id': 'synthetic', 'start': 0, 'frames': 6, 'matched': False}
        ev = SimpleNamespace(lock=Mock(return_value=nullcontext()), context=Mock(return_value=context),
            checked=Mock(), render_schedule=Mock(return_value=[entry]),
            stage_receipt=Mock(return_value={'pass': False}), digest=Mock(return_value='synthetic'))
        original_import = builtins.__import__
        def no_project_import(name, *args, **kwargs):
            if name == 'lightsem' or name.startswith('lightsem.'):
                raise AssertionError('Blocked preflight must not allocate/import GPU')
            return original_import(name, *args, **kwargs)
        error = r.ResourceBlocked(out, 'storage_preflight', 100, 1000)
        with patch.object(r, 'validate_context'), patch.object(r, 'storage_preflight', side_effect=error), \
                patch.object(r, 'record_storage_block') as record, patch('builtins.__import__', side_effect=no_project_import):
            with self.assertRaises(r.ResourceBlocked) as caught:
                r.run(out, 1, ev)
        self.assertIs(caught.exception, error)
        record.assert_called_once_with(out, context, entry, error, ev)

    def test_real_evaluator_bookkeeping_survives_blocked_start_at_both_edges(self):
        # Import CPU utilities only. No context/freeze/map/study arrays are opened.
        evaluator = module('real_eval_storage_integration', ROOT/'outputs/eval_pose_stage_study.py')
        for edge in ('free_reserve', 'study_cap'):
            with self.subTest(edge=edge), tempfile.TemporaryDirectory(prefix='pose-stage-real-lock-cpu-') as tmp:
                out = Path(tmp); (out/'freeze.json').write_text('{}')
                source = out/'mapped'; source.mkdir(); (source/'map.json').write_text('{}')
                entry = {'id': 'blocked', 'item_id': 'synthetic', 'start': 0, 'frames': 6, 'matched': False}
                ctx = {'plan': {'items': [{'id': 'synthetic'}]},
                    'render_authorization': {'schedule': [entry], 'stage_receipts': {}}}
                state = evaluator.storage_state(out)
                if edge == 'free_reserve':
                    state['filesystem_free_bytes'] = evaluator.FREE_RESERVE_BYTES+2**20
                    state['study_artifact_bytes'] = 0
                else:
                    state['filesystem_free_bytes'] = 100*2**30
                    state['study_artifact_bytes'] = evaluator.STUDY_LIMIT_BYTES-2**20
                state['remaining_study_budget_bytes'] = evaluator.STUDY_LIMIT_BYTES-state['study_artifact_bytes']
                original = r.ResourceBlocked(out, 'storage_preflight', state['filesystem_free_bytes'], 10**12)
                with evaluator.lock(out), patch.object(evaluator, 'storage_state', return_value=state), \
                        patch.object(evaluator, 'mapped', return_value=(source, {})):
                    r.record_storage_block(out, ctx, entry, original, evaluator)
                    self.assertFalse((out/'renders/blocked/started.json').exists())
                    failed = evaluator.loadjson(out/'renders/blocked/failed.json')
                    self.assertEqual(failed['outcome'], 'resource_blocked')
                    self.assertFalse(failed['started_receipt_written'])
                    self.assertEqual(len(failed['secondary_recording_errors']), 1)
                    self.assertIsNone(failed['gates']['complete_software_media'])
                    r.report(out, ctx, [entry], evaluator)
                    terminal = evaluator.loadjson(out/'render_report.json')
                    self.assertEqual(terminal['coverage_counts']['resource_blocked'], 1)
                    self.assertFalse(terminal['all_approved_entries_rendered'])
                    self.assertIsNone(terminal['all_candidate_pixel_gates_pass'])
                self.assertTrue(any('Ordinary started receipt not written' in note for note in original.__notes__))
                self.assertIsNone(evaluator._ACTIVE_OUTPUT)

    def test_budget_two_rejected_before_lock_context_or_gpu(self):
        ev = Mock()
        for budget in (0, 2, -1, True):
            with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'exactly --budget 1'):
                r.run(Path('/synthetic-never-created'), budget, ev)
        self.assertEqual(ev.mock_calls, [])

    def test_review_schedule(self):
        item = {'key': 'shpongle', 'start': 0}
        entry = {'start': 70*30, 'frames': 600}
        got = r.samples(self.plan, item, entry)
        self.assertEqual(got['chronological_1hz_last'], list(range(0, 600, 30))+[599])
        self.assertEqual(got['fixed_fast'], list(range(300, 360, 4)))
        entry = {'start': 1200, 'frames': 600}
        self.assertEqual(r.samples(self.plan, {'key': '083', 'start': 1200}, entry)['fixed_fast'], [270, 274, 278, 326])
        self.assertEqual(r.samples(self.plan, item, {'start': 0, 'frames': 1})['chronological_1hz_last'], [0])
    def test_pixel_gate_strict_boundaries(self):
        key = 'outside_bulbs_anychannel_ge250'
        self.assertTrue(r.pixel_gate({key: {'mean': .009, 'max': .049}}))
        self.assertFalse(r.pixel_gate({key: {'mean': .01, 'max': .049}}))
        self.assertFalse(r.pixel_gate({key: {'mean': 0, 'max': .05}}))
    def test_full_arrays_no_padding(self):
        worlds = {'world': np.zeros((6,22,3)), 'times': np.arange(6)/30}
        streams = {'baseline': np.zeros((8,512), np.uint8), 'candidate': np.zeros((8,512), np.uint8), 'time': np.arange(8)/40}
        item = {'full_frames': 6}
        r.validate_arrays(worlds, streams, item, {'start': 0, 'frames': 6})
        with self.assertRaisesRegex(ValueError, 'padding'):
            r.validate_arrays(worlds, streams, item, {'start': 1, 'frames': 6})
        streams['candidate'] = streams['candidate'].astype(float)
        with self.assertRaisesRegex(ValueError, 'byte DMX'):
            r.validate_arrays(worlds, streams, item, {'start': 0, 'frames': 6})
    def test_software_commands_no_padding(self):
        command = r.encode_command('/bin/ffmpeg', Path('v.mp4'), 6, 's.wav', 4.0)
        self.assertIn('libx264', command)
        self.assertIn('-n', command)
        self.assertNotIn('apad', ' '.join(command))
        self.assertNotIn('nvenc', ' '.join(command))
        self.assertIn('none', r.decode_command('/bin/ffmpeg', 'v.mp4'))
    def test_global_dim_rounding_and_repeated_hash(self):
        scales = optics.brightness_scalars([60, 100])
        np.testing.assert_array_equal(scales, [1, .6])
        rgb = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, 2)
        dim = optics.dim_rgb(rgb, float(scales[1]))
        self.assertLessEqual(np.max(np.abs(dim.astype(float)-.6*rgb)), .5+1e-9)
        self.assertFalse(np.array_equal(r.raw_hash(rgb), r.raw_hash(dim)))
        with self.assertRaises(ValueError):
            optics.brightness_scalars([0, 1])

@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'Software media executables unavailable')
class SyntheticMediaTests(unittest.TestCase):
    def test_full_decode_brightness_repeat_and_create_only(self):
        with tempfile.TemporaryDirectory(prefix='pose-stage-renderer-cpu-') as tmp:
            out = Path(tmp)
            ev = FakeEval(out)
            (out/'freeze.json').write_text('{}')
            source = out/'mapped'
            source.mkdir()
            (source/'map.json').write_text('{}')
            ev.savenpz(source/'full_world.npz', world=np.zeros((6,22,3)), times=np.arange(6)/30)
            baseline, candidate = np.full((8,512), 60, np.uint8), np.full((8,512), 100, np.uint8)
            ev.savenpz(source/'full_dmx.npz', baseline=baseline, candidate=candidate, time=np.arange(8)/40)
            audio = out/'synthetic.wav'
            with wave.open(str(audio), 'wb') as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(48000)
                f.writeframes((np.sin(np.arange(9600)*2*np.pi*440/48000)*1000).astype('<i2').tobytes())
            item = {'id': 'shpongle_prior_flow0', 'key': 'shpongle', 'full_frames': 6, 'start': 0,
                    'frames': 6, 'source_arm': 'prior_native_flow', 'audio_path': str(audio), 'audio_offset_s': 0.}
            cfg = SimpleNamespace(width=640, height=360)
            mask = np.ones((360,640), bool)
            fake_optics = SimpleNamespace(pixel_masks=lambda rig, cfg: ({'outside_bulbs': mask, 'bulbs_only': mask, 'outside_caption': mask}, np.zeros((1,2))),
                pixel_stats=optics.pixel_stats, brightness_scalars=optics.brightness_scalars, dim_rgb=optics.dim_rgb)
            ctx = {'plan': {'items': [item], 'renderer': {}, 'fixed_fast_samples':
                {'shpongle_intervals_s': [[0,2]], 'stride_frames': 4}}, 'cfg': cfg, 'rig': None,
                'helpers': {'review': CPUHelpers, 'optics': fake_optics},
                'freeze': {'executables': {shutil.which('ffmpeg'): 'synthetic', shutil.which('ffprobe'): 'synthetic'}}}
            first = {'id': 'full', 'item_id': item['id'], 'start': 0, 'frames': 6, 'matched': False}
            second = dict(first, id='dim', matched=True)
            bad = dict(second, id='bad-repeat')
            decode_bad = dict(second, id='decode-failure')
            resource_bad = dict(second, id='runtime-resource-block')
            receipt_bad = dict(second, id='receipt-save-failure')
            schedule = [first, second, bad, decode_bad, resource_bad, receipt_bad]
            ctx['render_authorization'] = {'schedule': schedule, 'stage_receipts': {}}
            font = ImageFont.load_default()
            receipt = r.render_one(out, ctx, first, schedule, ev, FakeRenderer(), font)
            self.assertEqual(receipt['media']['software_video_frames'], 6)
            self.assertGreater(receipt['media']['decoded_audio_samples_stereo_48000'], 0)
            self.assertEqual(receipt['sample_schedule']['chronological_1hz_last'], [0,5])
            self.assertTrue(receipt['gates']['candidate_pixels'])
            self.assertIsNone(receipt['human_music_fit_acceptance'])
            matched = r.render_one(out, ctx, second, schedule, ev, FakeRenderer(), font)
            self.assertTrue(matched['brightness_reference']['all_repeated_frame_rawhashes_equal'])
            self.assertEqual(matched['brightness_reference']['scalars_baseline_candidate'], [1., .6])
            self.assertAlmostEqual(matched['brightness_reference']['matched_mean_difference_255'], 0.)
            r.verified_render(out, second, ctx, ev)
            observation = out/'observer.json'
            observed = {'entry_id': first['id'], 'reviewer': 'synthetic-test-observer',
                'notes': 'Synthetic test only, no real acceptance.', 'independent_of_renderer': True,
                'coverage': 'all_frozen_decoded_samples', 'viewed_sample_frames': [0,5],
                'viewed_sheets': [s['path'] for pages in receipt['media']['review_sheets'].values() for s in pages],
                'comparisons': {'body': 'Synthetic constant world.', 'baseline_vs_candidate': 'Synthetic constants.',
                    'clipping_and_visibility': 'No synthetic clipping.'}, 'limitations': ['Synthetic pixels only.']}
            observation.write_text(json.dumps(observed))
            with self.assertRaisesRegex(ValueError, 'Every chronological and fixed-fast'):
                r.record_review(out, ctx, schedule, observation, ev)
            observed['viewed_sample_frames'] = [0,4,5]
            observation.write_text(json.dumps(observed))
            r.record_review(out, ctx, schedule, observation, ev)
            reviewed = r.verify_review(out, ctx, first, ev)
            self.assertIsNone(reviewed['human_music_fit_acceptance'])
            with self.assertRaisesRegex(ValueError, 'next missing'):
                r.record_review(out, ctx, schedule, observation, ev)
            with self.assertRaises(FileExistsError):
                r.render_one(out, ctx, second, schedule, ev, FakeRenderer(), font)
            bad = dict(second, id='bad-repeat')
            with self.assertRaisesRegex(ValueError, 'Repeated original raw hash differs'):
                r.render_one(out, ctx, bad, schedule, ev, FakeRenderer(1), font)
            self.assertTrue((out/'renders/bad-repeat/failed.json').exists())
            self.assertFalse((out/'renders/bad-repeat/render.json').exists())
            with patch.object(r, 'audit_media', side_effect=ValueError('Synthetic decode failure')):
                with self.assertRaisesRegex(ValueError, 'Synthetic decode failure'):
                    r.render_one(out, ctx, decode_bad, schedule, ev, FakeRenderer(), font)
            failure = ev.loadjson(out/'renders/decode-failure/failed.json')
            self.assertEqual(failure['phase'], 'software_media_audit')
            self.assertTrue(failure['gates']['candidate_pixels'])
            self.assertFalse(failure['gates']['complete_software_media'])
            self.assertEqual(failure['freeze_sha256'], ev.digest(out/'freeze.json'))
            self.assertEqual(failure['render_authorization'], ctx['render_authorization'])
            real_guard = r.storage_guard
            def guard_decode(path, phase, *args):
                if phase == 'software_video_decode_and_samples':
                    raise r.ResourceBlocked(path, phase, 100, 1000)
                return real_guard(path, phase, *args)
            with patch.object(r, 'storage_guard', side_effect=guard_decode):
                with self.assertRaises(r.ResourceBlocked):
                    r.render_one(out, ctx, resource_bad, schedule, ev, FakeRenderer(), font)
            blocked = ev.loadjson(out/'renders/runtime-resource-block/failed.json')
            self.assertEqual(blocked['outcome'], 'resource_blocked')
            self.assertTrue(blocked['gates']['candidate_pixels'])
            self.assertIsNone(blocked['gates']['complete_software_media'])
            self.assertTrue((out/'renders/runtime-resource-block/matched.mp4').exists())
            original_error = ValueError('original synthetic media failure')
            real_save = ev.savejson
            def save_except_failure(path, obj):
                if Path(path).name == 'failed.json':
                    raise OSError('secondary synthetic receipt failure')
                return real_save(path, obj)
            with patch.object(r, 'audit_media', side_effect=original_error), patch.object(ev, 'savejson', side_effect=save_except_failure):
                with self.assertRaises(ValueError) as caught:
                    r.render_one(out, ctx, receipt_bad, schedule, ev, FakeRenderer(), font)
            self.assertIs(caught.exception, original_error)
            self.assertTrue(any('secondary synthetic receipt failure' in note for note in original_error.__notes__))
            self.assertTrue((out/'renders/receipt-save-failure/matched.mp4').exists())
            # Tampered decoded review media cannot be silently skipped on restart.
            shot = out/'renders/full'/receipt['media']['samples'][0]['path']
            shot.chmod(0o644); shot.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Changed test artifact'):
                r.verified_render(out, first, ctx, ev)

if __name__ == '__main__':
    unittest.main()
