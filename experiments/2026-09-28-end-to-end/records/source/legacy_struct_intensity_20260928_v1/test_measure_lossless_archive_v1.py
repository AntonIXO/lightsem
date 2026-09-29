#!/usr/bin/env python3
"""Pure synthetic format/contract tests; source authoring is not a test result.

Future execution reads only the sibling driver SOURCE before installing guards.
No scientific fixture, real metadata JSON, checkpoint, cache, vault, or CLI is
opened. Every NPZ/NPY payload below exists only in small in-memory byte streams.
The driver source is executed under guards; its operational entry points are
blocked for each test. A plain native unittest result is still required later.
"""
from __future__ import annotations

# Eager imports precede all guards, including NumPy's format loader. Do not use
# numpy.testing: its lazy package inspection can read installed-package files.
import argparse
import ast
import builtins
import contextlib
import copy
import hashlib
import io
import json
import lzma  # Eagerly initialize the second declared ZIP codec before guards.
import math
import os
from pathlib import Path
import re
import struct
import stat
import sys
import time
import types
import unittest
from unittest import mock
import zipfile
import zlib
import numpy as np
from numpy.lib import format as npformat

DRIVER_PATH = Path(__file__).with_name('measure_lossless_archive_v1.py')
EXPECTED_TEST_COUNT = 67  # Authored method count, NOT an execution attestation.
REAL_IMPORT = builtins.__import__
REAL_NUMPY_LOAD = np.load
REAL_FILE_IO = io.FileIO
# Force any lazy format-attribute imports before the filesystem is closed.
NPY_HEADER_WRITER = npformat.write_array_header_1_0


def forbidden_io(*args, **kwargs):
    raise AssertionError('filesystem/process/scientific I/O forbidden in pure tests')


class NoDiskFileIO(REAL_FILE_IO):
    def __init__(self, *args, **kwargs):
        forbidden_io(*args, **kwargs)


def memory_load(file, *args, **kwargs):
    if not isinstance(file, io.BytesIO):
        raise AssertionError('NumPy loads must use an in-memory BytesIO')
    return REAL_NUMPY_LOAD(file, *args, **kwargs)


def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'lightsem', 'tensorflow', 'jax', 'cupy',
                              'pickle', 'joblib', 'subprocess', 'socket'}:
        raise AssertionError('scientific/process import forbidden: ' + name)
    return REAL_IMPORT(name, *args, **kwargs)


@contextlib.contextmanager
def source_guards():
    """No path, file descriptor, subprocess, or NumPy file API is available."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(builtins, 'open', side_effect=forbidden_io))
        stack.enter_context(mock.patch.object(builtins, '__import__', side_effect=guarded_import))
        for name in ('open', 'open_code'):
            stack.enter_context(mock.patch.object(io, name, side_effect=forbidden_io))
        stack.enter_context(mock.patch.object(io, 'FileIO', NoDiskFileIO))
        for name in ('open', 'fdopen', 'read', 'write', 'pread', 'pwrite', 'readv',
                     'writev', 'stat', 'lstat', 'fstat', 'statvfs', 'fstatvfs',
                     'listdir', 'scandir', 'readlink', 'mkdir', 'makedirs',
                     'unlink', 'remove', 'rmdir', 'removedirs', 'rename', 'replace',
                     'link', 'symlink', 'truncate', 'ftruncate', 'chmod', 'fchmod',
                     'chown', 'fchown', 'utime', 'fsync', 'fdatasync', 'chdir',
                     'system', 'popen', 'fork', 'execv', 'execve', 'spawnv', 'spawnve'):
            if hasattr(os, name):
                stack.enter_context(mock.patch.object(os, name, side_effect=forbidden_io))
        for name in ('open', 'read_text', 'read_bytes', 'write_text', 'write_bytes',
                     'iterdir', 'glob', 'rglob', 'stat', 'lstat', 'touch', 'mkdir',
                     'unlink', 'rmdir', 'rename', 'replace', 'resolve'):
            stack.enter_context(mock.patch.object(Path, name, side_effect=forbidden_io))
        for name in ('fromfile', 'memmap', 'save', 'savez', 'savez_compressed',
                     'loadtxt', 'savetxt', 'genfromtxt'):
            stack.enter_context(mock.patch.object(np, name, side_effect=forbidden_io))
        stack.enter_context(mock.patch.object(np, 'load', side_effect=memory_load))
        stack.enter_context(mock.patch.object(npformat, 'open_memmap', side_effect=forbidden_io))
        yield


class LosslessArchiveV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sole future source read. There are no plan/readiness/data file reads.
        cls.source = DRIVER_PATH.read_text(encoding='utf-8')
        cls.tree = ast.parse(cls.source, filename=str(DRIVER_PATH))
        cls.driver = types.ModuleType('synthetic_lossless_archive_v1')
        cls.driver.__file__ = str(DRIVER_PATH)
        with source_guards():
            exec(compile(cls.tree, str(DRIVER_PATH), 'exec'), cls.driver.__dict__)
        # The dispatcher regression uses this pure function only with its
        # metadata validator and all candidate work replaced by in-memory mocks.
        cls.measure_function = staticmethod(cls.driver.measure)
        # This version-only line is for the later root's prospective binding.
        # It attests no test result and calls no scientific or operational API.
        version_line = 'CODEC_VERSIONS=' + json.dumps(cls.driver.codec_versions(), sort_keys=True)
        if len(version_line.encode('ascii')) > 2048:
            raise AssertionError('codec version line exceeds its source bound')
        print(version_line, flush=True)

    def setUp(self):
        self.d = self.driver
        self.enterContext(source_guards())
        for name in ('read_control', 'write_receipt', 'no_symlink', 'main', 'measure'):
            self.enterContext(mock.patch.object(self.d, name, side_effect=forbidden_io))
        self.plan, self.ready, self.contract = self.metadata()
        self.rows = self.ready['all40_eligibility']

    @staticmethod
    def fake_ref(name):
        return {'path': '/SYNTHETIC_NEVER_OPEN/' + name,
                'size_bytes': 1, 'sha256': 'a' * 64}

    def metadata(self):
        d = self.d
        sections = [
            [{'start_frame': 1, 'end_frame_exclusive': 81, 'intersection_frames': 80,
              'section_id': 7, 'role': 'breakdown', 'retained': True, 'reason': None},
             {'start_frame': 83, 'end_frame_exclusive': 85, 'intersection_frames': 2,
              'section_id': 42, 'role': 'build-up', 'retained': False,
              'reason': 'INTERSECTION_LT80_FRAMES'},
             {'start_frame': 86, 'end_frame_exclusive': 166, 'intersection_frames': 80,
              'section_id': 90, 'role': 'drop', 'retained': True, 'reason': None}],
            [{'start_frame': 0, 'end_frame_exclusive': 82, 'intersection_frames': 82,
              'section_id': 13, 'role': 'breakdown', 'retained': True, 'reason': None}],
        ]
        frozen, rows = [], []
        for i, T in enumerate((166, 82)):
            start = 0.013 if i == 0 else 1.0
            i0 = round(start * 40)
            frozen.append({'clip': 'synthetic_%02d' % i, 'T_metadata': T,
                           'recording_id': 'recording_%d' % i,
                           'original_shuffled_donor_recording_id': 'donor_%d' % i,
                           'start_s_metadata': start, 'current_partition': 'train',
                           'current_Stage11_TEST_aliases': [],
                           'opaque_frozen_field': {'keep': ['original', i]}})
            retained = {role: 0 for role in d.ROLES}
            for section in sections[i]:
                if section['retained']:
                    retained[section['role']] += section['intersection_frames']
            valid = sum(s['intersection_frames'] for s in sections[i])
            rows.append({'clip': frozen[i]['clip'], 'recording_id': frozen[i]['recording_id'],
                         'donor_recording_id': frozen[i]['original_shuffled_donor_recording_id'],
                         'T': T, 'fps': 40, 'raw_start_s': start, 'i0': i0,
                         'rounded_origin_s': i0 / 40, 'roundoff_s': i0 / 40 - start,
                         'eligible': i == 0,
                         'reasons': [] if i == 0 else ['DROP_RETAINED_LT80_FRAMES'],
                         'status': 'INPUT_QUALIFIED' if i == 0 else 'ROLE_SUPPORT_INELIGIBLE',
                         'ratio_or_prediction_computed': False, 'sections': sections[i],
                         'target_valid_role_frames': valid,
                         'target_invalid_role_frames': T - valid,
                         'retained_role_frames': retained,
                         'opaque_qualification_field': {'preserve': [i, 'not discarded']}})
        plan = {'document_revision': 6, 'cohorts': {'clip_index': frozen},
                'opaque_plan_field': {'no_schema_projection_loss': True}}
        ready = {'status': 'PREPARATION_COMPLETE_NO_SCIENTIFIC_EVALUATION',
                 'GPU_or_study_admission_granted': False,
                 'model_constructions': 0, 'model_forwards': 0, 'GPU_jobs': 0,
                 'TEST_payload_reads': 0, 'raw_manifest_reads': 0, 'replacement_clips': 0,
                 'original40_inventory_preserved': True, 'science_scores_computed': 0,
                 'plan_ref': {'path': str(d.PLAN), 'sha256': d.PLAN_SHA, 'size_bytes': 285563},
                 'all40_eligibility': rows,
                 'input_refs': [self.fake_ref('input0.npz'), self.fake_ref('input1.npz')],
                 'source_refs': [self.fake_ref('source.py')],
                 'checkpoint_refs': {'none': self.fake_ref('never_loaded.pt')},
                 'opaque_readiness_field': {'preserve': ['every', 'field']}}
        contract = {'clips': 2, 'frames': 248, 'sections': 4, 'invalid': 4,
                    'ineligible': 1, 'short_sections': 1, 'input_refs': 2}
        return plan, ready, contract

    def validate(self):
        # This is the ONLY place a small contract replaces the frozen counts.
        return self.d.validate_metadata(self.plan, self.ready, contract=self.contract)

    def encoded_members(self):
        entries = []
        for i, row in enumerate(self.rows):
            vectors = self.d.integer_vectors(row)
            for field, dtype in self.d.FIELDS:
                name = 'c%02d_%s.npy' % (i, field)
                entries.append((name, self.d.encode_npy(vectors[field], dtype, row['T'])))
        return entries

    @staticmethod
    def make_zip(entries, compression=zipfile.ZIP_DEFLATED, comment=b''):
        sink = io.BytesIO()
        with zipfile.ZipFile(sink, 'w', allowZip64=False) as archive:
            archive.comment = comment
            for name, payload in entries:
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o600 << 16
                info.compress_type = compression
                archive.writestr(info, payload, compress_type=compression, compresslevel=9)
        return sink.getvalue()

    @staticmethod
    def with_compressed_trailer(raw, trailer=b'JUNK', keep_bytes=None):
        # Insert bytes (or truncate to keep_bytes) INSIDE the first compressed
        # range, while keeping all local/central sizes and offsets consistent.
        # This is not an EOCD test: the decoder must enforce the stream EOF.
        changed = bytearray(raw)
        local = struct.unpack('<4s5H3L2H', changed[:30])
        central_before = struct.unpack('<4s4H2LH', changed[-22:])[6]
        count = struct.unpack('<4s4H2LH', changed[-22:])[4]
        start = 30 + local[9] + local[10]
        end = start + local[7]
        if keep_bytes is None:
            changed[end:end] = trailer
            delta = len(trailer)
        else:
            if not 0 <= keep_bytes < local[7]:
                raise AssertionError('synthetic truncation must shorten this stream')
            del changed[start + keep_bytes:end]
            delta = keep_bytes - local[7]
        struct.pack_into('<L', changed, 18, local[7] + delta)
        cursor = central_before + delta
        for i in range(count):
            if i == 0:
                size = struct.unpack_from('<L', changed, cursor + 20)[0]
                struct.pack_into('<L', changed, cursor + 20, size + delta)
            else:
                offset = struct.unpack_from('<L', changed, cursor + 42)[0]
                struct.pack_into('<L', changed, cursor + 42, offset + delta)
            name_n, extra_n, comment_n = struct.unpack_from('<3H', changed, cursor + 28)
            cursor += 46 + name_n + extra_n + comment_n
        struct.pack_into('<L', changed, len(changed) - 6, central_before + delta)
        return bytes(changed)

    def archive(self):
        return self.d.encode_integer_archive(self.rows)[0]

    def resources(self):
        d = self.d
        return {'write_pattern': 'WRITE_ONCE_FLOAT_NPY_PLUS_SINGLE_INTEGER_NPZ',
                'float_payload_copy_count': 1,
                'future_unspent_code_control_metrics_bytes': d.FUTURE_RESERVE,
                'distinct_GPU_ledger_charge_bytes': d.GPU_LEDGER_RESERVE,
                'header_reservation_bytes': d.HEADER_RESERVE,
                'table_manifest_reservation_bytes': d.TABLE_MANIFEST_RESERVE,
                'retained_packet_bytes_excluding_distinct_GPU_ledger_charge': 100000,
                'other_unwritten_control_reserve_bytes': 10000,
                'global_existing_plus_other_reservations_excluding_row8_growth_and_GPU_ledger': 200000,
                'free_root_bytes': 2**30 + d.PACKET_CAP,
                'free_mnt_bytes': 2**30 + d.PACKET_CAP,
                'all_retained_and_prototype_controls_counted': True,
                'prior_failures_and_eight_protocols_preserved': True}

    def authorization(self):
        d = self.d
        return {'schema_version': 1, 'status': 'ROOT_AUTHORIZED_CPU_FORMAT_PROTOTYPE_ONLY',
                'scope': 'METADATA_INTEGER_CODEC_ONLY_NO_SCIENCE',
                'source_binding': {'driver_sha256': 'b' * 64, 'tests_sha256': 'c' * 64,
                                   'test_count': EXPECTED_TEST_COUNT,
                                   'root_verified_source_bytes': True},
                'native_test_attestation': {'exit_code': 0, 'tests_run': EXPECTED_TEST_COUNT,
                    'failures': 0, 'errors': 0, 'skipped': 0, 'expected_failures': 0,
                    'unexpected_successes': 0, 'plain_OK': True,
                    'native_receipt_ref': {
                        'path': str(d.ROOT / 'NATIVE_LOSSLESS_ARCHIVE_TEST_RESULT_V1.json'),
                        'sha256': 'e' * 64, 'size_bytes': 100}},
                'codec_versions': d.codec_versions(),
                'plan_ref': {'path': str(d.PLAN), 'sha256': d.PLAN_SHA, 'size_bytes': 285563},
                'readiness_ref': {'path': str(d.READY), 'sha256': d.READY_SHA, 'size_bytes': 202345},
                'envelope_ref': {'path': str(d.ROOT / 'LOSSLESS_ARCHIVE_EXTERNAL_ENVELOPE_V1.json'),
                                 'sha256': 'd' * 64, 'size_bytes': 100},
                'resources': self.resources(), 'snapshot_epoch_s': 1000,
                'root_attests_fresh_exact_command_accounting': True}

    def envelope(self):
        d = self.d
        return {'status': 'ENFORCED_BEFORE_EXEC',
                'module_cli_argv': [str(d.DRIVER), 'measure', '--authorization', str(d.AUTH)],
                'MemoryMax_bytes': 536870912, 'MemorySwapMax_bytes': 0,
                'RuntimeMaxSec': 50, 'acceptance_wall_seconds': 60,
                'PrivateDevices': True, 'GPU_devices_visible': False,
                'CUDA_VISIBLE_DEVICES': '', 'process_tree_enforced': True}

    def function(self, name):
        return next(node for node in self.tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name)

    @staticmethod
    def call_name(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return LosslessArchiveV1Tests.call_name(node.value) + '.' + node.attr
        return ''

    def test_01_integer_bounds_and_exact_types(self):
        self.assertEqual(self.d.integer(0, 0, 2), 0)
        self.assertEqual(self.d.integer(2, 0, 2), 2)
        for bad in (-1, 3, True, False, 1.0, '1', np.int64(1)):
            with self.assertRaises(self.d.ContractError):
                self.d.integer(bad, 0, 2)

    def test_02_finite_values_reject_bool_nan_infinity(self):
        self.assertEqual(self.d.finite(-0.25), -0.25)
        self.assertEqual(self.d.finite(3), 3)
        for bad in (True, False, float('nan'), float('inf'), -float('inf'), '0'):
            with self.assertRaises(self.d.ContractError):
                self.d.finite(bad)

    def test_03_bounded_json_exact_byte_cap_and_plain_types(self):
        raw = b'{"n":3,"x":-0.25,"ok":true,"z":null,"a":[1,2]}'
        value = self.d.bounded_json(raw, len(raw))
        self.assertEqual(value, {'n': 3, 'x': -0.25, 'ok': True, 'z': None, 'a': [1, 2]})
        with self.assertRaises(self.d.ContractError):
            self.d.bounded_json(raw, len(raw) - 1)
        with self.assertRaises(self.d.ContractError):
            self.d.bounded_json(raw.decode('ascii'), len(raw))

    def test_04_json_duplicate_keys_at_each_depth_rejected(self):
        for raw in (b'{"x":1,"x":2}', b'{"a":[{"x":1,"x":2}]}'):
            with self.assertRaises(self.d.ContractError):
                self.d.bounded_json(raw, 128)

    def test_05_json_nonfinite_tokens_and_exponent_overflow_rejected(self):
        for token in (b'NaN', b'Infinity', b'-Infinity', b'1e999'):
            with self.assertRaises(self.d.ContractError):
                self.d.bounded_json(b'{"x":' + token + b'}', 128)

    def test_06_json_depth_cap_is_enforced(self):
        raw = b'[' * (self.d.MAX_JSON_DEPTH + 1) + b'0' + b']' * (self.d.MAX_JSON_DEPTH + 1)
        with self.assertRaises(self.d.ContractError):
            self.d.bounded_json(raw, 128)

    def test_07_json_list_and_node_count_caps(self):
        raw = b'[' + b'0,' * 10000 + b'0]'
        with self.assertRaises(self.d.ContractError):
            self.d.bounded_json(raw, len(raw))
        self.assertEqual(self.d.MAX_JSON_NODES, 100000)
        with mock.patch.object(self.d, 'MAX_JSON_NODES', 4):
            with self.assertRaises(self.d.ContractError):
                self.d.bounded_json(b'[0,1,2,3]', 32)

    def test_08_json_key_and_string_caps(self):
        for value in ({'k' * 257: 0}, {'k': 'x' * 16385}):
            raw = json.dumps(value).encode('utf-8')
            with self.assertRaises(self.d.ContractError):
                self.d.bounded_json(raw, len(raw))

    def test_09_small_metadata_validates_without_mutation(self):
        before = copy.deepcopy((self.plan, self.ready, self.contract))
        self.assertEqual(self.validate(), self.rows)
        self.assertEqual((self.plan, self.ready, self.contract), before)
        self.assertEqual(self.rows[0]['status'], 'INPUT_QUALIFIED')
        self.assertEqual(self.rows[1]['status'], 'ROLE_SUPPORT_INELIGIBLE')
        self.assertIs(type(self.ready['science_scores_computed']), int)

    def test_10_small_metadata_cannot_replace_production_frozen_contract(self):
        self.assertEqual(self.d.FROZEN, {'clips': 40, 'frames': 371393, 'sections': 613,
            'invalid': 3, 'ineligible': 7, 'short_sections': 1, 'input_refs': 62})
        with self.assertRaises(self.d.ContractError):
            self.d.validate_metadata(self.plan, self.ready)

    def test_11_metadata_overlapping_sections_rejected(self):
        self.rows[0]['sections'][1]['start_frame'] = 80
        self.rows[0]['sections'][1]['intersection_frames'] = 5
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_12_metadata_malformed_or_out_of_range_interval_rejected(self):
        original = copy.deepcopy(self.rows[0]['sections'][0])
        for a, b in ((5, 5), (9, 4), (-1, 80), (0, 167), (False, 81)):
            section = dict(original, start_frame=a, end_frame_exclusive=b,
                           intersection_frames=b - a)
            self.rows[0]['sections'][0] = section
            with self.assertRaises(self.d.ContractError):
                self.validate()
        self.rows[0]['sections'][0] = original

    def test_13_metadata_global_and_local_count_drift_rejected(self):
        for key in ('frames', 'sections', 'invalid', 'ineligible', 'short_sections', 'input_refs'):
            original = self.contract[key]
            self.contract[key] += 1
            with self.assertRaises(self.d.ContractError):
                self.validate()
            self.contract[key] = original
        self.rows[0]['sections'][0]['intersection_frames'] = 79
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_14_metadata_original_ids_must_increase_without_renumbering(self):
        self.assertEqual([s['section_id'] for s in self.validate()[0]['sections']], [7, 42, 90])
        for bad in (7, 6, 65535, -1, True):
            self.rows[0]['sections'][1]['section_id'] = bad
            with self.assertRaises(self.d.ContractError):
                self.validate()

    def test_15_uncovered_gap_frames_keep_exact_sentinels(self):
        row = self.validate()[0]
        vectors = self.d.integer_vectors(row)
        for index in (0, 81, 82, 85):
            self.assertEqual(int(vectors['target_role'][index]), 255)
            self.assertEqual(int(vectors['section_id'][index]), 65535)
        self.assertEqual(int((vectors['target_role'] == 255).sum()), 4)
        self.assertEqual(int((vectors['section_id'] == 65535).sum()), 4)

    def test_16_rejected_short_sections_preserve_role_and_original_id(self):
        row = self.validate()[0]
        vectors = self.d.integer_vectors(row)
        self.assertEqual(vectors['target_role'][83:85].tobytes(), bytes([2, 2]))
        self.assertEqual(vectors['section_id'][83:85].tobytes(), struct.pack('<2H', 42, 42))
        self.assertFalse(row['sections'][1]['retained'])
        self.assertEqual(row['retained_role_frames']['build-up'], 0)
        row['sections'][1]['retained'] = True
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_17_metadata_clip_names_order_and_duplicate_inventory_rejected(self):
        self.ready['all40_eligibility'] = list(reversed(self.rows))
        with self.assertRaises(self.d.ContractError):
            self.validate()
        self.ready['all40_eligibility'] = self.rows
        self.rows[1]['clip'] = self.rows[0]['clip']
        self.plan['cohorts']['clip_index'][1]['clip'] = self.rows[0]['clip']
        with self.assertRaises(self.d.ContractError):
            self.validate()
        self.rows[0]['clip'] = '../escape'
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_18_metadata_donor_clock_partition_and_alias_drift_rejected(self):
        for field, value in (('donor_recording_id', 'replacement'), ('i0', 0),
                             ('fps', 40.0), ('rounded_origin_s', 10), ('T', 165)):
            previous = self.rows[0][field]
            self.rows[0][field] = value
            with self.assertRaises(self.d.ContractError):
                self.validate()
            self.rows[0][field] = previous
        old = self.plan['cohorts']['clip_index'][0]
        old['current_partition'] = 'TEST'
        with self.assertRaises(self.d.ContractError):
            self.validate()
        old['current_partition'] = 'train'
        old['current_Stage11_TEST_aliases'] = ['known_test']
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_19_metadata_eligibility_role_coverage_reason_drift_rejected(self):
        for field, value in (('eligible', False), ('status', 'ELIGIBLE'),
                             ('target_valid_role_frames', 163), ('target_invalid_role_frames', 3)):
            previous = self.rows[0][field]
            self.rows[0][field] = value
            with self.assertRaises(self.d.ContractError):
                self.validate()
            self.rows[0][field] = previous
        self.rows[0]['sections'][1]['reason'] = None
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_20_metadata_science_flags_require_exact_no_execution_values(self):
        for field in ('model_constructions', 'model_forwards', 'GPU_jobs',
                      'TEST_payload_reads', 'raw_manifest_reads', 'replacement_clips',
                      'science_scores_computed'):
            for bad in (1, False):
                self.ready[field] = bad
                with self.assertRaises(self.d.ContractError):
                    self.validate()
            self.ready[field] = 0
        self.ready['GPU_or_study_admission_granted'] = True
        with self.assertRaises(self.d.ContractError):
            self.validate()
        self.ready['GPU_or_study_admission_granted'] = False
        self.rows[0]['ratio_or_prediction_computed'] = True
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_21_metadata_revision_readiness_status_and_frozen_ref(self):
        for bad in ('V6', 5, True):
            self.plan['document_revision'] = bad
            with self.assertRaises(self.d.ContractError):
                self.validate()
        self.plan['document_revision'] = 6
        self.ready['status'] = 'SCIENTIFIC_EVALUATION_COMPLETE'
        with self.assertRaises(self.d.ContractError):
            self.validate()
        self.ready['status'] = 'PREPARATION_COMPLETE_NO_SCIENTIFIC_EVALUATION'
        self.ready['plan_ref']['sha256'] = '0' * 64
        with self.assertRaises(self.d.ContractError):
            self.validate()

    def test_22_integer_schema_has_exact_ordered_names_dtypes_shapes(self):
        expected = {}
        for i, T in enumerate((166, 82)):
            for field, dtype in (('frame_index', '<i4'), ('target_role', '|u1'), ('section_id', '<u2')):
                expected['c%02d_%s.npy' % (i, field)] = (dtype, T, i, field)
        schema = self.d.integer_schema(self.validate())
        self.assertEqual(schema, expected)
        self.assertEqual(list(schema), list(expected))
        self.assertEqual(len(schema), 6)

    def test_23_vectors_have_exact_dtype_shape_and_expected_payload_bits(self):
        vectors = self.d.integer_vectors(self.validate()[0])
        self.assertEqual(list(vectors), ['frame_index', 'target_role', 'section_id'])
        expected_roles = bytes([255] + [1] * 80 + [255, 255, 2, 2, 255] + [3] * 80)
        expected_ids = [65535] + [7] * 80 + [65535, 65535, 42, 42, 65535] + [90] * 80
        self.assertEqual(vectors['frame_index'].tobytes(), struct.pack('<166i', *range(166)))
        self.assertEqual(vectors['target_role'].tobytes(), expected_roles)
        self.assertEqual(vectors['section_id'].tobytes(), struct.pack('<166H', *expected_ids))
        for field, dtype in self.d.FIELDS:
            self.assertEqual(vectors[field].dtype.str, dtype)
            self.assertEqual(vectors[field].shape, (166,))
            self.assertTrue(vectors[field].flags.c_contiguous)

    def test_24_npy_header_magic_shape_dtype_and_fixed_header_cap(self):
        for dtype in ('<i4', '|u1', '<u2', '<f4'):
            header = self.d.npy_header(dtype, 3)
            self.assertEqual(header[:8], b'\x93NUMPY\x01\x00')
            self.assertLessEqual(len(header), 256)
            self.assertEqual(len(header), 10 + struct.unpack('<H', header[8:10])[0])
            parsed = ast.literal_eval(header[10:].decode('ascii').strip())
            self.assertEqual(parsed, {'descr': dtype, 'fortran_order': False, 'shape': (3,)})

    def test_25_npy_header_forbidden_dtype_and_frame_bounds(self):
        for dtype in ('<f8', '>f4', '|O', '<i8', '|S4'):
            with self.assertRaises(self.d.ContractError):
                self.d.npy_header(dtype, 3)
        for T in (0, -1, 11404, True, 2.0):
            with self.assertRaises(self.d.ContractError):
                self.d.npy_header('<f4', T)

    def test_26_npy_integer_roundtrip_uses_plain_byte_equality(self):
        row = self.validate()[0]
        vectors = self.d.integer_vectors(row)
        for field, dtype in self.d.FIELDS:
            original = vectors[field]
            raw = self.d.encode_npy(original, dtype, row['T'])
            restored = self.d.decode_npy(raw, dtype, row['T'])
            self.assertEqual(restored.tobytes(), original.tobytes())
            self.assertEqual(restored.dtype.str, dtype)
            self.assertEqual(restored.shape, (row['T'],))

    def test_27_float32_signed_zero_and_tiny_payload_bits_are_unchanged(self):
        bits = (0x00000000, 0x80000000, 0x00000001, 0x80000001,
                0x007FFFFF, 0x00800000, 0x3F800001, 0xBF800001)
        payload = struct.pack('<8I', *bits)
        original = np.frombuffer(payload, dtype='<f4').copy()
        raw = self.d.encode_npy(original, '<f4', len(bits))
        self.assertEqual(raw[len(self.d.npy_header('<f4', len(bits))):], payload)
        restored = self.d.decode_npy(raw, '<f4', len(bits))
        self.assertEqual(restored.tobytes(), payload)
        self.assertEqual(struct.unpack('<8I', restored.tobytes()), bits)
        self.assertTrue(math.copysign(1.0, float(restored[1])) < 0)

    def test_28_npy_wrong_endianness_or_dtype_is_rejected(self):
        for dtype in ('>f4', '<f8', '<i4'):
            value = np.array([0, 1, 2], dtype=dtype)
            with self.assertRaises(self.d.ContractError):
                self.d.encode_npy(value, '<f4', 3)
        raw = self.d.encode_npy(np.array([0, 1, 2], dtype='<i4'), '<i4', 3)
        with self.assertRaises(self.d.ContractError):
            self.d.decode_npy(raw, '<f4', 3)

    def test_29_npy_wrong_shape_and_noncontiguous_views_rejected(self):
        for value in (np.zeros((3, 1), dtype='<f4'), np.zeros(4, dtype='<f4'),
                      np.arange(6, dtype='<f4')[::2]):
            with self.assertRaises(self.d.ContractError):
                self.d.encode_npy(value, '<f4', 3)

    def test_30_npy_noncanonical_or_corrupt_header_rejected(self):
        raw = self.d.encode_npy(np.zeros(3, dtype='<f4'), '<f4', 3)
        for offset in (0, 6, 10):
            changed = bytearray(raw)
            changed[offset] ^= 1
            with self.assertRaises(self.d.ContractError):
                self.d.decode_npy(bytes(changed), '<f4', 3)
        wrong_shape = raw.replace(b'(3,)', b'(4,)', 1)
        self.assertNotEqual(wrong_shape, raw)
        with self.assertRaises(self.d.ContractError):
            self.d.decode_npy(wrong_shape, '<f4', 3)

    def test_31_npy_truncation_and_trailing_data_rejected(self):
        raw = self.d.encode_npy(np.zeros(3, dtype='<f4'), '<f4', 3)
        for changed in (raw[:-1], raw + b'\0', raw + raw):
            with self.assertRaises(self.d.ContractError):
                self.d.decode_npy(changed, '<f4', 3)

    def test_32_bounded_bytesio_rewrites_preserve_high_water(self):
        with self.d.BoundedBytesIO(8) as sink:
            self.assertEqual(sink.write(b'abcd'), 4)
            self.assertEqual(sink.seek(1), 1)
            sink.write(b'X')
            self.assertEqual(sink.high_water, 4)
            self.assertEqual(sink.getvalue(), b'aXcd')
            sink.seek(4)
            sink.write(b'efgh')
            self.assertEqual(sink.high_water, 8)
            self.assertEqual(sink.getvalue(), b'aXcdefgh')

    def test_33_bounded_bytesio_write_limit_is_checked_before_growth(self):
        with self.d.BoundedBytesIO(4) as sink:
            sink.write(b'abcd')
            with self.assertRaises(self.d.ContractError):
                sink.write(b'e')
            self.assertEqual(sink.getvalue(), b'abcd')
            self.assertEqual(sink.high_water, 4)
        for limit in (0, True, self.d.MAX_ARCHIVE + 1):
            with self.assertRaises(self.d.ContractError):
                self.d.BoundedBytesIO(limit)

    def test_34_bounded_bytesio_seek_and_truncate_caps(self):
        with self.d.BoundedBytesIO(4) as sink:
            sink.write(b'abcd')
            sink.seek(2)
            self.assertEqual(sink.truncate(), 2)
            self.assertEqual(sink.getvalue(), b'ab')
            with self.assertRaises(self.d.ContractError):
                sink.truncate(5)
            with self.assertRaises(self.d.ContractError):
                sink.seek(5)
            with self.assertRaises(ValueError):
                sink.seek(-1)

    def test_35_integer_archive_exact_schema_counts_sizes_and_roundtrip(self):
        self.validate()
        raw, high_water = self.d.encode_integer_archive(self.rows)
        with mock.patch.object(self.d, 'NP_LOAD', wraps=self.d.NP_LOAD) as loader:
            info = self.d.inspect_integer_archive(raw, self.rows)
        whole_loads = [call for call in loader.call_args_list
                       if isinstance(call.args[0], io.BytesIO)
                       and call.args[0].getvalue()[:4] == b'PK\x03\x04']
        self.assertEqual(len(whole_loads), 1)
        self.assertIs(whole_loads[0].kwargs['allow_pickle'], False)
        self.assertIs(info['NumPy_standard_NPZ_roundtrip'], True)
        self.assertEqual(high_water, len(raw))
        self.assertEqual(info['members'], 6)
        self.assertEqual(info['integer_raw_data_bytes'], 248 * 7)
        headers = sum(len(self.d.npy_header(dtype, row['T']))
                      for row in self.rows for _, dtype in self.d.FIELDS)
        self.assertEqual(info['integer_NPY_expanded_bytes'], 248 * 7 + headers)
        self.assertEqual(info['integer_archive_bytes'], len(raw))
        self.assertEqual(info['ZIP_container_bytes'] + info['compressed_integer_NPY_stream_bytes'], len(raw))
        self.assertEqual(info['integer_archive_sha256'], hashlib.sha256(raw).hexdigest())
        # Actual NumPy whole-NPZ loading is allowed ONLY with this validated,
        # small in-memory archive, never a path or scientific fixture.
        with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=256) as loaded:
            self.assertEqual(loaded.files, [name[:-4] for name in self.d.integer_schema(self.rows)])
            for name, (dtype, T, clip_index, field) in self.d.integer_schema(self.rows).items():
                value = loaded[name[:-4]]
                self.assertEqual(value.dtype.str, dtype)
                self.assertEqual(value.shape, (T,))
                self.assertEqual(value.tobytes(), self.d.integer_vectors(self.rows[clip_index])[field].tobytes())
        self.assertEqual(list(info['member_NPY_sha256']), list(self.d.integer_schema(self.rows)))
        self.assertIs(info['bytewise_roundtrip'], True)

    def test_36_integer_archive_deterministic_bytes_and_member_hashes(self):
        first, second = self.archive(), self.archive()
        self.assertEqual(first, second)
        info = self.d.inspect_integer_archive(first, self.rows)
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            for name, payload in self.encoded_members():
                self.assertEqual(archive.read(name), payload)
                self.assertEqual(info['member_NPY_sha256'][name], hashlib.sha256(payload).hexdigest())

    def test_37_integer_archive_extra_member_rejected(self):
        entries = self.encoded_members() + [('extra.npy', b'not authorized')]
        with self.assertRaises(self.d.ContractError):
            self.d.inspect_integer_archive(self.make_zip(entries), self.rows)

    def test_38_integer_archive_missing_member_rejected(self):
        with self.assertRaises(self.d.ContractError):
            self.d.inspect_integer_archive(self.make_zip(self.encoded_members()[:-1]), self.rows)

    def test_39_integer_archive_reordered_or_duplicate_names_rejected(self):
        entries = self.encoded_members()
        entries[0], entries[1] = entries[1], entries[0]
        with self.assertRaises(self.d.ContractError):
            self.d.inspect_integer_archive(self.make_zip(entries), self.rows)
        raw = self.archive()
        # Equal-width names allow a duplicate local/central filename without
        # asking ZipFile.writestr to emit warnings or touch a warning registry.
        duplicate = raw.replace(b'c01_frame_index.npy', b'c00_frame_index.npy')
        with self.assertRaises(self.d.ContractError):
            self.d.inspect_integer_archive(duplicate, self.rows)

    def test_40_integer_archive_unsafe_names_rejected(self):
        for name in ('../c00_frame_index.npy', '/absolute.npy', 'a\\b.npy', 'directory/'):
            entries = self.encoded_members()
            entries[0] = (name, entries[0][1])
            with self.assertRaises(self.d.ContractError):
                self.d.inspect_integer_archive(self.make_zip(entries), self.rows)

    def test_41_integer_archive_bad_crc_is_not_accepted(self):
        raw = bytearray(self.archive())
        central = struct.unpack('<4s4H2LH', raw[-22:])[6]
        crc = struct.unpack_from('<L', raw, 14)[0] ^ 1
        struct.pack_into('<L', raw, 14, crc)
        struct.pack_into('<L', raw, central + 16, crc)
        with self.assertRaises(zipfile.BadZipFile):
            self.d.inspect_integer_archive(bytes(raw), self.rows)

    def test_42_integer_archive_corrupt_compressed_payload_rejected(self):
        raw = bytearray(self.archive())
        local = struct.unpack('<4s5H3L2H', raw[:30])
        payload_offset = 30 + local[9] + local[10]
        raw[payload_offset] ^= 0xFF
        with self.assertRaises((self.d.ContractError, zipfile.BadZipFile, zlib.error, EOFError)):
            self.d.inspect_integer_archive(bytes(raw), self.rows)

    def test_43_integer_archive_declared_oversize_and_external_byte_cap(self):
        raw = bytearray(self.archive())
        central = struct.unpack('<4s4H2LH', raw[-22:])[6]
        struct.pack_into('<L', raw, 22, 0x7FFFFFFF)
        struct.pack_into('<L', raw, central + 24, 0x7FFFFFFF)
        with self.assertRaises(self.d.ContractError):
            self.d.inspect_integer_archive(bytes(raw), self.rows)
        small = self.archive()
        with mock.patch.object(self.d, 'MAX_ARCHIVE', len(small) - 1):
            with self.assertRaises(self.d.ContractError):
                self.d.inspect_integer_archive(small, self.rows)
        with self.assertRaises(self.d.ContractError):
            self.d.encode_integer_archive(self.rows, cap=64)

    def test_44_integer_archive_bad_npy_header_with_valid_zip_crc(self):
        entries = self.encoded_members()
        name, payload = entries[0]
        entries[0] = (name, b'?' + payload[1:])
        with self.assertRaises(self.d.ContractError):
            self.d.inspect_integer_archive(self.make_zip(entries), self.rows)

    def test_45_integer_archive_trailing_hidden_or_truncated_data_rejected(self):
        raw = self.archive()
        for changed in (raw + b'junk', b'junk' + raw, raw[:-1], raw[:-22],
                        self.with_compressed_trailer(raw)):
            with self.assertRaises((self.d.ContractError, zipfile.BadZipFile)):
                self.d.inspect_integer_archive(changed, self.rows)

    def test_46_integer_archive_wrong_dtype_shape_or_bits_with_valid_crc(self):
        for replacement in (
                self.d.encode_npy(np.zeros(166, dtype='<f4'), '<f4', 166),
                self.d.encode_npy(np.zeros(165, dtype='<i4'), '<i4', 165),
                self.d.encode_npy(np.zeros(166, dtype='<i4'), '<i4', 166)):
            entries = self.encoded_members()
            entries[0] = (entries[0][0], replacement)
            with self.assertRaises(self.d.ContractError):
                self.d.inspect_integer_archive(self.make_zip(entries), self.rows)

    def test_47_integer_archive_noncanonical_zip_method_comment_zip64_rejected(self):
        entries = self.encoded_members()
        for raw in (self.make_zip(entries, compression=zipfile.ZIP_STORED),
                    self.make_zip(entries, comment=b'not allowed')):
            with self.assertRaises(self.d.ContractError):
                self.d.inspect_integer_archive(raw, self.rows)
        raw = bytearray(self.archive())
        central = struct.unpack('<4s4H2LH', raw[-22:])[6]
        struct.pack_into('<H', raw, 4, 45)
        struct.pack_into('<H', raw, central + 6, 45)
        with self.assertRaises(self.d.ContractError):
            self.d.inspect_integer_archive(bytes(raw), self.rows)

    def test_48_table_bound_keeps_short_rows_and_finite_float_statistics_slots(self):
        bound = self.d.table_bound(self.validate())
        self.assertEqual(bound['rows'], 16)
        self.assertEqual(bound['columns'], list(self.d.CSV_COLUMNS))
        self.assertEqual(bound['float64_fields_per_row'], 2)
        self.assertEqual(bound['max_ASCII_bytes_per_unknown_float64'], 25)
        expected = len((','.join(self.d.CSV_COLUMNS) + '\n').encode('ascii'))
        for i, row in enumerate(self.rows):
            for arm in range(4):
                for section in row['sections']:
                    n = section['intersection_frames']
                    cells = (i, arm, section['section_id'], self.d.ROLES.index(section['role']),
                             section['start_frame'], section['end_frame_exclusive'], n,
                             n if section['retained'] else 0, int(section['retained']),
                             self.d.REASONS.index(section['reason']))
                    expected += len((','.join(map(str, cells)) + ',').encode('ascii')) + 52
        self.assertEqual(bound['max_bytes'], expected)
        self.assertIs(bound['statistics_computed'], False)
        self.assertNotIn('NA for', bound['float64_format'])
        for value in (0.0, -0.0, 5e-324, -5e-324, sys.float_info.max,
                      -sys.float_info.max, sys.float_info.min, 1.0000000000000002):
            text = self.d.format_finite_statistic(value)
            self.assertLessEqual(len(text.encode('ascii')), 25)
            self.assertEqual(struct.pack('<d', float(text)), struct.pack('<d', value))
            self.assertNotEqual(text, 'NA')
        self.assertTrue(self.d.format_finite_statistic(-0.0).startswith('-'))
        for bad in (0, True, 'NA', float('nan'), float('inf'), -float('inf')):
            with self.assertRaises(self.d.ContractError):
                self.d.format_finite_statistic(bad)

    def test_49_manifest_full_metadata_passthrough_without_writing(self):
        info = self.d.inspect_integer_archive(self.archive(), self.rows)
        with mock.patch.object(self.d, 'compact_json', wraps=self.d.compact_json) as capture:
            bound = self.d.manifest_bound(self.plan, self.ready, self.rows, info)
        self.assertEqual(capture.call_count, 1)
        manifest = capture.call_args.args[0]
        # The whole frozen index is retained. Only original section dictionaries
        # move to the CSV; ALL remaining readiness fields pass through intact.
        expected_ready = copy.deepcopy(self.ready)
        for row in expected_ready['all40_eligibility']:
            del row['sections']
        self.assertEqual(manifest['clip_index'], self.plan['cohorts']['clip_index'])
        self.assertEqual(manifest['readiness_except_section_table'], expected_ready)
        self.assertEqual(manifest['codec_versions'], self.d.codec_versions())
        decode = manifest['section_decode_contract']
        self.assertEqual(decode['recover_original_sections_from_arm'], 0)
        self.assertEqual(set(decode['original_section_keys']), set(self.rows[0]['sections'][0]))
        self.assertIn('no NA substitution', manifest['statistics_contract'])
        self.assertIs(bound['manifest_written'], False)
        self.assertIs(bound['unknown_hash_slots_are_size_only'], True)

    def test_50_manifest_exact_file_schema_hash_width_and_bound_arithmetic(self):
        info = self.d.inspect_integer_archive(self.archive(), self.rows)
        with mock.patch.object(self.d, 'compact_json', wraps=self.d.compact_json) as capture:
            bound = self.d.manifest_bound(self.plan, self.ready, self.rows, info)
        manifest = capture.call_args.args[0]
        self.assertEqual(bound['file_array_entries'], 14)
        self.assertEqual(bound['float_files'], 8)
        self.assertEqual(bound['integer_members'], 6)
        self.assertEqual(bound['future_binding_encoded_JSON_cap'], 16384)
        self.assertEqual(bound['manifest_max_bytes'],
                         bound['complete_known_manifest_with_fixed_width_hash_slots_bytes'] - 2 + 16384)
        self.assertEqual(bound['complete_known_manifest_with_fixed_width_hash_slots_bytes'],
                         len(self.d.compact_json(manifest)))
        self.assertEqual(manifest['section_table']['rows'], 16)
        self.assertEqual(manifest['arms'], ['none', 'struct', 'shuf', 'beat_trigger'])
        entries = manifest['files']
        self.assertEqual(len(entries), 14)
        float_names = [entry['file'] for entry in entries if entry['dtype'] == '<f4']
        self.assertEqual(float_names, ['c%02d_%s.npy' % (i, arm)
                                     for i in range(2) for arm in self.d.ARMS])
        for entry in entries:
            self.assertEqual(len(entry['sha256']), 64)
            self.assertRegex(entry['sha256'], r'^[0-9a-f]{64}$')
            self.assertIs(entry['fortran_order'], False)
            self.assertIn(entry['clip_ordinal'], (0, 1))
            self.assertEqual(entry['array_bytes'], entry['shape'][0] * np.dtype(entry['dtype']).itemsize)
            self.assertIn(entry['shape'], ([166], [82]))
            self.assertEqual(entry['npy_bytes'], entry['shape'][0] * np.dtype(entry['dtype']).itemsize
                             + len(self.d.npy_header(entry['dtype'], entry['shape'][0])))

    def test_51_budget_full_payload_peak_and_global_arithmetic_no_science(self):
        d = self.d
        resources = self.resources()
        bound = d.budget_bound(resources, 1000, 2000, 3000)
        peak = (100000 + 10000 + d.FUTURE_RESERVE + d.GPU_LEDGER_RESERVE + d.HEADER_RESERVE
                + d.TABLE_MANIFEST_RESERVE + d.FLOAT_PAYLOAD + d.RECEIPT_CAP + d.LOG_CAP + 1000)
        self.assertEqual(bound['whole_future_packet_total_and_peak_upper_bound_bytes'], peak)
        self.assertEqual(bound['global_peak_commitment_bytes'], 200000 + peak - 100000)
        self.assertEqual(bound['packet_margin_bytes'], d.PACKET_CAP - peak)
        self.assertEqual(bound['float_payload_bytes_no_discount'], 5942288)
        self.assertEqual(bound['prototype_disk_peak_before_future_science_bytes'],
                         110000 + d.RECEIPT_CAP + d.LOG_CAP)
        self.assertIs(bound['science_admission'], False)

    def test_52_budget_duplicate_float_payload_or_raw_then_archive_peak_rejected(self):
        resources = self.resources()
        for bad in (2, True, 1.0):
            resources['float_payload_copy_count'] = bad
            with self.assertRaises(self.d.ContractError):
                self.d.budget_bound(resources, 1000, 2000, 3000)
        resources['float_payload_copy_count'] = 1
        resources['write_pattern'] = 'RAW_THEN_ARCHIVE_WITH_TEMPORARY_DUPLICATE'
        with self.assertRaises(self.d.ContractError):
            self.d.budget_bound(resources, 1000, 2000, 3000)

    def test_53_budget_packet_cap_is_a_peak_not_just_retained_size(self):
        resources = self.resources()
        resources['retained_packet_bytes_excluding_distinct_GPU_ledger_charge'] = self.d.PACKET_CAP
        with self.assertRaises(self.d.ContractError):
            self.d.budget_bound(resources, 1000, 2000, 3000)

    def test_54_budget_global_commitment_cap_includes_future_growth(self):
        resources = self.resources()
        resources['global_existing_plus_other_reservations_excluding_row8_growth_and_GPU_ledger'] = self.d.GLOBAL_CAP
        with self.assertRaises(self.d.ContractError):
            self.d.budget_bound(resources, 1000, 2000, 3000)

    def test_55_budget_reserve_header_schema_and_preservation_fail_closed(self):
        d = self.d
        with self.assertRaises(d.ContractError):
            d.budget_bound(self.resources(), 1000, d.TABLE_MANIFEST_RESERVE + 1, 3000)
        with self.assertRaises(d.ContractError):
            d.budget_bound(self.resources(), 1000, 2000, d.HEADER_RESERVE + 1)
        for field in ('future_unspent_code_control_metrics_bytes', 'distinct_GPU_ledger_charge_bytes',
                      'header_reservation_bytes', 'table_manifest_reservation_bytes'):
            resources = self.resources()
            resources[field] -= 1
            with self.assertRaises(d.ContractError):
                d.budget_bound(resources, 1000, 2000, 3000)
        for argument_index in (0, 1, 2):
            for bad in (0, -1, True, 1.5):
                values = [1000, 2000, 3000]
                values[argument_index] = bad
                with self.assertRaises(d.ContractError):
                    d.budget_bound(self.resources(), *values)
        unknown_key = self.resources()
        unknown_key['unaccounted_extra_write'] = 1
        with self.assertRaises(d.ContractError):
            d.budget_bound(unknown_key, 1000, 2000, 3000)
        for field in ('all_retained_and_prototype_controls_counted',
                      'prior_failures_and_eight_protocols_preserved'):
            resources = self.resources()
            resources[field] = False
            with self.assertRaises(d.ContractError):
                d.budget_bound(resources, 1000, 2000, 3000)

    def test_56_budget_root_and_mnt_free_floors_require_all_peak_growth(self):
        for field in ('free_root_bytes', 'free_mnt_bytes'):
            resources = self.resources()
            bound = self.d.budget_bound(resources, 1000, 2000, 3000)
            growth = bound['whole_future_packet_total_and_peak_upper_bound_bytes'] - 100000
            resources[field] = 2**30 + growth - 1
            with self.assertRaises(self.d.ContractError):
                self.d.budget_bound(resources, 1000, 2000, 3000)
            resources[field] += 1
            self.assertEqual(self.d.budget_bound(resources, 1000, 2000, 3000), bound)

    def test_57_authorization_accepts_only_source_bound_cpu_format_fixture(self):
        self.assertEqual(self.d.EXPECTED_TEST_COUNT, EXPECTED_TEST_COUNT)
        auth = self.authorization()
        self.assertIs(self.d.validate_authorization(auth, 1010), auth)
        self.assertEqual(auth['scope'], 'METADATA_INTEGER_CODEC_ONLY_NO_SCIENCE')
        self.assertEqual(auth['native_test_attestation']['tests_run'], 67)

    def test_58_authorization_scope_source_reference_and_codec_mismatches(self):
        for field, bad in (('scope', 'ALLOW_SCIENCE'), ('status', 'APPROVED'),
                           ('root_attests_fresh_exact_command_accounting', False)):
            auth = self.authorization()
            auth[field] = bad
            with self.assertRaises(self.d.ContractError):
                self.d.validate_authorization(auth, 1010)
        for field, bad in (('driver_sha256', 'not a hash'), ('test_count', 63),
                           ('root_verified_source_bytes', False)):
            auth = self.authorization()
            auth['source_binding'][field] = bad
            with self.assertRaises(self.d.ContractError):
                self.d.validate_authorization(auth, 1010)
        for ref in ('plan_ref', 'readiness_ref', 'envelope_ref'):
            auth = self.authorization()
            auth[ref]['path'] = '/SYNTHETIC_NEVER_OPEN/different.json'
            with self.assertRaises(self.d.ContractError):
                self.d.validate_authorization(auth, 1010)
        auth = self.authorization()
        auth['codec_versions']['numpy'] = 'different'
        with self.assertRaises(self.d.ContractError):
            self.d.validate_authorization(auth, 1010)
        auth = self.authorization()
        auth['allow_science'] = True
        with self.assertRaises(self.d.ContractError):
            self.d.validate_authorization(auth, 1010)

    def test_59_authorization_rejects_stale_future_and_nonfinite_accounting(self):
        auth = self.authorization()
        for now in (999, 1301, float('nan'), float('inf')):
            with self.assertRaises(self.d.ContractError):
                self.d.validate_authorization(auth, now)
        self.assertIs(self.d.validate_authorization(auth, 1000), auth)
        self.assertIs(self.d.validate_authorization(auth, 1300), auth)

    def test_60_authorization_requires_plain_native_ok_no_skips_or_expected_failures(self):
        for field in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes'):
            for bad in (1, False):
                auth = self.authorization()
                auth['native_test_attestation'][field] = bad
                with self.assertRaises(self.d.ContractError):
                    self.d.validate_authorization(auth, 1010)
        for field, bad in (('exit_code', 1), ('tests_run', 63), ('plain_OK', False)):
            auth = self.authorization()
            auth['native_test_attestation'][field] = bad
            with self.assertRaises(self.d.ContractError):
                self.d.validate_authorization(auth, 1010)

    def test_61_envelope_accepts_exact_hidden_gpu_cpu_limits_only(self):
        envelope = self.envelope()
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            self.assertIsNone(self.d.validate_envelope(envelope))
        self.assertIs(envelope['GPU_devices_visible'], False)
        self.assertIs(envelope['process_tree_enforced'], True)
        self.assertEqual(envelope['module_cli_argv'][1], 'measure')

    def test_62_envelope_mismatches_types_argv_environment_and_proof_rejected(self):
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            for field, bad in (('status', 'PROMISED_NOT_ENFORCED'), ('MemoryMax_bytes', 536870913),
                ('MemorySwapMax_bytes', False), ('RuntimeMaxSec', 50.0),
                ('acceptance_wall_seconds', 61), ('PrivateDevices', 1),
                ('GPU_devices_visible', True), ('CUDA_VISIBLE_DEVICES', '0'),
                ('process_tree_enforced', False)):
                envelope = self.envelope()
                envelope[field] = bad
                with self.assertRaises(self.d.ContractError):
                    self.d.validate_envelope(envelope)
            envelope = self.envelope()
            envelope['module_cli_argv'][1] = 'science'
            with self.assertRaises(self.d.ContractError):
                self.d.validate_envelope(envelope)
        with mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaises(self.d.ContractError):
                self.d.validate_envelope(self.envelope())
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(self.d.ContractError):
                self.d.validate_envelope(self.envelope())

    def test_63_static_cli_is_measure_only_and_never_injects_small_contract(self):
        main = self.function('main')
        action_calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                        and self.call_name(n.func).endswith('.add_argument') and n.args
                        and isinstance(n.args[0], ast.Constant) and n.args[0].value == 'action']
        self.assertEqual(len(action_calls), 1)
        choices = next(k.value for k in action_calls[0].keywords if k.arg == 'choices')
        self.assertEqual(tuple(ast.literal_eval(choices)), ('measure',))
        metadata = self.function('validate_metadata')
        self.assertEqual(metadata.args.args[-1].arg, 'contract')
        self.assertIsInstance(metadata.args.defaults[-1], ast.Name)
        self.assertEqual(metadata.args.defaults[-1].id, 'FROZEN')
        calls = [n for n in ast.walk(self.function('measure')) if isinstance(n, ast.Call)
                 and self.call_name(n.func) == 'validate_metadata']
        self.assertEqual(len(calls), 1)
        self.assertEqual([self.call_name(n) for n in calls[0].args], ['plan', 'ready'])
        self.assertEqual(calls[0].keywords, [])
        reads = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                 and self.call_name(n.func) == 'read_control']
        self.assertEqual(len(reads), 4)
        main_reads = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                      and self.call_name(n.func) == 'read_control']
        self.assertEqual(len(main_reads), 4)

    def test_64_static_no_scientific_dispatch_and_guards_block_all_operational_io(self):
        forbidden = {'torch', 'lightsem', 'tensorflow', 'jax', 'cupy', 'pickle', 'joblib', 'subprocess'}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name.split('.')[0] not in forbidden for alias in node.names))
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or '').split('.')[0], forbidden)
            if isinstance(node, ast.Call):
                self.assertNotIn(self.call_name(node.func).split('.')[-1],
                                 {'forward', 'predict', 'train', 'fit', 'evaluate', 'score', 'smoke',
                                  'tofile', 'fromfile', 'memmap', 'loadtxt', 'savetxt', 'genfromtxt'})
        for operation in (
                lambda: builtins.open('/SYNTHETIC_NEVER_OPEN/x', 'rb'),
                lambda: io.open('/SYNTHETIC_NEVER_OPEN/x', 'wb'),
                lambda: os.open('/SYNTHETIC_NEVER_OPEN/x', os.O_RDONLY),
                lambda: Path('/SYNTHETIC_NEVER_OPEN/x').read_bytes(),
                lambda: np.load('/SYNTHETIC_NEVER_OPEN/x.npz'),
                lambda: np.save('/SYNTHETIC_NEVER_OPEN/x.npy', np.zeros(1)),
                lambda: self.d.NP_LOAD('/SYNTHETIC_NEVER_OPEN/x.npy'),
                lambda: self.d.read_control('/SYNTHETIC_NEVER_OPEN/x', 1),
                lambda: self.d.write_receipt({}), lambda: self.d.main([]),
                lambda: self.d.measure({}, {}, {})):
            with self.assertRaises(AssertionError):
                operation()

    def test_65_lzma_declared_roundtrip_properties_eof_and_unsupported_refusal(self):
        d = self.d
        self.validate()
        self.assertEqual(d.CODECS, ('deflate9', 'lzma_default'))
        self.assertEqual(d.codec_contract('lzma_default'), (zipfile.ZIP_LZMA, None, 2, 63))
        raw, high_water = d.encode_integer_archive(self.rows, codec='lzma_default')
        self.assertEqual(high_water, len(raw))
        self.assertEqual(raw, d.encode_integer_archive(self.rows, codec='lzma_default')[0])
        with mock.patch.object(d, 'NP_LOAD', wraps=d.NP_LOAD) as loader:
            info = d.inspect_integer_archive(raw, self.rows, codec='lzma_default')
        whole_loads = [call for call in loader.call_args_list
                       if isinstance(call.args[0], io.BytesIO)
                       and call.args[0].getvalue()[:4] == b'PK\x03\x04']
        self.assertEqual(len(whole_loads), 1)
        self.assertIs(whole_loads[0].kwargs['allow_pickle'], False)
        self.assertEqual(info['codec_candidate'], 'lzma_default')
        self.assertIs(info['NumPy_standard_NPZ_roundtrip'], True)
        self.assertIs(info['bytewise_roundtrip'], True)
        self.assertEqual(info['members'], 6)
        self.assertEqual(info['integer_raw_data_bytes'], 248 * 7)
        self.assertEqual(info['member_NPY_sha256'],
                         d.inspect_integer_archive(self.archive(), self.rows)['member_NPY_sha256'])
        with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=256) as loaded:
            self.assertEqual(loaded.files, [name[:-4] for name in d.integer_schema(self.rows)])
            for name, (dtype, T, clip_index, field) in d.integer_schema(self.rows).items():
                value = loaded[name[:-4]]
                self.assertEqual(value.dtype.str, dtype)
                self.assertEqual(value.shape, (T,))
                self.assertEqual(value.tobytes(), d.integer_vectors(self.rows[clip_index])[field].tobytes())
        # Price this second candidate independently. Unknown floats remain the
        # same undiscounted frozen payload; never choose/fallback to a codec.
        manifest = d.manifest_bound(self.plan, self.ready, self.rows, info)
        table = d.table_bound(self.rows)
        headers = info['ZIP_container_bytes'] + sum(len(d.npy_header('<f4', r['T'])) * 4
                                                     for r in self.rows)
        budget = d.budget_bound(self.resources(), info['compressed_integer_NPY_stream_bytes'],
                               manifest['manifest_max_bytes'] + table['max_bytes'], headers)
        self.assertEqual(budget['float_payload_bytes_no_discount'], d.FLOAT_PAYLOAD)
        self.assertIs(budget['science_admission'], False)
        local = struct.unpack('<4s5H3L2H', raw[:30])
        start = 30 + local[9] + local[10]
        self.assertEqual(raw[start:start + 5], b'\x09\x04\x05\x00\x5d')
        self.assertEqual(struct.unpack('<L', raw[start + 5:start + 9])[0], 8388608)
        for relative, value in ((0, 8), (2, 6), (4, 0), (8, 255)):
            changed = bytearray(raw)
            changed[start + relative] = value
            with self.assertRaises(d.ContractError):
                d.inspect_integer_archive(bytes(changed), self.rows, codec='lzma_default')
        with self.assertRaises(d.ContractError):
            d.inspect_integer_archive(self.with_compressed_trailer(raw), self.rows, codec='lzma_default')
        # Keep the valid nine-byte properties header and only one coded byte:
        # no output or complete LZMA EOF can exist, even though ZIP sizes agree.
        with self.assertRaises((d.ContractError, lzma.LZMAError, zipfile.BadZipFile)):
            d.inspect_integer_archive(self.with_compressed_trailer(raw, keep_bytes=10),
                                      self.rows, codec='lzma_default')
        with self.assertRaises(d.ContractError):
            d.inspect_integer_archive(raw, self.rows, codec='deflate9')
        with mock.patch.object(d, 'lzma', None):
            with self.assertRaisesRegex(d.ContractError, 'UNSUPPORTED_ZIP_LZMA'):
                d.encode_integer_archive(self.rows, codec='lzma_default')
            with self.assertRaisesRegex(d.ContractError, 'UNSUPPORTED_ZIP_LZMA'):
                d.inspect_integer_archive(raw, self.rows, codec='lzma_default')
        with mock.patch.object(zipfile, 'lzma', None):
            with self.assertRaisesRegex(d.ContractError, 'UNSUPPORTED_ZIP_LZMA'):
                d.codec_contract('lzma_default')
        for codec in ('auto', 'best', 'bzip2', 'unknown'):
            with self.assertRaises(d.ContractError):
                d.encode_integer_archive(self.rows, codec=codec)
            with self.assertRaises(d.ContractError):
                d.inspect_integer_archive(raw, self.rows, codec=codec)

    def test_66_unexpected_candidate_failure_clears_both_completion_flags(self):
        d = self.d
        # Shape/count metadata only. No vectors or archive are allocated here.
        rows = [{'T': 9284} for _ in range(39)] + [{'T': 9317}]
        info = {'members': 120, 'ZIP_container_bytes': 10,
                'compressed_integer_NPY_stream_bytes': 20, 'member_NPY_sha256': {}}
        table = {'rows': 2452, 'max_bytes': 100}
        manifest = {'file_array_entries': 280, 'manifest_max_bytes': 200}
        with mock.patch.object(d, 'encode_integer_archive', return_value=(b'mock-only', 9)), \
             mock.patch.object(d, 'inspect_integer_archive', return_value=info), \
             mock.patch.object(d, 'table_bound', return_value=table), \
             mock.patch.object(d, 'manifest_bound', return_value=manifest), \
             mock.patch.object(d, 'npy_header', return_value=b'h' * 128), \
             mock.patch.object(d, 'budget_bound', side_effect=RuntimeError('injected after complete')) as budget:
            for codec in d.CODECS:
                result = d.measure_candidate(self.plan, self.ready, rows, self.resources(), codec)
                self.assertEqual(result['status'], 'CODEC_MEASUREMENT_FAILED')
                self.assertEqual(result['codec_candidate'], codec)
                self.assertIs(result['measurement_complete'], False)
                self.assertIs(result['complete_format_bound_fits'], False)
                self.assertIs(result['scientific_execution_admitted'], False)
                self.assertIn('RuntimeError: injected after complete', result['error'])
            self.assertEqual(budget.call_count, 2)
            # Host exhaustion is not a codec result and must not start fallback.
            budget.side_effect = MemoryError('synthetic host exhaustion')
            with self.assertRaises(MemoryError):
                d.measure_candidate(self.plan, self.ready, rows, self.resources(), 'deflate9')

    def test_67_fixed_dual_candidate_dispatch_does_not_select_or_suppress_after_budget_rejection(self):
        d = self.d
        results = [
            {'codec_candidate': 'deflate9', 'measurement_complete': True,
             'complete_format_bound_fits': False, 'scientific_execution_admitted': False,
             'status': 'MEASURED_FORMAT_OR_BUDGET_BOUND_REJECTED'},
            {'codec_candidate': 'lzma_default', 'measurement_complete': True,
             'complete_format_bound_fits': True, 'scientific_execution_admitted': False,
             'status': 'MEASURED_FORMAT_BOUND_FITS_NO_SCIENCE'},
        ]
        resources = self.resources()
        with mock.patch.object(d, 'validate_metadata', return_value=self.rows) as validate, \
             mock.patch.object(d, 'measure_candidate', side_effect=copy.deepcopy(results)) as candidate, \
             mock.patch.object(d.time, 'monotonic', return_value=0.0):
            result = self.measure_function(self.plan, self.ready, resources)
        # No small contract is passed into production measure, even in this
        # mocked regression: the real production validator default stays FROZEN.
        validate.assert_called_once_with(self.plan, self.ready)
        self.assertEqual(candidate.call_args_list, [
            mock.call(self.plan, self.ready, self.rows, resources, 'deflate9'),
            mock.call(self.plan, self.ready, self.rows, resources, 'lzma_default')])
        self.assertEqual(result['candidate_order_fixed_before_measurement'], ['deflate9', 'lzma_default'])
        self.assertEqual(result['candidate_results'], results)
        self.assertIs(result['native_measurement_complete'], True)
        self.assertIs(result['automatic_selection'], False)
        self.assertIs(result['automatic_fallback'], False)
        self.assertIs(result['scientific_execution_admitted'], False)
        self.assertIs(result['unknown_brightness_or_statistics_generated'], False)
        for field in ('scientific_archive_files_written', 'model_calls', 'GPU_calls'):
            self.assertEqual(result[field], 0)
        self.assertEqual(result['status'], 'CPU_CODEC_CANDIDATES_MEASURED_NO_SCIENTIFIC_ADMISSION')


if __name__ == '__main__':
    unittest.main()
