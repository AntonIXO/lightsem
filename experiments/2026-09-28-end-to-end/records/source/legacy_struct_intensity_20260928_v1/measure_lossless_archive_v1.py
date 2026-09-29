"""CPU-only metadata/format prototype. No scientific archives are written.
This source supplies no model, GPU, score, training, or fallback dispatcher.
External enforcement and fresh root accounting are REQUIRED, not implemented here.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import stat
import sys
import time
import zipfile
import zlib
try:
    import lzma
except ImportError:
    lzma = None  # explicit unsupported candidate; never install or fall back
import numpy as np
from numpy.lib import format as npformat

NP_LOAD = np.load  # resolve NumPy's format loader before test filesystem guards
ROOT = Path('/root/lightsem/outputs/legacy_struct_intensity_20260928_v1')
DRIVER = ROOT / 'measure_lossless_archive_v1.py'
AUTH = ROOT / 'PARENT_BOUND_LOSSLESS_ARCHIVE_AUTHORIZATION_V1.json'
OUT = ROOT / 'native_lossless_archive_feasibility_v1.json'
PLAN = Path('/root/lightsem/docs/research/2026-09-28-end-to-end/PROSPECTIVE_STRUCT_INTENSITY_PROTOCOL_CANDIDATE_V6.json')
READY = Path('/mnt/lightsem-data/experiments/lightsem/end_to_end_20260928_v1/legacy_struct_intensity_v6/preparation_readiness_v6.json')
PLAN_SHA = '977241748163f747064cfc28e2998f707085431aee8189814c914707d7599267'
READY_SHA = '54049eb742d6e796f3356cb278e1044320a6a7e56cf9c4af04cd33536520a878'
PACKET_CAP = 12582912
GLOBAL_CAP = 1020054732
FLOAT_PAYLOAD = 5942288
HEADER_RESERVE = 131072
TABLE_MANIFEST_RESERVE = 524288
FUTURE_RESERVE = 1048576
GPU_LEDGER_RESERVE = 65536
RECEIPT_CAP = 65536
LOG_CAP = 32768
MAX_ARCHIVE = 4 * 1024 * 1024
MAX_JSON_DEPTH = 24
MAX_JSON_NODES = 100000
EXPECTED_TEST_COUNT = 67  # authored standalone methods; native pass still required
ROLES = ('intro', 'breakdown', 'build-up', 'drop', 'outro')
ARMS = ('none', 'struct', 'shuf', 'beat_trigger')
REASONS = (None, 'INTERSECTION_LT80_FRAMES')
FIELDS = (('frame_index', '<i4'), ('target_role', '|u1'), ('section_id', '<u2'))
FROZEN = {'clips': 40, 'frames': 371393, 'sections': 613, 'invalid': 3,
          'ineligible': 7, 'short_sections': 1, 'input_refs': 62}
CSV_COLUMNS = ('clip_ordinal', 'arm_ordinal', 'section_id', 'role_code',
               'start_frame', 'end_frame_exclusive', 'valid_count',
               'retained_count', 'retained', 'reason_code', 'sum', 'mean')
FLOAT64_TEXT_BYTES = 25  # format(value, '+.17e'): sign,digit,dot,17digits,e,sign,3digits
FUTURE_BINDING_JSON_CAP = 16384
CODECS = ('deflate9', 'lzma_default')
LZMA_DICT_BYTES = 8388608

def codec_contract(codec):
    if codec == 'deflate9':
        return zipfile.ZIP_DEFLATED, 9, 0, 20
    if codec == 'lzma_default':
        require(lzma is not None and zipfile.lzma is not None, 'UNSUPPORTED_ZIP_LZMA: stdlib support absent')
        return zipfile.ZIP_LZMA, None, 2, 63
    raise ContractError('undeclared codec; no fallback')

class ContractError(ValueError):
    pass

def require(condition, message):
    if not condition:
        raise ContractError(message)

def integer(value, lo=0, hi=2**63-1):
    require(type(value) is int and lo <= value <= hi, 'invalid bounded integer')
    return value

def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), 'invalid finite number')
    return value

def compact_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')

def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result

def reject_constant(value):
    raise ContractError('nonfinite JSON token')

def bounded_json(raw, cap):
    require(isinstance(raw, bytes) and len(raw) <= cap, 'JSON byte cap')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=no_duplicates,
                       parse_constant=reject_constant)
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        require(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH, 'JSON complexity cap')
        if isinstance(item, dict):
            require(all(type(k) is str and len(k) <= 256 for k in item), 'JSON key cap')
            pending.extend((v, depth+1) for v in item.values())
        elif isinstance(item, list):
            require(len(item) <= 10000, 'JSON list cap')
            pending.extend((v, depth+1) for v in item)
        elif isinstance(item, str):
            require(len(item.encode('utf-8')) <= 16384, 'JSON string cap')
        elif isinstance(item, float):
            require(math.isfinite(item), 'JSON finite cap')
        elif item is not None:
            require(type(item) in (int, bool), 'JSON value type')
    return value

def no_symlink(path):
    path = Path(path)
    require(path.is_absolute() and '..' not in path.parts, 'absolute canonical control path required')
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'symlink control path forbidden')
    return path

def read_control(path, cap, expected_sha=None, expected_bytes=None):
    """Only main's fixed CONTROL allowlist calls this; never dereference manifest paths."""
    path = no_symlink(path)
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        require(stat.S_ISREG(st.st_mode), 'regular CONTROL file required')
        require(st.st_size <= cap, 'control file byte cap')
        if expected_bytes is not None:
            require(st.st_size == expected_bytes, 'control exact size mismatch')
        raw = b''
        while len(raw) <= cap:
            block = os.read(fd, min(65536, cap+1-len(raw)))
            if not block:
                break
            raw += block
        require(len(raw) == st.st_size and len(raw) <= cap, 'control read size changed')
    finally:
        os.close(fd)
    if expected_sha is not None:
        require(hashlib.sha256(raw).hexdigest() == expected_sha, 'control hash mismatch')
    return bounded_json(raw, cap)

def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys), label + ' schema keys')

def validate_metadata(plan, ready, contract=FROZEN):
    require(type(plan.get('document_revision')) is int and plan['document_revision'] == 6, 'not frozen V6 plan')
    require(ready.get('status') == 'PREPARATION_COMPLETE_NO_SCIENTIFIC_EVALUATION', 'readiness status')
    require(ready.get('GPU_or_study_admission_granted') is False, 'readiness grants science')
    for key in ('model_constructions', 'model_forwards', 'GPU_jobs', 'TEST_payload_reads', 'raw_manifest_reads', 'replacement_clips', 'science_scores_computed'):
        require(ready.get(key) == 0 and type(ready.get(key)) is int, 'readiness zero execution fields')
    require(ready.get('original40_inventory_preserved') is True, 'inventory not preserved')
    require(ready.get('plan_ref', {}).get('sha256') == PLAN_SHA, 'readiness plan reference')
    frozen = plan['cohorts']['clip_index']
    rows = ready['all40_eligibility']
    require(isinstance(frozen, list) and isinstance(rows, list) and len(rows) == len(frozen) == contract['clips'], 'clip count')
    require(len(ready['input_refs']) == contract['input_refs'], 'input-ref count')
    ids = set()
    total = intervals = invalid = ineligible = short = 0
    for index, (old, row) in enumerate(zip(frozen, rows)):
        clip = row['clip']
        require(isinstance(clip, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,64}', clip), 'clip name')
        require(clip not in ids and clip == old['clip'], 'clip order/duplicate')
        ids.add(clip)
        T = integer(row['T'], 1, 11403)
        require(T == integer(old['T_metadata'], 1, 11403), 'T differs from frozen index')
        require(row['recording_id'] == old['recording_id'], 'recording differs')
        require(row['donor_recording_id'] == old['original_shuffled_donor_recording_id'], 'donor differs')
        require(row['fps'] == 40 and type(row['fps']) is int, 'fps changed')
        require(old['current_partition'] == 'train' and old['current_Stage11_TEST_aliases'] == [], 'partition/knownTEST changed')
        i0 = integer(row['i0'], 0, 2**31-1)
        start = finite(row['raw_start_s'])
        require(start == old['start_s_metadata'] and i0 == round(start*40), 'raw start/i0 changed')
        require(row['rounded_origin_s'] == i0/40 and row['roundoff_s'] == i0/40-start, 'clock fields changed')
        require(type(row['eligible']) is bool and isinstance(row['reasons'], list), 'eligibility shape')
        require(row['ratio_or_prediction_computed'] is False, 'prediction metadata')
        require(row['status'] in ('INPUT_QUALIFIED', 'ROLE_SUPPORT_INELIGIBLE'), 'qualification status')
        sections = row['sections']
        require(isinstance(sections, list) and 1 <= len(sections) <= 20, 'section count')
        previous_end = 0
        previous_id = -1
        valid = 0
        byrole = {role: 0 for role in ROLES}
        for section in sections:
            exact_keys(section, ('start_frame', 'end_frame_exclusive', 'intersection_frames', 'section_id', 'role', 'retained', 'reason'), 'section')
            a = integer(section['start_frame'], 0, T-1)
            b = integer(section['end_frame_exclusive'], 1, T)
            sid = integer(section['section_id'], 0, 65534)
            require(a < b and a >= previous_end and sid > previous_id, 'overlap/order/originalID')
            require(section['intersection_frames'] == b-a and type(section['intersection_frames']) is int, 'intersection drift')
            require(section['role'] in ROLES, 'unknown role')
            require(type(section['retained']) is bool, 'retained flag')
            retained = b-a >= 80
            require(section['retained'] == retained, 'short-section policy drift')
            require(section['reason'] == (None if retained else REASONS[1]), 'reason drift')
            valid += b-a
            if retained:
                byrole[section['role']] += b-a
            else:
                short += 1
            previous_end, previous_id = b, sid
        require(integer(row['target_valid_role_frames'], 0, T) == valid and integer(row['target_invalid_role_frames'], 0, T) == T-valid, 'coverage drift')
        require(row['retained_role_frames'] == byrole, 'retained role counts drift')
        require(row['eligible'] == (not row['reasons']), 'eligibility/reasons drift')
        require(row['status'] == ('INPUT_QUALIFIED' if row['eligible'] else 'ROLE_SUPPORT_INELIGIBLE'), 'eligibility status drift')
        total += T
        intervals += len(sections)
        invalid += T-valid
        ineligible += not row['eligible']
    require((total, intervals, invalid, ineligible, short) == tuple(contract[k] for k in ('frames', 'sections', 'invalid', 'ineligible', 'short_sections')), 'global count drift')
    return rows

def integer_vectors(row):
    T = integer(row['T'], 1, 11403)
    index = np.arange(T, dtype=np.dtype('<i4'))
    role = np.full(T, 255, dtype=np.dtype('|u1'))
    section_id = np.full(T, 65535, dtype=np.dtype('<u2'))
    for section in row['sections']:
        a, b = section['start_frame'], section['end_frame_exclusive']
        # Includes rejected short sections. Only uncovered frames keep sentinels.
        role[a:b] = ROLES.index(section['role'])
        section_id[a:b] = section['section_id']
    return {'frame_index': index, 'target_role': role, 'section_id': section_id}

def npy_header(dtype, T):
    dtype = np.dtype(dtype)
    require(dtype.str in ('<i4', '|u1', '<u2', '<f4'), 'dtype not allowlisted')
    integer(T, 1, 11403)
    sink = io.BytesIO()
    npformat.write_array_header_1_0(sink, {'descr': dtype.str, 'fortran_order': False, 'shape': (T,)})
    result = sink.getvalue()
    require(len(result) <= 256 and result[:8] == b'\x93NUMPY\x01\x00', 'NPY header format cap')
    return result

def encode_npy(array, dtype, T):
    require(isinstance(array, np.ndarray) and array.dtype.str == dtype and array.shape == (T,) and array.flags.c_contiguous, 'array field dtype/shape/layout')
    return npy_header(dtype, T) + array.tobytes(order='C')

def decode_npy(raw, dtype, T):
    header = npy_header(dtype, T)
    require(isinstance(raw, bytes) and len(raw) == len(header)+T*np.dtype(dtype).itemsize, 'NPY exact byte count')
    require(raw[:len(header)] == header, 'NPY exact header mismatch')
    stream = io.BytesIO(raw)
    value = NP_LOAD(stream, allow_pickle=False, max_header_size=256)
    require(stream.tell() == len(raw), 'NPY trailing bytes')
    require(value.dtype.str == dtype and value.shape == (T,) and value.flags.c_contiguous, 'NPY decoded dtype/shape/layout')
    require(value.tobytes(order='C') == raw[len(header):], 'NPY raw bit preservation')
    return value

class BoundedBytesIO(io.BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = integer(limit, 1, MAX_ARCHIVE)
        self.high_water = 0
    def write(self, data):
        end = self.tell()+len(data)
        require(end <= self.limit, 'in-memory archive byte cap')
        result = super().write(data)
        self.high_water = max(self.high_water, end)
        return result
    def seek(self, offset, whence=0):
        position = super().seek(offset, whence)
        require(0 <= position <= self.limit, 'in-memory seek cap')
        return position
    def truncate(self, size=None):
        target = self.tell() if size is None else size
        require(0 <= target <= self.limit, 'in-memory truncate cap')
        return super().truncate(target)

def integer_schema(rows):
    return {f'c{i:02d}_{field}.npy': (dtype, row['T'], i, field)
            for i, row in enumerate(rows) for field, dtype in FIELDS}

def encode_integer_archive(rows, cap=MAX_ARCHIVE, codec='deflate9'):
    method, level, flags, version = codec_contract(codec)
    sink = BoundedBytesIO(cap)
    schema = integer_schema(rows)
    require(len(schema) == len(rows)*3 <= 120, 'integer member count')
    with zipfile.ZipFile(sink, 'w', compression=method,
                         compresslevel=level, allowZip64=False) as archive:
        for i, row in enumerate(rows):
            arrays = integer_vectors(row)
            for field, dtype in FIELDS:
                name = f'c{i:02d}_{field}.npy'
                info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                info.compress_type = method
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, encode_npy(arrays[field], dtype, row['T']),
                                 compress_type=method, compresslevel=level)
    raw = sink.getvalue()
    require(len(raw) == sink.high_water, 'archive high-water mismatch')
    return raw, sink.high_water

def inspect_integer_archive(raw, rows, codec='deflate9'):
    method, level, flags, version = codec_contract(codec)
    require(isinstance(raw, bytes) and 22 <= len(raw) <= MAX_ARCHIVE, 'archive byte cap')
    schema = integer_schema(rows)
    count = len(schema)
    require(count == 3*len(rows) and 1 <= count <= 120, 'expected member count')
    eocd = struct.unpack('<4s4H2LH', raw[-22:])
    require(eocd[0] == b'PK\x05\x06' and eocd[1:3] == (0, 0) and eocd[3:5] == (count, count) and eocd[7] == 0, 'ZIP end record')
    central_size, central_offset = eocd[5:7]
    require(central_offset+central_size+22 == len(raw), 'ZIP central directory/EOF')
    compressed = expanded = offset = 0
    member_hashes = {}
    with zipfile.ZipFile(io.BytesIO(raw), 'r', allowZip64=False) as archive:
        infos = archive.infolist()
        require(len(infos) == count and [x.filename for x in infos] == list(schema), 'ZIP exact ordered names/count')
        require(archive.comment == b'', 'ZIP archive comment')
        for info in infos:
            name = info.filename
            dtype, T, clip_index, field = schema[name]
            encoded = name.encode('ascii')
            require(len(encoded) <= 64 and not info.is_dir() and '/' not in name and '\\' not in name, 'ZIP unsafe name')
            require(info.compress_type == method and info.flag_bits == flags and info.extra == b'' and info.comment == b'', 'ZIP method/flags/extras')
            require(info.extract_version == version and info.header_offset == offset, 'ZIP64/noncontiguous member')
            expected_bytes = len(npy_header(dtype, T))+T*np.dtype(dtype).itemsize
            require(info.file_size == expected_bytes and 0 < info.compress_size <= MAX_ARCHIVE, 'ZIP expanded/member size')
            require(offset+30 <= central_offset, 'ZIP local header range')
            local = struct.unpack('<4s5H3L2H', raw[offset:offset+30])
            require(local[0] == b'PK\x03\x04' and local[1] == version and local[2] == flags and local[3] == method, 'ZIP local format')
            require(local[6:9] == (info.CRC, info.compress_size, info.file_size) and local[9:] == (len(encoded), 0), 'ZIP local/central size or CRC disagreement')
            require(raw[offset+30:offset+30+len(encoded)] == encoded, 'ZIP local name')
            offset += 30+len(encoded)+info.compress_size
            require(offset <= central_offset, 'ZIP compressed range')
            compressed_start = offset-info.compress_size
            encoded_stream = raw[compressed_start:offset]
            if codec == 'deflate9':
                inflater = zlib.decompressobj(-15)
                bounded_member = inflater.decompress(encoded_stream, expected_bytes+1)
                require(len(bounded_member) == expected_bytes and inflater.eof and not inflater.unused_data and not inflater.unconsumed_tail, 'DEFLATE exact EOF/expanded cap')
            else:
                # Standard ZIP LZMA header + LZMA1 default filter properties.
                # Reject attacker-controlled dictionaries before constructing a decoder.
                require(len(encoded_stream) > 9 and encoded_stream[:4] == b'\x09\x04\x05\x00', 'ZIP LZMA properties header')
                require(encoded_stream[4] == 93 and struct.unpack('<L', encoded_stream[5:9])[0] == LZMA_DICT_BYTES, 'LZMA dictionary/property bounds')
                inflater = lzma.LZMADecompressor(format=lzma.FORMAT_RAW,
                    filters=[{'id': lzma.FILTER_LZMA1, 'dict_size': LZMA_DICT_BYTES, 'lc': 3, 'lp': 0, 'pb': 2}])
                bounded_member = inflater.decompress(encoded_stream[9:], max_length=expected_bytes+1)
                require(len(bounded_member) == expected_bytes and inflater.eof and not inflater.unused_data, 'LZMA exact EOF/expanded cap')
            del inflater  # release raw codec state before independent ZIP decoding
            with archive.open(info, 'r') as stream:
                member = stream.read(expected_bytes+1)
                require(len(member) == expected_bytes and stream.read(1) == b'', 'ZIP expanded EOF')
            del stream
            require(member == bounded_member, 'ZIP/codec reader disagreement')
            restored = decode_npy(member, dtype, T)
            expected = integer_vectors(rows[clip_index])[field]
            require(restored.tobytes(order='C') == expected.tobytes(order='C'), 'integer bit mismatch')
            compressed += info.compress_size
            expanded += info.file_size
            member_hashes[name] = hashlib.sha256(member).hexdigest()
            del bounded_member, member, restored, expected
        require(offset == central_offset, 'ZIP hidden bytes before central directory')
    # Actual NumPy standard-NPZ compatibility, not merely manual member decoding.
    # The immutable in-memory archive has already passed bounded stream/property,
    # name, dtype, header, expanded-byte, CRC and EOF checks above.
    with NP_LOAD(io.BytesIO(raw), allow_pickle=False, max_header_size=256) as loaded:
        require(loaded.files == [name[:-4] for name in schema], 'NumPy NPZ exact fields')
        for name, (dtype, T, clip_index, field) in schema.items():
            restored = loaded[name[:-4]]
            require(restored.dtype.str == dtype and restored.shape == (T,) and restored.flags.c_contiguous, 'NumPy NPZ dtype/shape/layout')
            require(restored.tobytes(order='C') == integer_vectors(rows[clip_index])[field].tobytes(order='C'), 'NumPy NPZ byte preservation')
    raw_data = sum(r['T'] for r in rows)*7
    require(expanded == raw_data + sum(len(npy_header(d, r['T'])) for r in rows for _, d in FIELDS), 'expanded total')
    return {'codec_candidate': codec, 'NumPy_standard_NPZ_roundtrip': True, 'members': count, 'compressed_integer_NPY_stream_bytes': compressed,
            'integer_NPY_expanded_bytes': expanded, 'integer_raw_data_bytes': raw_data,
            'ZIP_container_bytes': len(raw)-compressed, 'integer_archive_bytes': len(raw),
            'integer_archive_sha256': hashlib.sha256(raw).hexdigest(),
            'member_NPY_sha256': member_hashes, 'bytewise_roundtrip': True}

def format_finite_statistic(value):
    """Formatting contract only; this prototype never computes section statistics."""
    require(type(value) is float and math.isfinite(value), 'finite binary64 statistic required')
    text = format(value, '+.17e')
    require(len(text.encode('ascii')) <= FLOAT64_TEXT_BYTES and float(text).hex() == value.hex(), 'binary64 text bit preservation')
    return text

def table_bound(rows):
    header = (','.join(CSV_COLUMNS)+'\n').encode('ascii')
    known = len(header)
    count = 0
    for i, row in enumerate(rows):
        for arm in range(4):
            for section in row['sections']:
                n = section['intersection_frames']
                fields = (i, arm, section['section_id'], ROLES.index(section['role']),
                          section['start_frame'], section['end_frame_exclusive'], n,
                          n if section['retained'] else 0, int(section['retained']),
                          REASONS.index(section['reason']))
                # No simulated brightness or section statistic is created.
                known += len((','.join(str(x) for x in fields)+',').encode('ascii'))
                known += 2*FLOAT64_TEXT_BYTES + 2  # comma between unknown values plus LF
                count += 1
    return {'columns': list(CSV_COLUMNS), 'rows': count, 'max_bytes': known,
            'float64_fields_per_row': 2, 'max_ASCII_bytes_per_unknown_float64': FLOAT64_TEXT_BYTES,
            'encoding': 'ASCII CSV, LF, no quoting; normalized keys joined through manifest',
            'float64_format': '+.17e; finite float64 only, including signed zero; no NA substitution',
            'statistics_computed': False}

def manifest_bound(plan, ready, rows, archive_info):
    frozen = plan['cohorts']['clip_index']
    entries = []
    for i, row in enumerate(rows):
        for arm in ARMS:
            entries.append({'file': f'c{i:02d}_{arm}.npy', 'member': None, 'clip_ordinal': i,
                            'arm_ordinal': ARMS.index(arm), 'array_kind': 'brightness', 'fortran_order': False,
                            'array_bytes': row['T']*4, 'dtype': '<f4', 'shape': [row['T']],
                            'npy_bytes': row['T']*4+len(npy_header('<f4', row['T'])),
                            'sha256': '0'*64})
        for field, dtype in FIELDS:
            name = f'c{i:02d}_{field}.npy'
            entries.append({'file': 'integers.npz', 'member': name, 'clip_ordinal': i, 'arm_ordinal': None,
                            'array_kind': field, 'fortran_order': False, 'array_bytes': row['T']*np.dtype(dtype).itemsize, 'dtype': dtype,
                            'shape': [row['T']], 'npy_bytes': row['T']*np.dtype(dtype).itemsize+len(npy_header(dtype, row['T'])),
                            'sha256': archive_info['member_NPY_sha256'][name]})
    metadata_rows = [{k: v for k, v in row.items() if k != 'sections'} for row in rows]
    readiness_without_sections = {**ready, 'all40_eligibility': metadata_rows}
    manifest = {'schema': 'PROSPECTIVE_LOSSLESS_LAYOUT_V1_NOT_SCIENCE',
                'frozen_plan_ref': ready['plan_ref'],
                'readiness_ref': {'path': str(READY), 'sha256': READY_SHA, 'size_bytes': 202345},
                'clip_index': frozen, 'readiness_except_section_table': readiness_without_sections,
                'files': entries, 'codec_versions': codec_versions(),
                'integer_archive': {'file': 'integers.npz', 'codec_candidate': archive_info.get('codec_candidate', 'deflate9'), 'sha256': archive_info['integer_archive_sha256'], 'bytes': archive_info['integer_archive_bytes']},
                'section_table': {'file': 'sections.csv', 'sha256': '0'*64, 'bytes': TABLE_MANIFEST_RESERVE,
                                  'columns': list(CSV_COLUMNS), 'rows': sum(len(r['sections']) for r in rows)*4},
                'arms': list(ARMS), 'roles': list(ROLES), 'reasons': list(REASONS),
                'join_contract': 'clip_ordinal indexes both full clip_index and qualification rows; arm/role/reason codes index the explicit dictionaries. Original section IDs are never renumbered. All section fields remain reconstructible, including rejected short sections.',
                'statistics_contract': 'sum/mean are float64 accumulations over each original valid intersection from actual saved finite float32 arm files; retained_count is intersection_frames or0. Short intersections still have sum and mean over their valid positive-length intersection, even when retained_count=0. All values must be finite; no NA substitution.',
                'section_decode_contract': {'order': 'clip ordinal, arm ordinal, saved section order',
                    'recover_original_sections_from_arm': 0,
                    'intersection_frames': 'valid_count=end_frame_exclusive-start_frame',
                    'retained_count': 'valid_count if retained else0; NOT the statistic support',
                    'retained': '0/1 maps to original bool', 'role': 'roles[role_code]',
                    'reason': 'reasons[reason_code]', 'replicas': 'all4 arms have identical geometry/count/retained/reason rows',
                    'original_section_keys': ['start_frame', 'end_frame_exclusive', 'intersection_frames', 'section_id', 'role', 'retained', 'reason']},
                'manifest_encoding': 'compact ASCII JSON sort_keys=True, no final LF, noBOM',
                'unknown_hash_slots': 'SIZE_ONLY_64HEX_ZERO_PLACEHOLDERS_NOT_ACTUAL_SCIENTIFIC_HASHES',
                'future_binding': {}}
    base = len(compact_json(manifest))
    # Placeholder SHA bytes have exactly the future hex-string width. No hashes or
    # float values are fabricated as evidence; this manifest is never published.
    # Future binding is a schema-bound object, encoded size <=16384, including
    # braces. It contains future writer/config/authorization/amendment refs only.
    bound = base - 2 + FUTURE_BINDING_JSON_CAP
    return {'complete_known_manifest_with_fixed_width_hash_slots_bytes': base,
            'future_binding_encoded_JSON_cap': FUTURE_BINDING_JSON_CAP,
            'manifest_max_bytes': bound, 'file_array_entries': len(entries),
            'float_files': len(rows)*4, 'integer_members': len(rows)*3,
            'manifest_written': False, 'unknown_hash_slots_are_size_only': True}

def validate_resources(resources):
    keys = ('write_pattern', 'float_payload_copy_count', 'future_unspent_code_control_metrics_bytes',
            'distinct_GPU_ledger_charge_bytes', 'header_reservation_bytes', 'table_manifest_reservation_bytes',
            'retained_packet_bytes_excluding_distinct_GPU_ledger_charge', 'other_unwritten_control_reserve_bytes',
            'global_existing_plus_other_reservations_excluding_row8_growth_and_GPU_ledger',
            'free_root_bytes', 'free_mnt_bytes', 'all_retained_and_prototype_controls_counted',
            'prior_failures_and_eight_protocols_preserved')
    exact_keys(resources, keys, 'resources')
    require(resources['write_pattern'] == 'WRITE_ONCE_FLOAT_NPY_PLUS_SINGLE_INTEGER_NPZ', 'peak duplicate layout forbidden')
    require(integer(resources['float_payload_copy_count'], 1, 1) == 1, 'duplicate float payload forbidden')
    require(integer(resources['distinct_GPU_ledger_charge_bytes'], GPU_LEDGER_RESERVE, GPU_LEDGER_RESERVE) == GPU_LEDGER_RESERVE, 'GPU ledger reserve changed')
    require(integer(resources['header_reservation_bytes'], HEADER_RESERVE, HEADER_RESERVE) == HEADER_RESERVE, 'header reserve changed')
    require(integer(resources['table_manifest_reservation_bytes'], TABLE_MANIFEST_RESERVE, TABLE_MANIFEST_RESERVE) == TABLE_MANIFEST_RESERVE, 'table reserve changed')
    retained = integer(resources['retained_packet_bytes_excluding_distinct_GPU_ledger_charge'], 1, PACKET_CAP)
    extra = integer(resources['other_unwritten_control_reserve_bytes'], 0, PACKET_CAP)
    future = integer(resources['future_unspent_code_control_metrics_bytes'], FUTURE_RESERVE, PACKET_CAP)
    require(resources['all_retained_and_prototype_controls_counted'] is True and resources['prior_failures_and_eight_protocols_preserved'] is True, 'retained accounting attestation')
    # Before ANY output publication, reserve the complete prototype log+receipt
    # and keep future/GPU headroom untouched. Full science-format bound is later.
    growth = extra + RECEIPT_CAP + LOG_CAP + future + GPU_LEDGER_RESERVE
    require(retained+growth <= PACKET_CAP, 'prototype control peak or future headroom')
    global_base = integer(resources['global_existing_plus_other_reservations_excluding_row8_growth_and_GPU_ledger'], retained, GLOBAL_CAP)
    require(global_base+growth <= GLOBAL_CAP, 'prototype global control peak')
    for key in ('free_root_bytes', 'free_mnt_bytes'):
        require(integer(resources[key], 0) >= 1073741824+growth, 'prototype free floor plus writes')
    return resources

def budget_bound(resources, compressed_integer_bytes, table_manifest_bytes, header_bytes):
    validate_resources(resources)
    integer(compressed_integer_bytes, 1, MAX_ARCHIVE)
    integer(table_manifest_bytes, 1, TABLE_MANIFEST_RESERVE)
    integer(header_bytes, 1, HEADER_RESERVE)
    require(table_manifest_bytes <= TABLE_MANIFEST_RESERVE and header_bytes <= HEADER_RESERVE, 'schema/header reserve exceeded')
    require(resources['write_pattern'] == 'WRITE_ONCE_FLOAT_NPY_PLUS_SINGLE_INTEGER_NPZ', 'peak duplicate layout forbidden')
    require(resources['float_payload_copy_count'] == 1 and type(resources['float_payload_copy_count']) is int, 'duplicate float payload forbidden')
    require(resources['future_unspent_code_control_metrics_bytes'] >= FUTURE_RESERVE, 'future headroom consumed')
    require(resources['distinct_GPU_ledger_charge_bytes'] == GPU_LEDGER_RESERVE, 'GPU ledger reserve changed')
    require(resources['header_reservation_bytes'] == HEADER_RESERVE and resources['table_manifest_reservation_bytes'] == TABLE_MANIFEST_RESERVE, 'reserve changed')
    retained = integer(resources['retained_packet_bytes_excluding_distinct_GPU_ledger_charge'], 1, PACKET_CAP)
    extra = integer(resources['other_unwritten_control_reserve_bytes'], 0, PACKET_CAP)
    future = integer(resources['future_unspent_code_control_metrics_bytes'], FUTURE_RESERVE, PACKET_CAP)
    outputs = RECEIPT_CAP + LOG_CAP
    fixed = retained + extra + future + GPU_LEDGER_RESERVE + HEADER_RESERVE + TABLE_MANIFEST_RESERVE + FLOAT_PAYLOAD + outputs
    peak = fixed + compressed_integer_bytes
    require(peak <= PACKET_CAP, 'whole future packet/peak exceeds unchanged12MiB')
    global_base = integer(resources['global_existing_plus_other_reservations_excluding_row8_growth_and_GPU_ledger'], retained, GLOBAL_CAP)
    global_peak = global_base + peak-retained
    require(global_peak <= GLOBAL_CAP, 'whole global commitment cap')
    growth = peak-retained
    for field in ('free_root_bytes', 'free_mnt_bytes'):
        require(integer(resources[field], 0) >= 1073741824+growth, 'free floor plus peak growth')
    require(resources['all_retained_and_prototype_controls_counted'] is True, 'fresh retained controls not attested')
    require(resources['prior_failures_and_eight_protocols_preserved'] is True, 'old packet omitted')
    return {'retained_packet_bytes': retained, 'prototype_log_and_receipt_reserve_bytes': outputs,
            'other_unwritten_control_reserve_bytes': extra, 'future_unspent_reserve_bytes': future,
            'distinct_GPU_ledger_charge_bytes': GPU_LEDGER_RESERVE, 'float_payload_bytes_no_discount': FLOAT_PAYLOAD,
            'header_reserve_bytes': HEADER_RESERVE, 'table_manifest_reserve_bytes': TABLE_MANIFEST_RESERVE,
            'integer_compressed_stream_bytes': compressed_integer_bytes,
            'whole_future_packet_total_and_peak_upper_bound_bytes': peak,
            'packet_cap_bytes': PACKET_CAP, 'packet_margin_bytes': PACKET_CAP-peak,
            'global_peak_commitment_bytes': global_peak,
            'prototype_disk_peak_before_future_science_bytes': retained+extra+outputs,
            'peak_rule': 'Each future final float file once; one integer NPZ; one final CSV/manifest. No raw-then-archive, second payload copy, replacement output, or cleanup. Failed partials remain charged; a retry needs new full accounting.',
            'science_admission': False}

def measure_candidate(plan, ready, rows, resources, codec):
    # One candidate archive/codec at a time. No selected/fallback candidate exists.
    result = {'codec_candidate': codec, 'scientific_execution_admitted': False,
              'measurement_complete': False, 'complete_format_bound_fits': False}
    try:
        raw, high_water = encode_integer_archive(rows, codec=codec)
        info = inspect_integer_archive(raw, rows, codec=codec)
        table = table_bound(rows)
        manifest = manifest_bound(plan, ready, rows, info)
        require(info['members'] == 120 and table['rows'] == 2452 and manifest['file_array_entries'] == 280, 'complete layout counts')
        require(sum(row['T'] for row in rows)*16 == FLOAT_PAYLOAD, 'float payload count changed')
        float_headers = sum(len(npy_header('<f4', row['T'])) for row in rows)*4
        headers = float_headers+info['ZIP_container_bytes']
        total_meta = table['max_bytes']+manifest['manifest_max_bytes']
        proposed_peak = (resources['retained_packet_bytes_excluding_distinct_GPU_ledger_charge']
            + resources['other_unwritten_control_reserve_bytes'] + RECEIPT_CAP+LOG_CAP
            + resources['future_unspent_code_control_metrics_bytes'] + GPU_LEDGER_RESERVE
            + HEADER_RESERVE+TABLE_MANIFEST_RESERVE+FLOAT_PAYLOAD
            + info['compressed_integer_NPY_stream_bytes'])
        result.update({'integer_codec': {k: v for k, v in info.items() if k != 'member_NPY_sha256'},
            'in_memory_archive_high_water_bytes': high_water, 'float_header_bytes': float_headers,
            'actual_known_header_container_bytes': headers, 'section_table_bound': table,
            'complete_manifest_bound': manifest, 'table_and_manifest_max_bytes': total_meta,
            'proposed_packet_total_and_peak_IF_schema_reserves_hold_bytes': proposed_peak,
            'schema_header_reserves_hold': total_meta <= TABLE_MANIFEST_RESERVE and headers <= HEADER_RESERVE,
            'measurement_complete': True})
        try:
            result['storage'] = budget_bound(resources, info['compressed_integer_NPY_stream_bytes'], total_meta, headers)
            result.update(status='MEASURED_FORMAT_BOUND_FITS_NO_SCIENCE', complete_format_bound_fits=True)
        except ContractError as exc:
            result.update(status='MEASURED_FORMAT_OR_BUDGET_BOUND_REJECTED', error=str(exc)[:384])
        return result
    except MemoryError:
        raise  # resource exhaustion is not a reason to start another candidate
    except Exception as exc:
        result.update(status=('UNSUPPORTED' if 'UNSUPPORTED_ZIP_LZMA' in str(exc) else 'CODEC_MEASUREMENT_FAILED'),
                      measurement_complete=False, complete_format_bound_fits=False,
                      error=type(exc).__name__+': '+str(exc)[:384])
        return result

def measure(plan, ready, resources):
    rows = validate_metadata(plan, ready)
    validate_resources(resources)
    results = []
    started = time.monotonic()
    for codec in CODECS:
        if time.monotonic()-started >= 45:
            results.append({'codec_candidate': codec, 'status': 'NOT_RUN_TIME_GUARD',
                            'measurement_complete': False, 'complete_format_bound_fits': False,
                            'scientific_execution_admitted': False})
            continue
        results.append(measure_candidate(plan, ready, rows, resources, codec))
    complete = all(x['measurement_complete'] for x in results)
    return {'status': ('CPU_CODEC_CANDIDATES_MEASURED_NO_SCIENTIFIC_ADMISSION' if complete else 'CPU_CODEC_CANDIDATE_MEASUREMENT_INCOMPLETE'),
            'candidate_order_fixed_before_measurement': list(CODECS), 'candidate_results': results,
            'native_measurement_complete': complete, 'automatic_selection': False, 'automatic_fallback': False,
            'arrays_per_candidate': {'clips': 40, 'integer_arrays': 120, 'unknown_float_arrays': 160,
                                    'section_intersections': 613, 'section_arm_rows': 2452},
            'unknown_brightness_or_statistics_generated': False, 'scientific_archive_files_written': 0,
            'model_calls': 0, 'GPU_calls': 0, 'scientific_execution_admitted': False,
            'interpretation': 'Each codec was prospectively declared and reported independently. Nativeexit0 means measurements completed, NOT that either budget fits. No automatic choice, fallback, V6 amendment, archive publication, cache reread, scientific result or GPU budget grant.',
            'runtime_identity_scope': 'Recorded Python/NumPy/zlib identities and cached _lzma module path only. Linked liblzma version is not exposed. No binary/environment identity or future compressed-size invariance is established; changed runtime requires fresh validation before reuse.'}

def codec_versions():
    return {'python_major_minor': f'{sys.version_info.major}.{sys.version_info.minor}', 'python_full': sys.version,
            'lzma_backend_path': getattr(sys.modules.get('_lzma'), '__file__', None),
            'linked_liblzma_version': None,  # not exposed by stdlib; no binary identity claim
            'numpy': np.__version__, 'zlib_runtime': zlib.ZLIB_RUNTIME_VERSION,
            'zlib_compile': zlib.ZLIB_VERSION,
            'zip_candidates': [{'name': 'deflate9', 'ZIP_method': 8, 'level': 9},
                               {'name': 'lzma_default', 'ZIP_method': 14, 'level': 'stdlib_default',
                                'LZMA1_lc_lp_pb': [3, 0, 2], 'dict_bytes': LZMA_DICT_BYTES}],
            'stdlib_lzma_available': lzma is not None and zipfile.lzma is not None,
            'NPY_version': '1.0', 'ZIP64': False}

def validate_authorization(auth, now):
    exact_keys(auth, ('schema_version', 'status', 'scope', 'source_binding', 'native_test_attestation',
                      'codec_versions', 'plan_ref', 'readiness_ref', 'envelope_ref', 'resources',
                      'snapshot_epoch_s', 'root_attests_fresh_exact_command_accounting'), 'authorization')
    require(type(auth['schema_version']) is int and auth['schema_version'] == 1 and auth['status'] == 'ROOT_AUTHORIZED_CPU_FORMAT_PROTOTYPE_ONLY', 'not authorized')
    require(auth['scope'] == 'METADATA_INTEGER_CODEC_ONLY_NO_SCIENCE', 'scope changed')
    require(auth['root_attests_fresh_exact_command_accounting'] is True, 'root accounting attestation')
    age = finite(now)-finite(auth['snapshot_epoch_s'])
    require(0 <= age <= 300, 'stale/future resource accounting')
    require(auth['codec_versions'] == codec_versions(), 'codec/library versions differ from preregistration')
    binding = auth['source_binding']
    exact_keys(binding, ('driver_sha256', 'tests_sha256', 'test_count', 'root_verified_source_bytes'), 'source binding')
    require(binding['root_verified_source_bytes'] is True, 'source verification missing')
    require(all(isinstance(binding[k], str) and re.fullmatch('[0-9a-f]{64}', binding[k]) for k in ('driver_sha256', 'tests_sha256')), 'source hashes')
    require(binding['test_count'] == EXPECTED_TEST_COUNT and EXPECTED_TEST_COUNT > 0, 'test count not frozen')
    test = auth['native_test_attestation']
    exact_keys(test, ('exit_code', 'tests_run', 'failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes', 'plain_OK', 'native_receipt_ref'), 'test attestation')
    require(type(test['exit_code']) is int and test['exit_code'] == 0 and type(test['tests_run']) is int and test['tests_run'] == EXPECTED_TEST_COUNT and test['plain_OK'] is True, 'native test pass missing')
    require(all(type(test[k]) is int and test[k] == 0 for k in ('failures', 'errors', 'skipped', 'expected_failures', 'unexpected_successes')), 'nonplain test outcome')
    native = test['native_receipt_ref']
    exact_keys(native, ('path', 'sha256', 'size_bytes'), 'native test receipt ref')
    require(native['path'] in (str(ROOT/'NATIVE_LOSSLESS_ARCHIVE_TEST_RESULT_V1.json'), str(ROOT/'NATIVE_LOSSLESS_ARCHIVE_TEST_RESULT_V1.json.gz')), 'native test receipt path')
    require(isinstance(native['sha256'], str) and re.fullmatch('[0-9a-f]{64}', native['sha256']) is not None, 'native test receipt hash')
    integer(native['size_bytes'], 1, 16384)
    for key, path, sha, size in (('plan_ref', PLAN, PLAN_SHA, 285563), ('readiness_ref', READY, READY_SHA, 202345)):
        require(auth[key] == {'path': str(path), 'sha256': sha, 'size_bytes': size}, 'frozen CONTROL reference changed')
    envelope = auth['envelope_ref']
    exact_keys(envelope, ('path', 'sha256', 'size_bytes'), 'envelope ref')
    require(envelope['path'] == str(ROOT/'LOSSLESS_ARCHIVE_EXTERNAL_ENVELOPE_V1.json'), 'envelope path')
    require(re.fullmatch('[0-9a-f]{64}', envelope['sha256']) is not None, 'envelope hash')
    integer(envelope['size_bytes'], 1, 16384)
    return auth

def validate_envelope(envelope):
    require(envelope.get('status') == 'ENFORCED_BEFORE_EXEC', 'external enforcement proof missing')
    require(envelope.get('module_cli_argv') == [str(DRIVER), 'measure', '--authorization', str(AUTH)], 'external exact argv mismatch')
    required = {'MemoryMax_bytes': 536870912, 'MemorySwapMax_bytes': 0, 'RuntimeMaxSec': 50,
                'acceptance_wall_seconds': 60, 'PrivateDevices': True,
                'GPU_devices_visible': False, 'CUDA_VISIBLE_DEVICES': '',
                'process_tree_enforced': True}
    for key, value in required.items():
        require(type(envelope.get(key)) is type(value) and envelope[key] == value, 'external envelope mismatch: '+key)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA environment not hidden')

def write_receipt(result):
    raw = compact_json(result)+b'\n'
    require(len(raw) <= RECEIPT_CAP, 'receipt byte cap')
    path = no_symlink(OUT)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        total = 0
        while total < len(raw):
            n = os.write(fd, raw[total:])
            require(n > 0, 'short receipt write')
            total += n
        os.fsync(fd)
    finally:
        os.close(fd)
    return len(raw)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('measure',))
    parser.add_argument('--authorization', required=True, choices=(str(AUTH),))
    args = parser.parse_args(argv)
    started = time.monotonic()
    authorized = False
    try:
        auth = validate_authorization(read_control(args.authorization, 32768), time.time())
        ref = auth['envelope_ref']
        envelope = read_control(ref['path'], 16384, ref['sha256'], ref['size_bytes'])
        validate_envelope(envelope)
        require(envelope.get('driver_sha256') == auth['source_binding']['driver_sha256'] and envelope.get('tests_sha256') == auth['source_binding']['tests_sha256'], 'external source binding mismatch')
        validate_resources(auth['resources'])
        require(not OUT.exists() and not OUT.is_symlink(), 'receipt exists: preserve prior attempt')
        authorized = True
        plan = read_control(PLAN, 300000, PLAN_SHA, 285563)
        ready = read_control(READY, 220000, READY_SHA, 202345)
        result = measure(plan, ready, auth['resources'])
        result.update({'source_binding_root_attested': auth['source_binding'],
                       'codec_versions': codec_versions(), 'control_refs': {'plan': auth['plan_ref'], 'readiness': auth['readiness_ref'], 'envelope': ref},
                       'elapsed_internal_s': time.monotonic()-started,
                       'requires_root_verified_exit0_and_whole_job_wall_le60s': True,
                       'external_enforcement_is_root_bound_not_proven_by_CLI': True})
        write_receipt(result)
        print(json.dumps({'status': result['status'], 'receipt': str(OUT), 'scientific_execution_admitted': False}), flush=True)
        return 0 if result['native_measurement_complete'] else 2
    except Exception as exc:
        failure = {'status': 'CPU_FORMAT_PROTOTYPE_BLOCKED', 'error': type(exc).__name__+': '+str(exc)[:384],
                   'elapsed_internal_s': time.monotonic()-started, 'scientific_execution_admitted': False,
                   'no_archives_written': True, 'preserve_all_prior_controls_and_partial_receipts': True}
        if authorized:
            try:
                write_receipt(failure)
            except Exception:
                pass  # never overwrite/remove a partial or prior receipt
        print(json.dumps(failure), file=sys.stderr, flush=True)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
