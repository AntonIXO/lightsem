"""CPU contracts for the RTMPose all-partition fit, independent of extraction."""
from pathlib import Path
import hashlib
import json

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_normalization(row):
    """Validate both files and the scalar operation, rather than trust a label."""
    if sha(row['wav_path']) != row['wav_sha256']:
        raise ValueError('Waveform hash mismatch')
    if not row.get('normalization_receipt'):
        return None
    import soundfile as sf
    receipt = Path(row['normalization_receipt'])
    norm = json.loads(receipt.read_text())
    if (norm['recording_id'] != row['recording_id'] or
            norm['output_wav_sha256'] != row['wav_sha256'] or
            Path(norm['output_wav']).resolve() != Path(row['wav_path']).resolve() or
            sha(norm['source_wav']) != norm['source_wav_sha256']):
        raise ValueError('Normalization identity mismatch')
    gain = 10 ** (float(norm['gain_db']) / 20)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Invalid normalization gain')
    with sf.SoundFile(norm['source_wav']) as src, sf.SoundFile(norm['output_wav']) as dst:
        if (len(src), src.samplerate, src.channels) != (len(dst), dst.samplerate, dst.channels):
            raise ValueError('Normalization changed duration or channel layout')
        peak = 0.
        for x in src.blocks(blocksize=44100, dtype='float32', always_2d=True):
            y = dst.read(len(x), dtype='float32', always_2d=True)
            if not np.isfinite(y).all() or not np.allclose(y, x*gain, atol=2e-7, rtol=2e-6):
                raise ValueError('Normalization is not the declared scalar gain')
            peak = max(peak, float(np.max(np.abs(y))))
        if peak >= 1:
            raise ValueError('Normalized waveform clips')
    return {'receipt':str(receipt),'receipt_sha256':sha(receipt), 'gain_db':norm['gain_db']}


def _native_arrays(raw):
    """Validate and return the ragged RTMPose arrays in their raw layout.

    ``t`` is a source-frame clock.  The other arrays are detection rows and
    may therefore have a different first dimension; ``frame_index`` is the
    only permitted join between those two clocks.  This function deliberately
    does not fill missing detections.
    """
    required = ('t', 'K', 'C', 'box', 'track_id', 'candidate', 'frame_index')
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f'Missing native pose fields: {", ".join(missing)}')
    times = np.asarray(raw['t'], np.float64)
    k, c, boxes = (np.asarray(raw[name]) for name in ('K', 'C', 'box'))
    frame, ids, candidate = (np.asarray(raw[name]) for name in
                             ('frame_index', 'track_id', 'candidate'))
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid native PTS')
    n = k.shape[0] if k.ndim else -1
    if (k.ndim != 3 or k.shape != (n, 17, 2) or c.shape != (n, 17) or
            boxes.shape != (n, 4) or
            any(x.shape != (n,) for x in (frame, ids, candidate))):
        raise ValueError('Invalid ragged detection layout')
    if (frame.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu' or
            candidate.dtype.kind not in 'biu' or
            np.any(frame < 0) or np.any(frame >= len(times)) or
            np.any((ids >= 0) & ~candidate.astype(bool))):
        raise ValueError('Invalid frame/track assignment')
    if not np.isfinite(k).all() or not np.isfinite(c).all() or not np.isfinite(boxes).all():
        raise ValueError('Non-finite native detection values')
    return times, k, c, boxes, frame, ids, candidate


def native_metadata(raw):
    """Decode optional scalar metadata and check its source-frame count.

    Current caches use flat ``source_path``, ``source_sha256`` and ``n_frames``
    fields.  A nested ``source`` object is accepted for old diagnostic caches,
    but callers must not require it.
    """
    value = raw.get('meta') if hasattr(raw, 'get') else None
    if value is None:
        return {}
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError('Native metadata must be a scalar JSON string')
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if not isinstance(value, str):
        raise ValueError('Native metadata must be a JSON string')
    try:
        metadata = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError('Invalid native metadata JSON') from exc
    if not isinstance(metadata, dict):
        raise ValueError('Native metadata must be a JSON object')
    if 'n_frames' in metadata:
        try:
            n_frames = int(metadata['n_frames'])
        except (TypeError, ValueError) as exc:
            raise ValueError('Invalid native metadata n_frames') from exc
        if n_frames < 2:
            raise ValueError('Invalid native metadata n_frames')
    return metadata


def iter_native_tracks(raw):
    """Yield one independent track at a time; reject duplicate assignments."""
    times, k, c, boxes, frame, ids, candidate = _native_arrays(raw)
    metadata = native_metadata(raw)
    if 'n_frames' in metadata and int(metadata['n_frames']) != len(times):
        raise ValueError('Native metadata n_frames does not match t')
    for tid in np.unique(ids[ids >= 0]):
        selected = np.flatnonzero(ids == tid)
        index = frame[selected]
        if len(np.unique(index)) != len(index):
            raise ValueError('Multiple detections assigned to one track/frame')
        kk = np.full((len(times),17,2),np.nan,np.float32)
        cc = np.full((len(times),17),np.nan,np.float32)
        bb = np.full((len(times),4),np.nan,np.float32)
        kk[index], cc[index], bb[index] = k[selected], c[selected], boxes[selected]
        yield int(tid), kk, cc, bb


def window_union(starts, frames, width=480):
    starts = np.asarray(starts, np.int64)
    if np.any(starts < 0) or np.any(starts+width > frames):
        raise ValueError('Window outside recording')
    change = np.zeros(frames+1, np.int64)
    np.add.at(change, starts, 1)
    np.add.at(change, starts+width, -1)
    return np.cumsum(change[:-1]) > 0


def verify_muq(timeline, expected, waveform_sha256):
    """Check real per-track values, model pins, native clock and context recipe."""
    metadata = timeline.metadata
    model = metadata['muq']
    receipt = model['model_receipt']
    for field in ('model_id','revision'):
        if receipt.get(field) != expected[field]:
            raise ValueError(f'MuQ {field} mismatch')
    if receipt['artifacts']['model.safetensors']['sha256'] != expected['weights_sha256']:
        raise ValueError('MuQ weights mismatch')
    if model.get('layer') != expected['layer'] or model.get('frame_rate_hz') != expected['native_hz']:
        raise ValueError('MuQ actual layer/rate mismatch')
    if metadata.get('input_sha256') != waveform_sha256:
        raise ValueError('MuQ waveform mismatch')
    recipe = metadata['recipe']
    for key, field in [('muq_layer','layer'),('muq_native_hz','native_hz'),
                       ('muq_chunk_seconds','chunk_seconds')]:
        if recipe.get(key) != expected[field]:
            raise ValueError(f'MuQ {key} mismatch')
    if recipe.get('muq_boundary') not in (
        'nonoverlapping8s; final real8s crop contributes only timestamps after previous chunk',
        'overlap-aware native25Hz grid; overlapping real8s crops are timestamp-aligned and averaged',
    ):
        raise ValueError('MuQ context boundary mismatch')
    value, stamp = np.asarray(timeline.muq), np.asarray(timeline.muq_time)
    if (stamp.ndim != 1 or len(stamp) < 2 or value.shape != (len(stamp),1024) or
            not np.isfinite(stamp).all() or not np.isfinite(value).all() or
            np.any(np.diff(stamp) <= 0)):
        raise ValueError('Invalid native MuQ stream')
    hop=1/expected['native_hz']
    if (stamp[0] < 0 or stamp[0] > hop+1e-6 or
            stamp[-1] < metadata['duration_s']-hop-1e-6 or
            stamp[-1] > metadata['duration_s']+hop+1e-6 or
            np.any(np.diff(stamp)>hop+1e-6)):
        raise ValueError('Incomplete native MuQ duration')
    return {'frames':len(stamp),'first_s':float(stamp[0]),'last_s':float(stamp[-1]),
            'waveform_sha256':waveform_sha256,'model':model,'recipe':recipe}
