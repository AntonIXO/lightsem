"""Versioned, CPU-only Just Dance preparation for the frozen Stage 11 study.

Historical caches are inputs only.  No source index or cached tensor is reused
as identity truth.  Audio remains relative to the capture; scene and descriptor
arrays carry their explicit active-span offset.  ``prepare_data`` is resumable:
completed recording jobs are reused only when their dependency digest matches.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata

import numpy as np

CACHE = Path('/mnt/lightsem-data/cache')
SOURCE = Path('/mnt/lightsem-data/justdance')
REPO = Path(__file__).resolve().parents[2]
SPLIT_SALT = 'ls2-20260908/split/v1:'
FPS = 10.0
MIN_COVERAGE = .8
# Verified against the installed libchromaprint API for fpcalc algorithm 2.
FINGERPRINT_PERIOD_S = 1365 / 11025
FINGERPRINT_DELAY_S = 28666 / 11025
UI = ((0., 0., 1., .14), (.70, .72, 1., 1.),
      (0., .85, .45, 1.), (0., .30, .09, .80))

# These phrases were audited against every filename in the old collection
# before any model outcome was available.  They establish readable identities;
# audio matching still checks every recording, including this shortlist.
AUDITED_TITLES = (
    'bad boy|drum go dum|blinding lights|finesse|work work|dancing diva|i am the best|365|'
    '7 rings|bassa sababa|con altura|bangarang|old town road|get busy|keep in touch|'
    'policeman|vodovorot|baby shark|god is a woman|buscando|intoxicated|taki taki|'
    'come back home|last christmas|juice|boy you can keep it|bailando|heat seeker|'
    'dance monkey|feel special|kulikitaka|ice cream|lacrimosa|rainbow beats|'
    'here comes the spy|space cat|rare|stupid love|all the good girls go to hell|'
    'kick it|samba de janeiro|without me|que tire pa lante|uno|the other side|'
    'you ve got a friend in me|volar|the weekend|boss witch|yo le llego|levitating|'
    'jopping|sua cara|last friday night|level up|buttons|fit but you know it|'
    'tightrope|little apple|we can t stop|dibby dibby sound|georgia|i feel it coming|'
    '1999|white noise|heart of glass|crayon|djadja|jungle dances|hype|into the unknown|'
    'kill this love|la respuesta|magical morning|ma itu|le bal masque|panini|'
    'rain over me|skibidi|my new swag|stop movin|sucker|sushi|the color lab|'
    'so depois do carnaval|the frog concert|talk|tel aviv|woman like me|ez do dance|'
    'kitchen kittens|my friend the dragon|magenta riddim|me too|yameen yasar|'
    'what lovers do|dare to live|hot pot cool|bailemen swing|clown|gorgeous diva|'
    'tu zi wu|sweet sensation|water me|sangria wine|mad love|one kiss|calypso|'
    'rave in the grave|shape of you|rockabye|blow your mind|the greatest|youth|'
    'sax|imya 505|juju on that beat|je sais pas danser|little swing|lush life|'
    'peanut butter jelly'
).split('|')


def _json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    """Interrupted legacy-cache copies cannot masquerade as completed inputs."""
    partial = destination.with_suffix(destination.suffix+'.partial')
    shutil.copyfile(source, partial)
    os.replace(partial, destination)


def _digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _data_dir(root: Path) -> Path:
    root = Path(root).resolve()
    return root if root.name == 'data' else root / 'data'


def normalize_title(value: str) -> str:
    """Conservative title text; brackets and game/platform boilerplate removed."""
    value = unicodedata.normalize('NFKD', value.lower())
    value = ''.join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r'\[[^]]*\]', ' ', value)
    value = re.sub(r'just\s+dance\s*(?:20\d\d|[1-4])?', ' ', value)
    value = re.sub(r'\b(?:megastar|superstar|unlimited|unlimted|gameplay|official|'
                   r'extreme|extrema|alternate|alternative|version|nintendo|switch|'
                   r'playstation|xbox|kinect|wii|stars?|perfects?|1080p|720p|60fps|'
                   r'controller|walkthrough|edition|full|sweat|hud|plus|pc)\b', ' ', value)
    value = re.sub(r'\b20\d\d\b', ' ', value)
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', value).split())


def _plain_title(value: str) -> str:
    value = unicodedata.normalize('NFKD', value.lower())
    value = ''.join(c for c in value if not unicodedata.combining(c))
    return ' ' + ' '.join(re.sub(r'[^a-z0-9]+', ' ', value).split()) + ' '


def active_span(k: np.ndarray, times: np.ndarray, fps: float) -> tuple[int, int, float, float, np.ndarray]:
    if k.ndim == 3:
        k = k[:, None]
    seen = np.isfinite(k[:, :, 0, 0]).any(axis=1)
    if not seen.any():
        return 0, 0, 0., 0., np.zeros(k.shape[1])
    first = int(np.argmax(seen))
    last = int(len(seen) - np.argmax(seen[::-1]))
    start = float(times[first])
    duration = (last - first) / fps
    coverage = np.isfinite(k[first:last, :, 0, 0]).mean(axis=0)
    return first, last, start, duration, coverage


def normalize_scene(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit only recording-relative value dimensions, once before any crop."""
    if raw.ndim != 2 or raw.shape[1] != 6 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError('Scene targets must be nonempty, finite (time, 6) arrays')
    mean, scale = np.zeros(6, np.float32), np.ones(6, np.float32)
    mean[3:5] = raw[:, 3:5].mean(axis=0, dtype=np.float64)
    scale[3:5] = np.maximum(raw[:, 3:5].std(axis=0, dtype=np.float64), 1. / 255.)
    return ((raw - mean) / scale).astype(np.float32), mean, scale


def resize_mask(boxes: np.ndarray, original_wh: tuple[int, int], output_wh: tuple[int, int],
                *, include_ui: bool = True, dilation: float = .15) -> np.ndarray:
    """Pose boxes are original-image pixels, independent of pose network imgsz."""
    ow, oh = original_wh
    w, h = output_wh
    mask = np.full((h, w), 255, np.uint8)
    if include_ui:
        for x0, y0, x1, y1 in UI:
            mask[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = 0
    sx, sy = w / ow, h / oh
    for box in boxes:
        if not np.isfinite(box).all():
            continue
        x0, y0, x1, y1 = map(float, box)
        dx, dy = (x1-x0)*dilation, (y1-y0)*dilation
        a, b = max(0, math.floor((x0-dx)*sx)), min(w, math.ceil((x1+dx)*sx)+1)
        c, d = max(0, math.floor((y0-dy)*sy)), min(h, math.ceil((y1+dy)*sy)+1)
        if a < b and c < d:
            mask[c:d, a:b] = 0
    return mask


def recording_mask_dilation(boxes: np.ndarray, pose_times: np.ndarray,
                            absolute_times: np.ndarray, original_wh: tuple[int, int],
                            output_wh: tuple[int, int]) -> tuple[float, int]:
    """Use one padding policy per recording, avoiding mask-induced scene cuts.

    Entire detected coach boxes always remain masked. If 15% padding erases
    the observable background, remove padding for the complete recording.
    A recording without 100 observable pixels even then fails explicitly.
    """
    indices = np.unique(np.clip(np.searchsorted(pose_times, absolute_times, side='right')-1,
                                0, len(pose_times)-1))
    for dilation in (.15, 0.):
        minimum = output_wh[0]*output_wh[1]
        for index in indices:
            pixels = int(np.count_nonzero(resize_mask(boxes[index], original_wh, output_wh, dilation=dilation)))
            minimum = min(minimum, pixels)
            if pixels < 100:
                break
        if minimum >= 100:
            return dilation, minimum
    raise ValueError('Coach boxes leave fewer than 100 background pixels without any padding')


def reduce_histograms(hist: np.ndarray) -> np.ndarray:
    """The historical RPMC six-vector reduction, including entropy concentration."""
    h, s, v = hist[:, 0, :180].astype(float), hist[:, 1].astype(float), hist[:, 2].astype(float)
    bins = np.arange(256, dtype=float)
    hp = h / np.maximum(h.sum(1, keepdims=True), 1.)
    angles = np.arange(180) * 2*np.pi/180
    conc = 1 + (hp*np.log(hp+1e-12)).sum(1)/np.log(180.)
    sm = (s*bins).sum(1)/np.maximum(s.sum(1), 1.)/255
    vm = (v*bins).sum(1)/np.maximum(v.sum(1), 1.)/255
    vc = np.cumsum(v, axis=1)/np.maximum(v.sum(1, keepdims=True), 1.)
    p90 = np.argmax(vc >= .9, axis=1)/255
    return np.stack([(hp*np.cos(angles)).sum(1), (hp*np.sin(angles)).sum(1), sm, vm, p90, conc], 1).astype(np.float32)


def fingerprint_score(a: np.ndarray, b: np.ndarray, offset: int, window: int = 256) -> dict:
    """Best ~32-second aligned window, scored by bit errors, with overlap evidence."""
    a, b = np.asarray(a, np.uint32), np.asarray(b, np.uint32)
    ia, ib = max(0, offset), max(0, -offset)
    n = min(len(a)-ia, len(b)-ib)
    if n < window:
        return {'bit_error': 1., 'overlap_frames': max(0, n), 'offset_frames': offset}
    xor = np.bitwise_xor(a[ia:ia+n], b[ib:ib+n])
    lut = np.array([int(x).bit_count() for x in range(256)], np.uint8)
    errors = lut[xor.view(np.uint8).reshape(-1, 4)].sum(1).astype(float) / 32
    sums = np.r_[0., np.cumsum(errors)]
    roll = (sums[window:] - sums[:-window]) / window
    return {'bit_error': float(roll.min()), 'median_bit_error': float(np.median(errors)),
            'overlap_frames': n, 'offset_frames': offset, 'window_frames': window,
            'best_window_start': int(roll.argmin())}


def fingerprint_candidates(fingerprints: dict[str, np.ndarray]) -> dict[tuple[str, str], list[int]]:
    """Candidate retrieval by exact fingerprint words and offset votes, all pairs.

    This does not require menus to have equal length. Highly common words and
    within-track repetitions are omitted to avoid silence dominating retrieval.
    The final decision always uses the full 32-bit aligned error score.
    """
    index: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for rid, fp in fingerprints.items():
        words: dict[int, list[int]] = defaultdict(list)
        for position, word in enumerate(fp):
            words[int(word)].append(position)
        for word, positions in words.items():
            if 1 <= len(positions) <= 8 and word not in (0, 2**32-1):
                index[word].extend((rid, i) for i in positions)
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for hits in index.values():
        if not 2 <= len(hits) <= 40 or len({r for r, _ in hits}) < 2:
            continue
        for i, (ra, ia) in enumerate(hits):
            for rb, ib in hits[i+1:]:
                if ra == rb:
                    continue
                if ra > rb:
                    key, off = (rb, ra), ib-ia
                else:
                    key, off = (ra, rb), ia-ib
                votes[key][round(off/3)*3] += 1
    return {pair: [off for off, count in counts.most_common(3) if count >= 6]
            for pair, counts in votes.items() if counts.most_common(1)[0][1] >= 6}


def active_fingerprint(fp: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    """Keep fingerprint support inside the dance span, excluding shared menus."""
    first = max(0, math.ceil(start_s/FINGERPRINT_PERIOD_S))
    last = max(first, math.floor((end_s-FINGERPRINT_DELAY_S)/FINGERPRINT_PERIOD_S))
    return np.asarray(fp, np.uint32)[first:last]


def canonicalize(rows: list[dict], old_songs: dict, fingerprints: dict[str, np.ndarray],
                 *, test_count: int = 40, val_count: int = 40) -> dict:
    """Freeze canonical components and a deterministic split without outcome data."""
    by_id = {r['recording_id']: r for r in rows}
    fingerprints = {rid: active_fingerprint(fp, by_id[rid]['active_start_s'], by_id[rid]['active_end_s'])
                    if 'active_start_s' in by_id[rid] else fp for rid, fp in fingerprints.items()}
    parent = {rid: rid for rid in by_id}
    edges: list[dict] = []
    uncertain: set[str] = set()
    titles = {rid: normalize_title(Path(r['source_path']).stem) for rid, r in by_id.items()}
    readable: dict[str, list[str]] = defaultdict(list)

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str, reason: str, **evidence: object) -> None:
        if a not in parent or b not in parent:
            return
        aa, bb = find(a), find(b)
        parent[max(aa, bb)] = min(aa, bb)
        edges.append({'a': a, 'b': b, 'reason': reason, **evidence})

    prior_groups: dict[str, list[str]] = defaultdict(list)
    for r in old_songs.get('rows', []):
        if r['yt_id'] in by_id:
            prior_groups[str(r.get('song_key', r['yt_id']))].append(r['yt_id'])
    for ids in prior_groups.values():
        for rid in ids[1:]:
            union(ids[0], rid, 'existing_song_map')
    for ids in old_songs.get('duplicate_groups', {}).values():
        for rid in ids[1:]:
            union(ids[0], rid, 'existing_duplicate_link')

    exact: dict[str, list[str]] = defaultdict(list)
    for rid, title in titles.items():
        if len(title) >= 5:
            exact[title].append(rid)
    for title, ids in exact.items():
        for rid in ids[1:]:
            union(ids[0], rid, 'exact_normalized_title', title=title)
    for phrase in AUDITED_TITLES:
        ids = [rid for rid, row in by_id.items()
               if _plain_title(phrase) in _plain_title(Path(row['source_path']).stem)]
        for rid in ids:
            readable[rid].append(phrase)
        for rid in ids[1:]:
            union(ids[0], rid, 'audited_title_phrase', title=phrase)
    # Conservative broad title links: a multiword title contained in another
    # filename can be an alternate arrangement. Merge rather than split it.
    ids = sorted(by_id)
    for i, a in enumerate(ids):
        ta = titles[a]
        for b in ids[i+1:]:
            tb = titles[b]
            shorter, longer = sorted((ta, tb), key=len)
            if (len(shorter.split()) >= 2 and len(shorter) >= 8
                    and f' {shorter} ' in f' {longer} '
                    and len(shorter)/max(len(longer), 1) >= .45):
                union(a, b, 'conservative_title_containment', title=shorter)

    candidates = fingerprint_candidates(fingerprints)
    audio_checks = []
    for (a, b), offsets in sorted(candidates.items()):
        scores = [fingerprint_score(fingerprints[a], fingerprints[b], shift)
                  for offset in offsets for shift in range(offset-2, offset+3)]
        best = min(scores, key=lambda s: s['bit_error'])
        audio_checks.append({'a': a, 'b': b, **best})
        if best['bit_error'] <= .18:
            union(a, b, 'aligned_chromaprint', **best)
        elif best['bit_error'] <= .27:
            uncertain.update((a, b))

    groups: dict[str, list[str]] = defaultdict(list)
    for rid in ids:
        groups[find(rid)].append(rid)
    records, components, eligible = {}, [], []
    for members in groups.values():
        canonical_id = 'jd-' + hashlib.sha256('\n'.join(sorted(members)).encode()).hexdigest()[:20]
        is_old = any(by_id[r]['is_old'] for r in members)
        reasons = []
        if any(r not in fingerprints or len(fingerprints[r]) < 256 for r in members):
            reasons.append('missing_or_short_audio_fingerprint')
        if any(r in uncertain for r in members):
            reasons.append('unresolved_audio_similarity')
        known_titles = sorted({p for r in members for p in readable.get(r, [])})
        if not known_titles and not is_old:
            reasons.append('identity_not_in_audited_title_shortlist')
        if len(known_titles) > 1:
            reasons.append('multiple_audited_song_identities')
        qualified = any(by_id[r].get('coaches') for r in members)
        if not qualified:
            reasons.append('no_qualified_coach')
        c = {'canonical_id': canonical_id, 'recording_ids': sorted(members), 'is_old': is_old,
             'audited_titles': known_titles, 'holdout_exclusion_reasons': reasons,
             'eligible_new_holdout': not is_old and not reasons}
        components.append(c)
        if c['eligible_new_holdout']:
            eligible.append(canonical_id)
        for rid in members:
            records[rid] = {'canonical_id': canonical_id, 'identity_status': 'verified' if not reasons else 'conservative',
                            'holdout_exclusion_reasons': reasons, 'title_normalized': titles[rid]}
    needed = test_count + val_count
    if len(eligible) < needed:
        raise ValueError(f'Only {len(eligible)} verified new canonical families; require {needed} '
                         f'({test_count} test + {val_count} validation).')
    eligible.sort(key=lambda cid: hashlib.sha256((SPLIT_SALT+cid).encode()).hexdigest())
    tests, vals = set(eligible[:test_count]), set(eligible[test_count:needed])
    for rid, r in records.items():
        r['partition'] = ('test' if r['canonical_id'] in tests else
                          'val' if r['canonical_id'] in vals else 'train')
    for c in components:
        c['partition'] = 'test' if c['canonical_id'] in tests else 'val' if c['canonical_id'] in vals else 'train'
    return {'schema_version': 1, 'salt': SPLIT_SALT, 'records': records,
            'components': sorted(components, key=lambda c: c['canonical_id']), 'edges': edges,
            'audio_checks': audio_checks, 'n_audio_candidates': len(candidates),
            'n_verified_new_families': len(eligible), 'test': sorted(tests), 'val': sorted(vals),
            'audio_method': {'tool': 'Chromaprint fpcalc', 'length_s': 600,
                             'candidate_min_offset_votes': 6, 'confirmed_bit_error_max': .18,
                             'ambiguous_bit_error_max': .27, 'window_frames': 256,
                             'recording_active_span_only': True,
                             'fingerprint_period_s': FINGERPRINT_PERIOD_S,
                             'fingerprint_delay_s': FINGERPRINT_DELAY_S}}


def _worker_init() -> None:
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[key] = '1'
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _audio_job(job: dict) -> dict:
    from lightsem.audio.online import extract_bus
    from lightsem.motion.io import load_bus, save_bus
    rid, data = job['recording_id'], Path(job['data_dir'])
    wav, bus_path = data/'wav'/f'{rid}.wav', data/'audio'/f'{rid}.npz'
    fp_path, state_path = data/'fingerprints'/f'{rid}.json', data/'audio_state'/f'{rid}.json'
    t0 = time.monotonic()
    try:
        if state_path.exists():
            old = json.loads(state_path.read_text())
            if old.get('dependency_digest') == job['dependency_digest'] and all(Path(p).exists() for p in old['artifacts']):
                return old
            raise ValueError('Prepared audio input drift: use a new versioned campaign directory')
        for p in (wav, bus_path, fp_path, state_path):
            p.parent.mkdir(parents=True, exist_ok=True)
        old_wav = CACHE/'justdance_wav'/f'{rid}.wav'
        if not wav.exists():
            if old_wav.exists():
                atomic_copy(old_wav, wav)
            else:
                tmp = wav.with_suffix('.partial.wav')
                subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', job['source_path'],
                                '-vn', '-ac', '1', '-ar', '44100', '-y', str(tmp)],
                               capture_output=True, check=True)
                os.replace(tmp, wav)
        old_bus = CACHE/'justdance_audio'/f'{rid}.npz'
        if not bus_path.exists():
            if old_bus.exists():
                atomic_copy(old_bus, bus_path)
            else:
                bus = extract_bus(str(wav), causal=True)
                tmp = bus_path.with_suffix('.partial.npz')
                save_bus(bus, tmp)
                os.replace(tmp, bus_path)
        bus = load_bus(bus_path)
        if not np.isfinite(bus.mel).all() or any(not np.isfinite(v).all() for v in bus.channels.values()):
            raise ValueError('Nonfinite audio bus')
        if abs(bus.fps-40.) > 1e-6:
            raise ValueError(f'Expected 40 Hz audio, got {bus.fps}')
        if not fp_path.exists():
            cp = subprocess.run(['fpcalc', '-raw', '-json', '-length', '600', str(wav)],
                                capture_output=True, text=True)
            if not cp.stdout.strip().startswith('{'):
                raise ValueError(f'fpcalc failed: {cp.stderr[:180]}')
            fp = json.loads(cp.stdout)
            if len(fp.get('fingerprint', [])) < 256:
                raise ValueError('Short Chromaprint fingerprint')
            fp.update({'recording_id': rid, 'wav_sha256': sha256(wav), 'stderr': cp.stderr.strip()})
            _json(fp_path, fp)
        artifacts = {str(p): sha256(p) for p in (wav, bus_path, fp_path)}
        result = {'recording_id': rid, 'status': 'ok', 'dependency_digest': job['dependency_digest'],
                  'audio_path': str(bus_path), 'wav_path': str(wav), 'fingerprint_path': str(fp_path),
                  'duration_s': float(bus.duration_s), 'elapsed_s': time.monotonic()-t0, 'artifacts': artifacts}
        _json(state_path, result)
        return result
    except Exception as error:
        return {'recording_id': rid, 'status': 'error', 'error': f'{type(error).__name__}: {error}',
                'elapsed_s': time.monotonic()-t0}


def _audio52(wav_path: str, absolute_times: np.ndarray) -> tuple[np.ndarray, list[str], float]:
    import librosa
    from lightsem.audio.online import causal_resample, ema
    # The legacy module reads argv at import time; isolate that side effect.
    argv = sys.argv
    try:
        sys.argv = sys.argv[:1]
        from extract_rpmc_causal import features_from_mel, SR, HOP
    finally:
        sys.argv = argv
    audio, _ = librosa.load(wav_path, sr=SR)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=2048, hop_length=HOP, n_mels=128)
    channels, grid = features_from_mel(librosa.power_to_db(mel).astype(np.float32))
    fps_in = SR/HOP
    times = np.arange(mel.shape[1])/fps_in
    cols, names = [], []
    for name, values in channels.items():
        if not np.isfinite(values).all():
            raise ValueError(f'Nonfinite causal audio channel: {name}')
        for tau in (.2, 1., 4.):
            cols.append(causal_resample(ema(values, fps_in, tau), times, absolute_times))
            names.append(f'{name}_ema{tau}')
    per = grid['period_s']
    phase = np.mod((absolute_times-grid['phase_s'])/per, 1.)
    cols.extend((np.sin(2*np.pi*phase), np.cos(2*np.pi*phase), (1-phase)**2.5,
                 np.full(len(phase), per)))
    names.extend(('beat_sin', 'beat_cos', 'beat_accent', 'beat_period_s'))
    x = np.stack(cols, 1).astype(np.float32)
    if x.shape[1] != 52 or not np.isfinite(x).all():
        raise ValueError('Expected finite 52-channel scene audio')
    return x, names, float(grid['bpm'])


def _histogram_video(row: dict, boxes: np.ndarray, pose_times: np.ndarray,
                     pose_meta: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    import cv2
    cv2.setNumThreads(1)
    ow, oh = int(pose_meta['width']), int(pose_meta['height'])
    w, h = min(640, ow), max(2, round(oh*min(640, ow)/ow/2)*2)
    start, duration = row['active_start_s'], row['active_end_s']-row['active_start_s']
    mask_times = start+np.arange(int(math.ceil(duration*FPS))+1)/FPS
    dilation, minimum_pixels = recording_mask_dilation(boxes, pose_times, mask_times, (ow, oh), (w, h))
    command = ['ffmpeg', '-v', 'error', '-threads', '1', '-ss', f'{start:.9f}', '-i', row['source_path'],
               '-t', f'{duration:.9f}', '-an', '-vf', f'fps=10:round=up:start_time=0,scale={w}:{h}',
               '-threads', '1', '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']
    stderr_file = tempfile.TemporaryFile()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file)
    hist, frame_n, remaining = [], 0, w*h*3
    try:
        while True:
            chunks, needed = [], remaining
            while needed:
                part = process.stdout.read(needed)
                if not part:
                    break
                chunks.append(part)
                needed -= len(part)
            raw = b''.join(chunks)
            if not raw:
                break
            if len(raw) != remaining:
                raise ValueError('Truncated decoded video frame')
            img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
            t = start + frame_n/FPS
            pi = int(np.clip(np.searchsorted(pose_times, t, side='right')-1, 0, len(pose_times)-1))
            mask = resize_mask(boxes[pi], (ow, oh), (w, h), dilation=dilation)
            if (mask != 0).sum() < 100:
                raise ValueError(f'Insufficient unmasked background pixels at {t:.3f}s')
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hh = np.zeros((3, 256), np.float32)
            hh[0, :180] = cv2.calcHist([hsv], [0], mask, [180], [0, 180]).ravel()
            hh[1] = cv2.calcHist([hsv], [1], mask, [256], [0, 256]).ravel()
            hh[2] = cv2.calcHist([hsv], [2], mask, [256], [0, 256]).ravel()
            hist.append(hh)
            frame_n += 1
        returncode = process.wait(timeout=30)
        stderr_file.seek(0)
        stderr = stderr_file.read().decode(errors='replace')
        if returncode != 0:
            raise ValueError(f'ffmpeg visual decode: {stderr[:200]}')
    finally:
        if process.poll() is None:
            process.stdout.close()
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        process.stdout.close()
        stderr_file.close()
    if len(hist) < 600:
        raise ValueError(f'Active scene too short for 600-frame crop: {len(hist)} frames')
    return np.stack(hist), start + np.arange(len(hist))/FPS, {
        'histogram_width': w, 'histogram_height': h, 'pose_original_width': ow,
        'pose_original_height': oh, 'pose_box_scale_xy': [w/ow, h/oh],
        'coach_box_dilation': dilation, 'minimum_visible_background_pixels': minimum_pixels,
        'decode_fps': FPS, 'decode_rounding': 'up', 'pose_selection': 'previous_timestamp',
        'max_video_sampling_uncertainty_s': 1/max(float(pose_meta['fps_src']), 1.)}


def _feature_job(job: dict) -> dict:
    from lightsem.motion.io import load_bus, save_descriptors
    from lightsem.motion.pose2d import to_skeleton
    from lightsem.motion.descriptors import extract
    from lightsem.eval.bas import circular_phase
    rid, data = job['recording_id'], Path(job['data_dir'])
    state_path = data/'feature_state'/f'{rid}.json'
    visual_path = data/'visual'/f'{rid}.npz'
    t0 = time.monotonic()
    try:
        if state_path.exists():
            prior = json.loads(state_path.read_text())
            if prior.get('dependency_digest') == job['dependency_digest'] and all(Path(p).exists() for p in prior['artifacts']):
                return prior
            raise ValueError('Prepared feature input drift: use a new versioned campaign directory')
        with np.load(job['pose_path'], allow_pickle=False) as z:
            k, c, boxes, times = z['K'], z['C'], z['box'], z['t']
            meta = json.loads(str(z['meta']))
        if k.ndim == 3:
            k, c, boxes = k[:, None], c[:, None], boxes[:, None]
        f0, f1, offset, duration, coverages = active_span(k, times, float(meta['fps_eff']))
        if abs(offset-job['active_start_s']) > 1e-6:
            raise ValueError('Pose active offset changed after identity scan')
        bus = load_bus(job['audio_path'])
        coaches = []
        for coach in job['coaches']:
            pi, cid = coach['coach_index'], coach['coach_id']
            path = data/'motion'/f'{cid}.npz'
            stats: dict = {}
            sk = to_skeleton(k[f0:f1, pi].astype(float), c[f0:f1, pi].astype(float),
                             float(meta['fps_eff']), name=cid, stats=stats)
            beats = bus.beats-offset
            beats = beats[(beats >= 0) & (beats < sk.duration_s)]
            d = extract(sk, beats=beats, tempo=bus.tempo)
            if any(not np.isfinite(v).all() for v in d.channels.values()):
                raise ValueError(f'Nonfinite motion descriptor: {cid}')
            ph = circular_phase(d.events['footfall'], beats)
            fields = {'corpus': 'jd', 'genre': 'jd', 'setting': 'jd', 'camera': 'c',
                      'dancer': cid, 'music_id': rid, 'choreo_id': f'c{pi}',
                      'song_key': job['canonical_id'], 'canonical_id': job['canonical_id'],
                      'partition': job['partition'], 'n_coaches': int(k.shape[1]), 'coach_index': pi,
                      'view': 'frontal', 'view_provenance': 'asserted', 'coverage': float(coverages[pi]),
                      'start_s': offset, 'end_s': offset+sk.duration_s, 'duration_s': sk.duration_s,
                      'fps_src': meta['fps_src'], 'fps_eff': meta['fps_eff'], 'pose_model': meta.get('model'),
                      'pose_imgsz': meta.get('imgsz'), 'filled_frac': stats.get('filled_frac'),
                      'long_gap_frac': stats.get('long_gap_frac'), 'tempo': float(bus.tempo),
                      'beat_phase_R': float(ph['R']) if np.isfinite(ph['R']) else 0.,
                      'beat_offset_s': float(ph['offset_s']) if np.isfinite(ph['offset_s']) else 0.}
            d.meta.update(fields)
            path.parent.mkdir(parents=True, exist_ok=True)
            save_descriptors(d, path)
            coaches.append({**coach, **fields, 'name': cid, 'motion_path': str(path)})
        hist, absolute_times, visual_meta = _histogram_video(job, boxes, times, meta)
        raw_y = reduce_histograms(hist)
        y, y_mean, y_scale = normalize_scene(raw_y)
        x, names, bpm = _audio52(job['wav_path'], absolute_times)
        visual_meta.update({'normalization_scope': 'complete_active_recording_before_crop',
                            'normalized_columns': [3, 4], 'recording_id': rid,
                            'canonical_id': job['canonical_id'], 'partition': job['partition'],
                            'active_start_s': offset, 'duration_s': len(y)/FPS,
                            'masked_histograms': True, 'hue_concentration': 'one_minus_normalized_entropy'})
        visual_path.parent.mkdir(parents=True, exist_ok=True)
        temp = visual_path.with_suffix('.partial.npz')
        np.savez_compressed(temp, x=x, y=y, raw_y=raw_y, hue_hist=hist[:, 0, :180],
                            saturation_hist=hist[:, 1], value_hist=hist[:, 2],
                            y_recording_mean=y_mean, y_recording_scale=y_scale,
                            names=np.asarray(names), fps=np.float32(FPS), offset_s=np.float64(offset),
                            times=(absolute_times-offset).astype(np.float32), bpm=np.float32(bpm),
                            meta=json.dumps(visual_meta))
        os.replace(temp, visual_path)
        artifacts = {str(p): sha256(p) for p in [visual_path]+[Path(c['motion_path']) for c in coaches]}
        result = {'recording_id': rid, 'status': 'ok', 'dependency_digest': job['dependency_digest'],
                  'visual_path': str(visual_path), 'visual_frames': len(y), 'coaches': coaches,
                  'artifacts': artifacts, 'elapsed_s': time.monotonic()-t0}
        _json(state_path, result)
        return result
    except Exception as error:
        return {'recording_id': rid, 'status': 'error', 'error': f'{type(error).__name__}: {error}',
                'elapsed_s': time.monotonic()-t0}


def _jobs(fn, jobs: list[dict], workers: int, label: str, data: Path) -> dict[str, dict]:
    results = {}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as executor:
        pending = {executor.submit(fn, job): job['recording_id'] for job in jobs}
        for future in as_completed(pending):
            rid = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = {'recording_id': rid, 'status': 'error', 'error': f'{type(error).__name__}: {error}'}
            results[rid] = result
            print(f'{label} [{len(results)}/{len(jobs)}] {rid} {result["status"]} '
                  f'{result.get("error", "")} elapsed={time.monotonic()-started:.0f}s', flush=True)
            _json(data/f'{label}_progress.json', {'completed': len(results), 'total': len(jobs),
                                                'elapsed_s': time.monotonic()-started,
                                                'errors': {k: v for k, v in results.items() if v['status'] != 'ok'}})
    _json(data/f'{label}_results.json', results)
    return results


def _prepare_data(root: Path, workers: int = 4) -> dict:
    """Prepare all usable captures, then freeze the leakage-safe study manifest."""
    if not 1 <= workers <= 4:
        raise ValueError('CPU preparation supports one to four workers')
    data = _data_dir(Path(root))
    data.mkdir(parents=True, exist_ok=True)
    if (data/'manifest.json').exists():
        return load_manifest(root)
    started = time.monotonic()
    sources = json.loads((CACHE/'justdance_manifest.json').read_text())
    old_songs = json.loads((CACHE/'justdance_songs.json').read_text())
    old_ids = {r['yt_id'] for r in old_songs['rows']}
    deps = {str(p): sha256(p) for p in [Path(__file__), CACHE/'justdance_manifest.json',
            CACHE/'justdance_songs.json', REPO/'lightsem/audio/online.py', REPO/'lightsem/audio/beatgrid.py',
            REPO/'lightsem/motion/descriptors.py', REPO/'lightsem/motion/pose2d.py',
            REPO/'lightsem/motion/skeleton.py', REPO/'lightsem/motion/io.py', REPO/'extract_rpmc.py',
            REPO/'extract_rpmc_causal.py']}
    snapshot = {'schema_version': 1, 'dependencies': deps, 'workers': workers, 'source_cache': str(CACHE)}
    snap_path = data/'preparation_inputs.json'
    if snap_path.exists() and json.loads(snap_path.read_text())['dependencies'] != deps:
        raise ValueError('Preparation dependencies changed; inspect and explicitly version the data directory')
    _json(snap_path, snapshot)
    rows, exclusions = [], []
    for src in sources:
        rid = src.get('yt_id')
        if not rid or 'error' in src or (src.get('mean_db') if src.get('mean_db') is not None else -99) <= -40:
            exclusions.append({'recording_id': rid, 'reason': 'unusable_source_audio', 'source': src})
            continue
        source, pose = SOURCE/src['file'], CACHE/'justdance_pose'/f'{rid}.npz'
        if not source.exists() or not pose.exists():
            raise ValueError(f'Missing source or pose for usable capture {rid}')
        with np.load(pose, allow_pickle=False) as z:
            k, times = z['K'], z['t']
            meta = json.loads(str(z['meta']))
            _, _, offset, duration, cov = active_span(k, times, float(meta['fps_eff']))
        row = {'recording_id': rid, 'is_old': rid in old_ids, 'source_path': str(source),
               'pose_path': str(pose), 'source_sha256': sha256(source), 'pose_sha256': sha256(pose),
               'active_start_s': offset, 'active_end_s': offset+duration, 'duration_s': float(src['duration_s']),
               'pose_provenance': meta, 'coaches': [
                   {'coach_id': f'{rid}_c{i}', 'coach_index': i, 'coverage': float(v),
                    'start_s': offset, 'duration_s': duration}
                   for i, v in enumerate(cov) if v >= MIN_COVERAGE],
               'coach_exclusions': [{'coach_index': i, 'coverage': float(v), 'reason': 'coverage_below_0.8'}
                                    for i, v in enumerate(cov) if v < MIN_COVERAGE]}
        row['eligible_core'] = bool(row['coaches']) and duration >= 60.
        if not row['coaches']:
            row['exclusion_reason'] = 'no_qualified_coach'
        elif duration < 60.:
            row['exclusion_reason'] = 'active_span_below_60_seconds'
            row['coach_exclusions'].extend({**coach, 'reason': 'recording_active_span_below_60_seconds'}
                                            for coach in row['coaches'])
            row['coaches'] = []
        row['data_dir'] = str(data)
        row['dependency_digest'] = _digest({'dependencies': deps, 'source_sha256': row['source_sha256'],
                                            'pose_sha256': row['pose_sha256']})
        rows.append(row)
    _json(data/'recordings_scan.json', rows)
    _json(data/'source_exclusions.json', exclusions)
    print(f'Scan: {len(rows)} usable recordings, {sum(bool(r["coaches"]) for r in rows)} qualified, '
          f'{sum(len(r["coaches"]) for r in rows)} coaches', flush=True)
    audio = _jobs(_audio_job, rows, workers, 'audio', data)
    failures = {rid: a for rid, a in audio.items() if a['status'] != 'ok'}
    if failures:
        raise RuntimeError(f'Audio preparation failed for {len(failures)} recordings; see audio_results.json')
    fingerprints = {}
    for row in rows:
        a = audio[row['recording_id']]
        row.update({k: a[k] for k in ('audio_path', 'wav_path', 'fingerprint_path')})
        fp = json.loads(Path(a['fingerprint_path']).read_text())['fingerprint']
        fingerprints[row['recording_id']] = np.asarray(fp, np.int64).astype(np.uint32)
    identities = canonicalize(rows, old_songs, fingerprints)
    _json(data/'canonical_map.json', identities)
    split = {'schema_version': 1, 'salt': SPLIT_SALT, 'test': identities['test'], 'val': identities['val'],
             'recordings': {rid: {'canonical_id': v['canonical_id'], 'partition': v['partition']}
                            for rid, v in identities['records'].items()}}
    _json(data/'split_manifest.json', split)
    for row in rows:
        row.update(identities['records'][row['recording_id']])
        row['dependency_digest'] = _digest({'preparation': row['dependency_digest'], 'canonical_id': row['canonical_id'],
                                            'partition': row['partition'], 'audio': audio[row['recording_id']]['artifacts']})
    print(f'Identities: {len(identities["components"])} canonical families, '
          f'{identities["n_verified_new_families"]} eligible new, 40 validation + 40 locked test', flush=True)
    features = _jobs(_feature_job, [r for r in rows if r['coaches']], workers, 'features', data)
    failed = {rid: f for rid, f in features.items() if f['status'] != 'ok'}
    if failed:
        raise RuntimeError(f'Feature preparation failed for {len(failed)} recordings; see features_results.json')
    artifacts, motion_index = {}, []
    for row in rows:
        artifacts.update(audio[row['recording_id']]['artifacts'])
        result = features.get(row['recording_id'])
        if result:
            row.update({k: result[k] for k in ('visual_path', 'coaches', 'visual_frames')})
            artifacts.update(result['artifacts'])
            motion_index.extend(row['coaches'])
        else:
            row['visual_path'] = None
        row.pop('dependency_digest')
        row.pop('data_dir')
    _json(data/'motion_index.json', motion_index)
    for p in (data/'canonical_map.json', data/'split_manifest.json', data/'motion_index.json'):
        artifacts[str(p)] = sha256(p)
    counts = Counter(r['partition'] for r in rows)
    manifest = {'schema_version': 1, 'data_root': str(data), 'recordings': sorted(rows, key=lambda r: r['recording_id']),
                'source_exclusions': exclusions, 'counts': {'recordings': len(rows), 'coaches': len(motion_index),
                    'partitions': dict(counts), 'canonical_families': len(identities['components']),
                    'verified_new_families': identities['n_verified_new_families'], 'test_families': len(identities['test']),
                    'val_families': len(identities['val']), 'missing': 0, 'nonfinite': 0},
                'canonical_map_path': str(data/'canonical_map.json'), 'split_path': str(data/'split_manifest.json'),
                'motion_index_path': str(data/'motion_index.json'), 'artifact_hashes': artifacts,
                'dependency_hashes': deps, 'preparation_elapsed_s': time.monotonic()-started,
                'normalization': {'scene_y': 'columns 3/4 recording mean/std, floor 1/255; fit before crop',
                                  'audio_and_motion': 'unfitted; caller must fit old training songs only'},
                'visual_sampling': {'fps': FPS, 'width_max': 640, 'offset_source': 'first detected coach pose timestamp'}}
    _json(data/'manifest.json', manifest)
    _json(data/'manifest.sha256.json', {'sha256': sha256(data/'manifest.json')})
    print(f'Data frozen: {manifest["counts"]}; elapsed={manifest["preparation_elapsed_s"]:.0f}s', flush=True)
    return manifest


def prepare_data(root: Path, workers: int = 4) -> dict:
    """Serialize preparation/resume writers, waiting for any existing writer."""
    import fcntl
    data = _data_dir(Path(root))
    data.mkdir(parents=True, exist_ok=True)
    with (data/'preparation.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _prepare_data(root, workers)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_manifest(root: Path) -> dict:
    data = _data_dir(Path(root))
    path = data/'manifest.json'
    result = json.loads(path.read_text())
    if result.get('schema_version') != 1:
        raise ValueError('Unsupported Stage 11 manifest schema')
    sidecar = data/'manifest.sha256.json'
    if not sidecar.is_file():
        raise ValueError('Frozen data manifest checksum sidecar is missing')
    if sha256(path) != json.loads(sidecar.read_text())['sha256']:
        raise ValueError('Frozen data manifest hash mismatch')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/lightsem-data/experiments/lightsem/stage11_20260908'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    prepare_data(args.root, args.workers)


if __name__ == '__main__':
    main()
