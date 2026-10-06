"""Audit and publish direct D7/repaired-C plots without changing saved movies.

The original all-three, full-song scales are retained as historical diagnostics.
Direct comparisons share a scale between D7 and repaired C in the stated window;
all extrema are included. Neither motion nor numerical traces are filtered.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.rendering.native_repair_traces import ROWS, body_traces, song_traces, sha

METHODS = ('D7_reference', 'C3', 'C4')
SONGS = ('shpongle', 'cure_for_me')
BODY_ROWS = ROWS[:6]
LIGHT_ROWS = ROWS[6:]
COLORS = ('#ffa950', '#4bd6e6')


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def selected_pair(data, name, interval):
    time, old = data['values']['D7_reference'][name]
    other_time, repaired = data['values']['C4'][name]
    if not np.array_equal(time, other_time):
        raise ValueError('Direct comparison requires matching measurement timestamps')
    mask = (time >= interval[0]) & (time <= interval[1])
    if not mask.any():
        raise ValueError('Empty comparison window')
    return time[mask], old[mask], repaired[mask]


def shared_pair_scale(old, repaired, name):
    """Use both roles and every sample; never inherit broken-C extrema."""
    low = float(min(old.min(), repaired.min()))
    high = float(max(old.max(), repaired.max()))
    if name == 'brightness':
        return [0., 1.]
    if name == 'root_vertical_jerk':
        high = max(abs(low), abs(high), 1e-8)
        return [-high * 1.04, high * 1.04]
    if name == 'root_y':
        margin = max((high - low) * .04, .001)
        return [low - margin, high + margin]
    return [0., max(high, 1e-8) * 1.04]


def statistics(old, repaired):
    delta = repaired - old
    correlation = float(np.corrcoef(old, repaired)[0, 1]) if min(old.std(), repaired.std()) > 0 else None
    return {
        'exactly_equal': bool(np.array_equal(old, repaired)),
        'old_mean': float(old.mean()), 'repaired_mean': float(repaired.mean()),
        'mean_delta_percent': float(100 * (repaired.mean() / old.mean() - 1)) if abs(old.mean()) > 1e-12 else None,
        'old_p95': float(np.quantile(old, .95)), 'repaired_p95': float(np.quantile(repaired, .95)),
        'pearson_r': correlation, 'difference_RMSE': float(np.sqrt(np.mean(delta ** 2))),
        'difference_max_absolute': float(np.max(np.abs(delta))),
        'old_max': float(old.max()), 'repaired_max': float(repaired.max()),
    }


def authenticate_media(root, manifest):
    """Authenticate original role-to-artifact binding, including raw trace inputs."""
    pins = {}
    if len(manifest['clips']) != 6 or manifest.get('baseline_promoted') is not False:
        raise ValueError('Require the six unpromoted verified original movies')
    for row in manifest['clips']:
        method, song = row['method'], row['song']
        directory = root / 'review_3d_v1' / method / song
        audit = read(directory / 'AUDIT.json')
        result = read(directory / 'RENDER.json')
        if audit['inputs'] != result['inputs'] or audit['render_sha256'] != sha(directory / 'RENDER.json'):
            raise ValueError('Changed media input receipt')
        movie = directory / 'preview.mp4'
        movie_sha = sha(movie)
        if movie_sha != row['sha256'] or movie_sha != audit['movie_sha256'] or movie_sha != result['sha256']:
            raise ValueError('Changed original movie')
        if any(audit.get(k) is not True for k in ('full_software_AV_decode', 'pts_uniform', 'audio_matches_prepared_reference')):
            raise ValueError('Original full AV audit did not pass')
        if audit['decoded_frames'] != row['video_frames'] or audit['pcm_sha256'] != result['inputs']['audio']['decoded_pcm_sha256']:
            raise ValueError('Changed full-song timing or audio')
        for path, expected in audit['inputs']['trace_inputs'].items():
            if sha(path) != expected:
                raise ValueError('Changed trace input: ' + path)
            pins[path] = expected
        for path in (movie, directory / 'AUDIT.json', directory / 'RENDER.json', directory / 'smoke/TRACE.npz'):
            pins[str(path)] = sha(path)
    return pins


def audit_song(root, song):
    data = song_traces(str(root), song)
    provenance, bodies, checks = {}, {}, {}
    for method in METHODS:
        path = root / 'bodies' / method / song / 'body.npz'
        with np.load(path, allow_pickle=False) as source:
            bodies[method] = {key: source[key].astype(float) for key in ('world', 'native', 'time')}
        metadata = read(path.with_suffix('.json'))['metadata']
        provenance[method] = {'body_sha256': sha(path), 'path': str(path),
                              'checkpoint': metadata.get('checkpoint', metadata.get('generator')),
                              'noise_sha256': metadata.get('noise_sha256'), 'draw_seed': metadata.get('draw_seed'),
                              'motion_origin': metadata.get('motion_origin')}
        independent = body_traces(bodies[method]['world'], bodies[method]['time'])
        exported = root / 'review_3d_v1' / method / song / 'smoke/TRACE.npz'
        with np.load(exported, allow_pickle=False) as raw:
            for name, (time, values) in data['values'][method].items():
                if not np.array_equal(time, raw['current_' + name + '_time']) or not np.array_equal(values, raw['current_' + name + '_raw']):
                    raise ValueError('Plotted trace differs from original movie data: ' + method + '/' + name)
            for name, (time, values) in independent.items():
                if not np.array_equal(time, raw['current_' + name + '_time']) or not np.array_equal(values, raw['current_' + name + '_raw']):
                    raise ValueError('Original movie trace not derived from its own body: ' + method + '/' + name)
        checks[method] = {'own_body_recomputed': True, 'all_nine_saved_movie_traces_exactly_match': True}
    old, repaired = bodies['D7_reference'], bodies['C4']
    if not np.array_equal(old['time'], repaired['time']):
        raise ValueError('Native body timestamps differ')
    relative_delta = (repaired['world'] - repaired['world'][:, :1]) - (old['world'] - old['world'][:, :1])
    identity = {'world_exactly_equal': bool(np.array_equal(old['world'], repaired['world'])),
                'native139_exactly_equal': bool(np.array_equal(old['native'], repaired['native'])),
                'pelvis_relative_joint_vector_RMS_difference_m': float(np.sqrt(np.mean(np.sum(relative_delta ** 2, axis=-1)))),
                'world_max_absolute_difference_m': float(np.max(np.abs(old['world'] - repaired['world'])))}
    windows = {}
    for scope, interval in (('full_song', data['interval_s']), ('90_150', [90., 150.])):
        measurements = {}
        for name, label in ROWS:
            time, a, b = selected_pair(data, name, interval)
            stats = statistics(a, b)
            stats.update(label=label, samples=len(time), pair_scale=shared_pair_scale(a, b, name),
                         original_three_role_full_song_scale=data['scales'][name])
            if name not in ('brightness', 'root_y', 'root_vertical_jerk'):
                stats['pair_max_fraction_of_original_axis'] = max(stats['old_max'], stats['repaired_max']) / data['scales'][name][1]
            measurements[name] = stats
        windows[scope] = {'interval_s': interval, 'measurements': measurements}
    return data, {'song': song, 'provenance': provenance, 'checks': checks, 'identity': identity, 'windows': windows}


def plot_graph(data, song, rows, interval, path, *, difference=False):
    """Raw values; each pixel gets min/max so subpixel peak samples survive."""
    from scripts.rendering.native_repair_traces import pixel_envelope
    figure, axes = plt.subplots(len(rows), 1, figsize=(14, 1.32 * len(rows) + .75), sharex=True, layout='constrained')
    figure.patch.set_facecolor('#10151d')
    title = 'Shpongle' if song == 'shpongle' else 'AURORA — Cure For Me'
    kind = 'Repaired C − Old D7 | signed raw differences' if difference else 'Orange: Old D7   ·   Cyan: Repaired native C'
    figure.suptitle(f'{title} | {interval[0]:g}–{interval[1]:g} s | {kind}', color='#e3ebf4', fontsize=13)
    evidence = {}
    for axis, (name, label) in zip(np.atleast_1d(axes), rows):
        axis.set_facecolor('#10151d')
        time, old, repaired = selected_pair(data, name, interval)
        series = (repaired - old,) if difference else (old, repaired)
        scale = shared_pair_scale(old, repaired, name)
        if difference:
            high = max(float(np.max(np.abs(series[0]))), 1e-8) * 1.04
            scale = [-high, high]
            axis.axhline(0, color='#718295', linewidth=.7)
        for values, color in zip(series, (COLORS[1],) if difference else COLORS):
            minimum, maximum = pixel_envelope(time, values, left=interval[0], right=interval[1], pixels=1700)
            centers = interval[0] + (np.arange(1700) + .5) / 1700 * (interval[1] - interval[0])
            valid = np.isfinite(minimum)
            axis.vlines(centers[valid], minimum[valid], maximum[valid], color=color, linewidth=.7, alpha=.9)
            # Native samples are connected too; extrema above always retain raw peaks.
            axis.plot(time, values, color=color, linewidth=.55, alpha=.75)
            if values.min() < scale[0] or values.max() > scale[1]:
                raise ValueError('Graph would clip a raw sample')
        axis.set_ylim(*scale)
        axis.set_xlim(*interval)
        axis.set_ylabel(('Δ ' if difference else '') + label.replace('Raw ', ''), color='#e3ebf4', fontsize=9)
        axis.tick_params(colors='#cbd7e6', labelsize=8)
        axis.grid(color='#344254', linewidth=.4, alpha=.6)
        for spine in axis.spines.values():
            spine.set_color('#445269')
        evidence[name] = {'scale': scale, 'raw_samples_per_role': len(time), 'clipped_samples': 0}
    np.atleast_1d(axes)[-1].set_xlabel('Original song time (seconds) | raw native30Hz body; raw40Hz emitted lights | no smoothing', color='#cbd7e6', fontsize=10)
    figure.savefig(path, dpi=120, facecolor=figure.get_facecolor())
    plt.close(figure)
    return evidence


def graph_item(path, song, scope, kind, rows, interval, evidence):
    return {'song': song, 'scope': scope, 'kind': kind, 'interval_s': interval,
            'path': str(path), 'sha256': sha(path), 'measurement': evidence,
            'src': 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode()}


def publish(root, plan_sha, revision):
    root = root.resolve()
    if any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in revision) or not revision:
        raise ValueError('Safe nonempty revision required')
    if sha(root / 'PLAN.json') != plan_sha:
        raise ValueError('PLAN identity differs')
    if shutil.disk_usage(root).free < 20 * 2 ** 30 + 16 * 2 ** 20:
        raise ValueError('20GiB free-space reserve reached')
    review = root / 'review_3d_v1'
    parent = read(review / 'MANIFEST.json')
    archive_parent = review / 'revisions' / parent['revision']
    if parent != read(archive_parent / 'MANIFEST.json') or sha(review / 'index.html') != sha(archive_parent / 'index.html'):
        raise ValueError('Current page is not an authenticated archived revision')
    pins = authenticate_media(root, parent)
    dest = review / 'revisions' / revision
    dest.mkdir(exist_ok=False)
    graphs, audits = [], []
    for song in SONGS:
        data, audit = audit_song(root, song)
        audits.append(audit)
        for scope, interval in (('full_song', data['interval_s']), ('90_150', [90., 150.])):
            for kind, rows, difference in (('body', BODY_ROWS, False), ('lights', LIGHT_ROWS, False), ('difference', BODY_ROWS[:3], True)):
                image = dest / f'{song}_{scope}_{kind}.png'
                measured = plot_graph(data, song, rows, interval, image, difference=difference)
                graphs.append(graph_item(image, song, scope, kind, rows, interval, measured))
    report = {'scope': 'Descriptive audit of the saved displayed bodies and their original traces; no new training or artistic verdict.',
              'inputs': pins, 'plan_sha256': plan_sha, 'parent_revision': parent['revision'],
              'publisher_sha256': sha(Path(__file__)), 'trace_code_sha256': sha(Path(__file__).with_name('native_repair_traces.py')),
              'cause': 'Full-song maxima of broken C set every axis, including 90–150s details; only17vertical pixels per row in original graph.',
              'fix': 'Direct oldD7/repairedC overlay; common per-pair/per-window linear scales include all samples. Separate signed regional differences. Original movies/three-role charts retained.',
              'songs': audits, 'posthoc_motion_processing': False, 'baseline_promoted': False,
              'interpretation': 'Outputs differ, with substantially correlated movement. Stable dynamics versus broken C do not demonstrate superior musical choreography versus old D7. Same audio waveform/brightness law are intentional shared controls. Historical oldD7 and new C use different noise arrays; this is descriptive, not a paired training estimate.'}
    save(dest / 'SIMILARITY_AUDIT.json', report)
    # Start with the archived current page; its six sealed media/evidence bindings
    # remain untouched. Presentation extension has a separate receipt/version.
    html = (archive_parent / 'index.html').read_text()
    section = Path(__file__).with_name('native_repair_pair_section.html').read_text()
    payload = {'graphs': graphs, 'audit': report}
    section = section.replace('__PAIR_DATA_JSON__', json.dumps(payload, allow_nan=False).replace('<', '\\u003c'))
    if '<details id="diagnostic">' not in html:
        raise ValueError('Expected original review template')
    html = html.replace('<details id="diagnostic">', section + '\n<details id="diagnostic">', 1)
    html = html.replace('<summary>90–150 s diagnostic view — same full-song videos</summary>', '<summary>Archived three-role plots — original axes dominated by broken C</summary>')
    html = html.replace('90–150 s detail</button>', 'Direct D7 / repaired C plots</button>')
    html = html.replace('Reserved for the newly retrained joint native planner, seed 0.', 'Repaired joint native planner, seed 0; raw output after35,500 training updates.')
    html = html.replace('An unfinished repair remains an empty, labeled pane until its own output is ready.', 'All three movies are verified; broken C remains a labeled failure diagnostic.')
    manifest = copy.deepcopy(parent)
    manifest.update(revision=revision, presentation_parent_revision=parent['revision'],
                    direct_comparison={'audit_path': str(dest / 'SIMILARITY_AUDIT.json'), 'audit_sha256': sha(dest / 'SIMILARITY_AUDIT.json'),
                                       'graphs': [{k: v for k, v in graph.items() if k != 'src'} for graph in graphs],
                                       'section_sha256': sha(Path(__file__).with_name('native_repair_pair_section.html'))})
    save(dest / 'MANIFEST.json', manifest)
    with (dest / 'index.html').open('x') as stream:
        stream.write(html)
    save(dest / 'PUBLICATION.json', {'index_sha256': sha(dest / 'index.html'), 'manifest_sha256': sha(dest / 'MANIFEST.json'),
                                     'publisher_sha256': sha(Path(__file__)), 'unchanged_movie_hashes': [c['sha256'] for c in parent['clips']],
                                     'new_graphs': len(graphs), 'baseline_promoted': False})
    for name in ('MANIFEST.json', 'index.html'):
        temporary = review / (name + '.publishing')
        with temporary.open('xb') as stream:
            stream.write((dest / name).read_bytes())
        os.replace(temporary, review / name)
    return {'revision': revision, 'graphs': len(graphs), 'audit': str(dest / 'SIMILARITY_AUDIT.json'),
            'index': str(review / 'index.html'), 'index_sha256': sha(review / 'index.html')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--revision', required=True)
    args = parser.parse_args()
    print(json.dumps(publish(args.root, args.plan_sha256, args.revision), indent=2))


if __name__ == '__main__':
    main()
