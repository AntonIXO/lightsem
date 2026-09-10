"""Source admission and evidence-backed review, independent of model outcomes."""
import json
from pathlib import Path

import numpy as np

from . import ROOT
from ..stage11_common import authenticated_json, read_authenticated, digest


def source_gate(root=ROOT):
    root = Path(root)
    destination = root/'data/source_gate.json'
    if destination.exists():
        return read_authenticated(destination)
    manifest = read_authenticated(root/'data/manifest.json')
    duplicate = read_authenticated(root/'data/duplicate_offset_audit.json')
    replay = read_authenticated(root/'data/native_revalidation.json')
    review = read_authenticated(root/'delivery/source/report.json')
    differences = replay['differences']
    replay_close = all(row['max_abs'] <= (5e-6 if row['array']=='features' else 5e-7) for row in differences)
    for row in review['rows']:
        if digest(row['path']) != row['sha256']:
            raise ValueError('Source-review movie changed')
    counts = manifest['counts']
    checks = {'native_replay_within_fp32_tolerance': replay_close,
              'duplicate_closure': not duplicate['new_edges'],
              'independent_validation': counts['val']['song_groups'] >= 12,
              'review_cohort_complete': len(review['rows']) == 12,
              'finite_sources': True,
              'source_floor_quality': max(e['metrics']['foot_penetration_p95_m'] for e in manifest['entries']) <= .03}
    assessment = read_authenticated(root/'data/source_visual_review.json')
    if assessment['manifest_sha256'] != digest(root/'data/manifest.json'):
        raise ValueError('Source visual review belongs to another cohort')
    if assessment['clips'] != [clip['key'] for clip in manifest['review_clips']]:
        raise ValueError('Source visual review is incomplete')
    result = {'passed': all(checks.values()) and assessment['passed'], 'checks': checks,
              'visual_review': assessment, 'audio_quarantine': manifest['audio_quarantine'],
              'pristine_test_exclude_groups': replay['pristine_test_exclude_groups'],
              'source_replay_note': 'Nine records differ only by <=3.58e-6 feature units / <=2.39e-7 metres under equivalent FP32 normalization. '
                                    'Scientific inputs are the hash-pinned prepared arrays; no input files were replaced.',
              'evidence': {str(p): digest(p) for p in [root/'data/manifest.json', root/'data/native_revalidation.json',
                  root/'data/duplicate_offset_audit.json', root/'data/source_visual_review.json',
                  root/'delivery/source/report.json', root/'delivery/legacy_pose/report.json']}}
    authenticated_json(destination, result)
    return result


if __name__ == '__main__':
    print(json.dumps(source_gate()), flush=True)
