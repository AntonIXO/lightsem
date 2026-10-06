"""Run the separately frozen, supplemental CPU musical-transfer audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

from lightsem.experiments.dynamic_dance.campaign import ROOT, METHODS, sha, write_new
from lightsem.experiments.dynamic_dance.transfer_audit import AUDIT_PARAMETERS, audit_pair


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--method', choices=tuple(METHODS))
    parser.add_argument('--song')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--print-parameters', action='store_true')
    args = parser.parse_args(argv)
    if args.print_parameters:
        print(json.dumps(AUDIT_PARAMETERS, indent=2, sort_keys=True, allow_nan=False))
        return
    if args.method is None or args.song is None or args.output is None:
        parser.error('--method, --song and a NEW --output are required')
    output = args.output.resolve()
    root = args.root.resolve()
    if output.suffix != '.json' or any(output.is_relative_to(root/part) for part in ('bodies', 'timelines', 'sources')):
        parser.error('Write a new JSON audit outside original bodies/timelines/sources')
    seal = Path(str(output)+'.sha256.json')
    if output.exists() or seal.exists():
        raise FileExistsError('Preserve previous supplemental evidence: '+str(output))
    report = audit_pair(root, args.method, args.song)
    write_new(output, report)
    write_new(seal, {'sha256': sha(output)})
    print(json.dumps({'output': str(output), 'sha256': sha(output), 'status': report['status'],
                      'scope': 'Supplemental descriptive audit; no original gate changed'}, sort_keys=True))


if __name__ == '__main__':
    main()
