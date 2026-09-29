#!/usr/bin/python3 -B
"""One-time, two-record MCP repair. Does not import the scientific vault writer."""
import difflib
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

BASE = Path(__file__).resolve().parent.parent
VAULT = Path('/root/lightsem-vault')
MANIFEST_HASH = '87431c538e5713ba735b022434e895a1dca33854eee96721fe729aaefe0d9d13'
SNAPSHOT_HASH = 'bb0b6d6850eb80329a22ee2e0ef5f28629209742c9746128bb0ccb2159fb3f75'
IDS = ('exp-dance-physical-objective-20260911', 'exp-dance-optics-full-20260911')
IDENTITY = {'name': 'lightsem-two-record-notes-recovery', 'version': '1'}
TOOL = 'restore_experiment_notes_snapshot'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_regular(path, cap=65536):
    path = Path(path)
    require(path.is_absolute(), 'absolute path required')
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'symlink rejected')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, 'regular unaliased file required')
        require(before.st_size <= cap, 'input size cap')
        chunks = []
        left = cap + 1
        while left:
            chunk = os.read(fd, min(left, 65536))
            if not chunk:
                break
            chunks.append(chunk)
            left -= len(chunk)
        raw = b''.join(chunks)
        after = os.fstat(fd)
        require(len(raw) == before.st_size <= cap, 'size drift')
        require((before.st_ino, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_ino, after.st_mtime_ns, after.st_ctime_ns), 'input changed during read')
        return raw
    finally:
        os.close(fd)


def split_document(raw):
    require(raw.startswith(b'---\n'), 'frontmatter opening')
    end = raw.index(b'\n---\n', 4) + 5
    return raw[:end], raw[end:]


def without_interpretation(body):
    text = body.decode('utf-8')
    require(len(re.findall(r'^## Interpretation$', text, re.M)) == 1, 'unique Interpretation required')
    return re.sub(r'^## Interpretation\n.*?(?=^## |\Z)', '', text,
                  flags=re.M | re.S).strip()


def validate_insertion(damaged, candidate, original_body):
    old_meta, old_body = split_document(damaged)
    new_meta, new_body = split_document(candidate)
    require(old_meta == new_meta, 'frontmatter changed')
    require(re.search(br'^closed: true$', old_meta, re.M), 'closed record required')
    for body in (old_body, new_body):
        require(len(re.findall(br'^## Notes$', body, re.M)) == 1, 'unique Notes required')
        require(body.index(b'## Interpretation\n') < body.index(b'## Notes\n'), 'Interpretation must precede Notes')
    require(without_interpretation(new_body) == original_body.strip(), 'original body not restored')
    old_footer = re.search(br'\n(#[^\n]+)\n?$', old_body)
    new_footer = re.search(br'\n(#[^\n]+)\n?$', new_body)
    require(old_footer and new_footer and old_footer.group(0) == new_footer.group(0), 'tag footer changed')
    # Compare the content separately: a heading and the tag footer share '#',
    # which can shift SequenceMatcher's otherwise equivalent insertion boundary.
    old_content, new_content = old_body[:old_footer.start()], new_body[:new_footer.start()]
    changes = [x for x in difflib.SequenceMatcher(None, old_content, new_content,
                                                 autojunk=False).get_opcodes() if x[0] != 'equal']
    require(len(changes) == 1 and changes[0][0] == 'insert', 'only one insertion allowed')
    _, a, b, c, d = changes[0]
    require(a >= old_body.index(b'## Notes\n') + len(b'## Notes\n'), 'insertion outside Notes')
    require(old_content[:a] == new_content[:c] and old_content[b:] == new_content[d:], 'prefix/suffix changed')
    require(a <= old_footer.start(), 'insertion after footer')
    return len(candidate) - len(damaged)


def atomic_replace_checked(path, expected, candidate):
    """Preserve failed temporary files; never overwrite a previous attempt."""
    require(read_regular(path) == expected, 'live file drift before staging')
    info = path.stat()
    tmp = path.with_name(path.name + '.notes-recovery-v1.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, stat.S_IMODE(info.st_mode))
    try:
        os.fchmod(fd, stat.S_IMODE(info.st_mode))
        os.fchown(fd, info.st_uid, info.st_gid)
        view = memoryview(candidate)
        while view:
            n = os.write(fd, view)
            require(n > 0, 'short write')
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)
    require(read_regular(tmp) == candidate, 'staged bytes differ')
    require(read_regular(path) == expected, 'live file drift before replacement')
    os.replace(tmp, path)
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    require(read_regular(path) == candidate, 'post-write readback differs')


def apply_preflighted(rows):
    # Preflight ALL targets before any write. A later drift still stops immediately.
    for row in rows:
        live = read_regular(row['path'])
        require(live in (row['damaged'], row['candidate']), 'live hash drift')
    results = []
    try:
        for row in rows:
            live = read_regular(row['path'])
            if live == row['candidate']:
                action = 'already_restored_no_write'
            else:
                require(live == row['damaged'], 'live hash drift')
                atomic_replace_checked(row['path'], row['damaged'], row['candidate'])
                action = 'restored'
            results.append({'experiment_id': row['id'], 'path': str(row['path']),
                            'action': action, 'before_sha256': digest(live),
                            'after_sha256': digest(read_regular(row['path'])),
                            'inserted_bytes': row['inserted_bytes'],
                            'whole_frontmatter_and_interpretation_preserved': True})
        return {'ok': True, 'records': results, 'scientific_state_recomputed': False}
    except Exception as exc:
        return {'ok': False, 'records': results, 'error': str(exc),
                'partial_completion_possible': True, 'automatic_retry': False}


def restore(arguments):
    require(arguments == {'manifest_sha256': MANIFEST_HASH}, 'exact manifest argument required')
    require(os.environ.get('VIBESCIENCE_VAULT', '').rstrip('/') == str(VAULT), 'wrong vault')
    # A parent supervisor temporarily suspends the installed LightSem writer.
    require(os.environ.get('LIGHTSEM_RECOVERY_WRITERS_QUIESCED') == '1', 'writer quiescence required')
    raw = read_regular(BASE / 'RESTORATION_CANDIDATES_V1.json')
    require(digest(raw) == MANIFEST_HASH, 'manifest drift')
    manifest = json.loads(raw)
    raw = read_regular(BASE / 'BEFORE_SNAPSHOT_V1.json.gz')
    require(digest(raw) == SNAPSHOT_HASH, 'snapshot drift')
    snapshot = json.loads(gzip.decompress(raw))
    require(tuple(r['experiment_id'] for r in manifest['candidates']) == IDS, 'target allowlist')
    rows = []
    for item in manifest['candidates']:
        eid = item['experiment_id']
        path = VAULT / 'experiments' / (eid + '.md')
        require(item['live_path'] == str(path), 'target path drift')
        candidate_path = BASE / ('RESTORE_CANDIDATE_' + eid + '.md')
        require(item['candidate']['path'] == str(candidate_path), 'candidate path drift')
        candidate = read_regular(candidate_path)
        damaged = read_regular(BASE / ('DROPPED_BODY_' + eid + '.md'))
        require(digest(candidate) == item['candidate']['sha256'] and
                len(candidate) == item['candidate']['bytes'], 'candidate hash/size drift')
        require(digest(damaged) == item['current_dropped_body_sha256'], 'damaged snapshot drift')
        original = snapshot['target_records'][eid]
        require(original['path'] == str(path) and original['metadata']['closed'] is True, 'snapshot target drift')
        inserted = validate_insertion(damaged, candidate, original['body'])
        rows.append({'id': eid, 'path': path, 'damaged': damaged,
                     'candidate': candidate, 'inserted_bytes': inserted})
    directory_fd = os.open(VAULT / 'experiments', os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(directory_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return apply_preflighted(rows)
    finally:
        os.close(directory_fd)


def serve(source=sys.stdin, sink=sys.stdout, handler=restore):
    for line in source:
        require(len(line) <= 16384, 'request too large')
        request = json.loads(line)
        if 'id' not in request:
            continue
        method = request.get('method')
        try:
            if method == 'initialize':
                result = {'protocolVersion': '2024-11-05', 'capabilities': {'tools': {}}, 'serverInfo': IDENTITY}
            elif method == 'tools/list':
                result = {'tools': [{'name': TOOL, 'description': __doc__, 'inputSchema': {
                    'type': 'object', 'properties': {'manifest_sha256': {'type': 'string', 'const': MANIFEST_HASH}},
                    'required': ['manifest_sha256'], 'additionalProperties': False}}]}
            elif method == 'tools/call':
                params = request.get('params', {})
                require(params.get('name') == TOOL, 'unknown tool')
                data = handler(params.get('arguments', {}))
                result = {'content': [{'type': 'text', 'text': json.dumps(data)}], 'isError': not data['ok']}
            else:
                raise ValueError('unknown method')
            response = {'jsonrpc': '2.0', 'id': request['id'], 'result': result}
        except Exception as exc:
            response = {'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32602, 'message': str(exc)}}
        sink.write(json.dumps(response) + '\n')
        sink.flush()


if __name__ == '__main__':
    serve()
