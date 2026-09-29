"""Recovery tests use disposable copies only; production restore is never called."""
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import restore_notes_mcp as r


class NotesRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.snapshot = json.loads(gzip.decompress((r.BASE / 'BEFORE_SNAPSHOT_V1.json.gz').read_bytes()))
        self.rows = []
        for eid in r.IDS:
            damaged = (r.BASE / ('DROPPED_BODY_' + eid + '.md')).read_bytes()
            candidate = (r.BASE / ('RESTORE_CANDIDATE_' + eid + '.md')).read_bytes()
            path = self.root / (eid + '.md')
            path.write_bytes(damaged)
            path.chmod(0o640)
            inserted = r.validate_insertion(damaged, candidate, self.snapshot['target_records'][eid]['body'])
            self.rows.append({'id': eid, 'path': path, 'damaged': damaged, 'candidate': candidate, 'inserted_bytes': inserted})

    def test_exact_recovery_of_both_real_document_copies(self):
        result = r.apply_preflighted(self.rows)
        self.assertTrue(result['ok'])
        for row in self.rows:
            self.assertEqual(row['path'].read_bytes(), row['candidate'])
            self.assertEqual(row['path'].stat().st_mode & 0o777, 0o640)

    def test_repeated_recovery_is_no_write_including_mtime(self):
        r.apply_preflighted(self.rows)
        before = [x['path'].stat().st_mtime_ns for x in self.rows]
        result = r.apply_preflighted(self.rows)
        self.assertTrue(all(x['action'] == 'already_restored_no_write' for x in result['records']))
        self.assertEqual(before, [x['path'].stat().st_mtime_ns for x in self.rows])

    def test_second_target_drift_prevents_first_write(self):
        self.rows[1]['path'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'drift'):
            r.apply_preflighted(self.rows)
        self.assertEqual(self.rows[0]['path'].read_bytes(), self.rows[0]['damaged'])

    def test_changed_metadata_and_interpretation_are_rejected(self):
        row = self.rows[0]
        body = self.snapshot['target_records'][row['id']]['body']
        for old, new in ((b'closed: true', b'closed: false'), (b'## Interpretation\n', b'## Interpretation\nCHANGED\n')):
            with self.assertRaises(ValueError):
                r.validate_insertion(row['damaged'], row['candidate'].replace(old, new), body)

    def test_ambiguous_notes_and_interpretation_boundaries_rejected(self):
        row = self.rows[0]
        for heading in (b'## Notes\n', b'## Interpretation\n'):
            with self.assertRaises(ValueError):
                r.validate_insertion(row['damaged'], row['candidate'] + b'\n' + heading,
                                     self.snapshot['target_records'][row['id']]['body'])

    def test_snapshot_disagreement_and_footer_change_rejected(self):
        row = self.rows[0]
        with self.assertRaises(ValueError):
            r.validate_insertion(row['damaged'], row['candidate'], 'not the original')
        with self.assertRaises(ValueError):
            r.validate_insertion(row['damaged'], row['candidate'] + b'changed footer',
                                 self.snapshot['target_records'][row['id']]['body'])

    def test_symlinks_and_hardlinks_rejected(self):
        row = self.rows[0]
        path = self.root / 'alias'
        path.symlink_to(row['path'])
        with self.assertRaisesRegex(ValueError, 'symlink'):
            r.read_regular(path)
        path.unlink()
        path.hardlink_to(row['path'])
        with self.assertRaisesRegex(ValueError, 'unaliased'):
            r.read_regular(path)

    def test_failed_replace_preserves_live_and_temporary_bytes(self):
        row = self.rows[0]
        with mock.patch.object(r.os, 'replace', side_effect=OSError('injected failure')):
            result = r.apply_preflighted(self.rows)
        self.assertFalse(result['ok'])
        self.assertEqual(row['path'].read_bytes(), row['damaged'])
        self.assertEqual(row['path'].with_name(row['path'].name + '.notes-recovery-v1.tmp').read_bytes(), row['candidate'])
        self.assertEqual(self.rows[1]['path'].read_bytes(), self.rows[1]['damaged'])

    def test_late_target_drift_is_not_overwritten(self):
        row = self.rows[0]
        native_read = r.read_regular
        calls = []
        def reader(path, *args):
            if path == row['path']:
                calls.append(1)
                if len(calls) == 2:
                    path.write_bytes(b'new writer content')
            return native_read(path, *args)
        with mock.patch.object(r, 'read_regular', side_effect=reader):
            with self.assertRaisesRegex(ValueError, 'drift'):
                r.atomic_replace_checked(row['path'], row['damaged'], row['candidate'])
        self.assertEqual(row['path'].read_bytes(), b'new writer content')

    def test_mcp_protocol_only_exposes_recovery_tool(self):
        messages = [
            {'id': 1, 'method': 'initialize'}, {'method': 'notifications/initialized'},
            {'id': 2, 'method': 'tools/list'},
            {'id': 3, 'method': 'tools/call', 'params': {'name': 'close_experiment'}},
            {'id': 4, 'method': 'tools/call', 'params': {'name': r.TOOL, 'arguments': {'manifest_sha256': r.MANIFEST_HASH}}},
        ]
        sink = io.StringIO()
        handler = mock.Mock(return_value={'ok': True})
        r.serve(io.StringIO(''.join(json.dumps(x) + '\n' for x in messages)), sink, handler)
        replies = [json.loads(x) for x in sink.getvalue().splitlines()]
        self.assertEqual(replies[0]['result']['serverInfo'], r.IDENTITY)
        self.assertEqual([x['name'] for x in replies[1]['result']['tools']], [r.TOOL])
        self.assertIn('error', replies[2])
        handler.assert_called_once_with({'manifest_sha256': r.MANIFEST_HASH})

    def test_production_dispatch_rejects_nonexact_arguments_without_io(self):
        with mock.patch.object(r, 'read_regular', side_effect=AssertionError('must not read')):
            for arguments in ({}, {'manifest_sha256': 'wrong'}, {'manifest_sha256': r.MANIFEST_HASH, 'path': '/tmp/other'}):
                with self.assertRaisesRegex(ValueError, 'argument'):
                    r.restore(arguments)


if __name__ == '__main__':
    unittest.main()
