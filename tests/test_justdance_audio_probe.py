import json

from scripts.data import extract_justdance as extractor


def test_reprobe_audio_refreshes_stale_existing_level(tmp_path, monkeypatch):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps([{
        'yt_id': 'abcdefghijk', 'file': 'old.webm', 'mean_db': -91.0, 'max_db': -91.0,
    }]))
    monkeypatch.setattr(extractor, 'MANIFEST', manifest)
    monkeypatch.setattr(extractor, 'list_captures', lambda: [('abcdefghijk', 'new.webm')])
    monkeypatch.setattr(extractor, 'probe_capture', lambda fn: {
        'yt_id': 'abcdefghijk', 'file': fn, 'mean_db': -26.9, 'max_db': -1.5,
        'duration_s': 10.0,
    })

    rows = extractor.refresh_manifest(workers=1, reprobe_audio=True)
    assert rows[0]['file'] == 'new.webm'
    assert rows[0]['mean_db'] == -26.9
    assert rows[0]['max_db'] == -1.5
