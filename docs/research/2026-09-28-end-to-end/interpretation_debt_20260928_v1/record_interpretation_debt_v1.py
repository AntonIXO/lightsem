from log_to_vault import Client
from pathlib import Path
import gzip, hashlib, json
D=Path('/root/lightsem/docs/research/2026-09-28-end-to-end/interpretation_debt_20260928_v1/NATIVE_INTERPRETATION_DEFINITION_V1.json')
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checked(ref):
    p=Path(ref['path'])
    assert not p.is_symlink() and p.is_file(),str(p)
    assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256'],str(p)
    return p
p=json.loads(D.read_text())
assert p['allow_only_exact_interpretation_writes'] is True
checked(p['source']); checked(p['content_review']); snapshot=checked(p['before_snapshot'])
before=json.loads(gzip.decompress(snapshot.read_bytes()))
assert len(p['payloads'])==20 and {r['experiment_id'] for r in p['payloads']}==set(before['target_records'])
for eid,record in before['target_records'].items():
    path=Path(record['path'])
    assert not path.is_symlink() and sha(path)==record['sha256'],eid
    assert record['metadata']['interpretation'] is None and record['metadata']['closed'] is True,eid
for path,expected in before['protected_markdown_sha256'].items():
    assert not Path(path).is_symlink() and sha(path)==expected,path
root=Path(p['receipt_directory']);root.mkdir(exist_ok=False)
c=Client();index=[]
try:
    assert c.initialize()['name']=='vibescience-lightsem'
    for i,request in enumerate(p['payloads']):
        eid=request['experiment_id']
        # All target hashes were checked together; no automatic retry or resume.
        r=c.call('interpret_experiment',request)
        out=root/f'{i:02d}_receipt.json'
        with out.open('x') as f:json.dump(r,f,indent=2)
        assert isinstance(r,dict) and r.get('experiment_id')==eid,repr(r)
        for key in ['lesson','failure_mode','transfers_to']:
            assert r.get(key)==request[key],repr(r)
        assert r.get('verdict')==before['target_records'][eid]['metadata']['verdict'],repr(r)
        index.append({'experiment_id':eid,'path':str(out),'bytes':out.stat().st_size,'sha256':sha(out)})
        print('Exact native interpretation verified',eid,flush=True)
    with (root/'INDEX.json').open('x') as f:json.dump({'count':len(index),'receipts':index},f,indent=2)
finally:c.close()
