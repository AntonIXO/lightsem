"""Append-only human feedback for the new blind campaign materials."""
from pathlib import Path
import json
import subprocess
import time
from .stage11_common import digest


def answer(question,input_fn=input):
    while True:
        try:value=input_fn(question+' [A/B/= tie/u cannot judge/r replay/q quit]: ').strip().lower()
        except (EOFError,KeyboardInterrupt):return 'q'
        if value in {'a','b','=','u','r','q'}:return value


def rate(root,rater='anton',*,input_fn=input,player=None):
    directory=Path(root)/'results/renders';path=directory/'pairs.json'
    batch=json.loads(path.read_text());out=directory/'ratings.jsonl'
    if not rater.strip():raise ValueError('A rater identity is required')
    previous=[json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    done={r['id'] for r in previous if r['rater']==rater}
    player=player or (lambda p:subprocess.run(['mpv','--really-quiet','--fs',str(p)],check=True))
    for pair in batch['pairs']:
        if pair['id'] in done:continue
        while True:
            for side in ('A','B'):
                clip=directory/pair['videos'][side]
                if digest(clip)!=pair['video_sha256'][side]:raise RuntimeError('Rating video changed: '+str(clip))
                print(f'{pair["id"]}: playing {side}',flush=True);player(clip)
            music=answer('1) Which fits the music better?',input_fn)
            if music=='q':return False
            if music=='r':continue
            appearance=answer('2) Which looks more deliberately designed?',input_fn)
            if appearance=='q':return False
            if appearance=='r':continue
            break
        record={'id':pair['id'],'rater':rater,'fits_music':music,'engineer':appearance,
                't':time.time(),'schema':'stage11_rating_v1','pairs_sha256':digest(path),
                'unknown_token':'u','tie_token':'='}
        with out.open('a') as f:
            f.write(json.dumps(record)+'\n');f.flush()
        done.add(pair['id'])
    return True
