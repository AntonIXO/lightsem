#!/usr/bin/env python3
"""Local source-annotation workspace with allowlisted media and separate drafts.

It exposes no model predictions or other annotators' drafts. Two people should
use independent sessions. Saving a draft does not qualify or adjudicate labels.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,mimetypes,re
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote,urlsplit

ROOT=Path('/root/lightsem/runs/visual_next_20260929/annotations')
HTML=Path(__file__).with_name('visual_next_annotation.html')


def validate_draft(value,pack):
    if not isinstance(value,dict) or value.get('schema_version')!=1:
        raise ValueError('Unknown annotation schema')
    identity=value.get('annotator','')
    if not isinstance(identity,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',identity):
        raise ValueError('Use a simple annotator ID (letters/numbers, up to40characters)')
    known={r['canonical_id']:r for r in pack['records']}
    if value.get('enrollment_sha256')!=pack['enrollment_sha256']:
        raise ValueError('Wrong enrolled source cohort')
    clips=value.get('clips',{})
    if not isinstance(clips,dict) or not set(clips)<=set(known):raise ValueError('Unknown source clip')
    kinds={'content_color','hard_edit','gradual_edit','camera_only','people_ui_leakage','ambiguous'}
    for cid,row in clips.items():
        if not isinstance(row,dict):raise ValueError('Invalid clip record')
        events=row.get('events',[])
        if not isinstance(events,list) or len(events)>2000:raise ValueError('Invalid event count')
        for e in events:
            if not isinstance(e,dict):raise ValueError('Invalid event record')
            if e.get('kind') not in kinds:raise ValueError('Unknown event kind')
            if not (isinstance(e.get('start'),(float,int)) and isinstance(e.get('end'),(float,int))
                    and 0<=e['start']<=e['end']<64):raise ValueError('Event outside source window')
            if e['kind'] in {'gradual_edit','camera_only','people_ui_leakage','ambiguous'} and e['start']==e['end']:
                raise ValueError('Interval event requires a nonempty span')
        masks=row.get('masks',{})
        if not isinstance(masks,dict) or not set(masks)<=set(str(i) for i in range(10)):
            raise ValueError('Unknown mask frame')
        for m in masks.values():
            if not isinstance(m,dict) or not isinstance(m.get('polygons',[]),list):raise ValueError('Invalid mask')
            for polygon in m.get('polygons',[]):
                if not isinstance(polygon,dict):raise ValueError('Invalid polygon')
                points=polygon.get('points',[])
                if not isinstance(points,list) or polygon.get('kind') not in {'foreground','ui','unknown'} or len(points)<3:
                    raise ValueError('A polygon needs a class and3points')
                if len(points)>5000 or any(not isinstance(p,list) or len(p)!=2 or
                    any(not isinstance(v,(int,float)) or not 0<=v<=1 for v in p) for p in points):
                    raise ValueError('Polygon coordinates must be normalized within the image')
    return identity


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_HEAD(self):self.send_file(head=True)
    def do_GET(self):self.send_file()
    def send_file(self,head=False):
        route=unquote(urlsplit(self.path).path)
        if route in ('/','/index.html'):path=HTML
        elif route=='/pack.json':path=ROOT/'PACK.json'
        else:
            pack=json.loads((ROOT/'PACK.json').read_text())
            allowed={r['video'] for r in pack['records']}|{s['path'] for r in pack['records'] for s in r['stills']}
            relative=route.removeprefix('/')
            if relative not in allowed:self.send_error(404);return
            path=ROOT/relative
        size=path.stat().st_size;start,end,status=0,size-1,200
        request=self.headers.get('Range')
        if request:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',request)
            if not match or not any(match.groups()):self.send_error(416);return
            if match[1]:
                start=int(match[1]);end=min(end,int(match[2])) if match[2] else end
            else:start=max(0,size-int(match[2]))
            if start>end or start>=size:self.send_error(416);return
            status=206
        self.send_response(status)
        self.send_header('Content-Type',mimetypes.guess_type(path)[0] or 'application/octet-stream')
        self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
        self.send_header('Cache-Control','no-store')
        if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        if head:return
        try:
            with path.open('rb') as f:
                f.seek(start);left=end-start+1
                while left:
                    block=f.read(min(left,1024**2))
                    if not block:break
                    self.wfile.write(block);left-=len(block)
        except (BrokenPipeError,ConnectionResetError):pass

    def do_POST(self):
        if self.path!='/save':self.send_error(404);return
        if self.headers.get('Origin') not in (None,'http://'+self.headers.get('Host','')):
            self.send_error(403);return
        try:
            count=int(self.headers.get('Content-Length','0'))
            if count<1 or count>16*1024**2:raise ValueError('Invalid draft size')
            raw=self.rfile.read(count);value=json.loads(raw)
            identity=validate_draft(value,json.loads((ROOT/'PACK.json').read_text()))
            folder=ROOT/'drafts'/identity;folder.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256(raw).hexdigest()
            path=folder/(digest+'.json')
            if not path.exists():
                with path.open('xb') as f:f.write(raw)
            result={'status':'draft_saved_not_adjudicated','sha256':digest,
                    'saved_utc':datetime.now(timezone.utc).isoformat()}
            data=json.dumps(result).encode();self.send_response(200)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        except (ValueError,TypeError,KeyError) as e:self.send_error(400,str(e))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT);ap.add_argument('--port',type=int,default=8767)
    args=ap.parse_args();ROOT=args.root.resolve()
    assert (ROOT/'PACK.json').is_file()
    print(f'Annotation workspace: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
