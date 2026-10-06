"""Loopback-only, six-file allowlisted Range server for human-review previews."""
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse,quote
import argparse,json,hashlib,re
from scripts.rendering import serve_light_review as base

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def manifest(root):
    rows=[]
    for song in ('shpongle','cure_for_me'):
        for method in ('D7_reference','C3','C4'):
            folder=root/method/song;record=json.loads((folder/'RENDER.json').read_text());audit=json.loads((folder/'AUDIT.json').read_text());movie=folder/'preview.mp4'
            if not movie.resolve().is_relative_to(root.resolve()):raise ValueError('Media escapes preview root')
            if not (audit.get('full_software_AV_decode') and audit.get('pts_uniform') and audit.get('audio_matches_prepared_reference') and audit.get('decoded_frames')==1800):raise ValueError('Incomplete AV audit')
            if sha(movie)!=record['sha256'] or audit['movie_sha256']!=record['sha256'] or audit['render_sha256']!=sha(folder/'RENDER.json'):raise ValueError('Changed or unaudited preview')
            rows.append({'id':f'{len(rows)+1:02d}','status':'complete','path':str(movie.resolve()),'route':'/'+str(movie.relative_to(root)),'sha256':record['sha256'],'bytes':movie.stat().st_size,'download_name':f'{song}_{method}_90-150s_3d.mp4'})
    value={'variants':rows,'baseline_promoted':False,'scope':'Human-review previews; source90–150s, allA/B/C forbothsongs.'};p=root/'MANIFEST.json'
    if p.exists():
        if json.loads(p.read_text())!=value:raise ValueError('Changed published media manifest')
    else:p.write_text(json.dumps(value,indent=2)+'\n')
    return value

class Handler(base.Handler):
    routes={};names={}
    def respond(self,head=False):
        parsed=urlparse(self.path)
        if parsed.path in self.routes:self.path='/media/'+self.routes[parsed.path]+'.mp4'+(('?'+parsed.query) if parsed.query else '')
        return super().respond(head=head)
    def send_header(self,key,value):
        if key.lower()=='content-disposition':
            match=re.fullmatch(r'/media/(\d{2})\.mp4',urlparse(self.path).path)
            if match:value="attachment; filename*=UTF-8''"+quote(self.names[match[1]],safe='')
        return super().send_header(key,value)


def byte_range(value, size):
    if size < 1:
        raise ValueError('Empty media')
    if value is None:
        return 0, size - 1, 200
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', value)
    if not match or not any(match.groups()):
        raise ValueError('Invalid single byte range')
    if not match[1]:
        length = int(match[2])
        if length <= 0:
            raise ValueError('Invalid suffix byte range')
        return max(0, size - length), size - 1, 206
    start = int(match[1]); end = min(size - 1, int(match[2])) if match[2] else size - 1
    if start >= size or start > end:
        raise ValueError('Unsatisfiable byte range')
    return start, end, 206


def make_server(root, port):
    """Low-level loopback preview server; CLI authenticates all AV receipts first."""
    root = Path(root).resolve()
    allowed = {}
    index = 0
    for song in ('shpongle', 'cure_for_me'):
        for method in ('D7_reference', 'C3', 'C4'):
            index += 1
            path = root / method / song / 'preview.mp4'
            if path.is_file() and path.resolve().is_relative_to(root):
                allowed[f'/{method}/{song}/preview.mp4'] = (path, f'{song}_{method}_90-150s.mp4')
                allowed[f'/media/{index:02d}.mp4'] = allowed[f'/{method}/{song}/preview.mp4']
    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self): self.respond(False)
        def do_HEAD(self): self.respond(True)
        def log_message(self, *args): pass
        def respond(self, head):
            parsed = urlparse(self.path)
            if parsed.path == '/manifest.json' and (root / 'MANIFEST.json').is_file():
                data = (root / 'MANIFEST.json').read_bytes()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data))); self.end_headers()
                if not head: self.wfile.write(data)
                return
            if parsed.path in ('/', '/index.html'):
                data = (root / 'index.html').read_bytes()
                self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(data))); self.end_headers()
                if not head: self.wfile.write(data)
                return
            if parsed.path not in allowed:
                self.send_error(404); return
            path, filename = allowed[parsed.path]
            if not path.is_file() or not path.resolve().is_relative_to(root):
                self.send_error(404); return
            size = path.stat().st_size
            try: start, end, status = byte_range(self.headers.get('Range'), size)
            except ValueError:
                self.send_response(416); self.send_header('Content-Range', f'bytes */{size}'); self.end_headers(); return
            self.send_response(status); self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes'); self.send_header('Content-Length', str(end-start+1))
            if status == 206: self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            if 'download=1' in parsed.query:
                self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + quote(filename, safe=''))
            self.end_headers()
            if head: return
            try:
                with path.open('rb') as stream:
                    stream.seek(start); remaining = end-start+1
                    while remaining:
                        chunk = stream.read(min(remaining, 1024*1024))
                        if not chunk: break
                        self.wfile.write(chunk); remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError): pass
    return ThreadingHTTPServer(('127.0.0.1', port), PreviewHandler)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--port',type=int,default=8780);a=p.parse_args();base.ROOT=a.root.resolve();m=manifest(base.ROOT);Handler.routes={x['route']:x['id'] for x in m['variants']};Handler.names={x['id']:x['download_name'] for x in m['variants']}
    server=make_server(base.ROOT,a.port);print(f'http://127.0.0.1:{a.port}/',flush=True);server.serve_forever()
