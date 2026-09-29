"""Local-only review page with allowlisted media and byte-range seeking."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import re
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parent


class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.respond(head=True)

    def do_GET(self):
        self.respond()

    def respond(self, head=False):
        parsed = urlparse(self.path)
        manifest_path = ROOT / ('MANIFEST.json' if (ROOT / 'MANIFEST.json').exists() else 'PLAN.json')
        if parsed.path in ('/', '/index.html', '/manifest.json'):
            path = manifest_path if parsed.path == '/manifest.json' else ROOT / 'index.html'
            data = path.read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json' if parsed.path == '/manifest.json' else 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            if not head:
                self.wfile.write(data)
            return
        match = re.fullmatch(r'/media/(\d{2})\.mp4', parsed.path)
        if not match:
            self.send_error(404)
            return
        manifest = json.loads(manifest_path.read_text())
        item = next((v for v in manifest['variants'] if v['id'] == match[1]), None)
        if not item or item['status'] == 'planned':
            self.send_error(404)
            return
        path = Path(item['path'])
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        request_range = self.headers.get('Range')
        if request_range:
            m = re.fullmatch(r'bytes=(\d*)-(\d*)', request_range)
            if not m or not any(m.groups()):
                self.send_error(416)
                return
            if not m[1]:
                start = max(0, size - int(m[2]))
            else:
                start = int(m[1])
                if m[2]:
                    end = min(size - 1, int(m[2]))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header('Content-Range', 'bytes */' + str(size))
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header('Content-Type', 'video/mp4')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        if parse_qs(parsed.query).get('download') == ['1']:
            self.send_header('Content-Disposition', f'attachment; filename="shpongle_{item["id"]}_1080p_av1.mp4"')
        self.end_headers()
        if head:
            return
        try:
            with path.open('rb') as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining:
                    data = f.read(min(1024 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8765), Handler)
    print('http://127.0.0.1:8765', flush=True)
    server.serve_forever()
