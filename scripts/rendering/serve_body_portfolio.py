"""Local portfolio review using the existing allowlisted Range-capable server."""
import argparse
import json
from pathlib import Path
import re
from http.server import ThreadingHTTPServer
from urllib.parse import quote, urlparse

from scripts.rendering import serve_light_review as base


class PortfolioHandler(base.Handler):
    def send_header(self, keyword, value):
        if keyword.lower() == 'content-disposition':
            match = re.fullmatch(r'/media/(\d{2})\.mp4', urlparse(self.path).path)
            manifest = json.loads((base.ROOT / 'MANIFEST.json').read_text())
            row = next((r for r in manifest['variants'] if match and r['id'] == match[1]), None)
            if row:
                filename = Path(row.get('download_name', Path(row['path']).name)).name
                value = "attachment; filename*=UTF-8''" + quote(filename, safe='')
        super().send_header(keyword, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8773)
    args = parser.parse_args()
    base.ROOT = args.root.resolve()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), PortfolioHandler)
    print(f'http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()
