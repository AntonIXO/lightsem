import http.client
import threading
import pytest
from scripts.rendering.serve_native_dynamics_3d import byte_range, make_server

@pytest.mark.parametrize('value,expected', [(None,(0,99,200)),('bytes=0-63',(0,63,206)),('bytes=98-',(98,99,206)),('bytes=-4',(96,99,206)),('bytes=5-200',(5,99,206))])
def test_ranges(value,expected):
    assert byte_range(value,100)==expected

@pytest.mark.parametrize('value',['bytes=','bytes=-0','bytes=100-','bytes=4-2','bytes=0-2,5-6'])
def test_invalid_ranges(value):
    with pytest.raises(ValueError): byte_range(value,100)


def test_loopback_exact_routes_and_range_bytes(tmp_path):
    root=tmp_path/'review'; root.mkdir(); (root/'index.html').write_text('gallery')
    movie=root/'C4'/'shpongle'/'preview.mp4'; movie.parent.mkdir(parents=True); movie.write_bytes(bytes(range(100)))
    (root/'secret.txt').write_text('private')
    escaped=root/'C3'/'shpongle'/'preview.mp4'; escaped.parent.mkdir(parents=True); escaped.symlink_to(tmp_path/'outside')
    (tmp_path/'outside').write_bytes(b'private')
    server=make_server(root,0); worker=threading.Thread(target=server.serve_forever,daemon=True); worker.start()
    try:
        conn=http.client.HTTPConnection('127.0.0.1',server.server_port)
        conn.request('GET','/C4/shpongle/preview.mp4',headers={'Range':'bytes=0-63'})
        response=conn.getresponse(); assert response.status==206 and response.getheader('Content-Range')=='bytes 0-63/100'; assert response.read()==bytes(range(64))
        conn.request('HEAD','/C4/shpongle/preview.mp4?download=1'); response=conn.getresponse(); assert response.status==200 and response.getheader('Content-Length')=='100'; assert 'shpongle_C4_90-150s.mp4' in response.getheader('Content-Disposition'); assert response.read()==b''
        for route in ['/secret.txt','/../outside','/%2e%2e/outside','/C3/shpongle/preview.mp4','/PLAN.json']:
            conn.request('GET',route); response=conn.getresponse(); assert response.status==404; response.read()
        conn.close()
    finally:
        server.shutdown(); server.server_close(); worker.join()
