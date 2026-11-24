"""BanglaLPR showcase web app: test V1 / V2 on photos, videos and a live camera; every result is stored locally.

  D:/498R/_dataset_build/venv/Scripts/python.exe webapp/server.py [--port 8498] [--preload]

Only the Python standard library is used for the server (no Flask/FastAPI to install): http.server + sqlite3.
Data: webapp/data/lpr.db (SQLite) and webapp/data/media/ (photos, plate crops, annotated photos and videos).
Nothing is sent to the internet.
"""
import argparse
import csv
import io
import json
import mimetypes
import os
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
sys.path.insert(0, ROOT)
from lpr import vocab  # noqa: E402
from lpr.pipeline import LPRSystem, draw  # noqa: E402
from lpr.store import Store, event_payload  # noqa: E402

STATIC = os.path.join(HERE, 'static')
mimetypes.add_type('application/manifest+json', '.webmanifest')
DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
STORE = Store(os.path.join(HERE, 'data'))
SYSTEMS, LOAD_LOCK, JOBS = {}, threading.Lock(), {}
RECENT = {}                                  # live camera: last time each plate was stored (duplicate suppression)
DEDUP_S = 10.0
MAX_UPLOAD = 300 * 2 ** 20


def system(v):
    if v not in ('v1', 'v2'):
        raise ValueError('version must be v1 or v2')
    with LOAD_LOCK:
        if v not in SYSTEMS:
            t0 = time.perf_counter()
            s = LPRSystem(os.path.join(ROOT, v.upper()))          # the standalone V1/ or V2/ model folder
            for smp in samples()[:2]:            # warm-up: the first GPU call per input shape is slow
                img = cv2.imread(os.path.join(DATA, 'e2e_test', 'images', smp['file']))
                if img is not None:
                    s.process(img)
            SYSTEMS[v] = s
            print(f'loaded {v} in {time.perf_counter() - t0:.1f} s', flush=True)
        return SYSTEMS[v]


def save_jpg(img, prefix):
    name = f'{prefix}_{uuid.uuid4().hex[:10]}.jpg'
    cv2.imwrite(STORE.media(name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return name


def public(plate, version, run_id=None, save=True):
    """JSON-safe plate dict (+ saved crop, + DB row when save)."""
    p = {k: v for k, v in plate.items() if not k.startswith('_')}
    c = plate.get('_crop')
    p['crop'] = save_jpg(c, 'crop') if (save and c is not None and c.size) else None
    msg = event_payload(p, version)
    p['event_message'] = msg
    p['event_bytes'] = len(msg.encode('utf-8'))
    if save and run_id is not None:
        p['id'], p['watch_hit'] = STORE.add_plate(run_id, version, p, p['crop'])
    else:
        n = vocab.norm_text(p.get('text', ''))
        p['watch_hit'] = int(bool(n) and any(w['text_norm'] == n for w in STORE.watchlist()))
    return p


def recognize(img, version, filename, kind='photo', save=True):
    s = system(version)
    res = s.process(img)
    image = annotated = None
    run_id = None
    if save:
        image = save_jpg(img, 'photo')
        annotated = save_jpg(draw(img, res), 'annot')
        run_id = STORE.add_run(kind, filename, version, res, image, annotated,
                               {'retry_used': res['retry_used']})
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return {'version': version, 'run_id': run_id, 'image': image, 'annotated': annotated, 'ms': res['ms'],
            'retry_used': res['retry_used'], 'photo_bytes': int(buf.size) if ok else None,
            'size': [int(img.shape[1]), int(img.shape[0])],
            'plates': [public(p, version, run_id, save) for p in res['plates']]}


def run_video(job, path, version, filename):
    from lpr.video import process_video
    try:
        s = system(version)
        out_name = f'video_{uuid.uuid4().hex[:10]}.mp4'

        def prog(i, n):
            job.update(frame=i, total=n)
        res = process_video(s, path, STORE.media(out_name), progress=prog)
        run_id = STORE.add_run('video', filename, version, {'events': res['events'], 'ms': {}}, None, out_name,
                               {k: v for k, v in res.items() if k != 'events'})
        events = []
        for e in res['events']:
            p = {**e, 'valid': vocab and bool(e['text']), 'conf': None}
            from lpr import metrics
            p['valid'] = metrics.is_valid_plate(e['text'])
            events.append(public(p, version, run_id))
        job.update(state='done', result={**{k: v for k, v in res.items() if k != 'events'}, 'events': events,
                                         'video': out_name, 'run_id': run_id})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.update(state='error', error=str(e))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def report():
    r = {}
    for name in ('system_gpu', 'system_cpu', 'readers', 'detectors', 'detector_combo', 'fusion_selection',
                 'char_verifier_val', 'sr_check_val'):
        f = os.path.join(ROOT, 'eval', 'results', f'{name}.json')
        if os.path.exists(f):
            d = json.load(open(f, encoding='utf-8'))
            if name == 'fusion_selection':
                d = {k: {kk: vv for kk, vv in v.items() if kk != 'top8_by_val'} if isinstance(v, dict) else v
                     for k, v in d.items()}
            r[name] = d
    r['configs'] = {v: json.load(open(os.path.join(ROOT, v.upper(), 'models', 'config.json'), encoding='utf-8')) for v in ('v1', 'v2')}
    return r


def samples():
    f = os.path.join(DATA, 'e2e_test', 'e2e_test.csv')
    if not os.path.exists(f):
        return []
    rows = list(csv.DictReader(open(f, encoding='utf-8')))
    pick = [r for r in rows if r['source'] == 'blpd_ext'][:16] + [r for r in rows if r['source'] != 'blpd_ext'][:8]
    return [{'file': r['file'], 'text': r['text'], 'source': r['source']} for r in pick]


class Handler(BaseHTTPRequestHandler):
    server_version = 'BanglaLPR/1.0'

    def log_message(self, fmt, *args):
        msg = fmt % args
        if '/api/frame' not in msg:                  # the live camera would flood the log
            sys.stderr.write('%s - %s\n' % (self.log_date_time_string(), msg))

    # ------------------------------------------------------------------ helpers
    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def send_file(self, path, download=None):
        if not os.path.isfile(path):
            return self.send_error(HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        if path.endswith('.js'):
            ctype = 'text/javascript'
        size = os.path.getsize(path)
        self.send_response(200)
        self.send_header('Content-Type', ctype + ('; charset=utf-8' if ctype.startswith('text/') else ''))
        self.send_header('Content-Length', str(size))
        if download:
            self.send_header('Content-Disposition', f'attachment; filename="{download}"')
        self.end_headers()
        with open(path, 'rb') as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_UPLOAD:
            raise ValueError('file too large (max 300 MB)')
        return self.rfile.read(n) if n else b''

    def image_from_body(self):
        data = self.body()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('could not read the image (send a JPEG/PNG/WebP file)')
        return img

    @staticmethod
    def safe(base, rel):
        p = os.path.realpath(os.path.join(base, unquote(rel)))
        if not p.startswith(os.path.realpath(base) + os.sep):
            raise PermissionError(rel)
        return p

    # ------------------------------------------------------------------ routes
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                return self.send_file(os.path.join(STATIC, 'index.html'))
            if u.path in ('/app', '/app/'):                   # the mobile app (installable PWA)
                return self.send_file(os.path.join(STATIC, 'mobile.html'))
            if u.path == '/manifest.webmanifest':
                return self.send_file(os.path.join(STATIC, 'manifest.webmanifest'))
            if u.path == '/sw.js':                            # service worker must sit at the root to cover /app
                return self.send_file(os.path.join(STATIC, 'sw.js'))
            if u.path.startswith('/static/'):
                return self.send_file(self.safe(STATIC, u.path[len('/static/'):]))
            if u.path.startswith('/media/'):
                return self.send_file(self.safe(STORE.media(''), u.path[len('/media/'):]))
            if u.path.startswith('/sample/'):
                return self.send_file(self.safe(os.path.join(DATA, 'e2e_test', 'images'), u.path[len('/sample/'):]))
            if u.path == '/api/info':
                return self.send_json({'loaded': sorted(SYSTEMS), 'report': report(), 'samples': samples()})
            if u.path == '/api/history':
                return self.send_json(STORE.plates(q.get('q', ''), q.get('status', ''), q.get('version', ''),
                                                   int(q.get('limit', 200)), int(q.get('offset', 0))))
            if u.path == '/api/stats':
                return self.send_json(STORE.stats())
            if u.path == '/api/watchlist':
                return self.send_json(STORE.watchlist())
            if u.path.startswith('/api/job/'):
                j = JOBS.get(u.path.rsplit('/', 1)[-1])
                return self.send_json(j if j else {'state': 'missing'}, 200 if j else 404)
            if u.path == '/api/export.csv':
                rows = STORE.plates(limit=100000)
                buf = io.StringIO()
                w = csv.writer(buf)
                cols = ['id', 'created', 'version', 'kind', 'filename', 'text', 'text_en', 'status', 'conf', 'det_conf',
                        'track_id', 'frames', 'watch_hit', 'payload_bytes', 'crop']
                w.writerow(cols)
                for r in rows:
                    r['created'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r['created']))
                    w.writerow([r.get(c) for c in cols])
                b = ('\ufeff' + buf.getvalue()).encode('utf-8')      # BOM: Excel shows the Bangla text
                self.send_response(200)
                self.send_header('Content-Type', 'text/csv; charset=utf-8')
                self.send_header('Content-Disposition', 'attachment; filename="banglalpr_results.csv"')
                self.send_header('Content-Length', str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            return self.send_error(HTTPStatus.NOT_FOUND)
        except PermissionError:
            return self.send_error(HTTPStatus.FORBIDDEN)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self.send_json({'error': str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        name = unquote(self.headers.get('X-Filename', 'upload'))[:200]
        try:
            if u.path == '/api/recognize':
                img = self.image_from_body()
                vs = ['v1', 'v2'] if q.get('version') == 'both' else [q.get('version', 'v2')]
                save = q.get('save', '1') == '1'
                return self.send_json({'results': [recognize(img, v, name, 'photo', save) for v in vs]})
            if u.path == '/api/sample':
                f = self.safe(os.path.join(DATA, 'e2e_test', 'images'), q['file'])
                img = cv2.imread(f)
                vs = ['v1', 'v2'] if q.get('version') == 'both' else [q.get('version', 'v2')]
                truth = next((s['text'] for s in samples() if s['file'] == q['file']), None)
                res = [recognize(img, v, q['file'], 'sample') for v in vs]
                return self.send_json({'results': res, 'truth': truth})
            if u.path == '/api/frame':                       # live camera: store only new readable plates
                img = self.image_from_body()
                v = q.get('version', 'v2')
                r = recognize(img, v, 'camera', 'camera', save=False)
                now, stored = time.time(), []
                keep = [p for p in r['plates'] if p['valid']]
                fresh = [p for p in keep if now - RECENT.get(vocab.norm_text(p['text']), 0) > DEDUP_S]
                if fresh:
                    res = SYSTEMS[v].process(img)                # re-run to get crops for storage
                    run_id = STORE.add_run('camera', 'camera', v, res, None, save_jpg(draw(img, res), 'annot'))
                    for p in res['plates']:
                        n = vocab.norm_text(p['text'])
                        if p['valid'] and now - RECENT.get(n, 0) > DEDUP_S:
                            RECENT[n] = now
                            stored.append(public(p, v, run_id))
                r['stored'] = stored
                return self.send_json(r)
            if u.path == '/api/video_demo':                  # the synthetic clip made by tools/make_demo_video.py
                src = os.path.join(HERE, 'demo', 'demo_from_test_photos.mp4')
                if not os.path.exists(src):
                    raise ValueError('run tools/make_demo_video.py first')
                tmp = STORE.media(f'upload_{uuid.uuid4().hex[:10]}.mp4')
                with open(src, 'rb') as fi, open(tmp, 'wb') as fo:
                    fo.write(fi.read())
                jid = uuid.uuid4().hex[:12]
                job = {'state': 'running', 'frame': 0, 'total': 0, 'file': 'demo_from_test_photos.mp4',
                       'version': q.get('version', 'v2')}
                JOBS[jid] = job
                threading.Thread(target=run_video, args=(job, tmp, job['version'], job['file']), daemon=True).start()
                return self.send_json({'job': jid})
            if u.path == '/api/video':
                data = self.body()
                ext = os.path.splitext(name)[1].lower() or '.mp4'
                tmp = STORE.media(f'upload_{uuid.uuid4().hex[:10]}{ext}')
                with open(tmp, 'wb') as f:
                    f.write(data)
                jid = uuid.uuid4().hex[:12]
                job = {'state': 'running', 'frame': 0, 'total': 0, 'file': name, 'version': q.get('version', 'v2')}
                JOBS[jid] = job
                threading.Thread(target=run_video, args=(job, tmp, job['version'], name), daemon=True).start()
                return self.send_json({'job': jid})
            if u.path == '/api/watchlist':
                d = json.loads(self.body() or b'{}')
                if not vocab.norm_text(d.get('text', '')):
                    raise ValueError('empty plate text')
                STORE.watch_add(d['text'].strip(), d.get('note', '')[:200])
                return self.send_json({'ok': True})
            if u.path == '/api/retention':
                d = json.loads(self.body() or b'{}')
                return self.send_json({'deleted_plates': STORE.delete_older_than(float(d.get('days', 30)))})
            if u.path == '/api/clear':
                STORE.clear()
                return self.send_json({'ok': True})
            if u.path == '/api/load':
                for v in (['v1', 'v2'] if q.get('version') == 'both' else [q.get('version', 'v2')]):
                    system(v)
                return self.send_json({'loaded': sorted(SYSTEMS)})
            return self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as e:
            return self.send_json({'error': str(e)}, 400)
        except PermissionError:
            return self.send_error(HTTPStatus.FORBIDDEN)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self.send_json({'error': str(e)}, 500)

    def do_HEAD(self):
        self.send_response(200 if urlparse(self.path).path in ('/', '/index.html') else 404)
        self.end_headers()

    def do_DELETE(self):
        u = urlparse(self.path)
        if u.path.startswith('/api/watchlist/'):
            STORE.watch_del(int(u.path.rsplit('/', 1)[-1]))
            return self.send_json({'ok': True})
        return self.send_error(HTTPStatus.NOT_FOUND)


def lan_ips():
    """IPv4 addresses other devices on the Wi-Fi can use to reach this PC."""
    import socket
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('10.255.255.255', 1))                # no packet is sent; picks the outgoing interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        ips.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(i for i in ips if not i.startswith(('127.', '169.254.')))


def self_signed_cert(ips):
    """A self-signed certificate for this PC's addresses (phones only allow the camera on https pages).
    Made once with openssl (Git for Windows ships it) and remade when the PC's address changes."""
    import shutil
    import subprocess
    d = os.path.join(HERE, 'data', 'certs')
    os.makedirs(d, exist_ok=True)
    crt, key, names = os.path.join(d, 'server.crt'), os.path.join(d, 'server.key'), os.path.join(d, 'names.txt')
    san = ','.join(['DNS:localhost', 'IP:127.0.0.1'] + [f'IP:{i}' for i in ips])
    if os.path.exists(crt) and os.path.exists(key) and open(names).read() == san:
        return crt, key
    exe = shutil.which('openssl') or r'C:\Program Files\Git\usr\bin\openssl.exe'
    subprocess.run([exe, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825', '-keyout', key, '-out', crt,
                    '-subj', '/CN=BanglaLPR local', '-addext', f'subjectAltName={san}'],
                   check=True, capture_output=True)
    with open(names, 'w') as f:
        f.write(san)
    return crt, key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8498)))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--preload', action='store_true', help='load V1 and V2 before serving')
    p.add_argument('--phone', action='store_true',
                   help='serve on the Wi-Fi with https (self-signed), so a phone can use the live camera')
    a = p.parse_args()
    if a.preload:
        system('v1'); system('v2')
    host = '0.0.0.0' if a.phone else a.host
    srv = ThreadingHTTPServer((host, a.port), Handler)
    scheme = 'http'
    if a.phone:
        import ssl
        ips = lan_ips()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(*self_signed_cert(ips))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        scheme = 'https'
        print('Open on your phone (same Wi-Fi): ' + '  '.join(f'https://{i}:{a.port}' for i in ips), flush=True)
        print('The browser warns "not private" (self-signed certificate): tap Advanced -> Proceed.', flush=True)
    print(f'BanglaLPR web app on {scheme}://{host}:{a.port}  (data: {STORE.folder})', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
