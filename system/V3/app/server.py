"""BanglaLPR V3 web app: read Bangla plates from photos, videos or a live camera; every plate is saved to the
records folder as a plate image + a row in plates.csv.

  D:/498R/_dataset_build/venv/Scripts/python.exe app/server.py            (then open http://127.0.0.1:8499)
  options: --port 8499  --phone (https on the Wi-Fi so a phone can use its camera)  --cpu  --preload

Only the Python standard library is used for the server (http.server + sqlite3). Nothing is sent to the internet.
Records: V3/records/plates.csv, V3/records/plate_images/, V3/records/photos/, V3/records/exports/.
"""
import argparse
import csv
import json
import mimetypes
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # the V3 folder
sys.path.insert(0, ROOT)
from lpr import metrics, vocab  # noqa: E402
from lpr.pipeline import LPRSystem, draw  # noqa: E402
from lpr.store import Store, event_payload  # noqa: E402

STATIC = os.path.join(HERE, 'static')
DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
DEMO_CLIP = os.path.join(ROOT, '..', 'webapp', 'demo', 'demo_from_test_photos.mp4')
RECORDS = os.environ.get('BANGLALPR_RECORDS', os.path.join(ROOT, 'records'))
STORE = Store(RECORDS, 'v3')
ENGINE = {'lpr': None, 'state': 'not loaded', 'device': None, 'seconds': None}
LOAD_LOCK, JOBS, RECENT = threading.Lock(), {}, {}
DARK_HINT = 63.0       # night_level < this -> the page SUGGESTS night mode; chosen on val: flags 5 % of day photos, catches 49 % of night ones (test: 4.0 % / 40.5 %)
DEDUP_S = 10.0                                                 # live camera: store a plate once per 10 s
MAX_UPLOAD = 300 * 2 ** 20
mimetypes.add_type('application/manifest+json', '.webmanifest')


def engine():
    with LOAD_LOCK:
        if ENGINE['lpr'] is None:
            ENGINE['state'] = 'loading'
            t0 = time.perf_counter()
            lpr = LPRSystem(ROOT, device=ENGINE['device'])
            lpr.process(np.full((720, 1280, 3), 127, np.uint8))            # warm-up: first GPU call is slow
            for s in samples()[:1]:
                img = cv2.imread(os.path.join(DATA, 'e2e_test', 'images', s['file']))
                if img is not None:
                    lpr.process(img)
            ENGINE.update(lpr=lpr, state='ready', seconds=round(time.perf_counter() - t0, 1))
            print(f'V3 engine loaded in {ENGINE["seconds"]} s', flush=True)
        return ENGINE['lpr']


def plate_out(p, rec=None):
    """JSON-safe plate for the page (+ the stored record's id / image path when it was saved)."""
    d = {k: v for k, v in p.items() if not k.startswith('_')}
    msg = event_payload(d, 'v3')
    d['event_bytes'] = len(msg.encode('utf-8'))
    if rec:
        d.update(id=rec['id'], crop=rec['crop'], watch_hit=rec['watch_hit'])
    else:
        n = vocab.norm_text(d.get('text', ''))
        d['crop'] = None
        d['watch_hit'] = int(bool(n) and any(w['text_norm'] == n for w in STORE.watchlist()))
    return d


def recognize(img, filename, kind='photo', save=True, night=False):
    res = engine().process(img, night=night)
    for p in res['plates']:
        p['night'] = res['night_mode']
    out = {'ms': res['ms'], 'retry_used': res['retry_used'], 'size': [int(img.shape[1]), int(img.shape[0])],
           'run_id': None, 'annotated': None, 'plates': [], 'night_mode': res['night_mode'],
           'night_level': res['night_level'], 'looks_dark': res['night_level'] < DARK_HINT}
    if save and res['plates']:                                 # nothing to keep when no plate was found
        run_id, ann = STORE.add_run(kind, filename, res, draw(img, res),
                                    extra={'retry_used': res['retry_used'], 'night_mode': res['night_mode'],
                                           'night_level': res['night_level']})
        out.update(run_id=run_id, annotated=ann)
        out['plates'] = [plate_out(p, STORE.add_plate(run_id, p, p.get('_crop'), kind, filename, ann))
                         for p in res['plates']]
    else:
        out['plates'] = [plate_out(p) for p in res['plates']]
    return out


def run_video(job, path, filename, night=False):
    from lpr.video import process_video
    try:
        lpr = engine()
        rel = f'videos/video_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:6]}.mp4'
        os.makedirs(os.path.dirname(STORE.path(rel)), exist_ok=True)
        res = process_video(lpr, path, STORE.path(rel), progress=lambda i, n: job.update(frame=i, total=n), night=night)
        info = {k: v for k, v in res.items() if k != 'events'}
        run_id, _ = STORE.add_run('video', filename, {'events': res['events'], 'ms': {}}, None, rel, info)
        events = []
        for e in res['events']:
            p = {**e, 'valid': metrics.is_valid_plate(e['text']), 'conf': None, 'night': night}
            events.append(plate_out(p, STORE.add_plate(run_id, p, e.get('_crop'), 'video', filename, rel)))
        job.update(state='done', result={**info, 'events': events, 'video': rel, 'run_id': run_id})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        job.update(state='error', error=str(e))
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def samples():
    f = os.path.join(DATA, 'e2e_test', 'e2e_test.csv')
    if not os.path.exists(f):
        return []
    rows = list(csv.DictReader(open(f, encoding='utf-8')))
    pick = [r for r in rows if r['source'] == 'blpd_ext'][:9] + [r for r in rows if r['source'] != 'blpd_ext'][:3]
    return [{'file': r['file'], 'text': r['text'], 'source': r['source']} for r in pick]


def model_info():
    cfg = json.load(open(os.path.join(ROOT, 'models', 'config.json'), encoding='utf-8'))
    summ = os.path.join(ROOT, 'results', 'summary.json')
    s = json.load(open(summ, encoding='utf-8')) if os.path.exists(summ) else {}
    g = s.get('gpu', {})
    return {'name': cfg.get('name'), 'changes': cfg.get('v3_changes'), 'papers': cfg.get('built_from_papers'),
            'measured_on': s.get('measured_on'), 'measured_version': s.get('version'),
            'e2e': g.get('e2e_photos', {}), 'crops': g.get('ocr_test_crops', {}),
            'cpu': s.get('cpu', {}).get('e2e_photos', {})}


def open_in_explorer(path):
    if sys.platform.startswith('win'):
        if os.path.isfile(path):
            import subprocess
            subprocess.Popen(['explorer', '/select,', os.path.normpath(path)])
        else:
            os.startfile(path)  # noqa: S606
    else:
        raise ValueError('opening folders works on Windows only; the path is ' + path)


class Handler(BaseHTTPRequestHandler):
    server_version = 'BanglaLPR-V3/1.0'

    def log_message(self, fmt, *args):
        msg = fmt % args
        if '/api/frame' not in msg and '/api/job/' not in msg:
            sys.stderr.write('%s - %s\n' % (self.log_date_time_string(), msg))

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(b)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(b)

    def send_file(self, path, download=None, cache=False):
        if not os.path.isfile(path):
            return self.send_error(HTTPStatus.NOT_FOUND)
        ctype = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        if path.endswith('.js'):
            ctype = 'text/javascript'
        if path.endswith('.csv'):
            ctype = 'text/csv'
        self.send_response(200)
        self.send_header('Content-Type', ctype + ('; charset=utf-8' if ctype.startswith('text/') else ''))
        self.send_header('Content-Length', str(os.path.getsize(path)))
        self.send_header('Cache-Control', 'max-age=3600' if cache else 'no-store')
        if download:
            self.send_header('Content-Disposition', f"attachment; filename*=UTF-8''{quote(download)}")
        self.end_headers()
        with open(path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile, 1 << 20)

    def body(self):
        n = int(self.headers.get('Content-Length') or 0)
        if n > MAX_UPLOAD:
            raise ValueError('file too large (max 300 MB)')
        return self.rfile.read(n) if n else b''

    def image_from_body(self):
        img = cv2.imdecode(np.frombuffer(self.body(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError('could not read the image (send a JPEG, PNG or WebP photo)')
        return img

    @staticmethod
    def safe(base, rel):
        p = os.path.realpath(os.path.join(base, unquote(rel)))
        if not p.startswith(os.path.realpath(base) + os.sep):
            raise PermissionError(rel)
        return p

    def local_client(self):
        return self.client_address[0] in ('127.0.0.1', '::1', '::ffff:127.0.0.1')

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ('/', '/index.html'):
                return self.send_file(os.path.join(STATIC, 'index.html'))
            if u.path == '/manifest.webmanifest':
                return self.send_file(os.path.join(STATIC, 'manifest.webmanifest'))
            if u.path.startswith('/static/'):
                return self.send_file(self.safe(STATIC, u.path[len('/static/'):]))
            if u.path.startswith('/records/'):
                rel = u.path[len('/records/'):]
                if rel.endswith('lpr.db'):
                    raise PermissionError(rel)
                return self.send_file(self.safe(STORE.folder, rel), cache=True)
            if u.path.startswith('/sample/'):
                return self.send_file(self.safe(os.path.join(DATA, 'e2e_test', 'images'), u.path[len('/sample/'):]), cache=True)
            if u.path == '/api/info':
                return self.send_json({'engine': ENGINE['state'], 'load_s': ENGINE['seconds'], 'model': model_info(),
                                       'samples': samples(), 'records': STORE.folder, 'csv': STORE.log.csv_path,
                                       'demo_clip': os.path.exists(DEMO_CLIP), 'local': self.local_client()})
            if u.path == '/api/history':
                return self.send_json(STORE.plates(q.get('q', ''), q.get('status', ''), int(q.get('limit', 200)),
                                                   int(q.get('offset', 0)), float(q['since']) if q.get('since') else None))
            if u.path == '/api/stats':
                return self.send_json(STORE.stats())
            if u.path == '/api/watchlist':
                return self.send_json(STORE.watchlist())
            if u.path.startswith('/api/job/'):
                j = JOBS.get(u.path.rsplit('/', 1)[-1])
                return self.send_json(j if j else {'state': 'missing'}, 200 if j else 404)
            if u.path == '/api/plates.csv':                     # the live records/plates.csv
                return self.send_file(STORE.log.csv_path, download=f'banglalpr_v3_plates_{time.strftime("%Y%m%d")}.csv')
            if u.path.startswith('/api/exports/'):              # a zip made by POST /api/export
                return self.send_file(self.safe(os.path.join(STORE.folder, 'exports'), u.path[len('/api/exports/'):]),
                                      download=os.path.basename(u.path))
            return self.send_error(HTTPStatus.NOT_FOUND)
        except PermissionError:
            return self.send_error(HTTPStatus.FORBIDDEN)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self.send_json({'error': str(e)}, 500)

    # ------------------------------------------------------------------ POST
    def start_video(self, path, name, night=False):
        jid = uuid.uuid4().hex[:12]
        JOBS[jid] = job = {'state': 'running', 'frame': 0, 'total': 0, 'file': name, 'night': night}
        threading.Thread(target=run_video, args=(job, path, name, night), daemon=True).start()
        return self.send_json({'job': jid})

    def do_POST(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        name = unquote(self.headers.get('X-Filename', 'upload'))[:200]
        night = q.get('night') == '1'                           # night mode: the page's switch (Auto by clock / On)
        try:
            if u.path == '/api/recognize':
                return self.send_json(recognize(self.image_from_body(), name, q.get('kind', 'photo')[:20],
                                                q.get('save', '1') == '1', night))
            if u.path == '/api/sample':
                img = cv2.imread(self.safe(os.path.join(DATA, 'e2e_test', 'images'), q['file']))
                r = recognize(img, q['file'], 'sample', night=night)
                r['truth'] = next((s['text'] for s in samples() if s['file'] == q['file']), None)
                return self.send_json(r)
            if u.path == '/api/frame':                          # live camera: store only new readable plates
                img = self.image_from_body()
                lpr = engine()
                res = lpr.process(img, night=night)
                for p in res['plates']:
                    p['night'] = night
                now, stored = time.time(), []
                fresh = [p for p in res['plates'] if p['valid'] and now - RECENT.get(vocab.norm_text(p['text']), 0) > DEDUP_S]
                if fresh:
                    run_id, ann = STORE.add_run('camera', 'live camera', res, draw(img, res))
                    for p in fresh:
                        RECENT[vocab.norm_text(p['text'])] = now
                        stored.append(plate_out(p, STORE.add_plate(run_id, p, p.get('_crop'), 'camera', 'live camera', ann)))
                return self.send_json({'ms': res['ms'], 'plates': [plate_out(p) for p in res['plates']], 'stored': stored,
                                       'night_mode': night, 'night_level': res['night_level'],
                                       'looks_dark': res['night_level'] < DARK_HINT})
            if u.path == '/api/video':
                ext = os.path.splitext(name)[1].lower() or '.mp4'
                tmp = os.path.join(STORE.folder, f'_upload_{uuid.uuid4().hex[:10]}{ext}')
                with open(tmp, 'wb') as f:
                    f.write(self.body())
                return self.start_video(tmp, name, night)
            if u.path == '/api/video_demo':
                if not os.path.exists(DEMO_CLIP):
                    raise ValueError('demo clip not found (BanglaLPR_System/webapp/demo)')
                tmp = os.path.join(STORE.folder, f'_upload_{uuid.uuid4().hex[:10]}.mp4')
                shutil.copyfile(DEMO_CLIP, tmp)
                return self.start_video(tmp, 'demo clip (made from test photos)', night)
            if u.path == '/api/export':
                d = json.loads(self.body() or b'{}')
                r = STORE.export(d.get('q', ''), d.get('status', ''), d.get('since'))
                r['download'] = '/api/exports/' + os.path.basename(r['zip'])
                return self.send_json(r)
            if u.path == '/api/open':                           # open a folder in Windows Explorer (this PC only)
                if not self.local_client():
                    raise PermissionError('open folder: only from this PC')
                d = json.loads(self.body() or b'{}')
                what = d.get('what', 'records')
                if what == 'csv':
                    target = STORE.log.csv_path
                elif what == 'images':
                    target = os.path.join(STORE.folder, 'plate_images')
                elif what == 'export':
                    target = self.safe(os.path.join(STORE.folder, 'exports'), d.get('name', ''))
                else:
                    target = STORE.folder
                open_in_explorer(target)
                return self.send_json({'ok': True, 'path': target})
            if u.path == '/api/watchlist':
                d = json.loads(self.body() or b'{}')
                if not vocab.norm_text(d.get('text', '')):
                    raise ValueError('type the plate text first')
                STORE.watch_add(d['text'].strip(), d.get('note', '')[:200])
                return self.send_json({'ok': True})
            if u.path == '/api/retention':
                d = json.loads(self.body() or b'{}')
                return self.send_json({'deleted_plates': STORE.delete_older_than(float(d.get('days', 30)))})
            if u.path == '/api/clear':
                STORE.clear()
                return self.send_json({'ok': True})
            if u.path == '/api/load':
                engine()
                return self.send_json({'engine': ENGINE['state'], 'load_s': ENGINE['seconds']})
            return self.send_error(HTTPStatus.NOT_FOUND)
        except (ValueError, KeyError) as e:
            return self.send_json({'error': str(e)}, 400)
        except PermissionError as e:
            return self.send_json({'error': f'not allowed: {e}'}, 403)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return self.send_json({'error': str(e)}, 500)

    def do_DELETE(self):
        u = urlparse(self.path)
        try:
            if u.path.startswith('/api/watchlist/'):
                STORE.watch_del(int(u.path.rsplit('/', 1)[-1]))
                return self.send_json({'ok': True})
            if u.path.startswith('/api/plate/'):
                STORE.delete_plate(int(u.path.rsplit('/', 1)[-1]))
                return self.send_json({'ok': True})
        except ValueError as e:
            return self.send_json({'error': str(e)}, 400)
        return self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self):
        self.send_response(200 if urlparse(self.path).path in ('/', '/index.html') else 404)
        self.end_headers()


def lan_ips():
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
    """Phones allow the camera only on https pages: a self-signed certificate made once with openssl."""
    import subprocess
    d = os.path.join(STORE.folder, 'certs')
    os.makedirs(d, exist_ok=True)
    crt, key, names = os.path.join(d, 'server.crt'), os.path.join(d, 'server.key'), os.path.join(d, 'names.txt')
    san = ','.join(['DNS:localhost', 'IP:127.0.0.1'] + [f'IP:{i}' for i in ips])
    if os.path.exists(crt) and os.path.exists(key) and open(names).read() == san:
        return crt, key
    exe = shutil.which('openssl') or r'C:\Program Files\Git\usr\bin\openssl.exe'
    subprocess.run([exe, 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '825', '-keyout', key, '-out', crt,
                    '-subj', '/CN=BanglaLPR V3 local', '-addext', f'subjectAltName={san}'], check=True, capture_output=True)
    with open(names, 'w') as f:
        f.write(san)
    return crt, key


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=int(os.environ.get('PORT', 8499)))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--phone', action='store_true', help='serve on the Wi-Fi with https so a phone can use its camera')
    p.add_argument('--cpu', action='store_true', help='run the model on the CPU')
    p.add_argument('--preload', action='store_true', help='load the model before serving')
    a = p.parse_args()
    ENGINE['device'] = 'cpu' if a.cpu else None
    if a.preload:
        engine()
    else:
        threading.Thread(target=engine, daemon=True).start()    # load in the background; the page shows progress
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
    print(f'BanglaLPR V3 on {scheme}://{"127.0.0.1" if host == "0.0.0.0" else host}:{a.port}', flush=True)
    print(f'Records (CSV + plate images): {STORE.folder}', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
