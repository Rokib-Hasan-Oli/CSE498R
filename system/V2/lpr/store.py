"""Local result store: one SQLite file + saved photos/crops under webapp/data/ (nothing leaves the PC).

Tables
  runs      one row per request (photo, video, webcam frame, batch item): version, timings, annotated image
  plates    one row per plate / vehicle event: Bangla + English text, status, readers' texts, crop file
  watchlist plates to alert on (e.g. stolen or wanted vehicles, a demo of the hot-list use case)
Retention: delete_older_than(days) implements a fixed retention period (see docs/IMPACT_AND_INNOVATION.md, privacy).
"""
import json
import os
import sqlite3
import threading
import time

from . import vocab

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, kind TEXT, filename TEXT, version TEXT,
  n_plates INTEGER, ms_total REAL, ms_detect REAL, ms_read REAL, image TEXT, annotated TEXT, extra TEXT);
CREATE TABLE IF NOT EXISTS plates (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id INTEGER, created REAL, version TEXT, text TEXT, text_norm TEXT,
  text_en TEXT, status TEXT, valid INTEGER, conf REAL, det_conf REAL, box TEXT, readers TEXT, crop TEXT,
  track_id INTEGER, frames INTEGER, payload_bytes INTEGER, watch_hit INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS plates_norm ON plates(text_norm);
CREATE INDEX IF NOT EXISTS plates_created ON plates(created);
CREATE TABLE IF NOT EXISTS watchlist (
  id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, text_norm TEXT UNIQUE, note TEXT, created REAL);
"""


def event_payload(plate, version, camera='demo-cam-01'):
    """The compact message an edge camera would send upstream (MQTT/HTTP) for one plate: no image."""
    return json.dumps({'cam': camera, 'ts': round(time.time(), 1), 'v': version, 'plate': plate.get('text', ''),
                       'en': plate.get('text_en', ''), 'st': plate.get('status', ''),
                       'c': plate.get('conf', 0)}, ensure_ascii=False, separators=(',', ':'))


class Store:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(os.path.join(folder, 'media'), exist_ok=True)
        self.db = sqlite3.connect(os.path.join(folder, 'lpr.db'), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def media(self, name):
        return os.path.join(self.folder, 'media', name)

    # ---------------------------------------------------------------- writes
    def add_run(self, kind, filename, version, result, image=None, annotated=None, extra=None):
        ms = result.get('ms', {})
        with self.lock:
            cur = self.db.execute(
                'INSERT INTO runs (created, kind, filename, version, n_plates, ms_total, ms_detect, ms_read, image, '
                'annotated, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (time.time(), kind, filename, version, len(result.get('plates', result.get('events', []))),
                 ms.get('total'), ms.get('detect'), ms.get('read'), image, annotated,
                 json.dumps(extra or {}, ensure_ascii=False)))
            self.db.commit()
            return cur.lastrowid

    def add_plate(self, run_id, version, p, crop=None):
        norm = vocab.norm_text(p.get('text', ''))
        hit = int(bool(norm) and self.db.execute('SELECT 1 FROM watchlist WHERE text_norm=?', (norm,)).fetchone() is not None)
        payload = event_payload(p, version)
        with self.lock:
            cur = self.db.execute(
                'INSERT INTO plates (run_id, created, version, text, text_norm, text_en, status, valid, conf, det_conf, '
                'box, readers, crop, track_id, frames, payload_bytes, watch_hit) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (run_id, time.time(), version, p.get('text', ''), norm, p.get('text_en', ''), p.get('status', ''),
                 int(bool(p.get('valid', False))), p.get('conf'), p.get('det_conf'), json.dumps(p.get('box')),
                 json.dumps(p.get('readers') or p.get('all_reads') or {}, ensure_ascii=False), crop,
                 p.get('track_id'), p.get('frames'), len(payload.encode('utf-8')), hit))
            self.db.commit()
            return cur.lastrowid, hit

    def delete_older_than(self, days):
        cut = time.time() - days * 86400
        with self.lock:
            files = [r[0] for r in self.db.execute('SELECT crop FROM plates WHERE created < ? AND crop IS NOT NULL', (cut,))]
            files += [x for r in self.db.execute('SELECT image, annotated FROM runs WHERE created < ?', (cut,)) for x in r if x]
            n = self.db.execute('DELETE FROM plates WHERE created < ?', (cut,)).rowcount
            self.db.execute('DELETE FROM runs WHERE created < ?', (cut,))
            self.db.commit()
        for f in files:
            try:
                os.remove(self.media(f))
            except OSError:
                pass
        return n

    # ---------------------------------------------------------------- reads
    def plates(self, q='', status='', version='', limit=200, offset=0):
        sql, args = 'SELECT p.*, r.kind, r.filename, r.annotated FROM plates p LEFT JOIN runs r ON r.id = p.run_id WHERE 1=1', []
        if q:
            sql += ' AND (p.text_norm LIKE ? OR p.text_en LIKE ?)'
            args += [f'%{vocab.norm_text(q)}%', f'%{q}%']
        if status:
            sql += ' AND p.status = ?'
            args.append(status)
        if version:
            sql += ' AND p.version = ?'
            args.append(version)
        sql += ' ORDER BY p.id DESC LIMIT ? OFFSET ?'
        args += [limit, offset]
        return [dict(r) for r in self.db.execute(sql, args)]

    def stats(self):
        one = lambda s, a=(): self.db.execute(s, a).fetchone()[0]
        out = {'runs': one('SELECT COUNT(*) FROM runs'), 'plates': one('SELECT COUNT(*) FROM plates'),
               'unique_plates': one("SELECT COUNT(DISTINCT text_norm) FROM plates WHERE valid=1"),
               'watch_hits': one('SELECT COUNT(*) FROM plates WHERE watch_hit=1'),
               'payload_bytes_avg': one('SELECT AVG(payload_bytes) FROM plates'), 'by_version': {}}
        for r in self.db.execute('SELECT version, COUNT(*), AVG(ms_total) FROM runs WHERE ms_total IS NOT NULL GROUP BY version'):
            out['by_version'][r[0]] = {'runs': r[1], 'avg_ms': round(r[2] or 0, 1)}
        out['by_status'] = {r[0]: r[1] for r in self.db.execute('SELECT status, COUNT(*) FROM plates GROUP BY status')}
        out['db_bytes'] = os.path.getsize(os.path.join(self.folder, 'lpr.db'))
        return out

    def watchlist(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM watchlist ORDER BY id DESC')]

    def watch_add(self, text, note=''):
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO watchlist (text, text_norm, note, created) VALUES (?,?,?,?)',
                            (text, vocab.norm_text(text), note, time.time()))
            self.db.execute('UPDATE plates SET watch_hit=1 WHERE text_norm=?', (vocab.norm_text(text),))
            self.db.commit()

    def watch_del(self, wid):
        with self.lock:
            self.db.execute('DELETE FROM watchlist WHERE id=?', (wid,))
            self.db.commit()

    def clear(self):
        with self.lock:
            self.db.execute('DELETE FROM plates')
            self.db.execute('DELETE FROM runs')
            self.db.commit()
        d = os.path.join(self.folder, 'media')
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
