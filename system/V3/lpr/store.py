"""V3 result store: SQLite index (search, stats, watchlist) + the records folder (plate images + plates.csv).

Everything lives in V3/records/ (see lpr/records.py). The CSV is appended on every new plate and rewritten after
deletions, so plates.csv, plate_images/ and the database always hold the same plates. Nothing leaves the PC.
"""
import json
import os
import sqlite3
import threading
import time

from . import vocab
from .records import PlateLog

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


def event_payload(plate, version, camera='cam-01'):
    """The compact message an edge camera would send upstream for one plate: text only, no image."""
    return json.dumps({'cam': camera, 'ts': round(time.time(), 1), 'v': version, 'plate': plate.get('text', ''),
                       'en': plate.get('text_en', ''), 'st': plate.get('status', ''),
                       'c': plate.get('conf', 0)}, ensure_ascii=False, separators=(',', ':'))


class Store:
    def __init__(self, folder, version='v3'):
        self.folder = os.path.abspath(folder)
        self.version = version
        self.log = PlateLog(self.folder, version)
        self.db = sqlite3.connect(os.path.join(self.folder, 'lpr.db'), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r[1] for r in self.db.execute('PRAGMA table_info(plates)')}
        if 'night' not in cols:                          # V3 night mode (22 Nov 2026): older databases get the column
            self.db.execute('ALTER TABLE plates ADD COLUMN night INTEGER DEFAULT 0')
            self.db.commit()
        self.lock = threading.Lock()
        if self.log.header_outdated():                   # plates.csv written before a column was added
            self.log.rewrite(list(reversed(self.plates(limit=10 ** 9))))

    def path(self, rel):
        return self.log.path(rel)

    # ---------------------------------------------------------------- writes
    def add_run(self, kind, filename, result, annotated_img=None, annotated_rel=None, extra=None):
        ms = result.get('ms', {})
        now = time.time()
        with self.lock:
            cur = self.db.execute(
                'INSERT INTO runs (created, kind, filename, version, n_plates, ms_total, ms_detect, ms_read, image, '
                'annotated, extra) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (now, kind, filename, self.version, len(result.get('plates', result.get('events', []))),
                 ms.get('total'), ms.get('detect'), ms.get('read'), None, annotated_rel,
                 json.dumps(extra or {}, ensure_ascii=False)))
            run_id = cur.lastrowid
            if annotated_img is not None:
                annotated_rel = self.log.save_photo(annotated_img, run_id, kind, now)
                self.db.execute('UPDATE runs SET annotated=? WHERE id=?', (annotated_rel, run_id))
            self.db.commit()
            return run_id, annotated_rel

    def add_plate(self, run_id, p, crop_img=None, kind='', filename='', annotated=None):
        """Store one plate: DB row, plate image file, CSV row. Returns the stored record."""
        norm = vocab.norm_text(p.get('text', ''))
        hit = int(bool(norm) and self.db.execute('SELECT 1 FROM watchlist WHERE text_norm=?', (norm,)).fetchone() is not None)
        payload = event_payload(p, self.version)
        now = time.time()
        with self.lock:
            cur = self.db.execute(
                'INSERT INTO plates (run_id, created, version, text, text_norm, text_en, status, valid, conf, det_conf, '
                'box, readers, crop, track_id, frames, payload_bytes, watch_hit, night) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (run_id, now, self.version, p.get('text', ''), norm, p.get('text_en', ''), p.get('status', ''),
                 int(bool(p.get('valid', False))), p.get('conf'), p.get('det_conf'), json.dumps(p.get('box')),
                 json.dumps(p.get('readers') or p.get('all_reads') or {}, ensure_ascii=False), None,
                 p.get('track_id'), p.get('frames'), len(payload.encode('utf-8')), hit, int(bool(p.get('night')))))
            pid = cur.lastrowid
            crop = None
            if crop_img is not None and getattr(crop_img, 'size', 0):
                crop = self.log.save_plate_image(crop_img, pid, p.get('text_en', ''), now)
                self.db.execute('UPDATE plates SET crop=? WHERE id=?', (crop, pid))
            self.db.commit()
        rec = {'id': pid, 'created': now, 'version': self.version, 'text': p.get('text', ''),
               'text_en': p.get('text_en', ''), 'status': p.get('status', ''), 'valid': p.get('valid'),
               'conf': p.get('conf'), 'det_conf': p.get('det_conf'), 'kind': kind, 'filename': filename,
               'crop': crop, 'annotated': annotated, 'watch_hit': hit, 'night': int(bool(p.get('night')))}
        self.log.append(rec)
        return rec

    def _remove_files(self, rels):
        for f in rels:
            if f:
                try:
                    os.remove(self.path(f))
                except OSError:
                    pass

    def sync_csv(self):
        self.log.rewrite(list(reversed(self.plates(limit=10 ** 9))))

    def delete_plate(self, pid):
        with self.lock:
            r = self.db.execute('SELECT crop FROM plates WHERE id=?', (pid,)).fetchone()
            self.db.execute('DELETE FROM plates WHERE id=?', (pid,))
            self.db.commit()
        if r:
            self._remove_files([r[0]])
        self.sync_csv()

    def delete_older_than(self, days):
        cut = time.time() - days * 86400
        with self.lock:
            files = [r[0] for r in self.db.execute('SELECT crop FROM plates WHERE created < ?', (cut,))]
            files += [r[0] for r in self.db.execute('SELECT annotated FROM runs WHERE created < ?', (cut,))]
            n = self.db.execute('DELETE FROM plates WHERE created < ?', (cut,)).rowcount
            self.db.execute('DELETE FROM runs WHERE created < ?', (cut,))
            self.db.commit()
        self._remove_files(files)
        self.sync_csv()
        return n

    def clear(self):
        with self.lock:
            files = [r[0] for r in self.db.execute('SELECT crop FROM plates')]
            files += [r[0] for r in self.db.execute('SELECT annotated FROM runs')]
            self.db.execute('DELETE FROM plates')
            self.db.execute('DELETE FROM runs')
            self.db.commit()
        self._remove_files(files)
        self.sync_csv()

    # ---------------------------------------------------------------- reads
    def plates(self, q='', status='', limit=200, offset=0, since=None):
        sql = ('SELECT p.*, r.kind, r.filename, r.annotated FROM plates p LEFT JOIN runs r ON r.id = p.run_id '
               'WHERE 1=1')
        args = []
        if q:
            sql += ' AND (p.text_norm LIKE ? OR p.text_en LIKE ?)'
            args += [f'%{vocab.norm_text(q)}%', f'%{q}%']
        if status:
            sql += ' AND p.status = ?'
            args.append(status)
        if since:
            sql += ' AND p.created >= ?'
            args.append(since)
        sql += ' ORDER BY p.id DESC LIMIT ? OFFSET ?'
        args += [limit, offset]
        return [dict(r) for r in self.db.execute(sql, args)]

    def stats(self):
        one = lambda s, a=(): self.db.execute(s, a).fetchone()[0]
        today = time.mktime(time.strptime(time.strftime('%Y-%m-%d'), '%Y-%m-%d'))
        out = {'runs': one('SELECT COUNT(*) FROM runs'), 'plates': one('SELECT COUNT(*) FROM plates'),
               'today': one('SELECT COUNT(*) FROM plates WHERE created >= ?', (today,)),
               'unique_plates': one("SELECT COUNT(DISTINCT text_norm) FROM plates WHERE valid=1"),
               'watch_hits': one('SELECT COUNT(*) FROM plates WHERE watch_hit=1'),
               'images': one('SELECT COUNT(*) FROM plates WHERE crop IS NOT NULL'),
               'avg_ms': one('SELECT AVG(ms_total) FROM runs WHERE ms_total IS NOT NULL')}
        out['by_status'] = {r[0]: r[1] for r in self.db.execute('SELECT status, COUNT(*) FROM plates GROUP BY status')}
        size = 0
        for root, _, files in os.walk(self.folder):
            if os.sep + 'exports' in root:
                continue
            size += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        out['folder_bytes'] = size
        out['folder'] = self.folder
        out['csv'] = self.log.csv_path
        return out

    def watchlist(self):
        return [dict(r) for r in self.db.execute('SELECT * FROM watchlist ORDER BY id DESC')]

    def watch_add(self, text, note=''):
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO watchlist (text, text_norm, note, created) VALUES (?,?,?,?)',
                            (text, vocab.norm_text(text), note, time.time()))
            self.db.execute('UPDATE plates SET watch_hit=1 WHERE text_norm=?', (vocab.norm_text(text),))
            self.db.commit()
        self.sync_csv()

    def watch_del(self, wid):
        with self.lock:
            r = self.db.execute('SELECT text_norm FROM watchlist WHERE id=?', (wid,)).fetchone()
            self.db.execute('DELETE FROM watchlist WHERE id=?', (wid,))
            if r:
                self.db.execute('UPDATE plates SET watch_hit=0 WHERE text_norm=?', (r[0],))
            self.db.commit()
        self.sync_csv()

    def export(self, q='', status='', since=None):
        return self.log.export(self.plates(q, status, limit=10 ** 9, since=since))
