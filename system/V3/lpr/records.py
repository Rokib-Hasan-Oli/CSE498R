"""V3 records: every plate is saved as a plate image + one CSV row that names that image (nothing leaves the PC).

records/
  plates.csv                      one row per plate, newest last; the "Plate image" column is the image's path
  plate_images/<date>/<no>_<plate>.jpg   the cut-out licence plate of that row
  photos/<date>/run<no>_<kind>.jpg       the whole photo with the plate boxed (web app only)
  videos/                         annotated videos (web app only)
  exports/export_<time>/          a bundle made on request: results.csv + plate_images/ + results.html (+ .zip)

The CSV starts with a UTF-8 BOM so Excel shows the Bangla text. Image paths in the CSV are relative to the CSV's
own folder, so the folder can be copied or zipped and the links still work; one extra column holds the full path.
"""
import csv
import html
import io
import os
import re
import shutil
import threading
import time
import zipfile

import cv2

COLS = ['No', 'Date', 'Time', 'Plate (Bangla)', 'Plate (English)', 'Status', 'BRTA layout OK', 'Read confidence',
        'Detector confidence', 'Source', 'File name', 'Plate image', 'Plate image (full path)', 'Vehicle photo',
        'Watchlist match', 'Model', 'Night mode']
STATUS_WORDS = {'verified': 'Verified (readers agree)', 'likely': 'Likely (check if important)',
                'review': 'Check manually'}


def slug(text_en, n=40):
    s = re.sub(r'[^A-Za-z0-9]+', '-', text_en or '').strip('-').upper()
    return (s or 'UNREADABLE')[:n]


def day(ts):
    return time.strftime('%Y-%m-%d', time.localtime(ts))


def write_jpg(path, img, q=92):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])   # imencode + write: works with any path
    if not ok:
        raise ValueError('could not encode image')
    with open(path, 'wb') as f:
        f.write(buf.tobytes())


def _fmt(x, nd=3):
    return '' if x is None else (round(float(x), nd) if isinstance(x, (int, float)) else x)


class PlateLog:
    """The folder copy of the results: plate images + plates.csv. Used by the web app and by predict.py."""

    def __init__(self, folder, model='v3'):
        self.folder = os.path.abspath(folder)
        self.model = model
        self.csv_path = os.path.join(self.folder, 'plates.csv')
        self.lock = threading.Lock()
        os.makedirs(os.path.join(self.folder, 'plate_images'), exist_ok=True)
        if not os.path.exists(self.csv_path):
            self.rewrite([])

    def header_outdated(self):
        try:
            with open(self.csv_path, encoding='utf-8-sig', newline='') as f:
                return next(csv.reader(f), None) != COLS
        except OSError:
            return False

    def path(self, rel):
        return os.path.join(self.folder, rel.replace('/', os.sep))

    def next_no(self):
        """Next row number when there is no database (predict.py): rows already in the CSV + 1."""
        with open(self.csv_path, encoding='utf-8-sig') as f:
            return sum(1 for _ in csv.reader(f))         # header counts as the +1

    # ------------------------------------------------------------------ images
    def save_plate_image(self, crop, no, text_en, ts):
        rel = f'plate_images/{day(ts)}/{int(no):06d}_{slug(text_en)}.jpg'
        write_jpg(self.path(rel), crop, 95)
        return rel

    def save_photo(self, img, run_no, kind, ts):
        rel = f'photos/{day(ts)}/run{int(run_no):06d}_{kind}.jpg'
        write_jpg(self.path(rel), img, 85)
        return rel

    # ------------------------------------------------------------------ CSV
    def row(self, r, base=None):
        """r: plate record (DB row or dict with the same keys). base: folder the image paths are relative to."""
        ts = r.get('created') or time.time()
        crop = r.get('crop') or ''
        full = self.path(crop) if crop else ''
        rel = crop if base is None else (os.path.relpath(full, base).replace(os.sep, '/') if crop else '')
        photo = r.get('annotated') or ''
        if photo and base is not None:
            photo = os.path.relpath(self.path(photo), base).replace(os.sep, '/')
        return [r.get('id'), day(ts), time.strftime('%H:%M:%S', time.localtime(ts)), r.get('text') or '',
                r.get('text_en') or '', STATUS_WORDS.get(r.get('status'), r.get('status') or ''),
                'yes' if r.get('valid') else 'no', _fmt(r.get('conf')), _fmt(r.get('det_conf')),
                r.get('kind') or '', r.get('filename') or '', rel, full, photo,
                'YES' if r.get('watch_hit') else '', (r.get('version') or self.model).upper(),
                'on' if r.get('night') else 'off']

    def append(self, r):
        with self.lock, open(self.csv_path, 'a', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(self.row(r))

    def rewrite(self, rows):
        """rows oldest first. Used after deletions so the CSV always matches the stored plates."""
        with self.lock:
            tmp = self.csv_path + '.tmp'
            with open(tmp, 'w', encoding='utf-8-sig', newline='') as f:
                w = csv.writer(f)
                w.writerow(COLS)
                for r in rows:
                    w.writerow(self.row(r))
            os.replace(tmp, self.csv_path)

    # ------------------------------------------------------------------ export bundle
    def export(self, rows, title='BanglaLPR V3 results'):
        """Copy the given plates (newest first is fine) into exports/export_<time>/ with results.csv,
        plate_images/ and results.html (a printable sheet that shows each plate image), and zip it."""
        stamp = time.strftime('%Y%m%d_%H%M%S')
        name = f'export_{stamp}'
        dest = os.path.join(self.folder, 'exports', name)
        img_dir = os.path.join(dest, 'plate_images')
        os.makedirs(img_dir, exist_ok=True)
        rows = sorted(rows, key=lambda r: r.get('id') or 0)
        out_rows = []
        for r in rows:
            r = dict(r)
            src = self.path(r['crop']) if r.get('crop') else None
            if src and os.path.exists(src):
                dst = os.path.join(img_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                r['_export_img'] = 'plate_images/' + os.path.basename(src)
            else:
                r['_export_img'] = ''
            out_rows.append(r)
        with open(os.path.join(dest, 'results.csv'), 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for r in out_rows:
                row = self.row(r)
                row[COLS.index('Plate image')] = r['_export_img']
                row[COLS.index('Plate image (full path)')] = os.path.join(dest, r['_export_img'].replace('/', os.sep)) if r['_export_img'] else ''
                row[COLS.index('Vehicle photo')] = ''          # photos stay in records/photos (not copied)
                w.writerow(row)
        with open(os.path.join(dest, 'results.html'), 'w', encoding='utf-8') as f:
            f.write(sheet_html(out_rows, title, self))
        zpath = os.path.join(self.folder, 'exports', name + '.zip')
        with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(dest):
                for fn in files:
                    full = os.path.join(root, fn)
                    z.write(full, os.path.join(name, os.path.relpath(full, dest)))
        return {'name': name, 'folder': dest, 'zip': zpath, 'plates': len(out_rows),
                'images': sum(1 for r in out_rows if r['_export_img'])}


def sheet_html(rows, title, log):
    """A printable page: one line per plate with its picture (open results.html in any browser, or print to PDF)."""
    e = html.escape
    trs = io.StringIO()
    for r in rows:
        st = r.get('status') or ''
        img = f'<img src="{e(r["_export_img"])}" alt="">' if r.get('_export_img') else '<span class="none">no image</span>'
        cells = log.row(r)
        trs.write(f'<tr class="{"hit" if r.get("watch_hit") else ""}"><td>{e(str(cells[0]))}</td><td>{img}</td>'
                  f'<td class="bn">{e(cells[3]) or "—"}</td><td class="mono">{e(cells[4])}</td>'
                  f'<td><span class="st {e(st)}">{e(cells[5])}</span>{"<b class=w>WATCHLIST</b>" if r.get("watch_hit") else ""}</td>'
                  f'<td>{e(cells[1])}<br><small>{e(cells[2])}</small></td><td>{e(cells[9])}<br><small>{e(cells[10])}</small></td></tr>\n')
    made = time.strftime('%d %b %Y, %H:%M')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(title)}</title><style>
body{{font:14px/1.45 'Segoe UI',system-ui,sans-serif;color:#17201b;margin:24px;background:#fff}}
h1{{font-size:20px;margin:0 0 4px}} p.sub{{color:#5d6b63;margin:0 0 16px}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #dfe5df;padding:8px 10px;text-align:left;vertical-align:middle}}
th{{font-size:12px;text-transform:uppercase;letter-spacing:.4px;color:#5d6b63;background:#f6f8f6}}
td img{{height:56px;max-width:220px;border:1px solid #ccd;border-radius:4px;display:block}}
.bn{{font-family:'Hind Siliguri','Nirmala UI','Vrinda',sans-serif;font-size:19px;font-weight:700;white-space:nowrap}}
.mono{{font-family:Consolas,monospace;font-size:13px;white-space:nowrap}} td:nth-child(6){{white-space:nowrap}} small{{color:#5d6b63}} .none{{color:#999}}
.st{{font-size:12px;font-weight:600;padding:2px 8px;border-radius:99px;background:#eef1ee;white-space:nowrap}}
.st.verified{{background:#e3f1ec;color:#006a4e}} .st.likely{{background:#fff1da;color:#8a5200}} .st.review{{background:#fbe6e9;color:#b01c30}}
b.w{{display:inline-block;margin-left:6px;background:#d3243a;color:#fff;font-size:11px;padding:1px 6px;border-radius:4px}}
tr.hit td{{background:#fbe6e9}} tr{{page-break-inside:avoid}}
</style></head><body><h1>{e(title)}</h1><p class="sub">{len(rows)} plates · exported {e(made)} · images in plate_images/ next to this file · same rows as results.csv</p>
<table><thead><tr><th>No</th><th>Plate image</th><th>Plate (Bangla)</th><th>English</th><th>Status</th><th>Date</th><th>Source</th></tr></thead>
<tbody>{trs.getvalue()}</tbody></table></body></html>"""
