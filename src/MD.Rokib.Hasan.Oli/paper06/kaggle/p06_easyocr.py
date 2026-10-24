"""Paper #6 OCR step: stock EasyOCR (Bangla, `['bn']`, no fine-tuning) as in the paper (p.8), scored on our data.

  python p06_easyocr.py crops --name easyocr_crops --project runs           # EasyOCR alone on the human OCR test crops
  python p06_easyocr.py e2e --det runs/det_v8n/weights/best.pt --name e2e_v8n --project runs   # whole system

crops: every crop of ocr/test.csv (human plate text), two inputs:
         raw   the crop as it is
         prep  4x upscale (Lanczos) + grey + CLAHE (the paper only says "preprocessing", p.8)
       + results per condition (clear / night / blurry / low_res tags), on the real fog copies and the synthetic rain
       copies (ocr/test_fog.csv, test_rain_synth.csv), per source, and EasyOCR ms per plate (batch 1, T4).
e2e  : e2e_test/ whole photos -> our YOLOv8 plate detector (conf 0.25) -> most confident plate, 8 % padding ->
       EasyOCR -> text. A photo with no plate box scores as an empty reading. ms per photo = detector + OCR.
EasyOCR returns each plate line separately; they are joined top line first (the paper never says how).
Scores (metrics.score_rows, after vocab.norm_text): plate exact, CER, char accuracy = 100 - CER (the closest defined
match to the paper's undefined "character recognition accuracy", p.11) and the mean difflib similarity (paper #8's
plate score).
"""
import argparse
import csv
import difflib
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import vocab


class StubReader:
    """Local smoke test only (LPR_LOCAL_TEST, no easyocr installed): reads nothing, so the pipeline still runs."""
    def readtext(self, img, **kw):
        return []


def reader():
    try:
        import easyocr
    except ImportError:
        if os.environ.get('LPR_LOCAL_TEST'):
            print('local test: easyocr is not installed, using a stub reader (all readings empty)', flush=True)
            return StubReader(), 'stub'
        raise
    import torch
    return easyocr.Reader(['bn'], gpu=torch.cuda.is_available(), verbose=False), easyocr.__version__


def read_lines(r, img):
    """EasyOCR boxes -> text in reading order: top line first, left to right inside a line."""
    items = r.readtext(np.array(img), detail=1, paragraph=False)
    if not items:
        return '', 0.0
    boxes = [(np.mean([q[1] for q in b]), min(q[0] for q in b), max(q[1] for q in b) - min(q[1] for q in b), t, c)
             for b, t, c in items]
    boxes.sort(key=lambda z: z[0])
    tol = 0.5 * float(np.median([z[2] for z in boxes]))
    lines = [[boxes[0]]]
    for z in boxes[1:]:
        if abs(z[0] - np.mean([q[0] for q in lines[-1]])) <= tol:
            lines[-1].append(z)
        else:
            lines.append([z])
    text = ' '.join(' '.join(z[3] for z in sorted(l, key=lambda q: q[1])) for l in lines)
    return text, float(np.mean([z[4] for z in boxes]))


def prep(img):
    import cv2
    g = np.array(ImageOps.grayscale(img.resize((img.width * 4, img.height * 4), Image.LANCZOS)))
    return Image.fromarray(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(g)).convert('RGB')


def similarity(rows):
    return round(100 * float(np.mean([difflib.SequenceMatcher(None, vocab.norm_text(t), vocab.norm_text(p or '')).ratio()
                                      for p, t in rows])) if rows else 0.0, 2)


def scores(rows):
    rows = list(rows)
    return dict(metrics.score_rows(rows), **{'similarity_%': similarity(rows)})


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def ocr_rows(r, split, style, max_n):
    root = common.find_dataset()
    rows = list(csv.DictReader(open(os.path.join(root, 'ocr', f'{split}.csv'), encoding='utf-8')))[:max_n]
    out, ms = [], []
    for x in rows:
        img = Image.open(os.path.join(root, 'ocr', 'images', split, x['file'])).convert('RGB')
        img = prep(img) if style == 'prep' else img
        sync(); t0 = time.perf_counter()
        text, conf = read_lines(r, img)
        sync(); ms.append(1000 * (time.perf_counter() - t0))
        out.append({'file': x['file'], 'true': x['text'], 'pred': text, 'conf': round(conf, 4),
                    'valid': metrics.is_valid_plate(text), 'source': x.get('source', '')})
    return out, float(np.mean(ms[5:] or ms or [0]))


def crops(a):
    r, ver = reader()
    root = common.find_dataset()
    res = {'run': a.name, 'kind': 'easyocr-zero-shot', 'labels': 'human plate text',
           'settings': {'model': f'EasyOCR {ver}', 'lang': ['bn'], 'paragraph': False, 'line_join': 'top line first',
                        'prep': '4x Lanczos + grey + CLAHE(2.0, 4x4)'},
           'note': 'stock EasyOCR, no fine-tuning (paper p.8)'}
    preds, ms = ocr_rows(r, 'test', 'raw', a.max)
    pprep, ms_prep = ocr_rows(r, 'test', 'prep', a.max)
    ot = {'rows_decoder': scores((p['pred'], p['true']) for p in preds),
          'prep': dict(scores((p['pred'], p['true']) for p in pprep), ms_per_plate=round(ms_prep, 2)),
          'ms_per_plate': round(ms, 2),
          'valid_format_%': round(100 * float(np.mean([p['valid'] for p in preds])), 2) if preds else 0,
          'by_source': {s: scores((p['pred'], p['true']) for p in preds if p['source'] == s)
                        for s in sorted({p['source'] for p in preds})}}
    # per condition, char accuracy too (EasyOCR's plate-exact numbers are small, char accuracy is the paper's view)
    ot['by_condition'] = metrics.condition_scores(preds, root)
    ot['extra_tests'] = {}
    for sp in metrics.extra_splits(root):
        ep, _ = ocr_rows(r, sp, 'raw', a.max)
        ot['extra_tests'][sp] = scores((p['pred'], p['true']) for p in ep)
    res['ocr_test'] = ot
    run = Path(a.project) / a.name
    for name, rows in (('test', preds), ('test_prep', pprep)):
        with open(run / f'ocr_preds_{name}.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    res['examples'] = [{'true': p['true'], 'pred': p['pred']} for p in preds[:12]]
    print('raw', ot['rows_decoder'], '| prep', ot['prep'], '| ms', ot['ms_per_plate'], flush=True)
    return res


def e2e(a):
    from ultralytics import YOLO
    import cv2
    r, ver = reader()
    det = YOLO(a.det)
    root = common.find_dataset()
    rows = list(csv.DictReader(open(os.path.join(root, 'e2e_test', 'e2e_test.csv'), encoding='utf-8')))[:a.max]
    preds, found, ms = [], 0, []
    for i, x in enumerate(rows):
        bgr = cv2.imread(os.path.join(root, 'e2e_test', 'images', x['file']))
        img = Image.fromarray(bgr[:, :, ::-1])
        sync(); t0 = time.perf_counter()
        pr = det.predict(bgr, imgsz=a.imgsz, conf=0.25, verbose=False)[0]
        text, conf = '', 0.0
        if len(pr.boxes):
            found += 1
            x1, y1, x2, y2 = pr.boxes.xyxy[int(pr.boxes.conf.argmax())].cpu().numpy().tolist()
            pw, ph = 0.08 * (x2 - x1), 0.08 * (y2 - y1)
            c = img.crop((max(0, int(x1 - pw)), max(0, int(y1 - ph)), min(img.width, int(x2 + pw)), min(img.height, int(y2 + ph))))
            text, conf = read_lines(r, c)
        sync(); t = 1000 * (time.perf_counter() - t0)
        if i >= 5:
            ms.append(t)                                # skip warm-up
        preds.append({'file': x['file'], 'true': x['text'], 'pred': text, 'conf': round(conf, 4), 'source': x['source']})
    s = scores((p['pred'], p['true']) for p in preds)
    out = {'photos': len(rows), 'plate_detected_%': round(100 * found / max(1, len(rows)), 2),
           'full_plate_exact_%': s['plate_exact_%'], 'char_acc_%': s['char_acc_%'], 'CER_%': s['CER_%'],
           'similarity_%': s['similarity_%'], 'ms_mean': round(float(np.mean(ms)), 1) if ms else None,
           'ms_max': round(float(np.max(ms)), 1) if ms else None,
           'by_source': {src: scores((p['pred'], p['true']) for p in preds if p['source'] == src)
                         for src in sorted({p['source'] for p in preds})}}
    run = Path(a.project) / a.name
    with open(run / 'e2e_preds.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        w.writeheader(); w.writerows(preds)
    print('e2e', out, flush=True)
    return {'run': a.name, 'kind': 'yolov8+easyocr', 'detector': a.det,
            'settings': {'model': f'{Path(a.det).parent.parent.name} + EasyOCR {ver}', 'lang': ['bn'], 'imgsz': a.imgsz,
                         'det_conf': 0.25, 'crop_pad': 0.08},
            'e2e': out}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['crops', 'e2e'])
    p.add_argument('--det', default='')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--max', type=int, default=None)
    p.add_argument('--name', required=True)
    p.add_argument('--project', required=True)
    a = p.parse_args()
    (Path(a.project) / a.name).mkdir(parents=True, exist_ok=True)
    res = crops(a) if a.mode == 'crops' else e2e(a)
    res['env'] = common.env_info()
    common.save_json(res, Path(a.project) / a.name / 'results.json')
    print('DONE', a.name)
