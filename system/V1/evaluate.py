"""Re-measure this model folder on the 498R All-in-One v1.1 test sets (the numbers in results/summary.json).

  python evaluate.py --data D:/498R/dataset [--cpu]

A. reader part on the 651 OCR test crops: plate exact, character accuracy, status levels
B. whole system on the 204 e2e test photos: plate found, plate exact, character accuracy, FPS (batch 1)
Writes results/evaluate_<gpu|cpu>.json.
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lpr import metrics, vocab  # noqa: E402
from lpr.pipeline import LPRSystem  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.environ.get('DATASET_ROOT', 'D:/498R/dataset'))
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    lpr = LPRSystem(HERE, device='cpu' if a.cpu else None)
    ok = lambda p_, t: vocab.norm_text(p_ or '') == vocab.norm_text(t)
    res = {'version': lpr.version, 'device': 'cpu' if a.cpu else 'gpu'}
    ocr = list(csv.DictReader(open(os.path.join(a.data, 'ocr', 'test.csv'), encoding='utf-8')))
    imgs = [cv2.imread(os.path.join(a.data, 'ocr', 'images', 'test', r['file'])) for r in ocr]
    lpr.read_crops(imgs[:8])
    reads = [x for im in imgs for x in lpr.read_crops([im])]
    lv = {}
    for r, x in zip(reads, ocr):
        s = lv.setdefault(r['status'], [0, 0]); s[0] += 1; s[1] += ok(r['text'], x['text'])
    res['ocr_test_crops'] = {**metrics.score_rows((r['text'], x['text']) for r, x in zip(reads, ocr)),
                             'status': {k: {'plates': n, 'exact_%': round(100 * c / n, 2)} for k, (n, c) in lv.items()}}
    print('test crops', res['ocr_test_crops'], flush=True)
    e2e = list(csv.DictReader(open(os.path.join(a.data, 'e2e_test', 'e2e_test.csv'), encoding='utf-8')))
    photos = [cv2.imread(os.path.join(a.data, 'e2e_test', 'images', r['file'])) for r in e2e]
    for im in photos[:5]:
        lpr.process(im)
    preds, ms, found = [], [], 0
    for im, r in zip(photos, e2e):
        t0 = time.perf_counter(); o = lpr.process(im); ms.append(1000 * (time.perf_counter() - t0))
        found += bool(o['plates'])
        best = next((q for q in o['plates'] if q['valid']), o['plates'][0] if o['plates'] else None)
        preds.append((best['text'] if best else '', r['text']))
    res['e2e_photos'] = {**metrics.score_rows(preds), 'plate_found_%': round(100 * found / len(e2e), 2),
                         'ms_per_photo_mean': round(float(np.mean(ms)), 1), 'fps_mean': round(1000 / float(np.mean(ms)), 1)}
    print('whole photos', res['e2e_photos'], flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'results', f"evaluate_{res['device']}.json"), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
