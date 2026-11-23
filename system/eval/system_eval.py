"""Step 4: test the finished V1 and V2 systems exactly as the web app runs them (lpr.pipeline.LPRSystem).

  python eval/system_eval.py [v1 v2] [--cpu]

A. reader part on the OCR test crops (651): plate exact, char accuracy, status levels, ms per plate
B. whole system on e2e_test (204 whole photos with plate text: 144 BLPD external + 60 NightLPBD night photos):
   plate found, plate exact, char accuracy, by source, ms per photo and FPS (batch 1, after warm-up)
C. --cpu: the same whole-photo run on the CPU only (proposal target: >= 10 FPS on standard hardware)
Also the McNemar test V2 vs V1 on the same plates. Writes eval/results/system_<device>.json.
"""
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from lpr import metrics, vocab  # noqa: E402
from lpr.pipeline import LPRSystem  # noqa: E402

DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')


def rows(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def right(p, t):
    return vocab.norm_text(p or '') == vocab.norm_text(t)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    versions = args or ['v1', 'v2']
    cpu = '--cpu' in sys.argv
    device = 'cpu' if cpu else None
    if cpu:
        import torch
        torch.set_num_threads(int(os.environ.get('LPR_CPU_THREADS', '8')))
    out = {'device': 'cpu' if cpu else 'gpu'}
    ocr = rows(os.path.join(DATA, 'ocr', 'test.csv'))
    e2e = rows(os.path.join(DATA, 'e2e_test', 'e2e_test.csv'))
    ok = {}
    for v in versions:
        s = LPRSystem(v, device=device)
        res = {}
        if not cpu:                                            # A. reader part on the test crops
            imgs = [cv2.imread(os.path.join(DATA, 'ocr', 'images', 'test', r['file'])) for r in ocr]
            s.read_crops(imgs[:8])
            t0 = time.perf_counter()
            reads = []
            for im in imgs:                                    # batch 1, like a camera with one plate
                reads += s.read_crops([im])
            ms = 1000 * (time.perf_counter() - t0) / len(imgs)
            sc = metrics.score_rows((r['text'], x['text']) for r, x in zip(reads, ocr))
            lv = {}
            for r, x in zip(reads, ocr):
                a = lv.setdefault(r['status'], [0, 0])
                a[0] += 1
                a[1] += int(right(r['text'], x['text']))
            res['ocr_test_crops'] = {**sc, 'ms_per_plate': round(ms, 2), 'status': {
                k: {'plates': a, 'share_%': round(100 * a / len(ocr), 1), 'exact_%': round(100 * b / max(1, a), 2)}
                for k, (a, b) in sorted(lv.items())}}
            ok[(v, 'ocr')] = [right(r['text'], x['text']) for r, x in zip(reads, ocr)]
            with open(os.path.join(HERE, 'results', f'ocr_test_preds_{v}.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['file', 'true', 'pred', 'status', 'readers'])
                for r, x in zip(reads, ocr):
                    w.writerow([x['file'], x['text'], r['text'], r['status'], json.dumps(r['readers'], ensure_ascii=False)])
            print(v, 'ocr crops', res['ocr_test_crops'], flush=True)
        # B / C. whole photos
        photos = [cv2.imread(os.path.join(DATA, 'e2e_test', 'images', r['file'])) for r in e2e]
        for im in photos[:5]:
            s.process(im)                                      # warm-up (first call per shape is slow)
        preds, found, ms, retry = [], 0, [], 0
        for im, r in zip(photos, e2e):
            t0 = time.perf_counter()
            o = s.process(im)
            ms.append(1000 * (time.perf_counter() - t0))
            found += int(bool(o['plates']))
            retry += int(bool(o['retry_used']))
            ps = o['plates']
            best = next((p for p in ps if p['valid']), ps[0] if ps else None)   # photo = its first valid plate
            preds.append({'file': r['file'], 'true': r['text'], 'pred': best['text'] if best else '',
                          'status': best['status'] if best else 'none', 'source': r['source']})
        sc = metrics.score_rows((p['pred'], p['true']) for p in preds)
        res['e2e_photos'] = {**sc, 'plate_found_%': round(100 * found / len(e2e), 2), 'retry_used_photos': retry,
                             'ms_per_photo_mean': round(float(np.mean(ms)), 1),
                             'ms_per_photo_median': round(float(np.median(ms)), 1),
                             'fps_mean': round(1000 / float(np.mean(ms)), 1),
                             'by_source': {src: metrics.score_rows((p['pred'], p['true']) for p in preds if p['source'] == src)
                                           for src in sorted({p['source'] for p in preds})}}
        ok[(v, 'e2e')] = [right(p['pred'], p['true']) for p in preds]
        with open(os.path.join(HERE, 'results', f'e2e_preds_{v}_{out["device"]}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(preds[0]))
            w.writeheader()
            w.writerows(preds)
        print(v, 'e2e', {k: x for k, x in res['e2e_photos'].items() if k != 'by_source'}, flush=True)
        out[v] = res
        del s
    if 'v1' in versions and 'v2' in versions:
        for part in ('ocr', 'e2e'):
            if ('v1', part) in ok:
                a, b = ok[('v2', part)], ok[('v1', part)]
                fixed = sum(x and not y for x, y in zip(a, b))
                broken = sum(y and not x for x, y in zip(a, b))
                out[f'v2_vs_v1_{part}'] = {'fixed': fixed, 'broken': broken, 'p_mcnemar': metrics.mcnemar_exact(fixed, broken)}
        print({k: x for k, x in out.items() if k.startswith('v2_vs')})
    json.dump(out, open(os.path.join(HERE, 'results', f'system_{out["device"]}.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)


if __name__ == '__main__':
    main()
