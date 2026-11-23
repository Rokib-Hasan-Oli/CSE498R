"""Step 3b: system-level detector choice on the detection VAL photos (then reported on TEST).

  python eval/detector_combo.py

For each candidate, boxes at the system's working point (conf 0.25) are matched to the human boxes (IoU >= 0.5):
  found_%   = matched plates / labelled plates          (what the reader can get)
  extra_%   = unmatched boxes / all boxes               (crops that waste reading time or give false reads)
  missed_photos = photos with plates where nothing at all was found
Strategies: each detector alone; V1 detector, and when it finds nothing, a V2 detector on the same photo or on a
CLAHE-enhanced copy (paper #35's low-light finding: enhancing every photo adds false boxes, so only retry on a miss).
Writes eval/results/detector_combo.json.
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from eval.detectors import CANDIDATES  # noqa: E402
from lpr.pipeline import PlateDetector, clahe  # noqa: E402

DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
V1_DET = 'p18_v8n_576'
V2_DETS = ['p28_v9s', 'p27_v10n', 'p27_v10s', 'p06_det_v8n', 'p06_det_v8s', 'p35_det_v8s', 'p21_v9s']


def labels(img_path, w, h):
    f = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'
    f = f.replace('/images/', '/labels/')
    out = []
    if os.path.exists(f):
        for line in open(f):
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                out.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def score(gts, preds):
    n_gt = sum(len(g) for g in gts)
    found = extra = n_box = missed = 0
    for g, p in zip(gts, preds):
        used = set()
        n_box += len(p)
        for gb in g:
            best = max(((iou(gb, pb), j) for j, pb in enumerate(p) if j not in used), default=(0, None))
            if best[0] >= 0.5:
                used.add(best[1]); found += 1
        extra += len(p) - len(used)
        missed += int(bool(g) and not p)
    return {'plates': n_gt, 'found_%': round(100 * found / max(1, n_gt), 2),
            'extra_box_%': round(100 * extra / max(1, n_box), 2), 'missed_photos': missed}


def main():
    out = {}
    for split in ('val', 'test'):
        imgs = sorted(glob.glob(os.path.join(DATA, 'detection', 'images', split, '*.jpg')))
        pics = [cv2.imread(p) for p in imgs]
        gts = [labels(p, im.shape[1], im.shape[0]) for p, im in zip(imgs, pics)]
        preds = {}
        for n in [V1_DET] + V2_DETS:
            file, imgsz, _ = CANDIDATES[n]
            d = PlateDetector(file, imgsz)
            preds[n] = [[b[:4] for b in d.detect(im)] for im in pics]
            miss = [i for i, p in enumerate(preds[n]) if not p]
            preds[n + '+clahe'] = list(preds[n])
            for i in miss:
                preds[n + '+clahe'][i] = [b[:4] for b in d.detect(clahe(pics[i]), conf=0.35)]
            preds[n + '_on_miss_of_v1'] = None
        res = {}
        for n in [V1_DET] + V2_DETS:
            res[n] = score(gts, preds[n])
            res[n + ' +CLAHE retry'] = score(gts, preds[n + '+clahe'])
        for n in V2_DETS:                                   # V1 detector first, V2 detector only on its misses
            comb = [p if p else q for p, q in zip(preds[V1_DET], preds[n])]
            res[f'{V1_DET} -> {n} on miss'] = score(gts, comb)
            comb2 = [p if p else q for p, q in zip(preds[V1_DET], preds[n + '+clahe'])]
            res[f'{V1_DET} -> {n} (+CLAHE) on miss'] = score(gts, comb2)
        out[split] = res
        print(split)
        for k, v in res.items():
            print(f'  {k:42s} {v}')
    json.dump(out, open(os.path.join(HERE, 'results', 'detector_combo.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
