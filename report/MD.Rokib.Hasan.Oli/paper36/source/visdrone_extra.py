"""Paper #36 report: the paper's own data next to ours (CPU, the original check's VisDrone folder).

  venv python visdrone_extra.py <c36 visdrone root (VisDrone2019-DET-test-dev + results/<setting>/result.json)>

1. Object size: width as a share of the photo width (the paper's own "small object" yardstick, p.3) and COCO area
   class, for the 75 k VisDrone test-dev objects and for our 1,011 test plates.
2. The authors' TOOD / FCOS / VFNet predictions at a working threshold (score >= 0.3, IoU 0.5, class-aware, COCO
   matching with max 500 boxes): boxes, true positives, precision, recall for FI and the SAHI settings. Shows whether
   slicing also costs precision on their data and how much recall it buys there.
Writes visdrone_extra.json.
"""
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

import sahi_common as C

sys.path.insert(0, str(C.KIT))
import c36_sahi  # noqa: E402  (the original check: ground_truth with the authors' id order)

ROOT = Path(sys.argv[1])
SETTINGS = ['tood_fi', 'tood_sahi_po', 'tood_sahi_fi_po', 'tood_sf_fi', 'tood_sf_sahi_fi_po',
            'fcos_fi', 'fcos_sahi_fi_po', 'vfnet_fi', 'vfnet_sahi_fi_po']


def working_point(gt, dets, thr=0.3):
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        e = COCOeval(gt, dt, 'bbox')
        e.params.maxDets = [500]
        e.params.iouThrs = np.array([0.5])
        e.params.areaRng = [e.params.areaRng[0]]
        e.params.areaRngLbl = ['all']
        e.evaluate()
    tp = fp = n_gt = 0
    for r in e.evalImgs:
        if r is None:
            continue
        gi = np.asarray(r['gtIgnore']).astype(bool)
        n_gt += int((~gi).sum())
        s = np.asarray(r['dtScores'])
        m = np.asarray(r['dtMatches'])[0]
        ig = np.asarray(r['dtIgnore'])[0].astype(bool)
        keep = (s >= thr) & ~ig
        tp += int(((m > 0) & keep).sum())
        fp += int(((m == 0) & keep).sum())
    return {'boxes': tp + fp, 'tp': tp, 'fp': fp, 'precision_%': round(100 * tp / max(1, tp + fp), 2),
            'recall_%': round(100 * tp / max(1, n_gt), 2), 'objects': n_gt}


def main():
    gt = c36_sahi.ground_truth(ROOT)
    W = {i: im['width'] for i, im in gt.imgs.items()}
    share = np.array([a['bbox'][2] / W[a['image_id']] for a in gt.anns.values()])
    area = np.array([a['area'] for a in gt.anns.values()])
    out = {'visdrone': {'objects': len(share), 'photos': len(gt.imgs),
                        'width_share_%': {q: round(100 * float(np.percentile(share, q)), 2) for q in (10, 25, 50, 75, 90)},
                        'width_lt_1pct_%': round(100 * float(np.mean(share < 0.01)), 1),
                        'width_lt_2pct_%': round(100 * float(np.mean(share < 0.02)), 1),
                        'coco_small_%': round(100 * float(np.mean(area < 32 ** 2)), 1),
                        'coco_medium_%': round(100 * float(np.mean((area >= 32 ** 2) & (area < 96 ** 2))), 1),
                        'coco_large_%': round(100 * float(np.mean(area >= 96 ** 2)), 1),
                        'hist_share': np.histogram(np.log10(np.clip(share, 1e-4, 1)), bins=40, range=(-3.5, 0))[0].tolist()}}
    sh, ar = [], []
    for f, g, (w, h) in C.test_photos():
        for b in g:
            sh.append((b[2] - b[0]) / w)
            ar.append((b[2] - b[0]) * (b[3] - b[1]))
    sh, ar = np.array(sh), np.array(ar)
    out['ours'] = {'objects': len(sh), 'width_share_%': {q: round(100 * float(np.percentile(sh, q)), 2) for q in (10, 25, 50, 75, 90)},
                   'width_lt_1pct_%': round(100 * float(np.mean(sh < 0.01)), 1),
                   'width_lt_2pct_%': round(100 * float(np.mean(sh < 0.02)), 1),
                   'coco_small_%': round(100 * float(np.mean(ar < 32 ** 2)), 1),
                   'coco_medium_%': round(100 * float(np.mean((ar >= 32 ** 2) & (ar < 96 ** 2))), 1),
                   'coco_large_%': round(100 * float(np.mean(ar >= 96 ** 2)), 1),
                   'hist_share': np.histogram(np.log10(np.clip(sh, 1e-4, 1)), bins=40, range=(-3.5, 0))[0].tolist()}
    out['objects_per_photo'] = {'visdrone': round(len(share) / len(gt.imgs), 1)}
    out['working_point'] = {}
    for s in SETTINGS:
        res = json.loads((ROOT / 'results' / s / 'result.json').read_text())
        out['working_point'][s] = working_point(gt, res)
        print(s, out['working_point'][s], flush=True)
    (C.HERE / 'visdrone_extra.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k != 'working_point'}, indent=0)[:1500])


if __name__ == '__main__':
    main()
