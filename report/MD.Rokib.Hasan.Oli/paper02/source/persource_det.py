"""Why is LPD test mAP50-95 only 46.2? Score the run's best.pt per test source and compare box shapes with the labels.

  D:/498R/_dataset_build/venv/Scripts/python.exe persource_det.py <best.pt> <dataset root> <work dir>

For every source of detection/images/test: Ultralytics mAP50 / mAP50-95, then (conf 0.25) each label box is matched to
the best predicted box and we record IoU and the predicted/label width and height ratios. A box-style mismatch shows up
as ratios far from 1 for one source. Writes persource_det.json next to this file.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
weights, root, work = sys.argv[1:4]
os.makedirs(work, exist_ok=True)
from ultralytics import YOLO  # noqa: E402

det = os.path.join(root, 'detection')
imgs = sorted(glob.glob(os.path.join(det, 'images', 'test', '*.jpg')))
by_src = defaultdict(list)
for p in imgs:
    by_src[os.path.basename(p).split('_')[0]].append(p)

model = YOLO(weights)
out = {}


def labels(img):
    lp = img.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'
    if not os.path.exists(lp):
        return np.zeros((0, 4))
    a = np.loadtxt(lp, ndmin=2)
    return a[:, 1:5] if len(a) else np.zeros((0, 4))


def iou(a, b):
    """a, b: xywh normalised."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-12)


for src, files in sorted(by_src.items()):
    lst = os.path.join(work, f'test_{src}.txt')
    with open(lst, 'w') as f:
        f.write('\n'.join(files))
    yml = os.path.join(work, f'test_{src}.yaml')
    with open(yml, 'w') as f:
        f.write(f'path: {det}\ntrain: {lst}\nval: {lst}\ntest: {lst}\nnames:\n  0: plate\n')
    m = model.val(data=yml, split='test', imgsz=640, batch=8, workers=0, verbose=False, plots=False, device=0)
    ious, wr, hr = [], [], []
    for s in range(0, len(files), 16):
        for p, r in zip(files[s:s + 16], model.predict(files[s:s + 16], imgsz=640, conf=0.25, verbose=False, device=0)):
            pred = r.boxes.xywhn.cpu().numpy()
            for g in labels(p):
                if not len(pred):
                    continue
                k = int(np.argmax([iou(g, q) for q in pred]))
                v = iou(g, pred[k])
                if v >= 0.5:
                    ious.append(v)
                    wr.append(pred[k][2] / g[2])
                    hr.append(pred[k][3] / g[3])
    out[src] = {'photos': len(files), 'mAP50': round(100 * float(m.box.map50), 1),
                'mAP50-95': round(100 * float(m.box.map), 1), 'matched_plates': len(ious),
                'mean_iou': round(100 * float(np.mean(ious)), 1) if ious else None,
                'median_w_ratio': round(float(np.median(wr)), 3) if wr else None,
                'median_h_ratio': round(float(np.median(hr)), 3) if hr else None}
    print(src, out[src], flush=True)

with open(os.path.join(HERE, 'persource_det.json'), 'w') as f:
    json.dump(out, f, indent=1)
