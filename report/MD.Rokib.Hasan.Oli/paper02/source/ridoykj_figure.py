"""Picture of the box-style mismatch: label box (green) vs our detector's box (orange) on ridoykj and KUET test photos,
plus the test mAP with ridoykj left out. Run with the 498R venv python:

  D:/498R/_dataset_build/venv/Scripts/python.exe ridoykj_figure.py <best.pt> <dataset root> <work dir>
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
weights, root, work = sys.argv[1:4]
from ultralytics import YOLO  # noqa: E402

det = os.path.join(root, 'detection')
model = YOLO(weights)
GREEN, ORANGE = (0, 131, 0), (52, 104, 235)          # BGR of #008300 and #eb6834


def label_boxes(img_path, w, h):
    lp = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'
    a = np.loadtxt(lp, ndmin=2) if os.path.exists(lp) else np.zeros((0, 5))
    return [((x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h) for _, x, y, bw, bh in a]


def tile(src, k):
    files = sorted(glob.glob(os.path.join(det, 'images', 'test', f'{src}_*.jpg')))
    files = files[::max(1, len(files) // (3 * k))]
    out = []
    for p in files:
        if len(out) == k:
            break
        im = cv2.imread(p)
        h, w = im.shape[:2]
        gts = label_boxes(p, w, h)
        r = model.predict(p, imgsz=640, conf=0.25, verbose=False, device=0)[0]
        prs = r.boxes.xyxy.cpu().numpy().tolist()
        if not gts:
            continue
        g = gts[0]
        cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        half = max(g[2] - g[0], (g[3] - g[1]) * 2.2) * 1.0
        x1, y1 = int(max(0, cx - half)), int(max(0, cy - half * 0.55))
        x2, y2 = int(min(w, cx + half)), int(min(h, cy + half * 0.55))
        t = max(2, int((x2 - x1) / 160))
        for b in prs:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), ORANGE, t)
        for b in gts:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREEN, t)
        crop = cv2.resize(im[y1:y2, x1:x2], (360, 198))
        out.append(crop)
    return out


rows = []
for src in ('ridoykj', 'kuet'):
    t = tile(src, 4)
    rows.append(np.hstack([np.pad(c, ((4, 4), (4, 4), (0, 0)), constant_values=255) for c in t]))
cv2.imwrite(os.path.join(HERE, 'fig_boxstyle.png'), np.vstack(rows))

# test mAP without ridoykj
files = [p for p in sorted(glob.glob(os.path.join(det, 'images', 'test', '*.jpg')))
         if not os.path.basename(p).startswith('ridoykj_')]
os.makedirs(work, exist_ok=True)
lst = os.path.join(work, 'test_no_ridoykj.txt')
open(lst, 'w').write('\n'.join(files))
yml = os.path.join(work, 'test_no_ridoykj.yaml')
open(yml, 'w').write(f'path: {det}\ntrain: {lst}\nval: {lst}\ntest: {lst}\nnames:\n  0: plate\n')
m = model.val(data=yml, split='test', imgsz=640, batch=8, workers=0, verbose=False, plots=False, device=0)
yml_all = os.path.join(work, 'test_all.yaml')
open(yml_all, 'w').write(f'path: {det}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n  0: plate\n')
m_all = model.val(data=yml_all, split='test', imgsz=640, batch=8, workers=0, verbose=False,
                  plots=False, device=0)
res = {'without_ridoykj': {'photos': len(files), 'mAP50': round(100 * float(m.box.map50), 1),
                           'mAP50-95': round(100 * float(m.box.map), 1)},
       'all_local': {'mAP50': round(100 * float(m_all.box.map50), 1), 'mAP50-95': round(100 * float(m_all.box.map), 1)}}
print(res)
json.dump(res, open(os.path.join(HERE, 'no_ridoykj.json'), 'w'), indent=1)
