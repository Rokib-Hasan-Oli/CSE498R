"""Step 3 of the V1/V2 build: score every candidate plate detector on the detection VAL split (choice) and the
TEST split (report), with the image size each was trained at.

  python eval/detectors.py [names ...]

Writes eval/results/detectors.json: mAP50, mAP50-95, precision, recall per split, and GPU ms per photo (batch 1,
model step, median of 100 test photos after warm-up). Version rule: highest val mAP50; a tie (<= 0.2 points)
goes to the faster model.
"""
import glob
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
W = os.path.join(HERE, '..', 'models', 'weights')
CANDIDATES = {
    # ---- V1 papers
    'p02_lpd_v12s': ('p02_lpd_v12s.pt', 640, 2),
    'p18_v8n_576': ('p18_v8n_576.pt', 576, 18),
    'p03_v8n_416': ('p03_v8n_416.pt', 416, 3),
    'p03_v8n_640': ('p03_v8n_640.pt', 640, 3),
    'p11_v8n_320': ('p11_v8n_320.pt', 320, 11),
    'p10_v8n_640': ('p10_v8n_640.pt', 640, 10),
    # ---- V2 papers
    'p28_v9s': ('p28_v9s.pt', 640, 28),
    'p28_v9t': ('p28_v9t.pt', 640, 28),
    'p27_v10n': ('p27_v10n.pt', 640, 27),
    'p27_v10s': ('p27_v10s.pt', 640, 27),
    'p21_v9s': ('p21_v9s.pt', 640, 21),
    'p06_det_v8n': ('p06_det_v8n.pt', 640, 6),
    'p06_det_v8s': ('p06_det_v8s.pt', 640, 6),
    'p35_det_v8s': ('p35_det_v8s.pt', 640, 35),
}


def main():
    from ultralytics import YOLO
    names = sys.argv[1:] or list(CANDIDATES)
    yaml = os.path.join(tempfile.gettempdir(), 'banglalpr_det.yaml')
    with open(yaml, 'w', encoding='utf-8') as f:
        f.write(f"path: {DATA}/detection\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: plate\n")
    resf = os.path.join(HERE, 'results', 'detectors.json')
    res = json.load(open(resf, encoding='utf-8')) if os.path.exists(resf) else {}
    test_imgs = sorted(glob.glob(os.path.join(DATA, 'detection', 'images', 'test', '*.jpg')))[:110]
    for n in names:
        file, imgsz, paper = CANDIDATES[n]
        m = YOLO(os.path.join(W, file))
        e = {'file': file, 'imgsz': imgsz, 'paper': paper}
        for split in ('val', 'test'):
            r = m.val(data=yaml, split=split, imgsz=imgsz, batch=16, workers=0, plots=False, verbose=False,
                      project=os.path.join(tempfile.gettempdir(), 'banglalpr_val'), name=n, exist_ok=True)
            e[split] = {'mAP50': round(100 * r.box.map50, 2), 'mAP50_95': round(100 * r.box.map, 2),
                        'P': round(100 * r.box.mp, 2), 'R': round(100 * r.box.mr, 2)}
        ms = []
        for p in test_imgs:
            t0 = time.perf_counter()
            m.predict(p, imgsz=imgsz, conf=0.25, verbose=False)
            ms.append(1000 * (time.perf_counter() - t0))
        e['gpu_ms_photo_batch1'] = round(float(np.median(ms[10:])), 2)
        e['params_M'] = round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2)
        res[n] = e
        print(n, e, flush=True)
        json.dump(res, open(resf, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
