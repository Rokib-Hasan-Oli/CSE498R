"""Local checks of the paper #6 v1.1 run (venv python, local GPU, one job at a time).

  D:/498R/_dataset_build/venv/Scripts/python.exe local_check.py <run dir> <paper #2 upload dir> [det] [e2e]

<run dir>  = the run's results.zip unpacked (code/, runs/det_v8n|det_v8s/weights/best.pt, runs/e2e_*/e2e_preds.csv)
<p02 dir>  = weights/paper02/MD.Rokib.Hasan.Oli/ (best_lpd_v12s.pt, best_lpnr_v12n.pt from the paper #2 v1.1 run)

det: both #6 detectors and paper #2's plate detector (lpd_v12s, same v1.1 split) on the detection test split:
     Ultralytics mAP50 / mAP50-95 on the whole test (must equal the Kaggle numbers), per source and without ridoykj;
     one line per human plate (found at conf 0.25 / IoU 0.5 with the kit's greedy matching, best IoU, box ratios)
     -> det_per_plate.csv, so the detectors can be compared on the same plates (McNemar).
e2e: the 204 e2e_test photos read by paper #2's character reader (lpnr_v12n) instead of EasyOCR:
     p06 protocol  = the kit's #6 crop (most confident box, 8 % padding), one read -> isolates the reader;
     p02 cascade   = paper #2's cascade code, plate-detector-only mode, with the LPD swapped -> isolates the detector.
     -> e2e_cross_preds.csv.
Writes local_check.json next to this file (parts not run keep their old values).
"""
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN, P02 = sys.argv[1], sys.argv[2]
PARTS = set(sys.argv[3:]) or {'det', 'e2e'}
DATA = os.environ.setdefault('DATASET_ROOT', r'D:\498R\dataset')
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')
sys.path.insert(0, os.path.join(RUN, 'code'))
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

import metrics  # noqa: E402
import vocab  # noqa: E402

WORK = os.path.join(os.environ.get('TEMP', HERE), 'p06_local_check')
os.makedirs(WORK, exist_ok=True)
OUT = os.path.join(HERE, 'local_check.json')
res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
DET = {'det_v8n': os.path.join(RUN, 'runs', 'det_v8n', 'weights', 'best.pt'),
       'det_v8s': os.path.join(RUN, 'runs', 'det_v8s', 'weights', 'best.pt'),
       'p02_lpd_v12s': os.path.join(P02, 'best_lpd_v12s.pt')}
LPNR = os.path.join(P02, 'best_lpnr_v12n.pt')
DEV = 0 if torch.cuda.is_available() else 'cpu'
res['env'] = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
              'dataset': json.load(open(os.path.join(DATA, 'dataset_manifest.json'), encoding='utf-8'))['version']}


def iou_matrix(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def gt(img_path, w, h):
    lp = img_path.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).rsplit('.', 1)[0] + '.txt'
    rows = [l.split() for l in open(lp).read().splitlines() if l.strip()] if os.path.exists(lp) else []
    b = np.array([[float(v) for v in r[1:5]] for r in rows]).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
                     (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h], 1) if len(b) else np.zeros((0, 4))


def yaml_for(name, files):
    lst = os.path.join(WORK, f'{name}.txt')
    open(lst, 'w').write('\n'.join(files))
    y = os.path.join(WORK, f'{name}.yaml')
    open(y, 'w').write(f'path: {os.path.join(DATA, "detection")}\ntrain: {lst}\nval: {lst}\ntest: {lst}\nnames:\n  0: plate\n')
    return y


def val(model, y):
    m = model.val(data=y, split='test', imgsz=640, batch=16, workers=0, verbose=False, plots=False, device=DEV)
    return {'P': round(100 * float(m.box.mp), 2), 'R': round(100 * float(m.box.mr), 2),
            'mAP50': round(100 * float(m.box.map50), 2), 'mAP50-95': round(100 * float(m.box.map), 2)}


def part_det():
    tdir = os.path.join(DATA, 'detection', 'images', 'test')
    imgs = sorted(os.path.join(tdir, f) for f in os.listdir(tdir) if f.endswith('.jpg'))
    by_src = defaultdict(list)
    for p in imgs:
        by_src[os.path.basename(p).split('_')[0]].append(p)
    out, rows = {}, []
    for name, w in DET.items():
        model = YOLO(w)
        r = {'test': val(model, yaml_for('test_all', imgs)),
             'test_without_ridoykj': val(model, yaml_for('test_no_ridoykj', [p for p in imgs if 'ridoykj_' not in p])),
             'by_source': {}}
        for src, files in sorted(by_src.items()):
            r['by_source'][src] = val(model, yaml_for(f'test_{src}', files))
        tp = n = 0
        for s in range(0, len(imgs), 32):               # the kit's plate_counts: conf 0.25, greedy by confidence, IoU 0.5
            for res_ in model.predict(imgs[s:s + 32], imgsz=640, conf=0.25, device=DEV, verbose=False):
                h, wd = res_.orig_shape
                g = gt(res_.path, wd, h)
                p = res_.boxes.xyxy.cpu().numpy()
                cf = res_.boxes.conf.cpu().numpy()
                order = np.argsort(-cf)
                m = iou_matrix(p[order], g)
                used = {}
                for i in range(len(order)):
                    j = int(np.argmax(m[i])) if m.shape[1] else -1
                    if j >= 0 and m[i, j] >= 0.5 and j not in used:
                        used[j] = i
                for j, gb in enumerate(g):
                    best = float(m[:, j].max()) if m.shape[0] else 0.0
                    k = used.get(j)
                    pb = p[order][k] if k is not None else None
                    rows.append({'model': name, 'file': os.path.basename(res_.path), 'plate': j,
                                 'source': os.path.basename(res_.path).split('_')[0],
                                 'gt_w': round(gb[2] - gb[0], 1), 'gt_h': round(gb[3] - gb[1], 1),
                                 'found': int(k is not None), 'best_iou': round(best, 4),
                                 'conf': round(float(cf[order][k]), 4) if k is not None else '',
                                 'w_ratio': round((pb[2] - pb[0]) / (gb[2] - gb[0]), 4) if pb is not None else '',
                                 'h_ratio': round((pb[3] - pb[1]) / (gb[3] - gb[1]), 4) if pb is not None else ''})
                tp += len(used); n += len(g)
        r['plates_found_%'] = round(100 * tp / max(1, n), 2)
        r['plates'] = n
        out[name] = r
        print(name, r['test'], r['plates_found_%'], flush=True)
    with open(os.path.join(HERE, 'det_per_plate.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    res['det'] = out


def part_e2e():
    import cv2
    sys.path.insert(0, os.path.join(HERE, 'p02_code'))
    import p02_cascade as cas                          # paper #2's kit code (identical lprkit files)
    rows = list(csv.DictReader(open(os.path.join(DATA, 'e2e_test', 'e2e_test.csv'), encoding='utf-8')))
    lpnr = YOLO(LPNR)

    class A:
        lpd_imgsz = 640
        lpnr_imgsz = 640
    out, preds = {}, []
    for name in ('det_v8n', 'det_v8s', 'p02_lpd_v12s'):
        det = YOLO(DET[name])
        for proto in ('p06_protocol', 'p02_cascade'):
            if name == 'p02_lpd_v12s' and proto == 'p06_protocol':
                continue
            pp, found, ms = [], 0, []
            for i, r in enumerate(rows):
                img = cv2.imread(os.path.join(DATA, 'e2e_test', 'images', r['file']))
                t0 = time.perf_counter()
                if proto == 'p02_cascade':
                    (text, ok, _), f, _, _ = cas.run_one(img, None, det, lpnr, False, A)
                else:
                    pr = det.predict(img, imgsz=640, conf=0.25, verbose=False, device=DEV)[0]
                    f, text = len(pr.boxes) > 0, ''
                    if f:
                        b = pr.boxes.xyxy[int(pr.boxes.conf.argmax())].cpu().numpy().tolist()
                        c, _ = cas.crop(img, b, 0.08)
                        text = cas.read_plate(lpnr, c, 640)[0] if c.size else ''
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                if i >= 5:
                    ms.append(1000 * (time.perf_counter() - t0))
                found += int(f)
                pp.append({'run': f'{name}+lpnr_v12n/{proto}', 'file': r['file'], 'true': r['text'], 'pred': text,
                           'source': r['source']})
            s = metrics.score_rows((x['pred'], x['true']) for x in pp)
            key = f'{name}+lpnr_v12n/{proto}'
            out[key] = {'photos': len(rows), 'plate_detected_%': round(100 * found / len(rows), 2),
                        'plate_exact_%': s['plate_exact_%'], 'char_acc_%': s['char_acc_%'],
                        'ms_mean_local': round(float(np.mean(ms)), 1),
                        'by_source': {src: metrics.score_rows((x['pred'], x['true']) for x in pp if x['source'] == src)
                                      ['plate_exact_%'] for src in sorted({x['source'] for x in pp})}}
            preds += pp
            print(key, out[key], flush=True)
    with open(os.path.join(HERE, 'e2e_cross_preds.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
        wr.writeheader(); wr.writerows(preds)
    res['e2e_cross'] = out


if __name__ == '__main__':
    if 'det' in PARTS:
        part_det()
    if 'e2e' in PARTS:
        part_e2e()
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('wrote', OUT)
