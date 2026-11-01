"""Paper #2 cascade, end to end: COCO YOLO12l finds vehicles -> plate detector (LPD) inside each vehicle -> character
detector (LPNR) inside the plate -> tokens joined into the plate text. Scored on e2e_test/ (whole photos with the
plate text: BLPD's own 150 external-validation photos + NightLPBD test photos).

  python p02_cascade.py --lpd runs/lpd_v12s/weights/best.pt --lpnr runs/lpnr_v12n/weights/best.pt --name cascade_ours
  python p02_cascade.py --official --name cascade_blpd_weights      (the authors' LPD_Model.pt / LPNR_Model.pt)

Reports exact plate %, character-wise accuracy, Levenshtein similarity (paper Table 10), plate detection %, and
latency per photo (mean and max, like Tables 11-13) with and without the vehicle stage.
"""
import argparse
import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import vocab

VEHICLES = {2, 3, 5, 7}   # COCO car, motorcycle, bus, truck
BLPD_RAW = 'https://raw.githubusercontent.com/prosensarker4/BLPD/main/'


def token_name(name):
    """Our models use Bangla tokens; the authors' LPNR uses English names (vocab.BLPD)."""
    if name in vocab.CLASS_ID:
        return name
    return vocab.canon('blpd', name)


def read_plate(lpnr, crop, imgsz):
    r = lpnr.predict(crop, imgsz=imgsz, conf=0.25, verbose=False)[0]
    toks = []
    for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        t = token_name(lpnr.names[int(c)])
        if t:
            toks.append([t] + b.tolist())
    text, ok = vocab.plate_text(toks)
    xs = vocab.tokens_to_text([t[0] for t in sorted(toks, key=lambda t: (t[1] + t[3]) / 2)])
    return text, ok, xs


def crop(img, b, pad):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(w, x2 + pw)), int(min(h, y2 + ph))
    return img[y1:y2, x1:x2], (x1, y1)


def run_one(img, veh, lpd, lpnr, use_vehicle, a):
    t0 = time.perf_counter()
    plates = []                                  # (conf, x1, y1, x2, y2) in photo coordinates
    if use_vehicle:
        r = veh.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
        boxes = [b for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()) if int(c) in VEHICLES]
        for b in boxes:
            vc, (ox, oy) = crop(img, b, 0.05)
            if vc.size == 0:
                continue
            pr = lpd.predict(vc, imgsz=a.lpd_imgsz, conf=0.25, verbose=False)[0]
            for pb, pc in zip(pr.boxes.xyxy.cpu().numpy(), pr.boxes.conf.cpu().numpy()):
                plates.append((float(pc), pb[0] + ox, pb[1] + oy, pb[2] + ox, pb[3] + oy))
    else:
        pr = lpd.predict(img, imgsz=a.lpd_imgsz, conf=0.25, verbose=False)[0]
        plates = [(float(c), *b.tolist()) for b, c in zip(pr.boxes.xyxy.cpu().numpy(), pr.boxes.conf.cpu().numpy())]
    reads = []
    for p in sorted(plates, reverse=True)[:3]:
        pc, _ = crop(img, p[1:], 0.08)
        if pc.size:
            reads.append(read_plate(lpnr, pc, a.lpnr_imgsz))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0)
    valid = [r for r in reads if r[1]]
    best = (valid or reads or [('', False, '')])[0]
    return best, len(plates) > 0, ms, reads


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lpd', default='')
    p.add_argument('--lpnr', default='')
    p.add_argument('--official', action='store_true', help="use the authors' weights from github.com/prosensarker4/BLPD")
    p.add_argument('--vehicle', default='yolo12l.pt')
    p.add_argument('--lpd-imgsz', type=int, default=640)
    p.add_argument('--lpnr-imgsz', type=int, default=640)
    p.add_argument('--name', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--max', type=int, default=None)
    a = p.parse_args()
    import cv2
    from ultralytics import YOLO
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    if a.official and os.environ.get('LPR_LOCAL_TEST'):
        print('local smoke test: the authors\' pickled weights are only loaded on Kaggle; skipped')
        common.save_json({'run': a.name, 'kind': 'cascade', 'skipped': 'local test'}, run / 'results.json')
        return
    if a.official:
        for f in ('LPD_Model.pt', 'LPNR_Model.pt'):
            if not (run / f).exists():
                urllib.request.urlretrieve(BLPD_RAW + f, run / f)
        a.lpd, a.lpnr = str(run / 'LPD_Model.pt'), str(run / 'LPNR_Model.pt')
    veh, lpd, lpnr = YOLO(a.vehicle), YOLO(a.lpd), YOLO(a.lpnr)
    root = common.find_dataset()
    rows = list(csv.DictReader(open(os.path.join(root, 'e2e_test', 'e2e_test.csv'), encoding='utf-8')))[:a.max]
    out = {}
    for use_vehicle in (True, False):
        key = 'with_vehicle_stage' if use_vehicle else 'plate_detector_only'
        preds, found, ms = [], 0, []
        for i, r in enumerate(rows):
            img = cv2.imread(os.path.join(root, 'e2e_test', 'images', r['file']))
            (text, ok, xs), f, t, _ = run_one(img, veh, lpd, lpnr, use_vehicle, a)
            found += int(f)
            if i >= 5:
                ms.append(t)                          # skip warm-up
            preds.append({'file': r['file'], 'true': r['text'], 'pred': text, 'pred_xsort': xs, 'source': r['source']})
        s = metrics.score_rows((x['pred'], x['true']) for x in preds)
        lev_sim = 100 * np.mean([1 - metrics.lev(vocab.norm_text(x['pred']), vocab.norm_text(x['true'])) /
                                 max(1, len(vocab.norm_text(x['true']))) for x in preds])
        out[key] = {'photos': len(rows), 'plate_detected_%': round(100 * found / max(1, len(rows)), 2),
                    'full_plate_exact_%': s['plate_exact_%'], 'char_acc_%': s['char_acc_%'],
                    'levenshtein_similarity_%': round(float(lev_sim), 2), 'token_acc_%': s['token_acc_%'],
                    'xsort_exact_%': metrics.score_rows((x['pred_xsort'], x['true']) for x in preds)['plate_exact_%'],
                    'ms_mean': round(float(np.mean(ms)), 1) if ms else None,
                    'ms_max': round(float(np.max(ms)), 1) if ms else None,
                    'by_source': {src: metrics.score_rows((x['pred'], x['true']) for x in preds if x['source'] == src)
                                  for src in sorted({x['source'] for x in preds})}}
        with open(run / f'e2e_preds_{key}.csv', 'w', newline='', encoding='utf-8') as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=list(preds[0].keys()))
            w.writeheader(); w.writerows(preds)
        print(key, out[key])
    res = {'run': a.name, 'kind': 'cascade', 'lpd': a.lpd, 'lpnr': a.lpnr, 'vehicle': a.vehicle, 'e2e': out,
           'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', 'env': common.env_info()}
    if a.official:                               # the authors' LPNR on our OCR test crops too
        import rec
        preds, ms = [], []
        crows = list(csv.DictReader(open(os.path.join(root, 'ocr', 'test.csv'), encoding='utf-8')))[:a.max]
        for r in crows:
            text, ok, xs = read_plate(lpnr, os.path.join(root, 'ocr', 'images', 'test', r['file']), a.lpnr_imgsz)
            preds.append((text, r['text']))
        res['ocr_test'] = {'rows_decoder': metrics.score_rows(preds)}
    common.save_json(res, run / 'results.json')
    print('DONE', a.name)


if __name__ == '__main__':
    main()
