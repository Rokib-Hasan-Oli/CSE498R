"""Original-paper check for paper #2 (BLPD, Sarker & Alam): the paper's OWN data, split, weights and external photos.

  python c02_blpd.py prep     --out yaml/                 # finds the BLPD input, writes LPD/LPNR yaml with real paths
  python c02_blpd.py official --name official --project runs
  python c02_blpd.py e2e --lpd runs/lpd_repro/weights/best.pt --lpnr runs/lpnr_repro/weights/best.pt --name e2e_repro ...

Kaggle input: `ferozahmedds/bangladeshi-license-plate-detection-and-recognition` (the authors' GitHub repo: LPD_Dataset
1,502/425/213, LPNR_Dataset 5,058/1,438/721, LPD_Model.pt, LPNR_Model.pt, 150 external photos with the true text).
Without that input the same repo is downloaded from GitHub (pinned commit), so the check never runs on nothing.
official: the authors' two models on their own val + test split (Tables 6, 7, 9) and their cascade on the 150 external
          photos (Table 10: plate found, full-plate exact, char-wise accuracy, Levenshtein similarity, ms per photo).
e2e     : the same external test with OUR retrained LPD + LPNR (trained by det.py on the same split, paper settings).
The vehicle stage is COCO YOLO12l like the paper (conf 0.25, car/motorcycle/bus/truck).
"""
import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import vocab


BLPD_COMMIT = 'd55b11cf66cb34d5a1650823a5e62e35aaf0818c'      # github.com/prosensarker4/BLPD main, 25 Jun 2026
DOWNLOAD = '/tmp/blpd' if os.path.exists('/kaggle') else os.path.expanduser('~/blpd')


def _find():
    hits = []
    for base in ('/kaggle/input', os.environ.get('BLPD_ROOT', ''), DOWNLOAD):
        if base and os.path.isdir(base):
            hits += glob.glob(os.path.join(base, '**', 'LPD_Model.pt'), recursive=True)
    return Path(sorted(hits, key=len)[0]).parent if hits else None


def blpd_root():
    """The BLPD folder (LPD_Model.pt, LPD_Dataset, ...): the attached Kaggle input, $BLPD_ROOT, or else the authors' GitHub
    repo at a pinned commit, downloaded once into /tmp (about 0.6 GB, 1-3 min on Kaggle; same files as the Kaggle copy)."""
    root = _find()
    if root or os.environ.get('LPR_NO_DOWNLOAD'):
        if not root:
            sys.exit('BLPD data not found: Add Input -> Datasets -> '
                     '"ferozahmedds/bangladeshi-license-plate-detection-and-recognition" (or turn Internet on)')
        return root
    import shutil
    import zipfile
    os.makedirs(DOWNLOAD, exist_ok=True)
    z = os.path.join(DOWNLOAD, 'blpd.zip')
    url = f'https://github.com/prosensarker4/BLPD/archive/{BLPD_COMMIT}.zip'
    print('BLPD Kaggle input not attached -> downloading the authors\' repo', url, flush=True)
    r = subprocess.run(['curl', '-sSL', '--retry', '3', '-o', z, url])
    if r.returncode != 0 or not zipfile.is_zipfile(z):
        sys.exit('Could not download BLPD from GitHub. Add Input -> Datasets -> '
                 '"ferozahmedds/bangladeshi-license-plate-detection-and-recognition", or check that Internet is on.')
    with zipfile.ZipFile(z) as f:
        f.extractall(DOWNLOAD)
    os.remove(z)
    shutil.rmtree(os.path.join(DOWNLOAD, '__MACOSX'), ignore_errors=True)
    root = _find()
    if not root:
        sys.exit(f'downloaded BLPD has no LPD_Model.pt (repo changed?): {DOWNLOAD}')
    print('BLPD ready:', root, flush=True)
    return root


def prep(a):
    root = blpd_root()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    res = {}
    for part in ('LPD', 'LPNR'):
        src = (root / f'{part}_data.yaml').read_text(encoding='utf-8')
        names = src[src.index('nc:'):]
        d = root / f'{part}_Dataset'
        f = out / f'{part.lower()}.yaml'
        f.write_text(f'path: {d}\ntrain: train/images\nval: valid/images\ntest: test/images\n{names}\n', encoding='utf-8')
        res[part] = str(f)
        print(part, {s: len(list((d / s / 'images').iterdir())) for s in ('train', 'valid', 'test')}, '->', f)
    common.save_json({'root': str(root), **res}, out / 'blpd_paths.json')


def external(root):
    ext = root / 'External Validated Images'
    rows = list(csv.DictReader(open(ext / 'Easy OCR Predicted' / 'Easy_OCR_Output.csv', encoding='utf-8-sig')))
    return [(ext / 'Input' / r['Original_Image_Name'], r['Original_License_Number (Bangla)'].strip()) for r in rows]


def cascade(lpd_w, lpnr_w, root, a, tag):
    import cv2
    from ultralytics import YOLO
    import p02_cascade as pc
    veh, lpd, lpnr = YOLO(a.vehicle), YOLO(lpd_w), YOLO(lpnr_w)
    rows = external(root)[:a.max]
    ns = argparse.Namespace(lpd_imgsz=640, lpnr_imgsz=640)
    out, preds = {}, []
    for use_vehicle in (True, False):
        found, ms, pr = 0, [], []
        for i, (p, truth) in enumerate(rows):
            img = cv2.imread(str(p))
            (text, ok, xs), f, t, _ = pc.run_one(img, veh, lpd, lpnr, use_vehicle, ns)
            found += int(f)
            if i >= 5:
                ms.append(t)
            pr.append((text, truth, p.name))
        s = metrics.score_rows((x[0], x[1]) for x in pr)
        lev_sim = 100 * np.mean([1 - metrics.lev(vocab.norm_text(x[0]), vocab.norm_text(x[1])) / max(1, len(vocab.norm_text(x[1])))
                                 for x in pr])
        key = 'with_vehicle_stage' if use_vehicle else 'plate_detector_only'
        out[key] = {'photos': len(rows), 'plate_detected_%': round(100 * found / max(1, len(rows)), 2),
                    'full_plate_exact_%': s['plate_exact_%'], 'char_acc_%': s['char_acc_%'],
                    'levenshtein_similarity_%': round(float(lev_sim), 2), 'CER_%': s['CER_%'],
                    'ms_mean': round(float(np.mean(ms)), 1) if ms else None, 'ms_max': round(float(np.max(ms)), 1) if ms else None}
        print(tag, key, out[key], flush=True)
        if use_vehicle:
            preds = pr
    with open(Path(a.project) / a.name / 'external_preds.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['file', 'true', 'pred'])
        w.writerows([(n, t, p) for p, t, n in preds])
    return out


def official(a):
    from ultralytics import YOLO
    from det import val_metrics
    root = blpd_root()
    paths = common.load_json(Path(a.yaml_dir) / 'blpd_paths.json')
    run = Path(a.project) / a.name
    res = {'run': a.name, 'kind': 'blpd-official-weights', 'weights': 'LPD_Model.pt / LPNR_Model.pt (authors, GitHub)'}
    for part, w in (('LPD', root / 'LPD_Model.pt'), ('LPNR', root / 'LPNR_Model.pt')):
        m = YOLO(str(w))
        res[part] = {s: val_metrics(m, paths[part], s, 640, 0 if a.device == '0' else a.device, run / part, batch=16)
                     for s in ('val', 'test')}
        print(part, res[part], flush=True)
    res['external_150'] = cascade(root / 'LPD_Model.pt', root / 'LPNR_Model.pt', root, a, 'official')
    return res


def e2e(a):
    root = blpd_root()
    return {'run': a.name, 'kind': 'blpd-e2e', 'lpd': a.lpd, 'lpnr': a.lpnr,
            'external_150': cascade(a.lpd, a.lpnr, root, a, 'ours')}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prep', 'official', 'e2e'])
    p.add_argument('--out', default='yaml')
    p.add_argument('--yaml-dir', default=os.path.join(common.WORK, 'yaml'))
    p.add_argument('--lpd', default='')
    p.add_argument('--lpnr', default='')
    p.add_argument('--vehicle', default='yolo12l.pt')
    p.add_argument('--device', default='0')
    p.add_argument('--max', type=int, default=None)
    p.add_argument('--name', default='official')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    if a.mode == 'prep':
        prep(a)
    else:
        (Path(a.project) / a.name).mkdir(parents=True, exist_ok=True)
        if a.mode == 'official' and os.environ.get('LPR_LOCAL_TEST'):
            sys.exit("local test: the authors' pickled weights are only loaded on Kaggle")
        r = official(a) if a.mode == 'official' else e2e(a)
        r['env'] = common.env_info()
        common.save_json(r, Path(a.project) / a.name / 'results.json')
        print('DONE', a.name)
