"""Paper #36 (SAHI), original-paper check: the authors' OWN released predictions for every VisDrone row of Table 1,
re-scored with pycocotools against the official VisDrone2019-DET test-dev labels.

  python c36_sahi.py prep --root /kaggle/working/visdrone      (test-dev photos + labels, the authors' 20 result zips)
  python c36_sahi.py eval --root /kaggle/working/visdrone --name visdrone_rescore --project runs

Sources (pinned):
  labels   VisDrone2019-DET-test-dev.zip from the Ultralytics assets mirror (the official VisDrone release; 1,610 photos)
  results  github.com/fcakyon/small-object-detection-benchmark release v0.0.1 (the paper's footnote 1 repo, renamed
           from fcakyon/sahi-benchmark): <detector>_<setting>_visdrone_results.zip, each with result.json (COCO
           detections, conf 0.01) and eval.json (the authors' own score)
Ground truth = the authors' visdrone_to_coco.py: classes 1-10 (pedestrian ... motor) -> 0-9, 'ignored regions' (0)
and 'others' (11) dropped, boxes with zero area dropped. Image ids = c36_visdrone_map.ID_TO_FILE (their converter's
id order, recovered and verified, see that file). Metric = the paper's (p.4): COCO AP at IoU 0.5, all / small /
medium / large, max 500 detections.
xView (Table 2) is not checked: its data needs a registration at xviewdataset.org.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from c36_visdrone_map import ID_TO_FILE

LABELS = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip'
RESULTS = 'https://github.com/fcakyon/small-object-detection-benchmark/releases/download/v0.0.1/{}_visdrone_results.zip'
# Table 1 (p.3): setting -> (AP50, AP50 small, AP50 medium, AP50 large)
TABLE1 = {
    'fcos_fi': (25.8, 14.2, 39.6, 45.1), 'fcos_sahi_po': (29.0, 18.9, 41.5, 46.4), 'fcos_sahi_fi_po': (31.0, 19.8, 44.6, 49.0),
    'fcos_sf_sahi_po': (38.1, 25.7, 54.8, 56.9), 'fcos_sf_sahi_fi_po': (38.5, 25.9, 55.4, 59.8),
    'vfnet_fi': (28.8, 16.8, 44.0, 47.5), 'vfnet_sahi_po': (32.0, 21.4, 45.8, 45.5), 'vfnet_sahi_fi_po': (33.9, 22.4, 49.1, 49.4),
    'vfnet_sf_sahi_po': (41.9, 29.7, 58.8, 60.6), 'vfnet_sf_sahi_fi_po': (42.2, 29.6, 59.2, 63.3),
    'tood_fi': (29.4, 18.1, 44.1, 50.0), 'tood_sahi': (31.9, 22.6, 44.0, 45.2), 'tood_sahi_po': (32.5, 22.8, 45.2, 43.6),
    'tood_sahi_fi': (34.6, 23.8, 48.5, 53.1), 'tood_sahi_fi_po': (34.7, 23.8, 48.9, 50.3), 'tood_sf_fi': (36.8, 24.4, 53.8, 66.4),
    'tood_sf_sahi': (42.5, 31.6, 58.0, 61.1), 'tood_sf_sahi_po': (43.1, 31.7, 59.0, 60.2), 'tood_sf_sahi_fi': (43.4, 31.7, 59.6, 65.6),
    'tood_sf_sahi_fi_po': (43.5, 31.7, 59.8, 65.4)}
ABSTRACT = {'fcos': (6.8, 12.7), 'vfnet': (5.1, 13.4), 'tood': (5.3, 14.5)}      # "SAHI +x AP", "with SF cumulative +y" (p.1)


def fetch(url, dst):
    if not dst.exists():
        print('download', url, flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=300) as r, open(dst, 'wb') as f:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                f.write(b)
    return dst


def prep(a):
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'VisDrone2019-DET-test-dev' / 'annotations').exists():
        with zipfile.ZipFile(fetch(LABELS, root / 'testdev.zip')) as z:
            z.extractall(root)
        (root / 'testdev.zip').unlink()
    for s in (list(TABLE1) if not a.max else list(TABLE1)[:a.max]):
        d = root / 'results' / s
        if (d / 'result.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        zp = fetch(RESULTS.format(s), root / f'{s}.zip')
        with zipfile.ZipFile(zp) as z:
            for n in ('result.json', 'eval.json'):
                (d / n).write_bytes(z.read(n))
        zp.unlink()
    print('test-dev photos:', len(list((root / 'VisDrone2019-DET-test-dev' / 'images').glob('*.jpg'))),
          '| result files:', len(list((root / 'results').glob('*/result.json'))))


def ground_truth(root):
    from PIL import Image
    from pycocotools.coco import COCO
    base = Path(root) / 'VisDrone2019-DET-test-dev'
    images, anns = [], []
    for iid, fn in sorted(ID_TO_FILE.items()):
        w, h = Image.open(base / 'images' / fn).size
        images.append({'id': iid, 'width': w, 'height': h, 'file_name': fn})
        for l in (base / 'annotations' / fn.replace('.jpg', '.txt')).read_text().splitlines():
            r = l.split(',')
            if len(r) > 5 and 1 <= int(r[5]) <= 10:
                bb = [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
                if bb[2] * bb[3] > 0:
                    anns.append({'id': len(anns) + 1, 'image_id': iid, 'category_id': int(r[5]) - 1, 'bbox': bb,
                                 'area': bb[2] * bb[3], 'iscrowd': 0})
    gt = COCO()
    gt.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': k, 'name': str(k)} for k in range(10)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
    return gt


def ap50(gt, dets):
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        e = COCOeval(gt, dt, 'bbox')
        e.params.maxDets = [1, 10, 500]
        e.evaluate(); e.accumulate()
    p = e.eval['precision'][0, :, :, :, -1]                     # IoU 0.5, recall points, classes, areas, maxDets 500
    return [round(100 * float(np.mean(p[:, :, k][p[:, :, k] > -1])), 1) for k in range(4)]


def evaluate(a):
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    gt = ground_truth(a.root)
    out = {'run': a.name, 'kind': 'sahi-check', 'photos': len(gt.imgs), 'boxes': len(gt.anns), 'settings': {}}
    for s, paper in TABLE1.items():
        d = Path(a.root) / 'results' / s
        if not (d / 'result.json').exists():
            continue
        res = json.loads((d / 'result.json').read_text())
        ev = json.loads((d / 'eval.json').read_text())
        ours = ap50(gt, res)
        own = [round(100 * ev[k], 1) for k in ('bbox_mAP50', 'bbox_mAP50_s', 'bbox_mAP50_m', 'bbox_mAP50_l')]
        out['settings'][s] = {'paper': dict(zip(('AP50', 'AP50_s', 'AP50_m', 'AP50_l'), paper)),
                              'authors_eval_json': dict(zip(('AP50', 'AP50_s', 'AP50_m', 'AP50_l'), own)),
                              'rescored': dict(zip(('AP50', 'AP50_s', 'AP50_m', 'AP50_l'), ours)),
                              'rescored_minus_paper_AP50_pp': round(ours[0] - paper[0], 1), 'detections': len(res)}
        print(s, 'paper', paper, '| their eval.json', own, '| re-scored', ours, flush=True)
    st = out['settings']
    gains = {}
    for det_, (sahi_claim, sf_claim) in ABSTRACT.items():
        fi = st.get(f'{det_}_fi')
        best_sahi = max((v['rescored']['AP50'] for k, v in st.items() if k.startswith(det_ + '_sahi')), default=None)
        best_all = max((v['rescored']['AP50'] for k, v in st.items() if k.startswith(det_ + '_')), default=None)
        if fi and best_sahi is not None:
            gains[det_] = {'abstract_sahi_gain': sahi_claim, 'table_sahi_gain': round(best_sahi - fi['rescored']['AP50'], 1),
                           'abstract_sf_gain': sf_claim, 'table_sf_gain': round(best_all - fi['rescored']['AP50'], 1)}
    out['abstract_vs_table'] = gains
    out['max_abs_diff_rescored_vs_paper_AP50'] = max((abs(v['rescored_minus_paper_AP50_pp']) for v in st.values()), default=None)
    out['env'] = common.env_info()
    common.save_json(out, run / 'results.json')
    print('abstract vs table:', json.dumps(gains))
    print('DONE', a.name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prep', 'eval'])
    p.add_argument('--root', default=os.path.join(common.WORK, 'visdrone'))
    p.add_argument('--max', type=int, default=None, help='prep: only the first N result files (quick test)')
    p.add_argument('--name', default='visdrone_rescore')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    prep(a) if a.mode == 'prep' else evaluate(a)
