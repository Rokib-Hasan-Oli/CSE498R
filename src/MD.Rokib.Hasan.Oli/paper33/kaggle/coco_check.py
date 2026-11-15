"""Original-paper check for the detector papers (#26 YOLOv12, #27 YOLOv10, #28 YOLOv9, #30 RT-DETR): the authors'
COCO weights on COCO val2017 (the papers' own test data) and latency on a T4 with TensorRT FP16 (the papers' own
hardware and runtime).

  python coco_check.py prep --root /kaggle/working/datasets
  python coco_check.py eval --models yolov10n.pt yolov10s.pt --name coco_v10 --project runs [--rtdetr] [--small]

prep: val2017 images (images.cocodataset.org, 5,000 photos) + Ultralytics' YOLO-format labels, which also carry the
      official annotations/instances_val2017.json, so the mAP is computed by pycocotools exactly as in the papers.
eval: for each model: mAP50-95 / mAP50 (pycocotools, conf 0.001, 640 px), then batch-1 latency on 200 val photos:
      PyTorch FP16 and TensorRT FP16, "inference only" (the papers' number) and end to end (pre + inference + NMS).
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

LABELS = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip'
VAL = 'http://images.cocodataset.org/zips/val2017.zip'


def fetch(url, dst):
    if not dst.exists():
        print('download', url, flush=True)
        urllib.request.urlretrieve(url, dst)
    return dst


def prep(a):
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    coco = root / 'coco'
    if not (coco / 'annotations' / 'instances_val2017.json').exists():
        with zipfile.ZipFile(fetch(LABELS, root / 'coco2017labels.zip')) as z:
            z.extractall(root)
    if len(list((coco / 'images' / 'val2017').glob('*.jpg'))) < 5000:      # the labels zip makes an empty folder
        (coco / 'images').mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(fetch(VAL, root / 'val2017.zip')) as z:
            z.extractall(coco / 'images')
    import yaml
    from ultralytics.utils import ROOT
    names = yaml.safe_load(open(ROOT / 'cfg' / 'datasets' / 'coco.yaml', encoding='utf-8'))['names']
    y = root / 'coco_val.yaml'
    y.write_text(yaml.safe_dump({'path': str(coco), 'train': 'val2017.txt', 'val': 'val2017.txt', 'names': names},
                                sort_keys=False))
    lines = (coco / 'val2017.txt').read_text().splitlines()[:200]          # quick-test subset
    (coco / 'val2017_small.txt').write_text('\n'.join(lines) + '\n')
    (root / 'coco_val_small.yaml').write_text(yaml.safe_dump({'path': str(coco), 'train': 'val2017_small.txt',
                                                              'val': 'val2017_small.txt', 'names': names}, sort_keys=False))
    print('val2017 photos:', len(os.listdir(coco / 'images' / 'val2017')), '->', y)


def latency(model, imgs, kw, n=200, warm=20):
    for f in imgs[:warm]:
        model.predict(f, imgsz=640, verbose=False, **kw)
    inf, e2e = [], []
    for f in imgs[:n]:
        r = model.predict(f, imgsz=640, verbose=False, **kw)[0]
        inf.append(r.speed['inference']); e2e.append(sum(r.speed.values()))
    return {'inference_ms': round(float(np.median(inf)), 2), 'end_to_end_ms': round(float(np.median(e2e)), 2),
            'fps_inference': round(1000 / float(np.median(inf)), 1)}


def cocoeval(root, pred_json):
    """Official COCO mAP with pycocotools on the saved predictions (the numbers the papers report)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    anno = COCO(str(Path(root) / 'coco' / 'annotations' / 'instances_val2017.json'))
    pred = anno.loadRes(str(pred_json))
    e = COCOeval(anno, pred, 'bbox')
    e.params.imgIds = sorted({d['image_id'] for d in json.load(open(pred_json))})
    e.evaluate(); e.accumulate(); e.summarize()
    return {'AP_%': round(100 * e.stats[0], 2), 'AP50_%': round(100 * e.stats[1], 2), 'AP_small_%': round(100 * e.stats[3], 2),
            'AP_medium_%': round(100 * e.stats[4], 2), 'AP_large_%': round(100 * e.stats[5], 2)}


def evaluate(a):
    import torch
    from ultralytics import RTDETR, YOLO
    from det import legacy_onnx_export, prec
    load = RTDETR if a.rtdetr else YOLO
    yml = Path(a.root) / ('coco_val_small.yaml' if a.small else 'coco_val.yaml')
    imgs = sorted(str(p) for p in (Path(a.root) / 'coco' / 'images' / 'val2017').iterdir())
    res = {'run': a.name, 'kind': 'coco-check', 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
           'models': {}}
    for w in a.models:
        m = load(w)
        vdir = (Path(a.project) / a.name / Path(w).stem).resolve()       # absolute: Ultralytics nests relative paths
        v = m.val(data=str(yml), split='val', imgsz=640, batch=32, conf=0.001, iou=0.7, save_json=not a.small,
                  plots=False, workers=common.WORKERS, project=str(vdir.parent), name=vdir.name, exist_ok=True,
                  verbose=False)
        r = {'ultralytics_mAP50-95_%': round(100 * float(v.box.map), 2), 'ultralytics_mAP50_%': round(100 * float(v.box.map50), 2)}
        pj = Path(getattr(v, 'save_dir', vdir)) / 'predictions.json'
        if not a.small and pj.exists():
            try:
                r['coco'] = cocoeval(a.root, pj)
            except Exception as e:
                r['coco'] = f'failed: {e}'[:200]
        try:
            r['params_M'] = round(sum(p.numel() for p in m.model.parameters()) / 1e6, 2)
        except Exception:
            pass
        r['pytorch_fp16'] = latency(load(w), imgs, prec(16), n=a.n)
        if a.trt and torch.cuda.is_available():
            try:
                legacy_onnx_export()
                eng = load(w).export(format='engine', imgsz=640, **prec(16))
                r['tensorrt_fp16'] = latency(load(eng), imgs, {}, n=a.n)
            except Exception as e:
                r['tensorrt_fp16'] = f'failed: {e}'[:300]
        res['models'][Path(w).stem] = r
        print(Path(w).stem, json.dumps(r), flush=True)
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prep', 'eval'])
    p.add_argument('--root', default=os.path.join(common.WORK, 'datasets'))
    p.add_argument('--models', nargs='+', default=[])
    p.add_argument('--rtdetr', action='store_true')
    p.add_argument('--no-trt', dest='trt', action='store_false')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--small', action='store_true', help='quick test: 200 val photos, Ultralytics mAP only')
    p.add_argument('--name', default='coco')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    if a.mode == 'prep':
        prep(a)
    else:
        (Path(a.project) / a.name).mkdir(parents=True, exist_ok=True)
        r = evaluate(a)
        r['env'] = common.env_info()
        common.save_json(r, Path(a.project) / a.name / 'results.json')
        print('DONE', a.name)
