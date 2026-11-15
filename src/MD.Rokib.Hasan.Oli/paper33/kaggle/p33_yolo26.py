"""Paper #33 (YOLO26): the TWO heads of one trained YOLO26 model on our plate test split, and how they quantise.

  python p33_yolo26.py --weights runs/y26s/weights/best.pt --data det.yaml --name heads_y26s --project runs [--trt]

A trained YOLO26 keeps two detection heads (Ultralytics 8.4.170):
  one2many : the classic head; its boxes need NMS. Used when `nms` is not set (Ultralytics' default here).
  one2one  : the NMS-free head (`nms=False`), the paper's "end-to-end (e2e)" column (Table 2, p.8).
For each head: test P / R / mAP50 / mAP50-95, plates found at conf 0.25, recall by plate size, false alarms on
plate-free photos, batch-1 latency (pre + inference + post) with PyTorch FP16 (every photo warmed up once, median),
TensorRT FP16 and INT8 engines (--trt; INT8 also scored on the test split, the paper's claim that YOLO26 keeps its
accuracy under INT8, p.10) and ONNX on the CPU (50 photos).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import det

HEADS = {'one2many_nms': {}, 'one2one_e2e': {'nms': False}}


class Head:
    """A YOLO model whose predict / val always use one head (det.py helpers call model.predict / model.val)."""
    def __init__(self, model, kw):
        self.m, self.kw = model, kw

    def predict(self, *a, **k):
        return self.m.predict(*a, **{**self.kw, **k})

    def val(self, *a, **k):
        return self.m.val(*a, **{**self.kw, **k})


def latency(model, imgs, kw, dev, imgsz):
    """median ms per photo, batch 1, pre + inference + post; every photo is run once untimed first (new letterbox
    shapes are slow on the first FP16 call, see det.measure_speed)."""
    for f in imgs:
        model.predict(f, imgsz=imgsz, device=dev, verbose=False, **kw)
    ms, post = [], []
    for f in imgs:
        r = model.predict(f, imgsz=imgsz, device=dev, verbose=False, **kw)[0]
        ms.append(sum(r.speed.values())); post.append(r.speed.get('postprocess', 0.0))
    med = float(np.median(ms))
    return {'ms_median': round(med, 2), 'fps_median': round(1000 / med, 1), 'ms_mean': round(float(np.mean(ms)), 2),
            'postprocess_ms_median': round(float(np.median(post)), 2), 'images': len(imgs)}


def no_plate_rate(model, data, imgsz, dev, max_images=None):
    import json
    import yaml
    root = Path(yaml.safe_load(open(data, encoding='utf-8'))['path'])
    cfile = root / 'conditions.json'
    if not cfile.exists():
        return None
    cond = json.loads(cfile.read_text(encoding='utf-8'))
    neg = [str(root / l.strip()[2:]) for l in open(root / cond['real']['no_plate']['list'], encoding='utf-8') if l.strip()]
    neg = neg[:max_images]
    hits = 0
    for s in range(0, len(neg), 32):
        hits += sum(len(r.boxes) > 0 for r in model.predict(neg[s:s + 32], imgsz=imgsz, conf=0.25, device=dev, verbose=False))
    return {'photos': len(neg), 'false_alarm_%': round(100 * hits / max(1, len(neg)), 2)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--trt', action='store_true', help='also TensorRT FP16 + INT8 engines of both heads')
    p.add_argument('--n', type=int, default=200, help='photos for the latency test')
    p.add_argument('--max', type=int, default=None, help='quick test: score only N test photos')
    a = p.parse_args()
    from ultralytics import YOLO
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    dev = '0' if torch.cuda.is_available() else 'cpu'
    imgs = [str(x) for x in det.split_files(a.data, 'test')[0][:a.n if not a.max else min(a.n, a.max)]]
    out = {'run': a.name, 'kind': 'yolo26-heads', 'weights': a.weights, 'heads': {},
           'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    for head, kw in HEADS.items():
        m = Head(YOLO(a.weights), kw)
        h = {'test': det.val_metrics(m, a.data, 'test', a.imgsz, dev, run / head),
             'test_plates': det.plate_counts(m, a.data, 'test', a.imgsz, dev, max_images=a.max),
             'no_plate': no_plate_rate(m, a.data, a.imgsz, dev, a.max)}
        if torch.cuda.is_available():
            h['pytorch_fp16'] = latency(YOLO(a.weights), imgs, {**det.prec(16), **kw}, dev, a.imgsz)
        out['heads'][head] = h
        print(head, h['test'], h['test_plates']['found_%'], h.get('pytorch_fp16'), flush=True)
    if a.trt and torch.cuda.is_available():
        det.legacy_onnx_export()
        for head, kw in HEADS.items():
            nms_kw = {'nms': False} if kw else {}
            for prec_name, pk in (('tensorrt_fp16', det.prec(16)),
                                  ('tensorrt_int8', {'quantize': 8} if 'quantize' in det.prec(16) else {'int8': True})):
                try:
                    extra = dict(data=a.data, fraction=0.3) if prec_name.endswith('int8') else {}
                    eng = YOLO(a.weights).export(format='engine', imgsz=a.imgsz, device=dev, **pk, **nms_kw, **extra)
                    e = YOLO(eng)
                    r = latency(e, imgs, {}, dev, a.imgsz)
                    v = e.val(data=a.data, split='test', imgsz=a.imgsz, batch=1, device=dev, workers=common.WORKERS,
                              verbose=False, plots=False, project=str(run / head), name=f'eval_{prec_name}', exist_ok=True)
                    r.update({'test_mAP50': round(float(v.box.map50), 4), 'test_mAP50-95': round(float(v.box.map), 4)})
                    os.replace(eng, run / f'{head}_{prec_name}.engine') if os.path.exists(eng) else None
                except Exception as ex:
                    r = f'failed: {ex}'[:300]
                out['heads'][head][prec_name] = r
                print(head, prec_name, r, flush=True)
    try:                                                        # CPU: ONNX of each head (NMS-free graph vs raw + NMS)
        det.legacy_onnx_export()
        for head, kw in HEADS.items():
            onnx = YOLO(a.weights).export(format='onnx', imgsz=a.imgsz, simplify=False, dynamic=False,
                                          **({'nms': False} if kw else {}))
            out['heads'][head]['cpu_onnx'] = latency(YOLO(onnx), imgs[:50], {}, 'cpu', a.imgsz)
    except Exception as ex:
        out['cpu_onnx'] = f'failed: {ex}'[:300]
    o, e = out['heads']['one2many_nms'], out['heads']['one2one_e2e']
    out['e2e_minus_nms_mAP50-95_pp'] = round(100 * (e['test']['mAP50-95'] - o['test']['mAP50-95']), 2)
    if 'pytorch_fp16' in o:
        out['e2e_minus_nms_ms_fp16'] = round(e['pytorch_fp16']['ms_median'] - o['pytorch_fp16']['ms_median'], 2)
    out['env'] = common.env_info()
    common.save_json(out, run / 'results.json')
    print('DONE', a.name, out['e2e_minus_nms_mAP50-95_pp'], 'pp mAP50-95 (e2e - nms)')


if __name__ == '__main__':
    main()
