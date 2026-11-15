"""Paper #33 (YOLO26), original-paper check: the released YOLO26 weights on COCO val2017 with BOTH heads, as in the
paper's Table 2 (p.8: "mAPval 50-95" = one-to-many head + NMS, "(e2e)" = the NMS-free one-to-one head), plus T4
TensorRT FP16 latency of each head and ONNX on the CPU. Uses coco_check.py (the #26-#30 checks) for the data, the
pycocotools mAP and the latency loop.

  python coco_check.py prep --root /kaggle/working/datasets
  python c33_coco.py --models yolo26n.pt yolo26s.pt --name coco_gpu0 --project runs [--small] [--no-trt]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import coco_check

HEADS = {'one2many_nms': {}, 'one2one_e2e': {'nms': False}}


def evaluate(a):
    import torch
    from ultralytics import YOLO
    from det import legacy_onnx_export, prec
    yml = Path(a.root) / ('coco_val_small.yaml' if a.small else 'coco_val.yaml')
    imgs = sorted(str(p) for p in (Path(a.root) / 'coco' / 'images' / 'val2017').iterdir())
    res = {'run': a.name, 'kind': 'coco-check-yolo26', 'models': {},
           'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}
    for w in a.models:
        stem = Path(w).stem
        r = {}
        has_e2e = getattr(YOLO(w).model.model[-1], 'one2one_cv2', None) is not None     # YOLO11 has no NMS-free head
        for head, kw in (HEADS if has_e2e else {'one2many_nms': {}}).items():
            h = {}
            vdir = (Path(a.project) / a.name / f'{stem}_{head}').resolve()
            v = YOLO(w).val(data=str(yml), split='val', imgsz=640, batch=32, conf=0.001, iou=0.7, save_json=not a.small,
                            plots=False, workers=common.WORKERS, project=str(vdir.parent), name=vdir.name, exist_ok=True,
                            verbose=False, **kw)
            h['ultralytics_mAP50-95_%'] = round(100 * float(v.box.map), 2)
            h['ultralytics_mAP50_%'] = round(100 * float(v.box.map50), 2)
            pj = Path(getattr(v, 'save_dir', vdir)) / 'predictions.json'
            if not a.small and pj.exists():
                try:
                    h['coco'] = coco_check.cocoeval(a.root, pj)
                except Exception as e:
                    h['coco'] = f'failed: {e}'[:200]
            if torch.cuda.is_available():
                h['pytorch_fp16'] = coco_check.latency(YOLO(w), imgs, {**prec(16), **kw}, n=a.n)
                if a.trt:
                    try:
                        legacy_onnx_export()
                        eng = YOLO(w).export(format='engine', imgsz=640, **prec(16), **kw)
                        h['tensorrt_fp16'] = coco_check.latency(YOLO(eng), imgs, {}, n=a.n)
                    except Exception as e:
                        h['tensorrt_fp16'] = f'failed: {e}'[:300]
            try:
                legacy_onnx_export()
                onnx = YOLO(w).export(format='onnx', imgsz=640, simplify=False, dynamic=False, **kw)
                h['cpu_onnx'] = coco_check.latency(YOLO(onnx), imgs, {'device': 'cpu'}, n=min(a.n, 50), warm=5)
            except Exception as e:
                h['cpu_onnx'] = f'failed: {e}'[:300]
            r[head] = h
            print(stem, head, json.dumps(h), flush=True)
        try:
            r['params_M'] = round(sum(p.numel() for p in YOLO(w).model.parameters()) / 1e6, 2)
        except Exception:
            pass
        res['models'][stem] = r
    return res


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', default=os.path.join(common.WORK, 'datasets'))
    p.add_argument('--models', nargs='+', default=['yolo26n.pt', 'yolo26s.pt'])
    p.add_argument('--no-trt', dest='trt', action='store_false')
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--small', action='store_true', help='quick test: 200 val photos, Ultralytics mAP only')
    p.add_argument('--name', default='coco')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    (Path(a.project) / a.name).mkdir(parents=True, exist_ok=True)
    out = evaluate(a)
    out['env'] = common.env_info()
    common.save_json(out, Path(a.project) / a.name / 'results.json')
    print('DONE', a.name)
