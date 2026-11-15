"""Paper #33 local check (RTX 5060 Ti + Core Ultra 7 265K, dataset v1.1): re-scores the five trained detectors of the
Kaggle run, both heads of every YOLO26 model, plate by plate, so the report can use paired tests and clean timings.

  venv python local_check.py <unpacked After Train/results.zip> [--skip-speed]

1. Every model / head on the test split and its synthetic rain copy at conf 0.25, IoU 0.5 (det.plate_counts rules):
   one row per plate (found, best IoU, area, source, plates in the photo) and per photo (boxes, hits, duplicates).
   found_% must equal the Kaggle run's (same weights, same photos).
2. Ultralytics val (mAP) on the test split, and on the test split without ridoykj (its boxes sit on the plate edge,
   see the paper #6 report), so mAP50-95 can be read without that label style.
3. Timing, one job at a time on an idle PC: PyTorch FP16 batch 1 on the GPU (pre + inference + post, every photo warmed
   once, median of 200) and ONNX on the CPU (model only via onnxruntime, all cores and 4 threads like the Kaggle CPU;
   plus Ultralytics end to end).
4. MuSGD vs SGD optimiser step on YOLO26-N (the trainer's own parameter groups), to explain the slower epochs.
Writes local_check.json, det_per_plate.csv, det_per_photo.csv next to this file.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / 'kaggle'
sys.path.insert(0, str(KIT))
os.environ.setdefault('LPR_WORKERS', '0')
import det  # noqa: E402  (the kit's own iou_matrix / gt_boxes)

DATA = Path(r'D:\498R\dataset\detection')
MODELS = ['y11n', 'y26n', 'y26n_sgd', 'y26n_p2', 'y26s']
HEADS = {'nms': {}, 'e2e': {'nms': False}}
SPLITS = {'test': 'images/test', 'rain': 'images/test_rain_synth', 'fog': 'images/test_fog_synth'}


def yaml_for(name, test):
    f = HERE / '_yaml' / f'{name}.yaml'
    f.parent.mkdir(exist_ok=True)
    f.write_text(f'path: {DATA.as_posix()}\ntrain: images/train\nval: images/val\ntest: {test}\nnc: 1\n'
                 f'names:\n  0: plate\n', encoding='utf-8')
    return str(f)


def score(model, kw, split_dir, tag, plates, photos):
    imgs = sorted((DATA / split_dir).glob('*.jpg'))
    lab = lambda p: Path(str(p.parent).replace('images', 'labels', 1)) / (p.stem + '.txt')
    tp = n_gt = 0
    for s in range(0, len(imgs), 32):
        for r in model.predict([str(p) for p in imgs[s:s + 32]], imgsz=640, conf=0.25, device='0', verbose=False, **kw):
            h, w = r.orig_shape
            g, _ = det.gt_boxes(lab(Path(r.path)), w, h)
            p = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            order = np.argsort(-cf)
            m = det.iou_matrix(p[order], g)
            used, dup = {}, 0
            for i in range(len(order)):
                j = int(np.argmax(m[i])) if m.shape[1] else -1
                if j >= 0 and m[i, j] >= 0.5 and j not in used:
                    used[j] = float(m[i, j])
                elif j >= 0 and m[i, j] >= 0.5:
                    dup += 1                        # a second box on a plate that already has one
            name = Path(r.path).name
            src = name.split('_')[0]
            best = m.max(0) if m.shape[0] else np.zeros(len(g))
            for j, gb in enumerate(g):
                plates.append({**tag, 'file': name, 'plate': j, 'source': src, 'n_plates': len(g),
                               'area': round(float((gb[2] - gb[0]) * (gb[3] - gb[1])), 1),
                               'found': int(j in used), 'best_iou': round(float(best[j]), 4)})
            photos.append({**tag, 'file': name, 'source': src, 'n_plates': len(g), 'boxes': len(p), 'hits': len(used),
                           'dups': dup, 'top_conf': round(float(cf.max()), 4) if len(cf) else 0.0})
            tp += len(used)
            n_gt += len(g)
    return round(100 * tp / max(1, n_gt), 2)


def val(model, kw, yml):
    v = model.val(data=yml, split='test', imgsz=640, batch=16, device='0', workers=0, plots=False, verbose=False,
                  project=str(HERE / '_val'), name='v', exist_ok=True, **kw)
    return {'mAP50': round(100 * float(v.box.map50), 2), 'mAP50-95': round(100 * float(v.box.map), 2),
            'P': round(100 * float(v.box.mp), 2), 'R': round(100 * float(v.box.mr), 2)}


def gpu_latency(weights, kw, imgs):
    from ultralytics import YOLO
    m = YOLO(weights)
    for f in imgs:
        m.predict(f, imgsz=640, device='0', verbose=False, quantize=16, **kw)
    ms, post = [], []
    for f in imgs:
        r = m.predict(f, imgsz=640, device='0', verbose=False, quantize=16, **kw)[0]
        ms.append(sum(r.speed.values()))
        post.append(r.speed['postprocess'])
    return {'ms_median': round(float(np.median(ms)), 2), 'post_ms_median': round(float(np.median(post)), 2),
            'images': len(imgs)}


def cpu_latency(weights, kw, imgs, out_dir):
    """model-only onnxruntime time on one fixed 640x640 input, plus Ultralytics end to end (pre + ORT + post)."""
    import onnxruntime as ort
    from ultralytics import YOLO
    det.legacy_onnx_export()
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / Path(weights).name
    if not src.exists():
        src.write_bytes(Path(weights).read_bytes())
    onnx = YOLO(str(src)).export(format='onnx', imgsz=640, simplify=False, dynamic=False, device='cpu', **kw)
    x = np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32)
    res = {'onnx_MB': round(os.path.getsize(onnx) / 1e6, 2)}
    for label, threads in (('all_cores', 0), ('4_threads', 4)):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        s = ort.InferenceSession(onnx, so, providers=['CPUExecutionProvider'])
        name = s.get_inputs()[0].name
        for _ in range(10):
            s.run(None, {name: x})
        t = []
        for _ in range(100):
            t0 = time.perf_counter()
            s.run(None, {name: x})
            t.append(1000 * (time.perf_counter() - t0))
        res[f'model_ms_{label}'] = round(float(np.median(t)), 2)
        res['outputs'] = [list(o.shape) for o in s.get_outputs()]
    m = YOLO(onnx, task='detect')
    for f in imgs[:10]:
        m.predict(f, imgsz=640, device='cpu', verbose=False)
    ms, post = [], []
    for f in imgs:
        r = m.predict(f, imgsz=640, device='cpu', verbose=False)[0]
        ms.append(sum(r.speed.values()))
        post.append(r.speed['postprocess'])
    res['e2e_ms_median'] = round(float(np.median(ms)), 2)
    res['post_ms_median'] = round(float(np.median(post)), 2)
    return res


def optimiser_step(run):
    """time of one optimiser step (random gradients) for the trainer's MuSGD and SGD parameter groups on YOLO26-N."""
    from types import SimpleNamespace
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer
    out = {}
    for name in ('SGD', 'MuSGD'):
        model = YOLO(str(run / 'yolo26n.pt')).model.float().cuda().train()
        for p in model.parameters():
            p.requires_grad_(True)
        dummy = SimpleNamespace(args=SimpleNamespace(warmup_bias_lr=0.1), data={'nc': 1})
        opt = BaseTrainer.build_optimizer(dummy, model, name=name, lr=0.01, momentum=0.937, decay=5e-4)
        for p in model.parameters():
            p.grad = torch.randn_like(p) * 1e-3
        for _ in range(5):
            opt.step()
        torch.cuda.synchronize()
        t = []
        for _ in range(50):
            t0 = time.perf_counter()
            opt.step()
            torch.cuda.synchronize()
            t.append(1000 * (time.perf_counter() - t0))
        out[name] = {'step_ms_median': round(float(np.median(t)), 2), 'groups': len(opt.param_groups),
                     'lrs': sorted({round(g['lr'], 4) for g in opt.param_groups})}
        del model, opt
        torch.cuda.empty_cache()
    out['steps_per_epoch'] = 536
    out['extra_s_per_epoch'] = round((out['MuSGD']['step_ms_median'] - out['SGD']['step_ms_median']) * 536 / 1000, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run', help='unpacked After Train/results.zip (has runs/<name>/weights/best.pt)')
    ap.add_argument('--skip-speed', action='store_true')
    a = ap.parse_args()
    from ultralytics import YOLO
    run = Path(a.run)
    manifest = json.loads((DATA.parent / 'dataset_manifest.json').read_text(encoding='utf-8'))
    test_imgs = sorted((DATA / 'images' / 'test').glob('*.jpg'))
    nr = HERE / '_yaml' / 'test_no_ridoykj.txt'
    nr.parent.mkdir(exist_ok=True)
    nr.write_text('\n'.join(p.as_posix() for p in test_imgs if not p.name.startswith('ridoykj_')) + '\n', encoding='utf-8')
    ymls = {'test': yaml_for('test', 'images/test'), 'no_ridoykj': yaml_for('no_ridoykj', nr.as_posix())}
    out = {'dataset_version': manifest.get('version'), 'gpu': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'models': {}}
    import ultralytics
    out['ultralytics'] = ultralytics.__version__
    plates, photos = [], []
    for name in MODELS:
        w = str(run / 'runs' / name / 'weights' / 'best.pt')
        heads = HEADS if name != 'y11n' else {'nms': {}}
        for head, kw in heads.items():
            m = YOLO(w)
            r = {'found_%': {}}
            for split, d in SPLITS.items():
                r['found_%'][split] = score(m, kw, d, {'model': name, 'head': head, 'split': split}, plates, photos)
            r['val_test'] = val(m, kw, ymls['test'])
            r['val_test_no_ridoykj'] = val(m, kw, ymls['no_ridoykj'])
            out['models'][f'{name}/{head}'] = r
            print(name, head, json.dumps(r), flush=True)
    for f, rows in (('det_per_plate.csv', plates), ('det_per_photo.csv', photos)):
        with open(HERE / f, 'w', newline='', encoding='utf-8') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    if not a.skip_speed:
        imgs = [str(p) for p in test_imgs[:200]]
        out['speed'] = {}
        for name in MODELS:
            w = str(run / 'runs' / name / 'weights' / 'best.pt')
            for head, kw in (HEADS if name != 'y11n' else {'nms': {}}).items():
                k = f'{name}/{head}'
                out['speed'][k] = {'gpu_fp16': gpu_latency(w, kw, imgs),
                                   'cpu_onnx': cpu_latency(w, kw, imgs[:100], HERE / '_onnx' / f'{name}_{head}')}
                print('speed', k, json.dumps(out['speed'][k]), flush=True)
        out['optimiser_step'] = optimiser_step(run)
        print('optimiser', json.dumps(out['optimiser_step']), flush=True)
    import platform
    out['cpu'] = platform.processor()
    (HERE / 'local_check.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print('DONE')


if __name__ == '__main__':
    main()
