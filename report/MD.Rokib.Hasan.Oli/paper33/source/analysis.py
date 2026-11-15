"""Paper #33 report numbers -> analysis.json (system python; no GPU).

  python analysis.py <unpacked results.zip> <unpacked results original.zip/_output_.zip>

Sources: the v1.1 Kaggle run (runs/*/results.json, results.csv, logs, paper-33.log), the original-paper check
(runs/coco_gpu*/results.json) and the local check (local_check.json, det_per_plate.csv, det_per_photo.csv).
Paired tests: exact McNemar on plates (found / missed at conf 0.25, IoU 0.5) or on plate-free photos (false alarm),
95 % bootstrap interval of the difference in plates found, resampling photos (2000 draws, seed 0).
"""
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
RUN, CHECK = Path(sys.argv[1]), Path(sys.argv[2])
DATA = Path(r'D:\498R\dataset\detection')
MODELS = ['y11n', 'y26n', 'y26n_sgd', 'y26n_p2', 'y26s']
NAMES = {'y11n': 'YOLO11-N (SGD)', 'y26n': 'YOLO26-N (MuSGD)', 'y26n_sgd': 'YOLO26-N (SGD)',
         'y26n_p2': 'YOLO26-N-P2 (MuSGD)', 'y26s': 'YOLO26-S (MuSGD)'}
r2 = lambda x: round(float(x), 2)


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


# ------------------------------------------------------------------ Kaggle run
log = (PAPER / 'After Train' / 'paper-33.log').read_text(encoding='utf-8', errors='replace')
jobs = [{'job': m.group(1), 'code': int(m.group(2)), 'minutes': float(m.group(3))}
        for m in re.finditer(r'\[done\]\s+(\S+) code=(-?\d+) after ([\d.]+) min', log)]
last_t = max(float(m.group(1)) for m in re.finditer(r'^([\d.]+)s ', log, flags=re.M))
env = load(RUN / 'environment.json')
A = {'run': {'jobs': jobs, 'total_h': r2(last_t / 3600), 'dataset_version': env.get('dataset_version'),
             'env': {k: env.get(k) for k in ('torch', 'ultralytics', 'python', 'gpus', 'date')}}}

train, curves = {}, {}
for m in MODELS:
    d = load(RUN / 'runs' / m / 'results.json')
    rows = [{k.strip(): v for k, v in x.items()} for x in csv.DictReader(open(RUN / 'runs' / m / 'results.csv'))]
    ep = [int(float(x['epoch'])) for x in rows]
    t = [float(x['time']) for x in rows]
    curves[m] = {'epoch': ep, 'val_mAP50-95': [r2(100 * float(x['metrics/mAP50-95(B)'])) for x in rows],
                 'val_mAP50': [r2(100 * float(x['metrics/mAP50(B)'])) for x in rows],
                 'lr_pg0': [float(x['lr/pg0']) for x in rows]}
    logtxt = (RUN / 'logs' / f'{m}.log').read_text(encoding='utf-8', errors='replace')
    tp = d['test_plates']
    train[m] = {'name': NAMES[m], 'optimizer': d['settings']['optimizer'], 'model': d['settings']['model'],
                'epochs_planned': d['epochs_planned'], 'epochs_run': d['epochs_run'], 'best_epoch': d['best_epoch'],
                'time_cap_hit': '[time cap]' in logtxt, 'train_minutes': d['train_minutes'],
                's_per_epoch': r2(t[-1] / len(t)), 'params_M': d['params_M'], 'GFLOPs': d['GFLOPs'],
                'peak_gpu_mem_GB': d.get('peak_gpu_mem_GB'),
                'val': {k: r2(100 * d['val'][k]) for k in ('P', 'R', 'mAP50', 'mAP50-95')},
                'test': {k: r2(100 * d['test'][k]) for k in ('P', 'R', 'mAP50', 'mAP50-95')},
                'found_%': tp['found_%'], 'correct_%': tp['correct_%'], 'img_all_ok_%': tp['img_all_ok_%'],
                'iou70_acc_%': tp['iou70_acc_%'], 'mean_best_iou_%': tp['mean_best_iou_%'],
                'recall_by_size_%': tp['recall_by_size_%'], 'plates_by_size': tp['plates_by_size'],
                'found_by_source': {k: v['found_%'] for k, v in tp['found_%_by_source'].items()},
                'plates_by_source': {k: v['plates'] for k, v in tp['found_%_by_source'].items()},
                'by_condition': {k: ({'mAP50': r2(100 * v['mAP50']), 'mAP50-95': r2(100 * v['mAP50-95']),
                                      'found_%': v['plates']} if 'mAP50' in v else v)
                                 for k, v in d['test_by_condition'].items()},
                'nan_values': sum(1 for x in rows for v in x.values() if 'nan' in str(v).lower())}
    if 'mAP50' in d['test_by_condition'].get('no_plate', {}) or 'false_alarm_%' in d['test_by_condition'].get('no_plate', {}):
        train[m]['false_alarm_%'] = d['test_by_condition']['no_plate']['false_alarm_%']
A['train'], A['curves'] = train, curves
A['same_epoch_val'] = {e: {m: curves[m]['val_mAP50-95'][curves[m]['epoch'].index(e)] for m in MODELS}
                       for e in (5, 10, 20, 30, 40, 50)}

heads = {}
for m in ('y26n', 'y26s', 'y26n_p2'):
    d = load(RUN / 'runs' / f'heads_{m}' / 'results.json')
    hh = {}
    for h, v in d['heads'].items():
        x = {'test': {k: r2(100 * v['test'][k]) for k in ('P', 'R', 'mAP50', 'mAP50-95')},
             'found_%': v['test_plates']['found_%'], 'correct_%': v['test_plates']['correct_%'],
             'false_alarm_%': v['no_plate']['false_alarm_%'], 'pytorch_fp16_ms': v['pytorch_fp16']['ms_median'],
             'pytorch_fp16_post_ms': v['pytorch_fp16']['postprocess_ms_median']}
        for k in ('tensorrt_fp16', 'tensorrt_int8'):
            t = v[k]
            x[k] = {'ms': t['ms_median'], 'post_ms': t['postprocess_ms_median'], 'mAP50': r2(100 * t['test_mAP50']),
                    'mAP50-95': r2(100 * t['test_mAP50-95'])}
        x['cpu_onnx_ms'] = v['cpu_onnx']['ms_median']
        hh['nms' if h == 'one2many_nms' else 'e2e'] = x
    hh['e2e_minus_nms_mAP50-95_pp'] = d['e2e_minus_nms_mAP50-95_pp']
    heads[m] = hh
A['heads'] = heads
sp = load(RUN / 'runs' / 'y11n' / 'results.json')['speed']
A['y11n_speed'] = {'fp32_ms': sp['fp32']['ms_median'], 'fp16_ms': sp['fp16']['ms_median'],
                   'trt_fp16_ms': sp['tensorrt_fp16']['ms_median'], 'trt_int8_ms': sp['tensorrt_int8']['ms_median'],
                   'trt_int8_mAP50': r2(100 * sp['tensorrt_int8']['test_mAP50']), 'cpu_onnx_ms': sp['cpu_onnx']['ms_median']}
# which jobs ran at the same time (timings share the 4-core CPU): from the [start]/[done] times in the log
ev = [(float(m.group(1)), m.group(2), m.group(3)) for m in re.finditer(r'^([\d.]+)s \d+ \[(start|done)\]\s+(\S+)', log, flags=re.M)]
span = {}
for t, kind, job in ev:
    span.setdefault(job, [None, None])[0 if kind == 'start' else 1] = t
A['overlap'] = {j: [k for k, (a, b) in span.items() if k != j and a < span[j][1] and b > span[j][0]] for j in span}

# ------------------------------------------------------------------ original-paper check (COCO)
coco = {}
for g in ('coco_gpu0', 'coco_gpu1'):
    d = load(CHECK / 'runs' / g / 'results.json')
    for m, v in d['models'].items():
        for h in ('one2many_nms', 'one2one_e2e'):
            if h in v:
                x = v[h]
                coco[f'{m}/{h}'] = {'AP': x['coco']['AP_%'], 'AP50': x['coco']['AP50_%'], 'AP_small': x['coco']['AP_small_%'],
                                    'ultralytics_AP': x['ultralytics_mAP50-95_%'],
                                    'trt_inf_ms': x['tensorrt_fp16']['inference_ms'], 'trt_e2e_ms': x['tensorrt_fp16']['end_to_end_ms'],
                                    'pt_fp16_inf_ms': x['pytorch_fp16']['inference_ms'],
                                    'cpu_inf_ms': x['cpu_onnx']['inference_ms'], 'cpu_e2e_ms': x['cpu_onnx']['end_to_end_ms']}
        coco[f'{m}/params_M'] = v.get('params_M')
clog = (PAPER / 'After Train' / 'paper-33-original.log').read_text(encoding='utf-8', errors='replace')
A['coco'] = coco
A['coco_jobs'] = [{'job': m.group(1), 'code': int(m.group(2)), 'minutes': float(m.group(3))}
                  for m in re.finditer(r'\[done\]\s+(\S+) code=(-?\d+) after ([\d.]+) min', clog)]
A['coco_paper'] = {'yolo26n': (40.9, 40.1, 1.7, 38.9), 'yolo26s': (48.6, 47.8, 2.5, 87.2), 'yolo26m': (53.1, 52.5, 4.7, 220.0),
                   'yolo11n': (39.5, None, 1.5, 56.1)}

# ------------------------------------------------------------------ local check
L = load(HERE / 'local_check.json')
A['local'] = {k: L[k] for k in ('dataset_version', 'gpu', 'torch', 'ultralytics', 'cpu')}
A['local']['models'] = L['models']
A['local']['speed'] = L.get('speed')
A['local']['optimiser_step'] = L.get('optimiser_step')
rep = {}
for m in MODELS:
    loc = L['models'][f'{m}/nms']
    rep[m] = {'kaggle_found': train[m]['found_%'], 'local_found': loc['found_%']['test'],
              'kaggle_rain': train[m]['by_condition']['rain_synth']['found_%'], 'local_rain': loc['found_%']['rain'],
              'kaggle_mAP50-95': train[m]['test']['mAP50-95'], 'local_mAP50-95': loc['val_test']['mAP50-95'],
              'kaggle_mAP50': train[m]['test']['mAP50'], 'local_mAP50': loc['val_test']['mAP50']}
for m in ('y26n', 'y26s', 'y26n_p2'):
    rep[f'{m}/e2e'] = {'kaggle_found': heads[m]['e2e']['found_%'], 'local_found': L['models'][f'{m}/e2e']['found_%']['test']}
A['reproduction'] = rep

plates = list(csv.DictReader(open(HERE / 'det_per_plate.csv', encoding='utf-8')))
photos = list(csv.DictReader(open(HERE / 'det_per_photo.csv', encoding='utf-8')))
cond = {k: {l.strip().split('/')[-1] for l in open(DATA / f'cond_test_{k}.txt', encoding='utf-8') if l.strip()}
        for k in ('night', 'blurry', 'small_plates', 'clear', 'no_plate')}
P = defaultdict(dict)                  # (model/head, split) -> {(file, plate): found}
for r in plates:
    P[(f"{r['model']}/{r['head']}", r['split'])][(r['file'], int(r['plate']))] = int(r['found'])
meta = {(r['file'], int(r['plate'])): r for r in plates if r['model'] == 'y11n' and r['split'] == 'test'}


def paired(a, b, split='test', subset=None, boot=True):
    """a vs b on the same plates: found %, plates only a / only b found, McNemar p, bootstrap CI (photos)."""
    pa, pb = P[(a, split)], P[(b, split)]
    keys = [k for k in pa if (subset is None or k[0] in subset)]
    oa = sum(pa[k] and not pb[k] for k in keys)
    ob = sum(pb[k] and not pa[k] for k in keys)
    out = {'plates': len(keys), 'a_%': r2(100 * sum(pa[k] for k in keys) / len(keys)),
           'b_%': r2(100 * sum(pb[k] for k in keys) / len(keys)), 'only_a': oa, 'only_b': ob, 'p': mcnemar(oa, ob)}
    if boot and keys:
        byf = defaultdict(list)
        for k in keys:
            byf[k[0]].append(pa[k] - pb[k])
        files = list(byf)
        rng = random.Random(0)
        ds = []
        for _ in range(2000):
            s = n = 0
            for f in (rng.choice(files) for _ in files):
                s += sum(byf[f]); n += len(byf[f])
            ds.append(100 * s / n)
        ds.sort()
        out['diff_pp'] = r2(out['a_%'] - out['b_%'])
        out['ci95_pp'] = [r2(ds[49]), r2(ds[1949])]
    return out


PAIRS = [('y26n/nms', 'y11n/nms', 'YOLO26-N (MuSGD) vs YOLO11-N: the YOLO26 recipe vs the team baseline'),
         ('y26n_sgd/nms', 'y11n/nms', 'YOLO26-N vs YOLO11-N, both SGD: architecture alone'),
         ('y26n/nms', 'y26n_sgd/nms', 'MuSGD vs SGD on YOLO26-N (MuSGD stopped at epoch 50)'),
         ('y26n_p2/nms', 'y26n/nms', 'P2 head vs plain YOLO26-N (both MuSGD)'),
         ('y26s/nms', 'y26n/nms', 'YOLO26-S vs YOLO26-N (both MuSGD)'),
         ('y26n/e2e', 'y11n/nms', 'NMS-free YOLO26-N vs YOLO11-N + NMS: the deployment choice'),
         ('y26n_p2/e2e', 'y11n/nms', 'NMS-free YOLO26-N-P2 vs YOLO11-N + NMS'),
         ('y26n/e2e', 'y26n/nms', 'NMS-free vs NMS head, YOLO26-N'),
         ('y26n_sgd/e2e', 'y26n_sgd/nms', 'NMS-free vs NMS head, YOLO26-N (SGD)'),
         ('y26n_p2/e2e', 'y26n_p2/nms', 'NMS-free vs NMS head, YOLO26-N-P2'),
         ('y26s/e2e', 'y26s/nms', 'NMS-free vs NMS head, YOLO26-S')]
A['paired_test'] = {f'{a} vs {b}': {**paired(a, b), 'what': w} for a, b, w in PAIRS}
A['paired_rain'] = {f'{a} vs {b}': {**paired(a, b, 'rain'), 'what': w} for a, b, w in PAIRS[:6]}
A['paired_rain'].update({f'{a} vs {b}': {**paired(a, b, 'rain'), 'what': w} for a, b, w in (
    ('y26n_p2/nms', 'y26n_sgd/nms', 'MuSGD (P2, 60 epochs) vs SGD (plain)'),
    ('y26s/nms', 'y26n_sgd/nms', 'MuSGD (S, 60 epochs) vs SGD (N)'))})
A['paired_fog'] = {f'{a} vs {b}': paired(a, b, 'fog') for a, b, _ in PAIRS[:3]}
A['paired_night'] = {f'{a} vs {b}': paired(a, b, subset=cond['night']) for a, b, _ in PAIRS[:6]}
A['paired_small'] = {f'{a} vs {b}': paired(a, b, subset=cond['small_plates']) for a, b, _ in PAIRS[:6]}

# per head: rain / night / small found %, multi-plate photos, duplicates, false alarms, extra boxes
keys_all = sorted({k[0] for k in P})
H = {}
for mh in keys_all:
    pt, pr, pf = P[(mh, 'test')], P[(mh, 'rain')], P[(mh, 'fog')]
    multi = [k for k in pt if int(meta[k]['n_plates']) >= 2]
    ph = [r for r in photos if f"{r['model']}/{r['head']}" == mh and r['split'] == 'test']
    neg = [r for r in ph if int(r['n_plates']) == 0]
    H[mh] = {'found_test': r2(100 * sum(pt.values()) / len(pt)), 'found_rain': r2(100 * sum(pr.values()) / len(pr)),
             'found_fog': r2(100 * sum(pf.values()) / len(pf)),
             'found_night': r2(100 * sum(v for k, v in pt.items() if k[0] in cond['night']) / sum(k[0] in cond['night'] for k in pt)),
             'found_small': r2(100 * sum(v for k, v in pt.items() if k[0] in cond['small_plates']) / sum(k[0] in cond['small_plates'] for k in pt)),
             'multi_plates': len(multi), 'found_multi': r2(100 * sum(pt[k] for k in multi) / len(multi)),
             'dup_boxes': sum(int(r['dups']) for r in ph),
             'extra_boxes': sum(int(r['boxes']) - int(r['hits']) for r in ph),
             'false_alarm_photos': sum(int(r['boxes']) > 0 for r in neg), 'no_plate_photos': len(neg),
             'missed_plates': sum(1 - v for v in pt.values())}
A['by_head'] = H
A['counts'] = {'test_plates': len(P[('y11n/nms', 'test')]), 'night_plates': sum(k[0] in cond['night'] for k in P[('y11n/nms', 'test')]),
               'small_plates': sum(k[0] in cond['small_plates'] for k in P[('y11n/nms', 'test')]),
               'small_photos': len(cond['small_plates']), 'night_photos': len(cond['night']),
               'multi_photos': len({k[0] for k in P[('y11n/nms', 'test')] if int(meta[k]['n_plates']) >= 2}),
               'no_plate_photos': len(cond['no_plate'])}
# false alarms paired on plate-free photos
fa = {mh: {r['file']: int(r['boxes']) > 0 for r in photos if f"{r['model']}/{r['head']}" == mh and r['split'] == 'test'
           and int(r['n_plates']) == 0} for mh in keys_all}
A['paired_false_alarm'] = {}
for a, b, _ in PAIRS:
    oa = sum(fa[a][f] and not fa[b][f] for f in fa[a])
    ob = sum(fa[b][f] and not fa[a][f] for f in fa[a])
    A['paired_false_alarm'][f'{a} vs {b}'] = {'a': sum(fa[a].values()), 'b': sum(fa[b].values()), 'only_a': oa,
                                              'only_b': ob, 'p': mcnemar(oa, ob)}
# misses: are they the same plates? (overlap of the misses of the five NMS models)
miss = {m: {k for k, v in P[(f'{m}/nms', 'test')].items() if not v} for m in MODELS}
allm = set.intersection(*miss.values())
A['misses'] = {'per_model': {m: len(v) for m, v in miss.items()}, 'missed_by_all': len(allm),
               'missed_by_any': len(set.union(*miss.values())),
               'all_by_source': dict(sorted(defaultdict(int, {s: sum(meta[k]['source'] == s for k in allm)
                                                              for s in {meta[k]['source'] for k in allm}}).items())),
               'all_examples': sorted(f'{k[0]}#{k[1]}' for k in allm)[:40]}
A['local_no_ridoykj'] = {k: v['val_test_no_ridoykj'] for k, v in L['models'].items()}
(HERE / 'analysis.json').write_text(json.dumps(A, indent=1, ensure_ascii=False), encoding='utf-8')
print('written analysis.json')
for k, v in A['paired_test'].items():
    print(f"{k:28s} {v['a_%']:6.2f} {v['b_%']:6.2f}  {v['only_a']:3d}/{v['only_b']:3d}  p={v['p']:.4f}  CI {v['ci95_pp']}")
for k, v in A['paired_rain'].items():
    print('rain', f"{k:28s} {v['a_%']:6.2f} {v['b_%']:6.2f}  {v['only_a']:3d}/{v['only_b']:3d}  p={v['p']:.2g}  CI {v['ci95_pp']}")
