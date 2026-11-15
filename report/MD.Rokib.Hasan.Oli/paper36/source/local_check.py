"""Paper #36 local check (RTX 5060 Ti, dataset v1.1, sahi 0.12.8 + ultralytics 8.4.170 as on Kaggle).

  venv python local_check.py <unpacked After Train/results.zip> predict | predict_val | verify | read | valstats | timing

predict   raw boxes of the three trained detectors (conf 0.01), per photo: the full-image pass and every slice pass
          (slices 640 = paper / kit, 512 and 960; overlap 0.25), on the 1,153 test photos and the 204 e2e photos.
          Built from sahi's own pieces (slice_image, perform_batch_inference, convert_original_predictions), the
          same calls get_sliced_prediction makes. -> _cache/*.pkl
verify    (1) sahi_common.merge() vs the kit's Predictor (get_sliced_prediction) box by box on 60 photos per mode;
          (2) the kit's score() on the cached boxes for the 7 Kaggle configs vs the Kaggle results.json.
read      CRNN (the run's model_full.pt) reads the most confident plate of every e2e photo for every variant of
          sahi_common.VARIANTS -> e2e_reads.json
fragile   e2e photos whose reading flips when the FI box moves by 1 px -> e2e_fragile.json (the e2e noise floor)
valstats  Ultralytics val per photo (TP flags and scores, as #35's local_valstats.py), the kit's det.py settings
          (batch 16), on the test split and each condition list -> det_val_stats.jsonl (bootstrap of the kit's mAP).
timing    ms per photo of the kit's Predictor for every config on 300 test photos (median, idle PC), plus sahi with
          all slices in one batch -> timing.json
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

import sahi_common as C

RUN = Path(sys.argv[1])
STAGE = sys.argv[2]
import p36_sahi as P  # noqa: E402  (kit: Predictor, score, load_ocr, read)


def sahi_model(name):
    from sahi import AutoDetectionModel
    return AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path=str(C.weights(RUN, name)),
                                              confidence_threshold=0.01, device='cuda:0', image_size=C.MODELS[name][1])


def boxes_of(ops):
    return np.array([[float(o.score.value), *[float(v) for v in o.bbox.to_xyxy()]] for o in ops]).reshape(-1, 5)


def raw(m, path, sizes):
    from sahi.predict import get_prediction
    from sahi.slicing import slice_image
    from sahi.utils.cv import read_image_as_pil
    arr = np.asarray(read_image_as_pil(str(path)))
    h, w = arr.shape[:2]
    rec = {'wh': (w, h), 'fi': boxes_of(get_prediction(arr, m).object_prediction_list), 'by_size': {}}
    for s in sizes:
        sr = slice_image(image=arr, slice_height=s, slice_width=s, overlap_height_ratio=C.OVERLAP,
                         overlap_width_ratio=C.OVERLAP, auto_slice_resolution=True)
        rows, sl = [], []
        for k in range(len(sr)):
            img, (sx, sy) = sr.images[k], sr.starting_pixels[k]
            m.perform_batch_inference([np.ascontiguousarray(img)])
            m.convert_original_predictions(shift_amount=[[sx, sy]], full_shape=[[sr.original_image_height,
                                                                                  sr.original_image_width]])
            for o in m.object_prediction_list_per_image[0]:
                o = o.get_shifted_object_prediction()
                rows.append([float(o.score.value), *[float(v) for v in o.bbox.to_xyxy()], k])
            sl.append([sx, sy, sx + img.shape[1], sy + img.shape[0]])
        rec['by_size'][s] = {'sl': np.array(rows).reshape(-1, 6), 'slices': np.array(sl).reshape(-1, 4)}
    return rec


def entry(rec, s):
    """one photo's cache entry for slice size s, in the form sahi_common.merge() reads."""
    e = {'wh': rec['wh'], 'fi': rec['fi']}
    if s:
        e.update(rec['by_size'][s])
    else:
        e.update({'sl': np.zeros((0, 6)), 'slices': np.zeros((0, 4))})
    return e


def predict(val=False):
    """test + e2e photos for every model; with val=True the 896 val photos (to choose merge settings on val)."""
    C.CACHE.mkdir(exist_ok=True)
    photos = C.test_photos('val' if val else 'test')
    e2e = [C.ROOT / 'e2e_test' / 'images' / r['file'] for r in C.e2e_rows()]
    parts = [('val', [p[0] for p in photos])] if val else [('test', [p[0] for p in photos]), ('e2e', e2e)]
    for name in (['v8s_640', 'v8s_sf'] if val else C.MODELS):
        m = sahi_model(name)
        for part, files in parts:
            t0 = time.time()
            recs = {f.name: raw(m, f, C.SLICES[name]) for f in files}
            for s in [None] + C.SLICES[name]:
                with open(C.cache_file(name, s, part), 'wb') as fh:
                    pickle.dump({k: entry(r, s) for k, r in recs.items()}, fh)
            print(name, part, len(recs), f'{time.time() - t0:.0f} s', flush=True)
        del m
        torch.cuda.empty_cache()


class A:                                     # the kit's eval arguments (p36_sahi.py defaults)
    slice = 640
    overlap = 0.25


def verify():
    photos = C.test_photos()
    subsets = C.subsets()
    kag = json.load(open(RUN / 'runs' / 'sahi' / 'results.json', encoding='utf-8'))['configs']
    out = {'box_by_box': {}, 'rescore': {}}
    # (1) merge() == get_sliced_prediction on the first 60 photos that have more than one slice + 20 that fit in one
    many = [p for p in photos if max(p[2]) > 640][:60] + [p for p in photos if max(p[2]) <= 640][:20]
    for name in ('v8s_640', 'v8s_sf'):
        cache = C.load_cache(name, 640)
        for mode in ('FI', 'SAHI', 'SAHI_FI'):
            pr = P.Predictor(str(C.weights(RUN, name)), 640, mode, A, 'cuda:0')
            worst, n_box, diff_n = 0.0, 0, 0
            for f, _, _ in many:
                b = np.array(pr(f)[0]).reshape(-1, 5)
                mine = C.merge(cache[f.name], mode)
                if len(b) != len(mine):
                    diff_n += 1
                    continue
                key = lambda z: z[np.lexsort((z[:, 1], z[:, 2], -z[:, 0]))]
                if len(b):
                    worst = max(worst, float(np.abs(key(b) - key(mine)).max()))
                n_box += len(b)
            out['box_by_box'][f'{name}__{mode}'] = {'photos': len(many), 'boxes': n_box, 'photos_with_other_count': diff_n,
                                                    'max_abs_diff': round(worst, 5)}
            print(name, mode, out['box_by_box'][f'{name}__{mode}'], flush=True)
    # sahi's own GreedyNMMPostprocess vs sahi_common's array version, on the conf >= 0.25 boxes of 60 photos
    from sahi.postprocess.combine import GreedyNMMPostprocess
    from sahi.prediction import ObjectPrediction
    gp = GreedyNMMPostprocess(match_threshold=0.5, match_metric='IOS', class_agnostic=False)
    cache = C.load_cache('v8s_640', 640)
    worst = 0.0
    for f, _, wh in many:
        e = cache[f.name]
        allb = np.concatenate([e['sl'][:, :5], e['fi']]) if len(e['slices']) > 1 else e['sl'][:, :5]
        allb = allb[allb[:, 0] >= 0.25]
        if len(allb) < 2:
            continue
        ops = [ObjectPrediction(bbox=list(b[1:5]), category_id=0, category_name='plate', score=float(b[0]),
                                full_shape=[wh[1], wh[0]]) for b in allb]
        ref = boxes_of(gp(ops))
        mine = C._greedy_nmm(allb, 'IOS', 0.5)
        key = lambda z: z[np.lexsort((z[:, 1], z[:, 2], -z[:, 0]))]
        assert len(ref) == len(mine), (f.name, len(ref), len(mine))
        worst = max(worst, float(np.abs(key(ref) - key(mine)).max()))
    out['greedy_nmm_max_abs_diff'] = round(worst, 4)
    print('GNMM', worst)
    # (2) the kit's score() on the cached boxes
    for key in kag:
        name, mode = key.split('__')
        cache = C.load_cache(name, 640 if mode != 'FI' else None)
        preds = [[tuple(b) for b in C.merge(cache[f.name], mode)] for f, _, _ in photos]
        ns = [len(cache[f.name]['slices']) + (mode == 'SAHI_FI' and len(cache[f.name]['slices']) > 1)
              for f, _, _ in photos] if mode != 'FI' else []
        r = P.score(photos, preds, subsets, [0.0], ns)
        k = kag[key]
        cmp = {m: (k[m], r[m]) for m in ('AP50_%', 'AP50_small_%', 'AP50_medium_%', 'AP50_large_%', 'found_%',
                                         'correct_%', 'false_alarm_no_plate_%')}
        cmp['found_by_condition_%'] = (k['found_by_condition_%'], r['found_by_condition_%'])
        out['rescore'][key] = cmp
        print(key, cmp, flush=True)
    (C.HERE / 'verify.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


def read():
    import cv2
    rows = C.e2e_rows()
    ocr, c = P.load_ocr(str(RUN / 'runs' / 'crnn' / 'model_full.pt'), 'cuda')
    memo, out = {}, {}
    for v in C.VARIANTS:
        cache = C.load_cache(v['model'], v.get('size'), 'e2e')
        res = []
        for r in rows:
            b = C.merge(cache[r['file']], v['mode'], v.get('rule', 'NMS_IOU'), small_px=v.get('small_px'))
            b = b[b[:, 0] >= 0.25]
            text = ''
            if len(b):
                s, x1, y1, x2, y2 = b[np.argmax(b[:, 0])]
                k = (r['file'], round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1))
                if k not in memo:
                    im = cv2.imread(str(C.ROOT / 'e2e_test' / 'images' / r['file']))
                    pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
                    crop = im[int(max(0, y1 - ph)):int(y2 + ph), int(max(0, x1 - pw)):int(x2 + pw)]
                    memo[k] = P.read(ocr, c, crop, 'cuda') if crop.size else ''
                text = memo[k]
            res.append({'file': r['file'], 'text': text, 'gt': r['text'], 'detected': bool(len(b)),
                        'box': [round(float(z), 1) for z in b[np.argmax(b[:, 0])][1:5]] if len(b) else None})
        out[v['key']] = res
        print(v['key'], round(100 * np.mean([x['text'] == x['gt'] for x in res]), 2), flush=True)
    (C.HERE / 'e2e_reads.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')


def fragile():
    """how many e2e readings change when the FI plate box moves by 1 px (one side at a time): the e2e noise floor."""
    import cv2
    ocr, c = P.load_ocr(str(RUN / 'runs' / 'crnn' / 'model_full.pt'), 'cuda')
    cache = C.load_cache('v8s_640', None, 'e2e')
    out, n = [], 0
    for r in C.e2e_rows():
        b = C.merge(cache[r['file']], 'FI')
        b = b[b[:, 0] >= 0.25]
        if not len(b):
            continue
        n += 1
        s, x1, y1, x2, y2 = b[np.argmax(b[:, 0])]
        im = cv2.imread(str(C.ROOT / 'e2e_test' / 'images' / r['file']))
        res = set()
        for d in np.vstack([np.zeros(4), np.eye(4), -np.eye(4)]):
            X1, Y1, X2, Y2 = x1 + d[0], y1 + d[1], x2 + d[2], y2 + d[3]
            pw, ph = (X2 - X1) * 0.05, (Y2 - Y1) * 0.05
            crop = im[int(max(0, Y1 - ph)):int(Y2 + ph), int(max(0, X1 - pw)):int(X2 + pw)]
            res.add(P.read(ocr, c, crop, 'cuda') == r['text'])
        if len(res) > 1:
            out.append(r['file'])
    (C.HERE / 'e2e_fragile.json').write_text(json.dumps({'photos_with_box': n, 'flip_with_1px': len(out), 'files': out},
                                                         indent=1), encoding='utf-8')
    print('fragile', len(out), 'of', n)


def valstats():
    import yaml
    from ultralytics import YOLO
    from ultralytics.utils import metrics as um
    cap = []
    orig = um.DetMetrics.update_stats

    def update_stats(self, stat):
        cap.append({'file': Path(stat.get('im_name')).name, 'tp': np.asarray(stat['tp'])[:, 0].astype(int).tolist()
                    if len(stat['tp']) else [], 'conf': [round(float(x), 5) for x in stat['conf']],
                    'n_gt': int(len(stat['target_cls']))})
        return orig(self, stat)

    um.DetMetrics.update_stats = update_stats
    d = C.ROOT / 'detection'
    cond = json.load(open(d / 'conditions.json', encoding='utf-8'))
    jobs = [('all', 'images/test')] + [(k, v['list']) for k, v in cond['real'].items() if k != 'no_plate']
    ydir = C.HERE / '_work' / 'yaml'
    ydir.mkdir(parents=True, exist_ok=True)
    with open(C.HERE / 'det_val_stats.jsonl', 'w', encoding='utf-8') as fh:
        for name, (run, imgsz) in C.MODELS.items():
            model = YOLO(str(C.weights(RUN, name)))
            for tag, test in jobs:
                f = ydir / f'{tag}.yaml'
                f.write_text(yaml.safe_dump({'path': str(d), 'train': 'images/train', 'val': 'images/val',
                                             'test': test, 'names': {0: 'plate'}}), encoding='utf-8')
                cap.clear()
                m = model.val(data=str(f), split='test', imgsz=imgsz, batch=16, device=0, plots=False, workers=0,
                              verbose=False, project=str(C.HERE / '_work' / 'val'), name=f'{name}_{tag}', exist_ok=True)
                fh.write(json.dumps({'model': name, 'set': tag, 'mAP50': round(float(m.box.map50), 4),
                                     'P': round(float(m.box.mp), 4), 'R': round(float(m.box.mr), 4),
                                     'photos': list(cap)}) + '\n')
                print(name, tag, round(float(m.box.map50), 4), len(cap), flush=True)


def timing():
    photos = C.test_photos()
    big = [p for p in photos if max(p[2]) > 640]
    rnd = np.random.RandomState(0)
    pick = [photos[i] for i in sorted(rnd.choice(len(photos), 300, replace=False))]
    out = {'gpu': torch.cuda.get_device_name(0), 'photos': len(pick),
           'photos_gt_640_in_pick': sum(max(p[2]) > 640 for p in pick), 'configs': {}}
    from sahi.predict import get_sliced_prediction
    for name, (_, imgsz) in C.MODELS.items():
        for mode in (['FI', 'SAHI', 'SAHI_FI'] if imgsz == 640 else ['FI']):
            pr = P.Predictor(str(C.weights(RUN, name)), imgsz, mode, A, 'cuda:0')
            for f, _, _ in pick[:10]:
                pr(f)
            ms = []
            for f, _, _ in pick:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pr(f)
                torch.cuda.synchronize(); ms.append(1000 * (time.perf_counter() - t0))
            out['configs'][f'{name}__{mode}'] = {'median_ms': round(float(np.median(ms)), 2),
                                                 'mean_ms': round(float(np.mean(ms)), 2)}
            if mode == 'SAHI_FI':            # the same work with every slice of a photo in one batch (sahi batch_size)
                ms = []
                for f, _, _ in pick:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    get_sliced_prediction(str(f), pr.m, slice_height=640, slice_width=640, overlap_height_ratio=0.25,
                                          overlap_width_ratio=0.25, perform_standard_pred=True, postprocess_type='NMS',
                                          postprocess_match_metric='IOU', postprocess_match_threshold=0.5, verbose=0,
                                          batch_size=8)
                    torch.cuda.synchronize(); ms.append(1000 * (time.perf_counter() - t0))
                out['configs'][f'{name}__SAHI_FI_batch8'] = {'median_ms': round(float(np.median(ms)), 2),
                                                             'mean_ms': round(float(np.mean(ms)), 2)}
            print(name, mode, out['configs'], flush=True)
            del pr
            torch.cuda.empty_cache()
    out['big_photos_in_test'] = len(big)
    (C.HERE / 'timing.json').write_text(json.dumps(out, indent=1), encoding='utf-8')


if __name__ == '__main__':
    {'predict': predict, 'predict_val': lambda: predict(True), 'verify': verify, 'read': read, 'fragile': fragile,
     'valstats': valstats, 'timing': timing}[STAGE]()
