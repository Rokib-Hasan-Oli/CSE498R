"""Paper #36 report: every number the PDF uses -> analysis.json (CPU, venv python).

  venv python analysis.py <unpacked After Train/results.zip> <c36 check run dir (runs/visdrone_rescore)>

Reads the Kaggle run (results.json files, logs), the local caches and checks written by local_check.py
(_cache/*.pkl, verify.json, e2e_reads.json, det_val_stats.jsonl, timing.json) and visdrone_extra.json.
Scores every sahi_common.VARIANTS setting plate by plate on the 1,153 test photos with the kit's rules (found = conf
0.25 + IoU 0.5; AP50 = COCO, conf 0.01, max 500), sorts the extra boxes into kinds, runs the paired tests (exact
McNemar on plates / photos, photo bootstrap for AP50) and picks the val-best settings for the "small boxes only" merge.
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

import sahi_common as C
import p36_sahi as P  # noqa: E402  (kit: coco_ap50, match)

RUN = Path(sys.argv[1])
CHECK = Path(sys.argv[2])
RNG = np.random.RandomState(36)
NB = 2000
OUT = {}


def mcnemar(a, b):
    """a, b: 0/1 arrays over the same items. exact two-sided McNemar (binomial on the discordant pairs)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    only_a, only_b = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {'only_a': only_a, 'only_b': only_b, 'p': p}


# ------------------------------------------------------------------------------------------------- fast COCO AP50
def coco_flags(pred, g, max_det=500):
    """pycocotools matching for one photo, one class, area 'all': (scores, tp flags)."""
    pred = pred[np.argsort(-pred[:, 0], kind='mergesort')][:max_det] if len(pred) else pred.reshape(0, 5)
    if not len(pred):
        return np.zeros(0), np.zeros(0, int)
    if not len(g):
        return pred[:, 0], np.zeros(len(pred), int)
    iou = det_iou(pred[:, 1:5], g)
    used = np.zeros(len(g), bool)
    tp = np.zeros(len(pred), int)
    for i in range(len(pred)):
        best, m = 0.5 - 1e-10, -1
        for j in range(len(g)):
            if used[j] or iou[i, j] < best:
                continue
            best, m = iou[i, j], j
        if m >= 0:
            used[m] = True
            tp[i] = 1
    return pred[:, 0], tp


def det_iou(a, b):
    """pycocotools IoU for xyxy boxes (its box widths have no +1)."""
    import det
    return det.iou_matrix(a, b)


def coco_ap(scores, tps, n_gt):
    if n_gt == 0 or not len(scores):
        return 0.0
    o = np.argsort(-scores, kind='mergesort')
    tp = np.cumsum(tps[o]); fp = np.cumsum(1 - tps[o])
    rc = tp / n_gt
    pr = tp / np.maximum(tp + fp, np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, np.linspace(0, 1, 101), side='left')
    q = np.where(idx < len(pr), pr[np.minimum(idx, len(pr) - 1)], 0.0)
    return 100 * float(q.mean())


def ap_from_flags(fl, n_gt, idx=None):
    idx = range(len(fl)) if idx is None else idx
    s = np.concatenate([fl[i][0] for i in idx]) if len(idx) else np.zeros(0)
    t = np.concatenate([fl[i][1] for i in idx]) if len(idx) else np.zeros(0, int)
    return coco_ap(s, t, sum(n_gt[i] for i in idx))


# ------------------------------------------------------------------------------------------------ per-plate scoring
def kinds(pred, g, used):
    """what the boxes at conf >= 0.25 that are not a found plate are."""
    k = {'duplicate': 0, 'part_of_plate': 0, 'plate_plus_more': 0, 'background': 0}
    if not len(pred):
        return k
    m = det_iou(pred[:, 1:5], g) if len(g) else np.zeros((len(pred), 0))
    matched = set()
    for i in np.argsort(-pred[:, 0]):
        j = int(np.argmax(m[i])) if m.shape[1] else -1
        if j >= 0 and m[i, j] >= 0.5 and j not in matched:
            matched.add(j)
            continue
        if j >= 0 and m[i, j] >= 0.5:
            k['duplicate'] += 1
            continue
        if m.shape[1]:
            b = pred[i, 1:5]
            ix1 = np.maximum(b[0], g[:, 0]); iy1 = np.maximum(b[1], g[:, 1])
            ix2 = np.minimum(b[2], g[:, 2]); iy2 = np.minimum(b[3], g[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            pa = (b[2] - b[0]) * (b[3] - b[1]); ga = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
            if (inter / max(pa, 1e-9)).max() >= 0.5:
                k['part_of_plate'] += 1
                continue
            if (inter / np.maximum(ga, 1e-9)).max() >= 0.5:
                k['plate_plus_more'] += 1
                continue
        k['background'] += 1
    return k


def score_variant(v, photos, cache, sub):
    """one setting on the test photos: plate rows, photo rows, COCO flags."""
    plates, ph, flags = [], [], []
    for f, g, wh in photos:
        b = C.merge(cache[f.name], v['mode'], v.get('rule', 'NMS_IOU'), small_px=v.get('small_px'))
        flags.append(coco_flags(b, g))
        b25 = b[b[:, 0] >= 0.25]
        used = P.match([tuple(x) for x in b25], g)
        wd, ht = C.plate_size_640(g, wh) if len(g) else (np.zeros(0), np.zeros(0))
        for j in range(len(g)):
            plates.append({'file': f.name, 'j': j, 'found': int(j in used), 'w640': float(wd[j]), 'h640': float(ht[j]),
                           'area': float((g[j, 2] - g[j, 0]) * (g[j, 3] - g[j, 1])), 'src': f.name.split('_')[0]})
        kd = kinds(b25, g, used)
        ph.append({'file': f.name, 'n_gt': len(g), 'boxes': len(b25), 'hits': len(used), 'fp': len(b25) - len(used),
                   'no_plate': int(f.name in sub['no_plate']), 'any_box': int(len(b25) > 0), **kd})
    return plates, ph, flags


def summarise(plates, ph, flags, n_gt, photos, sub):
    fd = np.array([p['found'] for p in plates])
    w = np.array([p['w640'] for p in plates])
    bins = [(0, 24), (24, 40), (40, 64), (64, 128), (128, 1e9)]
    tp = sum(p['hits'] for p in ph); fp = sum(p['fp'] for p in ph)
    names = [f.name for f, _, _ in photos]
    idx_small = [i for i, n in enumerate(names) if n in sub['small_plates']]
    idx_night = [i for i, n in enumerate(names) if n in sub['night']]
    plate_sub = lambda key: np.array([p['found'] for p in plates if p['file'] in sub[key]])
    neg = [p for p in ph if p['no_plate']]
    s = {'AP50': round(ap_from_flags(flags, n_gt), 2), 'found_%': round(100 * fd.mean(), 2), 'found_n': int(fd.sum()),
         'correct_%': round(100 * tp / max(1, tp + fp), 2), 'extra_boxes': int(fp),
         'extra_by_kind': {k: int(sum(p[k] for p in ph)) for k in ('duplicate', 'part_of_plate', 'plate_plus_more', 'background')},
         'false_alarm_%': round(100 * np.mean([p['any_box'] for p in neg]), 2), 'false_alarm_n': int(sum(p['any_box'] for p in neg)),
         'found_by_w640_%': {f'{a}-{b if b < 1e8 else "max"}': [int(fd[(w >= a) & (w < b)].sum()), int(((w >= a) & (w < b)).sum())]
                             for a, b in bins},
         'small_plates_found': [int(plate_sub('small_plates').sum()), int(len(plate_sub('small_plates')))],
         'night_found': [int(plate_sub('night').sum()), int(len(plate_sub('night')))],
         'blurry_found': [int(plate_sub('blurry').sum()), int(len(plate_sub('blurry')))],
         'AP50_small_plate_photos': round(ap_from_flags(flags, n_gt, idx_small), 2),
         'AP50_night_photos': round(ap_from_flags(flags, n_gt, idx_night), 2)}
    return s


def boot_ap_diff(fa, fb, n_gt, idx=None):
    idx = np.arange(len(fa)) if idx is None else np.asarray(idx)
    d = []
    for _ in range(NB):
        r = RNG.choice(idx, len(idx), replace=True)
        d.append(ap_from_flags(fa, n_gt, r) - ap_from_flags(fb, n_gt, r))
    d = np.array(d)
    return {'diff': round(ap_from_flags(fa, n_gt, idx) - ap_from_flags(fb, n_gt, idx), 2),
            'ci95': [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
            'share_below_0': round(float(np.mean(d < 0)), 4)}


# -------------------------------------------------------------------------------------------- Ultralytics val stats
def ul_ap50(photos):
    tp = np.concatenate([np.array(p['tp'], int) for p in photos]) if photos else np.zeros(0)
    cf = np.concatenate([np.array(p['conf'], float) for p in photos]) if photos else np.zeros(0)
    n = sum(p['n_gt'] for p in photos)
    if n == 0 or not len(cf):
        return 0.0
    from ultralytics.utils.metrics import compute_ap
    o = np.argsort(-cf, kind='stable')
    tpc = np.cumsum(tp[o]); fpc = np.cumsum(1 - tp[o])
    ap, _, _ = compute_ap(tpc / (n + 1e-16), tpc / (tpc + fpc))
    return 100 * float(ap)


def valstats():
    rows = [json.loads(l) for l in open(C.HERE / 'det_val_stats.jsonl', encoding='utf-8')]
    V = {(r['model'], r['set']): r for r in rows}
    out = {'mAP50': {f'{m}|{s}': {'val_mAP50': round(100 * r['mAP50'], 2), 'mine': round(ul_ap50(r['photos']), 2)}
                     for (m, s), r in V.items()}}
    pairs = {}
    for s in sorted({s for _, s in V}):
        for a, b in (('v8s_sf', 'v8s_640'), ('v8s_1280', 'v8s_640'), ('v8s_sf', 'v8s_1280')):
            pa, pb = V[(a, s)]['photos'], V[(b, s)]['photos']
            assert [p['file'] for p in pa] == [p['file'] for p in pb] or sorted(p['file'] for p in pa) == sorted(p['file'] for p in pb)
            pb_by = {p['file']: p for p in pb}
            pb = [pb_by[p['file']] for p in pa]
            d = []
            for _ in range(NB):
                r = RNG.randint(0, len(pa), len(pa))
                d.append(ul_ap50([pa[i] for i in r]) - ul_ap50([pb[i] for i in r]))
            pairs[f'{a}-{b}|{s}'] = {'diff': round(ul_ap50(pa) - ul_ap50(pb), 2),
                                     'ci95': [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)],
                                     'photos': len(pa), 'plates': sum(p['n_gt'] for p in pa)}
    out['pairs'] = pairs
    return out


# --------------------------------------------------------------------------------------------------------- health
def health():
    log = C.HERE.parent / 'After Train' / 'paper-36.log'
    h = {'runs': {}}
    for r in ('det_v8s_640', 'det_v8s_sf', 'det_v8s_1280', 'crnn'):
        d = json.load(open(RUN / 'runs' / r / 'results.json', encoding='utf-8'))
        h['runs'][r] = {k: d.get(k) for k in ('epochs_planned', 'epochs_run', 'best_epoch', 'train_minutes', 'params_M',
                                               'GFLOPs', 'peak_gpu_mem_GB')}
        if 'test' in d:
            h['runs'][r].update({'val_mAP50': round(100 * d['val']['mAP50'], 2), 'test_mAP50': round(100 * d['test']['mAP50'], 2),
                                 'test_mAP50_95': round(100 * d['test']['mAP50-95'], 2), 'test_P': d['test']['P'],
                                 'test_R': d['test']['R'], 'found_%': d['test_plates']['found_%'],
                                 'correct_%': d['test_plates']['correct_%'],
                                 'by_condition': {k: (round(100 * v['mAP50'], 1) if isinstance(v, dict) and 'mAP50' in v else v)
                                                  for k, v in d['test_by_condition'].items()},
                                 'by_source': {k: v['found_%'] for k, v in d['test_plates']['found_%_by_source'].items()},
                                 'speed': d.get('speed')})
            hist = list(csv.DictReader(open(RUN / 'runs' / r / 'results.csv', encoding='utf-8')))
            key = [k for k in hist[0] if 'mAP50(B)' in k and '95' not in k][0]
            h['runs'][r]['val_curve'] = [round(100 * float(x[key]), 2) for x in hist]
            h['runs'][r]['nan'] = any(any(v.strip().lower() == 'nan' for v in x.values()) for x in hist)
        else:
            h['runs'][r].update({'test_plate_exact_%': d['test']['plate_exact_%'] if 'test' in d else None})
    if log.exists():
        txt = log.read_text(encoding='utf-8', errors='replace')
        h['jobs'] = [(m.group(1), int(m.group(2)), float(m.group(3))) for m in
                     re.finditer(r'\[done\]\s+(\S+) code=(-?\d+) after ([\d.]+) min', txt)]
        h['total_h'] = round(float(re.findall(r'^([\d.]+)s ', txt, re.M)[-1]) / 3600, 2)
        h['sf_set'] = json.loads(re.search(r'SF set: (\{.*?\})\n', txt).group(1).replace("'", '"'))
    h['env'] = json.load(open(RUN / 'environment.json', encoding='utf-8'))
    kit = C.KIT
    h['code_same'] = {f.name: (f.read_bytes().replace(b'\r\n', b'\n') == (kit / f.name).read_bytes().replace(b'\r\n', b'\n'))
                      for f in sorted((RUN / 'code').glob('*.py'))}
    sp = {}
    for r in ('det_v8s_640', 'det_v8s_1280'):
        d = json.load(open(RUN / 'runs' / r / 'results.json', encoding='utf-8'))
        sp[r] = d.get('speed')
    h['speed'] = sp
    return h


def main():
    OUT['health'] = health()
    kag = json.load(open(RUN / 'runs' / 'sahi' / 'results.json', encoding='utf-8'))
    OUT['kaggle'] = kag['configs']
    OUT['verify'] = json.load(open(C.HERE / 'verify.json', encoding='utf-8'))
    photos = C.test_photos()
    sub = C.subsets()
    n_gt = [len(g) for _, g, _ in photos]
    caches = {}
    per_plate, flags_all, ph_all, S = {}, {}, {}, {}
    for v in C.VARIANTS:
        ck = (v['model'], v.get('size'))
        if ck not in caches:
            caches[ck] = C.load_cache(*ck)
        pl, ph, fl = score_variant(v, photos, caches[ck], sub)
        per_plate[v['key']], ph_all[v['key']], flags_all[v['key']] = pl, ph, fl
        S[v['key']] = summarise(pl, ph, fl, n_gt, photos, sub)
        print(v['key'], {k: S[v['key']][k] for k in ('AP50', 'found_%', 'correct_%', 'false_alarm_%')}, flush=True)
    # pycocotools AP50 (all / small / medium / large) for the kit's 7 settings, same as the Kaggle job
    for key in kag['configs']:
        name, mode = key.split('__')
        cache = caches[(name, 640 if mode != 'FI' else None)]
        S[key]['coco'] = P.coco_ap50([(g, wh) for _, g, wh in photos], [[tuple(b) for b in C.merge(cache[f.name], mode)]
                                                                         for f, _, _ in photos])
    OUT['variants'] = S
    # paired tests on plates (found), plate-free photos (false alarm) and AP50 (photo bootstrap)
    pairs = [('v8s_640__SAHI', 'v8s_640__FI'), ('v8s_640__SAHI_FI', 'v8s_640__FI'), ('v8s_sf__SAHI_FI', 'v8s_sf__FI'),
             ('v8s_sf__FI', 'v8s_640__FI'), ('v8s_1280__FI', 'v8s_640__FI'), ('v8s_sf__FI', 'v8s_1280__FI'),
             ('v8s_sf__SAHI_FI', 'v8s_640__FI'), ('v8s_640__SAHI_FI__GNMM_IOS', 'v8s_640__SAHI_FI'),
             ('v8s_640__SAHI_FI__BORDER', 'v8s_640__SAHI_FI'), ('v8s_640__SAHI_FI__GNMM_IOS', 'v8s_640__FI'),
             ('v8s_640__SAHI_FI__BORDER', 'v8s_640__FI')]
    OUT['pairs'] = {}
    for a, b in pairs:
        fa = np.array([p['found'] for p in per_plate[a]]); fb = np.array([p['found'] for p in per_plate[b]])
        na = np.array([p['any_box'] for p in ph_all[a] if p['no_plate']]); nb = np.array([p['any_box'] for p in ph_all[b] if p['no_plate']])
        sm = lambda k: np.array([p['found'] for p in per_plate[k] if p['file'] in sub['small_plates']])
        OUT['pairs'][f'{a}|{b}'] = {'found': mcnemar(fa, fb), 'false_alarm': mcnemar(na, nb),
                                    'small_plates': mcnemar(sm(a), sm(b)),
                                    'AP50': boot_ap_diff(flags_all[a], flags_all[b], n_gt)}
        print(a, b, OUT['pairs'][f'{a}|{b}'], flush=True)
    # SF vs 640 on the small-plate photos and on night photos, AP from the same FI boxes
    names = [f.name for f, _, _ in photos]
    for tag in ('small_plates', 'night'):
        idx = [i for i, n in enumerate(names) if n in sub[tag]]
        OUT['pairs'][f'v8s_sf__FI|v8s_640__FI|AP_{tag}'] = boot_ap_diff(flags_all['v8s_sf__FI'], flags_all['v8s_640__FI'], n_gt, idx)
        OUT['pairs'][f'v8s_1280__FI|v8s_640__FI|AP_{tag}'] = boot_ap_diff(flags_all['v8s_1280__FI'], flags_all['v8s_640__FI'], n_gt, idx)
    # the "small boxes only" merge: threshold chosen on val (AP50, then found)
    OUT['small_merge_val'] = {}
    vph = C.test_photos('val')
    vng = [len(g) for _, g, _ in vph]
    for m in ('v8s_640', 'v8s_sf'):
        vc = C.load_cache(m, 640, 'val')
        vfi = [coco_flags(C.merge(vc[f.name], 'FI'), g) for f, g, _ in vph]
        res = {'FI': round(ap_from_flags(vfi, vng), 2)}
        for px in C.SMALL_PX:
            fl = [coco_flags(C.merge(vc[f.name], 'FI+SMALL', small_px=px), g) for f, g, _ in vph]
            found = sum(len(P.match([tuple(x) for x in (lambda b: b[b[:, 0] >= 0.25])(C.merge(vc[f.name], 'FI+SMALL', small_px=px))], g))
                        for f, g, _ in vph)
            res[px] = {'AP50': round(ap_from_flags(fl, vng), 2), 'found': found}
        res['found_FI'] = sum(len(P.match([tuple(x) for x in (lambda b: b[b[:, 0] >= 0.25])(C.merge(vc[f.name], 'FI'))], g))
                              for f, g, _ in vph)
        res['plates'] = sum(vng)
        best = max(C.SMALL_PX, key=lambda px: (res[px]['AP50'], res[px]['found']))
        res['best_px'] = best
        OUT['small_merge_val'][m] = res
        print('val', m, res, flush=True)
    # every slicing setting of a model scored on val too: the val-best one is the honest "best SAHI" for the test
    OUT['val_scores'] = {}
    for m in ('v8s_640', 'v8s_sf'):
        vcs = {s: C.load_cache(m, s, 'val') for s in (640, 512, 960)}
        rows = {}
        for v in [v for v in C.VARIANTS if v['model'] == m]:
            vc = vcs[v.get('size') or 640]
            b = [C.merge(vc[f.name], v['mode'], v.get('rule', 'NMS_IOU'), small_px=v.get('small_px')) for f, _, _ in vph]
            fl = [coco_flags(x, g) for x, (_, g, _) in zip(b, vph)]
            found = sum(len(P.match([tuple(z) for z in x[x[:, 0] >= 0.25]], g)) for x, (_, g, _) in zip(b, vph))
            fp = sum(int((x[:, 0] >= 0.25).sum()) for x in b) - found
            rows[v['key']] = {'AP50': round(ap_from_flags(fl, vng), 2), 'found_%': round(100 * found / sum(vng), 2),
                              'correct_%': round(100 * found / max(1, found + fp), 2)}
        sl = [k for k in rows if '__SAHI' in k or '+SMALL' in k]
        rows['best_slicing'] = max(sl, key=lambda k: (rows[k]['AP50'], rows[k]['found_%']))
        OUT['val_scores'][m] = rows
        print('val best', m, rows['best_slicing'], rows[rows['best_slicing']], 'FI', rows[f'{m}__FI'], flush=True)
        k, b = rows['best_slicing'], f'{m}__FI'
        fa = np.array([p['found'] for p in per_plate[k]]); fb = np.array([p['found'] for p in per_plate[b]])
        na = np.array([p['any_box'] for p in ph_all[k] if p['no_plate']]); nb = np.array([p['any_box'] for p in ph_all[b] if p['no_plate']])
        sm = lambda kk: np.array([p['found'] for p in per_plate[kk] if p['file'] in sub['small_plates']])
        OUT['pairs'][f'{k}|{b}'] = {'found': mcnemar(fa, fb), 'false_alarm': mcnemar(na, nb),
                                    'small_plates': mcnemar(sm(k), sm(b)), 'AP50': boot_ap_diff(flags_all[k], flags_all[b], n_gt)}
    for m in ('v8s_640', 'v8s_sf'):
        k = f"{m}__FI+SMALL{OUT['small_merge_val'][m]['best_px']}"
        fa = np.array([p['found'] for p in per_plate[k]]); fb = np.array([p['found'] for p in per_plate[f'{m}__FI']])
        sm = lambda kk: np.array([p['found'] for p in per_plate[kk] if p['file'] in sub['small_plates']])
        OUT['pairs'][f'{k}|{m}__FI'] = {'found': mcnemar(fa, fb), 'small_plates': mcnemar(sm(k), sm(f'{m}__FI')),
                                        'AP50': boot_ap_diff(flags_all[k], flags_all[f'{m}__FI'], n_gt)}
    # plates missed by every setting, and plates only slicing finds
    keys = [v['key'] for v in C.VARIANTS]
    F = np.array([[p['found'] for p in per_plate[k]] for k in keys])
    base = per_plate['v8s_640__FI']
    OUT['missed_by_all'] = [{'file': base[i]['file'], 'w640': round(base[i]['w640'], 1), 'src': base[i]['src']}
                            for i in np.where(F.sum(0) == 0)[0]]
    fi = F[keys.index('v8s_640__FI')]; sf = F[keys.index('v8s_640__SAHI_FI')]
    OUT['only_slicing_finds'] = [{'file': base[i]['file'], 'w640': round(base[i]['w640'], 1)} for i in np.where((sf == 1) & (fi == 0))[0]]
    OUT['only_fi_finds'] = [{'file': base[i]['file'], 'w640': round(base[i]['w640'], 1)} for i in np.where((sf == 0) & (fi == 1))[0]]
    w = np.array([p['w640'] for p in base])
    OUT['plate_w640'] = {'median': round(float(np.median(w)), 1), 'p10': round(float(np.percentile(w, 10)), 1),
                         'lt_24': int((w < 24).sum()), 'lt_40': int((w < 40).sum()), 'lt_64': int((w < 64).sum()), 'n': len(w),
                         'hist': np.histogram(w, bins=[0, 16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 200, 260, 400, 700])[0].tolist()}
    OUT['photo_sizes'] = {'gt_640': int(sum(max(wh) > 640 for _, _, wh in photos)), 'n': len(photos),
                          'slices_640_mean': round(float(np.mean([len(caches[('v8s_640', 640)][f.name]['slices']) for f, _, _ in photos])), 2),
                          'slices_512_mean': round(float(np.mean([len(caches[('v8s_640', 512)][f.name]['slices']) for f, _, _ in photos])), 2),
                          'slices_960_mean': round(float(np.mean([len(caches[('v8s_640', 960)][f.name]['slices']) for f, _, _ in photos])), 2)}
    # e2e reads
    E = json.load(open(C.HERE / 'e2e_reads.json', encoding='utf-8'))
    OUT['e2e'] = {k: {'plate_exact_%': round(100 * np.mean([x['text'] == x['gt'] for x in v]), 2),
                      'right': int(sum(x['text'] == x['gt'] for x in v)), 'n': len(v),
                      'detected': int(sum(x['detected'] for x in v))} for k, v in E.items()}
    ok = lambda k: np.array([x['text'] == x['gt'] for x in E[k]], int)
    OUT['e2e_pairs'] = {f'{a}|{b}': mcnemar(ok(a), ok(b)) for a, b in
                        [('v8s_640__SAHI_FI', 'v8s_640__FI'), ('v8s_640__SAHI', 'v8s_640__FI'), ('v8s_sf__FI', 'v8s_640__FI'),
                         ('v8s_1280__FI', 'v8s_640__FI'), ('v8s_640__SAHI_FI__GNMM_IOS', 'v8s_640__FI'),
                         ('v8s_640__SAHI_FI__BORDER', 'v8s_640__FI'), ('v8s_sf__SAHI_FI', 'v8s_sf__FI')]}
    for m in ('v8s_640', 'v8s_sf'):
        k = OUT['val_scores'][m]['best_slicing']
        OUT['e2e_pairs'][f'{k}|{m}__FI'] = mcnemar(ok(k), ok(f'{m}__FI'))
    # e2e: how often the most confident box is only a part of the plate (CRNN then reads half a plate)
    OUT['e2e_top_box'] = {}
    for k in ('v8s_640__FI', 'v8s_640__SAHI_FI', OUT['val_scores']['v8s_640']['best_slicing']):
        sizes = {k2: v2 for k2, v2 in ((r['file'], r) for r in E[k])}
        ref = {r['file']: r for r in E['v8s_640__FI']}
        narrower = sum(1 for f, r in sizes.items() if r['box'] and ref[f]['box'] and
                       (r['box'][2] - r['box'][0]) < 0.8 * (ref[f]['box'][2] - ref[f]['box'][0]))
        OUT['e2e_top_box'][k] = {'narrower_than_FI_box': narrower}
    # why SAHI hurts e2e: top box changed to a part of the plate
    OUT['valstats'] = valstats()
    OUT['timing'] = json.load(open(C.HERE / 'timing.json', encoding='utf-8')) if (C.HERE / 'timing.json').exists() else None
    OUT['visdrone'] = json.load(open(C.HERE / 'visdrone_extra.json', encoding='utf-8'))
    OUT['check'] = json.load(open(CHECK / 'results.json', encoding='utf-8'))
    (C.HERE / 'analysis.json').write_text(json.dumps(OUT, indent=1, default=float), encoding='utf-8')
    # per-plate table for the upload (found by every setting)
    with open(C.HERE / 'per_plate.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(['file', 'plate', 'source', 'w640', 'h640'] + keys)
        for i, p in enumerate(base):
            wr.writerow([p['file'], p['j'], p['src'], round(p['w640'], 1), round(p['h640'], 1)] + [int(F[k, i]) for k in range(len(keys))])
    print('DONE')


if __name__ == '__main__':
    main()
