"""Paper #2 report, v1.1 run: numbers that the kit does not print, computed from `After train/results.zip`.

Reads the run's results.json, e2e_preds_*.csv, ocr_preds_test.csv and results.csv straight from the zip (nothing is
executed from it) and writes analysis.json next to this file. Scoring uses our own kit (docs/_kit/lprkit), the same
code the run used. v1.0 numbers are from the 19 Oct Kaggle log (paper02-r.log), quoted in the first report.
"""
import csv
import io
import json
import math
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ZIP = os.path.join(PAPER, 'After train', 'results.zip')
sys.path.insert(0, os.path.join(PAPER, '..', '..', '_kit', 'lprkit'))
import metrics  # noqa: E402
import vocab  # noqa: E402

z = zipfile.ZipFile(ZIP)
J = lambda p: json.loads(z.read(p).decode('utf-8'))
C = lambda p: list(csv.DictReader(io.StringIO(z.read(p).decode('utf-8'))))
N = vocab.norm_text

# v1.0 run (19 Oct, Kaggle log): same code and e2e photos, LPNR trained on the v1.0 crops
V10 = {'lpd': {'P': 96.7, 'R': 96.2, 'mAP50': 98.9, 'mAP50-95': 46.2},
       'lpnr': {'mAP50': 92.6, 'mAP50-95': 63.9, 'ocr_exact': 83.56, 'blpd_on_ocr': 72.50},
       'ours_veh': {'exact': 83.33, 'char': 92.59, 'lev': 90.57, 'found': 93.14, 'ms': 85.8,
                    'blpd_ext': 91.67, 'night': 63.33},
       'ours_noveh': {'exact': 82.84, 'char': 93.57, 'ms': 31.5, 'blpd_ext': 90.97, 'night': 63.33},
       'auth_veh': {'exact': 78.43, 'char': 88.18, 'lev': 84.65, 'found': 88.24, 'ms': 91.2,
                    'blpd_ext': 95.14, 'night': 38.33},
       'auth_noveh': {'exact': 77.94, 'char': 88.84, 'ms': 36.3, 'blpd_ext': 93.75, 'night': 40.0},
       'cond_ocr': {'all': 83.56, 'clear': 86.67, 'night': 80.0, 'blurry': 72.32, 'fog': 76.92, 'rain': 69.43},
       'train_min': {'lpd': 182.9, 'lpnr': 93.1}}


def fields(text):
    """district words, metro?, letter, digits of a plate text (canonical form)."""
    t = vocab.text_tokens(text or '')
    kind = vocab.KIND
    return ([x for x in t if kind.get(x) == 'district'], [x for x in t if kind.get(x) == 'metro'],
            [x for x in t if kind.get(x) == 'letter'], [x for x in t if kind.get(x) == 'digit'])


def error_type(pred, true):
    if not (pred or '').strip():
        return 'no reading'
    pd, pm, pl, pg = fields(pred)
    td, tm, tl, tg = fields(true)
    bad = []
    if pd != td:
        bad.append('district')
    if bool(pm) != bool(tm):
        bad.append('metro')
    if pl != tl:
        bad.append('letter')
    if pg != tg:
        bad.append('digits')
    return '+'.join(bad) or 'other'


def mcnemar(b, c):
    """exact two-sided McNemar p-value: b, c = discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


out = {'v10': V10}
R = {r: J(f'runs/{r}/results.json') for r in ('lpd_v12s', 'lpnr_v12n', 'cascade_ours', 'cascade_blpd_weights')}
out['env'] = J('environment.json')
out['results'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('settings', 'env')} for k, v in R.items()}

# ---- end to end: per photo -------------------------------------------------------------------------------------
E = {}
for run in ('cascade_ours', 'cascade_blpd_weights'):
    for key in ('with_vehicle_stage', 'plate_detector_only'):
        E[run, key] = {r['file']: r for r in C(f'runs/{run}/e2e_preds_{key}.csv')}
files = sorted(E['cascade_ours', 'with_vehicle_stage'])
src = {f: E['cascade_ours', 'with_vehicle_stage'][f]['source'] for f in files}
ok = {k: {f: N(v[f]['pred']) == N(v[f]['true']) for f in files} for k, v in E.items()}
got = {k: {f: bool(v[f]['pred'].strip()) for f in files} for k, v in E.items()}

e2e = {}
for (run, key), rows in E.items():
    tag = f'{run}|{key}'
    d = {}
    for s in ('all', 'blpd_ext', 'nightlpbd'):
        fs = [f for f in files if s == 'all' or src[f] == s]
        errs = {}
        for f in fs:
            if not ok[run, key][f]:
                t = error_type(rows[f]['pred'], rows[f]['true'])
                errs[t] = errs.get(t, 0) + 1
        d[s] = {'photos': len(fs), 'exact': sum(ok[run, key][f] for f in fs),
                'read_something': sum(got[run, key][f] for f in fs), 'errors': dict(sorted(errs.items(), key=lambda x: -x[1]))}
    e2e[tag] = d
out['e2e'] = e2e

# vehicle stage: photos read without it but lost with it (and the reverse)
veh = {}
for run in ('cascade_ours', 'cascade_blpd_weights'):
    a, b = ok[run, 'with_vehicle_stage'], ok[run, 'plate_detector_only']
    lost = [f for f in files if b[f] and not a[f]]
    won = [f for f in files if a[f] and not b[f]]
    ga, gb = got[run, 'with_vehicle_stage'], got[run, 'plate_detector_only']
    veh[run] = {'right_only_without_vehicle': len(lost), 'right_only_with_vehicle': len(won),
                'lost_by_source': {s: sum(src[f] == s for f in lost) for s in ('blpd_ext', 'nightlpbd')},
                'no_reading_with_vehicle_but_reading_without': sum(gb[f] and not ga[f] for f in files),
                'mcnemar_p': round(mcnemar(len(lost), len(won)), 4)}
out['vehicle_stage'] = veh

# ours vs authors' weights on the same photos
cmp = {}
for key in ('with_vehicle_stage', 'plate_detector_only'):
    o, a = ok['cascade_ours', key], ok['cascade_blpd_weights', key]
    for s in ('all', 'blpd_ext', 'nightlpbd'):
        fs = [f for f in files if s == 'all' or src[f] == s]
        b = sum(o[f] and not a[f] for f in fs)
        c = sum(a[f] and not o[f] for f in fs)
        cmp[f'{key}|{s}'] = {'only_ours': b, 'only_authors': c, 'both': sum(o[f] and a[f] for f in fs),
                             'neither': sum(not o[f] and not a[f] for f in fs), 'mcnemar_p': round(mcnemar(b, c), 5)}
out['ours_vs_authors'] = cmp

# ---- plate reader on the 651 OCR test crops ----------------------------------------------------------------------
P = C('runs/lpnr_v12n/ocr_preds_test.csv')
right = [N(p['pred']) == N(p['true']) for p in P]
valid = [p['valid'] == 'True' for p in P]
xs_right = [N(p['pred_xsort']) == N(p['true']) for p in P]
errs = {}
for p, r in zip(P, right):
    if not r:
        t = error_type(p['pred'], p['true'])
        errs[t] = errs.get(t, 0) + 1
out['ocr'] = {
    'plates': len(P), 'exact': sum(right), 'xsort_exact': sum(xs_right),
    'valid': sum(valid), 'valid_and_right': sum(v and r for v, r in zip(valid, right)),
    'invalid': len(P) - sum(valid), 'invalid_and_wrong': sum((not v) and (not r) for v, r in zip(valid, right)),
    'errors': dict(sorted(errs.items(), key=lambda x: -x[1])),
    'non_blpd_exact': [sum(r for p, r in zip(P, right) if p['source'] != 'blpd'), sum(p['source'] != 'blpd' for p in P)],
}
w = [p for p, r in zip(P, right) if not r]
out['ocr']['wrong_conf_ge_0.9'] = sum(float(p['conf']) >= 0.9 for p in w)

# ---- training curves -------------------------------------------------------------------------------------------
curves = {}
for run in ('lpd_v12s', 'lpnr_v12n'):
    rows = C(f'runs/{run}/results.csv')
    g = lambda k: [float(r[k]) for r in rows]
    m50, m95 = g('metrics/mAP50(B)'), g('metrics/mAP50-95(B)')
    fit = [0.1 * a + 0.9 * b for a, b in zip(m50, m95)]   # Ultralytics fitness picks best.pt
    curves[run] = {'epoch': [int(float(r['epoch'])) for r in rows], 'mAP50': m50, 'mAP50-95': m95,
                   'train_box': g('train/box_loss'), 'val_box': g('val/box_loss'),
                   'train_cls': g('train/cls_loss'), 'val_cls': g('val/cls_loss'),
                   'best_epoch': 1 + max(range(len(fit)), key=fit.__getitem__),
                   'last': {'mAP50': m50[-1], 'mAP50-95': m95[-1]}, 'max_mAP50-95': max(m95),
                   'nan': any(math.isnan(x) for x in m50 + m95 + g('train/box_loss'))}
out['curves'] = curves

with open(os.path.join(HERE, 'analysis.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps({k: out[k] for k in ('e2e', 'vehicle_stage', 'ours_vs_authors', 'ocr')}, ensure_ascii=False, indent=1))
print({r: {k: v for k, v in c.items() if k in ('best_epoch', 'last', 'max_mAP50-95', 'nan')} for r, c in curves.items()})
