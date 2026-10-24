"""Paper #6 v1.1 run -> analysis.json (everything the report quotes that is not printed by the kit itself).

  python analysis.py <p06 run dir> <p02 run dir> <p24 run dir> <p08 run dir> <paper-6.log>

Run dirs = the results.zip of each paper's v1.1 run, unpacked (only runs/*/results.json and the prediction CSVs are
read). Local checks (local_check.json, det_per_plate.csv, e2e_cross_preds.csv, easyocr_check.json) are read from this
folder when present. Paired tests are exact McNemar tests on the same plates / photos (metrics.mcnemar_exact).
"""
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RUN, P02, P24, P08, LOG = sys.argv[1:6]
DATA = os.environ.get('DATASET_ROOT', r'D:\498R\dataset')
sys.path.insert(0, os.path.join(RUN, 'code'))
import metrics  # noqa: E402
import vocab  # noqa: E402

J = lambda *p: json.load(open(os.path.join(*p), encoding='utf-8'))
rd = lambda p: list(csv.DictReader(open(p, encoding='utf-8')))
nt = vocab.norm_text
A = {}


def mc(a_ok, b_ok):
    """a_ok, b_ok: dict key -> bool on the same items. Returns counts and the exact McNemar p."""
    keys = sorted(set(a_ok) & set(b_ok))
    b = sum(a_ok[k] and not b_ok[k] for k in keys)
    c = sum(b_ok[k] and not a_ok[k] for k in keys)
    return {'n': len(keys), 'a_%': round(100 * np.mean([a_ok[k] for k in keys]), 2),
            'b_%': round(100 * np.mean([b_ok[k] for k in keys]), 2), 'only_a': b, 'only_b': c,
            'p': metrics.mcnemar_exact(b, c)}


# ------------------------------------------------------------------------------------------------ run health
env = J(RUN, 'environment.json')
jobs = [{'name': m.group(1), 'code': int(m.group(2)), 'min': float(m.group(3))}
        for m in re.finditer(r'\[done\]\s+(\S+) code=(-?\d+) after ([\d.]+) min', open(LOG, encoding='utf-8').read())]
last = re.findall(r'^([\d.]+)s \d+ ', open(LOG, encoding='utf-8').read(), re.M)
A['run'] = {'dataset_version': env['dataset_version'], 'date': env['date'], 'torch': env['torch'],
            'ultralytics': env['ultralytics'], 'gpus': env['gpus'], 'jobs': jobs,
            'total_h': round(float(last[-1]) / 3600, 2)}

det = {}
for r in ('det_v8n', 'det_v8s'):
    d = J(RUN, 'runs', r, 'results.json')
    rows = rd(os.path.join(RUN, 'runs', r, 'results.csv'))
    k50, k95 = 'metrics/mAP50(B)', 'metrics/mAP50-95(B)'
    nan = sum('nan' in str(v).lower() for x in rows for v in x.values())
    det[r] = {'epochs_run': d['epochs_run'], 'epochs_planned': d['epochs_planned'], 'best_epoch': d['best_epoch'],
              'train_min': d['train_minutes'], 'params_M': d['params_M'], 'GFLOPs': d['GFLOPs'],
              'peak_gpu_mem_GB': d['peak_gpu_mem_GB'], 'nan_values': nan,
              'val': {k: round(100 * v, 2) for k, v in d['val'].items() if k != 'speed_ms'},
              'test': {k: round(100 * v, 2) for k, v in d['test'].items() if k != 'speed_ms'},
              'last_epoch_val': {'mAP50': round(100 * float(rows[-1][k50]), 2), 'mAP50-95': round(100 * float(rows[-1][k95]), 2)},
              'curve': {'epoch': [int(x['epoch']) for x in rows], 'mAP50': [float(x[k50]) for x in rows],
                        'mAP50-95': [float(x[k95]) for x in rows], 'train_box': [float(x['train/box_loss']) for x in rows],
                        'val_box': [float(x['val/box_loss']) for x in rows]},
              'plates': d['test_plates'], 'by_condition': d['test_by_condition'], 'speed': d['speed']}
A['det'] = det
# teammate's YOLOv8n on the same detection test (paper #3, Md. Yousuf, dataset v1.0: detection split unchanged in v1.1)
import zipfile  # noqa: E402
P03 = r'D:\498R\docs\results\paper03_results.zip'
if os.path.exists(P03):
    d = json.loads(zipfile.ZipFile(P03).read('runs/v8n_640/results.json'))
    A['p03_v8n_640'] = {'labels': d['labels'], 'epochs_run': d['epochs_run'], 'best_epoch': d['best_epoch'],
                        'test': {k: round(100 * v, 2) for k, v in d['test'].items() if k != 'speed_ms'},
                        'found_%': d['test_plates']['found_%'], 'train_min': d['train_minutes'],
                        'night_mAP50': round(100 * d['test_by_condition']['night']['mAP50'], 1)}

# ------------------------------------------------------------------------------------------------ EasyOCR on the crops
eo = J(RUN, 'runs', 'easyocr_crops', 'results.json')['ocr_test']
P = rd(os.path.join(RUN, 'runs', 'easyocr_crops', 'ocr_preds_test.csv'))
PP = {r['file']: r for r in rd(os.path.join(RUN, 'runs', 'easyocr_crops', 'ocr_preds_test_prep.csv'))}
ok = {r['file']: nt(r['pred']) == nt(r['true']) for r in P}
okp = {f: nt(r['pred']) == nt(r['true']) for f, r in PP.items()}
assert round(100 * np.mean(list(ok.values())), 2) == eo['rows_decoder']['plate_exact_%']


def lines(text):
    """(letters line, digits) of a reading after normalisation: digits = every digit in order, letters = the rest."""
    t = nt(text)
    return ''.join(c for c in t if c not in vocab.DIGITS), ''.join(c for c in t if c in vocab.DIGITS)


top_ok = [lines(r['pred'])[0] == lines(r['true'])[0] for r in P]
num_ok = [lines(r['pred'])[1] == lines(r['true'])[1] for r in P]
latin = [bool(re.search(r'[A-Za-z]', r['pred'])) for r in P]
ldig = [bool(re.search(r'[0-9]', r['pred'])) for r in P]
junk = [bool(re.search(r'[\[\]|!()\'"#%{}:;,.]', r['pred'])) for r in P]
empty = [not r['pred'].strip() for r in P]
conf = np.array([float(r['conf']) for r in P])
okv = np.array([ok[r['file']] for r in P])
valid = np.array([r['valid'] == 'True' for r in P])
hs = {}
for r in P:
    with Image.open(os.path.join(DATA, 'ocr', 'images', 'test', r['file'])) as im:
        hs[r['file']] = im.size
H = np.array([hs[r['file']][1] for r in P])
bins = [(0, 40), (40, 60), (60, 100), (100, 10 ** 6)]
by_h = {}
for lo, hi in bins:
    sel = [r for r, h in zip(P, H) if lo <= h < hi]
    by_h[f'{lo}-{hi if hi < 10 ** 6 else "max"}'] = dict(metrics.score_rows((r['pred'], r['true']) for r in sel))


def auc(score, label):
    pos, neg = score[label], score[~label]
    if not len(pos) or not len(neg):
        return None
    return round(float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])), 3)


thr = {}
for t in (0.5, 0.7, 0.9):
    s = conf >= t
    thr[str(t)] = {'kept_%': round(100 * s.mean(), 2), 'exact_among_kept_%': round(100 * okv[s].mean(), 2) if s.any() else None}
# which characters does EasyOCR miss most? (aligned on normalised strings)
miss, sub = Counter(), Counter()
import difflib  # noqa: E402
for r in P:
    a, b = nt(r['true']), nt(r['pred'])
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ('delete', 'replace'):
            miss.update(a[i1:i2])
        if op == 'replace' and i2 - i1 == j2 - j1:
            sub.update(zip(a[i1:i2], b[j1:j2]))
tok_true = Counter(c for r in P for c in nt(r['true']))
A['easyocr'] = {
    'kit': eo, 'n': len(P),
    'top_line_exact_%': round(100 * np.mean(top_ok), 2), 'digits_exact_%': round(100 * np.mean(num_ok), 2),
    'both_lines_%': round(100 * np.mean(okv), 2),
    'pred_with_latin_letters_%': round(100 * np.mean(latin), 2), 'pred_with_latin_digits_%': round(100 * np.mean(ldig), 2),
    'pred_with_brackets_or_punct_%': round(100 * np.mean(junk), 2), 'empty_%': round(100 * np.mean(empty), 2),
    'conf_auc': auc(conf, okv), 'conf_mean_right': round(float(conf[okv].mean()), 3),
    'conf_mean_wrong': round(float(conf[~okv].mean()), 3), 'conf_thresholds': thr,
    'valid_format_%': round(100 * valid.mean(), 2), 'exact_given_valid_%': round(100 * okv[valid].mean(), 2),
    'exact_given_invalid_%': round(100 * okv[~valid].mean(), 2),
    'crop_height_px': {'median': int(np.median(H)), 'min': int(H.min()), 'p10': int(np.percentile(H, 10)),
                       'under_24': int((H < 24).sum())},
    'by_crop_height': by_h,
    'prep_vs_raw': mc(okp, ok),
    'char_miss_rate_top': sorted(((c, n, round(100 * n / tok_true[c], 1)) for c, n in miss.items() if tok_true[c] >= 40),
                                 key=lambda x: -x[2])[:12],
    'top_substitutions': [(a, b, n) for (a, b), n in sub.most_common(10)],
    'examples': [{'true': r['true'], 'pred': r['pred'], 'conf': float(r['conf'])} for r in P[:12]],
}

if os.path.exists(os.path.join(HERE, 'easyocr_check.json')):     # local re-run + cheap fixes (easyocr_check.py)
    L = defaultdict(dict)
    for x in rd(os.path.join(HERE, 'easyocr_local_preds.csv')):
        L[x['variant']][x['file']] = nt(x['pred']) == nt(x['true'])
    A['easyocr_local'] = J(HERE, 'easyocr_check.json')
    A['easyocr_local']['paired_vs_local_raw'] = {v: mc(L[v], L['raw']) for v in L if v != 'raw'}
    A['easyocr_local']['local_raw_vs_kaggle_raw'] = mc(L['raw'], ok)

# ------------------------------------------------------------------------------------------------ same 651 crops: other readers
others = {'#2 lpnr_v12n (char-YOLO12n)': os.path.join(P02, 'runs', 'lpnr_v12n', 'ocr_preds_test.csv'),
          '#24 chars_v8n (char-YOLOv8n)': os.path.join(P24, 'runs', 'chars_v8n', 'ocr_preds_test.csv'),
          '#24 crnn (CRNN)': os.path.join(P24, 'runs', 'crnn', 'ocr_preds_test.csv'),
          '#24 vitstr (ViTSTR)': os.path.join(P24, 'runs', 'vitstr', 'ocr_preds_test.csv')}
same = {}
for name, f in others.items():
    o = {r['file']: nt(r['pred']) == nt(r['true']) for r in rd(f)}
    o_cer = metrics.score_rows((r['pred'], r['true']) for r in rd(f))
    same[name] = dict(mc(o, ok), char_acc_other=o_cer['char_acc_%'])
LP = {r['file']: r['pred'] for r in rd(others['#2 lpnr_v12n (char-YOLO12n)'])}
src = {r['file']: r['source'] for r in P}
pick = []                                          # example crops for the report: one of each typical case
for name, cond in (('right', lambda r: ok[r['file']] and src[r['file']] == 'anpr'),
                   ('Latin digits', lambda r: re.search(r'[0-9]', r['pred']) and lines(r['pred'])[0] == lines(r['true'])[0]),
                   ('metro word broken', lambda r: lines(r['pred'])[1] == lines(r['true'])[1] and not ok[r['file']]),
                   ('night', lambda r: src[r['file']] == 'nightlpbd' and not ok[r['file']] and r['pred'].strip()),
                   ('junk', lambda r: re.search(r'[\[\]|!]', r['pred']) and src[r['file']] == 'kuet')):
    r = next(r for r in P if cond(r))
    pick.append({'case': name, 'file': r['file'], 'true': r['true'], 'easyocr': r['pred'], 'conf': float(r['conf']),
                 'lpnr_v12n': LP[r['file']], 'lpnr_right': nt(LP[r['file']]) == nt(r['true'])})
A['easyocr']['picked_examples'] = pick
p8 = {r['file']: nt(r['pred']) == nt(r['true']) for r in rd(os.path.join(P08, 'runs', 'plates_cnn28', 'per_plate.csv'))
      if r['setting'] == 'gfpgan_fixed'}
if p8:
    same['#8 CNN + GFP-GAN + contours'] = mc(p8, ok)
A['same_651_crops'] = same            # a = the other reader, b = EasyOCR

# ------------------------------------------------------------------------------------------------ whole photos (204)
e2e = {}
E = {}
for r in ('e2e_v8n', 'e2e_v8s'):
    e2e[r] = J(RUN, 'runs', r, 'results.json')['e2e']
    E[r] = {x['file']: x for x in rd(os.path.join(RUN, 'runs', r, 'e2e_preds.csv'))}
okE = {r: {f: nt(x['pred']) == nt(x['true']) for f, x in E[r].items()} for r in E}
c2 = {x['file']: nt(x['pred']) == nt(x['true']) for x in rd(os.path.join(P02, 'runs', 'cascade_ours', 'e2e_preds_plate_detector_only.csv'))}
pairs = {'e2e_v8s vs e2e_v8n (EasyOCR)': mc(okE['e2e_v8s'], okE['e2e_v8n']),
         '#2 cascade (lpd_v12s + lpnr_v12n, Kaggle) vs e2e_v8n (EasyOCR)': mc(c2, okE['e2e_v8n'])}
nobox = {r: round(len(E[r]) * (1 - e2e[r]['plate_detected_%'] / 100)) for r in E}
empty = {r: sum(1 for x in E[r].values() if not x['pred'].strip()) - nobox[r] for r in E}   # box found, EasyOCR read nothing
if os.path.exists(os.path.join(HERE, 'e2e_cross_preds.csv')):
    X = defaultdict(dict)
    for x in rd(os.path.join(HERE, 'e2e_cross_preds.csv')):
        X[x['run']][x['file']] = nt(x['pred']) == nt(x['true'])
    pairs['det_v8n + lpnr_v12n (p06 crop) vs det_v8n + EasyOCR'] = mc(X['det_v8n+lpnr_v12n/p06_protocol'], okE['e2e_v8n'])
    pairs['det_v8s + lpnr_v12n (p06 crop) vs det_v8s + EasyOCR'] = mc(X['det_v8s+lpnr_v12n/p06_protocol'], okE['e2e_v8s'])
    pairs['#2 cascade with det_v8s vs #2 cascade with its lpd_v12s (local)'] = mc(X['det_v8s+lpnr_v12n/p02_cascade'],
                                                                                 X['p02_lpd_v12s+lpnr_v12n/p02_cascade'])
    pairs['#2 cascade with det_v8n vs with lpd_v12s (local)'] = mc(X['det_v8n+lpnr_v12n/p02_cascade'],
                                                                  X['p02_lpd_v12s+lpnr_v12n/p02_cascade'])
    pairs['local lpd_v12s cascade = Kaggle cascade (same plates right)'] = {
        'identical': all(X['p02_lpd_v12s+lpnr_v12n/p02_cascade'][f] == c2[f] for f in c2)}
# loss split for EasyOCR e2e: no box / box but wrong / right
A['e2e'] = {'kit': e2e, 'no_box_photos': nobox, 'paired': pairs,
            'loss_split': {r: {'no_box': nobox[r], 'box_but_empty_read': empty[r],
                               'read_wrong': sum(1 for f, v in okE[r].items() if not v) - nobox[r] - empty[r],
                               'read_right': sum(okE[r].values())} for r in E}}

# ------------------------------------------------------------------------------------------------ detectors on the same plates (local)
if os.path.exists(os.path.join(HERE, 'det_per_plate.csv')):
    D = defaultdict(dict)
    R = defaultdict(lambda: defaultdict(list))
    for x in rd(os.path.join(HERE, 'det_per_plate.csv')):
        D[x['model']][(x['file'], x['plate'])] = x['found'] == '1'
        if x['found'] == '1':
            R[x['model']][x['source']].append((float(x['best_iou']), float(x['w_ratio']), float(x['h_ratio'])))
    A['det_paired'] = {'v8s vs v8n': mc(D['det_v8s'], D['det_v8n']),
                       '#2 lpd_v12s vs v8n': mc(D['p02_lpd_v12s'], D['det_v8n']),
                       '#2 lpd_v12s vs v8s': mc(D['p02_lpd_v12s'], D['det_v8s'])}
    nl = defaultdict(dict)
    for x in rd(os.path.join(HERE, 'det_per_plate.csv')):
        if x['source'] == 'nightlpbd':
            nl[x['model']][(x['file'], x['plate'])] = x['found'] == '1'
    A['det_paired']['night: #2 lpd_v12s vs v8n'] = mc(nl['p02_lpd_v12s'], nl['det_v8n'])
    A['det_box_shape'] = {m: {s: {'mean_iou': round(100 * np.mean([v[0] for v in L]), 1),
                                  'median_w_ratio': round(float(np.median([v[1] for v in L])), 3),
                                  'median_h_ratio': round(float(np.median([v[2] for v in L])), 3), 'n': len(L)}
                              for s, L in sorted(R[m].items())} for m in R}

json.dump(A, open(os.path.join(HERE, 'analysis.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in A['easyocr'].items() if k not in ('kit', 'examples')}, ensure_ascii=False, indent=1)[:3500])
print(json.dumps(A['same_651_crops'], ensure_ascii=False, indent=1))
print(json.dumps(A['e2e']['paired'], ensure_ascii=False, indent=1), A['e2e']['loss_split'])
print(json.dumps(A.get('det_paired'), indent=1))
print(A['run'])
