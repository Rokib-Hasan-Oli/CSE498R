"""Paper #34 analysis of the v1.1 Kaggle run -> analysis.json (+ analysis_log.txt). Pure CPU, no model is loaded.

  python analysis.py <unpacked results.zip> <fusion dir>

<fusion dir> holds chars_v8n/ (model fusion run, v1.1, 22 Oct) and lpnr_v12n/ (cascaded YOLO run, v1.1, 21 Oct): their
ocr_preds_val.csv / ocr_preds_test.csv / results.json, and out/fusion34/results.json written by the kit's
p24_fusion.py over chars_v8n, lpnr_v12n, crnn, svtrv2_t_msr, svtrv2_s_msr. local_check.json (local_check.py) must exist.
Every p value is an exact two-sided McNemar test on the same plates; every CI is a 95 % bootstrap over plates
(5,000 resamples, seed 0).
"""
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

from PIL import Image

RUN, FUSE = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
sys.path.insert(0, os.path.join(RUN, 'code'))
sys.path.insert(0, 'D:/498R/docs/_kit/extras')
import metrics  # noqa: E402
import vocab  # noqa: E402
import p24_fusion as F  # noqa: E402

LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    LOG.append(s)
    print(s.encode('ascii', 'replace').decode())


SV = ['svtrv2_t_msr', 'svtrv2_s_msr', 'svtrv2_t_fixed', 'svtrv2_t_scratch']
OURS = SV + ['crnn']
OTHER = ['chars_v8n', 'lpnr_v12n']
ALL = OURS + OTHER
ok = lambda p, t: vocab.norm_text(p or '') == vocab.norm_text(t)
A = {}


def read_csv(p):
    with open(p, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def run_dir(r):
    return os.path.join(RUN, 'runs', r) if r in OURS else os.path.join(FUSE, r)


PRED = {s: {r: {x['file']: x for x in read_csv(os.path.join(run_dir(r), f'ocr_preds_{s}.csv'))} for r in ALL}
        for s in ('test', 'val')}
RES = {r: json.load(open(os.path.join(run_dir(r), 'results.json'), encoding='utf-8')) for r in ALL}
GT = {s: read_csv(os.path.join(DATA, 'ocr', f'{s}.csv')) for s in ('train', 'val', 'test')}
FILES = {s: [g['file'] for g in GT[s]] for s in ('val', 'test')}
for s in ('val', 'test'):
    for r in ALL:
        assert set(PRED[s][r]) == set(FILES[s]), (s, r)
TRUE = {s: {g['file']: g['text'] for g in GT[s]} for s in GT}
SRC = {s: {g['file']: g['source'] for g in GT[s]} for s in GT}
OK = {s: {r: [ok(PRED[s][r][f]['pred'], TRUE[s][f]) for f in FILES[s]] for r in ALL} for s in ('val', 'test')}


def boot(v, n=5000, seed=0):
    rnd, N = random.Random(seed), len(v)
    xs = sorted(sum(v[rnd.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return [round(100 * xs[int(0.025 * n)], 2), round(100 * xs[int(0.975 * n) - 1], 2)]


def boot_diff(a, b, n=5000, seed=0):
    rnd, N = random.Random(seed), len(a)
    d = [x - y for x, y in zip(a, b)]
    xs = sorted(sum(d[rnd.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return [round(100 * xs[int(0.025 * n)], 2), round(100 * xs[int(0.975 * n) - 1], 2)]


def pair(a, b, idx=None):
    """a vs b on the same plates (idx = subset of test indices)."""
    va, vb = OK['test'][a], OK['test'][b]
    if idx is not None:
        va, vb = [va[i] for i in idx], [vb[i] for i in idx]
    oa = sum(x and not y for x, y in zip(va, vb))
    ob = sum(y and not x for x, y in zip(va, vb))
    return {'n': len(va), 'a': round(100 * sum(va) / len(va), 2), 'b': round(100 * sum(vb) / len(vb), 2),
            'diff_pp': round(100 * (sum(va) - sum(vb)) / len(va), 2), 'only_a': oa, 'only_b': ob,
            'p': metrics.mcnemar_exact(oa, ob), 'ci95_diff': boot_diff(va, vb) if idx is None else None}


# ------------------------------------------------------------------------------------------------- 1. run health
logtxt = open(os.path.join(PAPER, 'After Train', 'paper-34.log'), encoding='utf-8').read()
jobs = {m.group(1): float(m.group(2)) for m in re.finditer(r'\[done\]\s+(\S+) code=0 after ([\d.]+) min', logtxt)}
total_s = max(float(m.group(1)) for m in re.finditer(r'^([\d.]+)s ', logtxt, re.M))
env = json.load(open(os.path.join(RUN, 'environment.json'), encoding='utf-8'))
health = {'dataset_version': env['dataset_version'], 'torch': env['torch'], 'gpus': env['gpus'], 'jobs_code0': jobs,
          'n_jobs_code0': len(jobs), 'total_h': round(total_s / 3600, 2), 'planned_h': 2.5,
          'nan_in_logs': sum(open(os.path.join(RUN, 'logs', f'{r}.log'), encoding='utf-8').read().lower().count('nan')
                             for r in OURS),
          'time_cap_hits': {r: ('[time cap]' in open(os.path.join(RUN, 'logs', f'{r}.log'), encoding='utf-8').read())
                            for r in OURS}}
for r in OURS:
    lg = RES[r]['log']
    b = min(lg, key=lambda x: x['val_loss'])
    health[r] = {'epochs_run': len(lg), 'epochs_planned': RES[r].get('epochs_planned', RES[r]['settings'].get('epochs')),
                 'best_epoch': b['epoch'], 'best_val_loss': b['val_loss'], 'last_val_loss': lg[-1]['val_loss'],
                 'min_train_loss': min(x['train_loss'] for x in lg), 'train_min': RES[r]['train_minutes'],
                 'init': RES[r]['settings'].get('init', 'from scratch (CRNN)'),
                 's_per_epoch': round(60 * RES[r]['train_minutes'] / len(lg), 1)}
    # one-cycle LR (torch OneCycleLR defaults: div 25, final div 1e4, cos) at the stop and at the best epoch
    if r in SV:
        E, pct = RES[r]['epochs_planned'], 0.075
        def lr_at(ep):
            f = ep / E
            if f <= pct:
                return 1 / 25 + (1 - 1 / 25) * (1 - math.cos(math.pi * f / pct)) / 2
            g = (f - pct) / (1 - pct)
            lo = 1 / 25 / 1e4
            return lo + (1 - lo) * (1 + math.cos(math.pi * g)) / 2
        health[r]['lr_share_at_stop'] = round(lr_at(len(lg)), 3)
        health[r]['lr_share_at_best'] = round(lr_at(b['epoch']), 3)
        health[r]['schedule_done_%'] = round(100 * len(lg) / E, 1)
A['health'] = health
log('health', json.dumps({k: v for k, v in health.items() if k not in OURS}, ensure_ascii=False))
for r in OURS:
    log(' ', r, health[r])

# -------------------------------------------------------------------------------------- 2. main table + CIs
main = {}
for r in ALL:
    o = RES[r].get('ocr_test') or {}
    sc = metrics.score_rows((PRED['test'][r][f]['pred'], TRUE['test'][f]) for f in FILES['test'])
    sv = metrics.score_rows((PRED['val'][r][f]['pred'], TRUE['val'][f]) for f in FILES['val'])
    main[r] = {'test': sc, 'val': sv, 'ci95': boot(OK['test'][r]), 'ms_kaggle': o.get('ms_per_plate'),
               'params_M': RES[r].get('params_M')}
    if r in OURS:
        assert abs(sc['plate_exact_%'] - o['rows_decoder']['plate_exact_%']) < 0.01, r
A['main'] = main
for r in ALL:
    log('main', r, main[r]['test']['plate_exact_%'], main[r]['ci95'], 'val', main[r]['val']['plate_exact_%'],
        'CER', main[r]['test']['CER_%'], 'ms', main[r]['ms_kaggle'])

# ---------------------------------------------------------------------------------------------- 3. paired tests
PAIRS = [('svtrv2_t_msr', 'crnn'), ('svtrv2_s_msr', 'crnn'), ('svtrv2_s_msr', 'svtrv2_t_msr'),
         ('svtrv2_t_msr', 'svtrv2_t_fixed'), ('svtrv2_t_msr', 'svtrv2_t_scratch'), ('svtrv2_t_fixed', 'crnn'),
         ('svtrv2_t_scratch', 'crnn'), ('chars_v8n', 'svtrv2_t_msr'), ('lpnr_v12n', 'svtrv2_t_msr'),
         ('chars_v8n', 'svtrv2_s_msr'), ('lpnr_v12n', 'svtrv2_s_msr')]
A['pairs'] = {f'{a}/{b}': pair(a, b) for a, b in PAIRS}
for k, v in A['pairs'].items():
    log('pair', k, v)

# ------------------------------------------------------------------------- 4. aspect ratio, MSR sizes (Table 1)
def size(split, f):
    with Image.open(os.path.join(DATA, 'ocr', 'images', split, f)) as im:
        return im.size        # (w, h)


def msr(R):
    return '64x64' if R < 1.5 else '48x96' if R < 2.5 else '40x112' if R < 3.5 else '32x128'


AR = {s: {g['file']: (lambda wh: wh[0] / wh[1])(size(s, g['file'])) for g in GT[s]} for s in ('train', 'val', 'test')}
WH = {f: size('test', f) for f in FILES['test']}
for f in FILES['test']:
    assert PRED['test']['svtrv2_t_msr'][f]['bucket'] == msr(AR['test'][f]), f
bins = [0, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 10]
A['aspect'] = {s: {'n': len(AR[s]), 'median': round(sorted(AR[s].values())[len(AR[s]) // 2], 3),
                   'buckets': dict(Counter(msr(x) for x in AR[s].values())),
                   'hist': [sum(bins[i] <= x < bins[i + 1] for x in AR[s].values()) for i in range(len(bins) - 1)]}
               for s in AR}
A['aspect']['bins'] = bins
A['aspect']['test_by_source'] = {src: {'n': n, 'median_R': round(sorted(AR['test'][f] for f in FILES['test'] if SRC['test'][f] == src)[n // 2], 2),
                                       'buckets': dict(Counter(msr(AR['test'][f]) for f in FILES['test'] if SRC['test'][f] == src))}
                                 for src, n in Counter(SRC['test'].values()).items()}
A['crop_px_test'] = {'median_w': sorted(w for w, h in WH.values())[325], 'median_h': sorted(h for w, h in WH.values())[325],
                     'h_below_32': sum(h < 32 for w, h in WH.values())}
log('aspect', A['aspect'])
idx_b = defaultdict(list)
for i, f in enumerate(FILES['test']):
    idx_b[msr(AR['test'][f])].append(i)
A['by_msr_size'] = {}
for b, idx in sorted(idx_b.items()):
    A['by_msr_size'][b] = {'n': len(idx), **{r: round(100 * sum(OK['test'][r][i] for i in idx) / len(idx), 2) for r in ALL},
                           'msr_vs_fixed': pair('svtrv2_t_msr', 'svtrv2_t_fixed', idx),
                           'msr_vs_crnn': pair('svtrv2_t_msr', 'crnn', idx),
                           'pre_vs_scratch': pair('svtrv2_t_msr', 'svtrv2_t_scratch', idx)}
    log('size', b, A['by_msr_size'][b])
A['paper_table1'] = {'R1 (<1.5) 64x64': {'msr': 87.4, 'fixed': 72.1}, 'R2 (1.5-2.5) 48x96': {'msr': 88.3, 'fixed': 83.1},
                     'R3 (2.5-3.5) 40x112': {'msr': 86.1, 'fixed': 84.1}, 'R4 (>=3.5)': {'msr': 87.5, 'fixed': 85.6}}
# finer: by aspect-ratio bin, T-MSR vs T-fixed vs CRNN (only bins with >= 20 plates)
A['by_ratio_bin'] = []
for i in range(len(bins) - 1):
    idx = [k for k, f in enumerate(FILES['test']) if bins[i] <= AR['test'][f] < bins[i + 1]]
    if len(idx) >= 20:
        A['by_ratio_bin'].append({'bin': [bins[i], bins[i + 1]], 'n': len(idx),
                                  **{r: round(100 * sum(OK['test'][r][k] for k in idx) / len(idx), 2) for r in ALL}})

# ------------------------------------------------------------------------------- 5. sources, conditions, extras
A['by_source'] = {}
for src in sorted(set(SRC['test'].values())):
    idx = [i for i, f in enumerate(FILES['test']) if SRC['test'][f] == src]
    A['by_source'][src] = {'n': len(idx), **{r: round(100 * sum(OK['test'][r][i] for i in idx) / len(idx), 2) for r in ALL},
                           't_vs_crnn': pair('svtrv2_t_msr', 'crnn', idx)}
A['val_by_source'] = {}
for src in sorted(set(SRC['val'].values())):
    idx = [i for i, f in enumerate(FILES['val']) if SRC['val'][f] == src]
    A['val_by_source'][src] = {'n': len(idx), **{r: round(100 * sum(OK['val'][r][i] for i in idx) / len(idx), 2) for r in ALL}}
A['source_share'] = {s: dict(Counter(SRC[s].values())) for s in ('train', 'val', 'test')}
log('by_source', A['by_source'])
log('val_by_source', A['val_by_source'])
cond = {g['file']: g for g in read_csv(os.path.join(DATA, 'ocr', 'test_conditions.csv'))}
A['by_condition'] = {}
for c in ('clear', 'night', 'blurry', 'low_res'):
    idx = [i for i, f in enumerate(FILES['test']) if cond[f][c] == '1']
    if idx:
        A['by_condition'][c] = {'n': len(idx), **{r: round(100 * sum(OK['test'][r][i] for i in idx) / len(idx), 2) for r in ALL},
                                't_vs_crnn': pair('svtrv2_t_msr', 'crnn', idx), 's_vs_crnn': pair('svtrv2_s_msr', 'crnn', idx),
                                'msr_vs_fixed': pair('svtrv2_t_msr', 'svtrv2_t_fixed', idx)}
A['extra_tests'] = {r: {k: v.get('plate_exact_%') for k, v in (RES[r].get('ocr_test') or {}).get('extra_tests', {}).items()}
                    for r in ALL}
A['extra_n'] = {k: len(read_csv(os.path.join(DATA, 'ocr', f'{k}.csv'))) for k in ('test_fog', 'test_rain_synth')}
log('by_condition', {c: {r: v[r] for r in ALL} for c, v in A['by_condition'].items()})
log('extra', A['extra_tests'], A['extra_n'])

# ------------------------------------------------------------------------------------------- 6. error anatomy
def kinds(text):
    tk = metrics.tokens(text)
    return {k: [t for t in tk if vocab.KIND.get(t, '?') == k] for k in ('district', 'metro', 'letter', 'digit', '?')}


def anatomy(r, split='test'):
    out = Counter()
    for f in FILES[split]:
        p, t = PRED[split][r][f]['pred'], TRUE[split][f]
        if ok(p, t):
            continue
        out['wrong'] += 1
        kp, kt = kinds(p), kinds(t)
        valid = metrics.is_valid_plate(p)
        out['invalid_layout'] += not valid
        if kp['district'] != kt['district']:
            out['district'] += 1
        if len(kp['metro']) != len(kt['metro']):
            out['metro'] += 1
        if kp['letter'] != kt['letter']:
            out['letter'] += 1
        dp, dt = ''.join(kp['digit']), ''.join(kt['digit'])
        if dp != dt:
            out['digits'] += 1
            if len(dp) > len(dt):
                out['digits_extra'] += 1
            elif len(dp) < len(dt):
                out['digits_missing'] += 1
            else:
                out['digits_substituted'] += 1
            # a doubled digit that CTC merged / split: truth has a repeat, prediction lost or gained one copy
            if len(dp) == len(dt) - 1 and any(dt[i] == dt[i + 1] for i in range(len(dt) - 1)):
                out['digit_repeat_lost'] += 1
        if p.strip() == '':
            out['empty'] += 1
    right_invalid = sum(ok(PRED[split][r][f]['pred'], TRUE[split][f]) and not metrics.is_valid_plate(PRED[split][r][f]['pred'])
                        for f in FILES[split])
    out['right_but_invalid'] = right_invalid
    out['invalid_total'] = out['invalid_layout'] + right_invalid
    return dict(out)


A['errors'] = {r: anatomy(r) for r in ALL}
A['errors_val'] = {r: anatomy(r, 'val') for r in ['svtrv2_t_msr', 'svtrv2_s_msr', 'crnn', 'chars_v8n', 'lpnr_v12n']}
truth_invalid = sum(not metrics.is_valid_plate(TRUE['test'][f]) for f in FILES['test'])
A['truth_invalid_test'] = truth_invalid
for r in ALL:
    log('errors', r, A['errors'][r])
# digits with a repeat in the truth (e.g. ৪৪, ৬৬): CTC needs a blank between two equal tokens
rep = [i for i, f in enumerate(FILES['test']) if any(a == b for a, b in zip(kinds(TRUE['test'][f])['digit'],
                                                                             kinds(TRUE['test'][f])['digit'][1:]))]
A['repeat_digits'] = {'n': len(rep), **{r: round(100 * sum(OK['test'][r][i] for i in rep) / len(rep), 2) for r in ALL}}
norep = [i for i in range(len(FILES['test'])) if i not in set(rep)]
A['no_repeat_digits'] = {'n': len(norep), **{r: round(100 * sum(OK['test'][r][i] for i in norep) / len(norep), 2) for r in ALL}}
log('repeat digits', A['repeat_digits'], A['no_repeat_digits'])

# ---------------------------------------------------------------------- 7. confidence: accuracy vs coverage
def coverage(r, split='test'):
    xs = sorted(((float(PRED[split][r][f]['conf'] or 0), ok(PRED[split][r][f]['pred'], TRUE[split][f])) for f in FILES[split]),
                key=lambda x: -x[0])
    n = len(xs)
    acc = lambda c: round(100 * sum(o for _, o in xs[:max(1, int(round(c * n)))]) / max(1, int(round(c * n))), 2)
    aurc = sum(1 - sum(o for _, o in xs[:k]) / k for k in range(1, n + 1)) / n
    curve = [[round(k / n, 4), round(100 * sum(o for _, o in xs[:k]) / k, 2)] for k in range(10, n + 1, 10)]
    return {'acc_at_50': acc(0.5), 'acc_at_80': acc(0.8), 'acc_at_90': acc(0.9), 'aurc_%': round(100 * aurc, 2),
            'curve': curve}


A['coverage'] = {r: coverage(r) for r in ALL}
for r in ALL:
    log('coverage', r, {k: v for k, v in A['coverage'][r].items() if k != 'curve'})

# -------------------------------------------------------------------------- 8. agreement, complementarity
def overlap(a, b):
    va, vb = OK['test'][a], OK['test'][b]
    wa = {i for i, x in enumerate(va) if not x}
    wb = {i for i, x in enumerate(vb) if not x}
    return {'both_right': sum(x and y for x, y in zip(va, vb)), 'only_a': sum(x and not y for x, y in zip(va, vb)),
            'only_b': sum(y and not x for x, y in zip(va, vb)), 'both_wrong': len(wa & wb),
            'jaccard_errors': round(len(wa & wb) / max(1, len(wa | wb)), 3),
            'same_text': sum(vocab.norm_text(PRED['test'][a][f]['pred']) == vocab.norm_text(PRED['test'][b][f]['pred'])
                             for f in FILES['test'])}


A['overlap'] = {f'{a}/{b}': overlap(a, b) for a, b in [('svtrv2_t_msr', 'svtrv2_s_msr'), ('svtrv2_t_msr', 'crnn'),
                                                      ('chars_v8n', 'svtrv2_t_msr'), ('chars_v8n', 'svtrv2_s_msr'),
                                                      ('chars_v8n', 'lpnr_v12n'), ('chars_v8n', 'crnn'),
                                                      ('lpnr_v12n', 'svtrv2_t_msr')]}
for k, v in A['overlap'].items():
    log('overlap', k, v)
agree = [i for i, f in enumerate(FILES['test']) if vocab.norm_text(PRED['test']['svtrv2_t_msr'][f]['pred']) ==
         vocab.norm_text(PRED['test']['chars_v8n'][f]['pred'])]
A['t_chars_agree'] = {'n': len(agree), 'share_%': round(100 * len(agree) / len(FILES['test']), 2),
                      'acc_when_agree_%': round(100 * sum(OK['test']['chars_v8n'][i] for i in agree) / len(agree), 2)}
log('agree', A['t_chars_agree'])

# ---------------------------------------------------------------------------------------------- 9. fusion
FU = json.load(open(os.path.join(FUSE, 'out', 'fusion34', 'results.json'), encoding='utf-8'))
Pv = {s: PRED[s] for s in ('val', 'test')}
single_val = {r: main[r]['val']['plate_exact_%'] for r in ALL}


def fused(models, rule, split='test'):
    bm = max(models, key=lambda n: single_val[n])
    texts, share = F.fuse_all(Pv[split], models, FILES[split], rule, bm)
    return [ok(t, TRUE[split][f]) for t, f in zip(texts, FILES[split])], round(100 * share, 2), texts


FUS = {
    'old_best_fallback': (['lpnr_v12n', 'chars_v8n'], 'fallback'),          # 88.79 % (model fusion report, 24 Oct)
    'old_kit_fallback': (['chars_v8n', 'crnn'], 'fallback'),                # 88.63 %
    'chars_t_fallback': (['chars_v8n', 'svtrv2_t_msr'], 'fallback'),
    'chars_s_fallback': (['chars_v8n', 'svtrv2_s_msr'], 'fallback'),
    'chars_t_s_fallback': (['chars_v8n', 'svtrv2_t_msr', 'svtrv2_s_msr'], 'fallback'),
    'chars_t_s_format': (['chars_v8n', 'svtrv2_t_msr', 'svtrv2_s_msr'], 'MV-BM+format'),
    'chars_lpnr_t_format': (['chars_v8n', 'lpnr_v12n', 'svtrv2_t_msr'], 'MV-BM+format'),
    'chars_lpnr_t_fallback': (['chars_v8n', 'lpnr_v12n', 'svtrv2_t_msr'], 'fallback'),
    'new_best_val': (['chars_v8n', 'lpnr_v12n', 'svtrv2_t_msr', 'svtrv2_s_msr'], 'MV-BM+format'),
    'new_best_fallback': (['chars_v8n', 'lpnr_v12n', 'svtrv2_t_msr', 'svtrv2_s_msr'], 'fallback'),
}
A['fusion'] = {'oracle_test_%': FU['oracle_any_model_right_test_%'], 'oracle_val_%': FU['oracle_any_model_right_val_%'],
               'n_combos': len(FU['combos']), 'chosen_on_val': {k: FU['best_fusion_chosen_on_val'][k] for k in ('models', 'rule')},
               'rows': {}}
fv = {}
for k, (ms, rule) in FUS.items():
    vt, share, _ = fused(ms, rule, 'test')
    vv, _, _ = fused(ms, rule, 'val')
    fv[k] = vt
    row = next(c for c in FU['combos'] if sorted(c['models']) == sorted(ms) and c['rule'] == rule)
    assert abs(row['test']['plate_exact_%'] - round(100 * sum(vt) / len(vt), 2)) < 0.01, k
    A['fusion']['rows'][k] = {'models': ms, 'rule': rule, 'val_%': round(100 * sum(vv) / len(vv), 2),
                              'test_%': round(100 * sum(vt) / len(vt), 2), 'ci95': boot(vt), 'helpers_called_%': share,
                              'test_CER_%': row['test']['CER_%'], 'ms_sum_kaggle': row['ms_per_plate_sum'],
                              'ms_kaggle': row['ms_per_plate']}
for k, vt in fv.items():
    for base in ('old_best_fallback', 'chars_v8n'):
        vb = fv[base] if base in fv else OK['test'][base]
        if k == base:
            continue
        oa = sum(x and not y for x, y in zip(vt, vb)); ob = sum(y and not x for x, y in zip(vt, vb))
        A['fusion']['rows'][k][f'vs_{base}'] = {'only_new': oa, 'only_base': ob, 'p': metrics.mcnemar_exact(oa, ob),
                                                'diff_pp': round(100 * (sum(vt) - sum(vb)) / len(vt), 2),
                                                'ci95_diff': boot_diff(vt, vb)}
for k, v in A['fusion']['rows'].items():
    log('fusion', k, v)
# val-chosen among the 156 combos: is it also the best on test?  rank of the chosen row on test
rank = sorted(FU['combos'], key=lambda c: -c['test']['plate_exact_%'])
A['fusion']['best_on_test'] = {k: rank[0][k] for k in ('models', 'rule')} | {'test_%': rank[0]['test']['plate_exact_%'],
                                                                              'val_%': rank[0]['val']['plate_exact_%']}
# fusion by condition (night) for the chosen fusion
for c in ('night', 'blurry', 'clear'):
    idx = [i for i, f in enumerate(FILES['test']) if cond[f][c] == '1']
    A['fusion'].setdefault('by_condition', {})[c] = {k: round(100 * sum(fv[k][i] for i in idx) / len(idx), 2)
                                                    for k in ('old_best_fallback', 'chars_t_fallback', 'new_best_fallback', 'new_best_val')}
log('fusion best_on_test', A['fusion']['best_on_test'], 'by condition', A['fusion']['by_condition'])

# ------------------------------------------------------------------------------------- 10. training curves
A['curves'] = {r: RES[r]['log'] for r in OURS}
A['released_weights'] = {}
for size, d in (('T', 'svtrv2_T_u14m'), ('S', 'svtrv2_S_u14m')):
    rows = [l for l in open(next(os.path.join(RUN, d, x) for x in os.listdir(os.path.join(RUN, d)) if x.endswith('.csv')),
                            encoding='utf-8').read().split('\n') if l.strip()]
    vals = [[float(x) for x in l.split(',')] for l in rows]
    cfg = open(os.path.join(RUN, d, 'config.yml'), encoding='utf-8').read()
    A['released_weights'][size] = {'common_avg': round(vals[0][-1], 2), 'u14m_avg': round(vals[-1][-1], 2),
                                   'MiB': round(os.path.getsize(os.path.join(RUN, d, 'best.pth')) / 2 ** 20, 1),
                                   'algorithm': re.search(r'algorithm: (\S+)', cfg).group(1),
                                   'decoder': re.search(r'Decoder:\s+name: (\S+)', cfg).group(1),
                                   'gtc_decoder': re.search(r'gtc_decoder:\s+name: (\S+)', cfg).group(1),
                                   'epochs': int(re.search(r'epoch_num: (\d+)', cfg).group(1)),
                                   'lr': float(re.search(r'lr: ([\d.e-]+)', cfg).group(1)),
                                   'batch_per_card': int(re.search(r'batch_size_per_card: (\d+)', cfg).group(1))}
log('released', A['released_weights'])

# ------------------------------------------------------------------------------- 11. examples for the figure
def pick_examples():
    out = {'t_right_crnn_wrong': [], 'crnn_right_t_wrong': [], 'msr_right_fixed_wrong': [], 'all_wrong': [],
           'fusion_fixed': []}
    for i, f in enumerate(FILES['test']):
        t, c, fx = OK['test']['svtrv2_t_msr'][i], OK['test']['crnn'][i], OK['test']['svtrv2_t_fixed'][i]
        row = {'file': f, 'true': TRUE['test'][f], 'R': round(AR['test'][f], 2), 'source': SRC['test'][f],
               **{r: PRED['test'][r][f]['pred'] for r in ALL}}
        if t and not c:
            out['t_right_crnn_wrong'].append(row)
        if c and not t:
            out['crnn_right_t_wrong'].append(row)
        if t and not fx and AR['test'][f] < 1.5:
            out['msr_right_fixed_wrong'].append(row)
        if not any(OK['test'][r][i] for r in ALL):
            out['all_wrong'].append(row)
        if fv['new_best_fallback'][i] and not fv['old_best_fallback'][i]:
            out['fusion_fixed'].append(row)
    return out


EX = pick_examples()
A['examples_counts'] = {k: len(v) for k, v in EX.items()}
A['examples'] = {k: v[:40] for k, v in EX.items()}
log('examples', A['examples_counts'])

A['local_check'] = json.load(open(os.path.join(HERE, 'local_check.json'), encoding='utf-8'))
json.dump(A, open(os.path.join(HERE, 'analysis.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
open(os.path.join(HERE, 'analysis_log.txt'), 'w', encoding='utf-8').write('\n'.join(LOG) + '\n')
print('wrote analysis.json')
