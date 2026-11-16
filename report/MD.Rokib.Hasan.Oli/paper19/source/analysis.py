"""Paper #19 analysis of the v1.1 Kaggle run + the local re-check + the local extra runs -> analysis.json.

  D:/498R/_dataset_build/venv/Scripts/python.exe analysis.py <unpacked results.zip>

Run order: local_check.py -> run_local_train.sh -> analysis.py -> charts.py -> build_report.py.
Inputs: <run>/runs/* (Kaggle), local_preds.csv + local_check.json (local_check.py), _local_runs/* (local_train.py),
the paper #24 results zip (char-YOLO and CRNN readings of the same 651 test crops), the dataset's ocr csvs.
"""
import csv
import io
import json
import math
import os
import random
import sys
import zipfile
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN / 'code'))
import numpy as np  # noqa: E402

import metrics  # noqa: E402
import p19_legibility as P  # noqa: E402
import vocab  # noqa: E402

DATA = Path(r'D:\498R\dataset')
P24_ZIP = next(Path(r'D:\498R\docs\Rokib Hasna').glob('24_*/After Train/github_upload/weights/paper24/*/paper24_results.zip'))
CLS = ['cls_b4_ema', 'cls_b4_ce', 'cls_qf_ema', 'cls_qf_ce']
COVERS = (100, 90, 80, 70, 60, 50)
A = {}
J = lambda p: json.load(open(p, encoding='utf-8'))
csvd = lambda p: list(csv.DictReader(open(p, encoding='utf-8')))
ok = lambda pred, true: vocab.norm_text(pred or '') == vocab.norm_text(true)
rng = np.random.default_rng(0)

# ------------------------------------------------------------------------------------------------ 1. the Kaggle run
res = {n: J(RUN / 'runs' / n / 'results.json') for n in CLS + ['gate', 'legibility_labels', 'readers_crnn', 'readers_slot']}
env = J(RUN / 'environment.json')
A['run'] = {'dataset_version': env['dataset_version'], 'torch': env['torch'], 'gpus': env['gpus'], 'date': env['date'],
            'labels': res['legibility_labels']['counts'], 'crnn_test': res['readers_crnn']['ocr_test'],
            'halves': res['readers_crnn']['halves'], 'reader_minutes': {'crnn': res['readers_crnn']['train_minutes'],
                                                                        'slot': res['readers_slot']['train_minutes']},
            'classifiers': {n: {k: res[n][k] for k in ('task', 'loss', 'epochs_run', 'train_minutes', 'ms_per_crop', 'params_M', 'val', 'test')}
                            for n in CLS},
            'gate': res['gate']}
for n in CLS:
    log = res[n]['log']
    A['run']['classifiers'][n]['best_epoch'] = int(np.argmax([l['val_acc'] for l in log]) + 1)
    A['run']['classifiers'][n]['best_val_acc'] = max(l['val_acc'] for l in log)
    A['run']['classifiers'][n]['curve'] = [[l['epoch'], l['train_loss'], l['val_acc']] for l in log]
    if 'test_quality_from_4class' in res[n]:
        A['run']['classifiers'][n]['quality_from_4class'] = res[n]['test_quality_from_4class']

# ------------------------------------------------------------------------------------------------ 2. readers + labels
rp = csvd(RUN / 'runs/readers_crnn/reader_preds.csv') + csvd(RUN / 'runs/readers_slot/reader_preds.csv')
rd = {}
for r in rp:
    key = (r['reader'].split('_half')[0] if 'half' in r['reader'] else r['reader'], r['split'], 'half0' in r['reader'])
    rd.setdefault(key, []).append(ok(r['pred'], r['text']))
A['readers'] = {f'{k[0]}|{k[1]}|{"half0" if k[2] else ("half1" if k[0] != "crnn_full" else "full")}': round(100 * np.mean(v), 2)
                for k, v in rd.items()}
A['readers_n'] = {f'{k[0]}|{k[1]}|{"half0" if k[2] else ("half1" if k[0] != "crnn_full" else "full")}': len(v) for k, v in rd.items()}
lab = {r['file']: r for r in csvd(RUN / 'runs/legibility_labels/labels.csv')}
src_of = {}
for s in ('train', 'val', 'test'):
    for r in csvd(DATA / 'ocr' / f'{s}.csv'):
        src_of[(s, r['file'])] = r['label_source']
by_ls = {}
for r in lab.values():
    if r['split'] == 'train':
        by_ls.setdefault(src_of[('train', r['file'])], {}).setdefault(r['label4'], 0)
        by_ls[src_of[('train', r['file'])]][r['label4']] += 1
A['train_labels_by_label_source'] = by_ls
by_src = {}
for r in lab.values():
    d = by_src.setdefault(r['source'], {c: 0 for c in P.CLASSES4})
    d[r['label4']] += 1
A['labels_by_source'] = by_src
cond = {r['file']: r for r in csvd(DATA / 'ocr/test_conditions.csv')}
conds = lambda f: [k for k in ('night', 'blurry', 'clear') if cond[f][k] == '1']

# test: the Kaggle run's own CRNN readings and classifier probabilities (so every gate row equals the run's);
# val: local readings and probabilities of the same weights (the run scored no val crops), used only to fit weights
dep = {r['file']: r for r in csvd(RUN / 'runs/legibility_labels/deploy_test.csv')}
LP = csvd(HERE / 'local_preds.csv')
kag = {n: {r['file']: r for r in csvd(RUN / 'runs' / n / 'test_scores.csv')} for n in CLS}
test = []
for f, r in dep.items():
    t = {'file': f, 'text': r['text'], 'pred': r['pred'], 'conf': float(r['conf']), 'ok': ok(r['pred'], r['text']),
         'label4': lab[f]['label4'], 'source': r['source'], 'cond': conds(f), 'valid': metrics.is_valid_plate(r['pred'])}
    for n in CLS:
        names = P.TASKS['baseline4' if 'b4' in n else 'quality']
        t[n] = np.array([float(kag[n][f][f'p_{c}']) for c in names])
    test.append(t)
val = []
for r in LP:
    if r['split'] != 'val':
        continue
    t = {'file': r['file'], 'conf': float(r['crnn_conf']), 'ok': r['crnn_ok'] == '1', 'pred': r['crnn_pred'],
         'valid': metrics.is_valid_plate(r['crnn_pred']), 'label4': r['label4']}
    for n in CLS:
        names = P.TASKS['baseline4' if 'b4' in n else 'quality']
        t[n] = np.array([float(r[f'{n}:{c}']) for c in names])
    val.append(t)
A['layout'] = {s: {'invalid': int(sum(not t['valid'] for t in rows)), 'invalid_right': int(sum(t['ok'] and not t['valid'] for t in rows))}
               for s, rows in (('test', test), ('val', val))}
suit = lambda t, n:float(t[n][0] + t[n][1]) if 'b4' in n else float(t[n][0])
A['n'] = {'test': len(test), 'val': len(val), 'test_ok': int(sum(t['ok'] for t in test)), 'val_ok': int(sum(t['ok'] for t in val))}

# label validity: how often the deployment CRNN reads each proxy class right (paper [10]: poor 72 %, good/perfect > 90 %)
A['deploy_by_label'] = {c: {'n': sum(t['label4'] == c for t in test),
                            'exact_%': round(100 * np.mean([t['ok'] for t in test if t['label4'] == c]), 2)} for c in P.CLASSES4}
# label consistency: the same plate (same text) photographed more than once in test
grp = {}
for t in test:
    grp.setdefault(vocab.norm_text(t['text']), []).append(t)
pairs = [(a, b) for g in grp.values() for i, a in enumerate(g) for b in g[i + 1:]]
A['same_plate'] = {'plates': sum(len(g) > 1 for g in grp.values()), 'crops': sum(len(g) for g in grp.values() if len(g) > 1),
                   'pairs': len(pairs),
                   'label_agree_%': round(100 * np.mean([a['label4'] == b['label4'] for a, b in pairs]), 1),
                   'label_far_%': round(100 * np.mean([abs(P.CLASSES4.index(a['label4']) - P.CLASSES4.index(b['label4'])) >= 2 for a, b in pairs]), 1),
                   'quality_agree_%': round(100 * np.mean([P.TO_QUALITY[a['label4']] == P.TO_QUALITY[b['label4']] for a, b in pairs]), 1),
                   'crnn_verdict_agree_%': round(100 * np.mean([a['ok'] == b['ok'] for a, b in pairs]), 1),
                   'cls_b4_ce_pred_agree_%': round(100 * np.mean([a['cls_b4_ce'].argmax() == b['cls_b4_ce'].argmax() for a, b in pairs]), 1)}
A['label_by_condition'] = {k: {c: sum(1 for t in test if k in t['cond'] and t['label4'] == c) for c in P.CLASSES4}
                           for k in ('night', 'blurry', 'clear')}

# ------------------------------------------------------------------------------------------------ 3. classifiers
def adjacent(cm):
    cm = np.array(cm)
    err = cm.sum() - np.trace(cm)
    adj = sum(cm[i, j] for i in range(4) for j in range(4) if abs(i - j) == 1)
    return {'errors': int(err), 'adjacent': int(adj), 'adjacent_%': round(100 * adj / max(1, err), 1)}


PAPER_FIG4 = [[1980, 101, 1, 0], [167, 1235, 65, 0], [0, 89, 797, 68], [0, 3, 42, 527]]
A['adjacent'] = {'paper_fig4': adjacent(PAPER_FIG4), 'paper_fig4_acc_%': round(100 * np.trace(PAPER_FIG4) / np.sum(PAPER_FIG4), 2)}
for n in ('cls_b4_ema', 'cls_b4_ce'):
    A['adjacent'][n] = adjacent(res[n]['test']['confusion'])


def mcnemar(a_ok, b_ok):
    b = int(sum(x and not y for x, y in zip(a_ok, b_ok)))
    c = int(sum(y and not x for x, y in zip(a_ok, b_ok)))
    return {'a_only': b, 'b_only': c, 'p': round(metrics.mcnemar_exact(b, c), 4)}


ts = {n: {r['file']: r for r in csvd(RUN / 'runs' / n / 'test_scores.csv')} for n in CLS}
files = list(ts['cls_b4_ema'])
A['ema_vs_ce'] = {}
for a, b in (('cls_b4_ema', 'cls_b4_ce'), ('cls_qf_ema', 'cls_qf_ce')):
    A['ema_vs_ce'][a] = mcnemar([ts[a][f]['label'] == ts[a][f]['pred'] for f in files], [ts[b][f]['label'] == ts[b][f]['pred'] for f in files])


# ------------------------------------------------------------------------------------------------ 4. the gate
def curve(rows, score):
    """accepted-plate exact (%) at every coverage k = 1..N (highest score first, ties keep file order)."""
    order = sorted(rows, key=lambda r: -score(r))
    c = np.cumsum([r['ok'] for r in order])
    return 100 * c / np.arange(1, len(order) + 1), order


def summary(rows, score):
    acc, order = curve(rows, score)
    N = len(order)
    out = {f'{c}%': round(float(acc[max(1, int(round(N * c / 100))) - 1]), 2) for c in COVERS}
    out['AURC_error_%'] = round(100 - float(acc.mean()), 2)
    k = int(round(N * 0.8))
    rej = order[k:]
    out['wrong_caught_of_rejected_20%'] = int(sum(not r['ok'] for r in rej))
    out['wrong_total'] = int(sum(not r['ok'] for r in order))
    return out


def auroc(rows, score):
    """P(score of a right reading > score of a wrong one): how well the score separates CRNN errors (0.5 = chance)."""
    pos = [score(r) for r in rows if r['ok']]
    neg = [score(r) for r in rows if not r['ok']]
    allv = np.array(pos + neg)
    ranks = np.argsort(np.argsort(allv, kind='mergesort'), kind='mergesort').astype(float) + 1
    # average ranks for ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


SC = {'ocr_confidence': lambda r: r['conf']}
for n in CLS:
    SC[n] = (lambda n: lambda r: suit(r, n))(n)
SC['cls_b4_ce x conf'] = lambda r: suit(r, 'cls_b4_ce') * r['conf']
SC['conf + layout'] = lambda r: r['conf'] if r['valid'] else r['conf'] - 1          # invalid BRTA layout -> last
SC['oracle'] = lambda r: float(r['ok']) + 1e-9 * r['conf']
SC['random'] = (lambda d: lambda r: d[r['file']])({t['file']: float(v) for t, v in zip(test, rng.random(len(test)))})


# learned combination, fitted on VAL (local CRNN readings = the #24 run's) and applied to test
def feats(r):
    c = min(max(r['conf'], 1e-4), 1 - 1e-4)
    return [1.0, math.log(c / (1 - c)), float(r['valid'])] + [float(v) for v in r['cls_b4_ce']] + [float(r['cls_qf_ema'][0])]


def logreg(X, y, l2=1.0, it=200):
    X, y = np.array(X), np.array(y, float)
    w = np.zeros(X.shape[1])
    for _ in range(it):                                                     # Newton / IRLS with a small ridge
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(H, g)
    return w


w_all = logreg([feats(r) for r in val], [r['ok'] for r in val])
w_conf = logreg([feats(r)[:3] for r in val], [r['ok'] for r in val])
SC['learned (val): conf + layout + classifiers'] = lambda r: float(np.dot(w_all, feats(r)))
SC['learned (val): conf + layout'] = lambda r: float(np.dot(w_conf, feats(r)[:3]))
A['learned_weights'] = {'all': [round(float(x), 3) for x in w_all], 'conf_layout': [round(float(x), 3) for x in w_conf],
                        'features': ['bias', 'logit(conf)', 'layout valid', 'b4ce p_perfect', 'b4ce p_good', 'b4ce p_poor',
                                     'b4ce p_illegible', 'qf_ema p_suitable']}
A['gate'] = {k: dict(summary(test, s), AUROC=auroc(test, s)) for k, s in SC.items()}
A['gate_curves'] = {k: [round(float(x), 2) for x in curve(test, SC[k])[0]] for k in
                    ('ocr_confidence', 'cls_b4_ce', 'cls_qf_ema', 'conf + layout', 'learned (val): conf + layout + classifiers', 'oracle', 'random')}
A['val_gate'] = {k: dict(summary(val, SC[k]), AUROC=auroc(val, SC[k])) for k in ('ocr_confidence', 'cls_b4_ce', 'cls_qf_ema', 'conf + layout')}

# Kaggle gate numbers re-computed from the CSVs (the kit's own selective())
rows_k = [{'file': f, 'ok': ok(r['pred'], r['text']), 'conf': float(r['conf'])} for f, r in dep.items()]
rep = {'ocr_confidence': P.selective(rows_k, lambda r: r['conf'])}
for n in CLS:
    s = P.suitability(RUN / 'runs' / n)
    rep[n] = P.selective(rows_k, lambda r: s[r['file']])
A['gate_reproduced'] = all(rep[k][c] == res['gate']['gates'][k][c] for k in rep for c in ('100%', '90%', '80%', '70%', '50%', 'AURC_error_%'))


# paired bootstrap over plates: gate X minus OCR confidence at 80 % coverage and in AURC
def boot(sa, sb, rows=None, B=2000):
    rows = rows or test
    d80, dau = [], []
    idx = np.arange(len(rows))
    for _ in range(B):
        smp = [rows[i] for i in rng.choice(idx, len(idx))]
        a, b = summary(smp, sa), summary(smp, sb)
        d80.append(a['80%'] - b['80%'])
        dau.append(a['AURC_error_%'] - b['AURC_error_%'])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]
    return {'d80_ci95': q(d80), 'dAURC_ci95': q(dau), 'p_cls_better_80': round(float(np.mean(np.array(d80) > 0)), 4)}


A['boot'] = {k: boot(SC[k], SC['ocr_confidence']) for k in
             ('cls_b4_ce', 'cls_b4_ema', 'cls_qf_ema', 'cls_qf_ce', 'conf + layout', 'learned (val): conf + layout + classifiers')}
A['boot']['classifiers_on_top_of_conf_layout'] = boot(SC['learned (val): conf + layout + classifiers'], SC['learned (val): conf + layout'])
A['majority'] = {'4class_%': round(100 * max(A['run']['labels']['test'].values()) / len(test), 2),
                 'quality_%': round(100 * max(sum(A['run']['labels']['test'][c] for c in ('perfect', 'good')),
                                              sum(A['run']['labels']['test'][c] for c in ('poor', 'illegible'))) / len(test), 2)}

# what each gate throws away at 80 % coverage (130 plates)
k80 = int(round(len(test) * 0.8))
rej = {k: {r['file'] for r in curve(test, SC[k])[1][k80:]} for k in ('ocr_confidence', 'cls_b4_ce', 'cls_qf_ema')}
wrong = {t['file'] for t in test if not t['ok']}
A['rejected_80'] = {
    'n_rejected': len(test) - k80, 'wrong_total': len(wrong),
    'conf_wrong': len(rej['ocr_confidence'] & wrong), 'cls_wrong': len(rej['cls_b4_ce'] & wrong),
    'both': len(rej['ocr_confidence'] & rej['cls_b4_ce']), 'both_wrong': len(rej['ocr_confidence'] & rej['cls_b4_ce'] & wrong),
    'cls_only_wrong': len((rej['cls_b4_ce'] - rej['ocr_confidence']) & wrong),
    'conf_only_wrong': len((rej['ocr_confidence'] - rej['cls_b4_ce']) & wrong),
    'cls_rejects_right': len(rej['cls_b4_ce'] - wrong), 'conf_rejects_right': len(rej['ocr_confidence'] - wrong)}

# per condition: reading accuracy, and the share each gate rejects at 80 % coverage
A['by_condition'] = {}
for k in ('night', 'blurry', 'clear'):
    sub = [t for t in test if k in t['cond']]
    A['by_condition'][k] = {'n': len(sub), 'exact_%': round(100 * np.mean([t['ok'] for t in sub]), 2),
                            'wrong': int(sum(not t['ok'] for t in sub))}
    for g in rej:
        r_ = [t for t in sub if t['file'] in rej[g]]
        A['by_condition'][k][f'rejected_by_{g}'] = len(r_)
        A['by_condition'][k][f'wrong_caught_by_{g}'] = int(sum(not t['ok'] for t in r_))
        acc = [t['ok'] for t in sub if t['file'] not in rej[g]]
        A['by_condition'][k][f'accepted_exact_{g}_%'] = round(100 * np.mean(acc), 2) if acc else None

# ------------------------------------------------------------------------------------------------ 5. a stronger reader
# paper #24's best reader: char-YOLOv8n alone (86.79 %) and its fallback fusion with the CRNN (88.63 %); same 651 crops
with zipfile.ZipFile(P24_ZIP) as z:
    rd24 = {(m, s): {r['file']: r for r in csv.DictReader(io.TextIOWrapper(z.open(f'runs/{m}/ocr_preds_{s}.csv'), encoding='utf-8'))}
            for m in ('chars_v8n', 'crnn') for s in ('val', 'test')}
cls_val = {t['file']: t for t in val}
cls_test = {t['file']: t for t in test}


def fused_rows(split):
    out = []
    for f, c in rd24[('chars_v8n', split)].items():
        cr = rd24[('crnn', split)][f]
        if metrics.is_valid_plate(c['pred']):
            pred = c['pred']
        else:
            ps = [('chars_v8n', c['pred'], float(c['conf'])), ('crnn', cr['pred'], float(cr['conf']))]
            pred = metrics.fuse(ps, 'MV-BM+format', 'chars_v8n')
        t = (cls_val if split == 'val' else cls_test)[f]
        out.append(dict(t, ok=ok(pred, c['true']), chars_ok=ok(c['pred'], c['true']), chars_conf=float(c['conf']),
                        crnn_conf=float(cr['conf']), chars_valid=metrics.is_valid_plate(c['pred']),
                        crnn_valid=metrics.is_valid_plate(cr['pred']), agree=ok(c['pred'], cr['pred']),
                        out_valid=metrics.is_valid_plate(pred or '')))
    return out


lg = lambda x: math.log(min(max(x, 1e-4), 1 - 1e-4) / (1 - min(max(x, 1e-4), 1 - 1e-4)))
f_free = lambda r: [1.0, lg(r['chars_conf']), float(r['chars_valid']), lg(r['crnn_conf']), float(r['crnn_valid']),
                    float(r['agree']), float(r['out_valid'])]
f_cls = lambda r: f_free(r) + [float(v) for v in r['cls_b4_ce']] + [float(r['cls_qf_ema'][0])]
sv, st = fused_rows('val'), fused_rows('test')
w_free = logreg([f_free(r) for r in sv], [r['ok'] for r in sv])
w_fcls = logreg([f_cls(r) for r in sv], [r['ok'] for r in sv])
A['strong'] = {'fallback_exact_%': round(100 * np.mean([r['ok'] for r in st]), 2),
               'fallback_val_exact_%': round(100 * np.mean([r['ok'] for r in sv]), 2),
               'chars_exact_%': round(100 * np.mean([r['chars_ok'] for r in st]), 2),
               'agree_share_%': round(100 * np.mean([r['agree'] for r in st]), 2),
               'exact_when_agree_%': round(100 * np.mean([r['ok'] for r in st if r['agree']]), 2),
               'exact_when_disagree_%': round(100 * np.mean([r['ok'] for r in st if not r['agree']]), 2),
               'w_free': [round(float(x), 3) for x in w_free], 'w_free_cls': [round(float(x), 3) for x in w_fcls]}
SS = {'char-YOLO confidence': lambda r: r['chars_conf'],
      'free signals (val-fitted)': lambda r: float(np.dot(w_free, f_free(r))),
      'cls_b4_ce': lambda r: suit(r, 'cls_b4_ce'), 'cls_qf_ema': lambda r: suit(r, 'cls_qf_ema'),
      'free signals + classifiers (val-fitted)': lambda r: float(np.dot(w_fcls, f_cls(r))),
      'oracle': lambda r: float(r['ok']) + 1e-9 * r['chars_conf']}
for nm, sc in SS.items():
    A['strong'][nm] = dict(summary(st, sc), AUROC=auroc(st, sc))
A['strong_curves'] = {k: [round(float(x), 2) for x in curve(st, SS[k])[0]] for k in SS}
A['strong']['boot_cls_vs_free'] = boot(SS['cls_b4_ce'], SS['free signals (val-fitted)'], st)
A['strong']['boot_freecls_vs_free'] = boot(SS['free signals + classifiers (val-fitted)'], SS['free signals (val-fitted)'], st)
chars_rows = [dict(r, ok=r['chars_ok']) for r in st]
for nm, sc in (('chars alone: own confidence', lambda r: r['chars_conf']), ('chars alone: cls_b4_ce', lambda r: suit(r, 'cls_b4_ce'))):
    A['strong'][nm] = dict(summary(chars_rows, sc), AUROC=auroc(chars_rows, sc))

# ------------------------------------------------------------------------------------------------ 6. local extra runs
loc_runs = {}
for d in sorted((HERE / '_local_runs').glob('*')):
    if not (d / 'results.json').exists():
        continue
    r = J(d / 'results.json')
    sc = {x['file']: x for x in csvd(d / 'test_scores.csv')}
    s = (lambda sc: lambda t: float(sc[t['file']]['p_perfect']) + float(sc[t['file']]['p_good'])
         if 'p_perfect' in next(iter(sc.values())) else float(sc[t['file']]['p_suitable']))(sc)
    g = summary(test, s)
    loc_runs[r['run']] = {'variant': r['variant'], 'task': r['task'], 'loss': r['loss'], 'seed': r['seed'],
                          'test_F1': r['test']['micro_F1_%'], 'macro_F1': r['test']['macro_F1_%'], 'per_class': r['test']['per_class_F1_%'],
                          'epochs_run': r['epochs_run'], 'best_epoch': r['best_epoch'], 'minutes': r['minutes'],
                          'gate80': g['80%'], 'AURC': g['AURC_error_%'], 'AUROC': auroc(test, s),
                          'qf_from_4': r.get('test_quality_from_4class', {}).get('micro_F1_%')}
A['local_runs'] = loc_runs
# seed groups: Kaggle seed 42 + local seeds 1, 2 for the kit recipe; local 42 / 1 / 2 for the other variants
groups = {}
for n in CLS:
    task, loss = ('baseline4' if 'b4' in n else 'quality'), n.split('_')[-1]
    g = groups.setdefault(f'kit_{task}_{loss}', [])
    gk = summary(test, SC[n])
    g.append({'seed': 'kaggle42', 'test_F1': res[n]['test']['micro_F1_%'], 'gate80': gk['80%'], 'AURC': gk['AURC_error_%'],
              'AUROC': auroc(test, SC[n]), 'best_epoch': A['run']['classifiers'][n]['best_epoch'],
              'per_class': res[n]['test']['per_class_F1_%']})
for k, r in loc_runs.items():
    groups.setdefault(f"{r['variant']}_{r['task']}_{r['loss']}", []).append(dict(r, seed=f"local{r['seed']}"))
A['seed_groups'] = {}
for k, g in groups.items():
    f1s = [x['test_F1'] for x in g]
    A['seed_groups'][k] = {'n': len(g), 'seeds': [x['seed'] for x in g], 'F1': f1s,
                           'F1_mean': round(float(np.mean(f1s)), 2), 'F1_sd': round(float(np.std(f1s, ddof=1)), 2) if len(g) > 1 else None,
                           'gate80': [x['gate80'] for x in g], 'gate80_mean': round(float(np.mean([x['gate80'] for x in g])), 2),
                           'AURC_mean': round(float(np.mean([x['AURC'] for x in g])), 2),
                           'AUROC_mean': round(float(np.mean([x['AUROC'] for x in g])), 4),
                           'best_epochs': [x['best_epoch'] for x in g],
                           'per_class_mean': [round(float(np.mean([x['per_class'][i] for x in g])), 2) for i in range(len(g[0]['per_class']))]}

# ------------------------------------------------------------------------------------------------ 7. paper facts
A['paper'] = {'abstract_F1': 89.5, 'table4_ema': 89.8, 'table4_more_images': 88.7, 'qf_ema': 96.7, 'qf_ce': 96.9,
              'fig4_acc_%': A['adjacent']['paper_fig4_acc_%'], 'lplcv1_poor_ocr_%': 72}
A['local_check'] = J(HERE / 'local_check.json')
A['examples'] = {}
for c in P.CLASSES4:
    xs = [t for t in test if t['label4'] == c]
    A['examples'][c] = [{'file': t['file'], 'text': t['text'], 'pred': t['pred'], 'ok': t['ok'], 'conf': t['conf'],
                         'suit_b4ce': round(suit(t, 'cls_b4_ce'), 3)} for t in random.Random(19).sample(xs, 4)]
# confident errors the classifier would have caught, and the reverse
srt = sorted(test, key=lambda t: -t['conf'])
A['examples']['conf_high_wrong'] = [{'file': t['file'], 'text': t['text'], 'pred': t['pred'], 'conf': t['conf'],
                                     'suit_b4ce': round(suit(t, 'cls_b4_ce'), 3)} for t in srt if not t['ok']][:6]
A['examples']['cls_low_right'] = [{'file': t['file'], 'text': t['text'], 'pred': t['pred'], 'conf': t['conf'],
                                   'suit_b4ce': round(suit(t, 'cls_b4_ce'), 3), 'cond': t['cond']}
                                  for t in sorted(test, key=lambda t: suit(t, 'cls_b4_ce')) if t['ok']][:6]
json.dump(A, open(HERE / 'analysis.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: A[k] for k in ('gate_reproduced', 'majority', 'strong')},
                 indent=1, ensure_ascii=False, default=float))
for k, v in A['gate'].items():
    print(f"{k:45s} 80% {v['80%']:6.2f}  50% {v['50%']:6.2f}  AURC {v['AURC_error_%']:6.2f}  AUROC {v['AUROC']:.3f}")
print(json.dumps(A['boot']))
for k, v in A['seed_groups'].items():
    print(k, v['F1'], v['F1_mean'], v['F1_sd'], 'gate80', v['gate80'], 'AUROC', v['AUROC_mean'], 'best', v['best_epochs'])
