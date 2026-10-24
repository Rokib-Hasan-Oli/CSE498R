"""Paper #24: deeper analysis of the v1.1 Kaggle run (run 22 Oct 2026, checked 24 Oct), from the per-plate predictions.

Reads the four models' ocr_preds_{val,test}.csv straight out of `After Train/results.zip` (no extraction), the OCR
condition tags and train list from dataset v1.1 (D:/498R/BanglaLPR_AllInOne_v1_1.zip, the version of this run), the
v1.0 test list (D:/498R/BanglaLPR_AllInOne_v1.zip) to find the 19 BLPD test rows that were re-cut in v1.1, and the kit's
own metrics/vocab/p24_fusion code (../kaggle). The first run (v1.0, 20 Oct) is only kept as numbers:
v1.0_run/analysis.json (its per-plate predictions were overwritten with results.zip).
Writes analysis.json next to this file; charts.py and build_report.py read it.

Everything new here (format filter, weighted vote, confidence-gated fallback) is chosen on VALIDATION and reported on
TEST, like p24_fusion.py. No model is retrained: these are offline what-ifs on the saved predictions.
"""
import collections
import csv
import io
import itertools
import json
import math
import os
import random
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PAPER, 'kaggle'))
import metrics     # noqa: E402
import p24_fusion  # noqa: E402
import vocab       # noqa: E402

ZIP = os.path.join(PAPER, 'After Train', 'results.zip')
DATA = 'D:/498R/BanglaLPR_AllInOne_v1_1.zip'          # the dataset version of this run
DATA_ROOT = 'BanglaLPR_AllInOne_v1_1/'
DATA_V10 = 'D:/498R/BanglaLPR_AllInOne_v1.zip'        # only for the old test list (which rows were re-cut)
BLPD_FIX = 'D:/498R/dataset/docs/blpd_crop_fix.md'    # old -> new file name of every re-cut BLPD row
V10 = json.load(open(os.path.join(HERE, 'v1.0_run', 'analysis.json'), encoding='utf-8'))
MODELS = ['chars_v8n', 'crnn', 'slot', 'vitstr']
SHOW = {'chars_v8n': 'char-YOLOv8n', 'crnn': 'CRNN-CTC', 'slot': 'slot CNN', 'vitstr': 'ViTSTR-tiny'}
RULES = ('HC', 'MV-HC', 'MV-BM', 'MVCP-HC')


def zip_rows(path, zpath, root):
    with zipfile.ZipFile(zpath) as z, z.open(root + path) as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding='utf-8')))


def data_rows(path):
    """rows of a csv inside the v1.1 dataset zip, e.g. data_rows('ocr/test.csv')"""
    return zip_rows(path, DATA, DATA_ROOT)


def data_image(path, v10=False):
    """an image inside the v1.1 (or v1.0) dataset zip as a BGR array, e.g. data_image('ocr/images/test/x.jpg')"""
    import cv2
    import numpy as np
    zp, root = (DATA_V10, 'BanglaLPR_AllInOne_v1/') if v10 else (DATA, DATA_ROOT)
    with zipfile.ZipFile(zp) as z:
        return cv2.imdecode(np.frombuffer(z.read(root + path), np.uint8), cv2.IMREAD_COLOR)


def recut_test():
    """{new v1.1 file: old v1.0 file} for the 19 BLPD test rows that were whole photos in v1.0."""
    out = {}
    for line in open(BLPD_FIX, encoding='utf-8'):
        m = re.match(r'\|\s*test\s*\|\s*(blpd_\w+\.jpg)\s*\|\s*(blpd_\w+\.jpg)\s*\|', line)
        if m:
            out[m.group(2)] = m.group(1)
    return out


def read_zip():
    z = zipfile.ZipFile(ZIP)
    P, R = {}, {}
    for m in MODELS:
        for s in ('val', 'test'):
            with z.open(f'runs/{m}/ocr_preds_{s}.csv') as f:
                P.setdefault(s, {})[m] = {r['file']: r for r in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8'))}
        R[m] = json.load(z.open(f'runs/{m}/results.json'))
    fusion = json.load(z.open('runs/fusion/results.json'))
    env = json.load(z.open('environment.json'))
    return P, R, fusion, env


def ok(r):
    return vocab.norm_text(r['pred'] or '') == vocab.norm_text(r['true'])


def same(a, b):
    return vocab.norm_text(a or '') == vocab.norm_text(b or '')


def score(pairs):
    return metrics.score_rows(pairs)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 2), round(100 * (c + h), 2))


def auroc(conf_right, conf_wrong):
    """P(conf of a right reading > conf of a wrong one); ties count half."""
    if not conf_right or not conf_wrong:
        return None
    import bisect
    wr = sorted(conf_wrong)
    tot = 0.0
    for c in conf_right:
        lo = bisect.bisect_left(wr, c)
        hi = bisect.bisect_right(wr, c)
        tot += lo + 0.5 * (hi - lo)
    return round(tot / (len(conf_right) * len(conf_wrong)), 3)


def valid(text):
    try:
        return metrics.is_valid_plate(text or '')
    except Exception:
        return False


def fuse_rule(preds, rule, bm, weights=None):
    """preds: [(model, text, conf)] for one plate. Paper rules and MV-BM+format go through metrics.fuse (same code as
    on Kaggle); the weighted votes are defined here."""
    if rule in RULES or rule == 'MV-BM+format':
        return metrics.fuse(preds, rule, bm)
    if rule == 'WMV':                   # weighted vote, weight = log-odds of the model's val accuracy
        g = collections.defaultdict(float)
        first = {}
        for m, t, c in preds:
            k = vocab.norm_text(t)
            g[k] += weights[m]
            first.setdefault(k, t)
        return first[max(g, key=g.get)]
    if rule == 'WMV+format':
        keep = [p for p in preds if valid(p[1])]
        return fuse_rule(keep or preds, 'WMV', bm, weights)
    raise ValueError(rule)


def boot_gain(new_ok, base_ok, n_boot=20000, seed=1):
    """95 % percentile bootstrap interval (plates resampled) of the accuracy gain in points."""
    rnd = random.Random(seed)
    n = len(new_ok)
    d = [int(a) - int(b) for a, b in zip(new_ok, base_ok)]
    gains = sorted(100 * sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return [round(gains[int(0.025 * n_boot)], 2), round(gains[int(0.975 * n_boot) - 1], 2)]


def main():
    P, R, fusion, env = read_zip()
    files = {s: sorted(set.intersection(*[set(P[s][m]) for m in MODELS])) for s in P}
    truth = {s: {f: P[s][MODELS[0]][f]['true'] for f in files[s]} for s in P}
    src = {f: P['test'][MODELS[0]][f]['source'] for f in files['test']}
    cond = {r['file']: r for r in data_rows('ocr/test_conditions.csv')}
    recut = recut_test()
    v10_test = {r['file'] for r in zip_rows('ocr/test.csv', DATA_V10, 'BanglaLPR_AllInOne_v1/')}
    assert set(recut) <= set(files['test']) and len(recut) == 19, len(recut)
    assert set(files['test']) - set(recut) == set(files['test']) & v10_test, 'unexpected test-list change'
    assert set(recut.values()) <= v10_test
    out = {'plates': {s: len(files[s]) for s in files}, 'models': MODELS, 'show': SHOW,
           'env': env, 'recut_test': recut}

    # 1. single models -------------------------------------------------------------------------------------------
    right = {s: {m: {f: ok(P[s][m][f]) for f in files[s]} for m in MODELS} for s in P}
    singles = {}
    for m in MODELS:
        k = sum(right['test'][m].values())
        n = len(files['test'])
        singles[m] = {
            'val_exact': score((P['val'][m][f]['pred'], truth['val'][f]) for f in files['val'])['plate_exact_%'],
            'test': score((P['test'][m][f]['pred'], truth['test'][f]) for f in files['test']),
            'test_right': k, 'test_ci95': wilson(k, n),
            'ms': R[m]['ocr_test']['ms_per_plate'],
            'params_M': R[m].get('params_M'),
            'train_min': R[m].get('train_minutes'),
            'valid_format_%': round(100 * sum(valid(P['test'][m][f]['pred']) for f in files['test']) / n, 2),
            'extra_tests': {k: v['plate_exact_%'] for k, v in (R[m]['ocr_test'].get('extra_tests') or {}).items()},
        }
        assert singles[m]['test']['plate_exact_%'] == fusion['single']['test'][m]['plate_exact_%']
    out['singles'] = singles
    best = max(MODELS, key=lambda m: singles[m]['val_exact'])
    assert best == fusion['best_single_on_val']
    out['best_single_val'] = best
    y = R['chars_v8n']
    out['chars_train'] = {'epochs_run': y.get('epochs_run'), 'epochs_planned': y.get('epochs_planned'),
                          'test_boxes': y.get('test_boxes'), 'val_boxes': y.get('val_boxes'),
                          'xsort_exact': y['ocr_test']['xsort_decoder']['plate_exact_%'],
                          'valid_format_%': y['ocr_test']['valid_format_%']}

    # 2. who gets which plate right (test) -----------------------------------------------------------------------
    pat = collections.Counter()
    for f in files['test']:
        pat[tuple(int(right['test'][m][f]) for m in MODELS)] += 1
    out['patterns'] = [{'right': [m for m, b in zip(MODELS, k) if b], 'n': v} for k, v in pat.most_common()]
    nt = len(files['test'])
    best_wrong = [f for f in files['test'] if not right['test'][best][f]]
    rescue = collections.Counter()
    for f in best_wrong:
        others = tuple(m for m in MODELS if m != best and right['test'][m][f])
        rescue[len(others)] += 1
    out['best_wrong'] = {'n': len(best_wrong),
                         'others_right_count': {str(k): rescue.get(k, 0) for k in range(4)},
                         'any_other_right': sum(v for k, v in rescue.items() if k > 0),
                         'crnn_right': sum(right['test']['crnn'][f] for f in best_wrong)}
    out['oracle_test'] = {}
    for k in range(1, len(MODELS) + 1):
        for combo in itertools.combinations(MODELS, k):
            out['oracle_test'][' + '.join(combo)] = round(100 * sum(any(right['test'][m][f] for m in combo)
                                                                    for f in files['test']) / nt, 2)
    assert out['oracle_test'][' + '.join(MODELS)] == fusion['oracle_any_model_right_test_%']

    # 3. pairwise error overlap (test) ---------------------------------------------------------------------------
    pairs = []
    for a, b in itertools.combinations(MODELS, 2):
        n11 = sum(right['test'][a][f] and right['test'][b][f] for f in files['test'])
        n00 = sum((not right['test'][a][f]) and (not right['test'][b][f]) for f in files['test'])
        n10 = sum(right['test'][a][f] and not right['test'][b][f] for f in files['test'])
        n01 = sum((not right['test'][a][f]) and right['test'][b][f] for f in files['test'])
        q = (n11 * n00 - n01 * n10) / max(1, (n11 * n00 + n01 * n10))
        same_wrong = sum((not right['test'][a][f]) and (not right['test'][b][f]) and
                         same(P['test'][a][f]['pred'], P['test'][b][f]['pred']) for f in files['test'])
        pairs.append({'a': a, 'b': b, 'both_right': n11, 'a_only': n10, 'b_only': n01, 'both_wrong': n00,
                      'Q': round(q, 3), 'disagree_%': round(100 * (n10 + n01) / nt, 2),
                      'same_wrong_string': same_wrong,
                      'p_b_wrong_given_a_wrong_%': round(100 * n00 / max(1, n00 + n01), 1)})
    out['pairs'] = pairs
    p = singles[best]['test']['plate_exact_%'] / 100
    out['independent_3_majority_%'] = round(100 * (p ** 3 + 3 * p ** 2 * (1 - p)), 1)

    # 4. confidence: can it tell right from wrong? -----------------------------------------------------------------
    conf = {}
    for m in MODELS:
        cr = [float(P['test'][m][f]['conf'] or 0) for f in files['test'] if right['test'][m][f]]
        cw = [float(P['test'][m][f]['conf'] or 0) for f in files['test'] if not right['test'][m][f]]
        bins = []
        allc = [(float(P['test'][m][f]['conf'] or 0), right['test'][m][f]) for f in files['test']]
        edges = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0001]
        for lo, hi in zip(edges, edges[1:]):
            sel = [r for c, r in allc if lo <= c < hi]
            bins.append({'lo': lo, 'hi': min(hi, 1.0), 'n': len(sel),
                         'acc_%': round(100 * sum(sel) / len(sel), 1) if sel else None})
        conf[m] = {'auroc': auroc(cr, cw), 'mean_right': round(sum(cr) / max(1, len(cr)), 3),
                   'mean_wrong': round(sum(cw) / max(1, len(cw)), 3),
                   'wrong_with_conf_ge_0.9_%': round(100 * sum(c >= 0.9 for c in cw) / max(1, len(cw)), 1),
                   'bins': bins}
    out['confidence'] = conf
    # HC in the best-model + CRNN pair: how often CRNN's confident wrong reading beats a right one
    def hc_pair(other):
        lost = sum(right['test'][best][f] and not right['test'][other][f] and
                   float(P['test'][other][f]['conf'] or 0) > float(P['test'][best][f]['conf'] or 0) for f in files['test'])
        won = sum((not right['test'][best][f]) and right['test'][other][f] and
                  float(P['test'][other][f]['conf'] or 0) > float(P['test'][best][f]['conf'] or 0) for f in files['test'])
        return {'pair': [best, other], 'right_overruled': lost, 'wrong_fixed': won}
    out['hc_pair'] = hc_pair('crnn')
    out['hc_pair_vitstr'] = hc_pair('vitstr')

    # 5. every subset x every rule (paper rules + ours), chosen on val ------------------------------------------
    acc_val = {m: singles[m]['val_exact'] / 100 for m in MODELS}
    W = {m: math.log(max(1e-3, acc_val[m]) / max(1e-3, 1 - acc_val[m])) for m in MODELS}
    rules = list(RULES) + ['MV-BM+format', 'WMV', 'WMV+format', 'fallback']
    combos = []
    fused = {}
    kit = {(frozenset(c['models']), c['rule']): c for c in fusion['combos']}   # the kit lists models in --runs order
    for k in range(2, len(MODELS) + 1):
        for combo in itertools.combinations(MODELS, k):
            bm = max(combo, key=lambda m: acc_val[m])
            ms_sum = round(sum(singles[m]['ms'] for m in combo), 2)
            for rule in rules:
                row = {'models': list(combo), 'rule': rule, 'k': k, 'ms_sum': ms_sum,
                       'ms_parallel': round(max(singles[m]['ms'] for m in combo), 2)}
                for s in ('val', 'test'):
                    if rule == 'fallback':      # the kit's rule, with the kit's own code
                        texts, share = p24_fusion.fuse_all(P[s], combo, files[s], 'fallback', bm)
                        preds = dict(zip(files[s], texts))
                        row[f'called_{s}_%'] = round(100 * share, 2)
                    else:
                        preds = {f: fuse_rule([(m, P[s][m][f]['pred'], float(P[s][m][f]['conf'] or 0)) for m in combo],
                                              rule, bm, W) for f in files[s]}
                    row[s] = score((preds[f], truth[s][f]) for f in files[s])
                    fused[(combo, rule, s)] = preds
                row['ms'] = ms_sum if rule != 'fallback' else round(
                    singles[bm]['ms'] + row['called_test_%'] / 100 * (ms_sum - singles[bm]['ms']), 2)
                kc = kit.get((frozenset(combo), rule))
                if kc:                         # the same rule on Kaggle: must give the same numbers
                    assert kc['test']['plate_exact_%'] == row['test']['plate_exact_%'], (combo, rule)
                    assert kc['val']['plate_exact_%'] == row['val']['plate_exact_%'], (combo, rule)
                combos.append(row)
    out['combos'] = combos
    out['kit_rows_checked'] = sum(1 for c in combos if (frozenset(c['models']), c['rule']) in kit)

    def pick(rows):                    # best on val, a tie goes to the faster row (p24_fusion.pick)
        return max(rows, key=lambda c: (c['val']['plate_exact_%'], -c['ms']))
    kag = pick([c for c in combos if c['rule'] in RULES])
    out['best_paper_rule'] = kag
    kit_rules = [c for c in combos if c['rule'] in RULES + ('MV-BM+format', 'fallback')]
    out['best_kit_rule'] = pick(kit_rules)
    out['best_any_rule'] = pick(combos)
    two = next(c for c in combos if c['models'] == [best, 'crnn'] and c['rule'] == 'fallback')
    out['fallback_2'] = two
    assert set(out['best_paper_rule']['models']) == set(fusion['best_paper_rule_chosen_on_val']['models'])
    assert set(out['best_kit_rule']['models']) == set(fusion['best_fusion_chosen_on_val']['models'])
    assert out['best_kit_rule']['rule'] == fusion['best_fusion_chosen_on_val']['rule']
    out['kaggle_fusion_check'] = {
        'kaggle_best_test': fusion['best_fusion_chosen_on_val']['test']['plate_exact_%'],
        'recomputed_best_test': out['best_kit_rule']['test']['plate_exact_%'],
        'kaggle_paper_rule_test': fusion['best_paper_rule_chosen_on_val']['test']['plate_exact_%'],
        'recomputed_paper_rule_test': kag['test']['plate_exact_%'],
        'kaggle_oracle_test': fusion['oracle_any_model_right_test_%'],
        'kit_rows': len(fusion['combos']), 'rows_equal': out['kit_rows_checked']}
    # top-N by val accuracy / by speed, like the paper's Tables 4 and 6
    order = sorted(MODELS, key=lambda m: -acc_val[m])
    speed = sorted(MODELS, key=lambda m: singles[m]['ms'])
    topn = []
    for name, rank in (('accuracy', order), ('speed', speed)):
        for k in range(1, len(MODELS) + 1):
            combo = tuple(sorted(rank[:k], key=MODELS.index))
            row = {'rank_by': name, 'k': k, 'added': rank[k - 1]}
            for rule in ('HC', 'MV-HC', 'MV-BM', 'MV-BM+format', 'fallback'):
                if k == 1:
                    row[rule] = singles[combo[0]]['test']['plate_exact_%']
                else:
                    row[rule] = next(c['test']['plate_exact_%'] for c in combos
                                     if tuple(c['models']) == combo and c['rule'] == rule)
            row['ms_sum'] = round(sum(singles[m]['ms'] for m in combo), 2)
            topn.append(row)
    out['topn'] = topn
    out['hc_range_with_crnn'] = [min(c['test']['plate_exact_%'] for c in combos if c['rule'] == 'HC' and
                                     {best, 'crnn'} <= set(c['models'])),
                                 max(c['test']['plate_exact_%'] for c in combos if c['rule'] == 'HC' and
                                     {best, 'crnn'} <= set(c['models']))]

    # 6. significance vs the best single model (exact McNemar + bootstrap interval of the gain) ------------------
    def fused_ok(c, s='test'):
        fp = fused[(tuple(c['models']), c['rule'], s)]
        return [same(fp[f], truth[s][f]) for f in files[s]]
    best_ok = [right['test'][best][f] for f in files['test']]

    def vs_best(c):
        fo = fused_ok(c)
        b = sum(x and not y for x, y in zip(fo, best_ok))
        cc = sum(y and not x for x, y in zip(fo, best_ok))
        return {'fixed': b, 'broken': cc, 'net': b - cc, 'p_mcnemar': metrics.mcnemar_exact(b, cc),
                'gain_pp': round(c['test']['plate_exact_%'] - singles[best]['test']['plate_exact_%'], 2),
                'gain_ci95': boot_gain(fo, best_ok)}
    for key in ('best_paper_rule', 'best_kit_rule', 'best_any_rule', 'fallback_2'):
        out['sig_' + key] = vs_best(out[key])
    for c in combos:
        c['vs_best'] = {k: v for k, v in vs_best(c).items() if k != 'gain_ci95'} if c['models'][0] == best else None

    # 7. confidence-gated fallback: read with the best model; only if its confidence is below t, or its text breaks
    #    the BRTA layout, ask the helpers and take MV-BM+format. t = 0 is the kit's fallback. t chosen on val.
    def cascade(s, t, helpers):
        res, called = {}, 0
        for f in files[s]:
            r0 = P[s][best][f]
            if float(r0['conf'] or 0) >= t and valid(r0['pred']):
                res[f] = r0['pred']
                continue
            called += 1
            ps = [(m, P[s][m][f]['pred'], float(P[s][m][f]['conf'] or 0)) for m in (best,) + helpers]
            res[f] = fuse_rule(ps, 'MV-BM+format', best, W)
        return res, called / max(1, len(files[s]))
    casc = []
    helper_sets = [h for k in (1, 2, 3) for h in itertools.combinations([m for m in MODELS if m != best], k)]
    for helpers in helper_sets:
        for t in [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]:
            row = {'helpers': list(helpers), 't': t}
            for s in ('val', 'test'):
                res, frac = cascade(s, t, helpers)
                row[s] = score((res[f], truth[s][f]) for f in files[s])['plate_exact_%']
                row[f'called_{s}_%'] = round(100 * frac, 1)
            row['ms_avg_test'] = round(singles[best]['ms'] + row['called_test_%'] / 100 *
                                       sum(singles[m]['ms'] for m in helpers), 2)
            casc.append(row)
    out['cascade'] = casc
    cbest = max(casc, key=lambda r: (r['val'], -r['ms_avg_test']))
    out['cascade_best'] = cbest
    out['cascade_t0_crnn'] = next(r for r in casc if r['helpers'] == ['crnn'] and r['t'] == 0.0)
    out['cascade_best_t_gt0'] = max([r for r in casc if r['t'] > 0], key=lambda r: (r['val'], -r['ms_avg_test']))

    # what the fallback does on the plates it sends to the helpers (2-model and val-chosen 3-model)
    def fallback_detail(c):
        s_out = {}
        for s in ('val', 'test'):
            fp = fused[(tuple(c['models']), 'fallback', s)]
            flagged = [f for f in files[s] if not valid(P[s][best][f]['pred'])]
            s_out[s] = {'flagged': len(flagged),
                        'flagged_wrong': sum(not right[s][best][f] for f in flagged),
                        'fixed': sum(same(fp[f], truth[s][f]) for f in flagged),
                        'still_wrong_valid_helper': sum((not same(fp[f], truth[s][f])) and valid(fp[f]) for f in flagged),
                        'kept_best_reading': sum(same(fp[f], P[s][best][f]['pred']) for f in flagged)}
        return s_out
    out['fallback_detail'] = {'2': fallback_detail(two), '3': fallback_detail(out['best_kit_rule'])}
    # where do the 3-model and the 2-model fallback differ?
    diff = {}
    for s in ('val', 'test'):
        f2, f3 = fused[(tuple(two['models']), 'fallback', s)], fused[(tuple(out['best_kit_rule']['models']), 'fallback', s)]
        d = [f for f in files[s] if not same(f2[f], f3[f])]
        diff[s] = {'plates': len(d), 'only_3_right': sum(same(f3[f], truth[s][f]) and not same(f2[f], truth[s][f]) for f in d),
                   'only_2_right': sum(same(f2[f], truth[s][f]) and not same(f3[f], truth[s][f]) for f in d)}
    out['fallback_2_vs_3'] = diff

    # 8. per source and per condition ---------------------------------------------------------------------------
    kp = fused[(tuple(kag['models']), kag['rule'], 'test')]
    fb = fused[(tuple(two['models']), 'fallback', 'test')]

    def group_rows(sel):
        fs = [f for f in files['test'] if sel(f)]
        if not fs:
            return None
        row = {'n': len(fs)}
        for m in MODELS:
            row[m] = round(100 * sum(right['test'][m][f] for f in fs) / len(fs), 1)
        row['fusion'] = round(100 * sum(same(kp[f], truth['test'][f]) for f in fs) / len(fs), 1)
        row['cascade'] = round(100 * sum(same(fb[f], truth['test'][f]) for f in fs) / len(fs), 1)
        row['oracle'] = round(100 * sum(any(right['test'][m][f] for m in MODELS) for f in fs) / len(fs), 1)
        return row
    out['by_source'] = {s: group_rows(lambda f, s=s: src[f] == s) for s in sorted(set(src.values()))}
    out['by_condition'] = {t: group_rows(lambda f, t=t: cond.get(f, {}).get(t) == '1')
                           for t in ('clear', 'night', 'blurry', 'low_res')}
    out['by_condition'] = {k: v for k, v in out['by_condition'].items() if v}
    out['unchanged_632'] = group_rows(lambda f: f not in recut)
    out['recut_19'] = group_rows(lambda f: f in recut)
    out['blpd_old_crops_11'] = group_rows(lambda f: src[f] == 'blpd' and f not in recut)
    out['recut_rows'] = [{'file': f, 'old': recut[f], 'true': truth['test'][f],
                          'preds': {m: [P['test'][m][f]['pred'], bool(right['test'][m][f])] for m in MODELS}}
                         for f in sorted(recut)]

    # 8a. v1.0 -> v1.1 (v1.0 numbers from v1.0_run/analysis.json; its per-plate predictions no longer exist)
    def v10_group(key):
        return V10[key]
    cmp_rows = []
    for m in MODELS:
        cmp_rows.append({'what': SHOW[m], 'v10_all': V10['singles'][m]['test']['plate_exact_%'],
                         'v11_all': singles[m]['test']['plate_exact_%'],
                         'v10_632': v10_group('crops_only')[m], 'v11_632': out['unchanged_632'][m],
                         'v10_19': v10_group('blpd_whole_photos')[m], 'v11_19': out['recut_19'][m],
                         'v10_val': V10['singles'][m]['val_exact'], 'v11_val': singles[m]['val_exact'],
                         'v10_ms': V10['singles'][m]['ms'], 'v11_ms': singles[m]['ms']})
    cmp_rows.append({'what': "paper's best rule (val-chosen)", 'v10_all': V10['best_paper_rule']['test']['plate_exact_%'],
                     'v11_all': kag['test']['plate_exact_%'], 'v10_632': V10['crops_only']['fusion'],
                     'v11_632': out['unchanged_632']['fusion'], 'v10_19': V10['blpd_whole_photos']['fusion'],
                     'v11_19': out['recut_19']['fusion']})
    cmp_rows.append({'what': 'format fallback (char-YOLO + CRNN)', 'v10_all': V10['cascade_best']['test'],
                     'v11_all': two['test']['plate_exact_%'], 'v10_632': V10['crops_only']['cascade'],
                     'v11_632': out['unchanged_632']['cascade'], 'v10_19': V10['blpd_whole_photos']['cascade'],
                     'v11_19': out['recut_19']['cascade']})
    cmp_rows.append({'what': 'oracle (any of the 4 right)', 'v10_all': V10['oracle_test']['chars_v8n + crnn + slot + vitstr'],
                     'v11_all': out['oracle_test']['chars_v8n + crnn + slot + vitstr'],
                     'v10_632': V10['crops_only']['oracle'], 'v11_632': out['unchanged_632']['oracle'],
                     'v10_19': V10['blpd_whole_photos']['oracle'], 'v11_19': out['recut_19']['oracle']})
    out['v10_vs_v11'] = cmp_rows
    out['v10_summary'] = {'sig_paper': V10['sig_best_paper_rule'], 'sig_fallback': V10['sig_cascade'],
                          'paper_rule': [V10['best_paper_rule']['models'], V10['best_paper_rule']['rule']],
                          'best_wrong': V10['best_wrong'], 'error_detector': V10['error_detector'],
                          'confidence_wrong_ge_0.9': {m: V10['confidence'][m]['wrong_with_conf_ge_0.9_%'] for m in MODELS},
                          'curves_best': {m: min(V10['curves'][m], key=lambda x: x['val_loss']) for m in ('crnn', 'slot', 'vitstr')},
                          'by_source': V10['by_source'], 'by_condition': V10['by_condition'],
                          'train_min': {m: V10['singles'][m]['train_min'] for m in MODELS}}

    # 8b. the BRTA layout and the confidence as error detectors for the best model (flag = do not trust)
    def detector(flag, s='test'):
        fl = [f for f in files[s] if flag(s, f)]
        wrong_all = sum(not right[s][best][f] for f in files[s])
        wrong_fl = sum(not right[s][best][f] for f in fl)
        acc = [f for f in files[s] if not flag(s, f)]
        return {'flagged': len(fl), 'flagged_%': round(100 * len(fl) / len(files[s]), 1),
                'flagged_wrong': wrong_fl,
                'precision_%': round(100 * wrong_fl / max(1, len(fl)), 1),
                'recall_%': round(100 * wrong_fl / max(1, wrong_all), 1),
                'accepted': len(acc), 'acc_when_accepted_%': round(100 * sum(right[s][best][f] for f in acc) / max(1, len(acc)), 2)}
    cb = lambda s, f: float(P[s][best][f]['conf'] or 0)
    inv = lambda s, f: not valid(P[s][best][f]['pred'])
    out['error_detector'] = {
        'format': detector(inv),
        'conf<0.8': detector(lambda s, f: cb(s, f) < 0.8),
        'format or conf<0.8': detector(lambda s, f: inv(s, f) or cb(s, f) < 0.8),
    }
    out['error_detector_val'] = {'format': detector(inv, 'val')}
    out['invalid_any_model_right'] = {m: sum(right['test'][m][f] and not valid(P['test'][m][f]['pred'])
                                             for f in files['test']) for m in MODELS}

    # 9. what kind of mistakes does the best model make? (token alignment) -------------------------------------
    kinds = collections.Counter()
    per_plate_errs = collections.Counter()
    for f in best_wrong:
        tp, tt = metrics.tokens(P['test'][best][f]['pred']), metrics.tokens(truth['test'][f])
        ops = align(tp, tt)
        per_plate_errs[min(3, len([o for o in ops if o[0] != 'ok']))] += 1
        for op, p_tok, t_tok in ops:
            if op == 'ok':
                continue
            kinds[(op, vocab.KIND.get(t_tok or p_tok, '?'))] += 1
    out['error_kinds'] = [{'op': k[0], 'kind': k[1], 'n': v} for k, v in kinds.most_common()]
    out['error_ops'] = dict(collections.Counter(k[0] for k in kinds.elements()))
    out['errors_per_wrong_plate'] = {str(k): v for k, v in sorted(per_plate_errs.items())}
    out['best_wrong_invalid_format'] = sum(not valid(P['test'][best][f]['pred']) for f in best_wrong)
    reasons = collections.Counter()
    for f in best_wrong:
        tk = metrics.tokens(P['test'][best][f]['pred'])
        kinds_ = [vocab.KIND.get(t) or ('multi_letter' if t and all(vocab.KIND.get(c) == 'letter' for c in t) else '?')
                  for t in tk]
        r = []
        if kinds_.count('multi_letter') or kinds_.count('letter') > 1:
            r.append('two letters')
        if kinds_.count('digit') < 6:
            r.append('fewer than 6 digits')
        if kinds_.count('digit') > 6:
            r.append('more than 6 digits')
        if kinds_.count('district') == 0:
            r.append('no district')
        if kinds_.count('district') > 1:
            r.append('two districts')
        if kinds_.count('letter') + kinds_.count('multi_letter') == 0:
            r.append('no letter')
        if not r and not valid(P['test'][best][f]['pred']):
            r.append('other layout error')
        for x in r:
            reasons[x] += 1
        if not r:
            reasons['valid layout, wrong symbol'] += 1
    out['best_wrong_reasons'] = dict(reasons.most_common())

    # 10. examples for the report (Bangla text + crop path) ---------------------------------------------------
    def ex(f, extra=None):
        e = {'file': f, 'source': src[f], 'true': truth['test'][f],
             'preds': {m: [P['test'][m][f]['pred'], round(float(P['test'][m][f]['conf'] or 0), 2),
                           bool(right['test'][m][f])] for m in MODELS},
             'fusion': kp[f], 'fusion_right': same(kp[f], truth['test'][f]),
             'fallback': fb[f], 'fallback_right': same(fb[f], truth['test'][f])}
        return e
    fixed = [f for f in files['test'] if not right['test'][best][f] and same(kp[f], truth['test'][f])]
    broken = [f for f in files['test'] if right['test'][best][f] and not same(kp[f], truth['test'][f])]
    missed = [f for f in best_wrong if any(right['test'][m][f] for m in MODELS) and not same(fb[f], truth['test'][f])]
    allwrong = [f for f in best_wrong if not any(right['test'][m][f] for m in MODELS)]
    fb_fixed = [f for f in files['test'] if not right['test'][best][f] and same(fb[f], truth['test'][f])]
    out['examples'] = {'paper_fixed': [ex(f) for f in fixed], 'paper_broken': [ex(f) for f in broken],
                       'fallback_fixed': [ex(f) for f in fb_fixed],
                       'missed': [ex(f) for f in missed], 'all_wrong': [ex(f) for f in allwrong[:8]]}
    out['counts'] = {'paper_fixed': len(fixed), 'paper_broken': len(broken), 'fallback_fixed': len(fb_fixed),
                     'missed_by_fallback_but_some_model_right': len(missed), 'all_wrong': len(allwrong)}

    # 11. training curves (for the health check) ---------------------------------------------------------------
    out['curves'] = {m: R[m]['log'] for m in ('crnn', 'slot', 'vitstr')}
    out['curves_best'] = {m: min(R[m]['log'], key=lambda x: x['val_loss']) for m in ('crnn', 'slot', 'vitstr')}
    out['steps_per_epoch'] = {m: R[m]['settings']['train_rows'] // R[m]['settings']['batch'] for m in ('crnn', 'slot', 'vitstr')}
    train_rows = data_rows('ocr/train.csv')
    out['train_sources'] = dict(collections.Counter(r['source'] for r in train_rows))
    out['train_label_source'] = dict(collections.Counter(r['label_source'] for r in train_rows))

    json.dump(out, open(os.path.join(HERE, 'analysis.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    summary(out)


def align(p, t):
    """Levenshtein alignment of predicted vs true tokens -> [(op, pred_tok, true_tok)]."""
    n, m = len(p), len(t)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (p[i - 1] != t[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (p[i - 1] != t[j - 1]):
            ops.append(('ok' if p[i - 1] == t[j - 1] else 'sub', p[i - 1], t[j - 1])); i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            ops.append(('extra', p[i - 1], None)); i -= 1
        else:
            ops.append(('missing', None, t[j - 1])); j -= 1
    return ops[::-1]


def summary(o):
    print('plates', o['plates'], 'dataset', o['env'].get('dataset_version'))
    for m, s in o['singles'].items():
        print(f"{m:10s} val {s['val_exact']:6.2f} test {s['test']['plate_exact_%']:6.2f} CI {s['test_ci95']} "
              f"CER {s['test']['CER_%']} char {s['test']['char_acc_%']} ms {s['ms']} valid {s['valid_format_%']} "
              f"extra {s['extra_tests']} train {s['train_min']}")
    print('chars train', o['chars_train'])
    print('patterns', o['patterns'])
    print('best wrong', o['best_wrong'])
    print('oracle', {k: v for k, v in o['oracle_test'].items() if '+' in k})
    for p in o['pairs']:
        print('pair', p)
    print('indep 3 majority', o['independent_3_majority_%'])
    for m, c in o['confidence'].items():
        print('conf', m, {k: v for k, v in c.items() if k != 'bins'}, [(b['lo'], b['n'], b['acc_%']) for b in c['bins']])
    print('hc pair', o['hc_pair'], o['hc_pair_vitstr'], 'HC range with crnn', o['hc_range_with_crnn'])
    print('kaggle check', o['kaggle_fusion_check'])
    for k in ('best_paper_rule', 'best_kit_rule', 'best_any_rule', 'fallback_2'):
        c = o[k]
        print(k, c['models'], c['rule'], 'val', c['val']['plate_exact_%'], 'test', c['test']['plate_exact_%'],
              'char', c['test']['char_acc_%'], 'ms', c['ms'], o['sig_' + k])
    for c in sorted(o['combos'], key=lambda c: -c['val']['plate_exact_%'])[:16]:
        print('  ', ' + '.join(c['models']), c['rule'], c['val']['plate_exact_%'], c['test']['plate_exact_%'], c['ms'])
    print('topn', o['topn'])
    print('cascade best', o['cascade_best'], '\n t0 crnn', o['cascade_t0_crnn'], '\n best t>0', o['cascade_best_t_gt0'])
    print('fallback detail', o['fallback_detail'], o['fallback_2_vs_3'])
    print('by source', json.dumps(o['by_source']))
    print('by cond', json.dumps(o['by_condition']))
    print('unchanged 632', o['unchanged_632'], '\nrecut 19', o['recut_19'], '\nblpd old crops', o['blpd_old_crops_11'])
    for r in o['v10_vs_v11']:
        print('v10->v11', r)
    print('error detector', json.dumps(o['error_detector']), o['error_detector_val'])
    print('invalid but right', o['invalid_any_model_right'])
    print('error kinds', o['error_kinds'], o['error_ops'], o['errors_per_wrong_plate'], 'invalid', o['best_wrong_invalid_format'])
    print('reasons', o['best_wrong_reasons'])
    print('counts', o['counts'])
    print('curves best', o['curves_best'], o['steps_per_epoch'], o['train_label_source'])


if __name__ == '__main__':
    main()
