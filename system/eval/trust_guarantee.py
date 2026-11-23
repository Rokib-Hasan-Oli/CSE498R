"""Part A of the final-paper plan: how far can the operator trust a V2 reading?

  python eval/trust_guarantee.py            (CPU only, about 1 minute)

Uses only the saved per-reader predictions in eval/preds (val 766 crops, test 651 crops); no model runs here.
For every plate it rebuilds the V2 reading (rule "verify", same code as select_fusion.py) and a trust score, then:

 1. Risk-coverage: accept the most trusted plates first; error among accepted plates vs share accepted (test).
 2. A calibrated review threshold. For a target "accepted plates are at least (1 - alpha) right" the threshold is
    chosen on calibration plates only, with a statistical test instead of a plain look at the accuracy:
    thresholds are tried from strict to loose (fixed-sequence testing, "Learn then Test", Angelopoulos et al. 2021);
    a threshold passes when the exact binomial p-value of "error rate >= alpha" is <= delta. The last passing
    threshold is used. So with probability >= 1 - delta the accepted plates meet the target, IF calibration and
    field plates come from the same distribution.
    Protocol 1 (as deployed): calibrate on val, score test once.
    Protocol 2 (same distribution): 2,000 random halves of test: calibrate on one half, score the other.
    "plain" = the usual way (largest threshold whose calibration accuracy reaches the target), for comparison.

Trust scores (higher = more trusted). "V2 cost" scores use only readers that V2 runs anyway:
  single_conf     baseline: reader p11 alone, (layout valid, its own confidence)             [another system]
  v2_status       V2's shipped rule: verified = 1, review = 0 (one operating point)
  v2_agree_conf   V2 cost: (layout valid, >= 2 readers agree, mean confidence of the agreeing readers)
  all5_votes_conf all five readers always run: (layout valid, number of readers that agree, their mean confidence)
  v2_logit        V2 cost: logistic regression on the V2 signals, fitted on val
  all5_logit      the same with the five-reader signals
Writes eval/results/trust_guarantee.json, trust_test_plates.csv and trust_risk_coverage.png.
"""
import csv
import json
import math
import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from lpr import fusion, metrics, vocab  # noqa: E402

PRED = os.path.join(HERE, 'preds')
RES = os.path.join(HERE, 'results')
DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
CFG = json.load(open(os.path.join(HERE, '..', 'models', 'v2', 'config.json'), encoding='utf-8'))
COMBO = [r['name'] for r in CFG['readers']]
RULE = CFG['rule']
ALPHAS = (0.05, 0.03, 0.02, 0.01)          # allowed error among accepted plates
DELTA = 0.10                               # allowed chance that the guarantee fails
SPLITS = 2000
SEED = 498
HOWS = ('certified', 'grid', 'plain')


def load(name, split):
    with open(os.path.join(PRED, f'{name}_{split}.csv'), encoding='utf-8') as f:
        return {r['file']: r for r in csv.DictReader(f)}


def right(p, t):
    return vocab.norm_text(p or '') == vocab.norm_text(t)


def plates(split):
    """One dict per plate: the V2 reading, whether it is right, and the trust signals."""
    P = {n: load(n, split) for n in COMBO}
    files = sorted(set.intersection(*[set(P[n]) for n in COMBO]))
    out = []
    for f in files:
        full = [(n, P[n][f]['pred'], float(P[n][f]['conf'])) for n in COMBO]
        true = P[COMBO[0]][f]['true']
        helpers = fusion.needs_helpers(RULE, full[0][1], full[1][1])
        ran = full if helpers else full[:2]
        text = fusion.fuse(ran + [None] * (len(full) - len(ran)), RULE)
        agree_ran = [c for n, t, c in ran if fusion.same(text, t)]
        agree_all = [c for n, t, c in full if fusion.same(text, t)]
        valid = int(metrics.is_valid_plate(text))
        out.append({
            'file': f, 'true': true, 'pred': text, 'ok': int(right(text, true)), 'valid': valid,
            'status': fusion.status(text, ran), 'helpers': int(helpers), 'n_ran': len(ran),
            'k_ran': len(agree_ran), 'k_all': len(agree_all),
            'conf_agree_ran': float(np.mean(agree_ran)) if agree_ran else 0.0,
            'conf_agree_all': float(np.mean(agree_all)) if agree_all else 0.0,
            'conf_p': full[0][2], 'conf_s': full[1][2], 'confs': [c for _, _, c in full],
            'single_pred': full[0][1], 'single_ok': int(right(full[0][1], true)),
            'single_valid': int(metrics.is_valid_plate(full[0][1])),
        })
    return out


def hand_scores(rows):
    """Fit-free scores. Lexicographic keys are packed into one number (confidences are in [0, 1])."""
    return {
        'single_conf': np.array([r['single_valid'] * 2 + r['conf_p'] for r in rows]),
        'v2_status': np.array([float(r['status'] == 'verified') for r in rows]),
        'v2_agree_conf': np.array([r['valid'] * 4 + (r['k_ran'] >= 2) * 2 + r['conf_agree_ran'] for r in rows]),
        'all5_votes_conf': np.array([r['valid'] * 20 + r['k_all'] * 2 + r['conf_agree_all'] for r in rows]),
    }


def feats(rows, all5):
    X = [[r['valid'], r['k_ran'] / r['n_ran'], r['helpers'], r['conf_p'], r['conf_s'], r['conf_agree_ran']]
         + ([r['k_all'] / len(COMBO), r['conf_agree_all']] + r['confs'] if all5 else []) for r in rows]
    return np.array(X, dtype=float)


def logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


def risk_coverage(score, ok):
    """Accept plates from the most to the least trusted. Returns coverage and error among accepted (ties together)."""
    order = np.argsort(-score, kind='stable')
    s, o = score[order], ok[order]
    last = np.r_[s[1:] != s[:-1], True]                       # last plate of each tied group
    n = np.arange(1, len(s) + 1)[last]
    err = np.cumsum(1 - o)[last]
    return n / len(s), err / n, s[last]


def aurc(cov, risk):
    """Area under the risk-coverage curve (mean error over all coverage levels; lower is better)."""
    c = np.r_[0.0, cov]
    return float(np.sum((c[1:] - c[:-1]) * risk))


def at_coverage(cov, risk, c):
    i = np.searchsorted(cov, c - 1e-12)
    return float(risk[min(i, len(risk) - 1)])


GRID = [c / 100 for c in range(10, 101, 5)]                # 19 coverage levels for the 'grid' method


def calibrate(score, ok, alpha, delta, how='certified'):
    """Threshold t: accept when score >= t. None = no threshold can be certified (send every plate to review).
    how: 'plain'     largest accepted set whose calibration accuracy reaches the target (no statistics)
         'certified' fixed-sequence test from strict to loose, stop at the first failure (the method planned first)
         'grid'      added after the first run, because one early wrong plate stops 'certified' at once: test the
                     19 fixed coverage levels 10 %, 15 %, ... 100 % each at delta / 19 (Bonferroni), keep the loosest
                     level that passes. Also a valid 1 - delta guarantee."""
    cov, risk, thr = risk_coverage(score, ok)
    n = np.rint(cov * len(score)).astype(int)
    k = np.rint(risk * n).astype(int)
    if how == 'plain':
        good = np.where(risk <= alpha)[0]
        return float(thr[good[-1]]) if len(good) else None
    if how == 'grid':
        best = None
        for c in GRID:
            i = min(np.searchsorted(cov, c - 1e-12), len(cov) - 1)
            if stats.binom.cdf(k[i], n[i], alpha) <= delta / len(GRID):
                best = float(thr[i])
        return best
    n_min = math.ceil(math.log(delta) / math.log(1 - alpha))  # fewest plates that can pass with zero errors
    best = None
    for i in np.where(n >= n_min)[0]:                         # strict -> loose; stop at the first failure
        if stats.binom.cdf(k[i], n[i], alpha) > delta:
            break
        best = float(thr[i])
    return best


def accept_stats(score, ok, t):
    if t is None:
        return {'accepted': 0, 'coverage_%': 0.0, 'review_%': 100.0, 'accepted_exact_%': None, 'wrong_accepted': 0}
    a = score >= t
    n, k = int(a.sum()), int((1 - ok[a]).sum())
    lo, hi = clopper(n - k, n)
    return {'accepted': n, 'coverage_%': round(100 * n / len(ok), 2), 'review_%': round(100 * (1 - n / len(ok)), 2),
            'accepted_exact_%': round(100 * (n - k) / n, 2) if n else None, 'wrong_accepted': k,
            'accepted_exact_95ci_%': [round(100 * lo, 2), round(100 * hi, 2)],
            'review_exact_%': round(100 * float(ok[~a].mean()), 2) if (~a).any() else None}


def clopper(k, n, conf=0.95):
    if n == 0:
        return 0.0, 1.0
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(stats.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def resample(score, ok, alpha, rng, how):
    """Protocol 2: random halves of the same set. Returns mean coverage, mean accepted accuracy, failure share."""
    cov, acc, fail, none = [], [], 0, 0
    n = len(ok)
    for _ in range(SPLITS):
        p = rng.permutation(n)
        cal, ev = p[:n // 2], p[n // 2:]
        t = calibrate(score[cal], ok[cal], alpha, DELTA, how)
        if t is None:
            none += 1
            cov.append(0.0)
            continue
        a = score[ev] >= t
        cov.append(float(a.mean()))
        if a.any():
            e = float(1 - ok[ev][a].mean())
            acc.append(1 - e)
            fail += int(e > alpha)
    return {'mean_coverage_%': round(100 * float(np.mean(cov)), 2),
            'mean_accepted_exact_%': round(100 * float(np.mean(acc)), 2) if acc else None,
            'target_missed_in_%_of_splits': round(100 * fail / SPLITS, 2),
            'no_threshold_in_%_of_splits': round(100 * none / SPLITS, 2)}


def main():
    val, test = plates('val'), plates('test')
    ok = {'val': np.array([r['ok'] for r in val]), 'test': np.array([r['ok'] for r in test])}
    single_ok = {'val': np.array([r['single_ok'] for r in val]), 'test': np.array([r['single_ok'] for r in test])}
    S = {'val': hand_scores(val), 'test': hand_scores(test)}
    for name, all5 in (('v2_logit', False), ('all5_logit', True)):
        Xv, Xt = feats(val, all5), feats(test, all5)
        oof = np.zeros(len(val))                              # val scores from models that did not see that plate
        for tr, te in KFold(5, shuffle=True, random_state=SEED).split(Xv):
            oof[te] = logit().fit(Xv[tr], ok['val'][tr]).predict_proba(Xv[te])[:, 1]
        S['val'][name] = oof
        S['test'][name] = logit().fit(Xv, ok['val']).predict_proba(Xt)[:, 1]
    names = list(S['test'])
    target = lambda n, s: single_ok[s] if n == 'single_conf' else ok[s]

    out = {'readers': COMBO, 'rule': RULE, 'plates': {'val': len(val), 'test': len(test)},
           'v2_exact_%': {s: round(100 * float(ok[s].mean()), 2) for s in ok},
           'single_exact_%': {s: round(100 * float(single_ok[s].mean()), 2) for s in ok},
           'helpers_ran_%': {'val': round(100 * np.mean([r['helpers'] for r in val]), 2),
                             'test': round(100 * np.mean([r['helpers'] for r in test]), 2)},
           'delta': DELTA, 'splits': SPLITS}

    # 1. risk-coverage on val and test
    out['risk_coverage'] = {}
    curves = {}
    for n in names:
        row = {}
        for s in ('val', 'test'):
            cov, risk, _ = risk_coverage(S[s][n], target(n, s))
            if s == 'test':
                curves[n] = (cov, risk)
            row[s] = {'AURC_%': round(100 * aurc(cov, risk), 3),
                      **{f'exact_at_{int(c * 100)}%_coverage': round(100 * (1 - at_coverage(cov, risk, c)), 2)
                         for c in (0.5, 0.8, 0.9, 0.95, 1.0)}}
        out['risk_coverage'][n] = row

    # 2. calibrated thresholds
    rng = np.random.default_rng(SEED)
    out['protocol1_val_to_test'], out['protocol2_test_halves'] = {}, {}
    for n in names:
        if n in ('single_conf', 'v2_status'):
            continue
        p1, p2 = {}, {}
        for a in ALPHAS:
            key = f'target_{100 - round(100 * a)}%'
            p1[key], p2[key] = {}, {}
            for how in HOWS:
                t = calibrate(S['val'][n], ok['val'], a, DELTA, how)
                p1[key][how] = {'threshold': t, 'val': accept_stats(S['val'][n], ok['val'], t),
                                'test': accept_stats(S['test'][n], ok['test'], t)}
                p2[key][how] = resample(S['test'][n], ok['test'], a, rng, how)
        out['protocol1_val_to_test'][n], out['protocol2_test_halves'][n] = p1, p2
    out['v2_status_point'] = {s: accept_stats(S[s]['v2_status'], ok[s], 1.0) for s in ('val', 'test')}

    # 2b. why the guarantee is hard: wrong readings that every reader that ran agrees on
    out['agreed_but_wrong'] = {}
    for s, rows in (('val', val), ('test', test)):
        bad = [r for r in rows if not r['ok']]
        out['agreed_but_wrong'][s] = {
            'wrong': len(bad), 'wrong_verified': sum(r['status'] == 'verified' for r in bad),
            'wrong_all_5_agree': sum(r['k_all'] == len(COMBO) for r in bad),
            'wrong_4_or_5_agree': sum(r['k_all'] >= len(COMBO) - 1 for r in bad),
            'all_5_agree_plates': sum(r['k_all'] == len(COMBO) for r in rows),
            'all_5_agree_exact_%': round(100 * np.mean([r['ok'] for r in rows if r['k_all'] == len(COMBO)]), 2)}
        with open(os.path.join(RES, f'trust_wrong_{s}.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file', 'true', 'pred', 'status', 'k_all', 'k_ran', 'n_ran', 'conf_p', 'conf_s'])
            for r in sorted(bad, key=lambda r: (-r['k_all'], -r['conf_agree_ran'])):
                w.writerow([r[c] for c in ('file', 'true', 'pred', 'status', 'k_all', 'k_ran', 'n_ran', 'conf_p', 'conf_s')])

    # 3. who is sent to review, by condition (protocol 1, the V2-cost score, every target)
    tagf = os.path.join(DATA, 'ocr', 'test_conditions.csv')
    if os.path.exists(tagf):
        tags = {r['file']: r for r in csv.DictReader(open(tagf, encoding='utf-8'))}
        cond = {}
        for a in ALPHAS:
            key = f'target_{100 - round(100 * a)}%'
            t = out['protocol1_val_to_test']['v2_agree_conf'][key]['plain']['threshold']
            cond[key] = {}
            for tag in ('clear', 'night', 'blurry'):
                m = np.array([tags.get(r['file'], {}).get(tag) == '1' for r in test])
                if m.any():
                    cond[key][tag] = {'plates': int(m.sum()), 'all_exact_%': round(100 * float(ok['test'][m].mean()), 2),
                                      **{k: v for k, v in accept_stats(S['test']['v2_agree_conf'][m], ok['test'][m], t).items()
                                         if k in ('coverage_%', 'review_%', 'accepted_exact_%', 'wrong_accepted')}}
        out['by_condition_test_v2_agree_conf'] = cond

    json.dump(out, open(os.path.join(RES, 'trust_guarantee.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    with open(os.path.join(RES, 'trust_test_plates.csv'), 'w', newline='', encoding='utf-8') as f:
        cols = ['file', 'true', 'pred', 'ok', 'status', 'valid', 'helpers', 'n_ran', 'k_ran', 'k_all', 'conf_p', 'conf_s']
        w = csv.writer(f)
        w.writerow(cols + names)
        for i, r in enumerate(test):
            w.writerow([r[c] for c in cols] + [round(float(S['test'][n][i]), 5) for n in names])

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        label = {'single_conf': 'one reader + its confidence', 'v2_agree_conf': 'V2: agreement, then confidence',
                 'all5_votes_conf': 'all 5 readers: votes, then confidence', 'v2_logit': 'V2 signals, logistic',
                 'all5_logit': '5-reader signals, logistic'}
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=160)
        for n, lab in label.items():
            cov, risk = curves[n]
            ax.step(100 * cov, 100 * risk, where='post', label=lab, lw=1.6)
        st = out['v2_status_point']['test']
        ax.plot([st['coverage_%']], [100 - st['accepted_exact_%']], 'k*', ms=11, label='V2 "verified" rule')
        ax.set_xlabel('plates accepted automatically (%)')
        ax.set_ylabel('wrong among accepted plates (%)')
        ax.set_xlim(40, 100)
        ax.set_ylim(0, 15)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
        ax.set_title(f'Risk-coverage on the test crops (n = {len(test)})', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(RES, 'trust_risk_coverage.png'))
    except Exception as e:                                    # the numbers do not depend on the picture
        print('figure skipped:', e)

    print('V2 exact val/test', out['v2_exact_%'], '| single', out['single_exact_%'], '| helpers', out['helpers_ran_%'])
    print('status point', out['v2_status_point'])
    for n in names:
        print(f'{n:16s} test', out['risk_coverage'][n]['test'])
        print(f'{n:16s} val ', out['risk_coverage'][n]['val'])
    print('agreed but wrong', out['agreed_but_wrong'])
    for n, p in out['protocol1_val_to_test'].items():
        for key, d in p.items():
            for how in HOWS:
                x, y = d[how]['test'], out['protocol2_test_halves'][n][key][how]
                print(f'{n:16s} {key} {how:9s} P1 test: cover {x["coverage_%"]:6.2f} exact {x["accepted_exact_%"]} '
                      f'wrong {x["wrong_accepted"]:2d} | P2: cover {y["mean_coverage_%"]:6.2f} exact '
                      f'{y["mean_accepted_exact_%"]} missed {y["target_missed_in_%_of_splits"]}% none '
                      f'{y["no_threshold_in_%_of_splits"]}%')
    print(json.dumps(out.get('by_condition_test_v2_agree_conf', {}), indent=1))


if __name__ == '__main__':
    main()
