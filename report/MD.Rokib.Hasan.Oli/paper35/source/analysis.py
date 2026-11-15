"""Paper #35 analysis: Kaggle run + check + local re-score -> analysis.json (every number the report quotes).

  python analysis.py <unpacked results.zip> <unpacked results-original.zip>

Reads local_check.json, det_preds.jsonl, det_per_plate.csv, ocr_per_crop.csv, e2e_per_photo.csv (local_check.py).
- reproduction: Kaggle results.json vs the local Ultralytics val / plates found / plate exact on the same sets;
- AP50 recomputed from the stored boxes (Ultralytics matching + 101-point interpolation), checked against val, then a
  paired photo bootstrap (5,000 resamples) of the AP50 difference enhancer - none on the night photos;
- McNemar (exact) on plates found and plates read, enhancer vs none, for every set;
- the gate: mean grey of the night-tagged photos / crops, and a gate sweep over the whole test split;
- the end-to-end test on the 60 NightLPBD + 144 BLPD e2e photos (three ways to place the enhancer);
- crop-level flips (which crops HVI breaks and fixes) for the figure.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ENH = ['none', 'gamma', 'clahe', 'hvi_gen', 'hvi_lolv2']
DARK = 60


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def ap50(rows):
    """rows: list of {'conf': [...], 'tp': [...], 'n_gt': n}. Ultralytics 8.4.170 ap_per_class + compute_ap (interp):
    precision falls to 0 right after the highest recall reached (sentinels [r_max, 1.0] / [0, 0])."""
    conf = np.concatenate([np.array(r['conf'], float) for r in rows]) if rows else np.zeros(0)
    tp = np.concatenate([np.array(r['tp'], float) for r in rows]) if rows else np.zeros(0)
    n_l = sum(r['n_gt'] for r in rows)
    if n_l == 0 or len(conf) == 0:
        return 0.0
    i = np.argsort(-conf)
    tp = tp[i]
    tpc, fpc = np.cumsum(tp), np.cumsum(1 - tp)
    recall = tpc / (n_l + 1e-16)
    precision = tpc / (tpc + fpc)
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def paired_pairs(a, b):
    """a, b: dict key -> 0/1 (same keys). Returns counts and McNemar p (b - a = only b right)."""
    only_a = sum(1 for k in a if a[k] and not b[k])
    only_b = sum(1 for k in a if b[k] and not a[k])
    return {'n': len(a), 'a': sum(a.values()), 'b': sum(b.values()), 'only_a': only_a, 'only_b': only_b,
            'p': round(mcnemar(only_a, only_b), 4)}


def main():
    run, chk = Path(sys.argv[1]), Path(sys.argv[2])
    K = json.load(open(run / 'runs' / 'lowlight' / 'results.json', encoding='utf-8'))
    DV = json.load(open(run / 'runs' / 'det_v8s' / 'results.json', encoding='utf-8'))
    CR = json.load(open(run / 'runs' / 'crnn' / 'results.json', encoding='utf-8'))
    L = json.load(open(HERE / 'local_check.json', encoding='utf-8'))
    preds = [json.loads(l) for l in open(HERE / 'det_preds.jsonl', encoding='utf-8')]
    plates = list(csv.DictReader(open(HERE / 'det_per_plate.csv', encoding='utf-8')))
    crops = list(csv.DictReader(open(HERE / 'ocr_per_crop.csv', encoding='utf-8')))
    e2e = list(csv.DictReader(open(HERE / 'e2e_per_photo.csv', encoding='utf-8')))
    grey = L['grey_test_photos']
    night, clear200 = L['night_files'], L['clear200_files']
    A = {}

    # ---------------- the Kaggle runs themselves
    env = json.load(open(run / 'environment.json', encoding='utf-8'))
    log = (run / 'logs').exists()
    A['run'] = {'dataset_version': env['dataset_version'], 'gpus': env['gpus'], 'torch': env['torch'],
                'ultralytics': env['ultralytics'],
                'det': {k: DV[k] for k in ('epochs_planned', 'epochs_run', 'best_epoch', 'train_minutes', 'params_M', 'GFLOPs')},
                'det_val': DV['val'], 'det_test': DV['test'], 'det_test_plates': DV['test_plates'],
                'det_by_condition': DV.get('test_by_condition'), 'det_speed': DV.get('speed'),
                'crnn': {'epochs': CR['settings']['epochs'], 'train_minutes': CR['train_minutes'], 'params_M': CR['params_M'],
                         'val': CR['val_ocr'], 'test': CR['ocr_test']['rows_decoder'], 'by_condition': CR['ocr_test']['by_condition'],
                         'by_source': CR['ocr_test']['by_source'], 'ms_per_plate': CR['ocr_test']['ms_per_plate'],
                         'last_val_loss': CR['log'][-1]['val_loss'], 'best_val_loss': min(x['val_loss'] for x in CR['log']),
                         'log': CR['log']},
                'kaggle_lowlight': K, 'logs_present': log}
    det_csv = list(csv.DictReader(open(run / 'runs' / 'det_v8s' / 'results.csv', encoding='utf-8')))
    det_csv = [{k.strip(): v for k, v in r.items()} for r in det_csv]
    A['run']['det_curve'] = [{'epoch': int(float(r['epoch'])), 'mAP50': float(r['metrics/mAP50(B)']),
                              'mAP50-95': float(r['metrics/mAP50-95(B)'])} for r in det_csv]
    A['run']['det_nan'] = sum(1 for r in det_csv for v in r.values() if v.strip().lower() == 'nan')
    C = {}
    for f in ('lolv1', 'lolv2_real'):
        C[f] = json.load(open(chk / 'runs' / f / 'results.json', encoding='utf-8'))
    A['check'] = C

    # ---------------- reproduction (Kaggle vs local)
    rep = {}
    for e in ENH:
        kd, ld = K['detection'][e], L['photo_val'][e]
        rep[e] = {s: {'kaggle_mAP50': round(100 * kd[s]['mAP50'], 2), 'local_mAP50': round(100 * ld[s]['mAP50'], 2),
                      'kaggle_mAP50-95': round(100 * kd[s]['mAP50-95'], 2), 'local_mAP50-95': round(100 * ld[s]['mAP50-95'], 2),
                      'kaggle_found': kd[s]['found_%']} for s in ('night', 'clear200', 'all_gated')}
    by = defaultdict(dict)
    for p in plates:
        by[p['enh']][(p['file'], p['plate'])] = int(p['found'])
    sets = {'night': set(night), 'clear200': set(clear200), 'all': set(grey)}
    for e in ENH:
        for s, files in (('night', sets['night']), ('clear200', sets['clear200'])):
            d = {k: v for k, v in by[e].items() if k[0] in files}
            rep[e][s]['local_found'] = round(100 * sum(d.values()) / max(1, len(d)), 2)
        g = {k: (by[e][k] if grey[k[0]] < DARK else by['none'][k]) for k in by['none']}
        rep[e]['all_gated']['local_found'] = round(100 * sum(g.values()) / len(g), 2)
    none_exact = {c['file']: int(c['exact']) for c in crops if c['enh'] == 'none'}
    for e in ENH:
        kr = K['reading'][e]
        ce = [c for c in crops if c['enh'] == e]
        nn = [int(c['exact']) for c in ce if c['night'] == '1']
        gated = [int(c['exact']) if float(c['grey_before']) < DARK else none_exact[c['file']] for c in ce]
        rep[e]['read'] = {'kaggle_night': kr['night']['plate_exact_%'], 'local_night': round(100 * np.mean(nn), 2),
                          'kaggle_gated': kr['all_gated']['plate_exact_%'], 'local_gated': round(100 * np.mean(gated), 2),
                          'night_crops': len(nn)}
    A['reproduction'] = rep

    # ---------------- AP50: val's own per-photo stats (local_valstats.py) and the predict() boxes (local_check.py)
    P = defaultdict(dict)                        # predict() at conf 0.001: letterbox like the kit's plate_counts
    for r in preds:
        P[r['enh']][r['file']] = r
    V = defaultdict(dict)                        # val(): exactly what the kit's mAP50 scores
    for l in open(HERE / 'det_val_stats.jsonl', encoding='utf-8'):
        r = json.loads(l)
        V[(r['enh'], r['set'])] = {p['file']: p for p in r['photos']}
    apc = {}
    for e in ENH:
        apc[e] = {'night_valstats': round(100 * ap50([V[(e, 'night')][f] for f in night]), 2),
                  'night_val': round(100 * L['photo_val'][e]['night']['mAP50'], 2),
                  'night_predict': round(100 * ap50([P[e][f] for f in night]), 2),
                  'all_valstats': round(100 * ap50(list(V[(e, 'all')].values())), 2),
                  'all_val': round(100 * L['photo_val'][e]['all_enhanced']['mAP50'], 2),
                  'all_predict': round(100 * ap50(list(P[e].values())), 2),
                  'clear200_predict': round(100 * ap50([P[e][f] for f in clear200]), 2),
                  'clear200_val': round(100 * L['photo_val'][e]['clear200']['mAP50'], 2)}
    A['ap_check'] = apc
    rng = np.random.default_rng(2026)

    def boot(rows_b, rows_e, B):
        idx = rng.integers(0, len(rows_b), (B, len(rows_b)))
        d = np.array([ap50([rows_e[i] for i in ii]) - ap50([rows_b[i] for i in ii]) for ii in idx])
        obs = ap50(rows_e) - ap50(rows_b)
        return {'diff_pp': round(100 * obs, 2), 'ci95': [round(100 * float(np.percentile(d, 2.5)), 2),
                                                        round(100 * float(np.percentile(d, 97.5)), 2)],
                'share_above_0_%': round(100 * float(np.mean(d > 0)), 1)}
    bt, bt_all, bt_pred = {}, {}, {}
    allp = sorted(grey)
    for e in ENH[1:]:
        bt[e] = boot([V[('none', 'night')][f] for f in night], [V[(e, 'night')][f] for f in night], 5000)
        bt_all[e] = boot([V[('none', 'all')][f] for f in allp], [V[(e, 'all')][f] for f in allp], 1000)
        bt_pred[e] = round(100 * (ap50([P[e][f] for f in night]) - ap50([P['none'][f] for f in night])), 2)
    A['night_ap_bootstrap'] = bt
    A['all_ap_bootstrap'] = bt_all
    A['night_ap_diff_predict'] = bt_pred

    # ---------------- McNemar on plates found
    mc = {}
    for s, files in (('night', sets['night']), ('clear200', sets['clear200']), ('all_enhanced', sets['all'])):
        base = {k: v for k, v in by['none'].items() if k[0] in files}
        for e in ENH[1:]:
            mc[f'{e}/{s}'] = paired_pairs(base, {k: by[e][k] for k in base})
    A['found_pairs'] = mc
    night_missed = {e: sorted(k[0] for k, v in by[e].items() if k[0] in sets['night'] and not v) for e in ENH}
    A['night_missed'] = night_missed
    A['night_missed_by_all'] = sorted(set.intersection(*[set(v) for v in night_missed.values()]))

    # ---------------- false alarms / boxes at conf 0.25 per photo (extra boxes after enhancement?)
    fa = {}
    for e in ENH:
        rows = [P[e][f] for f in night]
        fa[e] = {'night_boxes25': sum(r['boxes25'] for r in rows), 'night_plates': sum(r['n_gt'] for r in rows),
                 'night_tp25': sum(int(t) for r in rows for t, c in zip(r['tp'], r['conf']) if c >= 0.25)}
        nop = [r for f, r in P[e].items() if r['n_gt'] == 0]
        fa[e]['plate_free_photos'] = len(nop)
        fa[e]['plate_free_with_box_%'] = round(100 * np.mean([r['boxes25'] > 0 for r in nop]), 2)
    nofile = sorted(f for f, r in P['none'].items() if r['n_gt'] == 0)
    for e in ENH[1:]:
        fa[e]['plate_free_pairs'] = paired_pairs({f: int(P['none'][f]['boxes25'] > 0) for f in nofile},
                                                 {f: int(P[e][f]['boxes25'] > 0) for f in nofile})
    A['boxes'] = fa
    # what the night plates that no setting finds look like
    pn = {(p['file'], p['plate']): p for p in plates if p['enh'] == 'none' and p['file'] in sets['night']}
    miss = [p for k, p in pn.items() if all(by[e][k] == 0 for e in ENH)]
    A['night_missed_plates'] = [{'file': p['file'], 'area_px': float(p['area']), 'best_iou': float(p['best_iou']),
                                 'grey': float(p['grey_before'])} for p in miss]
    A['night_plate_area_median'] = float(np.median([float(p['area']) for p in pn.values()]))

    # ---------------- the dark gate
    gn = np.array([grey[f] for f in night])
    ga = np.array(list(grey.values()))
    src = lambda f: f.split('_')[0]
    A['gate'] = {'night_photos': len(night), 'night_below_60': int((gn < DARK).sum()),
                 'night_grey_median': round(float(np.median(gn)), 1), 'night_grey_min': round(float(gn.min()), 1),
                 'night_grey_max': round(float(gn.max()), 1),
                 'night_grey_quartiles': [round(float(np.percentile(gn, q)), 1) for q in (25, 50, 75)],
                 'test_photos': len(ga), 'test_below_60': int((ga < DARK).sum()),
                 'test_below_60_sources': dict(sorted(defaultdict(int, {}).items())),
                 'nightlpbd_below_60': int(sum(1 for f in night if src(f) == 'nightlpbd' and grey[f] < DARK)),
                 'nightlpbd_photos': int(sum(1 for f in night if src(f) == 'nightlpbd')),
                 'hist_night': np.histogram(gn, bins=range(0, 260, 10))[0].tolist(),
                 'hist_all': np.histogram(ga, bins=range(0, 260, 10))[0].tolist()}
    cnt = defaultdict(int)
    for f, g in grey.items():
        if g < DARK:
            cnt[src(f)] += 1
    A['gate']['test_below_60_sources'] = dict(cnt)
    nc = [c for c in crops if c['enh'] == 'none' and c['night'] == '1']
    gc = np.array([float(c['grey_before']) for c in nc])
    A['gate']['night_crops'] = len(nc)
    A['gate']['night_crops_below_60'] = int((gc < DARK).sum())
    A['gate']['night_crop_grey_median'] = round(float(np.median(gc)), 1)
    A['gate']['all_crops_below_60'] = int(sum(1 for c in crops if c['enh'] == 'none' and float(c['grey_before']) < DARK))
    sweep = {}
    VN, VA = V[('none', 'all')], None
    for e in ('clahe', 'hvi_gen', 'hvi_lolv2', 'gamma'):
        VA = V[(e, 'all')]
        sweep[e] = []
        for t in (40, 60, 80, 100, 120, 140, 256):
            rows = [VA[f] if grey[f] < t else VN[f] for f in allp]
            fnd = {k: (by[e][k] if grey[k[0]] < t else by['none'][k]) for k in by['none']}
            sweep[e].append({'thr': t, 'enhanced': int(sum(1 for f in allp if grey[f] < t)),
                             'all_AP50': round(100 * ap50(rows), 2), 'all_found_%': round(100 * np.mean(list(fnd.values())), 2)})
    A['gate_sweep'] = sweep
    A['gate_sweep_base'] = {'all_AP50': round(100 * ap50([VN[f] for f in allp]), 2),
                            'all_found_%': round(100 * np.mean(list(by['none'].values())), 2)}
    # "night-tag" gate: enhance exactly the night-tagged photos
    orc = {}
    for e in ENH[1:]:
        rows = [V[(e, 'all')][f] if f in sets['night'] else VN[f] for f in allp]
        orc[e] = round(100 * ap50(rows), 2)
    A['night_tag_gate_all_AP50'] = orc

    # ---------------- reading the crops: pairs, by grey, flips
    R = defaultdict(dict)
    for c in crops:
        R[c['enh']][c['file']] = c
    rp = {}
    for s, keep in (('night', lambda c: c['night'] == '1'), ('all', lambda c: True),
                    ('day', lambda c: c['night'] != '1')):
        base = {f: int(c['exact']) for f, c in R['none'].items() if keep(c)}
        for e in ENH[1:]:
            rp[f'{e}/{s}'] = paired_pairs(base, {f: int(R[e][f]['exact']) for f in base})
    A['read_pairs'] = rp
    try:
        sys.path.insert(0, str(HERE.parent / 'kaggle'))
        import metrics
        cer = {}
        for e in ENH:
            for s, keep in (('night', lambda c: c['night'] == '1'), ('all', lambda c: True)):
                cer[f'{e}/{s}'] = metrics.score_rows((c['pred'], c['true']) for c in R[e].values() if keep(c))
        A['read_scores'] = cer
        import vocab

        def parts(t):
            tk = metrics.tokens(t)
            kind = [vocab.KIND.get(x, '?') for x in tk]
            return ([x for x, k in zip(tk, kind) if k == 'district'], [x for x, k in zip(tk, kind) if k == 'letter'],
                    [x for x, k in zip(tk, kind) if k == 'digit'])
        err = {}
        for s, keep in (('night', lambda c: c['night'] == '1'), ('day', lambda c: c['night'] != '1')):
            for e in ('none', 'hvi_gen', 'clahe'):
                wrong = [c for c in R[e].values() if keep(c) and not int(c['exact'])]
                cnt = defaultdict(int)
                for c in wrong:
                    p, t = parts(c['pred']), parts(c['true'])
                    for name, a, b in zip(('district', 'letter', 'digits'), p, t):
                        cnt[name] += int(a != b)
                err[f'{e}/{s}'] = {'wrong_plates': len(wrong), **dict(cnt)}
        A['read_error_parts'] = err
    except Exception as ex:                                     # noqa: BLE001
        A['read_scores'] = f'metrics not importable: {ex}'
    flips = {}
    for e in ('clahe', 'hvi_gen', 'hvi_lolv2'):
        flips[e] = {'broken': [f for f, c in R['none'].items() if c['night'] == '1' and int(c['exact']) and not int(R[e][f]['exact'])],
                    'fixed': [f for f, c in R['none'].items() if c['night'] == '1' and not int(c['exact']) and int(R[e][f]['exact'])]}
    A['read_flips'] = flips
    A['flip_examples'] = {f: {e: R[e][f]['pred'] for e in ENH} | {'true': R['none'][f]['true'],
                                                                 'grey': float(R['none'][f]['grey_before']),
                                                                 'source': R['none'][f]['source']}
                          for f in sorted(set(flips['hvi_gen']['broken'][:6] + flips['hvi_gen']['fixed'][:3]))}
    ce = flips['hvi_gen']['broken'][:4] + flips['hvi_gen']['fixed'][:2]
    A['crop_examples'] = ce
    A['crop_right'] = {f: {e: bool(int(R[e][f]['exact'])) for e in ENH} for f in ce}
    for f in ce:
        A['flip_examples'].setdefault(f, {e: R[e][f]['pred'] for e in ENH} | {'true': R['none'][f]['true'],
                                                                             'grey': float(R['none'][f]['grey_before']),
                                                                             'source': R['none'][f]['source']})
    ng_sorted = sorted((f for f in night if src(f) == 'nightlpbd'), key=lambda f: grey[f])
    ex = [ng_sorted[0], ng_sorted[len(ng_sorted) // 2], ng_sorted[int(0.85 * len(ng_sorted))]]
    A['night_examples'] = ex
    A['example_grey'] = {f: grey[f] for f in ex}
    # crops by brightness: does HVI help the darkest crops?
    bins = [(0, 40), (40, 60), (60, 90), (90, 256)]
    bb = {}
    for lo, hi in bins:
        files = [f for f, c in R['none'].items() if lo <= float(c['grey_before']) < hi]
        bb[f'{lo}-{hi}'] = {'crops': len(files), **{e: round(100 * np.mean([int(R[e][f]['exact']) for f in files]), 2) if files else None
                                                   for e in ENH}}
    A['read_by_grey'] = bb
    sizes = [int(c['h']) * int(c['w']) for c in R['none'].values()]
    A['crop_size'] = {'median_h': int(np.median([int(c['h']) for c in R['none'].values()])),
                      'median_w': int(np.median([int(c['w']) for c in R['none'].values()])),
                      'median_pixels': int(np.median(sizes))}

    # ---------------- end to end
    E = defaultdict(dict)
    for r in e2e:
        E[(r['enh'], r['way'])][r['file']] = r
    ee = {}
    for (e, w), d in E.items():
        for s in ('nightlpbd', 'blpd_ext'):
            rows = [r for r in d.values() if r['source'] == s]
            ee[f'{e}/{w}/{s}'] = {'photos': len(rows), 'exact_%': round(100 * np.mean([int(r['exact']) for r in rows]), 2),
                                  'found_%': round(100 * np.mean([int(r['found']) for r in rows]), 2),
                                  'exact': sum(int(r['exact']) for r in rows)}
    A['e2e'] = ee
    ep = {}
    for e in ENH[1:]:
        for w in ('photo', 'detect', 'crop'):
            for s in ('nightlpbd', 'blpd_ext'):
                base ={f: int(r['exact']) for f, r in E[('none', 'photo')].items() if r['source'] == s}
                ep[f'{e}/{w}/{s}'] = paired_pairs(base, {f: int(E[(e, w)][f]['exact']) for f in base})
    A['e2e_pairs'] = ep
    ng = np.array([float(r['grey']) for r in E[('none', 'photo')].values() if r['source'] == 'nightlpbd'])
    A['e2e_night_grey'] = {'median': round(float(np.median(ng)), 1), 'below_60': int((ng < DARK).sum()), 'photos': len(ng)}
    # ---------------- speed / misc
    A['speed_local'] = L.get('speed')
    A['photo_ms_local'] = L.get('photo_ms')
    A['crop_ms_local'] = L.get('crop_ms')
    A['density_k'] = L['density_k']
    A['local'] = {'gpu': L['gpu'], 'torch': L['torch']}
    json.dump(A, open(HERE / 'analysis.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps({k: A[k] for k in ('reproduction', 'ap_check', 'night_ap_bootstrap', 'gate', 'e2e')}, indent=1,
                     ensure_ascii=False)[:6000])


if __name__ == '__main__':
    main()
