"""Paper #33 follow-up: why did the MuSGD-trained YOLO26 models find ~20 points more plates in synthetic rain?

A 2 x 2 design on the same network (YOLO26-N), data, augmentation and seeds; only the optimiser changes. Ultralytics
8.4.170's MuSGD differs from SGD in two ways (engine/trainer.py build_optimizer, optim/muon.py):
  (1) the Muon term: conv / linear weights get an extra orthogonalised step of 0.2 x lr next to the SGD step;
  (2) the classification-head parameters (cv3, one2one_cv3) train at 3 x the learning rate.

  arm      Muon term  head lr  how
  sgd      off        x1       optimizer=SGD (the team default)
  musgd    on         x3       optimizer=MuSGD (what optimizer=auto picks for trainings over 10k iterations)
  muon     on         x1       MuSGD with the 3x groups set back to lr0
  headlr   off        x3       MuSGD with the Muon step switched off in every group (= Nesterov SGD with a 3x head)

MuSGD's non-Muon branch is the same update as torch SGD with Nesterov momentum and the same weight decay (checked by
`selftest`), so each partial arm differs from SGD / MuSGD in exactly one factor.

  python p33_rain.py selftest                                  optimiser groups of the 4 arms + numeric equivalence
  python p33_rain.py train --arm muon [--score-max N] -- <det.py arguments>   train (det.py) + score this run
  python p33_rain.py score --run runs/rain_muon_s1 --data det.yaml [--score-max N]
  python p33_rain.py analyse --runs runs [other runs dirs ...] --out rain_analysis     (CPU only)

Scoring (det.plate_counts rules: conf 0.25, IoU 0.5, one box per label box): both heads of best.pt on the test split
and its synthetic rain and fog copies, one row per plate -> <run>/per_plate.csv, plus <run>/rain_scores.json.
Subsets: clean (all 1,011 test plates), night (fixed list), rain, fog (synthetic copies), and small / blurry / clear.
"""
import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from collections import defaultdict

from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARMS = ['sgd', 'musgd', 'muon', 'headlr']
ARM_INFO = {'sgd': (False, False), 'musgd': (True, True), 'muon': (True, False), 'headlr': (False, True)}  # Muon, 3x head
ARM_NAME = {'sgd': 'SGD', 'musgd': 'MuSGD', 'muon': 'Muon term only', 'headlr': '3x head lr only'}
HEADS = {'nms': {}, 'e2e': {'nms': False}}
SUBSETS = ['clean', 'night', 'rain', 'fog', 'small', 'blurry']
GROUPS = []                                  # optimiser groups of the last build (written to results.json)


# ------------------------------------------------------------------------------------------------ optimiser patch
def install_patch(arm):
    """Wrap BaseTrainer.build_optimizer: 'muon' removes MuSGD's 3x head lr, 'headlr' switches its Muon step off."""
    from ultralytics.engine.trainer import BaseTrainer
    orig = getattr(BaseTrainer.build_optimizer, '_orig', BaseTrainer.build_optimizer)

    def build_optimizer(self, model, name='auto', lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        opt = orig(self, model, name=name, lr=lr, momentum=momentum, decay=decay, iterations=iterations)
        if arm in ('muon', 'headlr'):
            assert type(opt).__name__ == 'MuSGD', f'arm {arm} needs optimizer=MuSGD, got {type(opt).__name__}'
        for g in opt.param_groups:
            if arm == 'muon':
                g['lr'] = lr
            elif arm == 'headlr':
                g['use_muon'] = False
        GROUPS[:] = [{'group': g.get('param_group', ''), 'tensors': len(g['params']),
                      'params': int(sum(p.numel() for p in g['params'])), 'lr': round(g['lr'], 6),
                      'use_muon': bool(g.get('use_muon', False)), 'weight_decay': g.get('weight_decay', 0.0)}
                     for g in opt.param_groups]
        print(f'[p33_rain] arm {arm}: {type(opt).__name__},',
              ', '.join(f"{x['group']} lr {x['lr']}{' +muon' if x['use_muon'] else ''} ({x['tensors']})" for x in GROUPS),
              flush=True)
        return opt
    build_optimizer._orig = orig
    BaseTrainer.build_optimizer = build_optimizer


def selftest():
    """Build the four arms on yolo26n.pt; check their groups, and that one step of MuSGD with the Muon step off and
    all lrs at lr0 equals one step of Ultralytics' SGD on the same gradients (so 'headlr' = SGD + a 3x head)."""
    import torch
    from types import SimpleNamespace
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    base = YOLO('yolo26n.pt').model.float()
    dummy = SimpleNamespace(args=SimpleNamespace(warmup_bias_lr=0.1), data={'nc': 1})
    torch.manual_seed(0)
    grads = {n: torch.randn_like(p) * 1e-2 for n, p in base.named_parameters()}

    def build(arm, name):
        install_patch(arm)
        m = copy.deepcopy(base).to(dev).train()
        for n, p in m.named_parameters():
            p.requires_grad_(True)
            p.grad = grads[n].to(dev).clone()
        return m, BaseTrainer.build_optimizer(dummy, m, name=name, lr=0.01, momentum=0.937, decay=5e-4)

    ok, out = True, {}
    for arm in ARMS:
        _, opt = build(arm, 'SGD' if arm == 'sgd' else 'MuSGD')
        lrs = sorted({round(g['lr'], 6) for g in opt.param_groups})
        muon = any(g.get('use_muon', False) for g in opt.param_groups)
        want_muon, want_x3 = ARM_INFO[arm]
        good = muon == want_muon and (lrs == [0.01, 0.03] if want_x3 else lrs == [0.01])
        out[arm] = {'optimizer': type(opt).__name__, 'groups': len(opt.param_groups), 'lrs': lrs, 'muon': muon, 'ok': good}
        ok &= good
    m_sgd, o_sgd = build('sgd', 'SGD')
    m_off, o_off = build('headlr', 'MuSGD')
    for g in o_off.param_groups:
        g['lr'] = 0.01                                           # headlr without the 3x = plain SGD?
    m_mu, o_mu = build('muon', 'MuSGD')
    for o in (o_sgd, o_off, o_mu):
        o.step()
    d_sgd = max((a - b).abs().max().item() for a, b in zip(m_sgd.parameters(), m_off.parameters()))
    d_muon = max((a - b).abs().max().item() for a, b in zip(m_sgd.parameters(), m_mu.parameters()))
    out['max |SGD - MuSGD(Muon off, x1)| after one step'] = d_sgd
    out['max |SGD - MuSGD(Muon on, x1)| after one step'] = d_muon
    ok &= d_sgd < 1e-6 and d_muon > 1e-5
    print(json.dumps(out, indent=1))
    print('SELFTEST', 'OK' if ok else 'FAILED')
    return ok


# ------------------------------------------------------------------------------------------------ scoring
def score(run, data, max_images=None, batch=32):
    import torch
    import yaml
    from ultralytics import YOLO
    import det
    run = Path(run)
    y = yaml.safe_load(open(data, encoding='utf-8'))
    root = Path(y['path'])
    cond = json.loads((root / 'conditions.json').read_text(encoding='utf-8'))
    tags = {k: {Path(l.strip()).name for l in open(root / v['list'], encoding='utf-8') if l.strip()}
            for k, v in cond['real'].items()}
    splits = {'test': root / 'images' / 'test'}
    for k, v in cond['synthetic'].items():
        splits[k.replace('_synth', '')] = root / 'images' / v['split']
    lab = lambda p: Path(str(p.parent).replace('images', 'labels', 1)) / (p.stem + '.txt')
    res = json.loads((run / 'results.json').read_text()) if (run / 'results.json').exists() else {}
    info = res.get('rain_exp', {})
    arm, seed = info.get('arm', ''), info.get('seed', '')
    dev = '0' if torch.cuda.is_available() else 'cpu'
    plates, photos = [], []
    for head, kw in HEADS.items():
        model = YOLO(str(run / 'weights' / 'best.pt'))
        for split, d in splits.items():
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})[:max_images]
            for s in range(0, len(imgs), batch):
                for r in model.predict([str(p) for p in imgs[s:s + batch]], imgsz=640, conf=0.25, device=dev,
                                       verbose=False, **kw):
                    h, w = r.orig_shape
                    g, _ = det.gt_boxes(lab(Path(r.path)), w, h)
                    p = r.boxes.xyxy.cpu().numpy()
                    order = np.argsort(-r.boxes.conf.cpu().numpy())
                    m = det.iou_matrix(p[order], g)
                    used, dups = {}, 0
                    for i in range(len(order)):
                        j = int(np.argmax(m[i])) if m.shape[1] else -1
                        if j >= 0 and m[i, j] >= 0.5:
                            if j in used:
                                dups += 1
                            else:
                                used[j] = float(m[i, j])
                    name = Path(r.path).name
                    t = {k: int(name in v) for k, v in tags.items()}
                    for j, gb in enumerate(g):
                        plates.append({'run': run.name, 'arm': arm, 'seed': seed, 'head': head, 'split': split,
                                       'file': name, 'plate': j, 'source': name.split('_')[0], 'n_plates': len(g),
                                       'area': round(float((gb[2] - gb[0]) * (gb[3] - gb[1])), 1),
                                       'found': int(j in used), 'iou': round(used.get(j, 0.0), 4),
                                       'night': t.get('night', 0), 'small': t.get('small_plates', 0),
                                       'blurry': t.get('blurry', 0), 'clear': t.get('clear', 0)})
                    photos.append({'head': head, 'split': split, 'file': name, 'n_plates': len(g), 'boxes': len(p),
                                   'hits': len(used), 'dups': dups})
    with open(run / 'per_plate.csv', 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=list(plates[0]))
        wr.writeheader()
        wr.writerows(plates)
    summ = {}
    for head in HEADS:
        sub = {}
        for name, sel in subset_filters().items():
            rows = [r for r in plates if r['head'] == head and sel(r)]
            sub[name] = {'plates': len(rows), 'found_%': round(100 * sum(r['found'] for r in rows) / max(1, len(rows)), 2)}
        ph = [x for x in photos if x['head'] == head and x['split'] == 'test']
        sub['false_alarm_photos'] = sum(x['boxes'] > 0 for x in ph if x['n_plates'] == 0)
        sub['no_plate_photos'] = sum(x['n_plates'] == 0 for x in ph)
        sub['double_boxes'] = sum(x['dups'] for x in ph)
        summ[head] = sub
    (run / 'rain_scores.json').write_text(json.dumps(summ, indent=1), encoding='utf-8')
    if res:
        res.setdefault('rain_exp', {})['scores'] = summ
        (run / 'results.json').write_text(json.dumps(res, indent=2), encoding='utf-8')
    print('[p33_rain] scored', run.name, json.dumps({h: {k: v for k, v in s.items() if isinstance(v, dict)} for h, s in summ.items()}))


def subset_filters():
    return {'clean': lambda r: r['split'] == 'test', 'night': lambda r: r['split'] == 'test' and int(r['night']),
            'rain': lambda r: r['split'] == 'rain', 'fog': lambda r: r['split'] == 'fog',
            'small': lambda r: r['split'] == 'test' and int(r['small']),
            'blurry': lambda r: r['split'] == 'test' and int(r['blurry'])}


# ------------------------------------------------------------------------------------------------ statistics
def binom_two_sided(k, n):
    """exact two-sided sign / McNemar test: P(X <= min(k, n-k)) * 2 under Binomial(n, 0.5). Python ints: a numpy
    int64 would overflow in 2 ** n and silently give p = 1."""
    k, n = int(k), int(n)
    if n == 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def welch(a, b):
    """difference of means with a 95 % Welch interval and p (seed level)."""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = math.sqrt(va + vb)
    if se == 0:                                # every seed gave the same value in both arms: no test possible
        return {'diff': round(d, 2), 'ci95': [round(d, 2), round(d, 2)], 'p': None}
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    t = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(d) / se, df)
    return {'diff': round(d, 2), 'ci95': [round(d - t * se, 2), round(d + t * se, 2)], 'p': float(p), 'df': round(df, 1)}


def factorial(cells):
    """balanced 2 x 2 ANOVA on seed-level values. cells[(muon, x3)] = list of n values (n >= 2)."""
    from scipy import stats
    n = min(len(v) for v in cells.values())
    y = {k: np.asarray(v[:n], float) for k, v in cells.items()}
    m = {k: v.mean() for k, v in y.items()}
    mse = sum(((v - v.mean()) ** 2).sum() for v in y.values()) / (4 * (n - 1))
    tcrit = stats.t.ppf(0.975, 4 * (n - 1))
    eff = {'Muon term': ((m[(1, 0)] + m[(1, 1)]) - (m[(0, 0)] + m[(0, 1)])) / 2,
           '3x head lr': ((m[(0, 1)] + m[(1, 1)]) - (m[(0, 0)] + m[(1, 0)])) / 2,
           'interaction': (m[(1, 1)] - m[(0, 1)]) - (m[(1, 0)] - m[(0, 0)])}
    se = {'Muon term': math.sqrt(mse / n), '3x head lr': math.sqrt(mse / n), 'interaction': math.sqrt(4 * mse / n)}
    out = {}
    for k, e in eff.items():
        if se[k] == 0:                         # no seed-to-seed variation at all (tiny subsets): no test possible
            out[k] = {'effect_pp': round(e, 2), 'ci95': [round(e, 2), round(e, 2)], 'F': None, 'p': None}
            continue
        tval = e / se[k]
        out[k] = {'effect_pp': round(e, 2), 'ci95': [round(e - tcrit * se[k], 2), round(e + tcrit * se[k], 2)],
                  'F': round(tval ** 2, 2), 'p': float(2 * stats.t.sf(abs(tval), 4 * (n - 1)))}
    out['seeds_per_cell'] = n
    out['residual_sd_pp'] = round(math.sqrt(mse), 2)
    return out


def pv(p):
    if p is None:
        return 'no seed variation'
    return 'p < 0.0001' if p < 0.0001 else f'p = {p:.4f}' if p < 0.01 else f'p = {p:.2f}'


# treatment first, so a positive difference means "better than the reference arm"
PAIRS = [('musgd', 'sgd'), ('muon', 'sgd'), ('headlr', 'sgd'), ('musgd', 'muon'), ('musgd', 'headlr'), ('muon', 'headlr')]
MAIN_SUBSETS = ['clean', 'night', 'rain', 'fog']        # the arm-vs-arm and factorial tables in the markdown


def analyse(dirs, out):
    rows, meta = [], {}
    for d in dirs:
        for f in sorted(Path(d).glob('rain_*/per_plate.csv')):
            if f.parent.name in meta:                # the same run attached twice (e.g. an earlier part's output)
                print('skipped duplicate run', f.parent)
                continue
            rows += list(csv.DictReader(open(f, encoding='utf-8')))
            res = json.loads((f.parent / 'results.json').read_text()) if (f.parent / 'results.json').exists() else {}
            meta[f.parent.name] = {'epochs_run': res.get('epochs_run'), 'epochs_planned': res.get('epochs_planned'),
                                   'train_minutes': res.get('train_minutes'), 'best_epoch': res.get('best_epoch'),
                                   'dir': str(f.parent)}
    if not rows:
        print('no rain_*/per_plate.csv found in', dirs)
        return
    F = subset_filters()
    P = defaultdict(dict)                     # (arm, seed, head, subset) -> {(split, file, plate): found}
    photo_of = {}
    for r in rows:
        k = (r['split'], r['file'], int(r['plate']))
        photo_of[k] = r['file']
        for s, sel in F.items():
            if sel(r):
                P[(r['arm'], int(r['seed']), r['head'], s)][k] = int(r['found'])
    arms = [a for a in ARMS if any(k[0] == a for k in P)]
    seeds = {a: sorted({k[1] for k in P if k[0] == a}) for a in arms}
    pct = lambda d: 100 * sum(d.values()) / max(1, len(d))
    A = {'runs': meta, 'arms': {a: seeds[a] for a in arms}, 'per_arm': {}, 'pairs': {}, 'factorial': {}}
    for head in HEADS:
        for s in SUBSETS:
            for a in arms:
                v = [pct(P[(a, sd, head, s)]) for sd in seeds[a] if P.get((a, sd, head, s))]
                if v:
                    A['per_arm'][f'{head}/{s}/{a}'] = {'seeds': seeds[a], 'found_%': [round(x, 2) for x in v],
                                                       'mean': round(float(np.mean(v)), 2),
                                                       'sd': round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else None,
                                                       'plates': len(P[(a, seeds[a][0], head, s)])}
    rng = random.Random(0)
    for head in HEADS:
        for a, b in [x for x in PAIRS if x[0] in arms and x[1] in arms]:
            for s in SUBSETS:
                va = [pct(P[(a, sd, head, s)]) for sd in seeds[a]]
                vb = [pct(P[(b, sd, head, s)]) for sd in seeds[b]]
                keys = sorted(set.intersection(*[set(P[(x, sd, head, s)]) for x in (a, b) for sd in seeds[x]]))
                if not keys or len(va) < 2 or len(vb) < 2:
                    continue
                ma = {k: np.mean([P[(a, sd, head, s)][k] for sd in seeds[a]]) for k in keys}
                mb = {k: np.mean([P[(b, sd, head, s)][k] for sd in seeds[b]]) for k in keys}
                up = int(sum(ma[k] > mb[k] for k in keys))
                down = int(sum(ma[k] < mb[k] for k in keys))
                byphoto = defaultdict(list)
                for k in keys:
                    byphoto[photo_of[k]].append(ma[k] - mb[k])
                ph = list(byphoto)
                boots = []
                for _ in range(2000):
                    tot = cnt = 0
                    for f in (rng.choice(ph) for _ in ph):
                        tot += sum(byphoto[f]); cnt += len(byphoto[f])
                    boots.append(100 * tot / cnt)
                boots.sort()
                matched = {}
                for sd in sorted(set(seeds[a]) & set(seeds[b])):
                    pa, pb = P[(a, sd, head, s)], P[(b, sd, head, s)]
                    oa = sum(pa[k] and not pb[k] for k in keys)
                    ob = sum(pb[k] and not pa[k] for k in keys)
                    matched[sd] = {'only_a': oa, 'only_b': ob, 'p': binom_two_sided(oa, oa + ob)}
                A['pairs'][f'{head}/{s}/{a} vs {b}'] = {
                    'seed_level': welch(va, vb), 'plates': len(keys), 'plates_a_more': up, 'plates_b_more': down,
                    'sign_test_p': binom_two_sided(up, up + down), 'boot_ci95_pp': [round(boots[49], 2), round(boots[1949], 2)],
                    'seed_matched': matched}
        if len(arms) == 4 and min(len(seeds[a]) for a in arms) >= 2:
            for s in SUBSETS:
                cells = {ARM_INFO[a]: [pct(P[(a, sd, head, s)]) for sd in seeds[a]] for a in arms}
                cells = {(int(k[0]), int(k[1])): v for k, v in cells.items()}
                A['factorial'][f'{head}/{s}'] = factorial(cells)
    Path(out + '.json').write_text(json.dumps(A, indent=1), encoding='utf-8')
    Path(out + '.md').write_text(render(A, arms, seeds), encoding='utf-8')
    print(render(A, arms, seeds))


def render(A, arms, seeds):
    L = ['# Paper #33 rain experiment: optimiser vs robustness (YOLO26-N)', '',
         'Same network, data, augmentation and seeds; only the optimiser changes. Found % = share of the plates found at '
         'confidence 0.25, IoU 0.5. Seed-level tests: Welch t on the per-seed found % (95 % interval of the difference). '
         'Plate-level: each plate\'s share of seeds that found it, arm a vs arm b; sign test over plates and a bootstrap '
         'interval over photos. Seed-matched: McNemar per seed (same seed = same data order). p-values are not adjusted '
         'for the number of comparisons; the 2 x 2 factorial is the main test.', '']
    L += ['## Runs', '', '| Run | Epochs run / planned | Best epoch | Train min |', '|---|---|---|---|']
    for r, m in sorted(A['runs'].items()):
        flag = '' if m['epochs_run'] == m['epochs_planned'] else ' **(stopped early)**'
        L.append(f"| {r} | {m['epochs_run']} / {m['epochs_planned']}{flag} | {m['best_epoch']} | {m['train_minutes']} |")
    for head, title in (('nms', 'NMS head (the default)'), ('e2e', 'NMS-free head')):
        L += ['', f'## Plates found per arm, {title}', '',
              '| Arm | Muon term | 3x head lr | ' + ' | '.join(SUBSETS) + ' |', '|---|---|---|' + '---|' * len(SUBSETS)]
        for a in arms:
            cells = []
            for s in SUBSETS:
                x = A['per_arm'].get(f'{head}/{s}/{a}')
                cells.append('–' if not x else f"{x['mean']:.2f}" + (f" ± {x['sd']:.2f}" if x['sd'] is not None else '')
                             + f" ({', '.join(f'{v:.1f}' for v in x['found_%'])})")
            mu, x3 = ARM_INFO[a]
            L.append(f"| {ARM_NAME[a]} (seeds {', '.join(map(str, seeds[a]))}) | {'on' if mu else 'off'} | "
                     f"{'on' if x3 else 'off'} | " + ' | '.join(cells) + ' |')
        L.append('')
        L.append('Mean ± sd over seeds (each seed in brackets). Plates: ' + ', '.join(
            f"{s} {A['per_arm'][f'{head}/{s}/{arms[0]}']['plates']}" for s in SUBSETS if f'{head}/{s}/{arms[0]}' in A['per_arm']) + '.')
        if head == 'e2e':
            continue
        L += ['', '## Arm vs arm, NMS head (positive = the first arm finds more plates)', '',
              '| Subset | a vs b | Mean diff, points (95 % CI, seeds) | Welch p | Plates a better / b better (sign test) | '
              'Bootstrap 95 % CI | Per seed: only a / only b (McNemar) |', '|---|---|---|---|---|---|---|']
        for s in MAIN_SUBSETS:
            for k, v in A['pairs'].items():
                h, ss, pair = k.split('/', 2)
                if h != head or ss != s:
                    continue
                a, b = pair.split(' vs ')
                w = v['seed_level']
                per = '; '.join(f"s{sd}: {m['only_a']}/{m['only_b']} ({pv(m['p'])})" for sd, m in v['seed_matched'].items())
                L.append(f"| {s} | {ARM_NAME[a]} vs {ARM_NAME[b]} | {w['diff']:+.2f} ({w['ci95'][0]:+.2f} to {w['ci95'][1]:+.2f}) | "
                         f"{pv(w['p'])} | {v['plates_a_more']} / {v['plates_b_more']} ({pv(v['sign_test_p'])}) | "
                         f"{v['boot_ci95_pp'][0]:+.2f} to {v['boot_ci95_pp'][1]:+.2f} | {per} |")
    if A['factorial']:
        L += ['', '## 2 x 2 factorial (seed-level found %, balanced ANOVA)', '',
              '| Head / subset | Muon term: effect, points (95 % CI), p | 3x head lr: effect (95 % CI), p | Interaction (95 % CI), p | residual sd |',
              '|---|---|---|---|---|']
        for k, f in A['factorial'].items():
            if k.split('/')[1] not in MAIN_SUBSETS:      # small / blurry stay in the json
                continue
            c = lambda e: f"{f[e]['effect_pp']:+.2f} ({f[e]['ci95'][0]:+.2f} to {f[e]['ci95'][1]:+.2f}), {pv(f[e]['p'])}"
            L.append(f"| {k} | {c('Muon term')} | {c('3x head lr')} | {c('interaction')} | {f['residual_sd_pp']} |")
        L += ['', 'Effect = mean found % with the factor on minus off (averaged over the other factor); interaction = '
              '(MuSGD - head-only) - (Muon-only - SGD).']
    else:
        L += ['', '*The 2 x 2 factorial needs all four arms: run `analyse` on the runs of both parts together.*']
    return '\n'.join(L) + '\n'


# ------------------------------------------------------------------------------------------------ main
def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        argv = sys.argv[2:]
        split = argv.index('--') if '--' in argv else len(argv)
        p = argparse.ArgumentParser()
        p.add_argument('--arm', choices=ARMS, required=True)
        p.add_argument('--score-max', type=int, default=None)
        a = p.parse_args(argv[:split])
        det_args = argv[split + 1:]
        install_patch(a.arm)
        det_args += ['--optimizer', 'SGD' if a.arm == 'sgd' else 'MuSGD', '--lr0', '0.01']
        sys.argv = ['det.py'] + det_args
        import det
        val = lambda k, d=None: det_args[det_args.index(k) + 1] if k in det_args else d
        det.main()
        run = Path(val('--project')) / val('--name')
        res = json.loads((run / 'results.json').read_text())
        mu, x3 = ARM_INFO[a.arm]
        res['rain_exp'] = {'arm': a.arm, 'muon_term': mu, 'head_lr_x3': x3, 'seed': int(val('--seed', 42)),
                           'optimizer_groups': GROUPS}
        (run / 'results.json').write_text(json.dumps(res, indent=2))
        score(run, val('--data'), a.score_max)
        return
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=['selftest', 'score', 'analyse'])
    p.add_argument('--run')
    p.add_argument('--data')
    p.add_argument('--score-max', type=int, default=None)
    p.add_argument('--runs', nargs='+', default=[])
    p.add_argument('--out', default='rain_analysis')
    a = p.parse_args()
    if a.cmd == 'selftest':
        sys.exit(0 if selftest() else 1)
    if a.cmd == 'score':
        score(a.run, a.data, a.score_max)
    else:
        analyse(a.runs, a.out)


if __name__ == '__main__':
    main()
