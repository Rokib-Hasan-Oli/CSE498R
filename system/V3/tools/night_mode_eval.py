"""Night mode for the V3 app: which changes help at night? Every choice on VAL, then test once.

  D:/498R/_dataset_build/venv/Scripts/python.exe tools/night_mode_eval.py

"Night" = the dataset's rule (mean grey < 60, or a NightLPBD image). The app cannot see that rule (night photos lit by
a flash or headlights are not dark), so in the app night mode is switched on by the clock or by the user; part 0
measures how well image brightness alone would do.

0. Night from the image alone: best brightness rule on val photos, then test.
1. Reading (OCR crops): the V2 rule ('verify') against night variants, scored on the night crops (val 46, test 65)
   and on all crops (what happens if night mode is on for a day plate).
     vote5          : all five readers always run, MV-BM+format
     clahe_on_doubt : 'verify'; when the plate is not verified, the five readers also read a CLAHE copy and all
                      readings vote (MV-BM+format)
     vote10         : five readers on the crop + five on a CLAHE copy, all vote
2. Detection (night val photos): detector confidence 0.25 (V2) / 0.15 / 0.10, retry chain as in V2; plates found
   and extra (unmatched) boxes.
3. Whole photos (e2e_test, 60 NightLPBD night + 144 day photos): V2 against the chosen night mode, through the real
   pipeline (lpr.pipeline.LPRSystem.process(night=...)).
Writes results/night_mode.json (run 1, where CLAHE readings also counted for the status: night_mode_run1_status_all_readings.json).
"""
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
sys.path.insert(0, V3)
from lpr import fusion, metrics, vocab  # noqa: E402
from lpr.pipeline import LPRSystem, clahe, night_level  # noqa: E402

DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
OUT = os.path.join(V3, 'results')
DARK = 60


def rows(p):
    return list(csv.DictReader(open(p, encoding='utf-8')))


def right(p, t):
    return vocab.norm_text(p or '') == vocab.norm_text(t)


def grey_mean(img):
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def is_night(file, img):
    return file.startswith('nightlpbd') or grey_mean(img) < DARK


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(a, b):
    """a, b: lists of bools on the same items (a = night mode, b = V2)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    return {'n': len(a), 'v2_%': round(100 * sum(b) / max(1, len(b)), 2), 'night_%': round(100 * sum(a) / max(1, len(a)), 2),
            'only_night_right': only_a, 'only_v2_right': only_b, 'p_mcnemar': round(mcnemar(only_a, only_b), 4)}


# ------------------------------------------------------------------------------------------------ 0. night from the image
def part0():
    feats = {}
    for sp in ('val', 'test'):
        d = os.path.join(DATA, 'detection', 'images', sp)
        out = []
        for f in sorted(os.listdir(d)):
            im = cv2.imread(os.path.join(d, f))
            out.append((night_level(im), is_night(f, im)))
        feats[sp] = out
    best = None
    v = np.array([x[0] for x in feats['val']])
    y = np.array([x[1] for x in feats['val']])
    for t in np.unique(np.round(np.percentile(v, np.linspace(1, 99, 99)), 1)):
        tpr, fpr = ((v < t) & y).sum() / y.sum(), ((v < t) & ~y).sum() / (~y).sum()
        if best is None or tpr - fpr > best[0]:
            best = (tpr - fpr, float(t), tpr, fpr)
    vt = np.array([x[0] for x in feats['test']])
    yt = np.array([x[1] for x in feats['test']])
    res = {'feature': 'mean grey of the top third of the photo (lpr.pipeline.night_level)',
           'best_threshold_on_val': best[1], 'val_night_caught_%': round(100 * best[2], 1), 'val_day_flagged_%': round(100 * best[3], 1),
           'test_night_caught_%': round(100 * ((vt < best[1]) & yt).sum() / yt.sum(), 1),
           'test_day_flagged_%': round(100 * ((vt < best[1]) & ~yt).sum() / (~yt).sum(), 1),
           'night_photos': {'val': int(y.sum()), 'test': int(yt.sum())}}
    print('[0] night from the image:', json.dumps(res), flush=True)
    return res


# ------------------------------------------------------------------------------------------------ 1. reading
def all_readings(s, imgs):
    """every reader on every crop, on the crop and on a CLAHE copy: per crop {(name, 'o'|'c'): Reading}."""
    out = [dict() for _ in imgs]
    for kind, batch in (('o', imgs), ('c', [clahe(i) for i in imgs])):
        for nm, rd in zip(s.names, s.readers):
            for k in range(0, len(batch), 64):
                for i, r in enumerate(rd.read(batch[k:k + 64])):
                    out[k + i][(nm, kind)] = r
    return out


def fuse_variant(s, R, variant):
    """offline emulation of LPRSystem.read_crops + the night variants (checked against read_crops for 'v2')."""
    n0, n1 = s.names[0], s.names[1]
    t = lambda nm, k: R[(nm, k)].text
    rd = lambda nm, k: (nm if k == 'o' else nm + '+clahe', R[(nm, k)].text, R[(nm, k)].conf)
    if variant in ('v2', 'clahe_on_doubt'):
        ran = [rd(n0, 'o'), rd(n1, 'o')]
        if fusion.needs_helpers('verify', t(n0, 'o'), t(n1, 'o')):
            ran += [rd(nm, 'o') for nm in s.names[2:]]
        text = fusion.fuse(ran, 'verify')
        st = fusion.status(text, ran)
        if variant == 'clahe_on_doubt' and st != 'verified':
            ran = [rd(nm, 'o') for nm in s.names] + [rd(nm, 'c') for nm in s.names]
            text = metrics.fuse(ran, 'MV-BM+format', n0)
            st = fusion.status(text, ran[:len(s.names)])
        return text, st
    ran = [rd(nm, 'o') for nm in s.names] + ([rd(nm, 'c') for nm in s.names] if variant == 'vote10' else [])
    text = metrics.fuse(ran, 'MV-BM+format', n0)
    # status from the five readers on the crop only: a CLAHE copy read by the same reader is not an independent
    # opinion (run 1 counted it, and 'verified' plates got less often right: val 94.87 -> 92.86 %)
    return text, fusion.status(text, ran[:len(s.names)])


VARIANTS = ['v2', 'vote5', 'clahe_on_doubt', 'vote10']


def part1(s):
    res, pick = {}, None
    split_rows = {sp: rows(os.path.join(DATA, 'ocr', f'{sp}.csv')) for sp in ('val', 'test')}
    tc = {r['file']: r for r in rows(os.path.join(DATA, 'ocr', 'test_conditions.csv'))}
    for sp in ('val', 'test'):
        rr = split_rows[sp]
        imgs = [cv2.imread(os.path.join(DATA, 'ocr', 'images', sp, r['file'])) for r in rr]
        night = [is_night(r['file'], im) for r, im in zip(rr, imgs)] if sp == 'val' else [tc[r['file']]['night'] == '1' for r in rr]
        R = all_readings(s, imgs)
        if sp == 'val':                                        # the emulation must equal the engine
            eng = s.read_crops(imgs[:120])
            same = sum(e['text'] == fuse_variant(s, R[i], 'v2')[0] for i, e in enumerate(eng))
            print(f'[1] emulation check: {same}/120 crops identical to LPRSystem.read_crops', flush=True)
            res['emulation_check'] = f'{same}/120'
        variants = VARIANTS if sp == 'val' else ['v2', pick]
        ok = {}
        for v in variants:
            pr = [fuse_variant(s, R[i], v) for i in range(len(rr))]
            ok[v] = [right(p[0], r['text']) for p, r in zip(pr, rr)]
            stv = [p[1] for p in pr]
            nn = [i for i in range(len(rr)) if night[i]]
            ver = [i for i in nn if stv[i] == 'verified']
            res[f'{sp}|{v}'] = {'night_crops': len(nn), 'night_exact_%': round(100 * sum(ok[v][i] for i in nn) / len(nn), 2),
                                'all_crops': len(rr), 'all_exact_%': round(100 * sum(ok[v]) / len(rr), 2),
                                'night_verified_%': round(100 * len(ver) / len(nn), 2),
                                'night_verified_right_%': round(100 * sum(ok[v][i] for i in ver) / max(1, len(ver)), 2),
                                'night_review_%': round(100 * sum(stv[i] == 'review' for i in nn) / len(nn), 2)}
            print(f'[1] {sp} {v}:', json.dumps(res[f'{sp}|{v}']), flush=True)
        if sp == 'val':
            # rule fixed before looking: best val night exact; must not lose more than 0.5 pt on all val crops;
            # ties -> the cheaper variant (order of VARIANTS)
            base_all = res['val|v2']['all_exact_%']
            ok_v = [v for v in VARIANTS if res[f'val|{v}']['all_exact_%'] >= base_all - 0.5]
            pick = max(ok_v, key=lambda v: (res[f'val|{v}']['night_exact_%'], -VARIANTS.index(v)))
            res['picked_on_val'] = pick
            print('[1] picked on val:', pick, flush=True)
        else:
            nn = [i for i in range(len(rr)) if night[i]]
            res['test_paired_night'] = paired([ok[pick][i] for i in nn], [ok['v2'][i] for i in nn])
            res['test_paired_all'] = paired(ok[pick], ok['v2'])
    return res


# ------------------------------------------------------------------------------------------------ 2. detection
def part2(s):
    d = os.path.join(DATA, 'detection', 'images', 'val')
    files = [f for f in sorted(os.listdir(d))]
    night = []
    for f in files:
        im = cv2.imread(os.path.join(d, f))
        if is_night(f, im):
            night.append((f, im))
    res = {'val_night_photos': len(night)}
    for conf in (0.25, 0.15, 0.10):
        found = plates = extra = 0
        for f, im in night:
            h, w = im.shape[:2]
            lab = os.path.join(DATA, 'detection', 'labels', 'val', os.path.splitext(f)[0] + '.txt')
            g = []
            if os.path.exists(lab):
                for l in open(lab).read().split('\n'):
                    v = l.split()
                    if len(v) >= 5:
                        cx, cy, bw, bh = map(float, v[1:5])
                        g.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            boxes = s.detect(im, night=True, conf=conf)
            used = set()
            for b in boxes:
                best, bj = 0, -1
                for j, gb in enumerate(g):
                    ix = max(0, min(b[2], gb[2]) - max(b[0], gb[0])) * max(0, min(b[3], gb[3]) - max(b[1], gb[1]))
                    u = (b[2] - b[0]) * (b[3] - b[1]) + (gb[2] - gb[0]) * (gb[3] - gb[1]) - ix
                    if u > 0 and ix / u > best:
                        best, bj = ix / u, j
                if best >= 0.5 and bj not in used:
                    used.add(bj)
                else:
                    extra += 1
            found += len(used)
            plates += len(g)
        res[f'conf_{conf}'] = {'plates': plates, 'found': found, 'found_%': round(100 * found / max(1, plates), 2), 'extra_boxes': extra}
        print(f'[2] val night, detector conf {conf}:', json.dumps(res[f'conf_{conf}']), flush=True)
    base = res['conf_0.25']
    # rule: lower confidence only if it finds more plates and adds at most as many extra boxes as plates it finds
    best = 0.25
    for conf in (0.15, 0.10):
        r = res[f'conf_{conf}']
        if r['found'] > base['found'] and (r['extra_boxes'] - base['extra_boxes']) <= (r['found'] - base['found']):
            best = conf if r['found'] > res[f'conf_{best}']['found'] else best
    res['picked_on_val'] = best
    print('[2] picked detector conf on val:', best, flush=True)
    return res


# ------------------------------------------------------------------------------------------------ 3. whole photos
def part3(s, read_variant, det_conf):
    e2e = rows(os.path.join(DATA, 'e2e_test', 'e2e_test.csv'))
    imgs = [cv2.imread(os.path.join(DATA, 'e2e_test', 'images', r['file'])) for r in e2e]
    s.night_cfg = {'det_conf': det_conf, 'read': read_variant}
    out = {}
    ok = {}
    for mode in (False, True):
        res = [s.process(im, night=mode) for im in imgs]
        top = [r['plates'][0]['text'] if r['plates'] else '' for r in res]
        ok[mode] = [right(t, r['text']) for t, r in zip(top, e2e)]
        stt = [r['plates'][0]['status'] if r['plates'] else 'none' for r in res]
        for sub, sel in (('night', lambda r: r['source'].startswith('nightlpbd')), ('day', lambda r: not r['source'].startswith('nightlpbd'))):
            idx = [i for i, r in enumerate(e2e) if sel(r)]
            out[f"{'night_mode' if mode else 'v2'}|{sub}"] = {
                'photos': len(idx), 'exact_%': round(100 * sum(ok[mode][i] for i in idx) / len(idx), 2),
                'found_%': round(100 * sum(bool(res[i]['plates']) for i in idx) / len(idx), 2),
                'verified_%': round(100 * sum(stt[i] == 'verified' for i in idx) / len(idx), 2),
                'ms_per_photo': round(float(np.mean([res[i]['ms']['total'] for i in idx])), 1)}
            print(f"[3] {'night mode' if mode else 'V2'} on {sub} photos:", json.dumps(out[f"{'night_mode' if mode else 'v2'}|{sub}"]), flush=True)
    for sub, sel in (('night', lambda r: r['source'].startswith('nightlpbd')), ('day', lambda r: not r['source'].startswith('nightlpbd'))):
        idx = [i for i, r in enumerate(e2e) if sel(r)]
        out[f'paired|{sub}'] = paired([ok[True][i] for i in idx], [ok[False][i] for i in idx])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    s = LPRSystem(V3)
    res = {'date': time.strftime('%Y-%m-%d %H:%M'), 'part0': part0(), 'part1': part1(s), 'part2': part2(s)}
    res['part3'] = part3(s, res['part1']['picked_on_val'], res['part2']['picked_on_val'])
    res['minutes'] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(os.path.join(OUT, 'night_mode.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved results/night_mode.json')


if __name__ == '__main__':
    main()
