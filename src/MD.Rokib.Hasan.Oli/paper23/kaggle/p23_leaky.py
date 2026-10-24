"""Paper #23 on our data: how much do near-duplicates between train and test inflate the scores?

One source (KUET, used by papers #1/#6) in two versions, same models and settings:
  leaky : KUET's own published train/valid/test with every Roboflow copy (extras/leaky_kuet/)
  fair  : the same KUET images, one copy each (detection|recognition/data_fair_kuet.yaml). Detection: our grouped
          split. Reading crops (dataset v1.1): the fair test is KUET's 194 published test crops, the other photos of
          those plates are left out, the rest is our split (our test crops join the fair train)

  python p23_leaky.py yaml                  -> writes the yaml files (absolute paths) and prints them as JSON
  (train det.py / rec.py on leaky and on fair with these yaml files; fair_epochs() gives the fair runs the
   epochs that match the leaky optimiser steps, since the fair train has 3-4x fewer images)
  python p23_leaky.py score --det-leaky runs/det_leaky --det-fair runs/det_fair \
                            --rec-leaky runs/rec_leaky --rec-fair runs/rec_fair --name leaky_vs_fair --project runs

score: every model on every test set
  leaky_test_all   : KUET's published test (what papers report)
  leaky_test_clean : its photos with NO near-duplicate (same scene group, or same plate text) in the published train
  leaky_test_dup   : the rest
  fair_test        : KUET photos in our test split
gap = all - clean. relative_gap_paper_% = gap / (100 - all): the rise of the error rate, as in paper #23 (p.4,
112-290 % on AOLP); relative_gap_% = gap / clean is the older form. Detection: mAP50, mAP50-95.
Recognition (character YOLO): mAP50 of the boxes and plate exact % (tokens joined by vocab.plate_text).

Seen / unseen: the two versions split KUET differently, so a model scored on the OTHER version's test may have trained
on some of those very items. In dataset v1.0 the fair reading split had 135 of the 194 published test crops in train
and the leaky train had 104 of the 183 fair test crops; from v1.1 the two reading tests are the same 194 crops and the
fair train holds none of their plates, so only the leaky reader's copies remain (the leak being measured). Each item
of the two whole test sets is put in a group per model:
  A = the identical image is in that model's train (same md5[:12] in the name kuet_<md5[:12]>[_<copy>].jpg)
  B = not A, but the same plate text is in that model's train (readers only)
  C = unseen
models.<m>.<set>.seen_in_train has the counts and `unseen` the scores on C only; rec_seen_unseen has the plate exact
of BOTH readers per group (groups by either reader's train, and the plates neither reader saw); report_extra.md shows
it as tables. ocr_test_without_kuet: both readers on the 498R OCR test without its KUET crops (by condition).
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import vocab

SETS = ('leaky_test_all', 'leaky_test_clean', 'leaky_test_dup', 'fair_test')
FULL = ('leaky_test_all', 'fair_test')            # the two whole test sets (clean / dup are parts of leaky_test_all)
GROUPS = ('A_same_crop_in_train', 'B_same_plate_text_in_train', 'C_unseen')
GLABEL = {GROUPS[0]: 'A: identical crop in train', GROUPS[1]: 'B: same plate text in train', GROUPS[2]: 'C: unseen'}
MODELS = (('det_leaky', 'detection', 'leaky'), ('det_fair', 'detection', 'fair'),
          ('rec_leaky', 'recognition', 'leaky'), ('rec_fair', 'recognition', 'fair'))
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def yamls(out_dir=None):
    root = Path(common.find_dataset())
    out_dir = Path(out_dir or os.path.join(common.WORK, 'yaml'))
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for part in ('detection', 'recognition'):
        names = open(root / part / 'data.yaml', encoding='utf-8').read().split('nc:', 1)[1]
        leaky = root / 'extras' / 'leaky_kuet' / part
        fair = root / part
        variants = {
            'leaky': (leaky, 'images/train', 'images/val', 'images/test'),
            'leaky_test_all': (leaky, 'images/train', 'images/val', 'images/test'),
            'leaky_test_clean': (leaky, 'images/train', 'images/val', 'test_clean.txt'),
            'leaky_test_dup': (leaky, 'images/train', 'images/val', 'test_dup.txt'),
            'fair': (fair, 'fair_kuet_train.txt', 'fair_kuet_val.txt', 'fair_kuet_test.txt'),
            'fair_test': (fair, 'fair_kuet_train.txt', 'fair_kuet_val.txt', 'fair_kuet_test.txt'),
        }
        for k, (path, tr, va, te) in variants.items():
            f = out_dir / f'{part}_{k}.yaml'
            f.write_text(f'path: {path}\ntrain: {tr}\nval: {va}\ntest: {te}\nnc:{names}', encoding='utf-8')
            res[f'{part}_{k}'] = str(f)
    return res


def subset_yaml(base_yaml, files, name):
    """copy of base_yaml whose test split is only `files` (absolute paths, listed in <name>.txt next to it)"""
    import yaml
    base = Path(base_yaml)
    lst = base.parent / f'{name}.txt'
    lst.write_text('\n'.join(files) + '\n', encoding='utf-8')
    y = yaml.safe_load(open(base, encoding='utf-8'))
    names = base.read_text(encoding='utf-8').split('nc:', 1)[1]
    f = base.parent / f'{name}.yaml'
    f.write_text(f"path: {y['path']}\ntrain: {y['train']}\nval: {y['val']}\ntest: {lst}\nnc:{names}", encoding='utf-8')
    return str(f)


def list_images(yaml_file, split='test'):
    import yaml
    y = yaml.safe_load(open(yaml_file, encoding='utf-8'))
    t = Path(y['path']) / y[split]
    if t.suffix == '.txt':
        lines = [l.strip() for l in open(t, encoding='utf-8') if l.strip()]
        return [str(t.parent / l[2:]) if l.startswith('./') else l for l in lines]
    return sorted(str(p) for p in t.iterdir() if p.suffix.lower() in IMG_EXT)


def fair_epochs(Y, part, epochs, batch):
    """epochs for the fair run that give about as many optimiser steps as `epochs` leaky epochs"""
    n = {v: len(list_images(Y[f'{part}_{v}'], 'train')) for v in ('leaky', 'fair')}
    return max(epochs, round(epochs * math.ceil(n['leaky'] / batch) / math.ceil(n['fair'] / batch)))


def file_id(path):
    """kuet_<md5[:12]>[_<copy>].jpg -> md5[:12]: the same id in both versions = the identical image"""
    return Path(path).stem.split('_')[1]


def label_of(img):
    return Path(img.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep).replace('/images/', '/labels/')).with_suffix('.txt')


def label_text(img, names, size=None):
    """(normalised plate text, valid) from the character labels of a crop; size = (w, h), else read with PIL"""
    lab = label_of(img)
    if not lab.exists():
        return '', False
    if size is None:
        from PIL import Image
        with Image.open(img) as im:
            size = im.size
    w, h = size
    toks = []
    for l in open(lab, encoding='utf-8').read().splitlines():
        if not l.strip():
            continue
        c, x, y, bw, bh = l.split()[:5]
        x, y, bw, bh = map(float, (x, y, bw, bh))
        toks.append([names[int(c)], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    text, valid = vocab.plate_text(toks)
    return vocab.norm_text(text), valid


def train_index(yaml_file, part, names):
    """(ids of the training images, plate texts in train): what a model trained on yaml_file has seen"""
    imgs = list_images(yaml_file, 'train')
    texts = set()
    if part == 'recognition':
        for p in imgs:
            t = label_text(p, names)[0]
            if t:                                   # any text, also a crop whose layout check failed
                texts.add(t)
    return {file_id(p) for p in imgs}, texts


def group(name, text, seen):
    ids, texts = seen
    if file_id(name) in ids:
        return GROUPS[0]
    return GROUPS[1] if text and text in texts else GROUPS[2]


def read_crops(model, imgs, names):
    """{file name: (true text, read right?)} for every crop whose label is one valid plate"""
    out = {}
    for s in range(0, len(imgs), 32):
        for p, r in zip(imgs[s:s + 32], model.predict(imgs[s:s + 32], imgsz=320, conf=0.25, verbose=False)):
            h, w = r.orig_shape
            truth, valid = label_text(p, names, (w, h))
            if not valid:
                continue
            pred = vocab.plate_text([[r.names[int(c)]] + b.tolist() for b, c in
                                     zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())])[0]
            out[Path(p).name] = (truth, vocab.norm_text(pred) == truth)
    return out


def exact(crops, keep=None):
    """(plate exact %, plates); the % is None when no plate is left"""
    v = [ok for n, (t, ok) in crops.items() if keep is None or n in keep]
    return (round(100 * sum(v) / len(v), 2) if v else None), len(v)


def box_map(m, data, part, a, run_name):
    v = m.val(data=data, split='test', imgsz=640 if part == 'detection' else 320, batch=16, workers=common.WORKERS,
              plots=False, verbose=False, project=str(Path(a.project) / a.name), name=run_name, exist_ok=True)
    return {'mAP50_%': round(float(v.box.map50) * 100, 2), 'mAP50-95_%': round(float(v.box.map) * 100, 2)}


def train_info(run, n_images):
    """epochs, batch and about how many optimiser steps a run trained for"""
    import yaml
    r = common.load_json(Path(run) / 'results.json') if (Path(run) / 'results.json').exists() else {}
    y = yaml.safe_load(open(Path(run) / 'args.yaml', encoding='utf-8')) if (Path(run) / 'args.yaml').exists() else {}
    ep, b, fr = r.get('epochs_run') or y.get('epochs'), y.get('batch') or 16, y.get('fraction') or 1.0
    return {'train_images': n_images, 'epochs_run': ep, 'batch': b, 'fraction': fr,
            'optimiser_steps': ep and int(ep * math.ceil(n_images * fr / b))}


def score_both(P, k, keep):
    L, F = (exact(P[t][k], keep) for t in ('rec_leaky', 'rec_fair'))
    return {'plates': L[1], 'rec_leaky_%': L[0], 'rec_fair_%': F[0],
            'leaky_minus_fair_pp': round(L[0] - F[0], 2) if L[1] else None}


def without_kuet(run, a, root):
    """a reader on the 498R OCR test (and its extra splits) without the KUET crops"""
    import rec
    f = Path(run) / 'ocr_preds_test.csv'
    w = Path(run) / 'weights' / 'best.pt'
    preds = list(csv.DictReader(open(f, encoding='utf-8'))) if f.exists() else rec.read_plates(w, 'test', 320, max_n=a.ocr_max)[0]
    preds = [p for p in preds if p['source'] != 'kuet']
    out = {'rows_decoder': metrics.score_rows((p['pred'], p['true']) for p in preds),
           'by_condition': metrics.condition_scores(preds, root), 'extra_tests': {}}
    for sp in metrics.extra_splits(root):
        ep = [p for p in rec.read_plates(w, sp, 320, max_n=a.ocr_max)[0] if p['source'] != 'kuet']
        out['extra_tests'][sp] = metrics.score_rows((p['pred'], p['true']) for p in ep)
    return out


def score(a):
    from ultralytics import YOLO
    Y = yamls()
    root = common.find_dataset()
    names = common.load_json(os.path.join(root, 'vocab.json'))['classes']
    res = {'run': a.name, 'kind': 'leaky-vs-fair', 'sets': {}, 'train': {}, 'models': {}}
    for k in SETS:
        res['sets'][k] = {part: len(list_images(Y[f'{part}_{k}'])) for part in ('detection', 'recognition')}
    P, seen_by = {}, {}                               # reader -> set -> crops; model -> what it trained on
    flags = []
    for tag, part, ver in MODELS:
        run = getattr(a, tag)
        if not run or not (Path(run) / 'weights' / 'best.pt').exists():
            continue
        seen = seen_by[tag] = train_index(Y[f'{part}_{ver}'], part, names)
        res['train'][tag] = train_info(run, len(list_images(Y[f'{part}_{ver}'], 'train')))
        m = YOLO(str(Path(run) / 'weights' / 'best.pt'))
        out = {}
        for k in SETS:
            out[k] = box_map(m, Y[f'{part}_{k}'], part, a, f'{tag}_{k}')
        for k in FULL:
            imgs = list_images(Y[f'{part}_{k}'])
            if part == 'recognition':
                crops = P.setdefault(tag, {})[k] = read_crops(m, imgs, names)
                g = {n: group(n, t, seen) for n, (t, ok) in crops.items()}
                out[k]['plate_exact_%'], out[k]['plates'] = exact(crops)
                out[k]['seen_in_train'] = {G: sum(v == G for v in g.values()) for G in GROUPS}
                out[k]['plate_exact_by_group'] = {G: dict(zip(('plate_exact_%', 'plates'), exact(crops, {n for n in g if g[n] == G})))
                                                  for G in GROUPS}
                unseen = [p for p in imgs if g.get(Path(p).name) == GROUPS[2]]
                n_seen = len(crops) - len(unseen)
            else:
                unseen = [p for p in imgs if file_id(p) not in seen[0]]
                out[k]['seen_in_train'] = {'A_same_photo_in_train': len(imgs) - len(unseen), 'not_in_train': len(unseen)}
                n_seen = len(imgs) - len(unseen)
            if n_seen and unseen:                     # scores again on the items this model never trained on
                u = box_map(m, subset_yaml(Y[f'{part}_{k}'], unseen, f'{part}_{k}_unseen_by_{tag}'), part, a, f'{tag}_{k}_unseen')
                if part == 'recognition':
                    u.update(out[k]['plate_exact_by_group'][GROUPS[2]])
                out[k]['unseen'] = dict(u, images=len(unseen))
            if n_seen:
                s = out[k]['seen_in_train']
                what = (f"{s[GROUPS[0]]} identical crops + {s[GROUPS[1]]} with the same plate text" if part == 'recognition'
                        else f"{s['A_same_photo_in_train']} identical photos")
                own = (ver == 'leaky') == (k == 'leaky_test_all')
                flags.append(f"{tag} on {k}: {what} of {n_seen + len(unseen)} are in its train"
                             + (' (its own published split: the near-duplicates the paper is about)' if own else
                                ' -> cross-score contaminated, use `unseen`'))
        if part == 'recognition':                     # clean / dup are parts of the published test: same predictions
            for k in ('leaky_test_clean', 'leaky_test_dup'):
                keep = {Path(p).name for p in list_images(Y[f'{part}_{k}'])}
                out[k]['plate_exact_%'], out[k]['plates'] = exact(P[tag]['leaky_test_all'], keep)
        main = 'plate_exact_%' if part == 'recognition' else 'mAP50-95_%'
        al, cl = out['leaky_test_all'][main], out['leaky_test_clean'][main]
        if al is not None and cl is not None:
            out['gap_all_minus_clean_pp'] = round(al - cl, 2)
            out['relative_gap_paper_%'] = round(100 * (al - cl) / max(1e-9, 100 - al), 2)
            out['relative_gap_%'] = round(100 * (al - cl) / max(1e-9, cl), 2)
        out['metric_for_gap'] = main
        res['models'][tag] = out
        print(tag, json.dumps(out), flush=True)
    for part, main in (('det', 'mAP50-95_%'), ('rec', 'plate_exact_%')):
        L, F = res['models'].get(f'{part}_leaky'), res['models'].get(f'{part}_fair')
        if L and F and None not in (L['leaky_test_all'][main], F['fair_test'][main]):
            g = L['leaky_test_all'][main] - F['fair_test'][main]      # each on its own test, as each paper would report it
            res[f'{part}_reported_leaky_vs_fair_pp'] = round(g, 2)
            res[f'{part}_reported_relative_gap_paper_%'] = round(100 * g / max(1e-9, 100 - L['leaky_test_all'][main]), 2)
    if len(P) == 2:                                   # both readers on the same plates, grouped by what each trained on
        res['rec_seen_unseen'] = {}
        for k in FULL:
            both = set(P['rec_leaky'][k]) & set(P['rec_fair'][k])
            g = {t: {n: group(n, P[t][k][n][0], seen_by[t]) for n in both} for t in P}
            d = {f'by_{t}_train': {G: score_both(P, k, {n for n in both if g[t][n] == G}) for G in GROUPS} for t in P}
            d['unseen_by_both'] = score_both(P, k, {n for n in both if all(g[t][n] == GROUPS[2] for t in P)})
            res['rec_seen_unseen'][k] = d
        res['ocr_test_without_kuet'] = {t: without_kuet(getattr(a, t), a, root) for t in P}
    res['cross_scores_with_training_data'] = flags
    res['env'] = common.env_info()
    common.save_json(res, Path(a.project) / a.name / 'results.json')
    md = report_md(res)
    (Path(a.project) / a.name / 'report_extra.md').write_text(md, encoding='utf-8')
    print(md)
    print('DONE', a.name)


def report_md(res):
    """tables for results_vs_paper.md: seen / unseen check, contaminated cross-scores, readers without KUET"""
    f = lambda v: '–' if v is None else str(v)
    pp = lambda v: '–' if v is None else f'{v:+.2f}'
    out = []
    su = res.get('rec_seen_unseen')
    if su:
        out += ['', '## Seen / unseen check (plate readers)', '',
                "The two KUET versions split the crops differently, so a reader scored on the other version's test has "
                'trained on some of those crops. Groups per reader: A = the identical crop is in its train, B = the same '
                'plate text is in its train (another crop), C = unseen. Plate exact % of both readers on each group; '
                'only the rows "unseen by both" compare the two readers fairly.', '',
                '| Test set | Groups by the train of | Group | Plates | rec_leaky (%) | rec_fair (%) | leaky - fair (pp) |',
                '|' + '---|' * 7]
        for k, d in su.items():
            for by, gs in d.items():
                if by == 'unseen_by_both':
                    continue
                for G, s in gs.items():
                    out.append(f"| {k} | {by[3:-6]} | {GLABEL[G]} | {s['plates']} | {f(s['rec_leaky_%'])} | {f(s['rec_fair_%'])} | "
                               f"{pp(s['leaky_minus_fair_pp'])} |")
            s = d['unseen_by_both']
            out.append(f"| {k} | both | **unseen by both** | {s['plates']} | {f(s['rec_leaky_%'])} | {f(s['rec_fair_%'])} | "
                       f"{pp(s['leaky_minus_fair_pp'])} |")
    if res.get('cross_scores_with_training_data'):
        out += ['', '## Scores that include training data', ''] + [f'- {x}' for x in res['cross_scores_with_training_data']]
        rows = [(t, k, d[k]['unseen']) for t, d in res['models'].items() for k in FULL
                if 'unseen' in d.get(k, {}) and t.endswith('_leaky') != (k == 'leaky_test_all')]
        if rows:
            out += ['', 'Cross-scores again on the items the model never trained on (identical photo / crop and, for '
                        'readers, the same plate text left out):', '',
                    '| Model | Test set | Images | mAP50 (%) | mAP50-95 (%) | plate exact (%) | on the whole set |',
                    '|' + '---|' * 7]
            for t, k, u in rows:
                main = res['models'][t]['metric_for_gap']
                out.append(f"| {t} | {k} | {u['images']} | {u['mAP50_%']} | {u['mAP50-95_%']} | {f(u.get('plate_exact_%'))} | "
                           f"{res['models'][t][k][main]} ({main}) |")
    tr = res.get('train')
    if tr:
        out += ['', '## Training length', '', '| Run | train images | epochs run | batch | optimiser steps (about) |',
                '|' + '---|' * 5] + [f"| {t} | {d['train_images']} | {f(d['epochs_run'])} | {d['batch']} | {f(d['optimiser_steps'])} |"
                                     for t, d in tr.items()]
    wk = res.get('ocr_test_without_kuet')
    if wk:
        out += ['', '## Plate readers by condition, without the KUET crops (plate exact %)', '',
                'Both readers trained on KUET only; with the KUET crops left out (rec_leaky trained on many of them) the '
                'rest of the 498R OCR test is a cross-dataset test for both.', '',
                '| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |', '|' + '---|' * 7]
        for t, d in wk.items():
            c, e = d.get('by_condition') or {}, d.get('extra_tests') or {}
            cells = [f((c.get(x) or {}).get('plate_exact_%')) for x in ('clear', 'night', 'blurry')]
            cells += [f((e.get(x) or {}).get('plate_exact_%')) for x in ('test_fog', 'test_rain_synth')]
            out.append(f"| {t} (no KUET, {d['rows_decoder']['plates']} plates) | {d['rows_decoder']['plate_exact_%']} | "
                       + ' | '.join(cells) + ' |')
    return '\n'.join(out) + ('\n' if out else '')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['yaml', 'score'])
    p.add_argument('--det-leaky', default='')
    p.add_argument('--det-fair', default='')
    p.add_argument('--rec-leaky', default='')
    p.add_argument('--rec-fair', default='')
    p.add_argument('--name', default='leaky_vs_fair')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    p.add_argument('--ocr-max', type=int, default=None, help='first N crops of each OCR split (quick test)')
    a = p.parse_args()
    if a.mode == 'yaml':
        print(json.dumps(yamls(), indent=1))
    else:
        (Path(a.project) / a.name).mkdir(parents=True, exist_ok=True)
        score(a)
