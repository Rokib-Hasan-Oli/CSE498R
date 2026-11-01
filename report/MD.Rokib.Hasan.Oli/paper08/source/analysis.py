"""Paper #8 v1.1 Kaggle run (22 Oct 2026): re-analysis of the saved outputs for the study report.

Reads `After Train/results.zip` directly (no extraction) and the 498R dataset v1.1 at D:/498R/dataset (token names,
train counts, condition tags). Writes analysis.json next to this file.
  - run health: jobs and exit codes (paper-8.log), epochs, dataset version
  - classifiers: accuracy per token group (digits / Metro / series letters / district words) from the best-val
    confusion matrices, macro F1 over the classes with >= 20 test crops, the most frequent confusions
  - plate pipeline: per-plate similarity with vs without GFP-GAN (same 651 plates, Wilcoxon signed-rank test),
    how many tokens the segmentation produced vs the truth, results per condition
Run with the system Python (numpy, scipy, yaml): python analysis.py
"""
import csv
import io
import json
import os
import re
import sys
import unicodedata
import zipfile
from collections import Counter

import numpy as np
import yaml
from PIL import Image
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
ZIP = os.path.join(PAPER_DIR, 'After Train', 'results.zip')
LOG = os.path.join(PAPER_DIR, 'After Train', 'paper-8.log')
DATA = 'D:/498R/dataset'
N = lambda s: unicodedata.normalize('NFC', s)

z = zipfile.ZipFile(ZIP)
J = lambda name: json.loads(z.read(name).decode('utf-8'))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(PAPER_DIR, 'kaggle'))   # the kit code; byte-identical to the run's code/ folder
import vocab  # noqa: E402

names = yaml.safe_load(open(f'{DATA}/recognition/data.yaml', encoding='utf-8'))['names']
names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
KIND = {n: vocab.KIND[N(n)] for n in names}
GROUP = {'digit': 'digits', 'metro': 'Metro', 'letter': 'series letters', 'district': 'district words'}

out = {}

# ------------------------------------------------------------------ run health
env = J('environment.json')
log = open(LOG, encoding='utf-8', errors='replace').read()
jobs = re.findall(r'\[done\]\s+(\S+) code=(\d+) after ([\d.]+) min', log)
out['run'] = {'dataset_version': env['dataset_version'], 'date': env['date'], 'torch': env['torch'], 'gpus': env['gpus'],
              'jobs': [{'name': n, 'code': int(c), 'min': float(m)} for n, c, m in jobs],
              'total_h': round(float(re.findall(r'^([\d.]+)s \d+ ', log, re.M)[-1]) / 3600, 2),
              'token_crops': dict(zip(('train', 'val', 'test'),
                                      map(int, re.findall(r'(\d+) token crops from', log)))),
              'classes_in_split': dict(zip(('train', 'val', 'test'),
                                           map(int, re.findall(r'plate crops, (\d+) classes', log)))),
              'test_class_without_train': re.findall(r"no train crop: \['(.+?)'\]", log)}
assert all(j['code'] == 0 for j in out['run']['jobs']) and len(jobs) == 5 and env['dataset_version'] == '1.1'


# ------------------------------------------------------------------ train counts per token (same rule as p08_chars.py)
def train_counts():
    c = Counter()
    root = f'{DATA}/recognition'
    for f in sorted(os.listdir(f'{root}/images/train')):
        lab = f'{root}/labels/train/{os.path.splitext(f)[0]}.txt'
        if not os.path.exists(lab):
            continue
        W, H = Image.open(f'{root}/images/train/{f}').size
        for line in open(lab).read().splitlines():
            if line.strip():
                k, _, _, w, h = line.split()[:5]
                if float(w) * W >= 4 and float(h) * H >= 4:
                    c[int(k)] += 1
    return c


TRAIN = train_counts()
assert sum(TRAIN.values()) == out['run']['token_crops']['train'], (sum(TRAIN.values()), out['run']['token_crops'])


# ------------------------------------------------------------------ classifiers
def per_class(conf):
    conf = np.array(conf)
    tp = np.diag(conf).astype(float)
    sup, pred = conf.sum(1), conf.sum(0)
    p = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    r = np.divide(tp, sup, out=np.zeros_like(tp), where=sup > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return conf, tp, sup, pred, p, r, f


clf = {}
for run in ('cnn28', 'resnet50'):
    r = J(f'runs/{run}/results.json')
    ids = [int(c[1:]) for c in r['classes']]           # folder cNNN -> data.yaml id NNN
    conf, tp, sup, pred, p, rr, f = per_class(r['test_best']['confusion'])
    groups = {}
    for kind, g in GROUP.items():
        m = np.array([KIND[names[i]] == kind for i in ids])
        groups[g] = {'n': int(sup[m].sum()), 'acc_%': round(100 * tp[m].sum() / max(1, sup[m].sum()), 2),
                     'classes_in_test': int((sup[m] > 0).sum())}
    big = sup >= 20
    rare = (sup > 0) & (sup < 20)
    conf_pairs = sorted(((int(conf[i, j]), names[ids[i]], names[ids[j]]) for i in range(len(ids)) for j in range(len(ids))
                         if i != j and conf[i, j] > 0), reverse=True)[:12]
    unseen = [names[ids[i]] for i in range(len(ids)) if sup[i] > 0 and TRAIN.get(ids[i], 0) == 0]
    lowtrain = (sup > 0) & np.array([TRAIN.get(i, 0) < 50 for i in ids])
    clf[run] = {
        'acc_best_%': r['chars_test']['best']['acc_%'], 'acc_last_%': r['chars_test']['last']['acc_%'],
        'macroF1_best_%': r['chars_test']['best']['F1_%'], 'macroF1_last_%': r['chars_test']['last']['F1_%'],
        'weightedF1_best_%': round(100 * r['test_best']['f1_weighted'], 2),
        'val_best_%': r['chars_val_best_%'], 'best_epoch': r['best_epoch'], 'epochs_run': r['epochs_run'],
        'epochs_planned': r['epochs_planned'], 'train_min': r['train_minutes'], 'params': r['params_total'],
        'params_trainable': r['params_trainable'], 'fps': r['speed']['fps_bs1'], 'ms': r['speed']['ms_per_image'],
        'groups': groups,
        'classes_with_test_crops': int((sup > 0).sum()),
        'classes_ge20_test': int(big.sum()), 'test_crops_in_ge20': int(sup[big].sum()),
        'macroF1_ge20_%': round(100 * f[big].mean(), 2),
        'acc_ge20_%': round(100 * tp[big].sum() / sup[big].sum(), 2),
        'classes_rare_test(1-19)': int(rare.sum()), 'acc_rare_%': round(100 * tp[rare].sum() / sup[rare].sum(), 2),
        'rare_classes_with_0_correct': int(((tp == 0) & rare).sum()),
        'acc_lowtrain(<50)_%': round(100 * tp[lowtrain].sum() / max(1, sup[lowtrain].sum()), 2),
        'n_lowtrain_test_crops': int(sup[lowtrain].sum()),
        'unseen_in_train_test_classes': unseen,
        'top_confusions': [{'n': n, 'true': a, 'pred': b} for n, a, b in conf_pairs],
        'curve': [dict(zip(('epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc'), map(float, row[:5])))
                  for row in list(csv.reader(io.StringIO(z.read(f'runs/{run}/log.csv').decode())))[1:]],
    }
out['classifiers'] = clf
c = clf['cnn28']['curve']
out['cnn_collapse'] = {'train_acc_ep5': c[4]['train_acc'], 'train_acc_ep40': c[39]['train_acc'],
                       'train_loss_min': min(e['train_loss'] for e in c), 'train_loss_ep40': c[39]['train_loss'],
                       'first_epoch_train_loss_up_for_good': next(e['epoch'] for e in c if e['train_loss'] > c[4]['train_loss'] and
                                                                 all(x['train_loss'] >= e['train_loss'] * 0.9 for x in c[int(e['epoch']):])),
                       'steps_per_epoch': out['run']['token_crops']['train'],
                       'paper_steps_total_cnn': 1292 * 40, 'our_steps_to_best': out['run']['token_crops']['train'] * 5}

# ------------------------------------------------------------------ plate pipeline
pr = J('runs/plates_cnn28/results.json')
out['plates'] = {k: v for k, v in pr['plates'].items()}
out['plates_ms'] = {'enhance': pr['ms_enhance'], 'read': pr['ms_read'], 'total': pr['ocr_test']['ms_per_plate']}
out['plates_by_condition'] = pr['ocr_test']['by_condition']

rows = list(csv.DictReader(io.StringIO(z.read('runs/plates_cnn28/per_plate.csv').decode('utf-8'))))
by = {}
for r in rows:
    by.setdefault(r['setting'], {})[r['file']] = r
files = list(by['none_paper'])
assert len(files) == 651 and all(len(v) == 651 for v in by.values())

VOC = sorted({N(n) for n in names}, key=len, reverse=True)


def split_tokens(s):
    """The pipeline's prediction is the CNN's tokens joined without spaces; cut it back greedily (longest token first)."""
    s, toks, i = N(s), [], 0
    while i < len(s):
        t = next((v for v in VOC if s.startswith(v, i)), None)
        if t is None:
            i += 1
            continue
        toks.append(t)
        i += len(t)
    return toks


paired = {}
for order in ('paper', 'fixed'):
    a = np.array([float(by[f'none_{order}'][f]['similarity']) for f in files])
    b = np.array([float(by[f'gfpgan_{order}'][f]['similarity']) for f in files])
    d = b - a
    paired[order] = {'mean_none_%': round(100 * a.mean(), 2), 'mean_gfpgan_%': round(100 * b.mean(), 2),
                     'mean_diff_points': round(100 * d.mean(), 2),
                     'gfpgan_better': int((d > 1e-9).sum()), 'gfpgan_worse': int((d < -1e-9).sum()),
                     'equal': int((np.abs(d) <= 1e-9).sum()),
                     'wilcoxon_p': float(f'{wilcoxon(b, a, zero_method="wilcox").pvalue:.3g}'),
                     'median_none': round(float(np.median(a)), 3), 'median_gfpgan': round(float(np.median(b)), 3)}
out['gfpgan_paired'] = paired

seg = {}
true_n = {f: len(vocab.text_tokens(by['none_fixed'][f]['true'])) for f in files}
for key in ('none_fixed', 'gfpgan_fixed'):
    pn = {f: len(split_tokens(by[key][f]['pred'])) for f in files}
    diff = np.array([pn[f] - true_n[f] for f in files])
    seg[key] = {'mean_pred_tokens': round(float(np.mean(list(pn.values()))), 2),
                'mean_true_tokens': round(float(np.mean(list(true_n.values()))), 2),
                'same_count_%': round(100 * float((diff == 0).mean()), 2),
                'too_many_%': round(100 * float((diff > 0).mean()), 2), 'too_few_%': round(100 * float((diff < 0).mean()), 2),
                'hist': {str(k): int(v) for k, v in sorted(Counter(np.clip(diff, -6, 8).tolist()).items())}}
out['segmentation_counts'] = seg

tags = {r['file']: r for r in csv.DictReader(open(f'{DATA}/ocr/test_conditions.csv', encoding='utf-8'))}
cond = {}
for tag in ('clear', 'night', 'blurry'):
    fs = [f for f in files if tags.get(f, {}).get(tag) == '1']
    cond[tag] = {'plates': len(fs)}
    for key in ('none_fixed', 'gfpgan_fixed'):
        cond[tag][key + '_sim_%'] = round(100 * float(np.mean([float(by[key][f]['similarity']) for f in fs])), 2)
out['similarity_by_condition'] = cond

ex = []          # examples for the report: one good, two typical, one bad plate (GFP-GAN + fixed order)
srt = sorted(files, key=lambda f: float(by['gfpgan_fixed'][f]['similarity']))
for f in (srt[-1], srt[len(srt) * 3 // 4], srt[len(srt) // 2], srt[5]):
    ex.append({'file': f, 'true': by['gfpgan_fixed'][f]['true'],
               **{k: {'pred': by[k][f]['pred'], 'sim': float(by[k][f]['similarity'])}
                  for k in ('none_paper', 'gfpgan_paper', 'none_fixed', 'gfpgan_fixed')}})
out['examples'] = ex

# ------------------------------------------------------------------ other readers on the same 651 OCR test crops (v1.1)
team = [r for r in csv.DictReader(open('D:/498R/docs/results/experiments.csv', encoding='utf-8'))
        if 'v1.1' in r['data'] and r['ocr_plate_exact_%'] and r['split'].startswith('All-in-One')]
out['same_test_other_readers'] = [{k: r[k] for k in ('run', 'paper', 'model', 'ocr_plate_exact_%', 'ocr_char_acc_%', 'fps_hardware')}
                                  for r in team]

out['enh_examples'] = sorted(n for n in z.namelist() if n.startswith('runs/plates_cnn28/enh_'))
json.dump(out, open(os.path.join(HERE, 'analysis.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ('classifiers', 'examples')}, ensure_ascii=False, indent=1)[:6000])
for run, v in clf.items():
    print(run, {k: v[k] for k in v if k not in ('curve', 'top_confusions')})
    print(run, 'confusions', v['top_confusions'][:8])
