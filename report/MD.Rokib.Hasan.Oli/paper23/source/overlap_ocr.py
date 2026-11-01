"""Paper #23 check: which KUET OCR test crops did each reader see in training (same file or same plate text)?
Uses the saved predictions only (no model is loaded)."""
import csv
import json
import os
import sys

from PIL import Image

KIT = r'D:\498R\docs\23_Do We Train on Test Data - The Impact of Near-Duplicates on License Plate Recognition\kaggle'
sys.path.insert(0, KIT)
import vocab  # noqa: E402

DS = r'D:\498R\dataset'
RES = sys.argv[1]
names = json.load(open(os.path.join(DS, 'vocab.json'), encoding='utf-8'))['classes']


def ids_dir(folder):
    return {f.split('_')[1].split('.')[0] for f in os.listdir(folder)}


def ids_txt(txt):
    return {os.path.basename(l.strip()).split('_')[1].split('.')[0] for l in open(txt, encoding='utf-8') if l.strip()}


def texts_of(img_dir):
    out = set()
    for f in os.listdir(img_dir):
        lab = os.path.join(img_dir.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep), os.path.splitext(f)[0] + '.txt')
        if not os.path.exists(lab):
            continue
        w, h = Image.open(os.path.join(img_dir, f)).size
        toks = []
        for l in open(lab, encoding='utf-8').read().splitlines():
            c, x, y, bw, bh = l.split()[:5]
            x, y, bw, bh = map(float, (x, y, bw, bh))
            toks.append([names[int(c)], (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
        t, ok = vocab.plate_text(toks)
        if ok:
            out.add(vocab.norm_text(t))
    return out


leaky = os.path.join(DS, 'extras', 'leaky_kuet', 'recognition', 'images')
L_train_ids = ids_dir(os.path.join(leaky, 'train'))
L_train_txt = texts_of(os.path.join(leaky, 'train'))
F_train_ids = ids_txt(os.path.join(DS, 'recognition', 'fair_kuet_train.txt'))
F_train_txt = texts_of_list = set()
for l in open(os.path.join(DS, 'recognition', 'fair_kuet_train.txt'), encoding='utf-8'):
    pass
# fair train texts from the OCR csv (same crops, human text)
ocr_train = {r['file'].split('_')[1].split('.')[0]: vocab.norm_text(r['text'])
             for r in csv.DictReader(open(os.path.join(DS, 'ocr', 'train.csv'), encoding='utf-8')) if r['source'] == 'kuet'}
F_train_txt = {ocr_train[i] for i in F_train_ids if i in ocr_train}
print('leaky train crops', len(L_train_ids), 'plate texts', len(L_train_txt), '| fair train crops', len(F_train_ids),
      'texts found', len(F_train_txt))

preds = {}
for run in ('rec_leaky', 'rec_fair'):
    for r in csv.DictReader(open(os.path.join(RES, 'runs', run, 'ocr_preds_test.csv'), encoding='utf-8')):
        if r['source'] == 'kuet':
            preds.setdefault(r['file'], {'true': vocab.norm_text(r['true'])})[run] = vocab.norm_text(r['pred']) == vocab.norm_text(r['true'])

groups = {}
for f, d in preds.items():
    i = f.split('_')[1].split('.')[0]
    seen_file = i in L_train_ids
    seen_plate = d['true'] in L_train_txt
    fair_seen = d['true'] in F_train_txt
    key = 'A: same crop file in leaky train' if seen_file else ('B: same plate (other photo) in leaky train' if seen_plate else 'C: plate never in leaky train')
    groups.setdefault(key, []).append(d)
    if fair_seen:
        groups.setdefault('(fair train has this plate - should be 0)', []).append(d)
out = {}
for k in sorted(groups):
    g = groups[k]
    n = len(g)
    a = 100 * sum(x['rec_leaky'] for x in g) / n
    b = 100 * sum(x['rec_fair'] for x in g) / n
    out[k] = {'crops': n, 'rec_leaky_%': round(a, 2), 'rec_fair_%': round(b, 2), 'diff_pp': round(a - b, 2)}
    print(f'{k:45s} n={n:4d}  leaky {a:6.2f}  fair {b:6.2f}  diff {a - b:+6.2f}')
n = len(preds)
print('all kuet', n, round(100 * sum(x['rec_leaky'] for x in preds.values()) / n, 2), round(100 * sum(x['rec_fair'] for x in preds.values()) / n, 2))
json.dump(out, open(os.path.join(RES, 'ocr_overlap_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
