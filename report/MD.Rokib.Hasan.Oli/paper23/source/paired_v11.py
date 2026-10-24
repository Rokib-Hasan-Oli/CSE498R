"""Paper #23, v1.1 run: per-plate reading of both readers on KUET's 194 published test crops (now the test of BOTH
versions) + exact McNemar tests. Also the 468 non-KUET plates of the 498R OCR test from the saved predictions.
Usage: python paired_v11.py <extracted results dir> <weights dir> <out json>"""
import csv
import json
import math
import os
import sys
from pathlib import Path

KIT = r'D:\498R\docs\Rokib Hasna\23_Do We Train on Test Data - The Impact of Near-Duplicates on License Plate Recognition\kaggle'
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')
sys.path.insert(0, KIT)
import p23_leaky as P  # noqa: E402
import vocab  # noqa: E402

DS = Path(r'D:\498R\dataset')
RES, W, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
names = json.load(open(DS / 'vocab.json', encoding='utf-8'))['classes']


def mcnemar(b, c):
    """exact two-sided McNemar p from the discordant counts"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(100 * (c - h), 1), round(100 * (c + h), 1)


def compare(L, F, keys):
    a = sum(L[k] and F[k] for k in keys)
    b = sum(L[k] and not F[k] for k in keys)
    c = sum(F[k] and not L[k] for k in keys)
    n = len(keys)
    return {'plates': n, 'leaky_%': round(100 * sum(L[k] for k in keys) / n, 2), 'fair_%': round(100 * sum(F[k] for k in keys) / n, 2),
            'leaky_ci95': wilson(sum(L[k] for k in keys), n), 'fair_ci95': wilson(sum(F[k] for k in keys), n),
            'both_right': a, 'only_leaky_right': b, 'only_fair_right': c, 'both_wrong': n - a - b - c,
            'mcnemar_p': round(mcnemar(b, c), 4)}


from ultralytics import YOLO  # noqa: E402

leaky_dir = DS / 'extras' / 'leaky_kuet' / 'recognition'
test = sorted(str(p) for p in (leaky_dir / 'images' / 'test').iterdir())
fair_test = [str(DS / 'recognition' / l.strip()[2:]) for l in open(DS / 'recognition' / 'fair_kuet_test.txt', encoding='utf-8') if l.strip()]
print('published test crops', len(test), '| fair test list', len(fair_test),
      '| same ids', len({P.file_id(p) for p in test} & {P.file_id(p) for p in fair_test}))

read = {}
for tag in ('rec_leaky', 'rec_fair'):
    m = YOLO(str(W / f'best_{tag}.pt'))
    read[tag] = P.read_crops(m, test, names)
    v = [ok for _, ok in read[tag].values()]
    print(tag, 'plates', len(v), 'exact %', round(100 * sum(v) / len(v), 2))

# what the leaky reader saw: plate texts of its training crops
leaky_train = sorted(str(p) for p in (leaky_dir / 'images' / 'train').iterdir())
seen_txt = set()
for p in leaky_train:
    t = P.label_text(p, names)[0]
    if t:
        seen_txt.add(t)
fair_train = [str(DS / 'recognition' / l.strip()[2:]) for l in open(DS / 'recognition' / 'fair_kuet_train.txt', encoding='utf-8') if l.strip()]
fair_txt = {P.label_text(p, names)[0] for p in fair_train} - {''}
print('leaky train crops', len(leaky_train), 'plates', len(seen_txt), '| fair train crops', len(fair_train), 'plates', len(fair_txt))

L = {k: ok for k, (t, ok) in read['rec_leaky'].items()}
F = {k: ok for k, (t, ok) in read['rec_fair'].items()}
txt = {k: t for k, (t, ok) in read['rec_leaky'].items()}
keys = sorted(set(L) & set(F))
B = [k for k in keys if txt[k] in seen_txt]
C = [k for k in keys if txt[k] not in seen_txt]
out = {'all_194': compare(L, F, keys), 'B_plate_in_leaky_train': compare(L, F, B), 'C_unseen_by_both': compare(L, F, C),
       'fair_train_plates_in_test': sum(txt[k] in fair_txt for k in keys),
       'leaky_train': {'crops': len(leaky_train), 'plates': len(seen_txt)}, 'fair_train': {'crops': len(fair_train), 'plates': len(fair_txt)}}

# the 468 non-KUET plates of the 498R OCR test (saved predictions)
pr = {}
for tag in ('rec_leaky', 'rec_fair'):
    pr[tag] = {r['file']: vocab.norm_text(r['pred']) == vocab.norm_text(r['true'])
               for r in csv.DictReader(open(RES / 'runs' / tag / 'ocr_preds_test.csv', encoding='utf-8')) if r['source'] != 'kuet'}
ks = sorted(set(pr['rec_leaky']) & set(pr['rec_fair']))
out['non_kuet_468'] = compare(pr['rec_leaky'], pr['rec_fair'], ks)
for k, v in out.items():
    print(k, v)
json.dump(out, open(OUT, 'w', encoding='utf-8'), indent=1)
