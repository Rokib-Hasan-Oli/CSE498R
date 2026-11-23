"""Part E of the final-paper plan: make the "lite pack", extra training crops for ONE small reader, labelled by the
five V2 readers together (runs on this PC; the Kaggle notebook paper27lite_kaggle.ipynb then trains with the pack).

  python tools/build_lite_pack.py [--out D:/498R/lite_pack] [--max 300] [--kmin 4]

Every candidate crop is read by all five V2 readers (models/v2/config.json). A pseudo-label is the character boxes of
a char-YOLO reader whose text is the accepted text. Three parts, all from the TRAIN split only:
  human_text : train crops that have a human plate text but no character boxes in recognition/ (the dataset's own
               labeller did not reproduce the human text). Accepted when a char-YOLO reader reads exactly the human
               text, so the label is checked by a person's text.
  relabel    : the dataset's auto-labelled Zenodo crops (one model made those labels). An auto label stays as it is
               when at least 3 of the 5 readers read the same text. Otherwise the crop is dropped from training, or,
               when at least KMIN readers agree on another valid plate, re-labelled with that reading.
  det_new    : plates cut from the detection TRAIN photos (box + 8 %, as V2 crops them) that have no text label.
               Accepted when at least KMIN of 5 readers agree on one valid plate, a char-YOLO reader is among them,
               and that text is not a val / test / e2e plate (so no test plate can enter training this way).
Before the pack is built the same rule is checked on the OCR val crops (saved readings in eval/preds): how often is
a text with k of 5 readers agreeing right? That is the expected purity of the pseudo-labels.
Writes <out>/BanglaLPR_lite_pack_v1/ (images/, labels/, lite_pack_manifest.csv, drop_recognition_train.txt,
summary.json, README.md) and <out>/BanglaLPR_lite_pack_v1.zip.
"""
import argparse
import collections
import csv
import json
import os
import shutil
import sys
import time
import zipfile

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
SYS = os.path.join(HERE, '..')
sys.path.insert(0, SYS)
from lpr import fusion, metrics, vocab  # noqa: E402
from lpr.pipeline import crop  # noqa: E402
from lpr.readers import build_reader  # noqa: E402

DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
CFG = json.load(open(os.path.join(SYS, 'models', 'v2', 'config.json'), encoding='utf-8'))
NAME = 'BanglaLPR_lite_pack_v1'
KEEP_AUTO = 3          # readers (of 5) that must read an existing auto label for it to stay
MIN_H, MIN_W, PER_TEXT = 20, 40, 2


def rows(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def val_purity():
    """On the OCR val crops: share of right texts among plates where k of 5 readers agree (and one is a char-YOLO)."""
    P = {r['name']: {x['file']: x for x in rows(os.path.join(SYS, 'eval', 'preds', f"{r['name']}_val.csv"))}
         for r in CFG['readers']}
    yolo = [r['name'] for r in CFG['readers'] if r['kind'] == 'charyolo']
    names = list(P)
    stat = collections.defaultdict(lambda: [0, 0])
    for f in sorted(set.intersection(*[set(v) for v in P.values()])):
        full = [(n, P[n][f]['pred'], float(P[n][f]['conf'])) for n in names]
        text = metrics.fuse(full, 'MV-BM+format', names[0])
        k = sum(fusion.same(text, t) for _, t, _ in full)
        if metrics.is_valid_plate(text) and any(fusion.same(text, P[n][f]['pred']) for n in yolo):
            stat[k][0] += 1
            stat[k][1] += int(fusion.same(text, P[names[0]][f]['true']))
    return {f'{k}_of_5': {'plates': a, 'right_%': round(100 * b / a, 2)} for k, (a, b) in sorted(stat.items())}


def yolo_lines(boxes, w, h, idx):
    out = []
    for tok, x1, y1, x2, y2, _ in boxes:
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        out.append(f'{idx[vocab.N(tok)]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='D:/498R/lite_pack')
    ap.add_argument('--max', type=int, default=None, help='dry run: at most this many candidates per part')
    ap.add_argument('--kmin', type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    purity = val_purity()
    print('val purity by agreement:', purity, flush=True)

    classes = json.load(open(os.path.join(DATA, 'vocab.json'), encoding='utf-8'))['classes']
    idx = {vocab.N(c): i for i, c in enumerate(classes)}
    readers = [(r['name'], r['kind'], build_reader(r)) for r in CFG['readers']]
    yolo = [n for n, k, _ in readers if k == 'charyolo']
    ocr = {s: rows(os.path.join(DATA, 'ocr', f'{s}.csv')) for s in ('train', 'val', 'test')}
    held = {vocab.norm_text(r['text']) for s in ('val', 'test') for r in ocr[s]}
    held |= {vocab.norm_text(r['text']) for r in rows(os.path.join(DATA, 'e2e_test', 'e2e_test.csv'))}
    man = rows(os.path.join(DATA, 'manifest.csv'))
    rec_train = {os.path.basename(m['file']): m for m in man if m['task'] == 'rec' and m['split'] == 'train'}

    out = os.path.join(a.out, NAME)
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, 'images'))
    os.makedirs(os.path.join(out, 'labels'))
    pack, drop, count = [], [], collections.Counter()

    def read_all(imgs):
        per = [r.read(imgs) for _, _, r in readers]
        return [[(readers[j][0], per[j][i]) for j in range(len(readers))] for i in range(len(imgs))]

    def vote(rd):
        full = [(n, x.text, x.conf) for n, x in rd]
        text = metrics.fuse(full, 'MV-BM+format', full[0][0])
        k = sum(fusion.same(text, t) for _, t, _ in full)
        src = next(((n, x) for n, x in rd if n in yolo and fusion.same(text, x.text) and x.boxes), None)
        return text, k, src

    def add(part, name, img, text, k, src, origin, source, copy_from=None):
        lines = yolo_lines(src[1].boxes, img.shape[1], img.shape[0], idx)
        if not lines:
            return False
        if copy_from:
            shutil.copy(copy_from, os.path.join(out, 'images', name))
        else:
            cv2.imwrite(os.path.join(out, 'images', name), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(out, 'labels', os.path.splitext(name)[0] + '.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        pack.append({'file': name, 'part': part, 'text': text, 'readers_agreeing': k, 'boxes_from': src[0],
                     'origin': origin, 'source': source})
        count[part] += 1
        return True

    def batches(items, load, n=32):
        for s in range(0, len(items), n):
            chunk = [(it, load(it)) for it in items[s:s + n]]
            chunk = [(it, im) for it, im in chunk if im is not None and im.shape[0] >= 8 and im.shape[1] >= 8]
            if chunk:
                yield chunk, read_all([im for _, im in chunk])

    # ---- part 1: human text, no character boxes
    cand = [r for r in ocr['train'] if r['label_source'] == 'human' and r['file'] not in rec_train][:a.max]
    count['human_text_candidates'] = len(cand)
    for chunk, reads in batches(cand, lambda r: cv2.imread(os.path.join(DATA, 'ocr', 'images', 'train', r['file']))):
        for (r, im), rd in zip(chunk, reads):
            src = next(((n, x) for n, x in rd if n in yolo and fusion.same(r['text'], x.text) and x.boxes), None)
            if src:
                k = sum(fusion.same(r['text'], x.text) for _, x in rd)
                add('human_text', 'ht_' + r['file'], im, r['text'], k, src, 'ocr/images/train/' + r['file'], r['source'],
                    os.path.join(DATA, 'ocr', 'images', 'train', r['file']))
    print('human_text', count['human_text'], 'of', len(cand), f'({time.time() - t0:.0f} s)', flush=True)

    # ---- part 2: the dataset's auto labels, checked by the five readers
    text_of = {r['file']: r['text'] for r in ocr['train']}
    cand = sorted(f for f, m in rec_train.items() if m['label_source'] == 'auto')[:a.max]
    count['auto_candidates'] = len(cand)
    for chunk, reads in batches(cand, lambda f: cv2.imread(os.path.join(DATA, 'recognition', 'images', 'train', f))):
        for (f, im), rd in zip(chunk, reads):
            old = text_of.get(f, '')
            if old and sum(fusion.same(old, x.text) for _, x in rd) >= KEEP_AUTO:
                count['auto_kept'] += 1
                continue
            drop.append(f)
            text, k, src = vote(rd)
            if (metrics.is_valid_plate(text) and k >= a.kmin and src and vocab.norm_text(text) not in held
                    and add('relabel', 'rl_' + f, im, text, k, src, 'recognition/images/train/' + f,
                            rec_train[f]['source'], os.path.join(DATA, 'recognition', 'images', 'train', f))):
                continue
            count['auto_dropped'] += 1
    print('auto kept', count['auto_kept'], 'relabelled', count['relabel'], 'dropped', count['auto_dropped'],
          f'({time.time() - t0:.0f} s)', flush=True)

    # ---- part 3: plates of the detection train photos
    det = [m for m in man if m['task'] == 'det' and m['split'] == 'train' and int(m['n_objects'] or 0) > 0]
    if a.max:
        det = det[::max(1, len(det) // a.max)][:a.max]
    seen = collections.Counter()
    plates = []
    for m in det:
        lab = os.path.join(DATA, m['file'].replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt')
        if os.path.exists(lab):
            for i, line in enumerate(open(lab, encoding='utf-8').read().split('\n')):
                v = line.split()
                if len(v) >= 5:
                    plates.append((m, i, tuple(map(float, v[1:5]))))
    count['det_plates'] = len(plates)
    cache = {}

    def load(p):
        m, i, (cx, cy, bw, bh) = p
        if m['file'] not in cache:
            cache.clear()
            cache[m['file']] = cv2.imread(os.path.join(DATA, m['file']))
        img = cache[m['file']]
        if img is None:
            return None
        H, W = img.shape[:2]
        c = crop(img, ((cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
        return c.copy() if c.shape[0] >= MIN_H and c.shape[1] >= MIN_W else None

    for chunk, reads in batches(plates, load):
        for ((m, i, _), im), rd in zip(chunk, reads):
            text, k, src = vote(rd)
            nt = vocab.norm_text(text)
            if not (metrics.is_valid_plate(text) and k >= a.kmin and src):
                count['det_not_agreed'] += 1
            elif nt in held:
                count['det_is_val_or_test_plate'] += 1
            elif seen[nt] >= PER_TEXT:
                count['det_over_2_per_plate'] += 1
            else:
                stem = os.path.splitext(os.path.basename(m['file']))[0]
                if add('det_new', f'dn_{stem}_{i}.jpg', im, text, k, src, m['file'], m['source']):
                    seen[nt] += 1
    print('det_new', count['det_new'], 'of', len(plates), 'plates', f'({time.time() - t0:.0f} s)', flush=True)

    with open(os.path.join(out, 'lite_pack_manifest.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(pack[0]))
        w.writeheader()
        w.writerows(pack)
    with open(os.path.join(out, 'drop_recognition_train.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted(drop)) + ('\n' if drop else ''))
    by_src = collections.Counter((p['part'], p['source']) for p in pack)
    summary = {'name': NAME, 'built': time.strftime('%Y-%m-%d %H:%M'), 'dataset_version': json.load(open(os.path.join(
        DATA, 'dataset_manifest.json'), encoding='utf-8')).get('version'), 'dry_run_max': a.max,
        'readers': [n for n, _, _ in readers], 'kmin': a.kmin, 'keep_auto_min_readers': KEEP_AUTO,
        'val_purity_by_agreement': purity, 'counts': dict(count), 'pack_images': len(pack),
        'drop_from_recognition_train': len(drop),
        'by_part_and_source': {f'{p}|{s}': n for (p, s), n in sorted(by_src.items())},
        'agreement_in_pack': dict(collections.Counter(f"{p['part']}|{p['readers_agreeing']}_of_5" for p in pack)),
        'minutes': round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    with open(os.path.join(out, 'README.md'), 'w', encoding='utf-8') as f:
        f.write(f"""# {NAME}

Extra training crops for one small plate reader, labelled by the five BanglaLPR V2 readers together.
Built {summary['built']} from the 498R BanglaLPR All-in-One v{summary['dataset_version']} **train** split only
(script: `BanglaLPR_System/tools/build_lite_pack.py`). Use it with `paper27lite_kaggle.ipynb`.

| Part | Images | What |
|---|---|---|
| `ht_*` human_text | {count['human_text']} | train crops with a human plate text; boxes from a reader that read exactly that text |
| `rl_*` relabel | {count['relabel']} | auto-labelled crops that fewer than {KEEP_AUTO} of 5 readers confirmed, re-labelled where at least {a.kmin} of 5 agree |
| `dn_*` det_new | {count['det_new']} | plates cut from detection train photos, at least {a.kmin} of 5 readers agree, not a val/test/e2e plate |

`drop_recognition_train.txt`: {len(drop)} auto-labelled recognition train crops to leave out (not confirmed by the readers).
`lite_pack_manifest.csv`: one row per image (text, readers agreeing, which reader gave the boxes, origin).
`summary.json`: all counts and the purity check on val.

These labels were made by models, not by people. Never use the pack for validation or testing. The source images
keep their own licences (several are non-commercial); see the dataset README.
""")
    z = os.path.join(a.out, NAME + '.zip')
    with zipfile.ZipFile(z, 'w', zipfile.ZIP_STORED) as zf:
        for d, _, fs in os.walk(out):
            for fn in fs:
                p = os.path.join(d, fn)
                zf.write(p, os.path.join(NAME, os.path.relpath(p, out)))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print('zip', z, round(os.path.getsize(z) / 1e6, 1), 'MB')


if __name__ == '__main__':
    main()
