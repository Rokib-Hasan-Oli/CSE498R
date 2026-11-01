"""Original-paper check for paper #8 (Afrin et al., ICCIT 2023): the authors' OWN character data (Data.zip), split and
settings, plus the plate-level test on the synthetic plates they cut their characters from.

  python c08_afrin.py prep --out /tmp/c08 --project runs        # Data.zip (pinned + sha256) -> /tmp/c08/Data, audit
  python c08_afrin.py evalclean --data /tmp/c08/Data --runs runs/cnn_s42 runs/resnet50_s42 ...
  python c08_afrin.py synth --out /tmp/c08/synth                 # needs Kaggle input furcifer/bangla-license-plates-synthetic
  python c08_afrin.py plates --csv /tmp/c08/synth/paper30.csv --weights runs/cnn_s42/best.pt --name plates_paper30 --project runs
  python c08_afrin.py summary --project runs --name ours_same_data

prep     : Data.zip from github.com/mhtusher131/... at commit 8dc72c7 (checksum-verified), then the audit: class counts
           vs Table I (p.3), image sizes, and val/test crops that are copies of a training crop (grey 32x32, mean
           |difference| <= 1) -> runs/audit/results.json, test_duplicates.csv, samples.png
evalclean: every trained run on all 303 test crops, on the "clean" ones (copies removed) and per symbol group
           (digits / series letters / region words), best.pt and last.pt -> <run>/eval_extra.json
synth    : plate lists from furcifer's synthetic plates (YOLO box per symbol; truth = top line, then bottom line):
           paper30 (30 plates with only the paper's 17 symbols, like p.5), eligible (all such), any200 (any symbol)
plates   : the paper's plate pipeline (p08_plates.py: GFP-GAN = Real-ESRGAN x2, Otsu, contours, CNN) on such a list,
           GFP-GAN off/on x paper order (x only) / fixed order (top line, then bottom line), difflib SequenceMatcher
summary  : mean +- std over the seeds of every variant -> runs/<name>/results.json (the table vs the paper)
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
import zipfile
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

REPO = 'mhtusher131/Bengali-License-Plate-Recognition_Unveling-Clarity-with-CNN-and-GFP-GAN'
COMMIT = '8dc72c7e2d39a7112bc86c9695ecffbcd6a7dfdd'
ZIP_SHA256 = '640ad3cf1fd0433b607325f2b8473adc2867fb500e5c8630179a3e4737bc66f1'
SPLITS = {'train': 'training_data', 'val': 'val_data', 'test': 'testing_data'}
# Table I, page 3 of the paper (train, validation, test), copied as printed, typos included.
TABLE_I = {
    'class_0': (90, 35, 22), 'class_1': (140, 35, 27), 'class_2': (130, 35, 23), 'class_3': (130, 35, 22),
    'class_4': (100, 35, 23), 'class_5': (140, 35, 24), 'class_6': (90, 35, 26), 'class_7': (120, 35, 20),
    'class_8': (110, 35, 21), 'class_9': (1, 45, 21), 'class_k': (13, 7, 7), 'class_ga': (15, 9, 9),
    'class_JHA': (30, 13, 14), 'class_LA': (18, 12, 12), 'class_DHAKA': (18, 11, 12),
    'class_NARAYANGANJ': (30, 15, 10), 'class_MATRO': (18, 12, 10),
}
GROUPS = {'digits': [f'class_{i}' for i in range(10)], 'letters': ['class_k', 'class_ga', 'class_JHA', 'class_LA'],
          'region_words': ['class_DHAKA', 'class_MATRO', 'class_NARAYANGANJ']}
NAMES = {'class_DHAKA': 'dhaka', 'class_JHA': 'jha', 'class_LA': 'la', 'class_MATRO': 'metro',
         'class_NARAYANGANJ': 'narayanganj', 'class_ga': 'ga', 'class_k': 'ka'}
PAPER_VOCAB = set('0123456789') | {'ka', 'ga', 'jha', 'la', 'dhaka', 'narayanganj'}   # metro is not on these plates


# ------------------------------------------------------------------ prep + audit
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def prep(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = out / 'Data'
    if not (data / 'training_data').exists():
        z = out / 'paper_Data.zip'
        if not z.exists() or sha256(z) != ZIP_SHA256:
            url = f'https://raw.githubusercontent.com/{REPO}/{COMMIT}/Data.zip'
            print('downloading', url, flush=True)
            subprocess.run(['curl', '-sSL', '--retry', '3', '-o', str(z), url], check=True)
        if sha256(z) != ZIP_SHA256:
            sys.exit('Data.zip checksum mismatch: the download was cut or the file changed; run the cell again')
        with zipfile.ZipFile(z) as f:
            for m in f.infolist():
                if '__MACOSX' not in m.filename and not m.filename.endswith('.DS_Store'):
                    f.extract(m, out)
    print({s: sum(1 for _ in (data / d).rglob('*.png')) for s, d in SPLITS.items()}, '->', data, flush=True)
    audit(data, Path(a.project) / 'audit')


def grey32(fs):
    from PIL import Image
    return np.stack([np.asarray(Image.open(f).convert('L').resize((32, 32), Image.BILINEAR), np.float32).ravel() for f in fs])


def nearest(A, B):
    d, j = [], []
    for i in range(0, len(A), 64):
        m = np.abs(A[i:i + 64, None, :] - B[None]).mean(-1)
        d.append(m.min(1))
        j.append(m.argmin(1))
    return np.concatenate(d), np.concatenate(j)


def audit(root, out, max_diff=1.0):
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    files = {s: sorted((root / d).rglob('*.png')) for s, d in SPLITS.items()}
    counts = {s: Counter(f.parent.name for f in fs) for s, fs in files.items()}
    rows = []
    for c in sorted(TABLE_I):
        ours = tuple(counts[s][c] for s in SPLITS)
        rows.append({'class': c, 'ours': ours, 'table_I': TABLE_I[c], 'match': ours == TABLE_I[c]})
    sizes = Counter(Image.open(f).size == (32, 32) for fs in files.values() for f in fs)
    Xtr = grey32(files['train'])
    md5_train = {hashlib.md5(f.read_bytes()).hexdigest() for f in files['train']}
    dup = {}
    for s in ('val', 'test'):
        d, j = nearest(grey32(files[s]), Xtr)
        flag = d <= max_diff
        per_class = Counter(f.parent.name for f, x in zip(files[s], flag) if x)
        dup[s] = {'copies': int(flag.sum()), 'of': len(files[s]),
                  'byte_identical': sum(hashlib.md5(f.read_bytes()).hexdigest() in md5_train for f in files[s]),
                  'per_class': {c: f'{per_class[c]}/{counts[s][c]}' for c in sorted(counts[s])}}
        print(f'{s}: {flag.sum()}/{len(files[s])} crops have a copy in train', flush=True)
        if s == 'test':
            with open(out / 'test_duplicates.csv', 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['test_file', 'nearest_train_file', 'mean_abs_diff', 'is_copy'])
                for f, dist, jj, x in zip(files[s], d, j, flag):
                    w.writerow([f.relative_to(root / SPLITS[s]).as_posix(),
                                files['train'][jj].relative_to(root / SPLITS['train']).as_posix(), f'{dist:.2f}', int(x)])
    classes = sorted(TABLE_I)
    grid = Image.new('RGB', (len(classes) * 70, 70), 'white')         # one example per class, like the paper's Fig. 2
    for i, c in enumerate(classes):
        grid.paste(Image.open(next(f for f in files['train'] if f.parent.name == c)).convert('RGB').resize((64, 64)), (i * 70 + 3, 3))
    grid.save(out / 'samples.png')
    res = {'run': 'audit', 'kind': 'data-audit', 'counts': {s: sum(c.values()) for s, c in counts.items()},
           'classes_matching_table_I': f"{sum(r['match'] for r in rows)}/{len(rows)}", 'counts_vs_table_I': rows,
           'share_32x32': f'{sizes[True]}/{sum(sizes.values())}', 'copy_rule': f'grey 32x32, mean |diff| <= {max_diff}',
           'test_copies': f"{dup['test']['copies']}/{dup['test']['of']}", 'val_copies': f"{dup['val']['copies']}/{dup['val']['of']}",
           'duplicates': dup}
    common.save_json(res, out / 'results.json')
    print('audit:', {k: res[k] for k in ('counts', 'classes_matching_table_I', 'share_32x32', 'test_copies', 'val_copies')})


# ------------------------------------------------------------------ re-test on the clean crops
def evalclean(a):
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from p08_train import TensorSet, build_model, list_paper_split, load_images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    items, classes = list_paper_split(a.data)
    test = items['test']
    dupes = Path(a.dupes or Path(a.project) / 'audit' / 'test_duplicates.csv')
    with open(dupes, newline='') as fh:
        copies = {r['test_file'] for r in csv.DictReader(fh) if r['is_copy'] == '1'}
    rel = [Path(f).relative_to(Path(a.data) / 'testing_data').as_posix() for f, _ in test]
    clean = torch.tensor([r not in copies for r in rel])
    cache = {}

    def score(y, p, mask):
        y, p = y[mask], p[mask]
        if len(y) == 0:
            return {'n': 0, 'acc': None, 'macro_f1': None}
        return {'n': int(len(y)), 'acc': float(accuracy_score(y, p)), 'macro_f1': float(f1_score(y, p, average='macro', zero_division=0))}

    for run in map(Path, a.runs):
        if not (run / 'best.pt').exists():
            print('skipped (no best.pt):', run)
            continue
        s = json.loads((run / 'results.json').read_text())['settings']
        if s['imgsz'] not in cache:
            cache[s['imgsz']] = load_images(test, s['imgsz'])
        x, y = cache[s['imgsz']]
        model = build_model(s['model'], len(classes), s['imgsz'], s['imagenet_norm'], pretrained=False).to(device)
        out = {}
        for ckpt in ('best', 'last'):
            model.load_state_dict(torch.load(run / f'{ckpt}.pt', map_location=device))
            model.eval()
            with torch.no_grad():
                pr = torch.cat([model(xb.to(device)).argmax(1).cpu()
                                for xb, _ in torch.utils.data.DataLoader(TensorSet(x, y), batch_size=128)])
            res = {'all': score(y, pr, torch.ones_like(clean)), 'clean': score(y, pr, clean)}
            for g, names in GROUPS.items():
                m = torch.tensor([classes[i] in names for i in y.tolist()])
                res[g] = score(y, pr, m)
                res[g + '_clean'] = score(y, pr, m & clean)
            out[ckpt] = res
        (run / 'eval_extra.json').write_text(json.dumps(out, indent=2))
        b = out['last']
        print(f"{run.name}: last epoch all {b['all']['acc']:.4f} | clean {b['clean']['acc']:.4f} (n={b['clean']['n']}) | "
              f"digits {b['digits']['acc']:.4f} letters {b['letters']['acc']:.4f} words {b['region_words']['acc']:.4f}", flush=True)


# ------------------------------------------------------------------ synthetic plates (the authors' plate source)
def synth(a):
    names_file = (sorted(glob.glob('/kaggle/input/**/lp.names', recursive=True)) +
                  sorted(glob.glob(os.path.join(os.environ.get('SYNTH_ROOT', '.'), '**', 'lp.names'), recursive=True)))
    if not names_file:
        print('Synthetic plate dataset not attached (Add Input -> furcifer/bangla-license-plates-synthetic): plate test skipped')
        return
    root = Path(names_file[0]).parent
    names = [l.strip() for l in Path(names_file[0]).read_text(encoding='utf-8').splitlines() if l.strip()]
    txts = [t for t in root.rglob('*.txt') if re.fullmatch(r'\d+_\d+', t.stem)]   # base plates only (no _006 / _-03 copies)

    def read_plate(txt):
        boxes = []
        for line in txt.read_text().split('\n'):
            parts = line.split()
            if len(parts) == 5:
                boxes.append((names[int(parts[0])], float(parts[1]), float(parts[2])))
        if not boxes:
            return None
        ys = sorted(b[2] for b in boxes)
        gaps = [(ys[i + 1] - ys[i], i) for i in range(len(ys) - 1)]
        gap, i = max(gaps) if gaps else (0, 0)
        cut = (ys[i] + ys[i + 1]) / 2 if gap > 0.15 else 2.0
        return [b[0] for b in sorted((b for b in boxes if b[2] < cut), key=lambda b: b[1]) +
                sorted((b for b in boxes if b[2] >= cut), key=lambda b: b[1])]

    plates = []
    for txt in sorted(txts):
        img = txt.with_suffix('.jpg')
        tokens = read_plate(txt)
        if tokens and img.exists():
            plates.append({'file': str(img.resolve()), 'truth': ' '.join(tokens), 'n_symbols': len(tokens),
                           'only_paper_symbols': int(all(t in PAPER_VOCAB for t in tokens))})
    eligible = [r for r in plates if r['only_paper_symbols']]
    sets = {'paper30': random.Random(a.seed).sample(eligible, min(30, len(eligible))), 'eligible': eligible[:a.max],
            'any200': random.Random(a.seed).sample(plates, min(200, len(plates)))}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in sets.items():
        with open(out / f'{name}.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f'base plates: {len(plates)} | only the paper\'s 17 symbols: {len(eligible)} -> {out}')


def plates(a):
    import cv2
    import torch
    import p08_plates as pp
    from p08_train import PaperCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = json.loads((Path(a.weights).parent / 'results.json').read_text())['classes']
    model = PaperCNN(len(classes), 28).to(device).eval()
    model.load_state_dict(torch.load(a.weights, map_location=device))
    rows = [r for r in csv.DictReader(open(a.csv, newline='', encoding='utf-8-sig')) if r.get('truth', '').strip()][:a.max]
    up = pp.load_upsampler(device)
    run = Path(a.project) / a.name
    norm = lambda s: ''.join(s.lower().split()).replace('-', '')
    per, summary = [], {}
    for enh in ('none', 'gfpgan'):
        imgs, ms_enh = {}, []
        for r in rows:
            bgr = cv2.imread(r['file'])
            if enh == 'gfpgan':
                pp.sync(device); t = time.perf_counter()
                bgr = pp.upsample_x2(up, bgr, device)
                pp.sync(device); ms_enh.append(1000 * (time.perf_counter() - t))
            imgs[r['file']] = pp.maximise_contrast(bgr)
        for order in ('paper', 'fixed'):
            scores, exact, ms_read = [], 0, []
            for r in rows:
                t = time.perf_counter()
                ids = pp.classify(model, [b[0] for b in pp.segment(imgs[r['file']], order)], device, 28)
                ms_read.append(1000 * (time.perf_counter() - t))
                pred = ''.join(NAMES.get(classes[i], classes[i].replace('class_', '')) for i in ids)
                s = SequenceMatcher(None, norm(r['truth']), norm(pred)).ratio()
                scores.append(s)
                exact += norm(r['truth']) == norm(pred)
                per.append({'file': Path(r['file']).name, 'setting': f'{enh}_{order}', 'truth': norm(r['truth']),
                            'pred': pred, 'score': round(s, 3)})
            sc = np.array(scores)
            summary[f'{enh}_{order}'] = {'plates': len(rows), 'similarity_%': round(float(100 * sc.mean()), 1),
                                         'ge95_%': round(float(100 * (sc >= 0.95).mean()), 1),
                                         'ge90_%': round(float(100 * (sc >= 0.90).mean()), 1),
                                         'exact_%': round(100 * exact / max(1, len(rows)), 1),
                                         'ms_enhance': round(float(np.mean(ms_enh[3:] or ms_enh or [0])), 1),
                                         'ms_read': round(float(np.mean(ms_read)), 1)}
            print(a.name, enh, order, summary[f'{enh}_{order}'], flush=True)
    with open(run / 'per_plate.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0]))
        w.writeheader()
        w.writerows(per)
    return {'run': a.name, 'kind': 'paper08-plates', 'list': a.csv, 'weights': a.weights, 'plates': summary,
            'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}


# ------------------------------------------------------------------ summary over seeds
def ms(xs, nd=2):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return f'{np.mean(xs):.{nd}f} ± {np.std(xs):.{nd}f} (n={len(xs)})' if len(xs) > 1 else f'{xs[0]:.{nd}f}'


def summary(a):
    runs = Path(a.project)
    var = {}
    for d in sorted(runs.iterdir()):
        if (d / 'results.json').exists() and (d / 'best.pt').exists():
            var.setdefault(d.name.rsplit('_s', 1)[0], []).append(d)
    out = {'run': a.name, 'kind': 'summary over seeds', 'variants': {k: [d.name for d in v] for k, v in var.items()}}
    for v, ds in var.items():
        rs = [json.loads((d / 'results.json').read_text()) for d in ds]
        ex = [json.loads((d / 'eval_extra.json').read_text()) if (d / 'eval_extra.json').exists() else {} for d in ds]
        g = lambda e, *k: (lambda x: None if x is None else 100 * x)(common_get(e, k))
        out[v] = {
            'acc_last': ms([r['chars_test']['last']['acc_%'] for r in rs]),
            'P_last': ms([r['chars_test']['last']['P_%'] for r in rs]), 'R_last': ms([r['chars_test']['last']['R_%'] for r in rs]),
            'F1_last': ms([r['chars_test']['last']['F1_%'] for r in rs]),
            'acc_best': ms([r['chars_test']['best']['acc_%'] for r in rs]),
            'val_best': ms([r['chars_val_best_%'] for r in rs]),
            'acc_clean_last': ms([g(e, 'last', 'clean', 'acc') for e in ex]),
            'n_clean': (common_get(ex[0], ('last', 'clean', 'n')) if ex and ex[0] else None),
            'digits_last': ms([g(e, 'last', 'digits', 'acc') for e in ex], 1),
            'letters_last': ms([g(e, 'last', 'letters', 'acc') for e in ex], 1),
            'letters_clean_last': ms([g(e, 'last', 'letters_clean', 'acc') for e in ex], 1),
            'words_last': ms([g(e, 'last', 'region_words', 'acc') for e in ex], 1),
            'fps': ms([common_get(r, ('speed', 'fps_bs1')) for r in rs], 0),
            'train_min': ms([r.get('train_minutes') for r in rs], 1)}
    for f in sorted(runs.glob('plates_*/results.json')):
        r = json.loads(f.read_text())
        out[f.parent.name] = r['plates']
    au = runs / 'audit' / 'results.json'
    if au.exists():
        au = json.loads(au.read_text())
        out['audit'] = {k: au[k] for k in ('test_copies', 'val_copies', 'share_32x32', 'classes_matching_table_I', 'counts')}
    out['env'] = common.env_info()
    (runs / a.name).mkdir(parents=True, exist_ok=True)
    common.save_json(out, runs / a.name / 'results.json')
    print(json.dumps({k: v for k, v in out.items() if k != 'env'}, indent=1, ensure_ascii=False))


def common_get(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prep', 'evalclean', 'synth', 'plates', 'summary'])
    p.add_argument('--out', default='/tmp/c08')
    p.add_argument('--data', default='/tmp/c08/Data')
    p.add_argument('--dupes', default='')
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--csv', default='')
    p.add_argument('--weights', default='')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max', type=int, default=None)
    p.add_argument('--name', default='ours_same_data')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    if a.mode == 'plates':
        (Path(a.project) / a.name).mkdir(parents=True, exist_ok=True)
        r = plates(a)
        r['env'] = common.env_info()
        common.save_json(r, Path(a.project) / a.name / 'results.json')
        print('DONE', a.name)
    else:
        {'prep': prep, 'evalclean': evalclean, 'synth': synth, 'summary': summary}[a.mode](a)
