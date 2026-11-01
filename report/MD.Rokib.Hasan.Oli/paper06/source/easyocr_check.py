"""Local EasyOCR re-run on the 651 OCR test crops: is the Kaggle number repeatable, and what do cheap fixes buy?

  set PYTHONPATH=<folder with easyocr 1.7.2 and its extra deps>
  D:/498R/_dataset_build/venv/Scripts/python.exe easyocr_check.py <run dir> <model dir>

Variants (stock `bn` model, no training, same line joining as the kit's p06_easyocr.read_lines):
  raw        the crop as it is (= the Kaggle job; compared reading by reading with ocr_preds_test.csv)
  prep       the kit's preprocessing: 4x Lanczos + grey + CLAHE (Kaggle: worse than raw)
  x2         2x Lanczos upscale only, colour kept
  allow      raw + allowlist = only the characters that occur in our training plate texts (no Latin letters,
             no Latin digits, no brackets) -> EasyOCR's decoder must pick a plate character
  allow_x2   allowlist + 2x upscale
Writes easyocr_check.json and easyocr_local_preds.csv next to this file.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RUN, MODELS = sys.argv[1], sys.argv[2]
DATA = os.environ.setdefault('DATASET_ROOT', r'D:\498R\dataset')
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')
sys.path.insert(0, os.path.join(RUN, 'code'))
import easyocr  # noqa: E402
import torch  # noqa: E402

import metrics  # noqa: E402
import p06_easyocr as kit  # noqa: E402
import vocab  # noqa: E402

rows = list(csv.DictReader(open(os.path.join(DATA, 'ocr', 'test.csv'), encoding='utf-8')))
train = list(csv.DictReader(open(os.path.join(DATA, 'ocr', 'train.csv'), encoding='utf-8')))
ALLOW = ''.join(sorted({c for r in train for c in vocab.N(r['text']) if not c.isspace()}))
kaggle = {r['file']: r['pred'] for r in csv.DictReader(open(os.path.join(RUN, 'runs', 'easyocr_crops', 'ocr_preds_test.csv'),
                                                            encoding='utf-8'))}
reader = easyocr.Reader(['bn'], gpu=torch.cuda.is_available(), verbose=False, model_storage_directory=MODELS,
                        download_enabled=True)


def read(img, **kw):
    """kit.read_lines with extra readtext arguments (allowlist)."""
    items = reader.readtext(np.array(img), detail=1, paragraph=False, **kw)
    if not items:
        return ''
    boxes = [(np.mean([q[1] for q in b]), min(q[0] for q in b), max(q[1] for q in b) - min(q[1] for q in b), t, c)
             for b, t, c in items]
    boxes.sort(key=lambda z: z[0])
    tol = 0.5 * float(np.median([z[2] for z in boxes]))
    lines = [[boxes[0]]]
    for z in boxes[1:]:
        if abs(z[0] - np.mean([q[0] for q in lines[-1]])) <= tol:
            lines[-1].append(z)
        else:
            lines.append([z])
    return ' '.join(' '.join(z[3] for z in sorted(l, key=lambda q: q[1])) for l in lines)


x2 = lambda im: im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
VARIANTS = {'raw': (lambda im: im, {}), 'prep': (kit.prep, {}), 'x2': (x2, {}),
            'allow': (lambda im: im, {'allowlist': ALLOW}), 'allow_x2': (x2, {'allowlist': ALLOW})}
out = {'easyocr': easyocr.__version__, 'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
       'allowlist_chars': len(ALLOW), 'variants': {}}
preds = []
for name, (fn, kw) in VARIANTS.items():
    pp, ms = [], []
    for i, r in enumerate(rows):
        img = fn(Image.open(os.path.join(DATA, 'ocr', 'images', 'test', r['file'])).convert('RGB'))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        text = read(img, **kw)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= 5:
            ms.append(1000 * (time.perf_counter() - t0))
        pp.append({'variant': name, 'file': r['file'], 'true': r['text'], 'pred': text})
    s = metrics.score_rows((p['pred'], p['true']) for p in pp)
    s['ms_per_plate_local'] = round(float(np.mean(ms)), 1)
    s['valid_format_%'] = round(100 * float(np.mean([metrics.is_valid_plate(p['pred']) for p in pp])), 2)
    if name == 'raw':
        s['same_text_as_kaggle_%'] = round(100 * float(np.mean([p['pred'] == kaggle[p['file']] for p in pp])), 2)
        s['same_normalised_as_kaggle_%'] = round(100 * float(np.mean(
            [vocab.norm_text(p['pred']) == vocab.norm_text(kaggle[p['file']]) for p in pp])), 2)
    out['variants'][name] = s
    preds += pp
    print(name, s, flush=True)
json.dump(out, open(os.path.join(HERE, 'easyocr_check.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
with open(os.path.join(HERE, 'easyocr_local_preds.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(preds[0].keys()))
    w.writeheader(); w.writerows(preds)
print('done')
