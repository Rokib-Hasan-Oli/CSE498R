"""Step 1 of the V1/V2 build: run every candidate reader on the OCR val (766) and test (651) crops of the
498R All-in-One v1.1 dataset and save one prediction file per reader and split.

  python eval/reader_preds.py [reader names ...]        (default: all candidates)

Writes eval/preds/<reader>_{val,test}.csv (file, true, pred, conf, valid, source) and eval/results/readers.json
(plate exact / char accuracy per reader and split, ms per plate on this PC, batch 1, GPU if present).
The same reader classes are used by the web app, so these numbers are what the app does.
"""
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from lpr import metrics  # noqa: E402
from lpr.readers import build_reader  # noqa: E402

DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
OUT = os.path.join(HERE, 'preds')
CANDIDATES = {
    # ---- V1 papers
    'p02_lpnr_v12n': {'kind': 'charyolo', 'file': 'p02_lpnr_v12n.pt', 'imgsz': 640, 'paper': 2},
    'p11_chars_v8n': {'kind': 'charyolo', 'file': 'p11_chars_v8n.pt', 'paper': 11},
    'p18_slot': {'kind': 'slot', 'file': 'p18_slot.pt', 'paper': 18},
    'p20_crnn_hr': {'kind': 'crnn', 'file': 'p20_crnn_hr.pt', 'paper': 20},
    # ---- V2 papers
    'p27_chars_v8n': {'kind': 'charyolo', 'file': 'p27_chars_v8n.pt', 'paper': 27},
    'p27_chars_v10n': {'kind': 'charyolo', 'file': 'p27_chars_v10n.pt', 'paper': 27},
    'p23_rec_fair': {'kind': 'charyolo', 'file': 'p23_rec_fair.pt', 'paper': 23},
    'p34_svtrv2_t': {'kind': 'svtrv2', 'file': 'p34_svtrv2_t.onnx', 'paper': 34},
    'p34_svtrv2_s': {'kind': 'svtrv2', 'file': 'p34_svtrv2_s.onnx', 'paper': 34},
    'p19_crnn': {'kind': 'crnn', 'file': 'p19_crnn_fp16.pt', 'paper': 19},
}


def rows(split):
    with open(os.path.join(DATA, 'ocr', f'{split}.csv'), encoding='utf-8') as f:
        return list(csv.DictReader(f))


def main():
    names = sys.argv[1:] or list(CANDIDATES)
    os.makedirs(OUT, exist_ok=True)
    resf = os.path.join(HERE, 'results', 'readers.json')
    res = json.load(open(resf, encoding='utf-8')) if os.path.exists(resf) else {}
    for name in names:
        spec = CANDIDATES[name]
        reader = build_reader(spec)
        entry = {'paper': spec['paper'], 'kind': spec['kind'], 'file': spec['file']}
        for split in ('val', 'test'):
            rs = rows(split)
            imgs = [cv2.imread(os.path.join(DATA, 'ocr', 'images', split, r['file'])) for r in rs]
            preds = []
            for s in range(0, len(imgs), 32):
                preds += reader.read(imgs[s:s + 32])
            with open(os.path.join(OUT, f'{name}_{split}.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['file', 'true', 'pred', 'conf', 'valid', 'source'])
                for r, p in zip(rs, preds):
                    w.writerow([r['file'], r['text'], p.text, round(p.conf, 4), int(p.valid), r['source']])
            entry[split] = metrics.score_rows((p.text, r['text']) for p, r in zip(preds, rs))
            entry[split]['valid_format_%'] = round(100 * np.mean([p.valid for p in preds]), 2)
        ms = []                                           # batch-1 latency, warm-up excluded
        for img in imgs[:60]:
            t0 = time.perf_counter()
            reader.read([img])
            ms.append(1000 * (time.perf_counter() - t0))
        entry['ms_per_plate_batch1'] = round(float(np.median(ms[10:])), 2)
        res[name] = entry
        print(name, 'val', entry['val']['plate_exact_%'], 'test', entry['test']['plate_exact_%'],
              'char', entry['test']['char_acc_%'], 'ms', entry['ms_per_plate_batch1'], flush=True)
        json.dump(res, open(resf, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
