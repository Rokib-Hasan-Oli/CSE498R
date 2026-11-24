"""Night-mode predictions of the PC engine, for the Android parity test (android/desktop-test/DesktopEval ... night):
every OCR test crop read with read_crops(night=True) and every e2e photo with process(night=True).

  D:/498R/_dataset_build/venv/Scripts/python.exe V3/tools/night_preds.py
Writes eval/results/ocr_test_preds_v2_night.csv and eval/results/e2e_preds_v2_night_gpu.csv (same columns as the V2 files).
"""
import csv
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
sys.path.insert(0, V3)
from lpr.pipeline import LPRSystem  # noqa: E402

DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
OUT = os.path.join(os.path.dirname(V3), 'eval', 'results')

s = LPRSystem(V3)
rows = list(csv.DictReader(open(os.path.join(DATA, 'ocr', 'test.csv'), encoding='utf-8')))
with open(os.path.join(OUT, 'ocr_test_preds_v2_night.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['file', 'true', 'pred', 'status'])
    for r in rows:
        x = s.read_crops([cv2.imread(os.path.join(DATA, 'ocr', 'images', 'test', r['file']))], night=True)[0]
        w.writerow([r['file'], r['text'], x['text'], x['status']])
rows = list(csv.DictReader(open(os.path.join(DATA, 'e2e_test', 'e2e_test.csv'), encoding='utf-8')))
with open(os.path.join(OUT, 'e2e_preds_v2_night_gpu.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['file', 'true', 'pred', 'status', 'source'])
    for r in rows:
        res = s.process(cv2.imread(os.path.join(DATA, 'e2e_test', 'images', r['file'])), night=True)
        best = next((p for p in res['plates'] if p['valid']), res['plates'][0] if res['plates'] else None)
        w.writerow([r['file'], r['text'], best['text'] if best else '', best['status'] if best else '', r['source']])
print('written to', OUT)
