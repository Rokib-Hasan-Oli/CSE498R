"""Build the two standalone model folders, BanglaLPR_System/V1 and BanglaLPR_System/V2.

  python tools/build_packages.py

Each folder runs on its own (copy it anywhere with Python + the requirements):
  lpr/                 the system code (same code for both versions)
  models/config.json   this version's components, with the reason for each choice
  models/weights/      only the weight files this version uses
  models/vocab.json    the 101 plate tokens
  predict.py           read plates in photos / a folder / a video
  evaluate.py          re-measure this version on the 498R test sets (GPU or --cpu)
  results/             this version's measured numbers (copied from eval/results) + training facts of its weights
  requirements.txt, README.md, and the explanation PDF (made by tools/build_explain_pdf.py)
"""
import csv
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
EV = os.path.join(ROOT, 'eval', 'results')
FACTS = json.load(open(os.path.join(HERE, 'training_facts.json'), encoding='utf-8'))
WEIGHTS = {
    'v1': ['p18_v8n_576.pt', 'p11_chars_v8n.pt', 'p02_lpnr_v12n.pt', 'p18_slot.pt', 'p20_crnn_hr.pt', 'p08_cnn28.pt',
           'p10_char_cnn32.pt', 'p10_class_names.json'],
    'v2': ['p18_v8n_576.pt', 'p06_det_v8s.pt', 'p11_chars_v8n.pt', 'p27_chars_v8n.pt', 'p02_lpnr_v12n.pt',
           'p34_svtrv2_s.onnx', 'p34_svtrv2_t.onnx'],
}
READERS = {'v1': ['p11_chars_v8n', 'p02_lpnr_v12n', 'p18_slot', 'p20_crnn_hr'],
           'v2': ['p11_chars_v8n', 'p27_chars_v8n', 'p02_lpnr_v12n', 'p34_svtrv2_s', 'p34_svtrv2_t',
                  'p27_chars_v10n', 'p23_rec_fair', 'p19_crnn']}
DETECTORS = {'v1': ['p18_v8n_576', 'p02_lpd_v12s', 'p10_v8n_640', 'p03_v8n_416', 'p03_v8n_640', 'p11_v8n_320'],
             'v2': ['p18_v8n_576', 'p06_det_v8n', 'p06_det_v8s', 'p27_v10s', 'p28_v9s', 'p21_v9s', 'p35_det_v8s',
                    'p27_v10n', 'p28_v9t']}

PREDICT = '''"""Read Bangla plates with this model folder.

  python predict.py photo.jpg [more.jpg ...]     photos
  python predict.py some_folder/                 every photo in a folder
  python predict.py clip.mp4                     a video (one reading per vehicle, writes clip_out.mp4)
  options: --cpu  --out results_folder

Prints each plate (Bangla, English, status) and writes <out>/results.json and annotated photos.
"""
import argparse
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lpr.pipeline import LPRSystem, draw  # noqa: E402

VIDEO = ('.mp4', '.avi', '.mov', '.mkv')
PHOTO = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--out', default='predictions')
    a = p.parse_args()
    lpr = LPRSystem(HERE, device='cpu' if a.cpu else None)
    import numpy as np
    lpr.process(np.full((720, 1280, 3), 127, np.uint8))     # warm-up: the first GPU call is slow
    os.makedirs(a.out, exist_ok=True)
    files = []
    for x in a.inputs:
        files += [os.path.join(x, f) for f in sorted(os.listdir(x))] if os.path.isdir(x) else [x]
    out = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in VIDEO:
            from lpr.video import process_video
            dst = os.path.join(a.out, os.path.splitext(os.path.basename(f))[0] + '_out.mp4')
            r = process_video(lpr, f, dst)
            ev = [{k: v for k, v in e.items() if not k.startswith('_')} for e in r['events']]
            for e in ev:
                print(f"{os.path.basename(f)} track {e['track_id']} {e['first_s']}-{e['last_s']} s: {e['text']}  {e['text_en']}  [{e['status']}]")
            out.append({'file': f, **{k: v for k, v in r.items() if k != 'events'}, 'events': ev, 'annotated': dst})
        elif ext in PHOTO:
            img = cv2.imread(f)
            if img is None:
                print('cannot read', f); continue
            r = lpr.process(img)
            dst = os.path.join(a.out, os.path.basename(f))
            cv2.imwrite(dst, draw(img, r))
            plates = [{k: v for k, v in pl.items() if not k.startswith('_')} for pl in r['plates']]
            for pl in plates:
                print(f"{os.path.basename(f)}: {pl['text']}  {pl['text_en']}  [{pl['status']}]  {r['ms']['total']} ms")
            if not plates:
                print(f'{os.path.basename(f)}: no plate found')
            out.append({'file': f, 'ms': r['ms'], 'plates': plates, 'annotated': dst})
    json.dump(out, open(os.path.join(a.out, 'results.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('saved', os.path.join(a.out, 'results.json'))


if __name__ == '__main__':
    main()
'''

EVALUATE = '''"""Re-measure this model folder on the 498R All-in-One v1.1 test sets (the numbers in results/summary.json).

  python evaluate.py --data D:/498R/dataset [--cpu]

A. reader part on the 651 OCR test crops: plate exact, character accuracy, status levels
B. whole system on the 204 e2e test photos: plate found, plate exact, character accuracy, FPS (batch 1)
Writes results/evaluate_<gpu|cpu>.json.
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lpr import metrics, vocab  # noqa: E402
from lpr.pipeline import LPRSystem  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=os.environ.get('DATASET_ROOT', 'D:/498R/dataset'))
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    lpr = LPRSystem(HERE, device='cpu' if a.cpu else None)
    ok = lambda p_, t: vocab.norm_text(p_ or '') == vocab.norm_text(t)
    res = {'version': lpr.version, 'device': 'cpu' if a.cpu else 'gpu'}
    ocr = list(csv.DictReader(open(os.path.join(a.data, 'ocr', 'test.csv'), encoding='utf-8')))
    imgs = [cv2.imread(os.path.join(a.data, 'ocr', 'images', 'test', r['file'])) for r in ocr]
    lpr.read_crops(imgs[:8])
    reads = [x for im in imgs for x in lpr.read_crops([im])]
    lv = {}
    for r, x in zip(reads, ocr):
        s = lv.setdefault(r['status'], [0, 0]); s[0] += 1; s[1] += ok(r['text'], x['text'])
    res['ocr_test_crops'] = {**metrics.score_rows((r['text'], x['text']) for r, x in zip(reads, ocr)),
                             'status': {k: {'plates': n, 'exact_%': round(100 * c / n, 2)} for k, (n, c) in lv.items()}}
    print('test crops', res['ocr_test_crops'], flush=True)
    e2e = list(csv.DictReader(open(os.path.join(a.data, 'e2e_test', 'e2e_test.csv'), encoding='utf-8')))
    photos = [cv2.imread(os.path.join(a.data, 'e2e_test', 'images', r['file'])) for r in e2e]
    for im in photos[:5]:
        lpr.process(im)
    preds, ms, found = [], [], 0
    for im, r in zip(photos, e2e):
        t0 = time.perf_counter(); o = lpr.process(im); ms.append(1000 * (time.perf_counter() - t0))
        found += bool(o['plates'])
        best = next((q for q in o['plates'] if q['valid']), o['plates'][0] if o['plates'] else None)
        preds.append((best['text'] if best else '', r['text']))
    res['e2e_photos'] = {**metrics.score_rows(preds), 'plate_found_%': round(100 * found / len(e2e), 2),
                         'ms_per_photo_mean': round(float(np.mean(ms)), 1), 'fps_mean': round(1000 / float(np.mean(ms)), 1)}
    print('whole photos', res['e2e_photos'], flush=True)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, 'results', f"evaluate_{res['device']}.json"), 'w', encoding='utf-8'),
              indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
'''


def build(v):
    dst = os.path.join(ROOT, v.upper())
    if os.path.isdir(os.path.join(dst, 'lpr')):
        shutil.rmtree(os.path.join(dst, 'lpr'))
    shutil.copytree(os.path.join(ROOT, 'lpr'), os.path.join(dst, 'lpr'), ignore=shutil.ignore_patterns('__pycache__'))
    os.makedirs(os.path.join(dst, 'models', 'weights'), exist_ok=True)
    for w in WEIGHTS[v]:
        s, d = os.path.join(ROOT, 'models', 'weights', w), os.path.join(dst, 'models', 'weights', w)
        if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(s):
            shutil.copy2(s, d)
    for f in os.listdir(os.path.join(dst, 'models', 'weights')):
        if f not in WEIGHTS[v]:
            os.remove(os.path.join(dst, 'models', 'weights', f))
    cfg = json.load(open(os.path.join(ROOT, 'models', v, 'config.json'), encoding='utf-8'))
    cfg['version'] = v
    json.dump(cfg, open(os.path.join(dst, 'models', 'config.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    shutil.copy2(os.path.join(ROOT, 'models', 'vocab.json'), os.path.join(dst, 'models', 'vocab.json'))
    open(os.path.join(dst, 'predict.py'), 'w', encoding='utf-8').write(PREDICT)
    open(os.path.join(dst, 'evaluate.py'), 'w', encoding='utf-8').write(EVALUATE)
    shutil.copy2(os.path.join(ROOT, 'requirements.txt'), os.path.join(dst, 'requirements.txt'))
    # ---- results of this version only
    R = os.path.join(dst, 'results')
    os.makedirs(R, exist_ok=True)
    J = lambda n: json.load(open(os.path.join(EV, n), encoding='utf-8'))
    fs = J('fusion_selection.json')[v.upper()]
    summary = {
        'version': v, 'measured_on': 'RTX 5060 Ti 16 GB / Core Ultra 7 265K, 31 Oct 2026, dataset v1.1',
        'gpu': J('system_gpu.json')[v], 'cpu': J('system_cpu.json')[v],
        'v2_vs_v1': {k: x for k, x in J('system_gpu.json').items() if k.startswith('v2_vs_v1')},
        'breakdown_ocr_test': J('breakdown_ocr_test.json')[v],
        'reader_selection_on_val': {k: x for k, x in fs.items() if k != 'top8_by_val'},
        'readers': {k: x for k, x in J('readers.json').items() if k in READERS[v]},
        'detectors': {k: x for k, x in J('detectors.json').items() if k in DETECTORS[v]},
        'detector_strategies': {s: {k: x for k, x in rows.items() if k.split(' ')[0] in DETECTORS[v]}
                                for s, rows in J('detector_combo.json').items()},
    }
    if v == 'v1':
        summary['char_verifier_val'] = J('char_verifier_val.json')
        summary['tiny_plate_upscale_val'] = J('sr_check_val.json')
    json.dump(summary, open(os.path.join(R, 'summary.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    facts = {w: FACTS[w] for w in WEIGHTS[v] if w in FACTS}
    json.dump(facts, open(os.path.join(R, 'training_facts.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    for n in (f'ocr_test_preds_{v}.csv', f'e2e_preds_{v}_gpu.csv'):
        shutil.copy2(os.path.join(EV, n), os.path.join(R, n))
    g, c = summary['gpu'], summary['cpu']
    papers = {'v1': 'Cloud Hybrid (3), Plate SR (20), Embedded LPR (18), Video LPR (11), Cascaded YOLO (2), CNN + GFP-GAN (8), BLPnet (10)',
              'v2': 'YOLOv10 (27), YOLOv9 (28), Advanced DL ALPR (21), HVI-CIDNet (35), Near-Duplicates (23), YOLOv8 + EasyOCR (6), LPLCv2 (19), SVTRv2 (34), on top of V1'}[v]
    readme = f"""# {cfg['name']}

Standalone model folder. **Read `BanglaLPR_{v.upper()}_Explained.pdf` first**: it covers how it works, which papers and
trained models are used and how, the results, how to improve it, and what to do next.

Built from the paper studies: {papers}. No new training: only the team's Kaggle-trained weights in `models/weights/`
(training facts in `results/training_facts.json`), combined and chosen on the validation split, tested once.

| Measured on the team PC (dataset v1.1 test) | |
|---|---|
| Whole plate exact, 204 test photos | {g['e2e_photos']['plate_exact_%']} % |
| Character accuracy, 204 test photos | {g['e2e_photos']['char_acc_%']} % |
| Whole plate exact / char accuracy, 651 test crops | {g['ocr_test_crops']['plate_exact_%']} % / {g['ocr_test_crops']['char_acc_%']} % |
| Speed, whole photo: GPU RTX 5060 Ti / CPU only | {g['e2e_photos']['fps_mean']} / {c['e2e_photos']['fps_mean']} FPS |

```bash
pip install -r requirements.txt                 # or use D:/498R/_dataset_build/venv/Scripts/python.exe
python predict.py car.jpg                       # photo(s), a folder, or a video (.mp4); add --cpu without a GPU
python evaluate.py --data D:/498R/dataset       # re-measure the numbers above
```

**Android app (offline, on the phone): `BanglaLPR_{v.upper()}.apk` in this folder.** Copy it to an Android 7+ phone and
install it; the model runs on the phone with no internet (install guide: `../android/README.md`).

Web app + mobile app for both versions: `python ../webapp/server.py --phone --port 8498`, then open the printed
https address on the PC, or that address + `/app` on a phone on the same Wi-Fi.

| Path | What |
|---|---|
| `models/config.json` | the components of this version and why each was chosen |
| `models/weights/` | the weight files this version uses |
| `lpr/` | the system code (detector, readers, fusion, video, storage) |
| `results/summary.json` | every measured number (test crops, whole photos, CPU, conditions, readers, detectors, selection on val) |
| `results/*.csv` | the prediction for every test crop and photo |
"""
    open(os.path.join(dst, 'README.md'), 'w', encoding='utf-8').write(readme)
    print('built', dst, sum(os.path.getsize(os.path.join(dst, 'models', 'weights', f)) for f in WEIGHTS[v]) // 2 ** 20, 'MB weights')


if __name__ == '__main__':
    build('v1')
    build('v2')
