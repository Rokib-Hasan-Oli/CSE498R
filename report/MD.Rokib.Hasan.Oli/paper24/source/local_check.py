"""Paper #24, local check of the v1.1 run's two upload models (24 Oct 2026, RTX 5060 Ti and CPU).

1. Makes the GitHub copies: best_chars_v8n.pt (the run's Ultralytics file, unchanged) and best_crnn_fp16.pt (the CRNN
   weights stored in FP16 plus its token classes and input settings: about 15 MB instead of 31 MB, under GitHub's 25 MB
   web-upload limit; loaded back as FP32).
2. Reads the 651 test and 766 val crops of dataset v1.1 with both models (kit code: rec.decode, seq.prep/ctc_decode)
   and compares every reading with the Kaggle predictions (ocr_preds_*.csv of the run).
3. Applies the format fallback (char-YOLO; CRNN only when char-YOLO's text breaks the BRTA layout) and scores it.
4. Times that reader per plate (batch 1, image already in memory) on the GPU and on the CPU.

  D:/498R/_dataset_build/venv/Scripts/python.exe local_check.py <extracted paper24_outputs.zip> <upload folder>
Writes local_check.json (and local_preds.csv) next to this file.
"""
import csv
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'kaggle'))
os.environ.setdefault('DATASET_ROOT', 'D:/498R/dataset')      # v1.1
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')

import cv2          # noqa: E402
import numpy as np  # noqa: E402
import torch        # noqa: E402

import common       # noqa: E402
import metrics      # noqa: E402
import vocab        # noqa: E402
from rec import decode         # noqa: E402
from seq import CRNN, ctc_decode, prep  # noqa: E402

OUT, UP = sys.argv[1], sys.argv[2]


def rows(split):
    root = common.find_dataset()
    with open(os.path.join(root, 'ocr', f'{split}.csv'), encoding='utf-8') as f:
        return [dict(r, path=os.path.join(root, 'ocr', 'images', split, r['file'])) for r in csv.DictReader(f)]


def kaggle(model, split):
    with open(os.path.join(OUT, 'runs', model, f'ocr_preds_{split}.csv'), encoding='utf-8') as f:
        return {r['file']: r for r in csv.DictReader(f)}


def make_upload():
    import shutil
    best = torch.load(os.path.join(OUT, 'runs/crnn/best.pt'), map_location='cpu', weights_only=True)
    full = torch.load(os.path.join(OUT, 'runs/crnn/model_full.pt'), map_location='cpu', weights_only=True)
    assert best.keys() == full['state_dict'].keys() and all(torch.equal(best[k], full['state_dict'][k]) for k in best)
    sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in best.items()}
    torch.save({'state_dict': sd16, 'kind': 'crnn', 'classes': full['classes'], 'H': full['H'], 'W': full['W'],
                'stack': full['stack'], 'stored_as': 'float16 (load with .float())',
                'source': 'paper #24 v1.1 Kaggle run, runs/crnn/best.pt (best validation loss, epoch 45 of 60)'},
               os.path.join(UP, 'best_crnn_fp16.pt'))
    shutil.copy2(os.path.join(OUT, 'runs/chars_v8n/weights/best.pt'), os.path.join(UP, 'best_chars_v8n.pt'))
    return best


def load_crnn(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = CRNN(len(ck['classes']), ck['H'])
    m.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck['state_dict'].items()})
    return m.eval().to(device), ck


@torch.no_grad()
def crnn_read(model, ck, imgs, device, bs=64):
    texts, confs = [], []
    for s in range(0, len(imgs), bs):
        x = torch.stack([prep(im, ck['H'], ck['W'], ck['stack'], False) for im in imgs[s:s + bs]]).to(device)
        toks, conf = ctc_decode(model(x).float())
        texts += [vocab.tokens_to_text([ck['classes'][k - 1] for k in t]) for t in toks]
        confs += conf
    return texts, confs


def yolo_read(yolo, imgs, device, bs=32):
    out = []
    for s in range(0, len(imgs), bs):
        # rect=False: square 320x320 letterbox for one image too, as in the kit's batched reading (Ultralytics would
        # otherwise letterbox a single image to a smaller rectangle and read a few plates differently)
        for res in yolo.predict(imgs[s:s + bs], imgsz=320, conf=0.10, rect=False, device=device, verbose=False):
            b = res.boxes
            text, _, _, mc = decode(yolo.names, b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())
            out.append((text, mc))
    return out


def fallback(y_text, y_conf, c_text, c_conf):
    if metrics.is_valid_plate(y_text):
        return y_text
    return metrics.fuse([('chars_v8n', y_text, y_conf), ('crnn', c_text, c_conf)], 'MV-BM+format', 'chars_v8n')


def main():
    from ultralytics import YOLO, __version__ as ul
    os.makedirs(UP, exist_ok=True)
    best32 = make_upload()
    dev = torch.device('cuda')
    res = {'date': '2026-10-24', 'gpu': torch.cuda.get_device_name(0), 'cpu': platform.processor(),
           'torch': torch.__version__, 'ultralytics': ul, 'cv2': cv2.__version__,
           'dataset': json.load(open(os.path.join(common.find_dataset(), 'dataset_manifest.json'),
                                     encoding='utf-8')).get('version'),
           'files': {f: os.path.getsize(os.path.join(UP, f)) for f in ('best_chars_v8n.pt', 'best_crnn_fp16.pt')}}
    crnn16, ck = load_crnn(os.path.join(UP, 'best_crnn_fp16.pt'), dev)
    crnn32 = CRNN(len(ck['classes']), ck['H'])
    crnn32.load_state_dict(best32)
    crnn32 = crnn32.eval().to(dev)
    yolo = YOLO(os.path.join(UP, 'best_chars_v8n.pt'))
    per_plate = []
    for split in ('test', 'val'):
        R = rows(split)
        imgs = [cv2.imread(r['path']) for r in R]
        assert all(im is not None for im in imgs)
        ky, kc = kaggle('chars_v8n', split), kaggle('crnn', split)
        y = yolo_read(yolo, imgs, 0)
        c16, cc16 = crnn_read(crnn16, ck, imgs, dev)
        c32, _ = crnn_read(crnn32, ck, imgs, dev)
        with torch.no_grad():               # FP16 storage vs the original FP32 weights: logit difference on 64 crops
            x = torch.stack([prep(im, ck['H'], ck['W'], ck['stack'], False) for im in imgs[:64]]).to(dev)
            dmax = float((crnn16(x) - crnn32(x)).abs().max())
        fb = [fallback(yt, yc, ct, cc) for (yt, yc), ct, cc in zip(y, c16, cc16)]
        kfb = [fallback(ky[r['file']]['pred'], float(ky[r['file']]['conf']), kc[r['file']]['pred'],
                        float(kc[r['file']]['conf'])) for r in R]
        same = lambda a, b: vocab.norm_text(a or '') == vocab.norm_text(b or '')
        right = lambda texts: sum(same(t, r['text']) for t, r in zip(texts, R))
        n = len(R)
        res[split] = {
            'plates': n,
            'chars_v8n_exact_%': round(100 * right([t for t, _ in y]) / n, 2),
            'chars_v8n_kaggle_exact_%': round(100 * right([ky[r['file']]['pred'] for r in R]) / n, 2),
            'chars_v8n_same_text_as_kaggle': sum(same(t, ky[r['file']]['pred']) for (t, _), r in zip(y, R)),
            'crnn_fp16_exact_%': round(100 * right(c16) / n, 2),
            'crnn_fp32_exact_%': round(100 * right(c32) / n, 2),
            'crnn_kaggle_exact_%': round(100 * right([kc[r['file']]['pred'] for r in R]) / n, 2),
            'crnn_fp16_same_text_as_fp32': sum(same(a, b) for a, b in zip(c16, c32)),
            'crnn_fp16_same_text_as_kaggle': sum(same(t, kc[r['file']]['pred']) for t, r in zip(c16, R)),
            'crnn_fp16_vs_fp32_max_logit_diff': round(dmax, 5),
            'fallback_exact_%': round(100 * right(fb) / n, 2),
            'fallback_kaggle_preds_exact_%': round(100 * right(kfb) / n, 2),
            'fallback_crnn_called': sum(not metrics.is_valid_plate(t) for t, _ in y),
        }
        print(split, res[split], flush=True)
        if split == 'test':
            per_plate = [{'file': r['file'], 'true': r['text'], 'chars_v8n': yt, 'crnn': ct, 'fallback': f,
                          'fallback_right': same(f, r['text'])} for r, (yt, _), ct, f in zip(R, y, c16, fb)]
            test_imgs = imgs
    with open(os.path.join(HERE, 'local_preds.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(per_plate[0]))
        w.writeheader()
        w.writerows(per_plate)

    # timing: the whole reader on one plate (image in memory): char-YOLO, layout check, CRNN only if needed
    def reader(im, device, crnn):
        yt, yc = yolo_read(yolo, [im], device, 1)[0]
        if metrics.is_valid_plate(yt):
            return yt, False
        ct, cc = crnn_read(crnn, ck, [im], device, 1)
        return fallback(yt, yc, ct[0], cc[0]), True

    def timed(device, crnn, n=200):
        for im in test_imgs[:20]:
            reader(im, device, crnn)
        ms_all, ms_y, calls = [], [], 0
        for im in test_imgs[:n]:
            if device != 'cpu':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _, called = reader(im, device, crnn)
            if device != 'cpu':
                torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - t0)
            ms_all.append(ms)
            calls += called
            if not called:
                ms_y.append(ms)
        return {'plates': n, 'ms_mean': round(float(np.mean(ms_all)), 2), 'ms_median': round(float(np.median(ms_all)), 2),
                'ms_mean_chars_only': round(float(np.mean(ms_y)), 2), 'crnn_called': calls,
                'plates_per_s': round(1000 / float(np.mean(ms_all)), 1)}
    res['timing_gpu'] = timed(0, crnn16)
    crnn_cpu, _ = load_crnn(os.path.join(UP, 'best_crnn_fp16.pt'), torch.device('cpu'))
    res['timing_cpu'] = timed('cpu', crnn_cpu)
    res['timing_cpu']['threads'] = torch.get_num_threads()
    print('timing', res['timing_gpu'], res['timing_cpu'])
    json.dump(res, open(os.path.join(HERE, 'local_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
