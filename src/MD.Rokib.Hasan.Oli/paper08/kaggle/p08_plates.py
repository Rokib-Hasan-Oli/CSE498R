"""Paper #8 on our data: read whole plates with the paper's pipeline and score them like the paper and like the kit.

  python p08_plates.py --cnn runs/cnn28 --name plates_cnn28 --project runs [--max 40]

Pipeline (the authors' Code/GFPGAN.ipynb + Image_Processing.ipynb + CNN.ipynb, as in the original-paper check):
  enhance -> grey -> top-hat/black-hat contrast -> 333x335 -> Otsu -> erode/dilate -> 15 largest contours
  -> order -> each blob to 28x28 -> the CNN trained on our token crops (p08_train.py) -> tokens -> text
enhance  none   : the crop as it is
         gfpgan : GFPGAN v1.3 as the authors ran it (-s 2, Real-ESRGAN background upsampler). A plate has no face, so the
                  output IS the RealESRGAN_x2plus upsampling; we run that network directly (spandrel)
order    paper  : all boxes left to right (the authors' code; it mixes the two plate lines)
         fixed  : drop too big / too small boxes, then top line left to right, then bottom line
gt_boxes        : the same CNN on the HUMAN token boxes of recognition/test (perfect segmentation), joined with the
                  kit's BRTA decoder (vocab.plate_text): the ceiling of a per-symbol classifier on our plates
On ocr/test.csv (human plate text). Scores: mean difflib SequenceMatcher similarity and the share >= 0.95 / >= 0.90
(paper p.5), plate exact / CER / char accuracy (metrics.score_rows), ms per plate for the enhancement and for
segmentation + CNN. ocr_test.rows_decoder = gfpgan + fixed order (for the team tables).
"""
import argparse
import csv
import difflib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import vocab
from p08_train import PaperCNN

ESRGAN_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'


def load_upsampler(device):
    if os.environ.get('LPR_LOCAL_TEST'):
        print('local test: bicubic x2 stands in for Real-ESRGAN x2 (third-party weights are only loaded on Kaggle)')
        return None
    from spandrel import ModelLoader
    path = Path(tempfile.gettempdir()) / 'RealESRGAN_x2plus.pth'   # not in the output folder
    if not path.exists():
        torch.hub.download_url_to_file(ESRGAN_URL, str(path))
    model = ModelLoader().load_from_file(str(path)).model.to(device).eval()
    return model.half() if device.type == 'cuda' else model


@torch.no_grad()
def upsample_x2(model, bgr, device):
    if model is None:
        return cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    h, w = bgr.shape[:2]
    img = cv2.copyMakeBorder(bgr, 0, (-h) % 4, 0, (-w) % 4, cv2.BORDER_REFLECT)
    x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(device)
    x = x.half() if device.type == 'cuda' else x
    y = model(x).float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (y[: 2 * h, : 2 * w, ::-1] * 255).round().astype(np.uint8)


def maximise_contrast(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.subtract(cv2.add(gray, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)),
                        cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k))


def segment(gray, order):
    """-> list of (blob 24x44 white-bordered, box x, y, w, h) in reading order (the authors' Image_Processing code)."""
    lp = cv2.resize(gray, (333, 335))
    _, binary = cv2.threshold(lp, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.erode(binary, (3, 3))      # the authors pass a tuple as the kernel; kept as-is
    binary = cv2.dilate(binary, (3, 3))
    binary[0:3, :] = 255
    binary[:, 0:3] = 255
    binary[332:335, :] = 255
    binary[:, 330:333] = 255
    cntrs, _ = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]]
    if order == 'fixed':
        H, W = binary.shape
        boxes = [b for b in boxes if b[2] < 0.6 * W and b[3] < 0.6 * H and b[2] * b[3] > 0.002 * W * H]
        if boxes:
            cy = sorted(b[1] + b[3] / 2 for b in boxes)
            gaps = [(cy[i + 1] - cy[i], i) for i in range(len(cy) - 1)]
            gap, i = max(gaps) if gaps else (0, 0)
            cut = (cy[i] + cy[i + 1]) / 2 if gap > 0.12 * H else H
            boxes = (sorted([b for b in boxes if b[1] + b[3] / 2 < cut], key=lambda b: b[0]) +
                     sorted([b for b in boxes if b[1] + b[3] / 2 >= cut], key=lambda b: b[0]))
    else:
        boxes = sorted(boxes, key=lambda b: b[0])
    out = []
    for x, y, w, h in boxes:
        canvas = np.full((44, 24), 255, np.uint8)   # 24x44 with a white border, as in the authors' code
        canvas[2:42, 2:22] = cv2.resize(binary[y:y + h, x:x + w], (20, 40))
        out.append((canvas, (x, y, w, h)))
    return out


@torch.no_grad()
def classify(model, imgs, device, imgsz):
    """imgs: grey (the authors' binarised blobs) or BGR crops -> class ids. Input as in training: RGB, /255."""
    if not imgs:
        return []
    x = []
    for im in imgs:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x.append(cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_AREA))
    x = torch.from_numpy(np.stack(x).astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
    return model(x).argmax(1).tolist()


def sim(p, t):
    return difflib.SequenceMatcher(None, vocab.norm_text(t), vocab.norm_text(p or '')).ratio()


def summary(rows):
    s = np.array([sim(p, t) for p, t in rows]) if rows else np.zeros(1)
    return dict(metrics.score_rows(rows), **{'similarity_%': round(float(100 * s.mean()), 2),
                                            'ge95_%': round(float(100 * (s >= 0.95).mean()), 2),
                                            'ge90_%': round(float(100 * (s >= 0.90).mean()), 2)})


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cnn', required=True, help='run folder of a p08_train.py CNN trained on our token crops')
    p.add_argument('--name', required=True)
    p.add_argument('--project', required=True)
    p.add_argument('--max', type=int, default=None)
    a = p.parse_args()
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r = json.loads((Path(a.cnn) / 'results.json').read_text())
    imgsz = r['settings']['imgsz']
    model = PaperCNN(len(r['classes']), imgsz).to(device).eval()
    model.load_state_dict(torch.load(Path(a.cnn) / 'best.pt', map_location=device))
    root = Path(common.find_dataset())
    names = yaml.safe_load(open(root / 'recognition' / 'data.yaml', encoding='utf-8'))['names']
    token = [names[int(c[1:])] for c in r['classes']]           # folder cNNN -> Bangla token
    rows = list(csv.DictReader(open(root / 'ocr' / 'test.csv', encoding='utf-8')))[:a.max]
    up = load_upsampler(device)

    res, per, ms_enh, ms_read = {}, [], [], []
    keep = {}
    for enh in ('none', 'gfpgan'):
        imgs = {}
        for x in rows:
            bgr = cv2.imread(str(root / 'ocr' / 'images' / 'test' / x['file']))
            if enh == 'gfpgan':
                sync(device); t = time.perf_counter()
                bgr = upsample_x2(up, bgr, device)
                sync(device); ms_enh.append(1000 * (time.perf_counter() - t))
                if len(ms_enh) <= 8:
                    cv2.imwrite(str(run / f'enh_{Path(x["file"]).stem}.png'), bgr)
            imgs[x['file']] = maximise_contrast(bgr)
        for order in ('paper', 'fixed'):
            key = f'{enh}_{order}'
            preds = []
            for x in rows:
                t = time.perf_counter()
                blobs = segment(imgs[x['file']], order)
                toks = [token[i] for i in classify(model, [b[0] for b in blobs], device, imgsz)]
                sync(device); ms_read.append(1000 * (time.perf_counter() - t))
                pred = ''.join(toks)            # tokens in the order the pipeline found them (the paper's string)
                preds.append({'file': x['file'], 'true': x['text'], 'pred': pred, 'source': x.get('source', '')})
                per.append({'file': x['file'], 'setting': key, 'true': x['text'], 'pred': pred,
                            'similarity': round(sim(pred, x['text']), 3)})
            res[key] = summary([(q['pred'], q['true']) for q in preds])
            keep[key] = preds
            print(key, res[key], flush=True)

    # perfect segmentation: the human token boxes of recognition/test, read by the same CNN
    rec_imgs = sorted((root / 'recognition' / 'images' / 'test').iterdir())[:a.max]
    gt_rows = []
    for f in rec_imgs:
        lab = root / 'recognition' / 'labels' / 'test' / (f.stem + '.txt')
        if not lab.exists():
            continue
        bgr = cv2.imread(str(f))
        H, W = bgr.shape[:2]
        boxes, truth = [], []
        for line in (l for l in lab.read_text().splitlines() if l.strip()):
            c, cx, cy, w, h = line.split()[:5]
            cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            b = [max(0, int(cx - w / 2)), max(0, int(cy - h / 2)), min(W, int(round(cx + w / 2))), min(H, int(round(cy + h / 2)))]
            if b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
                boxes.append(b)
                truth.append([names[int(c)]] + b)
        ids = classify(model, [bgr[b[1]:b[3], b[0]:b[2]] for b in boxes], device, imgsz)
        pred, _ = vocab.plate_text([[token[i]] + b for i, b in zip(ids, boxes)])
        gt_rows.append((pred, vocab.plate_text(truth)[0]))
    res['gt_boxes'] = dict(summary(gt_rows), crops=len(gt_rows))
    print('gt_boxes', res['gt_boxes'], flush=True)

    best = keep['gfpgan_fixed']
    out = {'run': a.name, 'kind': 'paper08-pipeline', 'cnn': str(a.cnn), 'n_classes': len(token),
           'settings': {'model': f"paper CNN {imgsz}x{imgsz} ({Path(a.cnn).name})",
                        'enhance': 'GFPGAN v1.3 -s 2 = RealESRGAN_x2plus' + (' (bicubic stand-in, local test)' if up is None else ''),
                        'segmentation': 'Otsu + 15 largest contours (authors\' code)'},
           'plates': res, 'ms_enhance': round(float(np.mean(ms_enh[3:] or ms_enh or [0])), 1),
           'ms_read': round(float(np.mean(ms_read)), 1),
           'ocr_test': {'rows_decoder': summary([(q['pred'], q['true']) for q in best]),
                        'ms_per_plate': round(float(np.mean(ms_enh[3:] or ms_enh or [0]) + np.mean(ms_read)), 1),
                        'by_condition': metrics.condition_scores(best, str(root)),
                        'note': 'rows_decoder = GFP-GAN + fixed (two-line) order'},
           'device': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', 'env': common.env_info()}
    with open(run / 'per_plate.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per[0]))
        w.writeheader(); w.writerows(per)
    common.save_json(out, run / 'results.json')
    print('DONE', a.name)


if __name__ == '__main__':
    main()
