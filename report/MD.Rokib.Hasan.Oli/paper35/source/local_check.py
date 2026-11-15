"""Paper #35 local check (RTX 5060 Ti, dataset v1.1) of the Kaggle run: the Kaggle-trained YOLOv8s plate detector and
CRNN reader, the five enhancers of p35_lowlight.py (none, gamma, clahe, hvi_gen, hvi_lolv2), scored plate by plate.

  venv python local_check.py <unpacked After Train/results.zip> [--skip-e2e] [--skip-speed]

1. Photos: every enhancer on ALL 1,153 test photos, written as JPEG exactly like the kit (cv2.imwrite), then
   - Ultralytics val on the kit's three sets (night 71, clear200, all_gated = enhance only mean grey < 60): must equal
     the Kaggle numbers;
   - one predict pass at conf 0.001: every box with its score and TP flag at IoU 0.5 (Ultralytics matching), so
     analysis.py can recompute AP50 for any subset or gate and bootstrap it, plus plates found at conf 0.25.
2. Crops: every enhancer on all 651 OCR test crops (in memory, as the kit), CRNN reading per crop.
3. End to end (new, not in the kit): the 204 e2e_test photos (60 NightLPBD night + 144 BLPD day photos with plate
   text): detector -> most confident plate (5 % padding) -> CRNN, in three ways per enhancer:
   photo  = enhance the photo, detect and crop on the enhanced photo;
   detect = enhance the photo for the detector only, crop the ORIGINAL pixels for the reader;
   crop   = detect on the original photo, enhance only the plate crop.
4. Speed: HVI-CIDNet FP32 / FP16 / autocast at 640x640, 1280x960 and a real crop size; NaN check for FP16.
5. The learned darkness density k of each weight file (the HVI collapse curve in the report).
Writes local_check.json, det_preds.jsonl, det_per_plate.csv, ocr_per_crop.csv, e2e_per_photo.csv next to this file.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / 'kaggle'
WORK = HERE / '_work'
os.environ.setdefault('LPR_WORK', str(WORK))
os.environ.setdefault('DATASET_ROOT', r'D:\498R\dataset')
os.environ.setdefault('LPR_WORKERS', '0')
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, str(KIT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
import det  # noqa: E402
import metrics  # noqa: E402
import p35_lowlight as p35  # noqa: E402
import seq  # noqa: E402

ENH = ['none', 'gamma', 'clahe', 'hvi_gen', 'hvi_lolv2']
DARK = p35.DARK


def ap_match(pred, gt, thr=0.5):
    """Ultralytics' match_predictions for one class: pairs by IoU (highest first), each box and plate used once."""
    tp = np.zeros(len(pred), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    iou = det.iou_matrix(gt, pred)                       # (gt, pred)
    m = np.array(np.nonzero(iou >= thr)).T
    if len(m):
        if len(m) > 1:
            m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
        tp[m[:, 1].astype(int)] = True
    return tp


def found_flags(pred, conf, gt, conf_thr=0.25, thr=0.5):
    """det.plate_counts rules: boxes at conf >= 0.25, by score, each takes its best unused plate at IoU >= 0.5."""
    keep = conf >= conf_thr
    p, c = pred[keep], conf[keep]
    order = np.argsort(-c)
    m = det.iou_matrix(p[order], gt)
    used = {}
    for i in range(len(order)):
        j = int(np.argmax(m[i])) if m.shape[1] else -1
        if j >= 0 and m[i, j] >= thr and j not in used:
            used[j] = float(m[i, j])
    best = m.max(0) if m.shape[0] else np.zeros(len(gt))
    return [int(j in used) for j in range(len(gt))], best, int(keep.sum())


def val_list(model, files, tag):
    """Ultralytics val on an explicit list of image paths (labels next to them in labels/test)."""
    d = WORK / 'lists'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{tag}.txt').write_text('\n'.join(str(f) for f in files) + '\n', encoding='utf-8')
    y = d / f'{tag}.yaml'
    y.write_text(f'path: {d.as_posix()}\ntrain: {tag}.txt\nval: {tag}.txt\ntest: {tag}.txt\nnames:\n  0: plate\n',
                 encoding='utf-8')
    v = model.val(data=str(y), split='test', imgsz=640, batch=16, device='0', workers=0, plots=False, verbose=False,
                  project=str(WORK / '_val'), name='v', exist_ok=True)
    return {'P': round(float(v.box.mp), 4), 'R': round(float(v.box.mr), 4), 'mAP50': round(float(v.box.map50), 4),
            'mAP50-95': round(float(v.box.map), 4)}


def photos(a, plate, dev, out):
    data = common.data_yaml('detection', 'human')
    imgs, lab = det.split_files(data, 'test')
    root = Path(common.find_dataset())
    byname = {f.name: f for f in imgs}
    lst = lambda k: [byname[Path(l.strip()).name] for l in open(root / 'detection' / f'cond_test_{k}.txt', encoding='utf-8')
                     if l.strip() and Path(l.strip()).name in byname]
    sets = {'night': [f.name for f in lst('night')], 'clear200': [f.name for f in lst('clear')[:200]]}
    out['sets'] = {k: len(v) for k, v in sets.items()}
    grey0 = {}
    pred_rows, plate_rows = [], []
    out['photo_val'], out['photo_ms'] = {}, {}
    for name in ENH:
        enh = p35.Enhancer(name, dev)
        fold = WORK / 'enh' / name
        (fold / 'images' / 'test').mkdir(parents=True, exist_ok=True)
        (fold / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
        ms, grey1 = [], {}
        for f in imgs:
            img = cv2.imread(str(f))
            grey0.setdefault(f.name, p35.grey(img))
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            img2 = enh(img)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - t0))
            grey1[f.name] = p35.grey(img2)
            cv2.imwrite(str(fold / 'images' / 'test' / f.name), img2)
            if lab(f).exists():
                shutil.copy(lab(f), fold / 'labels' / 'test' / lab(f).name)
        out['photo_ms'][name] = {'median': round(float(np.median(ms[5:])), 2), 'mean': round(float(np.mean(ms[5:])), 2)}
        files = sorted((fold / 'images' / 'test').glob('*.jpg'))
        for s in range(0, len(files), 32):
            for r in plate.predict([str(p) for p in files[s:s + 32]], imgsz=640, conf=0.001, device='0', verbose=False):
                h, w = r.orig_shape
                fn = Path(r.path).name
                g, _ = det.gt_boxes(fold / 'labels' / 'test' / (Path(fn).stem + '.txt'), w, h)
                p = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                tp = ap_match(p, g)
                fl, best, n25 = found_flags(p, cf, g)
                pred_rows.append({'enh': name, 'file': fn, 'n_gt': len(g), 'conf': [round(float(x), 5) for x in cf],
                                  'tp': [int(x) for x in tp], 'boxes25': n25})
                for j, gb in enumerate(g):
                    plate_rows.append({'enh': name, 'file': fn, 'plate': j, 'source': fn.split('_')[0],
                                       'area': round(float((gb[2] - gb[0]) * (gb[3] - gb[1])), 1), 'found': fl[j],
                                       'best_iou': round(float(best[j]), 4), 'grey_before': round(grey0[fn], 2),
                                       'grey_after': round(grey1[fn], 2)})
        none_dir = WORK / 'enh' / 'none' / 'images' / 'test'
        mine = fold / 'images' / 'test'
        gated = [(mine if grey0[f.name] < DARK else none_dir) / f.name for f in imgs]
        out['photo_val'][name] = {'night': val_list(plate, [mine / n for n in sets['night']], f'{name}_night'),
                                  'clear200': val_list(plate, [mine / n for n in sets['clear200']], f'{name}_clear200'),
                                  'all_gated': val_list(plate, gated, f'{name}_gated'),
                                  'all_enhanced': val_list(plate, list(files), f'{name}_all')}
        print(name, {k: v['mAP50'] for k, v in out['photo_val'][name].items()}, out['photo_ms'][name], flush=True)
        del enh
        torch.cuda.empty_cache()
    out['grey_test_photos'] = {k: round(v, 2) for k, v in grey0.items()}
    out['night_files'] = sets['night']
    out['clear200_files'] = sets['clear200']
    with open(HERE / 'det_preds.jsonl', 'w', encoding='utf-8') as fh:
        for r in pred_rows:
            fh.write(json.dumps(r) + '\n')
    with open(HERE / 'det_per_plate.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, list(plate_rows[0]))
        w.writeheader()
        w.writerows(plate_rows)


def crops(a, ocr, c, dev, out):
    root = Path(common.find_dataset())
    rows = list(csv.DictReader(open(root / 'ocr' / 'test.csv', encoding='utf-8')))
    tags = {r['file']: r for r in csv.DictReader(open(root / 'ocr' / 'test_conditions.csv', encoding='utf-8'))}
    ims = [cv2.imread(str(root / 'ocr' / 'images' / 'test' / r['file'])) for r in rows]
    keep = [(r, im) for r, im in zip(rows, ims) if im is not None]
    res = []
    out['crop_ms'] = {}
    for name in ENH:
        enh = p35.Enhancer(name, dev)
        e, ms = [], []
        for _, im in keep:
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            e.append(enh(im))
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - t0))
        preds = p35.read(ocr, c, e, dev)
        out['crop_ms'][name] = round(float(np.median(ms[5:])), 2)
        for (r, im), im2, pr in zip(keep, e, preds):
            res.append({'enh': name, 'file': r['file'], 'source': r['source'], 'night': tags.get(r['file'], {}).get('night', '0'),
                        'h': im.shape[0], 'w': im.shape[1], 'grey_before': round(p35.grey(im), 2),
                        'grey_after': round(p35.grey(im2), 2), 'true': r['text'], 'pred': pr,
                        'exact': int(seq.vocab.norm_text(pr) == seq.vocab.norm_text(r['text']))})
        night = [x for x in res if x['enh'] == name and x['night'] == '1']
        print(name, 'night crops exact', round(100 * np.mean([x['exact'] for x in night]), 2), out['crop_ms'][name], flush=True)
        del enh
        torch.cuda.empty_cache()
    with open(HERE / 'ocr_per_crop.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, list(res[0]))
        w.writeheader()
        w.writerows(res)


def best_box(plate, img):
    r = plate.predict(img, imgsz=640, conf=0.25, device='0', verbose=False)[0]
    if not len(r.boxes):
        return None, 0.0
    i = int(r.boxes.conf.argmax())
    return r.boxes.xyxy[i].cpu().numpy(), float(r.boxes.conf[i])


def crop(img, b, pad=0.05):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = b
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    return img[int(max(0, y1 - ph)):int(min(h, y2 + ph)), int(max(0, x1 - pw)):int(min(w, x2 + pw))]


def read_one(ocr, c, im, dev):
    return p35.read(ocr, c, [im], dev)[0] if im is not None and im.size else ''


def e2e(a, plate, ocr, c, dev, out):
    root = Path(common.find_dataset())
    rows = list(csv.DictReader(open(root / 'e2e_test' / 'e2e_test.csv', encoding='utf-8')))
    res = []
    base = {}
    for name in ENH:
        enh = p35.Enhancer(name, dev)
        for r in rows:
            img = cv2.imread(str(root / 'e2e_test' / 'images' / r['file']))
            if img is None:
                continue
            if r['file'] not in base:
                b0, c0 = best_box(plate, img)
                base[r['file']] = (b0, c0)
            b0, c0 = base[r['file']]
            ie = enh(img) if name != 'none' else img
            be, ce = best_box(plate, ie) if name != 'none' else (b0, c0)
            v = {'photo': read_one(ocr, c, crop(ie, be) if be is not None else None, dev),
                 'detect': read_one(ocr, c, crop(img, be) if be is not None else None, dev),
                 'crop': read_one(ocr, c, enh(crop(img, b0)) if b0 is not None else None, dev)}
            t = seq.vocab.norm_text(r['text'])
            for way, pr in v.items():
                res.append({'enh': name, 'way': way, 'file': r['file'], 'source': r['source'],
                            'grey': round(p35.grey(img), 2), 'found': int((be if way != 'crop' else b0) is not None),
                            'conf': round(ce if way != 'crop' else c0, 4), 'true': r['text'], 'pred': pr,
                            'exact': int(seq.vocab.norm_text(pr) == t)})
        night = [x for x in res if x['enh'] == name and x['source'] == 'nightlpbd']
        print(name, 'e2e night exact', {w: round(100 * np.mean([x['exact'] for x in night if x['way'] == w]), 2)
                                        for w in ('photo', 'detect', 'crop')}, flush=True)
        del enh
        torch.cuda.empty_cache()
    with open(HERE / 'e2e_per_photo.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, list(res[0]))
        w.writeheader()
        w.writerows(res)


def speed(dev, out):
    """HVI-CIDNet (generalization weights) FP32 vs FP16 (.half) vs autocast; batch 1, median of 30 after 5 warm-ups."""
    net = p35.load_cidnet('hvi_gen', dev)
    rng = np.random.default_rng(0)
    res = {}
    for label, (h, w) in {'640x640': (640, 640), '1280x960': (960, 1280), 'crop_224x256': (224, 256),
                          'crop_64x192': (64, 192)}.items():
        x = torch.from_numpy(rng.integers(0, 70, (1, 3, h, w)).astype(np.float32) / 255).to(dev)
        r = {}
        with torch.no_grad():
            y32 = net(x).clamp(0, 1)
            for mode in ('fp32', 'fp16', 'autocast'):
                if mode == 'fp16':
                    m, xx = net.half(), x.half()
                else:
                    m, xx = net.float(), x
                f = (lambda: m(xx)) if mode != 'autocast' else (lambda: _ac(m, xx))
                for _ in range(5):
                    y = f()
                ts = []
                for _ in range(30):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    y = f()
                    torch.cuda.synchronize()
                    ts.append(1000 * (time.perf_counter() - t0))
                y = y.float()
                nan = int(torch.isnan(y).sum())
                yc = torch.nan_to_num(y, 0).clamp(0, 1)
                mse = float(((yc - y32) ** 2).mean())
                r[mode] = {'ms_median': round(float(np.median(ts)), 2), 'nan_values': nan,
                           'psnr_vs_fp32_dB': round(10 * np.log10(1 / max(mse, 1e-12)), 2)}
        net.float()
        res[label] = r
        print('speed', label, r, flush=True)
    res['peak_gpu_GB_1280x960'] = None
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        net(torch.rand(1, 3, 960, 1280, device=dev) * 0.25)
    res['peak_gpu_GB_1280x960'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    out['speed'] = res


def _ac(m, x):
    with torch.autocast('cuda', dtype=torch.float16):
        return m(x)


def ks(dev, out):
    import safetensors.torch as sf
    from huggingface_hub import hf_hub_download
    out['density_k'] = {}
    for key, (repo, rev, sw) in p35.HF.items():
        sd = sf.load_file(hf_hub_download(repo_id=repo, filename='model.safetensors', revision=rev))
        out['density_k'][key] = {'k_code': round(float(sd['trans.density_k'].item()), 4), 'switches': sw, 'hf': [repo, rev],
                                 'params_M': round(sum(v.numel() for k, v in sd.items() if k != 'trans.density_k') / 1e6, 4)}
    print('k', out['density_k'], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--skip-photos', action='store_true')
    ap.add_argument('--skip-crops', action='store_true')
    ap.add_argument('--skip-e2e', action='store_true')
    ap.add_argument('--skip-speed', action='store_true')
    a = ap.parse_args()
    run = Path(a.run)
    os.environ['HVI_DIR'] = str(run / 'hvi_cidnet')
    from ultralytics import YOLO
    dev = torch.device('cuda')
    plate = YOLO(str(run / 'runs' / 'det_v8s' / 'weights' / 'best.pt'))
    ocr, c = p35.load_ocr(str(run / 'runs' / 'crnn' / 'model_full.pt'), dev)
    f = HERE / 'local_check.json'
    out = json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
    out.update({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'hvi_commit': p35.COMMIT,
                'dataset': common.find_dataset()})
    ks(dev, out)
    if not a.skip_photos:
        photos(a, plate, dev, out)
        f.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    if not a.skip_crops:
        crops(a, ocr, c, dev, out)
        f.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    if not a.skip_e2e:
        e2e(a, plate, ocr, c, dev, out)
        f.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    if not a.skip_speed:
        speed(dev, out)
    f.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding='utf-8')
    print('DONE')


if __name__ == '__main__':
    main()
