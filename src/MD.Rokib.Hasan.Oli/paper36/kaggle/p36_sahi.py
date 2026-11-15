"""Paper #36 (SAHI, ICIP 2022): slicing aided fine-tuning (SF) and slicing aided hyper inference (SAHI) for far /
small plates in our 1280 px photos (845 of 1,153 test photos are 1280 px; every detector kit reads them at 640 px).

  python p36_sahi.py slices --data det.yaml --out /kaggle/working/sf_data      (the SF training set: photos + slices)
  python p36_sahi.py eval --models v8s_640=runs/det_v8s_640/weights/best.pt@640 sf=runs/det_v8s_sf/weights/best.pt@640 \
         v8s_1280=runs/det_v8s_1280/weights/best.pt@1280 --ocr runs/crnn/model_full.pt --name sahi --project runs

slices  The paper's SF (p.2-3, Fig. 2): every training photo larger than 800 px is cut into overlapping 640 x 640
        patches (20 % overlap); a plate is kept in a patch when at least half of it is inside (clipped); a quarter of
        the plate-free patches are kept as negatives. Train list = all original photos + the patches; val / test are
        the normal (whole-photo) splits.
eval    For each model (weights@imgsz) on our human-labelled detection test photos, with the obss/sahi package
        (pinned) and the paper's merge rule (NMS, IoU 0.5, p.3):
          FI          full-image inference (the usual way)
          SAHI        640 x 640 slices with 25 % overlap (the paper's PO), no full image
          SAHI_FI     slices + full image (the paper's best setting for large objects)
          SAHI_FI_BORDER  slices + full image, but a slice box that touches (within 2 px) an edge of its slice that is
                      not an edge of the photo is dropped before the merge: the plate it cut is whole in a neighbouring
                      slice or in the full image. Not in the paper; on our plates the paper's merge keeps boxes on
                      plate parts, and this was the validation-best merge of the v1.1 study (29 Oct 2026).
        (a 1280 px model gets FI only: "just use a bigger input" is the fair alternative to slicing).
        Result keys: configs.<model>__<mode>. Scores: COCO AP50 overall and for small / medium / large plates (the
        paper's metric, max 500 detections, conf 0.01; only 5 of our 1,011 test plates are COCO-small), plates found /
        correct at conf 0.25 and IoU 0.5, found % by plate size and on the small-plate and night photos, false alarms
        on plate-free photos, ms per photo, slices per photo; on e2e_test photos the most confident plate is read by
        our CRNN (plate exact).
        Per-photo files for paired tests: per_plate.csv (every test plate: its width at 640 px and found 0/1 for every
        setting), e2e_reads.csv (every e2e photo: the CRNN reading for every setting), boxes.json (every setting: the
        boxes per test photo at conf >= 0.01, to re-score AP50).
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import det
import metrics
import seq


# ----------------------------------------------------------------------------------------------------------- slices
def windows(n, size, overlap):
    if n <= size:
        return [0]
    step = int(size * (1 - overlap))
    xs = list(range(0, n - size + 1, step))
    if xs[-1] != n - size:
        xs.append(n - size)
    return xs


def make_slices(a):
    import cv2
    import yaml
    rnd = random.Random(42)
    y = yaml.safe_load(open(a.data, encoding='utf-8'))
    imgs, lab = det.split_files(a.data, 'train')
    imgs = imgs[:a.max] if a.max else imgs
    out = Path(a.out)
    (out / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    listing, n_big, n_pos, n_neg, n_drop = [str(p) for p in imgs], 0, 0, 0, 0
    for f in imgs:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) <= a.min_side:
            continue
        n_big += 1
        g, c = det.gt_boxes(lab(f), w, h)
        for y0 in windows(h, a.size, a.overlap):
            for x0 in windows(w, a.size, a.overlap):
                x1, y1 = x0 + min(a.size, w), y0 + min(a.size, h)
                rows = []
                for b, k in zip(g, c):
                    ix1, iy1, ix2, iy2 = max(b[0], x0), max(b[1], y0), min(b[2], x1), min(b[3], y1)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area = (b[2] - b[0]) * (b[3] - b[1])
                    if area <= 0 or inter <= 0:
                        continue
                    if inter / area < a.min_vis:
                        n_drop += 1
                        continue
                    sw, sh = x1 - x0, y1 - y0
                    rows.append(f'{k} {((ix1 + ix2) / 2 - x0) / sw:.6f} {((iy1 + iy2) / 2 - y0) / sh:.6f} '
                                f'{(ix2 - ix1) / sw:.6f} {(iy2 - iy1) / sh:.6f}')
                if not rows:
                    if rnd.random() > a.neg:
                        continue
                    n_neg += 1
                else:
                    n_pos += 1
                name = f'{f.stem}_s{x0}_{y0}'
                cv2.imwrite(str(out / 'images' / 'train' / f'{name}.jpg'), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
                (out / 'labels' / 'train' / f'{name}.txt').write_text('\n'.join(rows) + ('\n' if rows else ''))
                listing.append(str(out / 'images' / 'train' / f'{name}.jpg'))
    (out / 'train_sf.txt').write_text('\n'.join(listing) + '\n', encoding='utf-8')
    sf = {k: y[k] for k in ('path', 'val', 'test', 'names') if k in y}
    sf['train'] = str(out / 'train_sf.txt')
    (out / 'sf.yaml').write_text(yaml.safe_dump(sf, sort_keys=False), encoding='utf-8')
    info = {'train_photos': len(imgs), 'photos_sliced': n_big, 'slices_with_plates': n_pos, 'slices_without': n_neg,
            'plates_cut_off_dropped': n_drop, 'train_images_total': len(listing), 'slice': a.size, 'overlap': a.overlap,
            'min_side': a.min_side, 'min_visible': a.min_vis, 'negatives_kept': a.neg}
    common.save_json(info, out / 'sf_info.json')
    print(json.dumps(info, indent=1))


# ------------------------------------------------------------------------------------------------------------- eval
def load_ocr(path, dev):
    c = torch.load(path, map_location='cpu', weights_only=False)
    n = len(c['classes'])
    m = {'crnn': lambda: seq.CRNN(n, c['H']), 'slot': lambda: seq.SlotCNN(n),
         'vitstr': lambda: seq.ViTSTR(n, c['H'], c['W'])}[c['kind']]()
    m.load_state_dict(c['state_dict'])
    return m.eval().to(dev), c


@torch.no_grad()
def read(ocr, c, img, dev):
    x = seq.prep(img, c['H'], c['W'], c['stack'], False)[None].to(dev)
    out = ocr(x).float()
    toks, _ = seq.ctc_decode(out) if c['kind'] == 'crnn' else seq.fixed_decode(out)
    return seq.vocab.tokens_to_text([c['classes'][k - 1] for k in toks[0]])


def sliced_border(m, path, size, overlap, iou=0.5, tol=2):
    """SAHI + FI with cut-off slice boxes dropped, built from the same sahi calls as get_sliced_prediction (slice_image,
    perform_batch_inference, convert_original_predictions, get_prediction for the full image), merged with NMS IoU."""
    from sahi.postprocess.combine import nms
    from sahi.predict import get_prediction
    from sahi.slicing import slice_image
    from sahi.utils.cv import read_image_as_pil
    arr = np.asarray(read_image_as_pil(str(path)))
    h, w = arr.shape[:2]
    sr = slice_image(image=arr, slice_height=size, slice_width=size, overlap_height_ratio=overlap,
                     overlap_width_ratio=overlap, auto_slice_resolution=True)
    rows = []
    for img, (sx, sy) in zip(sr.images, sr.starting_pixels):
        x1s, y1s, x2s, y2s = sx, sy, sx + img.shape[1], sy + img.shape[0]
        m.perform_batch_inference([np.ascontiguousarray(img)])
        m.convert_original_predictions(shift_amount=[[sx, sy]], full_shape=[[h, w]])
        for o in m.object_prediction_list_per_image[0]:
            b = o.get_shifted_object_prediction().bbox.to_xyxy()
            cut = ((b[0] - x1s <= tol and x1s > 0) or (b[1] - y1s <= tol and y1s > 0) or
                   (x2s - b[2] <= tol and x2s < w) or (y2s - b[3] <= tol and y2s < h))
            if not cut:
                rows.append([float(o.score.value), *[float(v) for v in b]])
    if len(sr) > 1:                                    # as sahi: a photo that is one slice gets no extra full-image pass
        rows += [[float(o.score.value), *[float(v) for v in o.bbox.to_xyxy()]] for o in get_prediction(arr, m).object_prediction_list]
    b = np.array(rows, dtype=float).reshape(-1, 5)
    if len(b) > 1:
        b = b[nms(np.concatenate([b[:, 1:5], b[:, :1], np.zeros((len(b), 1))], 1), match_metric='IOU', match_threshold=iou)]
    return [tuple(x) for x in b[np.argsort(-b[:, 0], kind='stable')]]


class Predictor:
    """one (model, mode) setting through the sahi package; returns [(score, x1, y1, x2, y2)] and the slice count."""
    def __init__(self, weights, imgsz, mode, a, dev):
        from sahi import AutoDetectionModel
        self.mode, self.a = mode, a
        self.m = AutoDetectionModel.from_pretrained(model_type='ultralytics', model_path=weights, confidence_threshold=0.01,
                                                    device=dev, image_size=imgsz)

    def __call__(self, path):
        from sahi.predict import get_prediction, get_sliced_prediction
        if self.mode == 'FI':
            r = get_prediction(str(path), self.m)
            n = 1
        elif self.mode == 'SAHI_FI_BORDER':
            return sliced_border(self.m, path, self.a.slice, self.a.overlap), None
        else:
            r = get_sliced_prediction(str(path), self.m, slice_height=self.a.slice, slice_width=self.a.slice,
                                      overlap_height_ratio=self.a.overlap, overlap_width_ratio=self.a.overlap,
                                      perform_standard_pred=self.mode == 'SAHI_FI', postprocess_type='NMS',
                                      postprocess_match_metric='IOU', postprocess_match_threshold=0.5, verbose=0)
            n = None
        boxes = [(float(o.score.value), *[float(v) for v in o.bbox.to_xyxy()]) for o in r.object_prediction_list]
        return boxes, n


def coco_ap50(gts, preds):
    """COCO AP50 overall / small / medium / large (pycocotools, max 500 detections as in the paper, p.4)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, anns, dets = [], [], []
    for i, (g, (w, h)) in enumerate(gts):
        images.append({'id': i, 'width': int(w), 'height': int(h)})
        for b in g:
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({'id': len(anns) + 1, 'image_id': i, 'category_id': 1, 'bbox': [float(b[0]), float(b[1]), bw, bh],
                         'area': bw * bh, 'iscrowd': 0})
        for s, x1, y1, x2, y2 in preds[i]:
            dets.append({'image_id': i, 'category_id': 1, 'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': s})
    gt = COCO()
    gt.dataset = {'images': images, 'annotations': anns, 'categories': [{'id': 1, 'name': 'plate'}]}
    gt.createIndex()
    if not dets:
        return {k: 0.0 for k in ('AP50_%', 'AP50_small_%', 'AP50_medium_%', 'AP50_large_%')}
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets)
        e = COCOeval(gt, dt, 'bbox')
        e.params.maxDets = [1, 10, 500]
        e.evaluate(); e.accumulate()
    p = e.eval['precision'][0, :, 0, :, -1]                  # IoU 0.5, all recall points, class 1, area, maxDets 500
    ap = lambda k: round(100 * float(np.mean(p[:, k][p[:, k] > -1])), 2) if (p[:, k] > -1).any() else None
    return {'AP50_%': ap(0), 'AP50_small_%': ap(1), 'AP50_medium_%': ap(2), 'AP50_large_%': ap(3)}


def match(pred, g, thr=0.5):
    if not len(pred) or not len(g):
        return set()
    m = det.iou_matrix(np.array([b[1:5] for b in pred]), g)
    used = set()
    for i in np.argsort([-b[0] for b in pred]):
        j = int(np.argmax(m[i]))
        if m[i, j] >= thr and j not in used:
            used.add(j)
    return used


def score(photos, preds, subsets, ms, slices, found=None):
    """found: an empty list to receive one 0/1 per test plate (photo order, then plate order)."""
    gts = [(g, wh) for _, g, wh in photos]
    out = coco_ap50(gts, preds)
    tp = fp = fn = neg = neg_hit = 0
    size_n, size_tp, sub_n, sub_tp = {}, {}, {}, {}
    for (f, g, _), pr in zip(photos, preds):
        pr = [b for b in pr if b[0] >= 0.25]
        used = match(pr, g)
        if found is not None:
            found += [int(j in used) for j in range(len(g))]
        tp += len(used); fp += len(pr) - len(used); fn += len(g) - len(used)
        if not len(g) and f.name in subsets.get('no_plate', set()):
            neg += 1; neg_hit += int(len(pr) > 0)
        for j, b in enumerate(g):
            a = (b[2] - b[0]) * (b[3] - b[1])
            k = 'small' if a < 32 ** 2 else 'medium' if a < 96 ** 2 else 'large'
            size_n[k] = size_n.get(k, 0) + 1; size_tp[k] = size_tp.get(k, 0) + int(j in used)
        for k, names in subsets.items():
            if f.name in names and len(g):
                sub_n[k] = sub_n.get(k, 0) + len(g); sub_tp[k] = sub_tp.get(k, 0) + sum(j in used for j in range(len(g)))
    out.update({'photos': len(photos), 'plates': tp + fn, 'found_%': round(100 * tp / max(1, tp + fn), 2),
                'correct_%': round(100 * tp / max(1, tp + fp), 2),
                'found_by_size_%': {k: round(100 * size_tp[k] / size_n[k], 2) for k in sorted(size_n)}, 'plates_by_size': size_n,
                'found_by_condition_%': {k: round(100 * sub_tp[k] / sub_n[k], 2) for k in sorted(sub_n)},
                'false_alarm_no_plate_%': round(100 * neg_hit / neg, 2) if neg else None,
                'ms_median': round(float(np.median(ms)), 1), 'ms_mean': round(float(np.mean(ms)), 1)})
    if slices:
        out['slices_per_photo'] = round(float(np.mean(slices)), 2)
    return out


def evaluate(a):
    import cv2
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    root = Path(common.find_dataset())
    data = common.data_yaml('detection', 'human')
    imgs, lab = det.split_files(data, 'test')
    imgs = imgs[:a.max] if a.max else imgs
    photos = []
    for f in imgs:
        im = cv2.imread(str(f))
        if im is None:
            continue
        h, w = im.shape[:2]
        photos.append((f, det.gt_boxes(lab(f), w, h)[0], (w, h)))
    cond = json.load(open(root / 'detection' / 'conditions.json', encoding='utf-8'))['real']
    subsets = {k: {Path(l.strip()).name for l in open(root / 'detection' / v['list'], encoding='utf-8') if l.strip()}
               for k, v in cond.items()}
    e2e = list(csv.DictReader(open(root / 'e2e_test' / 'e2e_test.csv', encoding='utf-8')))[:a.max]
    ocr = c = None
    if a.ocr and os.path.exists(a.ocr):
        ocr, c = load_ocr(a.ocr, 'cuda' if dev != 'cpu' else 'cpu')
    res = {'run': a.name, 'kind': 'sahi', 'sahi_settings': {'slice': a.slice, 'overlap': a.overlap, 'merge': 'NMS IoU 0.5',
                                                            'conf_for_AP': 0.01, 'conf_for_found': 0.25,
                                                            'SAHI_FI_BORDER': 'slice boxes within 2 px of an inner slice '
                                                            'edge dropped, then NMS IoU 0.5 with the full-image boxes'},
           'configs': {}, 'skipped': {}}
    plates_found, boxes_out, e2e_out = {}, {}, {}
    for spec in a.models:
        name, rest = spec.split('=', 1)
        w, imgsz = rest.rsplit('@', 1)
        if not os.path.exists(w):
            res['skipped'][name] = f'no weights at {w}'
            continue
        for mode in (a.modes if int(imgsz) <= a.slice else ['FI']):
            key = f'{name}__{mode}'
            pred = Predictor(w, int(imgsz), mode, a, dev)
            pred(photos[0][0])                                          # warm-up
            preds, ms, ns = [], [], []
            for f, g, (wd, ht) in photos:
                if dev != 'cpu':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                b, n = pred(f)
                if dev != 'cpu':
                    torch.cuda.synchronize()
                ms.append(1000 * (time.perf_counter() - t0)); preds.append(b)
                if mode != 'FI':
                    nx = len(windows(wd, a.slice, a.overlap)); ny = len(windows(ht, a.slice, a.overlap))
                    ns.append(nx * ny + (mode in ('SAHI_FI', 'SAHI_FI_BORDER')))
            plates_found[key] = []
            r = score(photos, preds, subsets, ms, ns, found=plates_found[key])
            boxes_out[key] = {f.name: [[round(b[0], 4)] + [round(v, 1) for v in b[1:5]] for b in pr]
                              for (f, _, _), pr in zip(photos, preds)}
            if ocr is not None:                                         # e2e photos: most confident plate -> CRNN
                rows = []
                for e in e2e:
                    p = root / 'e2e_test' / 'images' / e['file']
                    im = cv2.imread(str(p))
                    if im is None:
                        continue
                    bx = [x for x in pred(p)[0] if x[0] >= 0.25]
                    text = ''
                    if bx:
                        s, x1, y1, x2, y2 = max(bx)
                        pw, ph = (x2 - x1) * 0.05, (y2 - y1) * 0.05
                        crop = im[int(max(0, y1 - ph)):int(y2 + ph), int(max(0, x1 - pw)):int(x2 + pw)]
                        text = read(ocr, c, crop, dev) if crop.size else ''
                    rows.append((text, e['text'], bool(bx)))
                    e2e_out.setdefault(e['file'], {'gt': e['text']})[key] = text
                r['e2e'] = {**metrics.score_rows((x[0], x[1]) for x in rows),
                            'plate_detected_%': round(100 * float(np.mean([x[2] for x in rows])), 2) if rows else None}
            res['configs'][key] = r
            print(key, {k: r.get(k) for k in ('AP50_%', 'AP50_small_%', 'found_%', 'correct_%', 'ms_median')},
                  r['found_by_size_%'], (r.get('e2e') or {}).get('plate_exact_%'), flush=True)
            del pred
            torch.cuda.empty_cache() if dev != 'cpu' else None
    cf = res['configs']
    keys = list(cf)
    with open(run / 'per_plate.csv', 'w', newline='', encoding='utf-8') as fh:     # paired tests (McNemar) per plate
        wr = csv.writer(fh)
        wr.writerow(['file', 'plate', 'width_px', 'height_px', 'width_at_640'] + keys)
        k = 0
        for f, g, wh in photos:
            for j, b in enumerate(g):
                wr.writerow([f.name, j, round(float(b[2] - b[0]), 1), round(float(b[3] - b[1]), 1),
                             round(float(b[2] - b[0]) * 640 / max(wh), 1)] + [plates_found[c][k] for c in keys])
                k += 1
    if e2e_out:
        with open(run / 'e2e_reads.csv', 'w', newline='', encoding='utf-8') as fh:
            wr = csv.writer(fh)
            wr.writerow(['file', 'gt'] + keys)
            for f, d in e2e_out.items():
                wr.writerow([f, d['gt']] + [d.get(c, '') for c in keys])
    common.save_json(boxes_out, run / 'boxes.json')
    for name in {k.split('__')[0] for k in cf}:
        fi = cf.get(f'{name}__FI')
        for mode in ('SAHI', 'SAHI_FI', 'SAHI_FI_BORDER'):
            s = cf.get(f'{name}__{mode}')
            if fi and s:
                s['vs_FI'] = {'AP50_pp': round((s['AP50_%'] or 0) - (fi['AP50_%'] or 0), 2),
                              'AP50_small_pp': round((s['AP50_small_%'] or 0) - (fi['AP50_small_%'] or 0), 2),
                              'found_pp': round(s['found_%'] - fi['found_%'], 2),
                              'found_small_pp': round((s['found_by_size_%'].get('small') or 0) - (fi['found_by_size_%'].get('small') or 0), 2),
                              'ms_x': round(s['ms_median'] / max(1e-6, fi['ms_median']), 2)}
    res['env'] = common.env_info()
    common.save_json(res, run / 'results.json')
    print('DONE', a.name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['slices', 'eval'])
    p.add_argument('--data', default='')
    p.add_argument('--out', default=os.path.join(common.WORK, 'sf_data'))
    p.add_argument('--size', type=int, default=640, help='slices: patch size (SF)')
    p.add_argument('--overlap', type=float, default=None, help='slices 0.2 (default), eval 0.25 (the paper\'s PO)')
    p.add_argument('--min-side', type=int, default=800)
    p.add_argument('--min-vis', type=float, default=0.5)
    p.add_argument('--neg', type=float, default=0.25)
    p.add_argument('--models', nargs='*', default=[], help='name=weights@imgsz')
    p.add_argument('--ocr', default='')
    p.add_argument('--slice', type=int, default=640, help='eval: SAHI slice size')
    p.add_argument('--modes', nargs='*', default=['FI', 'SAHI', 'SAHI_FI', 'SAHI_FI_BORDER'],
                   help='eval: settings for models at <= slice px (larger models get FI only)')
    p.add_argument('--max', type=int, default=None)
    p.add_argument('--name', default='sahi')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    if a.overlap is None:
        a.overlap = 0.2 if a.mode == 'slices' else 0.25
    make_slices(a) if a.mode == 'slices' else evaluate(a)
