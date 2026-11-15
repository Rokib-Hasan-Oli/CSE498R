"""Paper #36 report: shared pieces for local_check.py (GPU) and analysis.py (CPU).

The kit scores SAHI with the obss/sahi package (pinned 0.12.8). Here every model's raw boxes are kept per photo: the
full-image pass (FI) and every slice pass separately, so the merge rule can be changed afterwards without running the
network again. merge() rebuilds the kit's three modes exactly (checked against get_sliced_prediction in
local_check.py verify) and adds the variants used to explain the Kaggle result.
"""
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / 'kaggle'
os.environ.setdefault('DATASET_ROOT', r'D:\498R\dataset')
os.environ.setdefault('LPR_WORK', str(HERE / '_work'))
os.environ.setdefault('LPR_WORKERS', '0')
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')
sys.path.insert(0, str(KIT))

import det  # noqa: E402  (the kit's iou_matrix / gt_boxes / split_files)

ROOT = Path(os.environ['DATASET_ROOT'])
CACHE = HERE / '_cache'
MODELS = {'v8s_640': ('det_v8s_640', 640), 'v8s_sf': ('det_v8s_sf', 640), 'v8s_1280': ('det_v8s_1280', 1280)}
# (model, slice size): 640 = the paper / kit, 512 and 960 = smaller / bigger slices than the network input
SLICES = {'v8s_640': [640, 512, 960], 'v8s_sf': [640, 512, 960], 'v8s_1280': []}
OVERLAP = 0.25


def weights(run, name):
    return Path(run) / 'runs' / MODELS[name][0] / 'weights' / 'best.pt'


def variants():
    """every (model, slices, mode, merge) setting that analysis.py scores and local_check.py read gives to the CRNN."""
    v = [{'key': 'v8s_1280__FI', 'model': 'v8s_1280', 'size': None, 'mode': 'FI'}]
    for m in ('v8s_640', 'v8s_sf'):
        v.append({'key': f'{m}__FI', 'model': m, 'size': None, 'mode': 'FI'})
        for mode in ('SAHI', 'SAHI_FI'):
            for rule in RULES:
                v.append({'key': f'{m}__{mode}' + ('' if rule == 'NMS_IOU' else f'__{rule}'), 'model': m, 'size': 640,
                          'mode': mode, 'rule': rule})
        for s in (512, 960):
            v.append({'key': f'{m}__SAHI_FI__s{s}', 'model': m, 'size': s, 'mode': 'SAHI_FI', 'rule': 'NMS_IOU'})
            v.append({'key': f'{m}__SAHI_FI__BORDER__s{s}', 'model': m, 'size': s, 'mode': 'SAHI_FI', 'rule': 'BORDER'})
        for px in SMALL_PX:
            v.append({'key': f'{m}__FI+SMALL{px}', 'model': m, 'size': 640, 'mode': 'FI+SMALL', 'small_px': px})
    return v


RULES = ['NMS_IOU', 'NMS_IOS', 'GNMM_IOS', 'BORDER', 'BORDER_GNMM']
SMALL_PX = [32, 48, 64, 96]
VARIANTS = variants()


def test_photos(split='test'):
    """The kit's p36_sahi.evaluate photo list: human-labelled detection test split, GT in photo pixels."""
    import cv2
    d = ROOT / 'detection'
    imgs = sorted(p for p in (d / 'images' / split).iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    lab = lambda p: Path(str(p.parent).replace('images', 'labels', 1)) / (p.stem + '.txt')
    out = []
    for f in imgs:
        im = cv2.imread(str(f))
        h, w = im.shape[:2]
        out.append((f, det.gt_boxes(lab(f), w, h)[0], (w, h)))
    return out


def subsets():
    cond = json.load(open(ROOT / 'detection' / 'conditions.json', encoding='utf-8'))['real']
    return {k: {Path(l.strip()).name for l in open(ROOT / 'detection' / v['list'], encoding='utf-8') if l.strip()}
            for k, v in cond.items()}


def e2e_rows():
    return list(csv.DictReader(open(ROOT / 'e2e_test' / 'e2e_test.csv', encoding='utf-8')))


def cache_file(name, size, part):
    return CACHE / f'{part}_{name}_{size or "fi"}.pkl'


def load_cache(name, size, part='test'):
    with open(cache_file(name, size, part), 'rb') as fh:
        return pickle.load(fh)


# ------------------------------------------------------------------------------------------------------------ merges
def _nms_keep(arr, metric, thr):
    from sahi.postprocess.combine import batched_nms
    if len(arr) <= 1:
        return list(range(len(arr)))
    a = np.concatenate([arr[:, 1:5], arr[:, :1], np.zeros((len(arr), 1))], 1)    # sahi order: x1 y1 x2 y2 score cat
    return list(batched_nms(a, match_metric=metric, match_threshold=thr))


def _greedy_nmm(arr, metric, thr):
    """sahi GreedyNMMPostprocess on arrays: kept box = union of the boxes merged into it, score = the best one."""
    from sahi.postprocess.combine import batched_greedy_nmm
    if len(arr) <= 1:
        return arr
    a = np.concatenate([arr[:, 1:5], arr[:, :1], np.zeros((len(arr), 1))], 1)
    keep = batched_greedy_nmm(a, match_metric=metric, match_threshold=thr)
    out = []
    for k, merged in keep.items():
        b = arr[[k] + list(merged)]
        out.append([b[:, 0].max(), b[:, 1].min(), b[:, 2].min(), b[:, 3].max(), b[:, 4].max()])
    return np.array(out).reshape(-1, 5)


def on_inner_border(boxes, slices, src, wh, tol=2):
    """True for a slice box that touches an edge of its slice that is not an edge of the photo (a cut-off plate)."""
    w, h = wh
    out = np.zeros(len(boxes), bool)
    for i, (b, k) in enumerate(zip(boxes, src)):
        if k < 0:
            continue
        x0, y0, x1, y1 = slices[int(k)]
        out[i] = ((b[1] - x0 <= tol and x0 > 0) or (b[2] - y0 <= tol and y0 > 0) or
                  (x1 - b[3] <= tol and x1 < w) or (y1 - b[4] <= tol and y1 < h))
    return out


def merge(rec, mode, rule='NMS_IOU', thr=0.5, small_px=None):
    """rec = one photo's cache entry {'fi': (n,5), 'sl': (m,6) with slice index, 'slices': (k,4), 'wh'}.
    mode FI | SAHI | SAHI_FI | FI+SMALL (FI boxes plus the slice boxes narrower than small_px at the 640 px network
    scale, cut-off ones dropped, NMS IoU 0.5).
    rule NMS_IOU (paper / kit) | NMS_IOS | GNMM_IOS (sahi default merge) | BORDER (drop cut-off slice boxes, then
    NMS_IOU) | BORDER_GNMM (drop cut-off slice boxes, then GNMM_IOS). Returns (n,5) [score x1 y1 x2 y2], best first."""
    fi = rec['fi'].reshape(-1, 5)
    if mode == 'FI':
        return fi
    sl = rec['sl'].reshape(-1, 6)
    one = len(rec['slices']) <= 1            # sahi: a photo that fits in one slice gets no extra full-image pass
    if rule.startswith('BORDER') and len(sl) or mode == 'FI+SMALL' and len(sl):
        sl = sl[~on_inner_border(sl, rec['slices'], sl[:, 5], rec['wh'])]
    if mode == 'FI+SMALL':
        # the slices only add boxes the full photo is too coarse for; a plate cut by a slice edge is never added
        keep = sl[(sl[:, 3] - sl[:, 1]) * 640 / max(rec['wh']) < small_px] if len(sl) else sl
        allb = np.concatenate([fi, keep[:, :5]]) if len(keep) else fi
        k = _nms_keep(allb, 'IOU', 0.5)
        return allb[k]
    allb = sl[:, :5]
    if mode == 'SAHI_FI' and not one:
        allb = np.concatenate([allb, fi]) if len(allb) else fi
    if len(allb) == 0:
        return allb.reshape(0, 5)
    if rule in ('NMS_IOU', 'BORDER'):
        out = allb[_nms_keep(allb, 'IOU', thr)]
    elif rule == 'NMS_IOS':
        out = allb[_nms_keep(allb, 'IOS', thr)]
    elif rule in ('GNMM_IOS', 'BORDER_GNMM'):
        out = _greedy_nmm(allb, 'IOS', thr)
    else:
        raise ValueError(rule)
    return out[np.argsort(-out[:, 0], kind='stable')]


def plate_size_640(g, wh):
    """plate width / height in pixels as the 640 px network sees the whole photo (letterbox scale 640 / long side)."""
    s = 640 / max(wh)
    return (g[:, 2] - g[:, 0]) * s, (g[:, 3] - g[:, 1]) * s
