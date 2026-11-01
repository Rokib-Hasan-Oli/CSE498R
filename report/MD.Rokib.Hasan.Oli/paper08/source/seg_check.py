"""Paper #8: WHERE does the plate pipeline fail on our plates? Local check with the run's CNN (cnn28/best.pt, read
straight from `After Train/results.zip`), CPU only, dataset v1.1 at D:/498R/dataset. No GFP-GAN (no third-party
weights are loaded here), so it is compared with the run's "none" rows.

On the 467 recognition/test crops (human token boxes):
  A  colour crop of each human box -> CNN                         (what the CNN was trained on)
  B  the same box cut from the pipeline's Otsu image, 24x44 blob   (perfect boxes, but the paper's binary input)
  C  the pipeline's own contours vs the human boxes: share of tokens found (IoU >= 0.5), blobs per plate,
     and the CNN on the blobs that do match a token
On the 651 ocr/test crops:
  D  the pipeline as in the run (no GFP-GAN, fixed order), to check that this local copy gives the run's numbers
  E  the same tokens joined by the kit's BRTA decoder (vocab.tokens_to_text: by token kind) instead of reading order
-> seg_check.json.   Run: python seg_check.py   (system Python, torch CPU)
"""
import csv
import io
import json
import os
import sys
import zipfile

import cv2
import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(PAPER_DIR, 'kaggle'))
import metrics  # noqa: E402
import p08_plates as P  # noqa: E402
import vocab  # noqa: E402
from p08_train import PaperCNN  # noqa: E402
from segtools import all_contours, binary_image, blob, iou  # noqa: E402

DATA = 'D:/498R/dataset'
torch.set_num_threads(4)
dev = torch.device('cpu')
z = zipfile.ZipFile(os.path.join(PAPER_DIR, 'After Train', 'results.zip'))
r = json.loads(z.read('runs/cnn28/results.json'))
model = PaperCNN(len(r['classes']), 28).eval()
model.load_state_dict(torch.load(io.BytesIO(z.read('runs/cnn28/best.pt')), map_location=dev, weights_only=True))
names = yaml.safe_load(open(f'{DATA}/recognition/data.yaml', encoding='utf-8'))['names']
names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
token = [names[int(c[1:])] for c in r['classes']]


# ------------------------------------------------------------------ A, B, C on recognition/test
root = f'{DATA}/recognition'
A = B = Cm = n_tok = 0
found_all = found_fixed = n_blobs_all = n_blobs_fixed = matched_blob_ok = matched_blob_n = 0
per_kind = {k: [0, 0, 0] for k in ('digit', 'metro', 'letter', 'district')}     # n, A ok, B ok
plates_all_found = 0
gt_plate = {}                       # file -> (CNN on the human boxes + BRTA decoder, truth): the run's gt_boxes row
imgs = sorted(os.listdir(f'{root}/images/test'))
for f in imgs:
    lab = f'{root}/labels/test/{os.path.splitext(f)[0]}.txt'
    if not os.path.exists(lab):
        continue
    bgr = cv2.imread(f'{root}/images/test/{f}')
    H, W = bgr.shape[:2]
    gray = P.maximise_contrast(bgr)
    b = binary_image(gray)
    sx, sy = 333 / W, 335 / H
    gt = []
    for line in (l for l in open(lab).read().splitlines() if l.strip()):
        c, cx, cy, w, h = line.split()[:5]
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        x1, y1, x2, y2 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2)), min(W, int(round(cx + w / 2))), min(H, int(round(cy + h / 2)))
        if x2 - x1 >= 2 and y2 - y1 >= 2:
            bx = (int(x1 * sx), int(y1 * sy), max(1, int((x2 - x1) * sx)), max(1, int((y2 - y1) * sy)))
            gt.append((names[int(c)], (x1, y1, x2, y2), bx))
    if not gt:
        continue
    pa = P.classify(model, [bgr[q[1]:q[3], q[0]:q[2]] for _, q, _ in gt], dev, 28)
    pb = P.classify(model, [blob(b, *bx) for _, _, bx in gt], dev, 28)
    gt_plate[f] = (vocab.plate_text([[token[i], *q] for i, (_, q, _) in zip(pa, gt)])[0],
                   vocab.plate_text([[t, *q] for t, q, _ in gt])[0])
    allb = all_contours(gray)
    fixb = [q[1] for q in P.segment(gray, 'fixed')]
    n_blobs_all += len(allb)
    n_blobs_fixed += len(fixb)
    hit_plate = True
    for (t, _, bx), ia, ib in zip(gt, pa, pb):
        k = vocab.KIND[vocab.N(t)]
        n_tok += 1
        per_kind[k][0] += 1
        A += token[ia] == t
        B += token[ib] == t
        per_kind[k][1] += token[ia] == t
        per_kind[k][2] += token[ib] == t
        ious = [iou(bx, q) for q in allb]
        if ious and max(ious) >= 0.5:
            found_all += 1
            q = allb[int(np.argmax(ious))]
            matched_blob_n += 1
            matched_blob_ok += token[P.classify(model, [blob(b, *q)], dev, 28)[0]] == t
        else:
            hit_plate = False
        if fixb and max(iou(bx, q) for q in fixb) >= 0.5:
            found_fixed += 1
    plates_all_found += hit_plate
n_plates = len(imgs)
res = {'recognition_test': {
    'plates': n_plates, 'tokens': n_tok,
    'A_colour_crops_acc_%': round(100 * A / n_tok, 2),
    'B_binary_blobs_gt_boxes_acc_%': round(100 * B / n_tok, 2),
    'per_kind': {k: {'n': v[0], 'A_%': round(100 * v[1] / max(1, v[0]), 2), 'B_%': round(100 * v[2] / max(1, v[0]), 2)}
                 for k, v in per_kind.items()},
    'C_tokens_found_iou50_all15_%': round(100 * found_all / n_tok, 2),
    'C_tokens_found_iou50_fixed_%': round(100 * found_fixed / n_tok, 2),
    'C_plates_all_tokens_found_%': round(100 * plates_all_found / n_plates, 2),
    'C_blobs_per_plate_all15': round(n_blobs_all / n_plates, 2),
    'C_blobs_per_plate_fixed': round(n_blobs_fixed / n_plates, 2),
    'C_tokens_per_plate': round(n_tok / n_plates, 2),
    'C_cnn_on_matched_blobs_acc_%': round(100 * matched_blob_ok / max(1, matched_blob_n), 2),
    'C_matched_blobs': matched_blob_n}}
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)

# ------------------------------------------------------------------ D, E on ocr/test (no GFP-GAN)
rows = list(csv.DictReader(open(f'{DATA}/ocr/test.csv', encoding='utf-8')))
D, E, Ep = [], [], []
for x in rows:
    bgr = cv2.imread(f'{DATA}/ocr/images/test/{x["file"]}')
    gray = P.maximise_contrast(bgr)
    toks = [token[i] for i in P.classify(model, [q[0] for q in P.segment(gray, 'fixed')], dev, 28)]
    D.append((''.join(toks), x['text']))
    E.append((vocab.tokens_to_text(toks), x['text']))
    toks_p = [token[i] for i in P.classify(model, [q[0] for q in P.segment(gray, 'paper')], dev, 28)]
    Ep.append((vocab.tokens_to_text(toks_p), x['text']))
res['ocr_test'] = {'D_local_none_fixed': P.summary(D), 'E_brta_decoder_on_fixed_tokens': P.summary(E),
                   'E_brta_decoder_on_paper_tokens': P.summary(Ep),
                   'E_valid_layout_%': round(100 * np.mean([metrics.is_valid_plate(p) for p, _ in E]), 2)}
print(json.dumps(res['ocr_test'], ensure_ascii=False, indent=1))

# ------------------------------------------------------------------ F: perfect segmentation vs paper #2's reader, same plates
# paper #2 (char-YOLO lpnr_v12n, the same v1.1 data, Kaggle 21 Oct) read all 651 ocr/test crops; the 467 recognition/test
# crops are a subset (same file names), so both readers can be compared plate by plate (exact McNemar test).
truth = {x['file']: x['text'] for x in rows}
P2 = os.path.join(os.path.dirname(os.path.dirname(PAPER_DIR)), 'Rokib Hasna',
                  '2_Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition',
                  'After train', 'results.zip')
p2 = {x['file']: x['pred'] for x in csv.DictReader(io.StringIO(
    zipfile.ZipFile(P2).read('runs/lpnr_v12n/ocr_preds_test.csv').decode('utf-8')))}
same = sorted(set(gt_plate) & set(p2) & set(truth))
ok8 = {f: vocab.norm_text(gt_plate[f][0]) == vocab.norm_text(truth[f]) for f in same}
ok2 = {f: vocab.norm_text(p2[f]) == vocab.norm_text(truth[f]) for f in same}
b8 = sum(ok8[f] and not ok2[f] for f in same)
b2 = sum(ok2[f] and not ok8[f] for f in same)
res['vs_paper02_same_plates'] = {
    'plates': len(same),
    'truth_matches_box_labels': sum(vocab.norm_text(gt_plate[f][1]) == vocab.norm_text(truth[f]) for f in same),
    'p08_gt_boxes_exact_%': round(100 * np.mean(list(ok8.values())), 2),
    'p08_gt_boxes': P.summary([(gt_plate[f][0], truth[f]) for f in same]),
    'p02_char_yolo_exact_%': round(100 * np.mean(list(ok2.values())), 2),
    'p02_char_yolo': P.summary([(p2[f], truth[f]) for f in same]),
    'only_p08_right': b8, 'only_p02_right': b2, 'mcnemar_p': metrics.mcnemar_exact(b8, b2)}
print(json.dumps(res['vs_paper02_same_plates'], ensure_ascii=False, indent=1))
json.dump(res, open(os.path.join(HERE, 'seg_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
