"""Paper #36 report: what are the 'background' boxes that only slicing adds? A seeded random sample of 48 of them
(640 model, val-best slicing = SAHI + FI with cut-off boxes dropped, conf >= 0.25, no labelled plate under the box,
no FI box at conf >= 0.25 with IoU > 0.3) cut out with context into a contact sheet, plus the full list -> fp_sample.json.
I label the sheet by eye (plate / not a plate) in fp_labels.json; analysis of that is in build_report.py.

  venv python fp_sample.py
"""
import json

import cv2
import numpy as np

import sahi_common as C
import det

KEY = 'v8s_640__SAHI_FI__BORDER'
v = next(x for x in C.VARIANTS if x['key'] == KEY)
sl = C.load_cache('v8s_640', 640)
fi = C.load_cache('v8s_640', None)
rows = []
for f, g, wh in C.test_photos():
    b = C.merge(sl[f.name], 'SAHI_FI', 'BORDER'); b = b[b[:, 0] >= 0.25]
    a = C.merge(fi[f.name], 'FI'); a = a[a[:, 0] >= 0.25]
    for x in b:
        if len(g) and det.iou_matrix(x[None, 1:5], g).max() > 0.05:
            continue
        if len(a) and det.iou_matrix(x[None, 1:5], a[:, 1:5]).max() > 0.3:
            continue
        rows.append({'file': f.name, 'score': round(float(x[0]), 3), 'box': [round(float(z), 1) for z in x[1:5]],
                     'w640': round(float((x[3] - x[1]) * 640 / max(wh)), 1), 'no_plate_photo': len(g) == 0})
rng = np.random.RandomState(36)
pick = sorted(rng.choice(len(rows), min(48, len(rows)), replace=False))
tiles = []
for k, i in enumerate(pick):
    r = rows[i]
    im = cv2.imread(str(C.ROOT / 'detection' / 'images' / 'test' / r['file']))
    x1, y1, x2, y2 = r['box']
    w, h = x2 - x1, y2 - y1
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(w * 2.2, h * 4, 60)
    X1, Y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
    crop = im[Y1:int(cy + s / 2), X1:int(cx + s / 2)].copy()
    cv2.rectangle(crop, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), (52, 104, 235), 1)
    crop = cv2.resize(crop, (200, 200))
    cv2.putText(crop, str(k), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    tiles.append(crop)
while len(tiles) % 8:
    tiles.append(np.full((200, 200, 3), 255, np.uint8))
sheet = np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)])
cv2.imwrite(str(C.HERE / '_fp_sheet.jpg'), sheet)
(C.HERE / 'fp_sample.json').write_text(json.dumps({'setting': KEY, 'n_boxes': len(rows),
                                                   'n_on_no_plate_photos': sum(r['no_plate_photo'] for r in rows),
                                                   'sample': [rows[i] for i in pick]}, indent=1), encoding='utf-8')
print(len(rows), 'slice-only background boxes;', sum(r['no_plate_photo'] for r in rows), 'on plate-free photos')
