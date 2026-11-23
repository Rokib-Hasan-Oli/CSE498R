"""Make a SYNTHETIC demo video from still e2e test photos (no real traffic video is in the dataset).

  python tools/make_demo_video.py [n_photos=8] [seconds_each=2.0]

Each photo is shown for a few seconds with a slow zoom towards its centre (like a car coming closer), so the tracker
and per-track voting of lpr.video can be exercised. It is NOT a measurement of real video accuracy.
Writes webapp/demo/demo_from_test_photos.mp4 and a .csv with the true plate of each segment.
"""
import csv
import os
import sys

import cv2

DATA = os.environ.get('DATASET_ROOT', 'D:/498R/dataset')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'webapp', 'demo')
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
fps, W, H = 25, 1280, 720
rows = list(csv.DictReader(open(os.path.join(DATA, 'e2e_test', 'e2e_test.csv'), encoding='utf-8')))
rows = rows[2:2 + n]                              # skip the 2 photos the web app uses for warm-up
os.makedirs(OUT, exist_ok=True)
vw = cv2.VideoWriter(os.path.join(OUT, 'demo_from_test_photos.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
with open(os.path.join(OUT, 'demo_from_test_photos.csv'), 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['segment', 'start_s', 'end_s', 'file', 'true_text'])
    for k, r in enumerate(rows):
        img = cv2.imread(os.path.join(DATA, 'e2e_test', 'images', r['file']))
        h, wd = img.shape[:2]
        frames = int(secs * fps)
        for i in range(frames):
            z = 1.0 + 0.25 * i / frames              # zoom 1.0 -> 1.25
            cw, ch = wd / z, h / z
            x0, y0 = (wd - cw) / 2, (h - ch) / 2
            c = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
            s = min(W / c.shape[1], H / c.shape[0])
            c = cv2.resize(c, (int(c.shape[1] * s), int(c.shape[0] * s)))
            canvas = cv2.copyMakeBorder(c, (H - c.shape[0]) // 2, H - c.shape[0] - (H - c.shape[0]) // 2,
                                        (W - c.shape[1]) // 2, W - c.shape[1] - (W - c.shape[1]) // 2,
                                        cv2.BORDER_CONSTANT, value=(20, 20, 20))
            vw.write(canvas)
        w.writerow([k + 1, round(k * secs, 2), round((k + 1) * secs, 2), r['file'], r['text']])
vw.release()
print('wrote', os.path.join(OUT, 'demo_from_test_photos.mp4'), n, 'segments')
