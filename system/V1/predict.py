"""Read Bangla plates with this model folder.

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
