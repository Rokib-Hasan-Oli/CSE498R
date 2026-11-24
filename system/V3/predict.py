"""Read Bangla plates with the V3 model folder; every plate is saved as a plate image + a CSV row.

  python predict.py photo.jpg [more.jpg ...]     photos
  python predict.py some_folder/                 every photo in a folder
  python predict.py clip.mp4                     a video (one reading per vehicle, writes clip_out.mp4)
  options: --cpu  --out results_folder (default: predictions)

Writes into <out>/:
  plates.csv                  one row per plate; "Plate image" = the picture of that plate (path relative to the CSV)
  plate_images/<date>/...jpg  the cut-out plate pictures
  annotated/                  the photos with plate boxes (and annotated videos)
  results.json                everything, for programs
Running again with the same --out adds to the same CSV.
"""
import argparse
import json
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lpr import metrics  # noqa: E402
from lpr.pipeline import LPRSystem, draw  # noqa: E402
from lpr.records import PlateLog, write_jpg  # noqa: E402

VIDEO = ('.mp4', '.avi', '.mov', '.mkv')
PHOTO = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('inputs', nargs='+')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--out', default='predictions')
    p.add_argument('--night', action='store_true', help='night mode: the night reading rule of models/config.json (night_mode)')
    a = p.parse_args()
    lpr = LPRSystem(HERE, device='cpu' if a.cpu else None)
    import numpy as np
    lpr.process(np.full((720, 1280, 3), 127, np.uint8))     # warm-up: the first GPU call is slow
    log = PlateLog(a.out, 'v3')
    ann_dir = os.path.join(a.out, 'annotated')
    os.makedirs(ann_dir, exist_ok=True)
    no = log.next_no()

    def save(pl, kind, filename, annotated_rel):
        nonlocal no
        now = time.time()
        crop = pl.get('_crop')
        rel = log.save_plate_image(crop, no, pl.get('text_en', ''), now) if crop is not None and crop.size else None
        log.append({'id': no, 'created': now, 'version': 'v3', 'text': pl.get('text', ''), 'text_en': pl.get('text_en', ''),
                    'status': pl.get('status', ''), 'valid': pl.get('valid'), 'conf': pl.get('conf'),
                    'det_conf': pl.get('det_conf'), 'kind': kind, 'filename': filename, 'crop': rel,
                    'annotated': annotated_rel, 'watch_hit': 0, 'night': a.night})
        no += 1
        return rel

    files = []
    for x in a.inputs:
        files += [os.path.join(x, f) for f in sorted(os.listdir(x))] if os.path.isdir(x) else [x]
    out = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        base = os.path.basename(f)
        if ext in VIDEO:
            from lpr.video import process_video
            dst = os.path.join(ann_dir, os.path.splitext(base)[0] + '_out.mp4')
            r = process_video(lpr, f, dst, night=a.night)
            ev = []
            for e in r['events']:
                e['valid'] = metrics.is_valid_plate(e['text'])
                e['plate_image'] = save(e, 'video', base, 'annotated/' + os.path.basename(dst))
                ev.append({k: v for k, v in e.items() if not k.startswith('_')})
                print(f"{base} vehicle {e['track_id']} {e['first_s']}-{e['last_s']} s: {e['text']}  {e['text_en']}  "
                      f"[{e['status']}]  -> {e['plate_image']}")
            out.append({'file': f, **{k: v for k, v in r.items() if k != 'events'}, 'events': ev, 'annotated': dst})
        elif ext in PHOTO:
            img = cv2.imread(f)
            if img is None:
                print('cannot read', f)
                continue
            r = lpr.process(img, night=a.night)
            write_jpg(os.path.join(ann_dir, os.path.splitext(base)[0] + '.jpg'), draw(img, r), 90)
            ann_rel = 'annotated/' + os.path.splitext(base)[0] + '.jpg'
            plates = []
            for pl in r['plates']:
                pl['plate_image'] = save(pl, 'photo', base, ann_rel)
                plates.append({k: v for k, v in pl.items() if not k.startswith('_')})
                print(f"{base}: {pl['text']}  {pl['text_en']}  [{pl['status']}]  {r['ms']['total']} ms  -> {pl['plate_image']}")
            if not plates:
                print(f'{base}: no plate found')
            out.append({'file': f, 'ms': r['ms'], 'plates': plates, 'annotated': ann_rel})
    with open(os.path.join(a.out, 'results.json'), 'w', encoding='utf-8') as fo:
        json.dump(out, fo, indent=1, ensure_ascii=False)
    print('CSV with plate images:', log.csv_path)
    print('Plate images:', os.path.join(log.folder, 'plate_images'))


if __name__ == '__main__':
    main()
