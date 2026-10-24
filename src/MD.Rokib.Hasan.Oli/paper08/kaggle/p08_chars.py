"""Paper #8 on our data: cut every labelled Bangla token out of the 498R recognition crops, so the paper's
per-symbol classifiers (CNN, ResNet-50) can be trained on them.

  python p08_chars.py --out /tmp/p08_chars [--max-images 300]

recognition/{images,labels}/{train,val,test} (YOLO boxes, 101 token classes: digits, মেট্রো, series letters, district
words) -> <out>/{training_data,val_data,testing_data}/cNNN/<crop>_<k>.png, the same folder layout as the authors'
Data.zip (one folder per class), so p08_train.py --task paper reads it unchanged. Folder cNNN = class id NNN of
recognition/data.yaml; tokens.json maps it back to the Bangla token. Train uses all train crops (human + verified +
auto boxes, like the other kits' REC data); val/test are human labels only. Each box gets 4 % padding; boxes under
4 px are skipped. The split is the dataset's grouped split, so no plate appears in two splits.
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

SPLITS = {'train': 'training_data', 'val': 'val_data', 'test': 'testing_data'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--pad', type=float, default=0.04)
    p.add_argument('--max-images', type=int, default=None, help='quick test: only N crops per split')
    a = p.parse_args()
    root = Path(common.find_dataset()) / 'recognition'
    names = yaml.safe_load(open(root / 'data.yaml', encoding='utf-8'))['names']
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
    out = Path(a.out)
    counts, plates = {}, {}
    for split, sub in SPLITS.items():
        c = Counter()
        imgs = sorted((root / 'images' / split).iterdir())[:a.max_images]
        for f in imgs:
            lab = root / 'labels' / split / (f.stem + '.txt')
            if not lab.exists():
                continue
            im = Image.open(f).convert('RGB')
            W, H = im.size
            for k, line in enumerate(l for l in lab.read_text().splitlines() if l.strip()):
                cls, cx, cy, w, h = line.split()[:5]
                cls, cx, cy, w, h = int(cls), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                if w < 4 or h < 4:
                    continue
                pw, ph = a.pad * w, a.pad * h
                box = (max(0, int(cx - w / 2 - pw)), max(0, int(cy - h / 2 - ph)),
                       min(W, int(round(cx + w / 2 + pw))), min(H, int(round(cy + h / 2 + ph))))
                d = out / sub / f'c{cls:03d}'
                d.mkdir(parents=True, exist_ok=True)
                im.crop(box).save(d / f'{f.stem}_{k}.png')
                c[cls] += 1
        counts[split] = c
        plates[split] = len(imgs)
        print(f'{split}: {sum(c.values())} token crops from {len(imgs)} plate crops, {len(c)} classes', flush=True)
    tokens = {f'c{i:03d}': n for i, n in enumerate(names)}
    rare = sorted((tokens[f'c{k:03d}'], v) for k, v in counts['train'].items() if v < 10)
    info = {'tokens': tokens, 'plates': plates,
            'crops': {s: sum(c.values()) for s, c in counts.items()},
            'per_class': {tokens[f'c{k:03d}']: [counts[s].get(k, 0) for s in SPLITS] for k in sorted(set().union(*counts.values()))},
            'test_classes_without_train_crops': [tokens[f'c{k:03d}'] for k in counts['test'] if k not in counts['train']],
            'rare_train_classes(<10)': rare}
    common.save_json(info, out / 'tokens.json')
    print('classes with test crops but no train crop:', info['test_classes_without_train_crops'])
    print('saved', out / 'tokens.json')


if __name__ == '__main__':
    main()
