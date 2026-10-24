"""Weights for the GitHub upload folder (web upload limit 25 MB per file).

cnn28/best.pt (14.8 MB) is copied as it is. resnet50/best.pt is 115 MB, but its backbone was FROZEN ImageNet ResNet-50
(BatchNorm in eval mode, so not even the running statistics moved). This script checks that every backbone tensor
equals torchvision's ResNet50_Weights.IMAGENET1K_V1 bit for bit, then saves only the trained part (the Dense head,
20.7 MB). README.md shows how to rebuild the full model; the full file stays in paper08_outputs.zip (Drive).

  python make_upload_weights.py <output folder>
"""
import hashlib
import io
import json
import os
import sys
import zipfile

import torch
import yaml
from torchvision import models

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1]
z = zipfile.ZipFile(os.path.join(PAPER_DIR, 'After Train', 'results.zip'))

open(os.path.join(OUT, 'best_cnn28.pt'), 'wb').write(z.read('runs/cnn28/best.pt'))
sd = torch.load(io.BytesIO(z.read('runs/resnet50/best.pt')), map_location='cpu', weights_only=True)
ref = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).state_dict()
names = list(ref)[:-2]                                    # everything but fc.weight / fc.bias
back = [k for k in sd if k.startswith('backbone.')]
assert len(back) == len(names), (len(back), len(names))
for k, rk in zip(back, names):                           # nn.Sequential(children[:-2]) keeps the order of the keys
    assert sd[k].shape == ref[rk].shape and torch.equal(sd[k], ref[rk]), k
head = {k: v for k, v in sd.items() if not k.startswith('backbone.')}
r = json.loads(z.read('runs/resnet50/results.json'))
assert r['classes'] == json.loads(z.read('runs/cnn28/results.json'))['classes']     # both models: the same class order
names = yaml.safe_load(open('D:/498R/dataset/recognition/data.yaml', encoding='utf-8'))['names']
names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
torch.save({'head_state_dict': head, 'classes': r['classes'], 'tokens': [names[int(c[1:])] for c in r['classes']],
            'imgsz': r['settings']['imgsz'],
            'imagenet_norm': r['settings']['imagenet_norm'],
            'backbone': 'torchvision resnet50, ResNet50_Weights.IMAGENET1K_V1 (frozen; checked bit-identical)',
            'source': 'paper #8 v1.1 Kaggle run 22 Oct 2026, runs/resnet50/best.pt (best-val epoch 20)'},
           os.path.join(OUT, 'best_resnet50_head.pt'))
for f in ('best_cnn28.pt', 'best_resnet50_head.pt'):
    p = os.path.join(OUT, f)
    print(f, os.path.getsize(p), hashlib.sha256(open(p, 'rb').read()).hexdigest())
print('full resnet50 best.pt sha256', hashlib.sha256(z.read('runs/resnet50/best.pt')).hexdigest(),
      '| backbone tensors checked:', len(back), '| head keys:', list(head))
