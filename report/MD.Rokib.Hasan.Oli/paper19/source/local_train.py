"""Paper #19 extra experiments on our PC (RTX 5060 Ti), same classifier recipe as the Kaggle kit.

  D:/498R/_dataset_build/venv/Scripts/python.exe local_train.py <unpacked results.zip> <variant> <task> <loss> <seed>

variant  kit       the kit's train_cls loop exactly (Adam 1e-5, batch 64, plateau x0.75 / 5, early stop 20 on val
                   accuracy, max 80 epochs, 96 x 192, AMP): only the seed changes (seed variation of the Kaggle rows)
         balanced  + the authors' BalancedSampler (dataset/dataset_utils.py, sampler "custom" in ema_br.json): every
                   epoch each class is cut to a random sample of the smallest class's size
         text      EMA loss with the weights the paper's TEXT describes ("higher weight for classes with higher error
                   counts", p.5): w = min((err + eps) / (min(err) + eps), 1.2) instead of the code's max(err) / err
         failure   a direct failure predictor: target = "the cross-fitted CRNN half-reader read this crop wrong"
                   (task quality: suitable = read right), i.e. the question the gate really asks
Writes _local_runs/<variant>_<task>_<loss>_s<seed>/{results.json, test_scores.csv, val_scores.csv}. The decoded crops
are cached in memory (no loader workers: spawned workers hang on Windows); everything else is the kit's code (<results>/code/p19_legibility.py).
"""
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
VARIANT, TASK, LOSS, SEED = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
HERE = Path(__file__).resolve().parent
os.environ.setdefault('DATASET_ROOT', r'D:\498R\dataset')
os.environ.setdefault('LPR_WORKERS', '0')
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')
sys.path.insert(0, str(RUN / 'code'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torchvision  # noqa: E402

import common  # noqa: E402
import p19_legibility as P  # noqa: E402
import vocab  # noqa: E402

_cache = {}
_load = P.CropDS.load


def cached(self, r):
    k = (r['split'], r['file'])
    if k not in _cache:
        _cache[k] = _load(self, r)
    return _cache[k]


P.CropDS.load = cached


class TextEMA(P.EMALoss):
    """The paper's sentence instead of its Eq. 1 / code: more errors -> more weight (1.0 for the best class, cap 1.2)."""
    def forward(self, logits, y):
        with torch.no_grad():
            conf = torch.zeros(self.n, self.n, device=logits.device)
            conf.index_put_((y, logits.argmax(1)), torch.ones_like(y, dtype=torch.float), accumulate=True)
            self.ema = conf if self.ema is None else self.alpha * conf + (1 - self.alpha) * self.ema
            mis = self.ema.sum(1) - torch.diagonal(self.ema)
            w = torch.clamp((mis + 1e-6) / (mis.min() + 1e-6), max=self.cap)
        return (F.cross_entropy(logits, y, reduction='none') * w[y]).mean()


class Balanced(torch.utils.data.Sampler):
    """The authors' BalancedSampler (static=False): each epoch, every class is cut to the smallest class's size."""
    def __init__(self, ys, bs):
        self.map, self.bs = {}, bs
        for i, y in enumerate(ys):
            self.map.setdefault(y, []).append(i)

    def __iter__(self):
        mn = min(len(v) for v in self.map.values())
        idx = []
        for v in self.map.values():
            idx += random.sample(v, mn)
        random.shuffle(idx)
        for i in range(0, len(idx), self.bs):
            yield idx[i:i + self.bs]

    def __len__(self):
        mn = min(len(v) for v in self.map.values())
        return -(-mn * len(self.map) // self.bs)


def main():
    dev = torch.device('cuda')
    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    name = f'{VARIANT}_{TASK}_{LOSS}_s{SEED}'
    out = HERE / '_local_runs' / name
    out.mkdir(parents=True, exist_ok=True)
    names = P.TASKS[TASK]
    root = common.find_dataset()
    rows = P.read_labels(RUN / 'runs/legibility_labels')
    for r in rows:
        if VARIANT == 'failure':
            r['y'] = 'suitable' if vocab.norm_text(r['crnn_pred']) == vocab.norm_text(r['text']) else 'unsuitable'
        else:
            r['y'] = r['label4'] if TASK == 'baseline4' else r['quality']
    sp = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    ds = lambda s, tr: P.CropDS(sp[s], names, root, 96, 192, tr)
    if VARIANT == 'balanced':
        dtr = ds('train', True)
        dl = torch.utils.data.DataLoader(dtr, batch_sampler=Balanced([dtr.idx[r['y']] for r in sp['train']], 64))
    else:
        dl = torch.utils.data.DataLoader(ds('train', True), batch_size=64, shuffle=True, drop_last=True)
    dv = torch.utils.data.DataLoader(ds('val', False), batch_size=64)
    dt = torch.utils.data.DataLoader(ds('test', False), batch_size=64)
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, len(names))
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.75, patience=5, min_lr=5e-6)
    lossf = (TextEMA(len(names)) if VARIANT == 'text' else P.EMALoss(len(names))) if LOSS == 'ema' else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    best, bad, log, t0, state = -1, 0, [], time.time(), None
    for ep in range(80):
        model.train()
        tl = n = 0
        te = time.time()
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast('cuda'):
                o = model(x)
            loss = lossf(o.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss.detach()) * len(y); n += len(y)
        tt = time.time() - te
        pv, yv = P.predict_probs(model, dv, dev)
        acc = float((pv.argmax(1) == yv).mean())
        sched.step(acc)
        log.append({'epoch': ep + 1, 'train_loss': round(tl / max(1, n), 4), 'val_acc': round(100 * acc, 2),
                    'sec_train': round(tt, 1), 'sec_all': round(time.time() - te, 1)})
        print(name, log[-1], flush=True)
        if acc > best:
            best, bad = acc, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= 20:
            break
    model.load_state_dict(state)
    pt, yt = P.predict_probs(model, dt, dev)
    pv, yv = P.predict_probs(model, dv, dev)
    res = {'run': name, 'variant': VARIANT, 'task': TASK, 'loss': LOSS, 'seed': SEED, 'classes': names,
           'epochs_run': len(log), 'best_epoch': int(np.argmax([l['val_acc'] for l in log]) + 1),
           'minutes': round((time.time() - t0) / 60, 1), 'val': P.f1_scores(yv, pv.argmax(1), len(names)),
           'test': P.f1_scores(yt, pt.argmax(1), len(names)), 'log': log, 'gpu': torch.cuda.get_device_name(0)}
    if TASK == 'baseline4':
        q = lambda p: (p[:, 2] + p[:, 3] > p[:, 0] + p[:, 1]).astype(int)
        res['test_quality_from_4class'] = P.f1_scores((yt >= 2).astype(int), q(pt), 2)
    for s, p in (('test', pt), ('val', pv)):
        with open(out / f'{s}_scores.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file', 'label', 'pred'] + [f'p_{c}' for c in names])
            for r, q in zip(sp[s], p):
                w.writerow([r['file'], r['y'], names[int(q.argmax())]] + [round(float(v), 5) for v in q])
    json.dump(res, open(out / 'results.json', 'w', encoding='utf-8'), indent=1)
    print('DONE', name, res['test']['micro_F1_%'], res['epochs_run'], res['minutes'], flush=True)


if __name__ == '__main__':
    main()
