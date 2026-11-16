"""Paper #19 (LPLCv2): a plate LEGIBILITY / quality classifier for our Bangla plates, and whether it is a better
"reject or read" gate than the OCR's own confidence.

  python p19_legibility.py reader --family crnn --name readers_crnn --project runs     (GPU 0)
  python p19_legibility.py reader --family slot --name readers_slot --project runs     (GPU 1)
  python p19_legibility.py labels --runs runs/readers_crnn runs/readers_slot --out runs/legibility_labels
  python p19_legibility.py train --labels runs/legibility_labels --task baseline4 --loss ema --name cls_b4_ema --project runs
  python p19_legibility.py filter --labels runs/legibility_labels --cls runs/cls_b4_ema runs/cls_qf_ema ... --name gate --project runs

Our crops have no human legibility labels (LPLCv2 itself is only given on e-mail request), so the label is an
OCR-DIFFICULTY proxy made by two reader families that never saw the crop (cross-fitting):
  reader   CRNN-CTC and the slot CNN (lprkit/seq.py models), each trained twice on one half of the OCR train split
           (halves = plate text hash, so one plate never sits in both halves) and once on all of it (CRNN only, the
           "deployment" reader). Each half model reads the other half; val / test are read by the half-A models,
           so every crop gets its label from readers of the same strength.
  labels   4 ordinal classes like the paper's perfect / good / poor / illegible (p.3-4):
             perfect   both readers exact            good      exactly one reader exact
             poor      neither exact, best reader within 1 token      illegible  neither, both >= 2 tokens wrong
           Quality Filter (paper p.4, Table III) = suitable (perfect + good) vs unsuitable (poor + illegible).
  train    ResNet-50 (ImageNet, all layers trained), 96 x 192 letterboxed RGB, Adam 1e-5, batch 64, LR x0.75 after 5
           epochs without a better val accuracy (min 5e-6), early stop after 20 (paper p.5), with the paper's EMA loss
           (authors' code: class weight = max(errors) / (errors + 1e-6), at most 1.2, from an EMA of the batch
           confusion matrix with alpha 0.8) or plain cross-entropy.
  filter   on the test crops: plate exact of the deployment CRNN on the plates each gate ACCEPTS at 100/90/80/70/50 %
           coverage, gate = legibility classifier vs the CRNN's own confidence vs both; share flagged per condition.
           If <dataset>/ocr/legibility_human.csv (file,label) exists, the classifier is also scored against it.
"""
import argparse
import csv
import hashlib
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import seq
import vocab

CLASSES4 = ['perfect', 'good', 'poor', 'illegible']
TASKS = {'baseline4': CLASSES4, 'quality': ['suitable', 'unsuitable']}
TO_QUALITY = {'perfect': 'suitable', 'good': 'suitable', 'poor': 'unsuitable', 'illegible': 'unsuitable'}
READER = {'crnn': dict(H=48, W=256, stack=True, epochs=60, lr=1e-3), 'slot': dict(H=64, W=128, stack=False, epochs=100, lr=1e-3)}


# ---------------------------------------------------------------------------------------------------------- readers
def half(text):
    return int(hashlib.md5(vocab.norm_text(text).encode('utf-8')).hexdigest(), 16) % 2


def train_reader(kind, tr, va, cls, epochs, dev, seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    c = READER[kind]
    model = (seq.CRNN(len(cls), c['H']) if kind == 'crnn' else seq.SlotCNN(len(cls))).to(dev)
    dl = torch.utils.data.DataLoader(seq.DS(tr, cls, c['H'], c['W'], c['stack'], True), batch_size=64, shuffle=True,
                                     num_workers=common.WORKERS, collate_fn=seq.collate, drop_last=True)
    dv = torch.utils.data.DataLoader(seq.DS(va, cls, c['H'], c['W'], c['stack'], False), batch_size=64,
                                     num_workers=min(2, common.WORKERS), collate_fn=seq.collate)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c['lr'], total_steps=epochs * max(1, len(dl)), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    best, state = 1e9, None
    for ep in range(epochs):
        tl = seq.run_epoch(model, dl, opt, sched, kind, dev, scaler, True)
        with torch.no_grad():
            vl = seq.run_epoch(model, dv, None, None, kind, dev, scaler, False)
        if vl < best:
            best, state = vl, {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f'  {kind} epoch {ep + 1}/{epochs} train {tl:.4f} val {vl:.4f}', flush=True)
    model.load_state_dict(state)
    return model, SimpleNamespace(H=c['H'], W=c['W'], stack=c['stack'])


def reader(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    cls = seq.class_list()
    tr, va, te = seq.load_rows('train'), seq.load_rows('val'), seq.load_rows('test')
    if a.fraction < 1:
        random.Random(0).shuffle(tr)
        tr, va, te = tr[:max(40, int(len(tr) * a.fraction))], va[:a.max_eval], te[:a.max_eval]
    for r, s in ((r, 'train') for r in tr):
        r['split'] = s
    for r in va:
        r['split'] = 'val'
    for r in te:
        r['split'] = 'test'
    halves = {0: [r for r in tr if half(r['text']) == 0], 1: [r for r in tr if half(r['text']) == 1]}
    epochs = a.epochs or READER[a.family]['epochs']
    out_rows, t0 = [], time.time()
    for h in (0, 1):
        print(f'{a.family}: half model {h} on {len(halves[h])} crops, reads {len(halves[1 - h])} train crops', flush=True)
        m, args = train_reader(a.family, halves[h], va, cls, epochs, dev)
        targets = halves[1 - h] + (va + te if h == 0 else [])
        preds, _ = seq.predict(m, targets, cls, a.family, args, dev)
        for r, p in zip(targets, preds):
            out_rows.append({'file': r['file'], 'split': r['split'], 'text': r['text'], 'source': r['source'],
                             'pred': p['pred'], 'conf': p['conf'], 'reader': f'{a.family}_half{h}'})
    res = {'run': a.name, 'kind': 'legibility-reader', 'family': a.family, 'epochs': epochs,
           'halves': {str(h): len(v) for h, v in halves.items()}}
    if a.family == 'crnn':                                   # the deployment reader, trained on all train crops
        m, args = train_reader('crnn', tr, va, cls, epochs, dev)
        preds, ms = seq.predict(m, te, cls, 'crnn', args, dev)
        for r, p in zip(te, preds):
            out_rows.append({'file': r['file'], 'split': 'test', 'text': r['text'], 'source': r['source'],
                             'pred': p['pred'], 'conf': p['conf'], 'reader': 'crnn_full'})
        res['ocr_test'] = {'rows_decoder': metrics.score_rows((p['pred'], p['true']) for p in preds), 'ms_per_plate': round(ms, 2)}
        torch.save({'state_dict': m.state_dict(), 'kind': 'crnn', 'classes': cls, 'H': args.H, 'W': args.W,
                    'stack': args.stack}, run / 'model_full.pt')
    with open(run / 'reader_preds.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    res['train_minutes'] = round((time.time() - t0) / 60, 1)
    res['env'] = common.env_info()
    common.save_json(res, run / 'results.json')
    print('DONE', a.name, res.get('ocr_test'))


# ----------------------------------------------------------------------------------------------------------- labels
def tok_err(pred, true):
    return metrics.lev(metrics.tokens(pred), metrics.tokens(true))


def labels(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in a.runs:
        rows += list(csv.DictReader(open(Path(r) / 'reader_preds.csv', encoding='utf-8')))
    by = {}
    for r in rows:
        by.setdefault((r['split'], r['file']), {})[r['reader']] = r
    lab, deploy = [], {}
    for (split, f), d in by.items():
        fam = {}
        for k, r in d.items():
            if k == 'crnn_full':
                deploy[f] = r
                continue
            if split in ('val', 'test') and not k.endswith('half0'):
                continue
            fam[k.split('_half')[0]] = r
        if set(fam) != {'crnn', 'slot'}:
            continue
        t = next(iter(fam.values()))['text']
        exact = {k: vocab.norm_text(v['pred']) == vocab.norm_text(t) for k, v in fam.items()}
        err = {k: tok_err(v['pred'], t) for k, v in fam.items()}
        n = sum(exact.values())
        lab4 = 'perfect' if n == 2 else 'good' if n == 1 else 'poor' if min(err.values()) <= 1 else 'illegible'
        lab.append({'file': f, 'split': split, 'text': t, 'source': fam['crnn']['source'], 'label4': lab4,
                    'quality': TO_QUALITY[lab4], 'crnn_pred': fam['crnn']['pred'], 'slot_pred': fam['slot']['pred']})
    with open(out / 'labels.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(lab[0]))
        w.writeheader(); w.writerows(lab)
    if deploy:
        with open(out / 'deploy_test.csv', 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['file', 'text', 'pred', 'conf', 'source'])
            w.writeheader()
            w.writerows({k: r[k] for k in ('file', 'text', 'pred', 'conf', 'source')} for r in deploy.values())
    counts = {s: {c: sum(1 for r in lab if r['split'] == s and r['label4'] == c) for c in CLASSES4} for s in ('train', 'val', 'test')}
    res = {'run': 'legibility_labels', 'kind': 'legibility-labels', 'counts': counts, 'rows': len(lab),
           'rule': 'perfect = both readers exact; good = one exact; poor = neither, best within 1 token; illegible = rest'}
    common.save_json(res, out / 'results.json')
    print(res)


# ------------------------------------------------------------------------------------------------------- classifier
class CropDS(torch.utils.data.Dataset):
    MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def __init__(self, rows, names, root, H, W, train):
        self.rows, self.idx, self.root, self.H, self.W, self.train = rows, {n: i for i, n in enumerate(names)}, root, H, W, train

    def __len__(self):
        return len(self.rows)

    def load(self, r):
        img = cv2.imread(os.path.join(self.root, 'ocr', 'images', r['split'], r['file']))
        return np.zeros((40, 120, 3), np.uint8) if img is None else img

    def __getitem__(self, i):
        r = self.rows[i]
        img = self.load(r)[:, :, ::-1]
        if self.train:                                   # mild photometric + geometric change (paper: albumentations)
            img = np.clip(img.astype(np.float32) * random.uniform(0.85, 1.15) + random.uniform(-15, 15), 0, 255).astype(np.uint8)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-3, 3), random.uniform(0.95, 1.05))
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        h, w = img.shape[:2]
        s = min(self.H / h, self.W / w)
        nh, nw = max(1, int(h * s)), max(1, int(w * s))
        canvas = np.full((self.H, self.W, 3), 127, np.uint8)
        y0, x0 = (self.H - nh) // 2, (self.W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(np.ascontiguousarray(img), (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = (canvas.astype(np.float32) / 255 - self.MEAN) / self.STD
        return torch.from_numpy(x.transpose(2, 0, 1)).float(), self.idx[r['y']]


class EMALoss(nn.Module):
    """The paper's EMA loss as in the authors' code (models/loss.py, FocalEMA): per-class weights from an EMA of the
    batch confusion matrix, weight = max(errors) / (errors + 1e-6), clamped at 1.2."""
    def __init__(self, n, alpha=0.8, cap=1.2):
        super().__init__()
        self.n, self.alpha, self.cap = n, alpha, cap
        self.ema = None

    def forward(self, logits, y):
        with torch.no_grad():
            conf = torch.zeros(self.n, self.n, device=logits.device)
            conf.index_put_((y, logits.argmax(1)), torch.ones_like(y, dtype=torch.float), accumulate=True)
            self.ema = conf if self.ema is None else self.alpha * conf + (1 - self.alpha) * self.ema
            mis = self.ema.sum(1) - torch.diagonal(self.ema)
            w = torch.clamp(mis.max() / (mis + 1e-6), max=self.cap)
        return (F.cross_entropy(logits, y, reduction='none') * w[y]).mean()


def f1_scores(y, p, n):
    cm = np.zeros((n, n), int)
    for t, q in zip(y, p):
        cm[t, q] += 1
    per = []
    for c in range(n):
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        per.append(round(100 * 2 * tp / max(1, 2 * tp + fp + fn), 2))
    return {'micro_F1_%': round(100 * np.trace(cm) / max(1, cm.sum()), 2), 'macro_F1_%': round(float(np.mean(per)), 2),
            'per_class_F1_%': per, 'confusion': cm.tolist()}


@torch.no_grad()
def predict_probs(model, dl, dev):
    model.eval()
    out, ys = [], []
    for x, y in dl:
        with torch.autocast('cuda', enabled=dev.type == 'cuda'):
            out.append(model(x.to(dev)).float().softmax(1).cpu())
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def read_labels(path):
    return list(csv.DictReader(open(Path(path) / 'labels.csv', encoding='utf-8')))


def train_cls(a):
    import torchvision
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    names = TASKS[a.task]
    root = common.find_dataset()
    rows = read_labels(a.labels)
    for r in rows:
        r['y'] = r['label4'] if a.task == 'baseline4' else r['quality']
    sp = {s: [r for r in rows if r['split'] == s] for s in ('train', 'val', 'test')}
    if a.fraction < 1:
        sp['train'] = sp['train'][:max(64, int(len(sp['train']) * a.fraction))]
    mk = lambda s, tr: torch.utils.data.DataLoader(CropDS(sp[s], names, root, a.H, a.W, tr), batch_size=a.batch, shuffle=tr,
                                                   num_workers=common.WORKERS, drop_last=tr)
    dl, dv, dt = mk('train', True), mk('val', False), mk('test', False)
    try:
        model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    except Exception as e:                                       # no internet in a local test
        print('ImageNet weights not available:', e); model = torchvision.models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, len(names))
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.75, patience=5, min_lr=5e-6)
    lossf = EMALoss(len(names)) if a.loss == 'ema' else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=dev.type == 'cuda')
    best, bad, log, t0 = -1, 0, [], time.time()
    for ep in range(a.epochs):
        model.train()
        tl = n = 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            with torch.autocast('cuda', enabled=dev.type == 'cuda'):
                out = model(x)
            loss = lossf(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += float(loss) * len(y); n += len(y)
        pv, yv = predict_probs(model, dv, dev)
        acc = float((pv.argmax(1) == yv).mean())
        sched.step(acc)
        log.append({'epoch': ep + 1, 'train_loss': round(tl / max(1, n), 4), 'val_acc': round(100 * acc, 2),
                    'lr': opt.param_groups[0]['lr']})
        print(log[-1], flush=True)
        if acc > best:
            best, bad = acc, 0
            torch.save(model.state_dict(), run / 'best.pt')
        else:
            bad += 1
        if bad >= a.patience or (a.max_hours and time.time() - t0 > a.max_hours * 3600):
            print(f'stop after epoch {ep + 1} (early stop / time cap)', flush=True)
            break
    model.load_state_dict(torch.load(run / 'best.pt', map_location=dev))
    pt, yt = predict_probs(model, dt, dev)
    pv, yv = predict_probs(model, dv, dev)
    x = next(iter(dt))[0][:1].to(dev)
    ms = []
    for _ in range(60):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter(); model(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ms.append(1000 * (time.perf_counter() - t1))
    res = {'run': a.name, 'kind': 'legibility-classifier', 'task': a.task, 'loss': a.loss, 'classes': names,
           'settings': {'model': 'resnet50 IMAGENET1K_V1, all layers', 'input_HW': [a.H, a.W], 'optimizer': f'Adam {a.lr}',
                        'batch': a.batch, 'plateau': 'x0.75 after 5, min 5e-6', 'early_stop': a.patience, 'seed': a.seed},
           'epochs_planned': a.epochs, 'epochs_run': len(log), 'train_minutes': round((time.time() - t0) / 60, 1),
           'val': f1_scores(yv, pv.argmax(1), len(names)), 'test': f1_scores(yt, pt.argmax(1), len(names)),
           'ms_per_crop': round(float(np.median(ms[10:])), 2), 'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           'log': log}
    if a.task == 'baseline4':                                 # the paper's other scenarios, read from the 4-class output
        q = lambda p: (p[:, 2] + p[:, 3] > p[:, 0] + p[:, 1]).astype(int)
        res['test_quality_from_4class'] = f1_scores((yt >= 2).astype(int), q(pt), 2)
    with open(run / 'test_scores.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['file', 'label', 'pred'] + [f'p_{n}' for n in names])
        for r, p in zip(sp['test'], pt):
            w.writerow([r['file'], r['y'], names[int(p.argmax())]] + [round(float(v), 5) for v in p])
    common.save_json(res, run / 'results.json')
    print('DONE', a.name, res['test']['micro_F1_%'], res['test']['per_class_F1_%'])


# ----------------------------------------------------------------------------------------------------------- filter
def suitability(run):
    rows = list(csv.DictReader(open(Path(run) / 'test_scores.csv', encoding='utf-8')))
    ok = lambda r: (float(r['p_perfect']) + float(r['p_good'])) if 'p_perfect' in r else float(r['p_suitable'])
    return {r['file']: ok(r) for r in rows}


def selective(rows, score, covers=(100, 90, 80, 70, 50)):
    order = sorted(rows, key=lambda r: -score(r))
    out = {}
    for c in covers:
        k = max(1, int(round(len(order) * c / 100)))
        acc = order[:k]
        out[f'{c}%'] = round(100 * np.mean([r['ok'] for r in acc]), 2)
    rej = order[int(round(len(order) * 0.8)):]
    out['wrong_among_rejected_20%'] = round(100 * np.mean([not r['ok'] for r in rej]), 2) if rej else None
    accs = [np.mean([r['ok'] for r in order[:k]]) for k in range(1, len(order) + 1)]
    out['AURC_error_%'] = round(100 * (1 - float(np.mean(accs))), 2)           # area under the risk-coverage curve
    return out


def filt(a):
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    root = common.find_dataset()
    dep = {r['file']: r for r in csv.DictReader(open(Path(a.labels) / 'deploy_test.csv', encoding='utf-8'))}
    for r in dep.values():
        r['ok'] = vocab.norm_text(r['pred']) == vocab.norm_text(r['text'])
        r['conf'] = float(r['conf'])
    cond = {}
    cpath = os.path.join(root, 'ocr', 'test_conditions.csv')
    if os.path.exists(cpath):
        for r in csv.DictReader(open(cpath, encoding='utf-8')):
            cond[r['file']] = [k for k in ('night', 'blurry', 'low_res', 'clear') if r.get(k) == '1']
    rows = list(dep.values())
    res = {'run': a.name, 'kind': 'legibility-gate', 'test_plates': len(rows),
           'deploy_reader_exact_%': round(100 * np.mean([r['ok'] for r in rows]), 2),
           'gates': {'ocr_confidence': selective(rows, lambda r: r['conf'])}}
    human_path = os.path.join(root, 'ocr', 'legibility_human.csv')
    human = {r['file']: r['label'] for r in csv.DictReader(open(human_path, encoding='utf-8'))} if os.path.exists(human_path) else {}
    for c in a.cls:
        name = Path(c).name
        if not (Path(c) / 'test_scores.csv').exists():
            continue
        s = suitability(c)
        res['gates'][name] = selective([r for r in rows if r['file'] in s], lambda r: s[r['file']])
        res['gates'][f'{name}+ocr_confidence'] = selective([r for r in rows if r['file'] in s],
                                                          lambda r: s[r['file']] * r['conf'])
        flagged = {k: [s[f] < 0.5 for f in s if k in cond.get(f, [])] for k in ('night', 'blurry', 'clear')}
        res['gates'][name]['flagged_unsuitable_%_by_condition'] = {k: round(100 * np.mean(v), 2) for k, v in flagged.items() if v}
        if human:
            hs = [(TO_QUALITY.get(human[f], human[f]) == 'suitable', s[f] >= 0.5) for f in s if f in human]
            res['gates'][name]['vs_human_quality_acc_%'] = round(100 * np.mean([x == y for x, y in hs]), 2) if hs else None
    g = res['gates']
    best = max((k for k in g if k != 'ocr_confidence' and '+' not in k), key=lambda k: g[k]['80%'], default=None)
    if best:
        res['best_classifier'] = best
        res['gain_at_80%_vs_ocr_confidence_pp'] = round(g[best]['80%'] - g['ocr_confidence']['80%'], 2)
    res['env'] = common.env_info()
    common.save_json(res, run / 'results.json')
    print('DONE', a.name, {k: (v['80%'], v['AURC_error_%']) for k, v in g.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['reader', 'labels', 'train', 'filter'])
    p.add_argument('--family', choices=list(READER), default='crnn')
    p.add_argument('--runs', nargs='*', default=[])
    p.add_argument('--out', default='')
    p.add_argument('--labels', default='')
    p.add_argument('--cls', nargs='*', default=[])
    p.add_argument('--task', choices=list(TASKS), default='baseline4')
    p.add_argument('--loss', choices=['ema', 'ce'], default='ema')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--H', type=int, default=96)
    p.add_argument('--W', type=int, default=192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--max-eval', type=int, default=None)
    p.add_argument('--max-hours', type=float, default=None)
    p.add_argument('--name', default='legibility')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    if a.mode == 'train' and a.epochs is None:
        a.epochs = 80
    {'reader': reader, 'labels': labels, 'train': train_cls, 'filter': filt}[a.mode](a)
