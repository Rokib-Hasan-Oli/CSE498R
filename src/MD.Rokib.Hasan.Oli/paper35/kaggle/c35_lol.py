"""Paper #35, original-paper check: the authors' released HVI-CIDNet weights on the paper's own test sets.

  python c35_lol.py prep --root /kaggle/working/lol
  python c35_lol.py eval --root /kaggle/working/lol --models hvi_lolv1 hvi_lolv1_perc --set lolv1 --name lolv1 --project runs

Data (public copies on Hugging Face, pinned revisions):
  LOLv1 eval15 (15 pairs)       geekyrakshit/LoL-Dataset  lol_dataset.zip  (the authors point to the LOL website)
  LOLv2-Real test (100 pairs)   okhater/lolv2-real        Test/Input, Test/GT
Weights: the authors' Hugging Face safetensors (README "all weights are public at Hugging Face"), with the switches of
their eval.py (LOLv1: trans.gated; LOLv2-Real best-PSNR: gated2, alpha 0.8; best-SSIM: alpha 0.82).
Metrics exactly as their measure.py: PSNR on 0-255 RGB, SSIM = mean of per-channel Gaussian-window SSIM (MATLAB
style), LPIPS (AlexNet, if the `lpips` package installs), each with and WITHOUT "GT mean" (the output brightness
rescaled to the ground truth's mean grey before scoring; the paper's LOLv1 column uses it, Table 1 caption, p.6).
Plus params and GPU ms per image (the paper: 1.88 M, 7.57 GFLOPs at 256 x 256, 0.053 s, Tables 1 and 3).
"""
import argparse
import glob
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import p35_lowlight as p35

LOLV1 = ('geekyrakshit/LoL-Dataset', 'bdef22d0f5803da8775114496a6689333265e157', 'lol_dataset.zip')
LOLV2 = ('okhater/lolv2-real', 'db2979107f7c22d372c6e4f1b06fd6203069af0a')


def prep(a):
    from huggingface_hub import hf_hub_download, snapshot_download
    root = Path(a.root)
    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'lolv2_real' / 'Test' / 'GT').exists():
        snapshot_download(repo_id=LOLV2[0], repo_type='dataset', revision=LOLV2[1], allow_patterns=['Test/*'],
                          local_dir=str(root / 'lolv2_real'))
    if not list((root / 'lolv1').glob('**/eval15/low/*.png')):
        z = hf_hub_download(repo_id=LOLV1[0], repo_type='dataset', filename=LOLV1[2], revision=LOLV1[1])
        with zipfile.ZipFile(z) as f:
            f.extractall(root / 'lolv1', members=[m for m in f.namelist() if 'eval15' in m])
    sets = {'lolv1': sorted(glob.glob(str(root / 'lolv1' / '**' / 'eval15' / 'low' / '*.png'), recursive=True)),
            'lolv2_real': sorted(glob.glob(str(root / 'lolv2_real' / 'Test' / 'Input' / '*.png')))}
    pairs = {}
    for k, lows in sets.items():
        pairs[k] = []
        for l in lows:
            h = l.replace(os.sep + 'low' + os.sep, os.sep + 'high' + os.sep) if k == 'lolv1' else \
                l.replace(os.sep + 'Input' + os.sep, os.sep + 'GT' + os.sep)
            if os.path.exists(h):
                pairs[k].append([l, h])
    common.save_json(pairs, root / 'pairs.json')
    print({k: len(v) for k, v in pairs.items()})


def _ssim(img1, img2):                                          # the authors' measure.py
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(k, k.transpose())
    mu1, mu2 = cv2.filter2D(img1, -1, w)[5:-5, 5:-5], cv2.filter2D(img2, -1, w)[5:-5, 5:-5]
    s1 = cv2.filter2D(img1 ** 2, -1, w)[5:-5, 5:-5] - mu1 ** 2
    s2 = cv2.filter2D(img2 ** 2, -1, w)[5:-5, 5:-5] - mu2 ** 2
    s12 = cv2.filter2D(img1 * img2, -1, w)[5:-5, 5:-5] - mu1 * mu2
    return (((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))).mean()


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(np.mean([_ssim(a[:, :, i], b[:, :, i]) for i in range(3)]))


def psnr(a, b):
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(10 * np.log10(255.0 ** 2 / (np.mean(d ** 2) + 1e-8)))


def gt_mean(out, gt):
    """measure.py's GT-mean step: means of the uint8 images, then the output scaled to the ground truth's brightness."""
    mo = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).mean()
    mt = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY).mean()
    return np.clip(out.astype(np.float64) * (mt / mo), 0, 255)


def evaluate(a):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    pairs = common.load_json(Path(a.root) / 'pairs.json')[a.set][:a.max]
    try:
        import lpips
        lp = lpips.LPIPS(net='alex', verbose=False).to(dev)
    except Exception as e:
        lp = None
        print('LPIPS not available:', e)
    out = {'run': a.name, 'kind': 'hvi-check', 'set': a.set, 'pairs': len(pairs), 'models': {}}
    for key in a.models:
        net = p35.load_cidnet(key, dev)
        s = {'plain': {'psnr': [], 'ssim': [], 'lpips': []}, 'gt_mean': {'psnr': [], 'ssim': [], 'lpips': []}}
        ms = []
        for low, high in pairs:
            x = torch.from_numpy(cv2.imread(low)[:, :, ::-1].copy()).float().div(255).permute(2, 0, 1)[None].to(dev)
            h, w = x.shape[2:]
            H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8
            xp = torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode='reflect') if (H, W) != (h, w) else x
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                y = net(xp).clamp(0, 1)[:, :, :h, :w]
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - t0))
            o = np.round(y[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)   # saved as PNG by eval.py
            g = cv2.imread(high)[:, :, ::-1].copy()
            for mode, img in (('plain', o.astype(np.float64)), ('gt_mean', gt_mean(o, g))):
                s[mode]['psnr'].append(psnr(img, g)); s[mode]['ssim'].append(ssim(img, g))
                if lp is not None:
                    t = lambda z: torch.from_numpy(np.ascontiguousarray(z)).float().permute(2, 0, 1)[None].to(dev) / 127.5 - 1
                    s[mode]['lpips'].append(float(lp(t(g), t(img))))
        r = {mode: {k: round(float(np.mean(v)), 4) if v else None for k, v in d.items()} for mode, d in s.items()}
        r['ms_median'] = round(float(np.median(ms[2:] if len(ms) > 4 else ms)), 2)
        r['params_M'] = round(sum(p.numel() for p in net.parameters()) / 1e6, 3)
        r['hf'] = p35.HF[key][:2]
        out['models'][key] = r
        print(key, a.set, r, flush=True)
        del net
    out['env'] = common.env_info()
    common.save_json(out, run / 'results.json')
    print('DONE', a.name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prep', 'eval'])
    p.add_argument('--root', default=os.path.join(common.WORK, 'lol'))
    p.add_argument('--set', choices=['lolv1', 'lolv2_real'], default='lolv1')
    p.add_argument('--models', nargs='*', default=['hvi_lolv1'])
    p.add_argument('--max', type=int, default=None)
    p.add_argument('--name', default='lol')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    prep(a) if a.mode == 'prep' else evaluate(a)
