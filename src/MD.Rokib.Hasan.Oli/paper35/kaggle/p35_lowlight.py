"""Paper #35 (HVI-CIDNet, CVPR 2025): low-light enhancement before our plate detector and plate reader.

  python p35_lowlight.py setup --dir /kaggle/working/hvi_cidnet        (git clone + checkout of the pinned commit)
  python p35_lowlight.py eval --det runs/det_v8s/weights/best.pt --ocr runs/crnn/model_full.pt --name lowlight --project runs

Enhancers (all zero-shot, nothing is trained on our plates):
  none        the photo as it is
  gamma       gamma 0.5 brightening (a classic baseline)
  clahe       CLAHE on the L channel of Lab (clip 2.0, 8x8 tiles)
  hvi_gen     HVI-CIDNet, the authors' "generalization" weights (LOLv2-Syn + random gamma, their recommended
              real-world model; Hugging Face Fediory/HVI-CIDNet-Generalization), unpaired setting (gated2, alpha 1.0)
  hvi_lolv2   HVI-CIDNet LOLv2-Real best-PSNR weights (Fediory/HVI-CIDNet-LOLv2-real-bestPSNR, alpha 0.8 as in
              the authors' eval.py)
Tests:
  detection   our plate detector on enhanced copies of: the night test photos (detection/cond_test_night.txt, the
              team rule "mean grey < 60 or NightLPBD"), 200 clear photos (does it hurt daylight?), and the whole test
              split with a GATE (only photos with mean grey < 60 are enhanced): mAP50 / mAP50-95 / plates found %
  reading     our CRNN on the night-tagged OCR test crops and on all crops with the same gate: plate exact, CER
  speed       enhancer ms per photo (640 x 640) and per plate crop (64 x 192) on the GPU, batch 1, median
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import det
import metrics
import seq

REPO = 'https://github.com/Fediory/HVI-CIDNet.git'
COMMIT = 'eb43d7d91e9a336c66856824ff9e4603ae41f408'          # master, 9 Mar 2026
HF = {'hvi_gen': ('Fediory/HVI-CIDNet-Generalization', '51481ef2546f870060c43eb6d6525399f5b3d2d3', dict(gated2=True, alpha=1.0)),
      'hvi_lolv2': ('Fediory/HVI-CIDNet-LOLv2-real-bestPSNR', 'c317f57dec7dd66704ac1b13789a2fa5f20746de', dict(gated2=True, alpha=0.8)),
      'hvi_lolv2_ssim': ('Fediory/HVI-CIDNet-LOLv2-real-bestSSIM', '5f759d5426d1326e46b9fbc4b6e4417c54a97226', dict(gated2=True, alpha=0.82)),
      'hvi_lolv1': ('Fediory/HVI-CIDNet-LOLv1-woperc', 'de3f94cef143b2dcf4d62de303ad17b4c27d7dc1', dict(gated=True)),
      'hvi_lolv1_perc': ('Fediory/HVI-CIDNet-LOLv1-wperc', '1fa9b6993413a8fe7df1ba433682ccf525349c88', dict(gated=True))}
DARK = 60                                                  # the team's night rule: mean grey < 60


def setup(a):
    d = Path(a.dir)
    if not (d / 'net').exists():
        subprocess.run(['git', 'clone', '-q', REPO, str(d)], check=True)
    subprocess.run(['git', '-C', str(d), 'checkout', '-q', COMMIT], check=True)
    print('HVI-CIDNet code ready at', d, 'commit', COMMIT)


def hvi_dir():
    d = os.environ.get('HVI_DIR', os.path.join(common.WORK, 'hvi_cidnet'))
    if not os.path.exists(os.path.join(d, 'net', 'CIDNet.py')):
        raise FileNotFoundError(f'HVI-CIDNet code not found at {d}: run `p35_lowlight.py setup --dir {d}` first')
    if d not in sys.path:
        sys.path.insert(0, d)
    return d


def load_cidnet(key, dev):
    """CIDNet with the authors' Hugging Face safetensors weights (no pickle) and the authors' test switches."""
    hvi_dir()
    from huggingface_hub import hf_hub_download
    import safetensors.torch as sf
    from net.CIDNet import CIDNet
    repo, rev, sw = HF[key]
    m = CIDNet()
    sd = sf.load_file(hf_hub_download(repo_id=repo, filename='model.safetensors', revision=rev))
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected and len(missing) < 3, f'{key}: weights do not fit ({len(missing)} missing, {len(unexpected)} unexpected)'
    for k, v in sw.items():
        setattr(m.trans, k, v)
    return m.eval().to(dev)


class Enhancer:
    def __init__(self, name, dev, max_side=1280):
        self.name, self.dev, self.max_side = name, dev, max_side
        self.net = load_cidnet(name, dev) if name.startswith('hvi') else None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.lut = np.array([255 * (i / 255) ** 0.5 for i in range(256)], np.uint8)

    @torch.no_grad()
    def __call__(self, bgr):
        if self.name == 'none':
            return bgr
        if self.name == 'gamma':
            return cv2.LUT(bgr, self.lut)
        if self.name == 'clahe':
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        h0, w0 = bgr.shape[:2]
        s = min(1.0, self.max_side / max(h0, w0))
        img = cv2.resize(bgr, (int(w0 * s), int(h0 * s)), interpolation=cv2.INTER_AREA) if s < 1 else bgr
        x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1])).float().div(255).permute(2, 0, 1)[None].to(self.dev)
        h, w = x.shape[2:]
        H, W = (h + 7) // 8 * 8, (w + 7) // 8 * 8                  # the network needs sizes divisible by 8
        x = torch.nn.functional.pad(x, (0, W - w, 0, H - h), mode='reflect') if (H, W) != (h, w) else x
        y = self.net(x).clamp(0, 1)[:, :, :h, :w]
        out = (y[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1] * 255).round().astype(np.uint8)
        return cv2.resize(out, (w0, h0), interpolation=cv2.INTER_CUBIC) if s < 1 else out


def grey(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean())


def write_subset(enh, files, lab, out, gate=False):
    """enhanced copies + labels in a YOLO folder; returns (yaml path, photos enhanced, mean grey before / after)."""
    import yaml
    (out / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (out / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
    n, before, after = 0, [], []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        g0 = grey(img)
        if not gate or g0 < DARK:
            img2 = enh(img)
            n += int(enh.name != 'none')
            before.append(g0); after.append(grey(img2))
            img = img2
        cv2.imwrite(str(out / 'images' / 'test' / f.name), img)
        if lab(f).exists():
            shutil.copy(lab(f), out / 'labels' / 'test' / lab(f).name)
    y = out / 'data.yaml'
    y.write_text(yaml.safe_dump({'path': str(out), 'train': 'images/test', 'val': 'images/test', 'test': 'images/test',
                                 'names': {0: 'plate'}}))
    return y, n, (round(float(np.mean(before)), 1) if before else None), (round(float(np.mean(after)), 1) if after else None)


def load_ocr(path, dev):
    c = torch.load(path, map_location='cpu', weights_only=False)
    n = len(c['classes'])
    m = {'crnn': lambda: seq.CRNN(n, c['H']), 'slot': lambda: seq.SlotCNN(n),
         'vitstr': lambda: seq.ViTSTR(n, c['H'], c['W'])}[c['kind']]()
    m.load_state_dict(c['state_dict'])
    return m.eval().to(dev), c


@torch.no_grad()
def read(ocr, c, imgs, dev):
    x = torch.stack([seq.prep(i, c['H'], c['W'], c['stack'], False) for i in imgs]).to(dev)
    out = torch.cat([ocr(x[k:k + 128]).float() for k in range(0, len(x), 128)])
    toks, _ = seq.ctc_decode(out) if c['kind'] == 'crnn' else seq.fixed_decode(out)
    return [seq.vocab.tokens_to_text([c['classes'][k - 1] for k in t]) for t in toks]


def ms_of(enh, shape, dev, n=30):
    img = np.random.default_rng(0).integers(0, 60, (*shape, 3), dtype=np.uint8)
    for _ in range(5):
        enh(img)
    ts = []
    for _ in range(n):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter(); enh(img)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    return round(float(np.median(ts)), 2)


def evaluate(a):
    from ultralytics import YOLO
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run = Path(a.project) / a.name
    run.mkdir(parents=True, exist_ok=True)
    root = Path(common.find_dataset())
    data = common.data_yaml('detection', 'human')
    imgs, lab = det.split_files(data, 'test')
    byname = {f.name: f for f in imgs}
    read_list = lambda k: [byname[Path(l.strip()).name] for l in open(root / 'detection' / f'cond_test_{k}.txt', encoding='utf-8')
                           if l.strip() and Path(l.strip()).name in byname]
    sets = {'night': read_list('night')[:a.max], 'clear200': read_list('clear')[:200][:a.max], 'all_gated': imgs[:a.max]}
    plate = YOLO(a.det)
    rows = {s: list(csv.DictReader(open(root / 'ocr' / f'{s}.csv', encoding='utf-8'))) for s in ('test',)}
    tags = {r['file']: r for r in csv.DictReader(open(root / 'ocr' / 'test_conditions.csv', encoding='utf-8'))}
    crops = [(r, cv2.imread(str(root / 'ocr' / 'images' / 'test' / r['file']))) for r in rows['test'][:a.max]]
    crops = [(r, im) for r, im in crops if im is not None]
    ocr = c = None
    if a.ocr and os.path.exists(a.ocr):
        ocr, c = load_ocr(a.ocr, dev)
    out = {'run': a.name, 'kind': 'lowlight', 'det_weights': a.det, 'ocr_weights': a.ocr, 'gate_mean_grey_below': DARK,
           'detection': {}, 'reading': {}, 'ms': {}, 'params_M': {}, 'skipped': {}}
    tmp = Path(a.tmp)
    for name in a.enhancers:
        try:
            enh = Enhancer(name, dev)
        except Exception as e:
            out['skipped'][name] = f'{e}'[:200]
            print('skip', name, e, flush=True)
            continue
        if enh.net is not None:
            out['params_M'][name] = round(sum(p.numel() for p in enh.net.parameters()) / 1e6, 2)
        out['ms'][name] = {'photo_640': ms_of(enh, (640, 640), dev), 'crop_64x192': ms_of(enh, (64, 192), dev)}
        out['detection'][name] = {}
        for s, files in sets.items():
            y, n, g0, g1 = write_subset(enh, files, lab, tmp / name / s, gate=(s == 'all_gated'))
            d = {'photos': len(files), 'enhanced': n, 'mean_grey_before': g0, 'mean_grey_after': g1,
                 **det.val_metrics(plate, str(y), 'test', 640, '0' if dev.type == 'cuda' else 'cpu', run / name / s),
                 'found_%': det.plate_counts(plate, str(y), 'test', 640, '0' if dev.type == 'cuda' else 'cpu')['found_%']}
            out['detection'][name][s] = d
            shutil.rmtree(tmp / name / s, ignore_errors=True)
        if ocr is not None:
            night = [(r, im) for r, im in crops if tags.get(r['file'], {}).get('night') == '1']
            res = {}
            for s, part, gate in (('night', night, False), ('all_gated', crops, True)):
                ims = [enh(im) if (not gate or grey(im) < DARK) else im for r, im in part]
                preds = read(ocr, c, ims, dev) if ims else []
                res[s] = {**metrics.score_rows(zip(preds, [r['text'] for r, _ in part])), 'crops': len(part),
                          'enhanced': sum(1 for _, im in part if not gate or grey(im) < DARK)}
            out['reading'][name] = res
        print(name, {s: (v['mAP50'], v['found_%']) for s, v in out['detection'][name].items()},
              {s: v.get('plate_exact_%') for s, v in out['reading'].get(name, {}).items()}, out['ms'][name], flush=True)
        if name.startswith('hvi') and sets['night']:                  # a before / after sheet for the report
            sheet = []
            for f in sets['night'][:6]:
                im = cv2.imread(str(f))
                im = cv2.resize(im, (320, int(320 * im.shape[0] / im.shape[1])))
                sheet.append(np.concatenate([im, enh(im)], 1))
            hmax = max(s.shape[0] for s in sheet)
            sheet = [cv2.copyMakeBorder(s, 0, hmax - s.shape[0], 0, 0, cv2.BORDER_CONSTANT) for s in sheet]
            cv2.imwrite(str(run / f'before_after_{name}.jpg'), np.concatenate(sheet, 0))
        del enh
        torch.cuda.empty_cache() if dev.type == 'cuda' else None
    base = out['detection'].get('none', {})
    for name, d in out['detection'].items():
        if name != 'none' and base:
            d['gain_night_mAP50_pp'] = round(100 * (d['night']['mAP50'] - base['night']['mAP50']), 2)
            d['gain_all_gated_mAP50_pp'] = round(100 * (d['all_gated']['mAP50'] - base['all_gated']['mAP50']), 2)
    rb = out['reading'].get('none', {})
    for name, d in out['reading'].items():
        if name != 'none' and rb:
            d['gain_night_exact_pp'] = round(d['night']['plate_exact_%'] - rb['night']['plate_exact_%'], 2)
    out['env'] = common.env_info()
    common.save_json(out, run / 'results.json')
    print('DONE', a.name)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['setup', 'eval'])
    p.add_argument('--dir', default=os.path.join(common.WORK, 'hvi_cidnet'))
    p.add_argument('--det', default='')
    p.add_argument('--ocr', default='')
    p.add_argument('--enhancers', nargs='*', default=['none', 'gamma', 'clahe', 'hvi_gen', 'hvi_lolv2'])
    p.add_argument('--tmp', default='/tmp/p35' if os.path.exists('/tmp') else os.path.join(common.WORK, 'p35_tmp'))
    p.add_argument('--max', type=int, default=None)
    p.add_argument('--name', default='lowlight')
    p.add_argument('--project', default=os.path.join(common.WORK, 'runs'))
    a = p.parse_args()
    setup(a) if a.mode == 'setup' else evaluate(a)
