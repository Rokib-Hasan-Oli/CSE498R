"""The BanglaLPR system: whole photo or video frame -> plates with Bangla text, English text, status and timings.

  from lpr.pipeline import LPRSystem
  lpr = LPRSystem('v2')                       # or 'v1'; reads models/<version>/config.json
  out = lpr.process(cv2.imread('car.jpg'))    # dict: plates [...], ms {...}

Stages (paper of origin in the version configs):
  detect   : one YOLO plate detector; optional low-light retry (CLAHE) when nothing is found (V2, paper #35 study)
  crop     : plate box + 8 % padding (paper #2 cascade)
  sr       : optional x2 super-resolution for tiny crops (paper #20)
  read     : primary reader on every crop; helpers per the fusion rule (lpr.fusion)
  status   : verified / likely / review (lpr.fusion.status; paper #19 study)
"""
import json
import os
import threading
import time

import cv2
import numpy as np
import torch

from . import fusion, metrics, vocab
from .readers import MODELS, CharVerifier, build_reader, pick_device, weight


def to_english(text):
    """Bangla plate text -> BRTA English form, e.g. 'Dhaka Metro-Ga 12-3456'."""
    if not text:
        return ''
    t = vocab.N(text).translate(str.maketrans(''.join(vocab.DIGITS), '0123456789'))
    for k in sorted(vocab.ENGLISH, key=len, reverse=True):
        if not k.isdigit() and k in t:
            t = t.replace(k, vocab.ENGLISH[k])
    return t


class PlateDetector:
    def __init__(self, file, imgsz=640, conf=0.25, device=None, wdir=None):
        from ultralytics import YOLO
        self.model = YOLO(weight(file, wdir))
        self.imgsz, self.conf, self.device = imgsz, conf, pick_device(device)

    def detect(self, img, conf=None):
        r = self.model.predict(img, imgsz=self.imgsz, conf=conf or self.conf, device=self.device, verbose=False)[0]
        return sorted(((*b.tolist(), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())),
                      key=lambda x: -x[4])


def clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def crop(img, box, pad=0.08):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box[:4]
    pw, ph = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
    x2, y2 = int(min(w, x2 + pw)), int(min(h, y2 + ph))
    return img[y1:y2, x1:x2]


class SR:
    """Paper #20 SR network (x2, RCAB body + bicubic skip), used only on crops lower than min_h pixels."""

    def __init__(self, file='p20_sr_x2.pt', min_h=32, device=None, wdir=None):
        import torch.nn as nn
        import torch.nn.functional as F

        class RCAB(nn.Module):
            def __init__(self, c, r=16):
                super().__init__()
                self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1))
                self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // r, 1), nn.ReLU(True),
                                        nn.Conv2d(c // r, c, 1), nn.Sigmoid())

            def forward(self, x):
                y = self.b(x)
                return x + y * self.ca(y)

        class SRNet(nn.Module):
            def __init__(self, scale=2, c=64, n=10):
                super().__init__()
                self.scale = scale
                self.head = nn.Conv2d(3, c, 3, 1, 1)
                self.body = nn.Sequential(*[RCAB(c) for _ in range(n)], nn.Conv2d(c, c, 3, 1, 1))
                self.up = nn.Sequential(nn.Conv2d(c, 3 * scale * scale, 3, 1, 1), nn.PixelShuffle(scale))

            def forward(self, x):
                f = self.head(x)
                return self.up(f + self.body(f)) + F.interpolate(x, scale_factor=self.scale, mode='bicubic',
                                                                 align_corners=False)
        ck = torch.load(weight(file, wdir), map_location='cpu', weights_only=True)
        self.net = SRNet(ck.get('scale', 2))
        self.net.load_state_dict(ck['state_dict'])
        self.device = torch.device(pick_device(device))
        self.net.to(self.device).eval()
        self.min_h = min_h

    @torch.no_grad()
    def __call__(self, c):
        if c.shape[0] >= self.min_h:
            return c, False
        x = torch.from_numpy(c[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)[None].to(self.device)
        y = self.net(x)[0].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        return np.ascontiguousarray(y), True


class LPRSystem:
    _lock = threading.Lock()                       # one GPU job at a time (web app threads share the models)

    def __init__(self, version='v1', device=None):
        """version: 'v1' / 'v2' (models/<version>/config.json + models/weights) or the path of a model folder
        (V1/ or V2/ package: <folder>/models/config.json + <folder>/models/weights)."""
        pkg = os.path.join(version, 'models', 'config.json')
        if os.path.exists(pkg):
            cfg_path, wdir = pkg, os.path.join(version, 'models', 'weights')
        else:
            cfg_path, wdir = os.path.join(MODELS, version, 'config.json'), None
        self.cfg = json.load(open(cfg_path, encoding='utf-8'))
        self.version = self.cfg.get('version', version)
        self.wdir = wdir
        d = self.cfg['detector']
        self.detector = PlateDetector(d['file'], d['imgsz'], d.get('conf', 0.25), device, wdir)
        self.reader_specs = self.cfg['readers']
        self.readers = [build_reader(s, device, wdir) for s in self.reader_specs]
        self.names = [s['name'] for s in self.reader_specs]
        self.rule = self.cfg['rule']
        st = self.cfg.get('status_reader')
        self.status_reader = build_reader(st, device, wdir) if st else None
        self.char_verifiers = [CharVerifier(k, device, wdir) for k in self.cfg.get('char_verifiers', [])]
        self.char_min = self.cfg.get('char_agreement_min', 0.8)
        up = self.cfg.get('tiny_upscale') or {}
        self.up_h = up.get('min_h', 0)
        self.sr = SR(min_h=self.up_h, device=device, wdir=wdir) if up.get('method') == 'sr' else None
        # retry chain when nothing is found: [{file, imgsz, conf, clahe}] (V2: paper #6 detector, CLAHE from #35)
        self.retry, dets = [], {d['file']: self.detector}
        for r in self.cfg.get('retry', []):
            if r['file'] not in dets:
                dets[r['file']] = PlateDetector(r['file'], r['imgsz'], r.get('conf', 0.25), device, wdir)
            self.retry.append((dets[r['file']], r))

    # ------------------------------------------------------------------------------------------------ reading
    def read_crops(self, crops):
        """Fused reading of each crop: list of dicts (text, status, conf, readers, ms)."""
        if not crops:
            return []
        t0 = time.perf_counter()
        n = len(crops)
        runs = [[None] * len(self.readers) for _ in range(n)]
        for i, r in enumerate(self.readers[0].read(crops)):
            runs[i][0] = r
        k0 = 2 if self.rule == 'verify' else 1
        if k0 == 2 and len(self.readers) > 1:
            for i, r in enumerate(self.readers[1].read(crops)):
                runs[i][1] = r
        need = [i for i in range(n) if len(self.readers) > k0 and
                fusion.needs_helpers(self.rule, runs[i][0].text, runs[i][1].text if k0 == 2 else None)]
        for j in range(k0, len(self.readers)):
            if need:
                for i, r in zip(need, self.readers[j].read([crops[i] for i in need])):
                    runs[i][j] = r
        stat = self.status_reader.read(crops) if self.status_reader else [None] * n
        out = []
        for i in range(n):
            rd = [(nm, r.text, r.conf) if r else None for nm, r in zip(self.names, runs[i])]
            text = fusion.fuse(rd, self.rule if len(self.readers) > 1 else 'single')
            ran = [x for x in rd if x]
            agree = None
            if self.char_verifiers and runs[i][0].boxes:
                agree = round(float(np.mean([v.agreement(crops[i], runs[i][0].boxes) for v in self.char_verifiers])), 3)
            s = fusion.status(text, ran, stat[i].text if stat[i] else None, agree, self.char_min)
            out.append({'text': text, 'text_en': to_english(text), 'valid': metrics.is_valid_plate(text), 'status': s,
                        'conf': round(float(runs[i][0].conf), 3),
                        'readers': {nm: r.text for nm, r in zip(self.names, runs[i]) if r},
                        'status_reader': stat[i].text if stat[i] else None, 'char_agreement': agree})
        ms = 1000 * (time.perf_counter() - t0)
        for o in out:
            o['read_ms'] = round(ms / n, 2)
        return out

    # ------------------------------------------------------------------------------------------------ photo
    def process(self, img, max_plates=5):
        with self._lock:
            t0 = time.perf_counter()
            boxes = self.detector.detect(img)
            retried = None
            for det, r in self.retry:
                if boxes:
                    break
                boxes = det.detect(clahe(img) if r.get('clahe') else img, conf=r.get('conf'))
                retried = r['file'] + (' + CLAHE' if r.get('clahe') else '')
            t1 = time.perf_counter()
            boxes = [b for b in boxes[:max_plates] if crop(img, b).size]
            crops, sr_used = [], []
            for b in boxes:
                c = crop(img, b)
                used = False
                if self.up_h and c.shape[0] < self.up_h:        # tiny plate: x2 before reading (paper #20 study)
                    c = self.sr(c)[0] if self.sr else cv2.resize(c, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    used = True
                crops.append(c)
                sr_used.append(used)
            reads = self.read_crops(crops)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
        plates = []
        for b, c, u, r in zip(boxes, crops, sr_used, reads):
            plates.append({'box': [round(float(v), 1) for v in b[:4]], 'det_conf': round(float(b[4]), 3),
                           'upscaled': u, **r, '_crop': c})
        return {'version': self.version, 'plates': plates, 'retry_used': retried,
                'ms': {'detect': round(1000 * (t1 - t0), 2), 'read': round(1000 * (t2 - t1), 2),
                       'total': round(1000 * (t2 - t0), 2)}}


def draw(img, result, color=(36, 170, 90)):
    """Boxes + English text on a copy of the photo (Bangla glyphs need a font; the web page shows the Bangla text)."""
    out = img.copy()
    cols = {'verified': (60, 170, 60), 'likely': (0, 165, 255), 'review': (40, 40, 220)}
    for i, p in enumerate(result['plates']):
        x1, y1, x2, y2 = map(int, p['box'])
        c = cols.get(p['status'], color)
        th = max(2, int(round(out.shape[1] / 500)))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, th)
        label = f"{i + 1}: {p['text_en'] or '?'}"
        fs = max(0.5, out.shape[1] / 1400)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        y0 = max(tht + 6, y1 - 4)
        lx = max(0, min(x1, out.shape[1] - tw - 8))          # keep the label inside the photo
        cv2.rectangle(out, (lx, y0 - tht - 6), (lx + tw + 6, y0 + 2), c, -1)
        cv2.putText(out, label, (lx + 3, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out
