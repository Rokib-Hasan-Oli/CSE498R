"""Plate readers: every reader takes a list of BGR plate crops and returns one Reading per crop.

Reader kinds (paper of origin in brackets):
  charyolo : character detector, one box per token, BRTA layout decoder (#2 LPNR, #11, #27, #23)
  crnn     : CNN + BiLSTM + CTC over the 101 tokens, two-line plates stacked into one row (#20, #19/#24)
  svtrv2   : SVTRv2 + CTC, multi-size resizing (MSR), run through ONNX Runtime (#34)
  slot     : fast-plate-ocr style CNN with 9 fixed slots (#18)
The token list is the dataset's vocab.json (101 classes). Readers never load third-party pickles: YOLO files are the
team's own Ultralytics checkpoints, the others are plain state_dicts (weights_only=True) or ONNX.
"""
import json
import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn as nn

from . import metrics, vocab

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, '..', 'models')
CLASSES = json.load(open(os.path.join(MODELS, 'vocab.json'), encoding='utf-8'))['classes']


@dataclass
class Reading:
    text: str
    conf: float
    valid: bool
    ms: float = 0.0
    boxes: list = field(default_factory=list)      # charyolo only: [token, x1, y1, x2, y2, conf]


def weight(name, wdir=None):
    return os.path.join(wdir or os.path.join(MODELS, 'weights'), name)


def pick_device(device=None):
    if device:
        return device
    return 'cuda:0' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------------------------------- char-YOLO
class CharYOLOReader:
    kind = 'charyolo'

    def __init__(self, file, imgsz=320, conf=0.10, min_conf=0.25, device=None, wdir=None):
        from ultralytics import YOLO
        self.model = YOLO(weight(file, wdir))
        self.imgsz, self.conf, self.min_conf = imgsz, conf, min_conf
        self.device = pick_device(device)
        self.names = self.model.names

    def read(self, crops):
        if not crops:
            return []
        t0 = time.perf_counter()
        # rect=False: square 320 input, as in the kit (a letterboxed rectangle reads a few plates differently)
        res = self.model.predict(crops, imgsz=self.imgsz, conf=self.conf, device=self.device, rect=False, verbose=False)
        ms = 1000 * (time.perf_counter() - t0) / len(crops)
        out = []
        for r in res:
            b = r.boxes
            keep = [(self.names[int(c)], bb.tolist(), float(s)) for bb, s, c in
                    zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()) if s >= self.min_conf]
            text, ok = vocab.plate_text([[n] + bb for n, bb, _ in keep])
            mc = float(np.mean([s for _, _, s in keep])) if keep else 0.0
            out.append(Reading(text, mc, ok, ms, [[n] + bb + [s] for n, bb, s in keep]))
        return out


# ---------------------------------------------------------------------------------------------------- CRNN (CTC)
class CRNN(nn.Module):
    """lprkit/seq.py CRNN: VGG-style CNN -> height collapsed -> 2-layer BiLSTM -> per-column classes (0 = blank)."""
    def __init__(self, n_cls, H=48):
        super().__init__()

        def blk(i, o, pool):
            return [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True)] + ([nn.MaxPool2d(pool)] if pool else [])
        self.cnn = nn.Sequential(*blk(1, 64, (2, 2)), *blk(64, 128, (2, 2)), *blk(128, 256, None),
                                 *blk(256, 256, (2, 1)), *blk(256, 384, None), *blk(384, 384, (2, 1)))
        self.rnn = nn.LSTM(384 * (H // 16), 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(512, n_cls + 1)

    def forward(self, x):
        f = self.cnn(x)
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)
        return self.fc(self.rnn(f)[0])


class SlotCNN(nn.Module):
    """lprkit/seq.py SlotCNN (papers #18, #24): 64x128 grey crop -> 9 fixed slots, 0 = empty."""
    def __init__(self, n_cls, L=9):
        super().__init__()

        def blk(i, o):
            return [nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(True)]
        self.f = nn.Sequential(*blk(1, 32), nn.MaxPool2d(2), *blk(32, 64), nn.MaxPool2d(2), *blk(64, 128), *blk(128, 128),
                               nn.MaxPool2d(2), *blk(128, 256), *blk(256, 256), nn.MaxPool2d(2), nn.Conv2d(256, 512, 1),
                               nn.ReLU(True))
        self.L, self.n = L, n_cls + 1
        self.head = nn.Conv2d(512, L * (n_cls + 1), 1)

    def forward(self, x):
        return self.head(self.f(x)).amax(dim=(2, 3)).view(x.shape[0], self.L, self.n)


def seq_prep(img, H, W, stack):
    """lprkit/seq.py prep (eval): grey, two-line plates cut and placed side by side, aspect-kept resize, pad 127."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape
    if stack and w / max(1, h) < 2.6:
        top, bot = g[: h // 2], g[h // 2:]
        g = np.concatenate([top, cv2.resize(bot, (w, top.shape[0]))], 1)
        h, w = g.shape
    s = min(H / h, W / w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    canvas = np.full((H, W), 127, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(canvas).float().div(255).sub(0.5).div(0.5)[None]


def ctc_greedy(probs, classes):
    """probs: (B, T, C) softmax. Returns [(text, conf)], conf = mean max-prob of the kept tokens (kit seq.py)."""
    best, pmax = probs.argmax(-1), probs.max(-1)
    out = []
    for s, p in zip(best, pmax):
        toks, cs, prev = [], [], 0
        for c, q in zip(s, p):
            if c != 0 and c != prev:
                toks.append(int(c)); cs.append(float(q))
            prev = c
        out.append((vocab.tokens_to_text([classes[k - 1] for k in toks if 0 < k <= len(classes)]),
                    float(np.mean(cs)) if cs else 0.0))
    return out


class TorchSeqReader:
    """CRNN or slot model from a plain state_dict."""

    def __init__(self, file, kind='crnn', H=48, W=256, stack=True, device=None, wdir=None):
        self.kind = kind
        ck = torch.load(weight(file, wdir), map_location='cpu', weights_only=True)
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        self.classes = ck.get('classes', CLASSES)
        self.H, self.W, self.stack = ck.get('H', H), ck.get('W', W), ck.get('stack', stack)
        self.model = CRNN(len(self.classes), self.H) if kind == 'crnn' else SlotCNN(len(self.classes))
        self.model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        self.device = torch.device(pick_device(device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def read(self, crops):
        if not crops:
            return []
        t0 = time.perf_counter()
        x = torch.stack([seq_prep(c, self.H, self.W, self.stack) for c in crops]).to(self.device)
        probs = self.model(x).float().softmax(-1).cpu().numpy()
        if self.kind == 'crnn':
            dec = ctc_greedy(probs, self.classes)
        else:                                            # fixed slots: stop at the first empty slot
            dec = []
            for s, p in zip(probs.argmax(-1), probs.max(-1)):
                toks, cs = [], []
                for c, q in zip(s, p):
                    if c == 0:
                        break
                    toks.append(int(c)); cs.append(float(q))
                dec.append((vocab.tokens_to_text([self.classes[k - 1] for k in toks]), float(np.mean(cs)) if cs else 0.0))
        ms = 1000 * (time.perf_counter() - t0) / len(crops)
        return [Reading(t, c, metrics.is_valid_plate(t), ms) for t, c in dec]


# ---------------------------------------------------------------------------------------------------- SVTRv2 (ONNX)
MSR = [(64, 64), (48, 96), (40, 112), (32, 128)]


def msr_size(h, w):
    r = w / max(1, h)
    return MSR[0 if r < 1.5 else 1 if r < 2.5 else 2 if r < 3.5 else 3]


class SVTRv2Reader:
    kind = 'svtrv2'

    def __init__(self, file, threads=4, wdir=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        prov = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in ort.get_available_providers()]
        self.sess = ort.InferenceSession(weight(file, wdir), so, providers=prov)
        self.classes = CLASSES

    def read(self, crops):
        from PIL import Image
        out = []
        for c in crops:                                  # one crop at a time: MSR sizes differ per crop
            t0 = time.perf_counter()
            H, W = msr_size(*c.shape[:2])
            x = Image.fromarray(np.ascontiguousarray(c[:, :, ::-1])).resize((W, H), Image.BICUBIC)
            x = (np.asarray(x, np.float32).transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
            probs = self.sess.run(None, {'x': x[None]})[0]      # RCTC head returns softmax
            best, pmax = probs.argmax(-1)[0], probs.max(-1)[0]
            toks, cs, prev = [], [], 0
            for k, q in zip(best, pmax):
                if k != 0 and k != prev:
                    toks.append(int(k)); cs.append(float(q))
                prev = k
            text = vocab.tokens_to_text([self.classes[k - 1] for k in toks])
            conf = float(np.prod(cs)) if cs else 0.0            # paper #34 kit: product of token probabilities
            out.append(Reading(text, conf, metrics.is_valid_plate(text), 1000 * (time.perf_counter() - t0)))
        return out


# ---------------------------------------------------------------------------------------------------- character verifier
class PaperCNN(nn.Module):
    """Paper #8 (Afrin et al.) CNN, 28x28 RGB character crop -> token class (kit p08_train.PaperCNN)."""
    def __init__(self, n_classes, imgsz=28):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 22, padding='same'), nn.ReLU(),
                                      nn.Conv2d(16, 32, 16, padding='same'), nn.ReLU(),
                                      nn.Conv2d(32, 64, 8, padding='same'), nn.ReLU(),
                                      nn.Conv2d(64, 128, 4, padding='same'), nn.ReLU(), nn.MaxPool2d(4))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * (imgsz // 4) ** 2, 512), nn.ReLU(), nn.Dropout(0.4),
                                  nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class BLPnetCharCNN(nn.Module):
    """Paper #10 (BLPnet) character CNN as trained in the team's paper10.ipynb ('guide' variant, 32x32 RGB):
    5 x [ZeroPad (0,1,0,1), Conv 2x2, ReLU, MaxPool 2, Dropout] -> global average pool -> 256 -> classes."""
    def __init__(self, n_classes, size=32):
        super().__init__()
        layers, c, spatial = [], 3, size
        for width in (16, 32, 64, 128, 512):
            layers += [nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(c, width, 2), nn.ReLU()]
            if spatial >= 2:
                layers.append(nn.MaxPool2d(2)); spatial //= 2
            layers.append(nn.Dropout(.2))
            c = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256), nn.ReLU(), nn.Dropout(.2),
                                  nn.Linear(256, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class CharVerifier:
    """Second opinion on every character box of a char-YOLO reading. Returns the share of boxes whose CNN class equals
    the YOLO class: a cheap agreement signal used in the status level, never to change the text.
      'p08': paper #8 CNN, 28x28 RGB, boxes padded 4 % (kit p08_chars.py), classes = vocab.json order
      'p10': paper #10 BLPnet CNN, 32x32 RGB, no padding, PIL resize (paper10.ipynb), its own class_names.json"""

    def __init__(self, kind='p08', device=None, wdir=None):
        self.kind = kind
        if kind == 'p08':
            self.classes, self.size, self.pad = CLASSES, 28, 0.04
            self.model = PaperCNN(len(self.classes))
            self.model.load_state_dict(torch.load(weight('p08_cnn28.pt', wdir), map_location='cpu', weights_only=True))
        else:
            self.classes = json.load(open(weight('p10_class_names.json', wdir), encoding='utf-8'))
            self.size, self.pad = 32, 0.0
            self.model = BLPnetCharCNN(len(self.classes))
            self.model.load_state_dict(torch.load(weight('p10_char_cnn32.pt', wdir), map_location='cpu', weights_only=True))
        self.device = torch.device(pick_device(device))
        self.model.to(self.device).eval()

    @torch.no_grad()
    def agreement(self, crop, boxes):
        if not boxes:
            return 0.0
        from PIL import Image
        xs, want = [], []
        h, w = crop.shape[:2]
        for tok, x1, y1, x2, y2, _ in boxes:
            pw, ph = self.pad * (x2 - x1), self.pad * (y2 - y1)
            c = crop[max(0, int(y1 - ph)):min(h, int(np.ceil(y2 + ph))), max(0, int(x1 - pw)):min(w, int(np.ceil(x2 + pw)))]
            if c.size == 0:
                continue
            rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            if self.kind == 'p08':
                rgb = cv2.resize(rgb, (28, 28), interpolation=cv2.INTER_LINEAR)
            else:
                rgb = np.asarray(Image.fromarray(rgb).resize((32, 32)))
            xs.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0)
            want.append(tok)
        if not xs:
            return 0.0
        pred = self.model(torch.stack(xs).to(self.device)).argmax(1).cpu().numpy()
        return float(np.mean([vocab.N(self.classes[p]) == vocab.N(t) for p, t in zip(pred, want)]))


# ---------------------------------------------------------------------------------------------------- factory
def build_reader(spec, device=None, wdir=None):
    """spec: {'name', 'kind', 'file', ...} from a version config."""
    k = spec['kind']
    if k == 'charyolo':
        return CharYOLOReader(spec['file'], spec.get('imgsz', 320), device=device, wdir=wdir)
    if k in ('crnn', 'slot'):
        return TorchSeqReader(spec['file'], k, spec.get('H', 48 if k == 'crnn' else 64),
                              spec.get('W', 256 if k == 'crnn' else 128), spec.get('stack', k == 'crnn'), device, wdir)
    if k == 'svtrv2':
        return SVTRv2Reader(spec['file'], wdir=wdir)
    raise ValueError(k)
