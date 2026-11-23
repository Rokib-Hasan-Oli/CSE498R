"""Export every model used by V1 and V2 to ONNX for the Android apps (ONNX Runtime on the phone).

  python tools/export_onnx_mobile.py

Writes BanglaLPR_System/android/onnx/<name>.onnx and checks each ONNX output against PyTorch on random input.
- plate detectors: dynamic height/width (the app letterboxes like Ultralytics predict: shortest side padded to a
  multiple of 32), output (1, 5, N) = cx, cy, w, h, score
- char-YOLO readers: fixed square input (the kit reads with rect=False), output (1, 4 + 101, N)
- CRNN / slot OCR: fixed grey input (48x256 / 64x128), raw logits
- character CNNs: batch dynamic, RGB 28x28 / 32x32, raw logits
- SVTRv2: the existing dynamic ONNX files are copied
"""
import os
import shutil
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)
from lpr import readers as R  # noqa: E402

W = os.path.join(ROOT, 'models', 'weights')
OUT = os.path.join(ROOT, 'android', 'onnx')
os.makedirs(OUT, exist_ok=True)


def check(path, feed, ref):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    got = s.run(None, {s.get_inputs()[0].name: feed})[0]
    d = float(np.abs(got - ref).max())
    print(f'  {os.path.basename(path)}: out {got.shape}, max |onnx - torch| = {d:.2e}, {os.path.getsize(path) / 2**20:.1f} MB', flush=True)
    return d


def export(model, x, name, dyn=None):
    path = os.path.join(OUT, name)
    model.eval()
    torch.onnx.export(model, x, path, input_names=['x'], output_names=['y'], opset_version=17,
                      dynamic_axes=dyn or {}, dynamo=False)
    with torch.no_grad():
        ref = model(x).float().numpy()
    check(path, x.numpy(), ref)


def yolo(file, imgsz, dynamic):
    """Ultralytics model -> ONNX through its own exporter wrapper (same graph as model.predict)."""
    from ultralytics import YOLO
    m = YOLO(os.path.join(W, file))
    net = m.model.float().eval()
    for p in net.parameters():
        p.requires_grad_(False)
    det = net.model[-1]
    det.export, det.format = True, 'onnx'              # export mode: one output tensor (1, 4 + nc, N)
    if hasattr(det, 'dynamic'):
        det.dynamic = dynamic
    x = torch.rand(1, 3, imgsz, imgsz)
    name = file.replace('.pt', '.onnx')
    dyn = {'x': {2: 'h', 3: 'w'}, 'y': {2: 'n'}} if dynamic else None
    class Wrap(torch.nn.Module):
        def __init__(self, n):
            super().__init__(); self.n = n
        def forward(self, t):
            y = self.n(t)
            return y[0] if isinstance(y, (list, tuple)) else y
    export(Wrap(net), x, name, dyn)
    return name


if __name__ == '__main__':
    print('detectors (dynamic size)')
    yolo('p18_v8n_576.pt', 576, True)
    yolo('p06_det_v8s.pt', 640, True)
    print('char-YOLO readers (square)')
    yolo('p11_chars_v8n.pt', 320, False)
    yolo('p27_chars_v8n.pt', 320, False)
    yolo('p02_lpnr_v12n.pt', 640, False)
    print('CRNN / slot')
    crnn = R.TorchSeqReader('p20_crnn_hr.pt', 'crnn', 48, 256, True, 'cpu').model
    export(crnn, torch.randn(1, 1, 48, 256), 'p20_crnn_hr.onnx', {'x': {0: 'b'}, 'y': {0: 'b'}})
    slot = R.TorchSeqReader('p18_slot.pt', 'slot', 64, 128, False, 'cpu').model
    export(slot, torch.randn(1, 1, 64, 128), 'p18_slot.onnx', {'x': {0: 'b'}, 'y': {0: 'b'}})
    print('character CNNs')
    export(R.CharVerifier('p08', 'cpu').model, torch.rand(2, 3, 28, 28), 'p08_cnn28.onnx', {'x': {0: 'b'}, 'y': {0: 'b'}})
    export(R.CharVerifier('p10', 'cpu').model, torch.rand(2, 3, 32, 32), 'p10_char_cnn32.onnx', {'x': {0: 'b'}, 'y': {0: 'b'}})
    for f in ('p34_svtrv2_s.onnx', 'p34_svtrv2_t.onnx', 'p10_class_names.json'):
        shutil.copy2(os.path.join(W, f), os.path.join(OUT, f))
    shutil.copy2(os.path.join(ROOT, 'models', 'vocab.json'), os.path.join(OUT, 'vocab.json'))
    print('done ->', OUT)
