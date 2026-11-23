"""One-off: export the paper #34 SVTRv2 readers (T and S, MSR) to ONNX so the system needs no OpenOCR code at run time.

  python tools/export_svtrv2_onnx.py <OpenOCR clone at commit 1ccfc6e> <kit extras dir>

Writes models/weights/p34_svtrv2_{t,s}.onnx (dynamic batch / height / width) and checks the ONNX output against
PyTorch on random MSR-sized inputs.
"""
import os
import sys

import numpy as np
import torch

OPENOCR, KIT = sys.argv[1], sys.argv[2]
sys.path[:0] = [OPENOCR, KIT, os.path.join(os.path.dirname(KIT), 'lprkit')]
import p34_svtrv2 as P  # noqa: E402

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'weights')
for src, dst in (('p34_svtrv2_t_msr.pt', 'p34_svtrv2_t.onnx'), ('p34_svtrv2_s_msr_fp16.pt', 'p34_svtrv2_s.onnx')):
    ck = torch.load(os.path.join(W, src), map_location='cpu', weights_only=True)
    m = P.build(ck['size'], len(ck['classes']) + 1)
    m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck['state_dict'].items()})
    m.eval()
    out = os.path.join(W, dst)
    torch.onnx.export(m, torch.randn(1, 3, 48, 96), out, input_names=['x'], output_names=['probs'], opset_version=17,
                      dynamic_axes={'x': {0: 'b', 2: 'h', 3: 'w'}, 'probs': {0: 'b', 1: 't'}}, dynamo=False)
    import onnxruntime as ort
    s = ort.InferenceSession(out, providers=['CPUExecutionProvider'])
    worst = 0.0
    for H, Wd in P.BUCKETS:
        x = torch.randn(2, 3, H, Wd)
        with torch.no_grad():
            ref = m(x).numpy()
        got = s.run(None, {'x': x.numpy()})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    print(dst, 'max |onnx - torch| =', worst, 'size MB', round(os.path.getsize(out) / 2 ** 20, 1))
