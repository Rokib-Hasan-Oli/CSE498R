"""Paper #34 local re-check (RTX 5060 Ti, dataset v1.1 in D:/498R/dataset) of the v1.1 Kaggle run.

  D:/498R/_dataset_build/venv/Scripts/python.exe local_check.py <unpacked results.zip> [score] [timing] [onnx]

score  : loads every run's best.pt (our own state_dicts, torch.load weights_only=True) into the authors' model code
         (the run's own openocr/ copy, commit 1ccfc6e) and the run's own lprkit code, reads the 651 test and 766 val
         crops again and compares plate by plate with the Kaggle CSVs. Also writes the FP16 copies for GitHub.
timing : batch-1 model time per MSR size (FP32 / FP16, GPU and 4-thread CPU), batch-64 throughput, and the whole
         reader on one plate (resize + model + CTC decode), median of 200 after 30 warm-up calls. Nothing else may
         use the GPU while this runs (the Kaggle numbers were taken while the other T4 was training).
onnx   : ONNX export of SVTRv2-T (one file, dynamic height / width), max difference to PyTorch on the test crops,
         onnxruntime CPU time per plate.
The authors' English checkpoints are NOT loaded here (third-party pickles; they are only loaded on Kaggle).
"""
import csv
import hashlib
import json
import os
import sys
import time

RUN = os.path.abspath(sys.argv[1])
TASKS = set(sys.argv[2:]) or {'score', 'timing', 'onnx'}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'local_check.json')
os.environ.setdefault('DATASET_ROOT', 'D:/498R/dataset')
os.environ['LPR_NO_DOWNLOAD'] = '1'
os.environ['LPR_WORKERS'] = '0'
os.environ['LPR_WORK'] = os.path.join(RUN, '_local')
os.environ['OPENOCR_DIR'] = os.path.join(RUN, 'openocr')
sys.path.insert(0, os.path.join(RUN, 'code'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import p34_svtrv2 as P  # noqa: E402
import seq  # noqa: E402
import metrics  # noqa: E402
import vocab  # noqa: E402

P.openocr_path()
dev = torch.device('cuda')
cls = seq.class_list()
SV = {'svtrv2_t_msr': ('T', 'msr'), 'svtrv2_s_msr': ('S', 'msr'), 'svtrv2_t_fixed': ('T', 'fixed'),
      'svtrv2_t_scratch': ('T', 'msr')}
res = json.load(open(OUT, encoding='utf-8')) if os.path.exists(OUT) else {}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def kaggle(run, split):
    with open(os.path.join(RUN, 'runs', run, f'ocr_preds_{split}.csv'), encoding='utf-8') as f:
        return {r['file']: r for r in csv.DictReader(f)}


def svtr(run, half=False, device=dev):
    size, _ = SV[run]
    m = P.build(size, len(cls) + 1)
    m.load_state_dict(torch.load(os.path.join(RUN, 'runs', run, 'best.pt'), map_location='cpu', weights_only=True))
    m = m.eval().to(device)
    return m.half() if half else m


def crnn(half=False, device=dev):
    full = torch.load(os.path.join(RUN, 'runs/crnn/model_full.pt'), map_location='cpu', weights_only=True)
    m = seq.CRNN(len(full['classes']), full['H'])
    m.load_state_dict(full['state_dict'])
    m = m.eval().to(device)
    return (m.half() if half else m), full


class A:   # seq.predict's settings object
    H, W, stack = 48, 256, True


same = lambda a, b: vocab.norm_text(a or '') == vocab.norm_text(b or '')

if 'score' in TASKS:
    S = {}
    rows = {s: seq.load_rows(s) for s in ('test', 'val')}
    for run in list(SV) + ['crnn']:
        S[run] = {}
        for split in ('test', 'val'):
            K = kaggle(run, split)
            if run == 'crnn':
                m, _ = crnn()
                preds, _ = seq.predict(m, rows[split], cls, 'crnn', A, dev)
            else:
                m = svtr(run)
                preds, _ = P.predict(m, rows[split], cls, SV[run][1], dev)
            d = [abs(float(p['conf']) - float(K[p['file']]['conf'])) for p in preds]
            S[run][split] = {
                'plates': len(preds),
                'local_exact_%': round(100 * sum(same(p['pred'], p['true']) for p in preds) / len(preds), 2),
                'kaggle_exact_%': round(100 * sum(same(K[p['file']]['pred'], p['true']) for p in preds) / len(preds), 2),
                'same_text_as_kaggle': sum(same(p['pred'], K[p['file']]['pred']) for p in preds),
                'conf_max_abs_diff': round(max(d), 4),
                'bucket_same': sum(p.get('bucket', K[p['file']].get('bucket')) == K[p['file']].get('bucket', p.get('bucket'))
                                   for p in preds)}
            if split == 'test':
                with open(os.path.join(HERE, f'local_preds_{run}.csv'), 'w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=list(preds[0]))
                    w.writeheader(); w.writerows(preds)
            print(run, split, S[run][split], flush=True)
            del m
            torch.cuda.empty_cache()
    # FP16 copies for GitHub (25 MB web limit): same text as FP32 on every test / val crop?
    up = os.path.join(RUN, '_fp16')
    os.makedirs(up, exist_ok=True)
    F16 = {}
    for run in ('svtrv2_t_msr', 'svtrv2_s_msr'):
        sd = torch.load(os.path.join(RUN, 'runs', run, 'best.pt'), map_location='cpu', weights_only=True)
        sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
        f = os.path.join(up, f'best_{run}_fp16.pt')
        size, resize = SV[run]
        torch.save({'state_dict': sd16, 'kind': f'svtrv2-{size}', 'size': size, 'resize': resize, 'classes': cls,
                    'openocr_commit': P.COMMIT, 'encoder': P.ENC[size], 'mixer': P.MIXER[size],
                    'source': f'paper #34 v1.1 Kaggle run 26 Oct 2026, runs/{run}/best.pt (best validation loss)'}, f)
        m32, m16 = svtr(run), P.build(size, len(cls) + 1)
        m16.load_state_dict({k: v.float() for k, v in sd16.items()})
        m16 = m16.eval().to(dev)
        out = {}
        for split in ('test', 'val'):
            a, _ = P.predict(m32, rows[split], cls, resize, dev)
            b, _ = P.predict(m16, rows[split], cls, resize, dev)
            out[split] = {'same_text': sum(same(x['pred'], y['pred']) for x, y in zip(a, b)), 'plates': len(a),
                          'exact_fp16_%': round(100 * sum(same(y['pred'], y['true']) for y in b) / len(b), 2)}
        F16[run] = {'file': os.path.basename(f), 'MiB': round(os.path.getsize(f) / 2 ** 20, 2), 'sha256': sha(f),
                    'fp32_MiB': round(os.path.getsize(os.path.join(RUN, 'runs', run, 'best.pt')) / 2 ** 20, 2), **out}
        print('fp16', run, F16[run], flush=True)
    res['score'], res['fp16'] = S, F16
    res['sha256'] = {run: sha(os.path.join(RUN, 'runs', run, 'best.pt')) for run in SV}
    res['sha256']['crnn_model_full'] = sha(os.path.join(RUN, 'runs/crnn/model_full.pt'))
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


# LOCAL_TIME_P10=1: another job shares the GPU / CPU, so report the 10th percentile of 3x more calls (the calls that
# did not wait for the other job) instead of the median; the statistic is written into local_check.json
P10 = os.environ.get('LOCAL_TIME_P10') == '1'


def t_ms(fn, n=200, warm=30, cuda=True):
    n = 3 * n if P10 else n
    for _ in range(warm):
        fn()
    ms = []
    for _ in range(n):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ms.append(1000 * (time.perf_counter() - t0))
    return round(float(np.percentile(ms, 10) if P10 else np.median(ms)), 3)


if 'timing' in TASKS:
    T = {'gpu': torch.cuda.get_device_name(0), 'cpu_threads': 4, 'torch': torch.__version__,
         'statistic': '10th percentile of 600 calls (another training job shared the GPU)' if P10 else 'median of 200 calls'}
    shapes = {'64x64': (64, 64), '48x96': (48, 96), '40x112': (40, 112), '32x128': (32, 128)}
    with torch.inference_mode():
        for run, size in (('svtrv2_t_msr', 'T'), ('svtrv2_s_msr', 'S')):
            T[size] = {}
            for half in (False, True):
                m = svtr(run, half)
                for k, (h, w) in shapes.items():
                    x = torch.randn(1, 3, h, w, device=dev, dtype=torch.half if half else torch.float)
                    T[size].setdefault(k, {})['gpu_fp16' if half else 'gpu_fp32'] = t_ms(lambda: m(x))
                    if not half:
                        xb = torch.randn(64, 3, h, w, device=dev)
                        T[size][k]['gpu_fp32_batch64_ms_per_plate'] = round(t_ms(lambda: m(xb), 50, 10) / 64, 4)
            torch.set_num_threads(4)
            m = svtr(run, device=torch.device('cpu'))
            for k, (h, w) in shapes.items():
                x = torch.randn(1, 3, h, w)
                T[size][k]['cpu_fp32'] = t_ms(lambda: m(x), 100, 10, cuda=False)
            print(size, T[size], flush=True)
        T['crnn'] = {}
        for half in (False, True):
            m, _ = crnn(half)
            x = torch.randn(1, 1, 48, 256, device=dev, dtype=torch.half if half else torch.float)
            T['crnn']['gpu_fp16' if half else 'gpu_fp32'] = t_ms(lambda: m(x))
        xb = torch.randn(64, 1, 48, 256, device=dev)
        m, _ = crnn()
        T['crnn']['gpu_fp32_batch64_ms_per_plate'] = round(t_ms(lambda: m(xb), 50, 10) / 64, 4)
        m, _ = crnn(device=torch.device('cpu'))
        T['crnn']['cpu_fp32'] = t_ms(lambda: m(torch.randn(1, 1, 48, 256)), 100, 10, cuda=False)
        print('crnn', T['crnn'], flush=True)
        # whole reader on one plate: crop in memory -> resize / normalise -> model -> greedy CTC -> text
        import cv2
        rows = seq.load_rows('test')[:200]
        imgs = [cv2.imread(r['path']) for r in rows]
        m_t = svtr('svtrv2_t_msr')
        m_c, full = crnn()
        ds_t = P.DS(rows, cls, 'msr', False)
        ds_c = seq.DS(rows, cls, 48, 256, True, False)
        def read_t(i):
            x = ds_t[i][0][None].to(dev)
            (toks, c), = P.decode(m_t(x).float())
            return vocab.tokens_to_text([cls[k - 1] for k in toks])
        def read_c(i):
            x = ds_c[i][0][None].to(dev)
            toks, _ = seq.ctc_decode(m_c(x).float())
            return vocab.tokens_to_text([cls[k - 1] for k in toks[0]])
        for name, fn in (('svtrv2_t_msr', read_t), ('crnn', read_c)):
            for i in range(30):
                fn(i)
            ms = []
            for i in range(len(rows)):
                torch.cuda.synchronize(); t0 = time.perf_counter(); fn(i); torch.cuda.synchronize()
                ms.append(1000 * (time.perf_counter() - t0))
            T[f'reader_{name}_ms'] = round(float(np.percentile(ms, 10) if P10 else np.median(ms)), 3)
        print('reader', T['reader_svtrv2_t_msr_ms'], T['reader_crnn_ms'], flush=True)
    res['timing'] = T
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

if 'onnx' in TASKS:
    import onnxruntime as ort
    O = {}
    m = svtr('svtrv2_t_msr', device=torch.device('cpu'))
    f = os.path.join(RUN, '_local', 'svtrv2_t_msr.onnx')
    os.makedirs(os.path.dirname(f), exist_ok=True)
    try:
        torch.onnx.export(m, torch.randn(1, 3, 48, 96), f, input_names=['x'], output_names=['probs'], opset_version=17,
                          dynamic_axes={'x': {0: 'b', 2: 'h', 3: 'w'}, 'probs': {0: 'b', 1: 't'}}, dynamo=False)
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        s = ort.InferenceSession(f, so, providers=['CPUExecutionProvider'])
        rows = seq.load_rows('test')
        ds = P.DS(rows, cls, 'msr', False)
        K = kaggle('svtrv2_t_msr', 'test')
        diffs, ok, sametext, per = [], 0, 0, {}
        with torch.inference_mode():
            for i, r in enumerate(rows):
                x = ds[i][0][None]
                a = m(x).float().numpy()
                b = s.run(None, {'x': x.numpy()})[0]
                diffs.append(float(np.abs(a - b).max()))
                (toks, _), = P.decode(torch.from_numpy(b))
                t = vocab.tokens_to_text([cls[k - 1] for k in toks])
                ok += same(t, r['text']); sametext += same(t, K[r['file']]['pred'])
        for k, (h, w) in {'64x64': (64, 64), '48x96': (48, 96)}.items():
            x = np.random.randn(1, 3, h, w).astype(np.float32)
            per[k] = t_ms(lambda: s.run(None, {'x': x}), 200, 20, cuda=False)
        O = {'ok': True, 'MiB': round(os.path.getsize(f) / 2 ** 20, 2), 'max_abs_diff': round(max(diffs), 6),
             'exact_%': round(100 * ok / len(rows), 2), 'same_text_as_kaggle': sametext, 'plates': len(rows),
             'ort_cpu4_ms': per, 'onnxruntime': ort.__version__}
    except Exception as e:      # noqa: BLE001
        O = {'ok': False, 'error': str(e)[:400]}
    print('onnx', O, flush=True)
    res['onnx'] = O
    json.dump(res, open(OUT, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
