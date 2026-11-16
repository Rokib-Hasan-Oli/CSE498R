"""Paper #19 local re-check (RTX 5060 Ti): the Kaggle v1.1 weights re-run on our PC, plate by plate.

  D:/498R/_dataset_build/venv/Scripts/python.exe local_check.py <unpacked results.zip>

1. The four ResNet-50 legibility classifiers (runs/cls_*/best.pt) on the 766 val and 651 test crops: probabilities vs
   the run's test_scores.csv, micro F1 vs results.json.
2. The deployment CRNN (runs/readers_crnn/model_full.pt) on val and test: test readings vs deploy_test.csv, val
   readings vs the paper #24 run's crnn/ocr_preds_val.csv (same weights, see local_check.json 'crnn_same_as_p24').
3. Batch-1 latency on GPU (FP32 and autocast FP16) and CPU (4 threads) for the classifier and the CRNN.
Writes local_check.json and local_preds.csv (val + test: CRNN reading/confidence, every classifier's probabilities),
the inputs of analysis.py. Uses the kit code exactly as run (<results>/code).
"""
import csv
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
os.environ.setdefault('DATASET_ROOT', r'D:\498R\dataset')
os.environ.setdefault('LPR_WORKERS', '0')
os.environ.setdefault('LPR_NO_DOWNLOAD', '1')
os.environ.setdefault('LPR_WORK', str(HERE / '_work'))
sys.path.insert(0, str(RUN / 'code'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision  # noqa: E402

import common  # noqa: E402
import p19_legibility as P  # noqa: E402
import seq  # noqa: E402
import vocab  # noqa: E402

P24_ZIP = next(Path(r'D:\498R\docs\Rokib Hasna').glob('24_*/After Train/github_upload/weights/paper24/*/paper24_results.zip'))
CLS = ['cls_b4_ema', 'cls_b4_ce', 'cls_qf_ema', 'cls_qf_ce']
dev = torch.device('cuda')
root = common.find_dataset()
out = {'dataset_version': json.load(open(os.path.join(root, 'dataset_manifest.json'), encoding='utf-8'))['version'],
       'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
labels = {r['file']: r for r in P.read_labels(RUN / 'runs/legibility_labels')}
split_rows = {s: [r for r in labels.values() if r['split'] == s] for s in ('val', 'test')}


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def load_cls(name):
    res = json.load(open(RUN / 'runs' / name / 'results.json', encoding='utf-8'))
    m = torchvision.models.resnet50()
    m.fc = nn.Linear(m.fc.in_features, len(res['classes']))
    m.load_state_dict(torch.load(RUN / 'runs' / name / 'best.pt', map_location='cpu'))
    return m.to(dev).eval(), res


probs = {}
out['classifiers'] = {}
for name in CLS:
    m, res = load_cls(name)
    names = res['classes']
    for s in ('val', 'test'):
        rows = [dict(r, y=r['label4'] if res['task'] == 'baseline4' else r['quality']) for r in split_rows[s]]
        dl = torch.utils.data.DataLoader(P.CropDS(rows, names, root, 96, 192, False), batch_size=64, shuffle=False)
        p, y = P.predict_probs(m, dl, dev)
        probs[(name, s)] = {r['file']: q for r, q in zip(rows, p)}
        if s == 'test':
            kag = {r['file']: np.array([float(r[f'p_{n}']) for n in names]) for r in
                   csv.DictReader(open(RUN / 'runs' / name / 'test_scores.csv', encoding='utf-8'))}
            d = np.array([np.abs(kag[f] - probs[(name, s)][f]).max() for f in kag])
            agree = np.mean([kag[f].argmax() == probs[(name, s)][f].argmax() for f in kag])
            f1 = P.f1_scores(y, p.argmax(1), len(names))
            out['classifiers'][name] = {'test_rows': len(kag), 'max_abs_prob_diff': round(float(d.max()), 5),
                                        'mean_abs_prob_diff': round(float(d.mean()), 6), 'argmax_agree_%': round(100 * agree, 2),
                                        'local_micro_F1_%': f1['micro_F1_%'], 'kaggle_micro_F1_%': res['test']['micro_F1_%'],
                                        'local_per_class': f1['per_class_F1_%'], 'kaggle_per_class': res['test']['per_class_F1_%'],
                                        'sha256': sha(RUN / 'runs' / name / 'best.pt')}
    print(name, out['classifiers'][name], flush=True)

# ---------------------------------------------------------------------------------------------------------- CRNN
ck = torch.load(RUN / 'runs/readers_crnn/model_full.pt', map_location='cpu', weights_only=False)
cls = ck['classes']
crnn = seq.CRNN(len(cls), ck['H'])
crnn.load_state_dict(ck['state_dict'])
crnn = crnn.to(dev).eval()
args = type('A', (), {'H': ck['H'], 'W': ck['W'], 'stack': ck['stack']})
reads = {}
for s in ('val', 'test'):
    rows = [r for r in seq.load_rows(s)]
    preds, _ = seq.predict(crnn, rows, cls, 'crnn', args, dev)
    reads[s] = {p['file']: p for p in preds}
dep = {r['file']: r for r in csv.DictReader(open(RUN / 'runs/legibility_labels/deploy_test.csv', encoding='utf-8'))}
same_text = sum(vocab.norm_text(reads['test'][f]['pred']) == vocab.norm_text(dep[f]['pred']) for f in dep)
cdiff = max(abs(float(dep[f]['conf']) - reads['test'][f]['conf']) for f in dep)
with zipfile.ZipFile(P24_ZIP) as z:
    p24 = {r['file']: r for r in csv.DictReader(io.TextIOWrapper(z.open('runs/crnn/ocr_preds_val.csv'), encoding='utf-8'))}
same_val = sum(vocab.norm_text(reads['val'][f]['pred']) == vocab.norm_text(p24[f]['pred']) for f in reads['val'] if f in p24)
ok = lambda d: 100 * np.mean([vocab.norm_text(p['pred']) == vocab.norm_text(p['true']) for p in d.values()])
out['crnn'] = {'sha256': sha(RUN / 'runs/readers_crnn/model_full.pt'), 'test_plates': len(dep),
               'test_same_text_as_kaggle': int(same_text), 'test_max_conf_diff': round(cdiff, 4),
               'local_test_exact_%': round(ok(reads['test']), 2), 'local_val_exact_%': round(ok(reads['val']), 2),
               'val_same_text_as_p24_run': int(same_val), 'val_plates': len(reads['val'])}
print('crnn', out['crnn'], flush=True)

# ---------------------------------------------------------------------------------------------------------- timing
def lat(model, x, n=200, amp=False):
    ms = []
    with torch.no_grad():
        for i in range(n):
            if x.is_cuda:
                torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.autocast('cuda', enabled=amp):
                model(x)
            if x.is_cuda:
                torch.cuda.synchronize()
            ms.append(1000 * (time.perf_counter() - t))
    return round(float(np.median(ms[20:])), 2)


m, _ = load_cls('cls_qf_ema')
xc = torch.randn(1, 3, 96, 192)
xr = seq.DS([seq.load_rows('test')[0]], cls, ck['H'], ck['W'], ck['stack'], False)[0][0][None]
torch.set_num_threads(4)
out['speed_ms_batch1'] = {
    'classifier_gpu_fp32': lat(m, xc.to(dev)), 'classifier_gpu_fp16': lat(m, xc.to(dev), amp=True),
    'crnn_gpu_fp32': lat(crnn, xr.to(dev)), 'crnn_gpu_fp16': lat(crnn, xr.to(dev), amp=True)}
m, crnn = m.cpu(), crnn.cpu()
out['speed_ms_batch1'].update({'classifier_cpu4': lat(m, xc, 60), 'crnn_cpu4': lat(crnn, xr, 60),
                               'cpu': 'Intel Core Ultra 7 265K, 4 threads'})
out['params_M'] = {'classifier': round(sum(p.numel() for p in m.parameters()) / 1e6, 2),
                   'crnn': round(sum(p.numel() for p in crnn.parameters()) / 1e6, 2)}
print(out['speed_ms_batch1'], out['params_M'])

# ---------------------------------------------------------------------------------------------------------- save
with open(HERE / 'local_preds.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    hdr = ['file', 'split', 'text', 'source', 'label4', 'crnn_pred', 'crnn_conf', 'crnn_ok']
    for name in CLS:
        hdr += [f'{name}:{n}' for n in P.TASKS['baseline4' if 'b4' in name else 'quality']]
    w.writerow(hdr)
    for s in ('val', 'test'):
        for r in split_rows[s]:
            p = reads[s][r['file']]
            row = [r['file'], s, r['text'], r['source'], r['label4'], p['pred'], p['conf'],
                   int(vocab.norm_text(p['pred']) == vocab.norm_text(r['text']))]
            for name in CLS:
                row += [round(float(v), 5) for v in probs[(name, s)][r['file']]]
            w.writerow(row)
json.dump(out, open(HERE / 'local_check.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('wrote local_check.json, local_preds.csv')
