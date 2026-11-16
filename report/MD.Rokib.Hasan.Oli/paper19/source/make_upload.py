"""Build `After train/github_upload/` for paper #19 (repository layout of github.com/Rokib-Hasan-Oli/CSE498R).

  D:/498R/_dataset_build/venv/Scripts/python.exe make_upload.py <unpacked results.zip>

Copies the run, the kit, the notes and report sources; writes the FP16 classifier; builds results/ and README.md on
top of the paper #36 upload folder (BASE: the combined #33, #35, #34 and #36 changes, with #19 deliberately left out),
adding back only the #19 lines; adds the run log and download file to `Logs and Download/` (the user's folder, as #36
does). The package README (weights/paper19/.../README.md) and UPLOAD_STEPS.md are written by hand; this script never
touches them. Idempotent: re-running replaces what it made.
"""
import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUN = Path(sys.argv[1]).resolve()
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
NAME = PAPER.name
UP = PAPER / 'After train' / 'github_upload'
ME = 'MD.Rokib.Hasan.Oli'
BASE = next(PAPER.parent.glob('36_*')) / 'After Train' / 'github_upload'     # holds #33, #35, #34, #36 (not #19)
W = UP / 'weights' / 'paper19' / ME
NB = UP / 'notebooks' / ME
SRC = UP / 'src' / ME / 'paper19'
REP = UP / 'report' / ME / 'paper19'
RES = UP / 'results'
LOGS = UP / 'Logs and Download' / ME
for d in (W, NB, SRC / 'kaggle', REP / 'source', RES, LOGS):
    d.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# 1. weights: the best classifier gate in FP16, the results zip, the comparison table, the PDF ----------------------
import torch  # noqa: E402

sd = torch.load(RUN / 'runs/cls_b4_ce/best.pt', map_location='cpu')
res = json.load(open(RUN / 'runs/cls_b4_ce/results.json', encoding='utf-8'))
torch.save({'state_dict': {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()},
            'arch': 'torchvision resnet50, fc -> 4', 'classes': res['classes'], 'input_HW': res['settings']['input_HW'],
            'letterbox_grey': 127, 'normalize': 'ImageNet mean / std, RGB', 'stored_as': 'float16 (load with .float())',
            'source': 'paper #19 v1.1 Kaggle run, runs/cls_b4_ce/best.pt (cross-entropy, best val accuracy, epoch 4 of 24)'},
           W / 'best_cls_b4_ce_fp16.pt')
shutil.copy2(RUN / 'paper19_results.zip', W / 'paper19_results.zip')
shutil.copy2(RUN / 'results_vs_paper.md', W / 'results_vs_paper.md')
pdf = PAPER / 'LPLCv2_Study_Report.pdf'
shutil.copy2(pdf, W / pdf.name)

# 2. notebooks: the run as it ran; the log and Kaggle's download file also in Logs and Download/ ----------------------
shutil.copy2(PAPER / 'After train' / 'paper-19.ipynb', NB / 'paper19_run.ipynb')
shutil.copy2(PAPER / 'After train' / 'paper-19.log', NB / 'paper19_run.log')
shutil.copy2(PAPER / 'After train' / 'paper-19.log', LOGS / 'paper-19.log')
shutil.copy2(PAPER / 'After train' / 'download.txt', LOGS / 'download-19.txt')

# 3. src: START_HERE (notes link -> repo path), guide, the kit as run -----------------------------------------------
notes_link = f'../../../report/{ME}/paper19/paper19_notes.md'
t = (PAPER / 'START_HERE.md').read_text(encoding='utf-8')
t = t.replace(f"]({NAME.replace(' ', '%20')}.md)", f']({notes_link})')
(SRC / 'START_HERE.md').write_text(t, encoding='utf-8', newline='\n')
shutil.copy2(PAPER / 'Kaggle_Training_Guide.md', SRC / 'Kaggle_Training_Guide.md')
for f in (PAPER / 'kaggle').iterdir():
    shutil.copy2(f, SRC / 'kaggle' / f.name)

# 4. report: notes (links -> repo paths) and the report sources ----------------------------------------------------
n = (PAPER / f'{NAME}.md').read_text(encoding='utf-8')
n = n.replace('(Kaggle_Training_Guide.md)', f'(../../../src/{ME}/paper19/Kaggle_Training_Guide.md)')
n = n.replace('(kaggle/paper19_kaggle.ipynb)', f'(../../../src/{ME}/paper19/kaggle/paper19_kaggle.ipynb)')
n = n.replace('(LPLCv2_Study_Report.pdf)', f'(../../../weights/paper19/{ME}/LPLCv2_Study_Report.pdf)')
n = n.replace('(sources in `report_src/`)', '(sources in [`source/`](source/))')
(REP / 'paper19_notes.md').write_text(n, encoding='utf-8', newline='\n')
skip = {'fonts', '_work', '__pycache__', 'local_train_log.txt'}
for f in HERE.iterdir():
    if f.name in skip:
        continue
    if f.is_dir():
        if f.name == '_local_runs':                         # results + scores of the extra runs (no weights were saved)
            for d in f.iterdir():
                (REP / 'source' / '_local_runs' / d.name).mkdir(parents=True, exist_ok=True)
                for g in d.iterdir():
                    shutil.copy2(g, REP / 'source' / '_local_runs' / d.name / g.name)
        continue
    shutil.copy2(f, REP / 'source' / f.name)
# the training log without the repeated torch warning lines
lines = [l for l in (HERE / 'local_train_log.txt').read_text(encoding='utf-8', errors='replace').splitlines()
         if 'UserWarning' not in l and 'tl += float' not in l and 'Consider using tensor.detach' not in l]
(REP / 'source' / 'local_train_log.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')

# 5. results: the #36 upload's table + the #19 row, rendered by add_results.py ------------------------------------
tmp = PAPER.parent.parent.parent / '_results_tmp19'                 # docs/_results_tmp19 (specs import works there)
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
master = PAPER.parent.parent.parent / 'results'
shutil.copy2(master / 'add_results.py', tmp / 'add_results.py')
base_md = (BASE / 'results' / 'results_table.md').read_text(encoding='utf-8')
assert '### Paper #19' not in base_md
for b in master.glob('paper*_vs_paper.md'):
    pid = b.name[5:7]
    if pid == '19' or f'### Paper #{int(pid)}:' in base_md or f'### Paper #{int(pid)} ' in base_md:   # GitHub's + #19
        shutil.copy2(b, tmp / b.name)
rows = list(csv.DictReader(open(BASE / 'results' / 'experiments.csv', encoding='utf-8')))
new = [r for r in csv.DictReader(open(master / 'experiments.csv', encoding='utf-8')) if r['paper'].startswith('#19 ')]
assert len(new) == 1 and not any(r['paper'].startswith('#19 ') for r in rows)
cols = list(rows[0])
with open(tmp / 'experiments.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader(); w.writerows(rows + new)
out = subprocess.run([sys.executable, str(tmp / 'add_results.py'), '--member', 'Rokib Hasan Oli'], capture_output=True,
                     encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8'))
print(out.stdout[-400:], out.stderr[-400:])
shutil.copy2(tmp / 'experiments.csv', RES / 'experiments.csv')
shutil.copy2(tmp / 'results_table.md', RES / 'results_table.md')
shutil.rmtree(tmp)

# 6. root README: the #36 upload's README (teammates' GitHub edits kept) + the #19 tracker row + the paper list -----
r = (BASE / 'README.md').read_text(encoding='utf-8')
row19 = ('| 19 | LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification | ✅ | '
         'A legibility classifier is **not** a better reject-or-read gate than the OCR\'s own confidence: keeping 80 % of '
         'plates, CRNN confidence 84.5 % right vs classifier 81.0 % (CI −5.6 to −1.3); confidence + BRTA layout + reader '
         'agreement gives **96.7 %** for the #24 fusion. Proxy labels: 4-class 51.9 % / Quality Filter 71.7 % (paper 89.8 / '
         '96.7); EMA loss: no gain on 4 classes (3 seeds) | '
         f'[results](weights/paper19/{ME}/) · [notebook](notebooks/{ME}/paper19_run.ipynb) · [kit](src/{ME}/paper19/) · '
         f'[notes](report/{ME}/paper19/) |')
anchor = next(l for l in r.splitlines() if l.startswith('| 23 |') or l.startswith('| 24 |'))
assert '| 19 |' not in r
r = r.replace(anchor, row19 + '\n' + anchor, 1)
old_list = '| 2, 6, 8, 23, 24, 33, 34, 35, 36 |'
assert r.count(old_list) == 1, 'member paper list changed in the base README'
r = r.replace(old_list, '| 2, 6, 8, 19, 23, 24, 33, 34, 35, 36 |')
(UP / 'README.md').write_text(r, encoding='utf-8', newline='\n')

print('upload tree:', UP)
for p in sorted(UP.rglob('*')):
    if p.is_file() and 'source' not in p.parts and 'kaggle' not in p.parts:
        print(f'  {p.relative_to(UP)}  {p.stat().st_size / 2 ** 20:.2f} MiB')
print('sha256 best_cls_b4_ce_fp16.pt', sha(W / 'best_cls_b4_ce_fp16.pt'))
print('sha256 paper19_results.zip', sha(W / 'paper19_results.zip'))
