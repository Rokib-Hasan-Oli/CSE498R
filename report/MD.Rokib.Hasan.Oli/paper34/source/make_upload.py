"""Build `After Train/github_upload/` for paper #34 (repository layout of github.com/Rokib-Hasan-Oli/CSE498R).

  D:/498R/_dataset_build/venv/Scripts/python.exe make_upload.py <unpacked results.zip> <fusion dir>

Copies the run, the (fixed) kit, the notes and the report sources; writes the two SVTRv2 readers with their class list
(T in FP32, S in FP16: 43 MiB in FP32 is over GitHub's 25 MB web-upload limit); builds results/ and README.md on top of
the newest upload folder of the chain #33 -> #35 -> #19 (BASE below), adding only the #34 lines. The package README
(weights/paper34/.../README.md) and UPLOAD_STEPS.md are written by hand; this script never touches them.
Also fills `After Train/for_drive/` with paper34_outputs.zip (from the run). Idempotent: re-running replaces what it made.
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

RUN, FUSE = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
NAME = PAPER.name
AT = PAPER / 'After Train'
UP = AT / 'github_upload'
ME = 'MD.Rokib.Hasan.Oli'
NEW5 = PAPER.parent
BASE = next(NEW5.glob('19_*')) / 'After train' / 'github_upload'      # holds the #33, #35 and #19 changes
W = UP / 'weights' / 'paper34' / ME
NB = UP / 'notebooks' / ME
SRC = UP / 'src' / ME / 'paper34'
REP = UP / 'report' / ME / 'paper34'
RES = UP / 'results'
for d in (W, NB, SRC / 'kaggle', REP / 'source', RES, AT / 'for_drive'):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(RUN / 'code'))
os.environ.setdefault('DATASET_ROOT', 'D:/498R/dataset')
os.environ['LPR_NO_DOWNLOAD'] = '1'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# 1. weights: the two readers (+ class list and settings), the results zip, the comparison table, the PDF ------------
import torch  # noqa: E402

import p34_svtrv2 as P  # noqa: E402
import seq  # noqa: E402

cls = seq.class_list()
for run, half, out in (('svtrv2_t_msr', False, 'svtrv2_t_msr.pt'), ('svtrv2_s_msr', True, 'svtrv2_s_msr_fp16.pt')):
    res = json.load(open(RUN / 'runs' / run / 'results.json', encoding='utf-8'))
    sd = torch.load(RUN / 'runs' / run / 'best.pt', map_location='cpu', weights_only=True)
    size = res['kind'][-1]
    torch.save({'state_dict': {k: (v.half() if half and v.is_floating_point() else v) for k, v in sd.items()},
                'kind': res['kind'], 'size': size, 'resize': res['settings']['resize'], 'classes': cls,
                'encoder': P.ENC[size], 'mixer': P.MIXER[size], 'openocr_commit': P.COMMIT,
                'input': 'BGR crop -> RGB, PIL bicubic resize to the MSR size, to_tensor, normalize mean 0.5 std 0.5',
                'stored_as': 'float16 (load with .float())' if half else 'float32',
                'source_sha256': sha(RUN / 'runs' / run / 'best.pt'),
                'source': f"paper #34 v1.1 Kaggle run 26 Oct 2026, runs/{run}/best.pt (best validation loss, epoch "
                          f"{min(res['log'], key=lambda x: x['val_loss'])['epoch']} of {res['epochs_run']})",
                'test_plate_exact_%': res['ocr_test']['rows_decoder']['plate_exact_%']}, W / out)
shutil.copy2(RUN / 'paper34_results.zip', W / 'paper34_results.zip')
shutil.copy2(RUN / 'results_vs_paper.md', W / 'results_vs_paper.md')
shutil.copy2(FUSE / 'out' / 'fusion34' / 'fusion_table.md', W / 'fusion_with_earlier_readers.md')
pdf = PAPER / 'SVTRv2_Study_Report.pdf'
if pdf.exists():
    shutil.copy2(pdf, W / pdf.name)

# 2. notebooks: the run as it ran ----------------------------------------------------------------------------------
shutil.copy2(AT / 'paper-34.ipynb', NB / 'paper34_run.ipynb')
shutil.copy2(AT / 'paper-34.log', NB / 'paper34_run.log')

# 3. src: START_HERE (notes link -> repo path), guide, the kit (fixed 27 Oct; the run's own code is in the run notebook)
notes_link = f'../../../report/{ME}/paper34/paper34_notes.md'
t = (PAPER / 'START_HERE.md').read_text(encoding='utf-8')
t = t.replace(f"]({NAME.replace(' ', '%20')}.md)", f']({notes_link})')
assert notes_link in t
(SRC / 'START_HERE.md').write_text(t, encoding='utf-8', newline='\n')
shutil.copy2(PAPER / 'Kaggle_Training_Guide.md', SRC / 'Kaggle_Training_Guide.md')
for f in (PAPER / 'kaggle').iterdir():
    if f.is_file():
        shutil.copy2(f, SRC / 'kaggle' / f.name)

# 4. report: notes (links -> repo paths) and the report sources -----------------------------------------------------
n = (PAPER / f'{NAME}.md').read_text(encoding='utf-8')
n = n.replace('(Kaggle_Training_Guide.md)', f'(../../../src/{ME}/paper34/Kaggle_Training_Guide.md)')
n = n.replace('(kaggle/paper34_kaggle.ipynb)', f'(../../../src/{ME}/paper34/kaggle/paper34_kaggle.ipynb)')
n = n.replace('(SVTRv2_Study_Report.pdf)', f'(../../../weights/paper34/{ME}/SVTRv2_Study_Report.pdf)')
n = n.replace('(sources in `report_src/`)', '(sources in [`source/`](source/))')
assert '(source/)' in n and f'weights/paper34/{ME}/SVTRv2_Study_Report.pdf' in n
(REP / 'paper34_notes.md').write_text(n, encoding='utf-8', newline='\n')
for f in HERE.iterdir():
    if f.is_file() and f.suffix in ('.py', '.json', '.txt', '.csv', '.png', '.jpg', '.md'):
        shutil.copy2(f, REP / 'source' / f.name)
for f in ('fusion_table.md', 'results.json'):
    shutil.copy2(FUSE / 'out' / 'fusion34' / f, REP / 'source' / f'fusion34_{f}')

# 5. results: the base upload's table + the #34 rows, rendered by add_results.py -------------------------------------
tmp = PAPER.parent.parent.parent / '_results_tmp34'                 # docs/_results_tmp34 (specs import works there)
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
master = PAPER.parent.parent.parent / 'results'
shutil.copy2(master / 'add_results.py', tmp / 'add_results.py')
base_md = (BASE / 'results' / 'results_table.md').read_text(encoding='utf-8')
for b in master.glob('paper*_vs_paper.md'):
    pid = b.name[5:7]
    if pid == '34' or f'### Paper #{int(pid)}:' in base_md or f'### Paper #{int(pid)} ' in base_md:
        shutil.copy2(b, tmp / b.name)
rows = list(csv.DictReader(open(BASE / 'results' / 'experiments.csv', encoding='utf-8')))
new = [r for r in csv.DictReader(open(master / 'experiments.csv', encoding='utf-8')) if r['paper'].startswith('#34 ')]
assert len(new) == 5 and not any(r['paper'].startswith('#34 ') for r in rows)
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

# 6. root README: the base README + the #34 tracker row + the member's paper list -----------------------------------
A = json.load(open(HERE / 'analysis.json', encoding='utf-8'))
FR = A['fusion']['rows']
r = (BASE / 'README.md').read_text(encoding='utf-8')
row34 = ('| 34 | SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition | ✅ | '
         f"SVTRv2-T (official code, English start) reads **{A['main']['svtrv2_t_msr']['test']['plate_exact_%']:.2f} %** of test plates "
         f"vs CRNN {A['main']['crnn']['test']['plate_exact_%']:.2f} % (p < 0.0001), below char-YOLO {A['main']['chars_v8n']['test']['plate_exact_%']:.2f} %; "
         f"from scratch {A['main']['svtrv2_t_scratch']['test']['plate_exact_%']:.2f} %. As a format fallback behind char-YOLO: "
         f"**{FR['chars_t_fallback']['test_%']:.2f} %** (22 fixed / 0 broken); 4-reader vote chosen on val **{FR['new_best_val']['test_%']:.2f} %** | "
         f'[results](weights/paper34/{ME}/) · [notebook](notebooks/{ME}/paper34_run.ipynb) · [kit](src/{ME}/paper34/) · '
         f'[notes](report/{ME}/paper34/) |')
assert '| 34 |' not in r
anchor = next(l for l in r.splitlines() if l.startswith('| 35 |'))
r = r.replace(anchor, row34 + '\n' + anchor, 1)
old_list = '| 2, 6, 8, 19, 23, 24, 33, 35 |'
assert old_list in r, 'member paper list changed in the base README'
r = r.replace(old_list, '| 2, 6, 8, 19, 23, 24, 33, 34, 35 |')
(UP / 'README.md').write_text(r, encoding='utf-8', newline='\n')

# 7. Drive: the run's outputs zip (all five models) -----------------------------------------------------------------
shutil.copy2(RUN / 'paper34_outputs.zip', AT / 'for_drive' / 'paper34_outputs.zip')

print('upload tree:', UP)
for p in sorted(UP.rglob('*')):
    if p.is_file() and 'source' not in p.parts and 'kaggle' not in p.parts:
        print(f'  {p.relative_to(UP)}  {p.stat().st_size / 2 ** 20:.2f} MiB')
for f in ('svtrv2_t_msr.pt', 'svtrv2_s_msr_fp16.pt', 'paper34_results.zip'):
    print('sha256', f, sha(W / f))
print('base:', BASE)
