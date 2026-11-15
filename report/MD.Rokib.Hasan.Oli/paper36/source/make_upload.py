"""Build `After Train/github_upload/` for paper #36 (repository layout of github.com/Rokib-Hasan-Oli/CSE498R).

  D:/498R/_dataset_build/venv/Scripts/python.exe make_upload.py <unpacked results.zip> <check zip dir> <CSE498R clone>

Copies the run, the kit, the notes and the report sources; the two new detectors (SF, 1280 px; the 640 model is the
#35 file and the CRNN the #24 file); builds results/ on top of the newest upload folder of the chain
#33 -> #35 -> #19 -> #34 (BASE below), adding only the #36 lines; builds README.md from that chain's README with the
teammates' README edits now on GitHub (origin/main) kept, plus the #36 tracker row; adds the run log and download file
to `Logs and Download/` (the folder the user started on GitHub on 29 Oct). The package README
(weights/paper36/.../README.md) and UPLOAD_STEPS.md are written by hand; this script never touches them.
Also fills `After Train/for_drive/` with paper36_outputs.zip. Idempotent: re-running replaces what it made.
"""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUN, CHK, REPO = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3]).resolve()
HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
NAME = PAPER.name
AT = PAPER / 'After Train'
UP = AT / 'github_upload'
ME = 'MD.Rokib.Hasan.Oli'
NEW5 = PAPER.parent
BASE = next(NEW5.glob('34_*')) / 'After Train' / 'github_upload'      # holds the #33, #35, #19 and #34 changes
# #19's upload folder is unfinished on 29 Oct (its local runs, PDF and README numbers are still open), so its rows are
# left out here; when #19 is done, rebuild its upload on top of this one.
SKIP = ('19',)
W = UP / 'weights' / 'paper36' / ME
NB = UP / 'notebooks' / ME
SRC = UP / 'src' / ME / 'paper36'
REP = UP / 'report' / ME / 'paper36'
RES = UP / 'results'
LOGS = UP / 'Logs and Download' / ME
for d in (W, NB, SRC / 'kaggle', REP / 'source', RES, LOGS, AT / 'for_drive'):
    d.mkdir(parents=True, exist_ok=True)


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


# 1. weights: the SF and 1280 px detectors, results zips, comparison tables, the PDF --------------------------------
for run, out in (('det_v8s_sf', 'best_det_v8s_sf.pt'), ('det_v8s_1280', 'best_det_v8s_1280.pt')):
    shutil.copy2(RUN / 'runs' / run / 'weights' / 'best.pt', W / out)
for f in ('paper36_results.zip', 'results_vs_paper.md'):
    shutil.copy2(RUN / f, W / f)
shutil.copy2(CHK / 'paper36_check_results.zip', W / 'paper36_check_results.zip')
shutil.copy2(CHK / 'results_original_check.md', W / 'results_original_check.md')
pdf = PAPER / 'SAHI_Study_Report.pdf'
shutil.copy2(pdf, W / pdf.name)

# 2. notebooks: the run as it ran; the log and Kaggle's download file also in Logs and Download/ ----------------------
shutil.copy2(AT / 'paper-36.ipynb', NB / 'paper36_run.ipynb')
shutil.copy2(AT / 'paper-36.log', NB / 'paper36_run.log')
shutil.copy2(AT / 'paper-36.log', LOGS / 'paper-36.log')
shutil.copy2(AT / 'download.txt', LOGS / 'download-36.txt')

# 3. src: START_HERE (notes link -> repo path), guides, the kit and the original-check notebook ----------------------
notes_link = f'../../../report/{ME}/paper36/paper36_notes.md'
t = (PAPER / 'START_HERE.md').read_text(encoding='utf-8')
t = t.replace(f"]({NAME.replace(' ', '%20')}.md)", f']({notes_link})')
assert notes_link in t
(SRC / 'START_HERE.md').write_text(t, encoding='utf-8', newline='\n')
for g in ('Kaggle_Training_Guide.md', 'Original_Check_Guide.md'):
    shutil.copy2(PAPER / g, SRC / g)
for f in (PAPER / 'kaggle').iterdir():
    if f.is_file():
        shutil.copy2(f, SRC / 'kaggle' / f.name)

# 4. report: notes (links -> repo paths) and the report sources -----------------------------------------------------
n = (PAPER / f'{NAME}.md').read_text(encoding='utf-8')
n = n.replace('(Kaggle_Training_Guide.md)', f'(../../../src/{ME}/paper36/Kaggle_Training_Guide.md)')
n = n.replace('(Original_Check_Guide.md)', f'(../../../src/{ME}/paper36/Original_Check_Guide.md)')
n = n.replace('(kaggle/paper36_kaggle.ipynb)', f'(../../../src/{ME}/paper36/kaggle/paper36_kaggle.ipynb)')
n = n.replace('(kaggle/paper36_original_check.ipynb)', f'(../../../src/{ME}/paper36/kaggle/paper36_original_check.ipynb)')
n = n.replace('(SAHI_Study_Report.pdf)', f'(../../../weights/paper36/{ME}/SAHI_Study_Report.pdf)')
n = n.replace('sources in `report_src/`)', 'sources in [`source/`](source/))')
assert '(source/)' in n and f'weights/paper36/{ME}/SAHI_Study_Report.pdf' in n
assert '(Original_Check_Guide.md)' not in n and '(Kaggle_Training_Guide.md)' not in n
(REP / 'paper36_notes.md').write_text(n, encoding='utf-8', newline='\n')
for f in HERE.iterdir():
    if f.is_file() and f.suffix in ('.py', '.json', '.jsonl', '.txt', '.csv', '.png', '.jpg', '.md'):
        shutil.copy2(f, REP / 'source' / f.name)

# 5. results: the base upload's table + the #36 rows, rendered by add_results.py -------------------------------------
tmp = PAPER.parent.parent.parent / '_results_tmp36'                 # docs/_results_tmp36 (specs import works there)
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir()
master = PAPER.parent.parent.parent / 'results'
shutil.copy2(master / 'add_results.py', tmp / 'add_results.py')
base_md = (BASE / 'results' / 'results_table.md').read_text(encoding='utf-8')
for b in master.glob('paper*_vs_paper.md'):
    pid = b.name[5:7]
    if pid in SKIP:
        continue
    if pid == '36' or f'### Paper #{int(pid)}:' in base_md or f'### Paper #{int(pid)} ' in base_md:
        shutil.copy2(b, tmp / b.name)
rows = [r for r in csv.DictReader(open(BASE / 'results' / 'experiments.csv', encoding='utf-8'))
        if not any(r['paper'].startswith(f'#{p} ') for p in SKIP)]
new = [r for r in csv.DictReader(open(master / 'experiments.csv', encoding='utf-8')) if r['paper'].startswith('#36 ')]
assert len(new) == 4 and not any(r['paper'].startswith('#36 ') for r in rows)
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

# 6. root README: the chain's README + the teammates' edits now on GitHub + the #36 row + the member's paper list -----
A = json.load(open(HERE / 'analysis.json', encoding='utf-8'))
V, P = A['variants'], A['pairs']
best = A['val_scores']['v8s_640']['best_slicing']
git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, encoding='utf-8', check=True).stdout
gh = git('show', 'origin/main:README.md')
old = git('show', 'eb53c93:README.md')
r = (BASE / 'README.md').read_text(encoding='utf-8')
o_lines, g_lines = old.splitlines(), gh.splitlines()
assert len(o_lines) == len(g_lines), 'README on GitHub gained or lost lines: merge by hand'
for a, b in zip(o_lines, g_lines):                       # carry every line a teammate changed on GitHub
    if a != b:
        assert r.count(a + '\n') == 1, a[:80]
        r = r.replace(a + '\n', b + '\n')
row36 = ('| 36 | SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection | ✅ | '
         f"The paper's slicing **hurts** our plates: AP50 {V['v8s_640__FI']['AP50']:.2f} → {V['v8s_640__SAHI_FI']['AP50']:.2f} "
         f"(3x false alarms, 3.5x time; slice edges cut large plates). Dropping cut-off slice boxes (chosen on val): "
         f"**{V[best]['AP50']:.2f}** (= full image, n.s.). Slice training (SF): small-plate photos mAP50 84.5 → **96.3**; "
         f"1280 px no gain. Paper Table 1 reproduced 80 / 80 | "
         f'[results](weights/paper36/{ME}/) · [notebook](notebooks/{ME}/paper36_run.ipynb) · [kit](src/{ME}/paper36/) · '
         f'[notes](report/{ME}/paper36/) |')
assert '| 36 |' not in r
anchor = next(l for l in r.splitlines() if l.startswith('| 35 |'))
r = r.replace(anchor + '\n', anchor + '\n' + row36 + '\n', 1)
for p in SKIP:                                           # the chain's row for a paper that is not uploaded yet
    r = r.replace(next(l for l in r.splitlines() if l.startswith(f'| {p} |') and '✅' in l) + '\n', '', 1)
old_list = '| 2, 6, 8, 19, 23, 24, 33, 34, 35 |'
assert old_list in r, 'member paper list changed in the base README'
r = r.replace(old_list, '| ' + ', '.join(x for x in '2 6 8 19 23 24 33 34 35 36'.split() if x not in SKIP) + ' |')
(UP / 'README.md').write_text(r, encoding='utf-8', newline='\n')

# 7. Drive: the run's outputs zip (all models) and the check zip ------------------------------------------------------
shutil.copy2(RUN / 'paper36_outputs.zip', AT / 'for_drive' / 'paper36_outputs.zip')
shutil.copy2(CHK / 'paper36_check_outputs.zip', AT / 'for_drive' / 'paper36_check_outputs.zip')

print('upload tree:', UP)
for p in sorted(UP.rglob('*')):
    if p.is_file() and 'source' not in p.parts and 'kaggle' not in p.parts:
        print(f'  {p.relative_to(UP)}  {p.stat().st_size / 2 ** 20:.2f} MiB')
for f in ('best_det_v8s_sf.pt', 'best_det_v8s_1280.pt', 'paper36_results.zip', 'paper36_check_results.zip'):
    print('sha256', f, sha(W / f))
print('base:', BASE)
