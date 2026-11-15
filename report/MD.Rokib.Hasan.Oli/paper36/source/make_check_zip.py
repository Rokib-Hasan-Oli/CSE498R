"""Paper #36: pack the full local run of the original-paper check (27 Oct 2026, the kit's c36_sahi.py on the CPU) the way
the check notebook's last cell does, so add_results.py can fill the notes' check table.

  venv python make_check_zip.py <c36 work dir with runs/visdrone_rescore/results.json and eval.log> <out dir>

Runs the notebook's own report cell (rows + report.compare_table + report.pack) with WORK = <out dir>.
-> <out dir>/paper36_check_outputs.zip, paper36_check_results.zip, results_original_check.md
"""
import glob
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parent / 'kaggle'
SRC, WORK = Path(sys.argv[1]), Path(sys.argv[2])
(WORK / 'runs' / 'visdrone_rescore').mkdir(parents=True, exist_ok=True)
(WORK / 'logs').mkdir(exist_ok=True)
(WORK / 'lprkit').mkdir(exist_ok=True)
res = json.loads((SRC / 'runs' / 'visdrone_rescore' / 'results.json').read_text(encoding='utf-8'))
assert len(res['settings']) == 20 and res['max_abs_diff_rescored_vs_paper_AP50'] == 0.0
shutil.copy2(SRC / 'runs' / 'visdrone_rescore' / 'results.json', WORK / 'runs' / 'visdrone_rescore' / 'results.json')
shutil.copy2(SRC / 'eval.log', WORK / 'logs' / 'visdrone_rescore.log')
(WORK / 'environment.json').write_text(json.dumps({**res['env'], 'note': 'local CPU run, 27 Oct 2026 (the check needs no GPU)'},
                                                  indent=1), encoding='utf-8')
for f in KIT.glob('*.py'):
    shutil.copy2(f, WORK / 'lprkit' / f.name)
nb = json.loads((KIT / 'paper36_original_check.ipynb').read_text(encoding='utf-8'))
cell = next(''.join(c['source']) for c in nb['cells'] if 'report.pack' in ''.join(c['source']))
os.environ['LPR_WORK'] = str(WORK)
sys.path.insert(0, str(WORK / 'lprkit'))
g = {'WORK': str(WORK), 'KIT': str(WORK / 'lprkit'), 'RUNS': str(WORK / 'runs'), 'os': os, 'glob': glob}
exec(cell, g)
print(sorted(p.name for p in WORK.glob('*.zip')))
