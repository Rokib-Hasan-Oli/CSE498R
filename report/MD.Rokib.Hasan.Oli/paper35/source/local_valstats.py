"""Paper #35 add-on to local_check.py: the per-photo statistics that Ultralytics val itself scores (TP flag at
IoU 0.5 and score of every box, plates per photo), so analysis.py can bootstrap exactly the mAP50 the kit reports.

  venv python local_valstats.py <unpacked After Train/results.zip>

Why: predict() and val() letterbox photos differently (val pads each batch to a shared rectangle), and on the 71
night photos that alone moves AP50 by up to 2 points. The kit's numbers are val numbers, so the tests use val's boxes.
Runs val on the enhanced copies that local_check.py wrote (_work/enh/<enhancer>/images/test), for the night list and
for all 1,153 test photos. Writes det_val_stats.jsonl.
"""
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('LPR_WORKERS', '0')
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import local_check as lc  # noqa: E402  (same environment variables, ENH, val_list)

CAPTURE = []


def main():
    from ultralytics import YOLO
    from ultralytics.utils import metrics as um
    orig = um.DetMetrics.update_stats

    def update_stats(self, stat):
        CAPTURE.append({'file': stat.get('im_name'), 'tp': np.asarray(stat['tp'])[:, 0].astype(int).tolist()
                        if len(stat['tp']) else [], 'conf': [round(float(c), 5) for c in stat['conf']],
                        'n_gt': int(len(stat['target_cls']))})
        return orig(self, stat)

    um.DetMetrics.update_stats = update_stats
    run = Path(sys.argv[1])
    plate = YOLO(str(run / 'runs' / 'det_v8s' / 'weights' / 'best.pt'))
    L = json.loads((HERE / 'local_check.json').read_text(encoding='utf-8'))
    out = []
    for name in lc.ENH:
        fold = lc.WORK / 'enh' / name / 'images' / 'test'
        for tag, files in (('night', [fold / f for f in L['night_files']]), ('all', sorted(fold.glob('*.jpg')))):
            CAPTURE.clear()
            m = lc.val_list(plate, files, f'vs_{name}_{tag}')
            assert len(CAPTURE) == len(files), (name, tag, len(CAPTURE), len(files))
            out.append({'enh': name, 'set': tag, 'val': m, 'photos': list(CAPTURE)})
            print(name, tag, m['mAP50'], len(CAPTURE), flush=True)
    with open(HERE / 'det_val_stats.jsonl', 'w', encoding='utf-8') as fh:
        for r in out:
            fh.write(json.dumps(r) + '\n')
    print('DONE')


if __name__ == '__main__':
    main()
