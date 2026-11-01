"""Collect the LOCAL run of paper #8's original-paper check (the authors' Data.zip, 3 seeds; same commands as
kaggle/paper08_original_check.ipynb cells 3-5, run one job at a time on an RTX 5060 Ti on 24 Oct 2026, no synthetic
plate test) into check_local.json (numbers as mean, sd, n).

  python check_collect.py <runs folder of the local check>
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1]


def get(d, *k):
    for x in k:
        if not isinstance(d, dict) or x not in d:
            return None
        d = d[x]
    return d


def ms(xs):
    xs = [x for x in xs if x is not None]
    return [round(float(np.mean(xs)), 2), round(float(np.std(xs)), 2), len(xs)] if xs else None


out = {'where': 'local RTX 5060 Ti, 24 Oct 2026 (Kaggle notebook not run yet)', 'per_run': {}}
var = {}
for d in sorted(os.listdir(RUNS)):
    p = os.path.join(RUNS, d)
    if os.path.exists(os.path.join(p, 'results.json')) and os.path.exists(os.path.join(p, 'best.pt')):
        var.setdefault(d.rsplit('_s', 1)[0], []).append(p)
for v, ds in var.items():
    rs = [json.load(open(os.path.join(d, 'results.json'))) for d in ds]
    ex = [json.load(open(os.path.join(d, 'eval_extra.json'))) for d in ds]
    pc = lambda e, *k: None if get(e, *k) is None else 100 * get(e, *k)
    out[v] = {
        'runs': [os.path.basename(d) for d in ds],
        'acc_last': ms([r['chars_test']['last']['acc_%'] for r in rs]),
        'acc_best': ms([r['chars_test']['best']['acc_%'] for r in rs]),
        'P_last': ms([r['chars_test']['last']['P_%'] for r in rs]),
        'R_last': ms([r['chars_test']['last']['R_%'] for r in rs]),
        'F1_last': ms([r['chars_test']['last']['F1_%'] for r in rs]),
        'val_best': ms([r['chars_val_best_%'] for r in rs]),
        'best_epoch': [r['best_epoch'] for r in rs],
        'acc_clean_last': ms([pc(e, 'last', 'clean', 'acc') for e in ex]),
        'acc_clean_best': ms([pc(e, 'best', 'clean', 'acc') for e in ex]),
        'n_clean': get(ex[0], 'last', 'clean', 'n'),
        'digits_last': ms([pc(e, 'last', 'digits', 'acc') for e in ex]),
        'letters_last': ms([pc(e, 'last', 'letters', 'acc') for e in ex]),
        'letters_clean_last': ms([pc(e, 'last', 'letters_clean', 'acc') for e in ex]),
        'n_letters_clean': get(ex[0], 'last', 'letters_clean', 'n'),
        'words_last': ms([pc(e, 'last', 'region_words', 'acc') for e in ex]),
        'words_clean_n': get(ex[0], 'last', 'region_words_clean', 'n'),
        'fps': ms([get(r, 'speed', 'fps_bs1') for r in rs]),
        'train_min': ms([r['train_minutes'] for r in rs]),
    }
    for d, r, e in zip(ds, rs, ex):
        out['per_run'][os.path.basename(d)] = {'acc_last': r['chars_test']['last']['acc_%'], 'acc_best': r['chars_test']['best']['acc_%'],
                                               'clean_last': pc(e, 'last', 'clean', 'acc'), 'best_epoch': r['best_epoch'],
                                               'val_best': r['chars_val_best_%']}
au = json.load(open(os.path.join(RUNS, 'audit', 'results.json')))
out['audit'] = {k: au[k] for k in ('counts', 'classes_matching_table_I', 'share_32x32', 'test_copies', 'val_copies', 'copy_rule')}
out['audit']['test_per_class'] = au['duplicates']['test']['per_class']
out['audit']['test_byte_identical'] = au['duplicates']['test']['byte_identical']
out['audit']['val_byte_identical'] = au['duplicates']['val']['byte_identical']
out['audit']['table_I_mismatch'] = [r for r in au['counts_vs_table_I'] if not r['match']]
json.dump(out, open(os.path.join(HERE, 'check_local.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'per_run'}, indent=1))
print(json.dumps(out['per_run'], indent=1))
