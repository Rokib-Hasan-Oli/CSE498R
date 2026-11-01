"""Paper #24 on our data: fuse the outputs of several OCR models, plate by plate.

  python p24_fusion.py --runs runs/chars_v8n runs/slot_ocr runs/crnn runs/vitstr --name fusion --project runs

Reads ocr_preds_val.csv and ocr_preds_test.csv (file, true, pred, conf, source) written by rec.py, seq.py, ved.py,
p22_vlm.py (lora) and p31_parseq.py; a run without both files is skipped with a message. For every subset of 2..N models
and every rule it scores plate exact / CER / token accuracy:
  HC, MV-HC, MV-BM, MVCP-HC : the paper's rules (metrics.fuse)
  MV-BM+format              : ours; drop readings that break the BRTA layout (district, [মেট্রো], letter, 6 digits),
                              then MV-BM; MV-BM on all readings if none fits
  fallback                  : ours; the subset's val-best model alone when its reading fits the layout, otherwise
                              MV-BM+format over it and the other models (the helpers), so the helpers only run on
                              the plates that need them: ms = best ms + share of plates x helper ms
MV-BM's tie-break model, the fallback's first model and every "best" row are chosen on VALIDATION, then reported on
test (no peeking at test); ties on val go to the faster row. Each row also has the exact McNemar test against the
val-best single model on the test plates (fixed / broken / p). Also: the oracle (any model right) = the ceiling for
fusion, and summed ms per plate (models run one after another).
"""
import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import metrics
import vocab

PAPER_RULES = ('HC', 'MV-HC', 'MV-BM', 'MVCP-HC')
RULES = PAPER_RULES + ('MV-BM+format', 'fallback')


def load(run, split):
    with open(Path(run) / f'ocr_preds_{split}.csv', encoding='utf-8') as f:
        return {r['file']: r for r in csv.DictReader(f)}


def usable(run):
    missing = [f'ocr_preds_{s}.csv' for s in ('val', 'test') if not (Path(run) / f'ocr_preds_{s}.csv').exists()]
    if missing:
        print(f'SKIP {run}: no {" / ".join(missing)} (runs from an older kit wrote only the test file: '
              f're-run them with the current kit to fuse them)', flush=True)
    return not missing


def ms_of(run):
    f = Path(run) / 'results.json'
    if not f.exists():
        return None
    r = json.load(open(f, encoding='utf-8'))
    return (r.get('ocr_test') or {}).get('ms_per_plate')


def conf(r):
    try:
        return float(r.get('conf') or 0)
    except ValueError:
        return 0.0


def fuse_all(preds, names, files, rule, best):
    """Fused text per plate, and the share of plates that ran more than the first model (fallback only)."""
    out, called = [], 0
    for f in files:
        if rule == 'fallback' and metrics.is_valid_plate(preds[best][f]['pred']):
            out.append(preds[best][f]['pred'])
            continue
        called += 1
        ps = [(n, preds[n][f]['pred'], conf(preds[n][f])) for n in names]
        out.append(metrics.fuse(ps, 'MV-BM+format' if rule == 'fallback' else rule, best))
    return out, called / max(1, len(files))


def right(texts, truth):
    return [vocab.norm_text(p or '') == vocab.norm_text(t) for p, t in zip(texts, truth)]


def vs_single(fused_ok, single_ok):
    """Exact McNemar test of a fusion against one model on the same plates."""
    b = sum(f and not s for f, s in zip(fused_ok, single_ok))
    c = sum(s and not f for f, s in zip(fused_ok, single_ok))
    return {'fixed': b, 'broken': c, 'net': b - c, 'p_mcnemar': metrics.mcnemar_exact(b, c)}


def ms_key(c):
    return c['ms_per_plate'] if c['ms_per_plate'] is not None else float('inf')


def pick(rows):
    """Best on val; a tie goes to the faster row."""
    return max(rows, key=lambda c: (c['val']['plate_exact_%'], -ms_key(c))) if rows else None


def describe(c, best_val):
    if c is None:
        return ''
    v = c['vs_best_single']
    s = (f"{' + '.join(c['models'])}, {c['rule']}: val {c['val']['plate_exact_%']} %, test {c['test']['plate_exact_%']} % "
         f"({v['gain_pp']:+.2f} pp vs {best_val}; fixed {v['fixed']}, broken {v['broken']}, exact McNemar p = {v['p_mcnemar']}), "
         f"{c['ms_per_plate']} ms per plate")
    if c['rule'] == 'fallback':
        s += f" (helpers run on {c['helpers_called_test_%']} % of the test plates)"
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--name', default='fusion')
    p.add_argument('--project', required=True)
    a = p.parse_args()
    runs = [r for r in a.runs if usable(r)]
    if len(runs) < 2:
        sys.exit(f'p24_fusion: need at least 2 runs with ocr_preds_val.csv and ocr_preds_test.csv, found {len(runs)}')
    names = []
    for r in runs:                                     # e.g. crnn from #24 and crnn from #31 -> crnn, crnn_2
        n, k = Path(r).name, 2
        while n in names:
            n, k = f'{Path(r).name}_{k}', k + 1
        names.append(n)
    P = {s: {n: load(r, s) for n, r in zip(names, runs)} for s in ('val', 'test')}
    files = {s: sorted(set.intersection(*[set(P[s][n]) for n in names])) for s in P}
    truth = {s: [P[s][names[0]][f]['true'] for f in files[s]] for s in P}
    single = {s: {n: metrics.score_rows(zip([P[s][n][f]['pred'] for f in files[s]], truth[s])) for n in names} for s in P}
    best_val = max(names, key=lambda n: single['val'][n]['plate_exact_%'])
    best_ok = right([P['test'][best_val][f]['pred'] for f in files['test']], truth['test'])
    res = {'run': a.name, 'kind': 'fusion', 'models': names, 'plates': {s: len(files[s]) for s in P},
           'skipped_runs': [r for r in a.runs if r not in runs],
           'single': single, 'best_single_on_val': best_val, 'rules': list(RULES),
           'ms_per_plate': {n: ms_of(r) for n, r in zip(names, runs)}, 'combos': []}
    for k in range(2, len(names) + 1):
        for combo in itertools.combinations(names, k):
            bm = max(combo, key=lambda n: single['val'][n]['plate_exact_%'])
            ms = [res['ms_per_plate'][n] for n in combo]
            ms_sum = round(sum(ms), 2) if all(m is not None for m in ms) else None
            for rule in RULES:
                row = {'models': list(combo), 'rule': rule}
                for s in ('val', 'test'):
                    texts, share = fuse_all(P[s], combo, files[s], rule, bm)
                    row[s] = metrics.score_rows(zip(texts, truth[s]))
                    if rule == 'fallback':
                        row[f'helpers_called_{s}_%'] = round(100 * share, 2)
                    if s == 'test':
                        row['vs_best_single'] = vs_single(right(texts, truth['test']), best_ok)
                        row['vs_best_single']['gain_pp'] = round(row['test']['plate_exact_%'] -
                                                                 single['test'][best_val]['plate_exact_%'], 2)
                row['ms_per_plate_sum'] = ms_sum
                row['ms_per_plate'] = ms_sum
                if rule == 'fallback':
                    row['first_model'] = bm
                    row['ms_per_plate'] = (round(res['ms_per_plate'][bm] + row['helpers_called_test_%'] / 100 *
                                                 (ms_sum - res['ms_per_plate'][bm]), 2) if ms_sum is not None else None)
                res['combos'].append(row)
    for s in ('val', 'test'):
        res[f'oracle_any_model_right_{s}_%'] = round(100 * sum(
            any(vocab.norm_text(P[s][n][f]['pred']) == vocab.norm_text(P[s][n][f]['true']) for n in names)
            for f in files[s]) / max(1, len(files[s])), 2)
    best = pick(res['combos'])
    res['best_fusion_chosen_on_val'] = best
    res['best_paper_rule_chosen_on_val'] = pick([c for c in res['combos'] if c['rule'] in PAPER_RULES])
    res['best_fallback_chosen_on_val'] = pick([c for c in res['combos'] if c['rule'] == 'fallback'])
    res['gain_over_best_single_test_pp'] = round(best['test']['plate_exact_%'] - single['test'][best_val]['plate_exact_%'], 2)
    res['mcnemar_vs_best_single'] = best['vs_best_single']
    common.save_json(res, Path(a.project) / a.name / 'results.json')

    def line(models, rule, val, test, cer, vs, ms):
        return f'| {models} | {rule} | {val} | {test} | {cer} | {vs} | {ms} |'
    lines = [f'Test plates: {len(files["test"])}, val plates: {len(files["val"])}. "vs {best_val}": the val-best single '
             f'model; fixed / broken = test plates the row gets right / wrong where {best_val} does not, p = exact McNemar. '
             f'ms: summed per plate; fallback = average (helpers only run on some plates).', '',
             line('models', 'rule', 'val exact %', 'test exact %', 'test CER %', f'fixed / broken vs {best_val} (p)', 'ms/plate'),
             '|---|---|---|---|---|---|---|']
    for n in names:
        lines.append(line(n, 'single', single['val'][n]['plate_exact_%'], single['test'][n]['plate_exact_%'],
                          single['test'][n]['CER_%'], '', res['ms_per_plate'][n]))
    shown = sorted(res['combos'], key=lambda c: (-c['val']['plate_exact_%'], ms_key(c)))[:15]
    for c in (res['best_paper_rule_chosen_on_val'], res['best_fallback_chosen_on_val']):
        if c not in shown:
            shown.append(c)
    for c in shown:
        v = c['vs_best_single']
        ms = c['ms_per_plate'] if c['rule'] != 'fallback' else f"{c['ms_per_plate']} (helpers on {c['helpers_called_test_%']} %)"
        lines.append(line(' + '.join(c['models']), c['rule'], c['val']['plate_exact_%'], c['test']['plate_exact_%'],
                          c['test']['CER_%'], f"{v['fixed']} / {v['broken']} (p {v['p_mcnemar']})", ms))
    lines += ['', f'- Best single on val: {best_val}, test {single["test"][best_val]["plate_exact_%"]} %.',
              f'- Chosen on val (all rules): {describe(best, best_val)}.',
              f'- Best of the paper\'s rules: {describe(res["best_paper_rule_chosen_on_val"], best_val)}.',
              f'- Best fallback: {describe(res["best_fallback_chosen_on_val"], best_val)}.',
              f'- Oracle (any model right): test {res["oracle_any_model_right_test_%"]} %.']
    if res['skipped_runs']:
        lines.append(f'- Skipped (no val predictions): {", ".join(res["skipped_runs"])}.')
    (Path(a.project) / a.name / 'fusion_table.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('DONE', a.name, 'best on val:', best['models'], best['rule'], '-> test', best['test'], best['vs_best_single'])


if __name__ == '__main__':
    main()
