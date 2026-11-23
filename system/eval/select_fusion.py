"""Step 2 of the V1/V2 build: choose each version's reader combination on VALIDATION only, then report it on test.

  python eval/select_fusion.py

For each pool (V1 = readers of papers 2, 11, 18, 20; V2 = readers of papers 27, 23, 34, 19, alone and on top of the
chosen V1 reader) it tries every ordered subset (primary reader first) with every rule of lpr.fusion, scores val,
picks the best val row (ties -> fewer ms), and only then looks at test. Also: the McNemar test of the chosen row
against the best single reader, the oracle ceiling, and how well the status levels separate right from wrong readings
(the always-on verifier = the best other reader).
Writes eval/results/fusion_selection.json and prints a summary.
"""
import csv
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from lpr import fusion, metrics, vocab  # noqa: E402

PRED = os.path.join(HERE, 'preds')
MAX_READERS = 5
RES = json.load(open(os.path.join(HERE, 'results', 'readers.json'), encoding='utf-8'))
POOLS = {
    'V1': ['p02_lpnr_v12n', 'p11_chars_v8n', 'p18_slot', 'p20_crnn_hr'],
    'V2_only': ['p27_chars_v8n', 'p27_chars_v10n', 'p23_rec_fair', 'p34_svtrv2_t', 'p34_svtrv2_s', 'p19_crnn'],
}


def load(name, split):
    with open(os.path.join(PRED, f'{name}_{split}.csv'), encoding='utf-8') as f:
        return {r['file']: r for r in csv.DictReader(f)}


def ms(name):
    return RES[name]['ms_per_plate_batch1']


def run(P, combo, rule, files):
    """Fused text per plate and the share of plates on which the helpers ran."""
    out, helped = [], 0
    for f in files:
        full = [(n, P[n][f]['pred'], float(P[n][f]['conf'])) for n in combo]
        k0 = 2 if rule == 'verify' else 1                    # readers that always run
        rd = full[:k0] + [None] * (len(combo) - k0)
        second = full[1][1] if rule == 'verify' else None
        if len(combo) > k0 and fusion.needs_helpers(rule, full[0][1], second):
            helped += 1
            rd = full
        out.append(fusion.fuse(rd, rule if len(combo) > 1 else 'single'))
    return out, helped / max(1, len(files))


def ok(texts, truth):
    return [vocab.norm_text(p) == vocab.norm_text(t) for p, t in zip(texts, truth)]


def search(pool, P, files, truth, fixed_primary=None):
    rows = []
    for k in range(1, min(MAX_READERS, len(pool)) + 1):
        for combo in itertools.permutations(pool, k):
            if fixed_primary and combo[0] != fixed_primary:
                continue
            lead = 2 if k > 2 else 1                          # verify: the second reader is special too
            if k > 2 and list(combo[lead:]) != sorted(combo[lead:]):
                continue                                      # helper order does not matter beyond the lead readers
            for rule in (('single',) if k == 1 else ('fallback', 'vote') if k == 2 else ('fallback', 'vote', 'verify')):
                r = {'readers': list(combo), 'rule': rule}
                for s in ('val', 'test'):
                    texts, share = run(P[s], combo, rule, files[s])
                    r[s] = metrics.score_rows(zip(texts, truth[s]))
                    r[f'helpers_{s}_%'] = round(100 * share, 2)
                    r[f'_ok_{s}'] = ok(texts, truth[s])
                    r[f'_texts_{s}'] = texts
                h = r['helpers_test_%'] / 100 if rule in ('fallback', 'verify') else 1.0
                k0 = 2 if rule == 'verify' else 1
                r['ms_per_plate'] = round(sum(ms(n) for n in combo[:k0]) + h * sum(ms(n) for n in combo[k0:]), 2)
                rows.append(r)
    return rows


def pick(rows):
    return max(rows, key=lambda r: (r['val']['plate_exact_%'], -r['ms_per_plate']))


def gate_table(row, P, files, truth, verifier):
    """Status levels on test for the chosen row with an always-on verifier reader."""
    texts, good = row['_texts_test'], row['_ok_test']
    lv = {'verified': [0, 0], 'likely': [0, 0], 'review': [0, 0]}
    for f, t, g in zip(files['test'], texts, good):
        combo = row['readers']
        rd = [(n, P['test'][n][f]['pred'], float(P['test'][n][f]['conf'])) for n in combo]
        if row['rule'] == 'fallback' and not fusion.needs_helpers('fallback', rd[0][1]):
            rd = rd[:1]
        if row['rule'] == 'verify' and not fusion.needs_helpers('verify', rd[0][1], rd[1][1]):
            rd = rd[:2]
        v = P['test'][verifier][f]['pred'] if verifier and verifier not in combo[:len(rd)] else None
        st = fusion.status(t, rd, v)
        lv[st][0] += 1
        lv[st][1] += int(g)
    n = len(files['test'])
    return {k: {'plates': a, 'share_%': round(100 * a / n, 1), 'exact_%': round(100 * b / max(1, a), 2)}
            for k, (a, b) in lv.items()}


def main():
    names = sorted({n for p in POOLS.values() for n in p})
    P = {s: {n: load(n, s) for n in names} for s in ('val', 'test')}
    files = {s: sorted(set.intersection(*[set(P[s][n]) for n in names])) for s in P}
    truth = {s: [P[s][names[0]][f]['true'] for f in files[s]] for s in P}
    out = {'plates': {s: len(files[s]) for s in files}}
    chosen = {}
    for ver, pool in POOLS.items():
        rows = search(pool, P, files, truth)
        best = pick(rows)
        single = pick([r for r in rows if r['rule'] == 'single'])
        chosen[ver] = (best, single, rows)
    # V2 = the readers of the chosen V1 + the V2 papers' readers (version 2 builds on version 1)
    pool = [n for n in chosen['V1'][0]['readers'] if n in ('p02_lpnr_v12n', 'p11_chars_v8n')] + POOLS['V2_only']
    rows = search(pool, P, files, truth)
    chosen['V2'] = (pick(rows), chosen['V1'][1], rows)
    for ver, (best, single, rows) in chosen.items():
        b, s = best['_ok_test'], single['_ok_test']
        fixed = sum(x and not y for x, y in zip(b, s))
        broken = sum(y and not x for x, y in zip(b, s))
        others = [n for n in (POOLS['V2_only'] if ver != 'V1' else POOLS['V1']) if n not in best['readers']]
        verifier = max(others, key=lambda n: RES[n]['val']['plate_exact_%']) if others else None
        top = sorted(rows, key=lambda r: (-r['val']['plate_exact_%'], r['ms_per_plate']))[:8]
        clean = lambda r: {k: v for k, v in r.items() if not k.startswith('_')}
        out[ver] = {'chosen_on_val': clean(best), 'best_single_on_val': clean(single),
                    'vs_best_single_test': {'fixed': fixed, 'broken': broken, 'p_mcnemar': metrics.mcnemar_exact(fixed, broken)},
                    'oracle_test_%': round(100 * sum(any(vocab.norm_text(P['test'][n][f]['pred']) == vocab.norm_text(t)
                                                         for n in best['readers']) for f, t in zip(files['test'], truth['test'])) / len(files['test']), 2),
                    'verifier': verifier, 'status_levels_test': gate_table(best, P, files, truth, verifier),
                    'top8_by_val': [clean(r) for r in top]}
        c = out[ver]['chosen_on_val']
        print(f"{ver}: {' + '.join(c['readers'])} [{c['rule']}] val {c['val']['plate_exact_%']} test {c['test']['plate_exact_%']} "
              f"char {c['test']['char_acc_%']} ms {c['ms_per_plate']} helpers {c['helpers_test_%']}% | single "
              f"{single['readers'][0]} test {single['test']['plate_exact_%']} | fixed/broken {fixed}/{broken} "
              f"p {out[ver]['vs_best_single_test']['p_mcnemar']} | verifier {verifier}: {out[ver]['status_levels_test']}")
        for r in top:
            print('   ', ' + '.join(r['readers']), r['rule'], r['val']['plate_exact_%'], r['test']['plate_exact_%'], r['ms_per_plate'])
    json.dump(out, open(os.path.join(HERE, 'results', 'fusion_selection.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)


if __name__ == '__main__':
    main()
