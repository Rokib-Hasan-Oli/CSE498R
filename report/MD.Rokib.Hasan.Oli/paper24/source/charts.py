"""Charts for the paper #24 study report, v1.1 update (static PDF, light mode). Run analysis.py first.

Numbers: the paper (arXiv 2309.04331v2, Tables 3-6) and analysis.json (the v1.1 Kaggle run of 22 Oct 2026, per-plate
predictions in `After Train/results.zip`; the v1.0 run of 20 Oct only as numbers, v1.0_run/analysis.json). Palette:
categorical slots 1-4, checked with the dataviz validator on 24 Oct (light: all checks pass; contrast WARN for slots 3-4
-> every bar is direct-labelled and every chart has a table in the report). Models keep one colour in every chart:
char-YOLOv8n = slot 1, CRNN = 2, slot CNN = 3, ViTSTR = 4. In "method" charts the highlight (our format-aware fallback)
is dark blue and everything else is grey.
"""
import json
import os
import shutil
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analysis import data_image  # noqa: E402

A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
LIGHT, OURS = '#c9c8c2', '#104281'
COL = {'chars_v8n': S1, 'crnn': S2, 'slot': S3, 'vitstr': S4}
NAME = {'chars_v8n': 'char-YOLOv8n', 'crnn': 'CRNN-CTC', 'slot': 'slot CNN', 'vitstr': 'ViTSTR-tiny'}
MODELS = A['models']
sg = A['singles']
kag = A['best_paper_rule']
fb = A['fallback_2']
SIGO = A['sig_fallback_2']
DET = A['error_detector']['format']
n = A['plates']['test']
ORA = A['oracle_test']['chars_v8n + crnn + slot + vitstr']

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.dpi': 220, 'legend.frameon': False,
})


def style(ax, axis='y'):
    for s in ('top', 'right', 'left' if axis == 'y' else 'bottom'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom' if axis == 'y' else 'left'].set_color(GRID)
    (ax.yaxis if axis == 'y' else ax.xaxis).grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)


def box(ax, x, y, w, h, text, fc='#ffffff', ec=INK2, fs=8, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.012', fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color,
            fontweight='bold' if bold else 'normal', linespacing=1.25)


def arrow(ax, a, b, color=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, lw=0.9, color=color, shrinkA=0, shrinkB=0))


# 1. How model fusion works (the paper) and what we add -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.6, 4.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.0, 0.97, "A. The paper's idea: read one plate crop with N different models, then combine the N strings",
        fontsize=9.5, fontweight='bold', color=INK, va='top')
box(ax, 0.0, 0.62, 0.11, 0.2, 'plate crop\n(from YOLOv4)', fc='#f4f3ef', fs=8)
rows = [('model 1', 'AIQ1056', '0.98', True), ('model 2', 'AIQ1Q56', '0.93', False), ('model 3', 'AIQ1056', '0.82', True),
        ('model 4', 'ATQ1056', '0.59', False)]
for i, (m, s, c, okk) in enumerate(rows):
    y = 0.835 - i * 0.075
    box(ax, 0.16, y - 0.028, 0.10, 0.056, m, fs=7.5)
    ax.text(0.28, y, f'"{s}"  ({c})', fontsize=7.8, va='center', color=INK if okk else S2, family='Consolas')
    arrow(ax, (0.11, 0.72), (0.16, y))
    arrow(ax, (0.415, y), (0.46, 0.72))
rules = [('HC', 'take the most confident string (even if only one model says it)'),
         ('MV', 'take the string most models agree on (sequence-level majority vote)'),
         ('MVCP', 'vote per character position, then join the winners'),
         ('ties', '-HC: most confident of the tied strings; -BM: the "best model"')]
for i, (k, v) in enumerate(rules):
    ax.text(0.47, 0.84 - i * 0.06, k, fontsize=8, fontweight='bold', color=INK, va='center')
    ax.text(0.53, 0.84 - i * 0.06, v, fontsize=8, color=INK2, va='center')
ax.text(0.47, 0.56, 'MV-HC result here: "AIQ1056" (2 votes beat 1 + 1); HC would also pick it (0.98).', fontsize=8,
        color=INK, va='center')
ax.text(0.0, 0.47, 'B. What we add for Bangla plates: the BRTA layout as a free error detector, and a second model only '
        'when it fires', fontsize=9.5, fontweight='bold', color=INK, va='top')
box(ax, 0.0, 0.12, 0.11, 0.22, f'plate crop\n(498R v1.1 test,\n{n} plates)', fc='#f4f3ef', fs=8)
box(ax, 0.15, 0.17, 0.14, 0.12, f"char-YOLOv8n\n{sg['chars_v8n']['ms']:.1f} ms", fc='#eaf2fc', ec=S1, fs=8, bold=True)
box(ax, 0.34, 0.15, 0.17, 0.16, 'BRTA layout OK?\ndistrict [Metro] - letter\n+ 6 digits', fs=7.6)
box(ax, 0.58, 0.27, 0.15, 0.09, f"yes ({100 - DET['flagged_%']:.0f} %): keep it", fs=7.8)
box(ax, 0.58, 0.05, 0.15, 0.13, f"no ({DET['flagged_%']:.0f} %): ask CRNN\n({sg['crnn']['ms']:.1f} ms), keep its\nreading if valid",
    fs=7.6)
box(ax, 0.79, 0.12, 0.2, 0.2, f"final text\n{fb['test']['plate_exact_%']:.1f} % exact (+{SIGO['gain_pp']:.1f} pp)\n"
    f"{fb['ms']:.1f} ms on average\n{SIGO['fixed']} fixed, {SIGO['broken']} broken", fc='#eaf2fc', ec=S1, fs=8, bold=True)
arrow(ax, (0.11, 0.23), (0.15, 0.23)); arrow(ax, (0.29, 0.23), (0.34, 0.23))
arrow(ax, (0.51, 0.25), (0.58, 0.31)); arrow(ax, (0.51, 0.20), (0.58, 0.12))
arrow(ax, (0.73, 0.31), (0.79, 0.25)); arrow(ax, (0.73, 0.12), (0.79, 0.18))
save(fig, 'fig_pipeline.png')

# 2. The paper's own numbers: Table 4 (top-N, four rules) and Table 6 (accuracy vs time) ---------------------------
topn = list(range(1, 13))
T4 = {'HC': [92.4, 94.1, 94.2, 95.2, 95.5, 96.0, 95.4, 95.7, 95.7, 95.5, 95.2, 95.0],
      'MV-HC': [92.4, 94.1, 94.9, 96.3, 96.6, 97.0, 97.2, 97.6, 97.5, 97.2, 97.0, 97.0],
      'MV-BM': [92.4, 92.4, 94.6, 95.9, 96.1, 97.1, 97.3, 97.5, 97.5, 97.4, 97.1, 97.0],
      'MVCP-HC': [92.4, 94.1, 94.2, 95.9, 96.2, 96.9, 97.0, 97.2, 97.5, 96.6, 96.5, 96.5]}
acc_rank = ([92.4, 94.1, 94.9, 96.3, 96.6, 97.0, 97.2, 97.6, 97.5, 97.2, 97.0, 97.0],
            [7.3, 14.4, 31.3, 36.6, 49.6, 52.6, 57.2, 59.7, 68.2, 84.2, 87.1, 89.4])
spd_rank = ([85.9, 90.2, 91.1, 95.4, 96.0, 96.6, 96.9, 96.9, 97.1, 97.1, 97.1, 97.1],
            [2.3, 4.9, 7.8, 10.8, 15.4, 20.7, 27.8, 35.0, 43.6, 56.6, 72.5, 89.4])
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
ax = axes[0]
for (k, v), c in zip(T4.items(), (S2, S1, S3, S4)):
    ax.plot(topn, v, color=c, lw=2, marker='o', ms=4, label=k, zorder=3)
style(ax); ax.set_ylim(91.5, 98.2); ax.set_xticks(topn)
ax.set_xlabel('number of fused models (top-N by accuracy)')
ax.set_title('Paper Table 4: mean recognition rate (%), 8 datasets', loc='left', fontsize=9)
ax.text(12.2, 95.0, 'HC', fontsize=7.5, color=INK2, va='center')
ax.legend(loc='lower right', ncol=2, fontsize=7.5, handlelength=1.2, bbox_to_anchor=(1.0, 0.04))
ax = axes[1]
ax.plot(acc_rank[1], acc_rank[0], color=S1, lw=2, marker='o', ms=4, label='models added by accuracy', zorder=3)
ax.plot(spd_rank[1], spd_rank[0], color=S2, lw=2, marker='s', ms=4, label='models added by speed', zorder=3)
style(ax); ax.set_ylim(85, 98.5)
ax.set_xlabel('fusion time per plate (ms, Quadro RTX 8000)')
ax.set_title('Paper Table 6: accuracy (MV-HC, %) vs time', loc='left', fontsize=9)
for i in (0, 3, 5):
    ax.annotate(f'{i + 1} fast', (spd_rank[1][i], spd_rank[0][i]), (spd_rank[1][i] + 6, spd_rank[0][i] - 1.6),
                fontsize=7.2, color=INK2, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
ax.annotate('best single\n(ViTSTR-Base)', (7.3, 92.4), (16, 88.6), fontsize=7.2, color=INK2,
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
ax.legend(loc='lower right', fontsize=7.5, handlelength=1.2)
fig.tight_layout(w_pad=2.5)
save(fig, 'fig_paper.png')


# 3. Our result: singles, the paper's fusion, our format-aware fallback, oracle (test) ------------------------------
def wil(p):
    import math
    k = round(p * n / 100)
    z = 1.96
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


short = {'chars_v8n': 'char-YOLO', 'crnn': 'CRNN', 'slot': 'slot', 'vitstr': 'ViTSTR'}
items = [(NAME[m], sg[m]['test']['plate_exact_%'], f"{sg[m]['ms']:.1f} ms", COL[m]) for m in reversed(MODELS)]
items += [(f"paper's best rule, val-chosen\n({' + '.join(short[m] for m in kag['models'])}, {kag['rule']})",
           kag['test']['plate_exact_%'], f"{kag['ms']:.1f} ms", MUTED),
          ('ours: char-YOLO + CRNN\nonly if BRTA layout fails', fb['test']['plate_exact_%'], f"{fb['ms']:.1f} ms avg", OURS),
          ('oracle: any of the 4 right\n(ceiling for any fusion)', ORA, '', None)]
fig, ax = plt.subplots(figsize=(9.6, 3.4))
for i, (lab, v, ms, c) in enumerate(items):
    lo, hi = wil(v)
    if c is None:
        ax.barh(i, v, 0.62, fc='none', ec=INK2, lw=1.0, ls='--', zorder=3)
    else:
        ax.barh(i, v, 0.62, color=c, zorder=3)
    ax.plot([lo, hi], [i, i], color=INK, lw=0.9, zorder=4)
    ax.text(hi + 0.8, i, f'{v:.2f} %' + (f'   ·   {ms}' if ms else ''), va='center', fontsize=8, color=INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items], fontsize=8)
style(ax, 'x'); ax.set_xlim(0, 105)
ax.set_xlabel(f'plates read exactly on the 498R v1.1 test split ({n} plates; line = 95 % confidence interval)   ·   '
              'ms per plate on a T4')
save(fig, 'fig_ours.png')

# 4. Where char-YOLO's errors go -----------------------------------------------------------------------------------
bw = A['best_wrong']
oc = bw['others_right_count']
fig, ax = plt.subplots(figsize=(9.6, 2.0))
segs = [('no other model reads it', oc['0'], LIGHT), ('1 other right', oc['1'], S2),
        ('2 others', oc['2'], S3), ('3 others', oc['3'], S4)]
x = 0
for k, (lab, v, c) in enumerate(segs):
    ax.barh(0, v - 0.3, 0.5, left=x, color=c, zorder=3)
    if v >= 12:
        ax.text(x + v / 2, 0, f'{lab}: {v}', ha='center', va='center', fontsize=8.5, color=INK)
    else:
        ax.text(x + v / 2 - 0.15, 0.3, f'{v}', ha='center', va='bottom', fontsize=8, color=INK)
    x += v
ax.text(oc['0'] + oc['1'] + (oc['2'] + oc['3']) / 2, -0.3, '2 / 3 others right', ha='center', va='top', fontsize=7.2,
        color=INK2)
style(ax, 'x'); ax.set_yticks([]); ax.set_xlim(0, bw['n'] + 1); ax.set_ylim(-0.75, 0.75)
ax.xaxis.grid(False)
ax.set_xticks(sorted({0, 20, 40, oc['0'], oc['0'] + oc['1'], bw['n']}))
ax.set_xlabel(f"the {bw['n']} test plates char-YOLOv8n reads wrongly (of {n}): could another model rescue them?")
save(fig, 'fig_overlap.png')

# 5. Is the confidence honest? accuracy inside each confidence bin -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), gridspec_kw={'width_ratios': [1.45, 1]})
ax = axes[0]
for m in MODELS:
    bins = [b for b in A['confidence'][m]['bins'] if b['n'] >= 10]
    xs = [(b['lo'] + b['hi']) / 2 for b in bins]
    ax.plot(xs, [b['acc_%'] for b in bins], color=COL[m], lw=2, marker='o', ms=4.5, label=NAME[m], zorder=3)
ax.plot([0.25, 1.0], [25, 100], color=MUTED, lw=0.8, ls='--', zorder=2)
ax.text(0.52, 58, 'honest: accuracy = confidence', fontsize=7.2, color=INK2, rotation=29)
style(ax); ax.set_xlim(0.2, 1.0); ax.set_ylim(-3, 103)
ax.set_xlabel('confidence the model reports (bins with ≥ 10 plates)')
ax.set_ylabel('plates right in that bin (%)')
ax.legend(loc='upper left', fontsize=7.5, handlelength=1.2)
ax.set_title('Same confidence, different meaning', loc='left', fontsize=9)
ax = axes[1]
vals = [A['confidence'][m]['wrong_with_conf_ge_0.9_%'] for m in MODELS]
bars = ax.bar(range(4), vals, 0.6, color=[COL[m] for m in MODELS], zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f} %', ha='center', fontsize=8, color=INK)
style(ax); ax.set_ylim(0, max(vals) + 12); ax.set_xticks(range(4))
ax.set_xticklabels([NAME[m] for m in MODELS], fontsize=7.6)
ax.set_title('Wrong readings with confidence ≥ 0.9', loc='left', fontsize=9)
fig.tight_layout(w_pad=2.5)
save(fig, 'fig_confidence.png')

# 6. Accuracy vs time, every subset (Table-6 style, ours) ---------------------------------------------------------
CODE = {'chars_v8n': 'Y', 'crnn': 'C', 'slot': 'S', 'vitstr': 'V'}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={'width_ratios': [1, 1.25]})
zx, zy = (6.5, 29.0), (85.6, 89.4)
for ax, zoom in zip(axes, (False, True)):
    for c in A['combos']:
        y = c['test']['plate_exact_%']
        if zoom and 'chars_v8n' not in c['models']:
            continue
        if c['rule'] == 'MV-BM':
            ax.scatter(c['ms'], y, s=26, color=MUTED, zorder=3, lw=0)
        if c['rule'] == 'MV-BM+format':
            ax.scatter(c['ms'], y, s=24, color=INK, marker='D', zorder=3, lw=0)
        if c['rule'] == 'fallback' and c['models'] != fb['models']:
            ax.scatter(c['ms'], y, s=30, facecolor='none', edgecolor=OURS, marker='^', zorder=3, lw=0.9)
        if zoom and ((c['rule'] == 'MV-BM' and len(c['models']) > 2) or c['rule'] == 'MV-BM+format'):
            code = '+'.join(CODE[m] for m in c['models'])
            ax.annotate(code, (c['ms'], y), (4, -2.5), textcoords='offset points', ha='left', fontsize=6.4, color=INK2)
    for m in MODELS:
        if zoom and m != 'chars_v8n':
            continue
        ax.scatter(sg[m]['ms'], sg[m]['test']['plate_exact_%'], s=60, color=COL[m], zorder=4, edgecolor='white', lw=1.5)
        ax.annotate(NAME[m] + (' alone' if zoom else ''), (sg[m]['ms'], sg[m]['test']['plate_exact_%']),
                    (0, -14) if zoom else (-7, -3), textcoords='offset points', fontsize=7.5, color=INK2,
                    ha='center' if zoom else 'right')
    ax.scatter(fb['ms'], fb['test']['plate_exact_%'], s=150, marker='*', color=OURS, zorder=5, edgecolor='white', lw=1)
    style(ax)
    ax.set_xlabel('ms per plate on a Kaggle T4 (batch 1; models in a row)')
axes[0].set_ylim(15, 95); axes[0].set_xlim(-3, 31)
axes[0].set_ylabel('test plates exact (%)')
axes[0].add_patch(plt.Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fc='none', ec=MUTED, lw=0.8, ls=':'))
axes[0].text(sum(zx) / 2, zy[0] - 2.5, 'zoom on the right', fontsize=7.2, color=INK2, ha='center', va='top')
axes[0].scatter([], [], s=26, color=MUTED, label='a subset, MV-BM (paper rule)')
axes[0].scatter([], [], s=24, color=INK, marker='D', label='same subset + BRTA filter')
axes[0].scatter([], [], s=30, facecolor='none', edgecolor=OURS, marker='^', label='same subset as a fallback')
axes[0].scatter([], [], s=110, color=OURS, marker='*', label='ours: CRNN only if layout fails')
axes[0].legend(loc='lower right', fontsize=7.2, handlelength=1.0)
ax = axes[1]
ax.set_xlim(*zx); ax.set_ylim(*zy)
ax.set_title('zoom: subsets that contain char-YOLO  (Y char-YOLO, C CRNN, S slot, V ViTSTR)', loc='left', fontsize=8)
ax.annotate(f"ours (Y+C)\n{fb['test']['plate_exact_%']:.2f} % at {fb['ms']:.1f} ms", (fb['ms'], fb['test']['plate_exact_%']),
            (14, 6), textcoords='offset points', ha='left', fontsize=7.5, color=INK,
            arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6))
fig.tight_layout(w_pad=2.0)
save(fig, 'fig_speed.png')

# 7. By condition and source: char-YOLO alone, paper fusion, ours, oracle -------------------------------------------
groups = [(f'all\n{n}', None)]
for key, lab in (('clear', 'clear'), ('night', 'night'), ('blurry', 'blurry')):
    groups.append((f"{lab}\n{A['by_condition'][key]['n']}", ('by_condition', key)))
for key, lab in (('kuet', 'KUET'), ('lpdba', 'LPDBA'), ('anpr', 'ANPR'), ('nightlpbd', 'NightLPBD'), ('blpd', 'BLPD')):
    groups.append((f"{lab}\n{A['by_source'][key]['n']}", ('by_source', key)))


def grp(g):
    if g is None:
        return {'chars_v8n': sg['chars_v8n']['test']['plate_exact_%'], 'fusion': kag['test']['plate_exact_%'],
                'cascade': fb['test']['plate_exact_%'], 'oracle': ORA}
    return A[g[0]][g[1]]


fig, ax = plt.subplots(figsize=(9.6, 3.1))
for i, (_, g) in enumerate(groups):
    d = grp(g)
    ax.plot([i, i], [d['chars_v8n'], d['cascade']], color=S1, lw=2, zorder=2, solid_capstyle='round')
    ax.scatter(i, d['oracle'], s=150, facecolor='none', edgecolor=INK2, lw=1.1, zorder=3)
    ax.scatter(i, d['chars_v8n'], s=42, color=LIGHT, edgecolor=INK2, lw=0.6, zorder=4)
    ax.scatter(i - 0.14, d['fusion'], s=34, color=MUTED, marker='s', zorder=4, lw=0)
    ax.scatter(i, d['cascade'], s=48, color=S1, zorder=5, edgecolor='white', lw=1)
    ax.text(i + 0.13, d['cascade'] + (0.5 if d['cascade'] == d['chars_v8n'] else 0), f"{d['cascade']:.1f}",
            va='center', fontsize=7.2, color=INK)
    if d['cascade'] != d['chars_v8n']:
        ax.text(i + 0.13, d['chars_v8n'], f"{d['chars_v8n']:.1f}", va='center', fontsize=6.8, color=INK2)
ax.scatter([], [], s=42, color=LIGHT, edgecolor=INK2, lw=0.6, label='char-YOLOv8n alone')
ax.scatter([], [], s=34, color=MUTED, marker='s', label="paper's best rule (val-chosen)")
ax.scatter([], [], s=48, color=S1, label='ours: format-aware fallback')
ax.scatter([], [], s=150, facecolor='none', edgecolor=INK2, lw=1.1, label='oracle (any model right)')
lo = min(min(grp(g)['chars_v8n'], grp(g)['fusion']) for _, g in groups)
style(ax); ax.set_ylim(lo - 3, 98)
ax.set_xlim(-0.5, len(groups) - 0.4)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g for g, _ in groups], fontsize=7.6)
ax.set_ylabel('plates exact (%)')
ax.legend(loc='upper center', ncol=4, fontsize=7.5, handlelength=1.0, bbox_to_anchor=(0.5, 1.14))
save(fig, 'fig_conditions.png')

# 8. Training curves of the three sequence models -----------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.5), sharey=False)
for ax, m in zip(axes, ('crnn', 'slot', 'vitstr')):
    log = A['curves'][m]
    ep = [x['epoch'] for x in log]
    ax.plot(ep, [x['train_loss'] for x in log], color=COL[m], lw=1.2, ls='--', label='train loss')
    ax.plot(ep, [x['val_loss'] for x in log], color=COL[m], lw=2, label='val loss')
    b = min(log, key=lambda x: x['val_loss'])
    ax.scatter([b['epoch']], [b['val_loss']], s=30, color=INK, zorder=4)
    ax.annotate(f"best val, epoch {b['epoch']}", (b['epoch'], b['val_loss']), (-4, 12), textcoords='offset points',
                fontsize=7.2, color=INK2, ha='right' if b['epoch'] > len(log) * 0.6 else 'left')
    style(ax)
    ax.set_title(f"{NAME[m]}: test exact {sg[m]['test']['plate_exact_%']:.1f} %", loc='left', fontsize=8.6)
    ax.set_xlabel('epoch')
    ax.set_ylim(0, max(x['val_loss'] for x in log[2:]) * 1.35)
axes[0].legend(loc='upper right', fontsize=7.2, handlelength=1.6)
fig.tight_layout(w_pad=1.6)
save(fig, 'fig_curves.png')

# 9. Why char-YOLO's readings are wrong (a plate can have several reasons) -----------------------------------------
rs = A['best_wrong_reasons']
order = ['more than 6 digits', 'two letters', 'no letter', 'two districts', 'fewer than 6 digits', 'no district',
         'valid layout, wrong symbol']
labels = ['more than 6 digits (a digit boxed twice)', 'two letters (one letter, two classes)', 'no letter found',
          'two districts', 'fewer than 6 digits', 'no district found', 'layout is valid, one symbol is wrong']
fig, ax = plt.subplots(figsize=(9.6, 2.5))
vals = [rs.get(k, 0) for k in order]
cols = [S1] * 6 + [MUTED]
ax.barh(range(len(order)), vals, 0.62, color=cols, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 0.5, i, str(v), va='center', fontsize=8, color=INK)
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=8); ax.invert_yaxis()
style(ax, 'x'); ax.set_xlim(0, max(vals) + 6)
ax.set_xlabel(f"char-YOLOv8n wrong plates by reason (blue = the BRTA layout check catches it: {DET['flagged']} of "
              f"{bw['n']} plates)")
save(fig, 'fig_errors.png')

# 10. v1.0 -> v1.1: the same models on the 632 unchanged plates and on the 19 re-cut BLPD rows --------------------
cmp = A['v10_vs_v11']
lab = {'char-YOLOv8n': ('char-YOLOv8n', S1), 'CRNN-CTC': ('CRNN-CTC', S2), 'slot CNN': ('slot CNN', S3),
       'ViTSTR-tiny': ('ViTSTR-tiny', S4), "paper's best rule (val-chosen)": ("paper's best rule", MUTED),
       'format fallback (char-YOLO + CRNN)': ('ours: format fallback', OURS),
       'oracle (any of the 4 right)': ('oracle (any of the 4)', INK2)}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), sharey=True)
for ax, (k10, k11, title) in zip(axes, (('v10_632', 'v11_632', f"{A['unchanged_632']['n']} test plates that did not change"),
                                         ('v10_19', 'v11_19', f"{A['recut_19']['n']} BLPD test rows: whole photo (v1.0) → plate crop (v1.1)"))):
    for i, r in enumerate(cmp):
        name, c = lab[r['what']]
        a, b = r[k10], r[k11]
        ax.plot([a, b], [i, i], color=GRID, lw=3, zorder=2, solid_capstyle='round')
        ax.scatter(a, i, s=40, facecolor='white', edgecolor=INK2, lw=1, zorder=3)
        ax.scatter(b, i, s=46, color=c, zorder=4, edgecolor='white', lw=1)
        right = b >= a
        ax.text(b + (2.4 if right else -2.4), i, f'{b:.1f}', va='center', ha='left' if right else 'right', fontsize=7.4,
                color=INK)
        ax.text(a + (-2.4 if right else 2.4), i, f'{a:.1f}', va='center', ha='right' if right else 'left', fontsize=6.8,
                color=MUTED)
    style(ax, 'x'); ax.set_xlim(-8, 108)
    ax.set_title(title, loc='left', fontsize=8.6)
    ax.set_xlabel('plates exact (%)   ·   hollow = v1.0 run, filled = v1.1 run')
axes[0].set_yticks(range(len(cmp))); axes[0].set_yticklabels([lab[r['what']][0] for r in cmp], fontsize=8)
axes[0].invert_yaxis()
fig.tight_layout(w_pad=1.5)
save(fig, 'fig_v10_v11.png')

# 11. BLPD before / after: v1.0 whole photo, v1.1 plate crop (6 of the 19 re-cut test rows) ------------------------
rr = A['recut_rows']
pick = [r for r in rr if r['preds']['chars_v8n'][1]][:4] + [r for r in rr if not r['preds']['chars_v8n'][1]][:2]
top, bot = [], []
for r in pick:
    old = cv2.resize(data_image('ocr/images/test/' + r['old'], v10=True), (200, 200))
    new = data_image('ocr/images/test/' + r['file'])
    s = min(190 / new.shape[1], 120 / new.shape[0])
    new = cv2.resize(new, (max(1, int(new.shape[1] * s)), max(1, int(new.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((130, 200, 3), 255, np.uint8)
    y0, x0 = (130 - new.shape[0]) // 2, (200 - new.shape[1]) // 2
    canvas[y0:y0 + new.shape[0], x0:x0 + new.shape[1]] = new
    top.append(np.pad(old, ((3, 3), (3, 3), (0, 0)), constant_values=255))
    bot.append(np.pad(canvas, ((3, 3), (3, 3), (0, 0)), constant_values=255))
sheet = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
fig, ax = plt.subplots(figsize=(9.6, 3.0))
ax.imshow(cv2.cvtColor(sheet, cv2.COLOR_BGR2RGB)); ax.axis('off')
ax.text(-8, 103, 'v1.0:\nwhole photo', ha='right', va='center', fontsize=8, color=INK)
ax.text(-8, 206 + 68, 'v1.1:\nplate crop', ha='right', va='center', fontsize=8, color=INK)
for j, r in enumerate(pick):
    okk = r['preds']['chars_v8n'][1]
    ax.text(j * 206 + 103, sheet.shape[0] + 14, 'char-YOLO: right' if okk else 'char-YOLO: wrong', ha='center',
            va='top', fontsize=7.4, color=INK if okk else INK2, fontweight='normal' if okk else 'bold')
save(fig, 'fig_recut.png')

# 12. example crops for the HTML tables (resized copies; Bangla text is set in the PDF, not in the image) ----------
ex_dir = os.path.join(HERE, 'ex')
shutil.rmtree(ex_dir, ignore_errors=True)       # the v1.0 examples are kept in v1.0_run/ex
os.makedirs(ex_dir)
ex_files = set()
for k in ('fallback_fixed', 'paper_broken', 'paper_fixed', 'all_wrong', 'missed'):
    for e in A['examples'][k][:16]:
        ex_files.add(e['file'])
for f in ex_files:
    im = data_image('ocr/images/test/' + f)
    h, w_ = im.shape[:2]
    s = 46 / h
    im = cv2.resize(im, (max(1, int(w_ * s * 2)), 92), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(ex_dir, f), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
for f in ('fig_blpd.png',):                     # v1.0-only figure (kept in v1.0_run/)
    if os.path.exists(os.path.join(HERE, f)):
        os.remove(os.path.join(HERE, f))
print('charts done', len(ex_files), 'example crops')
