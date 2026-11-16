"""Charts for the paper #19 study report (v1.1 run, static PDF, light mode).

  D:/498R/_dataset_build/venv/Scripts/python.exe charts.py [figure names ...]

Run order: local_check.py -> run_local_train.sh -> analysis.py -> charts.py -> build_report.py. Numbers come from
analysis.json; example crops from the dataset (D:/498R/dataset/ocr/images/test).
Palette (validated with the dataviz validator, light): gates = categorical slots in fixed order (OCR confidence blue,
legibility classifier orange, free signals aqua, combination yellow; oracle / random neutral grey, dashed); the four
legibility classes = one ordinal blue ramp (perfect light -> illegible dark). Contrast WARN for aqua / yellow -> every
line and bar carries a direct label.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
A = json.load(open(OUT / 'analysis.json', encoding='utf-8'))
DATA = Path(r'D:\498R\dataset')
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
NEUTRAL, NEUTRAL2 = '#a3a29d', '#6b6a66'
ORD = {'perfect': '#86b6ef', 'good': '#3987e5', 'poor': '#1c5cab', 'illegible': '#0d366b'}
CL = ['perfect', 'good', 'poor', 'illegible']
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
plt.rcParams.update({
    'font.family': 'Segoe UI', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
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
    if name.endswith('.jpg'):
        fig.savefig(OUT / name, pil_kwargs={'quality': 88})
    else:
        fig.savefig(OUT / name)
    plt.close(fig)


def box(ax, x, y, w, h, title, sub, fc='#f4f8fd', ec=S1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=1.2', fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h * 0.68, title, ha='center', va='center', fontsize=8.2, weight='bold', color=INK)
    ax.text(x + w / 2, y + h * 0.3, sub, ha='center', va='center', fontsize=6.8, color=INK2, linespacing=1.25)


def arrow(ax, a, b, c=INK2, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=c, lw=1.0, linestyle=ls))


# ------------------------------------------------------------------------------------------------ pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')
    R = A['run']
    box(ax, 0.5, 30, 18, 14, 'OCR train crops', f"5,607 (human + auto)\nsplit by plate text:\nhalf A {R['halves']['0']:,} · B {R['halves']['1']:,}")
    box(ax, 22.5, 30, 24, 14, 'Cross-fitted readers', 'CRNN + slot CNN trained\non one half read the OTHER\nhalf; val / test: half-A pair')
    box(ax, 50.5, 30, 23, 14, '4 proxy labels', 'perfect: both readers exact\ngood: one exact · poor: best\nwithin 1 token · illegible: rest', fc='#eef5ff')
    box(ax, 77.5, 30, 22, 14, 'ResNet-50 classifier', '96 x 192, Adam 1e-5,\nbatch 64, 4 classes or\nQuality Filter, EMA vs CE', fc='#fff3ee', ec=S2)
    for a, b in (((18.5, 37), (22.5, 37)), ((46.5, 37), (50.5, 37)), ((73.5, 37), (77.5, 37))):
        arrow(ax, a, b)
    box(ax, 1, 4, 22, 13, 'Test crop (651)', 'one plate crop from the\ndetector (here: the\nhuman-cut OCR test crop)', fc='#f6f6f4', ec=NEUTRAL)
    box(ax, 30, 12, 26, 11, 'Deployment CRNN', 'all train crops, 73.43 % exact\n= Model Fusion study file (same SHA-256)')
    box(ax, 30, 0.5, 26, 10, 'Legibility classifier (above)', 'P(perfect) + P(good)\n= "suitable" score', fc='#fff3ee', ec=S2)
    box(ax, 63, 4, 36, 15, 'Reject-or-read gate', 'sort test plates by a score, accept the top\n100 / 90 / 80 / 70 / 50 %: plate exact of the\naccepted plates (selective accuracy), AURC', fc='#effaf4', ec=S3)
    arrow(ax, (23, 13), (30, 17.5)); arrow(ax, (23, 8), (30, 5.5))
    arrow(ax, (56, 17.5), (63, 13), c=S1); arrow(ax, (56, 5.5), (63, 9), c=S2)
    ax.text(59.5, 18.6, 'confidence', fontsize=6.6, color=S1, ha='center')
    ax.text(59.5, 3.0, 'suitability', fontsize=6.6, color=S2, ha='center')
    save(fig, 'fig_pipeline.png')


# ------------------------------------------------------------------------------------------------ labels
def fig_labels():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={'width_ratios': [1.25, 1]})
    L = A['run']['labels']
    ls = A['train_labels_by_label_source']
    groups = [('train\nhuman', ls['human']), ('train\nauto', ls['auto']), ('val', L['val']), ('test', L['test'])]
    x = np.arange(len(groups))
    for i, (g, d) in enumerate(groups):
        tot = sum(d.get(c, 0) for c in CL)
        bottom = 0
        for c in CL:
            v = 100 * d.get(c, 0) / tot
            a1.bar(i, v, bottom=bottom, color=ORD[c], width=0.62, edgecolor=SURF, linewidth=1.5, zorder=3)
            if v > 6:
                a1.text(i, bottom + v / 2, f'{v:.0f}', ha='center', va='center', fontsize=7.2,
                        color=INK if c == 'perfect' else 'white')
            bottom += v
        a1.text(i, 101.5, f'n={tot:,}', ha='center', va='bottom', fontsize=7, color=INK2)
    a1.set_xticks(x, [g for g, _ in groups]); a1.set_ylim(0, 108); a1.set_ylabel('share of crops (%)')
    a1.set_title('A. Proxy labels per split', loc='left')
    for c in CL:
        a1.bar(0, 0, color=ORD[c], label=c)
    a1.legend(ncol=4, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), handlelength=1)
    style(a1)
    D = A['deploy_by_label']
    v = [D[c]['exact_%'] for c in CL]
    a2.bar(np.arange(4), v, color=[ORD[c] for c in CL], width=0.62, edgecolor=SURF, linewidth=2, zorder=3)
    for i, c in enumerate(CL):
        a2.text(i, v[i] + 1.5, f"{v[i]:.1f}", ha='center', va='bottom', fontsize=8, color=INK)
        a2.text(i, 3, f"n={D[c]['n']}", ha='center', va='bottom', fontsize=6.8, color=INK if c == 'perfect' else 'white')
    a2.axhline(72, color=S2, lw=1, ls=(0, (3, 2)))
    a2.text(3.45, 73.5, 'LPLCv1 "poor": 72 %\n(paper p.4)', fontsize=6.6, color=INK2, ha='right', va='bottom')
    a2.set_xticks(range(4), CL); a2.set_ylim(0, 112); a2.set_ylabel('plate exact (%)')
    a2.set_title('B. Deployment CRNN, test, per label', loc='left')
    style(a2)
    fig.tight_layout()
    save(fig, 'fig_labels.png')


# ------------------------------------------------------------------------------------------------ training curves
def fig_training():
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.7))
    C = A['run']['classifiers']
    for ax, names, title in ((axs[0], ('cls_b4_ema', 'cls_b4_ce'), 'A. 4 classes: validation accuracy'),
                             (axs[1], ('cls_qf_ema', 'cls_qf_ce'), 'B. Quality Filter: validation accuracy')):
        for n, col in zip(names, (S1, S2)):
            c = np.array(C[n]['curve'])
            ax.plot(c[:, 0], c[:, 2], color=col, lw=2, label=('EMA loss' if 'ema' in n else 'cross-entropy'))
            b = C[n]['best_epoch']
            ax.plot([b], [c[b - 1, 2]], 'o', ms=6, color=col, mec=SURF, mew=1.5, zorder=5)
            ax.text(c[-1, 0] + 0.4, c[-1, 2], f"{'EMA' if 'ema' in n else 'CE'}", fontsize=7.2, color=INK2, va='center')
        ax.set_xlabel('epoch'); ax.set_title(title, loc='left'); style(ax)
        ax.set_xlim(0, 27)
    axs[0].set_ylabel('val accuracy (%)')
    fig.tight_layout()
    save(fig, 'fig_training.png')


# ------------------------------------------------------------------------------------------------ confusion
def fig_confusion():
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
    ours = np.array(A['run']['classifiers']['cls_b4_ema']['test']['confusion'])
    paper = np.array([[1980, 101, 1, 0], [167, 1235, 65, 0], [0, 89, 797, 68], [0, 3, 42, 527]])
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for ax, m, t in ((axs[0], paper, 'A. Paper, one EMA fold (Fig. 4, p.5)'), (axs[1], ours, 'B. Ours, EMA, test (651 crops)')):
        pct = 100 * m / m.sum(1, keepdims=True)
        ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{pct[i, j]:.0f}%\n({m[i, j]})', ha='center', va='center', fontsize=7,
                        color='white' if pct[i, j] > 55 else INK)
        ax.set_xticks(range(4), CL, fontsize=7.5); ax.set_yticks(range(4), CL, fontsize=7.5)
        ax.set_xlabel('predicted'); ax.set_title(t, loc='left', fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    axs[0].set_ylabel('true label (row %)')
    fig.tight_layout()
    save(fig, 'fig_confusion.png')


# ------------------------------------------------------------------------------------------------ gate
def rc(ax, curves, spec, n):
    cov = 100 * np.arange(1, n + 1) / n
    for k, (lab, col, ls) in spec.items():
        y = np.array(curves[k])
        ax.plot(cov, y, color=col, lw=2 if ls == '-' else 1.3, ls=ls, label=lab)
    ax.set_xlim(100, 30); ax.set_xlabel('coverage: share of plates accepted (%)')
    style(ax)


def fig_gate():
    G, GC = A['gate'], A['gate_curves']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 4.0), gridspec_kw={'width_ratios': [1.45, 1]})
    spec = {'oracle': ('oracle (perfect gate)', NEUTRAL2, (0, (1.5, 1.5))),
            'learned (val): conf + layout + classifiers': ('conf + layout + classifiers (val-fitted)', S4, '-'),
            'conf + layout': ('OCR confidence + BRTA layout check', S3, '-'),
            'ocr_confidence': ('OCR confidence (CRNN)', S1, '-'),
            'cls_b4_ce': ('legibility classifier (4-class, CE)', S2, '-'),
            'cls_qf_ema': ('Quality Filter (EMA)', S2, (0, (4, 2))),
            'random': ('random', NEUTRAL, (0, (1.5, 1.5)))}
    rc(a1, GC, spec, len(GC['ocr_confidence']))
    a1.axvline(80, color=GRID, lw=1)
    a1.set_ylim(68, 101); a1.set_ylabel('plate exact of accepted plates (%)')
    a1.set_title('A. Risk-coverage on the 651 test plates', loc='left')
    a1.legend(fontsize=6.6, loc='upper center', bbox_to_anchor=(0.62, -0.2), ncol=2, handlelength=2.2)
    for k, dy in (('ocr_confidence', -1.3), ('cls_b4_ce', -1.6), ('learned (val): conf + layout + classifiers', 1.6)):
        a1.text(79, G[k]['80%'] + dy, f"{G[k]['80%']:.1f}", fontsize=7.2, color=INK, va='center')
    keys = ['random', 'cls_qf_ce', 'cls_qf_ema', 'cls_b4_ema', 'cls_b4_ce', 'ocr_confidence', 'conf + layout',
            'learned (val): conf + layout + classifiers']
    labs = ['random', 'QF, CE', 'QF, EMA', '4-class, EMA', '4-class, CE', 'OCR confidence', 'conf + layout', 'conf + layout\n+ classifiers']
    cols = [NEUTRAL, S2, S2, S2, S2, S1, S3, S4]
    v = [G[k]['AUROC'] for k in keys]
    y = np.arange(len(keys))
    a2.barh(y, v, color=cols, height=0.62, edgecolor=SURF, linewidth=2, zorder=3)
    for i, x in enumerate(v):
        a2.text(x + 0.008, i, f'{x:.3f}', va='center', fontsize=7.4, color=INK)
    a2.set_yticks(y, labs, fontsize=7.2); a2.set_xlim(0.4, 0.95)
    a2.set_xlabel('AUROC (0.5 = chance, 1 = perfect)')
    a2.set_title('B. Spotting CRNN errors', loc='left')
    style(a2, 'x')
    fig.tight_layout()
    save(fig, 'fig_gate.png')


def fig_strong():
    S, SCV = A['strong'], A['strong_curves']
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    spec = {'oracle': ('oracle', NEUTRAL2, (0, (1.5, 1.5))),
            'free signals + classifiers (val-fitted)': ('free signals + classifiers', S4, '-'),
            'free signals (val-fitted)': ('free signals: confidences + layout + agreement', S3, '-'),
            'char-YOLO confidence': ('char-YOLO confidence alone', S1, '-'),
            'cls_b4_ce': ('legibility classifier (4-class, CE)', S2, '-')}
    rc(ax, SCV, spec, len(SCV['oracle']))
    ax.axvline(80, color=GRID, lw=1)
    ax.set_ylim(85, 100.8); ax.set_ylabel('plate exact of accepted (%)')
    ax.set_title(f"Model Fusion study's best reader (char-YOLO + CRNN fallback, {S['fallback_exact_%']:.2f} % at full coverage)", loc='left')
    ax.legend(fontsize=6.8, loc='lower right', handlelength=2.2)
    for k, dy in (('free signals (val-fitted)', 0.5), ('cls_b4_ce', -0.6)):
        ax.text(79, S[k]['80%'] + dy, f"{S[k]['80%']:.1f}", fontsize=7.2, color=INK, va='center')
    fig.tight_layout()
    save(fig, 'fig_strong.png')


def fig_conditions():
    B = A['by_condition']
    conds = ['clear', 'blurry', 'night']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.9))
    x = np.arange(3)
    w = 0.27
    for j, (g, col, lab) in enumerate((('ocr_confidence', S1, 'OCR confidence'), ('cls_b4_ce', S2, 'legibility classifier'))):
        rej = [100 * B[c][f'rejected_by_{g}'] / B[c]['n'] for c in conds]
        a1.bar(x + (j - 0.5) * 0.36, rej, width=0.36, color=col, edgecolor=SURF, linewidth=2, zorder=3, label=lab)
        for i, vv in enumerate(rej):
            a1.text(x[i] + (j - 0.5) * 0.36, vv + 1, f'{vv:.0f}', ha='center', va='bottom', fontsize=7.2, color=INK)
    for j, (key, col, lab) in enumerate(((None, NEUTRAL, 'no gate (all plates)'), ('ocr_confidence', S1, 'OCR confidence'),
                                         ('cls_b4_ce', S2, 'legibility classifier'))):
        v = [B[c]['exact_%'] if key is None else B[c][f'accepted_exact_{key}_%'] for c in conds]
        a2.bar(x + (j - 1) * w, v, width=w, color=col, edgecolor=SURF, linewidth=2, zorder=3, label=lab)
        for i, vv in enumerate(v):
            a2.text(x[i] + (j - 1) * w, vv + 1, f'{vv:.0f}', ha='center', va='bottom', fontsize=7, color=INK)
    lab = [f"{c}\n(n={B[c]['n']})" for c in conds]
    a1.set_xticks(x, lab); a2.set_xticks(x, lab)
    a1.set_ylabel('% of these plates rejected'); a1.set_ylim(0, 65)
    a2.set_ylabel('plate exact (%)'); a2.set_ylim(0, 112)
    a1.set_title('A. Who gets rejected at 80 % coverage', loc='left')
    a2.set_title('B. Accuracy of the plates kept', loc='left')
    a1.legend(fontsize=7, loc='upper left')
    a2.legend(fontsize=6.6, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0), handlelength=1)
    style(a1); style(a2)
    fig.tight_layout()
    save(fig, 'fig_conditions.png')


def fig_ema():
    """Class weights of the EMA loss for an example error profile: the code / Eq. 1 vs the paper's sentence."""
    err = np.array([12.0, 40.0, 34.0, 18.0])
    code = np.minimum(err.max() / (err + 1e-6), 1.2)
    text = np.minimum((err + 1e-6) / (err.min() + 1e-6), 1.2)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(7.4, 2.4), gridspec_kw={'width_ratios': [1, 1.3]})
    a0.bar(range(4), err, color=[ORD[c] for c in CL], width=0.62, edgecolor=SURF, linewidth=2, zorder=3)
    for i, v in enumerate(err):
        a0.text(i, v + 1, f'{v:.0f}', ha='center', fontsize=7.4)
    a0.set_xticks(range(4), CL, fontsize=7.5); a0.set_title('A. Example: smoothed errors per class', loc='left', fontsize=9)
    a0.set_ylim(0, 48); style(a0)
    x = np.arange(4)
    for j, (v, col, lab) in enumerate(((code, S1, 'code and Eq. 1: max(err) / err, cap 1.2'),
                                       (text, S2, 'paper text: more errors -> more weight'))):
        a1.bar(x + (j - 0.5) * 0.36, v, width=0.36, color=col, edgecolor=SURF, linewidth=2, zorder=3, label=lab)
        for i, vv in enumerate(v):
            a1.text(x[i] + (j - 0.5) * 0.36, vv + 0.02, f'{vv:.2f}', ha='center', fontsize=6.8)
    a1.set_xticks(x, CL, fontsize=7.5); a1.set_ylim(0, 1.55); a1.set_title('B. Resulting class weights', loc='left', fontsize=9)
    a1.legend(fontsize=6.6, loc='upper left', ncol=1)
    style(a1)
    fig.tight_layout()
    save(fig, 'fig_ema.png')


def fig_seeds():
    SG = A['seed_groups']
    rows = [('kit_baseline4_ema', '4-class, EMA (kit)'), ('kit_baseline4_ce', '4-class, CE (kit)'),
            ('balanced_baseline4_ema', '4-class, EMA + balanced\nsampler (authors\' recipe)'),
            ('text_baseline4_ema', '4-class, EMA as the\npaper text says'),
            ('kit_quality_ema', 'Quality Filter, EMA'), ('kit_quality_ce', 'Quality Filter, CE'),
            ('failure_quality_ce', 'failure predictor\n("will the CRNN fail?")')]
    rows = [r for r in rows if r[0] in SG]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 0.55 * len(rows) + 1.0), sharey=True)
    y = np.arange(len(rows))[::-1]
    for yi, (k, lab) in zip(y, rows):
        g = SG[k]
        for s, f, gt in zip(g['seeds'], g['F1'], g['gate80']):
            mk = 'D' if s.startswith('kaggle') else 'o'
            a1.plot(f, yi, mk, color=S2 if 'quality' in k else S1, ms=6, mec=SURF, mew=1, zorder=4)
            a2.plot(gt, yi, mk, color=S2 if 'quality' in k else S1, ms=6, mec=SURF, mew=1, zorder=4)
        a1.text(max(g['F1']) + 0.8, yi, f"{g['F1_mean']:.1f}", va='center', fontsize=7.2)
        a2.text(max(g['gate80']) + 0.3, yi, f"{g['gate80_mean']:.1f}", va='center', fontsize=7.2)
    a2.axvline(A['gate']['ocr_confidence']['80%'], color=S1, lw=1.2, ls=(0, (3, 2)))
    a2.text(A['gate']['ocr_confidence']['80%'] - 0.2, y.max() + 0.55, 'OCR confidence 84.5', fontsize=6.8, color=INK2, ha='right')
    a1.set_yticks(y, [r[1] for r in rows], fontsize=7.2)
    a1.set_xlabel('test micro F1 (%) (= accuracy)'); a2.set_xlabel('gate: plate exact at 80 % coverage (%)')
    a1.set_title('A. Classifier score per seed', loc='left'); a2.set_title('B. As a gate for the CRNN', loc='left')
    a1.set_ylim(-0.7, len(rows) - 0.1)
    style(a1, 'x'); style(a2, 'x')
    a1.text(0.0, -0.5 / (0.55 * len(rows) + 1.0), 'diamond = Kaggle T4 run (seed 42); circles = our PC (seeds 42 / 1 / 2)',
            transform=a1.transAxes, fontsize=6.6, color=INK2, va='top')
    fig.tight_layout()
    save(fig, 'fig_seeds.png')


def fig_examples():
    E = A['examples']
    fig, axs = plt.subplots(4, 4, figsize=(7.4, 4.0))
    for i, c in enumerate(CL):
        for j, e in enumerate(E[c]):
            img = cv2.imread(str(DATA / 'ocr' / 'images' / 'test' / e['file']))[:, :, ::-1]
            ax = axs[i, j]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(ORD[c]); s.set_linewidth(2.2)
            ax.set_title(f"CRNN {'right' if e['ok'] else 'WRONG'} · conf {e['conf']:.2f} · suit {e['suit_b4ce']:.2f}",
                         fontsize=6.4, color=INK if e['ok'] else S2, pad=2)
        axs[i, 0].set_ylabel(c, fontsize=8.5, weight='bold', color=INK)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, 'fig_examples.jpg')


def fig_hard():
    """Confident CRNN errors (OCR confidence lets them through) and right readings the classifier rejects."""
    E = A['examples']
    fig, axs = plt.subplots(2, 5, figsize=(7.4, 2.4))
    for row, key, title in ((0, 'conf_high_wrong', 'CRNN wrong but confident'), (1, 'cls_low_right', 'CRNN right, classifier says unsuitable')):
        for j, e in enumerate(E[key][:5]):
            img = cv2.imread(str(DATA / 'ocr' / 'images' / 'test' / e['file']))[:, :, ::-1]
            ax = axs[row, j]
            ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(S2 if row == 0 else S1); s.set_linewidth(2)
            ax.set_title(f"conf {e['conf']:.2f} · suit {e['suit_b4ce']:.2f}", fontsize=6.4, pad=2)
        axs[row, 0].set_ylabel(title.replace(', ', ',\n').replace(' but ', '\nbut '), fontsize=6.8, color=INK)
    fig.tight_layout(h_pad=0.5, w_pad=0.3)
    save(fig, 'fig_hard.jpg')


FIGS = {f.__name__: f for f in (fig_pipeline, fig_labels, fig_training, fig_confusion, fig_gate, fig_strong, fig_conditions,
                                fig_ema, fig_seeds, fig_examples, fig_hard)}
if __name__ == '__main__':
    for n in (sys.argv[1:] or FIGS):
        if n == 'fig_seeds' and not any(k.startswith(('failure', 'balanced', 'text')) for k in A['seed_groups']):
            print('skip fig_seeds (no local runs yet)'); continue
        FIGS[n]()
        print('wrote', n)
