"""Charts for the paper #34 study report (v1.1 run, static PDF, light mode).

  D:/498R/_dataset_build/venv/Scripts/python.exe charts.py [fig names]

Run order: local_check.py -> analysis.py -> charts.py -> build_report.py. Numbers come from analysis.json (and its copy
of local_check.json); example crops are read from the local dataset v1.1 (D:/498R/dataset/ocr/images/test).
Palette (light): the SVTRv2 family in blues (T-MSR the main blue, S navy, fixed and scratch lighter), CRNN neutral
grey, the character-YOLO reader orange, the LPNR reader yellow, fusion green; every bar carries its value.
"""
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('DATASET_ROOT', 'D:/498R/dataset'))
A = json.load(open(OUT / 'analysis.json', encoding='utf-8'))
WANT = set(sys.argv[1:])
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
NAVY, LIGHT, PALE = '#104281', '#7fb0ea', '#c9dcf4'
NEUTRAL = '#a3a29d'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
COL = {'svtrv2_t_msr': S1, 'svtrv2_s_msr': NAVY, 'svtrv2_t_fixed': LIGHT, 'svtrv2_t_scratch': PALE, 'crnn': NEUTRAL,
       'chars_v8n': S2, 'lpnr_v12n': S4}
NAME = {'svtrv2_t_msr': 'SVTRv2-T\nMSR', 'svtrv2_s_msr': 'SVTRv2-S\nMSR', 'svtrv2_t_fixed': 'SVTRv2-T\nfixed 32x128',
        'svtrv2_t_scratch': 'SVTRv2-T\nfrom scratch', 'crnn': 'CRNN', 'chars_v8n': 'char-YOLO\n(Model Fusion)',
        'lpnr_v12n': 'LPNR YOLO\n(Cascaded YOLO)'}
ORDER = ['svtrv2_t_msr', 'svtrv2_s_msr', 'svtrv2_t_fixed', 'svtrv2_t_scratch', 'crnn', 'chars_v8n', 'lpnr_v12n']
plt.rcParams.update({
    'font.family': 'Segoe UI', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.dpi': 220, 'legend.frameon': False,
})


def want(n):
    return not WANT or n in WANT


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
    print('wrote', name)


def box(ax, x, y, w, h, title, sub, fc='#f4f8fd', ec=S1, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=1.2', fc=fc, ec=ec, lw=1.1,
                                linestyle='--' if dashed else '-'))
    ax.text(x + w / 2, y + h * 0.80, title, ha='center', va='center', fontsize=8.2, weight='bold', color=INK)
    ax.text(x + w / 2, y + h * 0.38, sub, ha='center', va='center', fontsize=6.6, color=INK2, linespacing=1.2)


def arrow(ax, a, b, c=INK2, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=c, lw=1.0, linestyle=ls))


def crop(f):
    im = cv2.imread(str(DATA / 'ocr' / 'images' / 'test' / f))
    return im[:, :, ::-1]


# ------------------------------------------------------------------------------------------------ 1. pipeline
if want('fig_pipeline.png'):
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis('off')
    box(ax, 0, 25, 13.5, 15, 'Plate crop', 'from the\ndetector,\nR = w / h')
    box(ax, 16.5, 25, 15.5, 15, 'MSR resize', 'R < 1.5: 64x64\n1.5-2.5: 48x96\n2.5-3.5: 40x112\nelse 32 x W')
    box(ax, 35, 25, 16.5, 15, 'Encoder', 'grouped convs,\nthen self-attention\n-> H/8 x W/4\nfeatures')
    box(ax, 54.5, 25, 15, 15, 'FRM', 'row attention +\na selecting token\nper column\n-> W/4 steps')
    box(ax, 72.5, 25, 12.5, 15, 'CTC', 'greedy:\nmerge repeats,\ndrop blanks')
    box(ax, 88, 25, 12, 15, 'Plate text', 'district,\n[metro], letter,\n6 digits', fc='#f1f9f4', ec=S3)
    for a, b in ((13.5, 16.5), (32, 35), (51.5, 54.5), (69.5, 72.5), (85, 88)):
        arrow(ax, (a, 32.5), (b, 32.5))
    box(ax, 33, 3, 52, 14, 'SGM: training only, NOT trained here',
        'left / right context strings (5 characters) look for the character\nin the visual features; '
        'needs language context, and plate text\n(random digits, a district from a list) has almost none', fc='#fbfbfa', ec=MUTED, dashed=True)
    arrow(ax, (43, 25), (43, 17), c=MUTED, ls='--')
    ax.text(0, 10, 'Our CRNN baseline instead cuts a\ntwo-line plate into its rows by\nhand and puts them side by side\n'
            '(48x256 grey) before CNN +\nBiLSTM + CTC.', fontsize=6.8, color=INK2, va='center')
    ax.text(50, 44, 'SVTRv2 at inference = MSR + encoder + FRM + CTC (one forward pass, no decoder loop)', ha='center',
            fontsize=8.4, color=INK, weight='bold')
    save(fig, 'fig_pipeline.png')

# -------------------------------------------------------------------------------------------- 2. main results
if want('fig_results.png'):
    M = A['main']
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    x = np.arange(len(ORDER))
    v = [M[r]['test']['plate_exact_%'] for r in ORDER]
    lo = [M[r]['test']['plate_exact_%'] - M[r]['ci95'][0] for r in ORDER]
    hi = [M[r]['ci95'][1] - M[r]['test']['plate_exact_%'] for r in ORDER]
    ax.bar(x, v, width=0.62, color=[COL[r] for r in ORDER], edgecolor=[S1 if r == 'svtrv2_t_scratch' else SURF for r in ORDER],
           linewidth=[1.0 if r == 'svtrv2_t_scratch' else 2 for r in ORDER], zorder=3)
    ax.errorbar(x, v, yerr=[lo, hi], fmt='none', ecolor=INK2, elinewidth=0.9, capsize=3, zorder=4)
    for i, val in enumerate(v):
        ax.text(i, M[ORDER[i]]['ci95'][1] + 1.2, f'{val:.1f}', ha='center', fontsize=8.4, color=INK, weight='bold')
    for fu, ls, lab in ((A['fusion']['rows']['chars_t_fallback']['test_%'], '-', 'char-YOLO + SVTRv2-T fallback'),
                        (A['fusion']['rows']['old_best_fallback']['test_%'], ':', 'previous best fusion')):
        ax.axhline(fu, color=S3 if ls == '-' else INK2, lw=1.2, ls=ls, zorder=2)
        ax.text(1.6, fu + (0.7 if ls == '-' else -2.0), f'{lab} {fu:.2f}', ha='left', fontsize=7.4,
                color=S3 if ls == '-' else INK2)
    ax.set_xticks(x, [NAME[r] for r in ORDER], fontsize=7.6)
    ax.set_ylim(50, 97); ax.set_ylabel('plate exact on 651 test crops (%)')
    ax.axvline(4.5, color=GRID, lw=1)
    ax.text(2.6, 94.8, 'this run (Kaggle, 26 Oct)', ha='center', fontsize=7.6, color=MUTED)
    ax.text(5.5, 94.8, 'earlier v1.1 runs', ha='center', fontsize=7.6, color=MUTED)
    style(ax)
    save(fig, 'fig_results.png')

# --------------------------------------------------------------------- 3. MSR vs fixed by size group vs paper
if want('fig_msr.png'):
    B, T1 = A['by_msr_size'], A['paper_table1']
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={'width_ratios': [1, 1.25]})
    ax = axs[0]
    groups = [('R1 (<1.5) 64x64', 'R < 1.5\n(64x64)'), ('R2 (1.5-2.5) 48x96', '1.5 <= R < 2.5\n(48x96)')]
    for i, (k, lab) in enumerate(groups):
        ax.bar(i - 0.17, T1[k]['fixed'], 0.32, color=LIGHT, zorder=3); ax.bar(i + 0.17, T1[k]['msr'], 0.32, color=S1, zorder=3)
        ax.text(i - 0.17, T1[k]['fixed'] + 0.8, f"{T1[k]['fixed']:.1f}", ha='center', fontsize=7.6)
        ax.text(i + 0.17, T1[k]['msr'] + 0.8, f"{T1[k]['msr']:.1f}", ha='center', fontsize=7.6, weight='bold')
        d = T1[k]['msr'] - T1[k]['fixed']
        ax.text(i, 96, f'+{d:.1f}', ha='center', fontsize=8.4, color=S1, weight='bold')
    ax.set_xticks(range(2), [g[1] for g in groups], fontsize=7.6)
    ax.set_ylim(60, 100); ax.set_title('Paper (English, Curve + MO text)', fontsize=9)
    ax.set_ylabel('word / plate accuracy (%)')
    style(ax)
    ax = axs[1]
    gs = [('64x64', f"R < 1.5 (64x64)\n{B['64x64']['n']} plates"), ('48x96', f"1.5 <= R < 2.5 (48x96)\n{B['48x96']['n']} plates")]
    for i, (k, lab) in enumerate(gs):
        for j, (r, c) in enumerate((('svtrv2_t_fixed', LIGHT), ('svtrv2_t_msr', S1), ('crnn', NEUTRAL))):
            ax.bar(i + (j - 1) * 0.26, B[k][r], 0.24, color=c, zorder=3, label=NAME[r].replace('\n', ' ') if i == 0 else None)
            ax.text(i + (j - 1) * 0.26, B[k][r] + 0.8, f'{B[k][r]:.1f}', ha='center', fontsize=7.4,
                    weight='bold' if r == 'svtrv2_t_msr' else None)
        p = B[k]['msr_vs_fixed']
        sig = f"p = {p['p']:.2f}" if p['p'] >= 0.01 else 'p < 0.01' if p['p'] >= 0.0001 else 'p < 0.0001'
        ax.text(i, 96, f"{p['diff_pp']:+.1f} ({p['only_a']} vs {p['only_b']}, {sig})", ha='center', fontsize=7.8, color=S1,
                weight='bold')
    ax.set_xticks(range(2), [g[1] for g in gs], fontsize=7.6)
    ax.set_ylim(60, 100); ax.set_title('Ours (Bangla plates, 651 test crops)', fontsize=9)
    ax.legend(loc='lower center', fontsize=7, ncol=3, bbox_to_anchor=(0.5, -0.42))
    style(ax)
    fig.tight_layout()
    save(fig, 'fig_msr.png')

# ------------------------------------------------------------------------------- 4. aspect ratios + resizing
if want('fig_aspect.jpg'):
    AS = A['aspect']
    fig = plt.figure(figsize=(7.4, 3.4))
    ax = fig.add_axes([0.07, 0.17, 0.42, 0.72])
    bins = AS['bins']
    lab = [f'{bins[i]:g}-{bins[i + 1]:g}' if bins[i + 1] < 10 else f'>= {bins[i]:g}' for i in range(len(bins) - 1)]
    x = np.arange(len(lab))
    for k, (s, c) in enumerate((('train', LIGHT), ('test', S1))):
        n = AS[s]['n']
        ax.bar(x + (k - 0.5) * 0.4, [100 * h / n for h in AS[s]['hist']], 0.38, color=c, zorder=3, label=f'{s} ({n} crops)')
    for xb, t in ((1.5, '64x64'), (5.5, '48x96'), (7.5, '40x112'), (8.5, '32x128')):
        ax.axvline(xb + 0.5 if xb != 1.5 else 2.5, color=INK2, lw=0.8, ls=':')
    ax.text(1.0, 31, '64x64', ha='center', fontsize=7.6, color=INK2); ax.text(4.5, 31, '48x96', ha='center', fontsize=7.6, color=INK2)
    ax.text(7.0, 31, '40x112', ha='center', fontsize=7.2, color=INK2)
    ax.set_xticks(x, lab, rotation=45, fontsize=6.8); ax.set_ylim(0, 34)
    ax.set_xlabel('aspect ratio R = width / height of the plate crop'); ax.set_ylabel('share of crops (%)')
    ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.0, 0.86)); style(ax)
    # one near-square crop resized three ways
    f = next(e['file'] for e in A['examples']['msr_right_fixed_wrong'] if 1.0 <= e['R'] < 1.4)
    im = crop(f)
    views = [('crop as photographed', im, None), ('MSR: 64x64', cv2.resize(im, (64, 64), interpolation=cv2.INTER_CUBIC), (64, 64)),
             ('fixed: 32x128 (squashed)', cv2.resize(im, (128, 32), interpolation=cv2.INTER_CUBIC), (32, 128)),
             ('CRNN: rows side by side', None, None)]
    g = cv2.cvtColor(im[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    h = g.shape[0]
    top, bot = g[: h // 2], g[h // 2:]
    st = np.concatenate([top, cv2.resize(bot, (g.shape[1], top.shape[0]))], 1)
    views[3] = ('CRNN: rows side by side (48x256)', np.stack([st] * 3, -1), None)
    pos = [(0.55, 0.52, 0.2, 0.36), (0.78, 0.52, 0.2, 0.36), (0.55, 0.13, 0.2, 0.3), (0.78, 0.13, 0.2, 0.3)]
    for (t, v, _), p in zip(views, pos):
        a = fig.add_axes(p); a.imshow(v); a.set_title(t, fontsize=7.2); a.axis('off')
    save(fig, 'fig_aspect.jpg')
    A.setdefault('figure_files', {})['aspect_example'] = f

# ---------------------------------------------------------------------------------------- 5. training curves
if want('fig_curves.png'):
    H = A['health']
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for r in ['svtrv2_t_msr', 'svtrv2_s_msr', 'svtrv2_t_fixed', 'svtrv2_t_scratch', 'crnn']:
        lg = A['curves'][r]
        e = [x['epoch'] for x in lg]
        c = COL[r] if r != 'svtrv2_t_scratch' else '#8fb8e8'
        ls = '--' if r == 'svtrv2_t_scratch' else '-'
        axs[0].plot(e, [x['val_loss'] for x in lg], color=c, lw=1.4, ls=ls, label=NAME[r].replace('\n', ' '))
        axs[0].plot(H[r]['best_epoch'], H[r]['best_val_loss'], 'o', color=c, ms=4.5, mec=INK, mew=0.6)
        axs[1].plot(e, [x['train_loss'] for x in lg], color=c, lw=1.4, ls=ls)
    for ax in axs:
        ax.axvline(100, color=MUTED, lw=0.8, ls=':')
        ax.axvspan(63, 100, color='#fbefe7', zorder=0)
        ax.set_xlim(0, 102); ax.set_xlabel('epoch'); style(ax)
    axs[0].text(81.5, 0.24, 'planned, never run\n(1.6 h cap)', ha='center', fontsize=7.0, color=S2)
    axs[1].text(81.5, 0.006, 'planned, never run', ha='center', fontsize=7.0, color=S2)
    axs[0].set_ylim(0.2, 1.0); axs[0].set_ylabel('val CTC loss'); axs[0].set_title('Validation loss (dot = kept checkpoint)', fontsize=8.6)
    axs[1].set_yscale('log'); axs[1].set_ylabel('train loss (log)'); axs[1].set_title('Training loss', fontsize=8.6)
    axs[0].legend(fontsize=6.4, loc='upper right', ncol=1)
    fig.tight_layout()
    save(fig, 'fig_curves.png')

# --------------------------------------------------------------------------------------------- 6. fusion
if want('fig_fusion.png'):
    R = A['fusion']['rows']
    rows = [('chars_v8n', 'char-YOLO alone', A['main']['chars_v8n']['test']['plate_exact_%'], A['main']['chars_v8n']['ci95'], S2),
            ('old_kit_fallback', 'char-YOLO + CRNN fallback', None, None, NEUTRAL),
            ('old_best_fallback', 'char-YOLO + LPNR fallback (best so far)', None, None, S4),
            ('chars_t_fallback', 'char-YOLO + SVTRv2-T fallback', None, None, S1),
            ('new_best_fallback', 'char-YOLO + LPNR + SVTRv2-T + S fallback', None, None, NAVY),
            ('new_best_val', 'same 4 readers, vote (MV-BM+format), chosen on val', None, None, S3)]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for i, (k, lab, v, ci, c) in enumerate(rows[::-1]):
        if v is None:
            v, ci = R[k]['test_%'], R[k]['ci95']
        ax.barh(i, v - 80, left=80, height=0.6, color=c, zorder=3)
        ax.errorbar(v, i, xerr=[[v - ci[0]], [ci[1] - v]], fmt='none', ecolor=INK2, elinewidth=0.9, capsize=3, zorder=4)
        extra = ''
        if k in R and 'vs_old_best_fallback' in R[k]:
            q = R[k]['vs_old_best_fallback']
            extra = f"   vs best so far: {q['only_new']} fixed / {q['only_base']} broken"
        ax.text(ci[1] + 0.25, i, f'{v:.2f}{extra}', va='center', fontsize=7.4, color=INK)
    ax.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]], fontsize=7.6)
    ax.set_xlim(80, 104); ax.set_xticks(range(80, 101, 5)); ax.axvline(90, color=MUTED, lw=0.8, ls=':')
    ax.set_xlabel('plate exact on the same 651 test crops (%)')
    style(ax, 'x')
    fig.tight_layout()
    save(fig, 'fig_fusion.png')

# ------------------------------------------------------------------------------------------- 7. coverage
if want('fig_coverage.png'):
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    for r in ['svtrv2_t_msr', 'svtrv2_s_msr', 'crnn', 'chars_v8n', 'lpnr_v12n']:
        cv = A['coverage'][r]['curve']
        ax.plot([100 * c for c, _ in cv], [a for _, a in cv], color=COL[r], lw=1.5, label=NAME[r].replace('\n', ' '))
    ax.axvline(80, color=MUTED, lw=0.8, ls=':')
    ax.set_xlabel('plates kept, most confident first (%)'); ax.set_ylabel('plate exact on the kept plates (%)')
    ax.set_ylim(70, 100.5); ax.set_xlim(0, 100); ax.legend(fontsize=6.6, loc='lower left'); style(ax)
    fig.tight_layout()
    save(fig, 'fig_coverage.png')

# --------------------------------------------------------------------------------------------- 8. errors
if want('fig_errors.png'):
    E = A['errors']
    rs = ['svtrv2_t_msr', 'svtrv2_s_msr', 'svtrv2_t_fixed', 'crnn', 'chars_v8n', 'lpnr_v12n']
    parts = [('letter', 'series letter', S1), ('district', 'district / metro word', S2), ('digits_substituted', 'digit swapped', S4),
             ('digits_missing', 'digit missing', '#8a6bd1'), ('digits_extra', 'digit extra', S3)]
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    x = np.arange(len(rs))
    for j, (k, lab, c) in enumerate(parts):
        ax.bar(x + (j - 2) * 0.15, [E[r].get(k, 0) for r in rs], 0.14, color=c, label=lab, zorder=3)
    for i, r in enumerate(rs):
        ax.text(i, max(E[r].get(k, 0) for k, _, _ in parts) + 4, f"{E[r]['wrong']} wrong", ha='center', fontsize=7.4, weight='bold')
    ax.set_xticks(x, [NAME[r] for r in rs], fontsize=7.4); ax.set_ylabel('wrong plates with this error')
    ax.set_ylim(0, 110); ax.legend(fontsize=6.8, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.13)); style(ax)
    save(fig, 'fig_errors.png')

# ------------------------------------------------------------------------------------------ 9. conditions
if want('fig_conditions.png'):
    C, X = A['by_condition'], A['extra_tests']
    cols = [('clear', f"clear\n{C['clear']['n']}"), ('night', f"night\n{C['night']['n']}"), ('blurry', f"blurry\n{C['blurry']['n']}"),
            ('test_fog', f"real fog\n{A['extra_n']['test_fog']}"), ('test_rain_synth', f"rain (synthetic)\n{A['extra_n']['test_rain_synth']}")]
    rs = ['svtrv2_t_msr', 'svtrv2_s_msr', 'svtrv2_t_fixed', 'crnn', 'chars_v8n']
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    x = np.arange(len(cols))
    for j, r in enumerate(rs):
        v = [C[k][r] if k in C else X[r][k] for k, _ in cols]
        ax.bar(x + (j - 2) * 0.16, v, 0.15, color=COL[r], label=NAME[r].replace('\n', ' '), zorder=3)
        for i, val in enumerate(v):
            if r in ('svtrv2_t_msr', 'crnn'):
                ax.text(i + (j - 2) * 0.16, val + 1, f'{val:.0f}', ha='center', fontsize=6.4)
    ax.set_xticks(x, [c[1] for c in cols], fontsize=7.6); ax.set_ylim(20, 100); ax.set_ylabel('plate exact (%)')
    ax.legend(fontsize=6.6, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.14)); style(ax)
    save(fig, 'fig_conditions.png')

# ------------------------------------------------------------------------------------------ 10. speed
LC = A.get('local_check', {})
if want('fig_speed.png') and 'timing' in LC:
    T = LC['timing']
    mix = A['by_msr_size']
    w = {k: v['n'] for k, v in mix.items()}
    tm = lambda size, key: sum(w[k] * T[size][k][key] for k in w) / sum(w.values())
    pts = [('svtrv2_t_msr', tm('T', 'gpu_fp32'), tm('T', 'gpu_fp16'), tm('T', 'cpu_fp32')),
           ('svtrv2_s_msr', tm('S', 'gpu_fp32'), tm('S', 'gpu_fp16'), tm('S', 'cpu_fp32')),
           ('crnn', T['crnn']['gpu_fp32'], T['crnn']['gpu_fp16'], T['crnn']['cpu_fp32'])]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for ax, j, title in ((axs[0], 1, 'GPU, batch 1, FP32 (RTX 5060 Ti)'), (axs[1], 3, 'CPU, 4 threads, batch 1, FP32')):
        for r, *ms in pts:
            ax.plot(ms[j - 1], A['main'][r]['test']['plate_exact_%'], 'o', color=COL[r], ms=8, mec=INK, mew=0.5)
            dy = {'svtrv2_t_msr': -1.6, 'svtrv2_s_msr': 1.4, 'crnn': 0}[r]
            ax.text(ms[j - 1] * 1.06, A['main'][r]['test']['plate_exact_%'] + dy, f"{NAME[r].replace(chr(10), ' ')}\n{ms[j - 1]:.2f} ms",
                    fontsize=7.0, va='center')
        ax.set_title(title, fontsize=8.6); ax.set_xlabel('ms per plate (model only)'); ax.set_ylabel('plate exact (%)')
        ax.set_ylim(70, 86); xs = [p[j] for p in pts]; ax.set_xlim(0, max(xs) * 1.7); style(ax)
    fig.tight_layout()
    save(fig, 'fig_speed.png')
    A.setdefault('figure_files', {})['speed_mix'] = {p[0]: [round(p[1], 3), round(p[2], 3), round(p[3], 3)] for p in pts}

# --------------------------------------------------------------------------------------- 11. example crops
if want('fig_examples.jpg'):
    E = A['examples']
    pick = []
    used = {}                                     # spread the examples over the data sources
    for k, n in (('t_right_crnn_wrong', 4), ('msr_right_fixed_wrong', 4), ('crnn_right_t_wrong', 2), ('all_wrong', 2)):
        for _ in range(n):
            cand = [e for e in E[k] if e['file'] not in [p[1]['file'] for p in pick]]
            if not cand:
                break
            e = min(cand, key=lambda e: (used.get(e['source'], 0), E[k].index(e)))
            used[e['source']] = used.get(e['source'], 0) + 1
            pick.append((k, e))
    fig = plt.figure(figsize=(7.4, 3.6))
    lab = {'t_right_crnn_wrong': 'SVTRv2-T right, CRNN wrong', 'msr_right_fixed_wrong': 'MSR right, fixed 32x128 wrong',
           'crnn_right_t_wrong': 'CRNN right, SVTRv2-T wrong', 'all_wrong': 'every reader wrong'}
    for i, (k, e) in enumerate(pick):
        a = fig.add_subplot(3, 4, i + 1)
        a.imshow(crop(e['file'])); a.axis('off')
        a.set_title(f"{i + 1}. {lab[k]}\nR = {e['R']:.2f}, {e['source']}", fontsize=6.4, color=INK2)
    fig.tight_layout(pad=0.4)
    save(fig, 'fig_examples.jpg')
    A.setdefault('figure_files', {})['examples'] = [{'n': i + 1, 'case': k, **e} for i, (k, e) in enumerate(pick)]

if 'figure_files' in A:
    json.dump(A, open(OUT / 'analysis.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
