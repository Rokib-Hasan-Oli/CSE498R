"""Charts for the paper #35 study report (v1.1 run, static PDF, light mode).

  D:/498R/_dataset_build/venv/Scripts/python.exe charts.py <unpacked results.zip>

Run order: local_check.py -> analysis.py -> charts.py -> build_report.py. Numbers come from analysis.json; example
photos come from local_check's enhanced copies (_work/enh), example crops are enhanced again here with the same kit code.
Palette (validated, light): no enhancement = neutral grey, gamma yellow, CLAHE aqua, HVI-gen blue, HVI-LOLv2 orange
(categorical slots 4, 3, 1, 2 in that adjacency; contrast WARN for yellow / aqua -> every bar carries its value).
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent
RUN = Path(sys.argv[1])
A = json.load(open(OUT / 'analysis.json', encoding='utf-8'))
ENH = ['none', 'gamma', 'clahe', 'hvi_gen', 'hvi_lolv2']
NAME = {'none': 'none', 'gamma': 'gamma 0.5', 'clahe': 'CLAHE', 'hvi_gen': 'HVI-CIDNet\ngeneralization',
        'hvi_lolv2': 'HVI-CIDNet\nLOLv2-Real'}
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
NEUTRAL = '#a3a29d'
COL = {'none': NEUTRAL, 'gamma': S4, 'clahe': S3, 'hvi_gen': S1, 'hvi_lolv2': S2}
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
    if name.endswith('.jpg'):                    # photo grids: JPEG keeps the PDF small
        fig.savefig(OUT / name, pil_kwargs={'quality': 88})
    else:
        fig.savefig(OUT / name)
    plt.close(fig)


def box(ax, x, y, w, h, title, sub, fc='#f4f8fd', ec=S1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=1.2', fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=8.4, weight='bold', color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha='center', va='center', fontsize=7.0, color=INK2)


def arrow(ax, a, b, c=INK2, ls='-'):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=c, lw=1.0, linestyle=ls))


def bars(ax, vals, labels, fmt='{:.1f}', ref=None, err=None):
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.62, color=[COL[e] for e in labels], edgecolor=SURF, linewidth=2, zorder=3)
    if err is not None:
        lo = [v - e[0] for v, e in zip(vals, err)]
        hi = [e[1] - v for v, e in zip(vals, err)]
        ax.errorbar(x, vals, yerr=[lo, hi], fmt='none', ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
    for i, v in enumerate(vals):
        top = v if err is None else max(v, err[i][1])
        bot = v if err is None else min(v, err[i][0])
        y = top if v >= 0 else bot
        ax.text(i, y + (0.6 if v >= 0 else -0.6), fmt.format(v), ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=8, color=INK, zorder=5, bbox=dict(boxstyle='square,pad=0.08', fc=SURF, ec='none'))
    ax.set_xticks(x, [NAME[e] for e in labels], fontsize=7.4)
    if ref is not None:
        ax.axhline(ref, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=2)


def fig_hvi():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.3, 2.55), gridspec_kw={'width_ratios': [1.25, 1, 1.25]})
    # (a) HSV hue line: red at both ends
    hue = np.linspace(0, 1, 360)
    rgb = np.array([cv2.cvtColor(np.uint8([[[int(h * 179), 255, 230]]]), cv2.COLOR_HSV2RGB)[0, 0] / 255 for h in hue])
    a1.imshow(rgb[None], extent=(0, 6, 0, 0.5), aspect='auto')
    for h in (0.15, 5.85):
        a1.plot([h], [0.25], 'o', ms=6, mfc='white', mec=INK, mew=1.2)
    a1.annotate('', (0.15, 0.75), (5.85, 0.75), arrowprops=dict(arrowstyle='<->', color=INK2, lw=0.9))
    a1.text(3, 0.85, 'two reds are 5.7 apart on the hue line', ha='center', fontsize=7.6, color=INK2)
    a1.set_xlim(-0.2, 6.2)
    a1.set_ylim(0, 1.15)
    a1.set_yticks([])
    a1.set_xticks([0, 1, 2, 3, 4, 5, 6], ['0\nred', '1', '2\ngreen', '3', '4\nblue', '5', '6\nred'], fontsize=7)
    for s in ('top', 'right', 'left'):
        a1.spines[s].set_visible(False)
    a1.set_title('(a) HSV: hue is a line', fontsize=9)
    # (b) polarised hue: a circle, the reds touch
    t = np.linspace(0, 2 * np.pi, 360)
    for i in range(359):
        a2.plot(np.cos(t[i:i + 2]), np.sin(t[i:i + 2]), color=rgb[i], lw=4, solid_capstyle='butt')
    for h in (0.15, 5.85):
        a2.plot([np.cos(np.pi * h / 3)], [np.sin(np.pi * h / 3)], 'o', ms=6, mfc='white', mec=INK, mew=1.2)
    a2.text(0, 0, 'HV plane:\n(cos πh/3, sin πh/3)\nx saturation x Ck(I)', ha='center', va='center', fontsize=7, color=INK2)
    a2.text(1.12, -0.52, 'the two reds\nare 0.31 apart', fontsize=7, color=INK2, ha='left')
    a2.set_aspect('equal')
    a2.set_xlim(-1.25, 1.95)
    a2.set_ylim(-1.25, 1.25)
    a2.axis('off')
    a2.set_title('(b) HVI: hue is an angle', fontsize=9)
    # (c) intensity collapse Ck(I) with the learned k of the released weights
    I = np.linspace(0, 1, 300)
    K = A['density_k']
    a3.plot(I, np.sin(np.pi * I / 2) ** 0.2, color=MUTED, lw=1.2, ls=(0, (3, 2)), label='start of training (k = 5)')
    for key, c, lab in (('hvi_lolv2', S2, 'LOLv2-Real weights'), ('hvi_gen', S1, 'generalization weights')):
        k = K[key]['k_code']
        a3.plot(I, np.sin(np.pi * I / 2) ** k, color=c, lw=2, label=f'{lab} (k = {1 / k:.2f})')
    a3.legend(fontsize=6.6, loc='lower right', handlelength=1.6)
    a3.set_xlabel('pixel intensity I = max(R, G, B)', fontsize=7.6)
    a3.set_ylabel('radius of the colour plane, Ck', fontsize=7.6)
    a3.set_xlim(0, 1)
    a3.set_ylim(0, 1.05)
    style(a3)
    a3.set_title('(c) dark pixels lose their colour', fontsize=9)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig_hvi.png')


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.3, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 36)
    ax.axis('off')
    box(ax, 0.5, 23, 11, 10, 'photo', 'night / day', fc='#f6f6f4', ec=MUTED)
    box(ax, 15, 23, 16, 10, 'dark gate', 'mean grey < 60 ?', fc='#fff7e8', ec=S4)
    box(ax, 34.5, 23, 19, 10, 'enhancer', 'gamma / CLAHE / HVI', fc='#eef5fd', ec=S1)
    box(ax, 57, 23, 12, 10, 'YOLOv8s', 'plate box', fc='#f6f6f4', ec=MUTED)
    box(ax, 72.5, 23, 12, 10, 'crop', '5 % padding', fc='#f6f6f4', ec=MUTED)
    box(ax, 88, 23, 11.5, 10, 'CRNN', 'plate text', fc='#f6f6f4', ec=MUTED)
    for a, b in (((11.5, 28), (15, 28)), ((31, 28), (34.5, 28)), ((53.5, 28), (57, 28)), ((69, 28), (72.5, 28)),
                 ((84.5, 28), (88, 28))):
        arrow(ax, a, b)
    ax.text(23, 20.6, 'no: skip the enhancer', ha='center', fontsize=6.8, color=INK2)
    rows = [('photo', 'enhance the photo; detect AND crop on the enhanced photo (the reader sees enhanced pixels)'),
            ('detect', 'enhance the photo for the detector only; crop the ORIGINAL pixels for the reader'),
            ('crop', 'detect on the original photo; enhance only the plate crop (what the kit\'s OCR test does)')]
    for i, (k, t) in enumerate(rows):
        y = 11.5 - 4.6 * i
        ax.text(2, y, k, fontsize=7.8, weight='bold', color=S1, va='center')
        ax.text(11, y, t, fontsize=7.6, color=INK2, va='center')
    ax.text(2, 16.2, 'Three places to put the enhancer (end-to-end test, section 12):', fontsize=7.8, color=INK, va='center')
    save(fig, 'fig_pipeline.png')


def fig_results():
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.9))
    b = A['night_ap_bootstrap']
    v = [0.0] + [b[e]['diff_pp'] for e in ENH[1:]]
    err = [(0.0, 0.0)] + [tuple(b[e]['ci95']) for e in ENH[1:]]
    bars(axs[0], v, ENH, fmt='{:+.1f}', ref=0, err=err)
    axs[0].set_title('Night photos: AP50 change (pp)', fontsize=8.6)
    axs[0].set_ylabel('vs no enhancement, 95 % CI', fontsize=7.4)
    lo = min(e[0] for e in err) - 1.5
    hi = max(e[1] for e in err) + 1.5
    axs[0].set_ylim(lo, hi)
    style(axs[0])
    rs = A['reproduction']
    v = [rs[e]['read']['local_night'] for e in ENH]
    bars(axs[1], v, ENH, ref=v[0])
    axs[1].set_title(f"Night crops read right (%), n = {rs['none']['read']['night_crops']}", fontsize=8.6)
    axs[1].set_ylim(0, 80)
    style(axs[1])
    ee = A['e2e']
    v = [ee[f'{e}/photo/nightlpbd']['exact_%'] for e in ENH]
    bars(axs[2], v, ENH, ref=v[0])
    axs[2].set_title(f"Night photos end to end (%), n = {ee['none/photo/nightlpbd']['photos']}", fontsize=8.6)
    axs[2].set_ylim(0, 80)
    style(axs[2])
    for a in axs:
        plt.setp(a.get_xticklabels(), rotation=90, fontsize=7)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_results.png')


def fig_gate():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.3, 2.7), gridspec_kw={'width_ratios': [1.15, 1]})
    G = A['gate']
    edges = np.arange(0, 260, 10)
    allh = np.array(G['hist_all'], float)
    nh = np.array(G['hist_night'], float)
    a1.bar(edges[:-1] + 5, allh - nh, width=8.6, color=NEUTRAL, edgecolor=SURF, linewidth=1, label='other test photos', zorder=3)
    a1.bar(edges[:-1] + 5, nh, bottom=allh - nh, width=8.6, color=S1, edgecolor=SURF, linewidth=1, label='night-tagged photos',
           zorder=3)
    a1.axvline(60, color=S2, lw=1.4, zorder=4)
    a1.text(146, max(allh) * 0.66, f"orange line = gate (mean grey < 60):\nonly {G['night_below_60']} of {G['night_photos']} night photos "
            f"and\n{G['test_below_60']} of {G['test_photos']:,} test photos are below it", fontsize=6.6, color=INK, va='top',
            ha='left')
    a1.set_xlabel('mean grey of the photo (0-255)', fontsize=7.6)
    a1.set_ylabel('test photos', fontsize=7.6)
    a1.legend(fontsize=7, loc='upper right')
    a1.set_xlim(0, 250)
    style(a1)
    a1.set_title('(a) Most night photos are not dark', fontsize=9)
    sw = A['gate_sweep']
    base = A['gate_sweep_base']['all_AP50']
    ends = []
    for e in ('clahe', 'hvi_gen', 'hvi_lolv2', 'gamma'):
        thr = [r['thr'] if r['thr'] < 256 else 160 for r in sw[e]]
        a2.plot(thr, [r['all_AP50'] - base for r in sw[e]], color=COL[e], lw=2, marker='o', ms=4.5, mec=SURF, mew=1)
        ends.append([sw[e][-1]['all_AP50'] - base, e])
    ends.sort()
    for i in range(1, len(ends)):                         # keep end labels at least 0.035 pp apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
    for y, e in ends:
        a2.text(163, y, NAME[e].replace('\n', ' '), fontsize=6.8, color=INK, va='center')
    a2.axhline(0, color=INK2, lw=0.9, ls=(0, (3, 2)))
    a2.set_xticks([40, 60, 80, 100, 120, 140, 160], ['40', '60', '80', '100', '120', '140', 'all'], fontsize=7)
    a2.set_xlim(35, 215)
    a2.set_xlabel('gate threshold (enhance photos darker than this)', fontsize=7.6)
    a2.set_ylabel('whole-test AP50 change (pp)', fontsize=7.6)
    style(a2)
    a2.set_title('(b) Gate sweep, all 1,153 test photos', fontsize=9)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig_gate.png')


def fig_speed():
    R = A['run']
    K = R['kaggle_lowlight']['ms']
    dsp = R['det_speed']
    rows = [('YOLOv8s detector, TensorRT FP16', dsp['tensorrt_fp16']['ms_median'], MUTED),
            ('YOLOv8s detector, PyTorch FP32', dsp['fp32']['ms_median'], MUTED),
            ('gamma 0.5', K['gamma']['photo_640'], COL['gamma']),
            ('CLAHE', K['clahe']['photo_640'], COL['clahe']),
            ('HVI-CIDNet, FP32', K['hvi_gen']['photo_640'], COL['hvi_gen'])]
    fig, ax = plt.subplots(figsize=(7.3, 2.1))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.6, edgecolor=SURF, linewidth=2, zorder=3)
    for yy, r in zip(y, rows):
        ax.text(r[1] + 2, yy, f'{r[1]:.1f} ms', va='center', fontsize=7.8, color=INK)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.6)
    ax.set_xlabel('ms per 640 x 640 photo on the Kaggle Tesla T4, batch 1 (median)', fontsize=7.6)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    style(ax, 'x')
    fig.tight_layout()
    save(fig, 'fig_speed.png')


def fig_training():
    R = A['run']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.3, 2.4))
    c = R['det_curve']
    ep = [r['epoch'] for r in c]
    a1.plot(ep, [100 * r['mAP50'] for r in c], color=S1, lw=2)
    a1.plot(ep, [100 * r['mAP50-95'] for r in c], color=S2, lw=2)
    a1.text(ep[-1] + 1, 100 * c[-1]['mAP50'], 'mAP50', fontsize=7.4, va='center', color=INK)
    a1.text(ep[-1] + 1, 100 * c[-1]['mAP50-95'], 'mAP50-95', fontsize=7.4, va='center', color=INK)
    a1.axvline(R['det']['best_epoch'], color=MUTED, lw=0.9, ls=(0, (3, 2)))
    a1.text(R['det']['best_epoch'] - 1, 76, f"best epoch {R['det']['best_epoch']}", fontsize=7, color=INK2, ha='right')
    a1.set_xlim(0, ep[-1] + 12)
    a1.set_xlabel('epoch', fontsize=7.6)
    a1.set_ylabel('validation (%)', fontsize=7.6)
    style(a1)
    a1.set_title('(a) YOLOv8s plate detector', fontsize=9)
    lg = R['crnn']['log']
    e2 = [r['epoch'] for r in lg]
    a2.plot(e2, [r['train_loss'] for r in lg], color=S1, lw=2)
    a2.plot(e2, [r['val_loss'] for r in lg], color=S2, lw=2)
    a2.text(e2[-1] + 1, lg[-1]['train_loss'], 'train', fontsize=7.4, va='center', color=INK)
    a2.text(e2[-1] + 1, lg[-1]['val_loss'], 'validation', fontsize=7.4, va='center', color=INK)
    a2.set_yscale('log')
    a2.set_xlim(0, e2[-1] + 12)
    a2.set_xlabel('epoch', fontsize=7.6)
    a2.set_ylabel('CTC loss (log scale)', fontsize=7.6)
    style(a2)
    a2.set_title('(b) CRNN plate reader', fontsize=9)
    fig.tight_layout(w_pad=1.5)
    save(fig, 'fig_training.png')


def tile(img, w):
    h = int(img.shape[0] * w / img.shape[1])
    return cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def fig_examples():
    """three night photos (darkest, median, brightest night-tagged) through every enhancer."""
    G = A['grey_night'] if 'grey_night' in A else None
    night = A['night_examples']
    fig, axs = plt.subplots(len(night), len(ENH), figsize=(7.3, 1.32 * len(night) + 0.3))
    for i, f in enumerate(night):
        for j, e in enumerate(ENH):
            im = cv2.imread(str(OUT / '_work' / 'enh' / e / 'images' / 'test' / f))
            axs[i, j].imshow(tile(im, 360))
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            for s in axs[i, j].spines.values():
                s.set_visible(False)
            if i == 0:
                axs[i, j].set_title(NAME[e].replace('\n', ' '), fontsize=7.6)
            if j == 0:
                axs[i, j].set_ylabel(f"grey {A['example_grey'][f]:.0f}", fontsize=7.4)
    fig.tight_layout(h_pad=0.3, w_pad=0.3)
    save(fig, 'fig_examples.jpg')


def fig_crops():
    sys.path.insert(0, str(OUT.parent / 'kaggle'))
    os.environ.setdefault('LPR_WORK', str(OUT / '_work'))
    os.environ.setdefault('DATASET_ROOT', r'D:\498R\dataset')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ['HVI_DIR'] = str(RUN / 'hvi_cidnet')
    import torch
    import p35_lowlight as p35
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = A['crop_examples']
    show = ['none', 'clahe', 'hvi_gen', 'hvi_lolv2']
    enh = {e: p35.Enhancer(e, dev) for e in show}
    fig, axs = plt.subplots(len(files), len(show), figsize=(7.3, 1.05 * len(files) + 0.3))
    for i, f in enumerate(files):
        im = cv2.imread(str(Path(r'D:\498R\dataset') / 'ocr' / 'images' / 'test' / f))
        for j, e in enumerate(show):
            out = enh[e](im)
            ax = axs[i, j]
            ax.imshow(tile(out, 300))
            ok = A['flip_examples'][f][e] is not None and A['crop_right'][f][e]
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(S3 if ok else S2)
                s.set_linewidth(2.2)
            ax.text(0.02, 0.04, 'read right' if ok else 'read wrong', transform=ax.transAxes, fontsize=6.8, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc=S3 if ok else '#c4441c', ec='none'))
            if i == 0:
                ax.set_title(NAME[e].replace('\n', ' '), fontsize=7.6)
            if j == 0:
                ax.set_ylabel(f'#{i + 1}', fontsize=8, rotation=0, labelpad=12)
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    save(fig, 'fig_crops.jpg')


if __name__ == '__main__':
    only = set(sys.argv[2:])                  # optional: names of the figures to redo, e.g. fig_gate
    for fn in (fig_hvi, fig_pipeline, fig_results, fig_gate, fig_speed, fig_training, fig_examples, fig_crops):
        if only and fn.__name__ not in only:
            continue
        fn()
        print('ok', fn.__name__)
