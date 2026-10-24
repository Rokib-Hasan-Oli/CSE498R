"""Charts for the paper #8 study report (v1.1 run, static PDF, light mode).

Run order: analysis.py -> seg_check.py -> (check_local.json from the local original check) -> charts.py -> build_report.py.
Numbers come from analysis.json, seg_check.json and check_local.json; images from `After Train/results.zip` and the
498R dataset v1.1. Palette: validated categorical slots 1-4 (validate_palette.js, all checks pass; values are
labelled on every bar because slots 3-4 are below 3:1 contrast).
"""
import io
import json
import os
import sys
import zipfile

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(OUT)
A = json.load(open(os.path.join(OUT, 'analysis.json'), encoding='utf-8'))
SC = json.load(open(os.path.join(OUT, 'seg_check.json'), encoding='utf-8'))
CK = os.path.join(OUT, 'check_local.json')
CK = json.load(open(CK, encoding='utf-8')) if os.path.exists(CK) else None
DATA = 'D:/498R/dataset'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
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
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


def bars(ax, labels, series, colors, names, ymax=112, fmt='{:.1f}', w=0.36, fs=7.6):
    k = len(series)
    for j, (vals, c, nm) in enumerate(zip(series, colors, names)):
        xs = [i + (j - (k - 1) / 2) * (w + 0.03) for i in range(len(labels))]
        rs = ax.bar(xs, vals, w, color=c, label=nm, zorder=3)
        for r, v in zip(rs, vals):
            if v is not None:
                ax.text(r.get_x() + r.get_width() / 2, v + 1.5, fmt.format(v), ha='center', va='bottom', fontsize=fs, color=INK)
    style(ax)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.8)


C = A['classifiers']

# ------------------------------------------------------------------ 1. the paper's pipeline, with what we measured
fig, ax = plt.subplots(figsize=(7.4, 2.45))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis('off')
steps = [('Plate crop', 'input is already\na cut plate', None),
         ('GFP-GAN x2', 'face model; on a\nplate only its\nReal-ESRGAN x2 acts', f"{A['plates_ms']['enhance']:.0f} ms / plate"),
         ('Contrast +\nOtsu', 'top-hat/black-hat,\n333x335, Otsu,\nerode/dilate', None),
         ('15 largest\ncontours', 'assumes one\nblob = one symbol', f"finds {SC['recognition_test']['C_tokens_found_iou50_all15_%']:.0f} % of\nthe characters"),
         ('Order the\nblobs', 'paper: by x only\n(mixes the\n2 plate lines)', None),
         ('CNN 28x28\nper blob', '17 classes (paper),\n101 tokens (ours)', f"{SC['recognition_test']['A_colour_crops_acc_%']:.0f} % colour crop\n{SC['recognition_test']['B_binary_blobs_gt_boxes_acc_%']:.0f} % binary blob")]
bw, gap, x0 = 13.2, 3.3, 1.0
for i, (t, sub, meas) in enumerate(steps):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, 19), bw, 13, boxstyle='round,pad=0.3,rounding_size=1.2',
                                fc='#eef4fc', ec=S1, lw=1.0))
    ax.text(x + bw / 2, 25.5, t, ha='center', va='center', fontsize=8.2, color=INK, fontweight='bold')
    ax.text(x + bw / 2, 13.4, sub, ha='center', va='center', fontsize=6.4, color=INK2, linespacing=1.15)
    if meas:
        ax.add_patch(FancyBboxPatch((x + 0.4, 1.0), bw - 0.8, 7.2, boxstyle='round,pad=0.25,rounding_size=1',
                                    fc='#fdf0ea', ec=S2, lw=0.9))
        ax.text(x + bw / 2, 4.6, meas, ha='center', va='center', fontsize=6.6, color=INK)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.5, 25.5), (x + bw + gap - 0.5, 25.5), arrowstyle='-|>',
                                     mutation_scale=9, color=INK2, lw=1))
ax.text(0.5, 34.5, 'The paper\'s plate reader (blue) and what our v1.1 run measured on Bangladeshi plates (orange)',
        fontsize=8.4, color=INK, fontweight='bold')
save(fig, 'fig_pipeline.png')

# ------------------------------------------------------------------ 2. training curves (the CNN collapse)
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
for ax, run, title in ((axs[0], 'cnn28', 'Paper CNN 28x28, Adam 1e-4 (40 epochs)'),
                       (axs[1], 'resnet50', 'Frozen ResNet-50 150x150, Adam 1e-3 (25 of 30 epochs)')):
    cv = C[run]['curve']
    ep = [e['epoch'] for e in cv]
    ax.plot(ep, [100 * e['train_acc'] for e in cv], color=S1, lw=2, label='train accuracy')
    ax.plot(ep, [100 * e['val_acc'] for e in cv], color=S2, lw=2, label='validation accuracy')
    b = C[run]['best_epoch']
    ax.plot([b], [100 * cv[b - 1]['val_acc']], 'o', ms=8, mfc=S2, mec=SURF, mew=2, zorder=5)
    ax.annotate(f'best val, epoch {b}: {100 * cv[b - 1]["val_acc"]:.1f} %\n(test {C[run]["acc_best_%"]:.1f} %)',
                (b, 100 * cv[b - 1]['val_acc']), xytext=(b + (6 if run == 'cnn28' else -14), 52 if run == 'cnn28' else 62),
                fontsize=7.2, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    if run == 'cnn28':
        ax.annotate(f'last epoch: test {C[run]["acc_last_%"]:.1f} %', (40, 100 * cv[-1]['val_acc']), xytext=(22, 18),
                    fontsize=7.2, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    style(ax)
    ax.set_title(title, fontsize=8.2, loc='left')
    ax.set_xlabel('epoch (one epoch = 47,992 steps of batch 1)', fontsize=7.6)
    ax.set_ylim(0, 102)
axs[0].set_ylabel('accuracy (%)')
axs[0].legend(loc='lower left', fontsize=7.4)
fig.tight_layout()
save(fig, 'fig_curves.png')

# ------------------------------------------------------------------ 3. accuracy per token group
g = ['digits', 'Metro', 'series letters', 'district words']
labels = [f'{x}\n({C["cnn28"]["groups"][x]["n"]} crops)' for x in g] + \
         [f'rare tokens\n(<20 test crops, {C["cnn28"]["classes_rare_test(1-19)"]} classes)']
fig, ax = plt.subplots(figsize=(7.2, 2.6))
bars(ax, labels, [[C['cnn28']['groups'][x]['acc_%'] for x in g] + [C['cnn28']['acc_rare_%']],
                  [C['resnet50']['groups'][x]['acc_%'] for x in g] + [C['resnet50']['acc_rare_%']]],
     [S1, S2], ['paper CNN (best-val checkpoint)', 'frozen ResNet-50 (best-val checkpoint)'])
ax.set_ylabel('test accuracy (%)')
ax.legend(loc='upper right', fontsize=7.4, ncol=2, bbox_to_anchor=(1, 1.14))
fig.tight_layout()
save(fig, 'fig_groups.png')

# ------------------------------------------------------------------ 4. where the plate is lost
R = SC['recognition_test']
V = SC['vs_paper02_same_plates']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={'width_ratios': [1, 1.25]})
ax = axs[0]
vals = [R['A_colour_crops_acc_%'], R['B_binary_blobs_gt_boxes_acc_%'], R['C_tokens_found_iou50_all15_%']]
rs = ax.bar(range(3), vals, 0.55, color=[S1, S1, S2], zorder=3)
for r, v in zip(rs, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 1.5, f'{v:.1f}', ha='center', fontsize=7.8, color=INK)
style(ax)
ax.set_ylim(0, 112)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xticks(range(3))
ax.set_xticklabels(['CNN on colour\ncrops (human\nboxes)', 'CNN on binary\nblobs (human\nboxes)', 'characters the\ncontours find\n(IoU >= 0.5)'], fontsize=7.2)
ax.set_title(f'Characters ({R["tokens"]:,} on {R["plates"]} test plates)', fontsize=8.4, loc='left')
ax.set_ylabel('%')
ax = axs[1]
pl = [('#2 char-YOLO\n(same 467)', V['p02_char_yolo_exact_%'], S3),
      ('#8 CNN, human\nboxes (467)', V['p08_gt_boxes_exact_%'], S1),
      ('#8 pipeline,\nGFP-GAN (651)', A['plates']['gfpgan_fixed']['plate_exact_%'], S2),
      ('#8 pipeline, no\nGFP-GAN (651)', A['plates']['none_fixed']['plate_exact_%'], S2)]
rs = ax.bar(range(4), [p[1] for p in pl], 0.55, color=[p[2] for p in pl], zorder=3)
for r, p in zip(rs, pl):
    ax.text(r.get_x() + r.get_width() / 2, p[1] + 1.5, f'{p[1]:.2f}' if p[1] < 1 else f'{p[1]:.1f}', ha='center', fontsize=7.8, color=INK)
style(ax)
ax.set_ylim(0, 112)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xticks(range(4))
ax.set_xticklabels([p[0] for p in pl], fontsize=7.2)
ax.set_title('Whole plates read exactly (%)', fontsize=8.4, loc='left')
fig.tight_layout()
save(fig, 'fig_where.png')

# ------------------------------------------------------------------ 5. GFP-GAN: does it help, what does it cost
P = A['gfpgan_paired']
SB = A['similarity_by_condition']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={'width_ratios': [1.55, 1]})
cats = ['all 651', f'clear ({SB["clear"]["plates"]})', f'night ({SB["night"]["plates"]})', f'blurry ({SB["blurry"]["plates"]})']
bars(axs[0], cats, [[P['fixed']['mean_none_%']] + [SB[k]['none_fixed_sim_%'] for k in ('clear', 'night', 'blurry')],
                    [P['fixed']['mean_gfpgan_%']] + [SB[k]['gfpgan_fixed_sim_%'] for k in ('clear', 'night', 'blurry')]],
     [S1, S2], ['without GFP-GAN', 'with GFP-GAN'], ymax=100, w=0.34)
axs[0].set_yticks([0, 20, 40, 60, 80])
axs[0].axhline(88, color=MUTED, lw=1, ls=(0, (3, 2)), zorder=2)
axs[0].text(3.55, 89.5, 'paper: 88 % (30 plates)', ha='right', fontsize=7, color=INK2)
axs[0].set_title('Mean plate similarity, two-line order (%)', fontsize=8.4, loc='left')
axs[0].legend(loc='upper left', fontsize=7.2, ncol=2, bbox_to_anchor=(0, 1.0))
ax = axs[1]
ms = [('segmentation\n+ CNN', A['plates_ms']['read'], S1), ('GFP-GAN\n(Real-ESRGAN x2)', A['plates_ms']['enhance'], S2)]
rs = ax.barh(range(2), [m[1] for m in ms], 0.5, color=[m[2] for m in ms], zorder=3)
for r, m in zip(rs, ms):
    ax.text(m[1] + 4, r.get_y() + r.get_height() / 2, f'{m[1]:.1f} ms', va='center', fontsize=7.8, color=INK)
style(ax, 'x')
ax.set_yticks(range(2))
ax.set_yticklabels([m[0] for m in ms], fontsize=7.4)
ax.set_xlim(0, 260)
ax.set_title('Time per plate, Tesla T4', fontsize=8.4, loc='left')
fig.tight_layout()
save(fig, 'fig_gfpgan.png')

# ------------------------------------------------------------------ 6. what the contours see: two test plates
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(PAPER_DIR, 'kaggle'))
import p08_plates  # noqa: E402

from segtools import all_contours, binary_image, iou  # noqa: E402


def overlay(stem):
    bgr = cv2.imread(f'{DATA}/recognition/images/test/{stem}.jpg')
    H, W = bgr.shape[:2]
    gray = p08_plates.maximise_contrast(bgr)
    b = cv2.cvtColor(binary_image(gray), cv2.COLOR_GRAY2RGB)
    sx, sy = 333 / W, 335 / H
    gt = []
    for line in open(f'{DATA}/recognition/labels/test/{stem}.txt').read().splitlines():
        if line.strip():
            _, cx, cy, w, h = map(float, line.split()[:5])
            gt.append((int((cx - w / 2) * W * sx), int((cy - h / 2) * H * sy), int(w * W * sx), int(h * H * sy)))
    cs = all_contours(gray)
    found = sum(max(iou(g_, c) for c in cs) >= 0.5 for g_ in gt)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), b, gt, cs, found


src = {}
for f in sorted(os.listdir(f'{DATA}/recognition/images/test')):   # the first test plate of three sources
    src.setdefault(f.split('_')[0], os.path.splitext(f)[0])
stems = [src[k] for k in ('blpd', 'lpdba', 'nightlpbd')]
fig, axs = plt.subplots(len(stems), 2, figsize=(7.0, 2.25 * len(stems)), gridspec_kw={'width_ratios': [1, 1]})
for row, stem in zip(axs, stems):
    rgb, b, gt, cs, found = overlay(stem)
    row[0].imshow(rgb)
    row[0].set_title(f'{stem.split("_")[0]} test crop', fontsize=7.6, loc='left')
    row[1].imshow(b)
    for (x, y, w, h) in cs:
        row[1].add_patch(Rectangle((x, y), w, h, fill=False, ec=S2, lw=1.1))
    for (x, y, w, h) in gt:
        row[1].add_patch(Rectangle((x, y), w, h, fill=False, ec=S3, lw=1.1, ls='--'))
    row[1].set_title(f'{len(cs)} contours (orange), {len(gt)} characters (green): {found} found', fontsize=7.6, loc='left')
    for a in row:
        a.axis('off')
fig.tight_layout()
save(fig, 'fig_contours.png')

# ------------------------------------------------------------------ 7. GFP-GAN examples (original vs x2 output)
z = zipfile.ZipFile(os.path.join(PAPER_DIR, 'After Train', 'results.zip'))
ex = ['anpr_02ed2068199b', 'anpr_070d0f167046']
fig, axs = plt.subplots(len(ex), 2, figsize=(6.4, 1.95 * len(ex)))
for row, stem in zip(axs, ex):
    o = cv2.cvtColor(cv2.imread(f'{DATA}/ocr/images/test/{stem}.jpg'), cv2.COLOR_BGR2RGB)
    e = cv2.cvtColor(cv2.imdecode(np.frombuffer(z.read(f'runs/plates_cnn28/enh_{stem}.png'), np.uint8), 1), cv2.COLOR_BGR2RGB)
    row[0].imshow(o)
    row[0].set_title(f'original {o.shape[1]}x{o.shape[0]}', fontsize=7.6, loc='left')
    row[1].imshow(e)
    row[1].set_title(f'GFP-GAN output {e.shape[1]}x{e.shape[0]} (Real-ESRGAN x2)', fontsize=7.6, loc='left')
    for a in row:
        a.axis('off')
fig.tight_layout()
save(fig, 'fig_enh.png')

# ------------------------------------------------------------------ 8. the original check (paper's own data)
if CK:
    cn = CK['cnn']
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.85), gridspec_kw={'width_ratios': [1, 1.15]})

    def wbars(ax, items):                         # items: (label, value, sd or None, colour)
        xs = np.arange(len(items))
        ax.bar(xs, [i[1] for i in items], 0.6, color=[i[3] for i in items], zorder=3)
        for x, (_, v, s, _) in zip(xs, items):
            if s:
                ax.errorbar([x], [v], yerr=[s], fmt='none', ecolor=INK2, elinewidth=1, capsize=3, zorder=4)
            ax.text(x, v + (s or 0) + 1.5, f'{v:.1f}', ha='center', fontsize=7.6, color=INK)
        style(ax)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_xticks(xs)
        ax.set_xticklabels([i[0] for i in items], fontsize=7)

    wbars(axs[0], [('paper\nCNN', 92.38, None, MUTED), ('ours CNN\n28x28 (code)', cn['acc_last'][0], cn['acc_last'][1], S1),
                   ('ours CNN\n32x32 (text)\n1 seed', CK['cnn32']['acc_last'][0], None, S1),
                   ('paper\nResNet-50', 87.50, None, MUTED),
                   ('ours\nResNet-50', CK['resnet50']['acc_last'][0], CK['resnet50']['acc_last'][1], S2)])
    axs[0].set_title('All 303 test crops: paper vs ours', fontsize=8.4, loc='left')
    axs[0].set_ylabel('test accuracy (%)')
    keys = [('all 303', 'acc_last'), (f'clean\n{cn["n_clean"]}', 'acc_clean_last'), ('digits', 'digits_last'),
            ('series\nletters', 'letters_last'), (f'letters,\nclean {cn["n_letters_clean"]}', 'letters_clean_last'),
            ('region words\n(all copies)', 'words_last')]
    wbars(axs[1], [(lab, cn[k][0], cn[k][1], S1) for lab, k in keys])
    axs[1].set_title('Our CNN 28x28 by subset (3 seeds)', fontsize=8.4, loc='left')
    fig.tight_layout()
    save(fig, 'fig_check.png')
print('charts written to', OUT)
