"""Charts for the paper #33 study report (v1.1 run, static PDF, light mode).

  D:/498R/_dataset_build/venv/Scripts/python.exe charts.py <unpacked results.zip>

Run order: local_check.py -> analysis.py -> charts.py -> build_report.py. Numbers come from analysis.json; the example
photos are re-predicted with the run's best.pt files on dataset v1.1. Palette: the validated categorical slots 1-3 (as
the #6 / #8 reports); colour means the optimiser (MuSGD blue, SGD orange) or the head, and every bar is labelled.
"""
import json
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
RUN = sys.argv[1]
A = json.load(open(os.path.join(OUT, 'analysis.json'), encoding='utf-8'))
DATA = 'D:/498R/dataset/detection'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
plt.rcParams.update({
    'font.family': 'Segoe UI', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.dpi': 220, 'legend.frameon': False,
})
T, H = A['train'], A['by_head']
MODELS = ['y11n', 'y26n_sgd', 'y26n', 'y26n_p2', 'y26s']
SHORT = {'y11n': 'YOLO11-N\nSGD', 'y26n_sgd': 'YOLO26-N\nSGD', 'y26n': 'YOLO26-N\nMuSGD*', 'y26n_p2': 'YOLO26-N-P2\nMuSGD',
         'y26s': 'YOLO26-S\nMuSGD'}
OPT = lambda m: S2 if T[m]['optimizer'] == 'SGD' else S1


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


def box(ax, x, y, w, h, title, sub, fc='#f4f8fd', ec=S1, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=1.2', fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h * 0.66, title, ha='center', va='center', fontsize=8.6, weight='bold', color=tc)
    ax.text(x + w / 2, y + h * 0.27, sub, ha='center', va='center', fontsize=7.2, color=INK2)


def arrow(ax, a, b, c=INK2):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color=c, lw=1.0))


# ------------------------------------------------------------------ 1. the two heads of one YOLO26 model
fig, ax = plt.subplots(figsize=(7.4, 2.9))
ax.set_xlim(0, 100)
ax.set_ylim(-4, 39)
ax.axis('off')
LSP = A['local']['speed']
box(ax, 0, 14, 11.5, 11, 'Photo', 'letterboxed\nto 640 x 640')
box(ax, 14.5, 14, 17, 11, 'Backbone + neck', 'C3k2 + attention,\nshared by both')
box(ax, 35, 27, 19, 11, 'One-to-many head', 'trained to fire several\nboxes per plate', ec=S2, fc='#fdf3ee')
box(ax, 35, 1, 19, 11, 'One-to-one head', 'trained to fire ONE\nbox per plate', ec=S3, fc='#eef9f4')
box(ax, 57.5, 27, 16, 11, 'NMS', 'drop boxes that\noverlap (IoU 0.7)', ec=S2, fc='#fdf3ee')
box(ax, 77, 27, 23, 11, 'Plates (the default)', f"found {H['y26n_p2/nms']['found_test']:.2f} %\n"
    f"CPU NMS {LSP['y26n_p2/nms']['cpu_onnx']['post_ms_median']:.1f} ms", ec=S2, fc='#fdf3ee')
box(ax, 57.5, 1, 42.5, 11, 'Plates, no NMS ("e2e", nms=False)',
    f"found {H['y26n_p2/e2e']['found_test']:.2f} % · CPU post-processing {LSP['y26n_p2/e2e']['cpu_onnx']['post_ms_median']:.2f} ms\n"
    f"no thresholds · {H['y26n_p2/e2e']['dup_boxes']} double boxes (NMS: {H['y26n_p2/nms']['dup_boxes']})",
    ec=S3, fc='#eef9f4')
arrow(ax, (11.5, 19.5), (14.5, 19.5))
arrow(ax, (31.5, 21.5), (35, 31))
arrow(ax, (31.5, 17.5), (35, 8))
arrow(ax, (54, 32.5), (57.5, 32.5))
arrow(ax, (73.5, 32.5), (77, 32.5))
arrow(ax, (54, 6.5), (57.5, 6.5))
ax.text(50, -3.6, 'Numbers: our YOLO26-N-P2 on the 1,011 test plates; post-processing per photo on our CPU (Core Ultra 7 '
        '265K, ONNX).\nThe P2 model scores 34,000 candidate boxes per photo, so NMS on the CPU is slow.', ha='center',
        fontsize=6.8, color=INK2)
save(fig, 'fig_heads.png')

# ------------------------------------------------------------------ 2. validation curves
C = A['curves']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
for ax, (title, ms) in zip(axs, (('Optimiser: MuSGD vs SGD (YOLO26-N) and YOLO11-N', ('y26n', 'y26n_sgd', 'y11n')),
                                 ('YOLO26 variants (all MuSGD)', ('y26n', 'y26n_p2', 'y26s')))):
    for m, c, ls in zip(ms, (S1, S2, S3), ('-', '-', '-')):
        ax.plot(C[m]['epoch'], C[m]['val_mAP50-95'], color=c, lw=1.6, ls=ls, label=T[m]['name'])
    style(ax)
    ax.set_title(title, fontsize=8.6, loc='left')
    ax.set_xlabel('epoch')
    ax.set_xlim(0, 61)
    ax.set_ylim(48, 62)
    ax.axvline(50, color=MUTED, lw=0.8, ls=':')
    ax.text(49.3, 48.6, 'YOLO26-N (MuSGD)\nstopped: time cap', ha='right', va='bottom', fontsize=6.6, color=INK2)
    ax.axvline(50.5, color=GRID, lw=0)
    ax.legend(fontsize=6.8, loc='lower right', bbox_to_anchor=(1.0, 0.18))
axs[0].set_ylabel('validation mAP50-95 (%)')
axs[0].text(51, 49.8, 'mosaic off\n(last 10)', fontsize=6.4, color=MUTED)
fig.tight_layout()
save(fig, 'fig_curves.png')

# ------------------------------------------------------------------ 3. five models: test accuracy (clean split)
LN = A['local_no_ridoykj']
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.55))
for ax, (title, vals, lo, hi) in zip(axs, (
        ('Test mAP50 (%)', [T[m]['test']['mAP50'] for m in MODELS], 96.5, 99.5),
        ('Plates found, conf 0.25 (%)', [T[m]['found_%'] for m in MODELS], 96.5, 99.0),
        ('mAP50-95 without ridoykj (%)', [LN[f'{m}/nms']['mAP50-95'] for m in MODELS], 63.0, 66.0))):
    rs = ax.barh(range(len(MODELS)), [v - lo for v in vals], left=lo, color=[OPT(m) for m in MODELS], height=0.62, zorder=3)
    for r, v in zip(rs, vals):
        ax.text(v + (hi - lo) * 0.015, r.get_y() + r.get_height() / 2, f'{v:.2f}', va='center', fontsize=7.2, color=INK)
    style(ax, 'x')
    ax.set_xlim(lo, hi)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([SHORT[m].replace('\n', ' ') for m in MODELS] if ax is axs[0] else [], fontsize=7.2)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8.2, loc='left')
    ax.tick_params(axis='x', labelsize=6.8)
fig.text(0.01, 0.01, 'Blue = MuSGD, orange = SGD. Axes do not start at 0 (differences are small). '
         '* stopped at epoch 50 of 60.', fontsize=6.8, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, 'fig_models.png')

# ------------------------------------------------------------------ 4. conditions: plates found per condition
conds = [('found_test', 'all test'), ('found_night', 'night'), ('found_small', 'small plates'),
         ('found_fog', 'fog (synthetic)'), ('found_rain', 'rain (synthetic)')]
fig, ax = plt.subplots(figsize=(7.4, 2.9))
w = 0.155
for j, m in enumerate(MODELS):
    vals = [H[f'{m}/nms'][k] for k, _ in conds]
    xs = [i + (j - 2) * (w + 0.012) for i in range(len(conds))]
    rs = ax.bar(xs, vals, w, color=OPT(m), zorder=3)
    for r, v in zip(rs, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 1.0, f'{v:.1f}', ha='center', va='bottom', fontsize=5.9,
                color=INK, rotation=90)
style(ax)
ax.set_ylim(0, 118)
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([f"{c}\n({n})" for (_, c), n in zip(conds, (
    A['counts']['test_plates'], A['counts']['night_plates'], A['counts']['small_plates'], A['counts']['test_plates'],
    A['counts']['test_plates']))], fontsize=7.6)
ax.set_ylabel('plates found (%)')
for i, m in enumerate(MODELS):
    ax.text(4 + (i - 2) * (w + 0.012), 2, SHORT[m].split('\n')[0].replace('YOLO', 'Y'), rotation=90, fontsize=5.6,
            color='white', ha='center', va='bottom', zorder=4)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=S2, label='trained with SGD'), Patch(color=S1, label='trained with MuSGD')], fontsize=7,
          loc='upper right', bbox_to_anchor=(1.0, 1.12), ncol=2)
ax.set_ylim(0, 112)
fig.text(0.01, 0.01, 'Bars in every group, left to right: YOLO11-N (SGD) · YOLO26-N (SGD) · YOLO26-N (MuSGD, stopped at '
         'epoch 50) · YOLO26-N-P2 (MuSGD) · YOLO26-S (MuSGD).\nNumber of plates in brackets. Rain and fog are synthetic '
         'copies of every test photo.', fontsize=6.6, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
save(fig, 'fig_conditions.png')

# ------------------------------------------------------------------ 5. speed
L = A['local']['speed']
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.8))
rows = [('YOLO11-N + NMS', 'y11n/nms', S2), ('YOLO26-N + NMS', 'y26n/nms', S1), ('YOLO26-N NMS-free', 'y26n/e2e', S3),
        ('YOLO26-N-P2 + NMS', 'y26n_p2/nms', S1), ('YOLO26-N-P2 NMS-free', 'y26n_p2/e2e', S3),
        ('YOLO26-S NMS-free', 'y26s/e2e', S3)]
groups = [('GPU: PyTorch FP16, whole call (ms)', lambda k: L[k]['gpu_fp16']['ms_median']),
          ('CPU: ONNX model only, 4 threads (ms)', lambda k: L[k]['cpu_onnx']['model_ms_4_threads']),
          ('CPU: post-processing incl. NMS (ms)', lambda k: L[k]['cpu_onnx']['post_ms_median'])]
for ax, (title, get) in zip(axs, groups):
    vals = [get(k) for _, k, _ in rows]
    rs = ax.barh(range(len(rows)), vals, color=[c for *_, c in rows], height=0.62, zorder=3)
    for r, v in zip(rs, vals):
        ax.text(v + max(vals) * 0.02, r.get_y() + r.get_height() / 2, f'{v:.1f}' if v >= 1 else f'{v:.2f}',
                va='center', fontsize=7.0, color=INK)
    style(ax, 'x')
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, *_ in rows] if ax is axs[0] else [], fontsize=7.0)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=7.6, loc='left')
    ax.tick_params(axis='x', labelsize=6.8)
fig.text(0.01, 0.01, 'Our PC, one job at a time: RTX 5060 Ti and Core Ultra 7 265K; batch 1, 640 px; median of 200 (GPU) / '
         '100 (CPU) test photos.\nOrange = YOLO11, blue = YOLO26 with NMS, green = YOLO26 NMS-free.', fontsize=6.4,
         color=INK2)
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, 'fig_speed.png')

# ------------------------------------------------------------------ 6. rain examples: MuSGD finds, SGD misses
import csv
from ultralytics import YOLO
plates = list(csv.DictReader(open(os.path.join(OUT, 'det_per_plate.csv'), encoding='utf-8')))
f = {}
for r in plates:
    if r['split'] == 'rain' and r['head'] == 'nms' and r['model'] in ('y26n', 'y26n_sgd'):
        f.setdefault(r['file'], {}).setdefault(r['model'], []).append(int(r['found']))
cand = sorted(k for k, v in f.items() if len(v) == 2 and all(v['y26n']) and not any(v['y26n_sgd'])
              and len(v['y26n']) == 1)
srcs = {}
pick = []
for k in cand:                                   # one photo per source, up to 3
    s = k.split('_')[0]
    if s not in srcs and s in ('kuet', 'lpdba', 'ridoykj', 'tusar', 'mahedi'):
        srcs[s] = k
        pick.append(k)
    if len(pick) == 3:
        break
mods = {m: YOLO(os.path.join(RUN, 'runs', m, 'weights', 'best.pt')) for m in ('y26n', 'y26n_sgd')}
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.3))
for j, k in enumerate(pick):
    img = cv2.imread(f'{DATA}/images/test_rain_synth/{k}')
    for i, m in enumerate(('y26n', 'y26n_sgd')):
        r = mods[m].predict(img, imgsz=640, conf=0.25, verbose=False)[0]
        im = img.copy()
        col = (214, 120, 42) if m == 'y26n' else (52, 104, 235)
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(im, (x1, y1), (x2, y2), col, max(3, im.shape[1] // 160))
            cv2.putText(im, f'{c:.2f}', (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, im.shape[1] / 450, col,
                        max(2, im.shape[1] // 300))
        ax = axs[i, j]
        ax.imshow(im[:, :, ::-1])
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        n = len(r.boxes)
        ax.set_title(f"{'YOLO26-N MuSGD' if m == 'y26n' else 'YOLO26-N SGD'}: {n} box{'es' if n != 1 else ''}",
                     fontsize=7.6, loc='left', color=INK)
fig.text(0.01, 0.005, 'Synthetic-rain copies of three test photos (' + ', '.join(pick) + ').\nSame architecture, same '
         'data, same seed; only the optimiser differs. Boxes at confidence >= 0.25 with their score.', fontsize=6.4,
         color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, 'fig_rain_examples.png')
json.dump({'rain_examples': pick}, open(os.path.join(OUT, 'charts_meta.json'), 'w'))
print('charts done', pick)
