"""Charts for the paper #36 study report (v1.1 run, static PDF, light mode).

  D:/498R/_dataset_build/venv/Scripts/python.exe charts.py [fig names]

Run order: local_check.py -> visdrone_extra.py -> analysis.py -> charts.py -> build_report.py. Numbers come from
analysis.json; photos and boxes from the local dataset v1.1 and the local box caches (_cache/).
Palette (light, validated with the dataviz validator: all checks pass, contrast WARN -> every bar carries its value):
full-image inference blue, slicing with the paper's merge orange, slicing with the cut-off-box filter aqua, the
slice-fine-tuned (SF) model yellow, the 1280 px model magenta.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import sahi_common as C  # noqa: E402
import det  # noqa: E402

OUT = C.HERE
A = json.load(open(OUT / 'analysis.json', encoding='utf-8'))
V = A['variants']
WANT = set(sys.argv[1:])
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
plt.rcParams.update({
    'font.family': 'Segoe UI', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.dpi': 220, 'legend.frameon': False,
})
BEST = A['val_scores']['v8s_640']['best_slicing']


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
        fig.savefig(OUT / name, pil_kwargs={'quality': 88}, bbox_inches='tight')
    else:
        fig.savefig(OUT / name, bbox_inches='tight')
    plt.close(fig)
    print('wrote', name)


# --------------------------------------------------------------------------------------------- 1. pipeline sketch
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 45); ax.axis('off')

    def box(x, y, w, h, text, fc='#f3f6fb', ec='#9ec5f4', bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2', fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6, color=INK,
                fontweight='bold' if bold else 'normal', linespacing=1.25)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=9, color=INK2, lw=0.9))
    ax.text(1, 43, 'SAHI: slicing aided hyper inference (test time)', fontsize=8.6, fontweight='bold', color=INK)
    box(1, 27, 15, 12, 'photo\n960 x 1280 px')
    # slice grid sketch
    gx, gy, gw, gh = 22, 25, 12, 16
    ax.add_patch(Rectangle((gx, gy), gw, gh, fc='white', ec=INK2, lw=0.8))
    for (sx, sy) in ((0, 0), (0.33, 0), (0, 0.375), (0.33, 0.375), (0, 0.5), (0.33, 0.5)):
        ax.add_patch(Rectangle((gx + sx * gw, gy + sy * gh), 0.67 * gw, 0.5 * gh, fill=False, ec=ORANGE, lw=0.7,
                               ls='--'))
    ax.text(gx + gw / 2, gy - 2.2, '6 slices 640 x 640\n25 % overlap', ha='center', va='top', fontsize=7, color=INK2)
    arrow(16.6, 33, 21.5, 33)
    box(39, 33, 17, 7, 'detector on each slice\n(+ whole photo = FI)')
    box(39, 24, 17, 7, 'shift boxes back to\nphoto coordinates')
    arrow(34.5, 33, 38.5, 36); arrow(47.5, 32.6, 47.5, 31.4)
    box(61, 24, 17, 16, 'merge all boxes\npaper: NMS, IoU 0.5\n(sahi default:\ngreedy merge, IOS 0.5)', fc='#fff4ee', ec='#f3c27a')
    arrow(56.6, 28, 60.5, 30)
    box(83, 27, 16, 10, 'plate boxes', bold=True)
    arrow(78.6, 32, 82.5, 32)
    ax.text(1, 14.5, 'SF: slicing aided fine-tuning (training time)', fontsize=8.6, fontweight='bold', color=INK)
    box(1, 1, 22, 10, 'train photos\n(8,574)')
    box(29, 1, 26, 10, '+ 640 px slices of every photo > 800 px\n(9,110 with a plate, 1,926 without)', fc='#fffaf0', ec='#f3c27a')
    box(61, 1, 17, 10, 'train the detector\non 19,610 images')
    box(83, 1, 16, 10, 'one model, used\nwith FI or SAHI')
    arrow(23.6, 6, 28.5, 6); arrow(55.6, 6, 60.5, 6); arrow(78.6, 6, 82.5, 6)
    save(fig, 'fig_pipeline.png')


# ---------------------------------------------------------------------------------------- 2. object size: theirs/ours
def fig_sizes():
    vd, us = A['visdrone']['visdrone'], A['visdrone']['ours']
    edges = np.linspace(-3.5, 0, 41)
    mid = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    for h, n, col, lab in ((vd['hist_share'], vd['objects'], ORANGE, f"VisDrone test-dev objects (n = {vd['objects']:,})"),
                           (us['hist_share'], us['objects'], BLUE, f"our test plates (n = {us['objects']:,})")):
        y = 100 * np.array(h) / n
        ax.fill_between(mid, y, step='mid', color=col, alpha=0.28, lw=0)
        ax.step(mid, y, where='mid', color=col, lw=2, label=lab)
    marks = (-2, np.log10(vd['width_share_%']['50'] / 100), np.log10(us['width_share_%']['50'] / 100))
    for x in marks:
        ax.axvline(x, color=MUTED, lw=0.8, ls=':')
    ymax = max(100 * np.array(vd['hist_share']) / vd['objects']) * 1.25
    ymax = max(ymax, max(100 * np.array(us['hist_share']) / us['objects']) * 1.12)
    ax.set_ylim(0, ymax)
    for x, t, yy, ha in ((-2, '1 % of the photo width\n(the paper\'s "small")', 0.97, 'right'),
                         (np.log10(vd['width_share_%']['50'] / 100), f"VisDrone median {vd['width_share_%']['50']:.1f} %", 0.80, 'left'),
                         (np.log10(us['width_share_%']['50'] / 100), f"our median {us['width_share_%']['50']:.1f} %", 0.97, 'left')):
        ax.text(x + (0.04 if ha == 'left' else -0.04), ymax * yy, t, fontsize=7.2, color=INK2, va='top', ha=ha)
    ax.set_xticks([-3, -2, -1, 0]); ax.set_xticklabels(['0.1 %', '1 %', '10 %', '100 %'])
    ax.set_xlabel('object width as a share of the photo width (log scale)')
    ax.set_ylabel('share of objects (%)')
    style(ax)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.62), fontsize=7.6)
    save(fig, 'fig_sizes.png')


# ------------------------------------------------------------------------------------------ 3. main kit settings
MAIN = [('v8s_640__FI', '640 model, full image (FI)', BLUE), ('v8s_640__SAHI', '640 model, SAHI slices only', ORANGE),
        ('v8s_640__SAHI_FI', '640 model, SAHI + FI', ORANGE), ('v8s_640__SAHI_FI__BORDER', '640 model, SAHI + FI,\ncut-off boxes dropped', AQUA),
        ('v8s_sf__FI', 'SF model, FI', BLUE), ('v8s_sf__SAHI_FI', 'SF model, SAHI + FI', ORANGE),
        ('v8s_sf__SAHI_FI__BORDER', 'SF model, SAHI + FI,\ncut-off boxes dropped', AQUA), ('v8s_1280__FI', '1280 px model, FI', BLUE)]


def fig_main():
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 3.6), sharey=True)
    y = np.arange(len(MAIN))[::-1]
    for ax, (key, title, lo) in zip(axs, (('AP50', 'AP50 (%)', 85), ('found_%', 'plates found (%)', 92),
                                          ('correct_%', 'boxes that are plates (%)', 50))):
        vals = [V[k][key] for k, _, _ in MAIN]
        ax.barh(y, [v - lo for v in vals], left=lo, color=[c for _, _, c in MAIN], height=0.62, edgecolor=SURF, lw=1.5)
        for yy, v in zip(y, vals):
            ax.text(v + (100 - lo) * 0.012, yy, f'{v:.1f}', va='center', fontsize=7.4, color=INK)
        ax.set_xlim(lo, 100 + (100 - lo) * 0.13)
        ax.set_title(title, fontsize=8.8, loc='left')
        style(ax, 'x')
    axs[0].set_yticks(y); axs[0].set_yticklabels([n for _, n, _ in MAIN], fontsize=7.4)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=BLUE, label='full image only'), Patch(color=ORANGE, label="slices, paper's merge (NMS IoU 0.5)"),
                        Patch(color=AQUA, label='slices, cut-off slice boxes dropped (ours)')],
               loc='lower center', ncol=3, fontsize=7.4, bbox_to_anchor=(0.55, -0.05))
    fig.subplots_adjust(bottom=0.16, wspace=0.12)
    save(fig, 'fig_main.png')


# ---------------------------------------------------------------------------------- 4. what the extra boxes are
def fig_kinds():
    rows = [('v8s_640__FI', 'FI (no slices)'), ('v8s_640__SAHI', 'SAHI, NMS IoU 0.5 (paper)'),
            ('v8s_640__SAHI_FI', 'SAHI + FI, NMS IoU 0.5 (paper)'), ('v8s_640__SAHI_FI__NMS_IOS', 'SAHI + FI, NMS IOS 0.5'),
            ('v8s_640__SAHI_FI__GNMM_IOS', 'SAHI + FI, greedy merge IOS\n(sahi default)'),
            ('v8s_640__SAHI_FI__BORDER', 'SAHI + FI, cut-off boxes\ndropped + NMS IoU'),
            ('v8s_640__SAHI_FI__s960', 'SAHI + FI, 960 px slices')]
    kinds = [('part_of_plate', 'part of a plate', ORANGE), ('background', 'background (no plate there)', BLUE),
             ('duplicate', 'second box on a found plate', YELLOW), ('plate_plus_more', 'plate + surroundings', AQUA)]
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for k, lab, col in kinds:
        vals = np.array([V[r]['extra_by_kind'][k] for r, _ in rows])
        ax.barh(y, vals, left=left, color=col, height=0.6, edgecolor=SURF, lw=1.5, label=lab)
        left += vals
    for yy, tot in zip(y, left):
        ax.text(tot + 8, yy, f'{int(tot)}', va='center', fontsize=7.6, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in rows], fontsize=7.4)
    ax.set_xlabel('boxes at conf 0.25 that are not a found plate (1,153 test photos, 640 model)')
    ax.set_xlim(0, left.max() * 1.1)
    style(ax, 'x')
    ax.legend(loc='lower right', fontsize=7.3)
    save(fig, 'fig_kinds.png')


# ------------------------------------------------------------------------------------- 5. misses by plate width
def fig_bins():
    ser = [('v8s_640__FI', '640 model, FI', BLUE), ('v8s_640__SAHI_FI', '640 model, SAHI + FI (paper)', ORANGE),
           (BEST, '640 model, val-best slicing', AQUA), ('v8s_sf__FI', 'SF model, FI', YELLOW),
           ('v8s_1280__FI', '1280 px model, FI', MAGENTA)]
    bins = list(V['v8s_640__FI']['found_by_w640_%'])
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    w = 0.16
    x = np.arange(len(bins))
    for i, (k, lab, col) in enumerate(ser):
        miss = [V[k]['found_by_w640_%'][b][1] - V[k]['found_by_w640_%'][b][0] for b in bins]
        xs = x + (i - 2) * w
        ax.bar(xs, miss, width=w * 0.92, color=col, label=lab, edgecolor=SURF, lw=1)
        for xx, m in zip(xs, miss):
            ax.text(xx, m + 0.15, str(m), ha='center', va='bottom', fontsize=6.6, color=INK)
    n = [V['v8s_640__FI']['found_by_w640_%'][b][1] for b in bins]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.replace('-max', '+')} px\n({k} plates)" for b, k in zip(bins, n)], fontsize=7.4)
    ax.set_xlabel('plate width in the 640 px picture the detector sees')
    ax.set_ylabel('plates missed')
    style(ax)
    ax.legend(fontsize=7, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.3))
    save(fig, 'fig_bins.png')


# ------------------------------------------------------------------------------------------ 6. training curves
def fig_curves():
    H = A['health']['runs']
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    imgs = {'det_v8s_640': 8574, 'det_v8s_sf': A['health']['sf_set']['train_images_total'], 'det_v8s_1280': 8574}
    for r, lab, col in (('det_v8s_640', 'YOLOv8s 640, 60 epochs', BLUE), ('det_v8s_sf', 'YOLOv8s SF (photos + slices), 26 epochs', YELLOW),
                        ('det_v8s_1280', 'YOLOv8s 1280, 20 epochs', MAGENTA)):
        c = H[r]['val_curve']
        x = np.arange(1, len(c) + 1) * imgs[r] / 1e3
        ax.plot(x, c, color=col, lw=2, label=f"{lab} (best {max(c):.1f})")
    ax.set_ylim(90, 98.5)
    ax.set_xlabel('training images seen (thousands)')
    ax.set_ylabel('val mAP50 (%)')
    style(ax)
    ax.legend(fontsize=7.4, loc='lower right')
    save(fig, 'fig_curves.png')


# ---------------------------------------------------------------------------------- 7. working points theirs / ours
def fig_wp():
    W = A['visdrone']['working_point']
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axs[0]
    for d, name in (('tood', 'TOOD'), ('fcos', 'FCOS'), ('vfnet', 'VFNet')):
        a, b = W[f'{d}_fi'], W[f'{d}_sahi_fi_po']
        ax.add_patch(FancyArrowPatch((a['recall_%'], a['precision_%']), (b['recall_%'], b['precision_%']),
                                     arrowstyle='-|>', mutation_scale=10, color=INK2, lw=1.4, shrinkA=5, shrinkB=5))
        ax.plot([a['recall_%']], [a['precision_%']], 'o', ms=8, color=BLUE, mec=SURF, mew=1.5)
        ax.plot([b['recall_%']], [b['precision_%']], 'o', ms=8, color=ORANGE, mec=SURF, mew=1.5)
        ax.text(a['recall_%'] - 0.8, a['precision_%'], name, fontsize=7.4, color=INK, ha='right', va='center')
    ax.text(36, 36.5, 'blue = FI, orange = SAHI + FI + PO', fontsize=7, color=INK2)
    ax.set_xlim(33, 66); ax.set_ylim(34, 74)
    ax.set_title('VisDrone, authors\' predictions (score >= 0.3)', fontsize=8.4, loc='left')
    ax.set_xlabel('recall (%)'); ax.set_ylabel('precision (%)')
    style(ax)
    ax = axs[1]
    pts = [('v8s_640__FI', 'FI', BLUE), ('v8s_640__SAHI_FI', 'SAHI + FI (paper)', ORANGE), (BEST, 'val-best slicing', AQUA)]
    P0 = V['v8s_640__FI']
    for k, lab, col in pts:
        v = V[k]
        if k != 'v8s_640__FI':
            ax.add_patch(FancyArrowPatch((P0['found_%'], P0['correct_%']), (v['found_%'], v['correct_%']),
                                         arrowstyle='-|>', mutation_scale=10, color=col, lw=1.6))
        ax.plot([v['found_%']], [v['correct_%']], 'o', ms=8, color=col, mec=SURF, mew=1.5)
        ax.text(v['found_%'] + 0.08, v['correct_%'] + 1.2, lab, fontsize=7, color=INK2)
    ax.set_xlim(96.5, 99.3); ax.set_ylim(55, 95)
    ax.set_title('our plates, 640 model (conf >= 0.25)', fontsize=8.4, loc='left')
    ax.set_xlabel('plates found = recall (%)'); ax.set_ylabel('boxes that are plates = precision (%)')
    style(ax)
    fig.subplots_adjust(wspace=0.32)
    save(fig, 'fig_wp.png')


# --------------------------------------------------------------------------------------------------- 8. speed
def fig_speed():
    K, T = A['kaggle'], A['timing']['configs']
    rows = [('v8s_640__FI', '640, FI'), ('v8s_640__SAHI', '640, SAHI'), ('v8s_640__SAHI_FI', '640, SAHI + FI'),
            ('v8s_1280__FI', '1280, FI')]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    y = np.arange(len(rows))[::-1]
    h = 0.36
    ax.barh(y + h / 2, [K[k]['ms_median'] for k, _ in rows], height=h * 0.92, color=BLUE, label='Kaggle T4 (the run)', edgecolor=SURF)
    ax.barh(y - h / 2, [T[k]['median_ms'] for k, _ in rows], height=h * 0.92, color='#8a8984', label='our RTX 5060 Ti', edgecolor=SURF)
    for yy, (k, _) in zip(y, rows):
        ax.text(K[k]['ms_median'] + 1.5, yy + h / 2, f"{K[k]['ms_median']:.0f}", va='center', fontsize=7.2, color=INK)
        ax.text(T[k]['median_ms'] + 1.5, yy - h / 2, f"{T[k]['median_ms']:.0f}", va='center', fontsize=7.2, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in rows], fontsize=7.6)
    ax.set_xlabel('ms per test photo through the sahi package (median; read + model + merge)')
    style(ax, 'x')
    ax.legend(fontsize=7.4, loc='lower right')
    save(fig, 'fig_speed.png')


# ------------------------------------------------------------------------------------------- 9. example photos
def boxes_for(name, key, part='test'):
    v = next(x for x in C.VARIANTS if x['key'] == key)
    rec = C.load_cache(v['model'], v.get('size'), part)[name]
    b = C.merge(rec, v['mode'], v.get('rule', 'NMS_IOU'), small_px=v.get('small_px'))
    return b[b[:, 0] >= 0.25], rec


def draw(ax, img, g, boxes, slices=None, crop=None, title=''):
    ax.imshow(img[:, :, ::-1])
    if slices is not None:
        for s in slices:
            ax.add_patch(Rectangle((s[0] + 3, s[1] + 3), s[2] - s[0] - 6, s[3] - s[1] - 6, fill=False, ec='white', lw=0.9, ls='--'))
    for b in g:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec='#00d000', lw=1.6))
    for b, col in boxes:
        ax.add_patch(Rectangle((b[1], b[2]), b[3] - b[1], b[4] - b[2], fill=False, ec=col, lw=1.4))
    if crop:
        ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
    ax.set_title(title, fontsize=7.6, loc='left')
    ax.axis('off')


def around(b, wh, sx, sy):
    cx, cy, w, h = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * sx, (b[3] - b[1]) * sy
    return (max(0, cx - w / 2), max(0, cy - h / 2), min(wh[0], cx + w / 2), min(wh[1], cy + h / 2))


def fig_examples():
    photos = {f.name: (f, g, wh) for f, g, wh in C.test_photos()}
    # (a) a large plate cut by slice edges -> part-of-plate boxes after the paper's merge
    best = None
    for n, (f, g, wh) in photos.items():
        if len(g) != 1:
            continue
        b, rec = boxes_for(n, 'v8s_640__SAHI_FI')
        m = det.iou_matrix(b[:, 1:5], g) if len(b) else np.zeros((0, 1))
        part = int(((m < 0.5) & (m > 0.05)).sum())
        if part >= 2 and (m >= 0.5).any() and (best is None or part > best[0]):
            best = (part, n)
    na = best[1]
    L = json.load(open(OUT / 'fp_labels.json', encoding='utf-8'))
    FS = json.load(open(OUT / 'fp_sample.json', encoding='utf-8'))['sample']
    # (b) an unlabelled far plate on a "plate-free" photo, (d) a real false box (a sign), both from the eye-checked sample
    ib = max((i for i in L['plate'] if FS[i]['no_plate_photo']), key=lambda i: FS[i]['w640'])
    idd = 36 if 36 in L['not_plate'] else L['not_plate'][0]
    nb, nd = FS[ib]['file'], FS[idd]['file']
    # (c) a far plate only slicing finds
    nc = A['only_slicing_finds'][0]['file']
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    f, g, wh = photos[na]
    img = cv2.imread(str(f))
    b, rec = boxes_for(na, 'v8s_640__SAHI_FI')
    fi, _ = boxes_for(na, 'v8s_640__FI')
    draw(axs[0, 0], img, g, [(x, ORANGE) for x in b], slices=rec['slices'], crop=around(g[0], wh, 4, 7),
         title=f"(a) a large plate cut by slice edges (white): {best[0]} extra\nboxes on parts of it after the paper's merge")
    f, g, wh = photos[nb]
    img = cv2.imread(str(f))
    bx = np.array([[FS[ib]['score'], *FS[ib]['box']]])
    draw(axs[0, 1], img, g, [(x, ORANGE) for x in bx], crop=around(FS[ib]['box'], wh, 9, 14),
         title=f"(b) a 'plate-free' test photo: the slices find a far\nplate ({FS[ib]['w640']:.0f} px at 640) that the labels leave out")
    f, g, wh = photos[nc]
    img = cv2.imread(str(f))
    b, rec = boxes_for(nc, 'v8s_640__SAHI_FI')
    fi, _ = boxes_for(nc, 'v8s_640__FI')
    draw(axs[1, 0], img, g, [(x, BLUE) for x in fi] + [(x, ORANGE) for x in b], crop=around(g[0], wh, 9, 10),
         title=f"(c) a labelled far plate ({A['only_slicing_finds'][0]['w640']:.0f} px at 640): the full-image\npass misses it, the slices find it")
    f, g, wh = photos[nd]
    img = cv2.imread(str(f))
    bx = np.array([[FS[idd]['score'], *FS[idd]['box']]])
    draw(axs[1, 1], img, g, [(x, ORANGE) for x in bx], crop=around(FS[idd]['box'], wh, 4, 6),
         title='(d) a real false box from the slices:\na house-number board')
    fig.text(0.01, 0.01, 'green = labelled plate, orange = box from slicing (conf >= 0.25), blue = full-image box (conf >= 0.25)',
             fontsize=7.2, color=INK2)
    fig.subplots_adjust(wspace=0.06, hspace=0.28, bottom=0.05)
    save(fig, 'fig_examples.jpg')
    return {'a': na, 'b': nb, 'c': nc, 'd': nd}


if __name__ == '__main__':
    for n, fn in [('fig_pipeline', fig_pipeline), ('fig_sizes', fig_sizes), ('fig_main', fig_main), ('fig_kinds', fig_kinds),
                  ('fig_bins', fig_bins), ('fig_curves', fig_curves), ('fig_wp', fig_wp), ('fig_speed', fig_speed),
                  ('fig_examples', fig_examples)]:
        if want(n):
            r = fn()
            if r:
                (OUT / 'figure_files.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
