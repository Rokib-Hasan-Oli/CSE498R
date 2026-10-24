"""Charts for the paper #6 study report (v1.1 run, static PDF, light mode).

  D:/498R/_dataset_build/venv/Scripts/python.exe charts.py <p06 run dir> <p02 run dir> <paper figure dir>

Run order: local_check.py -> easyocr_check.py -> analysis.py -> charts.py -> build_report.py.
Numbers come from analysis.json, local_check.json and easyocr_check.json; images from the run's best.pt files, the
498R dataset v1.1 and the paper PDF (figures extracted with PyMuPDF). Palette: validated categorical slots 1-4 (as the
paper #8 report); values are labelled on every bar.
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
RUN, P02, PFIG = sys.argv[1:4]
A = json.load(open(os.path.join(OUT, 'analysis.json'), encoding='utf-8'))
LC = json.load(open(os.path.join(OUT, 'local_check.json'), encoding='utf-8'))
EL = A.get('easyocr_local')
DATA = 'D:/498R/dataset'
S1, S2, S3, S4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
plt.rcParams.update({
    'font.family': 'Segoe UI', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.dpi': 220, 'legend.frameon': False,
})
D, E, EO = A['det'], A['e2e'], A['easyocr']


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


def bars(ax, labels, series, colors, names, ymax=112, fmt='{:.1f}', w=0.26, fs=7.0):
    k = len(series)
    for j, (vals, c, nm) in enumerate(zip(series, colors, names)):
        xs = [i + (j - (k - 1) / 2) * (w + 0.02) for i in range(len(labels))]
        rs = ax.bar(xs, [v or 0 for v in vals], w, color=c, label=nm, zorder=3)
        for r, v in zip(rs, vals):
            if v is not None:
                ax.text(r.get_x() + r.get_width() / 2, v + 1.2, fmt.format(v), ha='center', va='bottom', fontsize=fs,
                        color=INK, rotation=90 if k > 2 else 0)
    style(ax)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7.8)


# ------------------------------------------------------------------ 1. pipeline with what we measured
fig, ax = plt.subplots(figsize=(7.4, 2.45))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis('off')
n, s = D['det_v8n'], D['det_v8s']
steps = [('Street / CCTV\nphoto', 'any size; plate is a\nsmall part of it', None),
         ('YOLOv8\n1 class "plate"', 'COCO-pretrained,\nfine-tuned (n and s)',
          f"plates found {n['plates']['found_%']:.1f} / {s['plates']['found_%']:.1f} %\nmAP50 {n['test']['mAP50']:.1f} / {s['test']['mAP50']:.1f}"),
         ('Crop the most\nconfident box', '+8 % margin\n(paper: "segment")', None),
         ('EasyOCR "bn"\nCRAFT + CRNN', 'stock model,\nno plate training', f"char acc. {EO['kit']['rows_decoder']['char_acc_%']:.1f} %\nplates exact {EO['kit']['rows_decoder']['plate_exact_%']:.1f} %"),
         ('Join the two\nlines', 'top line first\n(paper: not said)', None),
         ('Plate text', 'scored after\nnormalising', f"whole system: exact\n{E['kit']['e2e_v8n']['full_plate_exact_%']:.1f} % of 204 photos")]
bw, gap, x0 = 13.2, 3.3, 1.0
for i, (t, sub, meas) in enumerate(steps):
    x = x0 + i * (bw + gap)
    ax.add_patch(FancyBboxPatch((x, 19), bw, 13, boxstyle='round,pad=0.3,rounding_size=1.2', fc='#eef4fc', ec=S1, lw=1.0))
    ax.text(x + bw / 2, 25.5, t, ha='center', va='center', fontsize=8.0, color=INK, fontweight='bold')
    ax.text(x + bw / 2, 13.6, sub, ha='center', va='center', fontsize=6.4, color=INK2, linespacing=1.15)
    if meas:
        ax.add_patch(FancyBboxPatch((x - 0.8, 0.6), bw + 1.6, 7.6, boxstyle='round,pad=0.25,rounding_size=1',
                                    fc='#fdf0ea', ec=S2, lw=0.9))
        ax.text(x + bw / 2, 4.4, meas, ha='center', va='center', fontsize=5.9, color=INK)
    if i < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((x + bw + 0.5, 25.5), (x + bw + gap - 0.5, 25.5), arrowstyle='-|>',
                                     mutation_scale=9, color=INK2, lw=1))
fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
ax.text(0.5, 34.5, 'The paper\'s system (blue) and what our v1.1 run measured on Bangladeshi plates (orange; YOLOv8n / YOLOv8s)',
        fontsize=8.4, color=INK, fontweight='bold')
save(fig, 'fig_pipeline.png')

# ------------------------------------------------------------------ 2. training curves
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6), sharey=True)
for ax, key, title in ((axs[0], 'mAP50', 'Validation mAP50'), (axs[1], 'mAP50-95', 'Validation mAP50-95')):
    for r, c, nm in (('det_v8n', S1, 'YOLOv8n'), ('det_v8s', S2, 'YOLOv8s')):
        cv = D[r]['curve']
        ax.plot(cv['epoch'], [100 * v for v in cv[key]], color=c, lw=1.6, label=nm)
        b = D[r]['best_epoch']
        ax.plot([b], [100 * cv[key][b - 1]], 'o', ms=6, mfc=c, mec=SURF, mew=1.5, zorder=5)
    ref = 95 if key == 'mAP50' else 59
    ax.axhline(ref, color=MUTED, lw=1, ls='--')
    ax.text(40, ref - 4.2, f'paper Fig. 6 (p.11), last epoch: about {ref}', fontsize=6.8, color=INK2)
    style(ax)
    ax.set_title(title, fontsize=8.6, loc='left')
    ax.set_xlabel('epoch (dots: best epoch, the saved best.pt)', fontsize=7.4)
axs[0].set_ylim(40, 102)
axs[0].set_ylabel('%')
axs[0].legend(loc='lower right', fontsize=7.4)
fig.tight_layout()
save(fig, 'fig_curves.png')

# ------------------------------------------------------------------ 3. mAP50-95 per source (local check)
srcs = ['kuet', 'nightlpbd', 'lpdba', 'mahedi', 'tusar', 'towniyan', 'ridoykj']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.75), gridspec_kw={'width_ratios': [2.3, 1]})
L = LC['det']
bars(axs[0], srcs, [[L[m]['by_source'][x]['mAP50-95'] for x in srcs] for m in ('det_v8n', 'det_v8s', 'p02_lpd_v12s')],
     [S1, S2, S3], ['#6 YOLOv8n', '#6 YOLOv8s', '#2 YOLO12s (LPD)'], ymax=100)
axs[0].set_title('Test mAP50-95 per source', fontsize=8.4, loc='left')
axs[0].legend(fontsize=6.8, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
grp = ['val', 'test', 'test w/o\nridoykj']
bars(axs[1], grp, [[D[m]['val']['mAP50-95'], L[m]['test']['mAP50-95'], L[m]['test_without_ridoykj']['mAP50-95']]
                   for m in ('det_v8n', 'det_v8s')], [S1, S2], ['v8n', 'v8s'], ymax=100, w=0.36)
axs[1].axhline(59, color=MUTED, lw=1, ls='--')
axs[1].text(0.62, 61, 'paper ≈ 59', fontsize=6.8, color=INK2)
axs[1].set_title('Where the 45 comes from', fontsize=8.4, loc='left')
fig.tight_layout()
save(fig, 'fig_persource.png')

# ------------------------------------------------------------------ 4. box style: ridoykj vs kuet, label vs prediction
from ultralytics import YOLO  # noqa: E402
m8n = YOLO(os.path.join(RUN, 'runs', 'det_v8n', 'weights', 'best.pt'))
tdir = os.path.join(DATA, 'detection', 'images', 'test')
picks = [sorted(f for f in os.listdir(tdir) if f.startswith('ridoykj_'))[3],
         sorted(f for f in os.listdir(tdir) if f.startswith('kuet_'))[7]]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.7))
for ax, f in zip(axs, picks):
    img = cv2.imread(os.path.join(tdir, f))
    h, w = img.shape[:2]
    lab = os.path.join(DATA, 'detection', 'labels', 'test', f.rsplit('.', 1)[0] + '.txt')
    g = [list(map(float, l.split()[1:5])) for l in open(lab).read().splitlines() if l.strip()]
    r = m8n.predict(img, imgsz=640, conf=0.25, verbose=False, device='cpu')[0]
    boxes = [(b, c) for b, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy())]
    gx = [((x - bw_ / 2) * w, (y - bh_ / 2) * h, (x + bw_ / 2) * w, (y + bh_ / 2) * h) for x, y, bw_, bh_ in g]
    allb = gx + [tuple(b) for b, _ in boxes]
    x1 = max(0, min(b[0] for b in allb) - 0.6 * (max(b[2] for b in allb) - min(b[0] for b in allb)))
    x2 = min(w, max(b[2] for b in allb) + 0.6 * (max(b[2] for b in allb) - min(b[0] for b in allb)))
    y1 = max(0, min(b[1] for b in allb) - 1.2 * (max(b[3] for b in allb) - min(b[1] for b in allb)))
    y2 = min(h, max(b[3] for b in allb) + 1.2 * (max(b[3] for b in allb) - min(b[1] for b in allb)))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for b in gx:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=S3, lw=1.8, ls='--'))
    for b, c in boxes:
        ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=S2, lw=1.6))
    ax.set_xlim(x1, x2)
    ax.set_ylim(y2, y1)
    ax.axis('off')
    sname = f.split('_')[0]
    bs = A['det_box_shape']['det_v8n'][sname]
    ax.set_title(f'{sname}: model box / label box = {bs["median_w_ratio"]:.2f} wide, {bs["median_h_ratio"]:.2f} high\n'
                 f'(median of all {bs["n"]} found {sname} test plates), mean IoU {bs["mean_iou"]:.0f} %', fontsize=7.2, loc='left')
fig.text(0.01, 0.01, 'green dashed: human label box · orange: YOLOv8n box (conf ≥ 0.25)', fontsize=7.2, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, 'fig_boxstyle.png')

# ------------------------------------------------------------------ 5. conditions: detector and EasyOCR
p02 = json.load(open(os.path.join(P02, 'runs', 'lpd_v12s', 'results.json'), encoding='utf-8'))['test_by_condition']
conds = [('clear', 'clear'), ('night', 'night'), ('blurry', 'blurry'), ('small_plates', 'small\nplates'),
         ('rain_synth', 'rain\n(synthetic)'), ('fog_synth', 'fog\n(synthetic)')]
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.9), gridspec_kw={'width_ratios': [1.55, 1]})
bars(axs[0], [c[1] for c in conds], [[100 * D[m]['by_condition'][c]['mAP50'] for c, _ in conds] for m in ('det_v8n', 'det_v8s')] +
     [[100 * p02[c]['mAP50'] for c, _ in conds]], [S1, S2, S3], ['#6 YOLOv8n', '#6 YOLOv8s', '#2 YOLO12s'], ymax=118)
axs[0].set_title('Detector test mAP50 by condition', fontsize=8.4, loc='left')
axs[0].legend(fontsize=6.6, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.0))
bc = EO['kit']['by_condition']
ext = EO['kit']['extra_tests']
lab = ['clean /\nclear', 'low light /\nnight', 'weather /\nreal fog', 'rain\n(synthetic)']
paper = [97.5, 91.5, 89.3, 89.3]
ours = [bc['clear']['char_acc_%'], bc['night']['char_acc_%'], ext['test_fog']['char_acc_%'], ext['test_rain_synth']['char_acc_%']]
bars(axs[1], lab, [paper, ours], [MUTED, S2], ['paper (Tables 2, 4; metric undefined)', 'ours: char accuracy'], ymax=118, w=0.36)
axs[1].set_title('EasyOCR by condition', fontsize=8.4, loc='left')
axs[1].legend(fontsize=6.4, loc='upper center', ncol=1, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
save(fig, 'fig_conditions.png')

# ------------------------------------------------------------------ 6. readers on the same 651 crops
rows = [('EasyOCR + 4x/CLAHE prep (kit)', EO['kit']['prep']['plate_exact_%'], EO['kit']['prep']['char_acc_%'], MUTED),
        ('EasyOCR, stock (paper #6)', EO['kit']['rows_decoder']['plate_exact_%'], EO['kit']['rows_decoder']['char_acc_%'], S2)]
if EL:
    v = EL['variants']
    best = max(('allow', 'allow_x2', 'x2'), key=lambda k: v[k]['plate_exact_%'])
    nm = {'allow': 'EasyOCR + plate allowlist (local)', 'allow_x2': 'EasyOCR + allowlist + 2x (local)', 'x2': 'EasyOCR + 2x upscale (local)'}[best]
    rows.append((nm, v[best]['plate_exact_%'], v[best]['char_acc_%'], S4))
sm = A['same_651_crops']
rows += [('ViTSTR (paper #24)', sm['#24 vitstr (ViTSTR)']['a_%'], sm['#24 vitstr (ViTSTR)']['char_acc_other'], S1),
         ('CRNN (paper #24)', sm['#24 crnn (CRNN)']['a_%'], sm['#24 crnn (CRNN)']['char_acc_other'], S1),
         ('char-YOLOv8n (paper #24)', sm['#24 chars_v8n (char-YOLOv8n)']['a_%'], sm['#24 chars_v8n (char-YOLOv8n)']['char_acc_other'], S3),
         ('char-YOLO12n (paper #2)', sm['#2 lpnr_v12n (char-YOLO12n)']['a_%'], sm['#2 lpnr_v12n (char-YOLO12n)']['char_acc_other'], S3)]
fig, ax = plt.subplots(figsize=(7.4, 2.9))
y = np.arange(len(rows))
ax.barh(y + 0.19, [r[1] for r in rows], 0.36, color=[r[3] for r in rows], zorder=3, label='plate exact')
ax.barh(y - 0.19, [r[2] for r in rows], 0.36, color=[r[3] for r in rows], alpha=0.35, zorder=3, label='char accuracy')
for i, r in enumerate(rows):
    ax.text(r[1] + 1, i + 0.19, f'{r[1]:.1f} % exact', va='center', fontsize=7, color=INK)
    ax.text(r[2] + 1, i - 0.19, f'{r[2]:.1f} % char.', va='center', fontsize=7, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=7.6)
ax.axvline(92.7, color=MUTED, lw=1, ls='--')
ax.text(93.6, 0.1, 'paper: 92.7 %\n"character\naccuracy"\n(own data)', fontsize=6.6, color=INK2)
style(ax, 'x')
ax.set_xlim(0, 118)
ax.set_title('The same 651 human-typed test plates: plate exact (dark) and char accuracy (light)', fontsize=8.4, loc='left')
fig.tight_layout()
save(fig, 'fig_readers.png')

# ------------------------------------------------------------------ 7. whole photos: where the 204 photos are lost
X = LC['e2e_cross']
cases = [('YOLOv8n + EasyOCR\n(paper #6, Kaggle)', 'e2e_v8n', None), ('YOLOv8s + EasyOCR\n(paper #6, Kaggle)', 'e2e_v8s', None),
         ('YOLOv8n + char-YOLO12n\n(same crops, local)', 'e2e_v8n', 'det_v8n+lpnr_v12n/p06_protocol'),
         ('YOLOv8s + char-YOLO12n\n(same crops, local)', 'e2e_v8s', 'det_v8s+lpnr_v12n/p06_protocol'),
         ('paper #2 cascade\n(YOLO12s + char-YOLO12n)', None, 'p02_lpd_v12s+lpnr_v12n/p02_cascade')]
fig, ax = plt.subplots(figsize=(7.4, 2.7))
for i, (lab, k, x) in enumerate(cases):
    if x is None:
        ls = E['loss_split'][k]
        nob, wrong, right = ls['no_box'], ls['read_wrong'] + ls['box_but_empty_read'], ls['read_right']
    else:
        right = round(204 * X[x]['plate_exact_%'] / 100)
        nob = round(204 * (1 - X[x]['plate_detected_%'] / 100))
        wrong = 204 - right - nob
    left = 0
    for val, c, nm in ((right, S3, 'read exactly'), (wrong, S2, 'plate found, read wrong'), (nob, MUTED, 'no plate box')):
        ax.barh(i, val, left=left, color=c, zorder=3, label=nm if i == 0 else None)
        if val >= 6:
            ax.text(left + val / 2, i, str(val), ha='center', va='center', fontsize=7.4, color='white' if c != MUTED else INK)
        left += val
ax.set_yticks(range(len(cases)))
ax.set_yticklabels([c[0] for c in cases], fontsize=7.2)
ax.invert_yaxis()
style(ax, 'x')
ax.set_xlim(0, 204)
ax.set_xlabel('photos of e2e_test (144 BLPD external + 60 NightLPBD)', fontsize=7.4)
ax.legend(fontsize=7, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout()
save(fig, 'fig_e2e.png')

# ------------------------------------------------------------------ 8. speed
sp = {m: D[m]['speed'] for m in ('det_v8n', 'det_v8s')}
lab = ['PyTorch FP32', 'PyTorch FP16', 'TensorRT FP16', 'CPU ONNX']
keys = ['fp32', 'fp16', 'tensorrt_fp16', 'cpu_onnx']
fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw={'width_ratios': [1.6, 1]})
bars(axs[0], lab, [[sp[m][k]['fps'] for k in keys] for m in ('det_v8n', 'det_v8s')], [S1, S2], ['YOLOv8n', 'YOLOv8s'],
     ymax=215, fmt='{:.0f}', w=0.36)
axs[0].axhline(78, color=MUTED, lw=1, ls='--')
axs[0].text(2.6, 81, 'paper: 78 FPS (GPU not named)', fontsize=6.6, color=INK2)
axs[0].set_title('Detector FPS, batch 1, Tesla T4 (CPU: Kaggle 4-core VM)', fontsize=8.2, loc='left')
axs[0].legend(fontsize=7, loc='upper left')
parts = [('detector\n(TRT FP16)', sp['det_v8n']['tensorrt_fp16']['ms_per_image']),
         ('EasyOCR\nper plate', EO['kit']['ms_per_plate']),
         ('whole photo\n(kit, PyTorch)', E['kit']['e2e_v8n']['ms_mean'])]
axs[1].bar(range(3), [p[1] for p in parts], 0.55, color=[S1, S2, S4], zorder=3)
for i, p in enumerate(parts):
    axs[1].text(i, p[1] + 2, f'{p[1]:.1f} ms', ha='center', fontsize=7.2)
style(axs[1])
axs[1].set_xticks(range(3))
axs[1].set_xticklabels([p[0] for p in parts], fontsize=7.2)
axs[1].set_ylim(0, 112)
axs[1].set_title('Milliseconds (T4)', fontsize=8.2, loc='left')
fig.tight_layout()
save(fig, 'fig_speed.png')

# ------------------------------------------------------------------ 9. paper figures (from the PDF) + EasyOCR example crops
fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={'width_ratios': [0.62, 1.55, 1.15]})
for ax, (f, t) in zip(axs, (('p06_0.png', 'Fig. 3 (p.6): augmentation\nBEFORE the split'),
                            ('p11_0.png', 'Fig. 6 (p.11): an Ultralytics results.png\n(mAP50 ≈ 0.95, mAP50-95 ≈ 0.59)'),
                            ('p14_1.png', 'Fig. 8 (p.14): two lines, two results\n(letters 0.52, digits 0.96 confidence)'))):
    ax.imshow(plt.imread(os.path.join(PFIG, f)))
    ax.axis('off')
    ax.set_title(t, fontsize=7.2, loc='left')
fig.tight_layout()
save(fig, 'fig_paper.png')

ex = EO['picked_examples']
fig, axs = plt.subplots(1, len(ex), figsize=(7.4, 1.25))
for ax, e in zip(axs, ex):
    im = cv2.cvtColor(cv2.imread(os.path.join(DATA, 'ocr', 'images', 'test', e['file'])), cv2.COLOR_BGR2RGB)
    ax.imshow(im)
    ax.axis('off')
    ax.set_title(f"({'abcde'[ex.index(e)]}) {e['case']}", fontsize=7.2, loc='left')
fig.tight_layout()
save(fig, 'fig_examples.png')
print('charts done')
