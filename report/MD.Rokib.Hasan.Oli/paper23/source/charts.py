"""Charts for the paper #23 study report (static PDF, light mode)."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
S1, S2 = '#2a78d6', '#eb6834'          # categorical slots 1, 2 (validated pair)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'

plt.rcParams.update({
    'font.family': 'Segoe UI', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlecolor': INK,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'savefig.dpi': 220, 'legend.frameon': False,
})


def style(ax, ymax=None):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if ymax:
        ax.set_ylim(0, ymax)


def bars(ax, groups, a, b, la, lb, fmt='{:.1f}', width=0.36):
    x = range(len(groups))
    gap = 0.02
    ra = ax.bar([i - width / 2 - gap for i in x], a, width, color=S1, label=la, zorder=3)
    rb = ax.bar([i + width / 2 + gap for i in x], b, width, color=S2, label=lb, zorder=3)
    for rects, vals in ((ra, a), (rb, b)):
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height(), fmt.format(v), ha='center', va='bottom',
                    fontsize=7.5, color=INK, zorder=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)


# 1. Paper: error rate, original vs fair split, small multiples ------------------------------
models = ['CNNG', 'Holistic', 'Multi-Task', 'STAR-Net', 'TRBA', 'ViTSTR']
data = {
    'AOLP-A (random 2:1 split)': ([98.88, 96.75, 97.33, 98.69, 99.18, 98.74], [95.63, 93.11, 93.79, 95.83, 96.94, 96.94]),
    'AOLP-B (train AC+LE, test RP)': ([98.91, 98.42, 98.42, 98.47, 98.75, 98.75], [96.80, 96.30, 95.29, 96.46, 97.47, 97.31]),
    'CCPD (standard split)': ([88.24, 77.01, 83.01, 78.53, 75.83, 79.06], [86.93, 75.41, 81.84, 73.33, 71.48, 76.37]),
}
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
for ax, (title, (orig, fair)) in zip(axes, data.items()):
    eo = [100 - v for v in orig]
    ef = [100 - v for v in fair]
    bars(ax, models, eo, ef, 'original split', 'fair split', fmt='{:.1f}', width=0.34)
    style(ax, max(ef) * 1.22)
    ax.set_title(title, loc='left', fontsize=9.5)
    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=7.8)
axes[0].set_ylabel('error rate (%) = 100 - recognition rate')
axes[0].legend(loc='upper left', fontsize=8, handlelength=1, ncol=2, bbox_to_anchor=(0, 1.0))
fig.tight_layout(w_pad=1.6)
fig.savefig(os.path.join(OUT, 'fig_paper_errors.png'))
plt.close(fig)

# 2. Ours: detection mAP50-95 on clean vs duplicated part of KUET's published test -----------
fig, ax = plt.subplots(figsize=(6.4, 2.9))
groups = ['published test: all\n(553 photos)', 'clean part\n(474, no copy in train)', 'duplicated part\n(79, copy in train)']
bars(ax, groups, [76.86, 76.04, 83.06], [74.45, 75.93, 68.24], 'leaky model (trained on KUET split with copies)',
     'fair model (trained on our grouped split)', fmt='{:.1f}', width=0.32)
style(ax, 100)
ax.set_ylabel('detection mAP50-95 (%)')
ax.legend(loc='upper left', fontsize=7.8, handlelength=1, ncol=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_ours_det.png'))
plt.close(fig)

# 3. Ours: OCR plate-exact on 183 KUET test plates split by what the leaky model saw ----------
fig, ax = plt.subplots(figsize=(6.4, 2.9))
groups = ['A: exact same crop\nwas in leaky train (104)', 'B: same plate, other photo\nin leaky train (41)',
          'C: plate never in\nany training set (38)']
bars(ax, groups, [94.23, 87.80, 68.42], [71.15, 75.61, 71.05], 'leaky reader', 'fair reader', fmt='{:.1f}', width=0.32)
style(ax, 110)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('whole plate read correctly (%)')
ax.legend(loc='upper right', fontsize=7.8, handlelength=1, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_ours_ocr.png'))
plt.close(fig)

# 4. Concept: split by file vs split by group ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.7))
plates = [('P1', 3), ('P2', 2), ('P3', 4), ('P4', 2), ('P5', 3)]
cols = ['#cde2fb', '#fde0d3', '#d2f0e4', '#fbe8b8', '#f6dbe6']


def draw(ax, assign, title):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    ax.set_title(title, loc='left', fontsize=9.5)
    for k, (name, x0) in enumerate((('TRAIN', 0.1), ('TEST', 6.3))):
        ax.add_patch(FancyBboxPatch((x0, 0.15), 3.6 if name == 'TEST' else 5.9, 3.5, boxstyle='round,pad=0.02,rounding_size=0.15',
                                    fc='#f7f7f5', ec=GRID, lw=1))
        ax.text(x0 + 0.15, 3.35, name, fontsize=8.5, color=INK2, fontweight='bold')
    pos = {'TRAIN': [0.35, 2.55], 'TEST': [6.55, 2.55]}
    for (name, n), c in zip(plates, cols):
        for j in range(n):
            side = assign(name, j)
            x, y = pos[side]
            ax.add_patch(FancyBboxPatch((x, y), 0.95, 0.5, boxstyle='round,pad=0.01,rounding_size=0.08', fc=c, ec='#9a9994', lw=0.6))
            ax.text(x + 0.475, y + 0.25, f'{name}.{j + 1}', ha='center', va='center', fontsize=6.8, color=INK)
            pos[side][0] += 1.1
            limit = 5.6 if side == 'TRAIN' else 9.6
            if pos[side][0] > limit - 0.9:
                pos[side][0] = 0.35 if side == 'TRAIN' else 6.55
                pos[side][1] -= 0.7


rand = {('P1', 2): 1, ('P2', 1): 1, ('P3', 3): 1, ('P5', 0): 1}
draw(axes[0], lambda p, j: 'TEST' if (p, j) in rand else 'TRAIN',
     'Split by FILE (leaky): photos of the same plate land on both sides')
draw(axes[1], lambda p, j: 'TEST' if p in ('P2', 'P4') else 'TRAIN',
     'Split by GROUP (fair): all photos of one plate stay together')
fig.text(0.01, 0.01, 'P1.1, P1.2 ... = different photos (or Roboflow copies) of plate P1. Same colour = same physical plate.',
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(os.path.join(OUT, 'fig_concept.png'))
plt.close(fig)
print('charts ok')
