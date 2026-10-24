"""v1.1-run charts for the paper #23 study report (static PDF, light mode). Numbers: runs/leaky_vs_fair/results.json
and paired_v11.json of the 22 Oct 2026 Kaggle run."""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
S1, S2 = '#2a78d6', '#eb6834'          # categorical slots 1, 2 (validated pair)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'

plt.rcParams.update({
    'font.family': 'Segoe UI', 'font.size': 9, 'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'figure.facecolor': SURF,
    'axes.facecolor': SURF, 'savefig.dpi': 220, 'legend.frameon': False,
})


def style(ax, ymax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_ylim(0, ymax)


def bars(ax, groups, a, b, la, lb, width=0.32):
    x = range(len(groups))
    ra = ax.bar([i - width / 2 - 0.02 for i in x], a, width, color=S1, label=la, zorder=3)
    rb = ax.bar([i + width / 2 + 0.02 for i in x], b, width, color=S2, label=lb, zorder=3)
    for rects, vals in ((ra, a), (rb, b)):
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, f'{v + 1e-6:.1f}', ha='center', va='bottom', fontsize=7.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)


# detection, v1.1 run (fair detector trained for the same number of steps)
fig, ax = plt.subplots(figsize=(6.6, 2.9))
bars(ax, ['published test: all\n(553 photos)', 'clean part\n(474, no copy in train)', 'duplicated part\n(79, copy in train)'],
     [76.86, 76.04, 83.06], [71.99, 73.06, 68.51],
     'leaky detector (KUET split with copies)', 'fair detector (grouped split, same steps)')
style(ax, 100)
ax.set_ylabel('detection mAP50-95 (%)')
ax.legend(loc='upper left', fontsize=7.8, handlelength=1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_v11_det.png'))
plt.close(fig)

# plate reading, v1.1 run: both readers on the SAME plates
fig, ax = plt.subplots(figsize=(7.0, 3.0))
bars(ax, ['all 194 KUET\ntest plates\nMcNemar p = 0.008', 'B: plate is in the\nleaky train (114)\np = 0.017',
          'C: plate unseen\nby both (80)\np = 0.45', 'other datasets,\nunseen by both (468)\np = 1.0'],
     [85.57, 90.35, 78.75, 50.21], [77.84, 79.82, 75.00, 50.00], 'leaky reader', 'fair reader')
style(ax, 112)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_ylabel('whole plate read correctly (%)')
ax.legend(loc='upper right', fontsize=7.8, handlelength=1, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_v11_ocr.png'))
plt.close(fig)
print('v1.1 charts ok')
