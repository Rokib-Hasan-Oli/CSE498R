"""Charts for the paper #2 study report, v1.1 run (static PDF, light mode).

Run order: analysis.py (reads After train/results.zip) -> persource_det.py / ridoykj_figure.py (local GPU, optional,
they write persource_det.json, no_ridoykj.json, fig_boxstyle.png) -> charts.py -> build_report.py.
Numbers come from analysis.json (the v1.1 Kaggle run of 21 Oct 2026; v1.0 values are the 19 Oct log), the paper
(PLOS One e0355858) and dataset/docs/dedupe_report.md. Palette: validated categorical slots 1-4 (validate_palette.js).
"""
import json
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(OUT, 'analysis.json'), encoding='utf-8'))
PS = json.load(open(os.path.join(OUT, 'persource_det.json')))
NR = json.load(open(os.path.join(OUT, 'no_ridoykj.json')))
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


def pairbars(ax, groups, a, b, ca, cb, la, lb, ymax=112, fmt='{:.1f}'):
    w, gap = 0.32, 0.02
    ra = ax.bar([i - w / 2 - gap for i in range(len(groups))], a, w, color=ca, label=la, zorder=3)
    rb = ax.bar([i + w / 2 + gap for i in range(len(groups))], b, w, color=cb, label=lb, zorder=3)
    for rects, vals in ((ra, a), (rb, b)):
        for r, v in zip(rects, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 1.5, fmt.format(v), ha='center', va='bottom', fontsize=7.8, color=INK)
    style(ax)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7.8)


E = A['e2e']
pct = lambda run, key, s: 100 * E[f'{run}|{key}'][s]['exact'] / E[f'{run}|{key}'][s]['photos']
R = A['results']

# 1. End to end, same photos: authors' weights vs ours (paper's pipeline, with the vehicle stage) ----------------------
groups = ['All 204 photos', 'BLPD external\n(144, daylight, the paper\'s own)', 'NightLPBD\n(60, night)']
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.15))
key = 'with_vehicle_stage'
a = [pct('cascade_blpd_weights', key, s) for s in ('all', 'blpd_ext', 'nightlpbd')]
b = [pct('cascade_ours', key, s) for s in ('all', 'blpd_ext', 'nightlpbd')]
pairbars(axes[0], groups, a, b, S2, S1, "authors' weights (BLPD repo)", 'our weights (498R v1.1 data)')
axes[0].set_title('Whole plate read exactly (%)', loc='left', fontsize=9.5)
ca = [R['cascade_blpd_weights']['e2e'][key]['char_acc_%']] + [R['cascade_blpd_weights']['e2e'][key]['by_source'][s]['char_acc_%'] for s in ('blpd_ext', 'nightlpbd')]
cb = [R['cascade_ours']['e2e'][key]['char_acc_%']] + [R['cascade_ours']['e2e'][key]['by_source'][s]['char_acc_%'] for s in ('blpd_ext', 'nightlpbd')]
pairbars(axes[1], groups, ca, cb, S2, S1, '', '')
axes[1].set_title('Characters correct (%)', loc='left', fontsize=9.5)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, fontsize=8, handlelength=1, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(w_pad=2, rect=(0, 0, 1, 0.92))
save(fig, 'fig_e2e.png')

# 2. Speed: ms per photo -------------------------------------------------------------------------------------------
eo, eb = R['cascade_ours']['e2e'], R['cascade_blpd_weights']['e2e']
rows = [('Paper, RTX 3050: vehicle + plate + characters', 117.16, S2),
        ("Our T4: authors' weights, with vehicle stage", eb['with_vehicle_stage']['ms_mean'], S1),
        ('Our T4: our weights, with vehicle stage', eo['with_vehicle_stage']['ms_mean'], S1),
        ("Our T4: authors' weights, no vehicle stage", eb['plate_detector_only']['ms_mean'], S1),
        ('Our T4: our weights, no vehicle stage', eo['plate_detector_only']['ms_mean'], S1)]
fig, ax = plt.subplots(figsize=(8.6, 2.75))
y = list(range(len(rows)))[::-1]
for yi, (lab, ms, c) in zip(y, rows):
    ax.barh(yi, ms, height=0.5, color=c, zorder=3)
    ax.text(ms + 2, yi, f'{ms:.1f} ms  ({1000 / ms:.1f} FPS)', va='center', fontsize=8, color=INK, zorder=5,
            bbox=dict(fc=SURF, ec='none', pad=3))
ax.axvline(100, color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
ax.text(101, 4.42, '100 ms = 10 FPS (our goal)', fontsize=7.5, color=INK2, va='bottom')
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=8)
ax.set_xlim(0, 160)
ax.set_ylim(-0.6, 4.9)
ax.set_xlabel('mean time per photo (ms), lower is better')
style(ax, axis='x')
fig.tight_layout()
save(fig, 'fig_speed.png')

# 3. v1.0 vs v1.1: same code, same e2e photos, same plate detector; only the character model was retrained --------------
V = A['v10']
ocr11 = R['lpnr_v12n']['ocr_test']['rows_decoder']['plate_exact_%']
groups = ['Plate crops\n(651, our test)', 'Cascade, 204 photos\n(with vehicle stage)', 'Cascade, 204 photos\n(no vehicle stage)',
          'BLPD external\n(no vehicle stage)', 'NightLPBD\n(no vehicle stage)']
a = [V['lpnr']['ocr_exact'], V['ours_veh']['exact'], V['ours_noveh']['exact'], V['ours_noveh']['blpd_ext'], V['ours_noveh']['night']]
b = [ocr11, pct('cascade_ours', 'with_vehicle_stage', 'all'), pct('cascade_ours', 'plate_detector_only', 'all'),
     pct('cascade_ours', 'plate_detector_only', 'blpd_ext'), pct('cascade_ours', 'plate_detector_only', 'nightlpbd')]
fig, ax = plt.subplots(figsize=(9.6, 3.0))
pairbars(ax, groups, a, b, S4, S1, 'v1.0 run (19 Oct)', 'v1.1 run (21 Oct)')
ax.set_title('Whole plate read exactly (%), our weights', loc='left', fontsize=9.5)
ax.legend(loc='upper center', ncol=2, fontsize=8, handlelength=1, bbox_to_anchor=(0.5, 1.13))
fig.tight_layout()
save(fig, 'fig_v10_v11.png')

# 4. Conditions (v1.1) --------------------------------------------------------------------------------------------
L = R['lpd_v12s']['test_by_condition']
det = [('all test', 100 * R['lpd_v12s']['test']['mAP50']), ('clear', 100 * L['clear']['mAP50']),
       ('night', 100 * L['night']['mAP50']), ('blurry', 100 * L['blurry']['mAP50']),
       ('small plates', 100 * L['small_plates']['mAP50']), ('fog (synthetic)', 100 * L['fog_synth']['mAP50']),
       ('rain (synthetic)', 100 * L['rain_synth']['mAP50'])]
O = R['lpnr_v12n']['ocr_test']
rec = [('all test', O['rows_decoder']['plate_exact_%']), ('clear', O['by_condition']['clear']['plate_exact_%']),
       ('night', O['by_condition']['night']['plate_exact_%']), ('blurry', O['by_condition']['blurry']['plate_exact_%']),
       ('fog (real copies)', O['extra_tests']['test_fog']['plate_exact_%']),
       ('rain (synthetic)', O['extra_tests']['test_rain_synth']['plate_exact_%'])]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 2.9))
for ax, data, title in ((axes[0], det, 'Plate detector (LPD): mAP50 (%)'),
                        (axes[1], rec, 'Character reader (LPNR): whole plate exact (%)')):
    yy = list(range(len(data)))[::-1]
    for yi, (lab, v) in zip(yy, data):
        ax.barh(yi, v, height=0.52, color=S1, zorder=3)
        ax.text(v + 1, yi, f'{v:.1f}', va='center', fontsize=8, color=INK)
    ax.set_yticks(yy)
    ax.set_yticklabels([d[0] for d in data], fontsize=8)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    style(ax, axis='x')
    ax.set_title(title, loc='left', fontsize=9.5)
fig.tight_layout(w_pad=2.5)
save(fig, 'fig_conditions.png')

# 5. LPD per test source: mAP50-95 (box tightness) ----------------------------------------------------------------------
names = {'kuet': 'KUET', 'towniyan': 'towniyan', 'nightlpbd': 'NightLPBD', 'mahedi': 'mahedi', 'lpdba': 'LPDB-A',
         'tusar': 'tusar', 'ridoykj': 'ridoykj'}
order = sorted(PS, key=lambda s: -PS[s]['mAP50-95'])
fig, ax = plt.subplots(figsize=(8.6, 2.7))
yy = list(range(len(order)))[::-1]
for yi, s in zip(yy, order):
    v = PS[s]['mAP50-95']
    ax.barh(yi, v, height=0.52, color=S2 if s == 'ridoykj' else S1, zorder=3)
    ax.text(v + 1, yi, f"{v:.1f}   (mAP50 {PS[s]['mAP50']:.1f}, {PS[s]['photos']} photos)", va='center', fontsize=7.8, color=INK)
ax.set_yticks(yy)
ax.set_yticklabels([names[s] for s in order], fontsize=8)
ax.set_xlim(0, 112)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel('test mAP50-95 (%) of our plate detector, per source')
style(ax, axis='x')
fig.tight_layout()
save(fig, 'fig_persource.png')

# 6. Training curves (validation, per epoch) ---------------------------------------------------------------------------
Cv = A['curves']
fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.8))
for ax, run, title in ((axes[0], 'lpd_v12s', 'Plate detector (YOLO12s), validation'),
                       (axes[1], 'lpnr_v12n', 'Character detector (YOLO12n), validation')):
    c = Cv[run]
    ax.plot(c['epoch'], [100 * v for v in c['mAP50']], color=S1, lw=2, label='mAP50', zorder=3)
    ax.plot(c['epoch'], [100 * v for v in c['mAP50-95']], color=S2, lw=2, label='mAP50-95', zorder=3)
    ax.axvline(c['best_epoch'], color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
    ax.text(c['best_epoch'] + 0.8, 8, f"best.pt = epoch {c['best_epoch']}", fontsize=7.5, color=INK2)
    ax.text(c['epoch'][-1], 100 * c['mAP50'][-1] + 3, f"{100 * c['mAP50'][-1]:.1f}", ha='right', fontsize=7.5, color=INK)
    ax.text(c['epoch'][-1], 100 * c['mAP50-95'][-1] + 3, f"{100 * c['mAP50-95'][-1]:.1f}", ha='right', fontsize=7.5, color=INK)
    style(ax)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, c['epoch'][-1] + 1)
    ax.set_xlabel('epoch')
    ax.set_title(title, loc='left', fontsize=9.5)
axes[0].set_ylabel('%')
axes[0].legend(loc='lower right', fontsize=8, handlelength=1.5, bbox_to_anchor=(1.0, 0.12))
fig.tight_layout(w_pad=2)
save(fig, 'fig_curves.png')

# 7. What our cleaning did to BLPD's published images (unchanged from the first report) -----------------------------------
cats = [('kept', S1), ('duplicate of another image', S2), ('masked digits / mosaic / incomplete labels', S3),
        ('synthetic template plate', S4)]
bars = [('Detection set\n(2,140 photos)', [1139, 1001, 0, 0]),
        ('Character set\n(7,217 plate crops)', [185, 4894, 2019, 119])]
fig, ax = plt.subplots(figsize=(9.4, 2.15))
for yi, (lab, vals) in zip((1, 0), bars):
    tot = sum(vals)
    left = 0.0
    for (cname, c), v in zip(cats, vals):
        if not v:
            continue
        p = 100 * v / tot
        ax.barh(yi, p - 0.25, left=left, height=0.5, color=c, zorder=3)
        if p > 6:
            ax.text(left + p / 2, yi, f'{v:,}', ha='center', va='center', fontsize=8, color=INK)
        left += p
    if yi == 0:
        ax.annotate('185 kept', xy=(1.2, 0.27), xytext=(4, 0.62), fontsize=7.8, color=INK,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7))
        ax.annotate('119', xy=(99.3, -0.27), xytext=(95, -0.62), fontsize=7.8, color=INK,
                    arrowprops=dict(arrowstyle='-', color=INK2, lw=0.7))
ax.set_yticks([1, 0])
ax.set_yticklabels([b[0] for b in bars], fontsize=8)
ax.set_xlim(0, 100)
ax.set_ylim(-0.8, 1.45)
ax.set_xlabel('share of the published images (%)')
style(ax, axis='x')
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in cats]
ax.legend(handles, [n for n, _ in cats], loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=7.8, handlelength=1)
fig.tight_layout()
save(fig, 'fig_cleaning.png')

# 8. Pipeline: the paper's cascade vs our planned pipeline (unchanged) --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis('off')


def box(x, y, w, h, text, new=False, kept=False):
    fc = '#fde0d3' if new else ('#cde2fb' if kept else '#f2f1ee')
    ec = S2 if new else (S1 if kept else '#9a9994')
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.8', fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.6, color=INK, linespacing=1.25)


def arrow(x1, x2, yy):
    ax.add_patch(FancyArrowPatch((x1, yy), (x2, yy), arrowstyle='-|>', mutation_scale=8, color=INK2, lw=0.9))


ax.text(0, 34, "The paper's cascade (Sarker & Alam 2026)", fontsize=9, fontweight='bold', color=INK)
top = [('Photo', 7), ('YOLO12l (COCO)\nfinds vehicles', 13), ('LPD: YOLOv12\nfinds the plate\ninside the vehicle', 15),
       ('LPNR: YOLOv12\n104 classes: digits,\nletters, Metro, districts', 17), ('order the labels\n(Algorithm 1: by x;\ntext: by type)', 13),
       ('"Dhaka Metro\nGa 32-8301"', 12)]
x = 0.5
for i, (t, w) in enumerate(top):
    box(x, 21, w, 10, t, kept=i in (2, 3))
    if i < len(top) - 1:
        arrow(x + w + 0.4, x + w + 2.6, 26)
    x += w + 3
ax.text(0, 16.2, 'Our planned pipeline (proposal, Fig. 1)', fontsize=9, fontweight='bold', color=INK)
bot = [('Video frame', 8), ('light plate detector\n(no vehicle stage)', 13), ('character YOLO\n101 Bangla tokens\n(clean class list)', 14),
       ('two-row decoder\n(words above,\ndigits below)', 13), ('BRTA grammar +\ndistrict dictionary\ncheck', 13),
       ('tracker +\nmulti-frame\nvoting', 11), ('plate text\n(+ flag if invalid)', 12)]
x = 0.5
for i, (t, w) in enumerate(bot):
    box(x, 3, w, 10, t, new=i in (1, 3, 4, 5), kept=i == 2)
    if i < len(bot) - 1:
        arrow(x + w + 0.4, x + w + 2.0, 8)
    x += w + 2.4
ax.add_patch(FancyBboxPatch((60, 0.2), 1.6, 1.2, boxstyle='round,pad=0.1', fc='#cde2fb', ec=S1, lw=0.8))
ax.text(62.3, 0.8, 'same idea as the paper', fontsize=7.2, color=INK2, va='center')
ax.add_patch(FancyBboxPatch((79, 0.2), 1.6, 1.2, boxstyle='round,pad=0.1', fc='#fde0d3', ec=S2, lw=0.8))
ax.text(81.3, 0.8, 'our change / addition', fontsize=7.2, color=INK2, va='center')
fig.tight_layout()
save(fig, 'fig_pipeline.png')
print('charts ok')
