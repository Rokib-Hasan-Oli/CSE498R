"""SAHI study report (dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Akyon, Altinuc and Temizel, ICIP 2022, arXiv 2202.06934 v5; page numbers = PDF pages); the v1.1
Kaggle run (`After Train/results.zip`, `paper-36.log`, 28 Oct 2026) via analysis.json; the local re-check on the
RTX 5060 Ti (local_check.py: raw boxes of every slice, box-by-box check against the sahi package, merge variants, val
stats, CRNN reads, timing); the authors' released VisDrone predictions (original check of 27 Oct, visdrone_extra.py);
my by-eye check of 48 slice-only boxes (fp_sample.py, fp_labels.json); the 498R proposal (RQ3, RQ4, G1, G3, G4, G6, O2,
O4, O5). Other papers are named, not numbered (team request).
Run order: local_check.py -> visdrone_extra.py -> analysis.py -> fp_sample.py -> charts.py -> build_report.py.
"""
import json
import math
import os
import sys

from part_a import part_a
from report_head import M, f1, f2, fig, pv, table, ul, write_pdf

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'SAHI_Study_Report.pdf')
J = lambda n: json.load(open(os.path.join(HERE, n), encoding='utf-8'))
A = J('analysis.json')
H, K, V, P, VS = A['health'], A['kaggle'], A['variants'], A['pairs'], A['valstats']
E, EP, T, VD, CK = A['e2e'], A['e2e_pairs'], A['timing']['configs'], A['visdrone'], A['check']
FR, VF = J('e2e_fragile.json'), J('verify.json')
FS, FL = J('fp_sample.json'), J('fp_labels.json')
R = H['runs']
BEST = A['val_scores']['v8s_640']['best_slicing']
BEST_SF = A['val_scores']['v8s_sf']['best_slicing']
VAL = A['val_scores']
W = VD['working_point']
N_PL, N_UN, N_NOT = len(FL['plate']), len(FL['unclear']), len(FL['not_plate'])


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


WL = wilson(N_PL, N_PL + N_UN + N_NOT)
q = lambda d: f"{d['only_a']} vs {d['only_b']}, {pv(d['p'])}"
ci = lambda d: f"{d['diff']:+.2f} (95 % CI {d['ci95'][0]:+.2f} to {d['ci95'][1]:+.2f})"
pair = lambda a, b: P[f'{a}|{b}']
nf = lambda k: f"{V[k]['found_n']}"
mins = {j: m for j, _, m in H['jobs']}
TS = H['speed']
TRT = {k: TS[k]['tensorrt_fp16']['fps'] for k in TS}
SFS = H['sf_set']
small = lambda k: V[k]['small_plates_found']
vsp = VS['pairs']
NAME = {'v8s_640__FI': '640 model, full image (FI)', 'v8s_640__SAHI': '640 model, SAHI (slices only)',
        'v8s_640__SAHI_FI': '640 model, SAHI + FI', 'v8s_sf__FI': 'SF model, FI', 'v8s_sf__SAHI': 'SF model, SAHI',
        'v8s_sf__SAHI_FI': 'SF model, SAHI + FI', 'v8s_1280__FI': '1280 px model, FI'}

# ---------------------------------------------------------------------------------------------------- the numbers
assert all(c == 0 for _, c, _ in H['jobs']) and len(H['jobs']) == 7 and H['env']['dataset_version'] == '1.1'
assert all(H['code_same'].values())
assert all(v['max_abs_diff'] == 0.0 and v['photos_with_other_count'] == 0 for v in VF['box_by_box'].values())
assert VF['greedy_nmm_max_abs_diff'] == 0.0
assert all(abs(a - b) < 0.0301 for r in VF['rescore'].values() for k, (a, b) in r.items() if k != "found_by_condition_%")
assert all(a == b for r in VF['rescore'].values() for k, (a, b) in r.items() if k == 'found_by_condition_%')
assert all(v['val_mAP50'] == v['mine'] for v in VS['mAP50'].values())
assert BEST == 'v8s_640__SAHI_FI__BORDER' and R['det_v8s_640']['epochs_run'] == 60 and R['det_v8s_1280']['epochs_run'] == 20
assert K['v8s_640__SAHI_FI']['plates_by_size']['small'] == 5 and N_PL + N_UN + N_NOT == 48
assert CK['max_abs_diff_rescored_vs_paper_AP50'] == 0.0 and len(CK['settings']) == 20

S = part_a(A)
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER STUDY REPORT · DATASET v1.1 RUN</p>
<h1 style="font-size:18pt; margin-bottom:3pt">SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection</h1>
<p style="font-family:H; font-size:11.5pt; color:#1c5cab; margin-bottom:8pt">The paper, our SAHI run on dataset v1.1, whether
the run is sound, what slicing does to Bangla plates, how to use it in our research, and what to show and say to the
faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 29 October 2026<br>
Paper: F. C. Akyon, S. O. Altinuc and A. Temizel, IEEE ICIP 2022 (arXiv 2202.06934 v5, 5 pages; page numbers are PDF
pages). Code: github.com/obss/sahi (MIT), pinned {M('sahi==0.12.8')}; the authors' prediction files from
github.com/fcakyon/small-object-detection-benchmark.<br>
Our run: {M('paper36_kaggle.ipynb')} on Kaggle, Tesla T4 x2, 28 Oct 2026, dataset <b>v1.1</b> (confirmed in
{M('environment.json')}), {H['total_h']:.2f} h, 7 jobs. Extra checks on our own PC on 29 Oct (RTX 5060 Ti): every
slice's boxes saved and re-merged, box-by-box check against the package, merge rules chosen on the validation split,
paired tests, end-to-end reading, timing, and a look inside the authors' VisDrone predictions.</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> To find small, far objects in big photos, run the detector on overlapping tiles
(slices) of the photo and merge the boxes (SAHI), and add tiles to the training set (SF). On drone photos (VisDrone)
this lifts TOOD from 29.4 to 34.7 AP50 by slicing alone and to 43.5 with slice training (p.3, Table 1). It works on top
of any detector, with no speed numbers and no "just use a bigger input" baseline.</p>
<p><b>Is our run OK?</b> <b>Yes.</b> Dataset v1.1, 7 of 7 jobs ended with code 0 in {H['total_h']:.2f} h, every training
ran all its epochs, no NaN, the code is the kit's. On our PC the same weights give the same numbers for all 7 settings, and
our rebuild of the package's merge equals the package box for box. The paper's own Table 1 is reproduced 80 of 80 from
the authors' released files.</p>
<p><b>What we found.</b></p>
{ul([
    f"<b>The paper's SAHI makes our detector worse.</b> 640 model: AP50 {f2(V['v8s_640__FI']['AP50'])} (full image) -> {f2(V['v8s_640__SAHI_FI']['AP50'])} (slices + full image), {ci(pair('v8s_640__SAHI_FI', 'v8s_640__FI')['AP50'])}; boxes that are plates {f1(V['v8s_640__FI']['correct_%'])} % -> {f1(V['v8s_640__SAHI_FI']['correct_%'])} %; plate-free photos with a box {V['v8s_640__FI']['false_alarm_n']} -> {V['v8s_640__SAHI_FI']['false_alarm_n']} of 160; 3.5x the time.",
    f"<b>Why:</b> our plates are big. Median plate width is {VD['ours']['width_share_%']['50']:.1f} % of the photo (VisDrone objects: {VD['visdrone']['width_share_%']['50']:.1f} %); only {K['v8s_640__FI']['plates_by_size']['small']} of 1,011 test plates are under 32 x 32 px. Slice edges cut plates into pieces ({V['v8s_640__SAHI_FI']['extra_by_kind']['part_of_plate']} extra boxes on parts of plates) that the paper's NMS rule cannot merge.",
    f"<b>A one-line fix</b> (drop slice boxes that touch an inner slice edge, chosen on validation) removes {V['v8s_640__SAHI_FI']['extra_by_kind']['part_of_plate'] - V[BEST]['extra_by_kind']['part_of_plate']} of the {V['v8s_640__SAHI_FI']['extra_by_kind']['part_of_plate']} part-of-plate boxes: AP50 {f2(V[BEST]['AP50'])} = full image (n.s.), {V[BEST]['found_n']} vs {V['v8s_640__FI']['found_n']} plates found ({q(pair(BEST, 'v8s_640__FI')['found'])}), all 24 far plates of the small-plate list found (full image 22).",
    f"<b>Half of the remaining \"false\" boxes are real plates.</b> In a seeded sample of 48 slice-only boxes, {N_PL} are number plates on far or secondary vehicles that our labels leave out, {N_NOT} are signs or painted text. Our test labels cannot reward far plates, so slicing looks worse than it is.",
    f"<b>Training on slices (SF) is the cheap win for far plates:</b> same speed as the normal detector, small-plate photos mAP50 {f1(VS['mAP50']['v8s_640|small_plates']['val_mAP50'])} -> {f1(VS['mAP50']['v8s_sf|small_plates']['val_mAP50'])} ({ci(vsp['v8s_sf-v8s_640|small_plates'])}), but {vsp['v8s_sf-v8s_640|clear']['diff']:+.1f} on clear photos. A 1280 px detector is no better overall and 1.5x slower.",
])}
<p><b>Use in our research:</b> keep the 640 px full-image detector for the pipeline (O2, the 10 FPS goal); use the
slice-edge-filtered SAHI only as an optional second pass for far plates (CCTV, the NSU street set); try SF when we
re-train. First fix the label policy for far plates (G3). What to show the faculty: Part E.</p>
</div>
<p class="small">Contents: Part A the paper (sections 1-6) · Part B the run and whether it is sound (7-9) · Part C results
and what they mean (10-17) · Part D how to use it in our research (18-20) · Part E what to show and explain to the
faculty (21-23) · Appendix.</p>
"""

S['b1'] = f"""
<p class="part">PART B · OUR RUN AND WHETHER IT IS SOUND</p>
<h1>7. What the notebook ran</h1>
<p>One Kaggle notebook ({M('paper36_kaggle.ipynb')}, cell for cell the kit of 27 Oct 2026, {M('FAST = False')}), Tesla T4 x2,
28 Oct 2026. Data: 498R BanglaLPR All-in-One <b>v1.1</b>, detection split 8,574 train / 896 val / 1,153 test photos (val and
test human-labelled; groups never cross splits). 1,050 of the 1,153 test photos are larger than 640 px on the long side
(most are 960 x 1280 phone photos), so the slicing has something to cut.</p>
{table(['Job', 'GPU', 'What', 'Epochs run / planned', 'Best epoch', 'Minutes'], [
    [M('sf_slices'), 'CPU', f"SF set: {SFS['photos_sliced']:,} train photos &gt; 800 px cut into 640 x 640 slices, 20 % overlap; a plate is kept in a slice when &ge; 50 % of it is inside ({SFS['plates_cut_off_dropped']:,} cut-off plate pieces dropped); {SFS['slices_with_plates']:,} slices with a plate + {SFS['slices_without']:,} without (a quarter kept) = {SFS['train_images_total']:,} training images", '–', '–', '1.8'],
    [M('det_v8s_640'), '0', 'YOLOv8s at 640 px (the usual detector, also used by SAHI)', f"{R['det_v8s_640']['epochs_run']} / 60", str(R['det_v8s_640']['best_epoch']), f1(mins['det_v8s_640'])],
    [M('det_v8s_sf'), '1', 'YOLOv8s at 640 px on photos + slices (SF); 26 epochs on 2.29x the images = the optimiser steps of 60 epochs', f"{R['det_v8s_sf']['epochs_run']} / 26", str(R['det_v8s_sf']['best_epoch']), f1(mins['det_v8s_sf'])],
    [M('det_v8s_1280'), '1 (after)', 'YOLOv8s at 1280 px, batch 8: "use a bigger input" instead of slicing', f"{R['det_v8s_1280']['epochs_run']} / 20", str(R['det_v8s_1280']['best_epoch']), f1(mins['det_v8s_1280'])],
    [M('crnn'), '0 (after)', 'CRNN plate reader for the end-to-end photos', '60 / 60', '–', f1(mins['crnn'])],
    [M('sahi'), '0 (after)', 'FI / SAHI / SAHI + FI for the 640 and SF models, FI for the 1280 model, through the sahi package with the paper\'s merge (NMS, IoU 0.5), 640 x 640 slices with 25 % overlap', '–', '–', f1(mins['sahi'])],
    [M('speed_*'), '0 (after)', 'TensorRT FP16 / ONNX timing of the 640 and 1280 models', '–', '–', f1(mins['speed_det_v8s_640'] + mins['speed_det_v8s_1280'])],
], [1.4, 0.8, 4.6, 1.2, 0.8, 0.8], num=(4, 5))}
<p>All YOLO runs: SGD lr 0.01, seed 42, the Ultralytics 8.4.170 defaults otherwise (mosaic, close_mosaic 10). AP50 in the
SAHI job is pycocotools on the package's boxes at conf 0.01 with up to 500 boxes per photo (the paper's metric); "plates
found" and "boxes that are plates" use conf 0.25 and IoU 0.5 as in every kit.</p>

<h1>8. Health check</h1>
{table(['Check', 'Result', 'OK?'], [
    ['Dataset version', f"{M('environment.json')}: dataset_version {H['env']['dataset_version']}; counts printed in the log = v1.1 manifest", '✓'],
    ['All jobs finished', f"7 of 7 {M('[done] ... code=0')} in {H['total_h']:.2f} h (planned 7.5 h)", '✓'],
    ['Training complete', f"60 / 60, 26 / 26 and 20 / 20 epochs, no time cap hit; best epochs {R['det_v8s_640']['best_epoch']}, {R['det_v8s_sf']['best_epoch']}, {R['det_v8s_1280']['best_epoch']}; no NaN in any results.csv", '✓'],
    ['Same code as the kit', f"{M('code/*.py')} = the kit's files (two differ only in line endings)", '✓'],
    ['Numbers reproduce', 'all 7 Kaggle settings re-scored on our PC from the same weights: AP50, small / medium / large, plates found, boxes correct, false alarms and found by condition equal to 0.03', '✓'],
    ['Our merge = the package', f"{M('local_check.py verify')}: our raw-slice rebuild of FI, SAHI and SAHI + FI equals {M('get_sliced_prediction')} box for box on 80 photos per setting (max difference 0.0); our greedy-merge rebuild equals sahi's {M('GreedyNMMPostprocess')}", '✓'],
    ['Condition numbers', 'Ultralytics val per subset re-run locally: every mAP50 equal to the Kaggle table (e.g. small plates 84.53 / 96.30 / 91.39)', '✓'],
    ['640 detector', 'tensor for tensor identical to the 640 detector of the HVI-CIDNet study run (the kit is deterministic); the CRNN file is the Model Fusion study\'s (SHA-256 0110631e…)', '✓'],
    ['Original check', f"the authors' 20 VisDrone result files re-scored: Table 1 reproduced 80 / 80 (largest difference {CK['max_abs_diff_rescored_vs_paper_AP50']:.1f})", '✓'],
    ['End-to-end reads', f"local e2e plate exact is 1 photo lower than Kaggle in 5 of 7 settings: {FR['flip_with_1px']} of {FR['photos_with_box']} e2e readings change when the plate box moves by 1 px, so tiny GPU differences flip one", '~ note'],
], [2.2, 6.6, 1.2])}
"""

S['b2'] = f"""
<h1>9. Points to keep in mind when reading the numbers</h1>
{ul([
    f"<b>COCO \"small\" means almost nothing here.</b> Only {K['v8s_640__FI']['plates_by_size']['small']} test plates are smaller than 32 x 32 px in the photo (medium {K['v8s_640__FI']['plates_by_size']['medium']}, large {K['v8s_640__FI']['plates_by_size']['large']}). The kit's \"AP50 small\" and \"small plates found\" rows (40 -> 80 %) are 2 of 5 plates. This report uses two better yardsticks: the plate's width in the 640 px picture the detector sees, and the team's small-plate list (22 photos, 24 plates whose area is below 32 x 32 px at 640).",
    f"<b>The 1280 px model got only 20 epochs</b> (the T4 time budget) and its best epoch was the 19th, so it was still improving. It saw a third of the images the 640 model saw. Its result is a lower bound for \"use a bigger input\".",
    f"<b>SF had the same number of optimiser steps</b> as the 640 model, but passed over the full photos only 26 times instead of 60.",
    f"<b>The test labels leave out far plates.</b> Several sources label only the main (readable) plate, and 160 test photos are listed as plate-free although some show far plates (section 13). Any box on such a plate counts as a false box, which hits slicing hardest.",
    f"<b>End-to-end differences of a few photos are noise:</b> {FR['flip_with_1px']} of {FR['photos_with_box']} readings flip when the box moves by one pixel.",
    "<b>ms per photo through the sahi package</b> include reading the JPEG and the package's Python work, not only the network; TensorRT FPS (section 17) are the network alone.",
    "<b>Two row labels in the kit's results table are loose</b> (no number is wrong): \"AP50 large plates: 640 FI / SAHI + FI\" shows only the SAHI + FI value, and the \"small plates found ... gain\" row is the +40 points on 5 plates. A hand note in the notes explains both; the kit of 29 Oct fixes them (section 20).",
])}
"""

S['c1'] = f"""
<p class="part">PART C · RESULTS AND WHAT THEY MEAN</p>
<h1>10. The kit's seven settings, and our fix</h1>
{table(['Setting', 'AP50', 'AP50 large', 'Plates found', 'Boxes that are plates', 'Plate-free photos with a box', 'Slices / photo', 'ms (T4)'], [
    [NAME[k], f2(V[k]['AP50']), f2(V[k]['coco']['AP50_large_%']), f"{V[k]['found_n']} ({f2(V[k]['found_%'])} %)", f1(V[k]['correct_%']),
     f"{V[k]['false_alarm_n']} / 160", f"{K[k].get('slices_per_photo', 1):.2f}", f1(K[k]['ms_median'])] for k in NAME] + [
    ['<b>640 model, SAHI + FI, cut-off boxes dropped</b> (val-best)', f"<b>{f2(V[BEST]['AP50'])}</b>", '–', f"<b>{V[BEST]['found_n']} ({f2(V[BEST]['found_%'])} %)</b>", f1(V[BEST]['correct_%']), f"{V[BEST]['false_alarm_n']} / 160", '5.96', '≈118'],
], [3.2, 0.8, 0.9, 1.3, 1.0, 1.2, 0.8, 0.8], num=(1, 2, 3, 4, 5, 6, 7))}
<p class="small">1,153 test photos, 1,011 plates, 160 photos without a labelled plate. AP50 = COCO at conf 0.01, max 500
boxes; the other columns at conf 0.25, IoU 0.5. All rows equal on Kaggle and on our PC.</p>
{fig('fig_main.png', 470, 'Figure 2. The settings side by side. Blue: full image only; orange: slices with the paper\'s merge; aqua: slices with cut-off slice boxes dropped (section 12).')}
<h3>Which differences are real? (paired tests on the same photos)</h3>
{table(['Comparison', 'AP50 difference (photo bootstrap)', 'Plates found: only A vs only B', 'Plate-free photos with a box: only A vs only B'], [
    ['640: SAHI (slices only) vs FI', ci(pair('v8s_640__SAHI', 'v8s_640__FI')['AP50']), q(pair('v8s_640__SAHI', 'v8s_640__FI')['found']), q(pair('v8s_640__SAHI', 'v8s_640__FI')['false_alarm'])],
    ['640: SAHI + FI vs FI', ci(pair('v8s_640__SAHI_FI', 'v8s_640__FI')['AP50']), q(pair('v8s_640__SAHI_FI', 'v8s_640__FI')['found']), q(pair('v8s_640__SAHI_FI', 'v8s_640__FI')['false_alarm'])],
    ['SF: SAHI + FI vs FI', ci(pair('v8s_sf__SAHI_FI', 'v8s_sf__FI')['AP50']), q(pair('v8s_sf__SAHI_FI', 'v8s_sf__FI')['found']), q(pair('v8s_sf__SAHI_FI', 'v8s_sf__FI')['false_alarm'])],
    ['SF model FI vs 640 model FI', ci(pair('v8s_sf__FI', 'v8s_640__FI')['AP50']), q(pair('v8s_sf__FI', 'v8s_640__FI')['found']), q(pair('v8s_sf__FI', 'v8s_640__FI')['false_alarm'])],
    ['1280 model FI vs 640 model FI', ci(pair('v8s_1280__FI', 'v8s_640__FI')['AP50']), q(pair('v8s_1280__FI', 'v8s_640__FI')['found']), q(pair('v8s_1280__FI', 'v8s_640__FI')['false_alarm'])],
    ['640: fixed SAHI + FI vs paper SAHI + FI', ci(pair(BEST, 'v8s_640__SAHI_FI')['AP50']), q(pair(BEST, 'v8s_640__SAHI_FI')['found']), q(pair(BEST, 'v8s_640__SAHI_FI')['false_alarm'])],
    ['<b>640: fixed SAHI + FI vs FI</b>', ci(pair(BEST, 'v8s_640__FI')['AP50']), q(pair(BEST, 'v8s_640__FI')['found']), q(pair(BEST, 'v8s_640__FI')['false_alarm'])],
], [2.8, 2.9, 2.2, 2.1])}
<p>Plates: exact McNemar test on the 1,011 plates; plate-free photos: the same on the 160 photos; AP50: 2,000 bootstrap
resamples of the 1,153 photos. <b>Reading:</b> the paper's slicing costs 5.7-8.6 AP50 and triples the false alarms; the
fixed merge recovers everything and finds 5 more plates (not significant), but the false alarms stay.</p>
"""

S['c2'] = f"""
<h1>11. Why slicing hurts on our photos: what the extra boxes are</h1>
<p>We sorted every box at conf 0.25 that is not a found plate into four kinds: a <b>part of a plate</b> (the box lies at
least half inside a labelled plate but covers too little of it for IoU 0.5), <b>plate + surroundings</b> (too big),
a <b>second box</b> on a plate already found, and <b>background</b> (no labelled plate under it).</p>
{fig('fig_kinds.png', 455, 'Figure 3. Boxes at conf 0.25 that are not a found plate, by kind (640 model). The paper\'s merge adds about 270 boxes on parts of plates and about 250 on background.')}
{ul([
    f"<b>Parts of plates</b> ({V['v8s_640__FI']['extra_by_kind']['part_of_plate']} -> {V['v8s_640__SAHI_FI']['extra_by_kind']['part_of_plate']}). A 960 x 1280 photo has inner slice edges at x = 320 and 640 and at y = 480, 640 and 1120, and our plates are 90 px wide at 640 (180 px in the photo) on the median, so many plates cross an edge. Each slice then sees half a plate and boxes it with high confidence (Figure 9a). Half a plate against the whole plate has an IoU of about 0.5 or less, so the paper's NMS at IoU 0.5 keeps both. The paper names this effect for its large objects (p.4) but its objects are rarely large.",
    f"<b>Background</b> ({V['v8s_640__FI']['extra_by_kind']['background']} -> {V['v8s_640__SAHI_FI']['extra_by_kind']['background']}). In a slice everything is shown 2x larger than in the full photo, so far signs, painted letters and small plates on other vehicles become plate-sized. Section 13 shows that about half of these are real plates the labels leave out.",
    "<b>The NMS IoU rule cannot fix this</b>, because the problem is not two boxes on the same object but a box on a different-looking part of it.",
])}
<h3>Merge rules, same boxes (640 model, SAHI + FI, 640 px slices)</h3>
{table(['Merge rule', 'AP50 test', 'Found', 'Boxes that are plates', 'Plate-free photos with a box', 'AP50 val'], [
    [n, f2(V[k]['AP50']), f2(V[k]['found_%']), f1(V[k]['correct_%']), f"{V[k]['false_alarm_n']}", f2(VAL['v8s_640'][k]['AP50'])] for n, k in (
        ('NMS, IoU 0.5 (the paper, the kit)', 'v8s_640__SAHI_FI'), ('NMS, IOS 0.5', 'v8s_640__SAHI_FI__NMS_IOS'),
        ('greedy merge, IOS 0.5 (sahi default)', 'v8s_640__SAHI_FI__GNMM_IOS'),
        ('<b>drop cut-off slice boxes, then NMS IoU 0.5</b>', 'v8s_640__SAHI_FI__BORDER'),
        ('drop cut-off slice boxes, then greedy merge', 'v8s_640__SAHI_FI__BORDER_GNMM'),
        ('960 px slices (2.3 per photo), NMS IoU 0.5', 'v8s_640__SAHI_FI__s960'),
        ('960 px slices, cut-off boxes dropped', 'v8s_640__SAHI_FI__BORDER__s960'),
        ('512 px slices (7.2 per photo), NMS IoU 0.5', 'v8s_640__SAHI_FI__s512'),
        ('full image only (FI), for reference', 'v8s_640__FI'))
], [3.6, 1.0, 0.9, 1.3, 1.5, 1.0], num=(1, 2, 3, 4, 5))}
<p><b>"Drop cut-off slice boxes"</b>: a box from a slice is thrown away when it touches (within 2 px) an edge of its slice
that is not an edge of the photo; the plate it cut is whole in a neighbouring slice (25 % overlap) or in the full-image
pass. This is a standard tiling trick, but it is neither in the paper nor in the package's default merge. Two package
facts matter here: the default greedy merge is better than the paper's NMS (+{pair('v8s_640__SAHI_FI__GNMM_IOS', 'v8s_640__SAHI_FI')['AP50']['diff']:.1f}
AP50), but the package <b>quietly replaces it with NMS / IoU</b> (only a log warning) whenever the model's confidence threshold is below 0.1, which is
exactly the setting used to measure AP; and smaller slices (512) make everything worse (more cut plates).</p>
"""

S['c3'] = f"""
<h1>12. The best slicing, chosen on validation, against the full image</h1>
<p>To avoid tuning on the test photos, every slicing setting of section 11 (merge rule x slice size, and a "full image +
small slice boxes only" merge) was scored on the 896 validation photos first. The validation winner for the 640 model is
<b>SAHI + FI with cut-off slice boxes dropped</b> (val AP50 {f2(VAL['v8s_640'][BEST]['AP50'])} vs FI {f2(VAL['v8s_640']['v8s_640__FI']['AP50'])},
found {f2(VAL['v8s_640'][BEST]['found_%'])} vs {f2(VAL['v8s_640']['v8s_640__FI']['found_%'])} %). On the test photos:</p>
{table(['', 'Full image (FI)', 'Val-best slicing', 'Difference'], [
    ['AP50', f2(V['v8s_640__FI']['AP50']), f2(V[BEST]['AP50']), ci(pair(BEST, 'v8s_640__FI')['AP50']) + ', n.s.'],
    ['Plates found (of 1,011)', nf('v8s_640__FI'), nf(BEST), q(pair(BEST, 'v8s_640__FI')['found'])],
    ['Small-plate list (24 plates)', f"{small('v8s_640__FI')[0]}", f"{small(BEST)[0]}", q(pair(BEST, 'v8s_640__FI')['small_plates'])],
    ['Night photos (71 plates)', f"{V['v8s_640__FI']['night_found'][0]}", f"{V[BEST]['night_found'][0]}", f"{V[BEST]['night_found'][0] - V['v8s_640__FI']['night_found'][0]:+d}"],
    ['Boxes that are plates', f1(V['v8s_640__FI']['correct_%']) + ' %', f1(V[BEST]['correct_%']) + ' %', f"{V[BEST]['extra_boxes']} vs {V['v8s_640__FI']['extra_boxes']} extra boxes"],
    ['Plate-free photos with a box', f"{V['v8s_640__FI']['false_alarm_n']} / 160", f"{V[BEST]['false_alarm_n']} / 160", q(pair(BEST, 'v8s_640__FI')['false_alarm'])],
    ['End-to-end plate exact (204 photos)', f"{E['v8s_640__FI']['right']}", f"{E[BEST]['right']}", q(EP[f'{BEST}|v8s_640__FI'])],
    ['ms per photo (T4 / our GPU)', f"{K['v8s_640__FI']['ms_median']:.0f} / {T['v8s_640__FI']['median_ms']:.0f}", f"about {K['v8s_640__SAHI_FI']['ms_median']:.0f} / {T['v8s_640__SAHI_FI']['median_ms']:.0f}", f"{T['v8s_640__SAHI_FI']['median_ms'] / T['v8s_640__FI']['median_ms']:.1f}x"],
], [2.8, 1.6, 1.6, 3.6])}
<div class="box"><p><b>Verdict.</b> With the fix, slicing is as good as the full image in AP50 and finds a few more far plates
(not significant on our test set), for about 4x the time and three times the plate-free photos with a box. On our
current test set it is not worth it as the default. Its value is in scenes with far plates, which our labels barely
contain (next section).</p></div>
<p>For the SF model the validation winner (cut-off boxes dropped + greedy merge) does <i>not</i> hold on test: AP50
{f2(V[BEST_SF]['AP50'])} vs its own FI {f2(V['v8s_sf__FI']['AP50'])} ({ci(pair(BEST_SF, 'v8s_sf__FI')['AP50'])}). A model trained on slices
already sees far plates well in the full photo, so slicing it again adds mostly false boxes.</p>
"""

S['c4'] = f"""
<h1>13. Far plates</h1>
{fig('fig_bins.png', 455, 'Figure 4. Plates missed, by plate width in the 640 px picture the detector sees (1,011 test plates). Slicing only helps below 40 px, where we have 33 plates; the paper\'s merge loses plates in every larger group.')}
{ul([
    f"<b>Our test set has few far plates:</b> median plate width {A['plate_w640']['median']:.0f} px at 640; only {A['plate_w640']['lt_40']} plates are under 40 px and {A['plate_w640']['lt_24']} under 24 px. The small-plate list (24 plates) is found 22 times by the full image and 24 times by the paper's slicing and by the val-best setting.",
    f"<b>Plates only slicing finds</b> (paper merge vs FI): {len(A['only_slicing_finds'])}, five of them 20-48 px wide (Figure 9c). Plates only the full image finds: {len(A['only_fi_finds'])} (29-292 px): in every one the full-image box (IoU 0.58-0.92 with the label) was removed by a higher-scoring slice box on part of the plate (IoU 0.36-0.49), which then fails the IoU 0.5 test. {len(A['missed_by_all'])} plates are missed by every setting (three far plates of 32-60 px and one large plate).",
    f"<b>The test labels leave out many far plates.</b> I looked at a seeded random sample of 48 of the {FS['n_boxes']} slice-only \"background\" boxes of the val-best setting (Figure 9b, d; sheet in the source folder): <b>{N_PL} are number plates</b> (far cars, rickshaws, trucks; median 10 px wide at 640; 95 % CI {WL[0]:.0f}-{WL[1]:.0f} %), {N_UN} are unclear, and only {N_NOT} are true false boxes (shop and house-number boards, painted text on trucks, a CNG sticker). Of the 10 sampled boxes on \"plate-free\" photos, 8 are plates.",
    "<b>So two different questions get two answers.</b> \"Read the main plate of the vehicle in front of the camera\" (what our labels measure): slicing adds nothing. \"Find every plate in a street scene\" (traffic monitoring, CCTV): slicing finds plates our labels do not even count, and most of them are too small to read anyway.",
])}
"""

S['c4b'] = f"""
{fig('fig_examples.jpg', 470, 'Figure 9. (a) The paper\'s merge leaves boxes on parts of a large plate that crosses slice edges. (b) A test photo listed as plate-free: a slice finds a truck plate the labels leave out. (c) A labelled far plate that only the slices find. (d) One of the real false boxes: a house-number board.')}
"""

S['c5'] = f"""
<h1>14. Training on slices (SF) and a bigger input (1280 px)</h1>
<p>These two change the <i>model</i>, not the inference, so they cost nothing extra at test time (SF) or 1.5x (1280).
Numbers are the kit's Ultralytics val on each condition subset (re-run locally, identical), with 2,000-photo bootstrap
intervals for the difference to the 640 model:</p>
{table(['Test subset (photos / plates)', '640 model', 'SF model', 'SF - 640 (95 % CI)', '1280 model', '1280 - 640 (95 % CI)'], [
    [f"{n} ({vsp[f'v8s_sf-v8s_640|{s}']['photos']} / {vsp[f'v8s_sf-v8s_640|{s}']['plates']})", f2(VS['mAP50'][f'v8s_640|{s}']['val_mAP50']),
     f2(VS['mAP50'][f'v8s_sf|{s}']['val_mAP50']), f"{vsp[f'v8s_sf-v8s_640|{s}']['diff']:+.2f} ({vsp[f'v8s_sf-v8s_640|{s}']['ci95'][0]:+.1f} to {vsp[f'v8s_sf-v8s_640|{s}']['ci95'][1]:+.1f})",
     f2(VS['mAP50'][f'v8s_1280|{s}']['val_mAP50']), f"{vsp[f'v8s_1280-v8s_640|{s}']['diff']:+.2f} ({vsp[f'v8s_1280-v8s_640|{s}']['ci95'][0]:+.1f} to {vsp[f'v8s_1280-v8s_640|{s}']['ci95'][1]:+.1f})"]
    for n, s in (('all test', 'all'), ('clear', 'clear'), ('night', 'night'), ('blurry', 'blurry'), ('<b>small plates</b>', 'small_plates'))
], [2.4, 1.1, 1.1, 2.0, 1.1, 2.0], num=(1, 2, 3, 4, 5))}
<p class="small">mAP50 in %. The same subsets in the kit's table also give synthetic rain: 640 {f1(R['det_v8s_640']['by_condition']['rain_synth'])},
SF {f1(R['det_v8s_sf']['by_condition']['rain_synth'])}, 1280 {f1(R['det_v8s_1280']['by_condition']['rain_synth'])}; synthetic fog all about 97.7.</p>
{ul([
    f"<b>SF is the only change with a significant gain:</b> +{vsp['v8s_sf-v8s_640|small_plates']['diff']:.1f} mAP50 on the small-plate photos (the same plates are found, 22 of 24, but SF puts fewer confident false boxes on these photos: precision 0.82 vs 0.77). It pays with {vsp['v8s_sf-v8s_640|clear']['diff']:+.1f} on clear photos (interval just below zero) and {vsp['v8s_sf-v8s_640|all']['diff']:+.1f} overall (n.s.). On night photos +{vsp['v8s_sf-v8s_640|night']['diff']:.1f} (n.s.; 67 vs 64 of 71 night plates found). It also gives the fewest extra boxes of all settings (precision {f1(V['v8s_sf__FI']['correct_%'])} %).",
    f"<b>The 1280 px model is no better</b> overall ({vsp['v8s_1280-v8s_640|all']['diff']:+.2f}, n.s.) or on small plates ({vsp['v8s_1280-v8s_640|small_plates']['diff']:+.1f}, interval includes 0), finds fewer night plates (60 vs 64), and runs at {TRT['det_v8s_1280']:.0f} instead of {TRT['det_v8s_640']:.0f} FPS (TensorRT FP16, T4). Only 20 epochs, still rising (caveat in section 9).",
    "<b>The paper never ran the bigger-input baseline</b>; on our data it ties with slicing at a third of slicing's cost, and SF beats both on far plates.",
])}
{fig('fig_curves.png', 440, 'Figure 5. Validation mAP50 during training, by training images seen. SF needs more images to reach the same validation level (its slices are a different distribution from the full validation photos); the 1280 model stops at 171 k images.')}
"""

S['c6'] = f"""
<h1>15. Their data and ours: why the same idea flips</h1>
{fig('fig_sizes.png', 455, 'Figure 6. Object width as a share of the photo width. VisDrone objects (orange) are mostly 0.3-5 % of the photo; our plates (blue) are 10-30 %. The paper\'s own yardstick for "small" is 1 % (p.3).')}
{table(['', 'VisDrone test-dev (the paper)', 'Our test split'], [
    ['Objects / photos', f"{VD['visdrone']['objects']:,} / {VD['visdrone']['photos']:,} ({VD['objects_per_photo']['visdrone']} per photo)", f"{VD['ours']['objects']:,} / 1,153 (0.9 per photo)"],
    ['Median width, share of photo width', f"{VD['visdrone']['width_share_%']['50']:.1f} %", f"{VD['ours']['width_share_%']['50']:.1f} %"],
    ['Narrower than 1 % of the photo', f"{VD['visdrone']['width_lt_1pct_%']:.0f} %", f"{VD['ours']['width_lt_1pct_%']:.0f} %"],
    ['COCO small / medium / large', f"{VD['visdrone']['coco_small_%']:.0f} / {VD['visdrone']['coco_medium_%']:.0f} / {VD['visdrone']['coco_large_%']:.0f} %", f"{VD['ours']['coco_small_%']:.1f} / {VD['ours']['coco_medium_%']:.0f} / {VD['ours']['coco_large_%']:.0f} %"],
    ['Recall of the full-image detector', f"TOOD {W['tood_fi']['recall_%']:.1f} % (score 0.3)", f"{f1(V['v8s_640__FI']['found_%'])} % (conf 0.25)"],
], [3.0, 3.5, 3.5])}
{fig('fig_wp.png', 460, 'Figure 7. Precision and recall before and after slicing. Left: the authors\' own VisDrone predictions (Table 1 files) at score 0.3: slicing costs 13-14 points of precision and buys about 12 points of recall. Right: our plates: a bigger precision cost (30 points with the paper merge), and almost no recall left to buy.')}
<p><b>The mechanism is the same on both datasets</b>: slicing adds false boxes. The paper never shows this, because
AP50 over many tiny objects rewards the recall gain more than it punishes the false boxes. Our detector already finds
98 % of the plates, so the trade can only lose. Slicing is a tool for recall-starved, tiny-object problems.</p>
"""

S['c7'] = f"""
<h1>16. Reading the plates end to end</h1>
<p>On the 204 end-to-end photos (BLPD external + NightLPBD, never used for training) the most confident box is cut out
and read by our CRNN (73.4 % plate exact on the OCR test crops). Paired with the full-image 640 setting:</p>
{table(['Setting', 'Plates read exactly (of 204)', 'vs 640 FI: only A vs only B'], [
    [NAME['v8s_640__FI'], f"{E['v8s_640__FI']['right']}", '–'],
    [NAME['v8s_640__SAHI'], f"{E['v8s_640__SAHI']['right']}", q(EP['v8s_640__SAHI|v8s_640__FI'])],
    [NAME['v8s_640__SAHI_FI'], f"{E['v8s_640__SAHI_FI']['right']}", q(EP['v8s_640__SAHI_FI|v8s_640__FI'])],
    ['640 model, sahi default greedy merge', f"{E['v8s_640__SAHI_FI__GNMM_IOS']['right']}", q(EP['v8s_640__SAHI_FI__GNMM_IOS|v8s_640__FI'])],
    ['640 model, cut-off boxes dropped (val-best)', f"{E[BEST]['right']}", q(EP[f'{BEST}|v8s_640__FI'])],
    [NAME['v8s_sf__FI'], f"{E['v8s_sf__FI']['right']}", q(EP['v8s_sf__FI|v8s_640__FI'])],
    [NAME['v8s_1280__FI'], f"{E['v8s_1280__FI']['right']}", q(EP['v8s_1280__FI|v8s_640__FI'])],
], [4.2, 2.2, 3.6])}
<p>The paper's slicing loses plates here too: in {A['e2e_top_box']['v8s_640__SAHI_FI']['narrower_than_FI_box']} photos its most
confident box is a part of the plate (narrower than 80 % of the full-image box), and the CRNN reads half a plate. With
cut-off boxes dropped this never happens (0 photos). Everything else is within the noise of {FR['flip_with_1px']} fragile
readings.</p>

<h1>17. Speed</h1>
{fig('fig_speed.png', 440, 'Figure 8. Median ms per test photo through the sahi package (decode + network + merge), Kaggle T4 and our RTX 5060 Ti.')}
{table(['', '640 FI', '640 SAHI', '640 SAHI + FI', '640 SAHI + FI, slices batched (8)', '1280 FI'], [
    ['T4 (the run), ms / photo', f1(K['v8s_640__FI']['ms_median']), f1(K['v8s_640__SAHI']['ms_median']), f1(K['v8s_640__SAHI_FI']['ms_median']), '–', f1(K['v8s_1280__FI']['ms_median'])],
    ['RTX 5060 Ti, ms / photo', f1(T['v8s_640__FI']['median_ms']), f1(T['v8s_640__SAHI']['median_ms']), f1(T['v8s_640__SAHI_FI']['median_ms']), f1(T['v8s_640__SAHI_FI_batch8']['median_ms']), f1(T['v8s_1280__FI']['median_ms'])],
    ['Network alone, TensorRT FP16 on T4', f"{TRT['det_v8s_640']:.0f} FPS", '≈ x5 passes', '≈ x6 passes', '–', f"{TRT['det_v8s_1280']:.0f} FPS"],
], [3.0, 1.1, 1.1, 1.3, 2.0, 1.1], num=(1, 2, 3, 4, 5))}
<p>The paper says only that time "increases linearly" with the slices (p.4). Measured: 3.5x on the T4 and 3.9x on our GPU
for about 6 passes per photo (5 slices + the full photo on average); batching the slices saves only 11 %. With SAHI + FI
the detector alone would use about 120 ms of the 100 ms per frame that our 10 FPS goal allows on a T4.</p>
"""

S['d1'] = f"""
<p class="part">PART D · HOW TO USE THIS IN OUR RESEARCH</p>
<h1>18. Where SAHI fits in the 498R pipeline</h1>
<pre>photo / frame ─► YOLO detector, 640 px, full image ─► plate crops ─► reader ─► BRTA check
                     │
                     └─ optional far-plate pass (CCTV, NSU street set only):
                        640 px slices, 25 % overlap, cut-off slice boxes
                        dropped, NMS IoU 0.5, merged with the full-image boxes</pre>
{ul([
    '<b>Keep the full-image 640 px detector as the default.</b> On our test set nothing beats it significantly, and it is the only setting that fits the 10 FPS budget with room for the reader.',
    '<b>Never use the paper\'s merge on plates.</b> If slicing is used, drop the slice boxes that touch an inner slice edge (our val-chosen fix); the paper\'s NMS at IoU 0.5 leaves boxes on parts of plates, and the package\'s better default merge is switched off at low confidence thresholds.',
    '<b>Use SF when we re-train the detector</b> for far plates (it costs training time, not test time), and check clear-photo mAP when we do.',
    '<b>Do not train at 1280 px</b> for this: 1.5x slower and no gain in our run.',
    f"<b>Weights:</b> {M('best_det_v8s_sf.pt')} (SF) and {M('best_det_v8s_1280.pt')} in the GitHub upload; the 640 model is the HVI-CIDNet study's file. The slice-edge merge ships as {M('sahi_border_predict.py')} next to the weights (same boxes as the analysis on all 1,153 test photos).",
])}

<h1>19. What we gain for the research</h1>
{table(['Proposal item', 'What this study adds'], [
    ['O2: the plate detector with the best accuracy-latency trade-off', f"a measured answer for the three standard ways to see small objects: slicing (3.5-4x time, no gain), a bigger input (1.5x, no gain), slice training (free at test time, +{vsp['v8s_sf-v8s_640|small_plates']['diff']:.0f} mAP50 on small-plate photos). The 640 px full-image detector stays"],
    ['RQ3 / O4: 10 FPS on a laptop', f"SAHI + FI needs about {K['v8s_640__SAHI_FI']['ms_median']:.0f} ms per photo for the detector alone on a T4, so it can only be an occasional second pass"],
    ['RQ4 / G4: robustness per condition', 'plate width in the network input as a new evaluation axis (Figure 4), next to night, blur, rain and fog; far plates under 40 px are where every detector fails'],
    ['G3: data and labels', f"a concrete label-policy problem: about half of the slice-only false boxes are real far plates ({N_PL} of 48), and 8 of 10 sampled boxes on \"plate-free\" photos are plates. Metrics for far plates need a written rule (label every visible plate, or ignore plates under a size)"],
    ['G1: accuracy vs speed', 'ms per photo for every setting on a T4 and on our GPU, plus TensorRT FPS'],
    ['G6: reproducibility', 'the paper\'s Table 1 reproduced 80 / 80 from its released files; two abstract numbers that do not match it; our box caches, merge code and per-plate table are released'],
    ['Literature review', 'a widely used small-object method tested on road photos for the first time in our project, with the reason it fails there (object size, cut plates) and a fix'],
], [3.4, 6.6])}

<h1>20. Next steps (in order)</h1>
{ul([
    '<b>Agree a label rule for far plates</b> with the team (label every visible plate, or mark plates under about 24 px at 640 as "ignore" so they count neither way). Without it no far-plate method can be scored fairly.',
    '<b>Test the far-plate pass where far plates exist:</b> the NSU street set and CCTV / video frames (RQ3, P3-P5). There, the fixed slicing and SF should show their value; on our current test set they cannot.',
    '<b>Kit updated (29 Oct, done):</b> the SAHI job now also scores SAHI + FI with cut-off slice boxes dropped (on the run\'s 640 model it gives exactly the numbers of this report: AP50 97.51, 997 plates, 148 e2e), writes per_plate.csv, e2e_reads.csv and boxes.json for paired tests, and the two loose table rows are split and reworded. A re-run is not needed for this report; use the new kit for the next run.',
    '<b>SF, done properly:</b> 3 seeds of SF vs the 640 model (the small-plate gain rests on 24 plates), and SF with slices only from photos that have small plates, to keep the clear-photo mAP.',
    '<b>Fair bigger-input baseline</b> (optional): 1280 px with the same number of images seen (about 60 epochs, about 7 h), or a short 1280 fine-tune from the 640 weights.',
    '<b>Small-object heads instead of slicing</b> (optional): the YOLO26 study\'s P2-head model found 98.42 % of the same test plates in one pass; combine a P2 head with SF.',
])}
"""

S['e1'] = f"""
<p class="part">PART E · FOR THE FACULTY</p>
<h1>21. What to show (5 slides or 5 printed figures)</h1>
{table(['#', 'Show', 'One-line message'], [
    ['1', 'Figure 1 (SAHI and SF)', 'A widely used ICIP 2022 method: run the detector on overlapping tiles (SAHI) and train on tiles (SF), for small far objects; we reproduced the paper\'s Table 1 exactly from the authors\' files'],
    ['2', 'Figure 2 + the paired-test table of section 10', f"On our plates the paper's slicing is worse than the normal detector (AP50 {f1(V['v8s_640__SAHI_FI']['AP50'])} vs {f1(V['v8s_640__FI']['AP50'])}, 3x the false alarms, 3.5x the time)"],
    ['3', 'Figure 6 + Figure 3', 'Why: drone objects are 1 % of the photo, our plates 18 %; slice edges cut plates into pieces the paper\'s merge keeps'],
    ['4', 'Figure 9 + the table of section 12', 'Our fix (drop cut-off slice boxes) brings slicing level with the normal detector; half of the remaining "false" boxes are real far plates our labels leave out'],
    ['5', 'The table of section 14', f"Training on tiles (SF) is the useful part for us: +{vsp['v8s_sf-v8s_640|small_plates']['diff']:.0f} mAP50 on far-plate photos at no test-time cost; a 1280 px detector gives nothing"],
], [0.4, 3.0, 6.6])}

<h1>22. What to say (about two minutes)</h1>
<div class="box">
<p>"This week I studied SAHI from ICIP 2022. Detectors shrink a big photo to 640 pixels, so far objects nearly vanish.
SAHI cuts the photo into overlapping tiles, runs the detector on every tile and merges the boxes; SF also trains the
detector on tiles. On drone photos it raises AP50 from 29 to 43. I first checked the paper: re-scoring the authors'
released prediction files gives all 80 numbers of their Table 1 exactly; only two numbers in the abstract do not match."</p>
<p>"Then I ran it on our dataset with YOLOv8s: the normal detector, SAHI on top of it, a detector trained with SF, and a
detector at 1280 pixels, the obvious alternative the paper never tests. On our plates the paper's SAHI is clearly worse:
AP50 drops from 97.3 to 91.6, it triples the false alarms and it is 3.5 times slower. The reason is size: drone objects
are about 1 % of the photo, our plates about 18 %, so tile edges cut plates in half and the paper's merge rule keeps the
halves. Dropping boxes that touch a tile edge, chosen on the validation set, brings it back level with the normal
detector."</p>
<p>"Two findings matter for our project. First, half of the boxes we count as false are real far plates that our labels
leave out, so we need a team rule for labelling far plates. Second, training on tiles helps far plates by 12 mAP50 points
at no extra test time, while the 1280-pixel model gives nothing. So we keep the 640-pixel detector, consider SF in the
next training, and test slicing only on the street and CCTV data, where far plates really occur."</p>
</div>
"""

S['e2'] = f"""
<h1>23. Questions they may ask, and short answers</h1>
{table(['Question', 'Answer'], [
    ['So SAHI does not work?', 'It works for what it was made for: many tiny objects in big aerial photos (we reproduced that). Our test photos are mostly close-ups with one large plate; there it cannot add recall and its false boxes cost more than it gains.'],
    ['Is the fix chosen on the test set?', 'No. All merge rules and slice sizes were scored on the 896 validation photos; the winner is reported on test. For the SF model the validation winner did not hold on test, and we report that too.'],
    ['Is your fix new?', 'No: dropping boxes cut by a tile edge is a standard tiling trick. It is just not in the paper or the package\'s default merge, and on plates it decides whether slicing helps or hurts.'],
    ['Why not just use 1280 px?', f"We tested it: no better overall or on small plates and {TRT['det_v8s_640'] / TRT['det_v8s_1280']:.1f}x slower; it had only 20 epochs, so a longer run may close some of the gap."],
    ['Is the SF gain real?', f"On the 22 small-plate photos the bootstrap interval is +{vsp['v8s_sf-v8s_640|small_plates']['ci95'][0]:.1f} to +{vsp['v8s_sf-v8s_640|small_plates']['ci95'][1]:.1f} mAP50, so yes for these photos, but it is one seed and 24 plates, and clear photos lose {abs(vsp['v8s_sf-v8s_640|clear']['diff']):.1f}. We plan 3 seeds."],
    ['How do you know half of the false boxes are plates?', f"I looked at a seeded random sample of 48 of them: {N_PL} are plates, {N_UN} unclear, {N_NOT} not plates (95 % interval {WL[0]:.0f}-{WL[1]:.0f} %). The sheet is in the source folder."],
    ['Why is AP50 small so low (16-27)?', 'Only 5 of 1,011 plates are smaller than 32 x 32 px in the photo, so that number is noise; we use plate width at 640 px and the small-plate list instead.'],
    ['Did you reproduce the paper?', 'Its Table 1 yes, exactly, from the authors\' own prediction files on the official VisDrone labels. We did not retrain their mmdetection models; xView needs a registration.'],
], [3.0, 7.0])}
<div class="warn"><p><b>Limits to state yourself:</b> one seed per model; our test set has few far plates (33 under 40 px), so
far-plate differences rest on small numbers; the 1280 model was cut at 20 epochs; the labels leave out far plates, which
biases every precision number against slicing; the label check is my own eye on 48 boxes.</p></div>
"""

S['f1'] = f"""
<h1>Appendix: files, settings and sources</h1>
{table(['What', 'Where'], [
    ['Notes (passes 1-3, results block)', M('docs/Rokib Hasna/new5/36_SAHI .../36_SAHI ... .md')],
    ['Kaggle run (as run)', M('After Train/paper-36.ipynb') + ', ' + M('paper-36.log') + ', ' + M('results.zip') + ' (whole /kaggle/working, 756 MiB)'],
    ['Kit (updated 29 Oct)', M('kaggle/paper36_kaggle.ipynb') + ', ' + M('_kit/extras/p36_sahi.py') + ' (test of the new setting on the run\'s 640 model: ' + M('kit_test_sahi_results.json') + ', ' + M('kit_test_per_plate.csv') + '); original check ' + M('paper36_original_check.ipynb') + ', ' + M('c36_sahi.py')],
    ['Local re-check', M('report_src/local_check.py') + ' (predict, predict_val, verify, read, fragile, valstats, timing) + ' + M('sahi_common.py') + ' → ' + M('_cache/') + ', ' + M('verify.json') + ', ' + M('e2e_reads.json') + ', ' + M('det_val_stats.jsonl') + ', ' + M('timing.json')],
    ['Their data vs ours', M('report_src/visdrone_extra.py') + ' → ' + M('visdrone_extra.json')],
    ['Analysis, label check', M('analysis.py') + ' → ' + M('analysis.json') + ', ' + M('per_plate.csv') + '; ' + M('fp_sample.py') + ' → ' + M('_fp_sheet.jpg') + ', ' + M('fp_labels.json')],
    ['Charts / this PDF', M('charts.py') + ', ' + M('build_report.py') + ' (+ ' + M('part_a.py') + ', ' + M('report_head.py') + ')'],
    ['GitHub upload folder', M('After Train/github_upload/') + ' with ' + M('UPLOAD_STEPS.md')],
], [3.0, 7.0])}
{table(['Paper named in this report', 'Full title (team folder)'], [
    ['YOLO26 study', 'Sapkota et al., YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection, arXiv 2509.25164'],
    ['HVI-CIDNet study', 'Yan et al., HVI: A New Color Space for Low-light Image Enhancement, CVPR 2025'],
    ['Model Fusion study', 'Laroca et al., Leveraging Model Fusion for Improved License Plate Recognition, CIARP 2023'],
    ['TOOD / FCOS / VFNet', 'Feng et al., ICCV 2021; Tian et al., ICCV 2019; Zhang et al., CVPR 2021 (the paper\'s detectors)'],
], [3.0, 7.0])}
<pre>@inproceedings{{akyon2022sahi,
  title     = {{Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection}},
  author    = {{Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin}},
  booktitle = {{2022 IEEE International Conference on Image Processing (ICIP)}},
  pages     = {{966--970}},
  year      = {{2022}},
  doi       = {{10.1109/ICIP46576.2022.9897990}}
}}</pre>
"""

FLOWS = [['cover'], ['a1'], ['a2'], ['a3'], ['b1', 'b2'], ['c1', 'c2', 'c3'], ['c4'], ['c4b', 'c5'], ['c6'], ['c7'],
         ['d1'], ['e1', 'e2'], ['f1']]

if __name__ == '__main__':
    n = write_pdf(S, FLOWS, OUT, 'SAHI study report · dataset v1.1 run · 498R BanglaLPR',
                  {'title': 'SAHI study report (dataset v1.1 run)', 'author': 'MD. Rokib Hasan Oli',
                   'subject': 'Paper study: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (ICIP 2022)',
                   'keywords': 'SAHI, slicing, small object detection, YOLOv8, Bangla license plates', 'creator': 'build_report.py'})
    print('pages', n, OUT)
