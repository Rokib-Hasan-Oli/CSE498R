"""Paper #6 study report (dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Ahamed & Ismail, Authorea preprint 2025, doi 10.22541/au.173748276.69471512/v1; page numbers = PDF
pages) and the paper notes; the v1.1 Kaggle run (`After Train/results.zip`, `paper-6.log`, 24 Oct 2026) via
analysis.json; the local checks local_check.json (detectors per source, reader swap on the 204 photos) and
easyocr_check.json (EasyOCR re-run + cheap fixes); the 498R proposal (RQ1-RQ4, O1-O5, targets).
Run order: local_check.py -> easyocr_check.py -> analysis.py -> charts.py -> build_report.py.
"""
import io
import json
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'Paper06_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
LC = json.load(open(os.path.join(HERE, 'local_check.json'), encoding='utf-8'))
FONT_DIR = os.path.join(HERE, 'fonts')
D, EO, E, EL, SM = A['det'], A['easyocr'], A['e2e'], A['easyocr_local'], A['same_651_crops']
n8, s8 = D['det_v8n'], D['det_v8s']
K = EO['kit']
RAW, PREP = K['rows_decoder'], K['prep']
LD, X = LC['det'], LC['e2e_cross']
P03 = A['p03_v8n_640']
V = EL['variants']
DP = A['det_paired']
EP = E['paired']


def bangla_font():
    """Nirmala.ttc cannot be loaded by MuPDF's CSS, but its first face saved as a .ttf works (shaping included)."""
    f = os.path.join(FONT_DIR, 'nirmala.ttf')
    if not os.path.exists(f):
        from fontTools.ttLib import TTCollection
        os.makedirs(FONT_DIR, exist_ok=True)
        TTCollection(r'C:\Windows\Fonts\Nirmala.ttc').fonts[0].save(f)
    return f


f1 = lambda x: f'{x:.1f}'
f2 = lambda x: f'{x:.2f}'
pv = lambda p: 'p &lt; 0.0001' if p < 0.0001 else f'p = {p:.4f}'.rstrip('0').rstrip('.') if p < 0.01 else f'p = {p:.2f}'
x8n = X['det_v8n+lpnr_v12n/p06_protocol']
x8s = X['det_v8s+lpnr_v12n/p06_protocol']
c8s = X['det_v8s+lpnr_v12n/p02_cascade']
c8n = X['det_v8n+lpnr_v12n/p02_cascade']
c02 = X['p02_lpd_v12s+lpnr_v12n/p02_cascade']
lp2 = SM['#2 lpnr_v12n (char-YOLO12n)']
cy24 = SM['#24 chars_v8n (char-YOLOv8n)']
fix = max(('allow', 'allow_x2', 'x2'), key=lambda k: V[k]['plate_exact_%'])
FIXN = {'allow': 'allowlist', 'allow_x2': 'allowlist + 2x upscale', 'x2': '2x upscale'}[fix]
pvr = EL['paired_vs_local_raw']
e8n, e8s = E['kit']['e2e_v8n'], E['kit']['e2e_v8s']
ls8n = E['loss_split']['e2e_v8n']
sp_n, sp_s = n8['speed'], s8['speed']
BC = K['by_condition']

# the text below quotes these numbers; stop if the data ever disagrees with it
assert A['run']['dataset_version'] == '1.1' and len(A['run']['jobs']) == 7 and all(j['code'] == 0 for j in A['run']['jobs'])
assert n8['epochs_run'] == s8['epochs_run'] == 120 and n8['nan_values'] == s8['nan_values'] == 0
assert n8['best_epoch'] == 90 and s8['best_epoch'] == 79
assert n8['test']['mAP50'] == 97.39 and s8['test']['mAP50'] == 97.61 and n8['test']['mAP50-95'] == 44.97
assert LD['det_v8n']['test']['mAP50'] == 97.39 and LD['det_v8n']['test']['mAP50-95'] == 45.0      # local = Kaggle
assert LD['det_v8s']['test']['mAP50-95'] == 45.88 and LD['p02_lpd_v12s']['test']['mAP50-95'] == 46.18
assert LD['det_v8n']['plates_found_%'] == n8['plates']['found_%'] == 96.44 and LD['det_v8s']['plates_found_%'] == 97.13
assert LD['det_v8n']['test_without_ridoykj']['mAP50-95'] == 64.27 and LD['det_v8n']['by_source']['ridoykj']['mAP50-95'] == 22.12
assert RAW['plate_exact_%'] == 13.52 and RAW['char_acc_%'] == 60.45 and PREP['plate_exact_%'] == 4.92
assert V['raw']['char_acc_%'] == RAW['char_acc_%'] and abs(V['raw']['plate_exact_%'] - RAW['plate_exact_%']) < 0.5
assert e8n['full_plate_exact_%'] == 11.76 and e8s['full_plate_exact_%'] == 10.29
assert x8n['plate_detected_%'] == e8n['plate_detected_%'] and x8s['plate_detected_%'] == e8s['plate_detected_%']
assert x8n['plate_exact_%'] == 84.8 and x8s['plate_exact_%'] == 88.24 and c02['plate_exact_%'] == 86.76
assert EP['local lpd_v12s cascade = Kaggle cascade (same plates right)']['identical']
assert lp2['a_%'] == 87.71 and lp2['only_a'] == 487 and lp2['only_b'] == 4
assert DP['#2 lpd_v12s vs v8n']['only_a'] == 24 and DP['#2 lpd_v12s vs v8n']['only_b'] == 5
assert DP['v8s vs v8n']['only_a'] == 17 and DP['v8s vs v8n']['only_b'] == 10 and DP['v8s vs v8n']['p'] > 0.2
assert P03['test']['mAP50'] == 98.73 and P03['found_%'] == 98.12 and P03['labels'] == 'human+auto'
assert EO['prep_vs_raw']['only_a'] == 13 and EO['prep_vs_raw']['only_b'] == 69
assert V['allow']['plate_exact_%'] == 13.98 and pvr['allow']['p'] > 0.5 and V['x2']['plate_exact_%'] < V['raw']['plate_exact_%']
assert E['loss_split']['e2e_v8n']['no_box'] == 11 and E['loss_split']['e2e_v8s']['no_box'] == 8

CSS = """
@font-face {font-family: B; src: url(arial.ttf);}
@font-face {font-family: B; src: url(arialbd.ttf); font-weight: bold;}
@font-face {font-family: B; src: url(ariali.ttf); font-style: italic;}
@font-face {font-family: B; src: url(arialbi.ttf); font-weight: bold; font-style: italic;}
@font-face {font-family: H; src: url(arialbd.ttf); font-weight: bold;}
@font-face {font-family: H; src: url(arial.ttf);}
@font-face {font-family: M; src: url(consola.ttf);}
@font-face {font-family: BN; src: url(nirmala.ttf);}
* {font-family: B;}
body {font-size: 9.4pt; line-height: 1.32; color: #1d1d1b;}
p {margin: 0 0 5pt 0; text-align: left;}
h1 {font-family: H; font-weight: bold; font-size: 15pt; color: #104281; margin: 4pt 0 6pt 0;}
h2 {font-family: H; font-weight: bold; font-size: 11.5pt; color: #1c5cab; margin: 10pt 0 4pt 0;}
h3 {font-family: B; font-weight: bold; font-size: 9.8pt; color: #1d1d1b; margin: 6pt 0 2pt 0;}
.part {font-family: H; font-size: 9pt; color: #6b6a66; letter-spacing: 1pt; margin: 0 0 2pt 0;}
ul, ol {margin: 0 0 5pt 0; padding-left: 14pt;}
li {margin: 0 0 2.5pt 0;}
table {border-collapse: collapse; width: 100%; margin: 3pt 0 8pt 0;}
th {font-weight: bold; color: #104281; text-align: left; font-size: 8.2pt; padding: 3pt 4pt;
    border: 0.6pt solid #c9d3e0; border-bottom: 1.4pt solid #5598e7;}
td {font-size: 8.2pt; padding: 2.6pt 4pt; border: 0.6pt solid #c9d3e0; vertical-align: top; line-height: 1.2;}
td.n, th.n {text-align: right;}
.box {border: 0.8pt solid #9ec5f4; border-left: 3.5pt solid #2a78d6; padding: 6pt 8pt; margin: 4pt 0 8pt 0;}
.warn {border: 0.8pt solid #f3c27a; border-left: 3.5pt solid #eb6834; padding: 6pt 8pt; margin: 4pt 0 8pt 0;}
.good {border: 0.8pt solid #9fd3a4; border-left: 3.5pt solid #008300; padding: 6pt 8pt; margin: 4pt 0 8pt 0;}
.box p, .warn p, .good p {margin: 0 0 3pt 0;}
.cap {font-size: 8.6pt; color: #52514e; margin: 2pt 0 9pt 0;}
.fig {text-align: center; margin: 4pt 0 0 0;}
.mono {font-family: M; font-size: 8.6pt;}
.bn {font-family: BN; font-size: 8.6pt;}
pre {font-family: M; font-size: 8.3pt; border: 0.6pt solid #c9c8c2; padding: 5pt 6pt;
     margin: 2pt 0 7pt 0; line-height: 1.25;}
.small {font-size: 9pt; color: #52514e;}
.q {font-weight: bold; margin: 6pt 0 1pt 0;}
"""


def table(header, rows, widths=None, num=()):
    w = widths or [1] * len(header)
    tot = sum(w)
    h = ''.join(f'<th width="{100 * x / tot:.1f}%"{" class=n" if i in num else ""}>{c}</th>'
                for i, (c, x) in enumerate(zip(header, w)))
    b = ''.join('<tr>' + ''.join(f'<td{" class=n" if i in num else ""}>{c}</td>' for i, c in enumerate(r)) + '</tr>'
                for r in rows)
    return f'<table><tr>{h}</tr>{b}</table>'


def fig(name, width, caption):
    return f'<div class="fig"><img src="{name}" width="{width}"></div><p class="cap">{caption}</p>'


def ul(items, tag='ul'):
    return f'<{tag}>' + ''.join(f'<li>{i}</li>' for i in items) + f'</{tag}>'


M = lambda s: f'<span class="mono">{s}</span>'
BN = lambda s: f'<span class="bn">{s}</span>'
ADD = M('python add_results.py paper06_results.zip --member "Rokib Hasan Oli"')
S = {}  # flow name -> HTML; each flow starts on a new page
jobs = {j['name']: j['min'] for j in A['run']['jobs']}
fps_n, fps_s = sp_n['fp32']['fps'], sp_s['fp32']['fps']
trt_n, trt_s = sp_n['tensorrt_fp16']['fps'], sp_s['tensorrt_fp16']['fps']

# ======================================================================================== cover + one-page summary
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER #6 STUDY REPORT · DATASET v1.1 RUN</p>
<h1 style="font-size:19pt; margin-bottom:3pt">YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles</h1>
<p style="font-family:H; font-size:12pt; color:#1c5cab; margin-bottom:8pt">The paper, our re-run on dataset v1.1, why the
detector is fine but EasyOCR is not, what the numbers mean for our system, and what to show and say to the faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 24 October 2026<br>
Paper: I. Ahamed and G. M. Ismail, Authorea preprint (not peer-reviewed), posted 21 Jan 2025,
doi 10.22541/au.173748276.69471512/v1; 22 pages, text on p.1-17. Page numbers are PDF pages.<br>
Code and data: none published. Our re-implementation: Ultralytics YOLOv8 8.4.170 + EasyOCR 1.7.2 ({M('paper06_kaggle.ipynb')}).<br>
Our run: Tesla T4 x2 on Kaggle, 24 Oct 2026, dataset <b>v1.1</b> (confirmed in {M('environment.json')}),
{A['run']['total_h']:.1f} h; full output in {M('After Train/results.zip')}. Extra checks run locally the same day (sections 10-13).</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> A COCO-pretrained YOLOv8 is fine-tuned to find Bangladeshi number plates in 2,600
own photos, and the stock EasyOCR Bangla model reads the cropped plate. The authors report precision 93.5 %, recall
92.3 %, mAP 94.0 % and 78 FPS for the detector, and 92.7 % "character recognition accuracy" for EasyOCR (p.9-12). No
metric is defined, the photos were augmented before the split, and no code, data or settings are given.</p>
</div>
<div class="good">
<p><b>Is the v1.1 training OK? Yes.</b> All 7 jobs ended with {M('code=0')}, dataset version 1.1 is confirmed, both
detectors ran all 120 epochs with no NaN (best epochs {n8['best_epoch']} and {s8['best_epoch']}), and the run took
{A['run']['total_h']:.1f} h of the 12 h limit. A local re-check with the run's {M('best.pt')} files gives the Kaggle test
numbers to the last digit (mAP50 {f2(LD['det_v8n']['test']['mAP50'])} and {f2(LD['det_v8s']['test']['mAP50'])}), and a local
EasyOCR re-run gives the same character accuracy ({f2(V['raw']['char_acc_%'])} %). The one number that looks wrong,
mAP50-95 = {f1(n8['test']['mAP50-95'])} (paper about 59), is explained by one test source with tighter label boxes, not by
the training (section 10).</p>
</div>
<div class="box">
<p><b>Five findings.</b> (1) <b>The detector matches the paper and our targets:</b> YOLOv8n mAP50 {f1(n8['test']['mAP50'])} %,
YOLOv8s {f1(s8['test']['mAP50'])} % (paper 94.0 %, proposal target 95 %); {f1(n8['plates']['found_%'])} / {f1(s8['plates']['found_%'])} %
of the {n8['plates']['plates']:,} test plates found. (2) <b>mAP50-95 is a labelling effect:</b> the ridoykj source (274
test photos) is boxed tightly; it scores {f1(LD['det_v8n']['by_source']['ridoykj']['mAP50-95'])} there and
{f1(LD['det_v8n']['test_without_ridoykj']['mAP50-95'])} without it. (3) <b>EasyOCR is the weak link:</b> {f1(RAW['char_acc_%'])} %
character accuracy (paper 92.7 %) and only <b>{f1(RAW['plate_exact_%'])} %</b> of 651 plates read exactly; {EO['pred_with_latin_digits_%']:.0f} %
of its readings contain English digits, and the paper-style preprocessing makes it worse ({f1(PREP['plate_exact_%'])} %).
(4) <b>Swap only the reader and the system works:</b> on the same 204 whole photos with the same YOLOv8 crops, a
character detector (paper #2's char-YOLO12n) reads {f1(x8n['plate_exact_%'])} % (v8n) and {f1(x8s['plate_exact_%'])} % (v8s)
exactly instead of {f1(e8n['full_plate_exact_%'])} / {f1(e8s['full_plate_exact_%'])} % with EasyOCR. (5) <b>Speed is no problem:</b>
YOLOv8n runs at {fps_n:.0f} FPS in PyTorch FP32 and {trt_n:.0f} FPS with TensorRT FP16 on a T4 (paper 78 FPS); EasyOCR
costs {f1(K['ms_per_plate'])} ms per plate, 17x the detector.</p>
<p><b>What to tell the faculty in one sentence.</b> "We rebuilt this paper on our duplicate-free benchmark: its YOLOv8
plate detector is as good as claimed (97 % mAP50), but stock EasyOCR reads only 14 % of real Bangladeshi plates
correctly, so in our system YOLOv8 stays as the detector and EasyOCR stays only as the baseline that our character-level
reader must beat, which it does with 88 % on the same plates."</p>
</div>
<h2>How this report is organised</h2>
{table(['Part', 'Sections', 'Read it when'], [
    ['A. Understand the paper', '1-6: problem, terms, data, method, results, weaknesses', 'before you explain the paper'],
    ['B. Our v1.1 experiment', '7-15: what we ran, is it OK, results vs paper, detector, EasyOCR, fixes, whole system, speed, findings', 'before you quote any number'],
    ['C. Use it', '16-22: research use, benefits, faculty slides, script, Q&amp;A, next steps, GitHub/Drive', 'before the faculty meeting'],
], [2.2, 4.8, 3.0])}
"""

# ======================================================================================== part A (the paper)
S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What problem does the paper solve?</h1>
<p>Bangladesh has more than 4.8 million registered vehicles and low-resolution CCTV, and its plates have two lines with
Bangla words, a series letter and six Bangla digits (p.1). The paper builds an automatic plate reader for these plates:
<b>find</b> the plate in a street photo, then <b>read</b> its Bangla text. It claims to work under low resolution, rain,
dust, night and partly hidden plates (p.3, p.12-13).</p>
<p>The method is the simplest full pipeline possible with off-the-shelf tools: a COCO-pretrained <b>YOLOv8</b> fine-tuned
as a one-class plate detector, then the stock <b>EasyOCR</b> Bangla model on the plate crop (p.5-8). Nothing new is
trained for reading. That simplicity is exactly why it matters for us: it is the baseline our proposal promised
("EasyOCR after our detector", proposal section on baselines), and every better reader must beat it.</p>
{fig('fig_pipeline.png', 470, 'Figure 1. The paper\'s system (p.5-8) with what our v1.1 run measured at each step (orange: YOLOv8n / YOLOv8s). Sections 9-13 explain the numbers.')}

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['YOLOv8 (n / s)', 'a one-stage object detector by Ultralytics (PyTorch, anchor-free). "n" (nano, 3.0 M parameters) and "s" (small, 11.1 M) are two sizes; the paper does not say which size it used'],
    ['COCO-pretrained, fine-tuned', 'the model starts from weights learned on the 80-class COCO photos and is then trained on plate boxes only (one class)'],
    ['IoU', 'intersection over union: how much a predicted box overlaps the label box (1 = perfect). IoU 0.5 = half overlap'],
    ['Precision / recall', 'of the boxes the model draws, how many are real plates / of the real plates, how many get a box'],
    ['mAP50 and mAP50-95', 'average precision with a box counted right at IoU 0.5 (mAP50), or averaged over IoU 0.50 to 0.95 (mAP50-95, rewards very tight boxes). The paper\'s "mAP 94.0" is mAP50 (its Fig. 6, p.11)'],
    ['Plates found %', 'our plain number: share of human plate boxes with a model box at confidence &gt;= 0.25 and IoU &gt;= 0.5'],
    ['EasyOCR', 'a free OCR library (JaidedAI): CRAFT finds text lines, a CRNN reads each line. The "bn" model was trained on general Bangla text, not on plates'],
    ['Character accuracy / CER', 'CER = edits needed to turn the reading into the truth, divided by the truth length; char accuracy = 100 - CER. The paper never defines its "character recognition accuracy"'],
    ['Plate exact', 'the whole plate text is right (after removing spaces and hyphens). One wrong symbol = wrong plate. The number a user cares about'],
    ['Whole system (e2e)', 'photo in, plate text out: detector + crop + reader, scored on 204 photos with typed plate text'],
    ['TensorRT FP16', 'NVIDIA\'s optimised runtime with 16-bit numbers; the fastest way to run a model on an NVIDIA GPU'],
    ['McNemar test', 'compares two systems on the same plates using only the plates where exactly one is right; p &lt; 0.05 = a real difference'],
], [2.2, 7.8])}
"""

S['a2'] = f"""
<h1>3. The data</h1>
{table(['Item', 'Detail', 'Page'], [
    ['Photos', '2,600 own photos, different light, angles, rain, dust, low light; about 90 % taken from the left or right side', 'p.6-7'],
    ['Labels', 'boxes drawn by hand in Roboflow, one class "Bangladeshi License Plate"', 'p.7'],
    ['Augmentation', 'blur, noise and weather effects in Roboflow, applied <b>before</b> the split; three "dataset propositions", the best model is kept', 'p.7, Fig. 3'],
    ['Split', 'about 70 % train / 20 % val / 10 % test; test = 260 photos', 'p.7, p.9'],
    ['Text labels for OCR', 'not described; Table 2 counts 100 clean + 80 low-resolution + 50 damaged + 30 weather plates = 260', 'p.12'],
    ['Availability', 'not public, no link', '-'],
], [1.6, 7.0, 1.4])}
<div class="warn"><p><b>Why "augmented before the split" matters.</b> If a photo is copied with blur or noise and the copies are
then split at random, the test set can contain near-copies of training photos. The model has then "seen" the test. Our
own team results show how much this inflates scores (paper #23: 85.6 vs 77.8 % reading on the same plates). The paper
itself lists "addressing the limitations of augmented datasets" as future work (p.17).</p></div>

<h1>4. The method</h1>
{fig('fig_paper.png', 470, 'Figure 2. Three figures from the paper. Left: the workflow, with augmentation before the train/val/test split. Middle: the training plot is a standard Ultralytics results.png (about 118 epochs), although the text says TensorFlow. Right: EasyOCR returns the two plate lines as two separate results; the letters line has much lower confidence.')}
{table(['Step', 'What the paper says', 'Page'], [
    ['1. Detector', 'YOLOv8, pretrained on COCO, fine-tuned; "hyperparameters such as anchor boxes and learning rate" tuned (YOLOv8 has no anchors); model size, epochs, batch, image size not given', 'p.7'],
    ['2. Crop', '"YOLOv8 provides the bounding boxes"; the plate is "segmented to isolate individual characters" (not described)', 'p.8'],
    ['3. Preprocessing', '"preprocessing to enhance clarity" (not described)', 'p.8'],
    ['4. Reading', 'EasyOCR, "specifically configured" for Bangla; no fine-tuning mentioned', 'p.8'],
    ['Tools', 'Python, PyCharm, "TensorFlow" for YOLOv8, OpenCV, EasyOCR; training on Google Colab; Intel i5, 16 GB RAM', 'p.8'],
], [1.4, 7.4, 1.2])}
<p><b>Our re-implementation</b> fills the gaps with the most standard choice and writes each one down: YOLOv8n and YOLOv8s,
120 epochs (Fig. 6 shows about 118), 640 px, batch 16, SGD, seed 42; the most confident box is cropped with 8 % margin;
EasyOCR {M('bn')} reads it, and the lines are joined top line first. The paper's undefined "preprocessing" is tested as
4x upscale + grey + CLAHE (a common choice), next to the raw crop.</p>
"""

S['a3'] = f"""
<h1>5. What the paper reports</h1>
{table(['Result', 'Value', 'Page'], [
    ['Detection accuracy / precision / recall / mAP', '<b>94.8 / 93.5 / 92.3 / 94.0 %</b> on 260 test photos', 'p.9, Table 1'],
    ['Training plot, last epoch (read off)', 'mAP50 about 0.95, mAP50-95 about 0.59, P about 0.92, R about 0.90, about 118 epochs', 'p.11, Fig. 6'],
    ['Character recognition accuracy (EasyOCR)', '<b>92.7 %</b>', 'p.11'],
    ['... by case', 'clean 97.5 % (100 plates), low resolution 92.0 % (80), damaged 85.6 % (50), bad weather 89.3 % (30); weighted mean 92.6 %', 'p.12, Table 2'],
    ['Adverse conditions (metric not named)', 'low light 91.5, rain 89.3, dust 87.2, partly hidden 85.0 %', 'p.13, Table 4'],
    ['Speed and comparison', 'YOLOv8 78 FPS; YOLOv7 65, Faster R-CNN 42, SSD 85 FPS (other rows cited from other papers, other data; GPU not named)', 'p.12, Table 3'],
    ['Abstract', '"94.8 % detection and recognition accuracy" (two different things merged into one number)', 'p.1'],
    ['Training time, model size', 'not reported', '-'],
], [3.4, 5.3, 1.3])}

<h1>6. Weaknesses (with page numbers, so you can show them)</h1>
{ul([
    '<b>No metric is defined</b> (p.9, p.11): "detection accuracy" and "character recognition accuracy" could mean many things; the abstract merges them into one 94.8 % (p.1).',
    '<b>Augmentation before the split</b> (p.7, Fig. 3): copies of one photo can sit in train and test, which inflates every number.',
    '<b>No code, no data, no settings</b>: model size, epochs, image size, batch and learning rate are missing; the 2,600 photos are not public.',
    '<b>The text does not fit YOLOv8</b>: "TensorFlow" (p.8), but YOLOv8 is PyTorch and Fig. 6 is an Ultralytics plot; "anchor boxes" (p.7), but YOLOv8 is anchor-free; "Object Loss" (p.10, Fig. 5) has the values of the DFL loss.',
    '<b>Contradictions</b>: the discussion says Faster R-CNN has a higher mAP than YOLOv8 (p.13), Table 3 says the opposite (p.12); the conclusion says the work focuses "on detection rather than full character recognition" (p.17) while the abstract claims recognition accuracy; the YOLOv7 row cites the YOLOv1 paper (p.12).',
    '<b>Comparison table on other data</b>: Table 3 copies numbers from other papers (other photos, other hardware), so it is not a fair race.',
    '<b>EasyOCR is not explained</b>: no word on how the two plate lines are joined, though its own example (Fig. 8, p.14) shows two separate results.',
    '<b>Preprint</b>: not peer-reviewed.',
])}
<div class="box"><p><b>Fair credit.</b> The recipe is reasonable and easy to repeat, it names real problems of Bangladeshi
plates (two lines, Bangla script, CCTV), and it is the most direct published baseline for "YOLOv8 + general OCR". Treat
the numbers as weak evidence and the recipe as a baseline; that is how we use it.</p></div>
"""

# ======================================================================================== part B (our run)
cnt = A['run']
S['b1'] = f"""
<p class="part">PART B · OUR EXPERIMENT ON DATASET v1.1</p>
<h1>7. What we ran</h1>
<p>The notebook {M('paper06_kaggle.ipynb')} (built by {M('docs/_kit')} like every team kit) runs the paper's method on the
<b>498R BanglaLPR All-in-One v1.1</b> benchmark: all public Bangladeshi plate sets merged, near-duplicates kept in one
group, grouped 70/15/15 split with seed 42, validation and test with human labels only.</p>
{table(['Job', 'GPU', 'What it does', 'Time'], [
    [M('det_v8n'), '0', 'YOLOv8n (COCO), 120 epochs, 640 px, batch 16, SGD, seed 42, on the 4,067 human-boxed training photos; tests, per-condition tests, plate counts', f"{jobs['det_v8n']} min"],
    [M('det_v8s'), '1', 'the same with YOLOv8s', f"{jobs['det_v8s']} min"],
    [M('speed_det_v8n / v8s'), '0', 'batch-1 speed on an idle T4: PyTorch FP32 / FP16, TensorRT FP16, CPU ONNX', f"{jobs['speed_det_v8n']} / {jobs['speed_det_v8s']} min"],
    [M('easyocr_crops'), '0', 'stock EasyOCR 1.7.2 bn on the 651 human-typed OCR test crops: raw and with 4x/grey/CLAHE; per condition, real fog copies, synthetic rain', f"{jobs['easyocr_crops']} min"],
    [M('e2e_v8n / e2e_v8s'), '1', 'whole system on the 204 e2e_test photos: detector → most confident box + 8 % → EasyOCR', f"{jobs['e2e_v8n']} min each"],
], [2.3, 0.5, 6.0, 1.2])}
{table(['Data (v1.1)', 'Train', 'Val', 'Test'], [
    ['detection photos (human boxes used here)', '4,067 human (+4,507 auto, not used)', '896', '1,153 (1,011 plates; 160 photos without a plate)'],
    ['OCR crops with typed plate text', '-', '-', '651 (+26 real fog copies, +651 synthetic rain copies)'],
    ['e2e photos with plate text', '-', '-', '204 (144 BLPD external + 60 NightLPBD)'],
], [3.2, 2.5, 1.0, 3.3])}
<p><b>Why a v1.1 re-run was needed.</b> Dataset v1.1 changed only recognition and OCR crops: 19 BLPD test crops that were
whole vehicle photos were cut to the plate. The detection split is identical to v1.0. So the detectors would repeat, but
the EasyOCR test on the 651 crops had to be re-run, and it now is.</p>
<p><b>Differences from the paper</b> (it gives no settings): our data instead of its 2,600 photos; both YOLOv8n and v8s;
human boxes only (like its hand-drawn boxes); undefined metrics replaced by defined ones (mAP50, plates found %, char
accuracy = 100 - CER, plate exact); "preprocessing" tested as one explicit recipe; Ultralytics 8.4.170 (PyTorch 2.11,
CUDA 12.8, Python 3.13). The run started {cnt['date']} UTC on Tesla T4 x2.</p>
"""

S['b1ok'] = f"""
<h1>8. Is the training OK?</h1>
{table(['Check', 'Result', 'OK?'], [
    ['Dataset version', M('dataset_version: 1.1') + ' in environment.json; downloaded from the team Drive; all counts as expected (log lines 34-72)', 'yes'],
    ['Jobs', f"all 7 ended with code=0; {A['run']['total_h']:.1f} h of the 12 h limit (planned 3.5 h)", 'yes'],
    ['Epochs', f"120 of 120 for both (time caps 3 h / 4 h not reached: {n8['train_min']} / {s8['train_min']} min of training)", 'yes'],
    ['Learning', f"no NaN; validation mAP50 rises from {100 * n8['curve']['mAP50'][0]:.1f} (epoch 1) to {n8['val']['mAP50']} (v8n); best epochs {n8['best_epoch']} (v8n) and {s8['best_epoch']} (v8s); the last epoch is within {abs(n8['val']['mAP50-95'] - n8['last_epoch_val']['mAP50-95']):.1f} (v8n) and {abs(s8['val']['mAP50-95'] - s8['last_epoch_val']['mAP50-95']):.1f} (v8s) mAP50-95 points of the best", 'yes'],
    ['Over-fitting?', f"val box loss flat after about epoch 60, mAP50-95 flat; best.pt (the saved best epoch) is used everywhere, so a late dip does not matter", 'no problem'],
    ['Repeatable (detectors)', f"local re-test of both best.pt on the same 1,153 test photos (RTX 5060 Ti): mAP50 {f2(LD['det_v8n']['test']['mAP50'])} / {f2(LD['det_v8s']['test']['mAP50'])}, mAP50-95 {f2(LD['det_v8n']['test']['mAP50-95'])} / {f2(LD['det_v8s']['test']['mAP50-95'])}, plates found {f2(LD['det_v8n']['plates_found_%'])} / {f2(LD['det_v8s']['plates_found_%'])} %: identical to Kaggle", 'yes'],
    ['Repeatable (EasyOCR)', f"local re-run of EasyOCR 1.7.2: char accuracy {f2(V['raw']['char_acc_%'])} % (Kaggle {f2(RAW['char_acc_%'])}), plate exact {f2(V['raw']['plate_exact_%'])} % (Kaggle {f2(RAW['plate_exact_%'])}); {V['raw']['same_text_as_kaggle_%']:.0f} % of readings letter-for-letter the same (GPU and library versions differ)", 'yes'],
    ['Whole-system crops', f"the local reader-swap test found plates in exactly the same share of photos as Kaggle ({e8n['plate_detected_%']} / {e8s['plate_detected_%']} %)", 'yes'],
], [2.0, 6.9, 1.1])}
{fig('fig_curves.png', 470, 'Figure 3. Validation mAP50 and mAP50-95 per epoch for both detectors (dots: the best epoch, saved as best.pt). Dashed: the last-epoch values read off the paper\'s own Ultralytics plot (Fig. 6, p.11). Our validation curves end at or above the paper\'s.')}
<div class="box"><p><b>Validation 60.7 vs test 45.0 mAP50-95: is the model over-fitted to validation?</b> No. The validation
set has 101 photos from the ridoykj source (11 %), the test set 274 (24 %). That source draws tighter boxes (exactly on
the plate edge) than the others, and mAP50-95 punishes every pixel of disagreement. Section 10 shows it source by source: without ridoykj the test
mAP50-95 is {f1(LD['det_v8n']['test_without_ridoykj']['mAP50-95'])} (v8n), close to validation. mAP50 is the same on both
({n8['val']['mAP50']} val, {n8['test']['mAP50']} test).</p></div>
"""

exp_rows = [
    ['mAP50, precision, recall', 'lower than the paper', f"<b>not confirmed</b>: higher (mAP50 {f1(n8['test']['mAP50'])} / {f1(s8['test']['mAP50'])} vs 94.0)", 'more training photos (4,067 vs about 1,820), mostly large plates in our photos (739 of 1,011 test plates are "large"), clean human boxes'],
    ['mAP50-95', 'lower than about 59', f"confirmed on paper ({f1(n8['test']['mAP50-95'])}), <b>but</b> {f1(LD['det_v8n']['test_without_ridoykj']['mAP50-95'])} without ridoykj", 'one source\'s box style (section 10)'],
    ['FPS', 'v8n faster than 78', f"confirmed: {fps_n:.0f} (FP32) / {trt_n:.0f} (TensorRT FP16)", 'T4, batch 1, pre- and post-processing included'],
    ['EasyOCR character accuracy', 'much lower than 92.7', f"confirmed: {f1(RAW['char_acc_%'])} %; plate exact {f1(RAW['plate_exact_%'])} %", 'general Bangla model, not trained on plate fonts or two-line layout'],
    ['By condition', 'clear better than night / blur / fog', f"confirmed: clear {f1(BC['clear']['char_acc_%'])} &gt; fog {f1(K['extra_tests']['test_fog']['char_acc_%'])} &gt; night {f1(BC['night']['char_acc_%'])} &gt; rain {f1(K['extra_tests']['test_rain_synth']['char_acc_%'])} &gt; blurry {f1(BC['blurry']['char_acc_%'])}", 'as the paper\'s Table 2, only much lower'],
    ['YOLOv8n vs v8s', 's slightly better, n about 2x faster', f"partly: s +{s8['test']['mAP50'] - n8['test']['mAP50']:.1f} mAP50, 17 vs 10 plates ({pv(DP['v8s vs v8n']['p'])}, a tie); n only {fps_n / fps_s:.2f}x faster in FP32, {trt_n / trt_s:.2f}x with TensorRT", 'at batch 1 the fixed pre- and post-processing time dominates'],
]
S['b2'] = f"""
<h1>9. Results next to the paper</h1>
{table(['Metric', 'Paper (page)', 'YOLOv8n', 'YOLOv8s'], [
    ['Precision (%)', '93.5 (p.9)', f2(n8['test']['P']), f2(s8['test']['P'])],
    ['Recall (%)', '92.3 (p.9)', f2(n8['test']['R']), f2(s8['test']['R'])],
    ['mAP50 (%)', '94.0 ("mAP", p.9)', f"<b>{f2(n8['test']['mAP50'])}</b>", f"<b>{f2(s8['test']['mAP50'])}</b>"],
    ['mAP50-95 (%)', 'about 59 (Fig. 6, p.11)', f2(n8['test']['mAP50-95']), f2(s8['test']['mAP50-95'])],
    ['mAP50-95 without the ridoykj source (%)', '-', f2(LD['det_v8n']['test_without_ridoykj']['mAP50-95']), f2(LD['det_v8s']['test_without_ridoykj']['mAP50-95'])],
    ['Plates found, conf 0.25, IoU 0.5 (%)', '94.8 "detection accuracy" (p.9, undefined)', f2(n8['plates']['found_%']), f2(s8['plates']['found_%'])],
    ['Validation mAP50 / mAP50-95 (%)', '-', f"{n8['val']['mAP50']} / {n8['val']['mAP50-95']}", f"{s8['val']['mAP50']} / {s8['val']['mAP50-95']}"],
    ['FPS, T4, batch 1: PyTorch FP32 / TensorRT FP16', '78 (GPU not named, p.12)', f"{fps_n:.1f} / {trt_n:.1f}", f"{fps_s:.1f} / {trt_s:.1f}"],
    ['Training time (min, T4) / parameters', 'not reported', f"{n8['train_min']} / {n8['params_M']} M", f"{s8['train_min']} / {s8['params_M']} M"],
], [3.6, 2.8, 1.8, 1.8])}
{table(['EasyOCR and whole system', 'Paper (page)', 'Ours'], [
    ['EasyOCR char accuracy, 651 human crops (%)', '92.7 (p.11, undefined)', f"<b>{f2(RAW['char_acc_%'])}</b>"],
    ['EasyOCR plate exact (%)', 'not reported', f"<b>{f2(RAW['plate_exact_%'])}</b>"],
    ['EasyOCR with 4x / grey / CLAHE preprocessing: char acc / exact (%)', '"preprocessing" (p.8)', f"{f2(PREP['char_acc_%'])} / {f2(PREP['plate_exact_%'])}"],
    ['EasyOCR clear / night / blurry crops: char accuracy (%)', 'clean 97.5 / low light 91.5 / - (p.12-13)', f"{f1(BC['clear']['char_acc_%'])} / {f1(BC['night']['char_acc_%'])} / {f1(BC['blurry']['char_acc_%'])}"],
    ['EasyOCR real fog / synthetic rain: char accuracy (%)', 'weather 89.3 / rain 89.3 (p.12-13)', f"{f1(K['extra_tests']['test_fog']['char_acc_%'])} / {f1(K['extra_tests']['test_rain_synth']['char_acc_%'])}"],
    ['EasyOCR low-resolution plates', '92.0 (p.12)', 'no crop in our OCR test is under 24 px high (rule of the condition tag), so not measured'],
    ['EasyOCR ms per plate (T4)', 'not reported', f1(K['ms_per_plate'])],
    ['Whole system, 204 photos: plate found / char acc / exact (%), v8n', '-', f"{e8n['plate_detected_%']} / {e8n['char_acc_%']} / <b>{e8n['full_plate_exact_%']}</b>"],
    ['Whole system, v8s', '-', f"{e8s['plate_detected_%']} / {e8s['char_acc_%']} / <b>{e8s['full_plate_exact_%']}</b>"],
    ['Whole system ms per photo (detector + EasyOCR, PyTorch)', '-', f"{e8n['ms_mean']} (v8n), {e8s['ms_mean']} (v8s)"],
], [4.6, 2.9, 2.5])}
<h3>The predictions we wrote down before the run (notes, 11 and 21 Oct): were they right?</h3>
{table(['What', 'Expected', 'Result', 'Why'], exp_rows, [1.8, 1.8, 3.4, 3.0])}
<p class="small">Our photos and the paper's are different, so the rows show direction and size, not a race. The comparison
that <i>is</i> fair is ours against other models on the <b>same</b> test plates (sections 10, 11 and 13).</p>
"""

bs = A['det_box_shape']['det_v8n']
S['b3'] = f"""
<h1>10. The detector in detail</h1>
<h3>Why mAP50-95 is 45 and not 59: one source boxes tighter</h3>
{fig('fig_persource.png', 470, 'Figure 4. Left: test mAP50-95 for each source, for both #6 detectors and paper #2\'s YOLO12s plate detector (all three trained on our v1.1 split). Every model scores 22-24 on ridoykj and 52-73 elsewhere. Right: validation, test, and test without ridoykj.')}
{fig('fig_boxstyle.png', 430, 'Figure 5. A ridoykj and a KUET test photo with the human label (green, dashed) and the YOLOv8n box (orange). Ridoykj labels follow the plate edge exactly; the model, trained mostly on sources that box the plate with a margin, draws a larger box. Both boxes find the plate (IoU above 0.5), but mAP50-95 also counts the IoU 0.75-0.95 levels, where they disagree.')}
<p>All three detectors find almost every ridoykj plate (mAP50 {f1(LD['det_v8n']['by_source']['ridoykj']['mAP50'])}-{f1(LD['p02_lpd_v12s']['by_source']['ridoykj']['mAP50'])}),
but their boxes are {bs['ridoykj']['median_w_ratio']:.2f}x as wide and {bs['ridoykj']['median_h_ratio']:.2f}x as high as the
ridoykj labels (median), against {bs['kuet']['median_w_ratio']:.2f} / {bs['kuet']['median_h_ratio']:.2f} on KUET. Mean IoU is
{f1(bs['ridoykj']['mean_iou'])} % on ridoykj vs {f1(bs['kuet']['mean_iou'])} % on KUET. Paper #2's report found the same for its own
detector (24.2 on ridoykj), so this is a property of the labels, not of YOLOv8. <b>Lesson for our evaluation:</b> report
mAP50 and plates found as the main detector numbers, and mAP50-95 per source or without ridoykj.</p>
"""

S['b3b'] = f"""
<h3>By condition</h3>
{fig('fig_conditions.png', 470, 'Figure 6. Left: detector test mAP50 on the fixed team condition subsets (clear 868 photos, night 71, blurry 53, small plates 22; rain and fog are synthetic copies of the test). Right: EasyOCR character accuracy on our closest condition subsets next to the paper\'s Tables 2 and 4 (its metric is undefined and on its own photos).')}
{ul([
    f"<b>Clear, blurry and synthetic fog are easy</b> for both detectors (mAP50 {100 * n8['by_condition']['clear']['mAP50']:.1f}, {100 * n8['by_condition']['blurry']['mAP50']:.1f}, {100 * n8['by_condition']['fog_synth']['mAP50']:.1f} for v8n).",
    f"<b>Synthetic rain is the hardest</b>: {100 * n8['by_condition']['rain_synth']['mAP50']:.1f} / {100 * s8['by_condition']['rain_synth']['mAP50']:.1f} mAP50, and only {n8['by_condition']['rain_synth']['plates']:.0f} % of plates found by v8n. The paper's rain claim (89.3 %) is not supported on our rain copies.",
    f"<b>Night</b>: {100 * n8['by_condition']['night']['mAP50']:.1f} / {100 * s8['by_condition']['night']['mAP50']:.1f} mAP50; on NightLPBD v8n finds {n8['plates']['found_%_by_source']['nightlpbd']['found_%']:.0f} % of plates. Paper #2's detector, trained with the auto-labelled photos too, finds {DP['night: #2 lpd_v12s vs v8n']['a_%']:.0f} % ({DP['night: #2 lpd_v12s vs v8n']['only_a']} vs {DP['night: #2 lpd_v12s vs v8n']['only_b']} plates, {pv(DP['night: #2 lpd_v12s vs v8n']['p'])}).",
    f"<b>Small plates</b> (22 photos): v8n {100 * n8['by_condition']['small_plates']['mAP50']:.1f}, v8s {100 * s8['by_condition']['small_plates']['mAP50']:.1f}. Few photos, so the v8n-v8s gap there is noise. Only {n8['plates']['plates_by_size']['small']} test plates are COCO-small (under 32x32 px).",
    f"<b>False alarms</b>: a box on {n8['by_condition']['no_plate']['false_alarm_%']} % of the 160 plate-free photos (both sizes).",
])}
<h3>Against other detectors on the same test photos</h3>
{table(['Detector (all on the v1.1 detection split)', 'Training boxes', 'mAP50', 'mAP50-95', 'Plates found', 'Same plates vs #6 YOLOv8n'], [
    ['#6 YOLOv8n (this run)', 'human, 120 epochs', f2(n8['test']['mAP50']), f2(n8['test']['mAP50-95']), f"{f2(n8['plates']['found_%'])} %", '-'],
    ['#6 YOLOv8s (this run)', 'human, 120 epochs', f2(s8['test']['mAP50']), f2(s8['test']['mAP50-95']), f"{f2(s8['plates']['found_%'])} %", f"{DP['v8s vs v8n']['only_a']} vs {DP['v8s vs v8n']['only_b']} plates, {pv(DP['v8s vs v8n']['p'])} (tie)"],
    ['#3 YOLOv8n (Md. Yousuf)', 'human + auto, 60 epochs', f2(P03['test']['mAP50']), f2(P03['test']['mAP50-95']), f"{f2(P03['found_%'])} %", 'not paired (no per-plate file); same photos'],
    ['#2 YOLO12s plate detector (our #2 run)', 'human + auto, 60 epochs', f2(LD['p02_lpd_v12s']['test']['mAP50']), f2(LD['p02_lpd_v12s']['test']['mAP50-95']), f"{f2(LD['p02_lpd_v12s']['plates_found_%'])} %", f"{DP['#2 lpd_v12s vs v8n']['only_a']} vs {DP['#2 lpd_v12s vs v8n']['only_b']} plates, {pv(DP['#2 lpd_v12s vs v8n']['p'])}"],
], [3.1, 1.9, 0.9, 1.0, 1.1, 2.6])}
<div class="box"><p><b>What this says.</b> (1) YOLOv8n and YOLOv8s are tied on our data: take n for speed. (2) The same
YOLOv8n trained by a teammate with the <b>auto-labelled</b> photos as well (DIU, LPD4K, BLPD: 4,507 more photos) finds
more plates ({f1(P03['found_%'])} vs {f1(n8['plates']['found_%'])} %) in half the epochs, and paper #2's YOLO12s with the same
boxes beats our v8n on the same plates ({DP['#2 lpd_v12s vs v8n']['only_a']} vs {DP['#2 lpd_v12s vs v8n']['only_b']}, {pv(DP['#2 lpd_v12s vs v8n']['p'])}).
The paper's "human boxes only" choice costs about 1.5-2 points of plates found; for the final detector, train on human +
auto boxes.</p></div>
"""

ex = EO['picked_examples']
exrow = lambda i, e: [f"({'abcde'[i]}) {e['case']}", BN(e['true']), BN(e['easyocr'] or '(nothing)') + f" ({e['conf']:.2f})",
                      BN(e['lpnr_v12n']) + (' ✓' if e['lpnr_right'] else ' ✗')]
miss = EO['char_miss_rate_top'][:6]
S['b4'] = f"""
<h1>11. EasyOCR: why the reading fails</h1>
{fig('fig_readers.png', 470, 'Figure 7. The same 651 human-typed test crops read by EasyOCR (this paper) and by the plate readers our team trained on the same v1.1 data (papers #2 and #24). Dark: plate exact; light: character accuracy. Dashed: the paper\'s 92.7 % on its own data.')}
<p>On the same 651 crops, paper #2's character detector reads {f1(lp2['a_%'])} % of plates exactly, EasyOCR
{f1(lp2['b_%'])} %: the detector is right on <b>{lp2['only_a']}</b> plates where EasyOCR is wrong, EasyOCR on {lp2['only_b']}
where the detector is wrong ({pv(lp2['p'])}). Even our weakest trained reader, ViTSTR, beats it
({f1(SM['#24 vitstr (ViTSTR)']['a_%'])} %, {pv(SM['#24 vitstr (ViTSTR)']['p'])}). EasyOCR reaches the paper's level on no subset.</p>
{fig('fig_examples.png', 470, 'Figure 8. Five test crops, one per typical case (readings in the table below).')}
{table(['Case', 'Truth', 'EasyOCR (confidence)', 'Paper #2 char-YOLO12n'], [exrow(i, e) for i, e in enumerate(ex)], [1.7, 2.6, 3.3, 2.4])}
"""

S['b4b'] = f"""
<h3>Where the errors are</h3>
{table(['Check (651 crops)', 'Result', 'What it means'], [
    ['Letters line right (district, মেট্রো, series letter)', f"{f1(EO['top_line_exact_%'])} %", 'the Bangla words are the main problem'],
    ['Digits line right (6 digits)', f"{f1(EO['digits_exact_%'])} %", 'digits are easier, but still wrong on 2 of 3 plates'],
    ['Readings with English digits (0-9)', f"{f1(EO['pred_with_latin_digits_%'])} %", BN('৪') + ' is read as "8", ' + BN('০') + ' as "0": the bn model mixes scripts; scoring maps them by value, so "8" counts as ' + BN('৮') + ', a wrong digit'],
    ['Readings with brackets, bars or other punctuation', f"{f1(EO['pred_with_brackets_or_punct_%'])} %", 'plate frame, bolts and dirt read as characters'],
    ['Characters missed most (share of their occurrences)', ', '.join(f"{BN(c)} {r:.0f} %" for c, n_, r in miss), 'series letters (' + BN('গ ব হ') + ') and the ' + BN('ো') + ' and ' + BN('ট') + ' of ' + BN('মেট্রো') + '; most common swap: ' + BN('৪') + f' read as "8" ({EO["top_substitutions"][0][2]} times)'],
    ['Empty readings', f"{f1(EO['empty_%'])} %", 'CRAFT found no text'],
    ['Valid BRTA format', f"{f1(EO['valid_format_%'])} % of readings; of those, {f1(EO['exact_given_valid_%'])} % are exactly right", 'a format check is a strong "trust this reading" signal'],
    ['Confidence as a warning', f"AUC {EO['conf_auc']:.2f}; mean {EO['conf_mean_right']:.2f} on right vs {EO['conf_mean_wrong']:.2f} on wrong readings; at confidence &gt;= 0.9: {EO['conf_thresholds']['0.9']['exact_among_kept_%']:.0f} % right, but only {EO['conf_thresholds']['0.9']['kept_%']:.1f} % of plates kept", 'confidence ranks errors well, but almost nothing is both confident and right'],
], [3.0, 2.8, 4.2])}
<h3>The paper's "preprocessing" makes it worse</h3>
<p>With 4x upscale + grey + CLAHE, exact reading drops from {f1(RAW['plate_exact_%'])} to {f1(PREP['plate_exact_%'])} % (69 plates lost,
13 gained, {pv(EO['prep_vs_raw']['p'])}) and the time per plate rises from {f1(K['ms_per_plate'])} to {f1(PREP['ms_per_plate'])} ms. Our crops
are already large (median {EO['crop_height_px']['median']} px high, 10 % below {EO['crop_height_px']['p10']} px), so a 4x upscale makes
huge images, CRAFT then splits the text into more pieces, and CLAHE strengthens dirt and frame edges. Preprocessing has to be
tested, not assumed (proposal G4).</p>
<h3>By source and condition</h3>
{table(['Subset', 'Crops', 'Char accuracy', 'Plate exact'], [
    *[[f"source {s_}", K['by_source'][s_]['plates'], f"{f1(K['by_source'][s_]['char_acc_%'])} %", f"{f1(K['by_source'][s_]['plate_exact_%'])} %"] for s_ in ('anpr', 'lpdba', 'blpd', 'kuet', 'nightlpbd')],
    *[[f"condition {c}", BC[c]['plates'], f"{f1(BC[c]['char_acc_%'])} %", f"{f1(BC[c]['plate_exact_%'])} %"] for c in ('clear', 'night', 'blurry')],
    ['real fog copies', K['extra_tests']['test_fog']['plates'], f"{f1(K['extra_tests']['test_fog']['char_acc_%'])} %", f"{f1(K['extra_tests']['test_fog']['plate_exact_%'])} %"],
    ['synthetic rain copies', K['extra_tests']['test_rain_synth']['plates'], f"{f1(K['extra_tests']['test_rain_synth']['char_acc_%'])} %", f"{f1(K['extra_tests']['test_rain_synth']['plate_exact_%'])} %"],
], [4.0, 1.4, 2.3, 2.3], num=(1, 2, 3))}
<p>EasyOCR is best on the ANPR and LPDB-A crops and worst on KUET, BLPD and night photos.
At night and on blurry crops almost no plate is read exactly (1.5 % and 1.8 %). The paper's 91.5 % for "low light" and
89.3 % for rain cannot be reproduced on real plates.</p>
"""

vr = lambda k, nm: [nm, f2(V[k]['plate_exact_%']), f2(V[k]['char_acc_%']), f2(V[k]['valid_format_%']), f1(V[k]['ms_per_plate_local']),
                    '-' if k == 'raw' else f"{pvr[k]['only_a']} vs {pvr[k]['only_b']}, {pv(pvr[k]['p'])}"]
S['b5'] = f"""
<h1>12. Can EasyOCR be saved cheaply? (local check)</h1>
<p>Before dropping EasyOCR, we tried the two cheapest fixes on a local GPU (RTX 5060 Ti, EasyOCR 1.7.2, same 651 crops, same
line joining as the kit; {M('report_src/easyocr_check.py')}): an <b>allowlist</b> (EasyOCR may only output the
{EL['allowlist_chars']} characters that occur in our training plate texts, so no "8", "|" or "[") and a mild 2x upscale.</p>
{table(['Variant', 'Plate exact %', 'Char acc. %', 'Valid format %', 'ms / plate (local)', 'vs local raw (plates gained vs lost)'], [
    vr('raw', 'raw crop (= the Kaggle job)'), vr('prep', '4x + grey + CLAHE (kit)'), vr('x2', '2x upscale, colour'),
    vr('allow', 'allowlist'), vr('allow_x2', 'allowlist + 2x upscale'),
], [2.8, 1.2, 1.2, 1.3, 1.3, 2.4], num=(1, 2, 3, 4))}
<div class="box"><p><b>Result: no cheap fix works.</b> The allowlist removes the English digits and raises character
accuracy by {V['allow']['char_acc_%'] - V['raw']['char_acc_%']:.1f} points ({f1(V['raw']['char_acc_%'])} → {f1(V['allow']['char_acc_%'])} %),
but exact reading stays where it was: {f2(V['allow']['plate_exact_%'])} vs {f2(V['raw']['plate_exact_%'])} %
({pvr['allow']['only_a']} plates gained, {pvr['allow']['only_b']} lost, {pv(pvr['allow']['p'])}). The 2x upscale is slightly worse, with or
without the allowlist, and doubles the time. So the problem is not the output alphabet or the image size but the model:
the bn recogniser has never seen two-line plate layouts, plate fonts or মেট্রো, and when it is forced to pick a Bangla
character instead of "8" it often picks the wrong one. A character detector trained on our plates reads
{f1(lp2['a_%'])} % of the same crops. Fine-tuning EasyOCR on our 5,607 training plate texts would be the fair test of
"EasyOCR adapted to plates" (section 21).</p></div>
"""

S['b6'] = f"""
<h1>13. The whole system: detector or reader?</h1>
<p>The kit's whole-system test reads the 204 e2e photos with YOLOv8 + EasyOCR. To see which part loses the plates, we kept
<b>exactly the same detector crops</b> (most confident box, 8 % margin) and replaced only EasyOCR with paper #2's
character reader (char-YOLO12n, trained on the same v1.1 data), locally ({M('report_src/local_check.py')}). We also put the
#6 detectors into paper #2's cascade code in place of its own plate detector.</p>
{fig('fig_e2e.png', 470, 'Figure 9. What happens to the 204 photos. Grey: the detector found no plate; orange: plate found but read wrong; green: read exactly. The detector loses 8-11 photos in every system; EasyOCR loses most of the rest.')}
{table(['System (204 photos)', 'Plate found', 'Exact', 'Char acc.', 'BLPD ext. / night exact', 'ms per photo'], [
    ['#6 YOLOv8n + EasyOCR (Kaggle)', f"{e8n['plate_detected_%']} %", f"<b>{f2(e8n['full_plate_exact_%'])} %</b>", f"{f1(e8n['char_acc_%'])} %", f"{f1(e8n['by_source']['blpd_ext']['plate_exact_%'])} / {f1(e8n['by_source']['nightlpbd']['plate_exact_%'])} %", f"{e8n['ms_mean']} (T4)"],
    ['#6 YOLOv8s + EasyOCR (Kaggle)', f"{e8s['plate_detected_%']} %", f"<b>{f2(e8s['full_plate_exact_%'])} %</b>", f"{f1(e8s['char_acc_%'])} %", f"{f1(e8s['by_source']['blpd_ext']['plate_exact_%'])} / {f1(e8s['by_source']['nightlpbd']['plate_exact_%'])} %", f"{e8s['ms_mean']} (T4)"],
    ['#6 YOLOv8n + char-YOLO12n, same crops (local)', f"{x8n['plate_detected_%']} %", f"<b>{f2(x8n['plate_exact_%'])} %</b>", f"{f1(x8n['char_acc_%'])} %", f"{f1(x8n['by_source']['blpd_ext'])} / {f1(x8n['by_source']['nightlpbd'])} %", f"{x8n['ms_mean_local']} (local)"],
    ['#6 YOLOv8s + char-YOLO12n, same crops (local)', f"{x8s['plate_detected_%']} %", f"<b>{f2(x8s['plate_exact_%'])} %</b>", f"{f1(x8s['char_acc_%'])} %", f"{f1(x8s['by_source']['blpd_ext'])} / {f1(x8s['by_source']['nightlpbd'])} %", f"{x8s['ms_mean_local']} (local)"],
    ['#2 cascade with #6 YOLOv8s (local)', f"{c8s['plate_detected_%']} %", f"{f2(c8s['plate_exact_%'])} %", f"{f1(c8s['char_acc_%'])} %", f"{f1(c8s['by_source']['blpd_ext'])} / {f1(c8s['by_source']['nightlpbd'])} %", f"{c8s['ms_mean_local']} (local)"],
    ['#2 cascade with its own YOLO12s (local = Kaggle)', f"{c02['plate_detected_%']} %", f"{f2(c02['plate_exact_%'])} %", f"{f1(c02['char_acc_%'])} %", f"{f1(c02['by_source']['blpd_ext'])} / {f1(c02['by_source']['nightlpbd'])} %", f"{c02['ms_mean_local']} (local)"],
], [3.1, 1.0, 1.25, 1.0, 1.8, 1.35])}
{ul([
    f"<b>The reader is the bottleneck.</b> Same photos, same crops: the char-YOLO reader is right on {EP['det_v8n + lpnr_v12n (p06 crop) vs det_v8n + EasyOCR']['only_a']} photos where EasyOCR is wrong, EasyOCR on {EP['det_v8n + lpnr_v12n (p06 crop) vs det_v8n + EasyOCR']['only_b']} ({pv(EP['det_v8n + lpnr_v12n (p06 crop) vs det_v8n + EasyOCR']['p'])}).",
    f"<b>The #6 detector is good enough.</b> In paper #2's cascade, YOLOv8s gives {f2(c8s['plate_exact_%'])} % vs {f2(c02['plate_exact_%'])} % with #2's own YOLO12s ({EP['#2 cascade with det_v8s vs #2 cascade with its lpd_v12s (local)']['only_a']} vs {EP['#2 cascade with det_v8s vs #2 cascade with its lpd_v12s (local)']['only_b']} photos, {pv(EP['#2 cascade with det_v8s vs #2 cascade with its lpd_v12s (local)']['p'])}: a tie). YOLOv8n: {f2(c8n['plate_exact_%'])} % ({pv(EP['#2 cascade with det_v8n vs with lpd_v12s (local)']['p'])}).",
    f"<b>Best combination so far</b> on these 204 photos: #6 YOLOv8s + #2 char-YOLO12n, {f2(x8s['plate_exact_%'])} % exact (night {f1(x8s['by_source']['nightlpbd'])} %). It is not significantly better than #2's own cascade, so call it \"as good as\".",
    f"<b>Lost before reading:</b> {ls8n['no_box']} photos (v8n) and {E['loss_split']['e2e_v8s']['no_box']} (v8s) have no plate box at all, every one a NightLPBD night photo (#2's detector: 9); this is the detector's share of the loss.",
    "The local check reproduces #2's Kaggle cascade photo by photo (the same 177 photos right), so the swap test is trustworthy.",
])}
"""

S['b7'] = f"""
<h1>14. Speed</h1>
{fig('fig_speed.png', 470, 'Figure 10. Left: detector frames per second at batch 1 on one Tesla T4 (CPU: the Kaggle VM\'s 4 cores, ONNX Runtime), pre- and post-processing included. Dashed: the paper\'s 78 FPS (GPU not named). Right: where the time of one photo goes.')}
{table(['Speed, batch 1', 'YOLOv8n', 'YOLOv8s', 'Paper'], [
    ['T4 PyTorch FP32: FPS (ms)', f"{fps_n:.1f} ({sp_n['fp32']['ms_per_image']} ms)", f"{fps_s:.1f} ({sp_s['fp32']['ms_per_image']} ms)", '78 FPS (GPU and precision not named)'],
    ['T4 PyTorch FP16', f"{sp_n['fp16']['fps']:.1f}", f"{sp_s['fp16']['fps']:.1f}", '-'],
    ['T4 TensorRT FP16', f"<b>{trt_n:.1f}</b> ({sp_n['tensorrt_fp16']['ms_per_image']} ms)", f"<b>{trt_s:.1f}</b> ({sp_s['tensorrt_fp16']['ms_per_image']} ms)", '-'],
    ['CPU ONNX (Kaggle, 4 cores)', f"{sp_n['cpu_onnx']['fps']:.1f}", f"{sp_s['cpu_onnx']['fps']:.1f}", '-'],
    ['Parameters / GFLOPs', f"{n8['params_M']} M / {n8['GFLOPs']}", f"{s8['params_M']} M / {s8['GFLOPs']}", 'not given'],
    ['EasyOCR per plate (T4, raw crop)', f1(K['ms_per_plate']), '', '-'],
    ['Whole photo, detector + EasyOCR (T4, PyTorch)', f"{e8n['ms_mean']} ms = {1000 / e8n['ms_mean']:.1f} photos/s", f"{e8s['ms_mean']} ms", '-'],
    ['Char-YOLO readers per plate (T4; papers #2 / #24)', '23.6 / 9.4 ms', '', '-'],
], [3.6, 2.2, 2.0, 2.2])}
<p><b>Reading the table.</b> PyTorch FP16 is not faster than FP32 at batch 1 for such small models (the GPU waits for
Python); TensorRT is. v8n is only {fps_n / fps_s:.2f}x faster than v8s in FP32 because pre- and post-processing take the same
time for both. The detector is not the speed problem of this system: EasyOCR takes {K['ms_per_plate'] / sp_n['tensorrt_fp16']['ms_per_image']:.0f}x
the TensorRT detector time per plate. On the Kaggle CPU, v8n reaches {sp_n['cpu_onnx']['fps']:.0f} FPS, enough for our 10 FPS
target (RQ3) if the reader is light too.</p>
"""

S['b8'] = f"""
<h1>15. Findings, and the numbers to quote</h1>
{table(['#', 'Finding', 'Number to quote', 'Section'], [
    ['1', 'The v1.1 run is healthy and repeatable', f"7/7 jobs code=0; 120/120 epochs; local re-test identical (mAP50 {f2(n8['test']['mAP50'])})", '8'],
    ['2', 'The YOLOv8 detector is as good as the paper claims, and meets our 95 % target', f"mAP50 {f1(n8['test']['mAP50'])} / {f1(s8['test']['mAP50'])} % (paper 94.0); plates found {f1(n8['plates']['found_%'])} / {f1(s8['plates']['found_%'])} %", '9, 10'],
    ['3', 'Low mAP50-95 is a label-style effect of one source', f"ridoykj {f1(LD['det_v8n']['by_source']['ridoykj']['mAP50-95'])}; without it {f1(LD['det_v8n']['test_without_ridoykj']['mAP50-95'])} (paper about 59)", '10'],
    ['4', 'Auto-labelled photos help the detector', f"human+auto v8n (#3) {f1(P03['found_%'])} % vs human-only {f1(n8['plates']['found_%'])} % found; #2 YOLO12s beats v8n 24 vs 5 plates ({pv(DP['#2 lpd_v12s vs v8n']['p'])})", '10'],
    ['5', 'Stock EasyOCR cannot read Bangladeshi plates', f"char acc. {f1(RAW['char_acc_%'])} % (paper 92.7); exact {f1(RAW['plate_exact_%'])} %; letters line {f1(EO['top_line_exact_%'])} %", '11'],
    ['6', 'Preprocessing hurts EasyOCR; cheap fixes do not help', f"4x/CLAHE {f1(PREP['plate_exact_%'])} %; allowlist {f1(V['allow']['plate_exact_%'])} % ({pv(pvr['allow']['p'])}) vs char-YOLO {f1(lp2['a_%'])} %", '11, 12'],
    ['7', 'With a plate-trained reader the same system works', f"same 204 photos and crops: {f1(e8n['full_plate_exact_%'])} → {f1(x8n['plate_exact_%'])} % (v8n), {f1(e8s['full_plate_exact_%'])} → {f1(x8s['plate_exact_%'])} % (v8s)", '13'],
    ['8', 'Speed is not the issue; EasyOCR is the slow part', f"v8n {trt_n:.0f} FPS TensorRT, {fps_n:.0f} FPS FP32 (paper 78); EasyOCR {f1(K['ms_per_plate'])} ms/plate", '14'],
], [0.4, 3.8, 4.6, 0.9])}
<h3>What we keep and what we drop</h3>
{table(['From paper #6', 'Decision', 'Why'], [
    ['YOLOv8 one-class plate detector', '<b>keep</b> as a detector candidate (O2): v8n for speed, trained on human + auto boxes', 'findings 2, 4, 8'],
    ['Stock EasyOCR bn', '<b>drop</b> from the system; keep as the baseline row of our results table', 'findings 5, 7'],
    ['"Preprocessing" (4x + grey + CLAHE)', '<b>drop</b>; any enhancement goes into the G4 ablation with measured gains', 'finding 6'],
    ['mAP50-95 as the headline detector number', '<b>replace</b> with mAP50 + plates found; mAP50-95 per source', 'finding 3'],
    ['Undefined "character accuracy"', '<b>replace</b> with char accuracy = 100 - CER and plate exact', 'section 11'],
    ['Allowlist / format check on OCR output', '<b>keep the idea</b> for our decoder (O3: BRTA-format-aware correction)', 'section 11-12: valid format → 84 % right'],
], [3.2, 4.0, 2.8])}
"""

# ======================================================================================== part C (use it)
S['c1'] = f"""
<p class="part">PART C · USE IT</p>
<h1>16. How to use this paper in our research</h1>
{table(['Our research part (proposal)', 'How paper #6 is used', 'Evidence we now have (v1.1)'], [
    ['Baselines section ("EasyOCR after our detector")', 'it IS the promised baseline: YOLOv8 + stock EasyOCR, run on our benchmark', f"det mAP50 {f1(n8['test']['mAP50'])}; EasyOCR {f1(RAW['char_acc_%'])} % char acc., {f1(RAW['plate_exact_%'])} % exact; e2e {f1(e8n['full_plate_exact_%'])} %"],
    ['RQ2 / O3: character-level recognizer, &gt;= 90 % char accuracy', 'the general-OCR side of the comparison; shows why a plate-trained, character-level reader is needed', f"same 651 crops: char-YOLO {f1(lp2['char_acc_other'])} % char acc. / {f1(lp2['a_%'])} % exact vs EasyOCR {f1(RAW['char_acc_%'])} / {f1(RAW['plate_exact_%'])} % ({lp2['only_a']} vs {lp2['only_b']}, {pv(lp2['p'])})"],
    ['O2: detector choice (YOLOv8 / YOLO11 / YOLOv12)', 'YOLOv8n and v8s rows with speed; n = s on accuracy', f"v8n {trt_n:.0f} FPS TensorRT; v8s vs v8n {DP['v8s vs v8n']['only_a']} vs {DP['v8s vs v8n']['only_b']} plates (tie)"],
    ['G3 / O1: benchmark and labels', 'auto-labelled photos are worth using; label style differs per source', f"human+auto {f1(P03['found_%'])} vs human {f1(n8['plates']['found_%'])} % found; ridoykj mAP50-95 {f1(LD['det_v8n']['by_source']['ridoykj']['mAP50-95'])}"],
    ['RQ4 / G4: conditions, CLAHE and upscaling study', 'first measured data points', f"rain {100 * n8['by_condition']['rain_synth']['mAP50']:.1f} mAP50; EasyOCR night {f1(BC['night']['plate_exact_%'])} % exact; 4x+CLAHE −{RAW['plate_exact_%'] - PREP['plate_exact_%']:.1f} points"],
    ['RQ3 / G1: &gt;= 10 FPS', 'detector cost vs reader cost', f"detector {sp_n['tensorrt_fp16']['ms_per_image']} ms (T4 TRT), EasyOCR {f1(K['ms_per_plate'])} ms; CPU v8n {sp_n['cpu_onnx']['fps']:.0f} FPS"],
    ['O5: evaluation protocol', 'defined metrics and per-source reporting', 'undefined "accuracy" in the paper; mAP50-95 depends on label style'],
    ['Demo system', 'detector part of the demo', f"YOLOv8s + #2 char-YOLO12n: {f1(x8s['plate_exact_%'])} % exact on 204 photos (= #2 cascade {f1(c02['plate_exact_%'])} %, tie)"],
], [2.6, 3.4, 4.0])}
<div class="warn"><p><b>Fix one sentence in our proposal.</b> The related-work paragraph says Ahamed and Ismail "report 94.8 %
combined detection and recognition accuracy". Keep the citation, but add what we measured: on our benchmark the same
pipeline reads {f1(RAW['plate_exact_%'])} % of plates exactly ({f1(RAW['char_acc_%'])} % character accuracy).</p></div>

<h1>17. What benefit do you get from it?</h1>
{ul([
    '<b>A finished, honest baseline.</b> The proposal promised "EasyOCR after our detector" as a baseline; it now exists with defined metrics, per-condition results and released weights.',
    '<b>A strong argument for our design.</b> Same photos, same crops, only the reader swapped: 12 % → 85-88 %. That one paired number justifies the character-level reader (RQ2) better than any citation.',
    '<b>A ready plate detector.</b> YOLOv8n: 6.2 MB, 97.4 % mAP50, 183 FPS with TensorRT on a T4, 16 FPS on a 4-core CPU; usable in the demo today.',
    '<b>An explained anomaly.</b> The low mAP50-95 is traced to one source\'s box style, which protects every teammate\'s detector numbers from a wrong conclusion.',
    '<b>Evidence that preprocessing must be measured</b>: the obvious "upscale + CLAHE" cuts EasyOCR\'s exact rate to about a third (13.5 → 4.9 %).',
    '<b>Practice in reading weak papers</b>: undefined metrics, augmentation before the split and a TensorFlow claim for a PyTorch model are exactly what examiners ask about.',
    '<b>Pre-registered predictions</b>: we wrote the expected gaps down before the run; 4 of 6 held, 1 held partly (v8n vs v8s speed), and the one that failed (the detector is better than the paper, not worse) has a reason (section 9).',
])}
<h3>A paragraph you can adapt for the related-work chapter</h3>
<pre>Ahamed and Ismail [ref] fine-tune a COCO-pretrained YOLOv8 as a plate detector and read
the crop with stock EasyOCR, reporting 94.0% mAP and 92.7% character accuracy on 260 own
test photos; the metrics are not defined and the photos were augmented before splitting.
Re-implemented on our duplicate-free benchmark, the detector reaches 97.4% mAP50
(YOLOv8n), but stock EasyOCR reads only 60.5% of the characters and 13.5% of 651 plates
exactly, and neither an output allowlist nor upscaling changes that. With the same
detector crops, a character-level YOLO reader trained on our data reads 84.8-88.2% of
204 whole photos exactly, against 10-12% with EasyOCR.</pre>
"""

S['c2'] = f"""
<h1>18. What to show the faculty (8 slides)</h1>
{table(['#', 'Slide title', 'What is on it', 'Source in this report'], [
    ['1', 'The paper and its claim', 'YOLOv8 + stock EasyOCR; 94.0 % mAP, 78 FPS, 92.7 % char accuracy on 260 own photos; no code, no data', 'sections 1, 5'],
    ['2', 'What we checked first', 'undefined metrics, augmentation before the split, TensorFlow/anchor contradictions (with page numbers)', 'section 6, Figure 2'],
    ['3', 'Our setup', 'v1.1 benchmark, human boxes, YOLOv8n/s, 120 epochs, T4 x2; training healthy and repeatable', 'sections 7-8, Figure 3'],
    ['4', 'Detector: as good as claimed', f"mAP50 {f1(n8['test']['mAP50'])} / {f1(s8['test']['mAP50'])}; {fps_n:.0f} / {trt_n:.0f} FPS; mAP50-95 explained by ridoykj", 'section 9-10, Figures 4-5'],
    ['5', 'EasyOCR: the weak link', f"{f1(RAW['char_acc_%'])} % char accuracy, {f1(RAW['plate_exact_%'])} % exact; examples; English digits; preprocessing hurts", 'section 11, Figures 7-8'],
    ['6', 'The key experiment', f"same photos and crops, reader swapped: {f1(e8n['full_plate_exact_%'])} → {f1(x8n['plate_exact_%'])} % (v8n), {f1(x8s['plate_exact_%'])} % (v8s)", 'section 13, Figure 9'],
    ['7', 'Conditions and speed', 'rain and night are the hard cases; EasyOCR is 17x the detector time', 'Figures 6, 10'],
    ['8', 'What we take for our system', 'YOLOv8n detector (human + auto boxes); character-level reader; EasyOCR stays a baseline; defined metrics', 'section 15 table'],
], [0.3, 2.2, 5.0, 2.1])}
<div class="box"><p><b>Keep it honest:</b> call 97.4 % "mAP50 on our test split" and never claim we "beat" the paper (different
photos). Say "same photos" only for comparisons on our own test (sections 10, 11, 13). The reader-swap and EasyOCR-fix
tests ran on our own PC, not on Kaggle; say so if asked. The #3 comparison (auto labels) is not paired.</p></div>

<h1>19. What to explain: a 3-minute script</h1>
<p>"This 2025 preprint builds the simplest plate reader for Bangladesh: a YOLOv8 detector fine-tuned to find plates, and the
free EasyOCR Bangla model to read them. It reports 94 percent mAP, 78 frames per second and 92.7 percent character accuracy.
But it is a preprint without code or data, its metrics are not defined, and it augmented the photos before splitting
them, so test photos can be copies of training photos.</p>
<p>We rebuilt it on our own benchmark, where duplicates are kept in one split. Both YOLOv8n and YOLOv8s trained cleanly for
120 epochs, and a local re-test gives the same numbers. The detector is as good as claimed: 97.4 percent mAP50, 96 percent of
plates found, 183 frames per second with TensorRT. Its mAP50-95 looks low, 45, but that comes from one source whose labels sit
exactly on the plate edge while the others leave a margin; without it, the score is 64.</p>
<p>The reading is the problem. Stock EasyOCR reaches 60 percent character accuracy and reads only 13.5 percent of 651 real plates
exactly. It mixes in English digits, breaks the word Metro, and the paper's kind of preprocessing makes it worse.
Forcing it to output only plate characters does not help either: still 14 percent.</p>
<p>Our key test keeps the same detector and the same crops on 204 whole photos and only swaps EasyOCR for the character-level
reader we trained in paper 2: exact reading jumps from 12 to 85 percent with YOLOv8n and 88 percent with YOLOv8s. So in our
system we keep YOLOv8 as the detector, train it on all our boxes, use a character-level reader, and keep EasyOCR only as the
baseline we must beat."</p>
"""

S['c3'] = f"""
<h1>20. Questions the faculty may ask</h1>
<p class="q">Q1. Your mAP50 is higher than the paper's. Did you beat it?</p>
<p>No claim of that: different photos. Ours has more training photos (4,067 human-labelled), mostly large plates, and a
duplicate-free test; theirs is unknown and possibly leaky. It shows that YOLOv8 is a strong plate detector, which is all we
need from it.</p>
<p class="q">Q2. Your mAP50-95 is 45, theirs about 59. Is your box regression bad?</p>
<p>No. The ridoykj source (274 test photos) boxes the plate exactly at its edge, while the other sources leave a margin, so
the model's boxes are about 1.2x wider and 1.4x higher than ridoykj's labels. Every detector we trained, including
paper #2's YOLO12s, scores 22-24 there and 52-73 on the other sources. Without ridoykj our test mAP50-95 is
{f1(LD['det_v8n']['test_without_ridoykj']['mAP50-95'])}, and validation (fewer ridoykj photos) is {f1(n8['val']['mAP50-95'])}.</p>
<p class="q">Q3. EasyOCR gets 92.7 % in the paper and 60 % for you. Is your EasyOCR set up wrong?</p>
<p>We use the same library and language as the paper, without fine-tuning, as it says. A local re-run on another GPU gives the
same {f1(V['raw']['char_acc_%'])} %. The paper never defines its metric and tests on its own unpublished plates; its own example
(Fig. 8, p.14) already shows low confidence (0.52) on the letters line, which is where we see most errors.</p>
<p class="q">Q4. How do you join EasyOCR's two lines? Maybe that causes the errors.</p>
<p>Top line first, left to right, then scoring ignores spaces and hyphens. Even each line alone is mostly wrong: the letters line
is right on {f1(EO['top_line_exact_%'])} % of plates and the 6 digits on {f1(EO['digits_exact_%'])} %, so joining is not the cause.</p>
<p class="q">Q5. Why not fine-tune EasyOCR?</p>
<p>That is a different method from the paper (which uses it stock). It is on our next-step list as a fair "EasyOCR on plates"
test; our trained readers already reach {f1(lp2['a_%'])} % exact on the same crops, so it has a high bar to clear.</p>
<p class="q">Q6. Is the reader-swap test fair?</p>
<p>Yes: the same 204 photos, the same detector, the same crop for every photo; only the reader differs. Plates found is identical to
the Kaggle run ({e8n['plate_detected_%']} %), and our local copy of paper #2's cascade reproduces its Kaggle result photo by photo.</p>
<p class="q">Q7. YOLOv8n or YOLOv8s?</p>
<p>n. They tie on accuracy (17 vs 10 plates, {pv(DP['v8s vs v8n']['p'])}), n has 3.7x fewer parameters and is faster on CPU
({sp_n['cpu_onnx']['fps']:.1f} vs {sp_s['cpu_onnx']['fps']:.1f} FPS).</p>
<p class="q">Q8. Does the detector work at night and in rain?</p>
<p>Night is weaker (v8n finds {n8['plates']['found_%_by_source']['nightlpbd']['found_%']:.0f} % of NightLPBD plates; #2's detector with auto labels
{DP['night: #2 lpd_v12s vs v8n']['a_%']:.0f} %). Synthetic rain is the hardest case ({100 * n8['by_condition']['rain_synth']['mAP50']:.1f} mAP50). We have
no real rain photos; that is a data gap for RQ4.</p>
<p class="q">Q9. 78 FPS vs your 114-183: is that fair?</p>
<p>The paper names no GPU and no precision. Ours is batch 1 on a T4 with pre- and post-processing included, given per precision.
Either way the detector is fast; the reader decides the system speed.</p>
<p class="q">Q10. What is the one thing you take from this paper?</p>
<p>That a YOLOv8 plate detector is enough, and a general OCR is not: the paired swap test (12 % → 85-88 %) is the evidence for our
character-level reader.</p>
"""

S['c4'] = f"""
<h1>21. Next steps (in order)</h1>
{ul([
    f'<b>Upload</b> (section 22). The team table is filled: {ADD} in {M("docs/results")} (done on 24 Oct; check the notes\' results block).',
    '<b>Detector for the system:</b> retrain YOLOv8n on human + auto boxes (as #3/#2 did) with our kit and compare on the same plates; add YOLO11n (O2). Report mAP50, plates found, per-source mAP50-95.',
    '<b>Fine-tune EasyOCR\'s recogniser</b> on our 5,607 training plate texts (two lines as one image, or line by line) as the fair "general OCR adapted to plates" baseline; compare with char-YOLO on the same 651 crops.',
    '<b>Decoder with BRTA format and district dictionary</b> (O3): the format check already marks good readings (84 % right when valid); use it to choose between readers or frames.',
    '<b>Night and rain:</b> collect real rain photos; test night-specific augmentation for the detector (NightLPBD found rate 80 %).',
    '<b>Speed on a laptop CPU</b> (RQ3): YOLOv8n ONNX/OpenVINO + char-YOLO reader end to end; target 10 FPS.',
    '<b>Proposal text:</b> update the EasyOCR sentence in related work (section 16).',
], 'ol')}

<h1>22. GitHub and Drive</h1>
<p>{M('After Train/results.zip')} (368 MiB) is the whole Kaggle output; do not upload it. The upload folder is ready at
{M('After Train/github_upload/weights/paper06/MD.Rokib.Hasan.Oli/')} (same layout as papers #2, #8, #23 and #24):</p>
{table(['File', 'What'], [
    [M('README.md'), 'results, model cards, quick start (tested), checksums'],
    [M('best_det_v8n.pt'), '6.2 MB, YOLOv8n plate detector (best epoch 90)'],
    [M('best_det_v8s.pt'), '22.5 MB, YOLOv8s plate detector (best epoch 79); under GitHub\'s 25 MB web-upload limit'],
    [M('paper06_results.zip'), '2.4 MB, all results, logs, per-plate EasyOCR and whole-system predictions, code; no weights'],
    [M('results_vs_paper.md'), 'the kit\'s ours-vs-paper table'],
    [M('Paper06_Study_Report.pdf'), 'this report'],
], [3.0, 7.0])}
<p>Drive {M('498R / weights / paper06 /')}: {M('After Train/for_drive/paper06_outputs.zip')} (124 MiB: both models with best.pt,
last.pt and ONNX exports). Web upload on GitHub: open {M('weights')} → Add file → Upload files → drag the {M('paper06')} folder →
commit message "Add paper #6 results on dataset v1.1 (Rokib)".</p>
"""

S['c5'] = f"""
<h1>Appendix: where every number comes from</h1>
{table(['Number', 'Source'], [
    ['paper numbers', 'the paper PDF (Papers/6_1 .../06_2025_YOLOv8-Based LPR for Bangladeshi Vehicles.pdf), pages given in the text; figures in Figure 2 extracted from it'],
    ['v1.1 training, detector, EasyOCR, whole-system and speed results', M('After Train/results.zip') + ' → ' + M('runs/*/results.json') + ', ' + M('results.csv') + ', ' + M('ocr_preds_test*.csv') + ', ' + M('e2e_preds.csv') + '; ' + M('paper-6.log')],
    ['line analysis, error types, confidence, paired tests with papers #2, #8, #24 (their v1.1 runs), #3 numbers', M('report_src/analysis.py') + ' → ' + M('analysis.json')],
    ['per-source detector scores, same-plate detector tests, reader swap on 204 photos', M('report_src/local_check.py') + ' → ' + M('local_check.json') + ', ' + M('det_per_plate.csv') + ', ' + M('e2e_cross_preds.csv') + ' (RTX 5060 Ti, the run\'s best.pt, paper #2\'s uploaded weights, dataset v1.1)'],
    ['EasyOCR local re-run and cheap fixes', M('report_src/easyocr_check.py') + ' → ' + M('easyocr_check.json') + ', ' + M('easyocr_local_preds.csv') + ' (EasyOCR 1.7.2, bn)'],
    ['charts', M('report_src/charts.py')],
], [3.6, 6.4])}
<pre>@misc{{ahamed2025yolov8,
  title        = {{{{YOLOv8}}-Based License Plate Recognition for
                  {{Bangladeshi}} Vehicles}},
  author       = {{Ahamed, Istiak and Ismail, Gazi Mohammad}},
  year         = {{2025}},
  month        = jan,
  howpublished = {{Authorea preprint}},
  doi          = {{10.22541/au.173748276.69471512/v1}},
  note         = {{Not peer-reviewed}}
}}</pre>
"""

FLOWS = [['cover'], ['a1'], ['a2'], ['a3'], ['b1'], ['b1ok'], ['b2'], ['b3'], ['b3b'], ['b4'], ['b4b'], ['b5'], ['b6'],
         ['b7'], ['b8'], ['c1'], ['c2'], ['c3'], ['c4'], ['c5']]


def build():
    bangla_font()
    arch = fitz.Archive()
    arch.add(fitz.Archive(r'C:\Windows\Fonts'))
    arch.add(fitz.Archive(HERE))
    arch.add(fitz.Archive(FONT_DIR))
    PW, PH = 595, 842
    ML, MT, MB = 52, 50, 56
    raw = io.BytesIO()
    writer = fitz.DocumentWriter(raw)
    pages = 0
    for names in FLOWS:
        story = fitz.Story(html=''.join(S[n] for n in names), user_css=CSS, archive=arch)
        more = True
        while more:
            dev = writer.begin_page(fitz.Rect(0, 0, PW, PH))
            more, _ = story.place(fitz.Rect(ML, MT, PW - ML, PH - MB))
            story.draw(dev)
            writer.end_page()
            pages += 1
            if pages > 60:
                raise RuntimeError('layout runaway')
    writer.close()
    doc = fitz.open('pdf', raw.getvalue())
    n = doc.page_count
    for i, pg in enumerate(doc):
        pg.insert_text((ML, PH - 26), 'Paper #6 study report · dataset v1.1 run · 498R BanglaLPR', fontsize=7.5,
                       fontname='helv', color=(0.45, 0.45, 0.43))
        t = f'{i + 1} / {n}'
        pg.insert_text((PW - ML - fitz.get_text_length(t, 'helv', 7.5), PH - 26), t, fontsize=7.5, fontname='helv', color=(0.45, 0.45, 0.43))
    doc.set_metadata({'title': 'Paper #6 study report (dataset v1.1): YOLOv8-based LPR for Bangladeshi vehicles',
                      'author': 'Prepared for MD. Rokib Hasan Oli',
                      'subject': 'Paper study, v1.1 Kaggle results, detector and EasyOCR analysis, reader swap, faculty guide'})
    doc.save(OUT, garbage=3, deflate=True)
    doc.close()
    print('pages', n, OUT)


if __name__ == '__main__':
    build()
