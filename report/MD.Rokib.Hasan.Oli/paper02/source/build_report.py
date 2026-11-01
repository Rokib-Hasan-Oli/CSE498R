"""Paper #2 study report (update for the dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (PLOS One 21(8): e0355858, CC BY 4.0, figures reused with credit); the v1.1 Kaggle run
(`After train/results.zip`, `paper-2.log`, 21 Oct 2026) via analysis.json; the v1.0 run (19 Oct log, quoted in the
first report, kept as Paper02_Study_Report_v1.0_run.pdf); the local per-source check (persource_det.json,
no_ridoykj.json, fig_boxstyle.png); the local preview in the paper notes (13 Oct 2026); dataset/docs; the proposal.
Run order: analysis.py -> persource_det.py, ridoykj_figure.py (local GPU) -> charts.py -> build_report.py.
"""
import io
import json
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HERE), 'Paper02_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
PS = json.load(open(os.path.join(HERE, 'persource_det.json')))
NR = json.load(open(os.path.join(HERE, 'no_ridoykj.json')))
R = A['results']


def ex(run, key, s='all'):
    d = A['e2e'][f'{run}|{key}'][s]
    return round(100 * d['exact'] / d['photos'], 2)


# the text below quotes these numbers; stop if the data ever disagrees with it
assert ex('cascade_ours', 'with_vehicle_stage') == 85.29 and ex('cascade_ours', 'plate_detector_only') == 86.76
assert ex('cascade_blpd_weights', 'with_vehicle_stage', 'blpd_ext') == 95.14
assert ex('cascade_ours', 'with_vehicle_stage', 'nightlpbd') == 66.67 and ex('cascade_blpd_weights', 'with_vehicle_stage', 'nightlpbd') == 38.33
assert A['ours_vs_authors']['with_vehicle_stage|all']['mcnemar_p'] == 0.02006
assert A['ours_vs_authors']['with_vehicle_stage|blpd_ext']['mcnemar_p'] == 0.58105
assert A['vehicle_stage']['cascade_ours']['mcnemar_p'] == 0.5488
assert R['lpnr_v12n']['ocr_test']['rows_decoder']['plate_exact_%'] == 87.71 and A['ocr']['xsort_exact'] == 1
assert A['ocr']['invalid'] == 52 and A['ocr']['invalid_and_wrong'] == 52
assert PS['ridoykj']['mAP50-95'] == 24.2 and NR['without_ridoykj']['mAP50-95'] == 64.3 and NR['all_local']['mAP50-95'] == 46.2
assert R['lpd_v12s']['speed']['tensorrt_fp16']['fps'] == 131.6 and R['lpd_v12s']['speed']['fp16']['ms_per_image'] == 140.93
assert A['env']['dataset_version'] == '1.1'

CSS = """
@font-face {font-family: B; src: url(arial.ttf);}
@font-face {font-family: B; src: url(arialbd.ttf); font-weight: bold;}
@font-face {font-family: B; src: url(ariali.ttf); font-style: italic;}
@font-face {font-family: B; src: url(arialbi.ttf); font-weight: bold; font-style: italic;}
@font-face {font-family: H; src: url(arialbd.ttf); font-weight: bold;}
@font-face {font-family: H; src: url(arial.ttf);}
@font-face {font-family: M; src: url(consola.ttf);}
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
ADD = M('python add_results.py paper02_outputs.zip --member "Rokib Hasan Oli"')
CREDIT = 'Figure from Sarker &amp; Alam (2026), PLOS One, CC BY 4.0.'
S = {}  # flow name -> HTML; each flow starts on a new page

# ======================================================================================== cover + one-page summary
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER #2 STUDY REPORT · UPDATE FOR DATASET v1.1</p>
<h1 style="font-size:19pt; margin-bottom:3pt">Cascaded YOLO-based approach for Bangladeshi license plate
identification and number recognition</h1>
<p style="font-family:H; font-size:12pt; color:#1c5cab; margin-bottom:8pt">The paper, our re-run on dataset v1.1, how it
fits our research, and what to show and say to the faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 22 October 2026 ·
replaces the first report of 19 Oct (kept as {M('Paper02_Study_Report_v1.0_run.pdf')})<br>
Paper: P. Sarker and S. M. Shamsul Alam (Khulna University), <i>PLOS One</i> 21(8): e0355858, 10 Aug 2026,
DOI 10.1371/journal.pone.0355858, 42 pages, open access (CC BY 4.0). Page numbers are journal pages.<br>
Code and data: github.com/prosensarker4/BLPD (weights, data, a validation notebook; no training script).<br>
Our run: {M('paper02_kaggle.ipynb')} on Tesla T4 x2, 21 Oct 2026, dataset <b>v1.1</b> (confirmed in
{M('environment.json')}), 3.2 h; full output in {M('After train/results.zip')}.</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> Three YOLOv12 models run one after another: a COCO model finds the vehicle, a
plate model finds the plate inside it, and a 104-class model finds every digit, letter, "Metro" and whole district
name inside the plate. Ordering those labels gives the plate text with no OCR engine. On 150 external photos it reads
95.33 % of plates exactly, against 15.33 % for an OCR pipeline (p.34, p.38).</p>
</div>
<div class="good">
<p><b>Is the v1.1 training OK? Yes.</b> All 5 jobs ended with {M('code=0')}, every epoch ran (60/60 and 70/70), no NaN,
0 corrupt images, dataset version 1.1 confirmed. Two checks show the pipeline is repeatable: the plate detector
(whose data did not change in v1.1) gives the same test scores as on 19 Oct, and the authors' weights give the same
result to the last photo (160 of 204).</p>
<p><b>What v1.1 changed.</b> v1.1 cut 91 BLPD "plate crops" that were really whole vehicle photos down to the plate. Retrained
on the fixed crops, our character reader reads <b>87.7 %</b> of the 651 test plates exactly (v1.0: 83.6 %). On the
<i>unchanged</i> 204 end-to-end photos the cascade improves from 83.3 to <b>85.3 %</b> (and from 82.8 to <b>86.8 %</b>
without the vehicle stage).</p>
</div>
<div class="box">
<p><b>Four findings.</b> (1) <b>The paper reproduces:</b> its own weights read 95.14 % of its own photos in our run
(paper 95.33 %). (2) <b>Same photos, two models:</b> in daylight ours and theirs are statistically tied (93.1 vs 95.1 %,
McNemar p = 0.58); at night ours reads 66.7 % vs their 38.3 % (p &lt; 0.001); over all 204 photos ours is better (85.3 vs
78.4 %, p = 0.02). (3) <b>The vehicle stage is not needed:</b> without it accuracy does not drop (86.8 vs 85.3 %, p = 0.55)
and the pipeline is 2.6× faster (36 vs 93 ms per photo on a T4). (4) <b>Both open questions of the first report are
solved:</b> "7.1 FPS" is an averaging artefact (the detector runs at 131.6 FPS with TensorRT), and the low box score
mAP50-95 = 46.2 comes from one source, ridoykj, whose boxes are drawn tighter (64.3 without it; paper: 68.5).</p>
<p><b>What to tell the faculty in one sentence.</b> "We reproduced the best published Bangla system on its own photos,
matched it in daylight, beat it clearly at night, made it 2.6 times faster by removing a stage, and showed that
fixing 91 bad dataset images alone added 2-4 points."</p>
</div>
<h2>How this report is organised</h2>
{table(['Part', 'Sections', 'Read it when'], [
    ['A. Understand the paper', '1-7: problem, terms, data, method, results, critique, what breaks on our plates', 'before you explain the paper'],
    ['B. Our v1.1 experiment', '8-14: what we ran, is it OK, results vs paper, v1.0 vs v1.1, six findings, targets', 'before you quote any number'],
    ['C. Use it', '15-21: research use, benefits, faculty slides, script, Q&amp;A, next steps, GitHub', 'before the faculty meeting'],
], [2.2, 4.8, 3.0])}
"""

# ======================================================================================== part A (the paper)
S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What problem does the paper solve?</h1>
<p>Bangladesh had 5,759,353 registered motor vehicles by 30 June 2023, 72 % of them motorcycles (BRTA, p.2).
Traffic police, toll gates and parking need to read plates automatically. A Bangladeshi plate has four required parts:
a <b>city (district) code</b> written as a Bangla word, a <b>vehicle-type letter</b>, a <b>two-digit class number</b> and a
<b>four-digit registration number</b>. Plates of metropolitan areas add the word <b>"Metro"</b> after the city (p.2, Fig. 2).</p>
{fig('pfig_plate_format.png', 300, 'Figure 1. Parts of a standard Bangladeshi plate: city code, "Metro", vehicle type, class number and unique number (paper Fig. 2, p.3). ' + CREDIT)}
<p>Earlier Bangla systems had two problems (p.4-5, Table 1):</p>
{ul([
    '<b>Too few classes.</b> Character recognizers used 14 to 60 classes. Real plates need more than 100 (10 digits, about 30 letters, "Metro" and 64 district names).',
    '<b>General OCR engines fail on Bangla.</b> Tesseract and EasyOCR misread conjunct consonants (<i>jukto borno</i>) and the two-line layout. Reported OCR accuracy was 78 % (EasyOCR) and 22-73 % (Tesseract).',
])}
<p>The paper's answer: treat every piece of the plate, even a whole district name, as an <b>object</b> that a YOLO
detector can find. Then no OCR engine is needed (p.5).</p>

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['Cascade', 'models run one after another; each one only looks inside the crop given by the one before'],
    ['LPD / LPNR', 'License Plate Detection model (finds the plate) / License Plate Number Recognition model (finds the characters)'],
    ['OCR-free', 'the text comes from a detector\'s class labels, not from an OCR engine such as Tesseract or EasyOCR'],
    ['Class / token', 'one thing the detector can name: a digit, a letter, "Metro" or a district word such as "Dhaka"'],
    ['Precision / Recall', 'of all boxes the model drew, how many were right / of all real objects, how many it found'],
    ['mAP50', 'mean average precision when a box counts as correct if it overlaps the true box by at least 50 % (IoU 0.5)'],
    ['mAP50-95', 'the same, averaged over stricter overlaps 0.50, 0.55 ... 0.95; it rewards boxes drawn <i>exactly like the labels</i>'],
    ['Full-number exact match', 'the whole plate text is right; one wrong character = wrong plate'],
    ['Character-wise accuracy', 'share of characters that are right'],
    ['Levenshtein similarity', '1 - (edits needed to turn the prediction into the truth / length of the truth); a nearly right plate scores near 100 %'],
    ['McNemar test', 'compares two models on the same photos using only the photos where they disagree; p &lt; 0.05 = a real difference, not luck'],
    ['External validation', 'a test on photos that were never used for training or tuning'],
    ['Latency / FPS', 'time to process one photo (ms) / photos per second = 1000 / latency'],
    ['Near-duplicate', 'a different file that shows the same photo or plate (Roboflow copy, re-post, next video frame); see paper #23'],
], [2.2, 7.8])}
"""

S['a2'] = f"""
<h1>3. The data</h1>
{table(['Set', 'Images', 'Train / val / test', 'Classes', 'Boxes', 'Page'], [
    ['Dataset-01 (plates, for LPD)', '2,140', '1,502 / 425 / 213', '1', '2,251 (Fig. 3 shows 1,561)', 'p.6-7, Table 2'],
    ['Dataset-02 (plate crops, for LPNR)', '7,217', '5,058 / 1,438 / 721', '104', '63,879', 'p.6-10, Tables 2-3'],
    ['External validation', '150', 'test only', '-', '-', 'p.32'],
], [2.6, 0.9, 1.7, 0.8, 1.9, 1.4], num=(1,))}
{ul([
    '<b>Sources:</b> their own phone photos in Khulna (Gollamari, Sonadanga, Zero Point), all in <b>daylight and sunny weather</b>, plus web crawling and Roboflow (p.6).',
    '<b>Split:</b> 70 / 20 / 10, done by image. No de-duplication step is described.',
    '<b>Strong imbalance:</b> digit "1" has 7,396 boxes, "Metro" 6,292, "Dhaka" 4,947, but Munshiganj has 2, Nowabganj 11, Bagerhat 37, Joypurhat and Kustia 45 (p.9).',
    '<b>Class list:</b> some "different" classes are the same token written two ways: "Cox-s Bazar" (47) and "Cox-s-Bazar" (58), "Da" (402) and "DA" (141), "Tha" (405) and "THA" (179) (Table 3, p.10). So 104 classes are really about 101 tokens. Our cleaned vocabulary has 101.',
])}

<h1>4. The method</h1>
{fig('fig_pipeline.png', 490, 'Figure 2 (drawn for this report). Top: the paper\'s cascade (Algorithm 1, p.15-16). Bottom: our planned pipeline from the proposal. Blue = idea we keep, orange = what we change or add.')}
{table(['Stage', 'What it does', 'Model and settings'], [
    ['1. Vehicle', 'finds cars, buses, trucks and motorcycles; each vehicle box is cropped', 'Ultralytics YOLO12l pretrained on COCO, not retrained (26.4 M parameters, 88.9 GFLOPs) (p.15)'],
    ['2. Plate (LPD)', 'finds the plate inside the vehicle crop; the box is mapped back to the photo', 'YOLOv12 trained on Dataset-01, 100 epochs (p.15)'],
    ['3. Characters (LPNR)', 'finds digits, letters, "Metro" and district words in the plate crop', 'YOLOv12 trained on Dataset-02, 70 epochs (p.16)'],
    ['4. Decoding', 'Algorithm 1: "sort C by left-to-right order" and join (p.16). The text below it: labels are "organized as City-Metro (if available)-Character-Number", e.g. "Dhaka Metro Ga 328301" (p.16)', 'simple rule, no language model'],
], [1.6, 4.6, 3.8])}
<p><b>Training settings (Table 5, p.18):</b> 640×640 input, batch 16, optimizer "auto", lr0 0.01, weight decay
0.0005, NMS IoU 0.7, data augmentation "No", no cosine schedule, deterministic, no early stopping. Training ran on a
Colab Tesla T4: LPD 2.42 h (87.3 s per epoch), LPNR 4.35 h (223.6 s per epoch) (p.17-18, p.24). The model size
(n, s, m...) is not stated; the repo's files suggest a medium LPD (40.8 MB) and a nano LPNR (5.7 MB).</p>
<p><b>Why YOLOv12?</b> It adds attention ("area attention" and R-ELAN blocks) to YOLO and was the most accurate YOLO
for its speed when the work was done (p.13). On their character data it beat YOLOv8, v9, v10 and v11 (Table 14, p.37).</p>
"""

S['a3'] = f"""
<h1>5. Every result in the paper</h1>
{table(['Result', 'Value', 'Page'], [
    ['LPD validation (425 images)', 'P 0.993, R 0.970, mAP50 0.974, mAP50-95 0.674', 'p.21, Table 6'],
    ['<b>LPD test</b> (213 images, 226 plates)', '<b>P 0.986, R 0.965, mAP50 0.963, mAP50-95 0.685</b>; 219 found, 7 missed, 6 false', 'p.22-23, Table 7'],
    ['LPNR validation (1,438 images)', 'P 0.952, R 0.905, mAP50 0.950, mAP50-95 0.770', 'p.24, Table 8'],
    ['<b>LPNR test</b> (721 images, 6,292 boxes)', '<b>P 0.950, R 0.898, mAP50 0.949, mAP50-95 0.767</b>; digits mAP50 ≥ 0.985; weakest "Sa" 0.346, Joypurhat 0.42, "Pa" 0.60 (mAP50-95)', 'p.25-31, Table 9'],
    ['YOLO versions on Dataset-02 (test mAP50)', 'v8 0.936, v9 0.938, v10 0.897, v11 0.931, <b>v12 0.949</b>', 'p.37, Table 14'],
    ['<b>External 150 photos</b>', '<b>plate found 100 %, full number exact 95.33 % (143/150), characters 98.82 %, Levenshtein 99.57 %</b>', 'p.34, Table 10'],
    ['OCR pipeline on the same 150', '23 correct = 15.33 % (counted "ignoring minor mismatches")', 'p.37-38, Table 15'],
    ['Laptop CPU (i5-8265U), per photo', 'mean 2,311.53 ms (vehicle 911 + plate 1,091 + characters 233), max 6,438 ms', 'p.35, Table 11'],
    ['Desktop GPU (RTX 3050), per photo', 'mean 117.16 ms (39.6 + 45.2 + 19.5) ≈ 8.5 FPS, max 284.31 ms', 'p.35, Table 12'],
    ['Jetson Orin Nano, CPU only, per photo', 'mean 10.37 s, max 25.44 s', 'p.36, Table 13'],
], [2.8, 5.6, 1.6])}
{fig('pfig_failures.png', 420, 'Figure 3. The 7 external photos the cascade read wrongly; the paper calls every error partial, and the plate was found in all of them (paper Fig. 30, p.34). ' + CREDIT)}

<h1>6. Critical review</h1>
<h3>Strengths</h3>
{ul([
    '<b>Peer reviewed and recent</b> (PLOS One, Aug 2026), with data, weights and a validation notebook public. That is rare for Bangla LPR.',
    '<b>Good idea:</b> word-level classes for district names avoid the hardest OCR problem (conjunct letters), and 104 classes cover nearly every plate.',
    '<b>Several end-to-end metrics</b> (exact, character-wise, Levenshtein) and <b>per-stage latency on three devices</b>, with mean, std, min and max.',
    '<b>Honest about limits:</b> the conclusion names low light, weather, blur, occlusion, imbalance and lightweight models as future work (p.41).',
    '<b>Its numbers reproduce on its own data:</b> Table 10 in our Kaggle run, Tables 6, 7 and 9 in a local retraining (section 10).',
])}
"""

S['a3b'] = f"""
<h3>Weaknesses (with page numbers, so you can show them)</h3>
{table(['#', 'Problem', 'Why it matters'], [
    ['1', '<b>Test sets leak.</b> Split by image from Roboflow/web sources with no de-duplication. Our audit: 94 of the 213 LPD test photos (44.1 %) and 270 of the 721 LPNR test crops (37.4 %) have a near-duplicate in their own train set; 345 test crops show a plate whose text is also in train.', 'Tables 7 and 9 are partly measured on seen images, so they are optimistic for new photos (paper #23).'],
    ['2', '<b>The datasets are smaller than they look.</b> In our cleaning, only 185 of the 7,217 character crops survived: 4,894 were duplicates, 2,019 had masked digits, mosaics or incomplete labels, 119 were synthetic templates. Of 2,140 plate photos, 1,001 were duplicates (Figure 4).', '"9,357 images" overstates how much distinct data the models saw.'],
    ['3', '<b>Max and mean are mixed.</b> The abstract and Table 17 give 284.31 ms (GPU) and 25.439 s (Jetson), which are the <i>maximum</i> times, next to the <i>mean</i> CPU time 2,311.53 ms (p.1, p.39).', 'The GPU mean is 117.16 ms; readers can mis-compare.'],
    ['4', '<b>"Edge deployment on Jetson"</b> is claimed as a contribution (p.6), but the Jetson ran on its CPU only, at 10.4 s per photo (p.17, p.36).', 'Not real time; the Jetson GPU was never used.'],
    ['5', '<b>Daylight only.</b> Own photos were all taken in daylight and sunny weather (p.6).', 'Our run: 38 % exact at night with their weights.'],
    ['6', '<b>Decoding is described two ways.</b> Algorithm 1 sorts all boxes left to right (p.16); done literally on two-line plates this reads <b>1 of 651</b> of our test plates. The text orders labels by type (city, Metro, letter, number), which works.', 'Coding Algorithm 1 as written gives almost 0 %.'],
    ['7', '<b>Small, self-chosen external set</b> (150 photos, "online and offline sources", p.32); 6 of the 150 are near-duplicates of other photos in our check.', 'One external set cannot show generalisation.'],
    ['8', '<b>OCR baseline not described:</b> the OCR engine and its settings are not named (p.37).', 'The 15.33 % cannot be reproduced.'],
    ['9', '<b>Text errors:</b> section 2.2 describes the original 2016 YOLOv1 (S×S grid, 448 px, fully connected layers), not YOLOv12 (p.10-13); "accuracy" uses true negatives, which do not exist in object detection (p.16); 2,251 vs 1,561 boxes (p.6 vs Fig. 3); duplicate class names (Table 3).', 'Shows the review was light; do not copy these parts.'],
    ['10', '<b>Unequal YOLO comparison:</b> Table 14 does not give the model size or settings of each version, and each was run once.', 'The v12 win (0.949 vs 0.936) may be noise or size.'],
], [0.3, 5.6, 3.6])}
"""

S['a4'] = f"""
{fig('fig_cleaning.png', 400, 'Figure 4. What our de-duplication and label check did to BLPD\'s published images (dataset/docs/dedupe_report.md). "Duplicate" = a Roboflow copy, a re-post, or the same photo in another public dataset; one copy of each cluster was kept. BLPD\'s plate boxes are loose, so its kept photos are used only as training hints.')}

<h1>7. What would break on our plates (and what we do about it)</h1>
{table(['Risk', 'Why', 'Our answer'], [
    ['Rows mixed', 'a pure left-to-right sort mixes the two lines', 'two-row decoder: words on top, digits below, each sorted by x'],
    ['Rare districts missed', '2 to 50 training boxes for some districts', 'merged sources; synthetic rare plates (G6); district dictionary'],
    ['Night, rain, blur, fog', 'training photos were daylight only', 'night photos in train; per-condition tests; CLAHE study (RQ4)'],
    ['Vehicle not found', 'COCO has no CNG/rickshaw class; dark photos', 'skip the vehicle stage (finding 3)'],
    ['Too slow', 'three models, 8.5 FPS on an RTX 3050', 'two light models, TensorRT/OpenVINO, tracking (RQ3)'],
    ['Wrong but plausible text', 'no check that the result is a legal plate', 'BRTA format check; flag or re-read invalid text (O3)'],
], [1.9, 3.6, 4.5])}
"""

# ======================================================================================== part B (our v1.1 run)
o = R['lpnr_v12n']['ocr_test']
d = R['lpd_v12s']
S['b1'] = f"""
<p class="part">PART B · OUR EXPERIMENT ON DATASET v1.1</p>
<h1>8. What we ran, and what changed in v1.1</h1>
<p>The notebook {M('paper02_kaggle.ipynb')} rebuilds the paper's cascade on our <b>498R BanglaLPR All-in-One</b>
dataset (merged from every public Bangla source, de-duplicated, grouped 70/15/15 split, seed 42, val/test with human
labels only) and also runs the <b>authors' own weights</b>, so both can be compared on the same photos.</p>
<h3>Dataset v1.1 (20 Oct): the only change that matters here</h3>
<p>In v1.0, <b>91 BLPD rows</b> of the character set (train 63, val 9, test 19) were not plate crops but whole vehicle
photos with tiny character boxes. v1.1 cuts each one to its plate (from the union of its character boxes). Nothing
was dropped and the splits did not move. The plate-detection set and the 204 end-to-end photos did not change.</p>
{table(['Job', 'GPU', 'What it does', 'Time'], [
    [M('lpd_v12s'), '0', 'YOLO12s plate detector, 60 epochs, 640, batch 16, SGD; train 8,574 photos (4,067 human + 4,507 auto boxes), val 896, test 1,153 (1,011 plates)', '182.2 min'],
    [M('lpnr_v12n'), '1', 'YOLO12n character detector, 70 epochs, 640, batch 16, 101 Bangla tokens; train 5,465 crops, val 536, test 467 (boxes); plus 651 test crops with typed plate text', '95.5 min'],
    [M('cascade_ours'), '0', 'YOLO12l (COCO) → our LPD → our LPNR on the 204 end-to-end photos, with and without the vehicle stage', '0.8 min'],
    [M('cascade_blpd_weights'), '1', 'the same cascade with the authors\' LPD_Model.pt and LPNR_Model.pt; also their LPNR on our 651 text crops', '1.2 min'],
    [M('speed_lpd_v12s'), '0', 'batch-1 speed of the plate detector: PyTorch FP32/FP16, TensorRT FP16, CPU ONNX', '6.0 min'],
], [2.0, 0.5, 6.0, 1.1])}
<p><b>End-to-end photos (204):</b> 144 of the paper's own 150 external photos (6 near-duplicates removed in our build)
and 60 NightLPBD night photos, each with its plate text typed in.</p>
<p><b>Differences from the paper:</b> LPD 60 epochs instead of 100; SGD instead of "auto"; sizes s and n (the paper does
not state its sizes); 101 tokens with the duplicate classes merged; our two-row decoder; a different, harder test set.
Software: Ultralytics 8.4.170, PyTorch 2.11 + CUDA 12.8, Python 3.13 (Kaggle, 21 Oct 2026 17:03).</p>
"""

S['b1ok'] = f"""
<h1>9. Is the training OK?</h1>
{table(['Check', 'Result', 'OK?'], [
    ['Dataset version', M('dataset_version: 1.1') + ' in environment.json; folder BanglaLPR_AllInOne_v1_1; counts as expected', 'yes'],
    ['Jobs', 'all 5 ended with code=0; total 3.2 h of the 12 h Kaggle limit', 'yes'],
    ['Epochs', 'LPD 60 of 60 (best.pt = epoch 36), LPNR 70 of 70 (best.pt = epoch 55); under the 5 h / 4 h caps', 'yes'],
    ['Data loading', '0 corrupt images in every split; 160 plate-free test photos as expected', 'yes'],
    ['Curves', 'no NaN; both validation curves rise and then stay flat (Figure 5); LPD has a short dip at epoch 3 (warm-up) and recovers by epoch 6', 'yes'],
    ['Repeatable', 'LPD test scores identical to the v1.0 run (P 96.7, R 96.2, mAP50 98.9); authors\' cascade identical (160 of 204 photos)', 'yes'],
    ['GPU memory', 'peak 7.7 GB of 16 GB on the T4', 'yes'],
    ['Speed row "7.10 FPS"', 'an averaging artefact of the PyTorch FP16 loop, not the model (section 13, finding 6)', 'ignore it'],
    ['LPD mAP50-95 46.2', 'caused by one test source with a different box style (finding 4)', 'explained'],
], [2.0, 6.8, 1.2])}
{fig('fig_curves.png', 470, 'Figure 5. Validation mAP per epoch. Both models converge; the dashed line marks the epoch Ultralytics kept as best.pt (best 0.1·mAP50 + 0.9·mAP50-95 on validation).')}
"""

S['b2'] = f"""
<h1>10. Results next to the paper</h1>
{table(['Metric', 'Paper (page)', 'Ours v1.0', '<b>Ours v1.1</b>', 'Authors\' weights (our run)'], [
    ['LPD test precision / recall', '98.6 / 96.5 (p.23)', '96.7 / 96.2', '96.7 / 96.2', '-'],
    ['LPD test mAP50 / mAP50-95', '96.3 / 68.5 (p.23)', '98.9 / 46.2', '<b>98.9</b> / 46.2 (64.3 w/o ridoykj)', '-'],
    ['LPNR test boxes mAP50 / mAP50-95', '94.9 / 76.7 (p.25)', '92.6 / 63.9', '<b>97.1</b> / 66.6', '-'],
    ['LPNR on our 651 text crops: plate exact', '-', '83.56', '<b>87.71</b>', '74.35'],
    ['Cascade: plate found in photo', '100 (p.34)', '93.14', '93.14', '88.24'],
    ['Cascade: whole plate exact (204 photos)', '95.33 (p.34, 150 photos)', '83.33', '<b>85.29</b>', '78.43'],
    ['Cascade: characters correct', '98.82 (p.34)', '92.59', '92.78', '88.18'],
    ['Cascade: Levenshtein similarity', '99.57 (p.34)', '90.57', '90.84', '84.65'],
    ['Cascade without vehicle stage: exact', '-', '82.84', '<b>86.76</b>', '77.94'],
    ['Time per photo, with / without vehicle stage', '117.16 ms, RTX 3050 (p.35)', '85.8 / 31.5 ms', '93.3 / 36.2 ms', '96.4 / 38.9 ms'],
    ['Training time LPD / LPNR', '145 / 261 min (p.17-24)', '183 / 93 min', '178 / 94 min', '-'],
], [3.2, 2.2, 1.5, 2.1, 1.5])}
<div class="warn"><p><b>Read this table carefully.</b> The paper's column is on the paper's data; ours is on the 498R
test split, which is larger, de-duplicated and harder. The true head-to-head is "ours" vs "authors' weights" on the
<i>same</i> 204 photos (finding 2). Times are on one Kaggle T4 (PyTorch, no TensorRT) and vary by about ±10 % between
runs.</p></div>

<h1>11. What the v1.1 fix changed</h1>
{fig('fig_v10_v11.png', 480, 'Figure 6. Our weights, v1.0 run vs v1.1 run. Only the character model changed between the runs (the plate detector and the 204 end-to-end photos are the same), so the gain on the four right-hand pairs comes from the fixed crops alone (63 training + 9 validation).')}
{ul([
    '<b>On unchanged photos the reader got better:</b> +4 photos with the vehicle stage (170 → 174 of 204) and +8 without it (169 → 177). Night: 63.3 → 70.0 %. BLPD external: 91.0 → 93.8 % (no vehicle stage).',
    '<b>On the 651 text crops</b> the gain is larger (83.6 → 87.7 %), but here 19 test crops were also fixed, so part of that gain is an easier, now-correct test.',
    '<b>Character box mAP50 92.6 → 97.1</b> is mostly the test fix: in v1.0 the 19 whole-photo test images had tiny character boxes that were almost impossible to hit.',
    '<b>We cannot test whether +4 / +8 photos is significant:</b> the v1.0 per-photo predictions were never downloaded. The direction is the same in every subset.',
])}
"""

S['b3'] = f"""
<h1>12. Six findings</h1>
<h2>Finding 1: the paper reproduces on its own data</h2>
{table(['On BLPD\'s own data', 'Paper (page)', 'Our check', 'Where'], [
    ['Authors\' weights, their external photos: whole plate exact', '95.33 (p.34)', '<b>95.14</b> (137 of 144)', 'Kaggle, 19 and 21 Oct (same result)'],
    ['Authors\' weights, their external photos: characters correct', '98.82 (p.34)', '98.93', 'Kaggle'],
    ['Retrained with paper settings on BLPD\'s split: LPD test mAP50 / mAP50-95', '96.3 / 68.5 (p.23)', '97.2 / 68.1', 'local preview, 13 Oct (RTX 5060 Ti)'],
    ['Retrained: LPNR test mAP50 / mAP50-95', '94.9 / 76.7 (p.25)', '96.7 / 80.0', 'local preview'],
    ['Retrained cascade, 150 external photos: exact / characters', '95.33 / 98.82 (p.34)', '90.67 / 98.05', 'local preview'],
], [4.6, 1.9, 1.8, 2.1])}
<p>The printed numbers are real on the paper's own data. This also proves our cascade code and scoring: given the
paper's models, they give the paper's number. The official Kaggle version of the retraining check
({M('paper02_original_check.ipynb')}, about 7.5 h) is still to be run.</p>

<h2>Finding 2: same photos, two models: a tie in daylight, a clear win at night</h2>
{fig('fig_e2e.png', 490, 'Figure 7. The paper\'s pipeline (with the vehicle stage), same 204 photos, same code, two sets of weights.')}
{table(['Same photos (paper\'s pipeline)', 'Only ours right', 'Only authors\' right', 'Both / neither', 'McNemar p', 'Verdict'], [
    ['All 204 photos', '23', '9', '151 / 21', '0.020', '<b>ours better</b>'],
    ['BLPD external (144, their photos)', '5', '8', '129 / 2', '0.58', 'tie'],
    ['NightLPBD (60)', '18', '1', '22 / 19', '0.00008', '<b>ours much better</b>'],
    ['BLPD external, no vehicle stage', '6', '6', '129 / 3', '1.0', 'exact tie (135 = 135)'],
], [3.2, 1.2, 1.4, 1.3, 1.2, 1.7])}
<p>On the authors' own daylight photos the 2-point gap (95.1 vs 93.1 %) is 3 photos and is not significant; without
the vehicle stage both read exactly 135 of 144. At night their model reads 23 of 60 and ours 40 of 60 (characters
58.1 vs 77.1 %). Most of their night failures are photos with no reading at all: 27 of 60 (ours: 14).</p>
"""

va = A['vehicle_stage']['cascade_ours']
S['b4'] = f"""
<h2 style="margin-top:2pt">Finding 3: the vehicle stage costs time and adds nothing</h2>
{fig('fig_speed.png', 420, 'Figure 8. Mean time per photo. The paper\'s number is on an RTX 3050; ours on one Kaggle T4 (PyTorch, no TensorRT). Removing the COCO vehicle model cuts the time by about 2.6×.')}
{table(['Our weights, 204 photos', 'Exact', 'Characters', 'Plate found', 'ms per photo (mean / max)'], [
    ['vehicle → plate → characters (paper)', '85.29 % (174)', '92.78 %', '93.14 %', '93.3 / 330.0'],
    ['plate → characters (no vehicle stage)', '<b>86.76 % (177)</b>', '<b>94.89 %</b>', '<b>95.59 %</b>', '<b>36.2 / 110.8</b>'],
], [3.6, 1.6, 1.3, 1.3, 2.2])}
<p>Without the vehicle stage {va['right_only_without_vehicle']} photos become right and {va['right_only_with_vehicle']}
become wrong (McNemar p = {va['mcnemar_p']:.2f}, no real difference); 6 of the 7 photos it rescues are night photos,
where the COCO model often misses the dark vehicle, so the plate is never searched. The vehicle model is the largest of the
three (YOLO12l) and takes about a third of the paper's GPU time (Table 12). Dropping it is the cheapest way to our
≥ 10 FPS goal: about 28 photos per second on a T4 already.</p>

<h2>Finding 4: the low box score (mAP50-95 46.2) is one dataset's box style</h2>
{fig('fig_persource.png', 410, 'Figure 9. Our plate detector scored per test source (local re-scoring of the Kaggle best.pt; the full test set gives the same 98.9 / 46.2 as on Kaggle).')}
{fig('fig_boxstyle.png', 440, 'Figure 10. Green = label box, orange = our detector\'s box. Top: ridoykj test photos: labels hug the inner plate; our boxes include the plate rim. Bottom: KUET: both agree.')}
<p>Our detector finds ridoykj plates perfectly (mAP50 99.5), but its boxes are 19 % wider and 36 % taller than
ridoykj's labels (median), so at strict overlaps they "miss" (mean IoU 62 %). ridoykj is 274 of the 1,153 test photos
(24 %), which pulls the whole test set down to 46.2. Without it, mAP50-95 is <b>64.3</b>, close to the paper's 68.5;
KUET alone is 73.2. This also explains why validation (60.4) is higher than test: ridoykj is a smaller part of
validation. It does not hurt reading: the cascade pads every plate crop by 8 %.</p>
"""

S['b5'] = f"""
<h2 style="margin-top:2pt">Finding 5: the decoder and a format check matter as much as the detector</h2>
{table(['Our reader on the 651 test crops', 'Plates', 'Exact'], [
    ['Algorithm 1 taken literally: all boxes sorted left to right', '651', '<b>1 (0.15 %)</b>'],
    ['Our two-row decoder (words above, digits below), as described in the paper\'s text', '651', '<b>571 (87.71 %)</b>'],
    ['... readings that pass the BRTA format check (district, [Metro], letter, 6 digits)', '599', '571 right, 28 wrong'],
    ['... readings that fail the format check', '52', '<b>all 52 wrong</b>'],
], [6.4, 1.2, 2.4])}
{ul([
    '<b>The format check is a free error detector:</b> it flags 52 of the 80 wrong readings (65 %) and never flags a right one. Flagged plates can be re-read by a second model; in paper #24 this "format fallback" added +1.84 points with no new error (p = 0.0005).',
    '<b>What goes wrong:</b> of the 80 errors, 29 are in the digits, 25 in the letter, 15 in the district word and 11 in several parts. Only 3 wrong readings have mean confidence ≥ 0.9, so confidence is also useful for flagging.',
    '<b>By source:</b> LPDB-A 96.0 %, ANPR 84.8 %, KUET 84.7 %, NightLPBD 80.8 %, BLPD 80.0 % (30 crops). <b>By condition</b> (Figure 11): clear 90.4 %, night 83.1 %, blurry 79.5 %, fog 76.9 %, synthetic rain 74.5 %.',
    'Validation crops (766) score lower, 80.6 %, so do not tune thresholds on test.',
])}
{fig('fig_conditions.png', 440, 'Figure 11. Per-condition test results (fixed team subsets; rain and detection-fog images are synthetic copies).')}

<h2>Finding 6: real speed numbers, and why "7.10 FPS" is wrong</h2>
{table(['Plate detector alone (YOLO12s, 640 px, batch 1)', 'ms per photo', 'FPS'], [
    ['Kaggle T4, PyTorch FP32', '18.9', '52.9'],
    ['Kaggle T4, <b>TensorRT FP16</b>', '<b>7.6</b>', '<b>131.6</b>'],
    ['Kaggle T4, PyTorch FP16 (the kit\'s table row)', '140.9 mean, but 17.9 for a single photo', '7.1 (artefact)'],
    ['Kaggle CPU, ONNX FP32', '166.8', '6.0'],
], [5.2, 3.0, 1.8])}
<p>The FP16 row averages 200 photos, and a few very slow calls inflate the mean: one photo takes 17.9 ms
(1.8 + 15.0 + 1.1 ms), nowhere near 141 ms. Quote TensorRT or FP32. The CPU number matters
for our laptop goal: YOLO12s at 640 px gives only 6 FPS on Kaggle's CPU, so the laptop version needs a nano model,
a smaller input or OpenVINO. The same FP16 row is used in 7 of the kit's notebooks, so teammates' "FPS" rows can have the
same problem.</p>

<h1>13. Ours vs the proposal's targets</h1>
{table(['Proposal target', 'Our result now (v1.1)', 'Status'], [
    ['Detection mAP50 ≥ 95 %', '98.9 % on the 1,153 test photos', 'met'],
    ['Character accuracy ≥ 90 %', '94.9 % on the 204 end-to-end photos without the vehicle stage (98.5 % daylight, 84.8 % night); 97.9 % on the 651 crops', 'met (night 84.8 %)'],
    ['≥ 10 FPS full pipeline on standard hardware', '27.6 photos/s on a T4 without the vehicle stage; detector alone 6 FPS on the Kaggle CPU', 'GPU yes, CPU not yet'],
    ['Duplicate-free vs standard split (RQ1)', 'BLPD audit: 44.1 % / 37.4 % of test images duplicated in train', 'evidence ready'],
], [3.0, 5.0, 2.0])}
"""

# ======================================================================================== part C
S['c1'] = f"""
<p class="part">PART C · USE IT</p>
<h1>14. How to use this paper in our research</h1>
{table(['Our research part', 'How paper #2 is used', 'Evidence we now have (v1.1)'], [
    ['Stage 2: character-level recognizer (O3)', 'design source: one YOLO detects digits, letters, "Metro" and whole district words; no OCR engine', '87.7 % of plates and 97.9 % of characters right on our 651 test crops'],
    ['Baseline for every table', 'the authors\' public weights, run on our photos with our code', 'same photos: tie in daylight (p 0.58), 66.7 vs 38.3 % at night (p &lt; 0.001)'],
    ['RQ1: duplicate-free split', 'BLPD is a second leaky dataset next to KUET (paper #23)', '44.1 % of LPD test and 37.4 % of LPNR test duplicated in train'],
    ['RQ2: ≥ 90 % characters with correction', 'the literal x-sort is the "no decoder" baseline; the format check is the first correction step', 'x-sort 0.15 % vs our decoder 87.7 %; format check flags 52 errors, 0 false alarms'],
    ['RQ3 / O4: real time', 'their 3-model cascade is the slow reference (8.5 FPS RTX 3050, 2.3 s laptop CPU)', 'no vehicle stage: 2.6× faster, same accuracy; TensorRT detector 131.6 FPS'],
    ['RQ4: robustness', 'their daylight-only model shows what happens without diverse data', 'night +28 points with our merged data; per-condition table'],
    ['O1: the benchmark itself', 'the v1.1 fix is a measured data-quality effect', '91 fixed crops → +4 / +8 photos on unchanged test photos'],
    ['O2: detector choice', 'their YOLO v8-v12 table is on leaky data', 'redo it on our split with equal sizes and 3 seeds'],
    ['Thesis writing', 'Introduction, Related work (their Table 1 lists 14 Bangla systems), Method, Experiments', 'BibTeX at the end'],
], [2.6, 4.0, 3.4])}

<h1>15. What benefit do you get from it?</h1>
{ul([
    '<b>A strong, citable baseline.</b> A 2026 peer-reviewed journal paper with public weights. Matching it in daylight and beating it at night, on the same photos and with a significance test, is a clear result examiners understand.',
    '<b>Proof that our evaluation code is right and repeatable.</b> It gives the paper\'s own number with the paper\'s weights (95.14 vs 95.33), and the same results in two runs two days apart.',
    '<b>A measured data-quality story.</b> Fixing 91 bad crops raised accuracy on unchanged photos. That supports the whole dataset effort (O1) with a number, not an opinion.',
    '<b>Design decisions with evidence:</b> OCR-free recognition (paper: 95.33 vs 15.33), no vehicle stage (finding 3), a real two-row decoder (finding 5), a format check (finding 5), a clean class list.',
    '<b>Honest metrics you can defend:</b> you can now explain every odd number (mAP50-95 46.2, 7.1 FPS) instead of hiding it.',
    '<b>The authors\' future work is our project:</b> low light, weather, blur, occlusion, imbalance and lightweight models (p.41).',
    '<b>Reusable assets:</b> their district/letter class scheme (mapped to our 101 tokens), their metrics, and their per-stage latency reporting.',
])}
<h3>A paragraph you can adapt for the related-work chapter</h3>
<pre>Sarker and Alam [ref] cascade three YOLOv12 models (a COCO vehicle detector, a plate
detector and a 104-class character detector that treats district names as single
objects) and report 95.33% full-plate accuracy on 150 external images, against 15.33%
for an OCR pipeline. Their data are daylight-only and split by image; in our audit
44.1% of their detection test images and 37.4% of their character test images have
a near-duplicate in training. Re-running their released weights we obtain 95.14% on
their external images but 38.3% on night images; a model trained on our de-duplicated
benchmark reaches 66.7% at night (McNemar p &lt; 0.001) and ties theirs in daylight.
Removing the vehicle stage does not change accuracy and cuts latency by 2.6x.</pre>
"""

S['c2'] = f"""
<h1>16. What to show the faculty (8 slides)</h1>
{table(['#', 'Slide title', 'What is on it', 'Source in this report'], [
    ['1', 'The problem', 'plate parts (city, Metro, letter, numbers); why OCR fails on Bangla; 14-60 classes in older work', 'Figure 1, section 1'],
    ['2', 'The paper\'s method', 'three YOLOv12 models in a cascade, 104 classes, ordering of labels', 'Figure 2 (top row)'],
    ['3', 'The paper\'s results', '95.33 % exact vs 15.33 % OCR; LPD mAP50 0.963, LPNR 0.949; 117 ms on RTX 3050, 2.3 s on CPU', 'section 5'],
    ['4', 'We reproduced it', 'authors\' weights on their photos 95.14 vs 95.33; retrained detectors match Tables 7 and 9', 'finding 1'],
    ['5', 'But: its data leaks', '44 % of test photos duplicated in train; 185 of 7,217 crops survive cleaning', 'Figure 4'],
    ['6', 'Same photos, two models', 'tie in daylight (p 0.58), 66.7 vs 38.3 % at night (p &lt; 0.001)', 'Figure 7 + table'],
    ['7', 'Faster and cleaner', 'no vehicle stage: 2.6× faster, same accuracy; v1.1 data fix: +4 / +8 photos', 'Figures 8 and 6'],
    ['8', 'How it shapes our system', 'our pipeline (Figure 2 bottom row), format check, next steps', 'Figure 2, section 20'],
], [0.3, 2.1, 5.0, 2.2])}
<div class="box"><p><b>Keep it honest:</b> say "on the same photos" for the comparison with the authors' weights and "on
our test split" for the rest. If you show mAP50-95, show Figure 9 with it. Never show the 7.1 FPS row.</p></div>

<h1>17. What to explain: a 3-minute script</h1>
<p>"This paper is the strongest recent work on Bangladeshi plates, a 2026 PLOS One article from Khulna University. They
chain three YOLOv12 models: one finds the vehicle, one the plate, and one finds every character and even whole district
names like 'Dhaka' as separate objects, so they need no OCR engine. They report 95 percent of plates read exactly
on 150 external photos, against 15 percent for OCR.</p>
<p>We ran their released weights on their own photos with our code and got 95.1 percent, so their result is real and
our code is correct. We also found that 44 percent of their plate test photos have a near-copy in their training set,
so their test scores are optimistic.</p>
<p>Then we trained the same design on our merged, de-duplicated dataset and compared both models on the same 204
photos. In daylight they tie: the difference is three photos and not significant. At night their model reads 38
percent of plates and ours 67 percent, which is highly significant. We also removed their first model, the vehicle
detector: accuracy did not drop and the pipeline became 2.6 times faster.</p>
<p>Our second dataset version fixed 91 images, 63 of them training images, that were whole car photos instead of plate
crops. Only that fix raised accuracy on the unchanged test photos by two to four points. And a simple plate-format check catches two
thirds of the reading errors without a single false alarm.</p>
<p>So we keep their best idea, the character-level YOLO with word classes, and change the rest: no vehicle stage, a
two-row decoder, a format check, tracking for video, and a fair benchmark. That is the plan in our proposal."</p>
"""

S['c3'] = f"""
<h1>18. Questions the faculty may ask</h1>
<p class="q">Q1. Why not just use their model?</p>
<p>It ties ours only in daylight; at night it reads 38 % (ours 67 %). It also uses a slow vehicle stage, a class list
with duplicate names, and was tested on a leaky split. We keep its idea, not its weights.</p>
<p class="q">Q2. Your daylight number is lower than theirs (93.1 vs 95.1). Is your model worse?</p>
<p>No. On those 144 photos the two models disagree on 13 photos (5 vs 8); McNemar p = 0.58, so this is chance. Without
the vehicle stage both read exactly 135. Their photos also come from the same sources as their training data.</p>
<p class="q">Q3. Why did your numbers change between the two runs?</p>
<p>Dataset v1.1 fixed 91 BLPD images that were whole photos instead of plate crops. The detector and the end-to-end
photos did not change, and the authors' result is identical in both runs, so the +2 to +4 points come from the fix.</p>
<p class="q">Q4. Why is your detector's mAP50-95 only 46?</p>
<p>One source, ridoykj (24 % of the test set), draws its boxes tighter than everyone else. Our detector finds those
plates (mAP50 99.5) but draws the box around the plate rim. Without ridoykj the score is 64.3, close to their 68.5.</p>
<p class="q">Q5. Is removing the vehicle stage safe?</p>
<p>Yes. 7 photos gained, 4 lost, p = 0.55, and it is 2.6× faster. It also rescues dark photos where COCO misses the
vehicle and vehicles COCO does not know (CNG, rickshaw).</p>
<p class="q">Q6. What is McNemar's test and why use it?</p>
<p>It compares two models on the same photos by counting only the photos where exactly one model is right. It tells
whether a gap like 95.1 vs 93.1 could be luck. p below 0.05 means a real difference.</p>
<p class="q">Q7. How do you know your evaluation code is correct?</p>
<p>With the authors' weights it reproduces their Table 10 (95.14 vs 95.33 % exact, 98.93 vs 98.82 % characters), and two
runs on different days give the same results.</p>
<p class="q">Q8. Algorithm 1 says "sort left to right". Did you do that?</p>
<p>Done literally on two-line plates it reads 1 of 651 plates. The paper's own text orders labels by type
(city, Metro, letter, number); our decoder does that (87.7 %).</p>
<p class="q">Q9. Is it real time?</p>
<p>On a T4 GPU: yes, about 28 photos per second without the vehicle stage, and the detector alone runs at 131.6 FPS
with TensorRT. On a CPU: not yet; the detector alone is 6 FPS on Kaggle's CPU. Next we try a nano model and OpenVINO.</p>
<p class="q">Q10. Is their 15 % OCR baseline fair?</p>
<p>We cannot tell: the paper does not name the OCR engine. Paper #6's kit runs EasyOCR on our data, so we will report
our own OCR baseline.</p>
"""

S['c4'] = f"""
<h1>19. Next steps (in order)</h1>
{ul([
    f'<b>Fill the notes and team table:</b> extract {M("paper02_outputs.zip")} from {M("After train/results.zip")}, then run {ADD} in {M("docs/results")}. It labels the row v1.1.',
    '<b>Upload</b> (section 20).',
    f'<b>Run {M("paper02_original_check.ipynb")}</b> on Kaggle (about 7.5 h) to make the reproduction of Tables 6-10 official.',
    '<b>Night:</b> the detector misses 9 of the 60 NightLPBD plates at confidence 0.25; every "no reading" at night is such a miss. Try a lower threshold at night, night augmentation and more night data.',
    '<b>Format fallback in the cascade:</b> flag invalid readings (52 of 80 errors) and re-read them with a second reader, as in paper #24.',
    '<b>Speed on a laptop:</b> YOLO12n/YOLOv8n plate detector, 416-512 px input, ONNX/OpenVINO; measure the full pipeline (≥ 10 FPS target).',
    '<b>Box style:</b> report mAP50 plus per-source mAP50-95, or re-draw ridoykj boxes in a dataset v1.2 (team decision).',
    '<b>Detector and reader choice (O2):</b> YOLOv8n / YOLO11n / YOLO12n with equal settings and 3 seeds on our split.',
    '<b>Kit fix for all teammates:</b> report TensorRT/FP32 FPS (and a median) instead of the PyTorch FP16 mean.',
], 'ol')}

<h1>20. GitHub and Drive</h1>
<p>{M('After train/results.zip')} (448 MB) is the whole Kaggle output; do not upload it. Inside it are the two files
the kit made for you. Layout as for paper #23 and Md. Yousuf's {M('weights/paper03/Md.Yousuf/')}:</p>
{table(['Where', 'What'], [
    [M('CSE498R / weights / paper02 / MD.Rokib.Hasan.Oli /'), 'README.md (main v1.1 results + limits), best_lpd_v12s.pt (18 MB), best_lpnr_v12n.pt, paper02_results.zip (2.3 MB), results_vs_paper.md, this PDF (optional)'],
    ['Team Drive ' + M('498R / weights / paper02 /'), 'paper02_outputs.zip (139.1 MB, with weights, ONNX and TensorRT files)'],
], [4.2, 5.8])}
<p class="small">Web upload: open the repo's {M('weights')} folder → Add file → Upload files → drag the {M('paper02')}
folder → commit message "Add paper #2 results on dataset v1.1 (Rokib)". Files must be under 25 MB each for the web upload.</p>

<h1>Appendix: where every number comes from</h1>
{table(['Number', 'Source'], [
    ['paper numbers', 'PLOS One e0355858, pages given in the text'],
    ['v1.1 training, cascade, condition, speed results', M('After train/results.zip') + ' → ' + M('runs/*/results.json') + ', ' + M('results.csv') + ', ' + M('logs/*.log') + '; ' + M('paper-2.log')],
    ['per-photo comparisons, McNemar tests, error types, format check', M('report_src/analysis.py') + ' → ' + M('analysis.json') + ' (from ' + M('e2e_preds_*.csv') + ', ' + M('ocr_preds_test.csv') + ')'],
    ['per-source mAP, box ratios, Figure 10', M('report_src/persource_det.py') + ', ' + M('ridoykj_figure.py') + ' (local RTX 5060 Ti, the run\'s best.pt, dataset v1.1)'],
    ['v1.0 numbers', 'Kaggle log of 19 Oct (paper02-r.log), quoted in ' + M('Paper02_Study_Report_v1.0_run.pdf')],
    ['retraining on BLPD\'s own split', 'local preview of 13 Oct 2026 in the paper #2 notes'],
    ['leakage and cleaning', M('dataset/docs/leakage_audit.md') + ', ' + M('dedupe_report.md') + ', ' + M('DATASET_v1_1_CHANGES.md')],
], [3.6, 6.4])}
<pre>@article{{sarker2026cascaded,
  title   = {{Cascaded {{YOLO}}-based approach for {{Bangladeshi}} license plate
             identification and number recognition}},
  author  = {{Sarker, Proshenjit and Alam, S. M. Shamsul}},
  journal = {{PLOS One}},
  volume  = {{21}}, number = {{8}}, pages = {{e0355858}}, year = {{2026}},
  doi     = {{10.1371/journal.pone.0355858}}
}}</pre>
"""

FLOWS = [['cover'], ['a1'], ['a2'], ['a3'], ['a3b', 'a4'], ['b1'], ['b1ok'], ['b2'], ['b3'], ['b4'], ['b5'], ['c1'], ['c2'],
         ['c3'], ['c4']]


def build():
    arch = fitz.Archive()
    arch.add(fitz.Archive(r'C:\Windows\Fonts'))
    arch.add(fitz.Archive(HERE))
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
        pg.insert_text((ML, PH - 26), 'Paper #2 study report · update for dataset v1.1 · 498R BanglaLPR', fontsize=7.5,
                       fontname='helv', color=(0.45, 0.45, 0.43))
        t = f'{i + 1} / {n}'
        pg.insert_text((PW - ML - fitz.get_text_length(t, 'helv', 7.5), PH - 26), t, fontsize=7.5, fontname='helv', color=(0.45, 0.45, 0.43))
    doc.set_metadata({'title': 'Paper #2 study report (dataset v1.1): Cascaded YOLO for Bangladeshi license plates',
                      'author': 'Prepared for MD. Rokib Hasan Oli',
                      'subject': 'Paper study, v1.1 Kaggle results, v1.0 vs v1.1, research use, faculty guide'})
    doc.save(OUT, garbage=3, deflate=True)
    doc.close()
    print('pages', n, OUT)


if __name__ == '__main__':
    build()
