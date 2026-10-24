"""Paper #23 study report -> PDF (PyMuPDF Story, A4, one column)."""
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'Paper23_Study_Report.pdf')

CSS = """
@font-face {font-family: B; src: url(arial.ttf);}
@font-face {font-family: B; src: url(arialbd.ttf); font-weight: bold;}
@font-face {font-family: B; src: url(ariali.ttf); font-style: italic;}
@font-face {font-family: B; src: url(arialbi.ttf); font-weight: bold; font-style: italic;}
@font-face {font-family: H; src: url(arialbd.ttf); font-weight: bold;}
@font-face {font-family: H; src: url(arial.ttf);}
@font-face {font-family: N; src: url(Nirmala.ttc);}
.bn {font-family: N;}
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
.big {font-family: H; font-weight: bold; font-size: 20pt; color: #104281;}
.q {font-weight: bold; margin: 6pt 0 1pt 0;}
"""


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


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


sections = []  # each entry = HTML that starts on a new page

# ------------------------------------------------------------------------------------------------ cover + summary
sections.append(f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER #23 STUDY REPORT</p>
<h1 style="font-size:21pt; margin-bottom:3pt">Do We Train on Test Data?</h1>
<p style="font-family:H; font-size:12.5pt; color:#1c5cab; margin-bottom:8pt">The Impact of Near-Duplicates on License Plate
Recognition: what the paper says, what our Kaggle run found, and how to use it</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader and data lead) · Supervisor: Dr. Md Mansur ul Mulk ·
18 October 2026<br>
Paper: R. Laroca, V. Estevam, A. S. Britto Jr., R. Minetto, D. Menotti, IJCNN 2023, DOI 10.1109/IJCNN54540.2023.10191584,
arXiv 2304.04653v2 (8 pages). Page numbers below are the arXiv PDF pages.<br>
Our run: Kaggle notebook <span class="mono">paper23_kaggle.ipynb</span>, finished 18 Oct 2026, 65.5 min on Tesla T4 x2.</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> The two most used LPR datasets (AOLP, CCPD) put different photos of the
<i>same plate</i> in both train and test. 47-68 % of AOLP test images and 19 % of CCPD test images have such a
"near-duplicate" in training. When the authors re-split the data so no plate is on both sides, six OCR models'
error rates more than double on AOLP, CCPD's average accuracy falls from 80.3 % to 77.6 %, and the "best"
model changes (p.1-5).</p>
</div>
<div class="good">
<p><b>Our run in three lines.</b> We trained the same YOLOv8n plate detector and plate reader twice on the KUET
Bangla dataset: once on KUET's own published split (with its Roboflow copies), once on our grouped split. In
KUET's published test set, 58.8 % of the plates (114 of 194) are also in training. The plate-reading accuracy a
paper would report is <b>85.57 %</b>; the fair number is <b>72.13 %</b>, so the error almost doubles (14.4 % to 27.9 %,
×1.93), the same effect as in the paper. On the 38 test plates neither model ever saw, the "leaky" model is
<i>not</i> better (68.4 % vs 71.1 %). Its advantage comes only from plates it had already seen (+12 to +23 points).</p>
</div>
<div class="box">
<p><b>Is the training OK?</b> Yes. All 5 jobs ended with <span class="mono">code=0</span>, all epochs ran, 0
corrupt images, no NaN. There are no errors, but there are 3 things you must know before you quote numbers (section 14): two
cross-scores in <span class="mono">results.json</span> are not valid, the per-condition reading table favours the
leaky model for the same reason, and the kit's "relative gap" is not the paper's formula.</p>
<p><b>What to tell the faculty.</b> "Public Bangla LPR splits leak test plates into training. On KUET this alone
makes plate reading look 13 points better than it is. That is why our dataset uses a grouped split, and why our
numbers are lower than some published ones: ours are honest." Show 5 things: the problem picture, the paper's
table, our leakage audit, our 85.6 → 72.1 result, and the seen/unseen chart (section 17).</p>
<p><b>GitHub.</b> Upload the ready folder <span class="mono">After Train/github_upload/weights/paper23/</span> into
<span class="mono">weights/</span> of the CSE498R repo; the steps are in section 21.</p>
</div>
<h2>How this report is organised</h2>
{table(['Part', 'Sections', 'Read it when'], [
    ['A. Understand the paper', '1-9: problem, terms, data, method, results, critique', 'before you explain the paper to anyone'],
    ['B. Our experiment', '10-14: design, leakage in KUET, results, comparison, health check', 'before you quote any of our numbers'],
    ['C. Use it', '15-21: research use, benefits, faculty slides, script, Q&amp;A, next steps, GitHub', 'before the faculty meeting'],
], [2.2, 4.6, 3.2])}
""")

# ------------------------------------------------------------------------------------------------ part A
sections.append(f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What problem does the paper study?</h1>
<p>An automatic license plate recognition (ALPR) system has two steps: <b>detection</b> (find the plate in the
photo) and <b>recognition</b> (read its characters, an OCR task). The paper looks only at recognition, because modern
systems crop the plate and <b>rectify</b> it (warp it to a flat rectangle) before reading it (p.1).</p>
<p>Researchers compare models on public datasets that are divided into train and test sets. The split is
usually by <i>file</i>: each image goes to train or to test. But one vehicle is often photographed several times
(another second, another angle, another subset). After cropping and rectifying, two such photos of the same plate look
almost identical. If one is in train and the other is in test, the model is partly tested on something it has
already memorised (p.1, Fig. 1-3).</p>
<p>So the question is: <b>how much of the reported accuracy is memory and how much is generalisation?</b> The idea
comes from Barz &amp; Denzler (2020), who found the same problem in CIFAR-10/100 ("Do we train on test data? Purging
CIFAR of near-duplicates") (p.2).</p>
{fig('fig_concept.png', 480, 'Figure 1 (drawn for this report). Left: a split by file lets photos of one plate land in both train and test. Right: a split by group keeps every photo of a plate on one side. This is the whole idea of the paper.')}

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning', 'Example'], [
    ['Exact duplicate', 'the same file twice (byte copy) or a re-encoded copy', 'the same JPEG saved twice'],
    ['Near-duplicate (this paper)', '<b>different images that show the same license plate</b> (p.1)', 'one car photographed in subset AC and again in subset RP'],
    ['Roboflow copy (our data)', 'an augmented copy of one photo (noise, brightness, flip) exported as a new file', 'KUET: about 3 files per photo in train'],
    ['Data leakage', 'information from the test set is present during training', 'test plate also in train'],
    ['Fair split / duplicate-free split', 'a split where no test image has a near-duplicate in train', 'AOLP-Fair-A, CCPD-Fair'],
    ['Group split', 'put all images of one group (plate, vehicle, photo cluster) into the same split', 'our 70/15/15 split, seed 42'],
    ['Recognition rate', 'plates read 100 % correctly / all test plates (one wrong character = wrong)', '96.94 %'],
    ['Gap', 'original accuracy minus fair accuracy, in percentage points (pp)', '99.18 - 96.94 = 2.24 pp'],
    ['Relative gap', 'gap / original error rate = gap / (100 % - original accuracy) (p.4)', '2.24 / 0.82 = 273 %'],
    ['Memorisation vs generalisation', 'remembering seen examples vs working on new ones', 'what a test set must measure'],
], [2.3, 4.6, 3.1])}
<div class="box"><p><b>Why the relative gap matters.</b> A drop from 99.18 % to 96.94 % sounds small (2.24 points),
but the error goes from 0.82 % to 3.06 %, which is 3.7 times as many wrong plates. The relative gap shows this: 2.24 / 0.82
= 273 %. When accuracy is near 100 %, always look at the error rate.</p></div>
""")

sections.append(f"""
<h1>3. Datasets and how people split them</h1>
{table(['Dataset', 'What it is', 'Split used in the literature', 'Test images with a near-duplicate in train (p.3, p.5)'], [
    ['AOLP (Taiwan)', '2,049 images in 3 subsets: access control AC 681, law enforcement LE 757, road patrol RP 611 (p.2)',
     '<b>AOLP-A</b>: random 2:1 train/test. <b>AOLP-B</b>: train on AC+LE, test on RP. 20 % of train is validation.',
     'AOLP-A: 320 of 683 = <b>46.9 %</b><br>AOLP-B: 413 of 611 = <b>67.6 %</b>'],
    ['CCPD (mainland China)', '366,789 images taken with handheld cameras in one city; subsets Base, Blur, Challenge, DB, FN, Green, Rotate, Tilt, Weather (p.2)',
     'standard split: Base 200k into 100k train + 100k val; all other subsets (about 157k) are test',
     '29,943 of about 157k = <b>19.1 %</b>'],
    ['ReId', '182,335 low-resolution plate crops, standard split 105,923 / 76,412 (p.5)', 'standard split', '52,394 of 76,412 = <b>68.6 %</b>'],
], [1.6, 3.4, 3.2, 2.4])}
<p>Note that AOLP-B, the "harder" protocol that tests on a different subset, has the <i>most</i> duplicates: the
same cars drive through the access-control gate and past the road-patrol camera.</p>

<h1>4. How the authors found the near-duplicates</h1>
{ul([
    '<b>Step 1, filter by plate text.</b> Only images whose plates have the same text can be duplicates (p.3).',
    '<b>Step 2, compare pixels.</b> Among those, they crop and rectify the plates and take the nearest neighbour by Euclidean distance in pixel space. Because step 1 already removed different plates, this simple distance "works reasonably well" (p.3).',
    '<b>Step 3, check by eye.</b> Fig. 1 shows pairs at the 10th, 50th and 90th percentile of distance. Even the 90th-percentile pair is clearly the same plate.',
], 'ol')}
<p>Their lists of duplicates and the fair splits are public at
<span class="mono">raysonlaroca.github.io/supp/lpr-train-on-test/</span>. There is no training code.</p>

<h1>5. How they built the fair splits</h1>
<p>They could not replace duplicates with new images (that would change the data distribution), so they re-split
while keeping each protocol's character and size (p.3-4):</p>
{table(['Fair split', 'What they changed', 'What stayed the same', 'Why this design'], [
    ['AOLP-Fair-A', 'new random 2:1 split, but all images of one vehicle/plate are kept in the same set', 'same numbers of train, val and test images', 'a like-for-like random split, only without leakage'],
    ['AOLP-Fair-B', '<b>only removed</b> the duplicates from the test set (RP)', 'train and val identical to AOLP-B', 'so nobody can say the drop comes from less training data'],
    ['CCPD-Fair', 'Base is still split 100k/100k, but every duplicate goes to <b>validation</b>, not train (same for Green)', 'test set unchanged (the same 157k images)', 'the test is identical, so any drop is caused by the training data'],
], [1.6, 3.4, 2.6, 3.0])}
<div class="box"><p><b>The clever part.</b> Fair-B and CCPD-Fair each keep one side fixed (train or test). That
removes the obvious objection "your fair split is just harder or smaller". Our own experiment copies these two ideas: the
"clean part of the published test" is the Fair-B idea, and the seen/unseen check is the controlled version.</p></div>

<h1>6. Models and training</h1>
{table(['Item', 'Detail (p.4)'], [
    ['Models (6)', 'LPR-specific multi-task CNNs: CNNG, Holistic-CNN, Multi-Task. Scene-text models: STAR-Net (CTC), TRBA (attention), ViTSTR-Base (Transformer)'],
    ['Keras models', 'Adam, learning rate 1e-3 (ReduceLROnPlateau patience 3, factor 0.1), batch 64, at most 100 epochs, early stop patience 7'],
    ['PyTorch models', 'deep-text-recognition-benchmark fork; Adadelta (rho 0.99), 300k iterations, batch 128'],
    ['Extra data', 'many synthetic plates: blank templates + random but realistic character sequences + perspective, noise, shadow and colour changes (also applied to real images) (p.4, Fig. 4)'],
    ['Hardware', 'AMD Threadripper 1920X, 96 GB RAM, NVIDIA Quadro RTX 8000 (48 GB). Training time not reported'],
    ['Fairness of comparison', 'all models trained by the authors with the same protocol in each experiment; no numbers copied from other papers (p.4, footnote 6)'],
], [1.8, 8.2])}
""")

paper_rows = [
    ['CNNG', '98.88 → 95.63 (3.25; 290 %)', '98.91 → 96.80 (2.11; 194 %)', '88.24 → 86.93 (1.31; 11 %)'],
    ['Holistic-CNN', '96.75 → 93.11 (3.64; 112 %)', '98.42 → 96.30 (2.12; 134 %)', '77.01 → 75.41 (1.60; 7 %)'],
    ['Multi-Task', '97.33 → 93.79 (3.54; 133 %)', '98.42 → 95.29 (3.13; 198 %)', '83.01 → 81.84 (1.17; 7 %)'],
    ['STAR-Net', '98.69 → 95.83 (2.86; 218 %)', '98.47 → 96.46 (2.01; 131 %)', '78.53 → 73.33 (<b>5.20</b>; 24 %)'],
    ['TRBA', '<b>99.18</b> → <b>96.94</b> (2.24; 273 %)', '98.75 → <b>97.47</b> (1.28; 102 %)', '75.83 → 71.48 (4.35; 18 %)'],
    ['ViTSTR-Base', '98.74 → <b>96.94</b> (1.80; 143 %)', '98.75 → 97.31 (1.44; 115 %)', '79.06 → 76.37 (2.69; 13 %)'],
]
sections.append(f"""
<h1>7. Results of the paper</h1>
<p>Recognition rate (%) on the original split → the fair split (gap in points; relative gap). Best in each column in
bold. Source: Tables I, II, III (p.4-5).</p>
{table(['Model', 'AOLP-A → AOLP-Fair-A', 'AOLP-B → AOLP-Fair-B', 'CCPD → CCPD-Fair'], paper_rows, [1.5, 2.9, 2.9, 2.9])}
{fig('fig_paper_errors.png', 495, 'Figure 2 (re-plotted from Tables I-III). Error rate = 100 % - recognition rate. On AOLP the orange bars (fair split) are 2-4 times as high as the blue ones (original split).')}
<h2>The four findings to remember</h2>
{ul([
    '<b>Errors more than double on AOLP.</b> Every model, both protocols. Relative gaps 102-290 % (p.4).',
    '<b>The ranking changes.</b> On AOLP-B, CNNG is 1st on the original split but only 3rd on the fair split. On AOLP-A, ViTSTR moves from 3rd to tied 1st (p.4-5). So a "new best model" claimed on a leaky split may only be better at memorising.',
    '<b>CCPD: smaller relative effect, big absolute effect.</b> Average 80.3 % → 77.6 %. STAR-Net loses 5.20 points and TRBA 4.35. Even the smallest gap (1.17 pp) is 1,800+ extra misread plates, and 5.2 pp is 8,000+ (p.5). The worst subset is Blur: TRBA 50.2 % → 38.7 % (Table IV).',
    '<b>Bigger models may hide it longer.</b> The ability to memorise grows with model capacity, so the problem may get worse as models get deeper (p.5).',
])}
<h2>Section IV: the problem is everywhere</h2>
{ul([
    'EnglishLP, Medialab LPR and PKU have no official split; people split randomly and get near-duplicates without noticing (p.5, Fig. 5).',
    'ReId: 68.6 % of test images have a near-duplicate in train, and works using it never mention it (p.5).',
    '<b>Across datasets:</b> CLPD (used as a "cross-dataset" test set) shares vehicles with ChineseLP. Training on one and testing on the other is then not really cross-dataset (p.6, Fig. 7).',
    'Good example: RodoSol-ALPR was split with no duplicates by design (p.6).',
])}
""")

sections.append(f"""
<h1>8. Critical reading: strengths and limits</h1>
<h3>Strengths</h3>
{ul([
    'Simple, important question; clear evidence on the two most used datasets and six model families.',
    'Careful control in AOLP-Fair-B and CCPD-Fair (one side kept fixed), so the drop cannot be blamed on less data.',
    'All models trained by the authors with one protocol, so the comparison is consistent.',
    'Duplicate lists and fair splits are public, so anyone can check or reuse them.',
    'Peer-reviewed (IJCNN 2023, IEEE), short and easy to cite.',
])}
<h3>Limits (good points to raise yourself before the faculty does)</h3>
{ul([
    '<b>Recognition only.</b> Detection is not tested. The authors argue full images vary more, but they give no numbers. (Our run adds detection.)',
    '<b>Duplicates need plate text.</b> Step 1 uses the plate text, so unlabelled images and duplicates of <i>different</i> plates of the same car are not covered.',
    '<b>AOLP-Fair-A changes which images are trained on</b>, so part of its drop may come from a different (maybe harder) test set. Fair-B controls for this, and Fair-B still shows big gaps.',
    '<b>One run per model.</b> No seeds, no confidence intervals. Small gaps (for example 1.28 pp on 611 images) could partly be noise.',
    '<b>No training code</b>, only the splits. CCPD corner labels are imprecise (p.2, footnote 2), which affects rectification.',
    'Synthetic data was added for all models. It reduces memorisation, so without it the gaps might be even larger. This is not tested.',
])}
<h1>9. Check yourself (2 minutes, without the paper)</h1>
<div class="box">
<p>In AOLP and CCPD, many test plates also appear (as other photos) in training: 47-68 % of AOLP test images and 19 %
of CCPD's. Re-splitting so that no plate is on both sides makes six OCR models' error rates more than double on AOLP
and drops CCPD's average from 80.3 % to 77.6 %. The "best" model even changes. Lesson: split by plate or photo
group, never by file, or the numbers are inflated.</p>
<p>Two weak points: OCR only (no detection), and duplicates are found through plate text (unlabelled images are not
covered). What we take from it: our grouped split, our leakage audit, and the leaky-vs-fair experiment on Bangla data.</p>
</div>
""")

# ------------------------------------------------------------------------------------------------ part B
sections.append(f"""
<p class="part">PART B · OUR EXPERIMENT ON BANGLA DATA</p>
<h1>10. What we ran</h1>
<p>We used the paper's idea on <b>KUET</b> (Bangladeshi License Plate Recognition Dataset, Hossain &amp; Hassan;
the dataset of paper #1 in our list). KUET has a plate-localisation part and a character-recognition part, each
with its own published train/valid/test split exported from Roboflow. The same source is trained in two versions,
with the same models and settings:</p>
{table(['Version', 'Data', 'Detection images train / val / test', 'Reading crops train / val / test'], [
    ['<b>leaky</b>', 'KUET\'s own published split with every Roboflow copy (<span class="mono">extras/leaky_kuet/</span>). This is how papers that use KUET trained.', '5,620 / 277 / 553', '3,391 / 320 / 194'],
    ['<b>fair</b>', 'the same KUET images in our grouped 70/15/15 split, seed 42, one copy per photo, every plate in one split (<span class="mono">fair_kuet_*.txt</span>)', '1,758 / 311 / 540', '908 / 200 / 183'],
], [1.0, 4.6, 2.2, 2.2])}
{table(['Setting', 'Value'], [
    ['Plate detector', 'YOLOv8n (3.01 M parameters, 8.2 GFLOPs) from COCO weights; 640 px; 50 epochs; batch 16'],
    ['Plate reader', 'YOLOv8n character detector over 101 tokens (10 digits, the word "Metro", 26 series letters, 64 district words); 320 px; 80 epochs; batch 32. A "rows" decoder turns the boxes into the two-row BRTA plate text'],
    ['Training', 'SGD, lr0 0.01, seed 42, deterministic, Ultralytics 8.4.170, PyTorch 2.11.0+cu128, Python 3.13.15'],
    ['Hardware / time', 'Kaggle 2× Tesla T4. GPU 0: det_leaky 47.3 min → det_fair 12.5 min → scoring 0.8 min. GPU 1: rec_leaky 28.8 min → rec_fair 9.7 min. Wall time 65.5 min'],
    ['Metrics', 'detection: mAP50 and mAP50-95; reading: plate exact % (the paper\'s recognition rate: whole plate correct)'],
    ['Test sets', '<b>leaky_test_all</b> = KUET published test (what papers report); <b>leaky_test_clean</b> = its part with no near-duplicate in the published train; <b>leaky_test_dup</b> = the rest; <b>fair_test</b> = KUET in our test split'],
], [1.8, 8.2])}
<h3>What counts as a duplicate here</h3>
{ul([
    '<b>Detection:</b> a test photo that is in the same near-duplicate cluster as a train photo. Clusters come from the dataset build: perceptual hashes (pHash, dHash) checked by normalised cross-correlation, plus Roboflow base names and MD5.',
    '<b>Reading:</b> a test crop whose <b>plate text</b> also occurs in train. No KUET reading test crop is the same <i>photo</i> as a train crop (audit: 0 of 218), so all of these are "same plate, other photo": exactly the paper\'s definition.',
])}

<h1>11. How leaky is KUET\'s published split?</h1>
{table(['Measure', 'KUET detection', 'KUET reading', 'Paper, for scale'], [
    ['train files vs distinct content', '5,687 published files from only 1,964 distinct photos (about 2.9 copies each); 5,620 used after label checks', '3,391 crops (3,356 distinct files) of only <b>447 plates</b> (about 7.5 crops per plate)', '-'],
    ['test items with a near-duplicate in train', '79 of 553 = <b>14.3 %</b>', '114 of 194 = <b>58.8 %</b>', 'AOLP-A 46.9 %, AOLP-B 67.6 %, CCPD 19.1 %'],
], [2.4, 2.6, 2.6, 2.4])}
<p>So KUET's reading test set is as leaky as AOLP: more than half of its test plates were already seen in training.
The same audit (<span class="mono">dataset/docs/leakage_audit.md</span>) finds that BLPD (paper #2) is worse for
detection: 94 of 213 test photos (44.1 %) have a copy in train, and 270 of 721 reading crops (37.4 %).</p>
""")

sections.append(f"""
<h1>12. Results</h1>
<h2>12.1 Detection: memory shows only at strict IoU</h2>
{table(['Detector (mAP50-95 %)', 'published test, all (553)', 'clean part (474)', 'duplicated part (79)', 'our fair test (540)'], [
    ['det_leaky', '76.86', '76.04', '<b>83.06</b>', '77.45*'],
    ['det_fair', '74.45', '75.93', '68.24', '74.68'],
    ['difference', '+2.41', '+0.11', '<b>+14.82</b>', '-'],
], [2.6, 2.0, 1.7, 1.9, 1.8], num=(1, 2, 3, 4))}
<p class="small">* det_leaky trained on 20 of these 540 photos. det_fair trained on none of KUET's published test photos, so the first three columns are a clean comparison. mAP50 is 97.1-99.5 % in every cell.</p>
{fig('fig_ours_det.png', 400, 'Figure 3. On the 474 clean test photos both detectors are equal. On the 79 photos that have a copy in the leaky training set, the leaky detector is 14.8 points better. That is memory, not skill.')}
<p><b>How to read it.</b> At IoU 0.5 (mAP50) plate detection is solved on KUET (about 99 %), so leakage cannot show there.
The paper's view that detection is less affected is right for the headline number: the overall gap is only 0.8-2.4
points. But at strict box precision (mAP50-95), on the photos it has seen copies of, the leaky detector is much better. For a
model that has never seen them, those 79 photos are <i>harder</i> than average (68.2 vs 75.9), yet the leaky model scores
83.1 on them.</p>

<h2>12.2 Plate reading: the paper's effect, on Bangla plates</h2>
<table><tr>
<td width="50%" style="border:none; text-align:center"><p class="small">What a paper using KUET's split would report</p><p class="big">85.57 %</p><p class="small">rec_leaky on the published test (194 plates)</p></td>
<td width="50%" style="border:none; text-align:center"><p class="small">The fair number</p><p class="big" style="color:#c2410c">72.13 %</p><p class="small">rec_fair on our grouped test (183 plates)</p></td>
</tr></table>
<p>The gap is 13.44 points. The error rate goes from 14.43 % to 27.87 %, ×1.93. Relative gap (paper formula) = 13.44 / 14.43 = <b>93 %</b>.
The paper's AOLP result was "more than twice as high"; ours is almost twice.</p>
{table(['Same model (rec_leaky), different parts of the published test', 'plates', 'plate exact %'], [
    ['all (what is reported)', '194', '85.57'],
    ['plate also in train (near-duplicates)', '114', '90.35'],
    ['clean: plate not in train', '80', '78.75'],
    ['gap all - clean (the AOLP-Fair-B idea: same model, duplicates removed from test)', '', '6.82 pp; relative gap 6.82 / 14.43 = 47 %'],
], [6.4, 1.0, 2.6], num=(1,))}
""")

sections.append(f"""
<h2>12.3 The controlled check: is the leaky model better, or does it remember?</h2>
<p>The two numbers above use different test sets. To compare the readers on <i>the same</i> plates, we took the 183 KUET
plates of our fair test (the fair reader never trained on any of them) and split them by what the <b>leaky</b>
reader saw in training. This uses only the saved predictions (<span class="mono">ocr_preds_test.csv</span>); no model was
re-run. The result is in <span class="mono">ocr_overlap_check.json</span>.</p>
{table(['Group of KUET test plates', 'plates', 'leaky reader', 'fair reader', 'difference'], [
    ['A: the identical crop was in the leaky training set', '104', '94.23', '71.15', '<b>+23.08</b>'],
    ['B: same plate, another photo in the leaky training set', '41', '87.80', '75.61', '<b>+12.20</b>'],
    ['C: plate in <i>neither</i> training set', '38', '68.42', '71.05', '<b>-2.63</b>'],
], [5.0, 1.0, 1.4, 1.4, 1.2], num=(1, 2, 3, 4))}
{fig('fig_ours_ocr.png', 400, 'Figure 4. On plates it has never seen (group C) the leaky reader is no better than the fair one. Its whole advantage is on plates it saw: +12 points for the same plate in another photo (the paper\'s near-duplicate), +23 for the identical crop.')}
<div class="good"><p><b>This is the strongest result of the run.</b> The paper says leaky splits measure memory. Here we see it
directly: same test plates, same metric, and the gap appears exactly where the model had seen the plate before.
Group B is the paper's case (same plate, other photo) and still gives +12 points.</p></div>
<p class="small">Caution: group C has only 38 plates, so its uncertainty is about ±15 points. Say "no sign of an
advantage", not "the fair model is 2.6 points better". On the four other sources, which neither reader saw (anpr,
blpd, lpdba, nightlpbd: 468 plates), the leaky reader is 48.1 % vs 46.2 %. That is a small real difference, which
fits its 3.7× more training updates.</p>

<h1>13. Our results next to the paper</h1>
{table(['', 'Paper (AOLP / CCPD)', 'Ours (KUET, Bangla)'], [
    ['Test items with a near-duplicate in train', 'AOLP-A 46.9 %, AOLP-B 67.6 %, CCPD 19.1 % (p.3)', 'reading 58.8 %; detection 14.3 %'],
    ['Accuracy on the leaky protocol', 'e.g. TRBA AOLP-A 99.18 %', '85.57 %'],
    ['Accuracy on the fair protocol', 'TRBA AOLP-Fair-A 96.94 %', '72.13 %'],
    ['Gap, fair protocol (pp)', 'AOLP 1.28-3.64; CCPD 1.17-5.20', '13.44'],
    ['Relative gap (gap / original error)', 'AOLP 102-290 %; CCPD 7-24 %', '93 %'],
    ['Same model, duplicates removed from test', 'AOLP-Fair-B: 1.28-3.13 pp (102-198 %)', '6.82 pp (47 %)'],
    ['Model ranking changes?', 'yes on AOLP', 'not tested (one reader architecture)'],
    ['Detection', 'not studied', 'headline +0.8-2.4 mAP50-95; on duplicated photos +14.8'],
    ['Runs / seeds', 'one run per model', 'one run, seed 42'],
], [3.0, 3.6, 3.4])}
<p>The relative gaps differ because they depend on the starting error: AOLP models start near 1 % error, so a few points of
drop is a large multiple. Our reader starts at 14 % error, like CCPD. The <b>absolute</b> gap (13.4 points) is bigger than anything in the paper:
KUET's reading split repeats each plate about 7.5 times in training, so there is a lot to memorise.</p>
<div class="warn"><p><b>Do not compare 85.57 % with the paper's 99.18 %.</b> Different country, plate format, data
and models. Only compare the <i>direction and size of the gap</i>. The kit's table says the same.</p></div>
""")

sections.append(f"""
<h1>14. Is my training OK? Health check of the Kaggle run</h1>
<div class="good">
<p><b>Yes, the run is correct and complete.</b></p>
{ul([
    'All five jobs ended with <span class="mono">code=0</span>: rec_leaky 29.8 min, rec_fair 10.7 min, det_leaky 47.8 min, det_fair 14.5 min, leaky_vs_fair 0.8 min.',
    'Full mode (FAST = False): 50 of 50 detector epochs and 80 of 80 reader epochs ran; no time cap was hit.',
    'Every dataset scan reports <b>0 corrupt</b> images; no NaN losses; losses fall smoothly.',
    'Both zips were packed: <span class="mono">paper23_outputs.zip</span> 50.9 MB (with weights) and <span class="mono">paper23_results.zip</span> 3.1 MB (no weights).',
    'The only warnings are harmless: Ray <span class="mono">get_trial_id</span> UserWarning, the debugger "frozen modules" note, and two nbconvert SyntaxWarnings at the very end. Ignore them.',
])}
</div>
<h2>Things to know before you quote numbers (not crashes, but they change the meaning)</h2>
{table(['#', 'Issue', 'Effect', 'What to do'], [
    ['1', '<b>Two cross-scores are not valid.</b> The grouped split for the reading crops does not keep KUET\'s published test crops in test. So rec_fair trained on 135 of the 194 published-test crops, and rec_leaky trained on 104 of the 183 fair-test crops.', 'rec_fair on the published test (91.24 %) and rec_leaky on the fair test (87.43 %) are inflated. Detection is fine: det_fair saw 0 published test photos.', 'Do not report these two. Use the controlled check (12.3). Fix the kit later (section 20).'],
    ['2', '<b>The per-condition reading table favours the leaky reader.</b> 104 of the 183 KUET crops in our OCR test were in its training data; 81 of the 112 "blurry" plates are KUET.', 'leaky "blurry" 75.89 % vs fair 59.82 % is mostly memory.', 'Quote only rec_fair\'s condition numbers, or the non-KUET sources.'],
    ['3', '<b>"Relative gap" in the kit is not the paper\'s formula.</b> The kit divides by the clean accuracy (8.66 %); the paper divides by the original error rate.', '8.66 % looks tiny next to the paper\'s 112-290 %.', 'Use 47 % (all vs clean) and 93 % (leaky vs fair protocol).'],
    ['4', '<b>The fair models got about 3× fewer updates</b> (the same epochs on 3-4× fewer files). det_fair\'s best epoch was the last one (50), so it was still improving.', 'Slightly favours the leaky models, yet they were not better on unseen data, so the conclusion gets <i>stronger</i>.', 'Rerun with equal steps (e.g. det_fair about 160 epochs) for the final report.'],
    ['5', '<b>rec_leaky memorises:</b> its validation mAP50-95 stopped improving at about 0.71-0.72 after epoch 20, while its training box loss kept falling (0.77 → 0.36).', 'Expected with about 7.5 crops per plate.', 'A good figure for the "memorisation" story (results.csv).'],
    ['6', '<b>xsort decoder = 0 %</b> in results.json.', 'Expected: Bangla plates have two rows, so sorting all characters left to right mixes the rows.', 'Ignore it; the "rows" decoder is the real one.'],
    ['7', '<b>Whole OCR test only 53-59 %.</b>', 'Both readers trained on KUET only and are tested on 5 sources (different fonts and cameras).', 'Not a bug; this is a domain-shift result.'],
    ['8', '<b>One seed, small subsets</b> (38 and 79 items).', 'About ±10-15 points of uncertainty on the small groups.', 'Say "consistent with"; add 3 seeds later.'],
], [0.35, 3.6, 3.0, 3.0])}
""")

# ------------------------------------------------------------------------------------------------ part C
sections.append(f"""
<p class="part">PART C · USE IT</p>
<h1>15. How to use this paper in your research</h1>
{table(['Use', 'What to do concretely'], [
    ['1. Justify our dataset design', 'In the report\'s Method section, cite Laroca et al. (2023) and our numbers to explain why the BanglaLPR All-in-One split is grouped (near-duplicate clusters + plate text + crops of one photo = one group). Sample sentence: <i>"Following Laroca et al. [x], who showed that near-duplicates between training and test sets inflate LPR accuracy, we split by group. On KUET\'s published split, 58.8 % of test plates also occur in training, and plate-reading accuracy falls from 85.6 % to 72.1 % when they are removed."</i>'],
    ['2. Read published Bangla numbers carefully', 'Papers that train and test on KUET\'s or BLPD\'s own split (for example #1 and #2 in our list) report numbers that include this leakage. When we compare with them, write "on the published (leaky) split" and give our fair number next to it.'],
    ['3. A reporting rule for the team', 'Never re-split the dataset randomly. Report on our fair test. For any OCR model also report <b>seen vs unseen plates</b> (as in 12.3). This costs nothing: it uses saved predictions.'],
    ['4. Make the leakage audit a result', '<span class="mono">dataset/docs/leakage_audit.md</span> covers 14 public Bangla splits (KUET 14.3 % / 58.8 %, BLPD 44.1 % / 37.4 %, NightLPBD 20.7 % / 27.6 %, Zenodo 22.1 %). Before calling it new, search Google Scholar for a similar audit of Bangla datasets. If you find none, it can be a section of the thesis or a short paper: "near-duplicates in Bangladeshi LPR datasets".'],
    ['5. Choose models fairly', 'The paper shows the ranking can change. Compare all our models (YOLO char reader, CRNN, PARSeq, TrOCR...) only on the grouped test, or a "better" model may just memorise better.'],
    ['6. Check across datasets too', 'Section IV warns about duplicates across datasets. Our build de-duplicated all 12 sources together, so a plate copied between two public sets is one group. Say this when you present cross-source results.'],
], [2.2, 7.8])}

<h1>16. What you gain from it</h1>
{ul([
    '<b>Credible numbers.</b> Faculty and reviewers can trust that our accuracy is not memory.',
    '<b>A defence, not a weakness.</b> When someone asks "why is your accuracy lower than paper X?", you have a peer-reviewed reason and our own measurement.',
    '<b>Our own finding on Bangla data,</b> not only a reproduction. The 85.6 → 72.1 result and the seen/unseen chart are new evidence.',
    '<b>Fair model choice.</b> Decisions based on the grouped test pick the model that generalises.',
    '<b>Saves time.</b> We will not chase a 95 %+ published number that was partly memory.',
    '<b>A reusable tool and rule</b> for the whole team (grouped split, leakage audit, seen/unseen breakdown).',
])}
""")

sections.append(f"""
<h1>17. What to show the faculty (8 slides)</h1>
{table(['#', 'Slide', 'Content (all files are in this folder or the dataset)'], [
    ['1', 'The claim', 'One sentence: "Public Bangla LPR splits leak test plates into training; on KUET this makes plate reading look 13 points better than it is."'],
    ['2', 'The problem', 'Figure 1 of this report (split by file vs by group).'],
    ['3', 'The paper', 'Laroca et al., IJCNN 2023: duplicate fractions (46.9 / 67.6 / 19.1 %) + Figure 2 (errors double).'],
    ['4', 'Our audit', 'Table from leakage_audit.md: KUET, BLPD, NightLPBD, Zenodo duplicate rates.'],
    ['5', 'Our experiment', 'Leaky vs fair design (section 10 tables): same models, same settings, two splits.'],
    ['6', 'Headline result', '85.57 % → 72.13 %; error ×1.93 (section 12.2).'],
    ['7', 'Proof it is memory', 'Figure 4 (groups A/B/C) and Figure 3 (detection: equal on clean photos, +14.8 on duplicated photos).'],
    ['8', 'So what + next', 'Our grouped split; our numbers are honest; limits (one seed, 38 plates); next steps (section 20).'],
], [0.4, 1.6, 8.0])}

<h1>18. What to explain: a 2-minute script</h1>
<div class="box">
<p>"Paper #23 is by Laroca and colleagues, IJCNN 2023. They noticed that the most used license-plate datasets put
different photos of the <i>same plate</i> in both train and test. After cropping, these photos look almost the same, so the
model is partly tested on what it memorised. In AOLP almost half to two-thirds of the test images have such a near-duplicate.
When they re-split so that no plate is on both sides, error rates of six OCR models more than doubled, and the best model changed.</p>
<p>We checked our Bangla data the same way. In KUET's published split, 59 % of the test plates also appear in training,
because Roboflow exports several copies of each photo and the same cars appear many times. We trained the same YOLOv8n models
on KUET's split and on our grouped split. Plate reading looks like 85.6 % on the leaky split but is 72.1 % on the fair one:
the error almost doubles, as in the paper. On plates that neither model had ever seen, the leaky model is not better at all.
Its advantage is only on plates it had already seen.</p>
<p>That is why our dataset is split by group, and why our numbers can be lower than some published Bangla results:
ours measure generalisation."</p>
</div>

<h1>19. Questions the faculty may ask, with answers</h1>
<p class="q">Q1. What exactly is a near-duplicate?</p>
<p>Different images showing the same physical plate (the paper's definition). In our data also Roboflow-augmented copies of one
photo. We find them with perceptual hashes checked by cross-correlation, Roboflow base names, MD5, and the plate text.</p>
<p class="q">Q2. Is the fair model worse just because it has fewer training images?</p>
<p>No. The distinct content is similar (447 plates in leaky train vs about 422 in fair train); the leaky set mainly has more
<i>copies</i>. The leaky models even got about 3× more updates. Still, on plates neither model saw (group C) the leaky reader is not better, and
on clean detection photos both detectors are equal (76.0 vs 75.9).</p>
<p class="q">Q3. 38 plates is small. Is that significant?</p>
<p>Not on its own; the uncertainty is about ±15 points. The claim is "no sign of an advantage on unseen plates". The large,
consistent gaps in groups A (+23) and B (+12) are the main evidence. Next: 3 seeds and the BLPD dataset.</p>
<p class="q">Q4. Does it matter for detection?</p>
<p>Less for the headline. mAP50 is about 99 % either way, which matches the paper's remark that full images vary more. At strict
IoU (mAP50-95), on duplicated photos the leaky detector is +14.8 points, so memorisation exists there too.</p>
<p class="q">Q5. Why not just delete duplicates from the test set instead of re-splitting?</p>
<p>We did both. "Clean part of the published test" is the paper's AOLP-Fair-B idea (85.6 → 78.8 for the same model). Re-splitting
is the Fair-A idea. Both show the same direction.</p>
<p class="q">Q6. Why is your accuracy lower than published Bangla papers?</p>
<p>Many published results use the leaky published splits. Our test is grouped and covers five sources. On KUET alone, leakage is
worth 13 points of plate reading.</p>
<p class="q">Q7. How did you find duplicates in images that have no plate text?</p>
<p>The dataset build compares perceptual hashes (pHash, dHash) and verifies candidate pairs with normalised cross-correlation. It also
links Roboflow copies by their base file name and byte copies by MD5. 51,348 images from 12 sources were checked together.</p>
<p class="q">Q8. Doesn't removing copies throw away useful augmentation?</p>
<p>We remove only duplicate <i>files</i>. Augmentation is done online during training (YOLO mosaic, HSV, flips), so the model still
sees variations. It just does not see them from the test set.</p>
<p class="q">Q9. What would you change in the experiment?</p>
<p>Equal training steps for both versions, 3 seeds, keep KUET's published test crops in test for the reading split too, add a stronger
reader (PARSeq, paper #31), and repeat on BLPD.</p>
""")

sections.append(f"""
<h1>20. Next steps (in order)</h1>
{ul([
    '<b>Upload the results</b> (section 21) and fill the team table: put <span class="mono">paper23_outputs.zip</span> in <span class="mono">docs/results/</span> and run <span class="mono">python add_results.py paper23_outputs.zip --member "Rokib Hasan Oli"</span>.',
    '<b>Fix the kit</b> (<span class="mono">docs/_kit</span>): keep KUET\'s published test crops in the fair reading test (or score cross-models only on unseen plates); add the paper-style relative gap; add the A/B/C seen/unseen table to <span class="mono">p23_leaky.py score</span>. Then rebuild the notebook with <span class="mono">build_kits.py</span>.',
    '<b>Rerun with equal steps</b> (fair models about 3× the epochs) and 3 seeds; report mean ± std.',
    '<b>Repeat on BLPD</b> (44.1 % of detection test photos leaky), the dataset of paper #2.',
    '<b>Stronger reader:</b> run leaky vs fair with PARSeq (kit #31). The paper expects higher-capacity models to memorise more.',
    '<b>Write it up:</b> one paragraph + Figure 4 in the Method/Dataset chapter; the audit table in the Dataset chapter.',
], 'ol')}

<h1>21. Uploading to GitHub (and Drive)</h1>
<p>Everything is prepared in <span class="mono">After Train/github_upload/weights/paper23/MD.Rokib.Hasan.Oli/</span>,
in the same layout as Md. Yousuf's <span class="mono">weights/paper03/Md.Yousuf/</span>:</p>
{table(['File', 'Size', 'Why'], [
    ['README.md', '-', 'what the package is, main results, limits (like your teammate\'s README)'],
    ['best_det_leaky.pt, best_det_fair.pt', '5.9 MiB each', 'the two plate detectors'],
    ['best_rec_leaky.pt, best_rec_fair.pt', '6.6 MiB each', 'the two plate readers'],
    ['paper23_results.zip', '3.0 MiB', 'all results, logs, curves and code, without weights'],
    ['results_vs_paper.md', '2 KB', 'the kit\'s ours-vs-paper table'],
    ['ocr_overlap_check.json', '1 KB', 'the controlled seen/unseen check (section 12.3)'],
    ['Paper23_Study_Report.pdf (optional)', '-', 'this report, if you want it in the repo'],
], [3.4, 1.4, 5.2])}
<p><b>Not to GitHub:</b> <span class="mono">paper23_outputs.zip</span> (48.5 MiB, includes last.pt) goes to the team
Drive folder <span class="mono">498R / weights / paper23 /</span>. It is ready in <span class="mono">After Train/for_drive/</span>.
Never upload <span class="mono">results.zip</span> (174 MB, the whole Kaggle output).</p>
<h3>Way 1: in the browser (easiest)</h3>
{ul([
    'Open github.com/Rokib-Hasan-Oli/CSE498R and click the <span class="mono">weights</span> folder.',
    'Click <b>Add file → Upload files</b>.',
    'From File Explorer, drag the whole folder <span class="mono">After Train\\github_upload\\weights\\paper23</span> into the page (Chrome/Edge keep the folder structure). Check that the list shows <span class="mono">paper23/MD.Rokib.Hasan.Oli/...</span> with 8 files (9 with the PDF).',
    'Commit message: <span class="mono">Add paper #23 results: leaky vs fair KUET split (Rokib)</span>. Optional description: "YOLOv8n detector + char reader; reading 85.57 % leaky vs 72.13 % fair; see README".',
    'Choose <b>Commit directly to the main branch</b> → <b>Commit changes</b>. Open the folder on GitHub and check that the README is shown.',
], 'ol')}
<h3>Way 2: with git (Git is installed on this PC)</h3>
<pre>cd D:\\498R
git clone https://github.com/Rokib-Hasan-Oli/CSE498R.git
   (now copy the folder After Train\\github_upload\\weights\\paper23
    into D:\\498R\\CSE498R\\weights\\ with File Explorer)
cd CSE498R
git add weights/paper23
git commit -m "Add paper #23 results: leaky vs fair KUET split (Rokib)"
git push</pre>
<p class="small">The first push asks you to sign in to GitHub in the browser. Files under 100 MB are fine with git (the web upload limit is 25 MB per file).</p>

<h1>Appendix: where every number comes from</h1>
{table(['Number', 'Source file'], [
    ['paper numbers', 'arXiv 2304.04653v2, pages given in the text (Tables I-IV on p.4-5)'],
    ['leaky/fair scores, gaps', '<span class="mono">results.zip → runs/leaky_vs_fair/results.json</span>'],
    ['training settings, times, curves', '<span class="mono">runs/&lt;run&gt;/results.json, args.yaml, results.csv</span>; Kaggle log'],
    ['seen/unseen groups A/B/C', '<span class="mono">ocr_overlap_check.json</span> (from <span class="mono">runs/rec_*/ocr_preds_test.csv</span> + KUET labels)'],
    ['overlap of the two splits (20, 104, 135 crops)', 'file-hash comparison of <span class="mono">extras/leaky_kuet/*/images</span> and <span class="mono">*/fair_kuet_*.txt</span>'],
    ['KUET/BLPD leakage, copies per photo', '<span class="mono">dataset/docs/leakage_audit.md</span>, <span class="mono">dedupe_report.md</span>'],
], [3.2, 6.8])}
<pre>@inproceedings{{laroca2023do,
  title     = {{Do We Train on Test Data? {{T}}he Impact of Near-Duplicates on License Plate Recognition}},
  author    = {{Laroca, Rayson and Estevam, Valter and Britto Jr., Alceu S.
               and Minetto, Rodrigo and Menotti, David}},
  booktitle = {{International Joint Conference on Neural Networks (IJCNN)}},
  pages     = {{1--8}},
  year      = {{2023}},
  doi       = {{10.1109/IJCNN54540.2023.10191584}}
}}</pre>
""")


def build():
    arch = fitz.Archive()
    arch.add(fitz.Archive(r'C:\Windows\Fonts'))
    arch.add(fitz.Archive(HERE))
    PW, PH = 595, 842
    ML, MT, MB = 52, 50, 56
    import io
    raw = io.BytesIO()
    writer = fitz.DocumentWriter(raw)
    pages = 0
    flows = [sections[0], sections[1], ''.join(sections[2:5]), ''.join(sections[5:7]), ''.join(sections[7:9]),
             ''.join(sections[9:])]
    for html in flows:
        story = fitz.Story(html=html, user_css=CSS, archive=arch)
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
        pg.insert_text((ML, PH - 26), 'Paper #23 study report · 498R BanglaLPR', fontsize=7.5, fontname='helv', color=(0.45, 0.45, 0.43))
        t = f'{i + 1} / {n}'
        pg.insert_text((PW - ML - fitz.get_text_length(t, 'helv', 7.5), PH - 26), t, fontsize=7.5, fontname='helv', color=(0.45, 0.45, 0.43))
    doc.set_metadata({'title': 'Paper #23 study report: Do We Train on Test Data?', 'author': 'Prepared for MD. Rokib Hasan Oli',
                      'subject': 'Near-duplicates in LPR: paper study, Kaggle results on KUET, faculty guide'})
    doc.save(OUT, garbage=3, deflate=True)
    doc.close()
    print('pages', n, OUT)


if __name__ == '__main__':
    build()
