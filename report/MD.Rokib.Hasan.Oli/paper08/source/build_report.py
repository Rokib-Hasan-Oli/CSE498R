"""Paper #8 study report (dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Afrin et al., ICCIT 2023, arXiv 2312.10701; page numbers = arXiv PDF pages) and the paper notes;
the v1.1 Kaggle run (`After Train/results.zip`, `paper-8.log`, 22 Oct 2026) via analysis.json; the local checks
seg_check.json (where the plate reader fails, vs paper #2 on the same plates) and check_local.json (the original-paper
check on the authors' Data.zip, run locally on 24 Oct 2026); the 498R proposal (RQ1-RQ4, O1-O5).
Run order: analysis.py -> seg_check.py -> check_collect.py <runs> -> charts.py -> build_report.py.
"""
import io
import json
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'Paper08_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
SC = json.load(open(os.path.join(HERE, 'seg_check.json'), encoding='utf-8'))
CK = json.load(open(os.path.join(HERE, 'check_local.json'), encoding='utf-8'))
FONT_DIR = os.path.join(HERE, 'fonts')
C, P, R, V = A['classifiers'], A['plates'], SC['recognition_test'], SC['vs_paper02_same_plates']
CN, RN, CN32, RNN = CK['cnn'], CK['resnet50'], CK['cnn32'], CK['resnet50norm']
GP, SB, SG = A['gfpgan_paired'], A['similarity_by_condition'], A['segmentation_counts']


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
msd = lambda m, nd=1: f'{m[0]:.{nd}f} ± {m[1]:.{nd}f}'

# the text below quotes these numbers; stop if the data ever disagrees with it
assert A['run']['dataset_version'] == '1.1' and all(j['code'] == 0 for j in A['run']['jobs'])
assert C['cnn28']['acc_best_%'] == 93.08 and C['cnn28']['acc_last_%'] == 36.67 and C['cnn28']['best_epoch'] == 5
assert C['resnet50']['acc_best_%'] == 92.89 and C['resnet50']['epochs_run'] == 25 and C['resnet50']['best_epoch'] == 20
assert P['gfpgan_fixed']['plate_exact_%'] == 0.15 and P['gfpgan_fixed']['similarity_%'] == 54.3
assert P['none_fixed']['similarity_%'] == 49.49 and P['gt_boxes']['plate_exact_%'] == 59.53
assert A['plates_ms']['enhance'] == 205.6 and A['plates_ms']['read'] == 6.1
assert GP['fixed']['gfpgan_better'] == 343 and GP['fixed']['gfpgan_worse'] == 224 and GP['fixed']['wilcoxon_p'] < 1e-10
assert R['A_colour_crops_acc_%'] == 92.91 and R['B_binary_blobs_gt_boxes_acc_%'] == 82.4
assert R['C_tokens_found_iou50_all15_%'] == 52.42 and R['C_plates_all_tokens_found_%'] == 8.57
assert SC['ocr_test']['D_local_none_fixed']['similarity_%'] == P['none_fixed']['similarity_%']   # local copy = Kaggle
assert V['p08_gt_boxes_exact_%'] == P['gt_boxes']['plate_exact_%'] and V['p02_char_yolo_exact_%'] == 88.87
assert V['only_p02_right'] == 144 and V['only_p08_right'] == 7
assert CK['audit']['test_copies'] == '48/303' and CK['audit']['val_copies'] == '320/429'
assert CN['acc_last'][:2] == [88.23, 1.48] and CN32['acc_last'][0] == 93.4 and RN['acc_last'][:2] == [87.46, 3.27]
assert CN['acc_clean_last'][0] == 86.01 and CN['words_last'][0] == 100.0 and CN['words_clean_n'] == 0

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
ADD = M('python add_results.py paper08_results.zip --member "Rokib Hasan Oli"')
S = {}  # flow name -> HTML; each flow starts on a new page

cc, rc = C['cnn28'], C['resnet50']
g8, g5 = cc['groups'], rc['groups']
pf, pn = P['gfpgan_fixed'], P['none_fixed']
fx = GP['fixed']
speedup = A['plates_ms']['enhance'] / A['plates_ms']['read']

# ======================================================================================== cover + one-page summary
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER #8 STUDY REPORT · DATASET v1.1 RUN</p>
<h1 style="font-size:19pt; margin-bottom:3pt">Bengali License Plate Recognition: Unveiling Clarity with CNN and GFP-GAN</h1>
<p style="font-family:H; font-size:12pt; color:#1c5cab; margin-bottom:8pt">The paper, our re-run on dataset v1.1, why the
plate reader fails on our plates, what GFP-GAN really buys, and what to show and say to the faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 24 October 2026<br>
Paper: N. Afrin, M. M. Hasan, M. F. E. Safin, K. R. Amin, M. Z. Haque, F. Ahmed and M. T. R. Shawon, <i>26th ICCIT</i>,
Cox's Bazar, 2023; arXiv 2312.10701, 6 pages. Page numbers are arXiv PDF pages.<br>
Code and data: github.com/mhtusher131/Bengali-License-Plate-Recognition_Unveling-Clarity-with-CNN-and-GFP-GAN (MIT; five
TensorFlow Colab notebooks and {M('Data.zip')}, no weights).<br>
Our run: {M('paper08_kaggle.ipynb')} on Tesla T4 x2, 22 Oct 2026, dataset <b>v1.1</b> (confirmed in {M('environment.json')}),
{A['run']['total_h']:.1f} h; full output in {M('After Train/results.zip')}. Extra checks run locally on 24 Oct 2026 (sections 12-13).</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> To read blurry Bangladeshi plates, the authors first sharpen the plate image with
GFP-GAN, then cut it into single symbols with classic OpenCV steps (Otsu threshold and contours), and classify each
symbol with a small CNN (17 classes). They report 92.38 % symbol accuracy (p.4) and 88 % average string similarity on
30 plates (p.5). They never test the reader without GFP-GAN.</p>
</div>
<div class="good">
<p><b>Is the v1.1 training OK? Yes, with one thing to understand.</b> All 5 jobs ended with {M('code=0')}, dataset version
1.1 is confirmed, and a local re-run of the plate test gives the Kaggle numbers to the last digit. The one oddity is real
but harmless: the paper's CNN recipe (batch 1, no normalisation, 40 epochs) became <b>unstable after epoch 5</b> on our
48,000 symbol crops, so the "last epoch" score is only {f1(cc['acc_last_%'])} %. The kit keeps the best-validation
checkpoint (epoch 5: <b>{f1(cc['acc_best_%'])} %</b>), and every later step used that one. ResNet-50 was stopped by the
time cap at epoch 25 of 30 after it had levelled off (best epoch 20).</p>
</div>
<div class="box">
<p><b>Five findings.</b> (1) <b>The symbol classifier is fine:</b> {f1(cc['acc_best_%'])} % of {A['run']['token_crops']['test']:,}
test symbols (paper: 92.38 % on 17 easy classes); {f1(cc['acc_ge20_%'])} % on the 21 common tokens, but only
{f1(cc['acc_rare_%'])} % on 39 rare district words and letters. (2) <b>The plate reader is not:</b> it reads
{f2(pf['plate_exact_%'])} % of our 651 test plates exactly (1 plate), with {f1(pf['similarity_%'])} % mean similarity
(paper: 88 %). (3) <b>Why:</b> the contour step finds only <b>{R['C_tokens_found_iou50_all15_%']:.0f} %</b> of the
characters and all of them on {R['C_plates_all_tokens_found_%']:.1f} % of plates; even with perfect human boxes the
per-symbol CNN reads {f1(V['p08_gt_boxes_exact_%'])} % of plates, against {f1(V['p02_char_yolo_exact_%'])} % for our
paper #2 character detector on the same {V['plates']} plates (144 vs 7 plates, p &lt; 0.0001). (4) <b>GFP-GAN:</b>
+{f1(fx['mean_diff_points'])} similarity points (significant, p &lt; 10<sup>-10</sup>) but no extra exact plate worth the
name, worse at night, and {speedup:.0f}x the reading time ({f1(A['plates_ms']['enhance'])} vs
{f1(A['plates_ms']['read'])} ms per plate). (5) <b>The paper's own numbers reproduce</b> on its own data with our code
(ResNet-50 {f1(RN['acc_last'][0])} %, paper 87.50 %; CNN {f1(CN32['acc_last'][0])} % at the paper's 32x32, paper 92.38 %), but
its split leaks: 48 of 303 test crops copy a training crop, and on the clean crops the CNN loses
{CN['acc_last'][0] - CN['acc_clean_last'][0]:.1f} points.</p>
<p><b>What to tell the faculty in one sentence.</b> "We tested this paper on 651 real Bangladeshi plates: its symbol
classifier is good, but cutting a plate into symbols with contours finds only half the characters, so it reads almost
no plate exactly, and GFP-GAN adds 34 times the cost for a few similarity points, so our system keeps a character
detector and drops restoration."</p>
</div>
<h2>How this report is organised</h2>
{table(['Part', 'Sections', 'Read it when'], [
    ['A. Understand the paper', '1-6: problem, terms, data, method, results, weaknesses', 'before you explain the paper'],
    ['B. Our v1.1 experiment', '7-14: what we ran, is it OK, results vs paper, symbols, plates, GFP-GAN, original check, findings', 'before you quote any number'],
    ['C. Use it', '15-21: research use, benefits, faculty slides, script, Q&amp;A, next steps, GitHub', 'before the faculty meeting'],
], [2.2, 4.8, 3.0])}
"""

# ======================================================================================== part A (the paper)
S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What problem does the paper solve?</h1>
<p>Street cameras in Bangladesh give small, blurry, low-resolution plate images. Earlier Bangla plate readers were
trained and tested on clean images, so they break on real camera footage (p.1-2). The paper asks: <b>if we first make the
plate image clearer with a GAN, can a simple CNN read the symbols?</b> The title word "Unveiling Clarity" refers to this
restoration step.</p>
<p>The paper's answer is a four-step pipeline: restore the plate with <b>GFP-GAN</b>, clean it with classic image
processing, cut it into single symbols with <b>contours</b>, and classify each symbol with a <b>4-layer CNN</b>. A frozen
<b>ResNet-50</b> is tried as a second classifier. There is <b>no plate detector</b>: the input is already a plate image.</p>
{fig('fig_pipeline.png', 470, 'Figure 1. The paper\'s plate reader (p.3-5, and the authors\' notebooks) with what our v1.1 run measured at each step (orange). Sections 11-12 explain the orange numbers.')}

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['GFP-GAN', 'a GAN that restores <i>faces</i> using a face prior learned by StyleGAN (Wang et al., CVPR 2021). It finds faces first; on an image with no face it only runs its <b>background upsampler</b>'],
    ['Real-ESRGAN x2', 'the background upsampler inside GFP-GAN v1.3: a super-resolution network that makes the image 2x bigger and sharper. On a plate, "GFP-GAN" = Real-ESRGAN x2'],
    ['Top-hat / black-hat', 'morphology filters that boost bright and dark details; used here to raise contrast'],
    ['Otsu threshold', 'picks one grey level that splits the image into black and white (binary image)'],
    ['Contour', 'the outline of one connected black or white region; the paper assumes one region = one symbol'],
    ['Segmentation', 'cutting the plate into one small image per symbol'],
    ['Symbol / token', 'one unit on the plate: a digit, a series letter, "মেট্রো" (Metro) or a whole district word such as "ঢাকা"'],
    ['SequenceMatcher similarity', 'Python difflib ratio = 2 x matching characters / (length of both strings). Partial credit: a plate with one wrong digit still scores about 0.9'],
    ['Plate exact', 'the whole plate text is right; one wrong symbol = wrong plate. The strict number'],
    ['CER / char accuracy', 'character error rate = edits needed / truth length; char accuracy = 1 - CER'],
    ['Macro vs weighted F1', 'macro = plain mean over classes (a rare class counts as much as digit "1"); weighted = mean weighted by class size'],
    ['Best-val vs last-epoch checkpoint', 'the model saved at the epoch with the best validation accuracy vs the model after the final epoch'],
    ['McNemar test', 'compares two readers on the same plates using only the plates where exactly one is right; p &lt; 0.05 = real difference'],
    ['Wilcoxon signed-rank test', 'compares two paired lists of numbers (here: similarity of each plate with and without GFP-GAN)'],
], [2.3, 7.7])}
"""

S['a2'] = f"""
<h1>3. The data</h1>
{table(['Item', 'Detail', 'Page'], [
    ['Symbol crops', '2,024 crops, 17 classes: digits 0-9, series letters ক গ ঝ ল, region words ঢাকা, মেট্রো, নারায়ণগঞ্জ', 'p.3, Table I'],
    ['Split', 'train 1,292 / val 429 / test 303 (the abstract calls the 1,292 "the dataset")', 'p.1, p.4'],
    ['Source', 'cut by hand from a Kaggle synthetic plate set (furcifer) and the BLRPS set', 'p.2'],
    ['Size', '"standardized to 32x32"; in Data.zip ' + CK['audit']['share_32x32'].replace('/', ' of ') + ' crops really are 32x32', 'p.3'],
    ['Balance', 'digits 90-140 training crops each; letters and words only 13-30', 'p.3, Table I'],
    ['Plate test', '30 plates (the 30 plates are not released)', 'p.5'],
], [1.6, 7.0, 1.4])}
<div class="warn"><p><b>Their split leaks (our audit of Data.zip, repeated on 24 Oct).</b> {CK['audit']['test_copies'].replace('/', ' of ')}
test crops and {CK['audit']['val_copies'].replace('/', ' of ')} validation crops are copies of a training crop (grey 32x32,
mean difference &lt;= 1; {CK['audit']['test_byte_identical']} test and {CK['audit']['val_byte_identical']} validation crops are
byte-identical files). <b>Every</b> test crop of ঢাকা (12), মেট্রো (10), নারায়ণগঞ্জ (10) and ঝ (14) is a copy, while the digits
are clean (1 of 229). So the letter and word part of 92.38 % is partly memory, and the 98.8 % validation accuracy is
inflated. Only {CK['audit']['classes_matching_table_I'].replace('/', ' of ')} classes match Table I (class 9 is printed as
1 training crop; Data.zip has 100).</p></div>

<h1>4. The method</h1>
<h3>Restoration and segmentation (p.3-4, notebooks GFPGAN.ipynb and Image_Processing.ipynb)</h3>
{ul([
    'GFP-GAN v1.3 with scale 2 and the Real-ESRGAN background upsampler. A plate has no face, so only the Real-ESRGAN x2 step changes the image (we checked the authors\' notebook).',
    'Grey image, top-hat plus black-hat contrast, resize to 333x335, Otsu threshold, erode and dilate (with a tuple as the kernel, a small bug kept as is).',
    'The 15 largest contours become 15 symbol candidates, sorted left to right <b>by x only</b>, each pasted on a 24x44 white canvas and resized to 28x28 for the CNN.',
])}
<h3>The classifiers (p.2-5, notebooks CNN.ipynb and resnet50.ipynb)</h3>
{table(['', 'CNN', 'ResNet-50'], [
    ['Layers', '4 conv (16/32/64/128 filters, kernels 22/16/8/4, "same"), max-pool 4, dense 512, dropout 0.4, dense 128, softmax', 'ImageNet ResNet-50, <b>all layers frozen</b>, Flatten, one Dense layer'],
    ['Input', '28x28 in the code (the text says 32x32), RGB / 255', '150x150, RGB / 255, no ImageNet normalisation'],
    ['Training', 'Adam lr 1e-4, <b>batch 1</b>, 40 epochs, small shifts', 'Adam (Keras default lr 1e-3; text says 1e-4), batch 1, 30 epochs, shear/zoom/flip'],
    ['Parameters', f"{cc['params'] / 1e6:.2f} M", f"{rc['params'] / 1e6:.1f} M ({rc['params_trainable'] / 1e6:.2f} M trainable)"],
], [1.3, 4.6, 4.1])}
<h3>Paper vs code (checked in the repository, commit 8dc72c7)</h3>
{table(['Topic', 'Paper', 'Authors\' code'], [
    ['Input size', '32x32x3 (p.3-4)', 'CNN 28x28, ResNet-50 150x150'],
    ['ResNet-50 test accuracy', '87.50 % (p.4)', 'the notebook prints 73.93 %'],
    ['Plate score', 'average of 30 plates (p.5)', 'averages 10 hard-coded scores (87.56 %)'],
    ['Plate demo', '-', 'index-to-label list does not match the class order; crops not scaled by 1/255'],
    ['Plate order', 'expected strings follow the x order (p.5, Table III)', 'x order mixes the two plate lines, so the "truth" is already mixed'],
], [2.0, 3.6, 4.4])}
"""

S['a3'] = f"""
<h1>5. What the paper reports</h1>
{table(['Result', 'Value', 'Page'], [
    ['CNN symbol test accuracy / precision / recall / F1', '<b>92.38 %</b> / 0.9667 / 0.9162 / 0.9476', 'p.4, Table II'],
    ['ResNet-50 test accuracy / precision / recall / F1', '87.50 % / 0.9343 / 0.8827 / 0.8984', 'p.4, Table II'],
    ['CNN validation accuracy / loss', '98.80 % / 0.0303 (the same paragraph also says "92 %")', 'p.4'],
    ['Whole plates: mean similarity', '<b>88 %</b> on 30 plates; 17 of 30 &gt;= 0.95, 24 of 30 &gt;= 0.90', 'p.5'],
    ['Example plate scores', '1.00, 0.95, 0.86, 0.84', 'p.5, Table III'],
    ['Comparison', 'proposed 88 %, ResNet-50 83 %, Kabiraj et al. (CNN + ESRGAN) 84 %, Haque et al. (ESRGAN) 78 %; different data, so not a fair race', 'p.6, Table IV'],
    ['Speed, hardware, training time', 'not reported', '-'],
], [3.6, 5.0, 1.4])}

<h1>6. Weaknesses (with page numbers, so you can show them)</h1>
{ul([
    '<b>No ablation of the main idea.</b> The title is about GFP-GAN, but no result without GFP-GAN is given (p.4-6). We measured it (section 12).',
    '<b>Leaky split</b> (section 3): 48 of 303 test crops and 320 of 429 validation crops copy training crops; all test crops of the 3 word classes and ঝ.',
    '<b>Only 17 symbols</b> (p.3): 3 of 64 districts and 4 of about 30 series letters. Most real plates contain a symbol the model cannot output.',
    '<b>x-only order</b> (p.5): Bangladeshi plates have two lines; sorting by x mixes them. Their expected strings use the same mixed order, so 88 % is easier than real reading.',
    '<b>Similarity, not exact match</b> (p.5): a mean SequenceMatcher ratio gives partial credit; the share of plates read fully right is not reported.',
    '<b>Tiny plate test</b>: 30 unreleased plates (p.5); no night, rain or real CCTV split.',
    '<b>GFP-GAN described wrongly</b> as a flow-based probabilistic model (p.2); it is a GAN with a face prior, and on plates its face part never runs.',
    '<b>Paper and code disagree</b> on input size, ResNet-50 accuracy and learning rate (section 4).',
    '<b>No speed numbers</b>, although GFP-GAN is the slowest step by far (section 12).',
])}
<div class="box"><p><b>Fair credit.</b> It is one of the few Bangla plate papers with public code and labelled data, and it asks
a good question (does restoration help?). That is why it is worth testing carefully rather than citing as proof.</p></div>
"""

# ======================================================================================== part B (our run)
jobs = {j['name']: j['min'] for j in A['run']['jobs']}
S['b1'] = f"""
<p class="part">PART B · OUR EXPERIMENT ON DATASET v1.1</p>
<h1>7. What we ran</h1>
<p>The notebook {M('paper08_kaggle.ipynb')} ports the authors' Keras code to PyTorch and runs it on our <b>498R BanglaLPR
All-in-One v1.1</b> data (all public Bangla sets merged, de-duplicated, grouped 70/15/15 split with seed 42, val/test with
human labels only). Every labelled symbol box of the recognition set became one crop, in the authors' folder layout.</p>
{table(['Job', 'GPU', 'What it does', 'Time'], [
    [M('cnn28'), '0', 'paper CNN 28x28, 40 epochs, batch 1, Adam 1e-4, on our 101 tokens', f"{jobs['cnn28']} min"],
    [M('resnet50'), '1', 'frozen ImageNet ResNet-50 + Dense, 150x150, 30 epochs planned, batch 1, Adam 1e-3, 3.2 h cap', f"{jobs['resnet50']} min"],
    [M('plates_cnn28'), '0', 'the paper\'s plate pipeline on the 651 OCR test crops: GFP-GAN off/on x paper order/two-line order, plus perfect segmentation (human boxes, 467 crops)', f"{jobs['plates_cnn28']} min"],
    [M('fps_cnn28 / fps_resnet50'), '1', 'classifier speed, batch 1, alone on an idle T4', '0.2 min each'],
], [2.2, 0.5, 6.2, 1.1])}
{table(['Symbol crops', 'Train', 'Val', 'Test'], [
    ['crops (from plate crops)', f"{A['run']['token_crops']['train']:,} (5,465)", f"{A['run']['token_crops']['val']:,} (536)", f"{A['run']['token_crops']['test']:,} (467)"],
    ['classes present (of 101)', A['run']['classes_in_split']['train'], A['run']['classes_in_split']['val'], A['run']['classes_in_split']['test']],
], [3.0, 2.4, 2.2, 2.4])}
<p>One test class, {BN(A['run']['test_class_without_train'][0])} (Munshiganj), has no training crop, so no model can get it right.
The plate test uses the 651 human-typed OCR test crops (the same crops every team plate reader is scored on).</p>
<p><b>Differences from the paper:</b> our data instead of theirs (101 tokens instead of 17, 37x more training crops); GFP-GAN
is run as the network that actually acts on plates (Real-ESRGAN x2, via spandrel, FP16); a "two-line" order and a perfect
segmentation test were added. Software: PyTorch 2.11 + CUDA 12.8, Python 3.13 (Kaggle, 22 Oct 2026 03:38).</p>
"""

S['b1ok'] = f"""
<h1>8. Is the training OK?</h1>
{table(['Check', 'Result', 'OK?'], [
    ['Dataset version', M('dataset_version: 1.1') + ' in environment.json; folder BanglaLPR_AllInOne_v1_1; counts as expected', 'yes'],
    ['Jobs', f"all 5 ended with code=0; {A['run']['total_h']:.1f} h of the 12 h Kaggle limit", 'yes'],
    ['Data', f"{A['run']['token_crops']['train']:,} training crops = exactly the number we count in the v1.1 labels", 'yes'],
    ['CNN epochs', f"40 of 40; best validation at epoch {cc['best_epoch']} ({f1(cc['val_best_%'])} %), then unstable (Figure 2)", 'yes, use best'],
    ['ResNet-50 epochs', f"{rc['epochs_run']} of 30: stopped by the 3.2 h time cap; best epoch {rc['best_epoch']}, flat since about epoch 12", 'yes'],
    ['Repeatable', f"a local re-run of the plate test with the run's best.pt (CPU) gives {f2(SC['ocr_test']['D_local_none_fixed']['similarity_%'])} % similarity without GFP-GAN and {f2(V['p08_gt_boxes_exact_%'])} % exact with human boxes: identical to Kaggle", 'yes'],
    ['Code', 'the original-paper check with the same code gives about the paper\'s numbers on the paper\'s data (section 13)', 'yes'],
], [2.0, 6.8, 1.2])}
{fig('fig_curves.png', 470, 'Figure 2. Accuracy per epoch. Left: the paper\'s CNN recipe reaches its best validation accuracy at epoch 5, then training accuracy itself falls (92 % to 52 %): the optimisation becomes unstable. Right: ResNet-50 (only the last layer trains) rises and stays flat.')}
<div class="warn"><p><b>Why did the CNN collapse, and is that a problem?</b> The paper trains 40 epochs on 1,292 crops with batch 1, which is
51,680 weight updates in total. One epoch on our data is {A['run']['token_crops']['train']:,} updates, so our epoch 1 already equals
the paper's whole training, and epoch 40 is 37 times more. Batch 1 with Adam, no batch normalisation, no learning-rate
decay and very large kernels (22x22 on a 28x28 image) slowly drifts: the training loss rises from
{A['cnn_collapse']['train_loss_min']:.2f} (epoch 5) to {A['cnn_collapse']['train_loss_ep40']:.2f} (epoch 40) and training
accuracy falls too, so this is not over-fitting but instability. <b>It is not a bug in the run:</b> the kit saves the
best-validation checkpoint, and the plate test and speed test used that model. The "last epoch" row in the kit's
table ({f1(cc['acc_last_%'])} %) only shows that the paper's recipe does not scale; do not quote it as our accuracy.</p></div>
"""

S['b2'] = f"""
<h1>9. Results next to the paper</h1>
{table(['Metric', 'Paper (page)', 'Ours: CNN', 'Ours: ResNet-50'], [
    ['Classes', '17 (p.3)', '101', '101'],
    ['Symbol test accuracy, best-val checkpoint', '-', f"<b>{f2(cc['acc_best_%'])}</b>", f"<b>{f2(rc['acc_best_%'])}</b>"],
    ['Symbol test accuracy, last epoch', '92.38 / 87.50 (p.4)', f"{f2(cc['acc_last_%'])} (unstable)", f2(rc['acc_last_%'])],
    ['Macro F1, best-val (all classes in test)', '94.76 / 89.84 (p.4)', f2(cc['macroF1_best_%']), f2(rc['macroF1_best_%'])],
    ['Macro F1, 21 tokens with &gt;= 20 test crops', '-', f2(cc['macroF1_ge20_%']), f2(rc['macroF1_ge20_%'])],
    ['Weighted F1, best-val', '-', f2(cc['weightedF1_best_%']), f2(rc['weightedF1_best_%'])],
    ['Best validation accuracy', '98.80 (p.4)', f2(cc['val_best_%']), f2(rc['val_best_%'])],
    ['Classifier speed, batch 1, T4', 'not reported', f"{cc['fps']} FPS ({cc['ms']} ms)", f"{rc['fps']} FPS ({rc['ms']} ms)"],
    ['Training time', 'not reported', f"{cc['train_min']} min", f"{rc['train_min']} min (25 epochs)"],
], [3.6, 2.2, 2.1, 2.1])}
{table(['Whole plates (CNN, 651 test plates unless noted)', 'Paper (page)', 'Ours'], [
    ['Mean similarity, GFP-GAN + x order (the paper\'s exact recipe)', '88 % on 30 plates (p.5)', f"{f2(P['gfpgan_paper']['similarity_%'])} %"],
    ['Plates with similarity &gt;= 0.95 / &gt;= 0.90', '57 % / 80 % (p.5)', f"{f2(P['gfpgan_paper']['ge95_%'])} % / {f2(P['gfpgan_paper']['ge90_%'])} %"],
    ['Mean similarity WITHOUT GFP-GAN, x order', 'never tested', f"{f2(P['none_paper']['similarity_%'])} %"],
    ['Mean similarity, GFP-GAN + two-line order', '-', f"{f2(pf['similarity_%'])} %"],
    ['Plate exact, GFP-GAN + two-line order', 'not reported', f"<b>{f2(pf['plate_exact_%'])} %</b> (1 plate)"],
    ['Plate exact, no GFP-GAN + two-line order', '-', f"{f2(pn['plate_exact_%'])} %"],
    ['Char accuracy (1 - CER), GFP-GAN + two-line order', '-', f"{f2(pf['char_acc_%'])} %"],
    ['Perfect segmentation (human boxes, 467 plates): exact / char acc.', '-', f"{f2(P['gt_boxes']['plate_exact_%'])} % / {f2(P['gt_boxes']['char_acc_%'])} %"],
    ['Time per plate: GFP-GAN / segmentation + CNN', 'not reported', f"{f1(A['plates_ms']['enhance'])} ms / {f1(A['plates_ms']['read'])} ms"],
], [5.0, 2.5, 2.5])}
<div class="box"><p><b>How to read the comparison.</b> The paper used its own 17 easy classes and a leaky test; we use 101
tokens on a grouped, duplicate-free split. So the classifier rows show direction and size, not a race. The plate rows
are different in kind: the paper scores 30 synthetic-looking plates with a forgiving similarity; we score 651 real
plates and also report the strict exact match. The comparison on the paper's <i>own</i> data is section 13.</p></div>
"""

conf = cc['top_confusions'][:6]
S['b3'] = f"""
<h1>10. The symbol classifier: good on common tokens, weak on rare ones</h1>
{fig('fig_groups.png', 460, 'Figure 3. Test accuracy per token group (best-validation checkpoints). Both models read digits and মেট্রো well; letters and district words are harder; the 39 rare tokens (1-19 test crops each) are where the CNN fails and the pretrained ResNet-50 helps.')}
{ul([
    f"<b>Digits carry the score:</b> {g8['digits']['n']:,} of the {A['run']['token_crops']['test']:,} test crops are digits ({100 * g8['digits']['n'] / A['run']['token_crops']['test']:.0f} %), read {f1(g8['digits']['acc_%'])} % right by the CNN.",
    f"<b>Why macro F1 is only {f1(cc['macroF1_best_%'])} %:</b> {cc['classes_with_test_crops']} classes appear in the test set; {cc['classes_rare_test(1-19)']} of them have 1-19 crops and the CNN never gets {cc['rare_classes_with_0_correct']} of these right. On the 21 tokens with &gt;= 20 test crops ({cc['test_crops_in_ge20']:,} crops, 97 % of the test) macro F1 is {f1(cc['macroF1_ge20_%'])} %.",
    f"<b>ResNet-50 vs CNN:</b> about equal overall ({f1(rc['acc_best_%'])} vs {f1(cc['acc_best_%'])} %), but ResNet-50 is better on letters (+{g5['series letters']['acc_%'] - g8['series letters']['acc_%']:.1f}), words (+{g5['district words']['acc_%'] - g8['district words']['acc_%']:.1f}) and rare tokens ({f1(rc['acc_rare_%'])} vs {f1(cc['acc_rare_%'])} %), because its ImageNet features need fewer examples. The CNN is 4.4x faster ({cc['fps']} vs {rc['fps']} FPS).",
    f"<b>Typical CNN mistakes</b> (count): " + '; '.join(f"{BN(c['true'])} read as {BN(c['pred'])} ({c['n']})" for c in conf) +
    ". Rare district words fall back to the most common word, ঢাকা; similar Bangla digits (৯/১, ০/৩) and ট/৮ are mixed up.",
])}
<p><b>Compared with the paper:</b> 93 % on 101 tokens with a clean split is at least as good as 92.38 % on 17 tokens with a
leaky test. The per-symbol CNN idea is not the weak part of the paper. The weak part comes next.</p>
"""

S['b4'] = f"""
<h1>11. Whole plates: where the reader breaks</h1>
<p>The plate reader is only as good as its weakest step. We split the loss into its parts on the 467 test plates that have
human symbol boxes ({R['tokens']:,} symbols), using the run's CNN locally ({M('report_src/seg_check.py')}).</p>
{fig('fig_where.png', 470, 'Figure 4. Left: the CNN reads 92.9 % of symbols cut by human boxes, 82.4 % when the same boxes are cut from the paper\'s black-and-white image, and the contours find only 52.4 % of the symbols at all. Right: whole plates read exactly. Paper #2\'s character detector (same v1.1 data) reads 88.9 % of the same 467 plates.')}
{table(['Step', 'What we measured', 'Loss'], [
    ['1. Cut the plate into symbols (Otsu + 15 largest contours)', f"a contour matches a symbol (IoU &gt;= 0.5) for <b>{f1(R['C_tokens_found_iou50_all15_%'])} %</b> of symbols; all symbols found on only {f1(R['C_plates_all_tokens_found_%'])} % of plates; {R['C_blobs_per_plate_all15']:.1f} blobs per plate for {R['C_tokens_per_plate']:.1f} symbols", 'largest'],
    ['2. Feed black-and-white blobs to a CNN trained on colour crops', f"{f1(R['A_colour_crops_acc_%'])} % → {f1(R['B_binary_blobs_gt_boxes_acc_%'])} % with perfect boxes (letters {f1(R['per_kind']['letter']['A_%'])} → {f1(R['per_kind']['letter']['B_%'])}, words {f1(R['per_kind']['district']['A_%'])} → {f1(R['per_kind']['district']['B_%'])})", '-10.5 points'],
    ['3. Classify the blobs that do match a symbol', f"{f1(R['C_cnn_on_matched_blobs_acc_%'])} % of {R['C_matched_blobs']:,} matched blobs right", 'small'],
    ['4. Order the symbols', f"x order {f1(P['none_paper']['similarity_%'])} % similarity, two-line order {f1(pn['similarity_%'])} %, our BRTA decoder {f1(SC['ocr_test']['E_brta_decoder_on_fixed_tokens']['similarity_%'])} %; exact stays 0", 'cannot fix 1-2'],
    ['5. Extra blobs', f"after the size filter the reader outputs {SG['none_fixed']['mean_pred_tokens']:.1f} symbols per plate for {SG['none_fixed']['mean_true_tokens']:.1f} true ones; only {f1(SG['none_fixed']['same_count_%'])} % of plates get the right count, {f1(SG['none_fixed']['too_many_%'])} % too many", 'garbage symbols'],
], [3.0, 5.8, 1.2])}
"""

ex = A['examples']
exrow = lambda e: [BN(e['true']), BN(e['none_paper']['pred']) + f" ({e['none_paper']['sim']:.2f})",
                   BN(e['gfpgan_fixed']['pred'] or '(nothing)') + f" ({e['gfpgan_fixed']['sim']:.2f})"]
S['b4b'] = f"""
{fig('fig_contours.png', 400, 'Figure 5. What the contours see on three test plates (left: the crop; right: the paper\'s 333x335 Otsu image). Orange: the 15 largest contours; green dashed: the human symbol boxes. Contours lock onto the plate frame, holes inside letters, bolts and background; joined or broken Bangla words are missed.')}
<h3>Four real plates (similarity in brackets)</h3>
{table(['Truth', 'Paper recipe without GFP-GAN, x order', 'GFP-GAN + two-line order'], [exrow(e) for e in ex], [2.6, 3.7, 3.7])}
<p class="small">Rows: the best plate (the only exact one), a plate at the 75th percentile, the median plate, and a bad night
plate. Even the median plate scores about 0.5 similarity while being completely unreadable: similarity hides failure.</p>
<div class="box"><p><b>Bottom line for section 11.</b> Classic contour segmentation does not work on real Bangladeshi plates
(two lines, joined Bangla words with মাত্রা, frames, bolts, tilt, glare, night). Even if segmentation were perfect, a
per-symbol classifier reads {f1(V['p08_gt_boxes_exact_%'])} % of plates, while paper #2's character detector, trained on the
same v1.1 data, reads {f1(V['p02_char_yolo_exact_%'])} % of the same {V['plates']} plates: it is right on 144 plates where the
CNN is wrong and wrong on only 7 where the CNN is right (McNemar p &lt; 0.0001). A detector finds and names each symbol
in one step, so there is no separate cutting step to fail.</p></div>
"""

S['b5'] = f"""
<h1>12. GFP-GAN: does restoration help, and what does it cost?</h1>
<p>This is the question the paper never answers. We ran the same 651 plates with and without the restoration step and the
same CNN, and compared every plate with itself.</p>
{fig('fig_gfpgan.png', 450, 'Figure 6. Left: mean similarity per condition, two-line order, with and without GFP-GAN (dashed: the paper\'s 88 % on 30 plates). Right: time per plate on a Tesla T4. GFP-GAN alone takes 34 times longer than segmentation and CNN together.')}
{table(['Paired result (651 plates)', 'Without', 'With GFP-GAN', 'Difference'], [
    ['Mean similarity, two-line order', f"{f2(fx['mean_none_%'])} %", f"{f2(fx['mean_gfpgan_%'])} %", f"+{f2(fx['mean_diff_points'])} points; {fx['gfpgan_better']} plates better, {fx['gfpgan_worse']} worse, {fx['equal']} equal; Wilcoxon p = {fx['wilcoxon_p']:.1e}"],
    ['Mean similarity, x order (paper)', f"{f2(GP['paper']['mean_none_%'])} %", f"{f2(GP['paper']['mean_gfpgan_%'])} %", f"+{f2(GP['paper']['mean_diff_points'])} points; p = {GP['paper']['wilcoxon_p']:.1e}"],
    ['Plates read exactly', f"{f2(pn['plate_exact_%'])} % (0)", f"{f2(pf['plate_exact_%'])} % (1)", '+1 plate'],
    ['Night (65 plates), similarity', f"{f2(SB['night']['none_fixed_sim_%'])} %", f"{f2(SB['night']['gfpgan_fixed_sim_%'])} %", f"{SB['night']['gfpgan_fixed_sim_%'] - SB['night']['none_fixed_sim_%']:+.1f} points (worse)"],
    ['Blurry (112 plates), similarity', f"{f2(SB['blurry']['none_fixed_sim_%'])} %", f"{f2(SB['blurry']['gfpgan_fixed_sim_%'])} %", f"{SB['blurry']['gfpgan_fixed_sim_%'] - SB['blurry']['none_fixed_sim_%']:+.1f} points"],
    ['Time per plate (T4)', f"{f1(A['plates_ms']['read'])} ms", f"{f1(A['plates_ms']['read'] + A['plates_ms']['enhance'])} ms", f"{(A['plates_ms']['read'] + A['plates_ms']['enhance']) / A['plates_ms']['read']:.0f}x slower in total; GFP-GAN alone allows at most {1000 / A['plates_ms']['enhance']:.1f} plates/s on a T4"],
], [2.6, 1.4, 1.5, 4.5])}
{fig('fig_enh.png', 290, 'Figure 7. Two test plates before and after "GFP-GAN" (its Real-ESRGAN x2 step). Edges get sharper and the image doubles in size, but no stroke that is missing in the original comes back, and the Otsu and contour steps that follow stay the same.')}
<div class="box"><p><b>Verdict.</b> Restoration gives a small, statistically real gain in partial similarity, mostly on clear
and blurry daylight plates, but it does not turn unreadable plates into readable ones (1 extra exact plate in 651), it
hurts at night, and it costs about 0.2 s per plate on a GPU, which alone breaks our 10 FPS laptop target (RQ3). The
problem here is segmentation, which restoration does not fix. Any new upscaling test belongs in front of paper #2's
detector, for small or blurry plates only (section 20).</p></div>
"""

CAP8 = ("Figure 8. The authors' own test crops, last epoch as the paper reports (whiskers: spread of 3 seeds). Left: paper "
        f"vs ours. Right: our CNN (28x28) by subset; \"clean\" = the {CN['n_clean']} test crops that do not copy a training "
        "crop; all region-word test crops are copies.")
S['b6'] = f"""
<h1>13. Original-paper check: the paper's own data with our code</h1>
<p>To be sure our port is right, we ran the kit's second notebook ({M('paper08_original_check.ipynb')}, the same commands)
locally on 24 Oct 2026 on an RTX 5060 Ti: the authors' {M('Data.zip')} (pinned to commit 8dc72c7, checksum-verified), their
split and settings, 3 seeds per model. The synthetic plate test of that notebook needs a Kaggle input and was not run.</p>
{fig('fig_check.png', 470, CAP8)}
{table(['On the authors\' data', 'Paper (page)', 'Ours, 3 seeds'], [
    ['CNN test accuracy, last epoch', '92.38 (p.4)', f"<b>{msd(CN['acc_last'], 2)}</b>"],
    ['CNN test accuracy, best-val checkpoint', '- (authors\' notebook: 93.40)', msd(CN['acc_best'], 2)],
    ['CNN macro precision / recall / F1, last epoch', '96.67 / 91.62 / 94.76 (p.4)', f"{CN['P_last'][0]:.1f} / {CN['R_last'][0]:.1f} / {CN['F1_last'][0]:.1f}"],
    ['CNN best validation accuracy', '98.80 (p.4)', msd(CN['val_best'], 2)],
    [f"CNN on the {CN['n_clean']} clean test crops, last epoch", 'not checked', msd(CN['acc_clean_last'], 2)],
    ['CNN digits / series letters / region words, last epoch', '-', f"{CN['digits_last'][0]:.1f} / {CN['letters_last'][0]:.1f} / {CN['words_last'][0]:.1f}"],
    [f"CNN series letters, clean crops only ({CN['n_letters_clean']})", '-', msd(CN['letters_clean_last'])],
    ['CNN at 32x32 (the paper\'s text), 1 seed, last epoch', '92.38 (p.3-4)', f2(CN32['acc_last'][0])],
    ['ResNet-50 test accuracy, last epoch', '87.50 (p.4; notebook 73.93)', msd(RN['acc_last'], 2)],
    ['ResNet-50 test accuracy, best-val checkpoint', '-', msd(RN['acc_best'], 2)],
    ['ResNet-50 with ImageNet normalisation (our fix), 1 seed, last / best-val', '-', f"{f2(RNN['acc_last'][0])} / {f2(RNN['acc_best'][0])}"],
    ['Training time per run (RTX 5060 Ti): CNN / ResNet-50', 'not reported', f"{CN['train_min'][0]:.1f} / {RN['train_min'][0]:.1f} min"],
], [4.6, 2.6, 2.8])}
<div class="good"><p><b>What the check shows.</b> (1) <b>Our port is the paper's method.</b> ResNet-50 gives
{msd(RN['acc_last'])} % (paper 87.50 %); the CNN at 32x32, the size the paper's text names, gives {f1(CN32['acc_last'][0])} %
(one seed; paper 92.38 %); at 28x28, the size in the authors' code, {msd(CN['acc_last'])} %. So the code we ran on our data
is not a broken copy. (2) <b>The leak inflates the score:</b> on the {CN['n_clean']} clean test crops the CNN drops by
{CN['acc_last'][0] - CN['acc_clean_last'][0]:.1f} points ({f1(CN['acc_clean_last'][0])} %) and on series letters by
{CN['letters_last'][0] - CN['letters_clean_last'][0]:.1f} points ({f1(CN['letters_last'][0])} → {f1(CN['letters_clean_last'][0])} %,
{CN['n_letters_clean']} clean crops). The three region words score 100 %, but every one of their test crops copies a training
crop: that 100 % is memory. (3) <b>Their validation set cannot choose a model:</b> {f1(CN['val_best'][0])} % validation accuracy,
because 3 of 4 validation crops copy training crops. (4) <b>One seed cannot rank the models:</b> last-epoch accuracy moves by up
to {max(v['acc_last'] for k, v in CK['per_run'].items() if k.startswith('resnet50_')) - min(v['acc_last'] for k, v in CK['per_run'].items() if k.startswith('resnet50_')):.0f}
points between seeds (ResNet-50 {min(v['acc_last'] for k, v in CK['per_run'].items() if k.startswith('resnet50_')):.1f} to
{max(v['acc_last'] for k, v in CK['per_run'].items() if k.startswith('resnet50_')):.1f} %), so Table II's single numbers (92.38 vs 87.50) do not show
that the CNN is better.</p></div>
<p class="small">On their small set (1,292 crops) the batch-1 CNN does not collapse: best epochs {', '.join(map(str, CN['best_epoch']))} of 40. This
supports the explanation in section 8: the recipe is fine for 52,000 updates and unstable for 1.9 million. Our earlier CPU
check (10 Oct, one seed, 28x28: 91.4 % last epoch) lies inside this spread. Kaggle run still to do: section 20.</p>
"""

S['b7'] = f"""
<h1>14. Findings, and the numbers to quote</h1>
{table(['#', 'Finding', 'Number to quote', 'Section'], [
    ['1', 'The v1.1 run is healthy; the CNN recipe is unstable on 48k crops, so use the best-val model', f"5/5 jobs code=0; best epoch {cc['best_epoch']}; local re-run identical", '8'],
    ['2', 'The symbol classifier works on common tokens', f"CNN {f1(cc['acc_best_%'])} % of {A['run']['token_crops']['test']:,} symbols; {f1(cc['acc_ge20_%'])} % on 21 common tokens", '10'],
    ['3', 'Rare tokens are the classifier\'s weak point', f"CNN {f1(cc['acc_rare_%'])} %, ResNet-50 {f1(rc['acc_rare_%'])} % on 39 rare tokens", '10'],
    ['4', 'The paper\'s plate reader fails on real plates', f"{f2(pf['plate_exact_%'])} % exact (1 of 651); {f1(pf['similarity_%'])} % similarity (paper 88 %)", '9, 11'],
    ['5', 'Segmentation is the cause', f"contours find {f1(R['C_tokens_found_iou50_all15_%'])} % of symbols; all of them on {f1(R['C_plates_all_tokens_found_%'])} % of plates", '11'],
    ['6', 'Per-symbol classification loses to a character detector even with perfect boxes', f"{f1(V['p08_gt_boxes_exact_%'])} vs {f1(V['p02_char_yolo_exact_%'])} % on the same {V['plates']} plates (144 vs 7, p &lt; 0.0001)", '11'],
    ['7', 'GFP-GAN: small similarity gain, no real exact gain, worse at night, very slow', f"+{f1(fx['mean_diff_points'])} points (p = {fx['wilcoxon_p']:.0e}); +1 exact plate; {f1(A['plates_ms']['enhance'])} ms/plate", '12'],
    ['8', 'The paper reproduces on its own (leaky) data with our code', f"ResNet-50 {f1(RN['acc_last'][0])} % (paper 87.50); CNN 32x32 {f1(CN32['acc_last'][0])} % (paper 92.38); 48/303 test crops copied", '3, 13'],
], [0.4, 4.0, 4.3, 0.9])}
<h3>What we keep and what we drop</h3>
{table(['From paper #8', 'Decision', 'Why'], [
    ['GFP-GAN / Real-ESRGAN restoration in front of the reader', '<b>drop</b> from the main pipeline', 'finding 7; breaks RQ3 speed'],
    ['Otsu + contour segmentation', '<b>drop</b>', 'findings 4-6'],
    ['Per-symbol CNN classifier', 'keep only as an idea for a second opinion on single symbols', 'finding 2; 760 FPS'],
    ['17-class Bangla symbol crops (Data.zip)', 'do not mix into training (leaky, synthetic, 17 classes)', 'section 3'],
    ['Similarity as the main metric', '<b>replace</b> with plate exact + CER (similarity as an extra)', 'section 11: 0.5 similarity = unreadable'],
    ['The restoration question', '<b>keep</b> as an ablation (proposal: "x2 upscaling")', 'section 20'],
], [3.4, 3.4, 3.2])}
"""

# ======================================================================================== part C (use it)
S['c1'] = f"""
<p class="part">PART C · USE IT</p>
<h1>15. How to use this paper in our research</h1>
{table(['Our research part', 'How paper #8 is used', 'Evidence we now have (v1.1)'], [
    ['RQ2 / O3: character-level recognizer', 'the "segment, then classify" baseline that motivates a character <i>detector</i>', f"same 467 plates: per-symbol CNN with perfect boxes {f1(V['p08_gt_boxes_exact_%'])} % vs char-YOLO {f1(V['p02_char_yolo_exact_%'])} %; real contours 0.15 %"],
    ['RQ4 / G4: enhancement ablation ("CLAHE, x2 upscaling" in the proposal)', 'the first measured data point for x2 upscaling', f"+{f1(fx['mean_diff_points'])} similarity points, +1 exact plate, night {SB['night']['gfpgan_fixed_sim_%'] - SB['night']['none_fixed_sim_%']:+.1f}"],
    ['RQ3 / O4: real time (&gt;= 10 FPS)', 'restoration cost reference', f"GFP-GAN {f1(A['plates_ms']['enhance'])} ms per plate on a T4 = at most {1000 / A['plates_ms']['enhance']:.1f} plates/s"],
    ['RQ1 / O1: duplicate-free split', 'a third leaky dataset next to KUET (#23) and BLPD (#2)', '48 of 303 test and 320 of 429 val crops copied from train'],
    ['O5: evaluation protocol', 'why we report plate exact, not only similarity', f"{f1(pf['similarity_%'])} % mean similarity while {f2(pf['plate_exact_%'])} % exact"],
    ['Method chapter (training)', 'why we keep the best-validation checkpoint and use early stopping', f"CNN last epoch {f1(cc['acc_last_%'])} % vs best-val {f1(cc['acc_best_%'])} %"],
    ['Related work', 'the restoration-based Bangla system (already cited as [afrin2023] in the proposal)', 'paragraph below'],
], [2.6, 3.6, 3.8])}

<h1>16. What benefit do you get from it?</h1>
{ul([
    '<b>A clear answer to an open question.</b> "Does restoration help Bangla plate reading?" now has a measured, paired answer with a cost, which the paper itself does not give.',
    '<b>A negative result with a mechanism.</b> You can say <i>why</i> the pipeline fails (contours find half the symbols), not only that it fails. Examiners value that.',
    '<b>Evidence for our design.</b> The same-plates comparison with paper #2 (88.9 vs 59.5 %, p &lt; 0.0001) justifies the character-detector design of our system with a number.',
    '<b>Proof that our code is right.</b> On the authors\' data our port gives about their numbers; locally re-running the plate test gives the Kaggle numbers exactly.',
    '<b>A metric lesson for the whole team:</b> partial-credit similarity can look like 50 % on plates that are completely wrong.',
    '<b>A training lesson:</b> a batch-1 recipe that works on 1,292 crops collapses on 48,000; best-val checkpoints protect every kit run.',
    '<b>Reusable pieces:</b> a fast symbol classifier (760 FPS) and a token crop set of 47,992 training crops in 101 classes, made by the kit from our labels.',
])}
<h3>A paragraph you can adapt for the related-work chapter</h3>
<pre>Afrin et al. [ref] restore Bangla plate images with GFP-GAN, segment symbols with
Otsu thresholding and contours, and classify them with a small CNN, reporting 92.38%
symbol accuracy on 17 classes and 88% mean string similarity on 30 plates. Their
split is leaky (48 of 303 test crops copy a training crop) and no result without
restoration is given. Re-implemented on our 101-token benchmark, the classifier reads
93.1% of symbols, but contour segmentation finds only 52% of the characters, and the
full pipeline reads 0.15% of 651 real plates exactly. Restoration (on plates, GFP-GAN
reduces to Real-ESRGAN x2) adds 4.8 similarity points at 34 times the reading cost,
and even with perfect symbol boxes the per-symbol approach reads 59.5% of plates
versus 88.9% for a character detector on the same plates.</pre>
"""

S['c2'] = f"""
<h1>17. What to show the faculty (8 slides)</h1>
{table(['#', 'Slide title', 'What is on it', 'Source in this report'], [
    ['1', 'The paper and its claim', 'blurry plates; GFP-GAN + contours + CNN; 92.38 % symbols, 88 % similarity on 30 plates', 'sections 1, 5'],
    ['2', 'The pipeline', 'the 6 steps, with our measured number under each', 'Figure 1'],
    ['3', 'We reproduced it on their data', f"ResNet-50 {f1(RN['acc_last'][0])} % (paper 87.50), CNN {f1(CN32['acc_last'][0])} % (paper 92.38); but 48 of 303 test crops are copies", 'Figure 8, section 3'],
    ['4', 'On our 101 tokens: the classifier is fine', f"{f1(cc['acc_best_%'])} % of symbols; rare tokens are the gap", 'Figure 3'],
    ['5', 'Whole plates fail', f"{f2(pf['plate_exact_%'])} % exact on 651 real plates; similarity hides it (example table)", 'section 9, 11'],
    ['6', 'Why: segmentation', f"contours find {f1(R['C_tokens_found_iou50_all15_%'])} % of symbols; perfect boxes {f1(V['p08_gt_boxes_exact_%'])} % vs detector {f1(V['p02_char_yolo_exact_%'])} %", 'Figures 4, 5'],
    ['7', 'GFP-GAN: the missing ablation', f"+{f1(fx['mean_diff_points'])} similarity points, +1 exact plate, worse at night, {speedup:.0f}x the time", 'Figure 6'],
    ['8', 'What we take for our system', 'drop restoration and contours; keep the character detector; report plate exact; upscaling as an ablation', 'section 14 table'],
], [0.3, 2.2, 5.0, 2.1])}
<div class="box"><p><b>Keep it honest:</b> call the 93 % "symbol accuracy on our split" and never compare it as a race with 92.38 %
(different classes and test). Say "same plates" only for the comparison with paper #2. Do not show the CNN's last-epoch
36.7 % without its explanation (section 8). The original check was run on our own PC, not on Kaggle; say so if asked.</p></div>

<h1>18. What to explain: a 3-minute script</h1>
<p>"This 2023 ICCIT paper tries to read blurry Bangladeshi plates. It first sharpens the plate with GFP-GAN, then cuts it into
symbols with a threshold and contours, and classifies each symbol with a small CNN. It reports 92 percent symbol accuracy
and 88 percent similarity on 30 plates, but it never tests without GFP-GAN.</p>
<p>We first ran their code on their own data: we get their numbers, 87.5 percent for ResNet-50 and 93 percent for the CNN,
so our port is correct. But 48 of their 303 test images are copies of training images, so their score is optimistic.</p>
<p>Then we trained the same models on our benchmark with 101 Bangla tokens. The classifier itself is good: 93 percent of
symbols. But on 651 real plates the full pipeline reads only one plate exactly. We found why: the contour step finds only
about half of the characters. It locks onto the plate frame, the holes inside letters and the bolts, and it breaks
joined Bangla words. Even if we give it perfect human boxes, it reads 60 percent of plates, while our character detector
from paper 2 reads 89 percent of the same plates.</p>
<p>Finally the missing test: with and without GFP-GAN on the same plates. GFP-GAN raises similarity by about 5 points, but it
makes only one more plate fully correct, it is worse at night, and it takes 0.2 seconds per plate, 34 times the rest of
the pipeline. So in our system we drop restoration and contour segmentation, keep the character detector, and report
exact plate accuracy, not only similarity."</p>
"""

S['c3'] = f"""
<h1>19. Questions the faculty may ask</h1>
<p class="q">Q1. Your CNN's last-epoch accuracy is 36.7 %. Did training fail?</p>
<p>No. The paper's recipe (batch 1, Adam, no normalisation) became unstable after epoch 5 on our 48,000 crops; one of our
epochs equals the paper's whole training. We keep the best-validation checkpoint ({f1(cc['acc_best_%'])} % test), as
every kit run does, and all plate results use it.</p>
<p class="q">Q2. Is your 93 % better than their 92.38 %?</p>
<p>It is not a fair race: 101 tokens on a clean split vs 17 tokens on a leaky test. It shows the classifier idea works at
our scale. On their own data our code gives their numbers: ResNet-50 {msd(RN['acc_last'])} % (3 seeds; paper 87.50 %),
CNN {f1(CN32['acc_last'][0])} % at 32x32 (paper 92.38 %).</p>
<p class="q">Q3. They report 88 %, you get almost 0 %. Why so different?</p>
<p>Three reasons: (1) similarity vs exact (they never report exact plates; our similarity is {f1(pf['similarity_%'])} %); (2)
their 30 test plates come from the clean synthetic set their symbols were cut from, ours are 651 real photos with night,
blur and tilt; (3) their expected strings use the same x order as the reader, which hides line-mixing errors.</p>
<p class="q">Q4. Maybe your implementation is wrong?</p>
<p>The same code reproduces their symbol accuracy on their data (section 13), the plate pipeline follows their notebook
step by step (Figure 1), and a local re-run gives the Kaggle numbers exactly. The failure is measured step by step: the
contours find {f1(R['C_tokens_found_iou50_all15_%'])} % of the symbols, and we can show it on images (Figure 5).</p>
<p class="q">Q5. So does GFP-GAN help or not?</p>
<p>A little, on similarity: +{f1(fx['mean_diff_points'])} points, 343 plates better and 224 worse, Wilcoxon p &lt; 10<sup>-10</sup>.
Not on exact reading (1 extra plate), not at night ({SB['night']['gfpgan_fixed_sim_%'] - SB['night']['none_fixed_sim_%']:+.1f}),
and at {f1(A['plates_ms']['enhance'])} ms per plate it is too slow for real time.</p>
<p class="q">Q6. You ran Real-ESRGAN, not GFP-GAN. Is that the same?</p>
<p>On a plate, yes. GFP-GAN v1.3 detects faces, restores them, and upsamples the rest of the image with Real-ESRGAN x2. A plate
has no face, so the output is the Real-ESRGAN x2 image. We ran exactly that network (RealESRGAN_x2plus, the one the authors' settings load).</p>
<p class="q">Q7. Why is macro F1 only 36 %?</p>
<p>Macro F1 averages over classes. 39 rare tokens have 1-19 test crops each and are mostly wrong; on the 21 common tokens
macro F1 is {f1(cc['macroF1_ge20_%'])} %. Rare districts are the real gap; the pretrained ResNet-50 does better on them.</p>
<p class="q">Q8. Could better segmentation fix the pipeline?</p>
<p>Only partly. With perfect human boxes the per-symbol CNN still reads {f1(V['p08_gt_boxes_exact_%'])} % of plates vs
{f1(V['p02_char_yolo_exact_%'])} % for the character detector on the same plates. Learning to find and name symbols together works better.</p>
<p class="q">Q9. Why compare with paper #2 at all?</p>
<p>Both were trained on the same v1.1 data and tested on the same plates by the same team code, so the comparison is
paired and fair (McNemar: 144 plates only the detector reads, 7 only the CNN reads).</p>
<p class="q">Q10. What will you do with restoration now?</p>
<p>Keep it as an ablation in our plan (x2 upscaling and CLAHE), but in front of the character detector and only for small
or blurry plates, and report exact accuracy and milliseconds together.</p>
"""

S['c4'] = f"""
<h1>20. Next steps (in order)</h1>
{ul([
    f'<b>Upload</b> (section 21) and fill the team table: {ADD} in {M("docs/results")} (done for you on 24 Oct; check the notes\' results block).',
    f'<b>Run {M("paper08_original_check.ipynb")} on Kaggle</b> (about 1 h) with the input {M("furcifer/bangla-license-plates-synthetic")}. It makes section 13 official and adds the one test we could not run: the plate test on the synthetic plates the authors cut their symbols from (the closest thing to their 30 plates).',
    '<b>Upscaling ablation done right:</b> bicubic x2, Real-ESRGAN x2 and CLAHE in front of paper #2\'s character detector, on the small-plate and blurry subsets; report plate exact, CER and ms per plate.',
    '<b>Rare tokens:</b> count district words with &lt; 50 training crops and plan targeted data (or synthetic plates) for them; this limits every reader, not only #8.',
    '<b>If a symbol classifier is ever reused</b> (e.g. as a second opinion for low-confidence district words): batch 64, learning-rate decay, early stopping, class-balanced sampling, ImageNet normalisation.',
    '<b>Board and notes:</b> paste the code link and the one-line result (section 15 paragraph, short form) on the research board.',
], 'ol')}

<h1>21. GitHub and Drive</h1>
<p>{M('After Train/results.zip')} (535 MB) is the whole Kaggle output; do not upload it. The upload folder is ready at
{M('After Train/github_upload/weights/paper08/MD.Rokib.Hasan.Oli/')} (same layout as paper #2 and #23):</p>
{table(['File', 'What'], [
    [M('README.md'), 'results, model cards, quick start (tested), checksums'],
    [M('best_cnn28.pt'), '14.8 MB, the paper CNN trained on our 101 tokens (best-val epoch 5)'],
    [M('best_resnet50_head.pt'), '20.7 MB, the trained head of ResNet-50. The full best.pt is 115 MB (over GitHub\'s limits), but its backbone is frozen ImageNet ResNet-50, checked bit-identical, so head + torchvision rebuild the exact model (README)'],
    [M('paper08_results.zip'), '0.14 MB, all results, logs, per-plate predictions and code, no weights'],
    [M('results_vs_paper.md'), 'the kit\'s comparison table'],
    [M('Paper08_Study_Report.pdf'), 'this report'],
    [M('local_checks.json'), 'the numbers of sections 11 and 13 (local checks)'],
], [3.0, 7.0])}
<p>Drive {M('498R / weights / paper08 /')}: {M('After Train/for_drive/paper08_outputs.zip')} (230 MiB: both models with best.pt and last.pt, full size).
Web upload on GitHub: open {M('weights')} → Add file → Upload files → drag the {M('paper08')} folder → commit message
"Add paper #8 results on dataset v1.1 (Rokib)". Every file is under 25 MB.</p>

"""

S['c5'] = f"""
<h1>Appendix: where every number comes from</h1>
{table(['Number', 'Source'], [
    ['paper numbers', 'arXiv 2312.10701, pages given in the text; paper-vs-code facts from the notes (repository commit 8dc72c7)'],
    ['v1.1 training, classifier, plate and speed results', M('After Train/results.zip') + ' → ' + M('runs/*/results.json') + ', ' + M('log.csv') + ', ' + M('per_plate.csv') + '; ' + M('paper-8.log')],
    ['token groups, rare classes, confusions, paired GFP-GAN test, counts per plate', M('report_src/analysis.py') + ' → ' + M('analysis.json')],
    ['segmentation diagnosis, binary-blob test, BRTA decoder, same plates vs paper #2', M('report_src/seg_check.py') + ' → ' + M('seg_check.json') + ' (CPU, the run\'s best.pt, dataset v1.1; paper #2 predictions from its v1.1 results.zip)'],
    ['original-paper check', M('report_src/check_collect.py') + ' → ' + M('check_local.json') + ' (local run of the kit commands, RTX 5060 Ti, 24 Oct 2026)'],
    ['upload weights', M('report_src/make_upload_weights.py') + ' (backbone check, head file)'],
], [3.6, 6.4])}
<pre>@inproceedings{{afrin2023bengali,
  title     = {{Bengali License Plate Recognition: Unveiling Clarity
               with {{CNN}} and {{GFP-GAN}}}},
  author    = {{Afrin, Noushin and Hasan, Md Mahamudul and Safin, Mohammed
               Fazlay Elahi and Amin, Khondakar Rifat and Haque, Md Zahidul
               and Ahmed, Farzad and Shawon, Md Tanvir Rouf}},
  booktitle = {{2023 26th International Conference on Computer and
               Information Technology (ICCIT)}},
  address   = {{Cox's Bazar, Bangladesh}},
  year      = {{2023}},
  note      = {{arXiv:2312.10701}}
}}</pre>
"""

FLOWS = [['cover'], ['a1'], ['a2'], ['a3'], ['b1'], ['b1ok'], ['b2'], ['b3'], ['b4'], ['b4b'], ['b5'], ['b6'], ['b7'],
         ['c1'], ['c2'], ['c3'], ['c4'], ['c5']]


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
        pg.insert_text((ML, PH - 26), 'Paper #8 study report · dataset v1.1 run · 498R BanglaLPR', fontsize=7.5,
                       fontname='helv', color=(0.45, 0.45, 0.43))
        t = f'{i + 1} / {n}'
        pg.insert_text((PW - ML - fitz.get_text_length(t, 'helv', 7.5), PH - 26), t, fontsize=7.5, fontname='helv', color=(0.45, 0.45, 0.43))
    doc.set_metadata({'title': 'Paper #8 study report (dataset v1.1): Bengali LPR with CNN and GFP-GAN',
                      'author': 'Prepared for MD. Rokib Hasan Oli',
                      'subject': 'Paper study, v1.1 Kaggle results, segmentation and GFP-GAN analysis, original check, faculty guide'})
    doc.save(OUT, garbage=3, deflate=True)
    doc.close()
    print('pages', n, OUT)


if __name__ == '__main__':
    build()
