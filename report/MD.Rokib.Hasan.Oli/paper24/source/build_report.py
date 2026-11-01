"""Paper #24 study report, v1.1 update -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Laroca et al., CIARP 2023, author version arXiv 2309.04331v2, re-read 24 Oct 2026), the v1.1 Kaggle
run of 22 Oct 2026 (`After Train/results.zip`, log `After Train/paper-24.log`), analysis.json (analysis.py: per-plate
re-analysis of the saved predictions, plus the v1.0 numbers from v1.0_run/analysis.json), local_check.json
(local_check.py: the GitHub weights re-scored on this PC), the 498R proposal (RQ1-RQ4, O1-O5) and the paper notes.
Order: analysis.py -> local_check.py -> charts.py -> build_report.py. The first report (v1.0 run) and its sources are
kept in v1.0_run/ and ../Paper24_Study_Report_v1.0_run.pdf.
"""
import io
import json
import os
import sys

import fitz

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'Paper24_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
L = json.load(open(os.path.join(HERE, 'local_check.json'), encoding='utf-8'))
XF = json.load(open(os.path.join(HERE, 'cross_fusion_p02_p24.json'), encoding='utf-8'))   # #2 lpnr_v12n + our runs
XB = XF['best_fusion_chosen_on_val']
FONT_DIR = os.path.join(HERE, 'fonts')


def bangla_font():
    """Nirmala.ttc cannot be loaded by MuPDF's CSS, but its first face saved as a .ttf works (shaping included)."""
    f = os.path.join(FONT_DIR, 'nirmala.ttf')
    if not os.path.exists(f):
        from fontTools.ttLib import TTCollection
        os.makedirs(FONT_DIR, exist_ok=True)
        TTCollection(r'C:\Windows\Fonts\Nirmala.ttc').fonts[0].save(f)
    return f


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
pre {font-family: M; font-size: 8.3pt; border: 0.6pt solid #c9c8c2; padding: 5pt 6pt;
     margin: 2pt 0 7pt 0; line-height: 1.25;}
.small {font-size: 9pt; color: #52514e;}
.q {font-weight: bold; margin: 6pt 0 1pt 0;}
.bn {font-family: BN; font-size: 8.6pt;}
.bad {color: #c0392b;}
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
f2 = lambda x: f'{x:.2f}'
f1 = lambda x: f'{x:.1f}'

# ----------------------------------------------------------------------------------------------- numbers used below
sg = A['singles']
Y, C, SL, V = (sg[m] for m in ('chars_v8n', 'crnn', 'slot', 'vitstr'))
ex = lambda m: f"{sg[m]['test']['plate_exact_%']:.2f}"
KAG = A['best_paper_rule']                      # the paper's rules, chosen on val (Kaggle's choice)
KIT = A['best_kit_rule']                        # every kit rule, chosen on val (Kaggle's headline row)
FB = A['fallback_2']                            # char-YOLO + CRNN fallback (the reader we ship)
SIGK, SIGO = A['sig_best_paper_rule'], A['sig_fallback_2']
ORA = A['oracle_test']['chars_v8n + crnn + slot + vitstr']
BW = A['best_wrong']
OC = BW['others_right_count']
PAIR_YC = next(p for p in A['pairs'] if p['a'] == 'chars_v8n' and p['b'] == 'crnn')
CONF = A['confidence']
DET = A['error_detector']
DETV = A['error_detector_val']['format']
FD = A['fallback_detail']['2']['test']
FDV = A['fallback_detail']['2']['val']
D23 = A['fallback_2_vs_3']
RS = A['best_wrong_reasons']
COND = A['by_condition']
SRC = A['by_source']
U632, R19 = A['unchanged_632'], A['recut_19']
CMP = {r['what']: r for r in A['v10_vs_v11']}
V10 = A['v10_summary']
HCP = A['hc_pair']
TOPN = A['topn']
CB = A['curves_best']
CT = A['chars_train']
NT, NV = A['plates']['test'], A['plates']['val']
MSY, MSC = Y['ms'], C['ms']
GAIN_K, GAIN_O = SIGK['gain_pp'], SIGO['gain_pp']
FBX = FB['test']['plate_exact_%']
KX = KAG['test']['plate_exact_%']
TG, TC = L['timing_gpu'], L['timing_cpu']
SHORT = {'chars_v8n': 'char-YOLO', 'crnn': 'CRNN', 'slot': 'slot CNN', 'vitstr': 'ViTSTR'}
KAG_NAME = ' + '.join(SHORT[m] for m in KAG['models'])
TIE = [c for c in A['combos'] if c['rule'] in ('HC', 'MV-HC', 'MV-BM', 'MVCP-HC')
       and c['val']['plate_exact_%'] == KAG['val']['plate_exact_%'] and c['models'] != KAG['models']]
v10 = lambda k: CMP[k]

# the sentences below quote these facts in words; stop if a re-run changes them
assert A['env']['dataset_version'] == '1.1' and NT == 651 and NV == 766
assert A['kaggle_fusion_check']['rows_equal'] == A['kaggle_fusion_check']['kit_rows'] == 66
assert KIT['models'] == ['chars_v8n', 'crnn', 'vitstr'] and KIT['rule'] == 'fallback'
assert FB['test']['plate_exact_%'] == KIT['test']['plate_exact_%'] and SIGO['broken'] == 0
assert DET['format']['precision_%'] == 100.0 and DETV['precision_%'] == 100.0
assert all(v == 0 for v in A['invalid_any_model_right'].values())
assert SIGK['p_mcnemar'] > 0.05 and SIGO['p_mcnemar'] < 0.01 and V10['sig_fallback']['broken'] == 0
assert D23['test']['only_3_right'] == D23['test']['only_2_right'] == 0 and D23['val']['only_3_right'] == 1
assert len(TIE) == 1 and TIE[0]['test']['plate_exact_%'] == Y['test']['plate_exact_%']
assert L['test']['chars_v8n_same_text_as_kaggle'] == NT and L['val']['chars_v8n_same_text_as_kaggle'] == NV
assert L['test']['fallback_exact_%'] == FBX and L['test']['crnn_fp16_exact_%'] == C['test']['plate_exact_%']
assert COND['night']['cascade'] == COND['night']['chars_v8n'] and SRC['nightlpbd']['cascade'] == SRC['nightlpbd']['chars_v8n']
assert CT['epochs_run'] == CT['epochs_planned'] == 100
assert XB['models'] == ['lpnr_v12n', 'chars_v8n'] and XB['rule'] == 'fallback' and XB['vs_best_single']['broken'] == 0
assert all(e['preds']['crnn'][0] == e['preds']['slot'][0] and not e['preds']['crnn'][2] for e in A['examples']['paper_broken'])
assert len(A['examples']['paper_broken']) == SIGK['broken'] and COND['night']['oracle'] == COND['night']['chars_v8n']
assert (round(R19['chars_v8n'] * 19 / 100) - 1) + (round(U632['chars_v8n'] * 6.32) - round(CMP['char-YOLOv8n']['v10_632'] * 6.32)) == round((Y['test']['plate_exact_%'] - CMP['char-YOLOv8n']['v10_all']) * 6.51)
assert all(c['fusion'] <= c['chars_v8n'] for c in list(COND.values()) + list(SRC.values()))
S = {}  # flow name -> HTML; each flow starts on a new page

# ======================================================================================== cover + one-page summary
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER #24 STUDY REPORT · v1.1 UPDATE</p>
<h1 style="font-size:19pt; margin-bottom:3pt">Leveraging Model Fusion for Improved License Plate Recognition</h1>
<p style="font-family:H; font-size:12pt; color:#1c5cab; margin-bottom:8pt">The paper, our re-run on the fixed dataset
(v1.1), whether the run is OK, how it fits our research, and what to show and say to the faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 24 October 2026<br>
Paper: R. Laroca, L. A. Zanlorensi, V. Estevam, R. Minetto and D. Menotti (Federal University of Paraná, Brazil),
<i>Iberoamerican Congress on Pattern Recognition (CIARP 2023)</i>, Springer LNCS, pp. 60-75,
DOI 10.1007/978-3-031-49249-5_5. Author version arXiv 2309.04331v2, 15 pages; page numbers below are its pages.<br>
Our run: Kaggle notebook {M('paper24_kaggle.ipynb')} on Tesla T4 x2, 22 Oct 2026, dataset <b>v1.1</b>, 79 min, all 5
jobs {M('code=0')}. Outputs: {M('After Train/results.zip')}. Every number of ours was recomputed from the per-plate
predictions in that zip ({M('report_src/analysis.py')}) and the two shared models were re-scored on this PC
({M('report_src/local_check.py')}). This report replaces the one of 21 Oct (v1.0 run), kept as
{M('Paper24_Study_Report_v1.0_run.pdf')}.</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> Read the same plate crop with several different OCR models and combine their answers.
The best rule is a plain majority vote over whole strings (ties broken by confidence or by the best model). With 12
models on 8 datasets the mean plate accuracy rises from 92.4 % (best single model) to 97.6 % (p.9), and on 4 unseen
datasets from 87.6 % to 90.3 % (p.11). Trusting the most confident model is the worst rule. Fusing 4-6 fast models gives
most of the gain for a fraction of the time (p.12).</p>
</div>
<div class="good">
<p><b>Is the v1.1 run OK? Yes.</b> It used dataset v1.1 and the current kit code; every job finished all its epochs; all
66 fusion rows on Kaggle match our recomputation; and the two models we share on GitHub give the same readings on this
PC as on Kaggle (all {NT} test plates for char-YOLO, {L['test']['crnn_fp16_same_text_as_kaggle']} of {NT} for CRNN,
the same accuracy). The dataset fix worked: the 19 BLPD test rows that were whole car photos are now plate crops, and
char-YOLO reads {R19['chars_v8n']:.0f} % of them instead of {v10('char-YOLOv8n')['v10_19']:.0f} %.</p>
</div>
<div class="good">
<p><b>What the run found.</b> (1) <b>The paper's recipe still gives nothing on our data:</b> the best voting rule chosen on
validation reads {KX:.2f} % against {ex('chars_v8n')} % for char-YOLOv8n alone ({GAIN_K:+.2f} points:
{SIGK['fixed']} plates fixed, {SIGK['broken']} broken, p = {SIGK['p_mcnemar']:.2f}). (2) <b>Why:</b> the models fail
together. On {OC['0']} of the {BW['n']} plates char-YOLO gets wrong, every other model is wrong too, so no vote can help
(ceiling: {ORA:.2f} %). (3) <b>Confidence cannot be compared across models:</b> CRNN gives
{CONF['crnn']['wrong_with_conf_ge_0.9_%']:.0f} % of its wrong readings a confidence of 0.9 or more, so "take the most
confident" drops to about {A['hc_range_with_crnn'][0]:.0f} %. (4) <b>Our Bangla-specific rule works, again:</b> keep
char-YOLO's reading when it fits the BRTA plate layout, otherwise ask CRNN. Every reading that broke the layout was wrong
({DET['format']['flagged']} of {DET['format']['flagged']} on test, {DETV['flagged']} of {DETV['flagged']} on
validation). Result: <b>{FBX:.2f} % ({GAIN_O:+.2f} points, {SIGO['fixed']} fixed, {SIGO['broken']} broken,
p = {SIGO['p_mcnemar']:.4f})</b> at {FB['ms']:.1f} ms per plate. The gain and the 12 / 0 split are the same as in the
v1.0 run, so the result <b>replicated</b>. (5) <b>New:</b> on real crops, char-YOLO's errors are mostly a digit boxed twice
or a letter read as two classes ({RS['more than 6 digits'] + RS['two letters']} of {BW['n']} wrong plates), so the next
gain is in its decoder.</p>
</div>
<div class="box">
<p><b>What changed since the first report.</b> All scores moved because 19 test images are now real crops
(char-YOLO {v10('char-YOLOv8n')['v10_all']:.2f} → {ex('chars_v8n')} %). The conclusions held. Two details did not: the
fallback no longer helps at night (it now helps most on blurry plates), and on the 632 unchanged plates the retrained
models moved by 1-3 points between the two runs (the slot CNN even by {U632['slot'] - v10('slot CNN')['v10_632']:+.1f}). So one run
cannot rank two models that are 1-2 points apart; we need seeds (section 11).</p>
<p><b>One sentence for the faculty.</b> "We re-ran the paper's model-fusion idea on our corrected Bangla dataset: plain
voting still does not help because our models make the same mistakes, but using the BRTA plate format to decide when to
ask a second model gives a significant +1.8 points at almost no cost, and it replicated across two runs."</p>
</div>
<p class="small"><b>Layout.</b> Part A (sections 1-7): understand the paper. Part B (8-14): our experiment; read it before
you quote our numbers. Part C (15-21): research use, benefits, slides, script, questions, next steps, files.</p>
"""

# ======================================================================================== part A
S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What problem does the paper solve?</h1>
<p>A plate reader has two stages: find the plate (detection) and read its characters (recognition, OCR). Detection is
nearly solved by YOLO-type models, so recent work focuses on recognition (p.1-2). But earlier studies, several by the
same group, showed that recognition models are <b>uneven</b>: a model that is best on one dataset can be poor on another,
because each dataset has its own layouts, fonts, tilt and image quality (p.2).</p>
<p>So the paper asks three plain questions (p.2):</p>
{ul(['Can we get clearly better plate reading by <b>combining the outputs</b> of several different recognition models?',
     'If yes, <b>how much</b> better?',
     '<b>How many and which</b> models should we combine?'], 'ol')}
<p>Before this paper, LPR fusion work had used copies of one model (Izidio et al., several Tiny-YOLOv3 nets on a private
set) or synthetic degraded images (Schirrmacher et al.); nobody had combined many <i>different</i> models across many
public datasets (p.4). The authors also stress a practical point: some applications (traffic tickets, forensics) prefer
accuracy over speed, so they can afford several models (p.2).</p>

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['Model fusion / ensemble', 'Run several models on the same input and combine their answers. It helps only when the models make <i>different</i> mistakes.'],
    ['Recognition rate (plate exact)', 'Share of plates whose <b>whole</b> string is right. One wrong character = wrong plate (p.8). This is the paper\'s only metric.'],
    ['HC (highest confidence)', 'Pick the string of the model that is most sure of itself, even if no other model agrees (p.7).'],
    ['MV (majority vote)', 'Pick the full string that most models output. Confidence is ignored unless there is a tie (p.7).'],
    ['MVCP (vote per character position)', 'Vote separately on character 1, character 2, ... and join the winners. Breaks when one model drops or adds a character, because the positions shift (p.10-11).'],
    ['-HC / -BM tie-break', 'If two strings get the same number of votes: take the more confident one (-HC) or the one from the "best model" (-BM) (p.7-8).'],
    ['Top-N', 'Fuse the N best models (ranked by mean accuracy). Table 4 tries N = 2 ... 12.'],
    ['Intra-dataset / cross-dataset', 'Test on the test split of a dataset used in training / on a dataset never seen in training (harder, closer to a new camera).'],
    ['Oracle (our term)', 'A plate counts as right if <i>any</i> model is right. No voting rule can beat it, so it is the ceiling for fusion.'],
    ['Calibration (our term)', 'A model is calibrated if, among readings with confidence 0.9, about 90 % are right. Uncalibrated confidences make HC fail.'],
    ['BRTA layout (our term)', 'The format of a Bangladeshi plate: one district word, optional ' + BN('মেট্রো') + ', one series letter, six digits. A reading that breaks it cannot be right.'],
    ['McNemar test (our term)', 'The standard test for "reader A beats reader B on the same plates": it only looks at the plates where they disagree (fixed vs broken).'],
], [2.3, 7.7])}

<h1>3. The 12 models and the 12 datasets</h1>
<p>The models were chosen because they cover the design patterns of modern text recognisers and are common baselines in
LPR (p.5). Each was trained <b>once</b> on the union of 8 training sets, in its own original framework:</p>
{table(['Family', 'Models', 'How they read', 'Framework, training (p.5)'], [
    ['Scene-text, CTC', 'CRNN, GRCNN, Rosetta, STAR-Net', 'CNN features, (BiLSTM), then CTC decoding', 'PyTorch (Clova AI fork), Adadelta, 300 k iterations, batch 128'],
    ['Scene-text, attention', 'R2AM, RARE, TRBA', 'CNN, (TPS rectification), BiLSTM, attention decoder', 'same'],
    ['Transformer', 'ViTSTR-Base', 'ViT patches; output tokens are the characters', 'same'],
    ['Multi-task CNN', 'Holistic-CNN, Multi-Task-LR', 'CNN, then one classifier per character slot', 'Keras, Adam 1e-3, batch 64, early stop'],
    ['YOLO character detectors', 'CR-NET, Fast-OCR', 'detect every character as an object, sort the boxes', 'Darknet, SGD, 90 k iterations, batch 64'],
], [1.7, 2.3, 3.0, 3.0])}
<p><b>Data (p.5-7, Table 2).</b> 8 datasets for training and testing: Caltech Cars, EnglishLP, UCSD-Stills, ChineseLP,
AOLP, SSIG-SegPlate, UFPR-ALPR, RodoSol-ALPR (20,000 images; Brazilian and Mercosur plates). 4 datasets only for
cross-dataset tests: OpenALPR-EU, PKU, CD-HARD (heavily tilted, mixed layouts) and CLPD. Training also used 772 web
images and synthetic plates made by <b>shuffling character positions</b>, so that models cannot memorise sequences
(p.6-7). Crops come from a YOLOv4 detector, not from the ground truth, to be realistic (p.8).</p>
<div class="box"><p><b>Mapping to our run.</b> We trained one model from each of four of these families on our data:
char-YOLOv8n (the CR-NET / Fast-OCR family), CRNN-CTC (CRNN), a slot CNN in the fast-plate-ocr style (Holistic-CNN /
Multi-Task-LR) and ViTSTR-tiny (ViTSTR). Section 8 gives the details.</p></div>
"""

S['a2'] = f"""
<h1>4. The method: three rules and two tie-breaks</h1>
{fig('fig_pipeline.png', 440, '<b>Figure 1.</b> A: the paper\'s fusion (example strings from its Fig. 4, p.10; orange = wrong). '
     'B: the change we test on Bangla plates (section 12, finding 4). Numbers in B are ours, on the 498R v1.1 test split.')}
<p>The whole method fits in a few lines. For each plate, every model gives a string and a confidence. Then:</p>
{ul(['<b>HC:</b> output the string with the highest confidence (p.7).',
     '<b>MV:</b> group identical strings and output the largest group. In the paper\'s example, five models say '
     '"ABC-123", "ABC-123", "ABC-124", "ABC-124", "ABC-125": a tie between two strings (p.7).',
     '<b>MVCP:</b> vote on every character position and join the winners (p.7).',
     '<b>Ties:</b> -HC takes the tied string with the higher confidence; -BM takes the tied string that the best single '
     'model output (p.7-8). With only <b>two</b> models every disagreement is a tie, so MV-BM is just "trust the best '
     'model" and MV-HC is just HC. You need at least three models before voting does anything.',
     '<b>Confidences are used raw.</b> The authors tried normalising them per model; it did not help (p.8).'])}
<p>The "best model" for -BM and the top-N order are taken from the mean accuracy over the datasets. The authors say the
validation ranking is "essentially the same, with only two models swapping positions" (p.9).</p>
"""

S['a2b'] = f"""
<h1>5. Every result in the paper</h1>
{fig('fig_paper.png', 460, '<b>Figure 2.</b> The paper\'s numbers, redrawn. Left: Table 4 (p.9), fusing the top-N models. '
     'Right: Table 6 (p.12), accuracy vs time, models added by accuracy or by speed.')}
{table(['Result', 'Value', 'Page'], [
    ['Best single model, 8 datasets (mean)', 'ViTSTR-Base 92.4 %; STAR-Net 92.2; TRBA 91.1; CR-NET 90.8; ... CRNN 87.1; Multi-Task-LR 85.9', 'p.9, Table 3'],
    ['Best fusion, 8 datasets', '<b>MV-HC (top 8) 97.6 %</b>; MV-BM (top 8) 97.5; MVCP-HC (top 9) 97.5; MVCP-BM 97.2; HC (top 6) 96.0', 'p.9, Table 3'],
    ['Worst cases removed', 'every single model is below 90 % on at least two datasets; every fusion rule is above 90 % on all eight', 'p.9'],
    ['Top-N curve (MV-HC)', 'top 2: 94.1, top 3: 94.9, top 4: 96.3, top 6: 97.0, top 8: 97.6, top 12: 97.0 %', 'p.9, Table 4'],
    ['Cross-dataset, best single', 'Fast-OCR 87.6 %; CR-NET 87.1; STAR-Net 84.4; ViTSTR-Base only 75.9', 'p.11, Table 5'],
    ['Cross-dataset, best fusion', '<b>MV-BM (top 8) 90.3 %</b>; MV-HC 90.1; MVCP-HC 87.6; HC 84.4 (below the best single model)', 'p.11, Table 5'],
    ['Hardest unseen set (CD-HARD)', 'best single 59.6 → MV fusion 65.4 %', 'p.11, Table 5'],
    ['Speed, single models', 'Multi-Task-LR 2.3 ms; Holistic-CNN 2.5; CRNN 2.9; Fast-OCR 3.0; ViTSTR-Base 7.3; TRBA 16.9 ms per plate', 'p.12, Table 6'],
    ['Speed, fusion', 'top 8 by accuracy: 59.7 ms (17 FPS). The 4 / 5 / 6 fastest: 95.4 / 96.0 / 96.6 % at 10.8 / 15.4 / 20.7 ms', 'p.12, Table 6'],
    ['Surprise', 'the 3 fastest fused (91.1 %) are below the best single model (92.4 %)', 'p.12'],
    ['Training time, seeds, significance tests', 'not reported', '-'],
], [2.6, 6.0, 1.4])}
<p><b>The authors' own explanation (p.10):</b> fusion "does not necessarily lead to significantly improved performance
compared to the best model in the ensemble. Instead, it reduces the likelihood of obtaining poor performance", because
diverse models make different errors but agree when they are right. HC is weak because every model is also confidently
wrong at times (p.11, Fig. 4). Keep this sentence in mind: our run shows exactly when that condition fails.</p>
"""

S['a3'] = f"""
<h1>6. Critical review</h1>
<h2>Strengths</h2>
{ul(['<b>Wide and honest test bed:</b> 12 models in their original frameworks, 12 public datasets, intra- and '
     'cross-dataset results, and detector crops instead of perfect crops (p.5-8).',
     '<b>Simple, reusable rules:</b> no training, about 40 lines of code; anyone can add it to an existing system.',
     '<b>Speed is reported next to accuracy</b> (Table 6), which gives a practical recipe: 4-6 fast models.',
     '<b>Useful negative results:</b> HC is the worst rule and confidence normalisation did not help (p.8, p.11).'])}
<h2>Weaknesses (what a reviewer would ask)</h2>
{ul(['<b>Test-informed choices.</b> The "best model" for -BM and the top-N order come from test-set accuracy across '
     'datasets. The authors say validation gives nearly the same order (p.9), but strictly every choice should be made '
     'on validation only. <i>We did that in our run.</i>',
     '<b>One training run per model, no confidence intervals, no significance test.</b> Several gains in Table 4 are '
     'differences of 0.1-0.3 points on small test sets (Caltech Cars has 46 plates, UCSD-Stills 60). Our own two runs '
     'show why this matters: the same model moved by 1-3 points between runs (section 11).',
     '<b>The headline gain partly measures "not knowing which model to use".</b> We recomputed it from Table 3: if you '
     'could pick the best single model <i>for each dataset</i>, the mean would already be <b>96.8 %</b>. Fusion (97.6 %) '
     'adds only <b>0.8 points</b> over that. The big +5.2 is against one fixed model (ViTSTR-Base), which is weak on the '
     'older US and EU sets. Cross-dataset the per-dataset best is Fast-OCR everywhere (87.6 %), so the +2.7 there is a '
     'real gain, mostly on CD-HARD and CLPD.',
     '<b>Not every model saw the same data:</b> CR-NET and Fast-OCR were not trained on RodoSol-ALPR, because they need '
     'character boxes (dagger in Table 3, p.9); their 59.0 / 56.7 % there is a data gap, not a model weakness.',
     '<b>Heavy for real time:</b> 8 models cost 59.7 ms per plate on a 48 GB workstation GPU; much slower on a laptop CPU.',
     '<b>Latin and Chinese plates only</b>; no complex script such as Bangla, where one character can be several Unicode '
     'code points.',
     '<b>Diversity is assumed, not measured.</b> The paper never reports how often the models fail together, which is '
     'exactly what decides whether fusion can work (our finding 2).'])}
<p><b>Verdict.</b> A careful, peer-reviewed and easy-to-copy study. Its main lesson is about <i>risk reduction</i>
across datasets rather than a large gain on one dataset, and its claim depends on having many strong and different
models. Both points matter for us.</p>

<h1>7. What would break on our Bangla plates (and what we did)</h1>
{table(['Problem', 'Why', 'What we did / do'], [
    ['MVCP on Unicode', 'Bangla vowel signs and conjuncts change the string length, so character positions shift (worse than in the paper)', 'vote on <b>tokens</b> (district, ' + BN('মেট্রো') + ', letter, 6 digits) instead of Unicode characters'],
    ['Correlated models', f"all our models learn from the same 5,607 crops; {A['train_label_source']['auto']:,} of their labels came from a character YOLO (auto-labels), so they can share its mistakes", 'measure it: pairwise overlap and the oracle (finding 2)'],
    ['Few strong models', 'we have one strong reader (char-YOLO); training several strong Bangla OCR models is expensive', 'test fusion with 2-4 models; teammates\' readers can join later (section 15)'],
    ['Confidence scales differ', 'a box score (char-YOLO) and a softmax mean (CRNN) do not mean the same thing', 'check calibration (finding 3); prefer MV-BM over HC'],
    ['Real time (≥ 10 FPS, RQ3)', 'running 4 models on every plate adds up', 'call a second model only when needed (finding 4)'],
    ['Two-row plates', 'most Bangla plates have two rows', 'char-YOLO reads boxes by token kind; CRNN / ViTSTR see the two rows side by side'],
], [2.0, 4.2, 3.8])}
"""

# ======================================================================================== part B
cfg_rows = [
    ['char-YOLOv8n', 'GPU 0', f"YOLOv8n from COCO weights; 100 epochs, 320 px, batch 32, SGD, no mirror flips; 101 token classes (the dataset's vocabulary)", f"{Y['train_min']:.0f} min"],
    ['slot CNN', 'GPU 0 (after)', f"6-block CNN, 9 slots, 64×128 grey input; 100 epochs, AdamW + OneCycle; {SL['params_M']} M parameters", f"{SL['train_min']:.0f} min"],
    ['CRNN-CTC', 'GPU 1', f"VGG-style CNN + 2-layer BiLSTM + CTC, 48×256, two rows side by side; 60 epochs; {C['params_M']} M parameters", f"{C['train_min']:.0f} min"],
    ['ViTSTR-tiny', 'GPU 1 (after)', f"ImageNet ViT-tiny (timm), 32×256, first 12 tokens classified; 60 epochs, lr 3e-4; {V['params_M']} M", f"{V['train_min']:.0f} min"],
    ['fusion', 'CPU', 'every subset of 2-4 models × HC, MV-HC, MV-BM, MVCP-HC (paper) and MV-BM+format, fallback (ours); best chosen on validation; exact McNemar test for every row', '0.3 min'],
]
S['b1'] = f"""
<p class="part">PART B · OUR EXPERIMENT</p>
<h1>8. What we ran and why</h1>
<p>No fusion code and most of the paper's datasets are not public (several need a request), so a run on the paper's own
data was not possible. Instead the kit asks the paper's question on <b>our</b> data: does fusing different OCR models
help on Bangladeshi plates, and at what cost? This is the second run. The first one (20 Oct, dataset v1.0) found 19 test
"crops" that were whole car photos; the dataset was fixed (v1.1) and everything was trained again from scratch.</p>
{table(['Run', 'Where', 'Model and settings', 'Train time'], cfg_rows, [1.5, 1.2, 5.6, 1.1])}
{table(['Item', 'Setting'], [
    ['Data', f"498R BanglaLPR All-in-One <b>v1.1</b> (downloaded by the notebook from the team Drive), OCR split: 5,607 train crops (3,471 human + 2,136 auto-labelled), {NV} validation, {NT} test (human labels only). Test sources: LPDBA {SRC['lpdba']['n']}, ANPR {SRC['anpr']['n']}, KUET {SRC['kuet']['n']}, NightLPBD {SRC['nightlpbd']['n']}, BLPD {SRC['blpd']['n']}."],
    ['What v1.1 changed', '91 BLPD rows (train 63, val 9, test 19) were whole vehicle photos; they are now cut to the plate. Splits, groups and counts did not change (docs/DATASET_v1_1_CHANGES.md).'],
    ['Split', 'grouped 70/15/15, seed 42, no near-duplicate across splits (team rule)'],
    ['Metric', 'plate exact (whole plate right after Unicode normalisation, digits as Bangla, no spaces or hyphens), plus CER and character accuracy'],
    ['Fair choice', 'MV-BM\'s "best model", the fallback\'s first model and every "best" row are chosen on <b>validation</b>; test is read once; a tie on validation goes to the faster row. (The paper chose on test.)'],
    ['Speed', 'ms per plate, batch 1, on one T4; a fusion costs the sum of its members (run one after another); the fallback costs char-YOLO plus the helpers on the share of plates that need them'],
    ['Hardware, software', f"Kaggle Tesla T4 x2, one job per GPU; Python {A['env']['python']}, torch {A['env']['torch']}, Ultralytics {A['env']['ultralytics']}, timm {A['env']['timm']}"],
], [1.6, 8.4])}
<p><b>After the run</b> we went further than the notebook (all in {M('report_src/analysis.py')}, no retraining): error
overlap between models, bootstrap intervals, confidence calibration, two more fusion rules (weighted votes), a
confidence-gated fallback, results per condition and per source, the 19 re-cut plates, and a comparison with the v1.0
run. Every extra rule was again chosen on validation and reported on test.</p>

<h1>9. Is the run OK? (health check)</h1>
{fig('fig_curves.png', 480, '<b>Figure 3.</b> Training (dashed) and validation (solid) loss of the three sequence models '
     '(v1.1 run). The black dot is the epoch whose weights were kept.')}
{table(['Check', 'Result', 'Meaning'], [
    ['Dataset version', f"environment.json: dataset_version {A['env']['dataset_version']}; the log shows the download from the \"All In One 1.1\" Drive folder", 'the run used the fixed data'],
    ['Jobs and time caps', 'all 5 jobs code=0; 79 min wall time on 2 GPUs', 'nothing failed or was cut short'],
    ['Code', 'the 9 kit files in the zip are identical to the current kit (line endings only)', 'the run is the kit you have'],
    ['Kaggle numbers = ours', f"all {A['kaggle_fusion_check']['kit_rows']} fusion rows recomputed from the per-plate predictions: identical (val and test); oracle {ORA:.2f} = {A['kaggle_fusion_check']['kaggle_oracle_test']:.2f}", 'the analysis reads the same predictions'],
    ['Shared models re-scored on this PC', f"char-YOLO: same text on all {NT} test and {NV} val plates; CRNN (FP16 copy): {L['test']['crnn_fp16_exact_%']:.2f} % test, same as Kaggle; fallback {L['test']['fallback_exact_%']:.2f} %", 'the GitHub files reproduce the report'],
    ['No test peeking', 'best model, tie-break model and every rule chosen on validation', 'test numbers are fair'],
    ['char-YOLOv8n', f"{CT['epochs_run']} / {CT['epochs_planned']} epochs; box mAP50 {CT['test_boxes']['mAP50']:.3f} on test (P {CT['test_boxes']['P']:.2f}, R {CT['test_boxes']['R']:.2f}); character accuracy {Y['test']['char_acc_%']:.1f} %", 'healthy (v1.0: mAP50 0.913)'],
    ['CRNN-CTC', f"val loss flat from epoch ~45, best at {CB['crnn']['epoch']}; test {ex('crnn')} %", 'healthy but weaker than char-YOLO'],
    ['ViTSTR-tiny', f"val loss lowest at epoch {CB['vitstr']['epoch']}, then rises; test {ex('vitstr')} %", f"<b>overfits and undertrained:</b> {A['steps_per_epoch']['vitstr'] * CB['vitstr']['epoch']:,} steps at its best epoch vs 300 k iterations in the paper; tiny model, 32-px input"],
    ['slot CNN', f"best at epoch {CB['slot']['epoch']}, val loss {CB['slot']['val_loss']:.2f} (v1.0: {V10['curves_best']['slot']['val_loss']:.2f}); test {ex('slot')} %", '<b>underfits:</b> 1.7 M parameters from scratch, global max-pool loses position; very sensitive to the run (section 11)'],
    ['Validation vs test', f"every model is better on test than on validation (char-YOLO {Y['val_exact']:.1f} vs {ex('chars_v8n')})", 'the validation plates are harder for every model; the ranking is the same on both, so the choices hold'],
    ['Auto-labels in training', f"{A['train_label_source']['auto']:,} of {A['train_label_source']['auto'] + A['train_label_source']['human']:,} OCR training labels are auto-labels from a character YOLO", 'a likely reason why the models share errors (finding 2)'],
    ['Single run', 'one seed per model, as in the paper', 'run-to-run spread is 1-3 points (section 11): add 3 seeds before final claims'],
], [2.0, 4.6, 3.4])}
<div class="good"><p><b>Verdict: the run is OK and can be quoted.</b> It is complete, it used v1.1, it matches the kit,
and every number in this report is reproducible from the saved predictions and from the shared weights. Its limits are
the ones of the design (two weak models, one seed), not errors in the run.</p></div>
"""

S['b2'] = f"""
<h1>10. Results next to the paper</h1>
{fig('fig_ours.png', 500, f'<b>Figure 4.</b> Our test results ({NT} plates, v1.1). The black lines are 95 % confidence '
     f'intervals (Wilson): about ±2.6 points for the best model, so a difference of 0.5 points between two readers cannot '
     f'be read from the bars; the paired McNemar test in section 12 can.')}
{table(['Metric', 'Paper (page)', 'Ours (498R v1.1 test)'], [
    ['Best single model, plate exact', '92.4 % ViTSTR-Base, mean of 8 sets (p.9)', f"<b>{ex('chars_v8n')} %</b> char-YOLOv8n (val {Y['val_exact']:.2f} %; 95 % CI {Y['test_ci95'][0]:.1f}-{Y['test_ci95'][1]:.1f})"],
    ['Other single models', '85.9-92.2 % (all 12 within 6.5 points)', f"CRNN {ex('crnn')}, slot CNN {ex('slot')}, ViTSTR-tiny {ex('vitstr')} % (spread {Y['test']['plate_exact_%'] - V['test']['plate_exact_%']:.0f} points)"],
    ['Best fusion with the paper\'s rules (val-chosen)', '97.6 % MV-HC top 8 (+5.2); cross 90.3 % (+2.7)', f"{KX:.2f} %, {KAG_NAME}, {KAG['rule']} ({GAIN_K:+.2f}; p = {SIGK['p_mcnemar']:.2f})"],
    ['HC (most confident)', '96.0 % intra; 84.4 % cross, below the best single model', f"{A['hc_range_with_crnn'][0]:.1f}-{A['hc_range_with_crnn'][1]:.1f} % for every subset with char-YOLO and CRNN, 10 points below the best single model"],
    ['Oracle (any model right)', 'not reported', f"{ORA:.2f} % (ceiling: only {ORA - Y['test']['plate_exact_%']:+.2f} points)"],
    ['Fusion time per plate', '59.7 ms top 8 (RTX 8000); 10.8 ms for the 4 fastest', f"{KAG['ms']:.1f} ms for the paper-rule fusion; {sum(sg[m]['ms'] for m in sg):.1f} ms for all 4 (T4)"],
    ['Our format-aware fallback', '-', f"<b>{FBX:.2f} %</b> ({GAIN_O:+.2f}; {SIGO['fixed']} fixed, {SIGO['broken']} broken; p = {SIGO['p_mcnemar']:.4f}) at {FB['ms']:.1f} ms"],
    ['Training time', 'not reported', f"{Y['train_min'] + SL['train_min'] + C['train_min'] + V['train_min']:.0f} GPU-min in total (79 min wall time on 2 GPUs)"],
], [2.6, 3.7, 3.7])}
<div class="warn"><p><b>Read the gap as a direction, not a contest.</b> The paper used 12 other datasets with Latin and
Chinese plates, larger training sets and 12 tuned models. Our numbers are on Bangla plates, with 4 models of which two
are undertrained. What carries over is the <i>pattern</i>: which rule wins, and when fusion can help at all.</p></div>
{table(['Top-N by validation accuracy (paper Table 4 style)', 'HC', 'MV-HC', 'MV-BM', '+ format', 'fallback', 'ms (sum)'], [
    [f"top {r['k']} (+ {A['show'][r['added']]})", f"{r['HC']:.2f}", f"{r['MV-HC']:.2f}", f"{r['MV-BM']:.2f}",
     f"{r['MV-BM+format']:.2f}", f"<b>{r['fallback']:.2f}</b>", f"{r['ms_sum']:.1f}"]
    for r in TOPN if r['rank_by'] == 'accuracy'], [3.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], num=(1, 2, 3, 4, 5, 6))}
<p class="small">The table shows the paper's own pattern on our data: HC and MV-HC fall by about 10 points as soon as
CRNN joins; MV-BM stays at or just below the best model; the BRTA filter (+ format) and the fallback lift every row. With
two models MV-BM equals the best model alone, as explained in section 4. "ms" is the time for all models of the row; the
fallback needs only {FB['ms']:.1f} ms on average because CRNN runs on {FB['called_test_%']:.1f} % of the plates.</p>
"""

S['b3'] = f"""
<h1>11. What changed from the v1.0 run, and what stayed</h1>
<p>The two runs used the same code, the same split and the same settings. Only 182 BLPD rows of the data changed
(91 character-box rows and the same 91 OCR rows), and the models were trained again from scratch. The v1.0 per-plate
predictions were overwritten, so the comparison below uses the numbers saved in the first analysis
({M('report_src/v1.0_run/analysis.json')}); a paired test between the two runs is not possible.</p>
{fig('fig_v10_v11.png', 500, f'<b>Figure 5.</b> The same readers in both runs. Left: the {U632["n"]} test plates whose '
     f'image did not change. Right: the {R19["n"]} BLPD test rows that were whole car photos in v1.0 and are plate '
     f'crops in v1.1.')}
{fig('fig_recut.png', 470, '<b>Figure 6.</b> Six of the 19 re-cut BLPD test rows: the v1.0 image (top) and the v1.1 '
     'plate crop (bottom), with char-YOLO\'s v1.1 result.')}
{table(['Plate exact (%)', 'v1.0 all', 'v1.1 all', 'v1.0 632 unchanged', 'v1.1 632 unchanged', 'v1.0 19 re-cut', 'v1.1 19 re-cut'], [
    [r['what'].replace(' (val-chosen)', ''), f2(r['v10_all']), f"<b>{r['v11_all']:.2f}</b>",
     f1(r['v10_632']), f1(r['v11_632']), f1(r['v10_19']), f1(r['v11_19'])] for r in A['v10_vs_v11']],
    [3.0, 1.1, 1.1, 1.3, 1.3, 1.1, 1.1], num=(1, 2, 3, 4, 5, 6))}
<h3>What the dataset fix did</h3>
{ul([f"On the 19 re-cut rows every reader improved: char-YOLO from 1 plate to {round(R19['chars_v8n'] * R19['n'] / 100)} "
     f"({v10('char-YOLOv8n')['v10_19']:.1f} → {R19['chars_v8n']:.1f} %). BLPD as a source went from "
     f"{V10['by_source']['blpd']['chars_v8n']:.1f} to {SRC['blpd']['chars_v8n']:.1f} %.",
     f"That alone explains the higher overall scores: char-YOLO +{Y['test']['plate_exact_%'] - v10('char-YOLOv8n')['v10_all']:.2f} "
     f"points overall comes from +{round(R19['chars_v8n'] * R19['n'] / 100) - 1} plates on the re-cut rows and "
     f"{round(U632['chars_v8n'] * U632['n'] / 100) - round(v10('char-YOLOv8n')['v10_632'] * U632['n'] / 100):+d} on the rest.",
     'The fusion results on the re-cut rows are all equal to char-YOLO alone: there, no other model rescued a plate.'])}
<h3>What two trainings of the same model tell us</h3>
<p>On the {U632['n']} plates that did not change, the retrained models moved by
{U632['chars_v8n'] - v10('char-YOLOv8n')['v10_632']:+.1f} (char-YOLO), {U632['crnn'] - v10('CRNN-CTC')['v10_632']:+.1f}
(CRNN), {U632['slot'] - v10('slot CNN')['v10_632']:+.1f} (slot CNN) and {U632['vitstr'] - v10('ViTSTR-tiny')['v10_632']:+.1f}
points (ViTSTR). Only 63 of 5,607 training rows changed, and the slot CNN improved on every source (KUET
{V10['by_source']['kuet']['slot']:.0f} → {SRC['kuet']['slot']:.0f}, LPDBA {V10['by_source']['lpdba']['slot']:.0f} →
{SRC['lpdba']['slot']:.0f}, ANPR {V10['by_source']['anpr']['slot']:.0f} → {SRC['anpr']['slot']:.0f} %), not just where
the data changed. So these are mostly <b>run-to-run differences</b> (GPU non-determinism and a different path through
training), not effects of the fix. Lesson: <b>one run cannot rank two readers that are 1-3 points apart</b>; the slot
CNN, trained from scratch, can swing by about 10 points.</p>
<h3>What replicated (the findings we can now state with more confidence)</h3>
{table(['Finding', 'v1.0 run (20 Oct)', 'v1.1 run (22 Oct)'], [
    ['Paper\'s best rule (val-chosen) vs best model', f"{v10('paper' + chr(39) + 's best rule (val-chosen)')['v10_all']:.2f} %, +0.62: {V10['sig_paper']['fixed']} fixed / {V10['sig_paper']['broken']} broken, p = {V10['sig_paper']['p_mcnemar']:.2f}", f"{KX:.2f} %, {GAIN_K:+.2f}: {SIGK['fixed']} fixed / {SIGK['broken']} broken, p = {SIGK['p_mcnemar']:.2f}"],
    ['Format fallback vs best model', f"{V10['sig_fallback']['fixed']} fixed / {V10['sig_fallback']['broken']} broken, +1.84, p = {V10['sig_fallback']['p_mcnemar']:.4f}", f"<b>{SIGO['fixed']} fixed / {SIGO['broken']} broken, {GAIN_O:+.2f}, p = {SIGO['p_mcnemar']:.4f}</b>"],
    ['Layout check as an error detector (test)', f"{V10['error_detector']['format']['flagged']} flagged, precision {V10['error_detector']['format']['precision_%']:.0f} %", f"{DET['format']['flagged']} flagged, precision {DET['format']['precision_%']:.0f} %"],
    ['char-YOLO errors that every model gets wrong', f"{V10['best_wrong']['others_right_count']['0']} of {V10['best_wrong']['n']}", f"{OC['0']} of {BW['n']}"],
    ['CRNN wrong readings with confidence ≥ 0.9', f"{V10['confidence_wrong_ge_0.9']['crnn']:.0f} %", f"{CONF['crnn']['wrong_with_conf_ge_0.9_%']:.0f} %"],
    ['Where the fallback helps most', f"night (+{V10['by_condition']['night']['cascade'] - V10['by_condition']['night']['chars_v8n']:.1f}), NightLPBD (+{V10['by_source']['nightlpbd']['cascade'] - V10['by_source']['nightlpbd']['chars_v8n']:.1f})", f"blurry (+{COND['blurry']['cascade'] - COND['blurry']['chars_v8n']:.1f}); night +0.0 (<i>did not replicate</i>)"],
    ['Main error type of char-YOLO', 'missed characters (mostly the 19 whole photos)', f"extra digit boxes and doubled letters (finding 5)"],
], [3.0, 3.5, 3.5])}
"""

S['b4'] = f"""
<h1>12. Six findings</h1>
<h2>Finding 1: the paper's recipe is not significant on our data (second time)</h2>
<p>The best paper-rule fusion chosen on validation is {KAG_NAME} with {KAG['rule']}: {KAG['val']['plate_exact_%']:.2f} % on
validation and {KX:.2f} % on test, against {ex('chars_v8n')} % for char-YOLOv8n alone. Per plate, it fixes
{SIGK['fixed']} readings and breaks {SIGK['broken']}. The exact McNemar test gives p = {SIGK['p_mcnemar']:.2f}, and a
bootstrap 95 % interval for the gain is {SIGK['gain_ci95'][0]:+.2f} to {SIGK['gain_ci95'][1]:+.2f} points: no evidence
of any gain. The choice itself was a tie on validation: {' + '.join(SHORT[m] for m in TIE[0]['models'])} with
{TIE[0]['rule']} also scored {TIE[0]['val']['plate_exact_%']:.2f} % and reads {TIE[0]['test']['plate_exact_%']:.2f} % on test
(exactly char-YOLO); the kit took the faster one. Either way: no gain, at {KAG['ms'] / MSY:.1f}x the time.</p>

<h2>Finding 2: our models fail together, so voting has little to work with</h2>
{fig('fig_overlap.png', 470, f'<b>Figure 7.</b> Of the {BW["n"]} plates char-YOLOv8n reads wrongly, {OC["0"]} are read '
     f'wrongly by all four models. Only {BW["any_other_right"]} could be rescued by any rule, so the oracle is {ORA:.2f} %.')}
{ul([f"When char-YOLO is wrong, CRNN is also wrong {PAIR_YC['p_b_wrong_given_a_wrong_%']:.0f} % of the time (Yule's Q = "
     f"{PAIR_YC['Q']:.2f}; 0 would mean independent errors, 1 fully shared).",
     f"If three readers each as good as char-YOLO made <i>independent</i> errors, a majority vote would be right "
     f"{A['independent_3_majority_%']:.1f} % of the time. Real errors are shared, which is why the vote gives {KX:.1f} %.",
     f"For a wrong plate to be fixed by voting, at least two other models must agree on the right string. That is possible "
     f"on only {OC['2'] + OC['3']} plates.",
     f"The vote also <i>breaks</i> plates: all {SIGK['broken']} it breaks are a series letter that CRNN and the slot CNN both "
     f"misread the same way ({BN('খ')}→{BN('গ')} / {BN('ক')}, {BN('ক')}→{BN('গ')}, {BN('ন')}→{BN('ব')}, {BN('জ')}→{BN('ড')}), outvoting "
     "char-YOLO's right answer (example table after finding 4).",
     'The paper\'s 12 models were all strong (85.9-92.4 %), so a wrong model was usually outvoted. Here ViTSTR-tiny and '
     'the slot CNN are wrong on most plates and cannot outvote anything.'])}

<h2>Finding 3: same confidence, different meaning (why HC fails)</h2>
{fig('fig_confidence.png', 480, f'<b>Figure 8.</b> Left: share of right readings in each confidence bin. Char-YOLO\'s '
     f'scores are honest above 0.8; CRNN\'s are not (only {CONF["crnn"]["bins"][4]["acc_%"]:.0f} % right between 0.9 and '
     f'0.95). Right: share of wrong readings with confidence ≥ 0.9.')}
<p>{CONF['crnn']['wrong_with_conf_ge_0.9_%']:.0f} % of CRNN's wrong readings have confidence ≥ 0.9 (mean
{CONF['crnn']['mean_wrong']:.2f} when wrong, {CONF['crnn']['mean_right']:.2f} when right), against
{CONF['chars_v8n']['wrong_with_conf_ge_0.9_%']:.0f} % for char-YOLO. So when the two disagree, HC usually believes CRNN
and loses: in the char-YOLO + CRNN pair, HC overrules {HCP['right_overruled']} right readings and fixes only
{HCP['wrong_fixed']}, and reads {A['topn'][1]['HC']:.1f} %, 10 points <i>below</i> char-YOLO alone. This is the paper's own
finding (p.11), only stronger: within one model the confidence ranks readings reasonably (AUROC
{min(CONF[m]['auroc'] for m in CONF):.2f}-{max(CONF[m]['auroc'] for m in CONF):.2f}), but across models the scales do not
match.</p>
"""


def ex_rows(items, with_all=False):
    import cv2
    rows = []
    for e in items:
        p = e['preds']
        f = e['file']
        im = cv2.imread(os.path.join(HERE, 'ex', f))
        w = round(26 * im.shape[1] / im.shape[0])
        cell = lambda m: (f'<span class="bn{" bad" if not p[m][2] else ""}">{p[m][0] or "(nothing)"}</span> '
                          f'<span class="small">{p[m][1]:.2f}</span>')
        r = [f'<img src="ex/{f}" height="26" width="{w}">', BN(e['true']), cell('chars_v8n'), cell('crnn')]
        if with_all:
            r += [cell('slot'), f'<span class="bn bad">{e["fusion"]}</span>', BN(e['fallback'])]
        else:
            r += [BN(e['fallback'])]
        rows.append(r)
    return rows


pick_fixed = {e['file']: e for e in A['examples']['fallback_fixed']}
fixed_show = [pick_fixed[f] for f in ('kuet_c2fd34f05670.jpg', 'lpdba_8281db012e93.jpg', 'anpr_a4298e28c6d4.jpg',
                                      'kuet_7dbe2a31180d.jpg')]
broken_show = A['examples']['paper_broken'][:4]
assert all(not e['preds']['chars_v8n'][2] and e['fallback_right'] for e in fixed_show)
assert all(e['preds']['chars_v8n'][2] and not e['fusion_right'] and e['fallback_right'] for e in broken_show)

S['b5'] = f"""
<h2>Finding 4 (our contribution): the BRTA layout is a free and safe fusion signal</h2>
<p>Every Bangladeshi plate has one layout: a district, an optional {BN('মেট্রো')}, exactly one letter, then six digits.
A reading that breaks it cannot be right. We used this in two ways, both chosen on validation:</p>
{ul([f"<b>As an error detector.</b> {DET['format']['flagged']} of char-YOLO's {NT} test readings ({DET['format']['flagged_%']:.1f} %) "
     f"break the layout, and <b>all {DET['format']['flagged']} are wrong</b> (on validation: all {DETV['flagged']} of "
     f"{DETV['flagged']}). That is {DET['format']['recall_%']:.0f} % of its test errors, found without any second model. "
     f"The readings that pass are {DET['format']['acc_when_accepted_%']:.1f} % right. In fact, on test not one reading of "
     "any of the four models that breaks the layout is right.",
     f"<b>As a fallback rule.</b> Keep char-YOLO's reading when the layout is valid; otherwise ask CRNN and take its reading "
     f"if that one is valid. CRNN runs on only {FB['called_test_%']:.1f} % of plates, so the mean cost is "
     f"{FB['ms']:.1f} ms instead of {MSY + MSC:.1f} ms. Result: <b>{FBX:.2f} %</b>, {SIGO['fixed']} plates fixed and "
     f"<b>none broken</b> (McNemar p = {SIGO['p_mcnemar']:.4f}; bootstrap 95 % interval of the gain "
     f"{SIGO['gain_ci95'][0]:+.2f} to {SIGO['gain_ci95'][1]:+.2f} points). Because it only replaces readings that are "
     "certainly wrong, it cannot break a right one.",
     f"<b>What happens to the {FD['flagged']} flagged plates:</b> CRNN fixes {FD['fixed']}; on {FD['kept_best_reading']} "
     f"CRNN's reading is invalid too, so char-YOLO's (wrong) reading stays; on {FD['still_wrong_valid_helper']} CRNN gives a "
     "valid but wrong reading. Those plates were wrong before and stay wrong, so nothing is lost, but they show where a "
     "better second reader would pay off.",
     f"<b>Kaggle's headline row adds ViTSTR as a third helper</b> (val {KIT['val']['plate_exact_%']:.2f} vs "
     f"{FB['val']['plate_exact_%']:.2f} %: one validation plate); on test both read {FBX:.2f} % and never differ on a right "
     f"answer. We ship the two-model version: one model less and {KIT['ms'] - FB['ms']:.1f} ms faster, with no test number "
     "changed.",
     f"<b>Confidence gates add nothing:</b> also asking CRNN when char-YOLO's confidence is below a threshold (0.5-0.95) "
     "did not raise the validation score, so the rule uses the layout alone. The same filter inside the vote "
     f"(MV-BM+format) also reaches {next(c for c in A['combos'] if c['models'] == ['chars_v8n', 'crnn'] and c['rule'] == 'MV-BM+format')['test']['plate_exact_%']:.2f} % "
     f"but always runs both models ({MSY + MSC:.1f} ms)."])}
{fig('fig_speed.png', 500, '<b>Figure 9.</b> Accuracy vs time for every subset (paper Table 6 style). Grey: the paper\'s '
     'MV-BM rule; black diamonds: the same subset with the BRTA filter; open triangles: the same subset as a fallback; '
     'star: our fallback (char-YOLO, then CRNN only when the layout fails), the cheapest point on the top edge.')}
"""

S['b6'] = f"""
<h3>Examples (test crops; red = wrong; small number = confidence)</h3>
{table(['Crop', 'Truth', 'char-YOLOv8n', 'CRNN', 'Our fallback'], ex_rows(fixed_show), [1.6, 2.0, 2.3, 2.0, 2.0])}
<p class="small">Above: plates the fallback fixes. Char-YOLO finds a letter twice under two classes ({BN('গঘ')}, {BN('হট')}),
loses a digit, or boxes digits twice (up to nine digits); each breaks the layout, so CRNN is asked and its valid
reading is kept. Below: plates the paper's vote <i>breaks</i>. CRNN and the slot CNN make the same letter mistake, so two
votes beat char-YOLO's right answer; our fallback never asks them because char-YOLO's reading is valid.</p>
{table(['Crop', 'Truth', 'char-YOLOv8n', 'CRNN', 'slot CNN', 'paper vote', 'our fallback'], ex_rows(broken_show, True),
       [1.3, 1.9, 1.9, 1.9, 1.9, 1.9, 1.9])}

<h2>Finding 5: on real crops, char-YOLO's errors are doubled boxes, so the next fix is its decoder</h2>
{fig('fig_errors.png', 480, f'<b>Figure 10.</b> Why char-YOLOv8n\'s {BW["n"]} wrong test readings are wrong (a plate can '
     f'have several reasons). Blue: the layout check catches it.')}
<p>In v1.0 most of char-YOLO's errors looked like <i>missed</i> characters; most of those came from the 19 whole photos.
On real crops the picture is different: of the {sum(a['n'] for a in A['error_kinds'])} token errors,
{A['error_ops']['sub']} are wrong symbols, {A['error_ops']['extra']} are <b>extra</b> tokens and only
{A['error_ops']['missing']} are missing ones. The layout check catches the extra and doubled ones
({DET['format']['flagged']} plates). The other {RS['valid layout, wrong symbol']} wrong plates have a valid layout with
one wrong symbol (mostly a letter or a digit confused with a similar one); no rule can catch those without a second
opinion. A decoder that keeps <b>one letter</b> (the most confident) and <b>six digits</b> (merge boxes that overlap)
would target {RS['more than 6 digits'] + RS['two letters']} of the {BW['n']} wrong plates before any fusion.</p>
"""

S['b7'] = f"""
<h2>Finding 6: where it helps (conditions and sources)</h2>
{fig('fig_conditions.png', 500, '<b>Figure 11.</b> Plate exact by condition and source. Grey dot: char-YOLO alone; square: '
     'the paper\'s val-chosen vote; blue dot: our fallback; ring: oracle. Where the blue dot sits in the ring, our rule '
     'already reaches what any fusion of these four models could reach.')}
{table(['Group (plates)', 'char-YOLO', 'CRNN', 'slot', 'ViTSTR', 'paper vote', 'our fallback', 'oracle'],
       [[f"{k} ({v['n']})", f"{v['chars_v8n']:.1f}", f"{v['crnn']:.1f}", f"{v['slot']:.1f}", f"{v['vitstr']:.1f}",
         f"{v['fusion']:.1f}", f"<b>{v['cascade']:.1f}</b>", f"{v['oracle']:.1f}"]
        for k, v in [('clear', COND['clear']), ('night', COND['night']), ('blurry', COND['blurry']),
                     ('KUET', SRC['kuet']), ('LPDBA', SRC['lpdba']), ('ANPR', SRC['anpr']), ('NightLPBD', SRC['nightlpbd']),
                     ('BLPD', SRC['blpd'])]],
       [2.2, 1.1, 1.0, 1.0, 1.0, 1.3, 1.3, 1.1], num=(1, 2, 3, 4, 5, 6, 7))}
{ul([f"The fallback helps most on <b>blurry</b> plates (+{COND['blurry']['cascade'] - COND['blurry']['chars_v8n']:.1f} points) "
     f"and on KUET (+{SRC['kuet']['cascade'] - SRC['kuet']['chars_v8n']:.1f}); there the oracle is still "
     f"{COND['blurry']['oracle'] - COND['blurry']['cascade']:.1f} / {SRC['kuet']['oracle'] - SRC['kuet']['cascade']:.1f} points "
     "higher, so a stronger second reader would help most there.",
     f"<b>Night: no gain this time</b> ({COND['night']['chars_v8n']:.1f} % for char-YOLO, the fallback and the oracle). In v1.0 "
     f"night was the best case (+{V10['by_condition']['night']['cascade'] - V10['by_condition']['night']['chars_v8n']:.1f}). "
     f"With {COND['night']['n']} night plates one plate is {100 / COND['night']['n']:.1f} points, so per-group gains of 2-3 "
     "points are not stable between runs; quote the overall test, and per group only the direction.",
     "The paper's vote is at or below char-YOLO alone in every group.",
     f"Weather tests from the notebook (char-YOLO alone): real fog copies {Y['extra_tests']['test_fog']:.1f} % (26 plates), "
     f"synthetic rain {Y['extra_tests']['test_rain_synth']:.1f} %. CRNN drops to {C['extra_tests']['test_fog']:.0f} % in fog, "
     "so it cannot help there."])}

<h1>13. Speed: does it fit our real-time target?</h1>
{table(['Where', 'char-YOLO alone', 'Our fallback (avg per plate)', 'Plates per second'], [
    ['Kaggle T4 (kit timing, batch 1, model step)', f"{MSY:.2f} ms", f"{FB['ms']:.2f} ms (CRNN on {FB['called_test_%']:.1f} %)", f"≈ {1000 / FB['ms']:.0f}"],
    [f"This PC, {L['gpu']} (whole reader: preprocess, model, decode, layout check)", f"{TG['ms_mean_chars_only']:.2f} ms", f"{TG['ms_mean']:.2f} ms (median {TG['ms_median']:.2f})", f"≈ {TG['plates_per_s']:.0f}"],
    [f"This PC, CPU only ({L['cpu']}, {TC['threads']} threads)", f"{TC['ms_mean_chars_only']:.2f} ms", f"{TC['ms_mean']:.2f} ms (median {TC['ms_median']:.2f})", f"≈ {TC['plates_per_s']:.0f}"],
], [3.6, 1.6, 3.0, 1.4])}
<p class="small">Local timing: the first {TG['plates']} test crops, already in memory, after 20 warm-up plates; CRNN ran on
{TG['crnn_called']} of them. This is the OCR step only (plate detection is not included). The CPU is a fast desktop
processor; a laptop CPU will be slower, so measure the full pipeline on the target machine before claiming RQ3.</p>
"""

S['b8'] = f"""
<h1>14. Ours vs the proposal's targets</h1>
{table(['Proposal target', 'Where we are (this run)', 'Status'], [
    ['Character recognition accuracy ≥ 90 % on test crops (RQ2)', f"char-YOLOv8n {Y['test']['char_acc_%']:.1f} %; with the fallback {FB['test']['char_acc_%']:.1f} %", 'met at character level'],
    ['Plate read exactly (our stricter view)', f"{ex('chars_v8n')} % → {FBX:.2f} % with the fallback (95 % CI of the plate rate about ±2.4)", 'close to 90 %; next fixes in section 20'],
    ['≥ 10 FPS full pipeline (RQ3, O4)', f"OCR step {FB['ms']:.1f} ms per plate on a T4, {TC['ms_mean']:.1f} ms on a desktop CPU", 'OCR is not the bottleneck; full pipeline not yet timed'],
    ['BRTA-format-aware correction (O3)', f"layout check finds {DET['format']['recall_%']:.0f} % of char-YOLO's errors with 100 % precision; fallback {GAIN_O:+.2f} points", 'first working piece, replicated'],
    ['Multi-frame voting (RQ3, O4)', 'not tested here (images only)', 'this paper\'s rule + our filter, next'],
], [3.2, 4.6, 2.2])}
"""

# ======================================================================================== part C
S['c1'] = f"""
<p class="part">PART C · USE IT</p>
<h1>15. How to use this paper in our research</h1>
{table(['Use', 'What to do', 'Links to'], [
    ['1. Final OCR choice', f"Use char-YOLO as the main reader and CRNN as a fallback when the BRTA layout fails ({FBX:.2f} %, {FB['ms']:.1f} ms). Report it as plate exact vs ms per plate, like the paper's Table 6 (our Figure 9).", 'O3, RQ2'],
    ['2. Grammar-aware fusion', 'Make the BRTA layout part of every fusion rule. It is the Bangla-specific part the paper does not have, and it never hurts a right reading.', 'O3 (format-aware correction)'],
    ['3. Multi-frame voting in video', 'The same rule, but the "models" are the frames of one tracked vehicle: drop invalid readings, take the majority string, stop OCR once 3 valid readings agree. The proposal already cites this paper for it.', 'RQ3, O4'],
    ['4. A team ensemble for free', 'p24_fusion.py accepts any run folder with ocr_preds_val.csv and ocr_preds_test.csv on the same split. The PARSeq (#31), VLM (#22) and ved.py scripts write both files, so teammates\' readers can join without retraining. ' +
     f"Tried on 24 Oct with paper #2's character model as the helper: {XB['test']['plate_exact_%']:.2f} % "
     f"({XB['vs_best_single']['fixed']} fixed / {XB['vs_best_single']['broken']} broken), and the oracle rises to "
     f"{XF['oracle_any_model_right_test_%']:.2f} %.", 'O5, team work'],
    ['5. A reporting standard', 'Choose on validation, report test once, give the confidence interval, use a McNemar test for "A beats B", and run more than one seed. This answers the weak points we found in the paper and the run-to-run spread we measured.', 'all RQs'],
    ['6. Robustness argument', 'Show per-condition results (night, blur, fog) for single vs fused readers: the paper\'s real claim is fewer bad cases, which is what RQ4 asks. Report direction per group, not small per-group gains.', 'RQ4'],
], [1.9, 6.3, 1.8])}

<h1>16. What benefit do you get from it?</h1>
{ul([f"<b>A real, significant and replicated gain at almost no cost:</b> {GAIN_O:+.2f} points plate exact "
     f"({SIGO['fixed']} fixed, {SIGO['broken']} broken, p = {SIGO['p_mcnemar']:.4f}) in two independent trainings, for "
     f"{FB['ms'] - MSY:+.2f} ms per plate and no new training.",
     f"<b>A free error detector:</b> the layout check flags {DET['format']['flagged_%']:.0f} % of readings, all of them wrong. "
     "In a live system you can reject them or wait for the next video frame; accepted readings are "
     f"{DET['format']['acc_when_accepted_%']:.1f} % right.",
     '<b>A clear answer to "why not just ensemble everything?"</b> We can show with numbers (oracle, error overlap, '
     'calibration) when fusion helps and when it does not. Faculty like a negative result that is explained.',
     '<b>A methods contribution for the final report:</b> grammar-aware fusion for Bangla plates, with fair validation-based '
     'selection, significance tests and a replication, which the original paper lacks.',
     '<b>Direction for the next experiments:</b> the oracle says better voters are needed, the error breakdown says the '
     'char-YOLO decoder should keep one letter and six digits, the calibration plot says never to use HC, and the '
     'run-to-run spread says to use seeds.',
     f"<b>A dataset fix for the whole team:</b> every teammate's OCR test contained the BLPD whole photos; v1.1 fixed them "
     f"and this run confirms it (char-YOLO {v10('char-YOLOv8n')['v10_19']:.0f} → {R19['chars_v8n']:.0f} % on those rows).",
     '<b>Ready-to-use weights</b> for the team: char-YOLOv8n and the CRNN fallback on GitHub, tested on this PC (section 21).',
     '<b>Literature value:</b> a peer-reviewed reference (CIARP, LNCS) for both model fusion and multi-frame voting in the '
     'related-work section.'])}
"""

S['c2'] = f"""
<h1>17. What to show the faculty (9 slides)</h1>
{table(['#', 'Slide title', 'What is on it', 'Source in this report'], [
    ['1', 'The question', 'single OCR models are uneven; can combining several help? how much? which?', 'section 1'],
    ['2', 'The paper\'s method', '12 models, HC / MV / MVCP, tie-breaks', 'Figure 1 A'],
    ['3', 'The paper\'s results', '92.4 → 97.6 % (8 sets), 87.6 → 90.3 % (unseen sets), HC worst, 4-6 fast models', 'Figure 2, section 5'],
    ['4', 'Our critique', 'chosen on test; one run; per-dataset best already 96.8 %, so fusion adds 0.8 there', 'section 6'],
    ['5', 'Our experiment', f"4 Bangla OCR models, dataset v1.1, val-chosen; best single {ex('chars_v8n')} %, paper's vote {KX:.2f} % (n.s.)", 'Figure 4'],
    ['6', 'Why plain fusion fails here', f"{OC['0']} of {BW['n']} errors shared by all; CRNN overconfident ({CONF['crnn']['wrong_with_conf_ge_0.9_%']:.0f} %)", 'Figures 7 and 8'],
    ['7', 'Our fix: BRTA-aware fallback', f"{FBX:.2f} % ({GAIN_O:+.2f}, {SIGO['fixed']} / {SIGO['broken']}, p = {SIGO['p_mcnemar']:.4f}) at {FB['ms']:.1f} ms; every invalid reading is wrong", 'Figure 1 B, Figure 9, example table'],
    ['8', 'Replication on the fixed dataset', f"same gain and same 12 / 0 in both runs; the 19 re-cut BLPD rows {v10('char-YOLOv8n')['v10_19']:.0f} → {R19['chars_v8n']:.0f} %; run-to-run spread 1-3 points", 'Figures 5 and 6, section 11'],
    ['9', 'What it means for our system', 'final OCR, decoder fix, video voting, seeds, team ensemble', 'sections 15 and 20'],
], [0.3, 2.1, 5.2, 2.4])}
<div class="box"><p><b>Keep it honest:</b> say "on our test split" for every number of ours, say that two of our four
models are undertrained, that all choices were made on validation, and that each number comes from one training run
(two runs for the main result). Do not compare 88.6 % with the paper's 97.6 % as if it were the same test.</p></div>

<h1>18. What to explain: a 3-minute script</h1>
<p>"This paper, from the Brazilian group that also wrote our paper #23 on near-duplicates, asks a simple question: if we
read the same plate with several different OCR models and combine the answers, do we read better? They trained 12
well-known models on 8 public datasets and combined them with three rules: trust the most confident model, take the
majority string, or vote character by character. The majority string wins. The mean accuracy goes from 92.4 to 97.6
percent, and on unseen datasets from 87.6 to 90.3. Trusting the most confident model is the worst rule, because models
are often confidently wrong. And four to six fast models give most of the gain.</p>
<p>When we read the tables closely, we found two weak points. They picked the best models on test results, and part of
the gain only replaces knowing which model is best for each dataset: per dataset, the best single model already averages
96.8 percent.</p>
<p>We repeated the idea on Bangla plates with four different models trained on our merged dataset, and we chose
everything on validation. We ran it twice: once before and once after we fixed 19 test images that were whole car
photos. Both times, the paper's voting gave no significant gain; this time it was even {abs(GAIN_K):.1f} points lower.
The reason is measurable: on {OC['0']} of the {BW['n']} plates our best model gets wrong, every other model is wrong too.
Fusion cannot invent an answer that no model has. We also saw the paper's confidence problem: CRNN gives
{CONF['crnn']['wrong_with_conf_ge_0.9_%']:.0f} percent of its wrong answers a confidence above 0.9.</p>
<p>Then we used something the paper does not have: the Bangladeshi plate format. A reading must have a district, one
letter and six digits. Every reading that breaks this rule was wrong. So we keep our character-YOLO reading when it is
valid, and ask CRNN only for the {DET['format']['flagged_%']:.0f} percent that are invalid. That gives {FBX:.1f} percent,
plus {GAIN_O:.1f} points, with {SIGO['fixed']} plates fixed and none broken, which is significant, and it costs less than
half a millisecond per plate on average. We got exactly the same gain in both runs.</p>
<p>For our project this means three things: the final reader is character-YOLO plus a format-aware fallback; the same
format filter and majority vote will combine readings over video frames; and to gain more, we should fix how
character-YOLO turns boxes into text and add stronger, more different second readers, with several seeds."</p>
"""

S['c3'] = f"""
<h1>19. Questions the faculty may ask</h1>
<p class="q">Q1. The paper gets +5 points from fusion. Why do you get nothing?</p>
<p>Fusion needs several strong models that make different mistakes. The paper had 12 models between 86 and 92 %. We have
one strong model, one medium and two weak ones, and their errors overlap: the oracle (any model right) is only
{ORA:.1f} %, so no voting rule can give more than {ORA - Y['test']['plate_exact_%']:+.1f}. Also, part of the paper's +5 is
against one fixed model; against the best model per dataset the gain is 0.8.</p>
<p class="q">Q2. Isn't +1.8 points also noise?</p>
<p>No. On the same {NT} plates the fallback fixes {SIGO['fixed']} and breaks {SIGO['broken']}; the exact McNemar test gives
p = {SIGO['p_mcnemar']:.4f} and the bootstrap interval of the gain is {SIGO['gain_ci95'][0]:+.1f} to
{SIGO['gain_ci95'][1]:+.1f} points. A second training on the corrected data gave the same 12 / 0. The paper's rule
({SIGK['fixed']} fixed, {SIGK['broken']} broken) gives p = {SIGK['p_mcnemar']:.2f}.</p>
<p class="q">Q3. Your numbers changed between the two runs. Which do we trust?</p>
<p>The v1.1 run: it uses the corrected data. Overall scores rose because the 19 whole photos became crops. On the 632
unchanged plates single models moved by 1-3 points (slot CNN about 10), which is normal run-to-run variation. That is why
we test gains on the same plates (McNemar) and why the next step is 3 seeds. The fallback's gain did not move.</p>
<p class="q">Q4. Did you tune on the test set?</p>
<p>No. The best model, the tie-break model, the rule and the subset were all chosen on the {NV} validation plates; the test
numbers were read once. This is stricter than the paper, which ranked models on test results.</p>
<p class="q">Q5. Is the format check cheating? It knows what plates look like.</p>
<p>It uses only the public BRTA format, never the label of the plate. It is the same knowledge a human reader has. For a
plate outside the format (army, diplomatic, old plates), char-YOLO's reading breaks the layout anyway; the rule then
keeps it unless CRNN produces a valid BRTA string. Such plates are not in our test set, so this case is not measured yet.</p>
<p class="q">Q6. Why are ViTSTR and the slot CNN so weak?</p>
<p>They were trained for about {A['steps_per_epoch']['slot'] * 100:,} (slot) and {A['steps_per_epoch']['vitstr'] * 60:,}
(ViTSTR) steps on 5,600 crops, against 300,000 iterations on tens of thousands of images in the paper. ViTSTR starts to
overfit after epoch {CB['vitstr']['epoch']}. We did not tune them because the question was fusion; the next step is to
train them longer with synthetic plates, or replace them with PARSeq from paper #31.</p>
<p class="q">Q7. Why did the gain at night disappear?</p>
<p>There are only {COND['night']['n']} night plates, so one plate is {100 / COND['night']['n']:.1f} points. In v1.0 the
fallback fixed two night plates. In v1.1 no other model reads any night plate that char-YOLO misses (the oracle at
night equals char-YOLO), so there is nothing for a fallback to fix. Gains on small groups move between runs; the overall
gain did not.</p>
<p class="q">Q8. Why not calibrate the confidences and use HC?</p>
<p>The paper tried normalising confidences and it did not help. Temperature scaling on validation is a cheap test we can
add, but MV-BM plus the format check already avoids the problem.</p>
<p class="q">Q9. How does this help with video?</p>
<p>A vehicle is seen in many frames. The same rule combines the frames: drop invalid readings, take the majority string,
and stop reading once three valid readings agree. That saves time and is our RQ3 and O4.</p>
<p class="q">Q10. Do you reach the 90 % target?</p>
<p>Character accuracy is {FB['test']['char_acc_%']:.1f} %, above the target. Whole-plate accuracy is {FBX:.1f} % with the
fallback; the decoder fix in section 20 targets {RS['more than 6 digits'] + RS['two letters']} of the {BW['n']} remaining
wrong plates.</p>

<h1>20. Next steps (in order)</h1>
{ul([f"<b>Decoder fix in char-YOLO (O3):</b> keep one letter (the most confident), keep six digits (merge boxes that overlap, "
     f"class-agnostic), retry with a lower box threshold when a district or letter is missing. It targets "
     f"{RS['more than 6 digits']} plates with extra digits and {RS['two letters']} with a doubled letter. Test it offline on "
     "the saved best.pt (no retraining), validation first.",
     '<b>Seeds:</b> 3 seeds for char-YOLO and CRNN, so the final table has mean ± spread (we measured 1-3 points between runs).',
     '<b>Better second readers:</b> PARSeq (#31) or a longer-trained ViTSTR with synthetic plates; re-run p24_fusion.py with '
     f"them and watch the oracle (now {ORA:.2f} %). Fusion only pays when it rises.",
     '<b>Video:</b> apply the format filter + majority vote over tracked frames and measure video-level accuracy and FPS (RQ3).',
     '<b>Team table:</b> done on 24 Oct (add_results.py, 4 v1.1 rows). There our char-YOLOv8n (86.79 %) and paper #2\'s '
     'YOLOv12n character model (87.71 %) are tied on the same plates (24 vs 18 plates, McNemar p = 0.44).'], 'ol')}
"""

S['c4'] = f"""
<h1>21. Files: GitHub and Drive</h1>
{table(['Where', 'File', 'What'], [
    ['GitHub upload folder', M('weights/paper24/MD.Rokib.Hasan.Oli/'), f"prepared in {M('After Train/github_upload/')}; upload the folder as it is"],
    ['', M('best_chars_v8n.pt'), f"char-YOLOv8n, the run's best.pt, unchanged ({L['files']['best_chars_v8n.pt'] / 1e6:.1f} MB)"],
    ['', M('best_crnn_fp16.pt'), f"CRNN-CTC weights stored in FP16 + its token list and input size ({L['files']['best_crnn_fp16.pt'] / 1e6:.1f} MB; the FP32 file is 30.8 MB, over GitHub's 25 MB web limit); same accuracy, tested"],
    ['', M('paper24_results.zip'), 'the run\'s results without weights: predictions, results.json, fusion table, logs, code'],
    ['', M('results_vs_paper.md'), 'the kit\'s ours-vs-paper table from the run'],
    ['', M('Paper24_Study_Report.pdf'), 'this report'],
    ['', M('README.md'), 'what is in the folder, quick start (tested), results, checksums'],
    ['Team Drive', M('498R / weights / paper24 / paper24_outputs.zip'), f"everything incl. all four models' weights (124 MB); prepared in {M('After Train/for_drive/')}"],
    ['This PC', M('After Train/results.zip'), 'the full Kaggle output (298 MB); the v1.0 output was overwritten by it'],
    ['This PC', M('report_src/'), f"rebuild: {M('analysis.py')} → {M('local_check.py')} → {M('charts.py')} → {M('build_report.py')}; v1.0 sources in {M('v1.0_run/')}"],
], [1.6, 3.2, 5.2])}

<h2>Appendix: where every number comes from</h2>
{table(['Number', 'Source'], [
    ['Paper numbers', 'arXiv 2309.04331v2: Table 3 (p.9), Table 4 (p.9), Table 5 (p.11), Table 6 (p.12), text p.2-12 (re-read 24 Oct 2026)'],
    ['Per-dataset-best 96.8 % / 87.6 %', 'our arithmetic on Table 3 and Table 5 (best value per column, then the mean)'],
    ['Single models, fusion table, oracle', 'Kaggle run 22 Oct 2026 (v1.1): results.zip → runs/*/results.json, runs/fusion/results.json'],
    ['Overlap, McNemar, bootstrap, calibration, extra rules, groups', 'analysis.py on runs/*/ocr_preds_{val,test}.csv → analysis.json'],
    ['v1.0 numbers', 'the first analysis (20-21 Oct): report_src/v1.0_run/analysis.json'],
    ['Conditions, training-label counts, re-cut rows', 'ocr/test_conditions.csv and ocr/train.csv in BanglaLPR_AllInOne_v1_1.zip; dataset/docs/blpd_crop_fix.md'],
    ['Local re-score and timing', f"local_check.py on this PC ({L['gpu']}, {L['cpu']}), 24 Oct 2026 → local_check.json"],
], [3.2, 6.8])}
"""

FLOWS = [['cover'], ['a1', 'a2'], ['a2b', 'a3'], ['b1'], ['b2'], ['b3'], ['b4'], ['b5'], ['b6'], ['b7'], ['b8'], ['c1', 'c2', 'c3'], ['c4']]
FIG_W = {'fig_pipeline.png': 440, 'fig_paper.png': 460, 'fig_ours.png': 500, 'fig_overlap.png': 470,
         'fig_confidence.png': 480, 'fig_speed.png': 500, 'fig_conditions.png': 500, 'fig_curves.png': 480,
         'fig_errors.png': 480, 'fig_v10_v11.png': 500, 'fig_recut.png': 470}


def build():
    bangla_font()
    arch = fitz.Archive()
    arch.add(fitz.Archive(r'C:\Windows\Fonts'))
    arch.add(fitz.Archive(FONT_DIR))
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
    figs = [(i + 1, round(r['bbox'][2] - r['bbox'][0])) for i, pg in enumerate(doc) for r in pg.get_image_info()
            if r['width'] > 1000]
    print('figures (page, width pt):', figs)
    if len(figs) != len(FIG_W) or min(w for _, w in figs) < 400:
        print('WARNING: a figure is missing or was shrunk to fit the end of a page')
    for i, pg in enumerate(doc):
        ys = [b[3] for b in pg.get_text('blocks') if b[4].strip()]
        print(f'  page {i + 1}: text ends at y={max(ys):.0f}' if ys else f'  page {i + 1}: empty')
    for i, pg in enumerate(doc):
        pg.insert_text((ML, PH - 26), 'Paper #24 study report (v1.1 update) · 498R BanglaLPR', fontsize=7.5, fontname='helv',
                       color=(0.45, 0.45, 0.43))
        t = f'{i + 1} / {n}'
        pg.insert_text((PW - ML - fitz.get_text_length(t, 'helv', 7.5), PH - 26), t, fontsize=7.5, fontname='helv', color=(0.45, 0.45, 0.43))
    doc.set_metadata({'title': 'Paper #24 study report (v1.1 update): Leveraging Model Fusion for Improved License Plate Recognition',
                      'author': 'Prepared for MD. Rokib Hasan Oli',
                      'subject': 'Paper study, v1.1 Kaggle re-run on 498R data, run check, research use, faculty guide'})
    doc.save(OUT, garbage=3, deflate=True)
    doc.close()
    print('pages', n, OUT)


if __name__ == '__main__':
    build()
