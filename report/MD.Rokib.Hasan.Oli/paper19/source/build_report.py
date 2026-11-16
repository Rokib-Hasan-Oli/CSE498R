"""Paper #19 study report (dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Wojcik et al., IJCNN 2026, arXiv 2604.08741 v1, 7 pages; page numbers = PDF pages); the authors'
code (github.com/UFPR-IPASP-PR/LPLCv2-Dataset at 7a1d3f6: models/loss.py, dataset/dataset_utils.py, configs/train/*.json,
scripts/run_experiments.sh); the v1.1 Kaggle run (`After train/results.zip`, `paper-19.log`, 26 Oct 2026) via
analysis.json; the local re-check (local_check.py) and the local extra runs (local_train.py) on the RTX 5060 Ti; the
paper #24 run's per-plate readings (char-YOLO, CRNN) of the same 651 test crops; the 498R proposal (RQ2-RQ4, O3, O5).
Run order: local_check.py -> run_local_train.sh -> analysis.py -> charts.py -> build_report.py.
"""
import json
import os
import sys

from part_a import part_a
from report_head import M, f1, f2, fig, pv, table, ul, write_pdf

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'LPLCv2_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
R, G, B, LC = A['run'], A['gate'], A['boot'], A['local_check']
C = R['classifiers']
ST, SG, BC, SP = A['strong'], A['seed_groups'], A['by_condition'], A['same_plate']
DL, RJ, MJ = A['deploy_by_label'], A['rejected_80'], A['majority']
LOGT = {'readers_crnn': 51.2, 'readers_slot': 46.2, 'legibility_labels': 0.2, 'cls_b4_ema': 12.5, 'cls_b4_ce': 12.7,
        'cls_qf_ema': 11.2, 'cls_qf_ce': 11.0, 'gate': 0.2, 'total_s': 4700.7}                       # paper-19.log
CL = ['perfect', 'good', 'poor', 'illegible']
LAB = R['labels']
g = lambda k, c='80%': G[k][c]
ci = lambda v: f"{v[0]:+.2f} to {v[1]:+.2f}"
SPD = LC['speed_ms_batch1']
LR = A['learned_weights']
FAIL = SG.get('failure_quality_ce')
BAL = SG.get('balanced_baseline4_ema')
TXT = SG.get('text_baseline4_ema')
N_LOCAL = len(A['local_runs'])
LY = A['layout']

# ---------------------------------------------------------------------------------------------- facts the text relies on
assert R['dataset_version'] == '1.1' and LC['dataset_version'] == '1.1'
assert R['crnn_test']['rows_decoder']['plate_exact_%'] == 73.43 and A['n']['test'] == 651 and A['n']['val'] == 766
assert LC['crnn']['sha256'].startswith('0110631e') and LC['crnn']['local_test_exact_%'] == 73.43
assert LC['crnn']['val_same_text_as_p24_run'] == 766 and LC['crnn']['test_same_text_as_kaggle'] == 648
assert all(LC['classifiers'][n]['argmax_agree_%'] == 100.0 and LC['classifiers'][n]['local_micro_F1_%'] == LC['classifiers'][n]['kaggle_micro_F1_%']
           for n in C)
assert A['gate_reproduced'] is True
assert [C[n]['best_epoch'] for n in ('cls_b4_ema', 'cls_b4_ce', 'cls_qf_ema', 'cls_qf_ce')] == [4, 4, 2, 1]
assert C['cls_b4_ema']['test']['micro_F1_%'] == 51.31 and C['cls_b4_ce']['test']['micro_F1_%'] == 51.92
assert C['cls_qf_ema']['test']['micro_F1_%'] == 71.74 and C['cls_qf_ce']['test']['micro_F1_%'] == 70.81
assert A['ema_vs_ce']['cls_b4_ema']['p'] > 0.05 and A['ema_vs_ce']['cls_qf_ema']['p'] > 0.05
assert g('ocr_confidence') == 84.45 and g('cls_b4_ce') == 81.0 and max(g(k) for k in C) == 81.0
assert all(B[k]['d80_ci95'][1] < 0 and B[k]['dAURC_ci95'][0] > 0 for k in C)                  # every classifier loses
assert B['conf + layout']['dAURC_ci95'][1] < 0 and B['learned (val): conf + layout + classifiers']['dAURC_ci95'][1] < 0
assert B['classifiers_on_top_of_conf_layout']['d80_ci95'][0] < 0 < B['classifiers_on_top_of_conf_layout']['d80_ci95'][1]
assert B['classifiers_on_top_of_conf_layout']['dAURC_ci95'][1] < 0
assert G['ocr_confidence']['AUROC'] > max(G[k]['AUROC'] for k in C)
assert ST['fallback_exact_%'] == 88.63 and ST['boot_cls_vs_free']['d80_ci95'][1] < 0
assert ST['boot_freecls_vs_free']['d80_ci95'][0] < 0 < ST['boot_freecls_vs_free']['d80_ci95'][1]
assert DL['perfect']['exact_%'] > DL['good']['exact_%'] > DL['poor']['exact_%'] > DL['illegible']['exact_%']
assert BC['night']['rejected_by_cls_b4_ce'] > BC['night']['rejected_by_ocr_confidence']
assert A['train_labels_by_label_source']['auto']['perfect'] == 1776
DRAFT = os.environ.get('DRAFT') == '1'                   # layout check before the local runs are all done
for k in () if DRAFT else ('failure_quality_ce', 'kit_baseline4_ema', 'kit_baseline4_ce', 'balanced_baseline4_ema', 'text_baseline4_ema',
          'kit_quality_ema', 'kit_quality_ce'):
    assert k in SG and SG[k]['n'] >= 2, f'local runs missing: {k}'
assert all(x < g('ocr_confidence') for k in SG for x in SG[k]['gate80'])                      # no run beats confidence

S = part_a(A)
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER STUDY REPORT · DATASET v1.1 RUN</p>
<h1 style="font-size:18pt; margin-bottom:3pt">LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification</h1>
<p style="font-family:H; font-size:11.5pt; color:#1c5cab; margin-bottom:8pt">The paper, our legibility run on dataset v1.1,
whether a "reject or read" legibility gate helps our Bangla plate reader, and what to show and say to the faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 29 October 2026<br>
Paper: L. Wojcik, E. A. F. Machoski, E. Nascimento Jr., R. Laroca and D. Menotti, IJCNN 2026 (arXiv 2604.08741 v1,
7 pages; page numbers are PDF pages). Code: github.com/UFPR-IPASP-PR/LPLCv2-Dataset (checked at commit 7a1d3f6); data on
e-mail request only.<br>
Our runs: (1) {M('paper19_kaggle.ipynb')}, Tesla T4 x2 on Kaggle, 26 Oct 2026, dataset <b>v1.1</b> (confirmed in
{M('environment.json')}), {LOGT['total_s'] / 60:.0f} min; (2) on our own PC (RTX 5060 Ti), 27-29 Oct 2026: every Kaggle number
re-checked crop by crop, plus {N_LOCAL} extra training runs (seeds, the authors' real
recipe, the paper's text version of the loss, a direct failure predictor) and new gate analyses.</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> A 4-class "how readable is this plate" label (perfect / good / poor / illegible) for
41,487 Brazilian radar-camera plates, the old benchmark's labels corrected, and a ResNet-50 trained with a class-reweighting
"EMA loss": 89.8 % micro F1 on 4 classes and 96.7 % on the binary <b>Quality Filter</b> (suitable vs unsuitable). The idea:
read good plates, enhance poor ones, drop illegible ones. The paper never tests whether this helps an OCR.</p>
</div>
<div class="good">
<p><b>Is the v1.1 run OK? Yes.</b> All 8 jobs ended with {M('code=0')} in {LOGT['total_s'] / 60:.0f} min (planned 2 h), dataset
v1.1 is confirmed, the code and notebook are exactly the kit. On our PC the four classifiers give the <b>same test F1</b>
and the same class for 100 % of the 651 crops, and the deployment CRNN is <b>byte-identical</b> to the Model Fusion study's (SHA-256
{M(LC['crnn']['sha256'][:8] + '…')}; {f2(R['crnn_test']['rows_decoder']['plate_exact_%'])} % plate exact). One thing to know: every
classifier reached its best validation accuracy at epoch 1-4 and then memorised the noisy labels; that is a property of
the labels, not a bug (section 7).</p>
</div>
<div class="warn">
<p><b>The answer for our research: a legibility classifier is NOT a better reject gate than the OCR's own confidence.</b>
(1) Our labels (made by two cross-fitted readers, no human legibility labels exist for Bangla plates) are learnable only
to {f1(C['cls_b4_ce']['test']['micro_F1_%'])} % on 4 classes and {f1(C['cls_qf_ema']['test']['micro_F1_%'])} % on the Quality Filter (paper:
89.8 / 96.7; majority-class guess: {f1(MJ['4class_%'])} / {f1(MJ['quality_%'])}). (2) <b>Gate:</b> keeping the 80 % most
confident plates, the CRNN is right on <b>{f2(g('ocr_confidence'))} %</b> of them with its own confidence, but only
<b>{f2(g('cls_b4_ce'))} %</b> with the best classifier (difference {ci(B['cls_b4_ce']['d80_ci95'])} points, 95 % CI); every
classifier loses. (3) The EMA loss changes nothing on 4 classes ({pv(A['ema_vs_ce']['cls_b4_ema']['p'])}; same mean over
three seeds) and adds about one point on the Quality Filter, which does not help the gate. (4) <b>Free signals win:</b>
confidence + the BRTA layout check gives {f2(g('conf + layout'))} %; for our best reader (the Model Fusion study's fusion reader, 88.63 %) the
agreement of char-YOLO and CRNN plus their confidences gives <b>{f2(ST['free signals (val-fitted)']['80%'])} %</b> at 80 % coverage,
the classifier {f2(ST['cls_b4_ce']['80%'])} %. (5) Added on top of the free signals, the classifiers add nothing measurable at
80 % coverage. (6) None of our {N_LOCAL} extra local runs, including the authors' real recipe and a
classifier trained directly on "will the CRNN fail?", beats the plain confidence.</p>
<p><b>What to tell the faculty in one sentence.</b> "We rebuilt the LPLCv2 legibility classifier for Bangla plates and
asked the question the paper skips: does it help the OCR? It does not; the reader's own confidence, our BRTA layout check
and the agreement of two readers already give a reject-or-read gate that keeps 80 % of the plates at
{f1(ST['free signals (val-fitted)']['80%'])} % accuracy for free, so our pipeline will report 'unreadable' with those signals
instead of a 23.5 M-parameter classifier."</p>
</div>
<p class="small"><b>Parts:</b> A. Understand the paper (sections 1-5) · B. Our v1.1 experiment (6-15, read before you quote
a number) · C. Use it (16-22: research use, benefits, slides, script, Q&amp;A, next steps, GitHub).</p>
"""

S['b1'] = f"""
<p class="part">PART B · OUR EXPERIMENT ON DATASET v1.1</p>
<h1>6. What we ran</h1>
<p>LPLCv2 is Brazilian and only on request, and nobody has labelled the legibility of Bangla plates. So the kit makes an
<b>OCR-difficulty label</b> for every one of our 7,024 OCR crops, trains the paper's classifier on it, and then asks the
practical question: <b>which plates should we refuse to read?</b></p>
{fig('fig_pipeline.png', 470, 'Figure 2. The kit. Top: how the training labels are made (cross-fitting, so no crop is judged by a reader that saw it) and the classifier. Bottom: the gate test on the 651 test crops; the classifier score and the CRNN confidence compete as the "accept this plate" score.')}
{table(['Job (GPU)', 'What it does', 'Time', 'Status'], [
    [M('readers_crnn') + ' (0)', f"CRNN-CTC x 3, 60 epochs, AdamW 1e-3 + OneCycle, 48 x 256: on half A ({R['halves']['0']:,} crops) and half B ({R['halves']['1']:,}) of the OCR train split (halves = hash of the plate text, so one plate never sits in both), each reads the other half; half A also reads val and test; a third CRNN on all train crops = the deployment reader", f"{LOGT['readers_crnn']:.0f} min", 'code=0'],
    [M('readers_slot') + ' (1)', 'slot CNN (one class per character position) x 2, 100 epochs, same halves: the second reader family', f"{LOGT['readers_slot']:.0f} min", 'code=0'],
    [M('legibility_labels') + ' (CPU)', 'perfect = both readers exact · good = exactly one exact · poor = neither, best within 1 token · illegible = the rest; Quality Filter = perfect + good vs poor + illegible', '0.2 min', 'code=0'],
    [M('cls_b4_ema / cls_b4_ce') + ' (0 / 1)', 'ResNet-50 (ImageNet, all layers), 96 x 192 letterbox, Adam 1e-5, batch 64, x0.75 after 5 flat epochs, early stop 20 on val accuracy, 4 classes, EMA loss (as the authors\' code) vs cross-entropy', '12.5 / 12.7 min', 'code=0'],
    [M('cls_qf_ema / cls_qf_ce') + ' (0 / 1)', 'the same for the binary Quality Filter', '11.2 / 11.0 min', 'code=0'],
    [M('gate') + ' (CPU)', 'test crops sorted by each score (CRNN confidence, classifier suitability, their product); plate exact of the accepted plates at 100 / 90 / 80 / 70 / 50 % coverage; AURC; share flagged per condition', '0.2 min', 'code=0'],
    ['local (RTX 5060 Ti)', f"re-run all four classifiers and the CRNN on val + test; timing; {N_LOCAL} extra classifier trainings ({M('local_train.py')}); new gate analyses ({M('analysis.py')}): bootstrap intervals, AUROC, layout check, a val-fitted combination, the Model Fusion study's stronger reader", 'about 2.5 h', 'done'],
], [1.6, 5.2, 0.9, 0.7])}
<p class="small">"Plate exact" = the whole reading is right (every district word, letter and digit). Coverage = the share of
test plates the gate lets through to the reader. The deployment CRNN is the reader whose answers the gate filters; the
half readers only make labels.</p>
"""

cls_rows = [[M(n), str(C[n]['epochs_run']), str(C[n]['best_epoch']), f1(C[n]['best_val_acc']), f"{C[n]['curve'][-1][1]:.3f}",
             f2(C[n]['test']['micro_F1_%']), f"{C[n]['train_minutes']:.1f}"] for n in ('cls_b4_ema', 'cls_b4_ce', 'cls_qf_ema', 'cls_qf_ce')]
S['b2'] = f"""
<h1>7. Is the training OK?</h1>
{table(['Check', 'Result', 'OK?'], [
    ['Dataset version', f"{M('dataset_version: 1.1')} in environment.json; OCR split 5,607 / 766 / 651 crops (train 3,471 human + 2,136 auto)", 'yes'],
    ['All jobs finished', f"8 / 8 {M('code=0')}, {LOGT['total_s'] / 60:.0f} min of the planned 2 h; no job hit its time cap", 'yes'],
    ['Code = kit', f"the 9 files in {M('code/')} are byte-identical to the kit ({M('ved.py')} differs only in line endings and is not used); the notebook's 25 cells equal the kit's, {M('FAST = False')}", 'yes'],
    ['Kaggle numbers reproduce', f"our PC re-ran the 4 classifiers on the 651 test crops: same micro and per-class F1, same predicted class for 100 % of crops, probabilities within {max(LC['classifiers'][n]['max_abs_prob_diff'] for n in C):.4f}; the kit's gate numbers recomputed from the CSVs = results.json", 'yes'],
    ['Deployment CRNN', f"SHA-256 {M(LC['crnn']['sha256'][:16] + '…')} = the CRNN of the Model Fusion and HVI-CIDNet study runs (the kit is deterministic); test {f2(LC['crnn']['local_test_exact_%'])} % on our PC as on Kaggle ({LC['crnn']['test_same_text_as_kaggle']} / 651 identical texts, the other 3 wrong both times); its val readings equal the Model Fusion run's on all 766 crops", 'yes'],
    ['Readers that make the labels', f"half-A CRNN: {f1(A['readers']['crnn|test|half0'])} % test / {f1(A['readers']['crnn|val|half0'])} % val; half-A slot: {f1(A['readers']['slot|test|half0'])} / {f1(A['readers']['slot|val|half0'])} %; cross-read train: CRNN {f1(A['readers']['crnn|train|half0'])} / {f1(A['readers']['crnn|train|half1'])} %, slot {f1(A['readers']['slot|train|half0'])} / {f1(A['readers']['slot|train|half1'])} %. Weaker than the full CRNN ({f2(R['crnn_test']['rows_decoder']['plate_exact_%'])} %), as expected with half the data", 'yes'],
    ['Classifiers', 'see the table below: each stopped by early stopping (patience 20), far below the 80-epoch cap', 'yes, but…'],
], [1.4, 6.0, 0.6])}
{table(['Run', 'Epochs run', 'Best epoch', 'Best val acc (%)', 'Final train loss', 'Test micro F1 (%)', 'Min'], cls_rows,
       [1.4, 0.8, 0.8, 1.0, 1.0, 1.1, 0.6], num=(1, 2, 3, 4, 5, 6))}
{fig('fig_training.png', 460, 'Figure 3. Validation accuracy per epoch (Kaggle run). Dots = the epoch whose weights were kept. After epoch 1-4 the training loss keeps falling to almost zero while validation accuracy drops: the network memorises the training labels.')}
<div class="warn"><p><b>The "but".</b> Every classifier was best at epoch 1-4 and then <b>memorised</b> its training labels
(training loss 0.01-0.04 at the end). With a learning rate of only 1e-5 this happens when labels are noisy: the network
first learns the shared signal, then the noise. Early stopping kept the right weights, so the numbers are valid; but it tells
us that our labels, not the training recipe, are the limit (section 8). The paper's 1,000-epoch budget would not help.</p></div>
"""

ex_ill = A['examples']['illegible']
S['b3'] = f"""
<h1>8. Our labels: what they measure</h1>
{fig('fig_labels.png', 470, 'Figure 4. A: the four proxy classes per split (train is split into human-labelled and auto-labelled crops). B: how often the deployment CRNN (which never made a label) reads each test class right; the dashed line is the paper\'s motivating number: "poor" LPLCv1 plates were read right only 72 % of the time (p.4).')}
{ul([
    f"<b>The labels are meaningful.</b> The deployment CRNN, which made none of the labels, reads {f1(DL['perfect']['exact_%'])} % of the \"perfect\" test plates right, {f1(DL['good']['exact_%'])} % of \"good\", {f1(DL['poor']['exact_%'])} % of \"poor\" and {f1(DL['illegible']['exact_%'])} % of \"illegible\". The same staircase as the paper's motivation (good / perfect above 90 %, poor 72 %, p.4).",
    f"<b>The auto-labelled training crops are almost all \"perfect\"</b> ({A['train_labels_by_label_source']['auto']['perfect']:,} of {sum(A['train_labels_by_label_source']['auto'].values()):,}): their texts were produced by a reader in the first place, so readers agree with them. They are 67 % of all \"perfect\" training crops, but val and test contain no auto-labelled crops ({LAB['val']['perfect']} and {LAB['test']['perfect']} perfect crops there, all human-labelled). The classifier can learn \"looks like a Zenodo-25k crop\" instead of \"easy to read\".",
    f"<b>The labels are noisy.</b> The test set has {SP['plates']} plates photographed more than once ({SP['crops']} crops, {SP['pairs']} pairs of the same plate). Two crops of the same plate get the same 4-class label only {f1(SP['label_agree_%'])} % of the time ({f1(SP['label_far_%'])} % of pairs are two or more classes apart) and the same Quality Filter label {f1(SP['quality_agree_%'])} % of the time. Some of that is real (one photo is sharper), much is reader luck. The classifier's own predictions agree on {f1(SP['cls_b4_ce_pred_agree_%'])} % of those pairs.",
    "<b>\"Illegible\" mostly means \"unusual for our readers\", not \"blurred\".</b> Figure 5 shows it: one-line plates, green CNG plates, rare district names and odd layouts are perfectly readable to a human but fail both half readers. A legibility classifier then has to guess the reader's blind spots from the pixels.",
])}
{fig('fig_examples.jpg', 470, 'Figure 5. Four random test crops per proxy class (frame colour = class). Above each: whether the deployment CRNN read it right, its confidence and the 4-class classifier\'s suitability score P(perfect) + P(good). The second "good" and the third "illegible" crop are the same plate from two sources (ANPR, LPDB-A).')}
{table(['"Illegible" crop', 'True text', 'Deployment CRNN reading', 'Right?'], [
    [M(e['file'][:22]), f"<span class='bn'>{e['text']}</span>", f"<span class='bn'>{e['pred']}</span>", 'yes' if e['ok'] else 'no']
    for e in ex_ill], [2.0, 2.2, 2.4, 0.6])}
<p class="small">Typical: the reader falls back on the most common district ("<span class='bn'>ঢাকা মেট্রো</span>") for a plate it
has rarely seen (<span class='bn'>যশোর</span>, <span class='bn'>চট্ট মেট্রো</span>). That is a reader-and-data problem (rare classes), which
the Model Fusion study's fusion reader and the BRTA layout check attack directly.</p>
"""

S['b4'] = f"""
<h1>9. Results next to the paper</h1>
{table(['Metric', 'Paper (page)', 'Ours (v1.1, 651 test crops)'], [
    ['4 classes, micro F1, EMA loss', '89.8 (p.5, Table IV)', f"{f2(C['cls_b4_ema']['test']['micro_F1_%'])} (majority-class guess {f1(MJ['4class_%'])})"],
    ['4 classes, micro F1, cross-entropy', '88.7 (+ more images, p.5)', f2(C['cls_b4_ce']['test']['micro_F1_%'])],
    ['Per class F1 perfect / good / poor / illegible, EMA', '92.7 / 86.2 / 86.3 / 93.0', ' / '.join(f1(x) for x in C['cls_b4_ema']['test']['per_class_F1_%'])],
    ['Per class F1, cross-entropy', '92.8 / 86.0 / 86.0 / 92.9', ' / '.join(f1(x) for x in C['cls_b4_ce']['test']['per_class_F1_%'])],
    ['Quality Filter, EMA / cross-entropy', '96.7 / 96.9 (p.6, Table V)', f"{f2(C['cls_qf_ema']['test']['micro_F1_%'])} / {f2(C['cls_qf_ce']['test']['micro_F1_%'])} (majority {f1(MJ['quality_%'])})"],
    ['Quality Filter read from the 4-class model', '-', f"{f2(C['cls_b4_ema']['quality_from_4class']['micro_F1_%'])} (EMA) / {f2(C['cls_b4_ce']['quality_from_4class']['micro_F1_%'])} (CE)"],
    ['EMA vs cross-entropy (same crops)', '+1.1 on 4 classes, -0.2 on QF (not significant)', f"4 classes {A['ema_vs_ce']['cls_b4_ema']['a_only']} vs {A['ema_vs_ce']['cls_b4_ema']['b_only']} crops, {pv(A['ema_vs_ce']['cls_b4_ema']['p'])}; QF {A['ema_vs_ce']['cls_qf_ema']['a_only']} vs {A['ema_vs_ce']['cls_qf_ema']['b_only']}, {pv(A['ema_vs_ce']['cls_qf_ema']['p'])} (McNemar)"],
    ['Errors between neighbouring classes', f"{A['adjacent']['paper_fig4']['adjacent_%']} % (one fold, Fig. 4)", f"{A['adjacent']['cls_b4_ema']['adjacent_%']} % (EMA), {A['adjacent']['cls_b4_ce']['adjacent_%']} % (CE)"],
    ['Test labels perfect / good / poor / illegible', '18,425 / 10,180 / 7,520 / 5,362 (all LPLCv2)', ' / '.join(str(LAB['test'][c]) for c in CL)],
    ['Training crops', '~33,000 (p.4)', '5,607'],
    ['Epochs (early stop)', 'up to 1,000, patience 20', f"{C['cls_b4_ema']['epochs_run']} / {C['cls_b4_ce']['epochs_run']} / {C['cls_qf_ema']['epochs_run']} / {C['cls_qf_ce']['epochs_run']}, best 4 / 4 / 2 / 1"],
    ['Training time', 'not reported (RTX 6000)', '11-13 min per classifier on one T4'],
    ['Speed, batch 1', 'not reported', f"{SPD['classifier_gpu_fp32']} ms GPU (RTX 5060 Ti), {SPD['classifier_cpu4']} ms CPU (4 threads); {LC['params_M']['classifier']} M parameters"],
], [2.5, 2.4, 3.1])}
{fig('fig_confusion.png', 430, 'Figure 6. Row-normalised confusion matrices. A: the paper\'s EMA fold (Fig. 4): a clean diagonal. B: ours (EMA, test): "good" is mostly predicted as "poor"; "perfect" and "illegible" are recognised best, the middle classes are guesswork.')}
<p><b>How to read the gap.</b> Our labels are not the paper's labels: theirs are human judgements of the image, ours say
whether two particular readers succeeded. The F1 rows therefore show the <i>direction</i> only; they do not mean that the
method is worse on Bangla plates. What our numbers do show: (1) the classifier learns something real (51.9 % vs a
30.6 % majority guess; Quality Filter 71.7 % vs 57.9 %); (2) the two middle classes are the hard ones in both studies,
"good" is the worst for us because "exactly one reader was right" is mostly luck; (3) the EMA loss makes no difference,
as the paper itself found for its Quality Filter.</p>
"""

def grow(k, name):
    v = G[k]
    return [name, f2(v['100%']), f2(v['90%']), f"<b>{f2(v['80%'])}</b>", f2(v['70%']), f2(v['50%']), f2(v['AURC_error_%']),
            f"{v['AUROC']:.3f}", f"{v['wrong_caught_of_rejected_20%']} / {v['wrong_total']}"]


S['b5'] = f"""
<h1>10. The real question: is a legibility gate better than the OCR's own confidence?</h1>
<p>Every plate reader already gives a confidence (for the CRNN: the product of its per-character probabilities). If the
legibility classifier is useful as a gate, accepting the plates it calls "suitable" should give a higher plate exact than
accepting the plates the CRNN is most sure about. It does not.</p>
{fig('fig_gate.png', 475, 'Figure 7. A: plate exact of the accepted plates as the gate gets stricter (read right to left). Blue = CRNN confidence; orange = the legibility classifiers; aqua = confidence with the BRTA layout check; yellow = confidence + layout + both classifiers, weights fitted on the validation crops. B: AUROC, how well each score separates the CRNN\'s right readings from its wrong ones.')}
{table(['Gate score', '100 %', '90 %', '80 %', '70 %', '50 %', 'AURC', 'AUROC', 'Wrong caught*'], [
    grow('oracle', 'oracle (perfect gate)'),
    grow('ocr_confidence', 'CRNN confidence'),
    grow('conf + layout', 'confidence + BRTA layout check'),
    grow('learned (val): conf + layout', 'confidence + layout, val-fitted'),
    grow('learned (val): conf + layout + classifiers', 'confidence + layout + classifiers, val-fitted'),
    grow('cls_b4_ce', 'classifier 4-class, CE'),
    grow('cls_b4_ema', 'classifier 4-class, EMA'),
    grow('cls_qf_ema', 'Quality Filter, EMA'),
    grow('cls_qf_ce', 'Quality Filter, CE'),
    grow('cls_b4_ce x conf', 'classifier x confidence (kit)'),
    grow('random', 'random order'),
], [3.0, 0.65, 0.65, 0.7, 0.65, 0.65, 0.7, 0.7, 1.0], num=(1, 2, 3, 4, 5, 6, 7, 8))}
<p class="small">Columns 100-50 % = plate exact (%) of the accepted plates at that coverage. AURC = mean error (%) over all
coverages (lower is better). *Wrong caught = wrong readings among the 130 rejected plates at 80 % coverage / all wrong
readings (173). The CRNN's validation crops gave the weights of the fitted rows ({M('analysis.py')}, logistic regression).</p>
{table(['Comparison (paired bootstrap over the 651 plates, 2,000 resamples)', 'Δ plate exact at 80 %', 'Δ AURC'], [
    ['classifier 4-class CE minus confidence', ci(B['cls_b4_ce']['d80_ci95']), ci(B['cls_b4_ce']['dAURC_ci95'])],
    ['classifier 4-class EMA minus confidence', ci(B['cls_b4_ema']['d80_ci95']), ci(B['cls_b4_ema']['dAURC_ci95'])],
    ['Quality Filter EMA minus confidence', ci(B['cls_qf_ema']['d80_ci95']), ci(B['cls_qf_ema']['dAURC_ci95'])],
    ['Quality Filter CE minus confidence', ci(B['cls_qf_ce']['d80_ci95']), ci(B['cls_qf_ce']['dAURC_ci95'])],
    ['confidence + layout minus confidence', ci(B['conf + layout']['d80_ci95']), ci(B['conf + layout']['dAURC_ci95'])],
    ['+ classifiers minus confidence + layout (both val-fitted)', ci(B['classifiers_on_top_of_conf_layout']['d80_ci95']), ci(B['classifiers_on_top_of_conf_layout']['dAURC_ci95'])],
], [4.6, 1.7, 1.7])}
<p class="small">95 % intervals in percentage points. A positive Δ plate exact or a negative Δ AURC is better. An interval
that excludes 0 is a significant difference.</p>
<div class="box"><p><b>Reading the result.</b> Every legibility classifier is <b>significantly worse</b> than the CRNN's own
confidence (by 3.5-5.8 points at 80 % coverage). Multiplying the two (the kit's combination) is also worse than confidence
alone. The BRTA layout check is a free, significant improvement. Fitted properly on the validation crops, the classifiers
add a small but real improvement in AURC on top of confidence + layout ({ci(B['classifiers_on_top_of_conf_layout']['dAURC_ci95'])}
points) but nothing measurable at the 80 % operating point ({ci(B['classifiers_on_top_of_conf_layout']['d80_ci95'])}). The
classifier and the confidence see partly different plates: at 80 % coverage they reject {RJ['both']} of the same 130 plates;
{RJ['cls_only_wrong']} wrong readings are caught only by the classifier, {RJ['conf_only_wrong']} only by the confidence, and the
classifier throws away {RJ['cls_rejects_right']} correctly read plates (confidence: {RJ['conf_rejects_right']}).</p></div>
{fig('fig_hard.jpg', 470, 'Figure 8. Top: the CRNN\'s most confident wrong readings (confidence 1.00); the classifier calls two of them unsuitable. Bottom: right readings the classifier rejects most strongly (suitability ≤ 0.06): clear plates of rare districts and green CNG plates. The classifier judges "unusual", not "unreadable".')}
"""

S['b6'] = f"""
<h1>11. Our best reader: the Model Fusion study's fusion</h1>
<p>The CRNN (73.4 %) is not our best reader. The Model Fusion study's fallback fusion (char-YOLOv8n; the CRNN only when char-YOLO's
reading breaks the BRTA layout) reads <b>{f2(ST['fallback_exact_%'])} %</b> of the same 651 crops. With two readers we get a
new free signal: <b>do they agree?</b> They agree on {f1(ST['agree_share_%'])} % of the test plates, and then the reading is
right {f1(ST['exact_when_agree_%'])} % of the time (when they disagree: {f1(ST['exact_when_disagree_%'])} %).</p>
{fig('fig_strong.png', 470, "Figure 9. Gates for the Model Fusion study's fusion reader. Aqua = the free signals (char-YOLO and CRNN confidences, both layout checks, agreement), weights fitted on the 766 validation crops; yellow = the same plus both classifiers; orange = the classifier alone.")}
{table(['Gate for the fusion reader', '90 %', '80 %', '50 %', 'AURC', 'AUROC'], [
    [n, f2(ST[k]['90%']), f"<b>{f2(ST[k]['80%'])}</b>", f2(ST[k]['50%']), f2(ST[k]['AURC_error_%']), f"{ST[k]['AUROC']:.3f}"]
    for n, k in (('oracle', 'oracle'), ('free signals (val-fitted)', 'free signals (val-fitted)'),
                 ('free signals + classifiers (val-fitted)', 'free signals + classifiers (val-fitted)'),
                 ('char-YOLO confidence alone', 'char-YOLO confidence'), ('classifier 4-class CE', 'cls_b4_ce'),
                 ('Quality Filter EMA', 'cls_qf_ema'))], [3.4, 0.8, 0.8, 0.8, 0.8, 0.8], num=(1, 2, 3, 4, 5))}
<p>Classifier minus free signals at 80 %: {ci(ST['boot_cls_vs_free']['d80_ci95'])} points (significantly worse). Adding the
classifiers to the free signals: {ci(ST['boot_freecls_vs_free']['d80_ci95'])} points at 80 % (no difference), AURC
{ci(ST['boot_freecls_vs_free']['dAURC_ci95'])} (a tiny gain). For char-YOLO alone (86.79 %) its own confidence gives
{f2(ST['chars alone: own confidence']['80%'])} % at 80 % coverage, the classifier {f2(ST['chars alone: cls_b4_ce']['80%'])} %.</p>
<div class="good"><p><b>The practical rule for our pipeline.</b> Read with the Model Fusion study's fusion; <b>accept the reading when
char-YOLO and the CRNN agree</b> ({f1(ST['agree_share_%'])} % of plates, {f1(ST['exact_when_agree_%'])} % right); otherwise use the
fitted score or report "unreadable / needs review". It costs nothing extra: both readers already run in the fusion.</p></div>

<h1>12. Night, blur and clear plates</h1>
{fig('fig_conditions.png', 460, 'Figure 10. At 80 % coverage. A: share of each condition the gate rejects. B: plate exact of the plates kept (grey = no gate).')}
<p>The classifier rejects <b>{BC['night']['rejected_by_cls_b4_ce']} of the {BC['night']['n']} night plates</b>
({100 * BC['night']['rejected_by_cls_b4_ce'] / BC['night']['n']:.0f} %; confidence: {BC['night']['rejected_by_ocr_confidence']}), but the
night plates it keeps are not read better ({f1(BC['night']['accepted_exact_cls_b4_ce_%'])} % vs {f1(BC['night']['accepted_exact_ocr_confidence_%'])} %): it
also throws away night plates the CRNN reads right ({BC['night']['rejected_by_cls_b4_ce'] - BC['night']['wrong_caught_by_cls_b4_ce']}
vs {BC['night']['rejected_by_ocr_confidence'] - BC['night']['wrong_caught_by_ocr_confidence']}). On the 480 clear plates the confidence keeps a clearly better set
({f1(BC['clear']['accepted_exact_ocr_confidence_%'])} % vs {f1(BC['clear']['accepted_exact_cls_b4_ce_%'])} %). The classifier is a
"dark photo" detector more than a "wrong reading" detector. On blurry plates the two are equal.</p>
"""

def srow(k, name):
    v = SG[k]
    seeds = ', '.join(s.replace('kaggle', 'K').replace('local', 'L') for s in v['seeds'])
    pc = ' / '.join(f1(x) for x in v['per_class_mean'])
    return [name, seeds, f"{f1(v['F1_mean'])} ± {f1(v['F1_sd'])}" if v['F1_sd'] is not None else f1(v['F1_mean']), pc, f1(v['gate80_mean']), f"{v['AUROC_mean']:.3f}",
            ', '.join(str(b) for b in v['best_epochs'])]


def seed_summary():
    if DRAFT:
        return '(local runs not all done yet)'
    e4, c4, eq, cq = SG['kit_baseline4_ema'], SG['kit_baseline4_ce'], SG['kit_quality_ema'], SG['kit_quality_ce']
    f4 = e4['F1'] + c4['F1']
    gates = [x for k in SG for x in SG[k]['gate80']]
    best_auc = max(r['AUROC'] for r in A['local_runs'].values())
    qf_pairs = [round(a - b, 2) for a, b in zip(eq['F1'], cq['F1'])]
    return f"""(1) <b>Seeds matter more than the loss.</b> The same 4-class recipe scores {f1(min(f4))}-{f1(max(f4))} %
across seeds; EMA and CE have the same mean ({f1(e4['F1_mean'])} vs {f1(c4['F1_mean'])} %), so the Kaggle run's 0.6-point gap was
noise. On the Quality Filter EMA is ahead in all three seed pairs ({', '.join(f'{d:+.1f}' for d in qf_pairs)} points; mean
{f1(eq['F1_mean'])} vs {f1(cq['F1_mean'])} %), the opposite of the paper's Table V (96.7 vs 96.9); as a gate it stays far behind the
confidence ({f1(eq['gate80_mean'])} vs {f1(cq['gate80_mean'])} % at 80 %). (2) <b>The authors' real recipe</b> (EMA + balanced sampler) lifts the
weak "good" class (F1 {f1(BAL['per_class_mean'][1])} vs {f1(e4['per_class_mean'][1])} %) at the cost of "perfect" ({f1(BAL['per_class_mean'][0])} vs
{f1(e4['per_class_mean'][0])}), with no gain overall ({f1(BAL['F1_mean'])} %); it gives the best gate of all runs, still
{f2(g('ocr_confidence') - max(BAL['gate80']))} points below the confidence. (3) The EMA weights as the paper's <b>text</b> describes them
change nothing ({f1(TXT['F1_mean'])} %). (4) The <b>failure predictor</b>, trained on the exact gate question, is no better a gate
({f1(FAIL['gate80_mean'])} % at 80 % coverage, AUROC {FAIL['AUROC_mean']:.3f}); a likely reason is that its target is the half-data
CRNN's failure (that reader gets {f1(A['readers']['crnn|test|half0'])} % of the test plates right), not the deployment CRNN's
({f2(R['crnn_test']['rows_decoder']['plate_exact_%'])} %). (5) <b>Over all {len(gates)} classifier trainings the gate at 80 % coverage
stays between {f2(min(gates))} and {f2(max(gates))} %</b>, every one below the CRNN's own confidence ({f2(g('ocr_confidence'))} %); the best
local AUROC is {best_auc:.3f} vs {G['ocr_confidence']['AUROC']:.3f}."""


S['b7'] = f"""
<h1>13. Extra runs on our PC: seeds, the authors' real recipe, the text's loss, a failure predictor</h1>
<p>One Kaggle run per setting cannot separate a 1-point effect from noise, and the kit did not copy the authors' class-balanced
sampler. So {N_LOCAL} more ResNet-50 trainings ran on our RTX 5060 Ti with exactly the kit's recipe
({M('local_train.py')}), one at a time, about 9 minutes each:</p>
{ul([
    '<b>kit, new seeds</b>: the four Kaggle settings with seeds 1 and 2 (Kaggle\'s seed 42 is the third);',
    '<b>balanced</b>: EMA loss + the authors\' BalancedSampler (each epoch every class is cut to the size of the smallest), seeds 42 and 1;',
    '<b>text</b>: the EMA loss as the paper\'s sentence describes it (more errors -> more weight, Figure 1), seeds 42 and 1;',
    '<b>failure</b>: the same ResNet-50 trained directly on the question the gate asks, "did the cross-fitted CRNN read this crop wrong?", seeds 42, 1 and 2.',
])}
{table(['Setting', 'Seeds', 'Test F1 mean ± SD', 'Per class (mean)', 'Gate 80 %', 'AUROC', 'Best epochs'], [
    srow(k, n) for k, n in (('kit_baseline4_ema', '4-class, EMA'), ('kit_baseline4_ce', '4-class, CE'),
                            ('balanced_baseline4_ema', '4-class, EMA + balanced sampler'),
                            ('text_baseline4_ema', '4-class, EMA as the text says'), ('kit_quality_ema', 'Quality Filter, EMA'),
                            ('kit_quality_ce', 'Quality Filter, CE'), ('failure_quality_ce', 'failure predictor (CRNN wrong?)'))
    if k in SG], [2.4, 1.0, 1.2, 1.8, 0.8, 0.7, 0.8], num=(2, 4, 5))}
<p class="small">K = Kaggle T4 run, L = our PC. Gate 80 % = plate exact of the deployment CRNN on the 80 % of test plates each
classifier rates most suitable (CRNN confidence: {f2(g('ocr_confidence'))} %). Per class = perfect / good / poor / illegible F1 (4-class
rows) or suitable / unsuitable (2-class rows).</p>
{fig('fig_seeds.png', 470, 'Figure 11. Every classifier run. A: its own test score. B: as a gate for the CRNN; the dashed line is the CRNN confidence. No run reaches it.')}
<div class="box"><p><b>What the extra runs say.</b> {seed_summary()}</p></div>

<h1>14. Speed</h1>
{table(['Model', 'Parameters', 'GPU, batch 1', 'CPU (4 threads), batch 1', 'T4 (Kaggle log)'], [
    ['ResNet-50 legibility classifier (96 x 192)', f"{LC['params_M']['classifier']} M", f"{SPD['classifier_gpu_fp32']} ms (FP32), {SPD['classifier_gpu_fp16']} ms (FP16 autocast)", f"{SPD['classifier_cpu4']} ms", f"{min(C[n]['ms_per_crop'] for n in C)}-{max(C[n]['ms_per_crop'] for n in C)} ms*"],
    ['CRNN reader (48 x 256)', f"{LC['params_M']['crnn']} M", f"{SPD['crnn_gpu_fp32']} ms (FP32), {SPD['crnn_gpu_fp16']} ms", f"{SPD['crnn_cpu4']} ms", f"{R['crnn_test']['ms_per_plate']} ms"],
    ['CRNN confidence, layout check, reader agreement', '0', 'free (already computed)', 'free', '-'],
], [2.6, 0.9, 2.0, 1.5, 1.3])}
<p class="small">GPU = RTX 5060 Ti, CPU = Intel Core Ultra 7 265K. *The four T4 numbers are the same network; the spread
comes from the other job sharing the machine. At batch 1 the FP16 autocast is slower than FP32 on our GPU (launch
overhead). On a CPU-only laptop, the classifier would cost more time than the reader it is supposed to protect.</p>
"""

S['b8'] = f"""
<h1>15. Findings, and the numbers to quote</h1>
{table(['#', 'Finding', 'Number to quote'], [
    ['1', 'The v1.1 run is healthy and reproducible', f"8 / 8 jobs, {LOGT['total_s'] / 60:.0f} min; same F1 and 100 % same classes on our PC; CRNN = the Model Fusion study's file"],
    ['2', 'Our OCR-difficulty labels are meaningful', f"CRNN plate exact per class {f1(DL['perfect']['exact_%'])} / {f1(DL['good']['exact_%'])} / {f1(DL['poor']['exact_%'])} / {f1(DL['illegible']['exact_%'])} %"],
    ['3', '…but noisy and partly about rare layouts', f"same plate, same 4-class label: {f1(SP['label_agree_%'])} %; best epoch 1-4"],
    ['4', 'Classifier scores far below the paper (different labels)', f"4 classes {f1(C['cls_b4_ce']['test']['micro_F1_%'])} % (paper 89.8), QF {f1(C['cls_qf_ema']['test']['micro_F1_%'])} % (paper 96.7)"],
    ['5', 'The EMA loss makes no real difference', f"4 classes {pv(A['ema_vs_ce']['cls_b4_ema']['p'])}, 3-seed means {f1(SG['kit_baseline4_ema']['F1_mean'])} vs {f1(SG['kit_baseline4_ce']['F1_mean'])} %; QF about +1 point (3 of 3 seeds), no gate gain"],
    ['6', '<b>OCR confidence beats every legibility gate</b>', f"80 % coverage: {f2(g('ocr_confidence'))} vs {f2(g('cls_b4_ce'))} %, Δ {ci(B['cls_b4_ce']['d80_ci95'])}; AUROC {G['ocr_confidence']['AUROC']:.3f} vs {G['cls_b4_ce']['AUROC']:.3f}"],
    ['7', 'The BRTA layout check is a free gain', f"{f2(g('conf + layout'))} % at 80 %, AURC {f2(G['conf + layout']['AURC_error_%'])} vs {f2(G['ocr_confidence']['AURC_error_%'])}"],
    ['8', 'Reader agreement is the best free gate for our best reader', f"fusion {f2(ST['fallback_exact_%'])} % -> {f2(ST['free signals (val-fitted)']['80%'])} % at 80 % coverage; agree -> {f1(ST['exact_when_agree_%'])} % right"],
    ['9', 'Classifiers add nothing measurable on top', f"Δ at 80 %: {ci(B['classifiers_on_top_of_conf_layout']['d80_ci95'])} (CRNN), {ci(ST['boot_freecls_vs_free']['d80_ci95'])} (fusion)"],
    ['10', 'The classifier over-rejects night plates', f"rejects {BC['night']['rejected_by_cls_b4_ce']} / {BC['night']['n']} night plates (confidence {BC['night']['rejected_by_ocr_confidence']}) with no gain in the kept accuracy"],
    ['11', 'The paper\'s EMA text contradicts its Eq. 1 and code; the code\'s EMA run also changes sampler and learning rate', 'p.5; ema_br.json, train.json'],
], [0.3, 3.8, 3.9])}
"""

S['c1'] = f"""
<p class="part">PART C · USE IT</p>
<h1>16. How to use this paper in our research</h1>
{table(['Proposal item', 'How LPLCv2 and this run feed it'], [
    ['<b>RQ2 / O3</b>: character-level reader with BRTA format correction', f'The layout check is also a <b>gate</b>: {LY["test"]["invalid"]} CRNN test readings break the BRTA layout and {LY["test"]["invalid_right"]} of them are right (val: {LY["val"]["invalid_right"]} of {LY["val"]["invalid"]}); adding the check to the confidence is a significant gain. Report "format-valid and confident" as the acceptance rule.'],
    ['<b>RQ3 / O4</b>: multi-frame voting in video', 'Use the same free score (confidence + layout + reader agreement) to <b>weight or select frames</b> before the vote: a vote over accepted frames only. A legibility classifier per frame would cost ~20 ms on a CPU; the free score costs nothing.'],
    ['<b>RQ4 / O5</b>: robustness per condition', 'Report <b>selective accuracy at 80 % coverage</b> and the rejected share per condition (night / blurry / clear), next to plain accuracy. It shows where the system knows it is unsure.'],
    ['Pipeline design', 'Final output per plate: the text, OR "unreadable" when the free score is low. Accepting only when char-YOLO and CRNN agree keeps 72.8 % of plates at 97.3 % accuracy (section 11).'],
    ['Literature review', 'Cite LPLCv2 as the legibility benchmark (it is already reference [wojcik2026] in our proposal) and our result that, for Bangla plates, reader confidence is the better gate. Our runs also fill the paper\'s own gap ("integrate legibility classification into a complete ALPR pipeline", p.7).'],
    ['Data contribution (optional)', 'Hand-label the legibility of our 651 OCR test crops with the paper\'s 4 definitions (about 1 hour for one person). The kit already scores against it: put the CSV at ' + M('ocr/legibility_human.csv') + ' and re-run the gate job. That gives the first Bangla legibility test set.'],
], [2.2, 5.8])}

<h1>17. What benefit do you get from it?</h1>
{ul([
    '<b>A clear, defensible negative result</b> with confidence intervals: "a 23.5 M-parameter legibility classifier is worse than the free OCR confidence as a reject gate on Bangla plates". Faculty value a well-tested "no" because it saves the team months.',
    '<b>A better gate for free</b>: confidence + BRTA layout check + reader agreement. It turns the 88.6 % fusion reader into a 96.7 %-precise reader on 80 % of plates, the honest way to report a deployable system.',
    '<b>A new evaluation for the whole project</b>: risk-coverage curves and selective accuracy. None of our other paper runs report how well the system knows when it is wrong; this one gives the code (' + M('analysis.py') + ': summary, auroc, boot).',
    '<b>Paper-reading skill</b>: you found that the paper\'s explanation of its own loss contradicts its equation, that the EMA ablation is confounded in the public code, and that the paper never tests its stated purpose. Good material for the literature-review chapter.',
    '<b>Evidence for label noise</b> in the readers\' view of our data (same plate, different labels; rare districts and layouts fail), pointing to the real fix: more data for rare districts and one-line / CNG layouts (the Model Fusion study\'s error analysis says the same).',
])}
"""

S['c2'] = f"""
<h1>18. What to show the faculty (7 slides)</h1>
{table(['#', 'Slide', 'Content'], [
    ['1', 'The question', 'LPLCv2 (IJCNN 2026): classify plate legibility before OCR. Its claim: 89.8 % F1, 96.7 % Quality Filter. Its gap: no test of whether it helps an OCR. Our question: does it beat the OCR\'s own confidence as a reject gate on Bangla plates?'],
    ['2', 'How we built labels', 'Figure 2 top (cross-fitted readers -> 4 proxy classes) and Figure 4 B (CRNN accuracy 99 / 88 / 61 / 17 % per class).'],
    ['3', 'Run health', '8 / 8 jobs, 78 min, v1.1; same numbers on our PC; CRNN identical to the Model Fusion study\'s. Best epoch 1-4 (noisy labels).'],
    ['4', 'Classifier vs paper', 'Section 9 table: 51.9 % / 71.7 % vs 89.8 / 96.7 (different labels, direction only); EMA = no effect on 4 classes (p = 0.42, same 3-seed mean), about +1 on QF.'],
    ['5', '<b>The main result</b>', 'Figure 7: at 80 % coverage confidence 84.5 % vs classifier 81.0 % (CI -5.6 to -1.3); AUROC 0.84 vs 0.77.'],
    ['6', 'What works instead', 'Figure 9: Model Fusion reader + free signals = 96.7 % at 80 % coverage; agree -> 97.3 % right.'],
    ['7', 'Decision and next steps', 'Use confidence + layout + agreement as the gate; no ResNet-50 gate. Next: video frame selection (RQ3), 651 human legibility labels.'],
], [0.3, 1.6, 6.1])}

<h1>19. What to explain: a 3-minute script</h1>
<div class="box"><p><i>"LPLCv2 is a paper from the Laroca and Menotti group, IJCNN 2026. They label 41 thousand Brazilian plates as perfect, good, poor or illegible and train a ResNet-50 to predict that, with a new EMA loss: 89.8 percent on four classes, 96.7 on suitable-versus-unsuitable. The idea is to read good plates, enhance poor ones and drop illegible ones. But the paper never checks whether this actually helps an OCR.</i></p><p><i>Their data is only on request and there are no legibility labels for Bangla plates, so we made our own: two reader types, trained on one half of our data, read the other half. Both right is 'perfect', one right is 'good', and so on. These labels make sense: our final CRNN reads 99 percent of 'perfect' plates and 17 percent of 'illegible' ones. We trained the paper's ResNet-50 with their loss, on dataset v1.1.</i></p><p><i>The run is healthy and reproduces on our PC. The classifier reaches 52 percent on four classes and 72 on the binary filter, far below the paper, because our labels are about reader difficulty and are noisy: two photos of the same plate get the same label only about half the time. The EMA loss makes no real difference, also over three seeds.</i></p><p><i>Then the real test. If we keep the 80 percent of plates the CRNN is most sure about, 84.5 percent are read right. If we keep the 80 percent the legibility classifier likes best, only 81. The confidence wins, and the difference is significant. Adding our BRTA layout check makes the free gate even better, and with the two readers of our Model Fusion study, simply asking 'do they agree?' gives 96.7 percent accuracy on 80 percent of the plates. Our extra runs, including the authors' real training recipe and a classifier trained directly to predict reader failure, never beat the confidence.</i></p><p><i>So our decision: no separate legibility network. The pipeline outputs a plate only when the reader is confident, the format is valid and the readers agree; otherwise it says 'unreadable'. We will use the same score to pick frames for video voting."</i></p></div>
"""

S['c3'] = f"""
<h1>20. Questions the faculty may ask</h1>
<p class="q">Q1. Your F1 is 52 % and theirs 90 %. Did you implement it wrong?</p>
<p>No. The model, loss and settings follow the paper and the authors' code, and every Kaggle number reproduces on our PC.
The labels are different: theirs are human judgements of the image (with corrections), ours say whether two readers
succeeded. Our labels are noisier (same plate, same label only {f1(SP['label_agree_%'])} % of the time) and partly measure reader
blind spots (rare districts, one-line plates), which no image classifier can see. The paper's own LPLCv1 numbers show the
same effect: noisy labels gave 74.5 %, corrected labels 84.4 % with the same network (p.5).</p>
<p class="q">Q2. Then is the gate comparison fair to the paper?</p>
<p>It tests the paper's method on the labels we can make. Two extra checks make the result robust: (1) a classifier
trained directly on "will the reader fail?" (the exact gate target) also loses to the confidence (section 13); (2) for
the Model Fusion study's char-YOLO the classifier is also far behind its own confidence. With human legibility labels a classifier
could be better at spotting truly illegible images, but it would still have to beat a confidence that already reaches
AUROC {G['ocr_confidence']['AUROC']:.2f}. That is why we propose labelling the 651 test crops by hand.</p>
<p class="q">Q3. Why do you trust the confidence intervals?</p>
<p>Paired bootstrap over the 651 test plates (2,000 resamples, both gates scored on the same resample), as in our
Model Fusion and HVI-CIDNet studies. The classifier's interval at 80 % coverage is {ci(B['cls_b4_ce']['d80_ci95'])} points: it never reaches 0.</p>
<p class="q">Q4. Isn't fitting the combination on validation data cheating?</p>
<p>No: the weights are fitted on the 766 validation crops (CRNN readings from the same model) and only applied to the
test crops. That is the standard way to choose thresholds without touching the test set.</p>
<p class="q">Q5. Does the EMA loss help at all?</p>
<p>Not on 4 classes: same-crop McNemar {pv(A['ema_vs_ce']['cls_b4_ema']['p'])}, and over three seeds EMA and CE have the same mean
({f1(SG['kit_baseline4_ema']['F1_mean'])} vs {f1(SG['kit_baseline4_ce']['F1_mean'])} %, seed spread about ±2 points). On the Quality Filter one run says nothing
({pv(A['ema_vs_ce']['cls_qf_ema']['p'])}), but EMA is ahead in all three seed pairs (mean {f1(SG['kit_quality_ema']['F1_mean'])} vs
{f1(SG['kit_quality_ce']['F1_mean'])} %): a small edge that does not make it a useful gate (section 13). The paper itself reports +1.1 on
4 classes and no gain on the Quality Filter; in the authors' code the EMA run also changes the sampler and the learning rate.</p>
<p class="q">Q6. The paper says the EMA loss gives more weight to classes with more errors. Is that true?</p>
<p>Not as written: Eq. 1 is max(errors) / errors, capped at 1.2, so the class with the MOST errors gets the LOWEST weight
(1.0); the code does the same (Figure 1). Our "text" runs implement the sentence instead (section 13).</p>
<p class="q">Q7. Why not use the classifier to send poor plates to super-resolution or enhancement?</p>
<p>Our HVI-CIDNet study found that enhancement does not help our night plates, and here the classifier's "poor" flag is mostly
"dark or unusual photo", not "readable after enhancement". Routing would need an enhancer that helps first.</p>
<p class="q">Q8. Is the classifier useless then?</p>
<p>As a gate in front of our readers, yes. It may still be useful (a) to describe a dataset's difficulty, (b) for a camera
health monitor (many "illegible" crops from one camera = dirty lens or bad focus), which does not need per-plate accuracy.</p>
<p class="q">Q9. What would change your conclusion?</p>
<p>Human legibility labels that a classifier learns well (F1 above ~85 %) AND that add to the free score on a held-out
set. The kit is ready for that test ({M('ocr/legibility_human.csv')}).</p>

<h1>21. Next steps (in order)</h1>
{ul([
    '<b>Adopt the free gate</b> in the pipeline code: accept = char-YOLO and CRNN agree, or the val-fitted score above a threshold chosen on validation for the target accuracy; otherwise "unreadable". Report selective accuracy at 80 / 90 % coverage in every end-to-end table.',
    '<b>Video (RQ3):</b> use the same score to choose frames for the multi-frame vote; compare voting over all frames vs accepted frames only.',
    '<b>Optional, one person-hour:</b> label the 651 test crops with the paper\'s 4 definitions and re-run the gate job against them.',
    '<b>Data:</b> the "illegible" proxy class is full of rare districts and layouts (one-line, CNG green); add such plates to training (the Model Fusion study reached the same conclusion from its errors).',
    '<b>Do not</b> spend Kaggle hours on a bigger legibility network or more EMA tuning for this project.',
], 'ol')}
"""

S['c4'] = f"""
<h1>22. GitHub and Drive</h1>
<p>The upload folder {M('After train/github_upload/')} mirrors the repository (steps in its {M('UPLOAD_STEPS.md')}). Upload it after the combined
upload of the YOLO26, HVI-CIDNet, SVTRv2 and SAHI studies: this folder's README and results build on that one.</p>
{table(['Repository path', 'Content'], [
    [M('weights/paper19/MD.Rokib.Hasan.Oli/'), 'README (what, results, quick start tested on our PC, SHA-256), the 4-class CE classifier in FP16 (45 MiB, the best classifier gate), ' + M('paper19_results.zip') + ' (no weights), ' + M('results_vs_paper.md') + ', this PDF. The CRNN is not uploaded again: it is byte-identical to ' + M('weights/paper24/…/best_crnn_fp16.pt') + '.'],
    [M('notebooks/MD.Rokib.Hasan.Oli/'), M('paper19_run.ipynb') + ' + ' + M('.log') + ' (the Kaggle run)'],
    [M('src/MD.Rokib.Hasan.Oli/paper19/'), 'START_HERE, the Kaggle guide, ' + M('kaggle/') + ' (the kit exactly as run)'],
    [M('report/MD.Rokib.Hasan.Oli/paper19/'), 'the notes and ' + M('source/') + ' (local check, extra runs, analysis, charts, report builder, their JSON / CSV / figures)'],
    [M('results/'), 'team table with the LPLCv2 row and the ours-vs-paper block'],
], [2.6, 5.4])}
<p><b>Drive</b> (by hand): {M('paper19_outputs.zip')} (380 MB: all four classifiers in FP32, the CRNN, readings and labels) to
{M('498R / weights / paper19 /')}. Do not upload {M('After train/results.zip')} (834 MB, the whole Kaggle working folder).</p>

<h1>Appendix: where every number comes from</h1>
{table(['Number', 'Source'], [
    ['Job times, label counts, Kaggle F1, gate rows', M('paper-19.log') + ', ' + M('runs/*/results.json') + ' in results.zip'],
    ['Classifier probabilities, CRNN readings (test)', M('runs/cls_*/test_scores.csv') + ', ' + M('runs/legibility_labels/deploy_test.csv')],
    ['Local re-check, val readings, timing', M('report_src/local_check.py') + ' -> ' + M('local_check.json') + ', ' + M('local_preds.csv')],
    ['Extra runs (section 13)', M('report_src/local_train.py') + ', ' + M('run_local_train.sh') + ' -> ' + M('_local_runs/*/results.json')],
    ['Gate tables, bootstrap, AUROC, conditions, same-plate pairs, fusion reader', M('report_src/analysis.py') + ' -> ' + M('analysis.json')],
    ['Fusion reader readings', 'Model Fusion study, v1.1 run, ' + M('paper24_results.zip') + ' ' + M('runs/{chars_v8n,crnn}/ocr_preds_{val,test}.csv')],
    ['Paper numbers', 'arXiv 2604.08741 v1, Tables I-VII, Fig. 4 (page given in each table)'],
    ['Code facts', 'UFPR-IPASP-PR/LPLCv2-Dataset @ 7a1d3f6: models/loss.py, dataset/dataset_utils.py, configs/train/{ema_br,train}.json, scripts/run_experiments.sh'],
], [3.0, 5.0])}
{table(['Paper named in this report', 'Full title (team folder)'], [
    ['Model Fusion study', 'Laroca et al., Leveraging Model Fusion for Improved License Plate Recognition, CIARP 2023'],
    ['HVI-CIDNet study', 'Yan et al., HVI: A New Color Space for Low-light Image Enhancement, CVPR 2025'],
    ['YOLO26 study', 'Sapkota et al., YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection, arXiv 2509.25164'],
    ['SVTRv2 study', 'Du et al., SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition, ICCV 2025'],
    ['SAHI study', 'Akyon et al., Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection, ICIP 2022'],
    ['Near-Duplicates', 'Laroca et al., Do We Train on Test Data? The Impact of Near-Duplicates on License Plate Recognition, IJCNN 2023'],
    ['Plate Super-Resolution', 'Nascimento et al., Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark, JBCS 2025'],
    ['Multinational LPR', 'Laroca et al., Advancing Multinational License Plate Recognition Through Synthetic and Real Data Fusion, IET ITS 2025'],
], [2.0, 6.0])}
"""

FLOWS = [['cover'], ['a1', 'a3', 'a4'], ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'], ['c1', 'c2', 'c3', 'c4']]
n = write_pdf(S, FLOWS, OUT, 'LPLCv2 study report · legibility classification · dataset v1.1 run · 498R BanglaLPR',
              {'title': 'LPLCv2 study report: legibility classification (v1.1 run)', 'author': 'MD. Rokib Hasan Oli (498R)',
               'subject': 'CSE 498R paper study', 'keywords': 'license plate, legibility, selective prediction, Bangla'})
print('wrote', OUT, n, 'pages')
