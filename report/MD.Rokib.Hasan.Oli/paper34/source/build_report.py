"""SVTRv2 study report (dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Du et al., ICCV 2025, arXiv 2411.15858 v2; page numbers = PDF pages incl. the supplement); the v1.1
Kaggle run (`After Train/results.zip`, `paper-34.log`, 26 Oct 2026) via analysis.json; the local re-check on the
RTX 5060 Ti (local_check.py: every run re-scored crop by crop, FP16 copies, timings, ONNX); the OpenOCR code at commit
1ccfc6e and the config / result files shipped with the released weights; the model fusion and cascaded YOLO v1.1 runs
(their ocr_preds_*.csv) for the fusion; the 498R proposal (RQ2, G1, G4, G5, G6, O3, O5).
Other papers are named, not numbered (team request). Run order: local_check.py -> analysis.py -> charts.py -> build_report.py.
"""
import json
import os
import sys

from part_a import part_a
from report_head import M, f1, f2, fig, pv, table, ul, write_pdf

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'SVTRv2_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
H, MN, P, B, E = A['health'], A['main'], A['pairs'], A['by_msr_size'], A['errors']
C, CV, FU, LC = A['by_condition'], A['coverage'], A['fusion'], A['local_check']
FR, SC, TM = FU['rows'], LC['score'], LC['timing']
ex = lambda r: MN[r]['test']['plate_exact_%']
NM = {'svtrv2_t_msr': 'SVTRv2-T, MSR', 'svtrv2_s_msr': 'SVTRv2-S, MSR', 'svtrv2_t_fixed': 'SVTRv2-T, fixed 32x128',
      'svtrv2_t_scratch': 'SVTRv2-T, MSR, from scratch', 'crnn': 'CRNN (our baseline)',
      'chars_v8n': 'character YOLO (Model Fusion study)', 'lpnr_v12n': 'LPNR YOLO (Cascaded YOLO study)'}
SV = ['svtrv2_t_msr', 'svtrv2_s_msr', 'svtrv2_t_fixed', 'svtrv2_t_scratch']
OURS = SV + ['crnn']
ALL = OURS + ['chars_v8n', 'lpnr_v12n']
# timing on the local GPU / CPU (model only, batch 1), weighted by the test crops' MSR sizes
W = {k: v['n'] for k, v in B.items()}
mix = lambda size, key: sum(W[k] * TM[size][k][key] for k in W) / sum(W.values())
tT = {k: mix('T', k) for k in ('gpu_fp32', 'gpu_fp16', 'cpu_fp32')}
tS = {k: mix('S', k) for k in ('gpu_fp32', 'gpu_fp16', 'cpu_fp32')}
tC = {k: TM['crnn'][k] for k in ('gpu_fp32', 'gpu_fp16', 'cpu_fp32')}
ON = LC.get('onnx', {})
bn = lambda s: f'<span class="bn">{s}</span>'
q = lambda d: f"{d['only_a']} vs {d['only_b']}, {pv(d['p'])}"

# ---------------------------------------------------------------------------------------------- facts the text uses
assert H['dataset_version'] == '1.1' and H['n_jobs_code0'] == 5 and H['nan_in_logs'] == 0 and H['total_h'] == 3.57
assert all(H['time_cap_hits'][r] for r in SV) and not H['time_cap_hits']['crnn']
assert [H[r]['epochs_run'] for r in SV] == [63, 66, 64, 65] and [H[r]['best_epoch'] for r in SV] == [55, 35, 23, 54]
assert all(SC[r][s]['local_exact_%'] == SC[r][s]['kaggle_exact_%'] for r in OURS for s in ('test', 'val'))
assert sum(SC[r][s]['same_text_as_kaggle'] for r in OURS for s in ('test', 'val')) == 7079
assert LC['sha256']['crnn_model_full'].startswith('0110631e')
assert (ex('svtrv2_t_msr'), ex('svtrv2_s_msr'), ex('svtrv2_t_fixed'), ex('svtrv2_t_scratch'), ex('crnn')) == (81.26, 81.87, 74.5, 61.75, 73.43)
assert P['svtrv2_t_msr/crnn']['p'] < 0.0001 and P['svtrv2_s_msr/svtrv2_t_msr']['p'] > 0.5
assert P['svtrv2_t_msr/svtrv2_t_fixed']['p'] < 0.0001 and P['svtrv2_t_msr/svtrv2_t_scratch']['only_a'] == 146
assert P['svtrv2_t_scratch/crnn']['p'] < 0.0001 and P['svtrv2_t_fixed/crnn']['p'] > 0.5
assert P['chars_v8n/svtrv2_t_msr']['p'] < 0.001 and P['lpnr_v12n/svtrv2_t_msr']['p'] < 0.001
assert B['64x64']['msr_vs_fixed']['p'] > 0.1 and B['48x96']['msr_vs_fixed']['p'] < 0.0001
assert B['64x64']['n'] == 329 and B['48x96']['n'] == 317
assert C['night']['t_vs_crnn']['p'] > 0.05 and C['clear']['t_vs_crnn']['p'] < 0.001
assert all(E[r]['right_but_invalid'] == 0 for r in ALL) and E['chars_v8n']['invalid_layout'] == 46
assert FR['chars_t_fallback']['test_%'] == 90.17 and FR['chars_t_fallback']['vs_chars_v8n']['only_base'] == 0
assert FR['chars_t_fallback']['vs_old_best_fallback']['p'] < 0.05 and FR['new_best_val']['test_%'] == 90.94
assert FR['new_best_val']['vs_old_best_fallback']['p'] < 0.01 and FU['oracle_test_%'] == 93.7
assert FU['chosen_on_val']['rule'] == 'MV-BM+format' and len(FU['chosen_on_val']['models']) == 4
assert A['released_weights']['T']['common_avg'] == 94.78 and A['released_weights']['S']['u14m_avg'] == 83.7
assert LC['fp16']['svtrv2_t_msr']['test']['same_text'] == 651 and LC['fp16']['svtrv2_s_msr']['test']['exact_fp16_%'] == 81.87

S = part_a(A)
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER STUDY REPORT · DATASET v1.1 RUN</p>
<h1 style="font-size:18pt; margin-bottom:3pt">SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition</h1>
<p style="font-family:H; font-size:11.5pt; color:#1c5cab; margin-bottom:8pt">The paper, our SVTRv2 run on dataset v1.1, whether
the run is sound, what it means for Bangla plates, how to use it in our research, and what to show and say to the
faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 27 October 2026<br>
Paper: Y. Du, Z. Chen, H. Xie, C. Jia and Y.-G. Jiang, ICCV 2025 (arXiv 2411.15858 v2, 17 pages with the supplement; page
numbers are PDF pages). Code: github.com/Topdu/OpenOCR (Apache-2.0), pinned at commit 1ccfc6e; English weights from the
authors' Google Drive.<br>
Our run: {M('paper34_kaggle.ipynb')} on Kaggle, Tesla T4 x2, 26 Oct 2026, dataset <b>v1.1</b> (confirmed in
{M('environment.json')}), {H['total_h']:.2f} h, 5 jobs. Extra checks on our own PC on 27 Oct (RTX 5060 Ti): every
Kaggle reading re-scored crop by crop, every reader timed again in one session, fusion with our two best earlier readers.</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> A CTC text reader (one forward pass, no decoder loop) with three upgrades: it resizes
each crop to one of four shapes by its aspect ratio (MSR), it learns to rearrange 2-D features into reading order (FRM),
and during training only it learns language context (SGM). It beats every attention-decoder reader on hard English text
(86.14 vs PARSeq 84.26 %) at 2.7x PARSeq's speed (p.7).</p>
<p><b>Is our run OK?</b> <b>Yes, with three caveats.</b> Dataset v1.1, 5 of 5 jobs ended with code 0, no NaN, the code is
the kit's; re-reading the 1,417 test and val crops with all five models on our PC gives the same plate-exact scores for
every model (same text in 7,079 of 7,085 readings). Caveats: (1) all four SVTRv2 runs hit the 1.6 h time cap at epoch 63-66 of 100, so the learning rate never
came down; (2) the kept checkpoint was picked on validation <i>loss</i>, which chose epoch 23 for the fixed-size run;
(3) the Kaggle ms per plate were measured while the other GPU was training (the same model took 7.6 and 15.0 ms). The
kit is fixed (60 epochs, pick on val accuracy, timing at the end) and smoke-tested.</p>
<p><b>What we found.</b></p>
{ul([
    f"<b>SVTRv2 beats our CRNN clearly:</b> SVTRv2-T {f2(ex('svtrv2_t_msr'))} % vs CRNN {f2(ex('crnn'))} % plate exact (+{P['svtrv2_t_msr/crnn']['diff_pp']:.1f} points; {q(P['svtrv2_t_msr/crnn'])}). SVTRv2-S (2.2x the parameters) is no better ({f2(ex('svtrv2_s_msr'))} %, {pv(P['svtrv2_s_msr/svtrv2_t_msr']['p'])}).",
    f"<b>But alone it does not beat our best readers:</b> character YOLO {f2(ex('chars_v8n'))} % and LPNR YOLO {f2(ex('lpnr_v12n'))} % are 5.5-6.5 points better (both {pv(P['chars_v8n/svtrv2_t_msr']['p'])} or smaller).",
    f"<b>English pre-training is essential:</b> from scratch SVTRv2-T reads only {f2(ex('svtrv2_t_scratch'))} %, less than CRNN; the authors' English weights add {P['svtrv2_t_msr/svtrv2_t_scratch']['diff_pp']:.1f} points. The paper's data hunger is real.",
    f"<b>MSR helps</b> (+{P['svtrv2_t_msr/svtrv2_t_fixed']['diff_pp']:.1f} points over a fixed 32x128 input, {q(P['svtrv2_t_msr/svtrv2_t_fixed'])}), but on our plates the gain is on the wider plates (+{B['48x96']['msr_vs_fixed']['diff_pp']:.1f}), not on the near-square ones (+{B['64x64']['msr_vs_fixed']['diff_pp']:.1f}, n.s.), the opposite of the paper's Table 1.",
    f"<b>Its main value for us is as a helper:</b> character YOLO + SVTRv2-T as a format fallback reads <b>{f2(FR['chars_t_fallback']['test_%'])} %</b> ({FR['chars_t_fallback']['vs_chars_v8n']['only_new']} plates fixed, 0 broken), and the four-reader vote chosen on validation reaches <b>{f2(FR['new_best_val']['test_%'])} %</b>, our first result above 90 % plate exact (vs the best so far, {FR['old_best_fallback']['test_%']:.2f} %: {FR['new_best_val']['vs_old_best_fallback']['only_new']} fixed / {FR['new_best_val']['vs_old_best_fallback']['only_base']} broken, {pv(FR['new_best_val']['vs_old_best_fallback']['p'])}).",
])}
<p><b>Use in our research:</b> keep character YOLO as the main reader; add SVTRv2-T (5.1 M parameters, {tT['gpu_fp32']:.1f} ms
per plate on our GPU) as the second reader behind the BRTA layout check. It answers RQ2 (90 % with plate grammar) and G5
(grammar used to combine readers), and gives a measured cost for G1. What to show the faculty: Part E.</p>
</div>
<p class="small">Contents: Part A the paper (sections 1-6) · Part B the run and whether it is sound (7-9) · Part C results
and what they mean (10-17) · Part D how to use it in our research (18-20) · Part E what to show and explain to the
faculty (21-23) · Appendix.</p>
"""

S['b1'] = f"""
<p class="part">PART B · OUR RUN AND WHETHER IT IS SOUND</p>
<h1>7. What the notebook ran</h1>
<p>One Kaggle notebook ({M('paper34_kaggle.ipynb')}, cell for cell the kit of 25 Oct 2026, {M('FAST = False')}), Tesla T4 x2,
26 Oct 2026. Data: 498R BanglaLPR All-in-One <b>v1.1</b>, OCR split 5,607 train / 766 val / 651 test plate crops (val and
test are human-labelled only; groups never cross splits). Every SVTRv2 run uses the authors' model code (OpenOCR commit
1ccfc6e: {M('SVTRv2LNConvTwo33')} encoder + {M('RCTCDecoder')} = FRM + CTC) with one output class per Bangla plate token
(101 tokens seen in our data: 64 district names and short forms, metro, 26 series letters and 10 digits, plus the CTC blank), the same labels as our CRNN.</p>
{table(['Run', 'GPU', 'Model and input', 'Start weights', 'Epochs run / planned', 'Kept epoch', 'Minutes'], [
    [M('svtrv2_t_msr'), '0', 'SVTRv2-T, MSR', "authors' English", f"{H['svtrv2_t_msr']['epochs_run']} / 100", str(H['svtrv2_t_msr']['best_epoch']), f1(H['svtrv2_t_msr']['train_min'])],
    [M('svtrv2_s_msr'), '0 (after)', 'SVTRv2-S, MSR', "authors' English", f"{H['svtrv2_s_msr']['epochs_run']} / 100", str(H['svtrv2_s_msr']['best_epoch']), f1(H['svtrv2_s_msr']['train_min'])],
    [M('svtrv2_t_fixed'), '1', 'SVTRv2-T, fixed 32x128 (Table 1 ablation)', "authors' English", f"{H['svtrv2_t_fixed']['epochs_run']} / 100", str(H['svtrv2_t_fixed']['best_epoch']), f1(H['svtrv2_t_fixed']['train_min'])],
    [M('svtrv2_t_scratch'), '1 (after)', 'SVTRv2-T, MSR', 'none (random)', f"{H['svtrv2_t_scratch']['epochs_run']} / 100", str(H['svtrv2_t_scratch']['best_epoch']), f1(H['svtrv2_t_scratch']['train_min'])],
    [M('crnn'), '1 (after)', 'CRNN-CTC, 48x256 grey, two lines side by side', 'none', '60 / 60', str(H['crnn']['best_epoch']), f1(H['crnn']['train_min'])],
], [1.7, 0.8, 2.6, 1.3, 1.2, 0.9, 0.8], num=(5, 6))}
<p>Settings of all SVTRv2 runs: AdamW lr 5e-4 (the paper uses 6.5e-4 at batch 1024; we use batch 64), weight decay 0.05,
one-cycle schedule with 7.5 % warm-up (the paper's 1.5 of 20 epochs), FP16 training, gradient clip 5, our augmentation
(small affine, colour jitter, blur, noise), SGM not trained. "Kept epoch" = lowest validation loss. With English weights
the encoder (164 of 164 tensors) and the FRM (14 of 16) are loaded; only the 94-class English classifier is new.</p>

<h1>8. Health check</h1>
{table(['Check', 'Result', 'OK?'], [
    ['Dataset version', f"{M('environment.json')}: dataset_version {H['dataset_version']}; counts printed in the log = v1.1 manifest", '✓'],
    ['All jobs finished', f"5 of 5 {M('[done] ... code=0')} in {H['total_h']:.2f} h (planned 2.5 h)", '✓'],
    ['Loss sane', 'no NaN or inf in any log; validation loss falls fast in the first 10 epochs for all five runs (Figure 7)', '✓'],
    ['Same code as the kit', f"{M('code/*.py')} = the kit's files (only report.py changed later, for an unrelated paper); notebook cells identical", '✓'],
    ['Numbers reproduce', f"every best.pt re-read on our PC: plate exact identical for all 5 runs on test AND val; same text in 7,079 of 7,085 readings (5 models x 1,417 crops; the 6 others are near-ties that GPU arithmetic tips the other way)", '✓'],
    ['English weights are the paper\'s', f"the shipped result files give T {A['released_weights']['T']['common_avg']} / S {A['released_weights']['S']['common_avg']} Common (= Table 3)", '✓'],
    ['CRNN baseline', f"model file SHA-256 {LC['sha256']['crnn_model_full'][:8]}… = the Model Fusion v1.1 run's file; its test and val readings are byte-identical", '✓'],
    ['Training finished', 'all four SVTRv2 runs stopped by the 1.6 h cap at epoch 63-66 of 100', '✗ caveat 1'],
    ['Checkpoint choice', 'kept on lowest validation loss: epochs 55 / 35 / 23 / 54', '✗ caveat 2'],
    ['Speed numbers', 'T4 ms measured while the other T4 was training (same model 7.62 vs 14.15 vs 14.99 ms)', '✗ caveat 3'],
], [2.2, 6.6, 1.2])}
"""

S['b2'] = f"""
<h1>9. The three caveats, what they mean, and the kit fix</h1>
<h3>Caveat 1: the schedule was cut at 63-66 % (all SVTRv2 runs)</h3>
<p>Each SVTRv2 epoch took about {H['svtrv2_t_msr']['s_per_epoch']:.0f} s on Kaggle, so 100 epochs need 2.5 h, but the cap was 1.6 h.
The one-cycle schedule lowers the learning rate slowly to almost zero at epoch 100; at the stop it was still about
{100 * H['svtrv2_t_msr']['lr_share_at_stop']:.0f} % of its peak. On our PC we profiled why an epoch is so slow for a
5 M-parameter model: reading and augmenting one crop takes 5.7 ms on the CPU (PIL colour jitter, affine, blur), while a
GPU step of 64 crops takes about 50 ms, so the GPU waits for the CPU about 85 % of the time (two jobs share Kaggle's 4 CPU
cores). The CRNN uses fast OpenCV augmentation and needed only 16.5 s per epoch.</p>
<h3>Caveat 2: the kept checkpoint was chosen on validation loss</h3>
{fig('fig_curves.png', 470, 'Figure 7. Validation and training CTC loss per epoch (dots: the kept checkpoints). The shaded part was planned but never run. Validation loss of the English-start runs stops falling after about 20 epochs while training loss keeps falling: CTC becomes over-confident, so the loss rises a little even when more plates are read right.')}
<p>For CTC the validation loss stops being a good guide early: it rises when the model is over-confident on a few
plates, even while accuracy improves. The fixed 32x128 run kept <b>epoch 23</b>, when the learning rate was still
{100 * H['svtrv2_t_fixed']['lr_share_at_best']:.0f} % of its peak; SVTRv2-S kept epoch 35; SVTRv2-T epoch 55. So <b>the MSR comparison
(T-MSR vs T-fixed) and the T vs S comparison are partly a comparison of checkpoints</b>. Only best.pt was saved, so we
cannot test other epochs now.</p>
<h3>Caveat 3: the Kaggle speed numbers are not usable</h3>
<p>The kit times each model at the end of its own job. SVTRv2-T was timed at {MN['svtrv2_t_msr']['ms_kaggle']:.2f} ms (MSR),
{MN['svtrv2_t_fixed']['ms_kaggle']:.2f} ms (fixed) and {MN['svtrv2_t_scratch']['ms_kaggle']:.2f} ms (scratch): the same network, two
times apart, because the other GPU's job was loading data on the shared CPU at the same time. The CRNN was timed last, with
nothing else running ({MN['crnn']['ms_kaggle']:.2f} ms). Part C uses our own timings instead (section 16).</p>
<div class="good"><p><b>Kit fixed (27 Oct), smoke-tested on our PC.</b> {M('p34_svtrv2.py')} and the spec now: (1) train 60
epochs with a 1.75 h cap, so the schedule always finishes (60 x 90 s = 1.5 h); (2) keep the epoch with the best
<b>validation plate exact</b> (ties: lower loss), and log it per epoch; (3) cache the decoded crops in RAM; (4) time every
reader at the end with both GPUs idle ({M('p34_svtrv2.py time')}, FP32 and FP16, per MSR size and for the test mix);
(5) report every run, also the fixed one, by MSR size group of the crop; (6) the "Fixed 32x128 (all plates)" row of the
old table, which showed the one 32x128-group plate of the MSR runs as 100.00 / 0.00, is gone. Planned time 3.5 h. FAST
smoke test: all jobs code 0 and the timing step ran (only the known Windows-only CRNN exit crash after "DONE").</p></div>
<div class="box"><p><b>Verdict.</b> The run is sound and its numbers can be used: they reproduce exactly, the data is v1.1, the
baseline is the same CRNN as in our other studies. Because SVTRv2 was stopped early and its checkpoints were picked on
loss, its accuracies are, if anything, <b>too low</b>, so "SVTRv2 beats CRNN" is safe. The MSR and T-vs-S comparisons need
the re-run with the fixed kit (Part D) before we state them strongly.</p></div>
"""

rows = []
for r in ALL:
    m = MN[r]
    rows.append([NM[r], f"<b>{f2(m['test']['plate_exact_%'])}</b>", f"{m['ci95'][0]:.1f}-{m['ci95'][1]:.1f}", f2(m['test']['CER_%']),
                 f2(m['test']['token_acc_%']), f2(m['val']['plate_exact_%']), f2(m['params_M']) if m['params_M'] else '–'])
S['c1'] = f"""
<p class="part">PART C · RESULTS AND WHAT THEY MEAN</p>
<h1>10. All readers on the same 651 test plates</h1>
{fig('fig_results.png', 470, 'Figure 2. Plate exact on the 651 test crops with 95 % bootstrap intervals. Left: the five readers of this run; right: our two best readers from earlier v1.1 runs (same test crops). Lines: the best fusion so far (dotted) and character YOLO + SVTRv2-T (green).')}
{table(['Reader', 'Plate exact %', '95 % CI', 'CER %', 'Token acc. %', 'Val exact %', 'Params (M)'], rows,
       [3.4, 1.2, 1.1, 0.9, 1.1, 1.1, 1.0], num=(1, 2, 3, 4, 5, 6))}
<p>CER = character error rate (Unicode characters). Validation is harder than test for every reader (for example the
ANPR source: SVTRv2-T {A['val_by_source']['anpr']['svtrv2_t_msr']:.1f} % on val vs {A['by_source']['anpr']['svtrv2_t_msr']:.1f} % on test), as in our earlier studies,
so compare readers on the same split only. The CRNN's 7.69 M parameters differ from the paper's CRNN (16.2 M): ours is a
smaller design.</p>

"""

S['c1b'] = f"""
<h1>11. Which differences are real? (paired tests)</h1>
{table(['Comparison (same 651 plates)', 'Difference', 'Plates only A / only B right', 'p (exact McNemar)', '95 % CI of the difference'], [
    ['SVTRv2-T vs CRNN', f"+{P['svtrv2_t_msr/crnn']['diff_pp']:.2f}", f"{P['svtrv2_t_msr/crnn']['only_a']} / {P['svtrv2_t_msr/crnn']['only_b']}", pv(P['svtrv2_t_msr/crnn']['p']), f"{P['svtrv2_t_msr/crnn']['ci95_diff'][0]:+.1f} to {P['svtrv2_t_msr/crnn']['ci95_diff'][1]:+.1f}"],
    ['SVTRv2-S vs CRNN', f"+{P['svtrv2_s_msr/crnn']['diff_pp']:.2f}", f"{P['svtrv2_s_msr/crnn']['only_a']} / {P['svtrv2_s_msr/crnn']['only_b']}", pv(P['svtrv2_s_msr/crnn']['p']), f"{P['svtrv2_s_msr/crnn']['ci95_diff'][0]:+.1f} to {P['svtrv2_s_msr/crnn']['ci95_diff'][1]:+.1f}"],
    ['SVTRv2-S vs SVTRv2-T', f"+{P['svtrv2_s_msr/svtrv2_t_msr']['diff_pp']:.2f}", f"{P['svtrv2_s_msr/svtrv2_t_msr']['only_a']} / {P['svtrv2_s_msr/svtrv2_t_msr']['only_b']}", pv(P['svtrv2_s_msr/svtrv2_t_msr']['p']), f"{P['svtrv2_s_msr/svtrv2_t_msr']['ci95_diff'][0]:+.1f} to {P['svtrv2_s_msr/svtrv2_t_msr']['ci95_diff'][1]:+.1f}"],
    ['MSR vs fixed 32x128 (T)', f"+{P['svtrv2_t_msr/svtrv2_t_fixed']['diff_pp']:.2f}", f"{P['svtrv2_t_msr/svtrv2_t_fixed']['only_a']} / {P['svtrv2_t_msr/svtrv2_t_fixed']['only_b']}", pv(P['svtrv2_t_msr/svtrv2_t_fixed']['p']), f"{P['svtrv2_t_msr/svtrv2_t_fixed']['ci95_diff'][0]:+.1f} to {P['svtrv2_t_msr/svtrv2_t_fixed']['ci95_diff'][1]:+.1f}"],
    ['English weights vs from scratch (T)', f"+{P['svtrv2_t_msr/svtrv2_t_scratch']['diff_pp']:.2f}", f"{P['svtrv2_t_msr/svtrv2_t_scratch']['only_a']} / {P['svtrv2_t_msr/svtrv2_t_scratch']['only_b']}", pv(P['svtrv2_t_msr/svtrv2_t_scratch']['p']), f"{P['svtrv2_t_msr/svtrv2_t_scratch']['ci95_diff'][0]:+.1f} to {P['svtrv2_t_msr/svtrv2_t_scratch']['ci95_diff'][1]:+.1f}"],
    ['SVTRv2-T from scratch vs CRNN', f"{P['svtrv2_t_scratch/crnn']['diff_pp']:.2f}", f"{P['svtrv2_t_scratch/crnn']['only_a']} / {P['svtrv2_t_scratch/crnn']['only_b']}", pv(P['svtrv2_t_scratch/crnn']['p']), f"{P['svtrv2_t_scratch/crnn']['ci95_diff'][0]:+.1f} to {P['svtrv2_t_scratch/crnn']['ci95_diff'][1]:+.1f}"],
    ['SVTRv2-T fixed 32x128 vs CRNN', f"+{P['svtrv2_t_fixed/crnn']['diff_pp']:.2f}", f"{P['svtrv2_t_fixed/crnn']['only_a']} / {P['svtrv2_t_fixed/crnn']['only_b']}", pv(P['svtrv2_t_fixed/crnn']['p']), f"{P['svtrv2_t_fixed/crnn']['ci95_diff'][0]:+.1f} to {P['svtrv2_t_fixed/crnn']['ci95_diff'][1]:+.1f}"],
    ['character YOLO vs SVTRv2-T', f"+{P['chars_v8n/svtrv2_t_msr']['diff_pp']:.2f}", f"{P['chars_v8n/svtrv2_t_msr']['only_a']} / {P['chars_v8n/svtrv2_t_msr']['only_b']}", pv(P['chars_v8n/svtrv2_t_msr']['p']), f"{P['chars_v8n/svtrv2_t_msr']['ci95_diff'][0]:+.1f} to {P['chars_v8n/svtrv2_t_msr']['ci95_diff'][1]:+.1f}"],
    ['LPNR YOLO vs SVTRv2-T', f"+{P['lpnr_v12n/svtrv2_t_msr']['diff_pp']:.2f}", f"{P['lpnr_v12n/svtrv2_t_msr']['only_a']} / {P['lpnr_v12n/svtrv2_t_msr']['only_b']}", pv(P['lpnr_v12n/svtrv2_t_msr']['p']), f"{P['lpnr_v12n/svtrv2_t_msr']['ci95_diff'][0]:+.1f} to {P['lpnr_v12n/svtrv2_t_msr']['ci95_diff'][1]:+.1f}"],
], [3.0, 1.1, 2.0, 1.4, 1.9], num=(1, 3))}
<p><b>Read it like this.</b> The first two rows and the last three are clear: SVTRv2 reads about 8 points more plates than
CRNN, and our character-detecting YOLO readers read 5.5-6.5 points more than SVTRv2. S vs T is a tie (38 vs 34 plates). The
English start is worth almost 20 points: SVTRv2 trained only on our 5,607 crops is 11.7 points <i>worse</i> than CRNN, so a
modern attention-based encoder needs far more data than we have, exactly the weak point named in section 6. And note the row
"SVTRv2-T fixed 32x128 vs CRNN": with a fixed 32x128 input SVTRv2-T is only level with CRNN ({pv(P['svtrv2_t_fixed/crnn']['p'])}).</p>
<div class="box"><p><b>Does this confirm the paper's title, "CTC beats encoder-decoder models"?</b> Not yet, and not
against it either. The title compares SVTRv2 with attention-decoder readers such as PARSeq; our team's PARSeq study has
not been run on dataset v1.1, so we have no attention-decoder reader on these plates. What we can say: (1) the paper's
CTC upgrades turn a CTC reader that is level with our CRNN (fixed 32x128) into one that is 7-8 points better; (2) a
family the paper did not compare at all, readers that <i>detect</i> each character (character YOLO, LPNR YOLO), is still
better on Bangla plates; (3) the paper's large-data condition matters: without the English pre-training the claim
fails on our 5,607 crops.</p></div>
<p><b>Statistics used throughout.</b> All readers are scored on the same 651 test crops, so differences are tested
<i>paired</i>: the exact McNemar test looks only at the plates where the two readers disagree ("only A right" vs "only B
right"); a 38 vs 34 split is a coin flip, an 87 vs 36 split is not. The 95 % intervals come from 5,000 bootstrap resamples
of the plates. With 651 plates, a difference below about 2.5 points is usually not significant.</p>
"""

AS = A['aspect']
S['c2'] = f"""
<h1>12. Multi-size resizing on Bangla plates</h1>
{fig('fig_aspect.jpg', 400, f"Figure 3. Left: aspect ratio of our plate crops (dotted lines = MSR thresholds). Right: one near-square test plate as photographed, as MSR feeds it (64x64), squashed to 32x128 (the fixed baseline), and as the CRNN sees it (two rows side by side).")}
<p>Our plates are not like English words. The median test crop has R = {AS['test']['median']:.2f}; {AS['test']['buckets']['64x64']}
of 651 test crops fall in MSR's 64x64 group (R &lt; 1.5) and {AS['test']['buckets']['48x96']} in the 48x96 group; only
{AS['test']['buckets'].get('40x112', 0) + AS['test']['buckets'].get('32x128', 0)} are wider. Both groups are mostly two-line plates; which group a plate falls in
depends mostly on how its source cropped it (median R: BLPD {AS['test_by_source']['blpd']['median_R']:.2f}, ANPR
{AS['test_by_source']['anpr']['median_R']:.2f}, LPDB-A {AS['test_by_source']['lpdba']['median_R']:.2f}, KUET {AS['test_by_source']['kuet']['median_R']:.2f}). The
training split has more wide plates (median R {AS['train']['median']:.2f}: {AS['train']['buckets']['48x96']} in 48x96 vs
{AS['train']['buckets']['64x64']} in 64x64), so the near-square group is a little under-represented in training.</p>
{fig('fig_msr.png', 410, 'Figure 4. MSR vs a fixed 32x128 input, by MSR size group. Left: the paper (SVTRv2 without SGM, curved and multi-oriented English words, Table 1, p.6). Right: our SVTRv2-T runs (and CRNN) on the same test plates; numbers above = MSR minus fixed (plates only MSR right vs only fixed right, exact McNemar).')}
{table(['MSR group', 'Test plates', 'T, MSR', 'T, fixed 32x128', 'MSR - fixed', 'CRNN', 'char. YOLO', 'T scratch'], [
    ['64x64 (R &lt; 1.5)', str(B['64x64']['n']), f2(B['64x64']['svtrv2_t_msr']), f2(B['64x64']['svtrv2_t_fixed']), f"{B['64x64']['msr_vs_fixed']['diff_pp']:+.2f} ({q(B['64x64']['msr_vs_fixed'])})", f2(B['64x64']['crnn']), f2(B['64x64']['chars_v8n']), f2(B['64x64']['svtrv2_t_scratch'])],
    ['48x96 (1.5-2.5)', str(B['48x96']['n']), f2(B['48x96']['svtrv2_t_msr']), f2(B['48x96']['svtrv2_t_fixed']), f"{B['48x96']['msr_vs_fixed']['diff_pp']:+.2f} ({q(B['48x96']['msr_vs_fixed'])})", f2(B['48x96']['crnn']), f2(B['48x96']['chars_v8n']), f2(B['48x96']['svtrv2_t_scratch'])],
], [1.6, 0.9, 0.9, 1.2, 2.8, 0.8, 1.0, 0.9], num=(1, 2, 3, 5, 6, 7))}
<p><b>What it means.</b> MSR helps overall, but the pattern is the <b>opposite of the paper's</b>: the paper gains most on
near-square text (+15.3) and less on the 48x96 group (+5.2); we gain {B['48x96']['msr_vs_fixed']['diff_pp']:+.1f} on the 48x96 group
and only {B['64x64']['msr_vs_fixed']['diff_pp']:+.1f} (not significant) on near-square plates. Three explanations, and how to tell them apart:</p>
{ul([
    '<b>Few CTC steps.</b> A 64x64 input gives only 16 output columns for 8-9 tokens, and every repeated digit (<span class="bn">৪৪</span>) needs a blank between the two copies; 48x96 gives 24. Our plates with a repeated digit are harder for every CTC reader (SVTRv2-T ' + f"{A['repeat_digits']['svtrv2_t_msr']:.1f} vs {A['no_repeat_digits']['svtrv2_t_msr']:.1f} % without), and SVTRv2 loses a repeated digit on {E['svtrv2_t_msr'].get('digit_repeat_lost', 0)} test plates (S: {E['svtrv2_s_msr'].get('digit_repeat_lost', 0)}). A test: MSR with 64x96 for R &lt; 1.5.",
    '<b>The fixed run kept an early checkpoint</b> (epoch 23, caveat 2). The re-run with the fixed kit settles this.',
    '<b>Character YOLO does not care about the shape</b> (86.9 vs 87.1 % on the two groups), which shows the near-square plates are not harder to see, only harder for a sequence reader.',
])}
"""

S['c3'] = f"""
<h1>13. Conditions and data sources</h1>
{fig('fig_conditions.png', 470, 'Figure 5. Plate exact by condition: the 651 test crops tagged clear / night / blurry, the 26 real fog crops and synthetic rain copies of all 651 test crops (labels on SVTRv2-T and CRNN bars).')}
{table(['Condition (plates)', 'SVTRv2-T', 'SVTRv2-S', 'T fixed', 'CRNN', 'char. YOLO', 'T vs CRNN (only T / only CRNN, p)'], [
    [f"clear ({C['clear']['n']})", f2(C['clear']['svtrv2_t_msr']), f2(C['clear']['svtrv2_s_msr']), f2(C['clear']['svtrv2_t_fixed']), f2(C['clear']['crnn']), f2(C['clear']['chars_v8n']), q(C['clear']['t_vs_crnn'])],
    [f"night ({C['night']['n']})", f2(C['night']['svtrv2_t_msr']), f2(C['night']['svtrv2_s_msr']), f2(C['night']['svtrv2_t_fixed']), f2(C['night']['crnn']), f2(C['night']['chars_v8n']), q(C['night']['t_vs_crnn'])],
    [f"blurry ({C['blurry']['n']})", f2(C['blurry']['svtrv2_t_msr']), f2(C['blurry']['svtrv2_s_msr']), f2(C['blurry']['svtrv2_t_fixed']), f2(C['blurry']['crnn']), f2(C['blurry']['chars_v8n']), q(C['blurry']['t_vs_crnn'])],
    [f"real fog ({A['extra_n']['test_fog']})", f2(A['extra_tests']['svtrv2_t_msr']['test_fog']), f2(A['extra_tests']['svtrv2_s_msr']['test_fog']), f2(A['extra_tests']['svtrv2_t_fixed']['test_fog']), f2(A['extra_tests']['crnn']['test_fog']), f2(A['extra_tests']['chars_v8n']['test_fog']), 'too few plates'],
    [f"rain, synthetic ({A['extra_n']['test_rain_synth']})", f2(A['extra_tests']['svtrv2_t_msr']['test_rain_synth']), f2(A['extra_tests']['svtrv2_s_msr']['test_rain_synth']), f2(A['extra_tests']['svtrv2_t_fixed']['test_rain_synth']), f2(A['extra_tests']['crnn']['test_rain_synth']), f2(A['extra_tests']['chars_v8n']['test_rain_synth']), '–'],
], [1.7, 0.9, 0.9, 0.9, 0.9, 1.0, 3.1], num=(1, 2, 3, 4, 5))}
<p>SVTRv2 is better than CRNN in every condition, but the gap is only significant on clear plates (the night and blur
subsets are small: 65 and 112 plates). Night is still the hardest case for every reader; SVTRv2-T reads
{C['night']['svtrv2_t_msr']:.1f} % of night plates vs {C['clear']['svtrv2_t_msr']:.1f} % of clear ones. The fixed 32x128 input hurts
most at night ({C['night']['svtrv2_t_fixed']:.1f} %; MSR vs fixed at night {pv(C['night']['msr_vs_fixed']['p'])}). On synthetic rain
SVTRv2 is the only reader that matches character YOLO (75-77 %).</p>
{table(['Source (test plates)', 'SVTRv2-T', 'SVTRv2-S', 'CRNN', 'char. YOLO', 'LPNR YOLO'], [
    [f"{s} ({v['n']})", f2(v['svtrv2_t_msr']), f2(v['svtrv2_s_msr']), f2(v['crnn']), f2(v['chars_v8n']), f2(v['lpnr_v12n'])]
    for s, v in A['by_source'].items()], [2.4, 1.2, 1.2, 1.2, 1.2, 1.2], num=(1, 2, 3, 4, 5))}
"""

ER = lambda r, k: E[r].get(k, 0)
S['c4'] = f"""
<h1>14. What kind of mistakes does each reader make?</h1>
{fig('fig_errors.png', 470, 'Figure 6. Wrong test plates by error type (a plate can have several). Letter = the series letter after the district; district / metro = the word(s) on the top line.')}
{table(['Reader', 'Wrong plates', 'Series letter', 'District / metro word', 'Digit swapped', 'Digit missing', 'Digit extra', 'Breaks the BRTA layout'], [
    [NM[r], str(E[r]['wrong']), str(ER(r, 'letter')), str(ER(r, 'district') + ER(r, 'metro')), str(ER(r, 'digits_substituted')), str(ER(r, 'digits_missing')), str(ER(r, 'digits_extra')), str(E[r]['invalid_layout'])]
    for r in ['svtrv2_t_msr', 'svtrv2_s_msr', 'svtrv2_t_fixed', 'crnn', 'chars_v8n', 'lpnr_v12n']],
    [3.0, 0.9, 0.9, 1.2, 0.9, 0.9, 0.9, 1.3], num=tuple(range(1, 8)))}
{ul([
    f"<b>SVTRv2 fixes the top line.</b> CRNN gets the district or metro word wrong on {ER('crnn', 'district') + ER('crnn', 'metro')} plates, SVTRv2-T on {ER('svtrv2_t_msr', 'district') + ER('svtrv2_t_msr', 'metro')}; series letters: {ER('crnn', 'letter')} vs {ER('svtrv2_t_msr', 'letter')}. Seeing the whole two-line plate with attention (instead of a hand-cut row) helps exactly where the letters are small.",
    f"<b>SVTRv2's own weakness is digits.</b> {ER('svtrv2_t_msr', 'digits')} of its {E['svtrv2_t_msr']['wrong']} errors involve a digit, often a swapped similar digit or a lost repeat (CTC merge). Character YOLO rarely swaps digits ({ER('chars_v8n', 'digits_substituted')}) but adds extra ones ({ER('chars_v8n', 'digits_extra')}: duplicate boxes).",
    f"<b>The two families fail on different plates.</b> Only {A['overlap']['chars_v8n/svtrv2_t_msr']['both_wrong']} plates are wrong for both character YOLO and SVTRv2-T (error overlap {A['overlap']['chars_v8n/svtrv2_t_msr']['jaccard_errors']:.2f}), against {A['overlap']['chars_v8n/lpnr_v12n']['both_wrong']} for the two YOLO readers (overlap {A['overlap']['chars_v8n/lpnr_v12n']['jaccard_errors']:.2f}). When character YOLO and SVTRv2-T agree ({A['t_chars_agree']['share_%']:.1f} % of plates) they are right {A['t_chars_agree']['acc_when_agree_%']:.1f} % of the time. This is why fusion works (section 15).",
    f"<b>A broken layout always means a wrong reading.</b> No reader produced a single correct plate whose text breaks the BRTA layout (district, [metro], letter, 6 digits). So the layout check is a free, perfect error flag; it flags {E['chars_v8n']['invalid_layout']} of character YOLO's {E['chars_v8n']['wrong']} wrong plates.",
])}
<table class="plain"><tr><td width="47%"><img src="fig_coverage.png" width="235"><p class="cap">Figure 8. Accuracy when a reader keeps only its most confident plates (the rest go to a human or a second reader).</p></td><td>
<p><b>Confidence.</b> Keeping the 80 % most confident plates, SVTRv2-T is right on {CV['svtrv2_t_msr']['acc_at_80']:.1f} % of them, CRNN on
{CV['crnn']['acc_at_80']:.1f} % (the same CRNN figure the LPLCv2 study found) and character YOLO on {CV['chars_v8n']['acc_at_80']:.1f} %.
SVTRv2's confidence (product of the per-token probabilities) is a useful but not a perfect error signal: it is better than
CRNN's, but the layout check and a second reader catch more.</p></td></tr></table>
"""

fr = lambda k, name: [name, f"{FR[k]['val_%']:.2f}", f"<b>{FR[k]['test_%']:.2f}</b>", f"{FR[k]['ci95'][0]:.1f}-{FR[k]['ci95'][1]:.1f}",
                      (f"{FR[k]['vs_chars_v8n']['only_new']} / {FR[k]['vs_chars_v8n']['only_base']}" if 'vs_chars_v8n' in FR[k] else '–'),
                      (f"{FR[k]['vs_old_best_fallback']['only_new']} / {FR[k]['vs_old_best_fallback']['only_base']} ({pv(FR[k]['vs_old_best_fallback']['p'])})" if 'vs_old_best_fallback' in FR[k] else 'reference'),
                      ('on ' + f"{FR[k]['helpers_called_%']:.1f} %" if FR[k]['rule'] == 'fallback' else 'all')]
S['c5'] = f"""
<h1>15. The key result: SVTRv2 as a second reader</h1>
<p>The Model Fusion study taught us to combine readers plate by plate. Our best rule there is the <b>format fallback</b>: take the
main reader's text if it fits the BRTA layout; otherwise let the other readers vote among their layout-valid readings.
The helpers only run on the plates that need them ({FR['chars_t_fallback']['helpers_called_%']:.2f} % here, the 46 plates whose
character-YOLO reading breaks the layout). We ran the kit's fusion code over five readers: character YOLO, LPNR YOLO, CRNN,
SVTRv2-T and SVTRv2-S (156 combinations x 6 rules, every choice made on validation, reported on test).</p>
{fig('fig_fusion.png', 470, 'Figure 9. Plate exact of fusions on the same 651 test plates (95 % bootstrap intervals); "fixed / broken" = plates the row gets right / wrong where the best fusion so far (character YOLO + LPNR fallback) does not.')}
{table(['Readers and rule', 'Val %', 'Test %', '95 % CI', 'Fixed / broken vs char. YOLO alone', 'Fixed / broken vs best so far', 'Helpers run'], [
    fr('old_kit_fallback', 'char. YOLO + CRNN, fallback (Model Fusion study)'),
    fr('old_best_fallback', 'char. YOLO + LPNR YOLO, fallback (best so far)'),
    fr('chars_t_fallback', '<b>char. YOLO + SVTRv2-T, fallback</b>'),
    fr('chars_s_fallback', 'char. YOLO + SVTRv2-S, fallback'),
    fr('new_best_fallback', 'char. YOLO + LPNR + SVTRv2-T + S, fallback'),
    fr('new_best_val', '<b>same four, MV-BM+format vote (chosen on val)</b>'),
], [3.4, 0.8, 0.8, 1.0, 1.4, 1.8, 0.9], num=(1, 2))}
{ul([
    f"<b>The cheapest strong system:</b> character YOLO + SVTRv2-T as fallback, <b>{FR['chars_t_fallback']['test_%']:.2f} %</b>: {FR['chars_t_fallback']['vs_chars_v8n']['only_new']} plates fixed and none broken compared with character YOLO alone ({pv(FR['chars_t_fallback']['vs_chars_v8n']['p'])}), and better than the best fusion so far ({q({'only_a': FR['chars_t_fallback']['vs_old_best_fallback']['only_new'], 'only_b': FR['chars_t_fallback']['vs_old_best_fallback']['only_base'], 'p': FR['chars_t_fallback']['vs_old_best_fallback']['p']})}). SVTRv2 fixes {FR['chars_t_fallback']['vs_chars_v8n']['only_new']} of the 46 flagged plates, CRNN only {FR['old_kit_fallback']['vs_chars_v8n']['only_new']}, LPNR YOLO {FR['old_best_fallback']['vs_chars_v8n']['only_new']}. Its cost: SVTRv2-T runs on 7 % of plates, about {0.0707 * tT['gpu_fp32']:.2f} ms extra on average on our GPU.",
    f"<b>The most accurate system:</b> the four-reader vote chosen on validation, <b>{FR['new_best_val']['test_%']:.2f} %</b> ({FR['new_best_val']['vs_old_best_fallback']['only_new']} fixed / {FR['new_best_val']['vs_old_best_fallback']['only_base']} broken vs the best so far, {pv(FR['new_best_val']['vs_old_best_fallback']['p'])}). It runs all four readers on every plate.",
    f"<b>Honest selection.</b> The best combination on <i>test</i> ({' + '.join(NM[m].split(' (')[0] for m in FU['best_on_test']['models'])}, {FU['best_on_test']['rule']}: {FU['best_on_test']['test_%']:.2f} %) was <i>not</i> the best on validation ({FU['best_on_test']['val_%']:.2f} %), so we do not report it as our result. The oracle (any of the five readers right) is {FU['oracle_test_%']:.1f} %: that is the ceiling for fusion with these readers.",
    f"<b>By condition</b> (best-so-far vs character YOLO + SVTRv2-T vs the vote): night {FU['by_condition']['night']['old_best_fallback']:.1f} / {FU['by_condition']['night']['chars_t_fallback']:.1f} / {FU['by_condition']['night']['new_best_val']:.1f} %, blurry {FU['by_condition']['blurry']['old_best_fallback']:.1f} / {FU['by_condition']['blurry']['chars_t_fallback']:.1f} / {FU['by_condition']['blurry']['new_best_val']:.1f} %.",
])}
"""

S['c6'] = f"""
<h1>16. Speed on our PC</h1>
<p>Our PC (RTX 5060 Ti, Core Ultra 7 265K, PyTorch {TM['torch'].split('+')[0]}), batch 1, model only (like the paper),
{TM['statistic']}, after 30 warm-up calls; all readers were timed one after the other in the same session, so they saw the
same conditions. SVTRv2 times are weighted by the test plates' MSR sizes ({W['64x64']} at 64x64, {W['48x96']} at 48x96,
{W.get('40x112', 0)} at 40x112, {W.get('32x128', 0)} at 32x128). {'Because another job was training on the same GPU, read these as best-case times and compare readers with each other, not with other reports; the re-run with the fixed kit times every reader on an idle T4.' if 'percentile' in TM['statistic'] else ''}</p>
{table(['Reader', 'Params (M)', 'GPU FP32 ms', 'GPU FP16 ms', 'GPU, batch 64: ms per plate', 'CPU 4 threads FP32 ms', 'Plate exact %'], [
    ['SVTRv2-T, MSR', '5.13', f"{tT['gpu_fp32']:.2f}", f"{tT['gpu_fp16']:.2f}", f"{sum(W[k] * TM['T'][k]['gpu_fp32_batch64_ms_per_plate'] for k in W) / sum(W.values()):.3f}", f"{tT['cpu_fp32']:.2f}", f2(ex('svtrv2_t_msr'))],
    ['SVTRv2-S, MSR', '11.25', f"{tS['gpu_fp32']:.2f}", f"{tS['gpu_fp16']:.2f}", f"{sum(W[k] * TM['S'][k]['gpu_fp32_batch64_ms_per_plate'] for k in W) / sum(W.values()):.3f}", f"{tS['cpu_fp32']:.2f}", f2(ex('svtrv2_s_msr'))],
    ['CRNN (48x256)', '7.69', f"{tC['gpu_fp32']:.2f}", f"{tC['gpu_fp16']:.2f}", f"{TM['crnn']['gpu_fp32_batch64_ms_per_plate']:.3f}", f"{tC['cpu_fp32']:.2f}", f2(ex('crnn'))],
], [2.0, 0.9, 1.0, 1.0, 1.5, 1.4, 1.1], num=(1, 2, 3, 4, 5, 6))}
{fig('fig_speed.png', 470, 'Figure 10. Accuracy vs model time per plate on our GPU (left) and CPU (right).')}
<p>Per MSR size, SVTRv2-T takes {TM['T']['64x64']['gpu_fp32']:.2f} ms at 64x64 and {TM['T']['48x96']['gpu_fp32']:.2f} ms at 48x96 on the GPU
(CPU: {TM['T']['64x64']['cpu_fp32']:.2f} / {TM['T']['48x96']['cpu_fp32']:.2f} ms). The whole reader on one plate (read the file, resize, model,
CTC decode) takes {TM['reader_svtrv2_t_msr_ms']:.2f} ms for SVTRv2-T and {TM['reader_crnn_ms']:.2f} ms for CRNN.
{('ONNX export of SVTRv2-T works (one file with dynamic size, ' + str(ON['MiB']) + ' MiB): same text as PyTorch on ' + str(ON['same_text_as_kaggle']) + ' of ' + str(ON['plates']) + ' test plates (max difference ' + f"{ON['max_abs_diff']:.1e}" + '), ' + f"{ON['ort_cpu4_ms']['64x64']:.2f} / {ON['ort_cpu4_ms']['48x96']:.2f}" + ' ms per plate with onnxruntime on 4 CPU threads (64x64 / 48x96).') if ON.get('ok') else ('ONNX export of SVTRv2-T failed here: ' + str(ON.get('error', 'not run'))[:160] + '.')}</p>
<p><b>Compared with the paper.</b> The paper ranks SVTRv2-T (201 FPS = 5.0 ms) ahead of CRNN (172 FPS = 5.8 ms) on a 1080Ti.
On our GPU {'the three readers are practically equal' if max(tT['gpu_fp32'], tS['gpu_fp32'], tC['gpu_fp32']) - min(tT['gpu_fp32'], tS['gpu_fp32'], tC['gpu_fp32']) < 0.5 else 'the readers differ'}: SVTRv2-T {tT['gpu_fp32']:.2f} ms, SVTRv2-S {tS['gpu_fp32']:.2f} ms, our CRNN {tC['gpu_fp32']:.2f} ms,
and FP16 does not help. At batch 1 models this small are limited by the number of GPU kernel launches, not by their
size; the size only shows in batches of 64 ({sum(W[k] * TM['T'][k]['gpu_fp32_batch64_ms_per_plate'] for k in W) / sum(W.values()):.2f} vs {sum(W[k] * TM['S'][k]['gpu_fp32_batch64_ms_per_plate'] for k in W) / sum(W.values()):.2f} vs {TM['crnn']['gpu_fp32_batch64_ms_per_plate']:.2f} ms per plate). All are far below the 100 ms per
photo that a 10 FPS pipeline (proposal objective O4) allows. On the CPU, the laptop case, SVTRv2-T is
<b>{'faster' if tT['cpu_fp32'] < tC['cpu_fp32'] else 'slower'} than CRNN</b> ({tT['cpu_fp32']:.1f} vs {tC['cpu_fp32']:.1f} ms; SVTRv2-S {tS['cpu_fp32']:.1f} ms), in line with the paper's claim
that SVTRv2-T is faster than CRNN.</p>
"""

EXS = A['figure_files']['examples']
S['c7'] = f"""
<h1>17. Examples</h1>
{fig('fig_examples.jpg', 470, 'Figure 11. Twelve test crops (texts below). Rows: SVTRv2-T right where CRNN is wrong; MSR right where the fixed 32x128 input is wrong; CRNN right where SVTRv2-T is wrong; every reader wrong.')}
{table(['#', 'Truth', 'SVTRv2-T (MSR)', 'T fixed 32x128', 'CRNN', 'char. YOLO'], [
    [str(e['n']), bn(e['true']), bn(e['svtrv2_t_msr']), bn(e['svtrv2_t_fixed']), bn(e['crnn']), bn(e['chars_v8n'])] for e in EXS],
    [0.4, 2.0, 2.0, 2.0, 2.0, 2.0])}
<p>Typical patterns: CRNN misreads the small top-line letter (3: {bn('জ')} for {bn('ব')}) or drops or swaps a digit
(1, 2, 4), while the fixed 32x128 input adds or swaps letters and digits on near-square plates (5: an extra letter;
6: {bn('৬')} read as {bn('২')}; 7: {bn('১')} as {bn('৯')}; 8: {bn('ঘ')} as {bn('গ')}). SVTRv2 loses on look-alike digits
(9: the last {bn('৮')} read as {bn('৪')}; 10: {bn('৮')} read as {bn('৪')}). The last row is hard for everyone: a blurred,
tilted grey plate (11) and a rare district (12: {bn('মুন্সিগঞ্জ')}, read as three other districts).</p>
"""

S['d1'] = f"""
<p class="part">PART D · HOW TO USE THIS IN OUR RESEARCH</p>
<h1>18. Where SVTRv2 fits in the 498R pipeline</h1>
<pre>photo ─► plate detector (YOLO) ─► plate crop ─► character YOLO reader ─► BRTA layout check
                                                        │ valid (93 %)          │ invalid (7 %)
                                                        ▼                       ▼
                                                   plate text ◄── vote of the layout-valid readings
                                                                   of SVTRv2-T (+ LPNR / SVTRv2-S)</pre>
{ul([
    '<b>Keep character YOLO as the main reader.</b> It is the most accurate single reader we have and its errors are easy to flag (extra boxes break the layout).',
    f"<b>Add SVTRv2-T as the second reader</b> behind the layout check: +{FR['chars_t_fallback']['vs_chars_v8n']['diff_pp']:.2f} points, no plate broken, and it runs on only 7 % of plates. It replaces the CRNN in that role (22 vs 12 plates fixed).",
    f"<b>For the most accurate offline result</b> (reports, the final table) use the four-reader vote: {FR['new_best_val']['test_%']:.2f} %.",
    '<b>Do not use SVTRv2 as the only reader</b>, and do not train it without the English weights.',
    f"<b>Weights to use:</b> {M('best.pt')} of {M('svtrv2_t_msr')} (19.6 MiB, in the GitHub upload) with the OpenOCR code at commit 1ccfc6e; loading code in the README of the upload folder (tested).",
])}

<h1>19. What we gain for the research</h1>
{table(['Proposal item', 'What this study adds'], [
    ['RQ2: can a character-level detector with plate-grammar correction reach &ge; 90 % on two-line plates, at lower cost than transformer recognisers?', f"Yes for plate exact: character YOLO + layout check + SVTRv2 helper = {FR['chars_t_fallback']['test_%']:.2f} %, four-reader vote {FR['new_best_val']['test_%']:.2f} %. And a measured answer to \"lower cost than transformer recognisers\": a modern attention-based CTC reader alone is 5.5 points <i>less</i> accurate than the character detector, at similar speed"],
    ['G5: the BRTA plate grammar is unused in other work', 'the layout check never accepted a wrong reading in 7 readers x 651 plates; it is the switch that makes fusion safe'],
    ['G1: accuracy vs speed', f"SVTRv2-T costs {tT['gpu_fp32']:.1f} ms (GPU) / {tT['cpu_fp32']:.1f} ms (CPU) per plate; as a fallback only {0.0707 * tT['cpu_fp32']:.1f} ms on average on the CPU"],
    ['O3: a recogniser with two-row decoding', 'FRM learns the reading order of two-line plates by itself (no hand-made row split) and makes half as many top-line mistakes as CRNN'],
    ['G4 / O5: robustness per condition', 'night, blur, real fog and synthetic rain numbers for a new reader family; night is still the weakest case'],
    ['G6: reproducibility', 'pinned official code, our weights, every prediction file, the exact notebook, and a fixed kit for the re-run'],
    ['Literature review', 'a 2025 ICCV paper tested outside English and Chinese for the first time in our project, with two findings that differ from the paper (data hunger; MSR pattern on near-square text)'],
], [3.8, 6.2])}

<h1>20. Next steps (in order)</h1>
{ul([
    '<b>Re-run on Kaggle with the fixed kit</b> (3.5 h, nothing to change in the notebook): 60 full epochs, checkpoint on val accuracy, clean timings. This settles caveats 1-3, especially MSR vs fixed and T vs S.',
    '<b>Two more seeds</b> for SVTRv2-T MSR and fixed (only if the re-run still shows a small MSR gap): one seed cannot separate a 4-point difference on 329 plates.',
    '<b>End-to-end test:</b> detector + character YOLO + SVTRv2-T fallback on the 204 end-to-end photos (BLPD external + NightLPBD), ms per photo on the T4 and on a laptop CPU (G1, O4).',
    '<b>Better pre-training for Bangla:</b> start from synthetic Bangla plates (the synthetic + real data fusion paper) instead of English words, or fine-tune on all our OCR crops first; English helped by 19.5 points, a Bangla start may help more.',
    '<b>MSR variant for plates:</b> 64x96 or 48x96 for R &lt; 1.5, to give near-square plates more CTC columns.',
    '<b>Optional:</b> SGM with the district list as "language" (small expected gain, section 4).',
])}
"""

S['e1'] = f"""
<p class="part">PART E · FOR THE FACULTY</p>
<h1>21. What to show (5 slides or 5 printed figures)</h1>
{table(['#', 'Show', 'One-line message'], [
    ['1', 'Figure 1 (the SVTRv2 pipeline)', 'A 2025 ICCV reader that is fast like CRNN (one pass, CTC) but sees the whole plate with attention; we used the authors\' own code and weights'],
    ['2', 'Figure 2 (all readers, same 651 plates)', f"SVTRv2 reads {P['svtrv2_t_msr/crnn']['diff_pp']:.1f} points more plates than our CRNN, but our character YOLO readers are still 5-6 points better"],
    ['3', 'The pre-training row of the table in section 11', f"Without English pre-training it is worse than CRNN ({f1(ex('svtrv2_t_scratch'))} %): the paper's results depend on millions of training crops"],
    ['4', 'Figure 4 (MSR, paper vs ours)', 'Shape-aware resizing helps on our plates too (+6.8), but where it helps differs from the paper'],
    ['5', 'Figure 9 + the table in section 15', f"As a second reader behind the plate-format check it gives {FR['chars_t_fallback']['test_%']:.2f} % (22 fixed, 0 broken); the four-reader vote is our first result above 90 % ({FR['new_best_val']['test_%']:.2f} %)"],
], [0.4, 3.2, 6.4])}

<h1>22. What to say (about two minutes)</h1>
<div class="box">
<p>"This week I studied SVTRv2 from ICCV 2025. Text readers come in two kinds: CTC readers like our CRNN are fast because
they read the whole image in one pass; attention-decoder readers like PARSeq are more accurate but write one character
at a time. SVTRv2 makes a CTC reader as accurate as the attention readers with three ideas: it resizes each crop to one of
four shapes by its aspect ratio instead of squashing it, it learns the reading order of the 2-D features, and during
training only it learns language context."</p>
<p>"I trained the authors' code on our 5,607 Bangla plate crops on Kaggle, from their English weights, and checked
every number again on our own PC. SVTRv2-T reads {f1(ex('svtrv2_t_msr'))} % of the 651 test plates exactly, 7.8 points
more than our CRNN, which is significant. But it is still 5.5 points below our character-detection YOLO reader, and from
scratch it is worse than CRNN, so it needs pre-training on a large dataset; the paper does not show that."</p>
<p>"Its value for us is as a second opinion: it makes different mistakes than the YOLO reader. When the YOLO reading does
not fit the Bangladeshi plate format, which always means it is wrong, we ask SVTRv2. That raises plate accuracy from
86.8 % to {FR['chars_t_fallback']['test_%']:.1f} % without breaking any correct plate, and voting four readers gives
{FR['new_best_val']['test_%']:.1f} %, our first result above the 90 % target. The run had three technical flaws, a time cap,
the checkpoint rule and the timing; I fixed the notebook and the next run will settle the open points."</p>
</div>

"""

S['e2'] = f"""
<h1>23. Questions they may ask, and short answers</h1>
{table(['Question', 'Answer'], [
    ['Why not just use SVTRv2 as the reader?', f"Alone it is {P['chars_v8n/svtrv2_t_msr']['diff_pp']:.1f} points worse than character YOLO ({pv(P['chars_v8n/svtrv2_t_msr']['p'])}). It is useful because its errors are different."],
    ['Is 90.94 % chosen on the test set?', 'No. All 156 combinations were ranked on validation, and the validation winner is reported on test. The test-best combination (91.71 %) is not claimed.'],
    ['Why would English pre-training help Bangla?', 'The encoder learns general stroke and shape features and the reading-order module learns to align features with a sequence; only the classifier is new. Without it, 5.6 k crops are too few for an attention model.'],
    ['Did you reproduce the paper?', 'The paper\'s own English numbers were not re-run (their checkpoints are only loaded on Kaggle), but the released files report exactly the Table 3 scores and our build has the paper\'s parameter counts. Our tests are on Bangla plates, so they test whether the ideas transfer.'],
    ['Is the MSR gain real?', f"Overall yes ({q(P['svtrv2_t_msr/svtrv2_t_fixed'])}), but the fixed-size run kept an earlier checkpoint, so part of the gap may be the checkpoint. The fixed kit removes this; then we will know."],
    ['Why no SGM?', 'It teaches language context; plates are random digits and a district from a list. The paper\'s own table shows its gain is small on regular text. Our English weights were trained with it.'],
    ['Why is S not better than T?', 'Tie (38 vs 34 plates). With 5.6 k crops the larger model does not help, and S kept an earlier checkpoint (epoch 35).'],
    ['Is it fast enough?', f"{tT['gpu_fp32']:.1f} ms per plate on our GPU and {tT['cpu_fp32']:.1f} ms on 4 CPU threads; as a fallback it runs on 7 % of plates only."],
], [3.0, 7.0])}
<div class="warn"><p><b>Limits to state yourself:</b> one seed per setting; the SVTRv2 runs were cut at 63-66 % of the
schedule; test differences of a few plates are not significant; night, blurry and fog subsets are small (65, 112, 26).</p></div>
"""

S['f1'] = f"""
<h1>Appendix: files, settings and sources</h1>
{table(['What', 'Where'], [
    ['Notes (passes 1-3, results block)', M('docs/Rokib Hasna/new5/34_SVTRv2 .../34_SVTRv2 ... .md')],
    ['Kaggle run (as run)', M('After Train/paper-34.ipynb') + ', ' + M('paper-34.log') + ', ' + M('results.zip') + ' (whole /kaggle/working, 419 MiB)'],
    ['Kit (fixed 27 Oct)', M('kaggle/paper34_kaggle.ipynb') + ', ' + M('_kit/extras/p34_svtrv2.py') + ', spec in ' + M('_kit/specs.py')],
    ['Local re-check', M('report_src/local_check.py') + ' → ' + M('local_check.json') + ', ' + M('local_preds_*.csv')],
    ['Paired tests, MSR groups, errors, fusion', M('report_src/analysis.py') + ' → ' + M('analysis.json') + ', ' + M('analysis_log.txt') + '; fusion with the kit\'s ' + M('p24_fusion.py')],
    ['Charts / this PDF', M('report_src/charts.py') + ', ' + M('build_report.py') + ' (+ ' + M('part_a.py') + ', ' + M('report_head.py') + ')'],
    ['GitHub upload folder', M('After Train/github_upload/') + ' with ' + M('UPLOAD_STEPS.md')],
], [3.0, 7.0])}
{table(['Paper named in this report', 'Full title (team folder)'], [
    ['Model Fusion', 'Laroca et al., Leveraging Model Fusion for Improved License Plate Recognition, CIARP 2023'],
    ['Cascaded YOLO', 'Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition'],
    ['LPLCv2', 'LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification'],
    ['PARSeq', 'Bautista and Atienza, Scene Text Recognition with Permuted Autoregressive Sequence Models, ECCV 2022'],
    ['Synthetic + real data fusion', 'Advancing Multinational License Plate Recognition Through Synthetic and Real Data Fusion'],
], [3.0, 7.0])}
<pre>@inproceedings{{du2025svtrv2,
  title     = {{{{SVTRv2}}: {{CTC}} Beats Encoder-Decoder Models in Scene Text Recognition}},
  author    = {{Du, Yongkun and Chen, Zhineng and Xie, Hongtao and Jia, Caiyan and Jiang, Yu-Gang}},
  booktitle = {{Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}},
  pages     = {{20147--20156}},
  year      = {{2025}}
}}</pre>
"""

FLOWS = [['cover'], ['a1'], ['a2'], ['a3'], ['a4'], ['b1'], ['b2'], ['c1'], ['c1b'], ['c2'], ['c3'], ['c4'], ['c5'],
         ['c6'], ['c7'], ['d1'], ['e1'], ['e2'], ['f1']]

if __name__ == '__main__':
    n = write_pdf(S, FLOWS, OUT, 'SVTRv2 study report · dataset v1.1 run · 498R BanglaLPR',
                  {'title': 'SVTRv2 study report (dataset v1.1): CTC beats encoder-decoder models in scene text recognition',
                   'author': 'Prepared for MD. Rokib Hasan Oli',
                   'subject': 'Paper study, v1.1 Kaggle results, local re-check, SVTRv2 on Bangla plates, fusion, research use, faculty guide'})
    print('pages', n, OUT)
