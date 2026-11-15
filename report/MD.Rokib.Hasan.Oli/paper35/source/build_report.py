"""Paper #35 study report (dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Yan et al., CVPR 2025, arXiv 2502.20272 v2; page numbers = PDF pages incl. the supplement); the v1.1
Kaggle run (`After Train/results.zip`, `paper-35.log`, 26 Oct 2026) and the LOL check (`results-original.zip`,
`paper-35-original.log`) via analysis.json; the local re-check on the RTX 5060 Ti (local_check.py, local_valstats.py:
every enhancer on all 1,153 test photos and 651 crops, the end-to-end test on the 204 e2e photos, timings); the
HVI-CIDNet code at commit eb43d7d (net/HVI_transform.py, net/CIDNet.py, eval.py, measure.py, train.py); the 498R
proposal (RQ4, G4, O5).
Run order: local_check.py -> local_valstats.py -> analysis.py -> charts.py -> build_report.py.
"""
import json
import os
import sys

from part_a import part_a
from report_head import M, f1, f2, fig, pv, table, ul, write_pdf

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'Paper35_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
R, K, C = A['run'], A['run']['kaggle_lowlight'], A['check']
RP, BT, BA, FP, RD = A['reproduction'], A['night_ap_bootstrap'], A['all_ap_bootstrap'], A['found_pairs'], A['read_pairs']
G, E, EP, SP, BX = A['gate'], A['e2e'], A['e2e_pairs'], A['speed_local'], A['boxes']
RS, EPARTS = A['read_scores'], A['read_error_parts']
V1, V2 = C['lolv1']['models'], C['lolv2_real']['models']
ENH = ['none', 'gamma', 'clahe', 'hvi_gen', 'hvi_lolv2']
NM = {'none': 'none', 'gamma': 'gamma 0.5', 'clahe': 'CLAHE', 'hvi_gen': 'HVI-CIDNet generalization',
      'hvi_lolv2': 'HVI-CIDNet LOLv2-Real'}
DV = R['det_test_plates']
CR = R['crnn']
LOGT = {'det_v8s': 108.0, 'crnn': 26.2, 'lowlight': 11.7, 'speed_det_v8s': 5.5, 'total_s': 7334}   # paper-35.log
night_map = {e: RP[e]['night']['kaggle_mAP50'] for e in ENH}
read_n = {e: RP[e]['read']['kaggle_night'] for e in ENH}
nn = lambda e: round(RD[f'{e}/night']['b']) if e != 'none' else RD['hvi_gen/night']['a']
e2n = lambda e, w='photo': E[f'{e}/{w}/nightlpbd']['exact']
e2d = lambda e, w='photo': E[f'{e}/{w}/blpd_ext']['exact']
ms_T4 = {e: K['ms'][e]['photo_640'] for e in ENH}
SRC = {'kuet': 'KUET', 'nightlpbd': 'NightLPBD', 'lpdba': 'LPDB-A', 'towniyan': 'Towniyan'}

# ---------------------------------------------------------------------------------------------- facts the text relies on
assert R['dataset_version'] == '1.1' and R['det']['epochs_run'] == 60 and R['det']['best_epoch'] == 55
assert R['det_nan'] == 0 and CR['epochs'] == 60 and CR['test']['plate_exact_%'] == 73.43
assert all(abs(RP[e][s]['kaggle_mAP50'] - RP[e][s]['local_mAP50']) <= 0.05 for e in ENH for s in ('night', 'clear200', 'all_gated'))
assert all(RP[e][s]['kaggle_found'] == RP[e][s]['local_found'] for e in ENH for s in ('night', 'clear200', 'all_gated'))
assert all(RP[e]['read']['kaggle_night'] == RP[e]['read']['local_night'] and RP[e]['read']['kaggle_gated'] == RP[e]['read']['local_gated'] for e in ENH)
assert all(abs(A['ap_check'][e]['night_valstats'] - A['ap_check'][e]['night_val']) <= 0.02 for e in ENH)
assert all(BT[e]['ci95'][0] < 0 < BT[e]['ci95'][1] for e in ENH[1:])
assert all(FP[f'{e}/night']['p'] == 1.0 for e in ENH[1:]) and FP['hvi_gen/night']['b'] == 64 and FP['hvi_gen/night']['n'] == 71
assert RD['gamma/night']['p'] < 0.01 and RD['hvi_gen/night']['p'] > 0.05 and RD['hvi_gen/day']['p'] < 0.01
assert RD['hvi_gen/night']['only_a'] == 7 and RD['hvi_gen/night']['only_b'] == 3
assert G['night_below_60'] == 17 and G['test_below_60'] == 20 and G['nightlpbd_below_60'] == 6
assert e2n('none') == 31 and e2n('clahe') == 35 and e2n('hvi_gen', 'detect') == 35 and all(p['p'] > 0.05 for p in EP.values())
assert BX['clahe']['plate_free_pairs']['p'] < 0.01 and BX['clahe']['plate_free_pairs']['b'] == 17
assert SP['640x640']['fp16']['nan_values'] == 0 and SP['640x640']['fp16']['ms_median'] < SP['640x640']['fp32']['ms_median'] / 2
assert abs(V1['hvi_lolv1']['gt_mean']['psnr'] - 28.2285) < 1e-3 and abs(V2['hvi_lolv2']['plain']['psnr'] - 23.8885) < 1e-3
assert len(A['night_missed_by_all']) == 6 and all(m['grey'] > 60 for m in A['night_missed_plates'])

S = part_a(A)
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER #35 STUDY REPORT · DATASET v1.1 RUN</p>
<h1 style="font-size:18pt; margin-bottom:3pt">HVI: A New Color Space for Low-light Image Enhancement (HVI-CIDNet)</h1>
<p style="font-family:H; font-size:11.5pt; color:#1c5cab; margin-bottom:8pt">The paper, our low-light run on dataset v1.1,
the check of the paper's own LOL numbers, whether a state-of-the-art enhancer helps our night plates, and what to show
and say to the faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 26 October 2026<br>
Paper: Q. Yan, Y. Feng, C. Zhang, G. Pang, K. Shi, P. Wu, W. Dong, J. Sun and Y. Zhang, CVPR 2025 (arXiv 2502.20272 v2,
30 pages with the supplement; page numbers are PDF pages). Code: github.com/Fediory/HVI-CIDNet (MIT), weights on
Hugging Face (safetensors).<br>
Our runs: (1) {M('paper35_kaggle.ipynb')}, Tesla T4 x2 on Kaggle, 26 Oct 2026, dataset <b>v1.1</b> (confirmed in
{M('environment.json')}), {LOGT['total_s'] / 3600:.2f} h; (2) {M('paper35_original_check.ipynb')} on LOLv1 / LOLv2-Real,
3 min. Extra checks on our own PC the same day (RTX 5060 Ti): every Kaggle number re-scored plate by plate, plus a new
end-to-end test.</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> A new colour space for brightening dark photos: hue becomes an angle (so red no longer
splits into two ends) and the colour of dark pixels is shrunk towards zero by a learnable curve (so their noise cannot
spread). A small two-branch network (CIDNet, 1.88 M parameters) brightens in that space. Best PSNR / SSIM on LOL and
8 other sets (p.6-7); the LOLv1 numbers use the ground truth's brightness ("GT mean"), and the authors lost the weights
behind the LOLv2-Real 24.11 dB.</p>
</div>
<div class="good">
<p><b>Is the v1.1 run OK? Yes, fully.</b> All four jobs ended with {M('code=0')} in {LOGT['total_s'] / 3600:.1f} h, dataset
v1.1 is confirmed, the YOLOv8s plate detector trained all 60 epochs (test mAP50 {f2(100 * R['det_test']['mAP50'])} %,
{f2(DV['found_%'])} % of 1,011 plates found), and the CRNN file is <b>byte-identical</b> to the one of the paper #24 run four
days earlier (same SHA-256), so the kit is reproducible. Our PC re-scored every Kaggle number with the same weights: the
same plates found and plates read, mAP50 within 0.04 points. The LOL check reproduces the authors' README to about
0.1 dB (LOLv2-Real {f2(V2['hvi_lolv2']['plain']['psnr'])} dB vs the paper's 24.11 from lost weights).</p>
</div>
<div class="warn">
<p><b>The answer for our research: no, HVI-CIDNet does not help our night plates</b> (zero-shot, as released).
(1) <b>Detection:</b> on the 71 night photos mAP50 rises {f2(night_map['none'])} -> {f2(night_map['hvi_gen'])} %, but the 95 %
interval of the gain includes zero ({f1(BT['hvi_gen']['ci95'][0])} to +{f1(BT['hvi_gen']['ci95'][1])}) and <b>not one extra plate is
found</b> ({FP['hvi_gen/night']['b']} of 71 with or without it); plain CLAHE does as well (+{f1(BT['clahe']['diff_pp'])}).
(2) <b>Reading:</b> enhanced night crops are read <b>worse</b> ({f1(read_n['none'])} -> {f1(read_n['hvi_gen'])} %, n.s.); enhancing
every crop costs {RD['hvi_gen/all']['a'] - RD['hvi_gen/all']['b']} of 651 plates ({pv(RD['hvi_gen/all']['p'])}).
(3) <b>End to end</b> on the 60 NightLPBD photos the best placement reads {e2n('hvi_gen', 'detect') - e2n('none')} more plates
({e2n('none')} -> {e2n('hvi_gen', 'detect')}, n.s.), the same as CLAHE. (4) <b>Why:</b> our night photos are not dark (median mean
grey {f1(G['night_grey_median'])}; only {G['night_below_60']} of 71 are below the team's gate of 60), and the plates the detector
misses are small or close-up, not dark. (5) <b>Cost:</b> {f1(ms_T4['hvi_gen'])} ms per photo on a T4, 24x the detector.</p>
<p><b>What to tell the faculty in one sentence.</b> "A state-of-the-art low-light enhancer makes our night photos look
better but does not make our detector or reader better, because our night failures come from glare, size and blur, not
darkness; this is the CLAHE and enhancement study of our proposal (RQ4), and it tells us to spend the effort on night
training data and task-aware training instead."</p>
</div>
<p class="small"><b>Parts:</b> A. Understand the paper (sections 1-5) · B. Our v1.1 experiment (6-15, read before you quote
a number) · C. Use it (16-22: research use, benefits, slides, script, Q&amp;A, next steps, GitHub).</p>
"""

S['b1'] = f"""
<p class="part">PART B · OUR EXPERIMENT ON DATASET v1.1</p>
<h1>6. What we ran</h1>
<p>The paper scores image quality against a well-lit photo of the same scene. We have no well-lit twin of our night
photos, so we score what we care about: <b>plates found and plates read</b>, with and without enhancement. Nothing is
trained on our plates; the enhancers are used <b>zero-shot</b>, exactly as released.</p>
{fig('fig_pipeline.png', 470, 'Figure 2. Where the enhancer sits. The Kaggle kit tests the "photo" route for the detector and the "crop" route for the reader; the local end-to-end test (section 12) compares all three routes on whole photos.')}
{table(['Job (GPU)', 'What it does', 'Time', 'Status'], [
    [M('det_v8s') + ' (0)', 'YOLOv8s plate detector, COCO start, 60 epochs, 640 px, batch 16, SGD, seed 42, human + auto boxes', f"{LOGT['det_v8s']:.0f} min", 'code=0'],
    [M('crnn') + ' (1)', 'CRNN-CTC plate reader, 60 epochs, AdamW + OneCycle, 48 x 256, two-line plates stacked', f"{LOGT['crnn']:.0f} min", 'code=0'],
    [M('lowlight') + ' (0)', 'five enhancers (none, gamma 0.5, CLAHE, HVI-CIDNet generalization and LOLv2-Real weights) on the 71 night photos, 200 clear photos and the whole test with a dark gate (mean grey &lt; 60): mAP50, mAP50-95, plates found; on the 65 night crops and all 651 crops with the gate: plate exact, CER; ms per photo and per crop', f"{LOGT['lowlight']:.0f} min", 'code=0'],
    [M('speed_det_v8s') + ' (1)', 'detector speed: PyTorch FP32 / FP16, TensorRT FP16, CPU ONNX', f"{LOGT['speed_det_v8s']:.1f} min", 'code=0'],
    [M('lolv1') + ', ' + M('lolv2_real'), 'original check: released weights on LOLv1 eval15 and LOLv2-Real test, PSNR / SSIM / LPIPS with and without GT mean', '0.5 + 2 min', 'code=0'],
    ['local (RTX 5060 Ti)', 'every enhancer on ALL 1,153 test photos and 651 crops, plate by plate; val statistics for paired bootstrap; end-to-end test on the 204 e2e photos (60 NightLPBD night + 144 BLPD day) in three routes; FP32 / FP16 timing', 'about 45 min', 'done'],
], [1.5, 6.1, 1.0, 0.9])}
<p class="small">Night photos = the dataset's {M('cond_test_night.txt')} (rule "mean grey &lt; 60 or NightLPBD"): 71 photos,
71 plates (60 from NightLPBD, 6 KUET, 4 LPDB-A, 1 Towniyan). Night crops = OCR test crops tagged night: 65.
Gamma 0.5 = every pixel value v becomes 255 (v / 255)^0.5; CLAHE = contrast-limited histogram equalisation on the
L channel of Lab (clip 2, 8 x 8 tiles). HVI generalization weights are the authors' real-world model (trained on
LOLv2 with random gamma); LOLv2-Real weights are their best-PSNR file with the 0.8 brightness factor of eval.py.</p>
"""

S['b2'] = f"""
<h1>7. Is the training OK?</h1>
{table(['Check', 'Result', 'OK?'], [
    ['Dataset version', f"{M('dataset_version: 1.1')} in environment.json; split counts in the log = v1.1 (detection 8,574 / 896 / 1,153; OCR 5,607 / 766 / 651)", 'yes'],
    ['All jobs finished', f"4 / 4 {M('code=0')}, {LOGT['total_s'] / 3600:.2f} h of the planned 4 h; nothing skipped ({M('skipped: {}')})", 'yes'],
    ['Detector trained fully', f"{R['det']['epochs_run']} / {R['det']['epochs_planned']} epochs, best epoch {R['det']['best_epoch']}, {R['det']['train_minutes']:.0f} min, no NaN in results.csv", 'yes'],
    ['Detector quality', f"test P {f1(100 * R['det_test']['P'])} / R {f1(100 * R['det_test']['R'])} / mAP50 {f2(100 * R['det_test']['mAP50'])} / mAP50-95 {f2(100 * R['det_test']['mAP50-95'])} %; plates found {f2(DV['found_%'])} % (NightLPBD {f2(DV['found_%_by_source']['nightlpbd']['found_%'])} %). Paper #6's YOLOv8s (human boxes only, 120 epochs): 97.6 / 45.9 %, 97.13 % found", 'yes'],
    ['CRNN quality', f"test plate exact {f2(CR['test']['plate_exact_%'])} %, CER {f2(CR['test']['CER_%'])} %; clear {f2(CR['by_condition']['clear']['plate_exact_%'])} %, night {f2(CR['by_condition']['night']['plate_exact_%'])} %, blurry {f2(CR['by_condition']['blurry']['plate_exact_%'])} %", 'yes'],
    ['Reproducible', f"{M('crnn/model_full.pt')} has the same SHA-256 ({M('0110631e...f033')}) as the CRNN of the paper #24 run of 22 Oct: same kit, same seed, a different Kaggle session, a bit-identical model", 'yes'],
    ['Same numbers on our PC', 'all 15 enhancer x set rows: plates found identical, mAP50 within 0.04 points (GPU rounding inside HVI-CIDNet), plate exact identical on night crops and gated crops (table below)', 'yes'],
    ['Notebooks', 'the two Kaggle notebooks are the kit notebooks cell for cell (FULL RUN, FAST = False); the run code in results.zip equals the kit files', 'yes'],
], [1.7, 7.5, 0.8])}
{fig('fig_training.png', 470, 'Figure 3. Training curves of the run: the detector\'s validation mAP flattens after epoch 30 (best epoch 55); the CRNN validation loss flattens after epoch 35 while the training loss keeps falling (the usual small-data overfit; the kit keeps the best-validation weights).')}
{table(['Set', 'Enhancer', 'Kaggle mAP50', 'Ours mAP50', 'Kaggle found %', 'Ours found %'],
       [[s, NM[e], f2(RP[e][s]['kaggle_mAP50']), f2(RP[e][s]['local_mAP50']), f2(RP[e][s]['kaggle_found']), f2(RP[e][s]['local_found'])]
        for s in ('night', 'clear200') for e in ENH], [1.1, 2.6, 1.2, 1.2, 1.3, 1.3], num=(2, 3, 4, 5))}
<p class="cap">Kaggle run vs our PC, same weights and photos. The gated whole-test rows (not shown) agree to 0.01.</p>
"""

lv1, lv1p, lv2, lv2s, gen = V1['hvi_lolv1'], V1['hvi_lolv1_perc'], V2['hvi_lolv2'], V2['hvi_lolv2_ssim'], V2['hvi_gen']
trip = lambda d: f"{d['psnr']:.2f} / {d['ssim']:.3f} / {d['lpips']:.3f}"
S['b3'] = f"""
<h1>8. The original-paper check: LOL</h1>
<p>The second notebook runs the authors' released weights on the paper's own test sets (LOLv1 eval15: 15 pairs; LOLv2-Real
test: 100 pairs) and scores them with the authors' {M('measure.py')} formulas, with and without GT mean. Same data, same
weights, so this <b>is</b> a direct check of Table 1.</p>
{table(['Weights', 'Test set', 'Plain PSNR / SSIM / LPIPS', 'GT mean', 'Paper (p.6)', 'Authors\' README', 'T4 ms'], [
    ['LOLv1, no perceptual loss', 'LOLv1', trip(lv1['plain']), '<b>' + trip(lv1['gt_mean']) + '</b>', '28.201 / 0.889 / <b>0.079</b>', '23.50 plain; 28.14 / 0.889 / 0.099', f1(lv1['ms_median'])],
    ['LOLv1, with perceptual loss', 'LOLv1', trip(lv1p['plain']), trip(lv1p['gt_mean']), '(same row)', '23.81 plain; 27.71 / 0.876 / 0.079', f1(lv1p['ms_median'])],
    ['LOLv2-Real best-PSNR', 'LOLv2-Real', '<b>' + trip(lv2['plain']) + '</b>', trip(lv2['gt_mean']), '24.111 / 0.871 / 0.108', '23.904 / 0.866 / 0.122 (24.111 file lost)', f1(lv2['ms_median'])],
    ['LOLv2-Real best-SSIM', 'LOLv2-Real', trip(lv2s['plain']), trip(lv2s['gt_mean']), '-', 'SSIM 0.8705', f1(lv2s['ms_median'])],
    ['Generalization (our kit)', 'LOLv2-Real', trip(gen['plain']), trip(gen['gt_mean']), '-', '-', f1(gen['ms_median'])],
], [2.0, 1.1, 1.9, 1.9, 1.8, 2.5, 0.7], num=(6,))}
{ul([
    f"<b>The released weights reproduce the README to about 0.1 dB</b> (LOLv1 GT mean {f2(lv1['gt_mean']['psnr'])} vs 28.14; LOLv2-Real {f2(lv2['plain']['psnr'])} vs 23.90). The code and the weights are honest.",
    f"<b>The paper's LOLv1 row needs GT mean and two files.</b> Without GT mean the same model gives {f2(lv1['plain']['psnr'])} dB, 4.6 dB less. PSNR 28.2 / SSIM 0.889 come from the model without perceptual loss, LPIPS 0.079 from the one with it ({lv1p['gt_mean']['lpips']:.3f}); the first one's LPIPS is {lv1['gt_mean']['lpips']:.3f}.",
    f"<b>LOLv2-Real 24.111 dB cannot be reproduced</b> with public weights: {f2(lv2['plain']['psnr'])} dB (-0.22).",
    f"<b>The generalization weights we use on plates</b> give {f2(gen['plain']['psnr'])} dB on LOLv2-Real (they are not tuned for it); with GT mean {f2(gen['gt_mean']['psnr'])} dB, almost the same as the specialised model, i.e. they mostly differ in overall brightness.",
    f"<b>Size and speed:</b> {lv1['params_M']:.3f} M parameters (paper 1.88 M); {f1(min(m['ms_median'] for m in (lv1, lv1p, lv2, lv2s, gen)))}-{f1(max(m['ms_median'] for m in (lv1, lv1p, lv2, lv2s, gen)))} ms per 600 x 400 image on a T4 in FP32 (paper 53 ms on an unnamed GPU; our RTX 5060 Ti: 42-53 ms).",
])}
"""

kt = lambda e, s: f2(100 * K['detection'][e][s]['mAP50'])
S['b4'] = f"""
<h1>9. Results next to the paper</h1>
<p>The paper reports no plate numbers, so the "Paper" column only has the two numbers that exist on both sides. This is
the kit's table ({M('results_vs_paper.md')}), checked against our local re-scoring.</p>
{table(['Metric', 'Paper (page)', 'Ours (Kaggle, v1.1)'], [
    ['Night photos (71): mAP50, no enhancement', '-', f"{kt('none', 'night')} %"],
    ['... after HVI-CIDNet generalization / LOLv2-Real', '-', f"{kt('hvi_gen', 'night')} / {kt('hvi_lolv2', 'night')} %"],
    ['... after CLAHE / gamma 0.5', '-', f"{kt('clahe', 'night')} / {kt('gamma', 'night')} %"],
    ['Night photos: plates found, none / HVI generalization', '-', f"{K['detection']['none']['night']['found_%']} / {K['detection']['hvi_gen']['night']['found_%']} % (64 of 71 both)"],
    ['Clear photos (200): mAP50, none / HVI generalization', '-', f"{kt('none', 'clear200')} / {kt('hvi_gen', 'clear200')} %"],
    ['Whole test with the dark gate: mAP50 change, HVI', '-', f"{K['detection']['hvi_gen']['gain_all_gated_mAP50_pp']:+.2f} points (20 of 1,153 photos enhanced)"],
    ['Night crops (65): plate exact, none / HVI / CLAHE / gamma', '-', f"{read_n['none']} / {read_n['hvi_gen']} / {read_n['clahe']} / {read_n['gamma']} %"],
    ['HVI-CIDNet ms per 640 x 640 photo', '53 ms per 600 x 400 image, GPU not named (p.7, Table 3)', f"{f1(ms_T4['hvi_gen'])} ms (T4, FP32)"],
    ['HVI-CIDNet ms per plate crop (64 x 192)', '-', f"{K['ms']['hvi_lolv2']['crop_64x192']:.1f}-{K['ms']['hvi_gen']['crop_64x192']:.1f} ms (T4)"],
    ['HVI-CIDNet parameters', '1.88 M (p.6, Table 1)', f"{K['params_M']['hvi_gen']} M"],
    ['Plate detector mAP50 / CRNN plate exact, all test', '-', f"{f2(100 * R['det_test']['mAP50'])} % / {CR['test']['plate_exact_%']} %"],
], [4.2, 3.0, 2.8])}
{fig('fig_results.png', 480, 'Figure 4. The three questions on night data. Left: change in mAP50 (AP50) on the 71 night photos vs no enhancement, with the 95 % interval of a paired photo bootstrap (5,000 draws, val\'s own boxes). Middle: night OCR crops read right. Right: the 60 NightLPBD photos read right end to end (route "photo"). Dashed line = no enhancement.')}
"""

pr = lambda k: f"{RD[k]['only_a']} / {RD[k]['only_b']}"
S['b5'] = f"""
<h1>10. Does enhancement help the detector?</h1>
{table(['Enhancer', 'Night mAP50', 'Change, 95 % CI', 'Night found (of 71)', 'Clear-200 mAP50', 'All 1,153 enhanced: mAP50 change', 'Plate-free photos with a box (of 160)'],
       [[NM[e], f2(night_map[e]), '-' if e == 'none' else f"{BT[e]['diff_pp']:+.2f} ({BT[e]['ci95'][0]:+.1f} to {BT[e]['ci95'][1]:+.1f})",
         str(FP['hvi_gen/night']['a'] if e == 'none' else FP[f'{e}/night']['b']), f2(RP[e]['clear200']['kaggle_mAP50']),
         '-' if e == 'none' else f"{BA[e]['diff_pp']:+.2f} ({BA[e]['ci95'][0]:+.1f} to {BA[e]['ci95'][1]:+.1f})",
         str(BX['none']['plate_free_pairs']['a'] if e == 'none' and 'plate_free_pairs' in BX['none'] else
             (BX['clahe']['plate_free_pairs']['a'] if e == 'none' else BX[e]['plate_free_pairs']['b'])) +
         ('' if e == 'none' else f" ({pv(BX[e]['plate_free_pairs']['p'])})")] for e in ENH],
       [2.3, 1.0, 1.7, 1.1, 1.1, 1.8, 1.6], num=(1, 3, 4))}
{ul([
    f"<b>No enhancer finds a single extra night plate.</b> At the working threshold (conf 0.25, IoU 0.5) the detector finds 64 of the 71 night plates with no enhancement, with gamma, with CLAHE and with HVI generalization (McNemar p = 1.0 for all); the LOLv2-Real model loses one.",
    f"<b>The mAP50 gains are within noise.</b> HVI generalization +{f2(BT['hvi_gen']['diff_pp'])} and CLAHE +{f2(BT['clahe']['diff_pp'])} points, but both 95 % intervals include zero (71 photos). mAP50 rewards a higher score on plates that are found anyway; it does not show new plates.",
    f"<b>The ranking even depends on how photos are resized.</b> Scoring the same boxes with Ultralytics' {M('predict')} letterbox instead of {M('val')}'s batch rectangles moves the night AP50 of every enhancer by up to 1.6 points, and the order changes (HVI +{f2(A['night_ap_diff_predict']['hvi_gen'])}, CLAHE +{f2(A['night_ap_diff_predict']['clahe'])}, LOLv2-Real {f2(A['night_ap_diff_predict']['hvi_lolv2'])}). Differences of 1-2 points on 71 photos are not evidence.",
    f"<b>Day photos are not hurt</b> (clear-200 mAP50 within 0.2 points, 201 of 202 plates found every time), but <b>false alarms rise</b> when every photo is enhanced: CLAHE puts a box on {BX['clahe']['plate_free_pairs']['b']} of 160 plate-free photos instead of {BX['clahe']['plate_free_pairs']['a']} ({pv(BX['clahe']['plate_free_pairs']['p'])}), HVI on {BX['hvi_gen']['plate_free_pairs']['b']} (n.s.). Another reason to enhance only behind a gate.",
    f"<b>The night misses are not dark.</b> The {len(A['night_missed_by_all'])} night plates that no setting finds sit in photos with mean grey {min(m['grey'] for m in A['night_missed_plates']):.0f}-{max(m['grey'] for m in A['night_missed_plates']):.0f}; four are small ({min(m['area_px'] for m in A['night_missed_plates'] if m['area_px'] < 20000) / 1000:.1f}-{max(m['area_px'] for m in A['night_missed_plates'] if m['area_px'] < 20000) / 1000:.1f} thousand px², median night plate {A['night_plate_area_median'] / 1000:.1f}) and two are close-ups half the photo wide (a tenth of its area; the training data has few plates that large). Brightness is not their problem.",
])}
{fig('fig_examples.jpg', 470, 'Figure 5. Three NightLPBD test photos (the darkest, the median and a bright one by mean grey) after each enhancer. HVI-CIDNet brightens and cleans the dark scene, as the paper promises; the plate itself was already readable in all three.')}
"""

S['b6'] = f"""
<h1>11. Does enhancement help the reader?</h1>
{table(['Enhancer', 'Night crops right (of 65)', 'Broken / fixed vs none', 'p', 'Night CER %', 'All 651 crops enhanced', 'Broken / fixed', 'p', 'Gate &lt; 60 (13 crops)'],
       [[NM[e], str(RD['hvi_gen/night']['a'] if e == 'none' else RD[f'{e}/night']['b']), '-' if e == 'none' else pr(f'{e}/night'),
         '-' if e == 'none' else f"{RD[f'{e}/night']['p']:.3f}", f2(RS[f'{e}/night']['CER_%']),
         f"{RS[f'{e}/all']['plate_exact_%']} %", '-' if e == 'none' else pr(f'{e}/all'),
         '-' if e == 'none' else f"{RD[f'{e}/all']['p']:.4f}", f"{RP[e]['read']['kaggle_gated']} %"] for e in ENH],
       [2.3, 1.1, 1.1, 0.7, 0.9, 1.2, 1.0, 0.8, 1.1], num=(1, 3, 4, 7))}
{ul([
    f"<b>Enhanced night crops are read worse, never better.</b> HVI generalization breaks {RD['hvi_gen/night']['only_a']} plates and fixes {RD['hvi_gen/night']['only_b']} (n.s.); CLAHE is neutral ({pr('clahe/night')}); gamma 0.5 is clearly harmful ({pr('gamma/night')}, {pv(RD['gamma/night']['p'])}).",
    f"<b>On normal crops the harm is significant.</b> Applied to every crop, HVI generalization loses {RD['hvi_gen/day']['a'] - RD['hvi_gen/day']['b']} of the 586 day plates ({pr('hvi_gen/day')}, {pv(RD['hvi_gen/day']['p'])}), CLAHE {RD['clahe/day']['a'] - RD['clahe/day']['b']} ({pv(RD['clahe/day']['p'])}). The CRNN learned the look of normal crops; an enhancer changes contrast, colour and edges in ways it never saw.",
    f"<b>Night crops are not dark either.</b> Median mean grey of the 65 night crops: {f1(G['night_crop_grey_median'])} (headlights and reflective plates); only {G['night_crops_below_60']} are below 60, so the gate touches 13 crops and changes the total by at most 0.3 points.",
    f"<b>Where the reader fails at night</b> (25 wrong night plates, no enhancement): district name {EPARTS['none/night']['district']}, series letter {EPARTS['none/night']['letter']}, digits {EPARTS['none/night']['digits']} (a plate can fail in several parts). With HVI: {EPARTS['hvi_gen/night']['wrong_plates']} wrong, digits {EPARTS['hvi_gen/night']['digits']}. Typical HVI errors: a digit or letter whose strokes become brighter and thinner (<span class='bn'>৭ / ১, ঘ / ক</span>), see Figure 6.",
])}
{fig('fig_crops.jpg', 280, 'Figure 6. Six night-tagged OCR crops (rows) after each enhancer, with the CRNN\'s verdict. Rows 1-4: HVI breaks a reading that was right; rows 5-6: HVI fixes one. The texts are in the table below.')}
{table(['#', 'True text', 'none', 'HVI generalization', 'CLAHE'],
       [[str(i + 1), f"<span class='bn'>{A['flip_examples'][f]['true']}</span>", f"<span class='bn'>{A['flip_examples'][f]['none']}</span>",
         f"<span class='bn'>{A['flip_examples'][f]['hvi_gen']}</span>", f"<span class='bn'>{A['flip_examples'][f]['clahe']}</span>"]
        for i, f in enumerate(A['crop_examples'])], [0.4, 2.4, 2.4, 2.4, 2.4])}
"""

way_row = lambda e: [NM[e]] + [f"{e2n(e, w)} ({e2n(e, w) - e2n('none'):+d})" if e != 'none' else str(e2n('none'))
                               for w in ('photo', 'detect', 'crop')] + \
                    [f"{e2d(e, w)} ({e2d(e, w) - e2d('none'):+d})" if e != 'none' else str(e2d('none')) for w in ('photo', 'detect')]
S['b7'] = f"""
<h1>12. End to end on the 60 NightLPBD photos (new test)</h1>
<p>The kit enhances whole photos for the detector but only the cut-out crop for the reader. An enhancer trained on whole
scenes might work better on the whole photo. So on our PC we ran the full pipeline (detector -> most confident plate ->
5 % padding -> CRNN) on the e2e test photos in three routes (Figure 2): <b>photo</b> (detect and read on the enhanced
photo), <b>detect</b> (detect on the enhanced photo, read the original pixels), <b>crop</b> (enhance only the crop).</p>
{table(['Enhancer', 'Night: photo', 'Night: detect', 'Night: crop', 'Day (BLPD, 144): photo', 'Day: detect'],
       [way_row(e) for e in ENH], [2.6, 1.3, 1.3, 1.3, 1.9, 1.6], num=(1, 2, 3, 4, 5))}
<p class="cap">Plates read exactly right, of 60 night photos and of 144 BLPD day photos (change vs none in brackets).
The detector finds a plate in 54 of the 60 night photos in every setting: the same 6 photos are missed by all of them,
so "run the enhancer only when no plate is found" would recover nothing.</p>
{ul([
    f"<b>The best routes gain 4 plates of 60</b>: CLAHE on the whole photo ({e2n('none')} -> {e2n('clahe')}, {EP['clahe/photo/nightlpbd']['only_b']} fixed, {EP['clahe/photo/nightlpbd']['only_a']} broken, {pv(EP['clahe/photo/nightlpbd']['p'])}) and HVI generalization for the detector only ({e2n('none')} -> {e2n('hvi_gen', 'detect')}, {EP['hvi_gen/detect/nightlpbd']['only_b']} fixed / {EP['hvi_gen/detect/nightlpbd']['only_a']} broken, {pv(EP['hvi_gen/detect/nightlpbd']['p'])}). Neither is significant; showing a gain of this size would take two to three times more night photos.",
    f"<b>Feeding enhanced pixels to the reader is the risky part:</b> the LOLv2-Real model loses {e2n('none') - e2n('hvi_lolv2')} plates in the photo and crop routes, but gains {e2n('hvi_lolv2', 'detect') - e2n('none')} in the detect route. Where it helps, enhancement helps <b>the box</b>, not the reading (a better crop position).",
    f"<b>Day photos are unchanged</b> within ±2 plates in the photo and detect routes (CLAHE on crops: {e2d('clahe', 'crop') - e2d('none'):+d}, n.s.).",
    f"The 60 e2e night photos are the same NightLPBD photos as in the detection test; their median mean grey is {f1(A['e2e_night_grey']['median'])} and only {A['e2e_night_grey']['below_60']} are below 60.",
])}
"""

sw = A['gate_sweep']
S['b8'] = f"""
<h1>13. The dark gate</h1>
<p>The kit only enhances photos with mean grey &lt; 60 (the team's night rule), so day photos stay untouched. This gate
is safe but almost never fires: <b>{G['test_below_60']} of 1,153 test photos</b> ({', '.join(f'{v} ' + SRC.get(k, k) for k, v in sorted(G['test_below_60_sources'].items(), key=lambda x: -x[1]))})
and only <b>{G['night_below_60']} of the 71 night photos</b> ({G['nightlpbd_below_60']} of the 60 NightLPBD photos). Night
photos have a median mean grey of {f1(G['night_grey_median'])} (quartiles {G['night_grey_quartiles'][0]:.0f} / {G['night_grey_quartiles'][2]:.0f}):
street lights, headlights and the cameras' automatic exposure make them as bright as many day photos.</p>
{fig('fig_gate.png', 480, 'Figure 7. (a) Mean grey of every test photo; blue = night-tagged. (b) Whole-test AP50 change when every photo darker than the threshold is enhanced (val boxes). Nothing changes until about 700 photos (threshold 120) are enhanced, and then the gain (0.2-0.5 points, n.s.) comes from ordinary day photos.')}
{ul([
    f"<b>Gate at 60:</b> whole-test mAP50 {f2(RP['none']['all_gated']['kaggle_mAP50'])} -> {f2(RP['hvi_gen']['all_gated']['kaggle_mAP50'])} % (HVI), plates found unchanged ({RP['hvi_gen']['all_gated']['kaggle_found']} %). Safe, and useless.",
    f"<b>Gate = the night tag</b> (enhance exactly the 71 night photos): whole-test AP50 {A['gate_sweep_base']['all_AP50']} -> {A['night_tag_gate_all_AP50']['hvi_gen']} % (HVI) / {A['night_tag_gate_all_AP50']['clahe']} % (CLAHE). Tiny.",
    f"<b>No gate</b> (enhance all 1,153): +{BA['hvi_gen']['diff_pp']:.2f} (HVI), +{BA['clahe']['diff_pp']:.2f} (CLAHE), +{BA['gamma']['diff_pp']:.2f} (gamma) points, all n.s., plus the extra false alarms of section 10 and the reading loss of section 11.",
    "<b>Lesson for the team:</b> \"night\" in our data is a capture condition (NightLPBD), not a brightness level. A brightness gate does not find the night photos; if a gate is needed, base it on the plate region or on the detector's confidence, and report night results by source.",
])}
"""

sp = SP
S['b9'] = f"""
<h1>14. Speed</h1>
{fig('fig_speed.png', 470, 'Figure 8. Cost per 640 x 640 photo on the Kaggle T4 (batch 1, median). HVI-CIDNet alone takes 24x the time of the TensorRT detector.')}
{table(['Where', 'HVI-CIDNet FP32', 'FP16 (.half)', 'autocast', 'Note'], [
    ['T4 (Kaggle), 640 x 640', f"{f1(ms_T4['hvi_gen'])} ms", '-', '-', f"LOL 600 x 400: {f1(gen['ms_median'])} ms; crop 64 x 192: {K['ms']['hvi_lolv2']['crop_64x192']:.0f}-{K['ms']['hvi_gen']['crop_64x192']:.0f} ms"],
    ['RTX 5060 Ti, 640 x 640', f"{f1(sp['640x640']['fp32']['ms_median'])} ms", f"<b>{f1(sp['640x640']['fp16']['ms_median'])} ms</b>", f"{f1(sp['640x640']['autocast']['ms_median'])} ms", f"FP16 output = FP32 to {sp['640x640']['fp16']['psnr_vs_fp32_dB']:.0f} dB PSNR, no NaN"],
    ['RTX 5060 Ti, 1280 x 960', f"{f1(sp['1280x960']['fp32']['ms_median'])} ms", f"{f1(sp['1280x960']['fp16']['ms_median'])} ms", f"{f1(sp['1280x960']['autocast']['ms_median'])} ms", f"peak GPU memory {sp['peak_gpu_GB_1280x960']} GB (FP32)"],
    ['RTX 5060 Ti, real crop 224 x 256', f"{f1(sp['crop_224x256']['fp32']['ms_median'])} ms", f"{f1(sp['crop_224x256']['fp16']['ms_median'])} ms", f"{f1(sp['crop_224x256']['autocast']['ms_median'])} ms", 'small inputs: launch overhead dominates, FP16 gains nothing'],
], [2.4, 1.3, 1.2, 1.1, 4.0])}
{ul([
    f"Our goal G1 is &gt;= 10 FPS for the whole pipeline on a laptop (100 ms per frame). HVI-CIDNet in FP32 needs {f1(ms_T4['hvi_gen'])} ms on a T4 by itself; even in FP16 it would take most of the budget. It can only run behind a gate that rarely fires, or offline.",
    f"<b>FP16 is safe</b> for this network (the paper's {M('eps = 1e-8')} does not underflow into NaN in our test) and {sp['640x640']['fp32']['ms_median'] / sp['640x640']['fp16']['ms_median']:.1f}x faster on 640 x 640 photos. Useful if the enhancer is kept for the demo display.",
    f"CLAHE costs {f1(ms_T4['clahe'])} ms on the CPU (no GPU) and gamma {f1(ms_T4['gamma'])} ms, so the classic methods are 15-250x cheaper for the same (lack of) effect.",
])}
"""

S['b10'] = f"""
<h1>15. Findings, and the numbers to quote</h1>
{table(['#', 'Finding', 'Number', 'Strength'], [
    ['1', 'The v1.1 run is complete and reproducible', 'all jobs code=0; CRNN bit-identical to the #24 run; local = Kaggle', 'strong'],
    ['2', 'The paper\'s released weights reproduce its README, not all of its Table 1', f"LOLv1 {f2(lv1['gt_mean']['psnr'])} with GT mean, {f2(lv1['plain']['psnr'])} without; LOLv2-Real {f2(lv2['plain']['psnr'])} vs 24.11", 'strong'],
    ['3', 'Enhancement finds no extra night plate', f"64 / 71 found with none, gamma, CLAHE, HVI (p = 1.0)", 'strong'],
    ['4', 'Night mAP50 gains are within noise', f"HVI {BT['hvi_gen']['diff_pp']:+.2f} ({BT['hvi_gen']['ci95'][0]:+.1f} to {BT['hvi_gen']['ci95'][1]:+.1f}), CLAHE {BT['clahe']['diff_pp']:+.2f} ({BT['clahe']['ci95'][0]:+.1f} to {BT['clahe']['ci95'][1]:+.1f})", 'strong (n = 71)'],
    ['5', 'Enhancing crops hurts the reader', f"night {nn('none')} -> {RD['hvi_gen/night']['b']} (n.s.); day 438 -> {RD['hvi_gen/day']['b']} ({pv(RD['hvi_gen/day']['p'])})", 'strong for day, weak for night'],
    ['6', 'End to end, the best routes gain 4 / 60 night plates', f"none {e2n('none')}; CLAHE photo {e2n('clahe')}; HVI detect-only {e2n('hvi_gen', 'detect')} (p &gt; 0.1)", 'weak (n = 60)'],
    ['7', 'Our night photos are not dark', f"median mean grey {f1(G['night_grey_median'])}; {G['night_below_60']} / 71 below 60; crops {f1(G['night_crop_grey_median'])}", 'strong'],
    ['8', 'CLAHE on all photos doubles false alarms', f"{BX['clahe']['plate_free_pairs']['a']} -> {BX['clahe']['plate_free_pairs']['b']} of 160 plate-free photos ({pv(BX['clahe']['plate_free_pairs']['p'])})", 'medium'],
    ['9', 'HVI-CIDNet is too slow for every frame', f"{f1(ms_T4['hvi_gen'])} ms (T4 FP32) vs detector {f1(R['det_speed']['tensorrt_fp16']['ms_median'])} ms; FP16 {sp['640x640']['fp32']['ms_median'] / sp['640x640']['fp16']['ms_median']:.1f}x faster, no NaN", 'strong'],
    ['10', 'Small mAP differences depend on the resize rule', 'up to 1.6 points on the 71 night photos (predict vs val letterbox)', 'strong'],
], [0.4, 3.8, 4.4, 1.4])}
<div class="box"><p><b>Bottom line.</b> HVI-CIDNet is a good image enhancer and an honest code release, but as a zero-shot
pre-processing step it does not improve Bangla plate detection or reading on our night data, and it costs 155 ms per
photo. CLAHE gives the same small detector effect for 10 ms. Our night problem is glare, small or close-up plates and
the reader, not darkness.</p></div>
"""

S['c1'] = f"""
<p class="part">PART C · USE IT</p>
<h1>16. How to use this paper in our research</h1>
{table(['Use', 'What to do', 'Where in the proposal'], [
    ['1. Report it as our pre-processing study', 'Put the five-row table (none / gamma / CLAHE / HVI x2) for night detection, night reading and end to end into the thesis as an ablation: "image-quality enhancement does not transfer to recognition". A clear negative result with paired tests is publishable and reviewers ask for it.', 'G4 + RQ4 ("CLAHE and upscaling study"), O5'],
    ['2. Keep the pipeline simple', 'Do not add an enhancer to the default pipeline. If anything, CLAHE on the photo for the detector, behind a gate, CPU only.', 'G1 / O4 (10 FPS)'],
    ['3. Fix how we define and test "night"', 'Report night by source (NightLPBD) and by plate brightness, not by a photo-brightness gate; add the gate finding to the dataset notes.', 'O5 per-condition tests'],
    ['4. Spend the night effort where the errors are', 'Small and close-up plates, glare, and the reader (digits and letters). Options: night-style augmentation in training (gamma, noise, blur, glare), more NightLPBD-like data, multi-frame voting on video (#24 fusion), super-resolution (#20).', 'G4, O3, O4'],
    ['5. If you want enhancement, make it task-aware', 'Train the reader on enhanced copies too, or fine-tune HVI-CIDNet with the detector / CRNN loss instead of PSNR (next steps, section 21).', 'future work'],
    ['6. Use HVI where humans look', 'In the demo, show the operator an HVI-enhanced view of a night plate (FP16, about 30 ms on our GPU, only when asked). Machines read the original.', 'demo / prototype'],
    ['7. Cite it correctly', 'Cite HVI-CIDNet as the state of the art in LLIE (CVPR 2025) and state the GT-mean and lost-weights caveats when quoting its numbers.', 'related work'],
], [2.4, 5.4, 2.2])}

<h1>17. What benefit do you get from it?</h1>
{ul([
    '<b>A finished answer to a question every reviewer asks</b> ("did you try low-light enhancement for night plates?"), with evidence on 71 night photos, 65 night crops, 60 end-to-end photos and paired statistics.',
    '<b>Saved effort and latency:</b> we now know not to put a 155 ms network in front of a 6.5 ms detector.',
    '<b>A better understanding of our data:</b> the night subset is defined by source, its photos are not dark, and the night misses are size and glare problems. This redirects the night work of the whole team.',
    '<b>Reproducibility evidence for the thesis:</b> a bit-identical CRNN across two Kaggle sessions, Kaggle numbers re-scored to the plate on a second machine, and the paper\'s README reproduced to 0.1 dB.',
    '<b>Method you can reuse:</b> paired McNemar tests per plate, a paired photo bootstrap on the validator\'s own boxes, and the warning that 1-2 mAP points on 71 photos move with the resize rule.',
    '<b>A critical-reading example</b> for the literature review: GT mean, a mixed results row, lost weights and test-time constants, each with a page number.',
])}
"""

S['c2'] = f"""
<h1>18. What to show the faculty (8 slides)</h1>
{table(['#', 'Slide', 'Content', 'From'], [
    ['1', 'The question', 'Night is our weakest condition (detector 88 % found on NightLPBD, reader 61.5 % on night crops vs 77.5 % clear). Does a CVPR 2025 enhancer fix it?', 'sections 1, 7'],
    ['2', 'HVI in one picture', 'Hue as an angle + collapse of dark colours; CIDNet 1.98 M parameters', 'Figure 1'],
    ['3', 'We checked the paper', 'Released weights = README to 0.1 dB; LOLv1 needs GT mean (28.2 vs 23.6 dB) and mixes two files; 24.11 dB weights lost', 'section 8 table'],
    ['4', 'Our run is solid', 'v1.1, all jobs ok, bit-identical CRNN, Kaggle = local', 'section 7, Figure 3'],
    ['5', 'Main result', 'Detection +1.3 mAP50 n.s., 0 extra plates; reading worse; end to end +4 / 60 n.s.; CLAHE = HVI', 'Figure 4'],
    ['6', 'Why it does not help', 'Night photos are not dark (17 / 71 below 60); misses are small / close-up; reader fails on digits and letters', 'Figures 6, 7'],
    ['7', 'Cost', '155 ms vs 6.5 ms per photo; FP16 2.6x faster', 'Figure 8'],
    ['8', 'Decision and next step', 'No enhancer in the pipeline; night augmentation and task-aware training next; HVI only for the operator\'s view', 'sections 16, 21'],
], [0.4, 2.0, 5.8, 1.8])}

<h1>19. What to explain: a 3-minute script</h1>
<p><b>(30 s) Problem.</b> "At night our pipeline loses plates: the detector finds 88 % of NightLPBD plates and the reader
gets 62 % of night plates right, against 78 % in daylight. The newest answer in the literature is low-light enhancement.
I studied HVI-CIDNet from CVPR 2025."</p>
<p><b>(40 s) The paper.</b> "It converts the photo into a new colour space where hue is an angle, so red does not jump from
one end of the scale to the other, and where the colour of very dark pixels is squeezed to zero, so their noise does not
get amplified. A small network of about 2 million parameters brightens in that space. It is best on the LOL benchmarks.
I checked those numbers with their weights: they match the README, but the LOLv1 score uses the ground truth's
brightness, and one weight file is lost."</p>
<p><b>(60 s) Our test.</b> "I put it, plus gamma and CLAHE, in front of our YOLOv8s detector and CRNN reader, trained on
dataset v1.1. On 71 night photos, the detector finds exactly the same 64 plates with and without it; mAP50 goes up 1.3
points, but that is inside the noise. The reader gets worse on enhanced crops. In a full end-to-end test on 60 night
photos, the best way to use it reads 4 more plates, which is not significant, and plain CLAHE does the same for a
fifteenth of the cost."</p>
<p><b>(30 s) Why.</b> "Our night photos are not actually dark: median brightness 85 of 255, only 17 of 71 below our
threshold. The plates we miss are small, very close, or in glare. Enhancement solves a problem we do not have."</p>
<p><b>(20 s) Decision.</b> "So we keep the pipeline without an enhancer, put the effort into night training data and
augmentation, and use HVI only to show the operator a clearer picture. This is the CLAHE and enhancement study we
promised in the proposal under RQ4."</p>
"""

S['c3'] = f"""
<h1>20. Questions the faculty may ask</h1>
<p class="q">Why did you not fine-tune HVI-CIDNet on our data?</p>
<p>It needs pairs of the same scene in dark and good light; we have none. Zero-shot use of the authors' real-world model
is how a team would actually deploy it. Fine-tuning it with the detector or reader loss is our proposed next step.</p>
<p class="q">The photo looks better. How can the reading get worse?</p>
<p>The CRNN learned what normal crops look like. The enhancer changes contrast, colour and stroke thickness (bright,
thin strokes; ঘ read as ক, ৭ as ১), which is a domain shift for the reader. Better for human eyes is not better for a
network trained on other pixels. On day crops the loss is significant ({pv(RD['hvi_gen/day']['p'])}).</p>
<p class="q">Are 71, 65 and 60 samples enough?</p>
<p>Enough to rule out a large effect, not enough to detect a small one. That is why we report paired tests and
intervals: the detector gain is between {BT['hvi_gen']['ci95'][0]:+.1f} and {BT['hvi_gen']['ci95'][1]:+.1f} points, and no extra plate is found.
To show a +4 / 60 end-to-end gain we would need two to three times more night photos.</p>
<p class="q">Why not score PSNR or SSIM on our photos like the paper?</p>
<p>They need a well-lit photo of the same scene, which does not exist for street photos. Our target is the plate text,
so plates found and plates read are the right measures.</p>
<p class="q">What is "GT mean" and why do you call it a problem?</p>
<p>Before scoring, the output is scaled so its average brightness equals the ground truth's. It hides any brightness
error, and a real system cannot do it because it needs the answer. Without it, the LOLv1 score falls from 28.2 to 23.6 dB.</p>
<p class="q">Maybe your gate threshold is wrong?</p>
<p>We swept it from 40 to "all photos": the whole-test change never exceeds +0.5 mAP50 points (n.s.), and it only appears
when hundreds of ordinary day photos are enhanced. Enhancing exactly the night-tagged photos changes the whole-test
AP50 by {A['night_tag_gate_all_AP50']['hvi_gen'] - A['gate_sweep_base']['all_AP50']:+.2f} points.</p>
<p class="q">Could FP16 or TensorRT make it fast enough?</p>
<p>FP16 is safe (no NaN) and {sp['640x640']['fp32']['ms_median'] / sp['640x640']['fp16']['ms_median']:.1f}x faster on our GPU; on a T4 that would be roughly 60 ms (not measured), still
about ten times the detector, for no accuracy gain.</p>
<p class="q">Did the CLAHE result surprise you?</p>
<p>CLAHE matches HVI on night detection (+1.7 n.s.) and keeps night reading unchanged, at 10 ms on the CPU. But on
all photos it doubles false boxes on plate-free photos ({BX['clahe']['plate_free_pairs']['a']} -> {BX['clahe']['plate_free_pairs']['b']},
{pv(BX['clahe']['plate_free_pairs']['p'])}), so even CLAHE should stay behind a gate.</p>
<p class="q">Is the experiment reproducible?</p>
<p>Yes: the same kit and seed produced a bit-identical CRNN in two Kaggle sessions four days apart, and our PC re-scored
every Kaggle number with the same weights (plates found identical, mAP50 within 0.04).</p>
"""

S['c4'] = f"""
<h1>21. Next steps (in order)</h1>
{table(['#', 'Step', 'Why', 'Cost'], [
    ['1', 'Task-aware reader: retrain the CRNN with 30 % of the training crops enhanced (CLAHE / HVI / gamma) as augmentation, then test on raw and enhanced night crops', 'tells whether the reading loss is only a domain shift', 'one Kaggle job, about 30 min'],
    ['2', 'Night-style augmentation for the detector (darken + noise + blur + glare) vs no augmentation; test on NightLPBD by source', 'the cheapest way to target our real night misses', 'one detector run, about 2 h'],
    ['3', 'Pool the NightLPBD validation photos (and new night photos) into a larger night test set', 'the current 60-71 photos cannot show a 5-point effect', 'data work'],
    ['4', 'Optional: fine-tune HVI-CIDNet with the detector or CRNN loss (frozen recogniser, enhancer trainable)', 'the only fair test of "enhancement for machines"', 'research-level, several days'],
    ['5', 'Demo: HVI-CIDNet FP16 as an on-demand "clear view" for the operator', 'useful for humans, harmless for the machine', 'small'],
], [0.4, 4.6, 3.0, 2.0])}

<h1>22. GitHub and Drive</h1>
<p>Everything is prepared in {M('After Train/github_upload/')} with the repository layout of
{M('github.com/Rokib-Hasan-Oli/CSE498R')}, and {M('UPLOAD_STEPS.md')} gives the copy command and the commit list (you make
the commits yourself). Upload paper #33 first: this folder's README and results tables already contain #33.</p>
{table(['Repository path', 'Content'], [
    [M('weights/paper35/MD.Rokib.Hasan.Oli/'), 'README (model card, results, how to reproduce), ' + M('best_det_v8s.pt') + ' (the detector of this run, 21.5 MiB), ' + M('paper35_results.zip') + ', ' + M('paper35_check_results.zip') + ', both comparison tables, this PDF. The CRNN is not uploaded again: it is the same file as ' + M('weights/paper24/.../best_crnn_fp16.pt')],
    [M('notebooks/MD.Rokib.Hasan.Oli/'), M('paper35_run.ipynb') + ' + log, ' + M('paper35_check_run.ipynb') + ' + log'],
    [M('src/MD.Rokib.Hasan.Oli/paper35/'), 'START_HERE, the two guides, the kit (' + M('kaggle/') + ')'],
    [M('report/MD.Rokib.Hasan.Oli/paper35/'), 'the paper notes and ' + M('source/') + ' (local check, analysis, charts, report builder, their outputs)'],
    [M('results/') + ', ' + M('README.md'), 'team table with the #35 rows, tracker row for #35'],
], [3.4, 6.6])}
<p>Team Drive {M('498R / weights / paper35 /')}: {M('After Train/for_drive/paper35_outputs.zip')} (151 MiB, with all weights,
TensorRT engine and ONNX) and {M('paper35_check_outputs.zip')}. Do not upload {M('results.zip')} (532 MiB) or
{M('results-original.zip')} (it contains the LOL images).</p>
"""

S['c5'] = f"""
<h1>Appendix: where every number comes from</h1>
{table(['Number', 'Source'], [
    ['paper numbers', 'the paper PDF (Papers/6_6 Added papers (new5)/35_2025_HVI Color Space for Low-light Image Enhancement.pdf), pages given in the text'],
    ['training, lowlight, speed (Kaggle)', M('After Train/results.zip') + ' → ' + M('runs/*/results.json') + ', ' + M('results.csv') + ', ' + M('logs/') + '; ' + M('paper-35.log')],
    ['LOL check', M('After Train/results-original.zip') + ' → ' + M('runs/lolv1|lolv2_real/results.json') + '; ' + M('paper-35-original.log')],
    ['plate-by-plate re-scoring, crops, end to end, timings', M('report_src/local_check.py') + ' → ' + M('local_check.json') + ', ' + M('det_preds.jsonl') + ', ' + M('det_per_plate.csv') + ', ' + M('ocr_per_crop.csv') + ', ' + M('e2e_per_photo.csv') + ' (RTX 5060 Ti, torch 2.13 cu130, Ultralytics 8.4.170, dataset v1.1)'],
    ['val\'s own per-photo boxes for the bootstrap', M('report_src/local_valstats.py') + ' → ' + M('det_val_stats.jsonl')],
    ['paired tests, bootstrap, gate sweep, error parts', M('report_src/analysis.py') + ' → ' + M('analysis.json')],
    ['how HVI and CIDNet work', 'HVI-CIDNet code at commit eb43d7d: net/HVI_transform.py, net/CIDNet.py, net/LCA.py, eval.py, measure.py, train.py'],
    ['charts', M('report_src/charts.py')],
], [3.6, 6.4])}
<pre>@inproceedings{{yan2025hvi,
  title     = {{{{HVI}}: A New Color Space for Low-light Image Enhancement}},
  author    = {{Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Pang, Guansong
               and Shi, Kangbiao and Wu, Peng and Dong, Wei and Sun, Jinqiu
               and Zhang, Yanning}},
  booktitle = {{Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)}},
  year      = {{2025}}
}}</pre>
"""

FLOWS = [['cover'], ['a1', 'a2', 'a3'], ['b1'], ['b2'], ['b3'], ['b4'], ['b5'], ['b6', 'b7'], ['b8'], ['b9'], ['b10'],
         ['c1'], ['c2'], ['c3'], ['c4'], ['c5']]

if __name__ == '__main__':
    n = write_pdf(S, FLOWS, OUT, 'Paper #35 study report · dataset v1.1 run · 498R BanglaLPR',
                  {'title': 'Paper #35 study report (dataset v1.1): HVI, a new colour space for low-light image enhancement',
                   'author': 'Prepared for MD. Rokib Hasan Oli',
                   'subject': 'Paper study, v1.1 Kaggle results, LOL check, low-light enhancement before plate detection and reading, faculty guide'})
    print('pages', n, OUT)
