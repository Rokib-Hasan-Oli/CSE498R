"""Paper #33 study report (dataset v1.1 run) -> PDF (PyMuPDF Story, A4, one column).

Sources: the paper (Sapkota et al., arXiv 2509.25164 v5, 16 Mar 2026; page numbers = PDF pages); the v1.1 Kaggle run
(`After Train/results.zip`, `paper-33.log`, 25-26 Oct 2026) and the original-paper check (`results original.zip`,
`paper-33-original.log`) via analysis.json; the local check (local_check.json: re-scoring plate by plate, timings,
optimiser step) on the RTX 5060 Ti; the Ultralytics 8.4.170 source (trainer.build_optimizer, optim/muon.py,
utils/loss.E2ELoss, utils/tal.py, engine/exporter.py); the 498R proposal (RQ1-RQ4, O1-O5, G1-G6).
Run order: local_check.py -> analysis.py -> charts.py -> build_report.py.
"""
import json
import os
import sys

from part_a import part_a
from report_head import M, f1, f2, fig, pv, table, ul, write_pdf

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(HERE)
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PAPER_DIR, 'Paper33_Study_Report.pdf')
A = json.load(open(os.path.join(HERE, 'analysis.json'), encoding='utf-8'))
T, HD, BH = A['train'], A['heads'], A['by_head']
PT, PR, PN, PF, PS, FA = (A['paired_test'], A['paired_rain'], A['paired_night'], A['paired_fog'], A['paired_small'],
                          A['paired_false_alarm'])
LS, OS, CO, LN, SE = A['local']['speed'], A['local']['optimiser_step'], A['coco'], A['local_no_ridoykj'], A['same_epoch_val']
LM = A['local']['models']
CNT, MISS, Y11 = A['counts'], A['misses'], A['y11n_speed']
MODELS = ['y11n', 'y26n_sgd', 'y26n', 'y26n_p2', 'y26s']
NM = {m: T[m]['name'] for m in MODELS}
jobs = {j['job']: j for j in A['run']['jobs']}
cjobs = {j['job']: j for j in A['coco_jobs']}
RNS = 'y26n/nms vs y26n_sgd/nms'                     # the optimiser pair
ARC = 'y26n_sgd/nms vs y11n/nms'                     # the architecture pair


def cpu_total(k):
    return LS[k]['cpu_onnx']['model_ms_4_threads'] + LS[k]['cpu_onnx']['post_ms_median']


def pair_row(label, d, plates=True):
    return [label, f"{f2(d['a_%'])} vs {f2(d['b_%'])}", f"{d['only_a']} vs {d['only_b']}", pv(d['p']),
            f"{d['ci95_pp'][0]:+.1f} to {d['ci95_pp'][1]:+.1f}"]


# the text below quotes these numbers; stop if the data ever disagrees with it
R = A['run']
assert R['dataset_version'] == '1.1' and len(R['jobs']) == 9 and all(j['code'] == 0 for j in R['jobs'])
assert all(j['code'] == 0 for j in A['coco_jobs']) and len(A['coco_jobs']) == 2
assert T['y26n']['epochs_run'] == 50 and T['y26n']['time_cap_hit'] and all(T[m]['epochs_run'] == 60 for m in MODELS if m != 'y26n')
assert all(T[m]['nan_values'] == 0 for m in MODELS)
assert all(v['kaggle_found'] == v['local_found'] for v in A['reproduction'].values())
assert all(abs(A['reproduction'][m]['kaggle_mAP50-95'] - A['reproduction'][m]['local_mAP50-95']) < 0.1 for m in MODELS)
assert CO['yolo26n/one2many_nms']['AP'] == 40.81 and CO['yolo26n/one2one_e2e']['AP'] == 40.06
assert CO['yolo26s/one2many_nms']['AP'] == 48.62 and CO['yolo26m/one2one_e2e']['AP'] == 52.5
assert all(PT[k]['p'] > 0.05 for k in list(PT)[:7]) and PT['y26n/e2e vs y26n/nms']['p'] < 0.01
assert PT['y26n/e2e vs y26n/nms']['only_a'] == 1 and PT['y26n/e2e vs y26n/nms']['only_b'] == 11
assert PR[RNS]['only_a'] == 232 and PR[RNS]['only_b'] == 16 and PR[RNS]['p'] < 1e-40
assert PR[ARC]['p'] > 0.05 and PF[ARC]['p'] < 0.001 and PF[RNS]['p'] > 0.5
assert BH['y26n/nms']['found_rain'] == 82.0 and BH['y26n_sgd/nms']['found_rain'] == 60.63
assert MISS['missed_by_all'] == 5 and CNT['test_plates'] == 1011 and CNT['small_plates'] == 24
assert OS['MuSGD']['step_ms_median'] > 10 * OS['SGD']['step_ms_median']
assert LS['y26n_p2/nms']['cpu_onnx']['post_ms_median'] > 50 and LS['y26n_p2/e2e']['cpu_onnx']['post_ms_median'] < 1
assert LS['y26n/e2e']['cpu_onnx']['model_ms_4_threads'] < LS['y11n/nms']['cpu_onnx']['model_ms_4_threads']
assert all(max(LN[f'{m}/nms']['mAP50-95'] for m in MODELS) - min(LN[f'{m}/nms']['mAP50-95'] for m in MODELS) < 0.5 for _ in [0])
assert all(PS[k]['p'] > 0.1 for k in PS) and PN['y26n_p2/nms vs y26n/nms']['p'] > 0.05

S = part_a(A)
p26 = lambda m, h='nms': LM[f'{m}/{h}']
speed_ratio_model = LS['y11n/nms']['cpu_onnx']['model_ms_4_threads'] / LS['y26n/e2e']['cpu_onnx']['model_ms_4_threads']
speed_ratio_total = cpu_total('y11n/nms') / cpu_total('y26n/e2e')
rain_gain = BH['y26n/nms']['found_rain'] - BH['y26n_sgd/nms']['found_rain']
musgd_rain = [BH[f'{m}/nms']['found_rain'] for m in ('y26n', 'y26n_p2', 'y26s')]
sgd_rain = [BH[f'{m}/nms']['found_rain'] for m in ('y11n', 'y26n_sgd')]
found_rng = (min(T[m]['found_%'] for m in MODELS), max(T[m]['found_%'] for m in MODELS))
LNr = (min(LN[f'{m}/nms']['mAP50-95'] for m in MODELS), max(LN[f'{m}/nms']['mAP50-95'] for m in MODELS))
int8 = {m: HD[m]['e2e']['tensorrt_fp16']['mAP50'] - HD[m]['e2e']['tensorrt_int8']['mAP50'] for m in HD}
dup_e2e = [BH[f'{m}/e2e']['dup_boxes'] for m in ('y26n', 'y26n_sgd', 'y26n_p2', 'y26s')]
dup_nms = [BH[f'{m}/nms']['dup_boxes'] for m in ('y26n', 'y26n_sgd', 'y26n_p2', 'y26s')]
BEST_EP = ', '.join(f"{NM[m]} {T[m]['best_epoch']}" for m in MODELS)
LR50, LR60 = A['curves']['y26n']['lr_pg0'][-1] / 3, A['curves']['y26n_sgd']['lr_pg0'][-1]   # MuSGD pg0 = 3x head group

# ======================================================================================== cover + one-page summary
S['cover'] = f"""
<p class="part">CSE 498R · BANGLA LICENSE PLATE RECOGNITION · PAPER #33 STUDY REPORT · DATASET v1.1 RUN</p>
<h1 style="font-size:18pt; margin-bottom:3pt">YOLO26: Key Architectural Enhancements and Performance Benchmarking
for Real-Time Object Detection</h1>
<p style="font-family:H; font-size:11.5pt; color:#1c5cab; margin-bottom:8pt">The paper, our YOLO26 vs YOLO11 run on dataset
v1.1, the check of the paper's own COCO numbers, what NMS-free, MuSGD, the P2 head and INT8 really do on Bangla plates,
and what to show and say to the faculty</p>
<p class="small">Prepared for MD. Rokib Hasan Oli (paper reader) · Supervisor: Dr. Md Mansur ul Mulk · 26 October 2026<br>
Paper: R. Sapkota, R. H. Cheppally, A. Sharda and M. Karkee, arXiv 2509.25164 v5 (16 Mar 2026), a review (not
peer-reviewed), 15 pages. Page numbers are PDF pages. Code: Ultralytics (AGPL-3.0), YOLO26 in {M('ultralytics==8.4.170')}.<br>
Our runs: (1) {M('paper33_kaggle.ipynb')}, Tesla T4 x2 on Kaggle, 25-26 Oct 2026, dataset <b>v1.1</b> (confirmed in
{M('environment.json')}), {R['total_h']:.2f} h; (2) {M('paper33_original_check.ipynb')} on COCO val2017, 26 Oct,
{max(j['minutes'] for j in A['coco_jobs']):.0f} min. Extra checks on our own PC the same day (RTX 5060 Ti, Core Ultra 7 265K).</p>

<h2>The whole report on one page</h2>
<div class="box">
<p><b>The paper in three lines.</b> A review of Ultralytics YOLO26 (Sept 2025): no DFL, a second NMS-free head, small-object
label assignment (STAL), progressive loss balancing and the MuSGD optimiser, for simpler and faster deployment on edge
devices. All its numbers are Ultralytics' COCO tables: YOLO26-N 40.9 mAP50-95 (40.1 NMS-free), 1.7 ms on a T4 with
TensorRT, 38.9 ms on a CPU (p.8). The promised Jetson benchmarks and any number for INT8, STAL or MuSGD are missing.</p>
</div>
<div class="good">
<p><b>Is the v1.1 training OK? Yes, with one flaw that is now fixed in the kit.</b> All 9 jobs ended with {M('code=0')},
dataset v1.1 is confirmed, no NaN, and our local re-scoring of all 8 model/head pairs gives the Kaggle plate counts to
the last plate. <b>The flaw:</b> YOLO26-N with MuSGD hit its 2.3 h time cap and stopped at <b>epoch 50 of 60</b>, so its
learning rate never decayed fully, the last 10 epochs without mosaic never ran, and its NMS-free head got less training
(section 7). Its numbers are a lower bound; every conclusion below holds with or without it. The original-paper check is
clean: the released weights reproduce the paper's COCO accuracy to 0.1 (YOLO26-N {f2(CO['yolo26n/one2many_nms']['AP'])} /
{f2(CO['yolo26n/one2one_e2e']['AP'])} vs 40.9 / 40.1).</p>
</div>
<div class="box">
<p><b>Six findings on our plates.</b> (1) <b>All five detectors tie on the normal test</b>: {f1(found_rng[0])}-{f1(found_rng[1])} %
of {CNT['test_plates']:,} plates found, no paired difference is significant (YOLO26-N vs YOLO11-N: {PT['y26n/nms vs y11n/nms']['only_a']}
vs {PT['y26n/nms vs y11n/nms']['only_b']} plates). Plate detection is saturated; reading is our bottleneck. (2) <b>NMS-free
works</b> on fully trained models (2-3 plates lost of 1,011, n.s.) and removes the CPU's NMS ({f1(LS['y26n/nms']['cpu_onnx']['post_ms_median'])} ms for
YOLO26-N, {f1(LS['y26n_p2/nms']['cpu_onnx']['post_ms_median'])} ms for the P2 model, vs {f2(LS['y26n/e2e']['cpu_onnx']['post_ms_median'])} ms),
but it puts a second box on {min(dup_e2e)}-{max(dup_e2e)} plates. (3) <b>The surprise: MuSGD-trained models are far more robust
to synthetic rain</b>: {f1(min(musgd_rain))}-{f1(max(musgd_rain))} % of plates found vs {f1(min(sgd_rain))}-{f1(max(sgd_rain))} % with SGD;
same YOLO26-N architecture: {PR[RNS]['only_a']} vs {PR[RNS]['only_b']} plates ({pv(PR[RNS]['p'])}). (4) The YOLO26
<b>architecture</b> helps synthetic fog (+{f1(PF[ARC]['diff_pp'])} points, {pv(PF[ARC]['p'])}), not rain. (5) <b>INT8 keeps the
accuracy</b> (NMS-free heads lose at most {f2(max(int8.values()))} mAP50 points) but is not faster at batch 1 on a T4.
(6) <b>The "43 % faster CPU" claim</b> is mostly the removed NMS: the network alone is {speed_ratio_model:.2f}x faster than
YOLO11-N on our CPU, network + post-processing {speed_ratio_total:.2f}x.</p>
<p><b>What to tell the faculty in one sentence.</b> "On our duplicate-free plate benchmark YOLO26 is not more accurate than
YOLO11 (both find about 98 % of plates), but its NMS-free head makes the CPU pipeline simpler and faster, INT8 keeps its
accuracy, and the MuSGD optimiser it ships with made our detectors find about 20 % more plates in synthetic rain, which
we will now test with more seeds and real rain photos."</p>
</div>
<p class="small"><b>Parts:</b> A. Understand the paper (sections 1-5, read before you explain it) · B. Our v1.1 experiment
(6-16, read before you quote a number) · C. Use it (17-23: research use, benefits, slides, script, Q&amp;A, next steps,
GitHub; read before the faculty meeting).</p>
"""

# ======================================================================================== part B
S['b1'] = f"""
<p class="part">PART B · OUR EXPERIMENT ON DATASET v1.1</p>
<h1>6. What we ran</h1>
<p>The notebook {M('paper33_kaggle.ipynb')} (built by {M('docs/_kit')} like every team kit) trains and tests on the <b>498R
BanglaLPR All-in-One v1.1</b> benchmark: all public Bangladeshi plate sets merged, near-duplicates kept in one group,
grouped 70/15/15 split (seed 42), validation and test with human boxes only. Detection: 8,574 training photos (4,067 human
+ 4,507 auto-labelled boxes), 896 validation, <b>1,153 test photos with 1,011 plates</b> ({CNT['no_plate_photos']} plate-free photos for false
alarms). Every run: 640 px, 60 epochs, batch 16, seed 42, deterministic, AMP, Ultralytics 8.4.170, PyTorch 2.11 (CUDA 12.8).</p>
{table(['Job', 'GPU', 'What it answers', 'Minutes'], [
    [M('y11n'), '1', 'YOLO11-N, SGD lr 0.01: the team baseline (same recipe as every detector kit)', f"{jobs['y11n']['minutes']:.0f}"],
    [M('y26n'), '0', 'YOLO26-N, MuSGD lr 0.01: "the YOLO26 recipe"', f"{jobs['y26n']['minutes']:.0f}"],
    [M('y26n_sgd'), '1', 'YOLO26-N with SGD: is it the architecture or the optimiser?', f"{jobs['y26n_sgd']['minutes']:.0f}"],
    [M('y26n_p2'), '1', 'YOLO26-N + P2 head (stride 4, for small plates), MuSGD; COCO backbone transferred (360 of 902 tensors)', f"{jobs['y26n_p2']['minutes']:.0f}"],
    [M('y26s'), '0', 'YOLO26-S, MuSGD: does a 4x larger model help?', f"{jobs['y26s']['minutes']:.0f}"],
    [M('heads_y26n / _y26s / _y26n_p2'), '0', 'the two heads of the same weights: mAP, plates, false alarms, PyTorch FP16, TensorRT FP16 + INT8 (with test mAP), CPU ONNX', f"{jobs['heads_y26n']['minutes']:.0f} / {jobs['heads_y26s']['minutes']:.0f} / {jobs['heads_y26n_p2']['minutes']:.0f}"],
    [M('speed_y11n'), '1', 'YOLO11-N: FP32 / FP16 / TensorRT FP16 + INT8 / CPU ONNX', f"{jobs['speed_y11n']['minutes']:.0f}"],
    [M('coco_gpu0 / coco_gpu1'), '0 / 1', 'original-paper check: released YOLO26-N/S/M and YOLO11-N on COCO val2017 (5,000 photos), both heads, pycocotools mAP, T4 TensorRT FP16 and CPU ONNX', f"{cjobs['coco_gpu0']['minutes']:.0f} / {cjobs['coco_gpu1']['minutes']:.0f}"],
], [2.6, 0.6, 5.8, 1.0])}
<p><b>Local checks (our PC, 26 Oct).</b> {M('report_src/local_check.py')} re-scored every trained model (both heads of each YOLO26)
plate by plate on the test split and on its synthetic rain and fog copies, re-ran mAP with and without the ridoykj
source, timed GPU and CPU one job at a time, and timed one MuSGD vs SGD optimiser step. {M('analysis.py')} turns that into
paired tests: exact McNemar on the plates where exactly one model finds the plate, and a 95 % bootstrap interval
(photos resampled) for the difference in plates found.</p>
"""

S['b2'] = f"""
<h1>7. Is the training OK?</h1>
{table(['Check', 'Result', 'OK?'], [
    ['All jobs finished', f"9 of 9 main jobs and 2 of 2 check jobs {M('code=0')}; main run {R['total_h']:.2f} h of the 12 h limit", 'yes'],
    ['Right dataset', f"{M('dataset_version: 1.1')} in environment.json; 8,574 / 896 / 1,153 photos as in the manifest", 'yes'],
    ['Epochs', f"YOLO11-N, YOLO26-N (SGD), YOLO26-N-P2, YOLO26-S: 60 of 60. <b>YOLO26-N (MuSGD): 50 of 60</b> (time cap 2.3 h)", '<b>no</b> (one run)'],
    ['NaN / divergence', 'no NaN in any results.csv; validation mAP50-95 climbs smoothly to 60 for every run (Figure 2)', 'yes'],
    ['Best epochs', BEST_EP, 'yes (late, no early overfitting)'],
    ['Validation mAP50-95', ', '.join(f"{f1(T[m]['val']['mAP50-95'])}" for m in MODELS) + ' (same order as above)', 'yes (all within 0.6)'],
    ['Repeatable', 'local re-scoring with the run\'s best.pt on dataset v1.1 gives the same plates found for all 8 model/head pairs (e.g. YOLO26-N 97.82, rain 82.00) and mAP50-95 within 0.06', 'yes'],
    ['Timing hygiene', f"{M('speed_y11n')} ran at the same time as {M('heads_y26n')} on the 4-core Kaggle CPU, so those two jobs' batch-1 timings are inflated; the other heads jobs ran alone", 'partly'],
    ['Report formatting', 'the notebook printed the TensorRT INT8 mAP50 row as a fraction (0.97 = 97 %)', 'cosmetic'],
], [2.0, 6.6, 1.4])}
<div class="warn"><p><b>What the time cap did to YOLO26-N (MuSGD).</b> Each training job has a {M('--max-hours')} cap so the notebook
fits Kaggle's 12 h. {M('y26n')}'s cap was 2.3 h, and its epochs took {f1(T['y26n']['s_per_epoch'])} s (YOLO26-N with SGD:
{f1(T['y26n_sgd']['s_per_epoch'])} s), so it stopped after epoch 50. Two reasons for the slow epochs: it shared the 4-core CPU
(data loading) with {M('y11n')} and then {M('y26n_sgd')}, and a MuSGD step is slower: {f2(OS['MuSGD']['step_ms_median'])} ms vs
{f2(OS['SGD']['step_ms_median'])} ms for SGD on our GPU (5 Newton-Schulz iterations per weight matrix), which adds about
{f1(OS['extra_s_per_epoch'])} s per epoch. Consequences: (1) its base learning rate ended at {LR50:.4f}, {LR50 / LR60:.0f}x the planned final
{LR60:.5f}; (2) the last 10
epochs without mosaic ({M('close_mosaic=10')}) never ran; (3) ProgLoss left the one-to-many weight at 0.21 instead of 0.10, so the
NMS-free head got less of the loss, which fits its weaker NMS-free result (section 11). Its scores are therefore a lower
bound for "YOLO26-N + MuSGD", and its comparison with {M('y26n_sgd')} is 50 vs 60 epochs; section 12 compares them at equal
epochs instead.</p>
<p><b>Fixed in the kit (26 Oct).</b> Caps are now 3.2 h (y26n), 3.6 h (y26s), 2.5 h (y11n), 2.7 h (y26n_sgd), 3.0 h (y26n_p2);
the speed job runs after the heads jobs on one GPU, so no timing shares the CPU; INT8 mAP is printed in %; the COCO check
runs its two jobs one after the other; its GitHub zip leaves out the 10-15 MB COCO prediction files. The notebooks were
rebuilt; the full run is planned at 8 h. Re-running only YOLO26-N (MuSGD) + its heads job takes about 3 h.</p></div>
{fig('fig_curves.png', 480, 'Figure 2. Validation mAP50-95 per epoch (896 photos). Left: the optimiser question; MuSGD is ahead in the first 25 epochs, then the curves meet. Right: the YOLO26 variants. The dotted line marks where YOLO26-N (MuSGD) stopped.')}
"""

coco_rows = []
for m, lab in (('yolo26n', 'YOLO26-N'), ('yolo26s', 'YOLO26-S'), ('yolo26m', 'YOLO26-M'), ('yolo11n', 'YOLO11-N')):
    pa = A['coco_paper'][m]
    nms, e2e = CO[f'{m}/one2many_nms'], CO.get(f'{m}/one2one_e2e')
    coco_rows.append([lab, f"{pa[0]} / {pa[1] if pa[1] else '–'}", f"<b>{f2(nms['AP'])}</b> / {('<b>' + f2(e2e['AP']) + '</b>') if e2e else '–'}",
                      f"{f2(e2e['AP'] - nms['AP']) if e2e else '–'}", f"{f1(nms['AP_small'])}",
                      f"{pa[2]}", f"{f2((e2e or nms)['trt_inf_ms'])}", f"{f2((e2e or nms)['trt_e2e_ms'])}"])
S['b3'] = f"""
<h1>8. The original-paper check: COCO val2017</h1>
<p>The paper's numbers are Ultralytics' numbers for the released weights, so the check runs those weights on the same 5,000
COCO val2017 photos with the official pycocotools scorer, once per head, plus a TensorRT FP16 engine per head on a T4.</p>
{table(['Model', 'Paper mAP50-95 NMS / e2e', 'Ours NMS / e2e', 'e2e - NMS', 'Ours AP small', 'Paper T4 ms', 'Ours T4 TRT ms (model)', 'Ours T4 TRT ms (+pre/post)'],
       coco_rows, [1.6, 1.5, 1.4, 0.8, 0.9, 0.9, 1.4, 1.5], num=(2, 3, 4, 5, 6, 7))}
<p class="cap">Paper: p.8, Table 2; YOLO11-N from Ultralytics' documentation (not in the paper). "ms" columns of ours are the
NMS-free engine for YOLO26 and the NMS engine for YOLO11, batch 1, median of 200 photos.</p>
<div class="good"><p><b>Accuracy: reproduced.</b> All seven mAP numbers are within 0.1 of the paper (largest gap 0.09), and the NMS-free
penalty is the same: -0.75 / -0.84 / -0.59 points (paper -0.8 / -0.8 / -0.6). The paper's accuracy table is honest.</p></div>
<div class="box"><p><b>Speed: same order, not the same numbers.</b> YOLO26-S and -M are within 0.25 ms of the paper; YOLO26-N is 0.8 ms
slower ({f2(CO['yolo26n/one2one_e2e']['trt_inf_ms'])} vs 1.7 ms) and YOLO11-N 0.7 ms ({f2(CO['yolo11n/one2many_nms']['trt_inf_ms'])} vs 1.5).
For the smallest models a fixed per-call cost (copying the image to the GPU, synchronising, TensorRT 11.4 instead of 10,
Kaggle's virtual machine) dominates, and the two jobs ran at the same time. Note that <b>YOLO11-N is faster than YOLO26-N on
the T4</b> in both our test and Ultralytics' own table (1.5 vs 1.7 ms): YOLO26's speed advantage is a CPU story, not a GPU one.
The Kaggle CPU numbers ({f1(CO['yolo26n/one2one_e2e']['cpu_e2e_ms'])} ms for YOLO26-N vs 38.9) come from 4 shared virtual cores with
another job running and cannot check the paper's CPU column; section 15 times the CPU properly on our PC.</p></div>
"""

main_rows = []
for m in MODELS:
    t = T[m]
    main_rows.append([NM[m] + (' *' if m == 'y26n' else ''), f"{t['params_M']:.2f} / {t['GFLOPs']:.1f}", f"{t['epochs_run']} ({t['best_epoch']})",
                      f"{t['train_minutes']:.0f}", f1(t['val']['mAP50-95']), f2(t['test']['mAP50']), f2(t['test']['mAP50-95']),
                      f2(LN[f'{m}/nms']['mAP50-95']), f2(t['found_%']), f"{BH[f'{m}/nms']['false_alarm_photos']} / 160"])
S['b4'] = f"""
<h1>9. Results next to the paper</h1>
{table(['Model', 'Params M / GFLOPs', 'Epochs (best)', 'Train min', 'Val mAP50-95', 'Test mAP50', 'Test mAP50-95', 'mAP50-95 w/o ridoykj', 'Plates found %', 'False alarms'],
       main_rows, [2.0, 1.2, 0.9, 0.7, 0.9, 0.9, 1.0, 1.1, 1.0, 1.0], num=(1, 2, 3, 4, 5, 6, 7, 8, 9))}
<p class="cap">Test = 1,153 photos / 1,011 plates; NMS head (Ultralytics' default). "w/o ridoykj" re-scored locally on the 879 test
photos without that source. False alarms = plate-free photos with any box at confidence 0.25. * stopped at epoch 50 of 60.
Params and GFLOPs as logged at the start of training (unfused); the fused YOLO26-N that is validated and exported has 2.38 M
parameters and 5.3 GFLOPs, close to the paper's 2.4 M / 5.4.</p>
<p><b>The paper's numbers are on COCO (80 classes, many tiny objects), ours on one class of mostly large plates</b>, so the
absolute values cannot be compared (our 45 mAP50-95 is not "better" than COCO's 40.9). What can be compared is every
<i>claim</i>:</p>
{table(['Paper claim', 'Paper (page)', 'Ours on plates', 'Verdict'], [
    ['NMS-free head costs little accuracy', '-0.8 mAP50-95 (p.8)', f"{HD['y26n']['e2e_minus_nms_mAP50-95_pp']:+.2f} / {HD['y26s']['e2e_minus_nms_mAP50-95_pp']:+.2f} / {HD['y26n_p2']['e2e_minus_nms_mAP50-95_pp']:+.2f} mAP50-95 (N / S / P2); 2-3 plates on fully trained models", 'holds'],
    ['INT8 / FP16: "minimal impact on accuracy"', 'no number (p.10)', f"NMS-free heads: -{f2(int8['y26n'])} / -{f2(int8['y26s'])} / -{f2(int8['y26n_p2'])} mAP50 vs TensorRT FP16", 'holds'],
    ['"up to 43 % faster CPU inference" (N)', 'phrase (p.7, p.11)', f"network {speed_ratio_model:.2f}x, network + post-processing {speed_ratio_total:.2f}x faster than YOLO11-N", 'holds only with NMS counted'],
    ['STAL / ProgLoss: better small objects', 'no number (p.7)', f"{CNT['small_plates']} small plates: YOLO26-N {f1(BH['y26n/nms']['found_small'])} % = YOLO11-N {f1(BH['y11n/nms']['found_small'])} %", 'not testable (too few)'],
    ['MuSGD: competitive accuracy in fewer epochs', 'no number (p.7)', f"+1.2 / +1.5 val mAP50-95 at epochs 10 / 20, equal from epoch 30; same test plates", 'weakly holds'],
    ['"faster inference speeds than YOLOv11"', 'p.7', f"GPU: T4 TensorRT {f2(CO['yolo26n/one2one_e2e']['trt_inf_ms'])} vs {f2(CO['yolo11n/one2many_nms']['trt_inf_ms'])} ms (COCO), PyTorch FP16 {f1(LS['y26n/e2e']['gpu_fp16']['ms_median'])} vs {f1(LS['y11n/nms']['gpu_fp16']['ms_median'])} ms; CPU: see the row above", 'CPU yes, GPU no'],
], [2.6, 1.6, 4.4, 1.4])}
"""

S['b5'] = f"""
<h1>10. Is YOLO26 better than YOLO11 on our plates?</h1>
{fig('fig_models.png', 480, 'Figure 3. The five detectors on the clean test split. Blue = MuSGD, orange = SGD. The axes are zoomed: the whole spread is about 1 point.')}
{table(['Comparison (same 1,011 plates)', 'Found % (a vs b)', 'Plates only a / only b', 'McNemar', '95 % interval, points'], [
    pair_row('YOLO26-N MuSGD* vs YOLO11-N', PT['y26n/nms vs y11n/nms']),
    pair_row('YOLO26-N SGD vs YOLO11-N (architecture)', PT[ARC]),
    pair_row('YOLO26-N MuSGD* vs SGD (optimiser)', PT[RNS]),
    pair_row('YOLO26-N-P2 vs YOLO26-N (P2 head)', PT['y26n_p2/nms vs y26n/nms']),
    pair_row('YOLO26-S vs YOLO26-N (size)', PT['y26s/nms vs y26n/nms']),
    pair_row('YOLO26-N-P2 NMS-free vs YOLO11-N + NMS', PT['y26n_p2/e2e vs y11n/nms']),
], [3.6, 1.6, 1.6, 1.1, 1.6])}
<p><b>No comparison is significant.</b> The detectors differ by a handful of plates out of 1,011, in both directions. Only
{MISS['missed_by_all']} plates are missed by all five models (2 NightLPBD, 2 LPDB-A, 1 KUET) and {MISS['missed_by_any']} by at least one: the misses
are mostly different plates, which is what noise between similar models looks like. False alarms are equal too (7-9 of 160
plate-free photos; every pair n.s.).</p>
<p><b>Why does mAP50-95 rank them (YOLO26-S 46.4 ... YOLO11-N 44.7)?</b> Because of one source. The ridoykj labels sit exactly on
the plate edge while the other sources leave a margin (found in the paper #6 report), so every model loses most of its
mAP50-95 on those 274 photos. Without ridoykj all five models score {f1(LNr[0])}-{f1(LNr[1])}, and YOLO11-N is the
highest. The test mAP50-95 ranking is a box-style effect, not a quality ranking.</p>
<div class="box"><p><b>Team context (results table, same split).</b> Every plate detector the team has trained on v1.1 finds 96.4-98.4 %
of the test plates: paper #2's YOLO12s 98.32 %, paper #3's YOLOv8n 98.12 %, our YOLO26-N-P2 98.42 %. Detection is solved
to within one or two plates in a hundred on this benchmark; the reader (paper #24: 88.6 % of plates exact) is where the
errors are. Choose the detector by speed, deployment and robustness, not by clean mAP.</p></div>
"""

head_rows = []
for m in ('y26n', 'y26n_sgd', 'y26n_p2', 'y26s'):
    n_, e_ = BH[f'{m}/nms'], BH[f'{m}/e2e']
    d = PT[f'{m}/e2e vs {m}/nms']
    mp = (f"{f1(HD[m]['nms']['test']['mAP50-95'])} / {f1(HD[m]['e2e']['test']['mAP50-95'])}" if m in HD else
          f"{f1(LM[f'{m}/nms']['val_test']['mAP50-95'])} / {f1(LM[f'{m}/e2e']['val_test']['mAP50-95'])}")
    head_rows.append([NM[m] + (' *' if m == 'y26n' else ''), mp, f"{f2(n_['found_test'])} / {f2(e_['found_test'])}",
                      f"{d['only_b']} vs {d['only_a']}, {pv(d['p'])}", f"{f1(n_['found_night'])} / {f1(e_['found_night'])}",
                      f"{f1(n_['found_multi'])} / {f1(e_['found_multi'])}", f"{n_['dup_boxes']} / {e_['dup_boxes']}",
                      f"{n_['false_alarm_photos']} / {e_['false_alarm_photos']}"])
S['b6'] = f"""
<h1>11. The NMS-free head vs the NMS head (same weights)</h1>
{table(['Model', 'mAP50-95 NMS / free', 'Found % NMS / free', 'Plates only NMS vs only free', 'Night found % (71)', 'Multi-plate photos (33 plates)', 'Double boxes', 'False alarms /160'],
       head_rows, [1.9, 1.2, 1.3, 1.6, 1.0, 1.2, 0.9, 0.9])}
<p class="cap">Each cell: NMS head / NMS-free head of the same best.pt. mAP from the Kaggle heads jobs (YOLO26-N SGD: local). Double
boxes = a second box with IoU &gt;= 0.5 on a plate that already has one, at confidence 0.25, over the 1,153 test photos.</p>
{ul([
    f"<b>Accuracy:</b> on the three fully trained models the NMS head finds 2-5 plates that the NMS-free head misses, the reverse happens 0-2 times (net 2-3 plates, all n.s.): the "
    f"paper's \"small cost\". On the stopped YOLO26-N it misses <b>11 more vs 1</b> ({pv(PT['y26n/e2e vs y26n/nms']['p'])}), and at night it "
    f"finds {f1(BH['y26n/e2e']['found_night'])} % vs {f1(BH['y11n/nms']['found_night'])} % for YOLO11-N (YOLO11-N finds {PN['y26n/e2e vs y11n/nms']['only_b']} night plates it misses, "
    f"the reverse happens {PN['y26n/e2e vs y11n/nms']['only_a']} times, {pv(PN['y26n/e2e vs y11n/nms']['p'])}). This is what an "
    f"under-trained one-to-one head looks like (section 7): ProgLoss gives that head most of the loss only in the last epochs.",
    f"<b>Double boxes:</b> without NMS nothing removes a second box on the same plate: {', '.join(map(str, dup_e2e))} double boxes "
    f"(N, N SGD, P2, S) vs {', '.join(map(str, dup_nms))} with NMS. For us this matters: each box becomes an OCR call. A one-line IoU merge "
    f"before the reader (or a higher confidence threshold) is needed.",
    "<b>Two plates in one photo</b> (15 photos, 33 plates) is where the notes predicted trouble for one-to-one heads; the NMS-free "
    "head finds the same or up to 6 points fewer of them, too few plates for a test.",
    f"<b>Speed:</b> on the T4 the NMS-free TensorRT engine is {f2(HD['y26s']['nms']['tensorrt_fp16']['ms'] - HD['y26s']['e2e']['tensorrt_fp16']['ms'])} ms (YOLO26-S) and "
    f"{f2(HD['y26n_p2']['nms']['tensorrt_fp16']['ms'] - HD['y26n_p2']['e2e']['tensorrt_fp16']['ms'])} ms (P2) faster per photo (both measured alone). On a CPU the gain is "
    f"large: NMS costs {f1(LS['y26n/nms']['cpu_onnx']['post_ms_median'])} ms for YOLO26-N and <b>{f1(LS['y26n_p2/nms']['cpu_onnx']['post_ms_median'])} ms for YOLO26-N-P2</b> "
    f"(34,000 candidate boxes instead of 8,400), the NMS-free output {f2(LS['y26n_p2/e2e']['cpu_onnx']['post_ms_median'])} ms (Figure 6).",
    "<b>Deployment:</b> the NMS-free ONNX file outputs the final 300 boxes ([1, 300, 6]); no IoU threshold to tune, no NMS code to port "
    "to a phone or microcontroller. That is the real benefit for the edge demo (paper #18).",
])}
"""

se_rows = [[f'epoch {e}', f1(SE[str(e)]['y26n']), f1(SE[str(e)]['y26n_sgd']), f"{SE[str(e)]['y26n'] - SE[str(e)]['y26n_sgd']:+.2f}",
            f1(SE[str(e)]['y11n'])] for e in (5, 10, 20, 30, 40, 50)]
S['b7'] = f"""
<h1>12. MuSGD vs SGD</h1>
<p><b>What "MuSGD" is in Ultralytics 8.4.170</b> (trainer.build_optimizer and optim/muon.py): every conv / linear weight gets the
normal SGD-momentum step at the learning rate <b>plus</b> a Muon step at 0.2 x the learning rate (its momentum orthogonalised by 5
Newton-Schulz iterations); biases and BatchNorm get plain SGD; and every group is split so the classification-head parameters
train at <b>3 x the learning rate</b>. So {M('y26n')} vs {M('y26n_sgd')} changes two things at once (the Muon term and the head
learning rate); we measure the recipe, not Muon alone.</p>
{table(['Validation mAP50-95 at', 'YOLO26-N MuSGD', 'YOLO26-N SGD', 'MuSGD - SGD', 'YOLO11-N SGD'], se_rows, [2.4, 1.6, 1.6, 1.4, 1.6], num=(1, 2, 3, 4))}
{ul([
    f"<b>Faster start, same end.</b> MuSGD is ahead by 0.9-1.5 points in the first 20 epochs and by 0.1-1 point later; at epoch 50 it "
    f"already has {f1(T['y26n']['val']['mAP50-95'])} (best.pt), what SGD reaches at epoch 55 ({f1(T['y26n_sgd']['val']['mAP50-95'])}). "
    "That is the paper's \"fewer epochs\" claim, in a weak form (one seed each; epoch-to-epoch noise is about 0.5 points).",
    f"<b>Same test result:</b> {PT[RNS]['only_a']} vs {PT[RNS]['only_b']} plates ({pv(PT[RNS]['p'])}), equal false alarms, equal fog ({PF[RNS]['only_a']} vs {PF[RNS]['only_b']}) and night ({PN[RNS]['only_a']} vs {PN[RNS]['only_b']}).",
    f"<b>Cost:</b> one optimiser step takes {f2(OS['MuSGD']['step_ms_median'])} ms instead of {f2(OS['SGD']['step_ms_median'])} ms on our RTX 5060 Ti "
    f"(8 parameter groups instead of 3), about {f1(OS['extra_s_per_epoch'])} s more per epoch of 536 steps; more on a T4. "
    "<b>But</b> on synthetic rain the MuSGD models are in a different class (next section).",
])}
"""

rain_rows = [pair_row(lab, PR[k]) for lab, k in (
    ('YOLO26-N MuSGD* vs YOLO26-N SGD (optimiser only)', RNS),
    ('YOLO26-N-P2 MuSGD vs YOLO26-N SGD', 'y26n_p2/nms vs y26n_sgd/nms'),
    ('YOLO26-S MuSGD vs YOLO26-N SGD', 'y26s/nms vs y26n_sgd/nms'),
    ('YOLO26-N MuSGD* vs YOLO11-N SGD', 'y26n/nms vs y11n/nms'),
    ('YOLO26-N SGD vs YOLO11-N SGD (architecture only)', ARC),
    ('YOLO26-N-P2 vs YOLO26-N (both MuSGD)', 'y26n_p2/nms vs y26n/nms'))]
S['b8'] = f"""
<h1>13. The surprise: MuSGD and synthetic rain</h1>
{fig('fig_conditions.png', 480, 'Figure 4. Plates found per condition (NMS head). Rain and fog are synthetic copies of all 1,153 test photos; night and small plates are fixed real subsets.')}
{table(['Rain copies, same 1,011 plates', 'Found % (a vs b)', 'Plates only a / only b', 'McNemar', '95 % interval, points'], rain_rows, [3.6, 1.6, 1.6, 1.1, 1.6])}
<p>The three MuSGD models find {f1(min(musgd_rain))}-{f1(max(musgd_rain))} % of the plates in the rain copies, the two SGD models
{f1(min(sgd_rain))}-{f1(max(sgd_rain))} %. With the <b>same architecture, data, augmentation and seed</b>, switching SGD to MuSGD gives
{PR[RNS]['only_a']} plates found only with MuSGD against {PR[RNS]['only_b']} found only with SGD (+{f1(rain_gain)} points, 95 % interval
{PR[RNS]['ci95_pp'][0]:+.1f} to {PR[RNS]['ci95_pp'][1]:+.1f}). The architecture alone does nothing for rain (YOLO26-N SGD vs YOLO11-N:
{PR[ARC]['only_a']} vs {PR[ARC]['only_b']}, {pv(PR[ARC]['p'])}) but helps fog: +{f1(PF[ARC]['diff_pp'])} points ({PF[ARC]['only_a']} vs {PF[ARC]['only_b']} plates,
{pv(PF[ARC]['p'])}), where the optimiser makes no difference. The effect also holds for mAP50 on rain (Kaggle: {f1(T['y26n']['by_condition']['rain_synth']['mAP50'])}
/ {f1(T['y26n_p2']['by_condition']['rain_synth']['mAP50'])} / {f1(T['y26s']['by_condition']['rain_synth']['mAP50'])} with MuSGD, {f1(T['y11n']['by_condition']['rain_synth']['mAP50'])}
/ {f1(T['y26n_sgd']['by_condition']['rain_synth']['mAP50'])} with SGD) and for the NMS-free heads.</p>
{fig('fig_rain_examples.png', 470, 'Figure 5. Three rain copies where YOLO26-N with MuSGD finds the plate and the same network trained with SGD finds nothing.')}
<div class="warn"><p><b>How strong is this?</b> Strong enough to follow up, not yet a claim for a paper. For: the pair differs only in the
optimiser; 232 vs 16 plates; three MuSGD models of different size and head all behave the same, two SGD models of different
architecture both do not. Against: (1) the rain is <b>synthetic</b> (streaks + darkening; our dataset has no real rain photos);
(2) <b>one seed per run</b>, and accuracy on a shifted test set varies much more between seeds than clean accuracy; (3) MuSGD
changes two things (Muon term and 3x head learning rate), so we do not know which one does it; (4) the MuSGD YOLO26-N
stopped at epoch 50 (but the P2 and S runs ran 60 epochs and show the same). Section 22 lists the experiment that settles it.</p></div>
"""

S['b9'] = f"""
<h1>14. Small plates, night and the P2 head (the STAL claim)</h1>
<p>The paper says STAL and ProgLoss give "a substantial accuracy boost" for small objects (p.7). Our test split cannot test that
well: only 5 plates are smaller than 32 x 32 pixels in the stored photo (COCO's "small"), and the fixed "small plates" subset
(every plate under 32 x 32 px after resizing to 640) has {CNT['small_photos']} photos with {CNT['small_plates']} plates. On those, YOLO26-N and
YOLO11-N both find {f1(BH['y26n/nms']['found_small'])} % (the same 23 plates), YOLO26-N-P2 all 24, YOLO26-S {f1(BH['y26s/nms']['found_small'])} %. None of
this is significant.</p>
{table(['Subset (plates)', 'YOLO11-N', 'YOLO26-N SGD', 'YOLO26-N MuSGD*', 'YOLO26-N-P2', 'YOLO26-S'], [
    [f"night ({CNT['night_plates']})"] + [f1(BH[f'{m}/nms']['found_night']) for m in MODELS],
    [f"small plates ({CNT['small_plates']})"] + [f1(BH[f'{m}/nms']['found_small']) for m in MODELS],
    ['NightLPBD source (60)'] + [f1(T[m]['found_by_source']['nightlpbd']) for m in MODELS],
    ['two or more plates in the photo (33)'] + [f1(BH[f'{m}/nms']['found_multi']) for m in MODELS],
], [2.6, 1.2, 1.3, 1.4, 1.3, 1.2], num=(1, 2, 3, 4, 5))}
<p><b>The P2 head is the most promising variant for hard plates.</b> It adds a stride-4 detection level and finds the most plates overall
({f2(T['y26n_p2']['found_%'])} %), at night ({f1(BH['y26n_p2/nms']['found_night'])} % vs {f1(BH['y26n/nms']['found_night'])} % for plain YOLO26-N, {PN['y26n_p2/nms vs y26n/nms']['only_a']} vs {PN['y26n_p2/nms vs y26n/nms']['only_b']} plates,
{pv(PN['y26n_p2/nms vs y26n/nms']['p'])}) and on multi-plate photos, but none of these gains is significant on our small subsets. It costs
1.6 ms more on the GPU, 8 ms more for the network on the CPU, and 80 ms of NMS on the CPU unless the NMS-free head is used.
YOLO11-N is the best of the plain models at night; the YOLO26 architecture does not help night photos.</p>
"""

int8_rows = []
for m in ('y26n', 'y26s', 'y26n_p2'):
    for h, lab in (('e2e', 'NMS-free'), ('nms', 'NMS')):
        x = HD[m][h]
        int8_rows.append([f"{NM[m]}{' *' if m == 'y26n' else ''}, {lab}", f"{f2(x['test']['mAP50'])}", f"{f2(x['tensorrt_fp16']['mAP50'])} / {f2(x['tensorrt_fp16']['mAP50-95'])}",
                          f"{f2(x['tensorrt_int8']['mAP50'])} / {f2(x['tensorrt_int8']['mAP50-95'])}",
                          f"{f2(x['tensorrt_fp16']['ms'])} / {f2(x['tensorrt_int8']['ms'])}", f"{f2(x['pytorch_fp16_ms'])}"])
int8_rows.append(['YOLO11-N, NMS', f2(T['y11n']['test']['mAP50']), '–', f"{f2(Y11['trt_int8_mAP50'])} / –",
                  f"{f2(Y11['trt_fp16_ms'])} / {f2(Y11['trt_int8_ms'])}", f2(Y11['fp16_ms'])])
S['b10'] = f"""
<h1>15. Speed and INT8</h1>
<h3>TensorRT on the Kaggle T4 (batch 1, pre + inference + post, median of 200 test photos)</h3>
{table(['Model, head', 'PyTorch mAP50', 'TRT FP16 mAP50 / 50-95', 'TRT INT8 mAP50 / 50-95', 'TRT ms FP16 / INT8', 'PyTorch FP16 ms'],
       int8_rows, [2.6, 1.1, 1.7, 1.7, 1.5, 1.2], num=(1, 2, 3, 4, 5))}
<p class="cap">YOLO26-N and YOLO11-N timings were taken while the other of the two jobs shared the CPU (section 7): read them as upper
bounds. INT8 calibration used 30 % of the validation split, never the test split.</p>
{ul([
    f"<b>INT8 keeps the accuracy.</b> NMS-free heads lose {f2(int8['y26n'])}, {f2(int8['y26s'])} and {f2(int8['y26n_p2'])} mAP50 points against their own TensorRT FP16 engine; "
    "NMS heads lose 0.1-0.8. Compare INT8 with TensorRT FP16, not with PyTorch: the engines use a fixed 640 x 640 square input "
    "while PyTorch validation uses rectangles, which alone moves mAP50 by up to 0.8 points.",
    "<b>INT8 is not faster here.</b> At batch 1 the network itself takes about 2.5 ms of the 5.5-6 ms; the rest is letterboxing and "
    "copying on the CPU, which INT8 does not touch. INT8 pays off on devices where the network dominates (Jetson, NPUs) or with "
    "batches, which the paper promised to show and did not.",
])}
{fig('fig_speed.png', 480, 'Figure 6. Clean timings on our PC, one job at a time. Left: GPU PyTorch FP16 per photo. Middle: the ONNX network alone on 4 CPU threads (like the Kaggle CPU). Right: post-processing, almost all of it NMS.')}
<p><b>The "43 % faster on CPU" claim.</b> Ultralytics' table gives YOLO11-N 56.1 ms and YOLO26-N 38.9 ms (1.44x). On our CPU the
network alone is only {speed_ratio_model:.2f}x faster ({f1(LS['y11n/nms']['cpu_onnx']['model_ms_4_threads'])} vs {f1(LS['y26n/e2e']['cpu_onnx']['model_ms_4_threads'])} ms, 4 threads;
{f1(LS['y11n/nms']['cpu_onnx']['model_ms_all_cores'])} vs {f1(LS['y26n/e2e']['cpu_onnx']['model_ms_all_cores'])} ms on all cores), but with post-processing it is {speed_ratio_total:.2f}x
({f1(cpu_total('y11n/nms'))} vs {f1(cpu_total('y26n/e2e'))} ms), because YOLO11-N still has to run NMS. The claim holds, and it is mostly the removed NMS.
On the GPU the order flips: YOLO26-N needs {f1(LS['y26n/e2e']['gpu_fp16']['ms_median'])} ms in PyTorch FP16 vs {f1(LS['y11n/nms']['gpu_fp16']['ms_median'])} ms for YOLO11-N.
For our laptop target (G1: the whole pipeline at &gt;= 10 FPS, i.e. 100 ms per frame) the NMS-free YOLO26-N (network +
post-processing, without resizing the photo) leaves about {100 - cpu_total('y26n/e2e'):.0f} ms for the reader; the P2 model with NMS
would use about {cpu_total('y26n_p2/nms'):.0f} ms on its own, the NMS-free P2 model {cpu_total('y26n_p2/e2e'):.0f} ms.</p>
"""

S['b11'] = f"""
<h1>16. Findings, and the numbers to quote</h1>
{table(['#', 'Finding', 'Numbers (all on dataset v1.1 unless COCO)', 'Strength'], [
    ['1', 'The paper\'s COCO accuracy table is reproduced', f"YOLO26-N/S/M {f2(CO['yolo26n/one2many_nms']['AP'])} / {f2(CO['yolo26s/one2many_nms']['AP'])} / {f2(CO['yolo26m/one2many_nms']['AP'])} (paper 40.9 / 48.6 / 53.1); NMS-free -0.6 to -0.8", 'strong'],
    ['2', 'On plates, YOLO26 = YOLO11 (and = YOLOv8 / YOLO12 of other kits)', f"{f1(found_rng[0])}-{f1(found_rng[1])} % found; YOLO26-N vs YOLO11-N {PT['y26n/nms vs y11n/nms']['only_a']} vs {PT['y26n/nms vs y11n/nms']['only_b']} plates; mAP50-95 w/o ridoykj {f1(LNr[0])}-{f1(LNr[1])}", 'strong (no difference)'],
    ['3', 'NMS-free costs 2-3 plates when fully trained, more when under-trained', f"net 2-3 plates, n.s.; stopped YOLO26-N 11 vs 1, {pv(PT['y26n/e2e vs y26n/nms']['p'])}", 'medium'],
    ['4', 'NMS-free gives double boxes', f"{min(dup_e2e)}-{max(dup_e2e)} vs {min(dup_nms)}-{max(dup_nms)} per 1,153 photos", 'strong'],
    ['5', 'NMS-free removes the CPU\'s NMS', f"{f1(LS['y26n/nms']['cpu_onnx']['post_ms_median'])} ms (N), {f1(LS['y26n_p2/nms']['cpu_onnx']['post_ms_median'])} ms (P2) -> {f2(LS['y26n/e2e']['cpu_onnx']['post_ms_median'])} ms", 'strong'],
    ['6', 'MuSGD + rain', f"{f1(min(musgd_rain))}-{f1(max(musgd_rain))} vs {f1(min(sgd_rain))}-{f1(max(sgd_rain))} % found; {PR[RNS]['only_a']} vs {PR[RNS]['only_b']} plates, {pv(PR[RNS]['p'])}", 'strong on synthetic rain, 1 seed'],
    ['7', 'YOLO26 architecture + fog', f"+{f1(PF[ARC]['diff_pp'])} points, {PF[ARC]['only_a']} vs {PF[ARC]['only_b']}, {pv(PF[ARC]['p'])}", 'medium (synthetic)'],
    ['8', 'MuSGD converges a little faster, costs a 13x slower optimiser step', f"+1.2 / +1.5 val at epochs 10 / 20; {f1(OS['MuSGD']['step_ms_median'])} vs {f1(OS['SGD']['step_ms_median'])} ms per step", 'weak (1 seed)'],
    ['9', 'INT8 keeps accuracy, gives no batch-1 speed on a T4', f"-{f2(min(int8.values()))} to -{f2(max(int8.values()))} mAP50 (NMS-free); {f2(HD['y26s']['e2e']['tensorrt_int8']['ms'])} vs {f2(HD['y26s']['e2e']['tensorrt_fp16']['ms'])} ms (S)", 'strong'],
    ['10', 'CPU: YOLO26-N faster than YOLO11-N mostly by losing NMS', f"network {speed_ratio_model:.2f}x, with post-processing {speed_ratio_total:.2f}x; GPU: YOLO11-N faster", 'strong'],
    ['11', 'P2 head: best overall, night and small, all n.s.', f"{f2(T['y26n_p2']['found_%'])} % found; night {f1(BH['y26n_p2/nms']['found_night'])} %; small 24/24", 'weak'],
], [0.4, 3.0, 4.6, 2.0])}
"""

# ======================================================================================== part C
S['c1'] = f"""
<p class="part">PART C · USE IT</p>
<h1>17. How to use this paper in our research</h1>
{table(['Our research part (proposal)', 'How paper #33 is used', 'Evidence we now have (v1.1)'], [
    ['O2: choose the detector among recent YOLOs (YOLOv8, YOLO11, YOLOv12)', 'add YOLO26 to the comparison; it is the newest model in our pinned Ultralytics', f"all tie on clean plates ({f1(found_rng[0])}-{f1(found_rng[1])} %); choose by CPU speed and robustness"],
    ['G1 / O4: &gt;= 10 FPS on a laptop; ONNX / OpenVINO export', 'the NMS-free ONNX graph: no NMS code, no thresholds, fixed [1, 300, 6] output', f"YOLO26-N network + post {f1(cpu_total('y26n/e2e'))} ms vs YOLO11-N {f1(cpu_total('y11n/nms'))} ms on 4 CPU threads"],
    ['RQ4 / G4: robustness to conditions', 'the optimiser is a new factor: MuSGD vs SGD on rain; architecture on fog', f"rain {f1(BH['y26n/nms']['found_rain'])} vs {f1(BH['y26n_sgd/nms']['found_rain'])} %; fog {f1(BH['y26n_sgd/nms']['found_fog'])} vs {f1(BH['y11n/nms']['found_fog'])} %"],
    ['P5: per-condition test sets', 'shows that a real rain subset is missing: synthetic rain is our only evidence', 'no real rain photos in v1.1'],
    ['O5: evaluation protocol', 'report both heads, INT8 vs TensorRT FP16, timings on an idle machine, per-source mAP50-95', 'kit fixed (timings, caps); ridoykj effect confirmed again'],
    ['Edge demo (paper #18 kit)', 'YOLO26-N NMS-free INT8 is the natural edge detector', f"INT8 -{f2(int8['y26n'])} mAP50; no NMS to port"],
    ['Related work chapter', 'cite YOLO26 (Ultralytics) and this review for the NMS-free / MuSGD ideas', 'paragraph below'],
], [2.8, 3.6, 3.6])}
<h3>A paragraph you can adapt for the related-work chapter</h3>
<pre>Ultralytics YOLO26 [ref] removes distribution focal loss and adds an NMS-free
one-to-one head, small-target-aware label assignment and the MuSGD optimiser;
Sapkota et al. [ref] review these changes and report 40.9 COCO mAP50-95 for
YOLO26-N (40.1 without NMS). We reproduced these COCO numbers to within 0.1. On our
duplicate-free Bangladeshi plate benchmark, however, YOLO26-N/S, YOLO26-N-P2 and
YOLO11-N find 97.6-98.4% of 1,011 test plates with no significant difference;
the NMS-free head removes 7.5 ms of CPU post-processing per frame, and detectors
trained with MuSGD found about 20 points more plates on synthetic rain.</pre>

<h1>18. What benefit do you get from it?</h1>
{ul([
    '<b>A tested answer to "should we switch to YOLO26?"</b> Not for accuracy (tie), yes for a CPU or edge pipeline (NMS-free) and possibly for robustness (MuSGD). That is a decision the proposal (O2) asked for, now backed by paired tests.',
    '<b>A possible research contribution.</b> "The optimiser changes how robust a plate detector is to rain" is a small, testable, new claim for Bangladeshi LPR; nobody has reported it. Three seeds and real rain photos turn it into a result (section 22).',
    '<b>A faster CPU detector for G1.</b> YOLO26-N NMS-free: about 16 ms per frame on 4 CPU threads including post-processing, released here as weights.',
    '<b>A verified source.</b> The COCO check lets you cite the YOLO26 numbers without doubt, and the review\'s missing parts (Jetson, INT8, STAL, MuSGD numbers) are exactly what our runs add.',
    '<b>Knowledge examiners ask about</b>: what NMS is and costs, one-to-one vs one-to-many assignment (YOLOv10, paper #27), INT8 calibration, why INT8 is not always faster, and why mAP50-95 depends on label style.',
    '<b>A better kit for everyone</b>: the run exposed a too-tight time cap, overlapping timing jobs and a formatting bug; all three are fixed for the next detector runs.',
])}
"""

S['c2'] = f"""
<h1>19. What to show the faculty (8 slides)</h1>
{table(['#', 'Slide title', 'What is on it', 'Source in this report'], [
    ['1', 'YOLO26 in one picture', 'the two heads (Figure 1), no DFL, STAL + ProgLoss, MuSGD; the review\'s claims and what it does not show', 'sections 1-5'],
    ['2', 'Setup and health', 'v1.1 benchmark, 5 detectors, both heads, T4 x2; 9/9 jobs OK; the epoch-50 stop and the kit fix', 'sections 6-7, Figure 2'],
    ['3', 'The paper\'s numbers are honest', 'COCO check table: 40.81 / 48.62 / 53.09 vs 40.9 / 48.6 / 53.1', 'section 8'],
    ['4', 'On plates: a tie', f"Figure 3 + the paired table; {f1(found_rng[0])}-{f1(found_rng[1])} % found; mAP50-95 ranking is the ridoykj box style", 'sections 9-10'],
    ['5', 'NMS-free: what it buys and costs', f"CPU NMS {f1(LS['y26n/nms']['cpu_onnx']['post_ms_median'])} / {f1(LS['y26n_p2/nms']['cpu_onnx']['post_ms_median'])} ms -> 0.27 ms; double boxes; 2-3 plates lost", 'section 11, Figure 6'],
    ['6', 'The rain result', f"Figure 4 (rain group) + Figure 5; {PR[RNS]['only_a']} vs {PR[RNS]['only_b']} plates; caveats", 'section 13'],
    ['7', 'Speed and INT8', 'INT8 keeps accuracy, not faster at batch 1; the "43 %" is the removed NMS', 'section 15'],
    ['8', 'Decision and next steps', 'YOLO26-N(-P2) NMS-free + MuSGD for the CPU / edge pipeline; seeds + real rain photos; re-run YOLO26-N', 'sections 17, 22'],
], [0.3, 2.2, 5.0, 2.1])}
<div class="box"><p><b>Keep it honest:</b> say "no significant difference" (not "YOLO26 is better") for clean plates; call the rain result
"on synthetic rain, one seed per model"; say the MuSGD YOLO26-N stopped at epoch 50; say that the CPU and GPU timings in
Figure 6 are from our own PC, and the TensorRT numbers from Kaggle. Never compare our mAP with the paper's COCO mAP.</p></div>

<h1>20. What to explain: a 3-minute script</h1>
<p>"YOLO26 is the newest Ultralytics detector, from September 2025, and this paper is a review of it. YOLO26 makes YOLO easier to
deploy: it drops the distribution focal loss, it adds a second head that is trained to give exactly one box per object, so no
non-maximum suppression is needed, it adds a label assignment for small objects, and it trains with a new optimiser, MuSGD,
borrowed from language-model training. The review copies Ultralytics' COCO table, 40.9 mAP for the nano model, but runs no
experiment of its own, and the Jetson and INT8 results it promises are missing.</p>
<p>So we did two things. First, we checked the paper's own numbers: the released weights give 40.8, 48.6 and 53.1 mAP on COCO, the
same as the paper. Second, we trained five detectors on our duplicate-free plate benchmark under the same settings: YOLO11
nano as the baseline, YOLO26 nano with both optimisers, YOLO26 with an extra small-object level, and YOLO26 small.</p>
<p>On normal test photos they all find about 98 percent of the 1,011 plates, and no difference is statistically significant. Plate
detection is solved on our data; reading is where our errors are. The NMS-free head costs almost nothing in accuracy, removes
7.5 milliseconds of NMS per frame on a CPU, and 80 for the small-object model, but it sometimes draws two boxes on one plate,
which we must merge before reading. INT8 keeps the accuracy but is not faster on a T4 at batch size one.</p>
<p>The surprise is rain. On synthetic rain copies of the test photos, every model trained with MuSGD finds about 80 percent of the
plates and every model trained with SGD about 61 percent; with the same network, MuSGD finds 232 plates that SGD misses and SGD
16 that MuSGD misses. This is one seed and synthetic rain, so our next step is three seeds per optimiser and real rainy-day
photos. For our system we will use YOLO26 nano, NMS-free, trained with MuSGD, as the CPU detector."</p>
"""

S['c3'] = f"""
<h1>21. Questions the faculty may ask</h1>
<p class="q">Q1. This is a review without experiments. Why spend a full run on it?</p>
<p>Because the model matters to us, not the review: YOLO26 is the newest detector in the Ultralytics version all our kits use, and
{M('optimizer=auto')} silently trains with MuSGD. Our runs supply exactly what the review lacks: ablations (optimiser, P2, heads),
INT8 accuracy and clean CPU timings, on our task.</p>
<p class="q">Q2. Your mAP50-95 is 45, the paper's 40.9. Is YOLO26 better on plates than on COCO?</p>
<p>The two numbers cannot be compared: COCO has 80 classes and many tiny objects, our task one class of mostly large plates. The fair
comparison with the paper is the COCO check (section 8), and there we match it to 0.1.</p>
<p class="q">Q3. So is YOLO26 better than YOLO11 for us?</p>
<p>Not in accuracy on clean photos: {PT['y26n/nms vs y11n/nms']['only_a']} vs {PT['y26n/nms vs y11n/nms']['only_b']} plates, {pv(PT['y26n/nms vs y11n/nms']['p'])}. It is better for a CPU pipeline
(NMS-free: {f1(cpu_total('y26n/e2e'))} vs {f1(cpu_total('y11n/nms'))} ms) and, if the rain result holds, when trained with MuSGD. On a GPU, YOLO11-N is slightly faster.</p>
<p class="q">Q4. One of your runs stopped at epoch 50. Does that change anything?</p>
<p>It makes that run (YOLO26-N with MuSGD) a lower bound and weakens its NMS-free head. No conclusion depends on it: the tie on clean
plates includes four complete runs, and the rain effect appears in the complete P2 and S runs too. The kit's time caps are fixed,
and a 3-hour re-run of that model is planned.</p>
<p class="q">Q5. Is the rain effect real or an artefact?</p>
<p>It is real for these models on these synthetic photos: same network, data and seed, 232 vs 16 plates, {pv(PR[RNS]['p'])}, and the same
pattern in five models. Whether it generalises is open: one seed each, synthetic rain only, and MuSGD changes two things (the Muon term
and a 3x head learning rate). That is why the next step is three seeds, each part of MuSGD separately, and real rain photos.</p>
<p class="q">Q6. Why would an optimiser change robustness?</p>
<p>We do not know yet and should not guess in a report. Plausible directions to test: the orthogonalised (Muon) updates spread learning over
more directions of each weight matrix, and the higher head learning rate changes how confident the classifier becomes; either could
change how features react to streaks and low contrast. The seed and ablation runs separate these.</p>
<p class="q">Q7. What exactly does "NMS-free" buy on plates?</p>
<p>No NMS step and no IoU threshold: {f1(LS['y26n/nms']['cpu_onnx']['post_ms_median'])} ms less per frame on a CPU (80 ms for the P2 model), 0.2-0.7 ms on a T4, and a simpler
export. It costs 2-3 plates out of 1,011 on fully trained models and gives some double boxes, which a simple overlap check removes.</p>
<p class="q">Q8. Why is INT8 not faster?</p>
<p>At batch 1 on a T4 the network is only about 2.5 ms of the 5.5 ms per photo; resizing and copying the photo on the CPU take the rest.
INT8 speeds up only the network. On a Jetson or an NPU, where the network dominates, it should help; that is the paper #18 experiment.</p>
<p class="q">Q9. Your T4 times are slower than the paper's 1.7 ms.</p>
<p>For YOLO26-S and -M we are within 0.2 ms. The nano models are about 0.8 ms slower because a fixed per-call cost dominates tiny
models on Kaggle's virtual machine (and two jobs ran at once). Our tables give both the model-only and the full per-photo times.</p>
<p class="q">Q10. Why not train YOLO26-M or -L?</p>
<p>Detection is already saturated (98 % found, 5 plates missed by every model) and our goal is &gt;= 10 FPS on a laptop. A larger
detector would cost CPU time the reader needs. YOLO26-S (4x the size) did not find more plates than the nano model.</p>
<p class="q">Q11. Why is test mAP50-95 45 while validation is 60?</p>
<p>The ridoykj source: its labels sit exactly on the plate edge, so tight IoU thresholds fail there. It has 274 of the 1,153 test photos;
without it, test mAP50-95 is {f1(LNr[0])}-{f1(LNr[1])}, like validation.</p>
"""

S['c4'] = f"""
<h1>22. Next steps (in order)</h1>
{ul([
    '<b>Upload</b> (section 23). The team table is already filled ({M("add_results.py")}, 26 Oct: 5 detector rows; the notes\' results and check blocks).',
    '<b>Re-run YOLO26-N with MuSGD for the full 60 epochs</b> with the fixed kit (only the {M("y26n")} job + {M("heads_y26n")}, about 3 h on one T4), to get its complete row and a fair NMS-free head.',
    '<b>The rain experiment</b> (RQ4): YOLO26-N with SGD and with MuSGD, 3 seeds each (6 runs, about 13 GPU-hours = one Kaggle session on 2 GPUs + one more); then split MuSGD into its parts (Muon term only; 3x head learning rate only). Report rain, fog, night and clean plates found with paired tests. The kit needs a small change for the two partial optimisers.',
    '<b>Real rain photos</b>: collect 100-200 rainy-day photos around NSU (part of P5); synthetic rain alone cannot carry a claim.',
    '<b>Detector in the system</b>: YOLO26-N (or -P2) NMS-free + an IoU merge of double boxes in front of paper #24\'s fallback reader; score the 204 whole photos and compare with paper #2\'s cascade.',
    '<b>CPU pipeline (G1)</b>: ONNX / OpenVINO YOLO26-N NMS-free + reader, frames per second end to end on a laptop; OpenVINO INT8.',
    '<b>Edge kit (paper #18)</b>: add YOLO26-N NMS-free INT8 next to YOLOv8n.',
    '<b>Proposal text</b>: add YOLO26 to O2\'s list and the NMS-free / MuSGD sentence to related work (section 17).',
], 'ol')}

<h1>23. GitHub and Drive</h1>
<p>{M('After Train/results.zip')} (659 MiB) and {M('results original.zip')} (2.2 GiB, mostly COCO images) are the whole Kaggle outputs; do not
upload them. The upload folder is ready at {M('After Train/github_upload/')} with the repository's layout:</p>
{table(['Path in the repository', 'What'], [
    [M('weights/paper33/MD.Rokib.Hasan.Oli/README.md'), 'results, model cards, quick start (tested), checksums'],
    [M('.../best_y26n_p2.pt'), '5.3 MiB, YOLO26-N-P2, MuSGD: most plates found (98.42 %); use NMS-free on a CPU'],
    [M('.../best_y26n_musgd.pt'), '5.1 MiB, YOLO26-N, MuSGD (stopped at epoch 50)'],
    [M('.../best_y26n_sgd.pt'), '5.1 MiB, YOLO26-N, SGD (optimiser ablation)'],
    [M('.../best_y26s.pt'), '19.3 MiB, YOLO26-S, MuSGD (under GitHub\'s 25 MB web-upload limit)'],
    [M('.../best_y11n.pt'), '5.2 MiB, YOLO11-N, SGD (baseline)'],
    [M('.../paper33_results.zip'), '5.4 MiB: every results.json / results.csv / log / plot and the kit code; no weights'],
    [M('.../paper33_check_results.zip'), 'the COCO check: results.json, logs, code (the 10-15 MB prediction files are on Drive)'],
    [M('.../results_vs_paper.md, results_original_check.md'), 'the kit\'s two comparison tables'],
    [M('.../Paper33_Study_Report.pdf'), 'this report'],
    [M('notebooks/MD.Rokib.Hasan.Oli/paper33_run.ipynb (+ .log), paper33_check_run.ipynb (+ .log)'), 'the two Kaggle notebooks as they ran'],
    [M('src/MD.Rokib.Hasan.Oli/paper33/'), 'START_HERE, both guides and the (fixed) kaggle/ kit'],
    [M('report/MD.Rokib.Hasan.Oli/paper33/'), 'the notes (paper33_notes.md) and source/ (the local check, analysis, charts and this report\'s builder)'],
], [4.6, 5.4])}
<p>Drive {M('498R / weights / paper33 /')}: {M('After Train/for_drive/paper33_outputs.zip')} (194 MiB: every best.pt, last.pt and ONNX export) and
{M('paper33_check_outputs.zip')} (83 MiB, with the COCO prediction files). Commit list: {M('github_upload/UPLOAD_STEPS.md')}.</p>
"""

S['c5'] = f"""
<h1>Appendix: where every number comes from</h1>
{table(['Number', 'Source'], [
    ['paper numbers', 'the paper PDF (Papers/6_6 Added papers (new5)/33_2025_YOLO26 Key Architectural Enhancements and Benchmarking.pdf), pages given in the text'],
    ['training, test, conditions, heads, TensorRT, INT8, Kaggle CPU', M('After Train/results.zip') + ' → ' + M('runs/*/results.json') + ', ' + M('results.csv') + ', ' + M('logs/') + '; ' + M('paper-33.log')],
    ['COCO check', M('After Train/results original.zip') + ' → ' + M('runs/coco_gpu*/results.json') + '; ' + M('paper-33-original.log')],
    ['plate-by-plate re-scoring (test, rain, fog), mAP without ridoykj, local GPU / CPU timings, optimiser step', M('report_src/local_check.py') + ' → ' + M('local_check.json') + ', ' + M('det_per_plate.csv') + ', ' + M('det_per_photo.csv') + ' (RTX 5060 Ti, Core Ultra 7 265K, Ultralytics 8.4.170, torch 2.13 cu130, dataset v1.1)'],
    ['paired tests, subsets, same-epoch curves, overlaps', M('report_src/analysis.py') + ' → ' + M('analysis.json')],
    ['how MuSGD, ProgLoss, STAL and the e2e export work', 'Ultralytics 8.4.170 source: engine/trainer.py (build_optimizer), optim/muon.py, utils/loss.py (E2ELoss), utils/tal.py, engine/exporter.py'],
    ['charts', M('report_src/charts.py')],
], [3.6, 6.4])}
<pre>@article{{sapkota2025yolo26,
  title   = {{{{YOLO26}}: Key Architectural Enhancements and Performance
             Benchmarking for Real-Time Object Detection}},
  author  = {{Sapkota, Ranjan and Cheppally, Rahul Harsha and Sharda, Ajay
             and Karkee, Manoj}},
  journal = {{arXiv preprint arXiv:2509.25164}},
  year    = {{2025}},
  note    = {{Version 5, 16 March 2026; review, not peer-reviewed}}
}}
@software{{ultralytics_yolo26,
  title   = {{Ultralytics {{YOLO26}}}},
  author  = {{Jocher, Glenn and Qiu, Jing}},
  year    = {{2025}},
  url     = {{https://github.com/ultralytics/ultralytics}},
  license = {{AGPL-3.0}}
}}</pre>
"""

FLOWS = [['cover'], ['a1'], ['a2'], ['a3'], ['b1'], ['b2'], ['b3'], ['b4'], ['b5'], ['b6', 'b7'], ['b8', 'b9'],
         ['b10'], ['b11'], ['c1'], ['c2'], ['c3'], ['c4'], ['c5']]

if __name__ == '__main__':
    n = write_pdf(S, FLOWS, OUT, 'Paper #33 study report · dataset v1.1 run · 498R BanglaLPR',
                  {'title': 'Paper #33 study report (dataset v1.1): YOLO26 key architectural enhancements and benchmarking',
                   'author': 'Prepared for MD. Rokib Hasan Oli',
                   'subject': 'Paper study, v1.1 Kaggle results, COCO check, NMS-free vs NMS, MuSGD, rain robustness, INT8, faculty guide'})
    print('pages', n, OUT)
