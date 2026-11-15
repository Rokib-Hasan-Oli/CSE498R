"""Part A of the paper #33 report: the paper itself (no run numbers except the COCO check summary in section 4)."""
from report_head import M, fig, table, ul


def part_a(A):
    co = A['coco']
    S = {}
    S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What is this paper, and what problem does it address?</h1>
<p><b>It is a review, not an experiment.</b> Sapkota, Cheppally, Sharda and Karkee (Cornell and Kansas State) describe
<b>Ultralytics YOLO26</b>, the YOLO version Ultralytics released in September 2025, and repeat Ultralytics' published
benchmark tables (p.1, p.8-9). The paper itself trains nothing and measures nothing; every number in it is Ultralytics'.</p>
<p><b>The problem YOLO26 addresses is deployment friction.</b> From YOLOv8 to YOLOv13, each version added accuracy, but kept
two parts that make export to phones, CPUs and small GPUs harder (p.4, p.6-7):</p>
{ul([
    '<b>NMS</b> (non-maximum suppression): a post-processing step that removes duplicate boxes. It costs time on the CPU after the network, and needs two hand-tuned thresholds (score and IoU).',
    '<b>DFL</b> (distribution focal loss): each box side is predicted as a probability distribution over 16 bins, which needs extra decoding operations that some export formats handle badly.',
])}
<p>YOLO26 removes both, and adds two training changes for small objects (ProgLoss, STAL) and a new optimiser (MuSGD). The
paper's message: a YOLO that is <b>simpler to deploy, at the same accuracy</b>, especially on edge devices.</p>
<div class="box"><p><b>Why it matters for us.</b> YOLO26 is the newest detector inside the Ultralytics version every team kit
pins (8.4.170), and {M('optimizer=auto')} in that version silently switches to MuSGD for long trainings. Our proposal (O2)
must choose a plate detector "with the best accuracy-latency trade-off among recent YOLO models", and our demo needs
&gt;= 10 FPS on a laptop (G1). So we need to know, on Bangla plates and not on COCO: is YOLO26 better than YOLO11, what does
NMS-free buy in milliseconds, does INT8 keep the accuracy, and what does MuSGD change?</p></div>

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['NMS', 'non-maximum suppression: after the network, keep the highest-scoring box and delete every other box that overlaps it more than an IoU threshold (Ultralytics: 0.7). Runs on the CPU in most pipelines'],
    ['One-to-many head', 'the classic YOLO head: during training several anchor points are "right" for one object, so at test time it fires several boxes per plate and needs NMS'],
    ['One-to-one head (NMS-free, "e2e")', 'a second head trained so that only ONE anchor point is right per object (the YOLOv10 idea, paper #27). Its top-300 scores are the final boxes: no NMS, no IoU threshold'],
    ['DFL / reg_max', 'DFL predicts each box side as a distribution over reg_max = 16 bins. YOLO26 sets reg_max = 1: one number per side, plain regression (an L1 loss replaces the DFL loss in the training log)'],
    ['STAL', 'small-target-aware label assignment: when deciding which anchor points learn an object, small objects are given a fair share (p.7)'],
    ['ProgLoss', 'progressive loss balancing: the weights of the loss terms change during training (p.7). In the code (E2ELoss): loss = w x one-to-many + (1 - w) x one-to-one, with w falling linearly from 0.8 to 0.1 over the planned epochs'],
    ['Muon / MuSGD', 'Muon (from LLM training) replaces each weight matrix\'s momentum update by its nearest orthogonal matrix (5 Newton-Schulz steps). MuSGD = an SGD step plus a smaller Muon step (section 3)'],
    ['mAP50 / mAP50-95', 'average precision with a box counted right at IoU 0.5 / averaged over IoU 0.50-0.95 (rewards tight boxes). The paper\'s Table 2 "mAP" is COCO mAP50-95'],
    ['Plates found %', 'our plain number: share of the 1,011 human test plates with a model box at confidence &gt;= 0.25 and IoU &gt;= 0.5'],
    ['TensorRT FP16 / INT8', 'NVIDIA\'s optimised GPU runtime with 16-bit floats / 8-bit integers. INT8 needs a calibration set (ours: 30 % of the validation split, never the test split)'],
    ['ONNX (CPU)', 'a portable model file run by onnxruntime on the CPU; how a laptop without a GPU would run the detector'],
    ['McNemar test', 'compares two detectors on the same plates using only the plates where exactly one finds the plate; p &lt; 0.05 = a real difference'],
], [2.1, 7.9])}
"""
    S['a2'] = f"""
<h1>3. The four changes, and what the code really does</h1>
{fig('fig_heads.png', 480, 'Figure 1. One trained YOLO26 model has two heads on the same backbone. Ultralytics 8.4.170 uses the one-to-many head + NMS unless <span class="mono">nms=False</span> is passed; the NMS-free branch is what the paper calls "end-to-end (e2e)". Numbers: our YOLO26-N (MuSGD) on the 1,153 test photos.')}
{table(['Change (paper)', 'What the paper says', 'What Ultralytics 8.4.170 actually does (we read the code)', 'Page'], [
    ['1. DFL removed', '"simpler, faster inference", easier export', f"{M('reg_max: 1')} in yolo26.yaml; the training log shows an {M('l1_loss')} column instead of {M('dfl_loss')}", 'p.6, Fig. 4a'],
    ['2. NMS-free head', 'final boxes straight from the network, "up to 43 % faster CPU inference" for the nano model', f"both heads are trained; {M('predict/val')} use the one-to-many head + NMS unless {M('nms=False')}; {M('export(nms=False)')} builds an end-to-end graph whose output is {M('[1, 300, 6]')} (box, score, class)", 'p.7, Fig. 4b'],
    ['3. ProgLoss + STAL', '"substantial accuracy boost" on small or occluded objects', 'ProgLoss: the one-to-many weight falls from 0.8 to 0.1 over the planned epochs, so the NMS-free head gets most of the loss at the end. STAL: label boxes smaller than the second stride (16 px; 8 px in the P2 model) are enlarged to it when choosing the anchor points that learn them. Neither can be switched off from the command line, so no clean ablation without code changes', 'p.7, Fig. 4c'],
    ['4. MuSGD', 'SGD + Muon, "competitive accuracy with fewer training epochs"', 'conv and linear weights get an SGD-momentum step (lr) <b>plus</b> a Muon step (0.2 x lr); biases and BatchNorm plain SGD; and the classification-head parameters get <b>3 x the learning rate</b> (all groups, "when finetuning"). So "MuSGD vs SGD" changes two things at once', 'p.7, Fig. 4d'],
], [1.5, 2.6, 4.8, 1.1])}
<div class="warn"><p><b>Two things a reader of the paper would not know.</b> (1) In the shipped software the NMS-free head is
<b>opt-in</b>: a YOLO26 model used with default settings still runs NMS. Our training runs are scored that way; the
{M('heads_*')} jobs score both heads of the same weights. (2) {M('optimizer=auto')} picks MuSGD (lr 0.01, momentum 0.9)
whenever a training has more than 10,000 iterations (ours: 60 epochs x 536 batches = 32,160), otherwise AdamW. Every
team kit therefore sets the optimiser by hand; the {M('y26n')} / {M('y26s')} / {M('y26n_p2')} runs ask for MuSGD on purpose.</p></div>
"""
    S['a3'] = f"""
<h1>4. What the paper reports</h1>
{table(['Model (COCO val2017, 640 px)', 'mAP50-95', 'mAP50-95 e2e', 'CPU ONNX ms', 'T4 TensorRT10 FP16 ms', 'Params (M)', 'GFLOPs'], [
    ['YOLO26-N', '40.9', '40.1', '38.9', '1.7', '2.4', '5.4'],
    ['YOLO26-S', '48.6', '47.8', '87.2', '2.5', '9.5', '20.7'],
    ['YOLO26-M', '53.1', '52.5', '220.0', '4.7', '20.4', '68.2'],
    ['YOLO26-L', '55.0', '54.4', '286.2', '6.2', '24.8', '86.4'],
    ['YOLO26-X', '57.5', '56.9', '525.8', '11.8', '55.7', '193.9'],
], [2.6, 1.1, 1.2, 1.2, 1.6, 1.1, 1.0], num=(1, 2, 3, 4, 5, 6))}
<p class="cap">Table 2 of the paper (p.8), copied from Ultralytics. Tables 3-5 (p.9) add classification (ImageNet top-1
71.4-79.9 %), pose (COCO 57.2-71.6) and oriented boxes (DOTA 52.4-56.7), all end-to-end. YOLO11-N, for comparison, is
39.5 mAP50-95, 56.1 ms CPU and 1.5 ms T4 in Ultralytics' documentation (not in the paper).</p>
{table(['Claim in the text', 'Number given?', 'Page'], [
    ['NMS-free costs only a little accuracy', 'yes: -0.6 to -0.8 mAP50-95 (Table 2)', 'p.8'],
    ['"up to 43 % faster CPU inference" for YOLO26-N', 'only as a phrase; it matches Ultralytics\' 56.1 vs 38.9 ms (YOLO11-N vs YOLO26-N, 1.44x)', 'p.3, p.7, p.11'],
    ['Benchmarks on Jetson Nano / Orin vs YOLOv8-v13 and transformers', '<b>promised in the abstract, absent from the paper</b>', 'p.1'],
    ['INT8 / FP16 with "minimal impact on accuracy", better than YOLO11 / YOLO12', 'no number, no table', 'p.10-11'],
    ['STAL + ProgLoss: "substantial accuracy boost" on small objects', 'no number, no ablation', 'p.7'],
    ['MuSGD: "competitive accuracy with fewer epochs", "fewer restarts"', 'no number, no curve', 'p.7'],
    ['Use cases: robotics, manufacturing ("early benchmarks suggest higher throughput")', 'no source, no number', 'p.10-11'],
], [5.6, 3.4, 1.0])}

<h1>5. Weaknesses (with page numbers, so you can show them)</h1>
{ul([
    '<b>No experiment of its own.</b> Every table is Ultralytics\' (the paper says so: "Ultralytics reports", Fig. 1 source link, p.1-2). A review can do that, but the title says "Performance Benchmarking".',
    '<b>The promised edge benchmarks are missing.</b> The abstract announces Jetson Nano and Orin results against YOLOv8, YOLOv11, YOLOv12, YOLOv13 and transformer detectors (p.1); no such table or figure exists.',
    '<b>No ablation.</b> Four changes arrive at once (DFL, NMS-free, ProgLoss + STAL, MuSGD); the paper cannot say which one gives what. The small-object, INT8 and MuSGD claims have no numbers at all (p.7, p.10).',
    '<b>The "default" is not what users get.</b> The paper describes YOLO26 as NMS-free (p.4-5, p.7); in the released package the NMS-free head is opt-in ({M("nms=False")}), so a user who follows the default still runs NMS.',
    '<b>MuSGD is described, not specified.</b> No learning rates, no mixing weights; the code mixes a 0.2 x Muon step with a full SGD step and triples the head learning rate, none of which is in the paper.',
    '<b>Figure 3 (p.5) is unreadable as a diagram</b>: its block sizes ("20 x 16 x 322,86,256,41") do not match YOLO26\'s layers, so the architecture has to be read from the YAML file.',
    '<b>Inconsistent naming and small slips</b>: "YOLOv26" (Table 1) vs "YOLO26"; "YOLO13" vs "YOLOv13" (p.11); YOLOv1 "(2015)" in Table 1 vs 2016 in the text (p.2).',
    '<b>Preprint review</b> (arXiv, version 5); not peer-reviewed.',
])}
<div class="box"><p><b>Fair credit.</b> It is a clear, well-organised summary of what changed in YOLO26 and why it matters for
deployment, with the full Ultralytics tables in one place, and our COCO check (section 8) shows those tables are honest:
the released weights reproduce them to 0.1 mAP ({co['yolo26n/one2many_nms']['AP']:.2f} vs 40.9 for YOLO26-N). Treat the
paper as documentation of the model, and our runs as the evidence on plates.</p></div>
"""
    return S
