"""Part A of the paper #35 report: the paper itself (run numbers only where the check settles a paper number)."""
from report_head import M, fig, table, ul


def part_a(A):
    K = A['density_k']
    C = A['check']
    v1 = C['lolv1']['models'] if 'models' in C['lolv1'] else C['lolv1']
    S = {}
    S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What is this paper, and what problem does it address?</h1>
<p><b>Low-light image enhancement (LLIE):</b> take a dark, noisy photo and produce the photo a camera would have taken in
good light. Yan, Feng, Zhang and colleagues (Northwestern Polytechnical University, Singapore Management University,
Xi'an University of Architecture and Technology) publish this at <b>CVPR 2025</b> (arXiv 2502.20272 v2, 30 pages with the
supplement). The code is MIT-licensed and every trained model is on Hugging Face.</p>
<p><b>The problem is the colour space the network works in</b> (p.1-3, Fig. 1):</p>
{ul([
    '<b>sRGB</b> (normal R, G, B channels): brightness and colour live in the same three numbers, so when a network brightens a dark photo it also shifts the colours ("colour bias").',
    '<b>HSV</b> (hue, saturation, value) separates brightness (V) from colour (H, S), which helps, but adds two kinds of noise: <b>red discontinuity</b> (hue is a number from 0 to 6 and red sits at BOTH ends, so two almost identical reds look very different to the network) and <b>black-plane noise</b> (for a nearly black pixel, hue and saturation are basically random, and the network amplifies that randomness when it brightens).',
])}
<p><b>Their answer:</b> a new colour space, <b>HVI</b>, that keeps HSV\'s separation of brightness and colour but removes both
noises, and a small two-branch network, <b>CIDNet</b>, that brightens in HVI and converts back to sRGB. They report the best
PSNR / SSIM / LPIPS on 10 benchmark sets with only 1.88 M parameters (p.6-7).</p>
<div class="box"><p><b>Why it matters for us.</b> Night photos are our weakest condition: in this run the plate detector
finds {A['run']['det_test_plates']['found_%_by_source']['nightlpbd']['found_%']:.1f} % of the NightLPBD plates (98.1 % over the whole test) and the
CRNN reads {A['run']['crnn']['by_condition']['night']['plate_exact_%']:.1f} % of the night plate crops right (clear crops:
{A['run']['crnn']['by_condition']['clear']['plate_exact_%']:.1f} %). Papers #2 and #21 found the same gap. A small, public,
state-of-the-art enhancer placed in front of the detector or the reader is the obvious idea. The paper never tests that:
its only measures are image-quality scores against a well-lit twin photo, and the authors list "visual recognition
tasks" as unexplored future work (p.20, section 11). Our run answers exactly that question.</p></div>

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['LLIE', 'low-light image enhancement: brighten a dark photo without adding noise or wrong colours'],
    ['Intensity I (Max-RGB)', 'the brightness of a pixel = its largest colour channel, max(R, G, B) (Eq. 1, p.3); the same as V in HSV'],
    ['Polarised hue', 'hue h (0-6) is turned into an angle: (cos πh/3, sin πh/3). On a circle, h = 0 and h = 6 are the same point, so red is continuous (p.4)'],
    ['Intensity collapse Ck', 'a radius factor between 0 and 1 that grows with intensity: Ck = (sin(πI/2) + ε)^(1/k). It shrinks the colour plane of dark pixels to a point, so their random hue cannot hurt (Eq. 4, p.4)'],
    ['Density k', 'the trainable number in Ck that sets how fast the colour plane opens up with brightness; learned together with the network'],
    ['HVI', 'the three maps Ĥ = Ck·S·cos, V̂ = Ck·S·sin and I = max(R, G, B) (Eq. 5): "Horizontal / Vertical - Intensity"'],
    ['CIDNet', 'Color and Intensity Decoupling Network: a U-Net with an HV branch (colour) and an I branch (brightness)'],
    ['LCA', 'Lighten Cross-Attention: each branch looks at the other one (6 blocks), so brightness guides the denoising of colour and back (p.4)'],
    ['PHVIT', 'the inverse transform HVI -> HSV -> sRGB; it has two optional scale knobs αS (saturation) and αI (brightness) (Eq. 6, p.5)'],
    ['PSNR / SSIM / LPIPS', 'pixel error in dB (higher is better) / structural similarity 0-1 (higher) / learned perceptual distance (lower). All need the well-lit photo of the same scene'],
    ['GT mean', 'before scoring, the output is scaled so its mean grey equals the ground truth\'s (p.17, Eq. 21). It removes the brightness error, which no real system can do because it needs the answer'],
    ['BRISQUE / NIQE', '"no-reference" quality scores for photos without a well-lit twin (lower is better)'],
    ['Zero-shot (for us)', 'we use the authors\' released weights as they are: nothing is trained on our plates'],
    ['Dark gate', 'a rule that runs the enhancer only on dark photos; ours: mean grey &lt; 60 (the team\'s night rule)'],
    ['mAP50 / AP50', 'detector average precision with a box counted right at IoU &gt;= 0.5 (one class, so mAP50 = AP50)'],
    ['Plates found / plate exact', 'share of plates with a detector box (conf &gt;= 0.25, IoU &gt;= 0.5) / share of plates whose whole text is read right'],
    ['McNemar / bootstrap', 'paired tests on the same plates: McNemar uses only the plates where exactly one setting is right; the bootstrap re-draws the photos 5,000 times to get a 95 % interval for a difference'],
], [2.0, 8.0])}
"""
    S['a2'] = f"""
<h1>3. How HVI and CIDNet work, and what the code really does</h1>
{fig('fig_hvi.png', 500, 'Figure 1. (a) In HSV, hue is a line from 0 to 6 and two almost identical reds sit at opposite ends. (b) HVI turns hue into an angle, so the same two reds are neighbours. (c) The collapse Ck shrinks the colour of dark pixels towards zero; the curves use the k that each released weight file learned (training starts at k = 5).')}
<p>A photo goes through three steps (p.4-5, Fig. 2): <b>HVIT</b> converts sRGB to the three HVI maps; the <b>dual-branch
U-Net</b> predicts a correction (the I branch sees only the intensity map, the HV branch sees HV plus intensity; six LCA
blocks link them); <b>PHVIT</b> converts the corrected HVI maps back to sRGB. The network adds its output to the input HVI
maps, so it learns "how much to change", not the whole photo. Training uses the same loss in both colour spaces
(L1 + SSIM + edge + VGG perceptual, p.17).</p>
{table(['Item', 'Paper', 'Code at commit eb43d7d (what we ran)', 'Page'], [
    ['Collapse Ck', 'Ck = k-th root of (sin(πI/2) + ε), k trainable', f"{M('pow(k)')} with {M('density_k')} initialised to 0.2: the code's k is <b>1 / paper k</b> (the code comment says so). Learned: generalization {K['hvi_gen']['k_code']:.2f} (paper k {1 / K['hvi_gen']['k_code']:.2f}), LOLv2-Real {K['hvi_lolv2']['k_code']:.2f} ({1 / K['hvi_lolv2']['k_code']:.2f}), LOLv1 {K['hvi_lolv1']['k_code']:.2f} ({1 / K['hvi_lolv1']['k_code']:.2f})", 'p.4, Eq. 4'],
    ['Inverse knobs αS, αI', '"customizing linear parameters"', f"eval.py turns them on per test set: {M('gated')} = saturation x 1.3 for LOLv1; {M('gated2')} + {M('alpha')} = brightness x 0.80 / 0.82 / 0.84 for the three LOLv2-Real weight files. How these constants were chosen is not stated", 'p.5, Eq. 6'],
    ['Attention', 'Lighten Cross-Attention', 'channel ("transposed") attention: a C x C map, so memory grows linearly with the number of pixels; full photos fit on a T4', 'p.4, Fig. 2'],
    ['Parameters', '1.88 M', f"{K['hvi_gen']['params_M']:.2f} M in every released file (counted in PyTorch; default channels 36/36/72/144)", 'p.6, Table 1'],
    ['Input size', 'any', 'height and width must be multiples of 8 (three down-samplings); the kit pads with reflection and crops back', 'code'],
    ['Weights', 'one model per data set', 'Hugging Face, safetensors (no pickle): Generalization (LOLv2-Syn + Real with random gamma 0.6-1.2, the authors\' "real-world" model, p.20), LOLv1 with / without perceptual loss, LOLv2-Real best PSNR / best SSIM, LOLv2-Syn, SICE, Sony-Total-Dark', 'README'],
], [1.6, 2.2, 5.2, 1.0])}
"""
    S['a3'] = f"""
<h1>4. What the paper reports</h1>
{table(['Method', 'Params (M)', 'GFLOPs', 'LOLv1 PSNR / SSIM / LPIPS', 'LOLv2-Real', 'LOLv2-Syn'], [
    ['Zero-DCE (curve, zero-shot)', '0.075', '4.83', '21.880 / 0.640 / 0.335', '16.059 / 0.580 / 0.313', '17.712 / 0.815 / 0.169'],
    ['SNR-Aware', '4.01', '26.35', '26.716 / 0.851 / 0.152', '21.480 / 0.849 / 0.163', '24.140 / 0.928 / 0.056'],
    ['RetinexFormer', '1.53', '15.85', '27.140 / 0.850 / 0.129', '22.794 / 0.840 / 0.171', '25.670 / 0.930 / 0.059'],
    ['GSAD (diffusion)', '17.36', '442.02', '27.605 / 0.876 / 0.092', '20.153 / 0.846 / 0.113', '24.472 / 0.929 / 0.051'],
    ['<b>CIDNet (HVI)</b>', '<b>1.88</b>', '<b>7.57</b>', '<b>28.201 / 0.889 / 0.079</b>', '<b>24.111 / 0.871 / 0.108</b>', '<b>25.705 / 0.942 / 0.045</b>'],
], [2.3, 1.0, 0.9, 2.2, 2.0, 2.0], num=(1, 2))}
<p class="cap">Table 1 of the paper (p.6), 4 of its 14 rows. The LOLv1 column uses GT mean for every method (caption).
FLOPs at 256 x 256.</p>
{table(['Claim', 'Number', 'Page'], [
    ['Extreme darkness (Sony-Total-Dark)', '22.904 dB / 0.676, +6.678 dB over the next method', 'p.6, Table 2'],
    ['Five unpaired sets (DICM, LIME, MEF, NPE, VV)', 'BRISQUE 23.521, NIQE 3.523 (best NIQE; RetinexNet has the better BRISQUE)', 'p.6, Table 2'],
    ['HVI as a plug-in for six other networks (LOLv2-Real)', '+0.38 to +3.56 dB PSNR each', 'p.7, Table 3'],
    ['Speed', '0.053 s per LOLv2 image (600 x 400), fastest of the seven; GPU not named (training used a 2080Ti or 3090)', 'p.7, Table 3'],
    ['Ablation (LOLv2-Real)', 'sRGB 20.06, HSV 21.35, polarisation only 21.56, Ck only 21.54 dB; full CIDNet 24.11', 'p.7, Table 4'],
    ['Cross-data test (train LOLv1, test LOLv2-Syn)', '19.457 dB, best of 8 methods (but LPIPS worse than Zero-DCE)', 'p.14, Table 5'],
    ['Random-gamma training (Generalization model)', 'NIQE 3.134 on the unpaired sets (-0.389)', 'p.20, Table 14'],
], [3.6, 5.4, 1.0])}

<h1>5. Weaknesses (with page numbers, so you can show them)</h1>
{ul([
    '<b>GT mean on LOLv1.</b> The headline LOLv1 numbers (28.201 dB) rescale every output to the ground truth\'s brightness (Table 1 caption, p.6; method p.17). Without it the same weights give about 23.6 dB (our check, section 8). The printed formula is also upside down: Eq. 21 (p.17) divides the output\'s mean by the target\'s, while the code ({M("measure.py")}) correctly multiplies by target / output.',
    f"<b>The LOLv1 row mixes two weight files.</b> PSNR 28.201 / SSIM 0.889 come from the model trained without perceptual loss, LPIPS 0.079 from the one trained with it (our check: {v1['hvi_lolv1']['gt_mean']['lpips']:.4f} vs {v1['hvi_lolv1_perc']['gt_mean']['lpips']:.4f}). No single model reaches all three.",
    '<b>Lost weights.</b> The authors\' README says the model behind the LOLv2-Real 24.111 dB was lost; the public best-PSNR weights give 23.904 (README) / 23.89 (our check). The ablation\'s "full model" row (p.7, Table 4) is that same lost model.',
    '<b>Test-time constants.</b> eval.py multiplies the LOLv2-Real outputs by 0.80-0.84 (a different constant for each weight file) and boosts saturation by 1.3 on LOLv1. The paper names the knobs (Eq. 6) but not the values or how they were picked; since PSNR is very sensitive to brightness, this is a form of tuning on the test set.',
    '<b>Ablation compares different networks.</b> The colour-space rows of Table 4 use a plain U-Net with self-attention (Fig. 5 caption, p.8), while the "full model" is CIDNet; all rows are one run each on one 100-image test set, with no seeds or error bars.',
    '<b>Parameter count.</b> The released model has 1.98 M parameters in PyTorch, not 1.88 M (all five files we loaded).',
    '<b>Image quality only.</b> Every number is PSNR / SSIM / LPIPS or a no-reference score. The paper never asks whether a detector or a reader works better on the enhanced photo; the authors say so themselves (p.20). A "nicer" photo is not automatically an easier one for a network trained on normal photos.',
    '<b>Separate weights per data set, unnamed GPU for the timing.</b> There is no single model for all conditions, and 0.053 s cannot be checked without the GPU name (our T4: 83 ms for the same image size).',
])}
<div class="box"><p><b>Fair credit.</b> The idea is simple, well motivated and cheap: two lines of maths (polarise the hue,
collapse the dark colours) fix two real HSV problems, the network is small, the code runs, and every released weight file
reproduces the authors' README to about 0.1 dB in our check. The weak points are in the reporting (GT mean, mixed rows,
lost weights, test-time constants), not in the method.</p></div>
"""
    return S
