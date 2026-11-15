"""Part A of the SVTRv2 study report: the paper itself (run numbers only where the code check settles a paper number).
Other papers are named, not numbered (team request for course-facing reports)."""
from report_head import M, fig, table, ul


def part_a(A):
    RW = A['released_weights']
    S = {}
    S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What is this paper, and what problem does it address?</h1>
<p><b>Scene text recognition (STR):</b> a model gets a small crop that contains one word (or, for us, one number plate)
and must output the text. Du, Chen, Xie, Jia and Jiang (Fudan University, USTC, Beijing Jiaotong University) publish
SVTRv2 at <b>ICCV 2025</b> (arXiv 2411.15858 v2, 17 pages with the supplement). The code is in the authors' OpenOCR
toolbox (Apache-2.0) with trained weights on Google Drive.</p>
<p><b>Two families of text readers exist</b> (p.1-2, Fig. 1):</p>
{ul([
    '<b>CTC readers</b> (CRNN, SVTR): one network looks at the image and gives a class score for every column; the CTC rule ("merge repeats, drop blanks") turns the column scores into text. One forward pass, so they are <b>fast</b>. Our own CRNN is one of these.',
    '<b>Encoder-decoder readers (EDTRs)</b> (PARSeq, MAERec, ABINet): an attention decoder writes the text one character at a time and can use what it has already written (language context). They are <b>more accurate</b> on hard text but <b>slower</b>, because the decoder runs once per character.',
])}
<p><b>Why CTC readers lose on hard text</b> (p.2): (1) every crop is squashed to one fixed size such as 32x128, which
distorts curved, rotated or near-square text; (2) CTC assumes the text runs left to right along the columns, which is
false for rotated or multi-line text (the "alignment puzzle"); (3) they have no language context, so occluded letters
cannot be guessed.</p>
<p><b>Their answer: SVTRv2</b>, a CTC reader with three upgrades that fix exactly these three problems, so that it beats
every EDTR on accuracy while staying fast. Headline: SVTRv2-B reads <b>86.14 %</b> of the hard Union14M benchmark words
correctly vs PARSeq 84.26 % and MAERec 85.17 %, at 143 FPS vs 52.6 and 17.1 FPS (p.7, Table 3).</p>
<div class="box"><p><b>Why it matters for us.</b> Our plate readers face the same trade-off. The most accurate readers in our
project are YOLO models that detect every character (character YOLO from the Model Fusion study, 86.79 %; the LPNR model of
the Cascaded YOLO study, 87.71 % plate exact), and our fastest is the CRNN (73.43 %). A Bangla plate also looks like
"irregular text": most plates have <b>two lines</b> (district + series letter on top, six digits below) and the crops are
almost square. SVTRv2 promises attention-level accuracy at CTC speed, and its resizing rule is built for exactly such
shapes. The paper never tests Bangla, plates or small training sets; our run does.</p></div>

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['CTC', 'connectionist temporal classification: the model predicts one class (or "blank") per column; decoding merges repeated classes and drops blanks. Two equal characters in a row (<span class="bn">৪৪</span>) need a blank between them, otherwise they merge into one'],
    ['EDTR', 'encoder-decoder text recogniser: an attention decoder writes one character per step (PARSeq, MAERec); accurate but slower'],
    ['Aspect ratio R', 'crop width / crop height. Our plate crops: median R = 1.50 on the test split (two-line plates are nearly square); English words are usually R = 3-6'],
    ['MSR', 'multi-size resizing: pick one of four input sizes by R (64x64, 48x96, 40x112, 32 x W) instead of always 32x128 (p.3)'],
    ['Local / global mixing', 'the two kinds of blocks in the SVTRv2 encoder: grouped 3x3 convolutions (look at neighbours) and multi-head self-attention (look at the whole plate) (p.3-4)'],
    ['FRM', 'feature rearrangement module: turns the 2-D feature map into the 1-D reading-order sequence CTC needs; first attention along each row, then a learned "selecting token" that picks, for every column, which row to read (p.4)'],
    ['SGM', 'semantic guidance module: during training only, the characters to the left / right of a position must find that character in the visual features, which teaches the encoder language context; removed at test time (p.4-5)'],
    ['U14M / Common', 'Union14M-Benchmark (7 hard English subsets: curved, multi-oriented, artistic, ...) and the 6 classic English benchmarks (IIIT5k, SVT, IC13, IC15, SVTP, CUTE80)'],
    ['U14M-Filter', 'the authors\' training set: Union14M-L (about 3.2 M real word crops) with every sample that also appears in a test set removed (p.5)'],
    ['Plate exact', 'our main score: the whole plate text right (district, metro, letter and all six digits)'],
], [2.2, 7.8])}
"""

    S['a2'] = f"""
<h1>3. How SVTRv2 works, step by step</h1>
{fig('fig_pipeline.png', 470, 'Figure 1. SVTRv2 as used here (after paper Fig. 2, p.3). At test time it is MSR + encoder + FRM + CTC: one forward pass with no decoder loop. The SGM branch is only used while training and was not trained in our run.')}
<h3>Step 1: multi-size resizing (MSR), p.3</h3>
<p>Measure R = w / h of the crop and resize it to <b>64x64 if R &lt; 1.5</b>, <b>48x96 if 1.5 &le; R &lt; 2.5</b>, <b>40x112 if
2.5 &le; R &lt; 3.5</b>, otherwise <b>32 x (&lfloor;R&rfloor; x 32)</b>. The shape stays roughly right, so a near-square
two-line plate is not squashed into a long thin strip. Training batches are grouped by size. The OpenOCR code does
exactly this: {M('tools/data/ratio_dataset_tvresize.py')} rounds R and indexes the list [[64, 64], [96, 48], [112, 40],
[128, 32]] (width, height), so the thresholds 1.5 / 2.5 / 3.5 are the rounding points.</p>
<h3>Step 2: the visual encoder, p.3-4</h3>
<p>Three stages shrink the image to H/8 x W/4 features. Early blocks are <b>local mixing</b> (two grouped 3x3
convolutions), later blocks are <b>global mixing</b> (self-attention over all positions). There is <b>no absolute position
embedding</b>, which is what makes several input sizes possible (the paper notes that ViT and SVTR cannot use MSR for this
reason, p.6). Three sizes: T (5.1 M parameters), S (11.3 M), B (19.8 M). Our build of the authors' code gives
5.13 M and 11.25 M parameters, the same as the paper and the OpenOCR readme.</p>
<h3>Step 3: feature rearrangement (FRM), p.4</h3>
<p>CTC needs one sequence in reading order, but the features are a 2-D grid. FRM learns the mapping in two steps:
<b>horizontal</b> (self-attention inside each feature row, so each row can reorder itself) and <b>vertical</b> (a learned
<b>selecting token</b> attends over the rows of every column and picks what that column should contribute). The result
is W/4 steps: 16 for a 64x64 plate, 24 for 48x96, 32 for 32x128. In code this is {M('RCTCDecoder')}. For a two-line
plate FRM must learn to read the top row before the bottom row; our CRNN does this by hand (it cuts the plate into its
two rows and puts them side by side).</p>
<h3>Step 4: semantic guidance (SGM), training only, p.4-5</h3>
<p>For each character, the string of up to 5 characters on its left (and, separately, on its right) is embedded and used
as an attention query on the visual features; the attended feature must classify the missing character. The encoder can
only make that work if its features already carry language context. Loss = 0.1 x CTC + 1 x SGM cross-entropy (p.5).
SGM is dropped at test time, so it costs nothing at inference. In the released code SGM is implemented with the
{M('GTCDecoder')} + {M('SMTRDecoder')} of the same group's SMTR paper (window 5, {M('sub_str_len: 5')} in the config).</p>
<div class="warn"><p><b>We did not train SGM.</b> Plate text has almost no language: six random digits, a series letter
and a district from a closed list of 64. The paper's own Table 2 shows SGM helps most where language helps most
(occluded words, OST +5.11) and least on regular text (Common +0.41, p.6). The kit trains the CTC branch (encoder + FRM)
only. Our English starting weights were, however, trained with SGM by the authors, so some of its effect is inside the
encoder we start from.</p></div>
"""

    S['a3'] = f"""
<h1>4. What the paper reports</h1>
<h3>Main comparison: 24 readers retrained on the same data (p.7, Table 3)</h3>
{table(['Reader', 'Type', 'Common avg', 'U14M avg', 'OST', 'LTB (long)', 'Params (M)', 'FPS (1080Ti)'], [
    ['CRNN', 'CTC', '90.21', '49.24', '58.0', '47.2', '16.2', '172'],
    ['SVTR-B', 'CTC', '94.58', '71.17', '69.6', '45.1', '18.1', '161'],
    ['PARSeq', 'EDTR', '96.40', '84.26', '79.9', '0.0', '23.8', '52.6'],
    ['MAERec', 'EDTR', '96.36', '85.17', '76.4', '9.8', '35.7', '17.1'],
    ['<b>SVTRv2-T</b>', 'CTC', '94.78', '79.05', '71.4', '47.8', '5.1', '201'],
    ['<b>SVTRv2-S</b>', 'CTC', '96.13', '83.70', '78.0', '47.6', '11.3', '189'],
    ['<b>SVTRv2-B</b>', 'CTC', '<b>96.57</b>', '<b>86.14</b>', '<b>80.0</b>', '<b>50.2</b>', '19.8', '143'],
], [1.6, 0.9, 1.1, 1.0, 0.8, 1.0, 1.0, 1.1], num=(2, 3, 4, 5, 6, 7))}
<p>Word accuracy in %. EDTRs cannot read long text (LTB, 25-35 characters) because their decoders stop at 25 characters;
CTC readers can (p.3, Fig. 5).</p>
<h3>Ablations (p.6, Tables 1-2)</h3>
{table(['What is changed', 'Result', 'Page'], [
    ['MSR vs fixed 32x128 (SVTRv2 with FRM, no SGM), irregular words grouped by R', 'R &lt; 1.5: <b>87.4 vs 72.1 (+15.3)</b>; 1.5-2.5: 88.3 vs 83.1 (+5.2); 2.5-3.5: 86.1 vs 84.1; &ge; 3.5: 87.5 vs 85.6', 'Table 1'],
    ['MSR vs a larger fixed 64x256', 'Fixed64x256 better than 32x128 but still below MSR (U14M 79.03 vs 83.86)', 'Table 1'],
    ['FRM (with MSR): none / horizontal / vertical / both', 'Multi-oriented words 83.73 / 85.76 / 84.44 / 86.19; U14M 82.22 / 82.94 / 82.70 / 83.86', 'Table 1'],
    ['SGM vs no SGM', 'Common 96.16 &rarr; 96.57, U14M 83.86 &rarr; 86.14, OST 74.92 &rarr; 80.03', 'Table 2'],
    ['MSR + FRM added to other encoders (ResNet+TF, FocalNet, ConvNeXtV2)', 'large gains (e.g. FocalNet U14M 71.63 &rarr; 78.94); ViT and SVTR cannot take MSR (absolute positions)', 'Table 1'],
], [3.6, 5.4, 1.0])}
<h3>How the speed is measured (supplement p.13, Sec. 9)</h3>
<p>PyTorch eager mode (no TensorRT, no ONNX), one NVIDIA 1080Ti, batch size 1, 3,000 images; the time is averaged per text
length, then over lengths. So FPS = model time only, on an old GPU, with no image loading. Our numbers in Part C are
measured the same way (model only, batch 1).</p>
<h3>Training recipe (p.5)</h3>
<p>AdamW (weight decay 0.05), learning rate 6.5e-4, batch 1024, one-cycle schedule with 1.5 warm-up epochs out of 20,
maximum text length 25, augmentation (rotation, perspective, motion blur, noise); English models first without SGM,
then with SGM; 4 x RTX 4090. The released configuration files confirm this: {RW['T']['epochs']} epochs, lr
{RW['T']['lr']}, {RW['T']['batch_per_card']} per GPU x 4 = 1024.</p>
"""

    S['a4'] = f"""
<h1>5. What we checked in the code and the released weights</h1>
{table(['Paper says', 'Code / weights say', 'Verdict'], [
    ['MSR sizes 64x64, 48x96, 40x112, 32xW by R thresholds 1.5 / 2.5 / 3.5 (p.3)', 'OpenOCR rounds R and indexes [[64,64],[96,48],[112,40],[128,32]]; R &ge; 4.5 gives 32 x 32&middot;round(R) (the paper writes &lfloor;R&rfloor;)', 'same for R &lt; 4.5 (all but 1 of our crops); tiny difference above'],
    ['SVTRv2-T 5.1 M, -S 11.3 M parameters (Table 3)', 'build_model with the authors\' T / S settings: 5.13 M / 11.25 M (OpenOCR readme: 5.13 / 11.25)', 'same'],
    ['SVTRv2-T 94.78 Common / 79.05 U14M; -S 96.13 / 83.70 (Table 3)', f"the result files shipped with the weights: T {RW['T']['common_avg']} / {RW['T']['u14m_avg']}, S {RW['S']['common_avg']} / {RW['S']['u14m_avg']}", 'the released weights ARE the Table 3 models'],
    ['SGM is a training-only guide (p.4-5)', f"config.yml of the weights: decoder {M(RW['T']['decoder'])} with {M(RW['T']['gtc_decoder'])} (window 5) + {M('RCTCDecoder')}; inference config uses the CTC branch", 'SGM = the SMTR decoder used as a guide'],
    ['FRM = horizontal + vertical rearranging (p.4)', M('RCTCDecoder') + ' (row attention, then a selecting token); our kit loads 14 of its 16 tensors, only the 94-class English classifier is replaced', 'same module, reused'],
    ['Training 20 epochs, lr 6.5e-4, batch 1024 (p.5)', f"config.yml: epoch_num {RW['T']['epochs']}, lr {RW['T']['lr']}, batch_size_per_card {RW['T']['batch_per_card']} (x 4 GPUs)", 'same'],
], [3.4, 4.6, 2.0])}
<p>So the code and weights we used are the paper's, not a re-implementation. We did <b>not</b> re-run the authors'
English benchmarks: their checkpoints are Python pickles and are only loaded on Kaggle, not on our own PC.</p>

<h1>6. Critical reading: strengths and weak points</h1>
<div class="good"><p><b>Strengths.</b> (1) One fair benchmark: 24 readers retrained on the same leakage-filtered data, the
same epochs and augmentation (supplement Sec. 12). (2) Clear ablations for each module, and the modules also help other
encoders. (3) Official code, configs, logs and weights are public (Apache-2.0); parameter counts and benchmark scores of the
weights match the paper. (4) CTC keeps long-text ability that attention decoders lose (LTB).</p></div>
<div class="warn"><p><b>Weak points we found.</b></p>
{ul([
    '<b>Data hunger.</b> Every number comes from about 3.2 M real training crops (or 14.5 M synthetic ones; supplement Fig. 6). Nothing says how SVTRv2 behaves with a few thousand crops, our case (5,607). Our run answers this: from scratch, SVTRv2-T is <b>worse than CRNN</b> (Part C).',
    '<b>MSR is tested only on two irregular subsets</b> (Curve and Multi-Oriented) grouped by R; there is no test on near-square, two-line text such as plates.',
    '<b>Speed on an old GPU in eager mode</b> (1080Ti, batch 1, no TensorRT). Rankings can change on a T4, on a CPU, or with export. Batch-1 latency of a small model is dominated by kernel launches, not by FLOPs.',
    '<b>SGM needs language.</b> Its gain is largest on occluded words; plates have little language, so the module is of limited use for us.',
    '<b>One run per setting</b>; no seeds or confidence intervals are reported, so gaps below about 0.5 points are not shown to be real.',
    '<b>English and Chinese only</b>; no script with conjuncts and vowel signs like Bangla.',
])}</div>
"""
    return S
