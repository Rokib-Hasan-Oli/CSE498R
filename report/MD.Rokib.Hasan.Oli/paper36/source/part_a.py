"""Part A of the SAHI study report: the paper itself, its code and the check of its own numbers.
Other papers are named, not numbered (team request for course-facing reports)."""
from report_head import M, fig, table, ul


def part_a(A):
    CK = A['check']
    VD = A['visdrone']
    WP = VD['working_point']
    ab = CK['abstract_vs_table']
    S = {}
    S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What is this paper, and what problem does it address?</h1>
<p>Akyon, Altinuc and Temizel (OBSS AI and Middle East Technical University, Ankara) publish <b>SAHI</b> at <b>IEEE ICIP
2022</b> (arXiv 2202.06934 v5, 5 pages). The method is released as the open-source {M('sahi')} Python package (MIT
licence, github.com/obss/sahi), which wraps Ultralytics YOLO, MMDetection, Detectron2 and other detectors. It is one of
the most widely used small-object tools in practice.</p>
<p><b>The problem</b> (p.1): detectors are pre-trained on COCO-like photos (about 640 x 480, large objects covering about
60 % of the image height). Drone and surveillance cameras give 4K photos where far objects are a few pixels wide. When
the photo is shrunk to the network input, those objects nearly vanish. The DORI surveillance rule the paper quotes needs
an object at least 10 % of the image height to detect it and 20 % to recognise it.</p>
<p><b>Their answer</b> has two parts, and both work on top of any existing detector, with no change to the network:</p>
{ul([
    '<b>Slicing aided hyper inference (SAHI)</b>, at test time: cut the photo into overlapping tiles, run the detector on each tile (so a far object fills more of the network input), move the boxes back to photo coordinates, optionally add one normal pass on the whole photo (FI), and merge everything with NMS.',
    '<b>Slicing aided fine-tuning (SF)</b>, at training time: add tiles cut from the training photos to the training set, so the detector also learns objects at the larger tile scale.',
])}
<div class="box"><p><b>Why it matters for us.</b> 845 of our 1,153 test photos are 1280 px on the long side, and every detector
in our kits reads them at 640 px, so a plate loses half its width before the network sees it. Far plates are the cases
our detectors miss. SAHI promises more far plates found without new data or a new model, and the authors' package
runs our YOLO weights directly. The paper never tests road photos or number plates; our run does.</p></div>

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['Slice / patch', 'a rectangular tile of the photo (the paper: 640 x 640 px for VisDrone, 400 x 400 for xView). Each slice is resized to the network input like a normal photo'],
    ['PO (patch overlap)', 'neighbouring slices overlap by 25 %, so an object on a slice border is whole in at least one slice'],
    ['FI (full inference)', 'the normal way: one pass on the whole photo, resized. In SAHI + FI, these boxes are merged with the slice boxes'],
    ['SF', 'slicing aided fine-tuning: slices of the training photos are added to the training set (the original photos stay in it)'],
    ['NMS', 'non-maximum suppression: when two boxes overlap more than a threshold, keep only the higher-scoring one'],
    ['IoU / IOS', 'two overlap measures for two boxes. IoU = shared area / combined area; IOS = shared area / area of the <i>smaller</i> box. A half-plate box inside a full-plate box has IoU about 0.5 but IOS 1.0'],
    ['Greedy NMM', 'non-maximum merging (the sahi package default): instead of deleting the weaker box, merge it into the stronger one (the boxes are united)'],
    ['AP50', 'average precision at IoU 0.5 (COCO rules, up to 500 boxes per photo): the area under the precision-recall curve; 100 = every object found with no false box ranked above it'],
    ['AP50 s / m / l', 'AP50 for objects smaller than 32 x 32 px, between 32 x 32 and 96 x 96, and larger, measured in photo pixels (COCO areas)'],
    ['VisDrone / xView', 'VisDrone2019-DET: 8,599 drone photos, 10 classes (pedestrian ... motor), about 47 objects per photo; xView: satellite photos, 60 classes'],
    ['FCOS / VFNet / TOOD', 'three one-stage, anchor-free detectors (2019-2021) in MMDetection that the paper tests SAHI on'],
    ['Plates found / boxes correct', 'our working-point scores (conf 0.25, IoU 0.5): share of labelled plates that get a box (recall) and share of boxes that are a found plate (precision)'],
], [2.2, 7.8])}
"""

    S['a2'] = f"""
<h1>3. How SAHI works, step by step</h1>
{fig('fig_pipeline.png', 480, 'Figure 1. SAHI at test time (top) and SF at training time (bottom), with the numbers of our run (after paper Fig. 2, p.2). A 960 x 1280 photo gives 6 slices of 640 x 640 with 25 % overlap, so 7 forward passes with FI instead of 1.')}
<h3>SF: slicing aided fine-tuning (p.2-3, p.4)</h3>
<p>Every training photo is cut into overlapping patches whose width and height are drawn from a range (VisDrone 480-640
px, xView 300-500 px). The patches are resized (keeping the aspect ratio) so that their width is 800-1333 px and added to
the original photos; the originals stay "to facilitate detection of large objects". The paper notes the risk itself: as
patches get smaller, large objects no longer fit in a patch and their detection can suffer (p.3).</p>
<h3>SAHI: slicing aided hyper inference (p.3)</h3>
<ol>
<li>Cut the query photo into l overlapping M x N patches (VisDrone 640 x 640, xView 400 x 400, 25 % overlap with PO).</li>
<li>Resize each patch (aspect kept) and run the detector on each one separately.</li>
<li>Optionally run the detector once on the whole photo (FI) "to detect larger objects".</li>
<li>Shift every patch box back to photo coordinates and merge all boxes with <b>NMS</b>: boxes whose IoU is above the
matching threshold T<sub>m</sub> = 0.5 are matched and the ones with a probability lower than T<sub>d</sub> are removed
(T<sub>d</sub> is never given a value).</li>
</ol>
<p><b>Cost.</b> The paper only says the computation grows linearly with the number of slices while the memory stays fixed
(p.4); no time is reported anywhere.</p>
<h3>What the code does (sahi 0.12.8, the version our kit pins)</h3>
{ul([
    f"{M('get_sliced_prediction()')} defaults to <b>greedy NMM with IOS 0.5</b>, not the paper's NMS with IoU. Our kit passes the paper's rule explicitly.",
    f"When the model's confidence threshold is below 0.1 (it must be 0.01 to measure AP), the package <b>quietly switches</b> any merge to NMS / IoU (only a log warning says so). So with the default settings, every AP measured through the package uses NMS IoU anyway.",
    f"The full-image pass is added only when the photo gives more than one slice: a 640 x 480 photo is one slice, so SAHI + FI = SAHI = FI for it.",
    f"Slices are run one at a time by default ({M('batch_size=1')}); the package can batch them.",
])}
"""

    S['a3'] = f"""
<h1>4. What the paper reports</h1>
<p>All numbers are AP50 on VisDrone test-dev (Table 1, p.3; 1,610 photos) and on a random 25 % xView split (Table 2).
Detectors: FCOS, VFNet and TOOD in MMDetection, SGD lr 0.01, momentum 0.9, weight decay 1e-4, 500-iteration warm-up,
decay at epochs 16 and 22 (p.4).</p>
{table(['Setting (Table 1)', 'FCOS', 'VFNet', 'TOOD', 'TOOD small / medium / large'], [
    ['FI (baseline)', '25.8', '28.8', '29.4', '18.1 / 44.1 / 50.0'],
    ['SAHI + PO', '29.0', '32.0', '32.5', '22.8 / 45.2 / 43.6'],
    ['SAHI + FI + PO', '31.0', '33.9', '34.7', '23.8 / 48.9 / 50.3'],
    ['SF + FI (no slicing at test time)', '–', '–', '36.8', '24.4 / 53.8 / 66.4'],
    ['SF + SAHI + PO', '38.1', '41.9', '43.1', '31.7 / 59.0 / 60.2'],
    ['<b>SF + SAHI + FI + PO</b>', '<b>38.5</b>', '<b>42.2</b>', '<b>43.5</b>', '31.7 / 59.8 / 65.4'],
    ['xView: FI -> best SF + SAHI', '2.2 -> 17.1', '2.1 -> 17.7', '2.1 -> 20.6', 'small 0.1 -> 14.9 (TOOD)'],
], [3.2, 1.0, 1.0, 1.0, 2.6], num=(1, 2, 3))}
<p><b>The paper's own reading</b> (p.4): slicing adds true positives on small objects; overlap helps small and medium
objects but "slightly decreases" large-object AP because of "false positives predicted from slices that match large
ground truth boxes" (pieces of large objects); FI brings the large objects back. The best small-object AP comes from SF
followed by slicing, the best large-object AP from SF followed by FI. Error analysis (Fig. 3-4): most room is in false
positives and missed objects, not in localisation or class confusion.</p>
<p><b>Read the table carefully.</b> The biggest single jump is SF: TOOD + SF + FI (36.8) uses <i>no slicing at test
time</i> and already beats every SAHI-only row (34.7 at best). So most of the gain comes from training at the slice
scale, which is cheap at test time. Slicing alone adds 3.1-5.3 AP50.</p>

<h1>5. Strengths, weak points and what the paper does not test</h1>
<h3>Strengths</h3>
{ul([
    'Generic and simple: works on any detector, no new layers; one package call for our YOLO weights.',
    'Honest about the large-object cost of slicing (p.3, p.4) and gives the full ablation (overlap, FI, SF) per size class.',
    f"Fully reproducible: every Table 1 prediction file is public, and we re-score all 20 of them to the paper's numbers ({CK['max_abs_diff_rescored_vs_paper_AP50']:.1f} difference, section 6).",
])}
<h3>Weak points we found</h3>
{table(['Point', 'Detail'], [
    ['Abstract vs Table 1', f"abstract: slicing adds 6.8 / 5.1 / 5.3 AP (FCOS / VFNet / TOOD) and with SF 12.7 / 13.4 / 14.5; Table 1 gives <b>{ab['fcos']['table_sahi_gain']}</b> / {ab['vfnet']['table_sahi_gain']} / {ab['tood']['table_sahi_gain']} and {ab['fcos']['table_sf_gain']} / {ab['vfnet']['table_sf_gain']} / <b>{ab['tood']['table_sf_gain']}</b>. FCOS +6.8 and TOOD +14.5 are in no table"],
    ['Conclusion numbers', 'the conclusion says SF adds "14.5 % AP for small objects" (Table 1: TOOD small 18.1 -> 31.7 = +13.6) and overlap "a further 2.9 %" (Table 1: +0.1 to +0.6; Table 2: up to +1.7)'],
    ['No bigger-input baseline', 'the obvious alternative, training or testing the detector at a larger input size, is never run; "memory stays fixed" is the only argument (p.4)'],
    ['No speed', 'slicing multiplies the forward passes (7 for one of our photos), but no ms or FPS appear'],
    ['SF is two changes at once', 'SF patches are also resized up to width 800-1333 px, so SF mixes "more scales in training" with "larger training inputs"; there is no SF-without-upscaling row'],
    ['Merge rule not pinned', 'the paper says NMS with IoU T<sub>m</sub> = 0.5 and leaves T<sub>d</sub> open; the released package defaults to greedy merging with IOS, and the benchmark repo is the only place where the exact call can be checked'],
    ['Aerial data only', 'VisDrone and xView are top-down views with dozens of tiny objects per photo; nothing on road-level photos with one or two medium objects'],
], [2.4, 7.6])}

<h1>6. The paper's own numbers: our check</h1>
<p>The authors released their prediction files for every Table 1 row (footnote 1 repo, now
github.com/fcakyon/small-object-detection-benchmark, release v0.0.1). Our original-check kit
({M('paper36_original_check.ipynb')}, run locally on 27 Oct 2026 on the CPU) downloads the official VisDrone2019-DET
test-dev labels ({CK['photos']:,} photos, {CK['boxes']:,} boxes) and re-scores all 20 files with pycocotools (AP at IoU 0.5,
500 boxes per photo, the paper's metric).</p>
<div class="good"><p><b>Result: Table 1 is exactly what the released predictions give.</b> All 20 rows x 4 columns (AP50, small,
medium, large) = 80 of 80 numbers reproduced with a largest difference of {CK['max_abs_diff_rescored_vs_paper_AP50']:.1f}; the
authors' own {M('eval.json')} files agree too. Only the two abstract numbers above are off. xView (Table 2) was not
checked (its data needs a registration).</p></div>
<p>One technical point: the image ids in the result files follow an unpublished file order of the authors' converter.
We recovered it from box overlap and froze it in {M('c36_visdrone_map.py')}; with it, every {M('eval.json')} is
reproduced. The same folder lets us look inside their predictions (section 15): at a working threshold, slicing also
<b>lowers precision</b> on VisDrone (TOOD {WP['tood_fi']['precision_%']:.1f} % -> {WP['tood_sahi_fi_po']['precision_%']:.1f} % at score 0.3),
but it buys {WP['tood_sahi_fi_po']['recall_%'] - WP['tood_fi']['recall_%']:.1f} points of recall there, because
{VD['visdrone']['coco_small_%']:.0f} % of the VisDrone objects are smaller than 32 x 32 px.</p>
"""
    return S
