"""Part A of the paper #19 report: the paper itself and the authors' code (run numbers only where they explain it)."""
from report_head import M, fig, table, ul


def part_a(A):
    R = A['run']
    S = {}
    S['a1'] = f"""
<p class="part">PART A · UNDERSTAND THE PAPER</p>
<h1>1. What is this paper, and what problem does it address?</h1>
<p><b>License plate legibility classification:</b> before the OCR reads a plate, decide <i>how readable</i> the plate crop
is. Wojcik, Machoski, Nascimento Jr., Laroca and Menotti (Federal University of Paraná, Paraná Military Police, PUCPR;
the same Laroca / Menotti group as four other papers in our study list: Multinational LPR, Plate Super-Resolution, Near-Duplicates and Model Fusion; full titles at the end) publish it at <b>IJCNN 2026</b> (arXiv 2604.08741,
7 pages). Code is public; the data is given on e-mail request.</p>
<p><b>Why anyone would want this</b> (p.1-2): real traffic cameras produce many plates that nobody can read (low
resolution, blur, compression, faulty cameras). A system that processes millions of photos a day cannot run expensive
super-resolution plus OCR on every crop. A legibility label lets it act early:</p>
{ul([
    '<b>perfect / good</b> plates: read them directly with the OCR;',
    '<b>poor</b> plates: send them to super-resolution (the Plate Super-Resolution paper) or enhancement (our HVI-CIDNet study) first;',
    '<b>illegible</b> plates: drop them, and report "unreadable" instead of a wrong plate number (fewer false alarms).',
])}
<p><b>What is new in v2</b> (p.2-4): the first benchmark (LPLCv1, SIBGRAPI 2025) had 10,210 photos and annotation errors,
and its best model reached only ~76 % in the most general setting. LPLCv2 (1) relabels 1,799 v1 plates, (2) adds 26,889
photos from two more days of 700+ Brazilian radar cameras (37,099 photos, 41,487 plates in total), (3) adds rich labels
(camera id, time of day, rain, faulty camera, vehicle make / model / colour), (4) trains ResNet-50 with a new
"EMA loss" and (5) adds a camera-aware split to measure camera contamination.</p>
<div class="box"><p><b>Why it matters for us.</b> Our best plate reader (the fusion reader of our Model Fusion study) is right on 88.6 % of the test
plates, so about 1 plate in 9 is printed <i>wrong</i>. In a real deployment a wrong number is worse than "cannot read".
A reject-or-read gate is therefore a real need for our system. The paper never shows that its classifier helps an
OCR (no end-to-end test, p.7 lists it as future work); our run tests exactly that, against the cheapest alternative:
the OCR's own confidence.</p></div>

<h1>2. Key terms in plain words</h1>
{table(['Term', 'Meaning'], [
    ['Legibility classes', '<b>perfect</b> = clear characters, no visible distortion; <b>good</b> = minor distortion, fully legible; <b>poor</b> = severe distortion but every character can still be identified; <b>illegible</b> = too degraded to validate the characters (p.3-4, Fig. 1). Human judgements.'],
    ['Scenarios (Table III, p.4)', '<b>Baseline</b> = the 4 classes. <b>Legibility Recognition</b> = suitable (perfect + good) vs poor, illegible removed. <b>Full Recognition</b> = suitable / poor / illegible. <b>Quality Filter</b> = suitable vs unsuitable (poor + illegible): the binary gate.'],
    ['Micro F1 ("Overall")', 'with one label per crop it is simply the accuracy (share of crops whose class is right). Per-class F1 = harmonic mean of precision and recall of that class.'],
    ['Confusion matrix', 'table of true class (rows) vs predicted class (columns); the diagonal is right.'],
    ['EMA', 'exponential moving average: a running average that forgets old values, here of the confusion matrix over training batches.'],
    ['5-bin, 10-fold protocol', 'plates are cut into 5 ordered bins; each fold uses 2 bins for training, 1 for validation and 2 for testing; 5 splits, plus 5 more with train and test swapped = 10 folds, results averaged (p.4).'],
    ['Camera contamination', 'photos from the same camera in training and test, so the model may learn the camera instead of legibility (a cousin of the near-duplicate problem of our Near-Duplicates study).'],
    ['Selective prediction', 'a reader that may say "I do not answer". <b>Coverage</b> = share of plates it answers; <b>selective accuracy</b> = plate exact on the answered plates. A good gate raises selective accuracy as coverage drops.'],
    ['AURC', 'area under the risk-coverage curve: the error of the answered plates averaged over every coverage from 1 plate to all. Lower is better; a perfect gate gives the smallest possible value.'],
    ['AUROC (error detection)', 'the chance that a right reading gets a higher score than a wrong one. 0.5 = coin flip, 1.0 = perfect separation.'],
    ['Cross-fitting (ours)', 'train a model on half A and let it judge half B (and the other way round), so no crop is judged by a model that saw it. We use it to make honest labels for the training crops.'],
], [1.4, 6.2])}
"""

    S['a3'] = f"""
<h1>3. How the method works, and what the code really does</h1>
<h3>3.1 The data (p.3-4)</h3>
<p>Photos from traffic radar cameras across the state of Paraná, three capture days. Boxes, text and legibility were first
made by models (YOLOv11, PARSeq-tiny, ResNet-50) and then <b>checked by hand</b>; vehicle data were looked up in the national
traffic database from the plate text. Counts: perfect 18,425, good 10,180, poor 7,520, illegible 5,362 (Table II). In the
relabelling of v1, 514 poor became good, 112 good became poor, 275 perfect became good, 111 good became perfect, and 787
illegible became poor after the OCR text was confirmed (p.4, Fig. 3).</p>
<h3>3.2 The model and the EMA loss (p.5)</h3>
<p>torchvision ResNet-50 (ImageNet weights), every layer trained, Adam 1e-5, learning rate x0.75 after 5 epochs without
gain (minimum 5e-6), early stop after 20 epochs on validation accuracy, up to 1,000 epochs, batch 64. The <b>EMA loss</b> is
cross-entropy with a weight per class that changes every batch:</p>
<pre>W(b)       = min( max(mispred(b)) / (mispred(b) + eps), M )      (Eq. 1)   eps = 1e-6, M = 1.2
smooth(b)  = alpha * confusion(b) + (1 - alpha) * smooth(b-1)    (Eq. 2, printed with confusion(b) twice)  alpha = 0.8
mispred(b) = row sums of smooth(b) - its diagonal                (Eq. 3)   = errors per true class</pre>
<p>The text says this gives "a higher weight for classes with higher error counts" (p.5). <b>Eq. 1 does the opposite</b>:
the class with the most errors gets max / max = 1.0, and every class with at least 17 % fewer errors than the worst one
hits the cap 1.2. So in practice the loss <b>down-weights the single worst class by 1 / 1.2</b>. The authors' code
({M('models/loss.py')}, class {M('FocalEMA')}) matches Eq. 1, not the sentence.</p>
{fig('fig_ema.png', 440, 'Figure 1. The EMA class weights for an example error profile. Blue = Eq. 1 and the authors\' code (the class with the most errors, "good", gets the lowest weight); orange = what the paper\'s sentence describes. Our local runs test both (section 13).')}
<h3>3.3 Paper vs code (authors' repository, commit 7a1d3f6, checked 27 Oct 2026)</h3>
{table(['Point', 'Paper', 'Code', 'Matters?'], [
    ['EMA weights', '"higher weight for classes with higher error counts" (p.5)', f"{M('max_mispred / (mispred_counts + 1e-6)')}, clamp 1.2 = Eq. 1: the worst class gets the LOWEST weight", 'yes: the sentence is wrong, the equation and code agree'],
    ['EMA update', 'Eq. 2 uses confusion(b) on both sides (typo)', f"{M('0.8 * batch_confusion + 0.2 * ema')}: the current batch dominates", 'small'],
    ['Sampler for the EMA run', 'not mentioned', f"{M('ema_br.json')}: {M('sampler: custom')} = BalancedSampler: every epoch each class is cut to a random sample of the smallest class's size", 'yes: the "+ EMA" row changes two things at once'],
    ['Epochs / patience of the EMA config', 'up to 1,000 / 20', f"{M('ema_br.json')}: 20 epochs, patience 5", 'yes (if this config produced Table IV)'],
    ['Learning rate of the cross-entropy rows', '1e-5 for everything', f"{M('train.json')} (used by {M('run_experiments.sh')}): Adam <b>1e-4</b>, plain shuffle", 'yes: "+ EMA" also lowers the learning rate 10x'],
    ['Batch size', '64', f"configs say 16, but {M('run_experiments.sh')} passes {M('-bs 64')}", 'no (consistent)'],
    ['Input size', 'not given', f"{M('imgsz [90, 180]')}, letterboxed", 'we use 96 x 192'],
    ['Citation for the EMA loss', '[31] = Morales-Brotons et al., "EMA of weights in deep learning"', 'that paper averages model WEIGHTS, not a confusion matrix', 'wrong citation'],
    ['Headline number', 'abstract: 89.5 % F1', 'Table IV: 89.8 %', 'small inconsistency'],
], [1.45, 2.0, 3.0, 1.45])}
<p class="small">So the paper's "+ EMA" improvement (+1.1 points, Table IV) may come from the loss, the balanced sampler or the
lower learning rate: the repository does not let us separate them, and the paper does not say. Our kit used the loss alone
(same learning rate, plain shuffle) and our local runs add the balanced sampler (section 13).</p>
"""

    S['a4'] = f"""
<h1>4. What the paper reports</h1>
{table(['Setting (ResNet-50, mean of 10 test folds)', 'Overall F1', 'Perfect', 'Good', 'Poor', 'Illegible', 'Page'], [
    ['LPLCv1 as published', '74.5', '84.5', '68.0', '56.7', '73.0', 'p.5, Table IV'],
    ['+ revised labels', '84.4', '92.1', '83.1', '83.8', '88.2', 'p.5, Table IV'],
    ['+ more images (33k training plates)', '88.7', '92.8', '86.0', '86.0', '92.9', 'p.5, Table IV'],
    ['<b>+ EMA loss</b>', '<b>89.8</b>', '92.7', '86.2', '86.3', '93.0', 'p.5, Table IV'],
    ['Intra-camera / cross-camera split (EMA)', '89.2 / 88.2', '93.4 / 93.5', '82.6 / 81.8', '84.8 / 85.7', '91.6 / 91.1', 'p.6, Table VI'],
], [3.2, 1.0, 0.8, 0.8, 0.8, 0.9, 1.1], num=(1, 2, 3, 4, 5))}
{table(['Scenario', 'LPLCv1', '+ labels', '+ images', '+ EMA', 'Intra / cross camera', 'Page'], [
    ['Legibility Recognition', '92.6', '95.3', '95.9', '96.5', '96.2 / 95.6', 'p.6, Tables V, VII'],
    ['Full Recognition', '87.2', '93.7', '95.2', '94.8', '94.5 / 93.8', 'p.6, Tables V, VII'],
    ['<b>Quality Filter</b>', '93.3', '96.4', '<b>96.9</b>', '96.7', '96.6 / 96.2', 'p.6, Tables V, VII'],
], [2.0, 0.9, 0.9, 0.9, 0.9, 1.5, 1.4], num=(1, 2, 3, 4, 5))}
{ul([
    f"<b>Label quality matters most:</b> relabelling alone gives +9.9 points (74.5 -> 84.4); 7x more training plates +4.3; the EMA loss +1.1 (p.5).",
    f"<b>Errors stay between neighbours:</b> in the confusion matrix of one EMA fold (Fig. 4, p.5) {A['adjacent']['paper_fig4']['adjacent']} of {A['adjacent']['paper_fig4']['errors']} errors ({A['adjacent']['paper_fig4']['adjacent_%']} %) are between adjacent classes; the fold's accuracy is {A['adjacent']['paper_fig4_acc_%']} %.",
    '<b>The EMA loss does not help the binary Quality Filter</b> (96.9 -> 96.7, "not statistically significant", p.6) and slightly hurts Full Recognition.',
    '<b>Camera contamination is small:</b> cross-camera is 1.0 point below intra-camera on 4 classes (p.6).',
    '<b>Not reported:</b> speed, training time, the effect on OCR accuracy, results of any other backbone in v2.',
])}

<h1>5. Weaknesses (with page numbers, so you can show them)</h1>
{table(['#', 'Weakness', 'Where'], [
    ['1', '<b>No downstream test.</b> The paper motivates the classifier as an OCR pre-filter but never measures whether filtering raises plate-reading accuracy, or compares it with the OCR\'s own confidence. Future work (p.7).', 'p.1-2, p.7'],
    ['2', '<b>The EMA explanation contradicts its own equation</b> (section 3.2), and the cited source [31] is about averaging network weights.', 'p.5, ref. [31]'],
    ['3', '<b>The "+ EMA" row is not a clean ablation:</b> in the public code the EMA configuration also switches on a class-balanced sampler and a 10x lower learning rate than the cross-entropy configuration.', 'p.5; configs'],
    ['4', '<b>Tiny, unreplicated gain:</b> +1.1 points on 4 classes from one run per fold, no seeds, no significance test for Table IV; the Quality Filter change is called not significant.', 'p.5-6'],
    ['5', '<b>Subjective labels:</b> 1,799 of the v1 labels had to be changed; the revision used the models\' own mistakes to find the errors, which can make the new labels easier for similar models.', 'p.4'],
    ['6', '<b>One country, one camera type:</b> Brazilian (Mercosur and old Brazilian) plates from fixed radar cameras; no cross-dataset test.', 'p.3, p.7'],
    ['7', '<b>Training data added only to training folds</b> while the test folds are LPLCv1-only, so "+ more images" also changes the train / test balance between days.', 'p.4'],
    ['8', '<b>Small inconsistencies:</b> abstract 89.5 % vs Table IV 89.8 %; Eq. 2 printed wrong; the paper links an old repository name (lmlwojcik/…, now UFPR-IPASP-PR/…).', 'p.1, p.5'],
], [0.3, 6.3, 1.1])}
"""
    return S
