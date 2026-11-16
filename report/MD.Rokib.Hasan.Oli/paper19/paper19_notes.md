# Paper #19: LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification

*Reader: MD. Rokib Hasan Oli · new5 (added 25 Oct 2026) · Board group: General LPR state of the art*
*PDF: `Papers/6_3 General LPR state of the art/19_2026_LPLCv2 License Plate Legibility Classification.pdf`
(arXiv 2604.08741, 7 pages; accepted at IJCNN 2026). Page numbers below are the PDF pages.*

---

## LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification (2026)
```
Problem:   Real traffic cameras produce many plates that nobody can read (blur, low resolution, faulty cameras). An
           LPR system should know that BEFORE the OCR: drop illegible plates, enhance poor ones, read good ones. The
           first benchmark (LPLCv1, 2025) was small and had label errors; the best model reached only ~76 % (p.1-2).
Method:    LPLCv2 = LPLCv1 relabelled (1,012 + 787 plates moved between classes) + 26,889 new photos from 700+
           Brazilian radar cameras: 37,099 photos, 41,487 plates in 4 legibility classes (perfect / good / poor /
           illegible), with text, vehicle make / model / colour, camera id, time of day, rain and faulty-camera
           flags (p.3-4, Tables I-II). ResNet-50 (ImageNet, all layers trained), Adam 1e-5, batch 64, an "EMA loss"
           that re-weights classes from a running confusion matrix (p.5, Eq. 1-3). 4 scenarios incl. the binary
           Quality Filter (suitable = perfect + good vs unsuitable) (p.4, Table III); 10-fold protocol 40/20/40.
Data:      LPLCv2 (on e-mail request); new intra-camera and cross-camera fold sets (p.4-5).
Results:   Baseline 4 classes, micro F1: 74.5 (LPLCv1) -> 84.4 (revised labels) -> 88.7 (+ images) -> 89.8 (+ EMA);
           per class 92.7 / 86.2 / 86.3 / 93.0 (p.5, Table IV). Quality Filter 96.7 (EMA) / 96.9 (no EMA) (p.6,
           Table V). Cross-camera 88.2 vs intra-camera 89.2 (p.6, Table VI).
Limits:    Brazilian radar plates only; human labels are subjective (most errors are between adjacent classes, p.5);
           the EMA loss helps the 4-class task by ~1 point and not the Quality Filter; the paper's text and the
           authors' code disagree on the EMA weights and the training settings (see "Paper vs. code").
For us:    A "reject or read" gate for the pipeline: say "unreadable" instead of printing a wrong Bangla number, or
           send poor plates to super-resolution (#20) or low-light enhancement (#35).
Code:      https://github.com/UFPR-IPASP-PR/LPLCv2-Dataset (training code public; data on request)
BibTeX:    wojcik2026lplcv2  (full entry at the bottom; confirm on Google Scholar)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** detect which plate crops are too degraded to read, so the system can reject or enhance them.
2. **How:** a bigger, relabelled 4-class legibility dataset and a ResNet-50 trained with a class-reweighting loss.
3. **How well:** 89.8 % micro F1 on 4 classes, 96.7 % on the binary Quality Filter (p.5-6).

**Useful for us?** Yes, as an idea for the pipeline. The paper is from the Laroca / Menotti group (also #17, #20, #23,
#24), peer-reviewed (IJCNN 2026) and has public code. Its data is Brazilian and only on request, so the kit trains the
same model on OCR-difficulty labels of our own plates and asks the practical question: is a legibility gate better
than the OCR's own confidence?

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Classes | perfect (clear), good (minor distortion), poor (severe but readable), illegible | p.3-4, Fig. 1 |
| Size | 37,099 photos, 41,487 plates: 18,425 / 10,180 / 7,520 / 5,362 per class | p.3-4, Tables I-II |
| Labels made by | YOLOv11 (boxes), PARSeq-tiny (text), ResNet-50 (legibility), all checked by hand | p.3 |
| Relabelling of v1 | 1,012 plates moved between perfect / good / poor; 787 illegible -> poor after OCR check | p.4, Fig. 3 |
| Scenarios | Baseline (4), Legibility Recognition (suitable, poor), Full Recognition (+ illegible), Quality Filter (2) | p.4, Table III |
| Protocol | 5 ordered bins, 10 folds 40/20/40; new photos only added to the training parts | p.4 |
| Model | torchvision ResNet-50 (ImageNet), all layers trained | p.5 |
| Training | Adam 1e-5, LR x0.75 after 5 epochs without gain (min 5e-6), early stop 20 on val accuracy, max 1,000 epochs, batch 64, RTX 6000 | p.5 |
| EMA loss | per-batch class weights from an EMA of the confusion matrix (alpha 0.8, eps 1e-6, cap M 1.2) | p.5, Eq. 1-3 |

**Every result number**
| Setting | Overall | Perfect | Good | Poor | Illegible | Page |
|---|---|---|---|---|---|---|
| LPLCv1 (as published) | 74.5 | 84.5 | 68.0 | 56.7 | 73.0 | p.5, Table IV |
| + revised labels | 84.4 | 92.1 | 83.1 | 83.8 | 88.2 | p.5, Table IV |
| + more images | 88.7 | 92.8 | 86.0 | 86.0 | 92.9 | p.5, Table IV |
| **+ EMA loss** | **89.8** | 92.7 | 86.2 | 86.3 | 93.0 | p.5, Table IV |
| Legibility Rec. / Full Rec. / Quality Filter (+ EMA) | 96.5 / 94.8 / 96.7 | | | | | p.6, Table V |
| same without EMA (+ more images) | 95.9 / 95.2 / 96.9 | | | | | p.6, Table V |
| Intra-camera / cross-camera (Baseline) | 89.2 / 88.2 | | | | | p.6, Table VI |
| Quality Filter intra / cross camera | 96.6 / 96.2 | | | | | p.6, Table VII |
| Speed / training time | not reported | | | | | – |

Terms I looked up:
- **Micro F1:** with one label per sample it equals accuracy; the paper calls it "Overall".
- **EMA (exponential moving average):** a running average, here of the confusion matrix over batches.
- **Camera contamination:** photos of the same camera in train and test, a cousin of the near-duplicate problem (#23).

**Weak points I found:**
- **Subjective labels:** the classes are human judgements, and 1,799 v1 labels had to be changed (p.4).
- **Small effect of the new loss:** +1.1 points on 4 classes, and -0.2 on the Quality Filter (not significant, p.6).
- **No downstream test:** the paper never shows that filtering improves an LPR system's accuracy; that is exactly
  what our kit measures.

### Paper vs. code (authors' repo, checked 25 Oct 2026)
| Point | Paper | Code (`models/loss.py`, `configs/train/ema_br.json`) |
|---|---|---|
| EMA class weights | "higher weight for classes with higher error counts" (p.5) | `max_mispred / (mispred + 1e-6)`, clamped at 1.2: the class with the MOST errors gets the LOWEST weight (1.0) |
| EMA update | Eq. 2 (garbled in the PDF) | `ema = 0.8 * batch_confusion + 0.2 * ema` (the current batch dominates) |
| Batch / patience / epochs | 64 / 20 / up to 1,000 | 16 / 5 / 20 |
| Input size | not given | `imgsz [90, 180]`, letterboxed with grey 127 |

The kit follows the code for the loss (it is what produced the numbers) and the paper for batch 64 / patience 20.

## Pass 3: rebuild it
```
plate crop ─► letterbox 96 x 192, ImageNet norm ─► ResNet-50 ─► 4 logits (perfect / good / poor / illegible)
loss: CE x w[class], w = clamp(max(err) / (err + 1e-6), 1.2), err from an EMA of the batch confusion matrix
use:  P(perfect) + P(good) = "suitable" score ─► accept the most suitable plates ─► OCR; the rest: reject / enhance
```
**Assumptions:** a human-like legibility label exists; it predicts OCR failure; the classifier is cheap enough to
run on every crop.

**What would fail on our Bangladeshi plates:**
- **No labels.** We have no legibility labels; the kit builds an OCR-difficulty proxy with two reader families that
  never saw the crop (cross-fitting). That is a different label (it partly measures the readers, not only the image).
- **Bangla script:** a "poor" Latin plate and a "poor" Bangla plate fail differently (conjuncts and district words are
  long and dense); the classes must be re-learned, not transferred.
- **Small data:** ~5.6 k training crops instead of 33 k plates.

**How I reproduce it:**
- **Readers (cross-fitting):** CRNN and slot CNN trained on each half of the OCR train split (halves = plate text
  hash), each half model reads the other half; val / test read by the half-A models; a CRNN on all train = the
  deployment reader.
- **Labels:** perfect = both exact, good = one exact, poor = best reader within one token, illegible = the rest.
- **Classifiers:** ResNet-50, 4 classes and Quality Filter, EMA loss vs cross-entropy (paper settings, 96 x 192 input).
- **Gate:** plate exact of the plates accepted at 100 / 90 / 80 / 70 / 50 % coverage, classifier vs the CRNN's own
  confidence vs both; share flagged at night / blurry.
- See the [Kaggle guide](../../../src/MD.Rokib.Hasan.Oli/paper19/Kaggle_Training_Guide.md). No original-paper check: the data is only on request.

## Check yourself (2-minute version)
> LPLCv2 grows a plate-legibility benchmark to 41k Brazilian plates in four classes (perfect, good, poor, illegible),
> fixes its labels and trains a ResNet-50 with a class-reweighting EMA loss: 89.8 % micro F1 on four classes and 96.7 %
> on the binary "suitable or not" filter. The idea is to reject or enhance unreadable plates before the OCR. The code
> weights the classes differently from what the text says, and the paper never shows the effect on LPR accuracy. For
> us: test a reject-or-read gate on Bangla plates against the OCR's own confidence.

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| LPLCv2: Fine-Grained License Plate Legibility Classification (Wojcik et al.) | 2026 | IJCNN 2026 | LPLCv2: 37,099 photos, 41,487 Brazilian plates, 4 legibility classes | (YOLOv11 used for labelling) | ResNet-50 legibility classifier (PARSeq-tiny for labels) | micro F1 89.8 % (4 classes), 96.7 % Quality Filter (p.5-6) | Yes (code; data on request) | Brazilian only; subjective labels; no end-to-end LPR test; text vs code differ | Reject-or-read gate before the Bangla OCR |

## Code check → Research board
- **Official repo:** https://github.com/UFPR-IPASP-PR/LPLCv2-Dataset (moved from lmlwojcik/LPLCv2-Dataset; last push
  24 Oct 2026; no licence file; academic use only). `main.py` + `configs/`, ResNet-50, EMA loss in `models/loss.py`,
  `gen_splits.py` for the folds. The data needs an e-mail ("2026 LPLCv2 Request Form").
- **The kit** re-implements the classifier and the EMA loss (as in their code) in `extras/p19_legibility.py`.

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://github.com/UFPR-IPASP-PR/LPLCv2-Dataset
```
```
Code public, data on request. My T4x2 kit: ResNet-50 + their EMA loss on OCR-difficulty labels of our plates, then a reject-or-read gate vs OCR confidence. Paper: 89.8 % F1 (4 classes), 96.7 % Quality Filter (p.5-6). Code weights classes opposite to the text.
```

## BibTeX
```bibtex
@inproceedings{wojcik2026lplcv2,
  title     = {{LPLCv2}: An Expanded Dataset for Fine-Grained License Plate Legibility Classification},
  author    = {Wojcik, Lucas and Machoski, Eduardo A. F. and Nascimento Jr., Eduil and Laroca, Rayson and Menotti, David},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  note      = {arXiv:2604.08741}
}
```
*Accepted at IJCNN 2026 (p.1); add pages and DOI from IEEE Xplore when published.*

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper19_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper19/kaggle/paper19_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper19/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 2 h for the full run |

| Run | GPU | What |
|---|---|---|
| `readers_crnn` | 0 | CRNN x 3 (half A, half B, all train), 60 epochs: cross-fitted readings + the deployment reader |
| `readers_slot` | 1 | slot CNN x 2 (half A, half B), 100 epochs |
| `legibility_labels` | CPU | 4-class OCR-difficulty labels for every OCR crop |
| `cls_b4_ema / cls_qf_ema` | 0 | ResNet-50, 4 classes / Quality Filter, EMA loss (paper), Adam 1e-5, batch 64 |
| `cls_b4_ce / cls_qf_ce` | 1 | the same with plain cross-entropy (the Table IV / V ablation) |
| `gate` | CPU | reject-or-read: accepted plates exact at 100-50 % coverage, classifier vs OCR confidence |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper19_outputs.zip` (team Drive `498R / weights / paper19 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #19: ours vs. the paper

| Metric | Paper (page) | readers_crnn | readers_slot | legibility_labels | cls_b4_ema | cls_b4_ce | cls_qf_ema | cls_qf_ce | gate |
|---|---|---|---|---|---|---|---|---|---|
| 4 classes (Baseline): micro F1 (%), EMA loss | 89.8 (LPLCv2, EMA) (p.5, Table IV) |  |  |  | 51.31 |  |  |  |  |
| 4 classes: micro F1 (%), cross-entropy | 88.7 (+ More Images, no EMA) (p.5, Table IV) |  |  |  |  | 51.92 |  |  |  |
| Per-class F1 perfect / good / poor / illegible (%) | 92.7 / 86.2 / 86.3 / 93.0 (p.5, Table IV) |  |  |  | [69.4, 27.56, 44.6, 57.97] |  |  |  |  |
| Quality Filter: F1 (%), EMA | 96.7 (p.6, Table V) |  |  |  |  |  | 71.74 |  |  |
| Quality Filter: F1 (%), cross-entropy | 96.9 (p.6, Table V) |  |  |  |  |  |  | 70.81 |  |
| Quality Filter read from the 4-class model (%) | - |  |  |  | 71.58 |  |  |  |  |
| Test labels perfect / good / poor / illegible | LPLCv2: 18,425 / 10,180 / 7,520 / 5,362 (all splits) (p.4, Table II) |  |  | {'perfect': 199, 'good': 178, 'poor': 176, 'illegible': 98} |  |  |  |  |  |
| Deployment CRNN: plate exact on all test plates (%) | - |  |  |  |  |  |  |  | 73.43 |
| 80 % accepted by OCR confidence: plate exact (%) | - |  |  |  |  |  |  |  | 84.45 |
| 80 % accepted by the Quality Filter (EMA): plate exact (%) | - |  |  |  |  |  |  |  | 80.81 |
| 80 % accepted by Quality Filter x OCR confidence (%) | - |  |  |  |  |  |  |  | 80.81 |
| Best classifier gate vs OCR confidence at 80 % (pp) | - |  |  |  |  |  |  |  | -3.45 |
| Risk-coverage area, error % (OCR confidence) | - |  |  |  |  |  |  |  | 9.65 |
| Risk-coverage area, error % (Quality Filter EMA) | - |  |  |  |  |  |  |  | 13.29 |
| Classifier ms per crop (T4, batch 1) | not reported | – | – | – | 10.85 | 6.91 | 6.53 | 12.02 | – |
| Epochs run (early stop) | up to 1,000, early stop 20 (p.5) | – | – | – | 24 | 24 | 22 | 21 | – |
| Training time (min) | not reported (RTX 6000 48 GB) (p.5) | 51.00 | 45.60 | – | 12.30 | 12.40 | 11.00 | 10.70 | – |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

Different labels: the paper's are human legibility judgements on Brazilian traffic-radar plates; ours are OCR-difficulty labels from two cross-fitted readers on Bangla plates, so the F1 rows show the direction only. The useful number for us is the gate: plate exact of the plates we accept vs the OCR's own confidence.

<!-- results:end -->

**How to read this table (hand note, 27 Oct; kept outside the markers so a re-run of add_results does not remove it).**
The v1.1 Kaggle run of 26 Oct is healthy: 8 / 8 jobs `code=0` in 78 min, dataset 1.1, code and notebook = the kit; our
PC re-ran the four classifiers (same F1, the same class for 100 % of the 651 test crops) and the deployment CRNN is
byte-identical to paper #24's (SHA-256 `0110631e…f033`). The F1 rows are low (51.9 % / 71.7 % vs 89.8 / 96.7) because our
labels are OCR-difficulty proxies (noisy: two crops of the same plate get the same label 53.6 % of the time), not
because the method was built wrong; every classifier peaked at epoch 1-4. **The answer is in the gate rows: the CRNN's
own confidence (84.45 % exact at 80 % coverage) beats every legibility classifier (best 81.00 %; Δ −5.57 to −1.34
points, bootstrap).** Free signals do better still: confidence + BRTA layout check 85.03 %, and for paper #24's fusion
reader (88.63 %) its confidences + layout + agreement of char-YOLO and CRNN give 96.74 % at 80 % coverage (classifier
93.47 %). The EMA loss changes nothing (McNemar p = 0.42 / 0.50). The `readers_crnn` row in the team table is this
deployment CRNN (= paper #24's `crnn`). Extra local runs (seeds, the authors' balanced sampler, the text's version of the
loss, a direct failure predictor) and the full write-up: [LPLCv2_Study_Report.pdf](../../../weights/paper19/MD.Rokib.Hasan.Oli/LPLCv2_Study_Report.pdf)
(sources in [`source/`](source/)).

## Original-paper check (the paper's own data)
<!-- check:start -->
*Not possible: LPLCv2 is given only on e-mail request (subject "2026 LPLCv2 Request Form", from a university address: your action); the code is public. The main kit uses OCR-difficulty labels on our plates instead.* The main kit above runs the paper's method on our data instead.
<!-- check:end -->
