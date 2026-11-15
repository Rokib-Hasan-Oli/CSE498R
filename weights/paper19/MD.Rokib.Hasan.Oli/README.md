# README.md — Paper #19: LPLCv2, Fine-Grained License Plate Legibility Classification (run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — experiments for *"LPLCv2: An Expanded Dataset for Fine-Grained License Plate Legibility Classification"* (Wojcik, Machoski, Nascimento Jr., Laroca & Menotti, IJCNN 2026, arXiv 2604.08741; Paper #19)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 29 October 2026 (Kaggle run of 26 October 2026, dataset **498R BanglaLPR All-in-One v1.1**; local re-check on 27 October and 15 extra runs on an RTX 5060 Ti on 27-29 October 2026)
> **Full study report:** [`LPLCv2_Study_Report.pdf`](LPLCv2_Study_Report.pdf) (paper study, run check, extra runs, findings, faculty guide; 16 pages)

This package tests one question: **is a plate-legibility classifier a better "reject or read" gate than the OCR's own
confidence?** The paper trains a ResNet-50 to sort plate crops into perfect / good / poor / illegible (and a binary
*Quality Filter*: suitable vs unsuitable) so that a system can drop unreadable plates before the OCR, but it never tests
the effect on plate reading. No legibility labels exist for Bangla plates and LPLCv2 is Brazilian and only on request,
so our kit builds **OCR-difficulty labels** for our 7,024 OCR crops with two cross-fitted readers, trains the paper's
classifier with and without its EMA loss, and compares it as a gate with the CRNN's confidence. Sources:
`paper19_results.zip` and the local checks in
[`report/MD.Rokib.Hasan.Oli/paper19/source/`](../../../report/MD.Rokib.Hasan.Oli/paper19/source/).

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [What Is in This Package](#2-what-is-in-this-package)
3. [Quick Start](#3-quick-start)
4. [Model Card](#4-model-card)
5. [Results Summary](#5-results-summary)
6. [Reproducing the Results](#6-reproducing-the-results)
7. [Limitations](#7-limitations)
8. [Differences from the Original Paper](#8-differences-from-the-original-paper)
9. [File Integrity](#9-file-integrity)
10. [Licence and Data Notes](#10-licence-and-data-notes)
11. [Citation](#11-citation)

---

## 1. At a Glance

| | |
|---|---|
| **Question** | Does a legibility classifier (LPLCv2 method) reject wrong plate readings better than the reader's own confidence? |
| **Answer** | **No.** Keeping the 80 % of test plates each score likes best, the CRNN is right on **84.45 %** with its own confidence and on **81.00 %** with the best classifier (Δ −3.45, 95 % CI −5.57 to −1.34); every classifier loses (AUROC 0.837 vs 0.72-0.77) |
| **Free signals win** | confidence + BRTA layout check: 85.03 %; for paper #24's fusion reader (88.63 %) confidences + layout + agreement of char-YOLO and CRNN: **96.74 %** at 80 % coverage vs classifier 93.47 %; when the two readers agree (72.8 % of plates) the reading is right 97.3 % of the time |
| **Classifier on its own task** | our proxy labels: 4 classes **51.92 %** (CE) / 51.31 % (EMA) micro F1, Quality Filter **71.74 %** (EMA) / 70.81 % (CE); paper (its human labels): 89.8 / 96.7 %; majority-class guess 30.6 / 57.9 % |
| **EMA loss** | no effect on 4 classes: same-crop McNemar p = 0.42, and over 3 seeds (Kaggle 42, local 1, 2) EMA 52.28 ± 1.82 vs CE 52.33 ± 2.18 %; Quality Filter: p = 0.50 on one run, but EMA is ahead in all 3 seed pairs (71.48 vs 70.40 %), with no gain as a gate |
| **Labels** | perfect / good / poor / illegible = 199 / 178 / 176 / 98 test crops; the deployment CRNN reads 99.0 / 87.6 / 61.4 / 17.4 % of them right; same plate photographed twice gets the same 4-class label only 53.6 % of the time (noisy) |
| **Training** | 2 × CRNN-half + 1 deployment CRNN (51 min), 2 × slot CNN (46 min), 4 × ResNet-50 (11-13 min each, best epoch 1-4); Kaggle Tesla T4 x2; 8 / 8 jobs `code=0`, 78 min |
| **Reproducibility** | our PC re-ran every classifier: same F1 and the same class for 100 % of the 651 crops; the deployment CRNN is **byte-identical** (SHA-256) to the paper #24 and #35 runs' CRNN |
| **Extra runs (local)** | 15 more ResNet-50 trainings on our RTX 5060 Ti (seeds 1 and 2 of the four Kaggle settings, the authors' balanced sampler, the EMA weights as the paper's text describes them, a direct "will the reader fail?" predictor): **none is a better gate**; over all 19 trainings the CRNN is right on 78.69-81.57 % of the plates kept at 80 % coverage vs **84.45 %** with its own confidence (section 5.5) |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · torchvision ResNet-50 IMAGENET1K_V1 · dataset v1.1 |

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `LPLCv2_Study_Report.pdf` | 3.9 MiB | Full study report, 16 pages |
| `best_cls_b4_ce_fp16.pt` | 45.1 MiB | The 4-class legibility classifier with the best gate result (cross-entropy, best epoch 4 of 24), weights stored in FP16 (the FP32 file is 90 MiB); a dict with `state_dict`, `classes`, `input_HW`, preprocessing notes |
| `paper19_results.zip` | 0.6 MiB | Results archive, **no model weights** (see 2.2) |
| `results_vs_paper.md` | — | The kit's ours-vs-paper table |

**The CRNN is not in this folder** because it is the same file as paper #24's: this run's `readers_crnn/model_full.pt`
has the same SHA-256 (`0110631e…f033`) as the #24 run of 22 October and the #35 run (same kit, same seed). Use
[`weights/paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt`](../../paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt).
The other three classifiers (4-class EMA, Quality Filter EMA / CE) are in `paper19_outputs.zip` on the team Drive
(`498R / weights / paper19 /`), with all four in FP32.

### 2.2 Inside `paper19_results.zip`

| Path | What |
|---|---|
| `runs/readers_crnn/`, `runs/readers_slot/` | `results.json` (halves, epochs, deployment CRNN test score), `reader_preds.csv` (every cross-fitted reading with confidence) |
| `runs/legibility_labels/` | `labels.csv` (file, split, text, source, 4-class label, Quality Filter label, both readings), `deploy_test.csv` (deployment CRNN readings of the 651 test crops) |
| `runs/cls_*/` | `results.json` (settings, val / test micro and per-class F1, confusion matrix, epoch log, ms per crop), `test_scores.csv` (class probabilities per test crop) |
| `runs/gate/results.json` | selective plate exact at 100 / 90 / 80 / 70 / 50 % coverage, AURC, share flagged per condition, for every gate |
| `logs/`, `code/`, `environment.json`, `results_vs_paper.md` | the job logs, the exact kit code, versions (incl. `dataset_version: 1.1`), the comparison table |

---

## 3. Quick Start

Tested on 27 Oct 2026 (Python 3.14, torch 2.13, torchvision, OpenCV) from a copy of the repository layout. Run from the
repository root; it needs this folder, `weights/paper24/…/best_crnn_fp16.pt` and the kit in
`src/MD.Rokib.Hasan.Oli/paper19/kaggle/`.

```python
import sys, cv2, numpy as np, torch, torchvision
sys.path.insert(0, 'src/MD.Rokib.Hasan.Oli/paper19/kaggle')      # the kit: seq.py (CRNN), metrics.py, vocab.py
import seq, metrics

# the legibility classifier (4 classes, cross-entropy), stored in FP16
ck = torch.load('weights/paper19/MD.Rokib.Hasan.Oli/best_cls_b4_ce_fp16.pt', map_location='cpu', weights_only=True)
cls = torchvision.models.resnet50()
cls.fc = torch.nn.Linear(cls.fc.in_features, len(ck['classes']))
cls.load_state_dict({k: v.float() for k, v in ck['state_dict'].items()})
cls.eval()
# the reader: this run's deployment CRNN is byte-identical to paper #24's
cr = torch.load('weights/paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt', map_location='cpu', weights_only=True)
ocr = seq.CRNN(len(cr['classes']), cr['H'])
ocr.load_state_dict({k: v.float() for k, v in cr['state_dict'].items()})
ocr.eval()


def letterbox(bgr, H=96, W=192):                  # the kit's classifier input (p19_legibility.CropDS, no augmentation)
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    s = min(H / h, W / w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    canvas = np.full((H, W, 3), 127, np.uint8)
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (canvas.astype(np.float32) / 255 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(x.transpose(2, 0, 1)).float()[None]


crop = cv2.imread('plate_crop.jpg')               # one plate crop (from the detector)
with torch.no_grad():
    p = cls(letterbox(crop)).softmax(1)[0]
    toks, conf = seq.ctc_decode(ocr(seq.prep(crop, cr['H'], cr['W'], cr['stack'], False)[None]).float())
text = seq.vocab.tokens_to_text([cr['classes'][k - 1] for k in toks[0]])
print(text, f'| CRNN confidence {conf[0]:.3f} |', 'layout OK' if metrics.is_valid_plate(text) else 'layout BROKEN',
      '|', {c: round(float(v), 3) for c, v in zip(ck['classes'], p)}, f'| suitable {float(p[0] + p[1]):.3f}')
```

On the test crop `ocr/images/test/anpr_0142111c1238.jpg` this prints
`ঢাকা মেট্রো-ল ৪৪-৫১৬০ | CRNN confidence 0.992 | layout OK | {'perfect': 0.006, 'good': 0.124, 'poor': 0.666, 'illegible': 0.205} | suitable 0.129`:
the reading is **correct and confident, yet the classifier calls the plate "poor"** — a typical case of why the
confidence is the better gate. The recommended gate (section 5.3) uses the confidence, the BRTA layout check and, with
paper #24's fusion, the agreement of char-YOLO and CRNN; its fitted weights and code are in `source/analysis.py`.

---

## 4. Model Card

### 4.1 `best_cls_b4_ce_fp16.pt` — ResNet-50 legibility classifier (4 proxy classes)

| | |
|---|---|
| Architecture | torchvision ResNet-50 (23.5 M parameters), ImageNet start (IMAGENET1K_V1), all layers trained, `fc` → 4 |
| Input | RGB plate crop, aspect kept, letterboxed to 96 × 192 on grey 127 (bicubic), ImageNet mean / std |
| Classes (ours) | `perfect` = both cross-fitted readers (CRNN, slot CNN) read the crop exactly · `good` = exactly one did · `poor` = neither, best within 1 token · `illegible` = the rest |
| Training data | All-in-One v1.1 OCR train crops (3,471 human + 2,136 auto texts) with cross-fitted labels 2,630 / 1,115 / 1,183 / 679; val 766, test 651 crops (human texts) |
| Settings | Adam 1e-5, batch 64, LR × 0.75 after 5 epochs without a better val accuracy (min 5e-6), early stop 20, max 80 epochs, light photometric + ±3° / ±5 % affine augmentation, AMP, seed 42, plain cross-entropy |
| Test (651 crops) | micro F1 **51.92 %**, macro 50.85 %; per class 70.42 / 30.66 / 45.19 / 57.14 %; as a Quality Filter 71.43 % |
| As a gate for the CRNN | plate exact of the accepted plates at 90 / 80 / 50 % coverage: 77.30 / **81.00** / 88.65 % (CRNN confidence: 79.86 / 84.45 / 91.72 %); AUROC 0.766 (confidence 0.837) |
| Speed (batch 1) | 3.2 ms RTX 5060 Ti (FP32), 19.9 ms CPU (Core Ultra 7 265K, 4 threads); T4 6.5-12 ms (Kaggle, shared machine) |
| Use | research only: a negative result. Do **not** put it in front of the reader as a gate; the free signals in 5.3 are better |

### 4.2 Readers that made the labels (not uploaded)

| Reader | Data | Test plate exact |
|---|---|---|
| CRNN half A / slot CNN half A | half of the OCR train split (hash of the plate text), read the other half, val and test | 54.07 % / 34.41 % |
| CRNN half B / slot CNN half B | the other half, read half A | — |
| Deployment CRNN (all train) | all OCR train crops | 73.43 % (= paper #24's CRNN) |

---

## 5. Results Summary

### 5.1 The classifiers (test, micro F1 %)

| Run | Task | Loss | Test F1 | Per class | Best epoch / run | Paper (human labels) |
|---|---|---|---|---|---|---|
| `cls_b4_ema` | 4 classes | EMA (authors' code) | 51.31 | 69.4 / 27.6 / 44.6 / 58.0 | 4 / 24 | 89.8 (92.7 / 86.2 / 86.3 / 93.0) |
| `cls_b4_ce` | 4 classes | cross-entropy | 51.92 | 70.4 / 30.7 / 45.2 / 57.1 | 4 / 24 | 88.7 |
| `cls_qf_ema` | Quality Filter | EMA | 71.74 | 73.3 / 69.9 | 2 / 22 | 96.7 |
| `cls_qf_ce` | Quality Filter | cross-entropy | 70.81 | 71.5 / 70.1 | 1 / 21 | 96.9 |

### 5.2 Gates for the deployment CRNN (651 test plates; plate exact % of the accepted plates)

| Gate score | 100 % | 90 % | 80 % | 70 % | 50 % | AURC | AUROC |
|---|---|---|---|---|---|---|---|
| oracle (perfect gate) | 73.43 | 81.57 | 91.75 | 100 | 100 | 3.91 | 1.000 |
| confidence + layout + classifiers (val-fitted) | 73.43 | 80.55 | 85.41 | 88.82 | 93.25 | 8.39 | 0.869 |
| CRNN confidence + BRTA layout check | 73.43 | 80.55 | 85.03 | 88.38 | 92.02 | 9.06 | 0.853 |
| **CRNN confidence** | 73.43 | 79.86 | **84.45** | 87.72 | 91.72 | 9.65 | 0.837 |
| classifier 4-class CE | 73.43 | 77.30 | 81.00 | 83.33 | 88.65 | 12.09 | 0.766 |
| classifier 4-class EMA | 73.43 | 77.30 | 80.81 | 83.55 | 88.04 | 12.10 | 0.767 |
| Quality Filter EMA | 73.43 | 77.13 | 80.81 | 83.33 | 88.65 | 13.29 | 0.758 |
| Quality Filter CE | 73.43 | 75.94 | 78.69 | 80.92 | 86.50 | 14.72 | 0.719 |

Paired bootstrap over the plates (2,000 draws), Δ plate exact at 80 % coverage vs the CRNN confidence: classifier 4-class
CE −5.57 to −1.34, Quality Filter EMA −5.95 to −1.16 points (worse); confidence + layout 0.00 to +1.73 (AURC −1.28 to
−0.11, better). Classifiers added to confidence + layout: −0.77 to +1.92 at 80 % (no difference), AURC −1.42 to −0.34.

### 5.3 Gates for paper #24's fusion reader (char-YOLOv8n, CRNN fallback; 88.63 % at full coverage)

| Gate score | 90 % | 80 % | 50 % | AURC | AUROC |
|---|---|---|---|---|---|
| **free signals** (char-YOLO and CRNN confidences, both layout checks, agreement; val-fitted) | 95.22 | **96.74** | 97.24 | 3.10 | 0.872 |
| free signals + classifiers (val-fitted) | 95.05 | 96.93 | 97.55 | 2.86 | 0.879 |
| char-YOLO confidence alone | 95.22 | 96.35 | 96.93 | 3.58 | 0.849 |
| classifier 4-class CE | 91.13 | 93.47 | 94.79 | 5.03 | 0.743 |

Classifier minus free signals at 80 %: −5.19 to −1.73 points; classifiers added to the free signals: −0.19 to +0.58.
Simple rule: **accept the reading when char-YOLO and the CRNN agree** (72.8 % of test plates, 97.3 % right); otherwise
use the fitted score or report "unreadable".

### 5.4 By condition (80 % coverage)

| Condition | Plates | CRNN exact | Rejected by confidence / classifier | Exact of the kept plates, confidence / classifier |
|---|---|---|---|---|
| clear | 480 | 77.50 % | 88 / 76 | 88.52 / 83.66 % |
| blurry | 112 | 64.29 % | 19 / 20 | 69.89 / 70.65 % |
| night | 65 | 61.54 % | 23 / **34** | 80.95 / 80.65 % |

### 5.5 Extra runs on our PC

15 more ResNet-50 trainings ran on our RTX 5060 Ti with exactly the kit's recipe (`local_train.py`, 5-10 min each,
27-29 Oct 2026): the four Kaggle settings with seeds 1 and 2, the authors' class-balanced sampler (their real "+ EMA"
recipe), the EMA weights as the paper's text describes them (more errors -> more weight), and a classifier trained
directly on "will the reader fail?". K = Kaggle T4 run, L = our PC; gate = plate exact of the deployment CRNN on the 80 %
of test plates each classifier rates most suitable (CRNN confidence: **84.45 %**). Per class = perfect / good / poor /
illegible F1 (4 classes) or suitable / unsuitable.

| Setting | Seeds | Test F1 mean ± SD | Per class (mean) | Gate at 80 % (mean) | AUROC (mean) | Best epochs |
|---|---|---|---|---|---|---|
| 4 classes, EMA (kit) | K42, L1, L2 | 52.28 ± 1.82 | 71.0 / 23.5 / 46.1 / 58.8 | 80.94 | 0.764 | 4, 4, 2 |
| 4 classes, CE (kit) | K42, L1, L2 | 52.33 ± 2.18 | 71.0 / 28.5 / 45.8 / 57.7 | 80.74 | 0.761 | 4, 3, 2 |
| 4 classes, EMA + authors' balanced sampler | L42, L1 | 51.61 ± 2.82 | 67.0 / 36.7 / 44.5 / 59.1 | 81.19 | 0.774 | 9, 6 |
| 4 classes, EMA weights as the paper's text says | L42, L1 | 50.46 ± 1.63 | 69.5 / 30.7 / 42.7 / 55.0 | 80.42 | 0.767 | 4, 3 |
| Quality Filter, EMA (kit) | K42, L1, L2 | 71.48 ± 0.32 | 72.4 / 70.5 | 80.36 | 0.753 | 2, 2, 2 |
| Quality Filter, CE (kit) | K42, L1, L2 | 70.40 ± 0.70 | 71.4 / 69.3 | 79.40 | 0.732 | 1, 1, 2 |
| failure predictor ("will the half-data CRNN fail?"), CE | L42, L1, L2 | 71.32 ± 0.64 | 70.3 / 72.3 | 79.91 | 0.748 | 1, 2, 2 |

- **Seeds matter more than the loss:** the same 4-class recipe scores 50.38-54.69 % across seeds, and EMA and CE have
  the same mean, so the Kaggle run's 0.6-point gap was noise. On the Quality Filter EMA is ahead in all three seed
  pairs (+0.9, +2.0, +0.3 points), the opposite of the paper's Table V (96.7 vs 96.9), with no gain as a gate.
- **The authors' balanced sampler** lifts the weak "good" class (F1 36.7 vs 23.5 %) at the cost of "perfect"; no gain
  overall. Its seed-1 run is the best gate of all (81.57 %), still 2.88 points below the confidence.
- **The failure predictor** is no better a gate (79.91 %), probably because its target is the half-data CRNN's failure (54.1 % of test
  plates read right), not the deployment CRNN's (73.4 %).
- **Over all 19 classifier trainings the gate at 80 % coverage stays between 78.69 and 81.57 %**, every one below the
  CRNN's own confidence (84.45 %); the best AUROC is 0.776 vs 0.837. Per-run results and test / val scores:
  [`source/_local_runs/`](../../../report/MD.Rokib.Hasan.Oli/paper19/source/_local_runs/), log:
  `source/local_train_log.txt`.

---

## 6. Reproducing the Results

1. **Kaggle run:** open [`src/MD.Rokib.Hasan.Oli/paper19/kaggle/paper19_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper19/kaggle/paper19_kaggle.ipynb)
   on Kaggle (GPU T4 x2, Internet on), Run All (about 80 min; the dataset downloads itself). Steps:
   [`Kaggle_Training_Guide.md`](../../../src/MD.Rokib.Hasan.Oli/paper19/Kaggle_Training_Guide.md). The executed run is
   [`notebooks/MD.Rokib.Hasan.Oli/paper19_run.ipynb`](../../../notebooks/MD.Rokib.Hasan.Oli/paper19_run.ipynb).
2. **Local re-check, extra runs and report:** [`report/MD.Rokib.Hasan.Oli/paper19/source/`](../../../report/MD.Rokib.Hasan.Oli/paper19/source/):
   `local_check.py` → `run_local_train.sh` (calls `local_train.py`, about 2 h on one GPU) → `analysis.py` → `charts.py`
   → `build_report.py`; `make_upload.py` builds this upload tree. The scripts take the unpacked `results.zip` as their
   first argument and expect the dataset in `D:\498R\dataset` (set `DATASET_ROOT` for another place).
3. **Team table:** `docs/results/add_results.py paper19_results.zip --member "Rokib Hasan Oli"` (row `readers_crnn`
   under paper #19 in [`results/`](../../../results/); the classifiers and the gate are in the paper #19 block).

---

## 7. Limitations

- **Proxy labels, not human labels.** Our classes say whether two particular readers succeeded; they are noisy (the
  same plate gets the same label in 53.6 % of pairs) and partly measure reader blind spots (rare districts, one-line
  and CNG plates). A classifier trained on human legibility labels might do better at spotting truly illegible images.
  The kit scores against human labels automatically if `ocr/legibility_human.csv` (file,label) is added to the dataset.
- **Small data:** 5,607 training crops (LPLCv2 trains on ~33,000 plates per fold).
- **One test split:** 651 crops; the bootstrap intervals cover the plate sampling, the seeds in 5.5 the training noise.
- **The fusion-reader gate** is fitted on the 766 validation crops with a small logistic regression; a different reader
  or dataset needs a refit.
- **Auto-labelled training crops** (2,136 Zenodo-25k texts) are 83 % "perfect" and absent from val / test: a possible
  shortcut for the classifier.

---

## 8. Differences from the Original Paper

| Paper | Here |
|---|---|
| Human legibility labels on 41,487 Brazilian radar plates (LPLCv2) | OCR-difficulty labels from two cross-fitted readers on 7,024 Bangla crops (All-in-One v1.1) |
| 5-bin, 10-fold 40 / 20 / 40 protocol, mean of 10 folds | the team's grouped 70 / 15 / 15 split (seed 42), one test split |
| Input 90 × 180 (code), batch 64, up to 1,000 epochs | 96 × 192, batch 64, up to 80 epochs (all runs stopped at 21-24) |
| EMA run in the code: class-balanced sampler, 20 epochs, patience 5; CE runs: learning rate 1e-4 | EMA and CE with the same learning rate 1e-5 and plain shuffling (the loss is the only change); the balanced sampler is a local extra run |
| Scores: micro F1 per scenario | the same, plus the gate test the paper does not do (selective plate exact, AURC, AUROC, bootstrap) |

Paper facts we found while checking (details in the PDF, sections 3 and 5): the text says the EMA loss gives "a higher
weight for classes with higher error counts", but Eq. 1 and the authors' code give the class with the most errors the
**lowest** weight (max(err) / err, capped at 1.2); Eq. 2 is printed with `confusion(b)` twice; the cited source for the
EMA loss (ref. [31]) is about averaging network weights; the abstract says 89.5 % and Table IV 89.8 %; in the public code
the "+ EMA" configuration also turns on a class-balanced sampler and uses a 10x lower learning rate than the
cross-entropy configuration, so the +1.1 point gain is not a clean ablation of the loss.

---

## 9. File Integrity

| File | SHA-256 |
|---|---|
| `best_cls_b4_ce_fp16.pt` | `20f7db692e02573ca4f02aef0ab508411b8ccd40a3b41b2f2ff222c6cfbcf14b` |
| `paper19_results.zip` | `53a96933603ede7232131448ac64a3b093058753e1aa9eeef3f8e818eb59bb59` |
| `LPLCv2_Study_Report.pdf` | `ef7a92a7d55f5bdfa4d34742bcfdec2ec97baedd2313eb39be4e830ab0622218` |
| CRNN (in `weights/paper24/…/best_crnn_fp16.pt`) | `cc70d5ad43b7e031e709d7c7c951c6450779a956a135199af904bae97d338022` |

Check on Windows: `certutil -hashfile best_cls_b4_ce_fp16.pt SHA256` · Linux/macOS: `sha256sum best_cls_b4_ce_fp16.pt`.

---

## 10. Licence and Data Notes

- The LPLCv2 code (github.com/UFPR-IPASP-PR/LPLCv2-Dataset, checked at commit `7a1d3f6`) has no licence file; it is
  not redistributed here. The kit re-implements the classifier and the EMA loss (`p19_legibility.py`).
- LPLCv2 data was not used (available only on e-mail request).
- The classifier and the readers were trained by us on the 498R All-in-One dataset v1.1 (sources and their licences in
  the dataset README). torchvision ImageNet weights: BSD-3-Clause.

---

## 11. Citation

```bibtex
@inproceedings{wojcik2026lplcv2,
  title     = {{LPLCv2}: An Expanded Dataset for Fine-Grained License Plate Legibility Classification},
  author    = {Wojcik, Lucas and Machoski, Eduardo A. F. and Nascimento Jr., Eduil and Laroca, Rayson and Menotti, David},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026},
  note      = {arXiv:2604.08741}
}
```
