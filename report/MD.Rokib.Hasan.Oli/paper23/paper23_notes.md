# Paper #23: Do We Train on Test Data? The Impact of Near-Duplicates on License Plate Recognition

*Reader: MD. Rokib Hasan Oli · Board group: General LPR state of the art (must-read for the data lead)*
*PDF: `Papers/6_3 General LPR state of the art/23_2023_Do We Train on Test Data Near-Duplicates LPR.pdf`
(arXiv 2304.04653v2, 8 pages; IJCNN 2023, DOI 10.1109/IJCNN54540.2023.10191584). Page numbers below are the PDF pages.*

---

## Do We Train on Test Data? The Impact of Near-Duplicates on License Plate Recognition (2023)
```
Problem:   Popular LPR datasets put different photos of the SAME plate in both train and test, so models are
           partly scored on memory, not generalisation. How big is the effect?
Method:    Find near-duplicates (same plate text, then nearest neighbours by pixel distance of the rectified
           crops) in AOLP and CCPD; build "fair" splits with no test plate in training while keeping the original
           split sizes and character; train 6 OCR models (CNNG, Holistic-CNN, Multi-Task in Keras; STAR-Net, TRBA,
           ViTSTR-Base in PyTorch) on both splits with the same settings + synthetic plates (p.2-4).
Data:      AOLP (2,049 Taiwan images; protocols A = random 2:1, B = train AC+LE / test RP) and CCPD (366,789
           images, mainland China; test ≈ 157 k) (p.2). Duplicates: AOLP-A 320 of 683 test images (46.9 %), AOLP-B
           413 of 611 (67.6 %), CCPD 29,943 of ≈157 k (19.1 %) (p.3); ReId dataset 68.6 % (p.5). Fair splits and
           duplicate lists are public.
Results:   AOLP-A → fair: TRBA 99.18 → 96.94 %, CNNG 98.88 → 95.63 %; error rates more than double (relative gap
           112-290 %) (p.4, Table I). AOLP-B → fair: CNNG 98.91 → 96.80 % and drops from 1st to 3rd place (p.4-5,
           Table II). CCPD → fair: average 80.3 → 77.6 %, STAR-Net −5.20, TRBA −4.35 points (p.5, Table III).
           Hardware: Threadripper 1920X + Quadro RTX 8000 (p.4). Training time not reported.
Limits:    Only OCR (recognition) is studied, not detection; two main datasets; duplicates found by plate text +
           pixel distance, so near-duplicates of DIFFERENT plates (same car, other angle) or unlabelled ones are
           not covered. Fair splits keep sizes but change which images are in train, so part of the drop could be
           harder test sets (they control for this in AOLP-B by only removing test duplicates).
For us:    The rule behind our whole dataset build: group every image by plate identity and by perceptual hash,
           and split by group; we also measure how much leaky public splits (KUET, BLPD) inflate our own scores.
Code:      no training code; duplicate lists + fair splits: https://raysonlaroca.github.io/supp/lpr-train-on-test/
BibTeX:    laroca2023do  (from the authors' page; full entry at the bottom)
```

---

## Pass 1: survey (3 lines)
1. **Problem:** test images that show plates also seen in training inflate LPR results.
2. **How:** find same-plate near-duplicates in AOLP and CCPD, build duplicate-free splits, retrain 6 OCR models on both.
3. **How well (how bad):** error rates more than double on AOLP fair splits; CCPD average 80.3 → 77.6 %; model ranking
   changes (p.4–5).

**Useful for us?** Yes, essential. Peer-reviewed (IJCNN) and directly about our biggest risk: the Bangla public sets
we merged contain Roboflow augmented copies and repeated plates (see "Our own audit" below).

## Pass 2: method and results
| Item | Detail | Page |
|---|---|---|
| Venue | IJCNN 2023 (IEEE) | p.1 |
| Near-duplicate definition | different images of the **same plate**; found via plate text, then the nearest neighbour by Euclidean distance in pixel space of rectified crops | p.2–3 |
| AOLP protocols | A: random 2:1 train/test (+20 % of train for val); B: train AC + LE, test RP | p.2 |
| CCPD protocol | Base 200 k → 100 k train / 100 k val; all other subsets (≈157 k) test | p.2 |
| Fair splits | AOLP-Fair-A: same sizes, same-plate images kept together; AOLP-Fair-B: same train/val, test duplicates removed; CCPD-Fair: duplicates moved from train to val, test unchanged | p.3–4 |
| OCR models | CNNG, Holistic-CNN, Multi-Task (Keras: Adam 1e-3, batch 64, ≤ 100 epochs, patience 7); STAR-Net, TRBA, ViTSTR-Base (PyTorch: Adadelta ρ 0.99, 300 k iterations, batch 128) | p.4 |
| Synthetic data | template plates (Taiwan, mainland China) + augmentation to avoid over-fitting | p.4 |
| Metric | recognition rate (whole plate correct); gap = original − fair; relative gap = gap / (100 % − original) | p.4 |

**Every result number**
| Dataset / protocol | Duplicates in test | Page |
|---|---|---|
| AOLP-A | 320 / 683 = 46.9 % | p.3 |
| AOLP-B | 413 / 611 = 67.6 % | p.3 |
| CCPD (latest) | 29,943 / ≈157 k = 19.1 % | p.3 |
| ReId | 52,394 / 76,412 = 68.6 % | p.5 |

| Model | AOLP-A → Fair | AOLP-B → Fair | CCPD → Fair | Page |
|---|---|---|---|---|
| CNNG | 98.88 → 95.63 | 98.91 → 96.80 | 88.24 → 86.93 | p.4–5, Tables I–III |
| Holistic-CNN | 96.75 → 93.11 | 98.42 → 96.30 | 77.01 → 75.41 | p.4–5 |
| Multi-Task | 97.33 → 93.79 | 98.42 → 95.29 | 83.01 → 81.84 | p.4–5 |
| STAR-Net | 98.69 → 95.83 | 98.47 → 96.46 | 78.53 → 73.33 | p.4–5 |
| TRBA | 99.18 → 96.94 | 98.75 → 97.47 | 75.83 → 71.48 | p.4–5 |
| ViTSTR-Base | 98.74 → 96.94 | 98.75 → 97.31 | 79.06 → 76.37 | p.4–5 |
| CCPD average | – | – | 80.3 → 77.6 | p.5 |

- Worst CCPD subset (Blur): TRBA 50.2 → 38.7 % (p.5, Table IV).
- The 1.17-point gap equals 1,800+ extra misread plates; the 5.2-point gap equals 8,000+ (p.5).

Terms I looked up:
- **Near-duplicate vs. exact duplicate:** an exact duplicate is the same file; a near-duplicate is a different photo
  that looks almost the same after cropping (the same plate a second later, from a slightly different angle, or
  re-encoded / augmented).
- **Relative gap:** how much the error grows; 200 % means the error tripled.
- **Group split:** keep all images of one group (plate, vehicle, video) in one split.

## Pass 3: rebuild it
```
all images ─► group key: plate text (same plate = same group) + perceptual-hash clusters (same photo, re-encoded/augmented)
           ─► split by group (train / val / test) ─► train OCR ─► test
           compare with the "leaky" split (by file, ignoring groups) ─► gap = memorisation effect
```
**Assumptions:** plate text identifies the vehicle; pixel distance after rectification finds the same plate's other
photos; the fair split keeps the same difficulty.

**What would fail or differ on our Bangladeshi data:**
- **Missing text labels.** Many of our detection photos have no plate text (DIU, KUET detection, BD-ALPDR). There,
  plate identity is unknown, so I group by perceptual hash (pHash/dHash) + Roboflow source names (`*_jpg.rf.<hash>`
  copies) instead.
- **Augmented copies.** Roboflow exports put 3–4 augmented copies of one photo in train (KUET, BLPD, lpd4k). These are
  near-duplicates **within** train, and when the base name also appears in val/test, across splits too.
- **Rotated/noised copies** fool a strict pixel distance; perceptual hashes are robust to noise and brightness but
  not to strong crops or rotations. That is why I also use the source file names (Roboflow base, NightLPBD `_C_/_R_/_S_`
  names).

**Our own audit (from the 498R dataset build, 12 Oct 2026):**
- KUET localisation (the paper #1 dataset): its 5,687 train images come from only 1,965 distinct source photos
  (mostly 4 Roboflow copies each, with salt-and-pepper noise and brightness changes), and 62 of its 553 test file
  bases also appear in train.
- BLPD (paper #2): plate train has 1,502 images from 830 bases; its recognition train has 5,058 images from 3,350
  bases, and ≈ 2,000 of them are 2×2 mosaic tiles with the digits masked.
- Exact cross-split near-duplicate counts are in `D:\498R\dataset\reports\leakage_audit.md` (written by the dataset build).

**How I reproduce it on our data:** the same detector and the same OCR, trained twice:
- **Leaky split:** each source's own published split with all its augmented copies (the way papers #1 and #2 trained).
- **Fair split:** our grouped 70/15/15 split (no duplicate across splits).
- **GPU 0:** YOLOv8n plate detector, leaky vs fair (two runs one after the other). **GPU 1:** character-YOLO (YOLOv8n)
  OCR, leaky vs fair.
- **Scoring:** both test sets, plus the fair test set for both models. "Gap" and "relative gap" as in the paper. See
  the [Kaggle guide](../../../src/MD.Rokib.Hasan.Oli/paper23/Kaggle_Training_Guide.md).

## Check yourself (2-minute version)
> In AOLP and CCPD, many test plates also appear (as other photos) in training: 47–68 % of AOLP test images and 19 % of
> CCPD's. Re-splitting so no plate is in both sets makes six OCR models' error rates double on AOLP and drops CCPD's
> average from 80.3 % to 77.6 %; the "best" model even changes. Lesson for us: split by plate / photo group, never by
> file, or our ≥ 95 % numbers will be inflated. Our own audit already finds 4× augmented copies in the Bangla public sets.

## Team literature table row
| Paper | Year | Venue | Dataset (size) | Detector | OCR method | Metrics | Code? | Limitation | How it helps us |
|---|---|---|---|---|---|---|---|---|---|
| Do We Train on Test Data? The Impact of Near-Duplicates on LPR (Laroca et al.) | 2023 | IJCNN 2023 (IEEE) | AOLP 2,049; CCPD 366,789 (+ ReId, EnglishLP, PKU, CLPD checks) | – (crops rectified) | 6 OCR models (CNNG, Holistic-CNN, Multi-Task, STAR-Net, TRBA, ViTSTR) | Dup. in test 46.9 / 67.6 / 19.1 % (p.3); AOLP error ×2+; CCPD 80.3 → 77.6 % (p.5) | Splits + duplicate lists (no code) | OCR only; plate-text-based duplicates; 2 main datasets | Basis of our grouped, leakage-free split; leaky-vs-fair experiment on Bangla data |

## Code check → Research board
- **Official material:** https://raysonlaroca.github.io/supp/lpr-train-on-test/ has `duplicates.zip` (near-duplicate lists)
  and `splits.zip` (original + fair splits for AOLP and CCPD). There is no training code. The OCR models come from
  https://github.com/roatienza/deep-text-recognition-benchmark (PyTorch) and the authors' Keras models.
- **What I built for us:** the de-duplication + grouped split in the 498R dataset build
  (`D:\498R\_dataset_build\scripts\dedupe_split.py`): md5 + pHash/dHash near-duplicates, Roboflow base names, plate
  keys. It also writes the leakage audit of the public sources.

Paste into the board (Code link + Short note, ≤ 300 chars):
```
https://raysonlaroca.github.io/supp/lpr-train-on-test/
```
```
Fair splits + duplicate lists (AOLP/CCPD), no code. Applied to our data: grouped split (pHash + Roboflow base + plate key), leakage audit of KUET/BLPD; T4x2 kit trains leaky vs fair (YOLOv8n det + char-YOLO). Paper: CCPD 80.3→77.6 % (p.5).
```

## BibTeX
```bibtex
@inproceedings{laroca2023do,
  title     = {Do We Train on Test Data? {T}he Impact of Near-Duplicates on License Plate Recognition},
  author    = {Laroca, Rayson and Estevam, Valter and Britto Jr., Alceu S. and Minetto, Rodrigo and Menotti, David},
  booktitle = {International Joint Conference on Neural Networks (IJCNN)},
  pages     = {1--8},
  year      = {2023},
  doi       = {10.1109/IJCNN54540.2023.10191584}
}
```
*From the authors' own BibTeX file (raysonlaroca.github.io/bibtex/laroca2023do.txt).*

## Settings I trained with
<!-- settings:start -->
| Item | Setting |
|---|---|
| Notebook | [`kaggle/paper23_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper23/kaggle/paper23_kaggle.ipynb), steps in [Kaggle_Training_Guide.md](../../../src/MD.Rokib.Hasan.Oli/paper23/Kaggle_Training_Guide.md) |
| Data | 498R BanglaLPR All-in-One v1.1 (downloaded by the notebook from the team Drive, or the Kaggle dataset `498r-bangla-lpr-all-in-one`): detection 8574 / 896 / 1153 photos (train: 4067 human + 4507 auto boxes); character boxes 5465 / 536 / 467 crops; OCR text 5607 / 766 / 651 crops; e2e test 204 photos |
| Split | team rule: grouped 70/15/15, seed 42, no near-duplicate across splits; val/test = human labels only |
| Hardware | Kaggle GPU T4 x2: one job per GPU at the same time (no DataParallel), CPU numbers on the Kaggle CPU |
| Software | ultralytics==8.4.170; PyTorch, CUDA and all versions are saved in `environment.json` and each `results.json` |
| Seed / optimiser | seed 42; YOLO runs use SGD (Ultralytics 8.4's `auto` would pick MuSGD), other settings in the table below |
| Time cap | every training job stops after the epoch that passes its `--max-hours` cap (12 h Kaggle limit); `epochs_run` in results.json shows if it did |
| Planned time | 2.5 h for the full run |

| Run | GPU | What |
|---|---|---|
| `det_leaky` | 0 | YOLOv8n plate detector on the published KUET split (with copies), 50 epochs |
| `det_fair` | 0 (after) | the same on our grouped KUET split, ~160 epochs (same optimiser steps) |
| `rec_leaky` | 1 | YOLOv8n character detector, published split, 80 epochs |
| `rec_fair` | 1 (after) | our split, ~290 epochs (same optimiser steps) |
| `leaky_vs_fair` | 0 (after) | all models x all test sets, gaps, seen/unseen groups |

The exact arguments of every run are in its `results.json` (`settings`) and, for YOLO runs, `args.yaml` inside
`paper23_outputs.zip` (team Drive `498R / weights / paper23 /`).
<!-- settings:end -->

## Results: ours vs. the paper
<!-- results:start -->
### Paper #23: ours vs. the paper

| Metric | Paper (page) | det_leaky | det_fair | rec_leaky | rec_fair | leaky_vs_fair |
|---|---|---|---|---|---|---|
| Detector: published test, mAP50-95 (%) | - |  |  |  |  | 76.86 |
| Detector: clean part of the published test (%) | - |  |  |  |  | 76.04 |
| Detector gap all - clean (pp) | - |  |  |  |  | 0.82 |
| Detector: fair model on our fair test, mAP50-95 (%) | - |  |  |  |  | 73.31 |
| OCR: leaky model on the published test, plate exact (%) | AOLP-A TRBA 99.18 (original split) (p.4, Table I) |  |  |  |  | 85.57 |
| OCR: fair model on our fair test, plate exact (%) | AOLP-A TRBA 96.94 (fair split) (p.4, Table I) |  |  |  |  | 77.84 |
| OCR gap: leaky vs fair, each on its own test (pp) | 1.8-3.6 (AOLP-A), 1.3-3.1 (AOLP-B), 1.2-5.2 (CCPD) (p.4-5, Tables I-III) |  |  |  |  | 7.73 |
| OCR relative gap of that = gap / (100 - leaky %) (%) | 112-290 (AOLP-A), 102-198 (AOLP-B), 7-24 (CCPD) (p.4-5, Tables I-III) |  |  |  |  | 53.57 |
| OCR: clean part of the published test, leaky model (%) | - |  |  |  |  | 78.75 |
| OCR gap all - clean, leaky model (pp) | - |  |  |  |  | 6.82 |
| OCR relative gap all vs clean = gap / (100 - all) (%) | 112-290 (AOLP-A, error rate) (p.4) |  |  |  |  | 47.26 |
| OCR fair test, plates neither reader trained on: leaky model (%) | - |  |  |  |  | 78.75 |
| OCR fair test, plates neither reader trained on: fair model (%) | - |  |  |  |  | 75.00 |
| Fair test crops identical to a crop in the leaky train | - |  |  |  |  | 0 |
| Published test crops identical to a crop in the fair train | - |  |  |  |  | 0 |
| Published test photos with no duplicate in train (clean) | - |  |  |  |  | 474 |
| Published test photos with a duplicate in train | - |  |  |  |  | 79 |
| Published test plates with the same plate in train | - |  |  |  |  | 114 |
| Epochs run | - | 50 | 160 | 80 | 377 | – |
| Training time (min) | not reported | 49.80 | 41.10 | 28.50 | 28.30 | – |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

Relative gap as in the paper: gap / (100 - accuracy on the original split), i.e. how much the error rate grows. Cross-scores (a model on the OTHER version's test) include items that model trained on: see "Scores that include training data" and the seen / unseen table at the end. The per-condition reader table below scores the whole 498R OCR test, and both readers trained on part of its 183 KUET crops (rec_leaky 104; from dataset v1.1 also rec_fair 120, as our test crops joined the fair train): use the last table, without the KUET crops, to compare the two readers there.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| det_fair | 99.2 | 95.2 | 72.7 | 78.1 | 82.8 | 74.5 | 88.2 | 2.50 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| rec_fair | 62.37 | 58.54 | 63.08 | 80.36 | 26.92 | 53.76 |
| rec_leaky | 60.68 | 58.33 | 53.85 | 75.89 | 23.08 | 58.37 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.

## Seen / unseen check (plate readers)

The two KUET versions split the crops differently, so a reader scored on the other version's test has trained on some of those crops. Groups per reader: A = the identical crop is in its train, B = the same plate text is in its train (another crop), C = unseen. Plate exact % of both readers on each group; only the rows "unseen by both" compare the two readers fairly.

| Test set | Groups by the train of | Group | Plates | rec_leaky (%) | rec_fair (%) | leaky - fair (pp) |
|---|---|---|---|---|---|---|
| leaky_test_all | rec_leaky | A: identical crop in train | 0 | – | – | – |
| leaky_test_all | rec_leaky | B: same plate text in train | 114 | 90.35 | 79.82 | +10.53 |
| leaky_test_all | rec_leaky | C: unseen | 80 | 78.75 | 75.0 | +3.75 |
| leaky_test_all | rec_fair | A: identical crop in train | 0 | – | – | – |
| leaky_test_all | rec_fair | B: same plate text in train | 0 | – | – | – |
| leaky_test_all | rec_fair | C: unseen | 194 | 85.57 | 77.84 | +7.73 |
| leaky_test_all | both | **unseen by both** | 80 | 78.75 | 75.0 | +3.75 |
| fair_test | rec_leaky | A: identical crop in train | 0 | – | – | – |
| fair_test | rec_leaky | B: same plate text in train | 114 | 90.35 | 79.82 | +10.53 |
| fair_test | rec_leaky | C: unseen | 80 | 78.75 | 75.0 | +3.75 |
| fair_test | rec_fair | A: identical crop in train | 0 | – | – | – |
| fair_test | rec_fair | B: same plate text in train | 0 | – | – | – |
| fair_test | rec_fair | C: unseen | 194 | 85.57 | 77.84 | +7.73 |
| fair_test | both | **unseen by both** | 80 | 78.75 | 75.0 | +3.75 |

## Scores that include training data

- det_leaky on leaky_test_all: 11 identical photos of 553 are in its train (its own published split: the near-duplicates the paper is about)
- det_leaky on fair_test: 20 identical photos of 540 are in its train -> cross-score contaminated, use `unseen`
- rec_leaky on leaky_test_all: 0 identical crops + 114 with the same plate text of 194 are in its train (its own published split: the near-duplicates the paper is about)
- rec_leaky on fair_test: 0 identical crops + 114 with the same plate text of 194 are in its train -> cross-score contaminated, use `unseen`

Cross-scores again on the items the model never trained on (identical photo / crop and, for readers, the same plate text left out):

| Model | Test set | Images | mAP50 (%) | mAP50-95 (%) | plate exact (%) | on the whole set |
|---|---|---|---|---|---|---|
| det_leaky | fair_test | 520 | 99.4 | 76.78 | – | 77.45 (mAP50-95_%) |
| rec_leaky | fair_test | 80 | 92.83 | 71.9 | 78.75 | 85.57 (plate_exact_%) |

## Training length

| Run | train images | epochs run | batch | optimiser steps (about) |
|---|---|---|---|---|
| det_leaky | 5620 | 50 | 16 | 17600 |
| det_fair | 1758 | 160 | 16 | 17600 |
| rec_leaky | 3391 | 80 | 32 | 8480 |
| rec_fair | 660 | 377 | 32 | 7917 |

## Plate readers by condition, without the KUET crops (plate exact %)

Both readers trained on KUET only; with the KUET crops left out (rec_leaky trained on many of them) the rest of the 498R OCR test is a cross-dataset test for both.

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| rec_leaky (no KUET, 468 plates) | 50.21 | 48.15 | 50.85 | 74.19 | 23.08 | 47.44 |
| rec_fair (no KUET, 468 plates) | 50.0 | 47.35 | 59.32 | 64.52 | 26.92 | 42.52 |

<!-- results:end -->

**How to read this table (v1.1 run of 22 Oct 2026, checked the same day).** This note is outside the markers, so `add_results.py` keeps it.
- Both readers are scored on **the same 194 KUET published test plates**. The row label "each on its own test" dates from before
  v1.1. Paired McNemar on those plates: 22 vs 7 discordant plates, **p = 0.008**.
  - Plates the leaky reader had seen (114): 90.35 vs 79.82 %, p = 0.017.
  - Plates unseen by both (80): 78.75 vs 75.00 %, p = 0.45.
  - Other datasets (468): 50.21 vs 50.00 %, p = 1.0.
  - Source: `report_src/paired_v11.py` → `paired_test_v11.json`.
- The per-condition reader table **with** KUET crops is not a fair comparison: both readers trained on part of the main OCR
  test's KUET crops. Use the table "without the KUET crops".
- det_fair (160 epochs) overfits (best epoch 102). The 50-epoch v1.0 fair detector scored 75.93 on clean photos, vs 73.06 here.
  The duplicated-photo advantage of the leaky detector (+14.6) holds in both runs.
- The v1.0 run (18 Oct: 85.57 vs 72.13 % on different test plates) is superseded. Full write-up: `Paper23_Study_Report.pdf`.

## Original-paper check (the paper's own data)
<!-- check:start -->
*Not possible: split lists are public, but AOLP needs a request and CCPD is 12 GB+ of Chinese plates.* The main kit above runs the paper's method on our data instead.
<!-- check:end -->
