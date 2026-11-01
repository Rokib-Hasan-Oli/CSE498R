# Paper #23: ours vs. the paper

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
