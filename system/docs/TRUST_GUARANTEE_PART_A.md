# Part A: how far can an operator trust a V2 reading?

**Run:** 16 Nov 2026 on this PC, CPU only. **Script:** `eval/trust_guarantee.py`. **Input:** the saved per-reader
predictions in `eval/preds/` (val 766 crops, test 651 crops, dataset v1.1). No model was trained or re-run.
**Every number below is in** `eval/results/trust_guarantee.json`. Picture: `eval/results/trust_risk_coverage.png`.

## Short answer

1. V2's own rule is a good gate at the 95 % level: **94.93 % of test plates are "verified" and 95.31 % of them are
   right** (95 % interval 93.33-96.84; val: 92.82 % verified, 93.95 % right).
2. **A guarantee of 97 %, 98 % or 99 % cannot be certified with the plates we have.** This is a negative result.
3. The usual way of picking a threshold (look at the accuracy on a calibration set) **misses its own target in about
   half of the random splits** (45-67 %). That is a useful warning for anyone who deploys a plate reader.
4. One reason for point 2 is the **dataset labels**: when all five readers agree and the label says something else,
   the picture usually shows that the label is wrong (see section 4). The labels need a human audit.

## 1. Risk-coverage on the test crops

Plates are accepted from the most trusted to the least trusted. "Exact" = accepted plates that are fully right.

| Trust score | Extra cost | Exact at 50 % accepted | 80 % | 90 % | 95 % | 100 % | AURC (lower is better) |
|---|---|---|---|---|---|---|---|
| One reader + its own confidence (char-YOLOv8n) | none | 97.57 | 95.97 | 95.05 | 90.63 | 86.18 | 2.87 |
| V2: readers agree, then mean confidence | none (V2 runs these anyway) | 97.55 | 96.55 | 95.90 | 95.15 | 91.71 | 2.50 |
| V2 signals, logistic model fitted on val | none | 97.55 | 96.55 | 96.76 | 95.15 | 91.71 | 2.25 |
| All 5 readers always run: votes, then confidence | about 3x reader time | 96.63 | 97.70 | 97.10 | 95.15 | 91.71 | 2.68 |
| 5-reader signals, logistic model fitted on val | about 3x reader time | 96.63 | 97.70 | 96.93 | 95.32 | 91.71 | 2.24 |

- Agreement between readers is a better trust signal than one reader's confidence (this repeats the LPLCv2 study's
  finding on the full V2 system).
- No score brings the accepted plates above about 97.7 % at a useful share of plates.
- Even when **all five readers agree**, the plate is "wrong" in 12 of 469 test plates (97.44 % right) and in 16 of
  462 val plates (96.54 %).

## 2. Calibrated review threshold

Target: "accepted plates are at least X % right". Allowed chance that the promise fails: delta = 10 %.

- **certified** = thresholds tested from strict to loose with an exact binomial test, stop at the first failure
  (fixed-sequence testing, as in "Learn then Test"). This was the method written before the first run.
- **grid** = added after the first run, because one early wrong plate stops the first method at once: 19 fixed
  coverage levels, each tested at delta / 19. It did not do better.
- **plain** = the largest accepted set whose calibration accuracy reaches the target (what people usually do).

**Protocol 1: calibrate on val (766), score test (651) once**

| Target | certified / grid | plain |
|---|---|---|
| 95 % | no threshold passes | 90.78 % of test plates accepted, 95.94 % right (24 wrong) |
| 97 % | no threshold passes | 37.33 % accepted, 97.94 % right (5 wrong) |
| 98 %, 99 % | no threshold passes | 1 plate accepted |

**Protocol 2: 2,000 random halves of test (calibrate on about 325, score about 326), score "V2: agree, then confidence"**

| Target | certified: plates accepted (mean) | right (mean) | target missed | plain: accepted (mean) | right (mean) | target missed |
|---|---|---|---|---|---|---|
| 95 % | 27.96 % | 98.73 % | 4.15 % of splits (allowed 10 %) | 94.30 % | 94.94 % | **51.15 %** |
| 97 % | 8.38 % (no threshold in 71.6 % of splits) | 97.18 % | 11.95 % | 65.80 % | 96.70 % | **56.30 %** |
| 98 % | 1.14 % (none in 97.35 %) | – | 2.65 % | 43.03 % | 97.64 % | **67.35 %** |
| 99 % | none | – | – | 25.91 % | 98.51 % | **58.00 %** |

Reading the table: the certified method keeps its promise but accepts few plates (at 95 %) or none (97-99 %).
The plain method accepts many plates but breaks its promise about half of the time.

## 3. Why the higher guarantees fail (my reading of the numbers)

- **Too few plates.** A rough binomial estimate: to certify 97 % when the accepted plates are truly 98 % right needs
  about 480 accepted calibration plates; to certify 99 % the accepted plates would have to be about 99.5 % right, and
  V2 is not there on any useful share of plates. (This is an estimate, not a measured number.)
- **The readers fail together.** The two always-on readers are the same network type trained on the same data, so
  they often make the same mistake.
- **Val is harder than test** (V2: 89.69 % vs 91.71 %; one reader: 80.81 % vs 86.18 %), so a threshold from val is
  not a fair promise for test. A guarantee only holds when calibration plates and field plates are alike.
- **Label noise**, next section.

## 4. Label check of the "all five agree but wrong" plates

I looked at the crops of all 28 such plates (12 test, 16 val) on 16 Nov 2026. This is my reading of the pictures,
**not ground truth; a Bangla reader must confirm each one.**

| | Plates | Label looks wrong | Label probably wrong | Unclear |
|---|---|---|---|---|
| Test | 12 | 6 | 3 | 3 |
| Val | 16 | 8 | 4 | 4 |

Examples where the picture is clear: test `anpr_ed89facc374d.jpg` (label ১১-১৭৩৭, plate shows ১৩-১৭৩৭), test
`anpr_14f5584fb020.jpg` (label ১১-৪৮৭২, plate shows ১১-৮৪৭২), val `anpr_905881ff0ccd.jpg` (label নারায়ণগঞ্জ-থ
১১-৩৫১৩, plate shows ঢাকা মেট্রো-গ ৩১-০৮৯১), val `anpr_b92cdfc27a35.jpg` (label ঢাকা মেট্রো, plate shows চট্ট মেট্রো).

All 133 plates that V2 reads differently from the label (54 test, 79 val) are listed in
`eval/results/label_audit_candidates.csv` with empty columns for two human checkers.

**What this means:** the test labels come from the public source datasets and some are wrong. Every accuracy in the
team's tables is therefore a little too low, and nobody knows by how much until people re-check the labels. Do not
correct only the plates where the readers disagree with the label (that would favour the readers). Two people should
re-read **all** 651 test and 766 val crops without seeing the predictions.

## 5. What the paper can say (and cannot)

- Can say: the "verified" flag accepts about 95 % of plates at about 95 % exact; agreement beats single-reader
  confidence; a plain threshold misses its target in about half of random splits; a certified 95 % guarantee is
  possible only on about 28 % of plates with 325 calibration plates.
- Cannot say: "auto-accepted plates are 99 % right", or any guarantee above 95 %.
- After the label audit: re-run `eval/reader_preds.py`, `eval/select_fusion.py`, `eval/system_eval.py` and
  `eval/trust_guarantee.py`, and report both the old-label and the audited-label numbers.

## Files

| File | What |
|---|---|
| `eval/trust_guarantee.py` | the analysis |
| `eval/results/trust_guarantee.json` | every number above |
| `eval/results/trust_risk_coverage.png` | risk-coverage picture |
| `eval/results/trust_test_plates.csv` | one row per test plate: reading, right/wrong, status, scores |
| `eval/results/trust_wrong_val.csv`, `trust_wrong_test.csv` | the wrong plates, most agreed first |
| `eval/results/label_audit_candidates.csv` | 133 plates for the human label check |
| `eval/results/dataset_check_2026-11-16.json` | split overlap, source mix, class balance |
