# Paper #19: ours vs. the paper

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
