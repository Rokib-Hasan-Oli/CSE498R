# Paper #2: ours vs. the paper

| Metric | Paper (page) | lpd_v12s | lpnr_v12n | cascade_ours | cascade_blpd_weights |
|---|---|---|---|---|---|
| LPD precision | 0.986 (p.23, Table 7) | 96.7 | – | – | – |
| LPD recall | 0.965 (p.23, Table 7) | 96.2 | – | – | – |
| LPD mAP50 | 0.963 (p.23, Table 7) | 98.9 | – | – | – |
| LPD mAP50-95 | 0.685 (p.23, Table 7) | 46.2 | – | – | – |
| LPNR mAP50 | 0.949 (p.25, Table 9) | – | 97.1 | – | – |
| LPNR mAP50-95 | 0.767 (p.25, Table 9) | – | 66.6 | – | – |
| OCR test crops: plate exact (%) | - | – | 87.71 | – | 74.35 |
| Cascade: plate found (%) | 100 (p.34, Table 10) | – | – | 93.14 | 88.24 |
| Cascade: full-plate exact (%) | 95.33 (p.34, Table 10) | – | – | 85.29 | 78.43 |
| Cascade: char-wise accuracy (%) | 98.82 (p.34, Table 10) | – | – | 92.78 | 88.18 |
| Cascade: Levenshtein similarity (%) | 99.57 (p.34, Table 10) | – | – | 90.84 | 84.65 |
| Cascade ms per photo (mean) | 117.16 (RTX 3050) (p.34, Table 11) | – | – | 93.30 | 96.40 |
| Without the vehicle stage: exact (%) | - | – | – | 86.76 | 77.94 |
| LPD FPS, T4 TensorRT FP16, batch 1 | - | 131.60 | – | – | – |
| Training time (min) | LPD 145, LPNR 261 (Colab T4) (p.17-18, p.24) | 178.40 | 94.40 | – | – |

Speed rows re-read from results.json by add_results.py: this zip comes from a kit before 22 Oct 2026, whose table showed the PyTorch FP16 mean timed without warming up every image shape (up to 8x too slow; paper #2: 7.1 FPS vs TensorRT 131.6). (fp32) = PyTorch FP32, the run has no TensorRT number; – = not measured (old kits had no FP16 median).

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

`cascade_blpd_weights` -> `e2e.*.by_source.blpd_ext` is the paper's own external set read with the paper's own weights (direct check of Table 10). Note: BLPD's external photos are near-duplicates of public LPDB-A images.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| lpd_v12s | 98.9 | 99.2 | 98.4 | 99.2 | 93.2 | 83.1 | 97.5 | 10.00 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| lpnr_v12n | 87.71 | 90.42 | 83.08 | 79.46 | 76.92 | 74.50 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
