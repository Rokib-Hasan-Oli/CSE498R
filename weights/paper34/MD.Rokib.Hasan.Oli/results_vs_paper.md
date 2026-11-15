# Paper #34: ours vs. the paper

| Metric | Paper (page) | svtrv2_t_msr | svtrv2_s_msr | svtrv2_t_fixed | svtrv2_t_scratch | crnn |
|---|---|---|---|---|---|---|
| Plate exact (%) | word accuracy, Common benchmarks: T 94.78, S 96.13 (English) (p.7, Table 3) | 81.26 | 81.87 | 74.50 | 61.75 | 73.43 |
| CER (%) | - | 2.49 | 3.07 | 4.47 | 8.55 | 5.53 |
| Token accuracy (%) | - | 97.16 | 97.12 | 95.49 | 92.47 | 95.05 |
| Two-line plates (48x96 MSR size): plate exact (%) | MSR vs fixed 32x128: +5.2 (1.5 <= R < 2.5) (p.6, Table 1) | 84.54 | 86.12 | – | 70.35 | – |
| Near-square plates (64x64 MSR size): plate exact (%) | MSR vs fixed 32x128: +15.3 (R < 1.5) (p.6, Table 1) | 78.42 | 78.12 | – | 53.50 | – |
| Fixed 32x128 (all plates): plate exact (%) | - | 100.00 | 0.00 | 74.50 | 0.00 | – |
| Night crops: plate exact (%) | - | 69.23 | 67.69 | 52.31 | 40.00 | 61.54 |
| ms per plate (T4, batch 1) | FPS: T 201, S 189, CRNN 172 (1080Ti) (p.7, Table 3) | 7.62 | 15.34 | 14.15 | 14.99 | 4.58 |
| Params (M) | T 5.1, S 11.3, CRNN 16.2 (p.7, Table 3) | 5.13 | 11.25 | 5.13 | 5.13 | 7.69 |
| Start weights | trained on U14M-Filter (English) (p.5) | best.pth: encoder 164 of 164 tensors, FRM head 14 of 16 (cla | best.pth: encoder 164 of 164 tensors, FRM head 14 of 16 (cla | best.pth: encoder 164 of 164 tensors, FRM head 14 of 16 (cla | from scratch | – |
| Epochs run | 20 (English, batch 1024) (p.5) | 63 | 66 | 64 | 65 | – |
| Training time (min) | not reported (4 x RTX 4090) (p.5) | 96.30 | 96.90 | 96.10 | 96.30 | 16.50 |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper reads English words (94 characters, millions of real training crops); we read Bangla plate tokens with about 5.6 k training crops. The semantic guidance module (SGM) is not trained here (plate text has almost no language context; see the notes). Compare svtrv2_t_msr with svtrv2_t_fixed for MSR, with svtrv2_t_scratch for the English pre-training, and every run with crnn and with PARSeq (#31).

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| crnn | 73.43 | 77.50 | 61.54 | 64.29 | 50.00 | 67.13 |
| svtrv2_s_msr | 81.87 | 84.58 | 67.69 | 79.46 | 57.69 | 76.80 |
| svtrv2_t_fixed | 74.50 | 79.17 | 52.31 | 67.86 | 53.85 | 66.97 |
| svtrv2_t_msr | 81.26 | 85.00 | 69.23 | 73.21 | 57.69 | 75.27 |
| svtrv2_t_scratch | 61.75 | 66.46 | 40.00 | 54.46 | 26.92 | 59.60 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
