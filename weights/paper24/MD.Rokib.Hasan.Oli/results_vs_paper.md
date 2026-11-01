# Paper #24: ours vs. the paper

| Metric | Paper (page) | chars_v8n | slot | crnn | vitstr | fusion |
|---|---|---|---|---|---|---|
| Single model plate exact (%) | best single 92.4 (ViTSTR-Base, intra) (p.9, Table 3) | 86.79 | 57.60 | 73.43 | 20.89 | – |
| Single model ms per plate | - | 9.36 | 1.15 | 6.47 | 10.18 | – |
| Best fusion of the paper's rules (chosen on val), test exact (%) | 97.6 (MV-HC, top 8) (p.9, Table 3) | – | – | – | – | 86.33 |
| Paper-rule fusion vs best single: exact McNemar p | - | – | – | – | – | 0.58 |
| Best fusion, any rule incl. MV-BM+format / fallback (chosen on val), test exact (%) | not in the paper | – | – | – | – | 88.63 |
| Chosen fusion rule | - | – | – | – | – | fallback |
| Gain over the best single model (pp) | +5.2 intra, +2.7 cross (p.9-11) | – | – | – | – | 1.84 |
| Plates the chosen fusion fixes vs the best single | - | – | – | – | – | 12 |
| Plates it breaks vs the best single | - | – | – | – | – | 0 |
| Chosen fusion vs best single: exact McNemar p | - | – | – | – | – | 0.0005 |
| Fusion ms per plate (sum; fallback: average) | 59.7 (top 8, RTX 8000) (p.12, Table 6) | – | – | – | – | 10.54 |
| Fallback: test plates that run the helpers (%) | - | – | – | – | – | 7.07 |
| Oracle: any model right (%) | - | – | – | – | – | 90.02 |
| Training time (min) | not reported | 52.10 | 23.70 | 23.90 | 24.50 | – |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| chars_v8n | 86.79 | 90.83 | 80.00 | 74.11 | 80.77 | 75.88 |
| crnn | 73.43 | 77.50 | 61.54 | 64.29 | 50.00 | 67.13 |
| slot | 57.60 | 62.29 | 27.69 | 54.46 | 42.31 | 51.46 |
| vitstr | 20.89 | 23.33 | 4.62 | 19.64 | 0.00 | 17.97 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
