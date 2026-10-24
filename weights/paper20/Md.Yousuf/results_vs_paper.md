# Paper #20: ours vs. the paper

| Metric | Paper (page) | sr_x2 | crnn_hr | sr_eval |
|---|---|---|---|---|
| OCR on HR plates (%) | 85.2 (p.8, Table 2) | – | – | 65.28 |
| OCR on LR plates (%) | 2.2 (p.8, Table 2) | – | – | 16.90 |
| OCR on bicubic (%) | - | – | – | 23.81 |
| OCR on one SR image (%) | PLNET 29.9 (p.8, Table 2) | – | – | 24.12 |
| OCR on Real-ESRGAN (%) | 20.2 (p.8, Table 2) | – | – | 12.29 |
| 5 SR copies + MVCP (%) | PLNET 40.9, LCDNet 42.3 (p.9, Table 3) | – | – | 28.26 |
| All but one token right, SR + MVCP (%) | 86.9 (>= 5 of 7, LCDNet) (p.9) | – | – | 50.23 |
| SR ms per plate (ours) | PLNET 24.88 (RTX 8000) (p.10-11, Table 6) | – | – | 5.50 |
| SR ms per plate (Real-ESRGAN) | 34.86 (p.10-11, Table 6) | – | – | 51.27 |
| Val PSNR SR / bicubic (dB) | - | 23.73 | – | – |
| Training time (min) | not reported | 21.50 | 19.30 | – |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

Our LR plates are synthetic (the paper warns this overstates SR gains, p.8); the 5 copies are differently degraded copies of one photo, not 5 video frames.

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| crnn_hr | 74.04 | 78.12 | 67.69 | 60.71 | 61.54 | 66.67 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
