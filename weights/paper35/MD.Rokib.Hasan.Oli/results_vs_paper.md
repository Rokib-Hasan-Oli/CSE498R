# Paper #35: ours vs. the paper

| Metric | Paper (page) | det_v8s | crnn | lowlight |
|---|---|---|---|---|
| Night photos: mAP50, no enhancement (%) | - |  |  | 94.4 |
| Night photos: mAP50 after HVI-CIDNet generalization (%) | - |  |  | 95.7 |
| Night photos: mAP50 after HVI-CIDNet LOLv2-Real (%) | - |  |  | 94.7 |
| Night photos: mAP50 after CLAHE (%) | - |  |  | 96.1 |
| Night photos: mAP50 after gamma 0.5 (%) | - |  |  | 93.8 |
| Gain on night photos, HVI generalization (pp) | - |  |  | 1.25 |
| Night photos: plates found after HVI (%) | - |  |  | 90.14 |
| Whole test with the dark gate: mAP50 gain, HVI (pp) | - |  |  | -0.0100 |
| Clear photos: mAP50 after HVI (does it hurt day?) (%) | - |  |  | 99.0 |
| Night crops: plate exact, no enhancement (%) | - |  |  | 61.54 |
| Night crops: plate exact after HVI (%) | - |  |  | 55.38 |
| Gain on night crops, HVI (pp) | - |  |  | -6.16 |
| HVI-CIDNet ms per 640x640 photo (T4) | 53 ms per LOLv2 image (GPU not named) (p.7, Table 3) |  |  | 154.87 |
| HVI-CIDNet ms per plate crop (T4) | - |  |  | 37.55 |
| HVI-CIDNet params (M) | 1.88 (7.57 GFLOPs at 256x256) (p.6, Table 1) |  |  | 1.98 |
| Plate detector mAP50, all test (%) | - | 98.5 |  |  |
| CRNN plate exact, all test (%) | - |  | 73.43 |  |
| LOLv2-Real PSNR / SSIM (paper only, see the check) | 24.111 / 0.871 (p.6, Table 1) |  |  |  |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper scores image quality (PSNR / SSIM against a well-lit photo of the same scene) on LOL; we have no well-lit twin of our night photos, so we score what we care about: plates found and plates read.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| det_v8s | 98.5 | 99.0 | 94.3 | 95.2 | 84.5 | 82.5 | 97.9 | 5.62 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| crnn | 73.43 | 77.50 | 61.54 | 64.29 | 50.00 | 67.13 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
