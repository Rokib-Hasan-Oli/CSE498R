# Paper #36: ours vs. the paper

| Metric | Paper (page) | det_v8s_640 | det_v8s_sf | det_v8s_1280 | crnn | sahi |
|---|---|---|---|---|---|---|
| Test mAP50, Ultralytics val on the full photo (%) | - | 98.5 | 97.5 | 98.0 | – | – |
| AP50, 640 model, full image (FI) (%) | TOOD + FI 29.4 (VisDrone) (p.3, Table 1) |  |  |  |  | 97.33 |
| AP50, 640 model, SAHI slices only (%) | TOOD + SAHI + PO 32.5 (p.3, Table 1) |  |  |  |  | 88.78 |
| AP50, 640 model, SAHI + FI (%) | TOOD + SAHI + FI + PO 34.7 (p.3, Table 1) |  |  |  |  | 91.63 |
| AP50, SF model, full image (%) | TOOD + SF + FI 36.8 (p.3, Table 1) |  |  |  |  | 96.65 |
| AP50, SF model, SAHI + FI (%) | TOOD + SF + SAHI + FI + PO 43.5 (p.3, Table 1) |  |  |  |  | 93.62 |
| AP50, 1280 px model, full image (%) | not in the paper |  |  |  |  | 97.10 |
| AP50 small plates: 640 FI (%) | TOOD + FI 18.1 (p.3, Table 1) |  |  |  |  | 26.01 |
| AP50 small plates: 640 SAHI + FI (%) | TOOD + SAHI + FI + PO 23.8 (p.3, Table 1) |  |  |  |  | 16.60 |
| AP50 small plates: SF SAHI + FI (%) | TOOD + SF + SAHI + FI + PO 31.7 (p.3, Table 1) |  |  |  |  | 12.58 |
| AP50 small plates: 1280 FI (%) | - |  |  |  |  | 16.07 |
| AP50 large plates: 640 FI / SAHI + FI (%) | TOOD 50.0 -> 50.3 (FI helps large objects) (p.3-4, Table 1) |  |  |  |  | 93.94 |
| Plates found at conf 0.25: 640 FI (%) | - |  |  |  |  | 98.12 |
| Plates found: 640 SAHI + FI (%) | - |  |  |  |  | 97.23 |
| Plates found: SF SAHI + FI (%) | - |  |  |  |  | 97.23 |
| Plates found: 1280 FI (%) | - |  |  |  |  | 97.63 |
| Small plates found: 640 FI / SAHI + FI gain (pp) | - |  |  |  |  | 40.00 |
| Small plates found: 1280 FI (%) | - |  |  |  |  | 60.00 |
| Boxes correct (precision) at conf 0.25: 640 SAHI + FI (%) | slices add false positives (p.4) (p.4) |  |  |  |  | 60.72 |
| False alarms on plate-free photos: 640 SAHI + FI (%) | - |  |  |  |  | 20.00 |
| ms per photo: 640 FI | - |  |  |  |  | 33.80 |
| ms per photo: 640 SAHI + FI | "increases linearly" with the slices (p.4) |  |  |  |  | 118.20 |
| ms per photo: 1280 FI | - |  |  |  |  | 46.10 |
| e2e photos plate exact: 640 FI (%) | - |  |  |  |  | 72.06 |
| e2e photos plate exact: 640 SAHI + FI (%) | - |  |  |  |  | 68.14 |
| e2e photos plate exact: 1280 FI (%) | - |  |  |  |  | 72.55 |
| Detector FPS, T4 TensorRT FP16 (640 / 1280 model) | - | 127.90 | – | 83.20 | – | – |
| Training time (min) | not reported (SGD lr 0.01, decay at epochs 16 and 22) (p.4) | 145.60 | 137.20 | 134.80 | 34.30 | – |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's numbers are AP50 on VisDrone test-dev (drone photos, 10 classes, mmdetection FCOS / VFNet / TOOD); ours are one class (plate) on our test photos with YOLOv8s. Read the gains (FI -> SAHI -> SF), not the absolute values. AP50 here = pycocotools on the sahi predictions (conf 0.01, max 500 boxes); 'plates found' = conf 0.25, IoU 0.5 as in every kit.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| det_v8s_1280 | 98.0 | 98.4 | 94.3 | 97.7 | 91.4 | 76.4 | 97.6 | 5.62 |
| det_v8s_640 | 98.5 | 99.0 | 94.3 | 95.2 | 84.5 | 82.5 | 97.9 | 5.62 |
| det_v8s_sf | 97.5 | 97.5 | 96.5 | 97.4 | 96.3 | 85.5 | 97.8 | 6.25 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| crnn | 73.43 | 77.50 | 61.54 | 64.29 | 50.00 | 67.13 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
