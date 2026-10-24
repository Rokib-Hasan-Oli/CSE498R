# Paper #6: ours vs. the paper

| Metric | Paper (page) | det_v8n | det_v8s | easyocr_crops | e2e_v8n | e2e_v8s |
|---|---|---|---|---|---|---|
| Precision (%) | 93.5 (p.9, Table 1) | 96.6 | 96.2 | – | – | – |
| Recall (%) | 92.3 (p.9, Table 1) | 94.1 | 94.7 | – | – | – |
| mAP50 (%) | 94.0 ("mAP"; = mAP@50 in Fig. 6) (p.9, Table 1) | 97.4 | 97.6 | – | – | – |
| mAP50-95 (%) | about 59 (read off Fig. 6) (p.11, Fig. 6) | 45.0 | 45.9 | – | – | – |
| Plates found (%), conf 0.25, IoU 0.5 | 94.8 "detection accuracy" (not defined) (p.9, Table 1) | 96.44 | 97.13 | – | – | – |
| Detector FPS, T4 PyTorch FP32, batch 1 | 78 (GPU and precision not named) (p.12, Table 3) | 113.80 | 109.60 | – | – | – |
| Detector FPS, T4 TensorRT FP16, batch 1 | - | 182.90 | 162.50 | – | – | – |
| Training time (min) | not reported | 130.90 | 144.50 | – | – | – |
| EasyOCR alone (human crops): char accuracy (%) | 92.7 ("character recognition accuracy", not defined) (p.11) | – | – | 60.45 | – | – |
| EasyOCR alone: plate exact (%) | not reported | – | – | 13.52 | – | – |
| EasyOCR alone: CER (%) | not reported | – | – | 39.55 | – | – |
| EasyOCR + preprocessing (4x, grey, CLAHE): char accuracy (%) | "preprocessing", no details (p.8) | – | – | 41.80 | – | – |
| EasyOCR, clear plates: char accuracy (%) | 97.5 (clean plates) (p.12, Table 2) | – | – | 66.84 | – | – |
| EasyOCR, small plates (low_res tag): char accuracy (%) | 92.0 (low resolution) (p.12, Table 2) | – | – | – | – | – |
| EasyOCR, night: char accuracy (%) | 91.5 (low light; metric not named) (p.13, Table 4) | – | – | 49.69 | – | – |
| EasyOCR, blurry: char accuracy (%) | - | – | – | 38.74 | – | – |
| EasyOCR, real fog copies: char accuracy (%) | 89.3 (bad weather) (p.12, Table 2) | – | – | 59.53 | – | – |
| EasyOCR, synthetic rain: char accuracy (%) | 89.3 (rain; metric not named) (p.13, Table 4) | – | – | 44.04 | – | – |
| EasyOCR ms per plate (T4) | not reported | – | – | 90.86 | – | – |
| Whole system (e2e photos): plate found (%) | - | – | – | – | 94.61 | 96.08 |
| Whole system: char accuracy (%) | - | – | – | – | 58.34 | 57.64 |
| Whole system: plate exact (%) | - | – | – | – | 11.76 | 10.29 |
| Whole system: ms per photo (detector + EasyOCR) | - | – | – | – | 68.70 | 69.50 |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's 2,600 photos are not public and were augmented BEFORE the split (p.7), so its test set can hold copies of training photos; ours is the grouped All-in-One test split with human boxes. Char accuracy = 100 - CER after normalisation (the paper never defines its metric). Its case groups (clean / low-res / damaged / weather) are matched to our closest condition tags; 'damaged' has no counterpart.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| det_v8n | 97.4 | 97.7 | 94.3 | 98.6 | 94.8 | 77.5 | 94.8 | 6.25 |
| det_v8s | 97.6 | 98.0 | 93.4 | 98.4 | 85.9 | 80.4 | 93.2 | 6.25 |

## Results by condition: plate readers (plate exact %)

| Run | all test | clear | night | blurry | fog (real fog copies) | rain (synthetic) |
|---|---|---|---|---|---|---|
| easyocr_crops | 13.52 | 17.71 | 1.54 | 1.79 | 15.38 | 5.68 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
