# Paper #28: ours vs. the paper

| Metric | Paper (page) | v9t | v9s | v8n | v8s |
|---|---|---|---|---|---|
| mAP50 (%) | COCO AP50: v9-S 63.4 (p.7, Table 1) | 98.3 | 98.8 | 98.7 | 98.5 |
| mAP50-95 (%) | COCO: v9-S 46.8, v8-S 44.9 (p.7, Table 1) | 45.2 | 46.3 | 45.2 | 45.8 |
| Small plates: recall (%) | - | 80.00 | 60.00 | 60.00 | 40.00 |
| Params (M) | v9-S 7.1 (p.7, Table 1) | 2.01 | 7.29 | 3.01 | 11.14 |
| GFLOPs | v9-S 26.4 (p.7, Table 1) | 7.80 | 27.40 | 8.20 | 28.60 |
| T4 TensorRT FP16 ms | not reported | 6.93 | 7.24 | 6.38 | 7.52 |
| T4 PyTorch FP16 ms | not reported | 31.56 | 27.28 | 16.05 | 23.22 |
| CPU ONNX ms | not reported | 116.20 | 296.93 | 105.83 | 300.71 |
| Training time (min) | not reported | 193.00 | 149.00 | 101.00 | 119.40 |

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| v8n | 98.7 | 99.0 | 94.5 | 97.8 | 92.8 | 83.8 | 96.3 | 6.88 |
| v8s | 98.5 | 99.0 | 94.3 | 95.2 | 84.5 | 82.5 | 97.9 | 5.62 |
| v9s | 98.8 | 99.0 | 96.8 | 98.0 | 90.8 | 77.0 | 96.6 | 6.25 |
| v9t | 98.3 | 97.9 | 97.5 | 98.1 | 93.2 | 88.3 | 95.0 | 5.62 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
