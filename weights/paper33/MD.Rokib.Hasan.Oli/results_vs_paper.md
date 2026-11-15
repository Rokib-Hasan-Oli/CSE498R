# Paper #33: ours vs. the paper

| Metric | Paper (page) | y26n | y26s | y11n | y26n_sgd | y26n_p2 | heads_y26n | heads_y26s | heads_y26n_p2 |
|---|---|---|---|---|---|---|---|---|---|
| mAP50 (%) | - | 98.7 | 99.1 | 97.8 | 98.5 | 98.0 | – | – | – |
| mAP50-95 (%) | COCO val: 26n 40.9, 26s 48.6 (p.8, Table 2) | 45.4 | 46.4 | 44.7 | 45.8 | 46.2 | – | – | – |
| mAP50-95, NMS-free one-to-one head (%) | COCO e2e: 26n 40.1, 26s 47.8 (p.8, Table 2) | – | – | – | – | – | 45.0 | 47.1 | 45.4 |
| mAP50-95, one-to-many head + NMS (%) | COCO: 26n 40.9, 26s 48.6 (p.8, Table 2) | – | – | – | – | – | 45.4 | 46.4 | 46.2 |
| NMS-free minus NMS head, mAP50-95 (pp) | -0.8 (26n and 26s, COCO) (p.8, Table 2) | – | – | – | – | – | -0.40 | 0.68 | -0.80 |
| Plates found at conf 0.25 (%) | - | 97.82 | 97.73 | 97.73 | 97.63 | 98.42 | – | – | – |
| Plates found, NMS-free head (%) | - | – | – | – | – | – | 96.83 | 97.53 | 98.12 |
| Small plates: recall (%) | STAL "boosts small objects", no number (p.7) | 40.00 | 20.00 | 60.00 | 60.00 | 80.00 | – | – | – |
| Small plates, NMS-free head: recall (%) | - | – | – | – | – | – | 40.00 | 20.00 | 80.00 |
| False alarms on plate-free photos, NMS-free head (%) | - | – | – | – | – | – | 5.00 | 5.00 | 3.75 |
| T4 PyTorch FP16 ms, median (YOLO26: NMS-free head; YOLO11: with NMS) | - | – | – | 12.06 (fp16) | – | – | 14.26 | 12.86 | 14.83 |
| T4 PyTorch FP16 ms, one-to-many + NMS (median) | - | – | – | – | – | – | 17.65 | 13.23 | 15.27 |
| T4 TensorRT FP16 ms (YOLO26: NMS-free; YOLO11: with NMS) | 26n 1.7, 26s 2.5 (TensorRT 10, model only) (p.8, Table 2) | – | – | 6.28 (tensorrt_fp16) | – | – | 5.55 | 5.78 | 5.85 |
| TensorRT INT8: test mAP50 (%) (YOLO26: NMS-free; YOLO11: with NMS) | "minimal impact on accuracy" (no number) (p.10) | – | – | 96.9 (tensorrt_int8) | – | – | 97.4 | 97.8 | 97.4 |
| TensorRT INT8 ms | - | – | – | 7.77 (tensorrt_int8) | – | – | 5.62 | 5.76 | 5.80 |
| CPU ONNX ms (YOLO26: NMS-free; YOLO11: with NMS) | 26n 38.9, 26s 87.2 (p.8, Table 2) | – | – | 90.98 (cpu_onnx) | – | – | 43.83 | 117.43 | 98.36 |
| Params (M) | 26n 2.4, 26s 9.5 (p.8, Table 2) | 2.50 | 9.95 | 2.59 | 2.50 | 2.52 | – | – | – |
| GFLOPs | 26n 5.4, 26s 20.7 (p.8, Table 2) | 5.90 | 22.70 | 6.50 | 5.90 | 7.60 | – | – | – |
| Best epoch | - | 50 | 49 | 55 | 55 | 55 | – | – | – |
| Training time (min) | not reported | 141.00 | 175.10 | 113.30 | 127.80 | 134.40 | – | – | – |

A value marked with (...) comes from `speed.fp16.ms_median`: that run has no number at `heads.one2one_e2e.pytorch_fp16.ms_median` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.tensorrt_fp16.ms_median`: that run has no number at `heads.one2one_e2e.tensorrt_fp16.ms_median` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.tensorrt_int8.test_mAP50`: that run has no number at `heads.one2one_e2e.tensorrt_int8.test_mAP50` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.tensorrt_int8.ms_median`: that run has no number at `heads.one2one_e2e.tensorrt_int8.ms_median` (e.g. the TensorRT export failed or was not run).

A value marked with (...) comes from `speed.cpu_onnx.ms_median`: that run has no number at `heads.one2one_e2e.cpu_onnx.ms_median` (e.g. the TensorRT export failed or was not run).

The paper used its own data; our numbers are on the 498R All-in-One test split (different data),
so the comparison shows the direction and size of the gap, not a head-to-head result.

The paper's Table 2 is Ultralytics' own COCO table (the paper runs no experiment of its own); its speed column is model-only TensorRT, ours is pre + inference + post at batch 1. In Ultralytics 8.4.170 a YOLO26 model runs its one-to-many head with NMS unless `nms=False` is given: the training runs (y26n, y26s, ...) are scored that way, the heads_* runs score both heads of the same weights.

*26 Oct 2026: the TensorRT INT8 mAP50 row was printed as a fraction by the notebook (0.97 = 97 %); it is shown here in % (report.py fix). Every other line is the run's own output; all results.json files are unchanged.*

## Results by condition: detectors (test mAP50 %)

| Run | all test | clear | night | blurry | small plates | rain (synthetic) | fog (synthetic) | false alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| y11n | 97.8 | 97.9 | 96.0 | 98.5 | 93.9 | 71.6 | 95.9 | 5.62 |
| y26n | 98.7 | 99.0 | 93.0 | 95.7 | 92.4 | 88.7 | 97.7 | 5.62 |
| y26n_p2 | 98.0 | 98.0 | 94.7 | 96.4 | 93.5 | 87.6 | 96.6 | 5.62 |
| y26n_sgd | 98.5 | 98.7 | 94.0 | 97.8 | 97.3 | 72.5 | 97.5 | 4.38 |
| y26s | 99.1 | 99.3 | 94.1 | 97.9 | 92.1 | 86.5 | 97.9 | 5.00 |

Condition subsets are fixed for the whole team (detection/conditions.json, ocr/conditions.json); rain and detection-fog copies are synthetic.
