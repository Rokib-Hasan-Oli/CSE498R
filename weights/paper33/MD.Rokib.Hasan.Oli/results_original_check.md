# Paper #33 on its own data: ours vs. the paper

| Metric | Paper (page) | coco_gpu0 | coco_gpu1 |
|---|---|---|---|
| YOLO26-N: COCO AP50-95, NMS head (%) | 40.9 (p.8, Table 2) | 40.81 |  |
| YOLO26-N: COCO AP50-95, NMS-free head (%) | 40.1 (e2e) (p.8, Table 2) | 40.06 |  |
| YOLO26-N: T4 TensorRT FP16, NMS-free, inference ms | 1.7 (p.8, Table 2) | 2.52 |  |
| YOLO26-N: CPU ONNX ms (end to end) | 38.9 (p.8, Table 2) | 96.64 |  |
| YOLO26-S: COCO AP50-95, NMS head (%) | 48.6 (p.8, Table 2) | 48.62 |  |
| YOLO26-S: COCO AP50-95, NMS-free head (%) | 47.8 (e2e) (p.8, Table 2) | 47.78 |  |
| YOLO26-S: T4 TensorRT FP16, NMS-free, inference ms | 2.5 (p.8, Table 2) | 2.64 |  |
| YOLO26-S: CPU ONNX ms (end to end) | 87.2 (p.8, Table 2) | 134.05 |  |
| YOLO26-M: COCO AP50-95, NMS head (%) | 53.1 (p.8, Table 2) |  | 53.09 |
| YOLO26-M: COCO AP50-95, NMS-free head (%) | 52.5 (e2e) (p.8, Table 2) |  | 52.50 |
| YOLO26-M: T4 TensorRT FP16, NMS-free, inference ms | 4.7 (p.8, Table 2) |  | 4.92 |
| YOLO11-N (baseline): COCO AP50-95 (%) | 39.5 (Ultralytics docs; not in the paper) |  | 39.45 |
| YOLO11-N: T4 TensorRT FP16, inference ms | 1.5 (Ultralytics docs) |  | 2.19 |

Same data and split as the paper (its own published data / weights), so this IS a direct check of
the paper's numbers. Remaining differences come from software versions, hardware and randomness.

The paper copies these numbers from Ultralytics; this check is therefore a check of the released weights. TensorRT 'inference ms' is the model only (the paper's column); `end_to_end_ms` in results.json adds pre- and post-processing (NMS for the one-to-many head).
