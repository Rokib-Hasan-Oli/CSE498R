# Paper #28 — YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information

**Model Card and Usage Guide — Bangla Licence Plate Detection Re-implementation**

Department of Electrical and Computer Engineering (ECE)  
North South University, Bashundhara, Dhaka-1229, Bangladesh

**Course:** CSE498R — Directed Research  
**Paper:** #28 — YOLOv9 (ECCV 2024)  
**Implementer:** Md. Yousuf  
**Semester:** Summer 2026

---

## 1. Overview

This repository contains the re-implementation of **Paper #28: YOLOv9 — Learning What You Want to Learn Using Programmable Gradient Information** on **Bangla (BRTA) licence plate detection**. The study trains and compares four real-time object detectors — **YOLOv9-T**, **YOLOv9-S**, **YOLOv8-N** and **YOLOv8-S** — on the team's 498R BanglaLPR All-in-One dataset.

The goal is to measure whether YOLOv9's two core ideas — **Programmable Gradient Information (PGI)** and **Generalized Efficient Layer Aggregation Network (GELAN)** — translate into better plate detection on Bangla plates compared with the YOLOv8 baseline, and to quantify the trade-offs in accuracy, speed, parameters and FLOPs.

### 1.1 What this project does

- Fine-tunes four COCO-pretrained detectors on Bangla plate detection (single class).
- Evaluates **mAP50**, **mAP50-95**, **small-plate recall**, **condition-specific performance** (clear, night, blurry, small, rain, fog) and **false alarms**.
- Measures **TensorRT FP16**, **PyTorch FP16** and **CPU ONNX** latency on a Tesla T4.
- Reports **parameters**, **FLOPs** and **training time** for each model.
- Compares the results with the paper's MS COCO benchmarks (directionally, not head-to-head).

---

## 2. Models

| Model | Weights | Params (M) | FLOPs (G) | COCO mAP50-95 | COCO mAP50 | Role in this study |
|---|---|---:|---:|---:|---:|---|
| **YOLOv9-T** | `yolov9t.pt` | 2.0 | 7.7 | 38.3 | 53.1 [reference:27] | Lightweight YOLOv9 variant |
| **YOLOv9-S** | `yolov9s.pt` | 7.2 | 26.7 | 46.8 | 63.4 [reference:28] | Standard YOLOv9 variant |
| **YOLOv8-N** | `yolov8n.pt` | 3.2 | 8.7 | 37.3 | 52.5 [reference:29] | Lightweight YOLOv8 baseline |
| **YOLOv8-S** | `yolov8s.pt` | 11.2 | 28.6 | 44.9 | 61.8 [reference:30] | Standard YOLOv8 baseline |

*The parameter and FLOP counts in the results tables (`2.01`, `7.29`, `3.01`, `11.14 M`) come from the saved run files and may differ slightly from the Ultralytics documentation because of detection-head accounting.*

### 2.1 Architecture summary

**YOLOv9 (Paper #28)** introduces two main components:

1. **Programmable Gradient Information (PGI)** — an auxiliary reversible branch that preserves gradient information through deep layers, solving the information bottleneck problem in very deep networks [reference:31].
2. **GELAN (Generalized Efficient Layer Aggregation Network)** — a lightweight backbone/neck built from CSP-ELAN blocks and conventional convolutions, achieving better parameter utilisation than depth-wise-convolution-based designs [reference:32].

The Ultralytics implementations used here combine GELAN and PGI. The **official PGI-vs-GELAN ablation** is not replicated in this study because the official repository requires separate NumPy fixes [reference:33].

---

## 3. Dataset

**498R BanglaLPR All-in-One v1.0 — detection split**

| Split | Purpose |
|---|---|
| `detection/train` | Fine-tuning all four detectors |
| `detection/val` | Best-checkpoint selection |
| `detection/test` | All reported test results |

**Condition tags** (`detection/conditions.json`): clear, night, blurry, small plates, rain (synthetic), fog (synthetic). Subsets may overlap.

**Sources:** `anpr`, `blpd`, `kuet`, `lpdba`, `nightlpbd` (same sources as Paper #20).

**False-alarm set:** plate-free photos used to measure the percentage of images with at least one false detection.

---

## 4. Training Configuration

| Setting | Value |
|---|---|
| Framework | Ultralytics 8.4.170 |
| Input size | 640 × 640 |
| Epochs | 60 |
| Batch size | 16 |
| Optimiser | SGD |
| Seed | 42 |
| GPUs | 2 × Tesla T4 |
| GPU allocation | GPU 0: YOLOv9-T → YOLOv9-S; GPU 1: YOLOv8-N → YOLOv8-S |
| Pretrained weights | COCO (`yolov9t.pt`, `yolov9s.pt`, `yolov8n.pt`, `yolov8s.pt`) |

### 4.1 Training commands

```python
from ultralytics import YOLO

# YOLOv9-T
model = YOLO("yolov9t.pt")
model.train(data="detection/data.yaml", epochs=60, batch=16, imgsz=640,
            optimizer="SGD", seed=42, device=0)

# YOLOv9-S
model = YOLO("yolov9s.pt")
model.train(data="detection/data.yaml", epochs=60, batch=16, imgsz=640,
            optimizer="SGD", seed=42, device=0)

# YOLOv8-N
model = YOLO("yolov8n.pt")
model.train(data="detection/data.yaml", epochs=60, batch=16, imgsz=640,
            optimizer="SGD", seed=42, device=1)

# YOLOv8-S
model = YOLO("yolov8s.pt")
model.train(data="detection/data.yaml", epochs=60, batch=16, imgsz=640,
            optimizer="SGD", seed=42, device=1)
```

---

## 5. Results

### 5.1 Main detection metrics (test split)

| Model | mAP50 (%) | mAP50-95 (%) | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|
| YOLOv9-T | 98.3 | 45.2 | 2.01 | 7.80 |
| YOLOv9-S | 98.8 | 46.3 | 7.29 | 27.40 |
| YOLOv8-N | 98.7 | 45.2 | 3.01 | 8.20 |
| YOLOv8-S | 98.5 | 45.8 | 11.14 | 28.60 |

### 5.2 Small-plate recall

| Model | Recall (%) |
|---|---:|
| YOLOv9-T | 80.00 |
| YOLOv9-S | 60.00 |
| YOLOv8-N | 60.00 |
| YOLOv8-S | 40.00 |

### 5.3 Performance by condition (mAP50 %)

| Model | Clear | Night | Blurry | Small | Rain | Fog | False alarms (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv9-T | 97.9 | 97.5 | 98.1 | 93.2 | 88.3 | 95.0 | 5.62 |
| YOLOv9-S | 99.0 | 96.8 | 98.0 | 90.8 | 77.0 | 96.6 | 6.25 |
| YOLOv8-N | 99.0 | 94.5 | 97.8 | 92.8 | 83.8 | 96.3 | 6.88 |
| YOLOv8-S | 99.0 | 94.3 | 95.2 | 84.5 | 82.5 | 97.9 | 5.62 |

### 5.4 Speed (ms per image, batch 1)

| Model | T4 TensorRT FP16 | T4 PyTorch FP16 | CPU ONNX |
|---|---:|---:|---:|
| YOLOv9-T | 6.93 | 31.56 | 116.20 |
| YOLOv9-S | 7.24 | 27.28 | 296.93 |
| YOLOv8-N | 6.38 | 16.05 | 105.83 |
| YOLOv8-S | 7.52 | 23.22 | 300.71 |

### 5.5 Training time

| Model | Time (min) |
|---|---:|
| YOLOv9-T | 193.0 |
| YOLOv9-S | 149.0 |
| YOLOv8-N | 101.0 |
| YOLOv8-S | 119.4 |

**Note:** v9 models ran sequentially on GPU 0 and v8 models on GPU 1; times reflect both model complexity and GPU contention.

---

## 6. File Manifest

| File | Description |
|---|---|
| `Paper28_Report.md` | Full re-implementation and evaluation report |
| `Paper28_README.md` | This model card and usage guide |
| `paper28_v9t_best.pt` | YOLOv9-T best checkpoint |
| `paper28_v9s_best.pt` | YOLOv9-S best checkpoint |
| `paper28_v8n_best.pt` | YOLOv8-N best checkpoint |
| `paper28_v8s_best.pt` | YOLOv8-S best checkpoint |
| `paper28_results.zip` | Results archive (logs, JSON, environment) |
| `results_vs_paper.md` | Short comparison table |
| `28_2024_YOLOv9 Learning What You Want to Learn PGI.pdf` | The paper (ECCV 2024) |
| `notebooks/paper-28-yolov9-learning-what-you-want-to-learn.ipynb` | Full notebook |

---

## 7. Quick-Start Usage

### 7.1 Load a trained model and run inference

```python
from ultralytics import YOLO

# Load the best YOLOv9-S checkpoint
model = YOLO("paper28_v9s_best.pt")

# Run inference on an image
results = model("path/to/plate_image.jpg", imgsz=640)

# Print detections
for r in results:
    boxes = r.boxes
    print(f"Detected {len(boxes)} plates")
    print(boxes.xyxy)   # bounding boxes
    print(boxes.conf)   # confidence scores
```

### 7.2 Evaluate on the test split

```python
metrics = model.val(data="detection/data.yaml", imgsz=640)
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")
```

### 7.3 Export for deployment

```python
# TensorRT FP16
model.export(format="engine", half=True, imgsz=640)

# ONNX for CPU
model.export(format="onnx", imgsz=640)
```

---

## 8. Choosing a Model

| Priority | Recommended model | Why |
|---|---|---|
| Highest mAP50-95 | YOLOv9-S | 46.3% mAP50-95, highest in the study |
| Best small-plate recall | YOLOv9-T | 80.0% recall on small plates |
| Lowest latency (TensorRT / CPU) | YOLOv8-N | 6.38 ms TensorRT, 105.83 ms CPU |
| Best parameter efficiency | YOLOv9-T | 45.2% mAP50-95 with 2.01 M params |
| Best night performance | YOLOv9-T / YOLOv9-S | 97.5% / 96.8% mAP50 at night |
| Lowest false alarms | YOLOv9-T / YOLOv8-S | 5.62% on plate-free photos |

---

## 9. Key Findings

- mAP50 is saturated on Bangla plates (98.3–98.8% for all four detectors). Use mAP50-95 and small-plate recall to differentiate models.
- YOLOv9-S achieves the highest mAP50-95 (46.3%), closely matching the paper's COCO mAP50-95 (46.8%) — but on a different dataset.
- YOLOv9-T is the most efficient (best mAP50-95 per parameter and per FLOP) and has the best small-plate recall (80%).
- YOLOv9 models are stronger at night (+2–3 points mAP50 over YOLOv8), consistent with PGI's gradient-preservation design.
- YOLOv8-N is the fastest on TensorRT, PyTorch and CPU, but YOLOv9-T is competitive on TensorRT and CPU.
- Rain is the hardest condition for all models (77–88% mAP50), followed by small plates (85–93%).

---

## 10. Limitations

- Single dataset (498R BanglaLPR All-in-One); no external benchmark.
- Single seed (42); no confidence intervals.
- PGI-vs-GELAN ablation not replicated.
- COCO-pretrained weights only; no training from scratch.
- Parallel GPU schedule confounds training-time comparisons.
- Speed measured on one T4; no error bars.
- Synthetic rain and fog; real weather may differ.
- False-alarm set size not reported.
- No statistical significance tests.
- Paper comparison is directional, not a reproduction of COCO results.

---

## 11. Reproducibility

- **Seed:** 42 (all training runs).
- **Framework:** Ultralytics 8.4.170, PyTorch 2.11.0+cu128.
- **Hardware:** 2 × Tesla T4 (Kaggle).
- **Dataset:** 498R BanglaLPR All-in-One v1.0 (`detection/` split).
- **Condition tags:** `detection/conditions.json` (fixed team-wide).
- **Notebook:** `notebooks/paper-28-yolov9-learning-what-you-want-to-learn.ipynb`.

---

## 12. License / Academic Note

This project was developed as coursework for CSE498R: Directed Research, North South University, Summer 2026. It is intended for academic and educational use. The YOLOv9 paper is by C.-Y. Wang, I.-H. Yeh and H.-Y. M. Liao (ECCV 2024). The Ultralytics implementations are used under their respective licences.

---

*End of `Paper28_README.md` — Md. Yousuf, 20 October 2026.*
