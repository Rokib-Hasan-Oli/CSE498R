# Paper28_README.md — Paper #28: YOLOv9 versus YOLOv8 for Bangla Plate Detection

> **Document name:** `Paper28_README.md`
> **Project:** 498R BanglaLPR — re-implementation of *"YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"* (Paper #28, arXiv:2402.13616)
> **Author / maintainer:** Md. Yousuf
> **Date:** 20 October 2026
<<<<<<< HEAD
> **Full technical report:** [`Paper28_Report.md`](Paper28_Report.md)
=======
> **Full technical report:** [`Paper28_Report.md`](../../../report/Md.Yousuf/Paper28_Report.md)
>>>>>>> b2e8c6800e4ff5030314c2d4fd0cba5b64ea80f5
> **Location in the team repository:** `weights/paper28/Md.Yousuf` in `Rokib-Hasan-Oli/CSE498R`

This package contains **four Bangla (BRTA) licence-plate detectors** (YOLOv9-T, YOLOv9-S, YOLOv8-N, YOLOv8-S) fine-tuned from COCO weights on the 498R BanglaLPR All-in-One detection split, their test, condition and speed results, the logs and the code. All numbers come from the saved run files in `paper28_results.zip`.

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [What Is in This Package](#2-what-is-in-this-package)
3. [Quick Start](#3-quick-start)
4. [Model Cards](#4-model-cards)
5. [Results Summary](#5-results-summary)
6. [How to Read These Results](#6-how-to-read-these-results)
7. [Reproducing the Results](#7-reproducing-the-results)
8. [Limitations](#8-limitations)
9. [Differences from the Original Paper](#9-differences-from-the-original-paper)
10. [File Integrity](#10-file-integrity)
11. [Licence and Data Notes](#11-licence-and-data-notes)
12. [Citation](#12-citation)

---

## 1. At a Glance

| | |
|---|---|
| **Task** | Detect Bangla BRTA licence plates (one class, `plate`) in photos; compare the GELAN-based YOLOv9 with YOLOv8 on the 498R detection split |
| **Models** | YOLOv9-T, YOLOv9-S, YOLOv8-N, YOLOv8-S (Ultralytics 8.4.170, COCO-pretrained start weights) |
| **Best accuracy** | **YOLOv9-S**: mAP50 **98.79 %**, mAP50-95 **46.33 %**, recall 97.20 % (test: 1,153 images, 1,011 plates) |
| **Smallest model** | **YOLOv9-T**: 2.01 M parameters, 7.8 GFLOPs, 4.4 MiB; mAP50 98.30 % |
| **Fastest / cheapest** | **YOLOv8-N**: 6.38 ms (TensorRT FP16, T4), 105.83 ms (CPU, ONNX), 101.0 min training; mAP50 98.73 % |
| **Conditions (mAP50)** | Night 94.3–97.5 % · blurry 95.2–98.1 % · small plates 84.5–93.2 % · synthetic rain 77.0–88.3 % · synthetic fog 95.0–97.9 % |
| **False alarms** | 5.62–6.88 % of 160 plate-free photos |
| **Training** | 101.0–193.0 min per model on a Tesla T4 (YOLOv9 queue on GPU 0, YOLOv8 queue on GPU 1, at the same time) |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 · TensorRT 11.4.0.106 |

The four models are within 0.49 points of mAP50 of each other, each was trained once with one seed, and YOLOv9-T stopped at epoch 54 of 60; see Sections 6 and 8. The paper's COCO numbers come from different data; see Section 5.4.

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `Paper28_README.md` | — | This guide |
| `paper28_v9t_best.pt` | 4.4 MiB | YOLOv9-T weights, best epoch 49 |
| `paper28_v9s_best.pt` | 14.5 MiB | YOLOv9-S weights, best epoch 54 |
| `paper28_v8n_best.pt` | 5.9 MiB | YOLOv8-N weights, best epoch 56 |
| `paper28_v8s_best.pt` | 21.5 MiB | YOLOv8-S weights, best epoch 55 |
| `paper28_results.zip` | 4.1 MiB | Results archive (no weights) |
| `results_vs_paper.md` | 1.5 KiB | Short ours-vs-paper table |
| `28_2024_YOLOv9 Learning What You Want to Learn PGI.pdf` | 4.7 MiB | The paper |

### 2.2 Contents of `paper28_results.zip`

```text
├── environment.json            # Python, PyTorch, CUDA, GPU and library versions
├── results_vs_paper.md         # ours vs the paper's reported numbers, plus results by condition
├── runs/
│   ├── v9t/  v9s/  v8n/  v8s/  # one folder per detector, each with:
│   │   ├── results.json        # settings, best epoch, val/test, plate counts, by source, by condition, speed
│   │   ├── results.csv         # per-epoch losses, P, R, mAP50, mAP50-95, learning rate, time
│   │   ├── args.yaml           # Ultralytics training arguments
│   │   ├── results.png  BoxPR_curve.png  confusion_matrix.png  labels.jpg   # Ultralytics plots
│   │   └── cond_yaml/          # clear, night, blurry, small_plates, rain_synth, fog_synth .yaml
├── logs/
│   ├── v9t.log  v9s.log  v8n.log  v8s.log                   # training logs
│   └── speed_v9t.log  speed_v9s.log  speed_v8n.log  speed_v8s.log   # speed-test logs
└── code/
    ├── det.py  common.py  report.py                         # used for Paper #28
    └── seq.py  metrics.py  vocab.py  rec.py  ved.py         # shared toolkit modules not used for Paper #28
```

---

## 3. Quick Start

### 3.1 Install

```bash
pip install ultralytics==8.4.170
```

To retrain or re-evaluate with the scripts, copy `code/det.py` and `code/common.py` into one folder (`det.py` imports `common`).

### 3.2 Detect plates with a checkpoint

This mirrors `load_model()` and `plate_counts()` in `det.py`; it has not been run separately from the saved scripts.

```python
from ultralytics import YOLO

model = YOLO("paper28_v9s_best.pt")            # or paper28_v9t / v8n / v8s _best.pt
res = model.predict("image.jpg", imgsz=640, conf=0.25, verbose=False)[0]

xyxy = res.boxes.xyxy.cpu().numpy()            # one row per plate: x1, y1, x2, y2
conf = res.boxes.conf.cpu().numpy()            # confidence of each box
print(model.names, xyxy, conf)
```

The checkpoints are Ultralytics checkpoints (the training log shows "Optimizer stripped" for each `best.pt`) with one class, `plate`. The working threshold of the plate counts is conf 0.25.

### 3.3 Re-evaluate on the test split and export

This mirrors `val_metrics()` and `measure_speed()` in `det.py`; it has not been run separately from the saved scripts. The dataset yaml is the one printed by the notebook:

```yaml
path: /tmp/498r_data/BanglaLPR_AllInOne_v1/detection
train: images/train
val: images/val
test: images/test
nc: 1
names:
  0: "plate"
```

```python
from ultralytics import YOLO

m = YOLO("paper28_v9s_best.pt")
r = m.val(data="detection_all.yaml", split="test", imgsz=640, batch=16)
print(r.box.mp, r.box.mr, r.box.map50, r.box.map)      # P, R, mAP50, mAP50-95

m.export(format="onnx", imgsz=640, simplify=False, dynamic=False)   # ONNX, used for the CPU speed test
```

### 3.4 Retrain the detectors and re-run the evaluation

The dataset (498R BanglaLPR All-in-One v1.0, detection part) must be available; the notebook downloads it from the team Drive. The commands below are the ones the notebook ran (from the training logs):

```bash
python code/det.py --data detection_all.yaml --model yolov9t.pt --name v9t --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 3.2   # GPU 0
python code/det.py --data detection_all.yaml --model yolov9s.pt --name v9s --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 4.0   # GPU 0, after v9t
python code/det.py --data detection_all.yaml --model yolov8n.pt --name v8n --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 2.8   # GPU 1
python code/det.py --data detection_all.yaml --model yolov8s.pt --name v8s --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 3.6   # GPU 1, after v8n
```

Speed test (needs the finished run folder `runs/<name>` with `weights/best.pt` and `results.json`):

```bash
python code/det.py --data detection_all.yaml --name v9t --project runs --imgsz 640 --fps-only --trt
```

`--test-max N` scores only N test images; the notebook's quick test (`FAST = True`) uses 1 epoch, `--fraction 0.03` and `--test-max 40`. `--trt int8` would also add INT8 TensorRT (not run in the notebook).

---

## 4. Model Cards

### 4.1 YOLOv9-T — `paper28_v9t_best.pt`

| Item | Value |
|---|---|
| Model | YOLOv9 (GELAN-based; layers `Conv`, `ELAN1`, `AConv`, `RepNCSPELAN4`, `SPPELAN`, `Detect` in the training log) |
| Parameters | 2.01 M as trained (2,005,603); 1,970,979 fused |
| GFLOPs | 7.8 as trained; 7.6 fused |
| Layers | 544 as trained; 197 fused |
| Input | Image resized to 640 (`imgsz` 640) |
| Output | Boxes of one class, `plate` (`nc` 1); plate counts use conf 0.25, IoU 0.5 |
| Training data | `detection/train` of 498R BanglaLPR All-in-One: 8,574 images (4,067 human + 4,507 auto labels; 466 background images) |
| Start weights | `yolov9t.pt` (COCO-pretrained); 1303/1339 items transferred |
| Optimiser | SGD, lr0 0.01, lrf 0.01, momentum 0.937, weight decay 5e-4, batch 16, seed 42, AMP on |
| Augmentation | Mosaic 1.0 (off for the last 10 epochs), HSV 0.015 / 0.7 / 0.4, translate 0.1, scale 0.5, fliplr 0.5, RandAugment, erasing 0.4; mixup, cutmix, copy-paste 0 |
| Epochs | 54 run of 60 planned (stopped by the 3.2 h time cap) |
| Checkpoint | Best fitness (0.1·mAP50 + 0.9·mAP50-95) on validation: epoch 49 |
| Training time | 193.0 min on one Tesla T4 (peak GPU memory 3.46 GB) |
| Test (1,153 images, 1,011 plates) | mAP50 **98.30 %** · mAP50-95 45.16 % · P 97.36 % · R 95.65 % |
| Validation (896 images) | mAP50 96.21 % · mAP50-95 60.40 % · P 96.31 % · R 94.35 % |
| Plates found / correct (conf 0.25, IoU 0.5) | 96.64 % / 92.61 % |
| Speed (T4, batch 1, 640) | TensorRT FP16 6.93 ms · PyTorch FP16 31.56 ms · PyTorch FP32 19.66 ms |
| Speed (CPU, ONNX) | 116.2 ms (8.6 FPS) |
| File | 4,610,249 bytes (4.4 MiB) |

By source (test, plates found): `mahedi` 100.0 % · `towniyan` 100.0 % · `kuet` 98.97 % · `ridoykj` 97.08 % · `lpdba` 96.68 % · `tusar` 93.1 % · `nightlpbd` 78.33 %.
By condition (test, mAP50): clear 97.9 % · night 97.5 % · blurry 98.1 % · small plates 93.2 % · rain (synthetic) 88.3 % · fog (synthetic) 95.0 %; false alarms on 160 plate-free photos 5.62 %.


### 4.2 YOLOv9-S — `paper28_v9s_best.pt`

| Item | Value |
|---|---|
| Model | YOLOv9 (GELAN-based; layers `Conv`, `ELAN1`, `AConv`, `RepNCSPELAN4`, `SPPELAN`, `Detect` in the training log) |
| Parameters | 7.29 M as trained (7,287,795); 7,167,475 fused |
| GFLOPs | 27.4 as trained; 26.7 fused |
| Layers | 544 as trained; 197 fused |
| Input | Image resized to 640 (`imgsz` 640) |
| Output | Boxes of one class, `plate` (`nc` 1); plate counts use conf 0.25, IoU 0.5 |
| Training data | `detection/train` of 498R BanglaLPR All-in-One: 8,574 images (4,067 human + 4,507 auto labels; 466 background images) |
| Start weights | `yolov9s.pt` (COCO-pretrained); 1333/1339 items transferred |
| Optimiser | SGD, lr0 0.01, lrf 0.01, momentum 0.937, weight decay 5e-4, batch 16, seed 42, AMP on |
| Augmentation | Mosaic 1.0 (off for the last 10 epochs), HSV 0.015 / 0.7 / 0.4, translate 0.1, scale 0.5, fliplr 0.5, RandAugment, erasing 0.4; mixup, cutmix, copy-paste 0 |
| Epochs | 60 run of 60 planned |
| Checkpoint | Best fitness (0.1·mAP50 + 0.9·mAP50-95) on validation: epoch 54 |
| Training time | 149.0 min on one Tesla T4 (peak GPU memory 6.45 GB) |
| Test (1,153 images, 1,011 plates) | mAP50 **98.79 %** · mAP50-95 46.33 % · P 97.04 % · R 97.20 % |
| Validation (896 images) | mAP50 97.10 % · mAP50-95 61.17 % · P 95.00 % · R 95.29 % |
| Plates found / correct (conf 0.25, IoU 0.5) | 97.82 % / 92.52 % |
| Speed (T4, batch 1, 640) | TensorRT FP16 7.24 ms · PyTorch FP16 27.28 ms · PyTorch FP32 23.43 ms |
| Speed (CPU, ONNX) | 296.93 ms (3.4 FPS) |
| File | 15,202,185 bytes (14.5 MiB) |

By source (test, plates found): `mahedi` 100.0 % · `ridoykj` 99.64 % · `kuet` 99.22 % · `lpdba` 97.16 % · `towniyan` 94.44 % · `tusar` 93.1 % · `nightlpbd` 85.0 %.
By condition (test, mAP50): clear 99.0 % · night 96.8 % · blurry 98.0 % · small plates 90.8 % · rain (synthetic) 77.0 % · fog (synthetic) 96.6 %; false alarms on 160 plate-free photos 6.25 %.


### 4.3 YOLOv8-N — `paper28_v8n_best.pt`

| Item | Value |
|---|---|
| Model | YOLOv8 (layers `Conv`, `C2f`, `SPPF`, `Detect` in the training log) |
| Parameters | 3.01 M as trained (3,011,043); 3,005,843 fused |
| GFLOPs | 8.2 as trained; 8.1 fused |
| Layers | 129 as trained; 72 fused |
| Input | Image resized to 640 (`imgsz` 640) |
| Output | Boxes of one class, `plate` (`nc` 1); plate counts use conf 0.25, IoU 0.5 |
| Training data | `detection/train` of 498R BanglaLPR All-in-One: 8,574 images (4,067 human + 4,507 auto labels; 466 background images) |
| Start weights | `yolov8n.pt` (COCO-pretrained); 319/355 items transferred |
| Optimiser | SGD, lr0 0.01, lrf 0.01, momentum 0.937, weight decay 5e-4, batch 16, seed 42, AMP on |
| Augmentation | Mosaic 1.0 (off for the last 10 epochs), HSV 0.015 / 0.7 / 0.4, translate 0.1, scale 0.5, fliplr 0.5, RandAugment, erasing 0.4; mixup, cutmix, copy-paste 0 |
| Epochs | 60 run of 60 planned |
| Checkpoint | Best fitness (0.1·mAP50 + 0.9·mAP50-95) on validation: epoch 56 |
| Training time | 101.0 min on one Tesla T4 (peak GPU memory 2.59 GB) |
| Test (1,153 images, 1,011 plates) | mAP50 **98.73 %** · mAP50-95 45.23 % · P 97.22 % · R 96.68 % |
| Validation (896 images) | mAP50 96.22 % · mAP50-95 59.73 % · P 95.56 % · R 94.82 % |
| Plates found / correct (conf 0.25, IoU 0.5) | 98.12 % / 92.36 % |
| Speed (T4, batch 1, 640) | TensorRT FP16 6.38 ms · PyTorch FP16 16.05 ms · PyTorch FP32 10.89 ms |
| Speed (CPU, ONNX) | 105.83 ms (9.4 FPS) |
| File | 6,222,698 bytes (5.9 MiB) |

By source (test, plates found): `mahedi` 100.0 % · `kuet` 99.48 % · `ridoykj` 98.91 % · `lpdba` 97.63 % · `tusar` 96.55 % · `towniyan` 94.44 % · `nightlpbd` 88.33 %.
By condition (test, mAP50): clear 99.0 % · night 94.5 % · blurry 97.8 % · small plates 92.8 % · rain (synthetic) 83.8 % · fog (synthetic) 96.3 %; false alarms on 160 plate-free photos 6.88 %.


### 4.4 YOLOv8-S — `paper28_v8s_best.pt`

| Item | Value |
|---|---|
| Model | YOLOv8 (layers `Conv`, `C2f`, `SPPF`, `Detect` in the training log) |
| Parameters | 11.14 M as trained (11,135,987); 11,125,971 fused |
| GFLOPs | 28.6 as trained; 28.4 fused |
| Layers | 129 as trained; 72 fused |
| Input | Image resized to 640 (`imgsz` 640) |
| Output | Boxes of one class, `plate` (`nc` 1); plate counts use conf 0.25, IoU 0.5 |
| Training data | `detection/train` of 498R BanglaLPR All-in-One: 8,574 images (4,067 human + 4,507 auto labels; 466 background images) |
| Start weights | `yolov8s.pt` (COCO-pretrained); 349/355 items transferred |
| Optimiser | SGD, lr0 0.01, lrf 0.01, momentum 0.937, weight decay 5e-4, batch 16, seed 42, AMP on |
| Augmentation | Mosaic 1.0 (off for the last 10 epochs), HSV 0.015 / 0.7 / 0.4, translate 0.1, scale 0.5, fliplr 0.5, RandAugment, erasing 0.4; mixup, cutmix, copy-paste 0 |
| Epochs | 60 run of 60 planned |
| Checkpoint | Best fitness (0.1·mAP50 + 0.9·mAP50-95) on validation: epoch 55 |
| Training time | 119.4 min on one Tesla T4 (peak GPU memory 4.5 GB) |
| Test (1,153 images, 1,011 plates) | mAP50 **98.51 %** · mAP50-95 45.82 % · P 96.90 % · R 96.93 % |
| Validation (896 images) | mAP50 97.07 % · mAP50-95 60.04 % · P 96.37 % · R 94.71 % |
| Plates found / correct (conf 0.25, IoU 0.5) | 98.12 % / 91.94 % |
| Speed (T4, batch 1, 640) | TensorRT FP16 7.52 ms · PyTorch FP16 23.22 ms · PyTorch FP32 14.29 ms |
| Speed (CPU, ONNX) | 300.71 ms (3.3 FPS) |
| File | 22,491,690 bytes (21.4 MiB) |

By source (test, plates found): `mahedi` 100.0 % · `ridoykj` 99.27 % · `kuet` 99.22 % · `lpdba` 98.58 % · `tusar` 93.1 % · `towniyan` 88.89 % · `nightlpbd` 88.33 %.
By condition (test, mAP50): clear 99.0 % · night 94.3 % · blurry 95.2 % · small plates 84.5 % · rain (synthetic) 82.5 % · fog (synthetic) 97.9 %; false alarms on 160 plate-free photos 5.62 %.


### 4.5 Evaluated scope

Evaluated: plate detection (one class) on the 498R BanglaLPR All-in-One detection test split; real clear, night, blurry and small-plate subsets; synthetic rain and synthetic fog copies of the test split; 160 plate-free photos; latency at batch 1 on a Tesla T4 (PyTorch FP32 / FP16, TensorRT FP16) and on a CPU (ONNX).

Not evaluated: the PGI-vs-GELAN ablation (the notebook states it is not included), training from scratch, larger models (YOLOv9-M/C/E, YOLOv8-M and up), INT8 TensorRT, several seeds, plate recognition (OCR).

---

## 5. Results Summary

All numbers: detection test split (1,153 images, 1,011 plates), from `runs/<run>/results.json` unless stated. One plate equals 0.099 points of plates found.

### 5.1 Accuracy and size

| Item | v9t | v9s | v8n | v8s |
|---|---|---|---|---|
| mAP50 (%) | 98.30 | 98.79 | 98.73 | 98.51 |
| mAP50-95 (%) | 45.16 | 46.33 | 45.23 | 45.82 |
| Precision (%) | 97.36 | 97.04 | 97.22 | 96.90 |
| Recall (%) | 95.65 | 97.20 | 96.68 | 96.93 |
| Plates found (%, conf 0.25) | 96.64 | 97.82 | 98.12 | 98.12 |
| Correct boxes (%, conf 0.25) | 92.61 | 92.52 | 92.36 | 91.94 |
| IoU > 0.7 accuracy (%) | 66.87 | 65.96 | 65.86 | 66.06 |
| Mean best IoU (%) | 75.82 | 76.33 | 76.4 | 76.48 |
| Recall of small plates (%, 5 plates) | 80.00 | 60.00 | 60.00 | 40.00 |
| Recall of medium plates (%, 267 plates) | 95.13 | 95.51 | 97.38 | 97.38 |
| Recall of large plates (%, 739 plates) | 97.29 | 98.92 | 98.65 | 98.78 |
| Parameters (M) | 2.01 | 7.29 | 3.01 | 11.14 |
| GFLOPs | 7.8 | 27.4 | 8.2 | 28.6 |
| Training time (min) | 193.0 | 149.0 | 101.0 | 119.4 |
| Epochs run | 54 | 60 | 60 | 60 |

### 5.2 By condition and by source

mAP50 (%) on the condition subsets (from `results_vs_paper.md`); rain and fog are synthetic copies of the test split:

| Run | All test | Clear | Night | Blurry | Small plates | Rain (synthetic) | Fog (synthetic) | False alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| v8n | 98.7 | 99.0 | 94.5 | 97.8 | 92.8 | 83.8 | 96.3 | 6.88 |
| v8s | 98.5 | 99.0 | 94.3 | 95.2 | 84.5 | 82.5 | 97.9 | 5.62 |
| v9s | 98.8 | 99.0 | 96.8 | 98.0 | 90.8 | 77.0 | 96.6 | 6.25 |
| v9t | 98.3 | 97.9 | 97.5 | 98.1 | 93.2 | 88.3 | 95.0 | 5.62 |

Plates found (%, conf 0.25, IoU 0.5) by image source:

| Source (plates / images) | v9t | v9s | v8n | v8s |
|---|---|---|---|---|
| `kuet` (387 / 540) | 98.97 | 99.22 | 99.48 | 99.22 |
| `lpdba` (211 / 205) | 96.68 | 97.16 | 97.63 | 98.58 |
| `mahedi` (32 / 32) | 100.0 | 100.0 | 100.0 | 100.0 |
| `nightlpbd` (60 / 60) | 78.33 | 85.0 | 88.33 | 88.33 |
| `ridoykj` (274 / 274) | 97.08 | 99.64 | 98.91 | 99.27 |
| `towniyan` (18 / 13) | 100.0 | 94.44 | 94.44 | 88.89 |
| `tusar` (29 / 29) | 93.1 | 93.1 | 96.55 | 93.1 |

### 5.3 Speed (batch 1, 640 px)

| Run | PyTorch FP32 (T4) | PyTorch FP16 (T4) | TensorRT FP16 (T4) | ONNX on CPU |
|---|---|---|---|---|
| v9t | 19.66 ms (50.9 FPS) | 31.56 ms (31.7 FPS) | 6.93 ms (144.4 FPS) | 116.2 ms (8.6 FPS) |
| v9s | 23.43 ms (42.7 FPS) | 27.28 ms (36.7 FPS) | 7.24 ms (138.2 FPS) | 296.93 ms (3.4 FPS) |
| v8n | 10.89 ms (91.8 FPS) | 16.05 ms (62.3 FPS) | 6.38 ms (156.7 FPS) | 105.83 ms (9.4 FPS) |
| v8s | 14.29 ms (70.0 FPS) | 23.22 ms (43.1 FPS) | 7.52 ms (133.0 FPS) | 300.71 ms (3.3 FPS) |

T4 jobs: first 200 test images (CPU: first 50) after 15 warm-up images; time = pre-process + inference + post-process (NMS included).

### 5.4 Compared with the paper

| Item | Paper (Table 1, MS COCO val) | This study (498R plate test split) |
|---|---|---|
| YOLOv9-S parameters / FLOPs | 7.1 M / 26.4 G | 7.29 M / 27.4 G as trained (7.17 M / 26.7 G fused) |
| YOLOv8-S parameters / FLOPs | 11.2 M / 28.6 G | 11.14 M / 28.6 G as trained |
| YOLOv8-N parameters / FLOPs | 3.2 M / 8.7 G | 3.01 M / 8.2 G as trained |
| YOLOv9-T | not listed | 2.01 M / 7.8 G as trained |
| AP50 / mAP50: YOLOv9-S | 63.4 % | 98.79 % |
| AP50 / mAP50: YOLOv8-S | 61.8 % | 98.51 % |
| AP50:95 / mAP50-95: YOLOv9-S | 46.8 % | 46.33 % |
| AP50:95 / mAP50-95: YOLOv8-S | 44.9 % | 45.82 % |
| AP50:95 / mAP50-95: YOLOv8-N | 37.3 % | 45.23 % |
| YOLOv9-S minus YOLOv8-S, AP50:95 / mAP50-95 | +1.9 points | +0.51 points |
| YOLOv9-S minus YOLOv8-S, AP50 / mAP50 | +1.6 points | +0.28 points |
| PGI ablation (Table 5, GELAN-S AP50:95: 46.7 → +PGI 46.8) | reported | not run |
| Speed and training time | not reported | see Section 5.3 and Section 5.1 |

The paper trains from scratch for 500 epochs on MS COCO (80 classes); this study fine-tunes COCO-pretrained Ultralytics models for 60 epochs on one class. As `results_vs_paper.md` states, the comparison shows the direction and size of the gap, not a head-to-head result.

---

## 6. How to Read These Results

**Evidence from the saved numbers and code.** Details are in `Paper28_Report.md`.

- **Evidence.** The four models differ by only 0.49 points of test mAP50 (98.30–98.79 %). YOLOv9-S is highest in mAP50, mAP50-95 (46.33 %) and recall (97.20 %); YOLOv8-N is second in mAP50 (98.73 %); YOLOv9-T is lowest in mAP50 and recall (95.65 %) but highest in precision (97.36 %).
- **Evidence.** YOLOv9-T found 96.64 % of the plates at conf 0.25 against 98.12 % for YOLOv8-N and YOLOv8-S, which is 15 of 1,011 plates. On the `nightlpbd` source it found 78.33 % against 85.00 % (YOLOv9-S) and 88.33 % (YOLOv8-N, YOLOv8-S).
- **Evidence.** YOLOv9-T was stopped after epoch 54 of 60 by the 3.2 h time cap (`[time cap] 3.2 h reached after epoch 54`). Its learning rate at the last epoch is 0.001255, the other three ended at 0.000265, and 4 of the planned 10 no-mosaic epochs were run.
- **Evidence.** The test split has only 5 small plates, so the "small plates recall" row (80 / 60 / 60 / 40 %) is 4, 3, 3 and 2 plates.
- **Evidence.** Test mAP50 is higher than validation mAP50 for all four models, and test mAP50-95 is lower than validation mAP50-95 (see the table below). Mean best IoU on the test split is 75.82–76.48 %.

| Run | Val mAP50 | Test mAP50 | Val mAP50-95 | Test mAP50-95 |
|---|---|---|---|---|
| v9t | 96.21 | 98.30 | 60.40 | 45.16 |
| v9s | 97.10 | 98.79 | 61.17 | 46.33 |
| v8n | 96.22 | 98.73 | 59.73 | 45.23 |
| v8s | 97.07 | 98.51 | 60.04 | 45.82 |

- **Evidence.** On the T4, PyTorch FP16 at batch 1 is slower than PyTorch FP32 for all four models, and TensorRT FP16 is the fastest mode for all four (6.38–7.52 ms). On the CPU the YOLOv9-T and YOLOv8-N models take 105.83–116.20 ms and the two small models 296.93–300.71 ms.
- **Evidence.** The training times were measured while the two GPU queues ran at the same time: YOLOv9-T ran together with YOLOv8-N and YOLOv8-S, YOLOv9-S ran together with YOLOv8-S for about its first 30.8 min only (calculated from the job times in notebook cell 14). Mean seconds per epoch: YOLOv9-T 213.4, YOLOv9-S 148.5, YOLOv8-S 118.8, YOLOv8-N 100.1.
- **Evidence.** Synthetic rain gives the widest spread of mAP50 between models (77.0 % YOLOv9-S to 88.3 % YOLOv9-T); on clear photos the models are within 1.2 points (97.9–99.0 %).
- **Evidence.** The notebook states that the official repository's PGI-vs-GELAN ablation is not in it; the measured comparison is between the Ultralytics YOLOv9 and YOLOv8 models. The paper reports the PGI ablation on COCO in its Table 5.
- **Evidence.** In `results.json`, `ms_breakdown` holds the stage times of the **last** measured image only (code of `measure_speed`); `ms_per_image` is the mean over all images.

---

## 7. Reproducing the Results

### 7.1 Environment

Exact versions are in `environment.json` (Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, Ultralytics 8.4.170, numpy 2.1.3, OpenCV 4.14.0, transformers 5.16.1, timm 1.0.29, peft 0.20.0). The run used two Tesla T4 GPUs: the YOLOv9 queue (v9t, then v9s) on GPU 0 and the YOLOv8 queue (v8n, then v8s) on GPU 1 at the same time. The speed jobs were started the same way, one queue per GPU. The dataset (3.39 GB) was downloaded from the team Google Drive at the start of the run.

### 7.2 Commands

<<<<<<< HEAD
Notebook route: open `notebooks/paper-28-yolov9-learning-what-you-want-to-learn.ipynb` on Kaggle (Accelerator **GPU T4 × 2**, Internet **On**), run it once with `FAST = True` and check that every cell finishes, then set `FAST = False` and use **Save Version → Save & Run All**. Download `paper28_outputs.zip` (216.4 MB, with weights) from the **Output** tab; `paper28_results.zip` (4.3 MB) is the copy without weights.
=======
Notebook route: open `notebooks/Md.Yousuf/paper28_run.ipynb` on Kaggle (Accelerator **GPU T4 × 2**, Internet **On**), run it once with `FAST = True` and check that every cell finishes, then set `FAST = False` and use **Save Version → Save & Run All**. Download `paper28_outputs.zip` (216.4 MB, with weights) from the **Output** tab; `paper28_results.zip` (4.3 MB) is the copy without weights.
>>>>>>> b2e8c6800e4ff5030314c2d4fd0cba5b64ea80f5

Script route: the commands in Section 3.4.

Wall times in the run (notebook cells 14 and 16): training 196.7 min (v9t), 152.3 min (v9s), 104.3 min (v8n), 123.2 min (v8s); speed tests 5.8, 4.0, 5.5 and 3.3 min.

### 7.3 Determinism

Seed 42 and `deterministic: true` for all four trainings (`args.yaml`). The test split is used once, at the end, with `best.pt`. The time cap stops training after the epoch that passes the limit and keeps the epoch plan.

---

## 8. Limitations

1. **Single run.** One seed per model and one measurement per speed mode; no confidence intervals. The models differ by 0.49 points of mAP50.
2. **Fine-tuning, not training from scratch.** The paper trains from scratch for 500 epochs; here COCO-pretrained Ultralytics models are fine-tuned for 60 epochs on one class.
3. **No PGI ablation.** The notebook does not include the official repository's PGI-vs-GELAN ablation.
4. **YOLOv9-T is not fully trained.** It stopped at epoch 54 of 60 (Section 6).
5. **Small subsets.** Only 5 small plates in the test split; the other condition subsets are fixed team-wide lists (`detection/conditions.json`), and rain and fog are synthetic copies.
6. **Labels.** The training set contains 4,507 auto-labelled images next to 4,067 human-labelled ones; validation and test labels are human.
7. **Speed.** Measured on a Tesla T4 (and one CPU) only. The speed jobs of the two GPUs ran at the same time (notebook cell 16), while the `det.py` docstring says speed is measured on an idle GPU. `ms_breakdown` is from the last image only.
8. **Training times** were measured with two GPU queues running at the same time (Section 6).
9. **Only detection.** No plate recognition was run on the detected boxes.
10. **Different data and metrics from the paper** (Section 9); the paper's numbers are COCO numbers.

---

## 9. Differences from the Original Paper

| Component | Paper | This study |
|---|---|---|
| Data | MS COCO 2017 (80 classes) | 498R BanglaLPR All-in-One detection split (1 class: `plate`) |
| Training | From scratch, 500 epochs | Fine-tuned from COCO-pretrained weights, 60 epochs (54 for YOLOv9-T) |
| Models | YOLOv9-S/M/C/E, GELAN-S/M/C/E, YOLOv8-N to X and others | YOLOv9-T/S and YOLOv8-N/S (Ultralytics) |
| PGI | Auxiliary reversible branch and multi-level auxiliary information; ablation in Table 5 | Ablation not run (notebook note) |
| Input size | 640 | 640 |
| Optimiser | SGD, lr 0.01 → 0.0001 linear, 3 warm-up epochs | SGD, lr 0.01, lrf 0.01 (final 0.0001), 3 warm-up epochs |
| Augmentation | scale 0.9, MixUp 0.15, copy & paste 0.3, mosaic off for the last 15 epochs | scale 0.5, MixUp 0, copy & paste 0, mosaic off for the last 10 epochs |
| Metrics | AP50:95, AP50, AP75, APS / M / L, parameters, FLOPs | mAP50, mAP50-95, P, R, plate counts, condition subsets, latency, training time |
| Speed | Not reported | T4 (PyTorch FP32 / FP16, TensorRT FP16) and CPU (ONNX) |
| Hardware | Not stated in the main text (NCHC computing resources acknowledged) | Kaggle, 2 × Tesla T4 |

---

## 10. File Integrity

SHA-256 of the files as downloaded from the repository folder:

```text
7dcd732236412163029a617e8cfc19ae70107bde62c730e8e76bb9e4499b4aa0  paper28_v9t_best.pt
f4f2c3b8219bd558bdab412a63c0a7a09a1b8823c71c3605097c1b28c03c6f91  paper28_v9s_best.pt
a3206979af636895f0fdf95f8409918cbaa4c6980399621e687138668e0455f6  paper28_v8n_best.pt
528a829bd8f58fe33fec476bb645e85f18da7e9c229eb3280a3194c0c21a1017  paper28_v8s_best.pt
4b977b133b388352f673935a84235cf49914c7859b1d3477741ded9ae4f0988e  paper28_results.zip
132a846f0b518bc939e88ac8711c8dd94590a9a205d3b22741d1211840bba1c2  results_vs_paper.md
5a67aef4802df6f1e32f84884b309b59818294df56b4dafe1c097287b222e9fd  28_2024_YOLOv9 Learning What You Want to Learn PGI.pdf
```

```bash
sha256sum paper28_v9t_best.pt paper28_v9s_best.pt paper28_v8n_best.pt paper28_v8s_best.pt paper28_results.zip
```

`.pt` files are PyTorch pickle archives; loading one can execute code. Load them only from sources you trust.

---

## 11. Licence and Data Notes

- **Dataset.** The 498R BanglaLPR All-in-One dataset combines several sources (detection test split: `kuet`, `ridoykj`, `lpdba`, `nightlpbd`, `mahedi`, `tusar`, `towniyan`). Verify the licence terms of each source before redistributing images. The dataset is not included in this package.
- **Paper and code.** The paper (arXiv:2402.13616), the Ultralytics library and the COCO-pretrained start weights (`yolov9t.pt`, `yolov9s.pt`, `yolov8n.pt`, `yolov8s.pt`) keep their own licences.
- **Privacy.** Licence plates can be personal data; follow the rules that apply where you use these models.
- **Error rates in our tests.** At conf 0.25 the detectors found 96.64–98.12 % of the test plates and raised a false alarm on 5.62–6.88 % of plate-free photos; where mistakes matter, detections should be checked by a person.

---

## 12. Citation

```bibtex
@article{wang2024yolov9,
  title   = {{YOLOv9}: Learning What You Want to Learn Using Programmable
             Gradient Information},
  author  = {Wang, Chien-Yao and Yeh, I-Hau and Liao, Hong-Yuan Mark},
  journal = {arXiv preprint arXiv:2402.13616},
  year    = {2024}
}

@misc{yousuf2026paper28,
  title  = {Re-implementation and Evaluation of Paper \#28 on the 498R BanglaLPR Dataset:
            YOLOv9 versus YOLOv8 for Bangla Plate Detection},
  author = {Md. Yousuf},
  year   = {2026},
  month  = oct,
  note   = {Paper28\_Report.md and Paper28\_README.md}
}
```

---

*End of `Paper28_README.md` — Md. Yousuf, 20 October 2026.*
