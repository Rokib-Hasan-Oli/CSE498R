# README.md — Paper #33: YOLO26, Key Architectural Enhancements and Performance Benchmarking (run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — experiments for *"YOLO26: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection"* (Sapkota, Cheppally, Sharda & Karkee, arXiv 2509.25164 v5, 2026; a review of Ultralytics YOLO26; Paper #33)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 26 October 2026 (Kaggle run of 25-26 October 2026, dataset **498R BanglaLPR All-in-One v1.1**; original-paper check on COCO val2017, 26 October 2026)
> **Full study report:** [`Paper33_Study_Report.pdf`](Paper33_Study_Report.pdf) (paper study, training check, COCO check, findings, faculty guide; 19 pages)

This package contains five **one-class plate detectors** trained under the same settings — YOLO11-N (the team
baseline), YOLO26-N with SGD and with MuSGD, YOLO26-N with the extra P2 (stride-4) level and YOLO26-S — plus the
results of both heads of every YOLO26 model (the classic one-to-many head + NMS and the new NMS-free one-to-one head),
TensorRT FP16 / INT8 tests, and the original-paper check of the released YOLO26 weights on COCO. The paper itself runs
no experiment; every number below is ours. All numbers come from the run files in `paper33_results.zip`,
`paper33_check_results.zip` and the local checks described in the PDF (sections 7-15).

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [What Is in This Package](#2-what-is-in-this-package)
3. [Quick Start](#3-quick-start)
4. [Model Cards](#4-model-cards)
5. [Results Summary](#5-results-summary)
6. [Reproducing the Results](#6-reproducing-the-results)
7. [Limitations](#7-limitations)
8. [Differences from the Original Paper](#8-differences-from-the-original-paper)
9. [File Integrity](#9-file-integrity)
10. [Licence and Data Notes](#10-licence-and-data-notes)
11. [Citation](#11-citation)

---

## 1. At a Glance

| | |
|---|---|
| **Task** | Find Bangladeshi number plates in a street / CCTV photo (one class, `plate`) |
| **Models** | `best_y11n.pt` YOLO11-N (SGD) · `best_y26n_sgd.pt` YOLO26-N (SGD) · `best_y26n_musgd.pt` YOLO26-N (MuSGD, stopped at epoch 50 of 60) · `best_y26n_p2.pt` YOLO26-N-P2 (MuSGD) · `best_y26s.pt` YOLO26-S (MuSGD) |
| **Detection test (1,153 photos, 1,011 plates)** | plates found (conf 0.25, IoU 0.5) **97.6-98.4 %** for all five; mAP50 97.8-99.1 %; **no paired difference is significant** (e.g. YOLO26-N vs YOLO11-N: 10 vs 9 plates, McNemar p = 1.0) |
| **mAP50-95** | 44.7-46.4 % — **64.6-65.0 % for all five without the ridoykj source**, whose labels sit exactly on the plate edge |
| **NMS-free head** (`nms=False`) | costs 2-3 plates of 1,011 on fully trained models (n.s.); removes CPU NMS: 7.5 ms → 0.27 ms (YOLO26-N), **80.4 ms → 0.27 ms (P2)**; but puts a second box on 13-24 plates per 1,153 photos (merge before OCR) |
| **Synthetic rain** | models trained with **MuSGD find 78.6-82.4 %** of plates, with SGD 60.6-63.0 %; same YOLO26-N network: 232 vs 16 plates, p < 0.0001 (synthetic rain, one seed per model) |
| **TensorRT INT8 (T4)** | NMS-free heads lose 0.04-0.52 mAP50 points vs TensorRT FP16; not faster at batch 1 (5.6-5.8 ms either way) |
| **CPU (Core Ultra 7 265K, ONNX, 4 threads)** | YOLO26-N NMS-free 15.9 + 0.3 ms vs YOLO11-N 18.5 + 7.7 ms (network + post-processing) = 1.62x faster |
| **COCO check (released weights)** | YOLO26-N / S / M mAP50-95 **40.81 / 48.62 / 53.09** (paper 40.9 / 48.6 / 53.1); NMS-free 40.06 / 47.78 / 52.50 (paper 40.1 / 47.8 / 52.5) |
| **Training** | 640 px, 60 epochs, batch 16, seed 42, Kaggle Tesla T4 x2 (one job per GPU), 113-175 min per model; 9/9 jobs `code=0` |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 · TensorRT 11.4 · dataset v1.1 |

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `Paper33_Study_Report.pdf` | 4.2 MiB | Full study report (paper, results, analysis, faculty guide), 19 pages |
| `best_y26n_p2.pt` | 5.3 MiB | YOLO26-N-P2, MuSGD, best epoch 55 of 60: most plates found (98.42 %) — **use NMS-free on a CPU** |
| `best_y26n_musgd.pt` | 5.1 MiB | YOLO26-N, MuSGD, stopped by the time cap after epoch 50 of 60 (best = last epoch) |
| `best_y26n_sgd.pt` | 5.1 MiB | YOLO26-N, SGD, best epoch 55 of 60 (optimiser ablation) |
| `best_y26s.pt` | 19.3 MiB | YOLO26-S, MuSGD, best epoch 49 of 60 |
| `best_y11n.pt` | 5.2 MiB | YOLO11-N, SGD, best epoch 55 of 60 (team baseline) |
| `paper33_results.zip` | 5.4 MiB | Results archive, **no model weights** (see 2.2) |
| `paper33_check_results.zip` | 77 KiB | COCO check: `results.json` of both jobs, logs, code (COCO prediction files are on Drive) |
| `results_vs_paper.md` | — | The kit's ours-vs-paper table and the per-condition table |
| `results_original_check.md` | — | The kit's COCO check table |

All weight files are under GitHub's 25 MB web-upload limit.

### 2.2 Inside `paper33_results.zip`

| Path | What |
|---|---|
| `runs/y11n/`, `runs/y26n/`, `runs/y26n_sgd/`, `runs/y26n_p2/`, `runs/y26s/` | `results.json` (val / test P, R, mAP, plates found by source and size, per-condition scores, all settings; `y11n` also FP32 / FP16 / TensorRT FP16 + INT8 / CPU speed), `results.csv` (per-epoch curves), `args.yaml`, `results.png`, PR curve, confusion matrix, `labels.jpg` |
| `runs/heads_y26n/`, `runs/heads_y26s/`, `runs/heads_y26n_p2/` | `results.json`: both heads of the same weights — test mAP, plates found, false alarms, PyTorch FP16, TensorRT FP16 and INT8 (ms and test mAP), CPU ONNX ms |
| `logs/` | the nine job logs |
| `code/` | the exact kit code used (`det.py`, `p33_yolo26.py`, `common.py`, `metrics.py`, `report.py`, ...) |
| `environment.json`, `results_vs_paper.md` | versions (incl. `dataset_version: 1.1`) and the comparison table |

`results_vs_paper.md` (inside the zip and in this folder): the notebook printed the TensorRT INT8 mAP50 row as a
fraction (0.97 = 97 %); that one row is shown in % here (noted under the table). Every `results.json` is unchanged.

### 2.3 Not in this folder

Team Drive **`498R / weights / paper33 /`**: `paper33_outputs.zip` (194 MiB: everything above plus `best.pt`, `last.pt`,
`best.onnx`, `best.fp16.onnx` and `best.int8.onnx` of every model) and `paper33_check_outputs.zip` (83 MiB: the COCO
check with its prediction files). The TensorRT `.engine` files were not kept (they only run on the GPU type and
TensorRT version they were built with; rebuild with `yolo export format=engine half=True`).

---

## 3. Quick Start

```bash
pip install ultralytics==8.4.170 onnxruntime onnxslim
```

```python
import torchvision
from ultralytics import YOLO

det = YOLO("best_y26n_p2.pt")                       # or best_y26n_musgd.pt / best_y26s.pt / best_y11n.pt
r = det.predict("car.jpg", imgsz=640, conf=0.25, nms=False, verbose=False)[0]   # nms=False: the NMS-free head
boxes, scores = r.boxes.xyxy, r.boxes.conf
keep = torchvision.ops.nms(boxes, scores, 0.5)      # one plate = one box: merge the rare double boxes before OCR
print(len(boxes), "box(es) ->", len(keep), "plate(s):", [round(s, 2) for s in scores[keep].tolist()],
      [[round(v) for v in b] for b in boxes[keep].tolist()])

r2 = det.predict("car.jpg", imgsz=640, conf=0.25, verbose=False)[0]               # default: one-to-many head + NMS
print("NMS head:", len(r2.boxes), "plate(s):", [round(s, 2) for s in r2.boxes.conf.tolist()])

onnx = det.export(format="onnx", imgsz=640, nms=False)    # NMS-free graph: output [1, 300, 6] = x1 y1 x2 y2 score class
r3 = YOLO(onnx, task="detect").predict("car.jpg", imgsz=640, conf=0.25, device="cpu", verbose=False)[0]
print("ONNX on the CPU:", len(r3.boxes), "box(es):", [round(s, 2) for s in r3.boxes.conf.tolist()])
```

We ran exactly this code (26 Oct 2026, local RTX 5060 Ti, Ultralytics 8.4.170) with `car.jpg` =
`blpd_ext_0127176e15c4.jpg` from the dataset's `e2e_test` (truth ঢাকা মেট্রো-গ ১৭-৪১১৯). Output:

```
2 box(es) -> 2 plate(s): [0.9, 0.29] [[107, 370, 175, 402], [247, 162, 265, 173]]
NMS head: 2 plate(s): [0.9, 0.29]
ONNX on the CPU: 1 box(es): [0.9]
```

The plate is found at 0.90 by both heads; the second, low-score box (18 x 11 px, far in the background) is near the
0.25 threshold, and the ONNX file (fixed 640 x 640 square letterbox instead of PyTorch's rectangle) scores it just
below. Load ONNX files with `device="cpu"` unless `onnxruntime-gpu` is installed: on a CUDA machine Ultralytics
otherwise tries to install `onnxruntime-gpu` by itself. To read the plate, crop the box and use a team reader (e.g.
`weights/paper24/MD.Rokib.Hasan.Oli/` or `weights/paper02/MD.Rokib.Hasan.Oli/best_lpnr_v12n.pt`).

---

## 4. Model Cards

All five: Ultralytics 8.4.170, COCO-pretrained start (`yolo11n.pt`, `yolo26n.pt`, `yolo26s.pt`; the P2 model is
`yolo26n-p2.yaml` with the `yolo26n.pt` backbone transferred, 360 of 902 tensors), one class `plate`, 640 px, 60 epochs
planned, batch 16, seed 42, deterministic, AMP, default Ultralytics augmentation (mosaic off in the last 10 epochs, HSV,
flips, RandAugment). **Data:** 498R All-in-One v1.1 detection — train 8,574 photos (4,067 human + 4,507 auto-labelled
boxes), val 896, test 1,153 (human boxes only; grouped duplicate-free split, seed 42).

| | YOLO11-N | YOLO26-N SGD | YOLO26-N MuSGD | YOLO26-N-P2 MuSGD | YOLO26-S MuSGD |
|---|---|---|---|---|---|
| File | `best_y11n.pt` | `best_y26n_sgd.pt` | `best_y26n_musgd.pt` | `best_y26n_p2.pt` | `best_y26s.pt` |
| Optimiser | SGD lr0 0.01 | SGD lr0 0.01 | MuSGD lr0 0.01 | MuSGD lr0 0.01 | MuSGD lr0 0.01 |
| Params (M) / GFLOPs (training log) | 2.59 / 6.5 | 2.50 / 5.9 | 2.50 / 5.9 | 2.52 / 7.6 | 9.95 / 22.7 |
| Epochs run (best) | 60 (55) | 60 (55) | **50 (50)**, time cap | 60 (55) | 60 (49) |
| Training time (min), peak GPU memory | 113, 2.9 GB | 128, 3.1 GB | 141, 3.1 GB | 134, 5.1 GB | 175, 6.0 GB |
| Val mAP50 / mAP50-95 | 97.09 / 60.19 | 97.36 / 59.98 | 97.65 / 60.10 | 97.64 / 60.52 | 97.32 / 60.28 |
| Test P / R | 96.25 / 97.23 | 97.05 / 97.13 | 97.79 / 96.36 | 97.04 / 97.28 | 97.56 / 96.74 |
| Test mAP50 / mAP50-95 | 97.80 / 44.68 | 98.51 / 45.83 | 98.69 / 45.36 | 97.98 / 46.16 | 99.10 / 46.39 |
| Test mAP50-95 without ridoykj | 65.01 | 64.73 | 64.63 | 64.80 | 64.73 |
| **Plates found** (NMS / NMS-free head) | 97.73 / – | 97.63 / 97.33 | 97.82 / 96.83 | **98.42 / 98.12** | 97.73 / 97.53 |
| Found: night (71) / small (24) | 93.0 / 95.8 | 87.3 / 95.8 | 88.7 / 95.8 | 94.4 / 100 | 88.7 / 87.5 |
| Found: synthetic fog / **rain** (1,011 each) | 93.1 / 63.0 | 95.5 / 60.6 | 95.8 / **82.0** | 94.5 / **82.4** | 96.1 / **78.6** |
| False alarms (160 plate-free photos) | 9 | 7 | 9 | 9 | 8 |
| GPU, PyTorch FP16 ms (RTX 5060 Ti; NMS / free) | 7.6 | 8.3 / 8.1 | 8.4 / 8.3 | 10.0 / 9.9 | 8.5 / 8.1 |
| CPU ONNX, network ms (4 threads; NMS / free) | 18.5 | 15.8 / 16.1 | 15.4 / 15.9 | 23.4 / 23.9 | 50.0 / 48.9 |
| CPU post-processing ms (NMS / free) | 7.7 | 11.0 / 0.27 | 7.5 / 0.27 | **80.4 / 0.27** | 6.7 / 0.27 |

Timings: our PC, one job at a time, batch 1, 640 px, median of 200 (GPU) / 100 (CPU) test photos; GPU = whole
`predict` call (pre + inference + post). Kaggle T4 TensorRT numbers are in section 5.3.

---

## 5. Results Summary

### 5.1 Which detector?

On the clean test split the five detectors are **statistically the same** (largest gap: YOLO26-N-P2 vs YOLO26-N,
13 vs 7 plates, p = 0.26; only 5 plates are missed by all five models). Choose by deployment:

* **CPU / laptop / edge:** YOLO26-N (or -P2) with the **NMS-free head** — no NMS code, no IoU threshold, ONNX output
  `[1, 300, 6]`; network + post-processing 16 ms on 4 CPU threads vs 26 ms for YOLO11-N. Merge double boxes before OCR.
* **GPU (PyTorch / TensorRT):** any of them; YOLO11-N is slightly faster on the GPU (7.6 vs 8.3 ms PyTorch FP16).
* **Hard plates:** YOLO26-N-P2 finds the most plates overall, at night and small (all n.s. on our small subsets); use it
  only NMS-free on a CPU (its NMS costs 80 ms).

### 5.2 NMS-free vs NMS head (same weights, 1,011 test plates)

| Model | Found % NMS / free | Plates only NMS vs only free | Double boxes NMS / free | CPU post ms NMS / free |
|---|---|---|---|---|
| YOLO26-N MuSGD (stopped at 50) | 97.82 / 96.83 | 11 vs 1 (p = 0.006) | 1 / 14 | 7.5 / 0.27 |
| YOLO26-N SGD | 97.63 / 97.33 | 3 vs 0 (p = 0.25) | 0 / 18 | 11.0 / 0.27 |
| YOLO26-N-P2 MuSGD | 98.42 / 98.12 | 5 vs 2 (p = 0.45) | 0 / 13 | 80.4 / 0.27 |
| YOLO26-S MuSGD | 97.73 / 97.53 | 2 vs 0 (p = 0.50) | 1 / 24 | 6.7 / 0.27 |

The stopped YOLO26-N run's NMS-free head is the only one that is significantly worse: ProgLoss shifts the training loss
to that head in the last epochs, which this run never reached (PDF section 7).

### 5.3 TensorRT on the Kaggle T4 (batch 1, pre + inference + post)

| Model, head | TRT FP16 mAP50 / ms | TRT INT8 mAP50 / ms |
|---|---|---|
| YOLO26-N MuSGD, NMS-free | 97.48 / 5.55 | 97.44 / 5.62 |
| YOLO26-S MuSGD, NMS-free | 97.83 / 5.78 | 97.78 / 5.76 |
| YOLO26-N-P2 MuSGD, NMS-free | 97.87 / 5.85 | 97.35 / 5.80 |
| YOLO26-S MuSGD, NMS | 98.75 / 6.43 | 97.95 / 6.60 |
| YOLO11-N, NMS | – / 6.28 | 96.91 / 7.77 |

INT8 keeps the accuracy but is not faster at batch 1: the network is about 2.5 ms of each call. The YOLO26-N and
YOLO11-N timings ran while another job shared the Kaggle CPU (upper bounds). INT8 calibration: 30 % of the validation
split.

### 5.4 Synthetic rain and fog (copies of all 1,153 test photos, NMS head)

| Comparison | Rain: found % (plates only a vs only b, p) | Fog: found % (plates, p) |
|---|---|---|
| YOLO26-N MuSGD vs YOLO26-N SGD (optimiser only) | 82.00 vs 60.63 (232 vs 16, p < 0.0001) | 95.75 vs 95.45 (19 vs 16, p = 0.74) |
| YOLO26-N SGD vs YOLO11-N SGD (architecture only) | 60.63 vs 63.01 (85 vs 109, p = 0.10) | 95.45 vs 93.08 (34 vs 10, p = 0.0004) |

One seed per model and synthetic weather only: a lead for the next experiment (3 seeds per optimiser, real rain
photos), not yet a result. Ultralytics' MuSGD also trains the classification head at 3x the learning rate, so the effect
cannot yet be pinned on the Muon term.

### 5.5 Original-paper check (COCO val2017, released weights, pycocotools)

| Model | Paper NMS / e2e | Ours NMS / e2e | Paper T4 ms | Ours T4 TensorRT FP16 ms (model / whole call) |
|---|---|---|---|---|
| YOLO26-N | 40.9 / 40.1 | **40.81 / 40.06** | 1.7 | 2.52 / 3.71 |
| YOLO26-S | 48.6 / 47.8 | **48.62 / 47.78** | 2.5 | 2.64 / 3.78 |
| YOLO26-M | 53.1 / 52.5 | **53.09 / 52.50** | 4.7 | 4.92 / 6.10 |
| YOLO11-N | 39.5 / – (docs) | **39.45** / – | 1.5 (docs) | 2.19 / 4.16 |

The paper's accuracy table is reproduced to within 0.1. The Kaggle CPU numbers in `results_original_check.md` came from
4 shared virtual cores with another job running and cannot check the paper's CPU column (see 4 for clean CPU timings).

---

## 6. Reproducing the Results

| Step | How |
|---|---|
| Training + heads + speed | Kaggle notebook `src/MD.Rokib.Hasan.Oli/paper33/kaggle/paper33_kaggle.ipynb` (GPU T4 x2, Internet on, `FAST = False`, about 8 h); the run as it happened: `notebooks/MD.Rokib.Hasan.Oli/paper33_run.ipynb` |
| COCO check | `paper33_original_check.ipynb` (about 1 h); run as it happened: `notebooks/MD.Rokib.Hasan.Oli/paper33_check_run.ipynb` |
| Local checks (paired tests, rain / fog, clean timings) | `report/MD.Rokib.Hasan.Oli/paper33/source/`: `local_check.py <unpacked results.zip>` → `analysis.py <run> <check>` → `charts.py` → `build_report.py` |
| Team table | `python docs/results/add_results.py paper33_results.zip --member "Rokib Hasan Oli"` (done 26 Oct 2026: 5 detector rows) |

The kit was fixed after this run (26 Oct 2026): longer time caps (YOLO26-N 3.2 h), timing jobs no longer share the CPU,
INT8 mAP printed in %, the COCO check runs its jobs one after the other, its GitHub zip leaves out the prediction files.

---

## 7. Limitations

* **YOLO26-N (MuSGD) stopped at epoch 50 of 60** (time cap 2.3 h): learning rate not fully decayed, no mosaic-free last
  10 epochs, NMS-free head under-trained. Its scores are a lower bound; a 3 h re-run with the fixed kit is planned.
* **One seed per model.** Clean-test differences are within noise; the rain effect needs more seeds.
* **Synthetic rain and fog only**; the dataset has no real rain photos. Night (71 plates) and small-plate (24 plates)
  subsets are too small for significant differences.
* **Timings:** Kaggle's YOLO11-N speed job and YOLO26-N heads job ran at the same time; use the local timings (4)
  or the YOLO26-S / P2 heads timings for clean comparisons. CPU numbers depend strongly on the CPU and thread count.
* **mAP50-95 depends on the label style** of each source (ridoykj boxes sit on the plate edge): compare detectors on
  plates found or on mAP50-95 without ridoykj.

---

## 8. Differences from the Original Paper

* The paper is a review that copies Ultralytics' COCO tables; it trains nothing. We trained on Bangladeshi plates (one
  class) and checked its COCO numbers with the released weights.
* The paper describes YOLO26 as NMS-free; in Ultralytics 8.4.170 the NMS-free head is opt-in (`nms=False`), the default
  still runs NMS. We report both heads.
* The paper's MuSGD has no settings; Ultralytics' code mixes a full SGD step with a 0.2 x Muon step on conv / linear
  weights and trains the classification head at 3 x the learning rate. Our "MuSGD vs SGD" tests that recipe.
* The promised Jetson, INT8 and small-object numbers are absent from the paper; our INT8 numbers are on a T4.

---

## 9. File Integrity

| File | SHA-256 |
|---|---|
| `best_y11n.pt` | `3f854c08f4604398614e2b0740550ce9650e62b95eeb09e769336d89b75ba930` |
| `best_y26n_musgd.pt` | `93146cb25afd684e2094399e8e6189c2303665945bc0dee2ff0f1a474287f808` |
| `best_y26n_p2.pt` | `3c528ea10839744fc27e4563205621406a5318a986ce1bb4ffc985badeeb547b` |
| `best_y26n_sgd.pt` | `0c7c00c2a47183c20ff35f04d6557c0b784b358f6de9e0e570caeff685a30841` |
| `best_y26s.pt` | `33c1a86e533137d8eb4a546753d72c922f5bd55ffbb92030ea92c3523de601ab` |
| `paper33_results.zip` | `b408c8fc01ba1a52f16b152a7169982c0f8b19c24ea8f7cc3bffa838add5a168` |
| `paper33_check_results.zip` | `7c992a28ec63d1279523ec187aedb24ac0d092fca545dd654a514f59eff275eb` |
| `Paper33_Study_Report.pdf` | `a87f92bee4f46981343a0288b65bcc8f21f8d241cedd8b4a0aad8375850ccc25` |

Check on Windows: `certutil -hashfile best_y26n_p2.pt SHA256`; Linux / macOS: `sha256sum best_y26n_p2.pt`.

---

## 10. Licence and Data Notes

* **Weights:** fine-tuned from Ultralytics' COCO-pretrained models, so they fall under **AGPL-3.0** like Ultralytics.
  Commercial use needs an Ultralytics Enterprise licence.
* **Training data:** 498R BanglaLPR All-in-One v1.1, built from public Bangladeshi plate datasets with their own
  licences (CC BY 4.0, CC BY-NC-SA 4.0, Apache-2.0, ...; see the dataset README). The BD-ALPDR part is non-commercial.
* **COCO val2017** (check only): images under Flickr terms, annotations CC BY 4.0.

---

## 11. Citation

```bibtex
@article{sapkota2025yolo26,
  title   = {{YOLO26}: Key Architectural Enhancements and Performance Benchmarking for Real-Time Object Detection},
  author  = {Sapkota, Ranjan and Cheppally, Rahul Harsha and Sharda, Ajay and Karkee, Manoj},
  journal = {arXiv preprint arXiv:2509.25164},
  year    = {2025}
}
@software{ultralytics_yolo26,
  title   = {Ultralytics {YOLO26}},
  author  = {Jocher, Glenn and Qiu, Jing},
  year    = {2025},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}
```
