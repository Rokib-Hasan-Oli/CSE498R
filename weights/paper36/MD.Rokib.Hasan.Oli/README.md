# README.md — Paper #36: SAHI, Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection (run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — experiments for *"Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection"* (Akyon, Altinuc & Temizel, IEEE ICIP 2022, arXiv 2202.06934 v5; Paper #36)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 29 October 2026 (Kaggle run of 28 October 2026, dataset **498R BanglaLPR All-in-One v1.1**; local re-check on an RTX 5060 Ti on 29 October; original-paper check on 27 October)
> **Full study report:** [`SAHI_Study_Report.pdf`](SAHI_Study_Report.pdf) (paper study, run check, results, research use, faculty guide; 19 pages)

This package tests whether **SAHI** — running a detector on overlapping 640 x 640 slices of a photo (slicing aided
hyper inference) and training it on such slices (slicing aided fine-tuning, SF) — finds more far / small **Bangla
number plates** than the normal full-image detector. YOLOv8s is trained three ways (640 px, SF, 1280 px) and scored
through the authors' own `sahi` package (pinned 0.12.8) with the paper's merge rule; then the merge is analysed and a
validation-chosen fix is tested. Sources: `paper36_results.zip` and the local checks in
[`report/MD.Rokib.Hasan.Oli/paper36/source/`](../../../report/MD.Rokib.Hasan.Oli/paper36/source/).

---

## Table of Contents

1. [At a Glance](#1-at-a-glance)
2. [What Is in This Package](#2-what-is-in-this-package)
3. [Quick Start](#3-quick-start)
4. [Model Card](#4-model-card)
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
| **Question** | Do slices at test time (SAHI), slice training (SF) or a 1280 px detector find more far plates in our 1280 px photos, and at what cost? |
| **Main result** | The paper's SAHI **hurts**: 640 model AP50 **97.33 → 91.64** with SAHI + full image (bootstrap −5.7, CI −7.5 to −4.1), boxes that are plates 90.6 → 60.7 %, plate-free photos with a box 10 → 32 of 160 (p < 0.0001), 3.5x the time |
| **Why** | Our plates are big (median 17.9 % of the photo width; VisDrone objects 1.4 %); slice edges cut plates and the paper's NMS at IoU 0.5 keeps the boxes on plate parts (8 → 278) |
| **Fix (chosen on val)** | Drop slice boxes that touch an inner slice edge, then NMS IoU 0.5: AP50 **97.51** (= full image, n.s.), 997 vs 992 plates found (7 vs 2, p = 0.18), small-plate list 24 / 24 (full image 22); false alarms stay (29 vs 10 photos) |
| **Labels** | In a seeded sample of 48 slice-only "false" boxes, **25 are real plates** on far / secondary vehicles that the test labels leave out (6 unclear, 17 signs / painted text) |
| **SF and 1280 px** | SF: small-plate photos mAP50 84.5 → **96.3** (+11.8, CI +2.8 to +20.5), clear photos −1.5, overall −1.0 (n.s.), no extra test time. 1280 px: no gain, 83 vs 128 FPS (TensorRT FP16, T4), only 20 epochs |
| **Original paper** | The authors' 20 released VisDrone prediction files re-scored: Table 1 reproduced **80 / 80**; abstract FCOS +6.8 (table +5.2) and TOOD +14.5 (table +14.1) are off |
| **Run health** | 7 / 7 jobs `code=0` in 5.15 h, dataset v1.1, all epochs run, no NaN; every Kaggle number re-scored locally; our merge rebuild = the package box for box |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 · sahi 0.12.8 · Kaggle Tesla T4 x2 · dataset v1.1 |

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `SAHI_Study_Report.pdf` | 2.9 MiB | Full study report, 19 pages |
| `best_det_v8s_sf.pt` | 21.4 MiB | **YOLOv8s trained with SF** (8,574 photos + 11,036 slices, 26 epochs, 640 px), Ultralytics checkpoint (FP16 weights) |
| `best_det_v8s_1280.pt` | 21.5 MiB | YOLOv8s trained and used at 1280 px (20 epochs, batch 8) |
| `sahi_border_predict.py` | — | SAHI + full image with the cut-off-box filter (the val-best setting), standalone, tested |
| `paper36_results.zip` | 2.8 MiB | Results archive of the run, **no model weights** (see 2.2) |
| `results_vs_paper.md` | — | The kit's ours-vs-paper table and the per-condition tables (read section 7 first) |
| `paper36_check_results.zip`, `results_original_check.md` | 0.1 MiB | Original-paper check: the authors' VisDrone files re-scored vs Table 1 |

**Not in this folder:** the 640 px model and the CRNN. The 640 model is tensor for tensor the HVI-CIDNet run's detector,
so use [`weights/paper35/MD.Rokib.Hasan.Oli/best_det_v8s.pt`](../../paper35/MD.Rokib.Hasan.Oli/best_det_v8s.pt); the
CRNN is paper #24's file (`crnn/model_full.pt` SHA-256 `0110631e…f033`). All models of the run are in
`paper36_outputs.zip` (288 MiB) on the team Drive, `498R / weights / paper36 /`.

### 2.2 Inside `paper36_results.zip`

| Path | What |
|---|---|
| `runs/<run>/results.json` | settings, epochs run, val / test mAP, plates found by size / source / condition, speed (detectors); plate exact (CRNN) |
| `runs/sahi/results.json` | the 7 SAHI settings: AP50 all / small / medium / large, plates found and correct, by size and condition, false alarms, ms per photo, e2e plate exact |
| `logs/*.log`, `environment.json`, `code/*.py` | the job logs, versions (dataset_version 1.1) and the exact kit code that ran |
| `results_vs_paper.md` | the same table as in this folder |

---

## 3. Quick Start

```bash
pip install ultralytics==8.4.170 sahi==0.12.8
```

**Full-image detection** (the recommended default; any of the three models):

```python
from ultralytics import YOLO
model = YOLO("best_det_v8s_sf.pt")                       # or weights/paper35/.../best_det_v8s.pt (640 px model)
r = model.predict("photo.jpg", imgsz=640, conf=0.25)[0]  # best_det_v8s_1280.pt: imgsz=1280
print(r.boxes.xyxy, r.boxes.conf)                        # one class: 0 = plate
```

**Far-plate pass** (SAHI + full image, cut-off slice boxes dropped, NMS IoU 0.5):

```bash
python sahi_border_predict.py best_det_v8s_sf.pt photo.jpg
```

Prints `conf x1 y1 x2 y2` per plate. Tested on 29 Oct 2026 on all 1,153 test photos: the same 1,154 boxes as the
analysis code (`sahi_common.merge(rec, 'SAHI_FI', 'BORDER')`, max difference 0). Use it only where far plates matter:
on our test set it is not better than full-image detection and costs about 4x the time.

---

## 4. Model Card

| Item | `best_det_v8s_sf.pt` | `best_det_v8s_1280.pt` |
|---|---|---|
| Architecture | YOLOv8s (11.1 M parameters, 28.6 GFLOPs at 640), 1 class | same, 114.6 GFLOPs at 1280 |
| Start weights | `yolov8s.pt` (COCO) | `yolov8s.pt` (COCO) |
| Training data | v1.1 detection train (4,067 human + 4,507 auto-labelled photos) + 11,036 slices of 640 x 640 (20 % overlap) cut from the 3,329 photos larger than 800 px (a plate kept when ≥ 50 % inside; 1,926 plate-free slices, a quarter kept) | v1.1 detection train, 8,574 photos |
| Schedule | 26 epochs (= the optimiser steps of 60 epochs on photos), batch 16, SGD lr 0.01, seed 42 | 20 epochs, batch 8, SGD lr 0.01, seed 42 |
| Best epoch (val mAP50) | 24 (96.91) | 19 (96.79), still rising |
| Test (1,153 photos, Ultralytics val) | mAP50 97.54, mAP50-95 46.2, plates found 97.73 % | mAP50 97.95, mAP50-95 46.4, plates found 97.43 % |
| Training time | 137 min, Tesla T4 | 135 min, Tesla T4 |
| Speed (T4, TensorRT FP16) | = the 640 model, 128 FPS | 83 FPS |
| Input | BGR photo, letterboxed to 640 | letterboxed to 1280 |

mAP50-95 on our test split is low for every detector because one source (ridoykj) draws its boxes on the plate edge
(see the paper #6 report); compare mAP50 and plates found.

---

## 5. Results Summary

### 5.1 The kit's settings on the 1,153 test photos (1,011 plates, 160 plate-free photos)

| Setting | AP50 | Plates found | Boxes that are plates | Plate-free photos with a box | ms / photo (T4) | e2e plate exact (204) |
|---|---|---|---|---|---|---|
| 640 model, full image (FI) | 97.33 | 992 (98.12 %) | 90.6 % | 10 | 33.8 | 146 |
| 640 model, SAHI (slices only) | 88.78 | 959 (94.86 %) | 60.9 % | 31 | 93.8 | 131 |
| 640 model, SAHI + FI (the paper) | 91.64 | 983 (97.23 %) | 60.7 % | 32 | 118.2 | 138 |
| **640 model, SAHI + FI, cut-off boxes dropped** (val-best) | **97.51** | **997 (98.62 %)** | 77.7 % | 29 | ≈118 | 148 |
| SF model, FI | 96.68 | 986 (97.53 %) | **93.8 %** | 9 | 32.5 | 149 |
| SF model, SAHI + FI | 93.62 | 983 (97.23 %) | 68.4 % | 21 | 119.0 | 140 |
| 1280 px model, FI | 97.10 | 987 (97.63 %) | 88.3 % | 8 | 46.1 | 148 |

AP50 = COCO at conf 0.01, max 500 boxes (the paper's metric); the other columns at conf 0.25, IoU 0.5. Our PC
re-scores the Kaggle numbers to 0.03 (the e2e column is the local read; Kaggle is 1 photo higher in 5 of 7 rows).

### 5.2 Ultralytics val by condition (test mAP50 %, bootstrap CI of the difference)

| Subset (photos / plates) | 640 | SF | SF − 640 | 1280 | 1280 − 640 |
|---|---|---|---|---|---|
| all (1,153 / 1,011) | 98.50 | 97.54 | −0.96 (−1.8 to +0.2) | 97.95 | −0.56 (−1.4 to +0.0) |
| clear (868 / 884) | 99.01 | 97.49 | −1.52 (−2.1 to −0.1) | 98.34 | −0.66 (−1.6 to −0.0) |
| night (71 / 71) | 94.34 | 96.51 | +2.18 (−3.0 to +7.1) | 94.31 | −0.02 (−4.7 to +4.0) |
| small plates (22 / 24) | 84.53 | **96.30** | **+11.77 (+2.8 to +20.5)** | 91.39 | +6.86 (−1.4 to +16.1) |

### 5.3 What to take from it

- Keep the 640 px full-image detector as the default (O2, the 10 FPS goal).
- Never use the paper's merge on plates; if slicing, drop the cut-off slice boxes (`sahi_border_predict.py`).
- SF is the cheap win for far plates (no test-time cost); re-check clear-photo mAP when using it.
- Agree a label rule for far plates before scoring any far-plate method.

---

## 6. Reproducing the Results

1. **Kaggle run:** [`src/MD.Rokib.Hasan.Oli/paper36/kaggle/paper36_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper36/kaggle/paper36_kaggle.ipynb)
   (GPU T4 x2, Internet on, about 5-7.5 h; it downloads dataset v1.1 itself). The run as it ran:
   [`notebooks/MD.Rokib.Hasan.Oli/paper36_run.ipynb`](../../../notebooks/MD.Rokib.Hasan.Oli/paper36_run.ipynb).
2. **Original-paper check:** `paper36_original_check.ipynb` in the same folder (CPU, about 0.5 h).
3. **Local analysis** (`report/.../source/`, needs the kit folder next to it as `../kaggle/` and dataset v1.1 in
   `DATASET_ROOT`): `local_check.py <unpacked results.zip> predict | predict_val | verify | read | fragile | valstats |
   timing` → `visdrone_extra.py` → `analysis.py` → `fp_sample.py` → `charts.py` → `build_report.py`.

---

## 7. Limitations

- One seed per model; the far-plate differences rest on small numbers (33 plates under 40 px wide at 640; small-plate
  list 24 plates).
- The test labels leave out many far plates (section 1), so every precision / false-alarm number is biased against
  slicing.
- The 1280 px model ran only 20 epochs (best at 19), a lower bound for "use a bigger input".
- **Loose rows in `results_vs_paper.md`:** "AP50 small" and "small plates found" are on 5 plates (COCO small); "AP50
  large plates: 640 FI / SAHI + FI" shows only the SAHI + FI value (FI: 97.61). No number is wrong. The kit in
  `src/MD.Rokib.Hasan.Oli/paper36/kaggle/` (updated 29 Oct) splits and rewords these rows and also scores SAHI + FI
  with cut-off slice boxes dropped (on this run's 640 model it reproduces section 5.1 exactly) and writes
  `per_plate.csv`, `e2e_reads.csv` and `boxes.json`; the notebook in `notebooks/` is the run as it ran (the older kit).
- e2e plate exact is fragile: 33 of 198 readings flip when the plate box moves by 1 px.

---

## 8. Differences from the Original Paper

| | Paper | Ours |
|---|---|---|
| Data | VisDrone2019-DET (drone, 10 classes, 47 objects per photo), xView | 498R All-in-One v1.1 (road photos, 1 class, 0.9 plates per photo) |
| Detectors | FCOS, VFNet, TOOD (MMDetection) | YOLOv8s (Ultralytics) |
| SF patches | random 480-640 px, resized to width 800-1333 | fixed 640 x 640 at native resolution, 20 % overlap, photos > 800 px |
| SAHI | 640 x 640, 25 % overlap, NMS IoU 0.5 | the same (through the authors' package), plus merge variants and the cut-off-box filter |
| Extra baselines | none | 1280 px detector; ms per photo; plates found / false alarms at a working threshold; e2e reading |

---

## 9. File Integrity

| File | SHA-256 |
|---|---|
| `best_det_v8s_sf.pt` | `e5a76ac910ffe4321abd360c5b0e73d7e77c4771f2dbbc2b2b86262ae69e9d0c` |
| `best_det_v8s_1280.pt` | `53b3e942aeea2bee50dbcc42831ee757456148cf7f0eadc8fd62cf56b763b239` |
| `paper36_results.zip` | `14b49341f3d92f016569122b2f40dabeea66d93cff80b354395e629fec7dfd6f` |
| `paper36_check_results.zip` | `54535e69ce80557b3af1b8654cbae17df8fced2845d9997422b32657cb1b2f72` |

The two `.pt` files are the run's `runs/<run>/weights/best.pt`, unchanged.

---

## 10. Licence and Data Notes

- `sahi` is MIT-licensed (github.com/obss/sahi); Ultralytics YOLOv8 is AGPL-3.0.
- The models were trained on the 498R All-in-One v1.1 dataset (team use; see its dataset card for the source licences).
- VisDrone images and the authors' prediction files are **not** included; the check notebook downloads them from
  their official sources.

---

## 11. Citation

```bibtex
@inproceedings{akyon2022sahi,
  title     = {Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection},
  author    = {Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  booktitle = {2022 IEEE International Conference on Image Processing (ICIP)},
  pages     = {966--970},
  year      = {2022},
  doi       = {10.1109/ICIP46576.2022.9897990}
}
```
