# README.md — Paper #6: YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles (re-run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — re-implementation of *"YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles"* (Ahamed & Ismail, Authorea preprint 2025, doi 10.22541/au.173748276.69471512/v1; Paper #6)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 24 October 2026 (Kaggle run of 24 October 2026, dataset **498R BanglaLPR All-in-One v1.1**)
> **Full study report:** [`Paper06_Study_Report.pdf`](Paper06_Study_Report.pdf) (paper study, training check, findings, faculty guide)
>
> **Other files for this paper in this repo:** [finished Kaggle notebook](../../../notebooks/MD.Rokib.Hasan.Oli/paper06_run.ipynb) and [its log](../../../notebooks/MD.Rokib.Hasan.Oli/paper06_run.log) · [Kaggle kit + START_HERE guide](../../../src/MD.Rokib.Hasan.Oli/paper06/) · [reading notes + report source](../../../report/MD.Rokib.Hasan.Oli/paper06/). The full `paper06_outputs.zip` (with every `last.pt`) is on the team Drive in `498R / weights / paper06 /`.

This package contains our two trained **one-class plate detectors** from the paper — **YOLOv8n** and **YOLOv8s**,
COCO-pretrained and fine-tuned on the human plate boxes of our dataset — plus the results of the paper's reader (stock
**EasyOCR** `bn`, no training) on our 651 human-typed test plates and of the whole system (detector + EasyOCR) on 204
whole photos, with logs, per-plate predictions and code. All numbers come from the run files in `paper06_results.zip`
and the local checks described in the PDF (section 8 and sections 10-13).

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
| **Task** | Find Bangladeshi number plates in a street / CCTV photo (one class, `plate`); the paper then reads the crop with stock EasyOCR |
| **Models** | `best_det_v8n.pt` — YOLOv8n, 3.01 M parameters · `best_det_v8s.pt` — YOLOv8s, 11.14 M parameters |
| **Detection test (1,153 photos, 1,011 plates)** | mAP50 **97.39 / 97.61 %** · plates found (conf 0.25, IoU 0.5) **96.44 / 97.13 %** (paper: mAP 94.0 %, on its own photos) |
| **mAP50-95** | 44.97 / 45.88 % — **64.27 / 64.45 % without the ridoykj source**, whose labels sit exactly on the plate edge (paper: about 59 %) |
| **Paper's reader, stock EasyOCR, 651 test plates** | char accuracy **60.45 %** · plates read exactly **13.52 %** (paper: 92.7 % "character accuracy", undefined) |
| **Whole system, 204 photos** | detector + EasyOCR: **11.76 / 10.29 %** exact · same crops read by paper #2's char-YOLO12n: **84.80 / 88.24 %** (local check) |
| **Speed (Tesla T4, batch 1)** | YOLOv8n 113.8 FPS PyTorch FP32, 182.9 FPS TensorRT FP16 · YOLOv8s 109.6 / 162.5 FPS · EasyOCR 90.9 ms per plate (paper: 78 FPS, GPU not named) |
| **Training** | 120 of 120 epochs each, 130.9 / 144.5 min, Kaggle Tesla T4 x2 (one job per GPU), best epochs 90 / 79 |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 · EasyOCR 1.7.2 · dataset v1.1 |

(Pairs of numbers: YOLOv8n / YOLOv8s.)

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `Paper06_Study_Report.pdf` | 5.3 MiB | Full study report (paper, results, analysis, faculty guide), 20 pages |
| `best_det_v8n.pt` | 5.9 MiB | YOLOv8n plate detector, Ultralytics checkpoint, best epoch 90 of 120 |
| `best_det_v8s.pt` | 21.5 MiB | YOLOv8s plate detector, Ultralytics checkpoint, best epoch 79 of 120 |
| `paper06_results.zip` | 2.3 MiB | Results archive, **no model weights** (see 2.2) |
| `results_vs_paper.md` | — | The kit's ours-vs-paper table (with the per-condition tables) |

Both weight files are under GitHub's 25 MB web-upload limit.

### 2.2 Inside `paper06_results.zip`

| Path | What |
|---|---|
| `runs/det_v8n/`, `runs/det_v8s/` | `results.json` (val/test P, R, mAP, plates found by source and size, per-condition scores, speed for FP32/FP16/TensorRT/CPU, all settings), `results.csv` (per-epoch curves), `args.yaml`, `results.png`, PR curve, confusion matrix, `labels.jpg` |
| `runs/easyocr_crops/` | `results.json` (EasyOCR on the 651 test crops: raw and 4x/grey/CLAHE, by source and condition, real fog, synthetic rain, ms per plate) and `ocr_preds_test.csv` / `ocr_preds_test_prep.csv` (one line per plate: truth, reading, confidence) |
| `runs/e2e_v8n/`, `runs/e2e_v8s/` | whole-system results on the 204 e2e photos and `e2e_preds.csv` (one line per photo) |
| `logs/` | the seven job logs |
| `code/` | the exact kit code used (`det.py`, `p06_easyocr.py`, `metrics.py`, `vocab.py`, `common.py`, ...) |
| `environment.json`, `results_vs_paper.md` | versions (incl. `dataset_version: 1.1`) and the comparison table |

### 2.3 Not in this folder

`paper06_outputs.zip` (124 MiB: everything above plus `best.pt`, `last.pt`, `best.onnx` and `best.fp16.onnx` of both
models) is on the team Drive: **`498R / weights / paper06 /`**. The TensorRT `.engine` files were not kept (they only
run on the GPU type they were built on; rebuild them with `yolo export format=engine half=True`).

---

## 3. Quick Start

```bash
pip install ultralytics==8.4.170 easyocr==1.7.2
unzip paper06_results.zip "code/*"        # the kit's EasyOCR line joining is in code/p06_easyocr.py
```

```python
import sys; sys.path.insert(0, "code")              # kit code from paper06_results.zip
import cv2, easyocr
from PIL import Image
from ultralytics import YOLO
from p06_easyocr import read_lines                  # EasyOCR boxes -> text, top line first (as in our tests)

det = YOLO("best_det_v8n.pt")                       # or best_det_v8s.pt
img = cv2.imread("car.jpg")                         # any street / CCTV photo (BGR)
r = det.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
print(len(r.boxes), "plate(s):", [round(c, 2) for c in r.boxes.conf.tolist()])

if len(r.boxes):
    x1, y1, x2, y2 = r.boxes.xyxy[int(r.boxes.conf.argmax())].tolist()     # the most confident plate
    pw, ph = 0.08 * (x2 - x1), 0.08 * (y2 - y1)                             # + 8 % margin
    crop = img[max(0, int(y1 - ph)):int(y2 + ph), max(0, int(x1 - pw)):int(x2 + pw)]
    text, conf = read_lines(easyocr.Reader(["bn"]), Image.fromarray(crop[:, :, ::-1]))
    print("EasyOCR (the paper's reader, baseline only):", text, round(conf, 2))

    import vocab                                    # better reader: paper #2's character detector
    lpnr = YOLO("best_lpnr_v12n.pt")                # weights/paper02/MD.Rokib.Hasan.Oli/best_lpnr_v12n.pt
    res = lpnr.predict(crop, imgsz=640, conf=0.25, verbose=False)[0]
    toks = [[lpnr.names[int(c)]] + b.tolist() for b, c in zip(res.boxes.xyxy.cpu(), res.boxes.cls.cpu())]
    print("char-YOLO12n (paper #2):", vocab.plate_text(toks)[0])
```

We ran exactly this code (24 Oct 2026, local RTX 5060 Ti) on `blpd_ext_0127176e15c4.jpg` of the dataset's `e2e_test`
(truth ঢাকা মেট্রো-গ ১৭-৪১১৯). Output:

```
1 plate(s): [0.79]
EasyOCR (the paper's reader, baseline only): 'ঢ ১৭-৪১১৮ 0.42
char-YOLO12n (paper #2): ঢাকা মেট্রো-গ ১৭-৪১১৯
```

The EasyOCR reading is the same as in our Kaggle run (`runs/e2e_v8n/e2e_preds.csv`), and the char-YOLO reading is the
same as in our local reader-swap test. **Use the detectors; do not use stock EasyOCR to read plates** (section 5.2):
EasyOCR is kept only as the baseline of our results table.

---

## 4. Model Cards

### 4.1 `best_det_v8n.pt` — YOLOv8n plate detector

| | |
|---|---|
| Architecture | Ultralytics YOLOv8n (anchor-free, 3.01 M parameters, 8.2 GFLOPs), COCO-pretrained `yolov8n.pt`, one class `plate` |
| Data | 498R All-in-One v1.1 detection: train = the 4,067 photos with **human** boxes (KUET, LPDB-A, ridoykj, mahedi, NightLPBD, tusar, towniyan; the 4,507 auto-labelled photos were not used, as the paper uses hand-drawn boxes), val 896, test 1,153 (human boxes only; grouped duplicate-free split, seed 42) |
| Training | 120 epochs, 640 px, batch 16, SGD (lr0 0.01, momentum 0.937), default Ultralytics augmentation (mosaic, HSV, flips, RandAugment), seed 42, deterministic, AMP; peak GPU memory 2.6 GB |
| Best epoch | 90 (validation mAP50 97.70 %, mAP50-95 60.68 %) |
| Test | P 96.57 % · R 94.07 % · mAP50 **97.39 %** · mAP50-95 44.97 % (64.27 % without ridoykj) · plates found **96.44 %** · plates found by source: KUET 98.5, LPDB-A 96.2, ridoykj 96.7, NightLPBD 80.0 % |
| By condition (test mAP50) | clear 97.7 · night 94.3 · blurry 98.6 · small plates 94.8 · synthetic rain 77.5 · synthetic fog 94.8 · false alarms on 160 plate-free photos 6.25 % |
| Speed (T4, batch 1, 640 px, incl. pre/post-processing) | PyTorch FP32 113.8 FPS (8.78 ms) · PyTorch FP16 107.7 · TensorRT FP16 **182.9 FPS (5.47 ms)** · CPU ONNX (Kaggle 4-core VM) 16.0 FPS |

### 4.2 `best_det_v8s.pt` — YOLOv8s plate detector

| | |
|---|---|
| Architecture | Ultralytics YOLOv8s (11.14 M parameters, 28.6 GFLOPs), COCO-pretrained `yolov8s.pt`, one class `plate` |
| Data / training | as 4.1 (120 epochs, 144.5 min) |
| Best epoch | 79 (validation mAP50 97.61 %, mAP50-95 61.43 %) |
| Test | P 96.18 % · R 94.71 % · mAP50 **97.61 %** · mAP50-95 45.88 % (64.45 % without ridoykj) · plates found **97.13 %** |
| By condition (test mAP50) | clear 98.0 · night 93.4 · blurry 98.4 · small plates 85.9 (22 photos) · synthetic rain 80.4 · synthetic fog 93.2 · false alarms 6.25 % |
| Speed (T4, batch 1) | PyTorch FP32 109.6 FPS (9.12 ms) · FP16 105.2 · TensorRT FP16 **162.5 FPS (6.15 ms)** · CPU ONNX 6.6 FPS |

**Which one?** On the same 1,011 test plates YOLOv8s finds 17 plates that YOLOv8n misses and misses 10 that YOLOv8n
finds (McNemar p = 0.25, a tie). Take **YOLOv8n** for speed (3.7x fewer parameters, 2.4x faster on CPU: 16.0 vs 6.6 FPS).

---

## 5. Results Summary

### 5.1 Detector: ours vs the paper (different data: a reference, not a duel)

| Metric | Paper (page) | YOLOv8n | YOLOv8s |
|---|---|---|---|
| Precision (%) | 93.5 (p.9, Table 1) | 96.57 | 96.18 |
| Recall (%) | 92.3 (p.9, Table 1) | 94.07 | 94.71 |
| mAP50 (%) | 94.0 ("mAP", p.9) | **97.39** | **97.61** |
| mAP50-95 (%) | about 59 (Fig. 6, p.11) | 44.97 | 45.88 |
| mAP50-95 without ridoykj (%) | — | 64.27 | 64.45 |
| Plates found, conf 0.25, IoU 0.5 (%) | 94.8 "detection accuracy" (undefined) | 96.44 | 97.13 |
| FPS, T4 batch 1: PyTorch FP32 / TensorRT FP16 | 78 (GPU not named, p.12) | 113.8 / 182.9 | 109.6 / 162.5 |

**Why mAP50-95 is low:** the ridoykj source (274 of 1,153 test photos) boxes the plate exactly at its edge, while the
other sources leave a margin. All our detectors find those plates (mAP50 96-99.5) but draw boxes 1.20x wider and 1.36x
higher than ridoykj's labels, so ridoykj alone scores 22-24 mAP50-95 for every model we trained (also paper #2's
YOLO12s). Validation has fewer ridoykj photos (11 % vs 24 %), hence 60.7 on validation.

**Against other detectors on the same test photos:** paper #2's YOLO12s plate detector (trained on human + auto boxes)
finds 98.32 % (24 vs 5 plates against YOLOv8n, p = 0.0005); Md. Yousuf's YOLOv8n from paper #3 (human + auto boxes,
60 epochs, same detection split) finds 98.12 % (not paired). Training on the auto-labelled photos as well helps.

### 5.2 The paper's reader: stock EasyOCR on our 651 human-typed test plates

| | Paper (page) | Ours |
|---|---|---|
| Character accuracy (100 − CER) | 92.7 % (p.11, undefined) | **60.45 %** |
| Plates read exactly | not reported | **13.52 %** |
| With 4x upscale + grey + CLAHE ("preprocessing", p.8) | — | 41.80 % char · 4.92 % exact (69 plates lost, 13 gained) |
| Clear / night / blurry crops, char accuracy | clean 97.5 / low light 91.5 / — | 66.84 / 49.69 / 38.74 % |
| Real fog copies / synthetic rain, char accuracy | weather 89.3 / rain 89.3 | 59.53 / 44.04 % |
| Time per plate (T4) | — | 90.9 ms |

On the **same 651 crops**, readers trained on our data read far more plates exactly: paper #2's char-YOLO12n 87.71 %
(487 plates only it reads vs 4 only EasyOCR reads, p < 0.0001), paper #24's char-YOLOv8n 86.79 %, CRNN 73.43 %.
EasyOCR gets the letters line right on 19.2 % of plates and the 6 digits on 32.7 %; 39 % of its readings contain
English digits (৪ read as "8"). A local re-run (EasyOCR 1.7.2) gives the same 60.45 % char accuracy; an output
allowlist of plate characters (13.98 % exact, p = 0.81) or a 2x upscale (13.06 %) does not help.

### 5.3 Whole system: 204 e2e photos (144 BLPD external + 60 NightLPBD)

| System | Plate found | Plates exact | ms per photo |
|---|---|---|---|
| YOLOv8n + EasyOCR (Kaggle) | 94.61 % | 11.76 % | 68.7 (T4) |
| YOLOv8s + EasyOCR (Kaggle) | 96.08 % | 10.29 % | 69.5 (T4) |
| YOLOv8n + paper #2 char-YOLO12n, **same crops** (local) | 94.61 % | **84.80 %** | 31.9 (RTX 5060 Ti) |
| YOLOv8s + paper #2 char-YOLO12n, **same crops** (local) | 96.08 % | **88.24 %** | 28.6 (RTX 5060 Ti) |
| Paper #2's own cascade (YOLO12s + char-YOLO12n), local = Kaggle | 95.59 % | 86.76 % | 30.2 (RTX 5060 Ti) |

The detector is good enough; the reader is the bottleneck. YOLOv8s + char-YOLO12n vs paper #2's cascade: 7 vs 4 photos
(p = 0.55, a tie). Every photo without a plate box (11 for v8n, 8 for v8s) is a NightLPBD night photo.

---

## 6. Reproducing the Results

1. Notebook `paper06_kaggle.ipynb` (kit folder `docs/Rokib Hasna/6_YOLOv8-Based License Plate Recognition for Bangladeshi Vehicles/kaggle/` in the team workspace; generated by `docs/_kit/build_kits.py`).
2. Kaggle: Accelerator **GPU T4 x2**, Internet on. The notebook downloads dataset v1.1 from the team Drive by itself.
3. *Save Version → Save & Run All*. About 2.8 h. Download `paper06_outputs.zip` from the Output tab.
4. Fill the team table: `python docs/results/add_results.py paper06_results.zip --member "<your name>"`.
5. Local checks (in the paper folder's `report_src/`): `local_check.py` (detectors per source, same-plate tests, reader swap on the 204 photos; needs the run's `results.zip` unpacked and paper #2's weights), `easyocr_check.py` (EasyOCR re-run and cheap fixes), then `analysis.py` → `charts.py` → `build_report.py` for the PDF.

---

## 7. Limitations

- **One seed** per model; the YOLOv8n/v8s difference is within run-to-run noise (section 4.2).
- **Human boxes only**, as in the paper; our teammates' detectors trained with the auto-labelled photos find 1.5-2 points more plates.
- **mAP50-95 depends on label style** across sources (ridoykj); quote mAP50 and plates found, and mAP50-95 per source.
- **Rain and fog detection tests are synthetic**; no real rain photos exist in the dataset. Night: NightLPBD only.
- **No low-resolution OCR test:** no OCR test crop is under 24 px high, so the paper's "low-resolution" group has no counterpart.
- **The reader-swap and EasyOCR-fix results are local checks** (RTX 5060 Ti, 24 Oct 2026), not part of the Kaggle run; their ms are not comparable with the T4 numbers.
- Still photos only, no video.

---

## 8. Differences from the Original Paper

| | Paper | Ours |
|---|---|---|
| Data | 2,600 own photos, augmented **before** the 70/20/10 split, not public | 498R All-in-One v1.1: grouped duplicate-free 70/15/15 split, 7 public sources, human boxes |
| Model | YOLOv8, size not given; text says TensorFlow | Ultralytics YOLOv8n and YOLOv8s (PyTorch) |
| Settings | not given (Fig. 6 shows about 118 epochs) | 120 epochs, 640 px, batch 16, SGD, seed 42 |
| Crop | "segmentation" (not described) | most confident box + 8 % margin |
| Preprocessing | "to enhance clarity" (not described) | tested: raw crop and 4x Lanczos + grey + CLAHE |
| OCR | EasyOCR "configured for Bangla" | EasyOCR 1.7.2 `bn`, no fine-tuning; lines joined top line first |
| Metrics | "detection accuracy", "character recognition accuracy" (undefined) | mAP50, mAP50-95, plates found, char accuracy = 100 − CER, plate exact |

---

## 9. File Integrity

SHA-256 checksums of the files in this folder (and the Drive file):

| File | SHA-256 |
|---|---|
| `best_det_v8n.pt` | `a7377298401ba689833f67a0c98eafba3ac6c0be2754a6fcd1f0d2c071e9c6f8` |
| `best_det_v8s.pt` | `1e245dc5ed4188acec06145581bd2eacbfebcadbb97867ff9517d3c1d91a06cd` |
| `paper06_results.zip` | `0e1dba2da55ea35f2dce51816e7a40321b3198c6934e64a68d2c2ed0ac43ff49` |
| `results_vs_paper.md` | `96cae6c90bf606fb1a2d06eb9d168c47416a882205a0469136ea6ad8b25720c8` |
| `Paper06_Study_Report.pdf` | `891b877e83dbb1cb50530a8d0c4d1c6d3db6b1089b1153c90dcb67f758cc4ded` |
| `paper06_outputs.zip` (Drive) | `51c35bd9622f990b8073acc95e2381d2f6b2742688c0e4f4c5040abde5d5c358` |

The two `.pt` files are byte-identical to `runs/det_v8n|det_v8s/weights/best.pt` inside `paper06_outputs.zip`.
Check on Windows: `certutil -hashfile best_det_v8n.pt SHA256` · Linux/macOS: `sha256sum best_det_v8n.pt`.

---

## 10. Licence and Data Notes

- The detectors were trained with Ultralytics YOLOv8 (AGPL-3.0); the weights follow Ultralytics' licence terms. COCO-pretrained starting weights: `yolov8n.pt` / `yolov8s.pt` from Ultralytics.
- EasyOCR (Apache-2.0) downloads its own `craft_mlt_25k` and `bengali` models; they are not redistributed here.
- The paper publishes no code; this is our re-implementation from the paper's text.
- Training data: 498R BanglaLPR All-in-One v1.1, built from public Bangla plate datasets; each source keeps its own licence (see the dataset's README and `docs/DATASET_SHARING_GUIDE.md`).

---

## 11. Citation

```bibtex
@misc{ahamed2025yolov8,
  title        = {{YOLOv8}-Based License Plate Recognition for {Bangladeshi} Vehicles},
  author       = {Ahamed, Istiak and Ismail, Gazi Mohammad},
  year         = {2025},
  month        = jan,
  howpublished = {Authorea preprint},
  doi          = {10.22541/au.173748276.69471512/v1},
  note         = {Not peer-reviewed}
}
```
