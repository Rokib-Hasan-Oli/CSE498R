# README.md — Paper #2: Cascaded YOLO for Bangladeshi Licence Plates (re-run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — re-implementation of *"Cascaded YOLO-based approach for Bangladeshi license plate identification and number recognition"* (Sarker & Alam, PLOS One 21(8): e0355858, 2026; Paper #2)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 22 October 2026 (Kaggle run of 21 October 2026, dataset **498R BanglaLPR All-in-One v1.1**)
> **Full study report:** [`Paper02_Study_Report.pdf`](Paper02_Study_Report.pdf) (15 pages: paper study, training check, findings, faculty guide)
>
> **Other files for this paper in this repo:** [finished Kaggle notebook](../../../notebooks/MD.Rokib.Hasan.Oli/paper02_run.ipynb) and [its log](../../../notebooks/MD.Rokib.Hasan.Oli/paper02_run.log) · [Kaggle kit + START_HERE guide](../../../src/MD.Rokib.Hasan.Oli/paper02/) · [reading notes + report source](../../../report/MD.Rokib.Hasan.Oli/paper02/). The full `paper02_outputs.zip` (with every `last.pt`) is on the team Drive in `498R / weights / paper02 /`.

This package contains our two trained models for the paper's cascade — a **YOLO12s plate detector (LPD)** and a
**YOLO12n character detector (LPNR)** that finds every Bangla digit, letter, "মেট্রো" and whole district word as an
object — plus the evaluation results, logs, per-photo predictions and code. All numbers come from the run files in
`paper02_results.zip`.

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
| **Task** | Read Bangladeshi plates end to end: find the plate, then find every token on it, then decode the text |
| **Models** | `best_lpd_v12s.pt` — plate detector, YOLO12s (9.25 M parameters, 23.5 GFLOPs) · `best_lpnr_v12n.pt` — character detector, YOLO12n, 101 Bangla tokens (2.63 M parameters, 7.8 GFLOPs) |
| **Plate detector, test** | mAP@0.5 **98.9 %**, P 96.7 %, R 96.2 %, plates found (conf 0.25) 98.3 % of 1,011 plates |
| **Character reader, 651 test plates** | whole plate exact **87.71 %**, characters correct 97.85 %, CER 2.15 % |
| **Full pipeline, 204 photos** | plate detector → reader: **86.76 %** exact (with the paper's extra vehicle stage: 85.29 %) |
| **Same photos vs the authors' released weights** | daylight tie (93.1 vs 95.1 %, McNemar p 0.58); **night 66.7 vs 38.3 %** (p < 0.001); all 204 photos 85.3 vs 78.4 % (p 0.02) |
| **Speed (Tesla T4, batch 1)** | plate detector 131.6 FPS TensorRT FP16 · 52.9 FPS PyTorch FP32 · full pipeline without vehicle stage 36.2 ms/photo (27.6 photos/s) |
| **Training** | LPD 60 epochs, 178.4 min · LPNR 70 epochs, 94.4 min · Kaggle Tesla T4 x2 (one job per GPU) |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 |

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `Paper02_Study_Report.pdf` | 5.6 MiB | Full study report (paper, results, analysis, faculty guide) |
| `best_lpd_v12s.pt` | 18.0 MiB | Plate detector (one class: `plate`), best epoch 36 of 60 |
| `best_lpnr_v12n.pt` | 5.4 MiB | Character detector (101 classes: Bangla digits, letters, মেট্রো, district words), best epoch 55 of 70 |
| `paper02_results.zip` | 2.2 MiB | Results archive, **no model weights** (see 2.2) |
| `results_vs_paper.md` | — | Ours-vs-paper table and per-condition tables |

### 2.2 Inside `paper02_results.zip`

| Path | What |
|---|---|
| `runs/lpd_v12s/` | `results.json` (all scores, speed, per-condition, per-source plates found), `results.csv` (per-epoch curves), `args.yaml`, plots |
| `runs/lpnr_v12n/` | `results.json`, `results.csv`, `args.yaml`, plots, `ocr_preds_test.csv` / `ocr_preds_val.csv` (one line per plate: truth, prediction, format check, confidence) |
| `runs/cascade_ours/`, `runs/cascade_blpd_weights/` | full-pipeline scores and `e2e_preds_*.csv` (one line per photo) for our weights and for the authors' weights |
| `logs/` | the five job logs |
| `code/` | the exact kit code used (`det.py`, `rec.py`, `p02_cascade.py`, `vocab.py`, `metrics.py`, ...) |
| `environment.json`, `results_vs_paper.md` | versions (incl. `dataset_version: 1.1`) and the kit's comparison table |

### 2.3 Not in this folder

`paper02_outputs.zip` (139.1 MB: everything above plus `last.pt`, ONNX and TensorRT exports, and the authors'
`LPD_Model.pt` / `LPNR_Model.pt` downloaded from github.com/prosensarker4/BLPD for the comparison) is too big for
GitHub. It is on the team Drive: **`498R / weights / paper02 /`**.

---

## 3. Quick Start

```bash
pip install ultralytics==8.4.170 opencv-python
unzip paper02_results.zip "code/*"        # only code/vocab.py is needed, for the decoder
```

```python
import sys; sys.path.insert(0, "code")
import cv2
import vocab                                   # 498R token list + two-row decoder + BRTA format check
from ultralytics import YOLO

lpd = YOLO("best_lpd_v12s.pt")                 # plate detector
lpnr = YOLO("best_lpnr_v12n.pt")               # character detector (101 Bangla tokens)

img = cv2.imread("car.jpg")
for x1, y1, x2, y2 in lpd.predict(img, conf=0.25, verbose=False)[0].boxes.xyxy.cpu().numpy():
    w, h = x2 - x1, y2 - y1                    # pad the plate crop by 8 %, as in the evaluation
    crop = img[max(0, int(y1 - 0.08 * h)):int(y2 + 0.08 * h), max(0, int(x1 - 0.08 * w)):int(x2 + 0.08 * w)]
    r = lpnr.predict(crop, imgsz=640, conf=0.25, verbose=False)[0]
    toks = [[lpnr.names[int(c)], *b] for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.cls.cpu().numpy())]
    text, valid = vocab.plate_text(toks)       # e.g. "ঢাকা মেট্রো-গ ১৭-৪১১৯"
    print(text, "" if valid else "(not a valid BRTA plate format: check it)")
```

`vocab.plate_text` builds the two rows from the token types (district, মেট্রো and letter on top; six digits below,
each row left to right) and returns `valid = False` when the result is not a legal BRTA plate. In our test every
reading flagged invalid was wrong (52 of 52).

---

## 4. Model Cards

### 4.1 `best_lpd_v12s.pt` — plate detector

| | |
|---|---|
| Architecture | Ultralytics YOLO12s, fine-tuned from COCO `yolo12s.pt` |
| Data | 498R All-in-One v1.1 detection: train 8,574 photos (4,067 human + 4,507 auto boxes), val 896, test 1,153 (human boxes only) |
| Training | 640 px, batch 16, 60 epochs, SGD lr0 0.01, seed 42, deterministic, AMP; best epoch 36 |
| Validation | P 96.4, R 94.5, mAP50 96.8, mAP50-95 60.4 |
| Test | P 96.7, R 96.2, mAP50 **98.9**, mAP50-95 46.2 (64.3 without the ridoykj source, see 7) |
| Plates found (conf 0.25, IoU 0.5) | 98.3 % overall; KUET 99.7, LPDB-A 97.6, ridoykj 100, NightLPBD 85.0 |
| Per condition (mAP50) | clear 99.2 · night 98.4 · blurry 99.2 · small plates 93.2 · fog (synthetic) 97.5 · rain (synthetic) 83.1 · false alarms on 160 plate-free photos 10 % |
| Speed, T4, batch 1, 640 px | TensorRT FP16 7.6 ms (131.6 FPS) · PyTorch FP32 18.9 ms (52.9 FPS) · ONNX on the Kaggle CPU 166.8 ms (6.0 FPS) |

### 4.2 `best_lpnr_v12n.pt` — character detector

| | |
|---|---|
| Architecture | Ultralytics YOLO12n, fine-tuned from COCO `yolo12n.pt`, 101 classes |
| Data | 498R All-in-One v1.1 recognition: train 5,465 plate crops, val 536, test 467 (character boxes); 651 test crops with typed plate text |
| Training | 640 px, batch 16, 70 epochs, SGD, seed 42; augmentation without horizontal flip (mirrored Bangla is invalid), mosaic 0.5, degrees 3, shear 2, scale 0.3; best epoch 55 |
| Character boxes, test | P 87.6, R 96.0, mAP50 **97.1**, mAP50-95 66.6 |
| Plate reading, 651 test plates | exact **87.71 %**, characters 97.85 %, CER 2.15 %, WER 3.75 %; 23.6 ms per plate (T4, batch 1) |
| By source (exact) | LPDB-A 96.0 · ANPR 84.8 · KUET 84.7 · NightLPBD 80.8 · BLPD 80.0 |
| By condition (exact) | clear 90.4 · night 83.1 · blurry 79.5 · fog (real copies) 76.9 · rain (synthetic) 74.5 |
| Validation (766 plates) | exact 80.55 % |

---

## 5. Results Summary

### 5.1 Ours vs the paper (different data: a reference, not a duel)

| Metric | Paper (page) | Ours (v1.1) |
|---|---|---|
| LPD test mAP50 / mAP50-95 | 96.3 / 68.5 (p.23) | 98.9 / 46.2 (64.3 without ridoykj) |
| LPNR test mAP50 / mAP50-95 | 94.9 / 76.7 (p.25) | 97.1 / 66.6 |
| Full pipeline, whole plate exact | 95.33 % on 150 photos (p.34) | 85.29 % on 204 photos (86.76 % without vehicle stage) |
| Characters correct | 98.82 % (p.34) | 92.78 % (94.89 % without vehicle stage) |
| Time per photo | 117.16 ms, RTX 3050 (p.35) | 93.3 ms with / 36.2 ms without vehicle stage, T4 |

### 5.2 Same 204 photos: our weights vs the authors' weights (paper's pipeline)

| Photos | Authors' weights | Ours | Only ours / only authors right | McNemar p |
|---|---|---|---|---|
| All 204 | 78.43 % | **85.29 %** | 23 / 9 | 0.020 |
| BLPD external (144, the paper's own photos) | 95.14 % (paper: 95.33 %) | 93.06 % | 5 / 8 | 0.58 (tie) |
| NightLPBD (60) | 38.33 % | **66.67 %** | 18 / 1 | 0.00008 |

The authors' weights reproduce the paper's Table 10 on its own photos (95.14 vs 95.33 %).

### 5.3 The vehicle stage is not needed

Removing the COCO YOLO12l vehicle stage: 7 photos become right, 4 wrong (McNemar p 0.55), and the time per photo drops
from 93.3 to 36.2 ms. 6 of the 7 rescued photos are night photos where the vehicle is missed.

### 5.4 Dataset v1.0 → v1.1

v1.1 cut 91 BLPD "plate crops" that were whole vehicle photos down to the plate. With the same plate detector and the
same 204 photos, the retrained reader went from 83.33 to 85.29 % (with vehicle stage) and from 82.84 to 86.76 %
(without). On the 651 test crops: 83.56 → 87.71 % (19 of those test crops were also fixed).

---

## 6. Reproducing the Results

1. Notebook `paper02_kaggle.ipynb` (kit folder `docs/Rokib Hasna/2_Cascaded YOLO-based ... recognition/kaggle/` in the team workspace; generated by `docs/_kit/build_kits.py`).
2. Kaggle: Accelerator **GPU T4 x2**, Internet on. The notebook downloads dataset v1.1 from the team Drive by itself.
3. *Save Version → Save & Run All*. About 3.2 h. Download `paper02_outputs.zip` from the Output tab.
4. Fill the team table: `python docs/results/add_results.py paper02_outputs.zip --member "<your name>"`.

Notebooks rebuilt after 22 Oct 2026 report TensorRT FP16 FPS and medians; this run was made with the earlier kit
(its own table printed a misleading "7.10 FPS" PyTorch FP16 mean — ignore it; the real values are above).

---

## 7. Limitations

- **Night detection:** at confidence 0.25 the plate detector misses 9 of 60 NightLPBD plates; every night photo with no reading is such a miss.
- **Box style:** the ridoykj source labels tight boxes; our detector draws boxes around the plate rim, so mAP50-95 on ridoykj is 24.2 (mAP50 99.5). It does not affect reading (the crop is padded).
- **CPU speed:** the plate detector alone runs at 6 FPS on the Kaggle CPU (YOLO12s, 640 px); a laptop version needs a smaller model, input or OpenVINO.
- **One seed;** LPD trained 60 epochs (paper 100). The end-to-end test has 204 photos; rain and detection-fog images are synthetic.
- **v1.0 vs v1.1** differences cannot be tested for significance (the v1.0 per-photo predictions were not kept).

---

## 8. Differences from the Original Paper

| | Paper | Ours |
|---|---|---|
| Data | own Khulna photos (daylight) + web + Roboflow, split by image | 498R All-in-One v1.1: merged public Bangla sets, de-duplicated, grouped 70/15/15 split |
| Model sizes | not stated | YOLO12s (LPD), YOLO12n (LPNR) |
| Classes | 104 (incl. duplicates such as "Cox-s Bazar"/"Cox-s-Bazar", "Da"/"DA", "Tha"/"THA") | 101 clean Bangla tokens |
| Epochs / optimiser | 100 and 70, "auto" | 60 and 70, SGD |
| Decoding | Algorithm 1: sort left to right (literally: 1 of 651 plates right); text: order by type | two-row decoder by token type + BRTA format check |
| Vehicle stage | always | measured with and without (not needed) |

---

## 9. File Integrity

SHA-256 checksums of the files in this folder (and the Drive file):

| File | SHA-256 |
|---|---|
| `best_lpd_v12s.pt` | `fe6d63d7ae9df225de77bdee9e671dc385579010ced22ddf92f87063d7f51218` |
| `best_lpnr_v12n.pt` | `5fe8a3c6959b20a02f68e89f1fa0ae8e098a02ebadbd63c7e622e3f2ee68107b` |
| `paper02_results.zip` | `e1980fde8021808d1ea25a544627f059bb4ba192aa7ac4a559f6d4cad0594483` |
| `results_vs_paper.md` | `f0879e4c76489d48845d8d2a35b090e54c0187ecf4e4e4e42a9b981b0c2fcae6` |
| `Paper02_Study_Report.pdf` | `449169da014d08fe9392eb45316e43f22a4c07e08477bfa696be1e818b2e9f1b` |
| `paper02_outputs.zip` (Drive) | `0e292d3a5f43f08c52b895d22a82ce5991793db51200328ef62afbf6ddcd1084` |

Check on Windows: `certutil -hashfile best_lpd_v12s.pt SHA256` · Linux/macOS: `sha256sum best_lpd_v12s.pt`.

---

## 10. Licence and Data Notes

- The models are fine-tuned from Ultralytics YOLO12 weights; Ultralytics models are released under **AGPL-3.0**, so these weights follow AGPL-3.0 terms.
- Training data: 498R BanglaLPR All-in-One v1.1, built from public Bangla plate datasets; each source keeps its own licence (see the dataset's README and `docs/DATASET_SHARING_GUIDE.md`).
- The authors' BLPD weights are **not** redistributed here; they are downloaded from github.com/prosensarker4/BLPD by the notebook.

---

## 11. Citation

```bibtex
@article{sarker2026cascaded,
  title   = {Cascaded {YOLO}-based approach for {Bangladeshi} license plate identification and number recognition},
  author  = {Sarker, Proshenjit and Alam, S. M. Shamsul},
  journal = {PLOS One},
  volume  = {21},
  number  = {8},
  pages   = {e0355858},
  year    = {2026},
  doi     = {10.1371/journal.pone.0355858}
}
```
