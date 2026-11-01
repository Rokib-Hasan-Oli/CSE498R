# README.md — Paper #24: Model Fusion for License Plate Recognition (re-run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — re-implementation of *"Leveraging Model Fusion for Improved License Plate Recognition"* (Laroca et al., CIARP 2023, LNCS pp. 60–75, arXiv 2309.04331; Paper #24)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 24 October 2026 (Kaggle run of 22 October 2026, dataset **498R BanglaLPR All-in-One v1.1**)
> **Full study report:** [`Paper24_Study_Report.pdf`](Paper24_Study_Report.pdf) (paper study, run check, findings, v1.0 → v1.1 comparison, faculty guide)
>
> **Other files for this paper in this repo:** [finished Kaggle notebook](../../../notebooks/MD.Rokib.Hasan.Oli/paper24_run.ipynb) and [its log](../../../notebooks/MD.Rokib.Hasan.Oli/paper24_run.log) · [Kaggle kit + START_HERE guide](../../../src/MD.Rokib.Hasan.Oli/paper24/) · [reading notes + report source](../../../report/MD.Rokib.Hasan.Oli/paper24/). The full `paper24_outputs.zip` (with every `last.pt`) is on the team Drive in `498R / weights / paper24 /`.

The paper reads each plate crop with several different OCR models and combines their answers (highest confidence,
majority vote, vote per character position). We trained four Bangla OCR models on our dataset — **char-YOLOv8n**,
**CRNN-CTC**, a **slot CNN** and **ViTSTR-tiny** — and tested every subset with every fusion rule, all choices made on
validation. The paper's rules do not help on our data, because our models fail on the same plates. Our own rule does:
keep char-YOLO's reading when it fits the **BRTA plate layout** (district, optional মেট্রো, one letter, six digits),
otherwise ask CRNN. This package contains the **two models of that reader**, the run's results, and the study report.
All numbers come from the run files in `paper24_results.zip`; the weights were re-checked on a local PC.

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
| **Task** | Read the text of a Bangladeshi plate crop; combine several OCR models (paper #24) |
| **Models here** | `best_chars_v8n.pt` — char-YOLOv8n (detects each token: district, মেট্রো, letter, digit) · `best_crnn_fp16.pt` — CRNN-CTC (reads the crop as a sequence), the fallback reader |
| **Test set** | 498R v1.1 OCR test: 651 plate crops (human labels), validation 766 |
| **Best single model** | char-YOLOv8n **86.79 %** plates exact (character accuracy 97.49 %) |
| **Paper's best rule (chosen on val)** | char-YOLO + slot CNN + CRNN, MV-BM: **86.33 %** (−0.46 points; 5 plates fixed, 8 broken; McNemar p = 0.58: no gain) |
| **Our format fallback** | char-YOLO, then CRNN only if the BRTA layout fails: **88.63 %** (+1.84 points; **12 fixed, 0 broken**; p = 0.0005) |
| **Why it works** | every reading that broke the layout was wrong: 46 of 46 on test, 92 of 92 on validation |
| **Replicated** | the v1.0 run (20 Oct) gave the same +1.84 points and the same 12 fixed / 0 broken |
| **Speed per plate** | T4 (Kaggle, model step): 9.36 ms char-YOLO, **9.82 ms** fallback (CRNN on 7.1 % of plates) · local RTX 5060 Ti (whole reader): 5.82 ms · local CPU (Core Ultra 7 265K): 12.06 ms |
| **Training** | char-YOLOv8n 100/100 epochs, 52.1 min · CRNN 60 epochs, 23.9 min (best epoch 45) · Kaggle Tesla T4 x2 |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 · dataset v1.1 |

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `Paper24_Study_Report.pdf` | 4.3 MiB | Full study report, 19 pages (paper, run check, results, faculty guide) |
| `best_chars_v8n.pt` | 6.6 MiB | char-YOLOv8n, the run's `runs/chars_v8n/weights/best.pt`, unchanged (Ultralytics, 101 token classes) |
| `best_crnn_fp16.pt` | 14.7 MiB | CRNN-CTC weights (best validation loss, epoch 45 of 60) **stored in FP16**, plus its token list and input settings |
| `paper24_results.zip` | 1.2 MiB | Results archive, **no model weights** (see 2.2) |
| `results_vs_paper.md` | — | The kit's ours-vs-paper table from the run |

**Why is the CRNN stored in FP16?** The original FP32 file is 30.8 MB, over GitHub's 25 MB web-upload limit. The FP16
copy is loaded back as FP32 (see Quick Start). We checked it on all 651 test and 766 validation plates: the same
accuracy as the FP32 file and as Kaggle (73.43 % test, 62.27 % val), the same text on all 766 val plates and on 649 of
651 test plates (largest logit difference 0.02). The full FP32 file is in `paper24_outputs.zip` on the team Drive.

**Why only two models?** These two are the reader we recommend. The kit's val-chosen row also asks ViTSTR as a third
helper; it gains one validation plate and changes no test result (88.63 % either way). The slot CNN and ViTSTR weights
are in `paper24_outputs.zip` on the Drive.

### 2.2 Inside `paper24_results.zip`

| Path | What |
|---|---|
| `runs/chars_v8n/` | `results.json` (box mAP, plate reading, per source / condition, fog and rain tests, speed), `results.csv` (per-epoch curves), `ocr_preds_test.csv` / `ocr_preds_val.csv` (one line per plate: truth, prediction, confidence), plots |
| `runs/crnn/`, `runs/slot/`, `runs/vitstr/` | `results.json` (scores, settings, per-epoch loss) and `ocr_preds_test.csv` / `ocr_preds_val.csv` |
| `runs/fusion/` | `results.json` (every subset × every rule, val and test, McNemar vs the best single model, oracle) and `fusion_table.md` |
| `logs/` | the five job logs |
| `code/` | the exact kit code used (`rec.py`, `seq.py`, `p24_fusion.py`, `metrics.py`, `vocab.py`, ...) |
| `environment.json`, `results_vs_paper.md` | versions (incl. `dataset_version: 1.1`) and the comparison table |

### 2.3 Not in this folder

`paper24_outputs.zip` (118 MiB: everything above plus all four models' weights — char-YOLOv8n `best.pt`/`last.pt`,
CRNN, slot CNN and ViTSTR `best.pt` and `model_full.pt`) is on the team Drive: **`498R / weights / paper24 /`**.

---

## 3. Quick Start

```bash
pip install torch ultralytics opencv-python numpy
unzip paper24_results.zip "code/*"        # the kit code: rec.py (char-YOLO decoder), seq.py (CRNN), metrics.py, vocab.py
```

```python
import sys; sys.path.insert(0, "code")
import cv2, torch
from ultralytics import YOLO
from rec import decode                    # char-YOLO boxes -> plate text (the kit's BRTA decoder)
from seq import CRNN, prep, ctc_decode    # the CRNN model and its pre/post-processing
import metrics, vocab

yolo = YOLO("best_chars_v8n.pt")
ck = torch.load("best_crnn_fp16.pt", map_location="cpu", weights_only=True)
crnn = CRNN(len(ck["classes"]), ck["H"]).eval()
crnn.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["state_dict"].items()})

def read_plate(img):                      # img: one plate crop as a BGR array (cv2.imread)
    b = yolo.predict(img, imgsz=320, conf=0.10, rect=False, device="cpu", verbose=False)[0].boxes
    text, _, _, conf = decode(yolo.names, b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())
    if metrics.is_valid_plate(text):      # BRTA layout OK -> keep char-YOLO's reading (93 % of test plates)
        return text
    with torch.no_grad():                 # otherwise ask CRNN and keep its reading if that one is valid
        out = crnn(prep(img, ck["H"], ck["W"], ck["stack"], False)[None]).float()
    toks, c = ctc_decode(out)
    c_text = vocab.tokens_to_text([ck["classes"][k - 1] for k in toks[0]])
    return metrics.fuse([("chars_v8n", text, conf), ("crnn", c_text, c[0])], "MV-BM+format", "chars_v8n")

print(read_plate(cv2.imread("plate.jpg")))    # e.g. ঢাকা মেট্রো-ল ৪৪-৫১৬০
```

The input is a **plate crop** (from a plate detector, e.g. our paper #2 model
`weights/paper02/MD.Rokib.Hasan.Oli/best_lpd_v12s.pt`), not a whole photo. Keep `rect=False`: it gives the square
320×320 input the kit used; without it Ultralytics letterboxes a single image to a smaller rectangle and reads a few
plates differently (88.17 % instead of 88.63 % in our test). Remove `device="cpu"` to run char-YOLO on a GPU.
**Tested:** this exact code, run on all 651 v1.1 test crops on a local CPU (24 Oct 2026), reads **88.63 %** of the
plates exactly, the same as the Kaggle run.

---

## 4. Model Cards

### 4.1 `best_chars_v8n.pt` — character YOLO (main reader)

| | |
|---|---|
| Architecture | Ultralytics YOLOv8n, started from COCO weights; 101 classes (the dataset's `vocab.json`) = 10 digits, মেট্রো, 26 series letters, 62 district words, 2 short metro forms (চট্ট, রাজ) |
| Data | 498R All-in-One v1.1 `recognition/` (character boxes): train 5,465 crops (2,399 human + 930 verified + 2,136 auto), val 536, test 467 |
| Training | 100 epochs, 320 px, batch 32, SGD, seed 42, no left-right flips (text), mosaic 0.5, ±3° rotation; 52.1 min on a T4 |
| Boxes (test) | mAP50 0.939 · mAP50-95 0.647 · P 0.899 · R 0.901 |
| Plate reading (651 OCR test crops) | **86.79 %** exact (95 % CI 84.0–89.2) · character accuracy 97.49 % · CER 2.51 % · val 82.38 % |
| Decoder | `rec.decode`: boxes with confidence ≥ 0.25; district/মেট্রো/letter on top, the six digits below (by token kind, not by y) |
| Typical errors | a digit boxed twice (25 of 86 wrong plates), one letter found under two classes (14), one wrong symbol in a valid layout (40) |
| Speed | 9.36 ms per plate on a T4 (Kaggle, batch 1) |

### 4.2 `best_crnn_fp16.pt` — CRNN-CTC (fallback reader)

| | |
|---|---|
| Architecture | `seq.CRNN`: VGG-style CNN (64-128-256-256-384-384) → 2-layer BiLSTM (256) → CTC over the same 101 tokens + blank; 7.69 M parameters |
| Input | grey crop, two-row plates cut into top and bottom halves placed side by side, resized with aspect kept and padded to 48×256 (`seq.prep`) |
| File content | `state_dict` (FP16), `classes` (token list), `H`, `W`, `stack`, `kind` |
| Data | 498R All-in-One v1.1 `ocr/`: train 5,607 crops (3,471 human + 2,136 auto texts), val 766, test 651 |
| Training | 60 epochs, batch 64, AdamW + OneCycle, lr 1e-3, light augmentation, seed 42; best validation loss at epoch 45; 23.9 min on a T4 |
| Alone | **73.43 %** exact on test (character accuracy 94.47 %) · val 62.27 % |
| Caution | over-confident: 65 % of its wrong readings have confidence ≥ 0.9, so never pick between models by confidence ("HC" gives about 76 %) |

---

## 5. Results Summary

### 5.1 Ours vs the paper (different data: a reference, not a duel)

| Metric | Paper (page) | Ours (v1.1, 651 test plates) |
|---|---|---|
| Best single model, plate exact | 92.4 % ViTSTR-Base, mean of 8 datasets (p.9) | 86.79 % char-YOLOv8n |
| Other single models | 85.9–92.2 % | CRNN 73.43, slot CNN 57.60, ViTSTR-tiny 20.89 % |
| Best fusion of the paper's rules | 97.6 % MV-HC top 8 (+5.2, p.9); cross-dataset 90.3 % (+2.7, p.11) | 86.33 % (−0.46; p = 0.58) |
| Highest confidence (HC) | 96.0 % (p.9); 84.4 % cross, below the best single model (p.11) | 76.2–76.7 % with char-YOLO + CRNN |
| Oracle: any model right | not reported | 90.02 % (the ceiling for any fusion) |
| Fusion time | 59.7 ms top 8 on an RTX 8000 (p.12) | 17.0 ms paper rule · **9.82 ms our fallback** (T4) |
| **Our BRTA-format fallback** | — | **88.63 %** (+1.84; 12 fixed / 0 broken; p = 0.0005; bootstrap 95 % CI of the gain +0.92 to +2.92) |

### 5.2 Why the paper's rules do not help here

| Measure | Value |
|---|---|
| char-YOLO's wrong test plates | 86, of which **65 are wrong in all four models** (only 21 could be rescued by any rule) |
| CRNN wrong when char-YOLO is wrong | 80 % (Yule's Q = 0.90; independent errors would give 0) |
| Plates the paper's vote breaks | 8 — all a series letter that CRNN and the slot CNN misread the same way (e.g. খ → গ) |
| HC in char-YOLO + CRNN | overrules 85 right readings, fixes 17 |

### 5.3 By condition and source (plates exact %)

| Group (plates) | char-YOLO | CRNN | paper's vote | our fallback | oracle |
|---|---|---|---|---|---|
| clear (480) | 90.8 | 77.5 | 90.4 | **92.3** | 92.7 |
| night (65) | 80.0 | 61.5 | 78.5 | **80.0** | 80.0 |
| blurry (112) | 74.1 | 64.3 | 74.1 | **78.6** | 84.8 |
| KUET (183) | 83.6 | 75.4 | 83.1 | **86.3** | 90.2 |
| LPDBA (202) | 93.1 | 81.7 | 93.1 | **95.0** | 95.5 |
| ANPR (184) | 86.4 | 68.5 | 85.3 | **88.0** | 88.6 |
| NightLPBD (52) | 76.9 | 59.6 | 76.9 | **76.9** | 76.9 |
| BLPD (30) | 83.3 | 60.0 | 83.3 | **83.3** | 83.3 |

Weather tests, char-YOLO alone: real fog copies 80.77 % (26 plates), synthetic rain 75.88 %.

### 5.4 v1.0 run (20 Oct) vs this v1.1 run (22 Oct)

| Plate exact (%) | v1.0 | v1.1 | 19 BLPD rows (whole photo → crop) | 632 unchanged plates (v1.0 → v1.1) |
|---|---|---|---|---|
| char-YOLOv8n | 85.56 | 86.79 | 5.3 → 78.9 | 88.0 → 87.0 |
| CRNN-CTC | 74.04 | 73.43 | 0.0 → 57.9 | 76.3 → 73.9 |
| slot CNN | 45.93 | 57.60 | 0.0 → 47.4 | 47.3 → 57.9 |
| paper's best rule | 86.18 | 86.33 | 5.3 → 78.9 | 88.6 → 86.6 |
| our fallback | 87.40 | 88.63 | 5.3 → 78.9 | 89.9 → 88.9 |

The dataset fix (19 BLPD test rows that were whole vehicle photos are now plate crops) explains the higher overall
scores. On the unchanged plates the retrained models move by 1–3 points (the slot CNN by +10.6): that is run-to-run
variation, so compare readers on the same plates (McNemar), not by one run's percentages.

---

## 6. Reproducing the Results

1. Notebook `paper24_kaggle.ipynb` (kit folder `docs/Rokib Hasna/24_Leveraging Model Fusion for Improved License Plate Recognition/kaggle/` in the team workspace; generated by `docs/_kit/build_kits.py`).
2. Kaggle: Accelerator **GPU T4 x2**, Internet on. The notebook downloads dataset v1.1 from the team Drive by itself.
3. *Save Version → Save & Run All*. About 80 min. Download the output (`paper24_outputs.zip`, `paper24_results.zip`).
4. Fusion of other runs (teammates' readers): `python code/p24_fusion.py --runs runs/chars_v8n runs/crnn <other run> --name fusion --project runs` — any run folder with `ocr_preds_val.csv` and `ocr_preds_test.csv` on the same split can join.
5. Fill the team table: `python docs/results/add_results.py paper24_results.zip --member "<your name>"`.
6. Report and local checks: `report_src/` next to the notebook (`analysis.py` → `local_check.py` → `charts.py` → `build_report.py`).

---

## 7. Limitations

- **One training run per model.** Two runs of the same model differed by 1–3 points on unchanged plates; the fallback's
  gain (+1.84, 12 / 0) was the same in both runs, but single-model rankings within ~3 points are not reliable.
- **Two weak voters.** ViTSTR-tiny (20.89 %) and the slot CNN (57.60 %) are undertrained (about 5–9 k steps), so the
  paper's voting is tested with fewer strong models than in the paper.
- **Plates outside the BRTA layout** (army, diplomatic, old formats) are not in the test set; for them the fallback keeps
  char-YOLO's reading unless CRNN finds a valid BRTA string.
- **Crops, not video:** 651 still crops (65 night, 112 blurry); multi-frame voting is not tested here.
- **Speed** is the OCR step only (no plate detector); the CPU numbers are from a fast desktop CPU.

---

## 8. Differences from the Original Paper

| | Paper | Ours |
|---|---|---|
| Data | 12 public datasets (Latin / Chinese plates); 8 for training and testing, 4 cross-dataset | 498R All-in-One v1.1 (Bangla plates, 5 sources); grouped, duplicate-free split |
| Models | 12 (CRNN, GRCNN, Rosetta, STAR-Net, R2AM, RARE, TRBA, ViTSTR-Base, Holistic-CNN, Multi-Task-LR, CR-NET, Fast-OCR) | 4, one per family: char-YOLOv8n, CRNN-CTC, slot CNN, ViTSTR-tiny |
| Training | 90 k–300 k iterations in each model's original framework | 60–100 epochs in PyTorch / Ultralytics on a T4 |
| "Best model" and top-N order | from test accuracy (validation said to be almost the same, p.9) | from **validation** only; ties go to the faster row |
| MVCP | votes per character position | votes per **token** (Bangla vowel signs and conjuncts shift character positions) |
| Rules | HC, MV-HC, MV-BM, MVCP-HC, MVCP-BM | the paper's four main ones + MV-BM+format and the format fallback (ours) |
| Statistics | none | exact McNemar test per row, Wilson and bootstrap intervals, oracle, two runs |

---

## 9. File Integrity

SHA-256 checksums of the files in this folder (and the Drive file):

| File | SHA-256 |
|---|---|
| `best_chars_v8n.pt` | `bd58dc78c17aa98efc7fd87ec72d5a7a7f45d353fdf30bbc8afdfe78397b48cf` |
| `best_crnn_fp16.pt` | `cc70d5ad43b7e031e709d7c7c951c6450779a956a135199af904bae97d338022` |
| `paper24_results.zip` | `8b96a8f324807d8f4c7bfb41ad840c3d8ef295ebea815f8f0ee20f704513d6a0` |
| `results_vs_paper.md` | `47ce5f9900a1c7f46a9c91d96cf350134a47f78b34b2269877ee70cb374a7c84` |
| `Paper24_Study_Report.pdf` | `6697588d82bdf171fa77ee8b5980349d518e94838d072de4509db3efd44ac49e` |
| `paper24_outputs.zip` (Drive) | `f25a0c6d6c38724d6cb0a771567d0c56cd48e018eeb62539cabb19c35342f6f2` |

Check on Windows: `certutil -hashfile best_chars_v8n.pt SHA256` · Linux/macOS: `sha256sum best_chars_v8n.pt`.

---

## 10. Licence and Data Notes

- The paper publishes no fusion code (only its split lists, raysonlaroca.github.io/supp/lpr-model-fusion/); the fusion
  rules and models here are our re-implementation in the 498R kit.
- char-YOLOv8n is trained with Ultralytics YOLOv8 (AGPL-3.0); the YOLOv8n COCO start weights come from Ultralytics.
- Training data: 498R BanglaLPR All-in-One v1.1, built from public Bangla plate datasets; each source keeps its own
  licence (see the dataset's README and `docs/DATASET_SHARING_GUIDE.md`).

---

## 11. Citation

```bibtex
@inproceedings{laroca2023leveraging,
  title     = {Leveraging Model Fusion for Improved License Plate Recognition},
  author    = {Laroca, Rayson and Zanlorensi, Luiz A. and Estevam, Valter and Minetto, Rodrigo and Menotti, David},
  booktitle = {Iberoamerican Congress on Pattern Recognition (CIARP)},
  pages     = {60--75},
  year      = {2023},
  doi       = {10.1007/978-3-031-49249-5_5}
}
```
