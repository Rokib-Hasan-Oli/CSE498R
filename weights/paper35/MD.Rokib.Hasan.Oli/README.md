# README.md — Paper #35: HVI, a New Color Space for Low-light Image Enhancement (run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — experiments for *"HVI: A New Color Space for Low-light Image Enhancement"* (Yan, Feng, Zhang, Pang, Shi, Wu, Dong, Sun & Zhang, CVPR 2025, arXiv 2502.20272 v2; HVI-CIDNet; Paper #35)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 26 October 2026 (Kaggle run of 26 October 2026, dataset **498R BanglaLPR All-in-One v1.1**; original-paper check on LOLv1 / LOLv2-Real the same day; local re-check on an RTX 5060 Ti)
> **Full study report:** [`Paper35_Study_Report.pdf`](Paper35_Study_Report.pdf) (paper study, run check, LOL check, findings, faculty guide; 19 pages)

This package tests one question: **does a state-of-the-art low-light enhancer help our night plate detector and plate
reader?** Five pre-processing settings — none, gamma 0.5, CLAHE, and HVI-CIDNet with the authors' *generalization* and
*LOLv2-Real* weights, all zero-shot — are placed in front of a YOLOv8s plate detector and a CRNN plate reader trained on
dataset v1.1, and scored on night photos, night crops, clear photos, the whole test split and (locally) end to end on
60 NightLPBD night photos. The paper itself only reports image-quality scores (PSNR / SSIM / LPIPS); every plate number
below is ours. Sources: `paper35_results.zip`, `paper35_check_results.zip` and the local checks in
[`report/MD.Rokib.Hasan.Oli/paper35/source/`](../../../report/MD.Rokib.Hasan.Oli/paper35/source/).

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
| **Question** | Does low-light enhancement before detection / reading improve Bangla plate recognition at night? |
| **Answer** | **No** (zero-shot HVI-CIDNet): no extra night plate found, reading gets worse, end-to-end gain +4 / 60 (n.s.); plain CLAHE does as well for 10 ms |
| **Night detection (71 photos)** | plates found **64 / 71 with and without** enhancement (p = 1.0); mAP50 94.41 → 95.66 % (HVI) / 96.10 % (CLAHE), gains inside the 95 % interval (HVI −0.1 to +2.8 points) |
| **Night reading (65 crops)** | 61.54 % → **55.38 %** with HVI (7 broken, 3 fixed, n.s.); gamma 44.62 % (p = 0.003); CLAHE 61.54 % |
| **All 651 crops enhanced** | HVI 73.43 → 69.74 % (p = 0.003); on day crops −20 of 586 plates (p = 0.007) |
| **End to end, 60 NightLPBD photos (local)** | none 31 right; best CLAHE on the photo 35, HVI for the detector only 35 (n.s.); the same 6 photos are missed by every setting |
| **Why** | our night photos are not dark: median mean grey 85, only 17 / 71 below the team's gate of 60; the missed night plates are small or close-ups |
| **Cost (T4, 640 × 640)** | HVI-CIDNet **154.9 ms** (FP32) vs detector 6.5 ms (TensorRT FP16), CLAHE 9.8 ms, gamma 0.6 ms; FP16 is 2.6x faster with no NaN (local) |
| **Original check (LOL)** | released weights = authors' README to ~0.1 dB: LOLv1 28.23 dB with GT mean (23.61 without); LOLv2-Real 23.89 dB (paper 24.111 from lost weights); 1.98 M parameters (paper 1.88 M) |
| **Training** | YOLOv8s 60 epochs (best 55), 105 min; CRNN 60 epochs, 26 min; Kaggle Tesla T4 x2; 4 / 4 jobs `code=0`, 2.0 h |
| **Reproducibility** | the CRNN file is **byte-identical** (SHA-256) to the paper #24 run's; local re-scoring = Kaggle (plates found and read identical, mAP50 within 0.04) |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 · HVI-CIDNet commit `eb43d7d` · dataset v1.1 |

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `Paper35_Study_Report.pdf` | 3.4 MiB | Full study report, 19 pages |
| `best_det_v8s.pt` | 21.5 MiB | YOLOv8s plate detector of this run (human + auto boxes, 60 epochs, best epoch 55), Ultralytics checkpoint, unchanged |
| `paper35_results.zip` | 1.1 MiB | Results archive, **no model weights** (see 2.2) |
| `paper35_check_results.zip` | 47 KiB | LOL check: `results.json` of both jobs, logs, code |
| `results_vs_paper.md` | — | The kit's ours-vs-paper table and the per-condition tables |
| `results_original_check.md` | — | The kit's LOL check table |

**The CRNN is not in this folder** because it is the same file as paper #24's: this run's `crnn/model_full.pt` has the
same SHA-256 (`0110631e…f033`) as the #24 run of 22 October (same kit, same seed). Use
[`weights/paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt`](../../paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt)
(its FP16 copy; we checked: a 0.0 difference to this run's FP32 file after the FP16 cast). The HVI-CIDNet weights are
the authors' own (Hugging Face, see 4.2); nothing was trained on them.

### 2.2 Inside `paper35_results.zip`

| Path | What |
|---|---|
| `runs/det_v8s/` | `results.json` (val / test P, R, mAP, plates found by source and size, per-condition scores, speed FP32 / FP16 / TensorRT / CPU ONNX, settings), `results.csv`, `args.yaml`, curves |
| `runs/crnn/` | `results.json` (val / test plate exact, CER, by source / condition, settings, loss log), `ocr_preds_val.csv`, `ocr_preds_test.csv` |
| `runs/lowlight/results.json` | every enhancer × set: photos, photos enhanced, mean grey before / after, P, R, mAP50, mAP50-95, plates found; night / gated reading scores; ms per photo and per crop; parameters |
| `logs/` | the four job logs |
| `code/` | the exact kit code used (`p35_lowlight.py`, `det.py`, `seq.py`, `common.py`, `metrics.py`, …) |
| `environment.json`, `results_vs_paper.md` | versions (incl. `dataset_version: 1.1`) and the comparison table |

The full outputs (all weights, TensorRT engine, ONNX files) are `paper35_outputs.zip` on the team Drive
(`498R / weights / paper35 /`).

---

## 3. Quick Start

Tested on 26 Oct 2026 (Python 3.14, torch 2.13, Ultralytics 8.4.170) from a copy of the repository layout. Run from
the repository root; it needs only this folder, `weights/paper24/…/best_crnn_fp16.pt` and the kit in
`src/MD.Rokib.Hasan.Oli/paper35/kaggle/`.

```python
import sys, cv2, torch
from ultralytics import YOLO
sys.path.insert(0, 'src/MD.Rokib.Hasan.Oli/paper35/kaggle')   # the kit: seq.py (CRNN) + vocab.py
import seq

det = YOLO('weights/paper35/MD.Rokib.Hasan.Oli/best_det_v8s.pt')
ck = torch.load('weights/paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt', map_location='cpu', weights_only=True)
ocr = seq.CRNN(len(ck['classes']), ck['H'])
ocr.load_state_dict({k: v.float() for k, v in ck['state_dict'].items()})
ocr.eval()

img = cv2.imread('night_photo.jpg')
if cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() < 60:       # optional: CLAHE behind the dark gate
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
r = det.predict(img, imgsz=640, conf=0.25, verbose=False)[0]
for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
    pw, ph = 0.05 * (x2 - x1), 0.05 * (y2 - y1)                # 5 % padding, as in the tests
    crop = img[max(0, int(y1 - ph)):int(y2 + ph), max(0, int(x1 - pw)):int(x2 + pw)]
    x = seq.prep(crop, ck['H'], ck['W'], ck['stack'], False)[None]
    with torch.no_grad():
        toks, _ = seq.ctc_decode(ocr(x).float())
    print(f'plate {c:.2f}:', seq.vocab.tokens_to_text([ck['classes'][k - 1] for k in toks[0]]))
```

On the NightLPBD test photo `nightlpbd_43895605d697.jpg` this prints `plate 0.84: ঢাকা মেট্রো-গ ২৮-৮১৬৯` (correct).
To try HVI-CIDNet itself: `src/…/kaggle/p35_lowlight.py setup --dir hvi_cidnet`, then
`p35_lowlight.Enhancer('hvi_gen', torch.device('cuda'))(bgr_image)` (downloads the authors' safetensors weights).

---

## 4. Model Card

### 4.1 `best_det_v8s.pt` — YOLOv8s plate detector

| | |
|---|---|
| Architecture | YOLOv8s (11.1 M parameters, 28.6 GFLOPs), COCO-pretrained start, one class `plate` |
| Training data | All-in-One v1.1 detection train (4,067 human + 4,507 auto boxes), val 896 human; grouped 70 / 15 / 15 split, seed 42 |
| Settings | 640 px, 60 epochs, batch 16, SGD, seed 42, deterministic, Ultralytics 8.4.170 defaults otherwise |
| Test (1,153 photos, 1,011 plates) | P 96.9 · R 96.9 · **mAP50 98.51** · mAP50-95 45.82 % · plates found 98.12 % (NightLPBD 88.33 %) |
| By condition (mAP50) | clear 99.0 · night 94.3 · blurry 95.2 · small plates 84.5 · synthetic rain 82.5 · synthetic fog 97.9 · false alarms on plate-free photos 5.62 % |
| Speed (T4, batch 1, 640 px) | TensorRT FP16 6.5 ms · PyTorch FP32 9.0 ms · CPU ONNX 245 ms (Kaggle CPU) |
| Compared with | paper #6's YOLOv8s (human boxes only, 120 epochs): mAP50 97.6 / mAP50-95 45.9 %, 97.13 % found |

### 4.2 Enhancers tested (not trained here)

| Name | What | Source |
|---|---|---|
| `gamma` | v → 255 (v / 255)^0.5 | OpenCV LUT |
| `clahe` | CLAHE on the L channel of Lab, clip 2.0, 8 × 8 tiles | OpenCV |
| `hvi_gen` | HVI-CIDNet *generalization* weights (LOLv2 + random gamma), `gated2`, alpha 1.0 | `Fediory/HVI-CIDNet-Generalization` @ `51481ef` |
| `hvi_lolv2` | HVI-CIDNet LOLv2-Real best-PSNR weights, `gated2`, alpha 0.8 (as the authors' eval.py) | `Fediory/HVI-CIDNet-LOLv2-real-bestPSNR` @ `c317f57` |

Code: github.com/Fediory/HVI-CIDNet at commit `eb43d7d91e9a336c66856824ff9e4603ae41f408` (MIT).

---

## 5. Results Summary

### 5.1 Detector (Kaggle T4; plates found at conf 0.25, IoU 0.5)

| Enhancer | Night mAP50 (71) | Night found | Clear-200 mAP50 | Whole test, gate < 60 | All 1,153 enhanced (local) | Plate-free photos with a box (local, of 160) |
|---|---|---|---|---|---|---|
| none | 94.41 | 64 / 71 | 99.16 | 98.50 | 98.50 | 9 |
| gamma 0.5 | 93.84 | 64 | 98.96 | 98.50 | 98.93 | 11 |
| CLAHE | 96.10 | 64 | 99.08 | 98.50 | 98.74 | **17 (p = 0.008)** |
| HVI generalization | 95.66 | 64 | 98.98 | 98.49 | 98.72 | 13 |
| HVI LOLv2-Real | 94.65 | 63 | 99.13 | 98.47 | 98.96 | 12 |

Paired photo bootstrap (val's own boxes, 5,000 draws) of the night mAP50 change: HVI generalization +1.28 (−0.1 to
+2.8), CLAHE +1.68 (−0.3 to +4.6), LOLv2-Real +0.27 (−2.3 to +2.5), gamma −0.56 (−3.9 to +3.8). All include zero.

### 5.2 Reader (CRNN, plate exact %)

| Enhancer | Night crops (65) | Broken / fixed | All 651 enhanced | Gate < 60 (13 crops) |
|---|---|---|---|---|
| none | 61.54 | — | 73.43 | 73.43 |
| gamma 0.5 | 44.62 | 12 / 1 (p = 0.003) | 69.74 | 72.96 |
| CLAHE | 61.54 | 2 / 2 | 70.97 | 73.58 |
| HVI generalization | 55.38 | 7 / 3 (n.s.) | 69.74 | 73.12 |
| HVI LOLv2-Real | 56.92 | 5 / 2 (n.s.) | 72.20 | 73.12 |

### 5.3 End to end (local; plates read right of 60 NightLPBD night photos / 144 BLPD day photos)

| Enhancer | Night: photo | Night: detect only | Night: crop only | Day: photo |
|---|---|---|---|---|
| none | 31 | 31 | 31 | 116 |
| gamma 0.5 | 31 | 33 | 27 | 115 |
| CLAHE | **35** | 31 | 33 | 113 |
| HVI generalization | 33 | **35** | 31 | 116 |
| HVI LOLv2-Real | 26 | 34 | 26 | 118 |

*photo* = detect and read on the enhanced photo; *detect only* = detect on the enhanced photo, read the original pixels;
*crop only* = enhance only the plate crop. No difference is significant (McNemar p ≥ 0.06).

---

## 6. Reproducing the Results

1. **Kaggle run:** open [`src/MD.Rokib.Hasan.Oli/paper35/kaggle/paper35_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper35/kaggle/paper35_kaggle.ipynb)
   on Kaggle (GPU T4 x2, Internet on), Run All (about 2 h; the dataset downloads itself). Steps:
   [`Kaggle_Training_Guide.md`](../../../src/MD.Rokib.Hasan.Oli/paper35/Kaggle_Training_Guide.md). The executed run is
   [`notebooks/MD.Rokib.Hasan.Oli/paper35_run.ipynb`](../../../notebooks/MD.Rokib.Hasan.Oli/paper35_run.ipynb).
2. **LOL check:** [`paper35_original_check.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper35/kaggle/paper35_original_check.ipynb) (about 5 min).
3. **Local re-check and report:** [`report/MD.Rokib.Hasan.Oli/paper35/source/`](../../../report/MD.Rokib.Hasan.Oli/paper35/source/):
   `local_check.py` → `local_valstats.py` → `analysis.py` → `charts.py` → `build_report.py` (the scripts expect the kit
   in `../kaggle` and the dataset in `D:\498R\dataset`; adjust `KIT` / `DATASET_ROOT` for another layout).
4. **Team table:** `docs/results/add_results.py paper35_outputs.zip --member "Rokib Hasan Oli"` (rows `det_v8s`, `crnn`
   under paper #35 in [`results/`](../../../results/)).

---

## 7. Limitations

- **Small night sets:** 71 night photos, 65 night crops, 60 end-to-end night photos. They rule out a large effect, not a
  small one; a +4 / 60 end-to-end gain would need two to three times more night photos to test.
- **Zero-shot only:** HVI-CIDNet was not fine-tuned on plates (no paired dark / bright plate photos exist). A
  task-aware enhancer (trained with the detector or reader loss) was not tested.
- **One seed** for the detector and the reader (the CRNN is deterministic: bit-identical across two sessions).
- **"Night" is a source tag:** the night set is mostly NightLPBD, whose photos are lit (median mean grey 85); results on
  truly dark footage (e.g. CCTV without lights) may differ.
- The T4 HVI timing of the 64 × 192 crop (24.9 / 37.6 ms) ran while another job used the machine; real crops are
  larger (median 253 × 416).

---

## 8. Differences from the Original Paper

| Paper | Here |
|---|---|
| Image quality (PSNR / SSIM / LPIPS) against a well-lit twin | plates found and plates read (no twin exists for street photos) |
| LOL, SICE, Sony-Total-Dark, 5 unpaired sets | 498R All-in-One v1.1 night / clear / whole test, NightLPBD e2e photos |
| One trained model per data set | the authors' released weights, zero-shot |
| LOLv1 scored with GT mean | no GT mean anywhere on our data (impossible in practice); the LOL check reports both |
| 0.053 s per 600 × 400 image (GPU not named) | T4: 81.9-87.3 ms per 600 × 400, 154.9 ms per 640 × 640 |

Paper facts we found while checking (details in the PDF, section 5): the LOLv1 row mixes two weight files (LPIPS 0.079
is the perceptual-loss model's); the LOLv2-Real 24.111 dB weights are lost (public best 23.90); eval.py uses
per-file brightness constants (0.80-0.84) and a saturation boost (×1.3) not given in the paper; the code's `density_k`
is 1 / paper k; the released model has 1.98 M parameters, not 1.88 M.

---

## 9. File Integrity

| File | SHA-256 |
|---|---|
| `best_det_v8s.pt` | `444db39bf26f63eefe2606b178e766787e8b66671aa09cf9812f64ad28cf3a81` |
| `paper35_results.zip` | `747de31f32286794e4d9e45f3d1419390fb729e1bb9948db2847f6cd56e64da8` |
| `paper35_check_results.zip` | `bd0deede56c421bc44df2151b96868c1df87b8032e932e13ef91d00e26d52c14` |
| CRNN (in `weights/paper24/…/best_crnn_fp16.pt`) | `cc70d5ad43b7e031e709d7c7c951c6450779a956a135199af904bae97d338022` |

Check on Windows: `certutil -hashfile best_det_v8s.pt SHA256` · Linux/macOS: `sha256sum best_det_v8s.pt`.

---

## 10. Licence and Data Notes

- HVI-CIDNet code and weights: MIT (Yan et al.); not redistributed here — the kit downloads them at a pinned commit and
  pinned Hugging Face revisions.
- Detector and CRNN: trained by us on the 498R All-in-One dataset v1.1 (sources and their licences in the dataset
  README; NightLPBD is CC BY 4.0). Ultralytics is AGPL-3.0.
- The LOL images used by the check are not in this repository.

---

## 11. Citation

```bibtex
@inproceedings{yan2025hvi,
  title     = {{HVI}: A New Color Space for Low-light Image Enhancement},
  author    = {Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Pang, Guansong and Shi, Kangbiao and Wu, Peng and Dong, Wei and Sun, Jinqiu and Zhang, Yanning},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025}
}
```
