# Paper20_README.md — Paper #20: Super-Resolution and Fusion for Bangla Plate Recognition

> **Document name:** `Paper20_README.md`
> **Project:** 498R BanglaLPR — re-implementation of *"Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark"* (Paper #20, arXiv:2505.06393)
> **Author / maintainer:** Md. Yousuf
> **Date:** 19 October 2026
> **Full technical report:** [`Paper20_Report.md`](../../../report/Md.Yousuf/Paper20_Report.md)
> **Location in the team repository:** `weights/paper20/Md.Yousuf` in `Rokib-Hasan-Oli/CSE498R`

This package contains a **CRNN OCR model for Bangla (BRTA) licence plates**, the results of putting a ×2 **super-resolution (SR) network** and **multi-copy fusion** in front of that OCR on synthetic low-resolution plates, the logs and the code. All numbers come from the saved run files in `paper20_results.zip`.

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
| **Task** | Read Bangla BRTA plate text from cropped plate images; test whether ×2 super-resolution and fusion of several copies help on low-resolution plates |
| **OCR model** | CRNN with CTC decoding, 7.69 M parameters, trained on HR plate crops |
| **OCR accuracy** | **74.04 %** plate exact, CER 6.74 %, token accuracy 93.47 % (651 test plates, original crops) |
| **SR network** | ×2 residual channel-attention network, 0.79 M parameters (`sr.pt`) |
| **SR effect (single image)** | LR 16.90 % → bicubic 23.81 % → our SR **24.12 %** plate exact (HR reference 65.28 %) |
| **Fusion of five copies** | Best **32.10 %** plate exact (our SR + majority vote, ties by confidence) |
| **Real-ESRGAN x4plus (zero-shot)** | 12.29 % — lower than the LR input |
| **Speed** | SR 5.50 ms/plate (T4); OCR 4.96 ms/plate |
| **Training** | OCR 19.3 min; SR 21.5 min (Tesla T4) |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · Ultralytics 8.4.170 (toolkit dependency) |

The low-resolution plates are **synthetic** (made from HR crops) and the five fused "frames" are five degradations of one photograph; see Section 8.

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `Paper20_README.md` | — | This guide |
| `Paper20_Report.md` | — | Full technical report |
| `sr.pt` | 3.0 MiB | SR network weights (`SRNet`, ×2), best validation PSNR (23.73 dB) |
| `paper20_crnn_best.pt` | 29.4 MiB | OCR weights (state dict only), best validation loss (epoch 52) |
| `paper20_crnn_model_full.pt` | 29.4 MiB | Same OCR weights plus class list, input size and stacking flag |
| `paper20_results.zip` | 75.4 KiB | Results archive (no weights) |
| `results_vs_paper.md` | — | Short ours-vs-paper table |
| `20_2025_License Plate Super-Resolution Dataset and Benchmark.pdf` | 4.7 MiB | The paper |

### 2.2 Contents of `paper20_results.zip`

```text
├── environment.json            # Python, PyTorch, CUDA, GPU and library versions
├── results_vs_paper.md         # ours vs the paper's reported numbers
├── runs/
│   ├── sr_x2/results.json      # SR training summary (PSNR, parameters, time)
│   ├── crnn_hr/
│   │   ├── results.json        # OCR settings, per-epoch loss, val/test, by source, by condition
│   │   ├── ocr_preds_test.csv  # per-plate predictions on the 651 test plates
│   │   └── ocr_preds_val.csv   # per-plate predictions on the 766 validation plates
│   └── sr_eval/results.json    # HR / LR / bicubic / SR / Real-ESRGAN and fusion results
├── logs/
│   ├── sr_x2.log               # SR training log (60 epochs of val PSNR)
│   ├── crnn_hr.log             # OCR training log
│   └── sr_eval.log             # evaluation log
└── code/
    ├── p20_sr.py               # Paper #20 SR training and evaluation
    ├── seq.py  metrics.py  vocab.py  common.py  report.py     # shared toolkit used by Paper #20
    └── det.py  rec.py  ved.py                                  # toolkit modules not used for Paper #20
```

---

## 3. Quick Start

### 3.1 Install

```bash
pip install torch opencv-python-headless numpy
```

Copy `code/seq.py`, `code/vocab.py`, `code/metrics.py` and `code/common.py` into one folder (the OCR model class and decoding live in `seq.py`). `common.py` looks for the dataset only when dataset functions are called.

### 3.2 Read a plate crop with the OCR

This mirrors `load_ocr()` and `read()` in `p20_sr.py`; it has not been run separately from the saved scripts.

```python
import sys, cv2, torch
sys.path.insert(0, "code")                       # folder holding seq.py, vocab.py, metrics.py, common.py
import seq

ckpt = torch.load("paper20_crnn_model_full.pt", map_location="cpu", weights_only=False)
model = seq.CRNN(len(ckpt["classes"]), ckpt["H"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

img = cv2.imread("plate_crop.jpg")               # a cropped plate image (BGR)
x = seq.prep(img, ckpt["H"], ckpt["W"], ckpt["stack"], False)[None]
with torch.no_grad():
    out = model(x).float()
toks, conf = seq.ctc_decode(out)
text = seq.vocab.tokens_to_text([ckpt["classes"][k - 1] for k in toks[0]])
print(text, conf[0])
```

The checkpoint stores `state_dict`, `kind` (`crnn`), `classes`, `H` (48), `W` (256) and `stack` (true). `weights_only=False` is required because the file contains a dictionary with non-tensor entries; load it only from a source you trust (Section 10).

### 3.3 Super-resolve a low-resolution plate

This mirrors `evaluate()` in `p20_sr.py`; it has not been run separately from the saved scripts. The input is a 32 × 64 (height × width) BGR plate crop; the output is 64 × 128.

```python
import sys, cv2, torch
sys.path.insert(0, "code")
import p20_sr

ckpt = torch.load("sr.pt", map_location="cpu", weights_only=False)   # {'state_dict', 'scale'}
sr = p20_sr.SRNet(ckpt["scale"])
sr.load_state_dict(ckpt["state_dict"])
sr.eval()

lr = cv2.imread("lr_plate.png")                  # 32 x 64 BGR
with torch.no_grad():
    y = sr(p20_sr.to_t(lr)[None]).float()
cv2.imwrite("sr_plate.png", p20_sr.to_img(y[0]))
```

`p20_sr.py` imports `seq`, `metrics` and `common`, so keep them in the same folder.

### 3.4 Retrain the SR network and re-run the evaluation

The dataset (498R BanglaLPR All-in-One v1.0, OCR part) must be available to `common.find_dataset()`.

```bash
python code/p20_sr.py train --name sr_x2 --project runs --epochs 60
python code/seq.py --model crnn --name crnn_hr --project runs --epochs 60
python code/p20_sr.py eval --sr runs/sr_x2/sr.pt --ocr runs/crnn_hr/model_full.pt \
                           --name sr_eval --project runs
```

`--no-esrgan` skips the Real-ESRGAN comparison (otherwise `RealESRGAN_x4plus.pth` is downloaded from the Real-ESRGAN release); `--max-test N` evaluates only N test plates.

---

## 4. Model Cards

### 4.1 OCR — `paper20_crnn_best.pt` / `paper20_crnn_model_full.pt`

| Item | Value |
|---|---|
| Model | CRNN with CTC decoding (`seq.py`, `--model crnn`) |
| Parameters | 7.69 M |
| Input | Grey, 48 × 256 (height × width); plates with width/height < 2.6 are cut into top and bottom halves placed side by side |
| Output | Bangla tokens (district, *Metro*, series letter, digits), joined to plate text |
| Training data | `ocr/train` of 498R BanglaLPR All-in-One: 5,607 crops (3,471 human + 2,136 auto labels), HR only |
| Optimiser | AdamW + OneCycle, lr 1e-3, weight decay 1e-4, batch 64, 60 epochs, seed 42 |
| Augmentation | Rotation ±4°, scale 0.92–1.05, brightness/contrast, blur, noise, random down-up resize; no flip |
| Checkpoint | Lowest validation loss: epoch 52 (0.3542) |
| Training time | 19.3 min on one Tesla T4 |
| Test (651 plates, original crops) | Plate exact **74.04 %** · CER 6.74 % · token accuracy 93.47 % · WER 9.69 % · 4.96 ms/plate |
| Validation (766 plates) | Plate exact 64.49 % · CER 10.58 % |

By source (test): `lpdba` 85.15 % · `kuet` 75.41 % · `anpr` 69.57 % · `nightlpbd` 67.31 % · `blpd` 30.00 % (30 plates).
By condition (test): clear 78.12 % · night 67.69 % · blurry 60.71 % · real fog copies 61.54 % (26 plates) · synthetic rain 66.67 %.

### 4.2 SR network — `sr.pt`, class `SRNet` in `p20_sr.py`

| Item | Value |
|---|---|
| Architecture | 3×3 input conv → 10 residual channel-attention blocks (64 channels) → conv → ×2 PixelShuffle, plus a bicubic skip connection |
| Parameters | 0.79 M |
| Input / output | LR 32 × 64 → SR 64 × 128 (height × width) |
| Training data | 5,607 HR crops resized to 64 × 128; random synthetic degradation each step |
| Degradation | Gaussian blur (σ 0.4–1.4), ÷2 resize (area / cubic / linear), Gaussian noise (σ 0–8), JPEG (quality 30–90) |
| Loss / optimiser | L1; Adam lr 2e-4, cosine decay; batch 32; 60 epochs; seed 42 |
| Validation PSNR | 23.73 dB (SR) vs 22.15 dB (bicubic), 300 validation crops |
| Training time | 21.5 min on one Tesla T4 |
| Speed | 5.50 ms/plate (T4, batch 1, AMP) |

### 4.3 Evaluated scope

Evaluated: Bangla BRTA plate crops from the 498R OCR split; synthetic ×2 low-resolution copies; five-copy fusion (N = 5); clear, night, blurry, real-fog and synthetic-rain subsets.

Not evaluated: real low-resolution Bangla plate tracks, real multi-frame video, N = 1 or N = 3 fusion, SR3 / LPSRGAN / LCDNet, ×4 scale, plate detection (see the Paper #3 package for detectors).

---

## 5. Results Summary

All numbers: `ocr/test`, 651 plates, from `runs/sr_eval/results.json` unless stated. One plate equals 0.154 points.

### 5.1 One image per plate

| Input to the OCR | Plate exact | CER | Token acc | All but one token | SR ms/plate |
|---|---|---|---|---|---|
| HR (64 × 128) | 65.28 % | 9.96 % | 91.75 % | 84.64 % | — |
| LR (32 × 64, synthetic) | 16.90 % | 31.37 % | 64.26 % | 31.49 % | — |
| Bicubic ×2 | 23.81 % | 28.47 % | 69.19 % | 40.86 % | — |
| **Our SR ×2** | **24.12 %** | **26.38 %** | **72.39 %** | **45.31 %** | 5.50 |
| Real-ESRGAN x4plus (zero-shot) | 12.29 % | 33.30 % | 60.89 % | 27.19 % | 51.27 |

### 5.2 Fusion of five differently degraded copies (plate exact %)

| Input | Single copy | HC | MV-HC | MVCP-HC |
|---|---|---|---|---|
| LR | 17.36 | **26.11** | 25.35 | 21.81 |
| Bicubic ×2 | 22.43 | 31.64 | **31.80** | 28.57 |
| Our SR ×2 | 24.88 | 29.80 | **32.10** | 28.26 |

HC = most confident reading; MV-HC = most frequent string, ties by confidence; MVCP-HC = vote per token position, ties by confidence.

### 5.3 Compared with the paper

| Item | Paper | This study |
|---|---|---|
| OCR on HR plates | 85.2 % | 65.28 % (74.04 % on original crops) |
| OCR on LR plates | 2.2 % | 16.90 % |
| One SR image | PLNET 29.9 %, LCDNet 29.9 % | 24.12 % |
| One Real-ESRGAN image | 20.2 % | 12.29 % (zero-shot) |
| 5 images, MVCP | PLNET 40.9 %, LCDNet 42.3 % | 28.26 % (MVCP-HC); best rule 32.10 % (MV-HC) |
| SR time | PLNET 24.88 ms, Real-ESRGAN 34.86 ms (RTX 8000) | Own SR 5.50 ms, Real-ESRGAN 51.27 ms (T4) |

The paper used real LR/HR pairs of Brazilian/Mercosur plates, GP_LPR as the OCR, five trained SR models and five real consecutive frames; this study uses synthetic LR Bangla plates, a CRNN, one own SR network and five degraded copies of one photograph. The comparison shows scale and direction only.

---

## 6. How to Read These Results

**Evidence from the saved numbers and code, and hypotheses that were not tested.** Details and the check for each hypothesis are in Section 11 of the report.

- **Evidence.** The SR network beats bicubic clearly in CER (−2.09 points), token accuracy (+3.20) and all-but-one-token (+4.45), but only by 0.31 points (two plates) in plate exact. *Hypothesis:* an L1-only network improves overall pixel fidelity but not the specific character details that decide an exact match; an OCR-guided loss would test this.
- **Evidence.** The HR reference is 74.04 % on original crops but 65.28 % on the 64 × 128 resized variant. In the code, every resized image has aspect ratio 2.0 and therefore takes the OCR's two-line path (width/height < 2.6). *Hypothesis:* this forced 2:1 shape lowers the HR reference; evaluating at the original aspect ratio would test this.
- **Evidence.** MVCP-HC has the lowest plate exact of the three fusion rules for all inputs, and its WER (136–138 %) is above 100 % because the fused string joins each token with a space while WER counts space-separated words. *Hypothesis:* voting by token index suffers when single readings drop or insert a token; aligning sequences before voting would test this.
- **Evidence.** Zero-shot Real-ESRGAN x4plus lowered recognition below the LR input (12.29 % vs 16.90 %). *Hypothesis:* the pre-trained natural-image model does not suit tiny plate crops; fine-tuning it on plates would test this.

---

## 7. Reproducing the Results

### 7.1 Environment

Exact versions are in `environment.json` (Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, Ultralytics 8.4.170, numpy 2.1.3, OpenCV 4.14.0). The run used two Tesla T4 GPUs: SR training on GPU 0 and OCR training on GPU 1 at the same time; evaluation on one T4.

### 7.2 Commands

```bash
python p20_sr.py train --name sr_x2 --project runs --epochs 60                 # GPU 0
python seq.py --model crnn --name crnn_hr --project runs --epochs 60            # GPU 1
python p20_sr.py eval --sr runs/sr_x2/sr.pt --ocr runs/crnn_hr/model_full.pt \
                      --name sr_eval --project runs
```

Wall times in the run: OCR 19.8 min, SR 22.2 min, evaluation 1.5 min.

### 7.3 Determinism

Seed 42 for OCR and SR training. The evaluation draws its degradations from fixed seeds (`10000 + i` for the single LR copy, `1000 × j + i` for fusion copy *j*), so the same degraded copies are generated each time.

---

## 8. Limitations

1. **Synthetic LR.** There are no real LR/HR pairs for Bangla plates; all LR images are generated from HR crops, and the SR network is trained and tested on the same degradation family.
2. **Copies, not frames.** The five fused inputs are five random degradations of one photograph, not five video frames; errors of the copies may be correlated.
3. **Different components from the paper.** One own PLNET-style SR network instead of five trained SR models; CRNN instead of GP_LPR; Real-ESRGAN used zero-shot.
4. **HR reference altered by resizing** (Section 6).
5. **Single run.** One seed for training and one set of degradation seeds for evaluation; no confidence intervals.
6. **OCR trained on HR crops only;** not trained or fine-tuned on LR or SR images.
7. **Small subsets.** `blpd` 30 plates, real fog 26, `nightlpbd` 52, night 65.
8. **PSNR on 300 validation crops only;** no SSIM or perceptual metrics.
9. **Fusion only with N = 5.**
10. **MVCP-HC WER** is not comparable with the other WER values.
11. **Speed** measured on a Tesla T4 only.

---

## 9. Differences from the Original Paper

| Component | Paper | This study |
|---|---|---|
| Data | UFPR-SR-Plates, real LR/HR pairs (Brazilian, Mercosur) | 498R BanglaLPR All-in-One OCR crops, synthetic LR |
| LR / HR size | 16 × 48 → 32 × 96 | 32 × 64 → 64 × 128 |
| OCR | GP_LPR | CRNN-CTC (7.69 M parameters) |
| SR models | SR3, Real-ESRGAN, LPSRGAN, PLNET, LCDNet (trained) | One PLNET-style L1 network; Real-ESRGAN x4plus zero-shot |
| Fusion | N = 1, 3, 5 real consecutive frames | N = 5 degraded copies of one photo |
| Fusion rules | HC, MV, MVCP | HC, MV-HC, MVCP-HC |
| Metric | All characters correct (+ ≥ 6, ≥ 5 of 7) | Plate exact, CER, token accuracy, WER, all-but-one-token |
| Test size | 4,005 tracks | 651 plates |
| Hardware | Ryzen 5950X + Quadro RTX 8000 | Kaggle Tesla T4 |

---

## 10. File Integrity

SHA-256 of the files as supplied:

```text
14d5c91cda2de396034e243b376b14f96549ccd7255b791ef99711ff698dc679  sr.pt
3159f2d57a74186de3726b19164ad22ac606d170f59dfe6e5776d64d86a5b6aa  paper20_crnn_best.pt
af7199b428b5d20a656d67d4eda4342d1290c3010253713d239cff26843ff869  paper20_crnn_model_full.pt
b26e207e3f490c01b7693bbdc1095427a0f4c2767e466afbbd4aa8e3dcb58a12  paper20_results.zip
4145926383f46fa7513d7d77a5cd8e185e603efc9f13f7aa3804d2ffba01a7dd  results_vs_paper.md
8c258b203cf3e02ebffd913709b30e8226013d785baf9bec54abd9314d3d6a14  20_2025_License Plate Super-Resolution Dataset and Benchmark.pdf
```

```bash
sha256sum sr.pt paper20_crnn_best.pt paper20_crnn_model_full.pt paper20_results.zip
```

`.pt` files are PyTorch pickle archives; loading one can execute code. Load them only from sources you trust.

---

## 11. Licence and Data Notes

- **Dataset.** The 498R BanglaLPR All-in-One dataset combines several sources (OCR split: `anpr`, `blpd`, `kuet`, `lpdba`, `nightlpbd`). The licence terms of each source were not checked in this study; verify them before redistributing images. The dataset is not included in this package.
- **Paper and code.** The paper (arXiv:2505.06393) and any third-party SR code or weights (for example Real-ESRGAN x4plus) keep their own licences.
- **Privacy.** Licence plates can be personal data; follow the rules that apply where you use these models.
- **Error rates in our tests.** The OCR reads 74.04 % of original test crops exactly and 16.90 % of synthetic low-resolution plates; where mistakes matter, readings should be checked by a person.

---

## 12. Citation

```bibtex
@article{nascimento2025toward,
  title   = {Toward Advancing License Plate Super-Resolution in Real-World Scenarios:
             A Dataset and Benchmark},
  author  = {Nascimento, Valfride and Lima, Gabriel E. and Ribeiro, Rafael O. and
             Schwartz, William Robson and Laroca, Rayson and Menotti, David},
  journal = {arXiv preprint arXiv:2505.06393},
  year    = {2025}
}

@misc{yousuf2026paper20,
  title  = {Re-implementation and Evaluation of Paper \#20 on the 498R BanglaLPR Dataset:
            Super-Resolution and Fusion for Bangla Plate Recognition},
  author = {Md. Yousuf},
  year   = {2026},
  month  = oct,
  note   = {Paper20\_Report.md and Paper20\_README.md}
}
```

---

*End of `Paper20_README.md` — Md. Yousuf, 19 October 2026.*
