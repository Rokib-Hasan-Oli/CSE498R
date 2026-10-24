# Paper20_Report.md — Re-implementation and Evaluation Report

> **Document name:** `Paper20_Report.md`
> **Title:** Re-implementation and Evaluation of *"Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark"* (Paper #20) on Bangla Licence Plates
> **Author / Implementer:** Md. Yousuf
> **Project:** 498R BanglaLPR — team paper re-implementation study
> **Date:** 19 October 2026
> **Companion file:** `Paper20_README.md` (model card, usage guide, file manifest)
> **Source folder:** `weights/paper20/Md.Yousuf` in the team repository `Rokib-Hasan-Oli/CSE498R`
> **Environment:** Kaggle Notebook, 2 × Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, Ultralytics 8.4.170

**Source of every number.** All results in this report come from the saved run files: `runs/sr_x2/results.json`, `runs/crnn_hr/results.json`, `runs/sr_eval/results.json`, `runs/crnn_hr/ocr_preds_*.csv`, `environment.json`, the logs in `logs/`, `results_vs_paper.md`, and the code in `code/`. Statements about the original paper come from the paper itself (arXiv:2505.06393). Where this study did not measure something, the report says so.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Objectives](#2-objectives)
3. [The Original Paper](#3-the-original-paper)
4. [Implementation versus the Paper](#4-implementation-versus-the-paper)
5. [Experimental Setup](#5-experimental-setup)
6. [Results A — The OCR Model Trained on HR Plates](#6-results-a--the-ocr-model-trained-on-hr-plates)
7. [Results B — Super-Resolution Training](#7-results-b--super-resolution-training)
8. [Results C — Super-Resolution in Front of the OCR](#8-results-c--super-resolution-in-front-of-the-ocr)
9. [Results D — Fusion over Five Copies](#9-results-d--fusion-over-five-copies)
10. [Results E — Speed](#10-results-e--speed)
11. [Discussion and Interpretation](#11-discussion-and-interpretation)
12. [Limitations of This Study](#12-limitations-of-this-study)
13. [Conclusions and Recommendations](#13-conclusions-and-recommendations)
14. [Reproducibility](#14-reproducibility)
- [Appendix A — OCR Training Arguments and Curve](#appendix-a--ocr-training-arguments-and-curve)
- [Appendix B — SR Training Curve](#appendix-b--sr-training-curve)
- [Appendix C — Full Metrics of the SR Evaluation](#appendix-c--full-metrics-of-the-sr-evaluation)
- [Appendix D — Paper Tables Used for Comparison](#appendix-d--paper-tables-used-for-comparison)
- [Appendix E — Artefact Manifest](#appendix-e--artefact-manifest)
- [References](#references)

---

## 1. Executive Summary

Paper #20 studies **licence-plate super-resolution (SR)** for recognition. It introduces the UFPR-SR-Plates dataset (real low-resolution / high-resolution plate pairs), benchmarks five SR models in front of an OCR model, and shows that fusing the OCR outputs of several super-resolved frames of the same plate raises recognition further. Its headline numbers: recognition rises from 2.2 % on low-resolution (LR) plates to 29.9 % after SR of a single image, and up to 42.3 % with five super-resolved images fused by Majority Vote by Character Position (MVCP).

This study re-implements the idea on Bangla (BRTA) plates from the team's 498R BanglaLPR All-in-One dataset. No real LR/HR pairs exist for Bangla plates, so LR images are **synthesised** from HR crops. We trained a CRNN OCR on HR plate crops and a small ×2 SR network, evaluated them on the 651-plate OCR test split, and tested fusion over five differently degraded copies of each plate.

### 1.1 Headline results (651 test plates, plate exact match)

| Input to the OCR | Plate exact | CER | Token accuracy |
|---|---|---|---|
| HR crop, 64 × 128 | **65.28 %** | 9.96 % | 91.75 % |
| LR (synthetic, 32 × 64) | 16.90 % | 31.37 % | 64.26 % |
| LR upsampled with bicubic ×2 | 23.81 % | 28.47 % | 69.19 % |
| LR → our SR ×2 | **24.12 %** | **26.38 %** | **72.39 %** |
| LR → Real-ESRGAN x4plus (zero-shot) | 12.29 % | 33.30 % | 60.89 % |

| Five-copy fusion (plate exact) | single copy | HC | MV-HC | MVCP-HC |
|---|---|---|---|---|
| LR | 17.36 % | 26.11 % | 25.35 % | 21.81 % |
| Bicubic ×2 | 22.43 % | 31.64 % | 31.80 % | 28.57 % |
| Our SR ×2 | 24.88 % | 29.80 % | **32.10 %** | 28.26 % |

| Other | Value |
|---|---|
| OCR on original test crops (no SR protocol) | 74.04 % plate exact, CER 6.74 %, token accuracy 93.47 % (4.96 ms/plate) |
| SR network | 0.79 M parameters, val PSNR 23.73 dB vs 22.15 dB bicubic, 5.50 ms/plate on a T4 |
| OCR network | CRNN-CTC, 7.69 M parameters, 19.3 min training |
| SR training time | 21.5 min |

### 1.2 Findings

1. A learned SR network improved over the LR input by 7.22 points of plate exact match (16.90 → 24.12 %), but plain bicubic upsampling already gave 6.91 of those points (→ 23.81 %). The SR network's edge over bicubic is **0.31 points** in plate exact match (2 plates of 651), and **larger** in the softer metrics: CER −2.09 points, token accuracy +3.20 points, "all but one token right" +4.45 points (40.86 → 45.31 %).
2. Zero-shot Real-ESRGAN x4plus **lowered** recognition (12.29 %) relative to the LR input (16.90 %). In the paper, Real-ESRGAN raised recognition from 2.2 % to 20.2 %.
3. Fusing five copies helped every input. The best result is 32.10 % (our SR, MV-HC), 7.22 points above a single super-resolved copy.
4. MVCP-HC was the **lowest** of the three fusion rules for all three inputs (LR, bicubic, SR). In the paper MVCP was the highest.
5. Our synthetic LR plates are much easier to read than the paper's real LR plates (16.90 % vs 2.2 % with no SR), as the paper warns about synthetic degradations. Our five "frames" are five random degradations of one photograph, not five video frames.
6. The paper's reported numbers (HR 85.2 %, single-image SR 29.9 %, five-image MVCP 42.3 %) come from a different dataset, OCR model, SR models, LR size, and fusion input; they are indicative only.

---

## 2. Objectives

Re-implement the paper's pipeline — SR in front of an OCR, and fusion over several low-resolution copies — on Bangla plates, using the assets available, and report what the results do and do not show.

| ID | Question | Answered in |
|---|---|---|
| RQ1 | How well does an OCR trained on HR Bangla plates read HR, LR and super-resolved plates? | §6, §8 |
| RQ2 | Does a trained ×2 SR network beat bicubic upsampling and a zero-shot general-purpose SR model? | §8 |
| RQ3 | Does fusing the OCR outputs of five copies help, and which rule works best? | §9 |
| RQ4 | How does speed compare? | §10 |
| RQ5 | How do the results relate to the paper's? | §3, §8.3, §11 |

---

## 3. The Original Paper

**Citation.** V. Nascimento, G. E. Lima, R. O. Ribeiro, W. R. Schwartz, R. Laroca and D. Menotti, "Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark," *Journal of the Brazilian Computer Society* (author-prepared version), arXiv:2505.06393v1 [cs.CV], 9 May 2025.

### 3.1 Contributions claimed

1. **UFPR-SR-Plates**, a public dataset of 10,000 tracks with 100,000 plate images: each track has five consecutive LR images (vehicle far from the camera) and five consecutive HR images (vehicle close) of the same plate.
2. A benchmark of five SR models — general-purpose SR3 and Real-ESRGAN, and plate-specific LPSRGAN, PLNET and LCDNet — with the OCR model GP_LPR reading the super-resolved plates.
3. Three fusion strategies over the OCR outputs of N super-resolved images of one plate: Highest Confidence (HC), Majority Vote (MV) and Majority Vote by Character Position (MVCP).

### 3.2 Dataset and protocol as described in the paper

| Item | Paper's setting |
|---|---|
| Source | Rolling-shutter camera at the Federal University of Paraná, Curitiba; daytime (7 a.m.–5 p.m.) over six months; night images excluded |
| Size | 10,000 tracks × 10 images; 5,000 tracks at 1280 × 960 and 5,000 at 1920 × 1080 |
| Layouts | Brazilian (3 letters + 4 digits) and Mercosur (3 letters, 1 digit, 1 letter, 2 digits), evenly split |
| Split | 3,965 train / 2,030 validation / 4,005 test tracks, with no plate shared between splits |
| LR / HR input size | LR 16 × 48 px → SR 32 × 96 px (×2); HR 32 × 96 px |
| OCR | GP_LPR; maximum decoding length 7 |
| N for fusion | 1, 3 and 5 consecutive images, from nearest to farthest |
| Optimiser | Adam for all models; per-model learning rates and schedules listed in the paper |
| Hardware | AMD Ryzen 5950X, 128 GB RAM, NVIDIA Quadro RTX 8000 |
| Metric | Recognition rate (all characters correct), plus partial matches (≥ 6 and ≥ 5 of 7 correct) |

### 3.3 Results reported by the paper

**Table 2 — OCR on a single image per track (all characters correct / ≥ 6 / ≥ 5)**

| Test images | All | ≥ 6 | ≥ 5 |
|---|---|---|---|
| HR | 85.2 % | 98.5 % | 99.8 % |
| LR (no SR) | 2.2 % | 8.2 % | 20.1 % |
| LR + SR3 | 18.4 % | 45.8 % | 68.3 % |
| LR + LPSRGAN | 19.6 % | 46.1 % | 67.4 % |
| LR + Real-ESRGAN | 20.2 % | 49.8 % | 71.8 % |
| LR + PLNET | 29.9 % | 57.8 % | 76.9 % |
| LR + LCDNet | 29.9 % | 59.2 % | 77.1 % |

**Table 3 — Fusion of N = 5 super-resolved images, all characters correct**

| SR model | HC | MV | MVCP |
|---|---|---|---|
| SR3 | 20.2 % | 25.1 % | 28.3 % |
| LPSRGAN | 25.4 % | 27.4 % | 28.8 % |
| Real-ESRGAN | 24.9 % | 27.7 % | 29.5 % |
| PLNET | 35.9 % | 39.5 % | **40.9 %** |
| LCDNet | 35.7 % | 40.7 % | **42.3 %** |

**Table 6 — SR inference time**

| Model | Time (ms), mean ± std | FPS |
|---|---|---|
| LPSRGAN | 4.82 ± 0.45 | 207.5 |
| PLNET | 24.88 ± 1.21 | 40.2 |
| Real-ESRGAN | 34.86 ± 2.66 | 28.7 |
| LCDNet | 61.88 ± 1.77 | 16.1 |
| SR3 | 12.31 ± 0.018 s (omitted from the table) | — |

The complete Tables 3–5 are in [Appendix D](#appendix-d--paper-tables-used-for-comparison).

### 3.4 Points in the paper that affect comparison

- The paper reports a large drop from synthetic to real data: LPSRGAN reaches 93.9 % on a synthetic benchmark but 19.6 % on real UFPR-SR-Plates, and the paper concludes that synthetic degradation pipelines do not replicate real-world noise (paper p. 8).
- The paper's partial-match metric counts correct *characters* of 7. Bangla plates in this study are scored on text normalised to Bangla tokens (district, *Metro*, series letter, six digits), which is a different unit.
- The paper trains and evaluates five SR models with their own training protocols, including GAN training and OCR-guided losses for the plate-specific ones.
- The paper lists limitations: daylight only, Brazilian/Mercosur layouts only, motorcycle plates excluded, systematic OCR confusions (for example B/8).

---

## 4. Implementation versus the Paper

| Component | Paper | This study |
|---|---|---|
| Data | UFPR-SR-Plates — real LR/HR pairs, Brazilian / Mercosur plates | 498R BanglaLPR All-in-One OCR crops (BRTA plates); **LR copies synthesised from HR crops** |
| Test size | 4,005 tracks | 651 plates (`ocr/test`, human labels) |
| HR / LR size | HR 32 × 96, LR 16 × 48 (height × width) | HR 64 × 128, LR 32 × 64 (height × width); HR crops must be ≥ 40 px high and are resized to 64 × 128 with area interpolation |
| LR generation | Real far-away captures | Gaussian blur (σ 0.4–1.4) → ÷2 resize with a random interpolation (area / cubic / linear) → Gaussian noise (σ 0–8) → JPEG (quality 30–90) |
| OCR | GP_LPR | CRNN-CTC trained on HR plate crops, 7.69 M parameters |
| SR models | SR3, Real-ESRGAN, LPSRGAN, PLNET, LCDNet, each trained | One PLNET-style ×2 network (own code, 0.79 M parameters) trained with L1 loss only; Real-ESRGAN x4plus used **zero-shot** (pre-trained weights, no training on plates) |
| Fusion input | N real consecutive frames, N ∈ {1, 3, 5} | 5 differently degraded copies of **one** photograph; N = 5 only |
| Fusion rules | HC, MV, MVCP | HC, MV-HC (majority string, ties broken by confidence), MVCP-HC (vote per token position, ties by confidence) |
| Main metric | Recognition rate, all 7 characters | Plate exact match, CER, token accuracy, WER, "all but one token right" |
| Hardware | Ryzen 5950X + Quadro RTX 8000 | Kaggle Tesla T4 |
| Not replicated | — | SR3, LPSRGAN, LCDNet; real LR/HR pairs; N = 3; resolution split; real video frames |

**The SR network in code (`p20_sr.py`).** `SRNet`: a 3×3 input convolution, 10 residual channel-attention blocks (RCAB, reduction 16, 64 channels), a convolution, a ×2 PixelShuffle upsampler, and a bicubic skip connection added to the output. It is described in the script as "PLNET-like"; it is not the authors' PLNET code and has no OCR-guided or adversarial loss.

---

## 5. Experimental Setup

### 5.1 Dataset — 498R BanglaLPR All-in-One v1.0 (OCR part)

| Split | Human-labelled | Auto-labelled | Used for |
|---|---|---|---|
| `ocr/train` | 3,471 | 2,136 | Training the OCR (5,607 rows) and the SR network (5,607 crops) |
| `ocr/val` | 766 | — | OCR model selection; SR network: first 300 crops for PSNR |
| `ocr/test` | 651 | — | All reported test results |
| `ocr/test_fog` | 26 | — | Real fog copies, OCR only |
| `ocr/test_rain_synth` | 651 | — | Synthetic rain copies of the test split, OCR only |

Test plates by source: `anpr` 184 · `blpd` 30 · `kuet` 183 · `lpdba` 202 · `nightlpbd` 52. Condition subsets are fixed team-wide (`ocr/conditions.json`): clear 480 · night 65 · blurry 112 plates (subsets may overlap).

The dataset (3.39 GB) was downloaded from the team's Google Drive folder at the start of the run.

### 5.2 Hardware and software

| Item | Value |
|---|---|
| Run date | 19 Oct 2026 |
| Platform | Kaggle notebook, Linux 6.18.48, x86-64 |
| GPUs | 2 × Tesla T4; SR training on GPU 0 and OCR training on GPU 1 ran at the same time; evaluation on one T4 |
| Python / PyTorch / CUDA | 3.13.15 / 2.11.0+cu128 / 12.8 |
| Other libraries | Ultralytics 8.4.170, numpy 2.1.3, OpenCV 4.14.0, transformers 5.16.1, timm 1.0.29, peft 0.20.0 |
| Seed | 42 (SR training and OCR training) |
| Wall time | OCR 19.8 min · SR 22.2 min · SR evaluation 1.5 min |

### 5.3 Models and training

**OCR (`seq.py --model crnn`, run `crnn_hr`)**

| Setting | Value |
|---|---|
| Model | CRNN with CTC decoding, 7.69 M parameters |
| Input | 48 × 256 grey; plates with width/height < 2.6 are treated as two-line and cut into top and bottom halves placed side by side |
| Labels | Bangla tokens (district, *Metro*, series letter, digits) |
| Optimiser | AdamW + OneCycle, lr 1e-3, weight decay 1e-4, batch 64, 60 epochs |
| Augmentation | Rotation ±4°, scale 0.92–1.05, brightness/contrast, blur, noise, random down-up resize |
| Training data | 5,607 HR crops (human + auto), no flip |
| Checkpoint | Lowest validation loss (epoch 52, 0.3542); `best.pt` and `model_full.pt` hold these weights |
| Training time | 19.3 min |

**SR network (`p20_sr.py train`, run `sr_x2`)**

| Setting | Value |
|---|---|
| Model | Residual channel-attention network (see §4), ×2, 0.79 M parameters |
| Training data | 5,607 HR crops resized to 64 × 128; a new random degradation is drawn for each crop in each step |
| Loss / optimiser | L1; Adam lr 2e-4 with cosine decay; batch 32; 60 epochs |
| Validation | PSNR on 300 validation crops with fixed degradations; best-PSNR checkpoint kept |
| Training time | 21.5 min |

### 5.4 Evaluation protocol (`p20_sr.py eval`)

- Image sizes are given as height × width.
- Test plates: all 651 `ocr/test` crops with height ≥ 40 px (none were dropped), each resized to 64 × 128.
- Variants read by the same OCR: **HR** (the 64 × 128 crop), **LR** (one synthetic degradation, 32 × 64), **bicubic** (LR upsampled ×2 with cubic interpolation), **SR_ours** (LR through our SR network), **RealESRGAN_x4** (LR through Real-ESRGAN x4plus, output resized to 64 × 128 with area interpolation).
- Fusion: five copies per plate, each degraded with a different random seed; each copy is read by the OCR; the five readings are fused with HC (most confident), MV-HC (most frequent normalised string, ties by mean confidence) and MVCP-HC (vote per token position, ties by confidence).
- Metrics: plate exact match; CER (character edit distance / truth length on normalised text); token accuracy; WER (words split at spaces and hyphens); "all but one token right" (token edit distance ≤ 1).
- SR speed: milliseconds per plate on the T4, batch 1, first 5 plates excluded, AMP on.

---

## 6. Results A — The OCR Model Trained on HR Plates

### 6.1 Validation and test

| Split | Plates | Plate exact | CER | Char accuracy | Token accuracy | WER |
|---|---|---|---|---|---|---|
| Validation | 766 | 64.49 % | 10.58 % | 89.42 % | 92.43 % | 12.39 % |
| **Test** | 651 | **74.04 %** | 6.74 % | 93.26 % | 93.47 % | 9.69 % |

Inference time: 4.96 ms per plate (test).

The 651 test plates are scored in two ways in this study. On the **original crops** (the OCR's own evaluation) plate exact is **74.04 %**. On the **64 × 128 resized "HR" variant** used by the SR evaluation it is **65.28 %** (§8). §11.1 discusses the difference.

### 6.2 By source (test, original crops)

| Source | Plates | Plate exact | CER | Token accuracy |
|---|---|---|---|---|
| `lpdba` | 202 | **85.15 %** | 3.42 % | 97.50 % |
| `kuet` | 183 | 75.41 % | 3.10 % | 96.05 % |
| `anpr` | 184 | 69.57 % | 7.05 % | 93.41 % |
| `nightlpbd` | 52 | 67.31 % | 16.87 % | 89.64 % |
| `blpd` | 30 | 30.00 % | 34.81 % | 57.09 % |

### 6.3 By condition (test, original crops)

| Condition | Plates | Plate exact | CER | Token accuracy |
|---|---|---|---|---|
| All test | 651 | 74.04 % | 6.74 % | 93.47 % |
| Clear | 480 | 78.12 % | 6.23 % | 93.85 % |
| Night | 65 | 67.69 % | 14.49 % | 90.71 % |
| Blurry | 112 | 60.71 % | 5.11 % | 93.60 % |
| Fog (real fog copies) | 26 | 61.54 % | 13.02 % | 89.38 % |
| Rain (synthetic copies of the test split) | 651 | 66.67 % | 9.32 % | 90.98 % |

### 6.4 Training behaviour

- Training loss falls from 9.88 (epoch 1) to 0.018 (epoch 60); validation loss reaches its minimum of 0.3542 at epoch 52 and is 0.360 at epoch 60.
- From about epoch 42 the validation loss changes by less than 0.02 while the training loss keeps falling (0.041 → 0.018).
- The full per-epoch curve is in Appendix A.

---

## 7. Results B — Super-Resolution Training

| Quantity | Value |
|---|---|
| Parameters | 0.79 M |
| Training crops | 5,607 |
| Validation PSNR, SR | 23.73 dB (best, from epoch 53 on) |
| Validation PSNR, bicubic | 22.15 dB |
| Gain over bicubic | +1.58 dB |
| Training time | 21.5 min |

- Validation PSNR rises from 22.64 dB (epoch 1) to 23.28 dB (epoch 10), 23.62 dB (epoch 30), 23.72 dB (epoch 46) and 23.73 dB (epoch 53), then stays at 23.73 dB (Appendix B).
- The SR checkpoint `sr.pt` (3,185,557 bytes) was supplied. Inspected without running: it is a PyTorch archive whose state dict uses the `SRNet` layer names (`head`, `body.N.b`, `body.N.ca`, `up`) and holds 3,160,016 bytes of float32 tensor data, i.e. 0.790 M parameters, matching the 0.79 M recorded in `runs/sr_x2/results.json`. The checkpoint was not re-evaluated for this report; all SR numbers come from the saved `sr_eval` run.

---

## 8. Results C — Super-Resolution in Front of the OCR

### 8.1 Single-image results (651 test plates)

| Variant | Plate exact | CER | Char acc | Token acc | WER | All but one token | SR ms/plate |
|---|---|---|---|---|---|---|---|
| HR | 65.28 % | 9.96 % | 90.04 % | 91.75 % | 12.56 % | 84.64 % | — |
| LR | 16.90 % | 31.37 % | 68.63 % | 64.26 % | 45.45 % | 31.49 % | — |
| Bicubic ×2 | 23.81 % | 28.47 % | 71.53 % | 69.19 % | 40.18 % | 40.86 % | — |
| **SR_ours ×2** | **24.12 %** | **26.38 %** | **73.62 %** | **72.39 %** | **36.85 %** | **45.31 %** | 5.50 |
| Real-ESRGAN x4plus (zero-shot) | 12.29 % | 33.30 % | 66.70 % | 60.89 % | 48.39 % | 27.19 % | 51.27 |

### 8.2 Differences between variants

| Comparison | Plate exact | CER | Token acc | All but one token |
|---|---|---|---|---|
| SR_ours − LR | +7.22 | −4.99 | +8.13 | +13.82 |
| Bicubic − LR | +6.91 | −2.90 | +4.93 | +9.37 |
| **SR_ours − bicubic** | **+0.31** | **−2.09** | **+3.20** | **+4.45** |
| Real-ESRGAN − LR | −4.61 | +1.93 | −3.37 | −4.30 |
| HR − SR_ours | +41.16 | −16.42 | +19.36 | +39.33 |

One plate equals 0.154 points on the 651-plate test split; the +0.31 points in plate exact between SR_ours and bicubic is two plates (157 vs 155).

### 8.3 Next to the paper's Table 2

| Metric | Paper | This study |
|---|---|---|
| OCR on HR plates | 85.2 % | 65.28 % (74.04 % on original crops) |
| OCR on LR plates (no SR) | 2.2 % | 16.90 % |
| OCR on bicubic LR | not reported | 23.81 % |
| OCR on one SR image | PLNET 29.9 %, LCDNet 29.9 % | 24.12 % (own PLNET-style network) |
| OCR on one Real-ESRGAN image | 20.2 % | 12.29 % (zero-shot) |
| Gain from SR over LR (points) | +27.7 (PLNET) | +7.22 |

The comparison shows scale and direction only: the datasets, LR sizes, LR generation, OCR models and SR models differ (§4). In particular, the paper's LR plates are real and nearly unreadable (2.2 %), while ours are synthetic and readable at 16.90 %, which leaves much less room for SR to gain.

---

## 9. Results D — Fusion over Five Copies

### 9.1 Plate exact match (%), five differently degraded copies of each plate

| Input | Single copy | HC | MV-HC | MVCP-HC |
|---|---|---|---|---|
| LR | 17.36 | **26.11** | 25.35 | 21.81 |
| Bicubic ×2 | 22.43 | 31.64 | **31.80** | 28.57 |
| SR_ours ×2 | 24.88 | 29.80 | **32.10** | 28.26 |

### 9.2 CER (%) and "all but one token right" (%)

| Input | Fusion | CER | All but one token | Token acc | WER |
|---|---|---|---|---|---|
| LR | single copy | 32.41 | 31.18 | 62.81 | 46.80 |
| LR | HC | 28.41 | 41.32 | 68.72 | 41.15 |
| LR | MV-HC | 28.63 | 40.55 | 68.58 | 41.41 |
| LR | MVCP-HC | 30.02 | 36.10 | 67.12 | 137.44 |
| Bicubic | single copy | 29.90 | 37.17 | 67.70 | 41.67 |
| Bicubic | HC | 25.34 | 48.69 | 73.75 | 35.40 |
| Bicubic | MV-HC | 25.37 | 49.00 | 73.96 | 34.82 |
| Bicubic | MVCP-HC | 26.55 | 44.85 | 72.70 | 138.40 |
| SR_ours | single copy | 26.73 | 42.55 | 72.21 | 37.11 |
| SR_ours | HC | 23.41 | 52.23 | 75.65 | 33.69 |
| SR_ours | **MV-HC** | **23.26** | **52.84** | **76.04** | **32.88** |
| SR_ours | MVCP-HC | 24.06 | 50.23 | 75.82 | 136.37 |

### 9.3 Observations

- Fusion raised plate exact match for every input: +8.75 points (LR, HC), +9.37 points (bicubic, MV-HC), +7.22 points (SR, MV-HC) over the single copy.
- The best plate exact match in the study is **32.10 %** (SR_ours, MV-HC). Bicubic with MV-HC reaches 31.80 % (two plates fewer: 207 vs 209 of 651).
- With HC, bicubic (31.64 %) is higher than SR_ours (29.80 %) in plate exact match; with CER, token accuracy and "all but one token right", SR_ours is higher under every rule.
- MVCP-HC has the lowest plate exact match of the three rules for all three inputs (21.81 / 28.57 / 28.26 %).
- The MVCP-HC WER values (136–138 %) are above 100 % for all inputs; §11.4 explains this from the code.
- The paper's ordering is HC < MV < MVCP (Table 3); in this study MVCP-HC is the lowest.

### 9.4 Next to the paper's Table 3 and Table 2 (five-image fusion)

| Item | Paper | This study |
|---|---|---|
| Single super-resolved image | PLNET 29.9 %, LCDNet 29.9 % | 24.88 % (single copy of the fusion run) |
| 5 images, HC | PLNET 35.9 %, LCDNet 35.7 % | 29.80 % |
| 5 images, MV | PLNET 39.5 %, LCDNet 40.7 % | 32.10 % (MV-HC) |
| 5 images, MVCP | PLNET 40.9 %, LCDNet 42.3 % | 28.26 % (MVCP-HC) |
| Metric "nearly right" | ≥ 5 of 7 characters: LCDNet MVCP 86.9 % | all but one token: SR_ours MVCP-HC 50.23 % |

The last row compares different metrics (characters of a 7-character plate versus tokens of a Bangla plate) and is shown only because the team's comparison table lists both.

---

## 10. Results E — Speed

| Model | Device | ms per plate | Notes |
|---|---|---|---|
| SR_ours (0.79 M parameters) | Tesla T4, batch 1, AMP | **5.50** | Input 32 × 64 → output 64 × 128 |
| Real-ESRGAN x4plus | Tesla T4, batch 1, AMP | 51.27 | Input 32 × 64 → 128 × 256, then resized |
| CRNN OCR | Test split inference | 4.96 | 7.69 M parameters |
| *Paper: PLNET* | *Quadro RTX 8000* | *24.88 ± 1.21* | *LR 16 × 48 → 32 × 96* |
| *Paper: Real-ESRGAN* | *Quadro RTX 8000* | *34.86 ± 2.66* | |
| *Paper: LCDNet* | *Quadro RTX 8000* | *61.88 ± 1.77* | |
| *Paper: LPSRGAN* | *Quadro RTX 8000* | *4.82 ± 0.45* | |

The paper's and our timings use different GPUs, input sizes and models; they are not like-for-like. Within this study, Real-ESRGAN is 9.3 times slower than our SR network (51.27 vs 5.50 ms).

---

## 11. Discussion and Interpretation

Each statement is marked **Evidence** (follows from the saved numbers or code) or **Hypothesis** (a reasoned explanation not tested here), with the check that would settle it.

### 11.1 The two "HR" numbers (74.04 % vs 65.28 %)

- **Evidence.** The same OCR reads the same 651 test plates at 74.04 % on the original crops and at 65.28 % on the 64 × 128 resized "HR" variant used in the SR evaluation.
- **Evidence.** In `hr_rows`, every crop is resized to 64 × 128 (aspect ratio 2.0) regardless of its original shape. In `seq.prep`, an image with width/height < 2.6 is treated as a two-line plate and cut into top and bottom halves. Every HR, LR, bicubic and SR image in the SR evaluation has aspect ratio 2.0 and therefore takes the two-line path.
- **Hypothesis.** Forcing all plates to 2:1 changes how single-line crops are presented to the OCR relative to its training, which lowers the HR reference. *Check:* evaluate the HR variant resized with the aspect ratio preserved (padding), or use the original crop size for the SR experiments.
- **Consequence.** The 65.28 % HR value is a ceiling for *this* SR protocol, not the OCR's best accuracy.

### 11.2 Why the trained SR network barely beats bicubic in exact match

- **Evidence.** SR_ours improves PSNR by 1.58 dB over bicubic on validation crops, and improves CER (−2.09), token accuracy (+3.20) and all-but-one-token (+4.45) on the test set, but improves plate exact by only 0.31 points.
- **Evidence.** The network is trained with L1 loss only; it has 0.79 M parameters and 5,607 training crops. The paper's plate-specific SR models are trained with perceptual/OCR-guided or adversarial objectives (it describes an OCR discriminator and layout-aware loss for LCDNet and a CTC-based perceptual loss for LPSRGAN).
- **Hypothesis.** A pixel-wise L1 objective recovers smooth structure (higher PSNR) and fixes some tokens, but does not target the character details that decide an exact match. *Check:* add an OCR-guided loss using the frozen CRNN (CTC loss on the super-resolved image) and compare plate exact against the L1 network.
- **Hypothesis.** The OCR was trained only on HR crops, so it may under-use the improvements in super-resolved images. *Check:* fine-tune the OCR on a mix of HR, LR and SR images.

### 11.3 Why zero-shot Real-ESRGAN lowered recognition

- **Evidence.** Real-ESRGAN x4plus was applied without any training on plates to 32 × 64 synthetic LR images; its 128 × 256 output was then resized to 64 × 128. Recognition fell from 16.90 % (LR) to 12.29 %.
- **Evidence.** In the paper, Real-ESRGAN was trained with the two-stage protocol the paper describes (PSNR-oriented pre-training, then GAN fine-tuning) and improved recognition from 2.2 % to 20.2 %.
- **Hypothesis.** The zero-shot model is tuned to natural-image degradations and may hallucinate texture on small, high-contrast characters; the ×4 output followed by area resizing may also discard detail. *Check:* fine-tune Real-ESRGAN on our plate crops (as in the paper's protocol) or feed it plate-sized crops at ×2 only.

### 11.4 Fusion rules

- **Evidence.** Fusion helps (+7 to +9 points of plate exact). MV-HC is best for bicubic and SR inputs and HC is best for LR; MVCP-HC is lowest in all three.
- **Evidence (code, WER).** `metrics.fuse` returns the MVCP-HC result as `' '.join(tokens)`, i.e. each token separated by a space, while `metrics.words` splits on spaces and hyphens. Each fused token (including every digit) therefore counts as one "word", which inflates WER above 100 %. Plate exact, CER and token metrics use text normalised without spaces and hyphens, so they are not affected. The MVCP-HC WER values are therefore not comparable with the other WER values.
- **Evidence (code, MVCP).** MVCP-HC votes by token *index* after padding all readings to the longest one. If one reading drops or inserts a token, every later token of that reading is shifted to the wrong position.
- **Hypothesis.** Positional shifts caused by insertions/deletions in individual readings explain why position-wise voting underperforms whole-string voting here. In the paper the readings come from a fixed-length (7-character) decoder, where positions cannot shift. *Check:* align the five token sequences (for example by edit-distance alignment) before voting.
- **Hypothesis.** The five copies are degradations of the same photograph, so their OCR errors may be correlated, limiting what voting can recover compared with five real frames of a moving vehicle. *Check:* evaluate on real multi-frame data, or use independent crops.

### 11.5 Synthetic LR versus real LR

- **Evidence.** LR plates are far more readable here (16.90 %) than in the paper (2.2 %). The paper reports a large drop from synthetic to real data (LPSRGAN 93.9 % on a synthetic benchmark versus 19.6 % on UFPR-SR-Plates) and concludes that synthetic degradation pipelines do not replicate real-world noise.
- **Evidence.** The SR network is trained and evaluated on the same synthetic degradation family (blur, ÷2 resize, noise, JPEG with fixed parameter ranges).
- **Consequence.** The measured SR gains describe this degradation family only; real low-resolution Bangla plates may behave differently. This was not tested.

### 11.6 Conditions and sources

- **Evidence.** Plate exact is highest on `lpdba` (85.15 %) and lowest on `blpd` (30.00 %, 30 plates, CER 34.81 %). `nightlpbd` has CER 16.87 % and plate exact 67.31 %.
- **Evidence.** Blurry plates have the lowest plate exact (60.71 %) but the lowest CER (5.11 %) of the condition subsets: most characters are right, but few plates are right in full.
- **Evidence.** Synthetic rain lowers plate exact from 74.04 % to 66.67 % on the same 651 plates.
- **Hypothesis.** `blpd` may differ from the other sources in plate layout or image style (30 plates, so also high variance). *Check:* inspect the `blpd` predictions in `ocr_preds_test.csv`.

### 11.7 Validation versus test

- **Evidence.** Plate exact is 64.49 % on validation (766 plates) and 74.04 % on test (651 plates); CER is 10.58 % versus 6.74 %.
- **Hypothesis.** The splits have different source mixes (for example the weak `blpd` source or night images may weigh differently). *Check:* compute validation scores per source and compare with §6.2.

---

## 12. Limitations of This Study

1. **Synthetic low-resolution data.** No real LR/HR pairs exist for Bangla plates; all LR images are generated from HR crops.
2. **Fusion over copies, not frames.** The five fused inputs are five degradations of one photograph.
3. **Different components from the paper.** One own PLNET-style SR network instead of five trained SR models; a CRNN instead of GP_LPR; a zero-shot Real-ESRGAN.
4. **HR reference changed by resizing.** The SR evaluation forces 64 × 128 inputs (§11.1).
5. **Single run.** One seed for OCR and SR training and one fixed set of random degradations for evaluation; no confidence intervals. On 651 plates, one plate is 0.154 points.
6. **OCR trained on HR only.** It was not trained or fine-tuned on LR or SR images.
7. **Small subsets.** `blpd` has 30 plates, the fog set 26, `nightlpbd` 52, night 65.
8. **PSNR on 300 validation crops** only; SSIM and perceptual metrics were not computed.
9. **Fusion with N = 5 only;** N = 1 and N = 3 were not run.
10. **WER of MVCP-HC** is not comparable with other WER values (§11.4).
11. **Speed.** Measured on a Tesla T4 only; paper timings come from a different GPU.
12. **Not replicated:** SR3, LPSRGAN, LCDNet, real LR/HR data, and the paper's resolution split (Tables 4 and 5).

---

## 13. Conclusions and Recommendations

### Conclusions

1. On synthetic ×2 low-resolution Bangla plates, a CRNN trained on HR plates reads 16.90 % of LR plates exactly. Bicubic upsampling raises this to 23.81 % and a trained residual SR network to 24.12 %, against 65.28 % on the HR reference.
2. The SR network's advantage over bicubic is clear in CER (−2.09), token accuracy (+3.20) and all-but-one-token (+4.45), and negligible in plate exact (+0.31).
3. Zero-shot Real-ESRGAN x4plus did not help (12.29 %).
4. Fusing five copies helped every input. The best plate exact is 32.10 % (SR_ours with MV-HC). MVCP-HC was the lowest fusion rule in this implementation.
5. The paper's numbers (single-image SR 29.9 %, five-image MVCP 42.3 %) could not be matched or directly compared, because the data, SR models, OCR and fusion inputs differ.

### Recommendations

| # | Recommendation | Based on |
|---|---|---|
| 1 | Report plate exact together with CER, token accuracy and all-but-one-token when judging SR | SR's gain over bicubic is visible only in the softer metrics (§8.2) |
| 2 | Use MV-HC or HC, not the present MVCP-HC, when fusing readings of one plate | §9.1 |
| 3 | Do not use zero-shot Real-ESRGAN x4plus in front of this OCR | 12.29 % vs 16.90 % without SR (§8.1) |
| 4 | Evaluate the HR reference at original aspect ratio before drawing conclusions about SR headroom | 74.04 % vs 65.28 % (§11.1) |
| 5 | Treat all SR gains as specific to the synthetic degradation used here | §11.5 |
| 6 | Use the supplied CRNN weights as the OCR baseline for Bangla plates | 74.04 % plate exact on the test split (§6.1) |

### Experiments that would settle the open questions

| # | Experiment | Settles |
|---|---|---|
| 1 | Evaluate the HR variant with aspect-preserving resizing | §11.1 |
| 2 | Train the SR network with an OCR-guided CTC loss from the frozen CRNN | §11.2 |
| 3 | Fine-tune the OCR on a mix of HR, LR and SR images | §11.2 |
| 4 | Fine-tune Real-ESRGAN on plate crops | §11.3 |
| 5 | Align token sequences before position-wise voting; compute WER on normalised text for MVCP | §11.4 |
| 6 | Run fusion with N = 1, 3, 5 and with independent crops or real multi-frame data | §11.4 |
| 7 | Multi-seed runs of OCR and SR training; several degradation seeds for evaluation | Variation (limitation 5) |
| 8 | Per-source validation scores | §11.7 |
| 9 | Collect real low-resolution Bangla plate tracks | §11.5 |

---

## 14. Reproducibility

### 14.1 Commands (from the script docstring and the run log)

```bash
# Run in parallel on two GPUs
python p20_sr.py train --name sr_x2 --project runs --epochs 60                 # GPU 0
python seq.py --model crnn --name crnn_hr --project runs --epochs 60            # GPU 1

# Evaluation after both finish
python p20_sr.py eval --sr runs/sr_x2/sr.pt --ocr runs/crnn_hr/model_full.pt \
                      --name sr_eval --project runs
```

### 14.2 Supplied artefacts

The supplied artefacts are `paper20_results.zip` (results, logs, code, environment), `sr.pt` (SR network weights), `paper20_crnn_best.pt` (OCR state dict), `paper20_crnn_model_full.pt` (OCR weights with class list and settings), `results_vs_paper.md`, and the paper PDF. See [Appendix E](#appendix-e--artefact-manifest).

### 14.3 Determinism

Seed 42 was set for OCR and SR training. The evaluation uses fixed random seeds for the degradations (`10000 + i` for the single LR copy and `1000 × j + i` for fusion copy *j*), so the same copies are produced on every run.

---

## Appendix A — OCR Training Arguments and Curve

| Argument | Value |
|---|---|
| Model | `crnn` (CTC) |
| Epochs / batch | 60 / 64 |
| Learning rate / optimiser | 1e-3 / AdamW (weight decay 1e-4) + OneCycle (10 % warm-up) |
| Input | 48 × 256 |
| Two-line stacking | true |
| Flip | false |
| Training rows | 5,607 (human 3,471 + auto 2,136) |
| Seed | 42 |

| Epoch | 1 | 5 | 10 | 20 | 30 | 40 | 50 | 52 | 60 |
|---|---|---|---|---|---|---|---|---|---|
| Train loss | 9.877 | 2.030 | 0.556 | 0.223 | 0.103 | 0.046 | 0.023 | 0.021 | 0.018 |
| Val loss | 3.331 | 2.166 | 0.806 | 0.473 | 0.388 | 0.366 | 0.363 | **0.354** | 0.360 |

---

## Appendix B — SR Training Curve

Validation PSNR (dB) on 300 validation crops; bicubic baseline 22.15 dB at every epoch.

| Epoch | 1 | 5 | 10 | 20 | 30 | 40 | 46 | 53 | 60 |
|---|---|---|---|---|---|---|---|---|---|
| SR | 22.64 | 23.11 | 23.28 | 23.48 | 23.62 | 23.68 | 23.72 | 23.73 | 23.73 |

---

## Appendix C — Full Metrics of the SR Evaluation

All values in %, 651 test plates. Columns: plate exact / CER / character accuracy / token accuracy / WER / all-but-one-token.

| Variant | Plate exact | CER | Char acc | Token acc | WER | All but one |
|---|---|---|---|---|---|---|
| HR | 65.28 | 9.96 | 90.04 | 91.75 | 12.56 | 84.64 |
| LR | 16.90 | 31.37 | 68.63 | 64.26 | 45.45 | 31.49 |
| Bicubic | 23.81 | 28.47 | 71.53 | 69.19 | 40.18 | 40.86 |
| SR_ours | 24.12 | 26.38 | 73.62 | 72.39 | 36.85 | 45.31 |
| RealESRGAN_x4 | 12.29 | 33.30 | 66.70 | 60.89 | 48.39 | 27.19 |

Fusion rows (five copies) are in §9.2.

---

## Appendix D — Paper Tables Used for Comparison

**Table 3 — all characters correct, both resolutions (HC / MV / MVCP)**

| SR model | N = 3 | N = 5 |
|---|---|---|
| SR3 | 19.9 / 21.4 / 24.1 | 20.2 / 25.1 / 28.3 |
| LPSRGAN | 24.0 / 24.5 / 25.3 | 25.4 / 27.4 / 28.8 |
| Real-ESRGAN | 23.5 / 24.1 / 25.6 | 24.9 / 27.7 / 29.5 |
| PLNET | 34.5 / 36.1 / 36.6 | 35.9 / 39.5 / 40.9 |
| LCDNet | 34.8 / 36.7 / 37.8 | 35.7 / 40.7 / 42.3 |

**Tables 4 and 5 — LCDNet and PLNET, N = 5, MVCP, all characters correct, by source resolution**

| Model | 1280 × 960 | 1920 × 1080 |
|---|---|---|
| PLNET | 22.7 % | 59.3 % |
| LCDNet | 23.5 % | 61.4 % |

---

## Appendix E — Artefact Manifest

| Path | Description |
|---|---|
| `Paper20_Report.md` | This report |
| `Paper20_README.md` | Model card, usage guide, file manifest |
| `sr.pt` | SR network weights (`SRNet`, ×2, 0.79 M parameters, ≈ 3.0 MiB) |
| `paper20_crnn_best.pt` | CRNN OCR weights, state dict only (≈ 29.4 MiB) |
| `paper20_crnn_model_full.pt` | CRNN OCR weights plus class list, input size and stacking flag (≈ 29.4 MiB) |
| `paper20_results.zip` | Results archive (no weights) |
| `results_vs_paper.md` | Short comparison table |
| `20_2025_License Plate Super-Resolution Dataset and Benchmark.pdf` | The paper (arXiv:2505.06393) |
| `runs/sr_x2/results.json` | SR training summary |
| `runs/crnn_hr/results.json`, `ocr_preds_test.csv`, `ocr_preds_val.csv` | OCR results and per-plate predictions |
| `runs/sr_eval/results.json` | SR evaluation and fusion results |
| `logs/sr_x2.log`, `logs/crnn_hr.log`, `logs/sr_eval.log` | Run logs |
| `environment.json` | Library and hardware versions |
| `code/p20_sr.py` | Paper #20 SR training and evaluation |
| `code/seq.py`, `code/metrics.py`, `code/vocab.py`, `code/common.py`, `code/report.py` | Shared toolkit used by Paper #20 |
| `code/det.py`, `code/rec.py`, `code/ved.py` | Toolkit modules (detector, character-YOLO OCR, transformer OCR) not used for Paper #20 |

---

## References

1. V. Nascimento, G. E. Lima, R. O. Ribeiro, W. R. Schwartz, R. Laroca and D. Menotti, "Toward Advancing License Plate Super-Resolution in Real-World Scenarios: A Dataset and Benchmark," arXiv:2505.06393v1 [cs.CV], 2025. (Paper #20.)
2. X. Wang et al., "Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data," 2021. (Cited by the paper; pre-trained x4plus weights from the Real-ESRGAN release.)
3. B. Shi, X. Bai and C. Yao, "An end-to-end trainable neural network for image-based sequence recognition" (CRNN). (Architecture family of the OCR used here.)
4. Y. Liu et al., GP_LPR (Global Perception License Plate Recognition), 2024. (OCR used by the paper.)
5. 498R BanglaLPR All-in-One v1.0 — team dataset (OCR split: sources `anpr`, `blpd`, `kuet`, `lpdba`, `nightlpbd`).

---

*End of `Paper20_Report.md` — Md. Yousuf, 19 October 2026.*
