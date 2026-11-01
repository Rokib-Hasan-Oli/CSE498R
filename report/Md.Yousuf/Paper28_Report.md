# Paper28_Report.md — Re-implementation and Evaluation Report

> **Document name:** `Paper28_Report.md`
> **Title:** Re-implementation and Evaluation of *"YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"* (Paper #28) on Bangla Licence Plates
> **Author / Implementer:** Md. Yousuf
> **Project:** 498R BanglaLPR — team paper re-implementation study
> **Date:** 21 October 2026
> **Companion file:** `Paper28_README.md` (model cards, usage guide, file manifest)
> **Source folder:** `weights/paper28/Md.Yousuf` in the team repository `Rokib-Hasan-Oli/CSE498R`
> **Environment:** Kaggle Notebook, 2 × Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu128, Ultralytics 8.4.170

**Source of every number.** All results in this report come from the saved run files: `runs/<run>/results.json`, `runs/<run>/results.csv`, `runs/<run>/args.yaml`, `environment.json`, the logs in `logs/`, `results_vs_paper.md`, the notebook run log, and the code in `code/`. Statements about the original paper come from the paper itself (arXiv:2402.13616v2). Where this study did not measure something, the report says so. The four runs are `v9t`, `v9s`, `v8n`, `v8s` (YOLOv9-T, YOLOv9-S, YOLOv8-N, YOLOv8-S).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Objectives](#2-objectives)
3. [The Original Paper](#3-the-original-paper)
4. [Implementation versus the Paper](#4-implementation-versus-the-paper)
5. [Experimental Setup](#5-experimental-setup)
6. [Results A — Detection Accuracy of the Four Models](#6-results-a--detection-accuracy-of-the-four-models)
7. [Results B — Training Behaviour and Cost](#7-results-b--training-behaviour-and-cost)
8. [Results C — Conditions and Sources](#8-results-c--conditions-and-sources)
9. [Results D — Plate Counts, Localisation and Plate Size](#9-results-d--plate-counts-localisation-and-plate-size)
10. [Results E — Speed](#10-results-e--speed)
11. [Discussion and Interpretation](#11-discussion-and-interpretation)
12. [Limitations of This Study](#12-limitations-of-this-study)
13. [Conclusions and Recommendations](#13-conclusions-and-recommendations)
14. [Reproducibility](#14-reproducibility)
- [Appendix A — Training Arguments and Validation Curve](#appendix-a--training-arguments-and-validation-curve)
- [Appendix B — Loss Curves](#appendix-b--loss-curves)
- [Appendix C — Full Metrics by Condition](#appendix-c--full-metrics-by-condition)
- [Appendix D — Paper Tables Used for Comparison](#appendix-d--paper-tables-used-for-comparison)
- [Appendix E — Artefact Manifest](#appendix-e--artefact-manifest)
- [References](#references)

---

## 1. Executive Summary

Paper #28 introduces **Programmable Gradient Information (PGI)** and the **Generalized Efficient Layer Aggregation Network (GELAN)**, and combines them into YOLOv9. PGI gives the main branch reliable gradients through an auxiliary reversible branch that is used in training only; GELAN is a lightweight backbone built from conventional convolutions. On MS COCO, trained from scratch for 500 epochs, the paper reports YOLOv9-S at 46.8 % AP (63.4 % AP50) with 7.1 M parameters, against 44.9 % AP (61.8 % AP50) for YOLOv8-S with 11.2 M.

This study fine-tunes four COCO-pretrained Ultralytics detectors — YOLOv9-T, YOLOv9-S, YOLOv8-N and YOLOv8-S — on the **one-class plate detection split** of the team's 498R BanglaLPR All-in-One dataset (8,574 training images), evaluates them on the 1,153-image test split (1,011 plates), on condition subsets (clear, night, blurry, small plates, synthetic rain, synthetic fog), on 160 plate-free photos, and measures speed on a Tesla T4 (PyTorch FP32/FP16, TensorRT FP16) and on a CPU (ONNX). The paper's PGI-vs-GELAN ablation was **not** run.

### 1.1 Headline results (test split, 1,153 images, 1,011 plates)

| Metric (test, 1,153 images, 1,011 plates) | YOLOv9-T | YOLOv9-S | YOLOv8-N | YOLOv8-S |
|---|---|---|---|---|
| mAP50 (%) | 98.30 | 98.79 | 98.73 | 98.51 |
| mAP50-95 (%) | 45.16 | 46.33 | 45.23 | 45.82 |
| Precision (%) | 97.36 | 97.04 | 97.22 | 96.90 |
| Recall (%) | 95.65 | 97.20 | 96.68 | 96.93 |
| Plates found at conf 0.25 (%) | 96.64 | 97.82 | 98.12 | 98.12 |
| Parameters (M) / GFLOPs | 2.01 / 7.8 | 7.29 / 27.4 | 3.01 / 8.2 | 11.14 / 28.6 |
| T4 TensorRT FP16 (ms) | 6.93 | 7.24 | 6.38 | 7.52 |
| CPU ONNX (ms) | 116.20 | 296.93 | 105.83 | 300.71 |
| Training time (min) | 193.0 | 149.0 | 101.0 | 119.4 |
| Epochs run / best epoch | 54 / 49 | 60 / 54 | 60 / 56 | 60 / 55 |

| Comparison | mAP50 | mAP50-95 | Parameters | TensorRT FP16 |
|---|---|---|---|---|
| YOLOv9-S − YOLOv8-S (this study) | +0.28 | +0.51 | 7.29 M vs 11.14 M (−34.6 %) | 7.24 vs 7.52 ms |
| YOLOv9-S − YOLOv8-S (paper, COCO) | +1.6 (AP50) | +1.9 (AP50:95) | 7.1 M vs 11.2 M | not reported |
| YOLOv9-T − YOLOv8-N (this study) | −0.43 | −0.07 | 2.01 M vs 3.01 M (−33.2 %) | 6.93 vs 6.38 ms |

| Other | Value |
|---|---|
| Conditions (test mAP50, range over the four models) | night 94.3–97.5 · blurry 95.2–98.1 · small plates 84.5–93.2 · synthetic rain 77.0–88.3 · synthetic fog 95.0–97.9 |
| False alarms on 160 plate-free photos | 5.62–6.88 % (9 to 11 photos) |
| Total wall time of the notebook | about 6.0 h (21,658 s to the results table) on 2 × Tesla T4 |

### 1.2 Findings

1. The four models are within **0.49 points** of test mAP50 (98.30–98.79 %) and 1.17 points of mAP50-95 (45.16–46.33 %). YOLOv9-S is highest in mAP50, mAP50-95 and recall (97.20 %).
2. YOLOv9-S beats YOLOv8-S by 0.28 points mAP50 and 0.51 points mAP50-95 with 34.6 % fewer parameters. The direction matches the paper's COCO result (+1.6 / +1.9); the size is smaller, and each model was trained once.
3. YOLOv9-T has the fewest parameters (2.01 M, 4.4 MiB) and costs about the same accuracy as YOLOv8-N (−0.43 mAP50), but it finds fewer plates at confidence 0.25 (96.64 % vs 98.12 %, 15 plates of 1,011), took 1.9 times as long to train (193.0 vs 101.0 min) and was **stopped at epoch 54 of 60** by the 3.2 h time cap.
4. Fewer parameters did **not** mean faster inference here. YOLOv8-N is the fastest in every PyTorch mode and on CPU; TensorRT FP16 puts all four within 6.38–7.52 ms. PyTorch FP16 at batch 1 was slower than FP32 for all four models.
5. Conditions separate the models more than the clean test set does: on synthetic rain the spread is 11.3 points (YOLOv9-S 77.0 % to YOLOv9-T 88.3 %), on clear photos 1.2 points.
6. Test mAP50 is higher than validation mAP50 for all four models (+1.4 to +2.5 points) but test mAP50-95 is **14.2–15.2 points lower** than validation mAP50-95; §11.2 discusses this.
7. The Ultralytics YOLOv9 model graph in the training log has one `Detect` head and no auxiliary branch. The comparison therefore measures GELAN-based YOLOv9 against YOLOv8 as implemented in Ultralytics, **not** the effect of PGI (§11.3).
8. The paper's COCO numbers (AP50 63.4 %, AP 46.8 %) come from different data, training and class count; they are indicative only.

---

## 2. Objectives

Re-implement the paper's comparison — YOLOv9 against YOLOv8 at small model sizes — on Bangla plates, using the assets available, and report what the results do and do not show.

| ID | Question | Answered in |
|---|---|---|
| RQ1 | How accurately do the four detectors find Bangla plates on the held-out test split? | §6, §9 |
| RQ2 | Does YOLOv9 beat YOLOv8 at comparable size, as in the paper's Table 1? | §6, §10, §11.1 |
| RQ3 | How robust are the models on night, blurry, small-plate, rain and fog images and on plate-free photos? | §8 |
| RQ4 | What do training and inference cost on a T4 and on a CPU? | §7, §10 |
| RQ5 | How do the results relate to the paper's? | §3, §8.4, §11 |

---

## 3. The Original Paper

**Citation.** C.-Y. Wang, I-H. Yeh and H.-Y. M. Liao, "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information," arXiv:2402.13616v2 [cs.CV], 29 February 2024. Code: github.com/WongKinYiu/yolov9.

### 3.1 Contributions claimed

1. **Programmable Gradient Information (PGI)**: an auxiliary reversible branch and multi-level auxiliary information that supply the main branch with complete input information and reliable gradients during training; the auxiliary branch is removed at inference, so it adds no inference cost.
2. **GELAN**, a lightweight network built from conventional convolutions (CSP blocks with planned RepConv computational blocks) designed by gradient path planning, claimed to use parameters better than depth-wise-convolution designs such as YOLO MS.
3. **YOLOv9**, the combination of GELAN and PGI, reported on MS COCO as better than other train-from-scratch real-time detectors and, in the paper's Figure 5, better than detectors pre-trained on large data such as RT DETR.

### 3.2 Dataset and protocol as described in the paper

| Item | Paper's setting |
|---|---|
| Dataset | MS COCO 2017 (80 classes), validation split for reported AP |
| Training | From scratch, 500 epochs; setup follows YOLOv7 AF |
| Optimiser | SGD, momentum 0.937, weight decay 0.0005; lr 0.01 to 0.0001 with linear decay; 3 warm-up epochs (warm-up momentum 0.8, bias lr 0.1) |
| Loss gains | box 7.5, class 0.5, DFL 1.5 |
| Augmentation | HSV saturation 0.7, value 0.4, translation 0.1, scale 0.9, mosaic 1.0, MixUp 0.15, copy & paste 0.3; mosaic off for the last 15 epochs |
| Input size | 640 |
| Models | YOLOv9-S/M/C/E and GELAN-S/M/C/E, compared with many detectors (Table 1) |
| Metrics | AP50:95, AP50, AP75, APS / APM / APL, parameters, FLOPs |
| Compute | Not stated in the main text; the paper thanks the National Center for High-performance Computing (NCHC) for computational resources |

### 3.3 Results reported by the paper

**Table 1 (extract) — MS COCO val, real-time detectors**

| Model | Params (M) | FLOPs (G) | AP50:95 (%) | AP50 (%) | AP75 (%) |
|---|---|---|---|---|---|
| YOLOv8-N | 3.2 | 8.7 | 37.3 | 52.6 | — |
| YOLOv8-S | 11.2 | 28.6 | 44.9 | 61.8 | — |
| GELAN-S | 7.1 | 26.4 | 46.7 | 63.0 | 50.7 |
| **YOLOv9-S** | 7.1 | 26.4 | **46.8** | **63.4** | 50.7 |

**Table 5 — ablation on PGI (AP50:95 / AP50 / AP75, %)**

| Model | AP50:95 | AP50 | AP75 |
|---|---|---|---|
| GELAN-S | 46.7 | 63.0 | 50.7 |
| + deep supervision (DS) | 46.5 | 62.9 | 50.5 |
| + PGI | 46.8 | 63.4 | 50.7 |

The paper's other comparisons (Tables 1–6 in full) are in [Appendix D](#appendix-d--paper-tables-used-for-comparison).

### 3.4 Points in the paper that affect comparison

- The paper trains **from scratch** for 500 epochs on 80 classes; this study fine-tunes pretrained weights for 60 epochs on one class (§4).
- The paper's headline claim for lightweight models is efficiency at equal accuracy: YOLOv9 uses about 10 % fewer parameters and 5–15 % fewer calculations than YOLO MS for 0.4–0.6 points more AP; YOLOv9-E uses 16 % fewer parameters and 27 % fewer calculations than YOLOv8-X for 1.7 points more AP.
- The PGI gain on the small model is small in the paper itself: +0.1 AP50:95 and +0.4 AP50 for GELAN-S (Table 5); the gain grows with model size (+0.6 AP50:95 for GELAN-E).
- The paper reports no latency or training-time figures.
- PGI is a **training-time** mechanism: the auxiliary reversible branch is removed at inference.

---

## 4. Implementation versus the Paper

| Component | Paper | This study |
|---|---|---|
| Data | MS COCO 2017, 80 classes | 498R BanglaLPR All-in-One detection split, 1 class: `plate` |
| Training | From scratch, 500 epochs | Fine-tuned from COCO-pretrained Ultralytics weights, 60 epochs planned (54 run for YOLOv9-T) |
| Models | YOLOv9-S/M/C/E, GELAN-S/M/C/E, YOLOv8-N to X and others | YOLOv9-T/S and YOLOv8-N/S (Ultralytics 8.4.170) |
| PGI | Auxiliary reversible branch and multi-level auxiliary information; ablation in Table 5 | **Not run**; the Ultralytics YOLOv9 graph has a single `Detect` head (§11.3) |
| Input size | 640 | 640 |
| Optimiser | SGD, lr 0.01 → 0.0001 linear, 3 warm-up epochs | SGD, lr0 0.01, lrf 0.01 (final 0.0001), 3 warm-up epochs, batch 16, seed 42, AMP on |
| Augmentation | scale 0.9, MixUp 0.15, copy & paste 0.3, mosaic off for the last 15 epochs | scale 0.5, MixUp 0, copy & paste 0, mosaic off for the last 10 epochs; plus fliplr 0.5, RandAugment, erasing 0.4 |
| Metrics | AP50:95, AP50, AP75, APS / M / L, parameters, FLOPs | mAP50, mAP50-95, P, R, plate counts, IoU quality, size recall, condition subsets, false alarms, latency, training time |
| Speed | Not reported | T4 (PyTorch FP32 / FP16, TensorRT FP16) and CPU (ONNX), batch 1 |
| Hardware | Not stated in the main text | Kaggle, 2 × Tesla T4 (one queue per GPU, run at the same time) |
| Not replicated | — | PGI ablation, training from scratch, YOLOv9-M/C/E, YOLOv8-M and larger, GELAN-only models, several seeds |

**The detector code in `det.py`.** `det.py` trains one Ultralytics detector with a time cap (`--max-hours`, checked after each epoch), then evaluates `best.pt` on validation and test and writes `results.json`. The plate counts use confidence 0.25 and IoU 0.5: *found %* is true positives over all ground-truth plates; *correct %* is true positives over all boxes (a precision at that threshold); *photos with every plate found* and *IoU > 0.7 accuracy* are per-photo measures. Plate size follows the COCO area rule on the stored image (small < 32², medium < 96², otherwise large). `condition_metrics` re-runs the validator on the condition lists and on the synthetic rain and fog copies of the test split, and counts photos with at least one box among the plate-free test photos at conf 0.25 (false alarms). `measure_speed` times the first 200 test images after 15 warm-up images (CPU: first 50).

---

## 5. Experimental Setup

### 5.1 Dataset — 498R BanglaLPR All-in-One v1.0 (detection part)

| Split | Images | Labels | Used for |
|---|---|---|---|
| `detection/train` | 8,574 (466 background images) | 4,067 human + 4,507 auto | Training |
| `detection/val` | 896 (99 background images; 850 plate instances) | human | Checkpoint selection (best fitness) |
| `detection/test` | 1,153 (160 plate-free photos; 1,011 plates) | human | All reported test results |
| `images/test_rain_synth` | 1,153 | as test | Synthetic rain copies of the test split |
| `images/test_fog_synth` | 1,153 | as test | Synthetic fog copies of the test split |

Test plates and images by source: `kuet` 387 / 540 · `lpdba` 211 / 205 · `ridoykj` 274 / 274 · `nightlpbd` 60 / 60 · `tusar` 29 / 29 · `mahedi` 32 / 32 · `towniyan` 18 / 13. Condition subsets are fixed team-wide lists (`detection/conditions.json`): clear 868 images; the other three real subsets have 71, 53 and 22 images (night, blurry and small plates, in the order the code evaluates them); subsets may overlap. The test split contains only **5 small plates** by the COCO area rule (267 medium, 739 large). The dataset (3.39 GB) was downloaded from the team Google Drive at the start of the run.

The dataset listing printed in the notebook also contains `leaky_kuet` subsets (for example `detection|fair_test` 540, `test_clean` 474, `test_dup` 79). The test split used here has 540 `kuet` images, the same count as `fair_test`; which `kuet` subset the main `test` split corresponds to was not checked in this study (§12).

### 5.2 Hardware and software

| Item | Value |
|---|---|
| Run date | 20 Oct 2026 (environment record 02:37) |
| Platform | Kaggle notebook, Linux 6.18.48, x86-64 |
| GPUs | 2 × Tesla T4; YOLOv9 queue (v9t, then v9s) on GPU 0 and YOLOv8 queue (v8n, then v8s) on GPU 1 at the same time |
| Python / PyTorch / CUDA | 3.13.15 / 2.11.0+cu128 / 12.8 |
| Other libraries | Ultralytics 8.4.170, TensorRT 11.4.0.106, numpy 2.1.3, OpenCV 4.14.0, transformers 5.16.1, timm 1.0.29, peft 0.20.0 |
| Seed | 42, `deterministic: true` for all four trainings |
| Wall time | Training 196.7 (v9t), 152.3 (v9s), 104.3 (v8n), 123.2 (v8s) min; speed tests 5.8, 4.0, 5.5, 3.3 min |

### 5.3 Models and training

| Setting | YOLOv9-T | YOLOv9-S | YOLOv8-N | YOLOv8-S |
|---|---|---|---|---|
| Start weights | `yolov9t.pt` | `yolov9s.pt` | `yolov8n.pt` | `yolov8s.pt` |
| Items transferred | 1303 / 1339 | 1333 / 1339 | 319 / 355 | 349 / 355 |
| Parameters (as trained / fused) | 2,005,603 / 1,970,979 | 7,287,795 / 7,167,475 | 3,011,043 / 3,005,843 | 11,135,987 / 11,125,971 |
| GFLOPs (as trained / fused) | 7.8 / 7.6 | 27.4 / 26.7 | 8.2 / 8.1 | 28.6 / 28.4 |
| Layers (as trained / fused) | 544 / 197 | 544 / 197 | 129 / 72 | 129 / 72 |
| Time cap (`--max-hours`) | 3.2 | 4.0 | 2.8 | 3.6 |
| Epochs run of 60 | 54 | 60 | 60 | 60 |
| Best epoch (fitness 0.1·mAP50 + 0.9·mAP50-95, validation) | 49 | 54 | 56 | 55 |

Shared settings: imgsz 640, batch 16, SGD lr0 0.01, lrf 0.01, momentum 0.937, weight decay 5e-4, warm-up 3 epochs, mosaic 1.0 (off for the last 10 epochs), HSV 0.015 / 0.7 / 0.4, translate 0.1, scale 0.5, fliplr 0.5, RandAugment, erasing 0.4, mixup / cutmix / copy-paste 0, AMP on, 4 workers. Full arguments are in [Appendix A](#appendix-a--training-arguments-and-validation-curve).

### 5.4 Evaluation protocol

- Test split used once, at the end, with each run's `best.pt`.
- Ultralytics validator: precision, recall, mAP50, mAP50-95 on `test`; the same on `val`.
- Plate counts at conf 0.25 and IoU 0.5 (§4), including by source and by plate size.
- Condition subsets: mAP on each real subset, on the synthetic rain and fog copies; false alarms on the 160 plate-free test photos at conf 0.25.
- Speed: batch 1, imgsz 640, first 200 test images after 15 warm-up images; time = pre-process + inference + post-process (NMS included). Modes: PyTorch FP32, PyTorch FP16, TensorRT FP16 (T4); ONNX on CPU (first 50 images).

---

## 6. Results A — Detection Accuracy of the Four Models

### 6.1 Validation and test

| Run | Val mAP50 | Test mAP50 | Δ | Val mAP50-95 | Test mAP50-95 | Δ |
|---|---|---|---|---|---|---|
| YOLOv9-T | 96.21 | 98.30 | +2.09 | 60.40 | 45.16 | -15.24 |
| YOLOv9-S | 97.10 | 98.79 | +1.69 | 61.17 | 46.33 | -14.84 |
| YOLOv8-N | 96.22 | 98.73 | +2.51 | 59.73 | 45.23 | -14.50 |
| YOLOv8-S | 97.07 | 98.51 | +1.44 | 60.04 | 45.82 | -14.22 |

| Run | Val P | Val R | Test P | Test R |
|---|---|---|---|---|
| YOLOv9-T | 96.31 | 94.35 | 97.36 | 95.65 |
| YOLOv9-S | 95.00 | 95.29 | 97.04 | 97.20 |
| YOLOv8-N | 95.56 | 94.82 | 97.22 | 96.68 |
| YOLOv8-S | 96.37 | 94.71 | 96.90 | 96.93 |

The test numbers are the headline table of §1.1. YOLOv9-S is highest in test mAP50 (98.79 %), mAP50-95 (46.33 %) and recall (97.20 %); YOLOv9-T is highest in precision (97.36 %) and lowest in recall (95.65 %).

### 6.2 Differences between models

| Comparison | mAP50 | mAP50-95 | Recall | Precision |
|---|---|---|---|---|
| YOLOv9-S − YOLOv8-S | +0.28 | +0.51 | +0.27 | +0.14 |
| YOLOv9-T − YOLOv8-N | −0.43 | −0.07 | −1.03 | +0.14 |
| YOLOv9-S − YOLOv9-T | +0.49 | +1.17 | +1.55 | −0.32 |
| YOLOv8-S − YOLOv8-N | −0.22 | +0.59 | +0.25 | −0.32 |

One plate equals 0.099 points of *found %* on the 1,011 test plates.

### 6.3 Next to the paper's Table 1

| Item | Paper (MS COCO val) | This study (498R test split) |
|---|---|---|
| YOLOv9-S parameters / FLOPs | 7.1 M / 26.4 G | 7.29 M / 27.4 G as trained (7.17 M / 26.7 G fused) |
| YOLOv8-S parameters / FLOPs | 11.2 M / 28.6 G | 11.14 M / 28.6 G |
| YOLOv8-N parameters / FLOPs | 3.2 M / 8.7 G | 3.01 M / 8.2 G |
| YOLOv9-T | not listed | 2.01 M / 7.8 G |
| AP50 / mAP50 — YOLOv9-S | 63.4 % | 98.79 % |
| AP50 / mAP50 — YOLOv8-S | 61.8 % | 98.51 % |
| AP50:95 / mAP50-95 — YOLOv9-S | 46.8 % | 46.33 % |
| AP50:95 / mAP50-95 — YOLOv8-S | 44.9 % | 45.82 % |
| AP50:95 / mAP50-95 — YOLOv8-N | 37.3 % | 45.23 % |
| YOLOv9-S − YOLOv8-S, AP50:95 / mAP50-95 | +1.9 | +0.51 |
| YOLOv9-S − YOLOv8-S, AP50 / mAP50 | +1.6 | +0.28 |
| PGI ablation (Table 5, GELAN-S 46.7 → +PGI 46.8) | reported | not run |

The comparison shows direction and size only: the data (80-class COCO vs one class), training (500 epochs from scratch vs 60 epochs fine-tuning) and metrics' difficulty differ (§4). Parameter and FLOP counts, which do not depend on the data, agree with the paper within 4 % for YOLOv9-S (7.29 vs 7.1 M, 27.4 vs 26.4 G) and match for YOLOv8-S (11.14 vs 11.2 M); the one-class head explains the small differences.

---

## 7. Results B — Training Behaviour and Cost

| Run | Epochs run | Best epoch | Train min (results.json) | Mean s / epoch | Peak GPU mem (GB) | Final lr (pg0) | Wall min in notebook |
|---|---|---|---|---|---|---|---|
| YOLOv9-T | 54 | 49 | 193.0 | 213.4 | 3.46 | 0.001255 | 196.7 |
| YOLOv9-S | 60 | 54 | 149.0 | 148.5 | 6.45 | 0.000265 | 152.3 |
| YOLOv8-N | 60 | 56 | 101.0 | 100.1 | 2.59 | 0.000265 | 104.3 |
| YOLOv8-S | 60 | 55 | 119.4 | 118.8 | 4.5 | 0.000265 | 123.2 |

- Mean seconds per epoch: YOLOv9-T 213.4, YOLOv9-S 148.5, YOLOv8-S 118.8, YOLOv8-N 100.1. YOLOv9-T was **slower per epoch than the larger YOLOv9-S** (213.4 vs 148.5 s).
- YOLOv9-T ran at the same time as YOLOv8-N and YOLOv8-S; YOLOv9-S shared the two GPUs with YOLOv8-S for only its first 30.8 min (calculated from the job times in notebook cell 14). The epoch times are therefore not measured on an idle GPU pair.
- YOLOv9-T stopped after epoch 54 (`[time cap] 3.2 h reached after epoch 54`; Ultralytics: "54 epochs completed in 3.202 hours"). Its learning rate at the last epoch is 0.00125, against 0.000265 for the other three, and 4 of the 10 planned no-mosaic epochs were run.
- The log prints a second `[time cap] ... after epoch 55` line at the start of the later test validation; `results.csv` has 54 rows, so 54 is the epoch count used here (§11.5).
- Validation mAP50-95 rises from 52.9 % (YOLOv9-T, epoch 1) to 60.4 % at epoch 49, from 52.1 % (YOLOv9-S) to 61.2 % at epoch 54, from 51.3 % (YOLOv8-N) to 59.7 % at epoch 56, from 54.4 % (YOLOv8-S) to 60.0 % at epoch 55 (Appendix A).
- Training box loss falls steadily to the last epoch for all four while validation box loss flattens at about 1.27–1.28 (Appendix B); validation mAP50-95 is flat or slightly lower after the best epoch in all runs.

---

## 8. Results C — Conditions and Sources

### 8.1 By condition (test mAP50, %)

| Run | All test | Clear | Night | Blurry | Small plates | Rain (synthetic) | Fog (synthetic) | False alarms on plate-free photos (%) |
|---|---|---|---|---|---|---|---|---|
| YOLOv9-T | 98.3 | 97.9 | 97.5 | 98.1 | 93.2 | 88.3 | 95.0 | 5.62 |
| YOLOv9-S | 98.8 | 99.0 | 96.8 | 98.0 | 90.8 | 77.0 | 96.6 | 6.25 |
| YOLOv8-N | 98.7 | 99.0 | 94.5 | 97.8 | 92.8 | 83.8 | 96.3 | 6.88 |
| YOLOv8-S | 98.5 | 99.0 | 94.3 | 95.2 | 84.5 | 82.5 | 97.9 | 5.62 |

Condition subsets are fixed for the whole team (`detection/conditions.json`); rain and fog copies are synthetic. mAP50-95 on the same subsets:

| Run | All test | Clear | Night | Blurry | Small plates | Rain (synthetic) | Fog (synthetic) |
|---|---|---|---|---|---|---|---|
| YOLOv9-T | 45.2 | 43.2 | 67.0 | 64.6 | 57.4 | 35.6 | 39.7 |
| YOLOv9-S | 46.3 | 44.4 | 65.3 | 64.6 | 53.5 | 32.3 | 43.0 |
| YOLOv8-N | 45.2 | 43.9 | 60.6 | 62.3 | 51.4 | 34.1 | 41.7 |
| YOLOv8-S | 45.8 | 44.6 | 61.8 | 59.8 | 51.0 | 31.8 | 43.8 |

### 8.2 Drop from clear photos to synthetic rain and fog (mAP50, points)

| Run | Clear | Rain | Drop (points) | Fog | Drop (points) |
|---|---|---|---|---|---|
| YOLOv9-T | 97.9 | 88.3 | -9.6 | 95.0 | -2.9 |
| YOLOv9-S | 99.0 | 77.0 | -22.0 | 96.6 | -2.4 |
| YOLOv8-N | 99.0 | 83.8 | -15.2 | 96.3 | -2.7 |
| YOLOv8-S | 99.0 | 82.5 | -16.5 | 97.9 | -1.1 |

### 8.3 By source (plates found, %, conf 0.25, IoU 0.5)

| Source (plates / images) | YOLOv9-T | YOLOv9-S | YOLOv8-N | YOLOv8-S |
|---|---|---|---|---|
| `kuet` (387 / 540) | 98.97 | 99.22 | 99.48 | 99.22 |
| `lpdba` (211 / 205) | 96.68 | 97.16 | 97.63 | 98.58 |
| `mahedi` (32 / 32) | 100.00 | 100.00 | 100.00 | 100.00 |
| `nightlpbd` (60 / 60) | 78.33 | 85.00 | 88.33 | 88.33 |
| `ridoykj` (274 / 274) | 97.08 | 99.64 | 98.91 | 99.27 |
| `towniyan` (18 / 13) | 100.00 | 94.44 | 94.44 | 88.89 |
| `tusar` (29 / 29) | 93.10 | 93.10 | 96.55 | 93.10 |

### 8.4 Next to the paper

The paper reports no robustness or condition results; it reports AP by object size (APS / APM / APL) on COCO. Plate size in this study is in §9.

---

## 9. Results D — Plate Counts, Localisation and Plate Size

| Item | YOLOv9-T | YOLOv9-S | YOLOv8-N | YOLOv8-S |
|---|---|---|---|---|
| Plates found (%, conf 0.25, IoU 0.5) | 96.64 | 97.82 | 98.12 | 98.12 |
| Correct boxes (%, conf 0.25) | 92.61 | 92.52 | 92.36 | 91.94 |
| Photos with every plate found (%) | 92.80 | 92.89 | 93.24 | 92.45 |
| IoU > 0.7 accuracy (%) | 66.87 | 65.96 | 65.86 | 66.06 |
| Mean best IoU (%) | 75.82 | 76.33 | 76.40 | 76.48 |
| Recall, small plates (%, 5 plates) | 80.00 | 60.00 | 60.00 | 40.00 |
| Recall, medium plates (%, 267 plates) | 95.13 | 95.51 | 97.38 | 97.38 |
| Recall, large plates (%, 739 plates) | 97.29 | 98.92 | 98.65 | 98.78 |

- All four models find 96.64–98.12 % of the plates at conf 0.25 and 91.94–92.61 % of the boxes they output are correct. *Correct %* is a precision at conf 0.25 and counts every extra or misplaced box, so it is not a second measure of plate finding.
- Only 66 % of photos have all plates boxed at IoU > 0.7 (65.86–66.87 %), and the mean best IoU is 75.8–76.5 %. Box tightness, not plate finding, limits mAP50-95 (§11.2).
- The test split holds 5 small plates: recall 80 / 60 / 60 / 40 % is 4, 3, 3 and 2 plates (YOLOv9-T, YOLOv9-S, YOLOv8-N, YOLOv8-S). Medium plates: YOLOv8 models 97.38 % vs YOLOv9 95.13–95.51 %. Large plates: 97.29–98.92 %, YOLOv9-S highest.

---

## 10. Results E — Speed

Batch 1, imgsz 640, T4 jobs on the first 200 test images after 15 warm-up images (CPU: first 50), time = pre-process + inference + post-process (NMS included).

| Run | PyTorch FP32 (T4) | PyTorch FP16 (T4) | TensorRT FP16 (T4) | ONNX on CPU |
|---|---|---|---|---|
| YOLOv9-T | 19.66 ms (50.9 FPS) | 31.56 ms (31.7 FPS) | 6.93 ms (144.4 FPS) | 116.20 ms (8.6 FPS) |
| YOLOv9-S | 23.43 ms (42.7 FPS) | 27.28 ms (36.7 FPS) | 7.24 ms (138.2 FPS) | 296.93 ms (3.4 FPS) |
| YOLOv8-N | 10.89 ms (91.8 FPS) | 16.05 ms (62.3 FPS) | 6.38 ms (156.7 FPS) | 105.83 ms (9.4 FPS) |
| YOLOv8-S | 14.29 ms (70.0 FPS) | 23.22 ms (43.1 FPS) | 7.52 ms (133.0 FPS) | 300.71 ms (3.3 FPS) |

Stage times (ms) — `ms_breakdown` in `results.json` holds the stage times of the **last** measured image only, so it is indicative:

| Run | Mode | Pre-process | Inference | Post-process |
|---|---|---|---|---|
| YOLOv9-T | PyTorch FP32 | 1.84 | 15.49 | 1.23 |
| YOLOv9-T | TensorRT FP16 | 1.91 | 3.95 | 1.29 |
| YOLOv9-T | CPU ONNX | 23.2 | 97.98 | 3.58 |
| YOLOv9-S | PyTorch FP32 | 1.9 | 22.29 | 1.31 |
| YOLOv9-S | TensorRT FP16 | 1.87 | 4.04 | 1.16 |
| YOLOv9-S | CPU ONNX | 6.87 | 285.38 | 1.17 |
| YOLOv8-N | PyTorch FP32 | 1.84 | 6.71 | 1.42 |
| YOLOv8-N | TensorRT FP16 | 1.95 | 3.0 | 1.26 |
| YOLOv8-N | CPU ONNX | 5.56 | 85.38 | 1.39 |
| YOLOv8-S | PyTorch FP32 | 2.5 | 8.55 | 1.27 |
| YOLOv8-S | TensorRT FP16 | 2.54 | 4.11 | 1.87 |
| YOLOv8-S | CPU ONNX | 3.81 | 248.41 | 2.3 |

- TensorRT FP16 is the fastest mode for all four models (6.38–7.52 ms; 133.0–156.7 FPS) and narrows the differences between them to 1.14 ms.
- PyTorch FP16 at batch 1 is **slower** than FP32 for all four models (for example YOLOv9-T 31.56 vs 19.66 ms).
- YOLOv8-N is the fastest in PyTorch FP32 (10.89 ms), FP16 (16.05 ms) and on CPU (105.83 ms). On CPU, YOLOv9-T (116.20 ms) is close to YOLOv8-N, and the two S models need about 300 ms.
- YOLOv9-S is **slower** than YOLOv8-S in PyTorch FP32 (23.43 vs 14.29 ms) and FP16 (27.28 vs 23.22 ms), and marginally faster in TensorRT FP16 (7.24 vs 7.52 ms) and on CPU (296.93 vs 300.71 ms).
- The paper reports no speed figures, so there is nothing to set these against. The two GPU queues ran their speed jobs at the same time (notebook cell 16), whereas the `det.py` docstring asks for an idle GPU.

---

## 11. Discussion and Interpretation

Each statement is marked **Evidence** (follows from the saved numbers or code) or **Hypothesis** (a reasoned explanation not tested here), with the check that would settle it.

### 11.1 YOLOv9 versus YOLOv8: a small difference

- **Evidence.** The four models are within 0.49 points of test mAP50 and 1.17 points of mAP50-95. YOLOv9-S − YOLOv8-S is +0.28 mAP50 and +0.51 mAP50-95; YOLOv9-T − YOLOv8-N is −0.43 and −0.07.
- **Evidence.** Each model was trained once with seed 42. Plates found differ by at most 15 plates (96.64 vs 98.12 %), false alarms by at most 2 photos of 160.
- **Hypothesis.** Differences of this size may be within seed-to-seed variation, so the ranking of YOLOv9-S, YOLOv8-N and YOLOv8-S in mAP50 is not established. *Check:* three or more seeds per model and a paired comparison.
- **Consequence.** What the data do support is the efficiency direction of the paper's claim: YOLOv9-S reaches at least YOLOv8-S accuracy with 34.6 % fewer parameters and 4.2 % fewer GFLOPs (27.4 vs 28.6).

### 11.2 Validation versus test, and mAP50 versus mAP50-95

- **Evidence.** Test mAP50 is higher than validation by 1.4–2.5 points for all four models, while test mAP50-95 is **lower** by 14.2–15.2 points (table in §6.1). On the test split the mean best IoU is 75.8–76.5 % and the IoU > 0.7 accuracy about 66 %.
- **Evidence.** Within the test split, mAP50-95 differs strongly by subset: clear 43.2–44.6 %, night 60.6–67.0 %, blurry 59.8–64.6 % (§8.1) — higher on the harder-looking subsets.
- **Hypothesis.** mAP50-95 is dominated by how tightly the boxes follow each source's annotation style, which differs between sources, rather than by image difficulty. The night and blurry subsets may come from sources whose box conventions differ from those of the large `kuet` and `ridoykj` sources. *Check:* mAP50-95 and mean IoU per source on val and test; inspect boxes of the lowest-IoU sources.
- **Consequence.** Choosing a model by mAP50-95 on the pooled test split mixes annotation style with model quality; mAP50 and plate counts are the safer comparison here.

### 11.3 What this comparison does not test: PGI

- **Evidence.** The notebook's own note says the official repository's PGI-vs-GELAN ablation is not included. The model summary of the YOLOv9-T run lists `Conv`, `ELAN1`, `AConv`, `RepNCSPELAN4`, `SPPELAN`, `Upsample`, `Concat` and a single `Detect` head on layers [15, 18, 21] (544 layers as trained, 197 fused). The paper's Table 2 (Appendix) also lists a single `Predict 15, 18, 21`, i.e. the inference topology.
- **Evidence.** The log contains no auxiliary-branch modules (no `CBLinear` / `CBFuse`).
- **Hypothesis.** The auxiliary reversible branch and the multi-level auxiliary supervision that define PGI were **not trained** here; the YOLOv9 models are GELAN-style networks trained with the standard Ultralytics detection loss. The notebook sentence that the Ultralytics v9 models "are the GELAN+PGI architectures" should be read with this in mind. *Check:* train the official-repository GELAN-S and YOLOv9-S (with auxiliary branch) on the same split and compare.
- **Evidence.** Even in the paper, the PGI effect on the small model is +0.1 AP50:95 (Table 5); a 0.28–0.51-point difference between YOLOv9-S and YOLOv8-S therefore cannot be attributed to PGI.

### 11.4 Fewer parameters, but not faster

- **Evidence.** YOLOv9-T has 33 % fewer parameters than YOLOv8-N and YOLOv9-S 35 % fewer than YOLOv8-S, but in PyTorch FP32 YOLOv9-T needs 19.66 ms against 10.89 ms and YOLOv9-S 23.43 against 14.29 ms. The YOLOv9 graphs have 544 layers as trained (197 fused) against 129 (72 fused) for YOLOv8.
- **Hypothesis.** At batch 1 on a T4 the time is dominated by the number of small kernels rather than by parameters or FLOPs; the many small blocks of the GELAN design (`AConv`, `RepNCSPELAN4`) cost more launches. TensorRT fuses and schedules them better, which is consistent with the gap shrinking to 0.55 ms (6.93 vs 6.38) for the tiny pair. *Check:* a profiler trace in PyTorch, and measurements at larger batch sizes.
- **Evidence.** PyTorch FP16 was slower than FP32 for all four models at batch 1. **Hypothesis:** casting and small-kernel overhead outweigh the gain of half precision at this batch size. *Check:* the same test at batch 8 or 16.
- **Evidence.** The speed jobs of the two GPUs ran at the same time, and training times were taken with two queues active (§7). **Hypothesis:** host-CPU contention may inflate pre- and post-processing times. *Check:* repeat the speed test with one job at a time.

### 11.5 YOLOv9-T was not fully trained

- **Evidence.** The time cap stopped YOLOv9-T after epoch 54 of 60; the final learning rate is 0.00125 and 4 of the 10 no-mosaic epochs were run. It had the lowest test recall (95.65 %) and plates found (96.64 %), and on `nightlpbd` it found 78.33 % of the plates against 85.00–88.33 % for the others (47 vs 51–53 of 60 plates).
- **Evidence.** Its validation mAP50-95 is flat over epochs 49–54 (60.28 % at epoch 50, 60.10 % at epoch 54, best 60.40 % at epoch 49) and it was the slowest model per epoch.
- **Hypothesis.** An unfinished schedule and the missing no-mosaic epochs contribute to its lower recall, but the cap cannot be separated from the architecture in this run. *Check:* re-train YOLOv9-T for 60 epochs without a cap.
- **Evidence (code).** The second `[time cap] ... epoch 55` line appears when the callback runs again during the later validation pass; `results.csv` has 54 rows. **Hypothesis:** this is a duplicate print, not a 55th epoch. *Check:* the callback in `det.py` (`trainer.epoch + 1`).

### 11.6 Conditions

- **Evidence.** The ranking of the models changes with the condition: YOLOv9-T is best on night (97.5 %), blurry (98.1 %), small plates (93.2 %) and synthetic rain (88.3 %) but lowest on clear photos (97.9 % vs 99.0 %) and on synthetic fog; YOLOv9-S is best on all-test and clear but **lowest on synthetic rain** (77.0 %, −22.0 points from clear).
- **Evidence.** On the `nightlpbd` source YOLOv9-T finds the fewest plates at conf 0.25 (78.33 %) yet has the highest night-subset mAP50 (97.5 %).
- **Hypothesis.** The two measures differ because mAP integrates over all confidence thresholds while *found %* uses a fixed 0.25; YOLOv9-T may rank night plates correctly but with lower confidence. *Check:* the PR curve per subset and the found % at lower thresholds (the saved `BoxPR_curve.png` is for the pooled test split).
- **Evidence.** The small-plate subset has 22 images and the test split 5 small plates; the rain and fog copies are synthetic, so 11.3 points of spread in rain is a statement about this synthetic generator.
- **Hypothesis.** The models' response to synthetic rain depends on how the streaks are drawn and may not transfer to real rain. *Check:* real rain photos.
- **Evidence.** False alarms are 5.62 / 6.25 / 6.88 / 5.62 % for YOLOv9-T / YOLOv9-S / YOLOv8-N / YOLOv8-S, i.e. 9, 10, 11 and 9 of 160 photos; the spread is 2 photos.

### 11.7 Sources

- **Evidence.** `mahedi` (32 plates) is found 100 % by all four. `nightlpbd` is the weakest source (78.33–88.33 %), `tusar` 93.1–96.55 % (29 plates), `towniyan` 88.89–100 % (18 plates).
- **Hypothesis.** Night images are the hard case for detection at conf 0.25, and the three small sources are too small (18–32 plates) to rank the models. *Check:* more night data in the test split and per-source confidence intervals.

---

## 12. Limitations of This Study

1. **Single run.** One seed per model and one measurement per speed mode; no confidence intervals. The models differ by 0.49 points of mAP50.
2. **Fine-tuning, not training from scratch.** The paper trains from scratch for 500 epochs on COCO; here COCO-pretrained Ultralytics models are fine-tuned for 60 epochs on one class.
3. **No PGI ablation, and PGI not trained** (§11.3).
4. **YOLOv9-T is not fully trained:** stopped at epoch 54 of 60 by the time cap.
5. **Small subsets.** 5 small plates in the test split, 22 images in the small-plate subset, 18–60 plates in four of the seven sources; rain and fog are synthetic copies.
6. **Labels.** 4,507 of the 8,574 training images are auto-labelled; validation and test labels are human.
7. **Possible source overlap.** The dataset has `leaky_kuet` subsets; whether the main `test` split shares images with training for `kuet` (the `copy` / `dup` subsets) was not checked here.
8. **Speed.** Measured on a Tesla T4 and one CPU only; both GPU queues ran at the same time; `ms_breakdown` is from the last image only.
9. **Training times** were measured with two GPU queues running at the same time.
10. **mAP50-95 depends on annotation style** (§11.2), so pooled mAP50-95 is not a clean model ranking.
11. **Only detection.** No plate recognition (OCR) was run on the detected boxes.
12. **Different data and metrics from the paper** (§4); the paper's numbers are COCO numbers.
13. **Not replicated:** PGI-vs-GELAN ablation, from-scratch training, YOLOv9-M/C/E, INT8 TensorRT.

---

## 13. Conclusions and Recommendations

### Conclusions

1. On the 498R plate detection test split, all four detectors reach 98.30–98.79 % mAP50 and find 96.64–98.12 % of the plates at conf 0.25. YOLOv9-S is highest in mAP50 (98.79 %), mAP50-95 (46.33 %) and recall (97.20 %).
2. YOLOv9-S matches or slightly beats YOLOv8-S (+0.28 mAP50, +0.51 mAP50-95) with 34.6 % fewer parameters — the direction of the paper's Table 1, at a smaller size, from one run per model.
3. Fewer parameters did not give lower latency: YOLOv8-N is the fastest model in PyTorch and on CPU; TensorRT FP16 puts all four within 6.38–7.52 ms.
4. The models differ most on synthetic rain (77.0–88.3 %) and on `nightlpbd` plates found (78.33–88.33 %); YOLOv9-T is best on the real night, blurry and small-plate subsets but is the least completely trained model.
5. The study does not test PGI: the Ultralytics YOLOv9 graph has no auxiliary branch.
6. The paper's numbers (AP50 63.4 %, AP 46.8 %) could not be matched or directly compared, because data, training and class count differ.

### Recommendations

| # | Recommendation | Based on |
|---|---|---|
| 1 | Use YOLOv9-S where accuracy and recall matter most (mAP50 98.79 %, recall 97.20 %) | §6.1 |
| 2 | Use YOLOv8-N where cost matters: 3.01 M parameters, 101.0 min training, 6.38 ms TensorRT FP16 and 105.83 ms on CPU, at 98.73 % mAP50 | §6.1, §7, §10 |
| 3 | Use YOLOv9-T only where model size (2.01 M, 4.4 MiB) is the constraint, after re-training without the time cap and checking night images | §11.5 |
| 4 | Deploy with TensorRT FP16 on a T4; do not use PyTorch FP16 at batch 1 | §10 |
| 5 | Judge models by mAP50 and plate counts alongside mAP50-95, and report per-source results | §11.2 |
| 6 | Describe these results as GELAN-based YOLOv9 versus YOLOv8, not as a PGI test | §11.3 |
| 7 | Have a person check detections where mistakes matter (false alarms 5.62–6.88 % of plate-free photos) | §8.1 |

### Experiments that would settle the open questions

| # | Experiment | Settles |
|---|---|---|
| 1 | Three or more seeds per model; paired comparison | §11.1 |
| 2 | Per-source mAP50-95 and mean IoU on val and test | §11.2 |
| 3 | Official-repository GELAN-S vs YOLOv9-S (with auxiliary branch) on the same split | §11.3 |
| 4 | Profiler trace and batch-size sweep (1, 8, 16) with one speed job at a time | §11.4 |
| 5 | Re-train YOLOv9-T for 60 epochs without a cap | §11.5 |
| 6 | PR curves per subset; found % at lower confidence thresholds | §11.6 |
| 7 | Real rain and fog photos in place of synthetic copies | §11.6 |
| 8 | Check the `kuet` test images against the `leaky_kuet` duplicate lists | Limitation 7 |
| 9 | Train from scratch for more epochs on a larger merged set | Limitation 2 |

---

## 14. Reproducibility

### 14.1 Commands (from the training logs and `det.py`)

```bash
# GPU 0 queue (YOLOv9)
python det.py --data detection_all.yaml --model yolov9t.pt --name v9t --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 3.2
python det.py --data detection_all.yaml --model yolov9s.pt --name v9s --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 4.0
# GPU 1 queue (YOLOv8), at the same time
python det.py --data detection_all.yaml --model yolov8n.pt --name v8n --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 2.8
python det.py --data detection_all.yaml --model yolov8s.pt --name v8s --project runs --epochs 60 --imgsz 640 --batch 16 --fraction 1.0 --max-hours 3.6
# Speed (after training), one job per GPU
python det.py --data detection_all.yaml --name v9t --project runs --imgsz 640 --fps-only --trt
```

Notebook route: open `notebooks/Md.Yousuf/paper28_run.ipynb` on Kaggle (GPU T4 × 2, Internet on), run once with `FAST = True` to check every cell, then set `FAST = False` and use Save Version → Save & Run All.

### 14.2 Supplied artefacts

The supplied artefacts are `paper28_v9t_best.pt`, `paper28_v9s_best.pt`, `paper28_v8n_best.pt`, `paper28_v8s_best.pt` (weights), `paper28_results.zip` (results, logs, code, environment), `results_vs_paper.md`, and the paper PDF. The SHA-256 values listed in `Paper28_README.md` were re-computed on the files downloaded from the repository folder and all seven match. See [Appendix E](#appendix-e--artefact-manifest).

### 14.3 Determinism

Seed 42 and `deterministic: true` for all four trainings (`args.yaml`). The test split is used once, at the end, with `best.pt`. The time cap stops training after the epoch that passes the limit and keeps the epoch plan (so the learning-rate schedule of a capped run is not completed).

---

## Appendix A — Training Arguments and Validation Curve

| Argument | Value |
|---|---|
| Task / optimiser | detect / SGD |
| Epochs / batch / imgsz | 60 / 16 / 640 |
| lr0 / lrf / momentum / weight decay | 0.01 / 0.01 / 0.937 / 0.0005 |
| Warm-up epochs / momentum / bias lr | 3.0 / 0.8 / 0.1 |
| Loss gains box / cls / dfl | 7.5 / 0.5 / 1.5 |
| Mosaic / close_mosaic | 1.0 / 10 |
| HSV h / s / v | 0.015 / 0.7 / 0.4 |
| translate / scale / fliplr | 0.1 / 0.5 / 0.5 |
| mixup / cutmix / copy_paste | 0.0 / 0.0 / 0.0 |
| auto_augment / erasing | randaugment / 0.4 |
| Seed / deterministic / AMP | 42 / True / True |
| patience / workers / fraction | 100 / 4 / 1.0 |
| Training rows | 8,574 (human 4,067 + auto 4,507) |

Validation mAP50-95 (%) per epoch:

| Run | Ep 1 | Ep 5 | Ep 10 | Ep 20 | Ep 30 | Ep 40 | Ep 50 | Ep 54 | Ep 60 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv9-T | 52.92 | 52.70 | 54.56 | 58.08 | 58.22 | 58.64 | 60.28 | 60.10 | — |
| YOLOv9-S | 52.14 | 50.23 | 56.09 | 57.77 | 58.93 | 60.40 | 60.69 | 61.21 | 60.39 |
| YOLOv8-N | 51.33 | 47.12 | 53.51 | 56.14 | 57.46 | 58.62 | 59.24 | 59.27 | 59.16 |
| YOLOv8-S | 54.39 | 51.40 | 54.14 | 57.78 | 58.58 | 59.66 | 59.99 | 59.82 | 59.62 |

Validation mAP50 (%) per epoch:

| Run | Ep 1 | Ep 5 | Ep 10 | Ep 20 | Ep 30 | Ep 40 | Ep 50 | Ep 54 | Ep 60 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv9-T | 93.17 | 93.38 | 94.58 | 96.07 | 96.18 | 96.38 | 96.93 | 96.98 | — |
| YOLOv9-S | 93.02 | 89.97 | 94.08 | 96.41 | 96.62 | 96.97 | 96.62 | 97.09 | 97.04 |
| YOLOv8-N | 91.02 | 90.84 | 93.40 | 95.44 | 96.79 | 96.93 | 96.17 | 96.06 | 96.20 |
| YOLOv8-S | 92.14 | 90.42 | 93.27 | 96.98 | 96.80 | 96.92 | 97.30 | 97.30 | 97.27 |

(— : the run had ended; YOLOv9-T ran 54 epochs.)

---

## Appendix B — Loss Curves

Training box loss per epoch:

| Run | Ep 1 | Ep 5 | Ep 10 | Ep 20 | Ep 30 | Ep 40 | Ep 50 | Ep 54 | Ep 60 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv9-T | 1.223 | 1.212 | 1.114 | 1.033 | 0.975 | 0.934 | 0.879 | 0.781 | — |
| YOLOv9-S | 1.206 | 1.182 | 1.089 | 0.995 | 0.931 | 0.892 | 0.824 | 0.728 | 0.679 |
| YOLOv8-N | 1.199 | 1.173 | 1.096 | 1.017 | 0.953 | 0.919 | 0.858 | 0.758 | 0.710 |
| YOLOv8-S | 1.202 | 1.171 | 1.074 | 0.984 | 0.913 | 0.862 | 0.792 | 0.694 | 0.642 |

Validation box loss per epoch:

| Run | Ep 1 | Ep 5 | Ep 10 | Ep 20 | Ep 30 | Ep 40 | Ep 50 | Ep 54 | Ep 60 |
|---|---|---|---|---|---|---|---|---|---|
| YOLOv9-T | 1.457 | 1.464 | 1.420 | 1.341 | 1.314 | 1.305 | 1.273 | 1.266 | — |
| YOLOv9-S | 1.457 | 1.491 | 1.399 | 1.343 | 1.312 | 1.274 | 1.262 | 1.257 | 1.268 |
| YOLOv8-N | 1.448 | 1.489 | 1.404 | 1.357 | 1.331 | 1.314 | 1.292 | 1.282 | 1.282 |
| YOLOv8-S | 1.423 | 1.472 | 1.388 | 1.339 | 1.310 | 1.297 | 1.270 | 1.270 | 1.269 |

---

## Appendix C — Full Metrics by Condition

All values in %, from the Ultralytics validator on each subset (`plates found` = plate counts at conf 0.25, IoU 0.5 on that subset). Rain and fog are synthetic copies of the whole test split.

**YOLOv9-T**

| Subset | P | R | mAP50 | mAP50-95 | Plates found (%) |
|---|---|---|---|---|---|
| All test | 97.36 | 95.65 | 98.30 | 45.16 | 96.64 |
| Clear | 97.61 | 97.10 | 97.85 | 43.24 | 97.74 |
| Night | 94.03 | 88.77 | 97.51 | 67.02 | 81.69 |
| Blurry | 100.00 | 88.54 | 98.14 | 64.56 | 88.68 |
| Small plates | 94.56 | 91.67 | 93.17 | 57.35 | 95.83 |
| Rain (synthetic) | 96.14 | 83.28 | 88.27 | 35.63 | 74.09 |
| Fog (synthetic) | 96.12 | 93.47 | 94.95 | 39.69 | 92.78 |

**YOLOv9-S**

| Subset | P | R | mAP50 | mAP50-95 | Plates found (%) |
|---|---|---|---|---|---|
| All test | 97.04 | 97.20 | 98.79 | 46.33 | 97.82 |
| Clear | 97.38 | 98.30 | 99.02 | 44.41 | 98.64 |
| Night | 95.51 | 89.91 | 96.79 | 65.30 | 87.32 |
| Blurry | 100.00 | 92.07 | 98.01 | 64.62 | 94.34 |
| Small plates | 87.85 | 90.39 | 90.75 | 53.53 | 95.83 |
| Rain (synthetic) | 84.28 | 81.11 | 77.03 | 32.30 | 66.17 |
| Fog (synthetic) | 96.38 | 96.14 | 96.63 | 43.03 | 95.65 |

**YOLOv8-N**

| Subset | P | R | mAP50 | mAP50-95 | Plates found (%) |
|---|---|---|---|---|---|
| All test | 97.22 | 96.68 | 98.73 | 45.23 | 98.12 |
| Clear | 97.99 | 98.08 | 98.97 | 43.85 | 98.76 |
| Night | 90.58 | 88.73 | 94.54 | 60.55 | 90.14 |
| Blurry | 97.95 | 90.10 | 97.75 | 62.32 | 92.45 |
| Small plates | 86.99 | 87.50 | 92.79 | 51.43 | 95.83 |
| Rain (synthetic) | 95.80 | 78.87 | 83.76 | 34.10 | 72.50 |
| Fog (synthetic) | 96.15 | 93.77 | 96.26 | 41.69 | 93.67 |

**YOLOv8-S**

| Subset | P | R | mAP50 | mAP50-95 | Plates found (%) |
|---|---|---|---|---|---|
| All test | 96.90 | 96.93 | 98.51 | 45.82 | 98.12 |
| Clear | 97.54 | 97.74 | 99.01 | 44.56 | 98.87 |
| Night | 98.05 | 88.73 | 94.34 | 61.75 | 90.14 |
| Blurry | 95.93 | 92.45 | 95.17 | 59.76 | 94.34 |
| Small plates | 77.11 | 91.67 | 84.53 | 50.98 | 91.67 |
| Rain (synthetic) | 93.57 | 80.51 | 82.47 | 31.81 | 74.38 |
| Fog (synthetic) | 96.72 | 96.14 | 97.88 | 43.79 | 95.85 |

---

## Appendix D — Paper Tables Used for Comparison

**Table 1 — extract (MS COCO val; params M / FLOPs G / AP50:95 / AP50 / AP75, %)**

| Model | Params | FLOPs | AP50:95 | AP50 | AP75 |
|---|---|---|---|---|---|
| YOLOv5-S r7.0 | 7.2 | 16.5 | 37.4 | 56.8 | — |
| YOLOv6-S v3.0 | 18.5 | 45.3 | 44.3 | 61.2 | — |
| YOLOv7-S AF | 11.0 | 28.1 | 45.1 | 61.8 | 48.9 |
| YOLOv8-N | 3.2 | 8.7 | 37.3 | 52.6 | — |
| YOLOv8-S | 11.2 | 28.6 | 44.9 | 61.8 | — |
| YOLO MS-S | 8.1 | 31.2 | 46.2 | 63.7 | 50.5 |
| GELAN-S | 7.1 | 26.4 | 46.7 | 63.0 | 50.7 |
| YOLOv9-S | 7.1 | 26.4 | 46.8 | 63.4 | 50.7 |
| YOLOv9-M | 20.0 | 76.3 | 51.4 | 68.1 | 56.1 |
| YOLOv9-C | 25.3 | 102.1 | 53.0 | 70.2 | 57.8 |
| YOLOv9-E | 57.3 | 189.0 | 55.6 | 72.8 | 60.6 |

**Table 5 — ablation on PGI (AP50:95, %)**

| Base | + DS | + PGI |
|---|---|---|
| GELAN-S 46.7 | 46.5 (−0.2) | 46.8 (+0.1) |
| GELAN-M 51.1 | 51.2 (+0.1) | 51.4 (+0.3) |
| GELAN-C 52.5 | 52.5 (=) | 53.0 (+0.5) |
| GELAN-E 55.0 | 55.3 (+0.3) | 55.6 (+0.6) |

**Appendix Table 1 — YOLOv9 hyper-parameters in the paper:** epochs 500; SGD; lr 0.01 → 0.0001, linear; momentum 0.937; weight decay 0.0005; warm-up 3 epochs, momentum 0.8, bias lr 0.1; box 7.5, class 0.5, DFL 1.5; HSV saturation 0.7, value 0.4; translation 0.1; scale 0.9; mosaic 1.0; MixUp 0.15; copy & paste 0.3; close mosaic 15 epochs.

---

## Appendix E — Artefact Manifest

| Path | Description |
|---|---|
| `Paper28_Report.md` | This report |
| `Paper28_README.md` | Model cards, usage guide, file manifest |
| `paper28_v9t_best.pt` | YOLOv9-T weights, best epoch 49 (4,610,249 bytes, 4.4 MiB) |
| `paper28_v9s_best.pt` | YOLOv9-S weights, best epoch 54 (15,202,185 bytes, 14.5 MiB) |
| `paper28_v8n_best.pt` | YOLOv8-N weights, best epoch 56 (6,222,698 bytes, 5.9 MiB) |
| `paper28_v8s_best.pt` | YOLOv8-S weights, best epoch 55 (22,491,690 bytes, 21.5 MiB) |
| `paper28_results.zip` | Results archive (no weights) |
| `results_vs_paper.md` | Short ours-vs-paper table |
| `28_2024_YOLOv9 Learning What You Want to Learn PGI.pdf` | The paper (arXiv:2402.13616) |
| `runs/<run>/results.json`, `results.csv`, `args.yaml` | Per-run summary, per-epoch curve, training arguments |
| `runs/<run>/results.png`, `BoxPR_curve.png`, `confusion_matrix.png`, `labels.jpg` | Ultralytics plots |
| `runs/<run>/cond_yaml/*.yaml` | Dataset files for clear, night, blurry, small plates, rain, fog |
| `logs/v9t.log`, `v9s.log`, `v8n.log`, `v8s.log` | Training logs |
| `logs/speed_*.log` | Speed-test logs |
| `environment.json` | Library and hardware versions |
| `code/det.py`, `code/common.py`, `code/report.py` | Used for Paper #28 |
| `code/seq.py`, `code/metrics.py`, `code/vocab.py`, `code/rec.py`, `code/ved.py` | Shared toolkit modules not used for Paper #28 |

---

## References

1. C.-Y. Wang, I-H. Yeh and H.-Y. M. Liao, "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information," arXiv:2402.13616v2 [cs.CV], 2024. (Paper #28.)
2. G. Jocher et al., Ultralytics YOLO (YOLOv8 / YOLOv9 implementations), version 8.4.170. (Library used for all four detectors.)
3. T.-Y. Lin et al., "Microsoft COCO: Common Objects in Context," ECCV 2014. (Dataset of the paper and source of the pretrained start weights.)
4. C.-Y. Wang, A. Bochkovskiy and H.-Y. M. Liao, "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors," CVPR 2023. (Training setup the paper follows.)
5. 498R BanglaLPR All-in-One v1.0 — team dataset (detection split; sources `kuet`, `lpdba`, `ridoykj`, `nightlpbd`, `tusar`, `mahedi`, `towniyan`).

---

*End of `Paper28_Report.md` — Md. Yousuf, 21 October 2026.*
