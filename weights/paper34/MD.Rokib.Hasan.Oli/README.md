# README.md — Paper #34: SVTRv2, CTC Beats Encoder-Decoder Models in Scene Text Recognition (run on dataset v1.1)

> **Document name:** `README.md`
> **Project:** 498R BanglaLPR — experiments for *"SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition"* (Du, Chen, Xie, Jia & Jiang, ICCV 2025, arXiv 2411.15858 v2; Paper #34)
> **Author / maintainer:** MD. Rokib Hasan Oli
> **Date:** 27 October 2026 (Kaggle run of 26 October 2026, dataset **498R BanglaLPR All-in-One v1.1**; local re-check on an RTX 5060 Ti on 27 October)
> **Full study report:** [`SVTRv2_Study_Report.pdf`](SVTRv2_Study_Report.pdf) (paper study, run check, results, research use, faculty guide; 19 pages)

This package tests whether **SVTRv2** — a CTC text reader with multi-size resizing (MSR), a feature rearrangement
module (FRM) and a training-only semantic guidance module (SGM) — is a good plate reader for **Bangla number plates**.
The authors' own model code (OpenOCR, pinned commit) is trained on our 101 plate tokens, from the authors' English
weights or from scratch, with MSR or a fixed 32x128 input, next to our CRNN baseline; then it is fused with our two best
earlier readers. Sources: `paper34_results.zip` and the local checks in
[`report/MD.Rokib.Hasan.Oli/paper34/source/`](../../../report/MD.Rokib.Hasan.Oli/paper34/source/).

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
| **Question** | Is SVTRv2 (ICCV 2025) a better Bangla plate reader than our CRNN, and how does it compare with our character-detection readers? |
| **Main result** | SVTRv2-T (English start, MSR) **81.26 %** plate exact on 651 test crops vs CRNN 73.43 % (87 vs 36 plates, p < 0.0001); SVTRv2-S 81.87 % (tie with T) |
| **Against our best readers** | character YOLO (Model Fusion, #24) 86.79 % and LPNR YOLO (Cascaded YOLO, #2) 87.71 % are still 5.5-6.5 points better (p ≤ 0.0004) |
| **Pre-training** | from scratch SVTRv2-T reads only 61.75 % (11.7 points **below** CRNN); the authors' English weights add +19.5 points |
| **MSR** | MSR vs fixed 32x128: +6.76 (77 vs 33, p < 0.0001); +9.78 on the 48x96 size group, +3.95 (n.s.) on near-square plates — the opposite of the paper's Table 1 |
| **As a second reader** | character YOLO + SVTRv2-T format fallback **90.17 %** (22 fixed / 0 broken; SVTRv2 runs on 7 % of plates); four-reader vote chosen on val **90.94 %** (16 fixed / 2 broken vs the best fusion so far, 88.79 %, p = 0.0013) |
| **Speed (our PC, batch 1, model only)** | GPU: T 4.40 / S 4.23 / CRNN 4.19 ms (all launch-bound); CPU 4 threads: **T 13.6 ms**, S 21.3 ms, CRNN 18.6 ms; ONNX of T: same text on 651 / 651, 7.9-8.7 ms on 4 CPU threads |
| **Run health** | 5 / 5 jobs `code=0` in 3.57 h, dataset v1.1, no NaN; local re-scoring = Kaggle for all 5 runs (same text in 7,079 of 7,085 readings) |
| **Caveats** | SVTRv2 runs stopped by the 1.6 h cap at epoch 63-66 of 100; checkpoint kept on val loss (epochs 55 / 35 / 23 / 54); T4 ms in `results_vs_paper.md` measured with the other GPU busy. The kit is fixed (60 epochs, val-accuracy checkpoint, timing at the end) |
| **Environment** | Python 3.13.15 · PyTorch 2.11.0+cu128 · OpenOCR commit `1ccfc6e` · Kaggle Tesla T4 x2 · dataset v1.1 |

---

## 2. What Is in This Package

### 2.1 Files

| File | Size | Description |
|---|---|---|
| `README.md` | — | This guide |
| `SVTRv2_Study_Report.pdf` | 4.1 MiB | Full study report, 19 pages |
| `svtrv2_t_msr.pt` | 19.6 MiB | **SVTRv2-T + FRM/CTC, MSR input** (the recommended reader), FP32, with its class list and settings |
| `svtrv2_s_msr_fp16.pt` | 21.5 MiB | SVTRv2-S, MSR input, stored in FP16 (the FP32 file is 43 MiB, over GitHub's 25 MB web limit); load with `.float()` |
| `paper34_results.zip` | 0.2 MiB | Results archive, **no model weights** (see 2.2) |
| `results_vs_paper.md` | — | The kit's ours-vs-paper table and the per-condition table (read the notes in section 7 first) |
| `fusion_with_earlier_readers.md` | — | Fusion of character YOLO (#24), LPNR YOLO (#2), CRNN, SVTRv2-T and -S: 156 combinations x 6 rules, chosen on val |

**Not in this folder:** the fixed-32x128 and from-scratch SVTRv2-T models (ablations only) and the CRNN. The CRNN is the
same file as paper #24's (`crnn/model_full.pt` SHA-256 `0110631e…f033`), so use
[`weights/paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt`](../../paper24/MD.Rokib.Hasan.Oli/best_crnn_fp16.pt). All five
models of the run are in `paper34_outputs.zip` (156 MiB) on the team Drive, `498R / weights / paper34 /`.

### 2.2 Inside `paper34_results.zip`

| Path | What |
|---|---|
| `runs/<run>/results.json` | settings (start weights, epochs run, MSR buckets), loss log per epoch, val / test plate exact, CER, token accuracy, by source / MSR size / condition, fog and rain tests, ms per plate, parameters |
| `runs/<run>/ocr_preds_test.csv`, `ocr_preds_val.csv` | every crop: truth, prediction, confidence, source, MSR size (the input of the fusion) |
| `logs/*.log`, `environment.json`, `code/*.py` | the job logs, versions (dataset_version 1.1) and the exact kit code that ran |
| `results_vs_paper.md` | the same table as in this folder |

---

## 3. Quick Start

Tested word for word on 27 Oct 2026 in a clean folder with this repository's `weights/` and `src/`: the result for the
example crop equals the Kaggle prediction, and over all 651 test crops `svtrv2_t_msr.pt` gives the same text as Kaggle on
651 / 651 (81.26 %), `svtrv2_s_msr_fp16.pt` on 650 / 651 (81.87 %).

```bash
# from the repository root
git clone https://github.com/Topdu/OpenOCR.git
git -C OpenOCR checkout 1ccfc6ee6161f7133b192e16af3a9265273997ba
pip install torch torchvision opencv-python pillow numpy tqdm
```

```python
import sys

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path[:0] = ['OpenOCR', 'src/MD.Rokib.Hasan.Oli/paper34/kaggle']     # the authors' model code + our kit
import p34_svtrv2 as P
import vocab

ck = torch.load('weights/paper34/MD.Rokib.Hasan.Oli/svtrv2_t_msr.pt', map_location='cpu', weights_only=True)
model = P.build(ck['size'], len(ck['classes']) + 1)
model.load_state_dict({k: v.float() for k, v in ck['state_dict'].items()})
model.eval()


def read_plate(path):
    img = cv2.imread(path)                                       # a plate crop (BGR), e.g. from the plate detector
    h, w = img.shape[:2]
    H, W = P.BUCKETS[P.bucket(h, w, ck['resize'])]               # MSR size: 64x64, 48x96, 40x112 or 32x128
    x = Image.fromarray(img[:, :, ::-1]).resize((W, H), Image.BICUBIC)
    x = TF.normalize(TF.to_tensor(x), [0.5] * 3, [0.5] * 3)[None]
    with torch.no_grad():
        (toks, conf), = P.decode(model(x))
    return vocab.tokens_to_text([ck['classes'][k - 1] for k in toks]), conf


print(read_plate('plate.jpg'))
```

For test crop `anpr_0142111c1238.jpg` (dataset v1.1, `ocr/images/test/`) this prints
`('ঢাকা মেট্রো-ল ৪৪-৫১৬০', 0.9255…)`. On a Windows console set `PYTHONIOENCODING=utf-8` first, or the Bangla text cannot
be printed. For SVTRv2-S use `svtrv2_s_msr_fp16.pt` (same code). To use it as the **fallback reader**: run it only when
the main reader's text fails `metrics.is_valid_plate` (same kit folder), as in `fusion_with_earlier_readers.md`.

---

## 4. Model Card

| | SVTRv2-T (`svtrv2_t_msr.pt`) | SVTRv2-S (`svtrv2_s_msr_fp16.pt`) |
|---|---|---|
| Architecture | OpenOCR `SVTRv2LNConvTwo33` encoder (dims 64/128/256, depths 3/6/3, heads 2/4/8) + `RCTCDecoder` (FRM + CTC) | dims 96/192/384, depths 3/6/3, heads 3/6/12, same head |
| Parameters | 5.13 M | 11.25 M |
| Classes | 101 plate tokens (64 district names / short forms, মেট্রো, 26 series letters, 10 digits) + CTC blank; list stored in the file | same |
| Input | RGB crop, PIL bicubic resize to its MSR size (64x64 if w/h < 1.5, 48x96 if < 2.5, 40x112 if < 3.5, else 32x128), mean / std 0.5 | same |
| Start weights | the authors' English SVTRv2-T (Union14M-Filter, trained with SGM; their result file: 94.78 Common, 79.05 U14M = paper Table 3); encoder 164 / 164 and FRM 14 / 16 tensors loaded, classifier new | the authors' SVTRv2-S (96.13 / 83.70) |
| Training | 5,607 v1.1 train crops, AdamW lr 5e-4, wd 0.05, one-cycle 7.5 % warm-up, batch 64, FP16, our augmentation; 63 of 100 planned epochs (time cap), kept epoch 55 (lowest val loss); 96 min on a T4 | 66 epochs run, kept epoch 35; 97 min |
| Test (651 crops) | plate exact 81.26 % (95 % CI 78.2-84.2), CER 2.49 %, token accuracy 97.16 % | 81.87 % (78.8-84.8), CER 3.07 %, token 97.12 % |
| Val (766 crops) | 73.76 % | 75.33 % |

---

## 5. Results Summary

All on the same 651 v1.1 test crops; p = exact McNemar test on the plates where two readers disagree.

| Reader | Plate exact % | CER % | Night (65) | Blurry (112) | Rain, synthetic (651) |
|---|---|---|---|---|---|
| SVTRv2-T, MSR, English start | **81.26** | 2.49 | 69.23 | 73.21 | 75.27 |
| SVTRv2-S, MSR, English start | 81.87 | 3.07 | 67.69 | 79.46 | 76.80 |
| SVTRv2-T, fixed 32x128 | 74.50 | 4.47 | 52.31 | 67.86 | 66.97 |
| SVTRv2-T, MSR, from scratch | 61.75 | 8.55 | 40.00 | 54.46 | 59.60 |
| CRNN (same file as #24) | 73.43 | 5.53 | 61.54 | 64.29 | 67.13 |
| character YOLO (#24 run) | 86.79 | 2.51 | 80.00 | 74.11 | 75.88 |
| LPNR YOLO (#2 run) | 87.71 | 2.15 | 83.08 | 79.46 | 74.50 |

| Fusion (chosen on validation) | Test % | vs character YOLO alone | vs best fusion so far (88.79 %) |
|---|---|---|---|
| character YOLO + SVTRv2-T, format fallback | **90.17** | 22 fixed / 0 broken | 11 / 2, p = 0.02 |
| character YOLO + LPNR + SVTRv2-T + S, MV-BM+format vote | **90.94** | 28 / 1 | 16 / 2, p = 0.0013 |

Error types, MSR size groups, sources, confidence curves and examples: report sections 12-17.

---

## 6. Reproducing the Results

1. **Kaggle:** import [`src/MD.Rokib.Hasan.Oli/paper34/kaggle/paper34_kaggle.ipynb`](../../../src/MD.Rokib.Hasan.Oli/paper34/kaggle/paper34_kaggle.ipynb)
   (the kit as fixed on 27 Oct: 60 epochs, checkpoint on val plate exact, timing at the end; about 3.5 h on T4 x2,
   Internet on, the dataset downloads itself). The notebook exactly as run on 26 Oct is
   [`notebooks/MD.Rokib.Hasan.Oli/paper34_run.ipynb`](../../../notebooks/MD.Rokib.Hasan.Oli/paper34_run.ipynb).
2. **Local re-check, analysis, figures, PDF:** in [`report/MD.Rokib.Hasan.Oli/paper34/source/`](../../../report/MD.Rokib.Hasan.Oli/paper34/source/):
   `local_check.py <unpacked results.zip>` (venv with CUDA torch) → `analysis.py <run> <fusion dir>` → `charts.py` →
   `build_report.py`. The fusion is the kit's `p24_fusion.py` over the five readers' `ocr_preds_*.csv`
   (`fusion34_results.json`).

---

## 7. Limitations

- **Training was cut short:** every SVTRv2 run stopped at epoch 63-66 of 100 (1.6 h cap; ~90 s per epoch because the
  PIL augmentation is CPU-bound), so the learning rate never annealed. SVTRv2's numbers are, if anything, too low.
- **Checkpoint on validation loss:** CTC val loss rises from over-confidence while accuracy still improves; the fixed
  32x128 run kept epoch 23, so the MSR-vs-fixed and S-vs-T comparisons are partly checkpoint comparisons.
- **`results_vs_paper.md`:** the "ms per plate" row was measured while the other T4 was training (the same SVTRv2-T:
  7.62 / 14.15 / 14.99 ms), and the "Fixed 32x128 (all plates)" row shows, for the MSR runs, the single test plate in the
  32x128 size group (100.00 / 0.00), not a fixed-size score. Use the timings above and the report.
- **Local timings** were taken while another training job shared our GPU (10th percentile of 600 calls); compare readers
  with each other, not with other reports.
- One seed per setting; small condition subsets (night 65, blurry 112, real fog 26 crops).

---

## 8. Differences from the Original Paper

| Paper | Here | Why |
|---|---|---|
| English words (94 characters), U14M-Filter (~3.2 M crops), 20 epochs, batch 1024, lr 6.5e-4, 4 x RTX 4090 | Bangla plate tokens (101 classes), 5,607 crops, 100 planned (63-66 run), batch 64, lr 5e-4, Kaggle T4 | our task and hardware |
| SGM trained in a second phase | not trained (CTC branch only); the English start weights were trained with it | plate text has almost no language context |
| Word accuracy on English benchmarks | plate exact on our test split, by MSR group, condition and source | different data: a direction, not a head-to-head |
| FPS on a 1080Ti, PyTorch eager, batch 1 | ms on an RTX 5060 Ti / 4 CPU threads, batch 1, model only | our hardware |

---

## 9. File Integrity

| File | SHA-256 |
|---|---|
| `svtrv2_t_msr.pt` | `dfaaaae29883839740ae651051a84cb6fed7c8b2a5e7ba27bf747dfa402e9ead` |
| `svtrv2_s_msr_fp16.pt` | `8bf41bd8a1a9f57674f493e7b940b056c13f7f566b501ee98531b1de869b83f5` |
| `paper34_results.zip` | `d7bf5590d681770725ff9b1d177cc53db3e2d9340625dc398e55e01263f215f5` |
| `SVTRv2_Study_Report.pdf` | `610183dfa7fc8e4db126b426b46bab21bbc2fcb48d6117aaf9881bf21049f503` |

The state dicts inside the two `.pt` files are the run's `runs/svtrv2_t_msr/best.pt` (unchanged values) and
`runs/svtrv2_s_msr/best.pt` (cast to FP16); each file stores the source file's SHA-256 under `source_sha256`.

---

## 10. Licence and Data Notes

The model code is the authors' OpenOCR (Apache-2.0, not copied here; clone it at the pinned commit). The start weights
come from the authors' public Google Drive folders. Our trained weights, code and results follow this repository's
terms. The plate crops are from the 498R BanglaLPR All-in-One dataset v1.1 (team Drive); no images are in this folder.

---

## 11. Citation

```bibtex
@inproceedings{du2025svtrv2,
  title     = {{SVTRv2}: {CTC} Beats Encoder-Decoder Models in Scene Text Recognition},
  author    = {Du, Yongkun and Chen, Zhineng and Xie, Hongtao and Jia, Caiyan and Jiang, Yu-Gang},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {20147--20156},
  year      = {2025}
}
```
